import datetime
import os
import time
import threading
from threading import Lock
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    PretrainedConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    RobertaTokenizer,
    RobertaTokenizerFast,
)

# ---------------------------------------------------------------------------
# Florence-2 設施接觸日誌檔案與記錄函式
# ---------------------------------------------------------------------------
FACILITY_CONTACT_LOG_FILE = "florence2_contact.log"

def log_facility_contact(cat_id, box_name, duration_sec, time_str=None, video_name=None, status="接觸超過 5 秒"):
    """
    紀錄貓咪接觸 Florence-2 設施 bounding box 的狀態至日誌 (包含終端機輸出與本地檔案保存)
    :param cat_id: 貓咪 ID (例如 1 或 '#1')
    :param box_name: Bounding box 名稱 (例如 'litter box' 或 'bowl')
    :param duration_sec: 接觸持續時間 (秒)
    :param time_str: 影片時間字串 (例如 '01:23')
    :param video_name: 來源影片檔案名稱
    :param status: 狀態說明 (預設 '接觸超過 5 秒')
    """
    now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_part = f" [影片時間: {time_str}]" if time_str else ""
    video_part = f" [影片: {video_name}]" if video_name else ""
    clean_cat_id = str(cat_id).replace("#", "")
    log_line = f"[{now_ts}]{time_part}{video_part} [Florence-2 設施接觸] 貓咪 ID: #{clean_cat_id} | 標記框名稱: {box_name} | 狀態: {status} | 持續時間: {duration_sec:.1f} 秒"
    
    try:
        print(log_line, flush=True)
    except Exception:
        pass

    try:
        with open(FACILITY_CONTACT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception as e:
        print(f"[Florence-2 Log] 寫入日誌失敗: {e}")


# ---------------------------------------------------------------------------
# Hugging Face Transformers 與 Florence-2 相容性修補 (Monkey-patching)
# ---------------------------------------------------------------------------
def _apply_florence2_patches():
    """修復新版 transformers (4.45+) 對 Florence-2 遠端程式碼的相容性問題"""
    if not hasattr(PretrainedConfig, 'forced_bos_token_id'):
        PretrainedConfig.forced_bos_token_id = None

    if not hasattr(RobertaTokenizer, 'additional_special_tokens') or not isinstance(RobertaTokenizer.additional_special_tokens, property):
        RobertaTokenizer.additional_special_tokens = property(
            lambda self: self.special_tokens_map.get('additional_special_tokens', [])
        )

    if not hasattr(RobertaTokenizerFast, 'additional_special_tokens') or not isinstance(RobertaTokenizerFast.additional_special_tokens, property):
        RobertaTokenizerFast.additional_special_tokens = property(
            lambda self: self.special_tokens_map.get('additional_special_tokens', [])
        )

_apply_florence2_patches()


class Florence2FacilityDetector:
    """
    使用 microsoft/Florence-2-large 模型進行靜態設施 (litter_box, bowl) 定期物件偵測
    採用非同步執行序機制，確保每分鐘 (60 秒) 偵測一次且不阻塞即時串流 (MJPEG Stream)
    """

    def __init__(
        self,
        model_id="microsoft/Florence-2-large",
        interval_sec=60.0,
        task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
        text_input="litter box",
        device=None,
    ):
        self.model_id = model_id
        self.interval_sec = interval_sec
        self.task_prompt = task_prompt
        self.text_input = text_input

        # 運算裝置判斷
        cuda_available = torch.cuda.is_available()
        if device is None:
            self.device = "cuda" if cuda_available else "cpu"
        else:
            self.device = device

        if self.device == "cuda" and not cuda_available:
            print("[Florence-2] [!] 未偵測到 CUDA GPU，自動切換至 CPU 模式。")
            self.device = "cpu"

        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = None
        self.processor = None
        self.is_loading = False
        self.is_ready = False

        self.lock = Lock()
        self.latest_detections = []
        self.last_infer_time = -999.0
        self.infer_busy = False

        # 非同步在背景載入模型
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        """背景載入 Florence-2 模型與 Processor"""
        try:
            self.is_loading = True
            print(f"[Florence-2 設施偵測器] [*] 正在載入模型: {self.model_id} (Device: {self.device})...")
            
            _apply_florence2_patches()

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                attn_implementation="eager",
                torch_dtype=self.torch_dtype,
            ).eval().to(self.device)

            # ⚡ 修復新版 transformers 遺漏 shared weight 綁定導致輸出亂碼的問題
            if hasattr(model.language_model.model, 'shared'):
                shared_weight = model.language_model.model.shared.weight
                model.language_model.model.encoder.embed_tokens.weight = shared_weight
                model.language_model.model.decoder.embed_tokens.weight = shared_weight
                model.language_model.lm_head.weight = shared_weight

            processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

            with self.lock:
                self.model = model
                self.processor = processor
                self.is_ready = True
                self.is_loading = False
            
            print(f"[Florence-2 設施偵測器] [+] 模型載入成功！已使用 {self.device.upper()} 就緒進行設施定位。")
        except Exception as e:
            self.is_loading = False
            print(f"[Florence-2 設施偵測器] [x] 載入模型失敗: {e}")

    def reset(self):
        """切換影片或重設時清空當前設施偵測結果與計時"""
        with self.lock:
            self.latest_detections.clear()
            self.last_infer_time = -999.0
            self.infer_busy = False

    def update_facilities(self, frame_bgr, current_video_time_sec):
        """
        每 interval_sec (預設 60 秒 / 每分鐘) 觸發一次 Florence-2 物件偵測推論 (非同步非阻塞)
        """
        if not self.is_ready or self.infer_busy:
            return

        # 檢查間隔時間 (每 60 秒一次)
        if current_video_time_sec - self.last_infer_time < self.interval_sec:
            return

        self.infer_busy = True
        self.last_infer_time = current_video_time_sec

        # 複製當前畫面供背景推論使用
        frame_copy = frame_bgr.copy()
        threading.Thread(
            target=self._async_infer_worker,
            args=(frame_copy, current_video_time_sec),
            daemon=True,
        ).start()

    def _async_infer_worker(self, frame_bgr, video_time_sec):
        """背景推論工作函式"""
        try:
            h_orig, w_orig = frame_bgr.shape[:2]
            # 轉為 RGB PIL Image
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)

            # Florence-2 DaViT 要求正方形輸入 (768, 768)
            resized_pil = image_pil.resize((768, 768))

            prompt = self.task_prompt + self.text_input

            with self.lock:
                if self.model is None or self.processor is None:
                    self.infer_busy = False
                    return
                model = self.model
                processor = self.processor

            inputs = processor(
                text=prompt,
                images=resized_pil,
                return_tensors="pt"
            ).to(self.device, self.torch_dtype)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False,
                )

            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]

            # 解析回答並縮放至原圖尺寸 (w_orig, h_orig)
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=self.task_prompt,
                image_size=(w_orig, h_orig),
            )

            # 解析設施偵測框 (litter_box, bowl)
            results = self._parse_detections(parsed_answer, w_orig, h_orig)

            with self.lock:
                self.latest_detections = results

            if results:
                print(f"[Florence-2 @ {video_time_sec:.1f}s] 偵測到 {len(results)} 個設施: {[d['cls_name'] for d in results]}")

        except Exception as e:
            print(f"[Florence-2 設施偵測器] 推論過程發生異常: {e}")
        finally:
            self.infer_busy = False

    def _parse_detections(self, parsed_answer, img_w, img_h):
        """將 Florence-2 輸出的 bounding boxes 標準化為系統相容格式"""
        detections = []
        if self.task_prompt not in parsed_answer:
            return detections

        data = parsed_answer[self.task_prompt]
        bboxes = data.get("bboxes", [])
        labels = data.get("labels", [])

        for box, raw_label in zip(bboxes, labels):
            x1, y1, x2, y2 = [int(round(v)) for v in box]

            # 邊界約束
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            norm_label = str(raw_label).strip().lower()

            # 類別正規化 (支援 litter box / bowl 各式別名)
            if any(k in norm_label for k in ["litter", "box", "toilet", "sand"]):
                cls_name = "litter_box"
                cls_id = 98
            elif any(k in norm_label for k in ["bowl", "dish", "plate", "food", "water"]):
                cls_name = "bowl"
                cls_id = 97
            else:
                cls_name = norm_label.replace(" ", "_")
                cls_id = 99

            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "stable_id": None,
                "conf": 0.95,
                "cls_id": cls_id,
                "cls_name": cls_name,
                "box_name": norm_label,
                "is_florence": True,
                "updated_by_sam": True, # 用於前端 UI 綠色框 / 標籤顯示 [Florence-2]
            })

        return detections

    def get_facility_detections(self):
        """取得當前最新的設施偵測列表 (執行序安全)"""
        with self.lock:
            return list(self.latest_detections)

    def log_contact(self, cat_id, box_name, duration_sec, time_str=None, video_name=None, status="接觸超過 5 秒"):
        """記錄貓咪接觸設施 bounding box 的日誌"""
        log_facility_contact(cat_id, box_name, duration_sec, time_str, video_name, status)

