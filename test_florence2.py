import os
import glob
import argparse
from PIL import Image
import torch
import cv2
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    PretrainedConfig,
    RobertaTokenizer,
    RobertaTokenizerFast,
)

# Optional Ultralytics Annotator
try:
    from ultralytics.utils.plotting import Annotator, colors
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


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


def load_florence2_model(model_id="microsoft/Florence-2-large", device=None):
    """
    載入 Florence-2 模型與 Processor
    """
    _apply_florence2_patches()
    cuda_available = torch.cuda.is_available()
    if device is None:
        device = "cuda" if cuda_available else "cpu"
    
    if device == "cuda" and not cuda_available:
        print("[!] 警告: 目前 PyTorch 未啟用 CUDA (GPU)，已自動切換回 CPU 模式！")
        device = "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[*] 🚀 正在使用 GPU: {gpu_name} (CUDA)")
    else:
        print("[*] ⚠️ 目前使用 CPU 模式 (未偵測到 CUDA 或未啟用 GPU)")

    print(f"[*] 正在載入模型: {model_id} 到裝置: {device}...")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # trust_remote_code=True 是 Florence-2 所必須的
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch_dtype
    ).eval().to(device)

    # ⚡ 修復新版 transformers (4.45+) 遺漏 shared weight 綁定導致輸出亂碼的問題
    if hasattr(model.language_model.model, 'shared'):
        shared_weight = model.language_model.model.shared.weight
        model.language_model.model.encoder.embed_tokens.weight = shared_weight
        model.language_model.model.decoder.embed_tokens.weight = shared_weight
        model.language_model.lm_head.weight = shared_weight

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    print("[*] 模型載入完成！")
    return model, processor, device


def run_inference(model, processor, device, image, task_prompt, text_input=None):
    """
    對單張 PIL Image 進行 Florence-2 推論
    
    常用 task_prompt:
      1. '<CAPTION_TO_PHRASE_GROUNDING>' : 依據 text_input 中的詞彙進行目標偵測 (例如 "a cat . a dog . a water bowl")
      2. '<OPEN_VOCABULARY_DETECTION>'  : 開放詞彙目標偵測 (例如 "cat, dog, bowl")
      3. '<OD>'                         : 一般物件偵測 (不需要 text_input)
      4. '<DENSE_REGION_CAPTION>'       : 密集區域描述 (包含位置與敘述)
    """
    # 組合 Prompt
    prompt = task_prompt if text_input is None else task_prompt + text_input

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Florence-2 DaViT 預期正方形 (768, 768) 輸入
    resized_image = image.resize((768, 768))

    # 預處理圖片與 Prompt
    inputs = processor(
        text=prompt,
        images=resized_image,
        return_tensors="pt"
    ).to(device, torch_dtype)

    # 模型生成輸出
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            early_stopping=False,
            do_sample=False,
            num_beams=1 if device == "cpu" else 3,
            use_cache=False
        )

    # 解碼為文字
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # 後處理解析 bounding boxes (還原至原始圖片解析度)
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer


def draw_bounding_boxes(image_pil, parsed_answer, task_key):
    """
    在影像上繪製 Bounding Box 與標籤
    優先使用 Ultralytics Annotator，若無則使用 OpenCV
    """
    # 轉換成 OpenCV 格式 (BGR numpy array)
    img_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    if task_key not in parsed_answer:
        return img_cv2

    result_data = parsed_answer[task_key]
    bboxes = result_data.get("bboxes", [])
    labels = result_data.get("labels", [])

    if HAS_ULTRALYTICS:
        annotator = Annotator(img_cv2, line_width=2, example=str(labels))
        for box, label in zip(bboxes, labels):
            # Florence-2 格式: [x1, y1, x2, y2]
            annotator.box_label(box, label=str(label), color=colors(hash(label) % 10, True))
        return annotator.result()
    else:
        # OpenCV 備用繪製
        for idx, (box, label) in enumerate(zip(bboxes, labels)):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = ((idx * 50) % 255, (idx * 80 + 100) % 255, (idx * 120 + 50) % 255)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_cv2, str(label), (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return img_cv2


def main():
    # ==========================
    # 可直接在程式碼中修改此區設定
    # ==========================
    DEFAULT_INPUT_DIR = "video_test"
    DEFAULT_OUTPUT_DIR = "video_test_output"
    
    # 模式一: 依據指定詞彙偵測 (Phrase Grounding)
    DEFAULT_TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"
    DEFAULT_TEXT_INPUT = "cat, litter box"

    # 模式二: 一般開放物件偵測 (若想測試一般物件偵測，可取消下方註解)
    # DEFAULT_TASK_PROMPT = "<OD>"
    # DEFAULT_TEXT_INPUT = None

    parser = argparse.ArgumentParser(description="Florence-2-large Object Detection & Phrase Grounding Test")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_INPUT_DIR, help="圖片來源資料夾")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="偵測結果儲存資料夾")
    parser.add_argument("--task_prompt", type=str, default=DEFAULT_TASK_PROMPT, help="Florence-2 任務 Prompt")
    parser.add_argument("--text_input", type=str, default=DEFAULT_TEXT_INPUT, help="要偵測的目標文字/類別 (多個類別用 . 隔開)")
    parser.add_argument("--model_id", type=str, default="microsoft/Florence-2-large", help="模型名稱")
    parser.add_argument("--device", type=str, default=None, help="指定運算裝置 (cuda 或 cpu，預設為自動判斷)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 搜尋支援的圖片格式
    supported_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths = []
    for ext in supported_exts:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        # 也搜尋一層子目錄 (若有)
        image_paths.extend(glob.glob(os.path.join(args.image_dir, "*", ext)))

    if not image_paths:
        print(f"[!] 在 '{args.image_dir}' 資料夾中找不到任何圖片！請先將測試照片放入該目錄。")
        return

    print(f"[*] 找到 {len(image_paths)} 張圖片，開始進行測試...")
    print(f"[*] 任務類型: {args.task_prompt}")
    print(f"[*] 偵測目標 (Text Input): {args.text_input}")

    # 載入模型
    model, processor, device = load_florence2_model(args.model_id, device=args.device)

    # 逐一處理圖片
    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        print(f"\n[-] 正在處理: {file_name}")
        
        try:
            image_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[x] 無法開啟圖片 {img_path}: {e}")
            continue

        # 執行推論
        parsed_answer = run_inference(
            model=model,
            processor=processor,
            device=device,
            image=image_pil,
            task_prompt=args.task_prompt,
            text_input=args.text_input
        )

        print(f"    推論結果: {parsed_answer}")

        # 繪製 Bounding Box
        result_img = draw_bounding_boxes(image_pil, parsed_answer, args.task_prompt)

        # 儲存結果
        save_path = os.path.join(args.output_dir, f"detected_{file_name}")
        cv2.imwrite(save_path, result_img)
        print(f"    已儲存標記圖片至: {save_path}")

    print(f"\n[✓] 所有圖片測試完成！標註結果儲存於 '{args.output_dir}' 目錄。")


if __name__ == "__main__":
    main()
