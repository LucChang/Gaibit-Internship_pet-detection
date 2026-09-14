import csv
import io
import json
import math
import os
import subprocess
import sys
import time
import threading
from collections import defaultdict, deque
from threading import Lock

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from ultralytics import YOLO
from florence2_facility_detector import Florence2FacilityDetector, log_facility_contact

app = Flask(__name__)

# 全域與執行序安全機制
data_lock = Lock()

def find_default_video():
    """動態搜尋第一個可用的影片檔"""
    for search_root in ["video_test", "."]:
        if os.path.exists(search_root):
            for r, _, files in os.walk(search_root):
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        return os.path.normpath(os.path.join(r, f))
    return os.path.join("video_test", "CCTV_1", "07-18-35-956452.mp4")

# 全域狀態
current_video_path = find_default_video()
model = None
model_path = None
global_facility_detector = None

# ROI 列表記憶與持久化機制 (分監視器/資料夾儲存於 roi_config.json)
ROI_CONFIG_FILE = "roi_config.json"

def get_camera_key(video_path=None):
    """取得目前影片所在的監視器/資料夾識別 Key"""
    path = video_path or current_video_path
    folder = os.path.dirname(path) or "."
    return os.path.normpath(folder).replace("\\", "/")

def load_roi_config(camera_key=None):
    """從 roi_config.json 讀取特定監視器/資料夾的 ROI 設定"""
    global rois
    if camera_key is None:
        camera_key = get_camera_key()
    
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                with data_lock:
                    rois = data.get(camera_key, [])
                    return
        except Exception as e:
            print(f"[警告] 讀取 ROI 記憶設定檔失敗: {e}")
    
    with data_lock:
        rois = []

def save_roi_config(camera_key=None):
    """將目前監視器/資料夾的 ROI 設定存入 roi_config.json"""
    if camera_key is None:
        camera_key = get_camera_key()
    
    all_configs = {}
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, "r", encoding="utf-8") as f:
                all_configs = json.load(f)
        except Exception:
            all_configs = {}
    
    with data_lock:
        all_configs[camera_key] = list(rois)
    
    try:
        with open(ROI_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(all_configs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[錯誤] 儲存 ROI 記憶設定檔失敗: {e}")

# 初始化 ROI 列表並自動載入記憶紀錄
rois = []
load_roi_config()

# 事件紀錄列表
event_logs = []
active_contacts = {} # key: f"{stable_id}_{roi_id}" -> { start_frame, start_sec, last_seen_frame, roi_category }
next_event_id = 1

# 自動擷取片段影片存放目錄
CLIPS_DIR = os.path.join("static", "clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

def generate_clip_async(video_path, start_sec, end_sec, event_ids, object_id, category):
    """背景非同步剪輯接觸事件前後 10 秒影片片段（包含 YOLO 標註與 ROI 標籤）"""
    def _worker():
        try:
            # 稍作等待以確保當前即時串流影片幀已推進過 (end_sec + 10 秒)
            time.sleep(10.0)
            
            if not os.path.exists(video_path):
                print(f"[警告] 來源影片檔不存在: {video_path}")
                return

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[警告] 無法開啟影片檔進行剪輯: {video_path}")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_dur = total_frames / fps
            
            c_start_sec = max(0.0, start_sec - 10.0)
            c_end_sec = min(total_dur, end_sec + 10.0)
            
            start_frame = int(c_start_sec * fps)
            end_frame = int(c_end_sec * fps)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            clean_cat = str(category).replace(" ", "_").replace("/", "_")
            clean_obj = str(object_id).replace("#", "")
            filename = f"clip_evt_{event_ids[0]}_cat_{clean_obj}_{clean_cat}_{int(start_sec)}.mp4"
            out_path = os.path.join(CLIPS_DIR, filename)
            rel_url = f"/static/clips/{filename}"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            # 載入獨立推論模型與 ROI 標籤
            clip_model = YOLO(find_best_pt_model())
            tracker_config = "strongsort_tuned.yaml" if os.path.exists("strongsort_tuned.yaml") else ("botsort.yaml" if os.path.exists("botsort.yaml") else "bytetrack.yaml")
            clip_smoother = TrackIDSmoother(max_dist=100.0)
            
            with data_lock:
                current_rois = list(rois)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            f_idx = start_frame
            clip_frame_count = 0
            
            while f_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                clip_frame_count += 1
                h_f, w_f, _ = frame.shape
                
                # 執行物件推論與標註
                results = clip_model.track(
                    source=frame,
                    conf=0.25,
                    iou=0.45,
                    imgsz=640,
                    tracker=tracker_config,
                    persist=True,
                    verbose=False
                )
                
                # 1. 繪製 ROI 框
                for r in current_rois:
                    rx1 = int(r['x'] * w_f)
                    ry1 = int(r['y'] * h_f)
                    rx2 = int((r['x'] + r['w']) * w_f)
                    ry2 = int((r['y'] + r['h']) * h_f)
                    color_bgr = hex_to_bgr(r['color'])
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color_bgr, 2)
                    badge_text = f"ROI: {r['category']}"
                    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (rx1, max(0, ry1 - 22)), (rx1 + tw + 10, ry1), color_bgr, -1)
                    cv2.putText(frame, badge_text, (rx1 + 5, max(12, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

                # 2. 繪製偵測物件與追蹤標籤
                active_ids = []
                positions = {}
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = clip_model.names.get(cls_id, str(cls_id)) if hasattr(clip_model, 'names') else str(cls_id)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        raw_id = int(box.id[0]) if box.id is not None else None
                        if cls_name not in ['bowl', 'litter_box']:
                            if raw_id is None:
                                raw_id = 99000 + (cls_id * 1000) + (cx % 1000)
                            stable_id = clip_smoother.update(raw_id, cx, cy, cls_id, clip_frame_count)
                            active_ids.append(stable_id)
                            positions[stable_id] = {'last_pos': (cx, cy), 'cls_id': cls_id}
                        else:
                            stable_id = None
                        
                        if cls_name in ['bowl', 'litter_box']:
                            obj_color = hex_to_bgr("#FF5722") if cls_name == 'bowl' else hex_to_bgr("#29B6F6")
                            label = f"{cls_name} ({conf:.2f})"
                        else:
                            obj_color = hex_to_bgr(COLOR_HEX_LIST[stable_id % len(COLOR_HEX_LIST)]) if stable_id else (0, 255, 0)
                            label = f"#{stable_id} ({conf:.2f})" if stable_id else f"{cls_name} ({conf:.2f})"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), obj_color, 2)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        ty1 = max(y1, th + 8)
                        cv2.rectangle(frame, (x1, ty1 - th - 4), (x1 + tw + 6, ty1 + 2), obj_color, -1)
                        cv2.putText(frame, label, (x1 + 3, ty1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                clip_smoother.post_frame_update(clip_frame_count, active_ids, positions)
                out.write(frame)
                f_idx += 1
                
            cap.release()
            out.release()
            
            with data_lock:
                for evt in event_logs:
                    if evt.get('id') in event_ids:
                        evt['clip_url'] = rel_url
            print(f"[系統] 已成功生成包含 YOLO 標註與 ROI 標籤之前後 10 秒影片片段: {filename}")
        except Exception as e:
            print(f"[錯誤] 剪輯影片過程發生異常: {e}")
            
    threading.Thread(target=_worker, daemon=True).start()

# 系統即時數據
system_status = {
    "fps": 0.0,
    "frame_count": 0,
    "total_frames": 0,
    "active_objects": 0,
    "active_contacts": 0,
    "total_events": 0,
    "current_video": "",
    "tracker": "StrongSORT"
}

# 鮮豔色彩對照
COLOR_HEX_LIST = ["#00E676", "#29B6F6", "#FF1744", "#FFEA00", "#E040FB", "#FF9100", "#A7FFEB", "#FF80AB"]

def hex_to_bgr(hex_color):
    """Hex 顏色字串 轉成 OpenCV BGR tuple"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (0, 255, 0)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def find_best_pt_model():
    """自動搜尋最優模型檔 best.pt"""
    candidates = [
        os.path.join("runs", "detect", "runs", "detect", "train_yolo26_small-2", "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "detect", "train_yolo26_small-3", "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "detect", "colab_yolo_p2_highres-2", "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "detect", "colab_yolo_p2_highres", "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "detect", "train_yolo11", "weights", "best.pt"),
        os.path.join("runs", "detect", "train_yolo26n_pet_boxes", "weights", "best.pt"),
        "yolo26n.pt"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    for root, _, files in os.walk("."):
        if "best.pt" in files:
            return os.path.join(root, "best.pt")
    return candidates[0]

class TrackIDSmoother:
    """空間-時間 ID 穩定與重映射器"""
    def __init__(self, max_dist=100.0, max_lost_frames=60):
        self.max_dist = max_dist
        self.max_lost_frames = max_lost_frames
        self.raw_to_stable = {}
        self.lost_tracks = {}
        self.active_in_prev_frame = {}
        self.next_stable_id = 1
        
    def update(self, raw_id, cx, cy, cls_id, current_frame):
        if raw_id in self.raw_to_stable:
            stable_id = self.raw_to_stable[raw_id]
            if stable_id in self.lost_tracks:
                del self.lost_tracks[stable_id]
            return stable_id

        best_stable_id = None
        min_dist = float('inf')

        for s_id, info in list(self.lost_tracks.items()):
            if (current_frame - info['last_frame']) <= self.max_lost_frames:
                lx, ly = info['last_pos']
                dist = math.hypot(cx - lx, cy - ly)
                if dist < self.max_dist and dist < min_dist:
                    min_dist = dist
                    best_stable_id = s_id

        if best_stable_id is not None:
            stable_id = best_stable_id
            del self.lost_tracks[stable_id]
        else:
            stable_id = self.next_stable_id
            self.next_stable_id += 1

        self.raw_to_stable[raw_id] = stable_id
        return stable_id

    def post_frame_update(self, current_frame, current_active_stable_ids, current_positions):
        for s_id, info in list(self.active_in_prev_frame.items()):
            if s_id not in current_active_stable_ids:
                self.lost_tracks[s_id] = {
                    'last_pos': info['last_pos'],
                    'last_frame': current_frame - 1,
                    'cls_id': info.get('cls_id', 0)
                }

        self.active_in_prev_frame = current_positions.copy()

        for s_id, info in list(self.lost_tracks.items()):
            if current_frame - info['last_frame'] > self.max_lost_frames:
                del self.lost_tracks[s_id]

def format_timestamp(seconds):
    """將秒數轉成 MM:SS 格式"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def find_videos_in_folder(folder_path):
    """取得特定資料夾中所有影片檔案的完整路徑列表"""
    v_files = []
    if os.path.exists(folder_path):
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    v_files.append(os.path.normpath(os.path.join(root, f)))
    v_files.sort()
    return v_files

def generate_video_stream():
    """MJPEG 影片串流與即時 YOLO11 + StrongSORT (BoT-SORT + GMC) + ROI 交集感應運算"""
    global model, current_video_path, system_status, next_event_id
    
    if model is None:
        m_path = find_best_pt_model()
        model = YOLO(m_path)

    cap = cv2.VideoCapture(current_video_path)
    if not cap.isOpened():
        print(f"[錯誤] 無法開啟影片: {current_video_path}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    if fps_src <= 0 or np.isnan(fps_src):
        fps_src = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 使用 StrongSORT / BoT-SORT 調校組態
    tracker_config = "strongsort_tuned.yaml" if os.path.exists("strongsort_tuned.yaml") else ("botsort.yaml" if os.path.exists("botsort.yaml") else "bytetrack.yaml")
    id_smoother = TrackIDSmoother(max_dist=100.0)

    # 初始化 Florence-2 靜態設施偵測器 (每 60 秒 / 每分鐘透過 Florence-2-large 偵測)
    global global_facility_detector
    if global_facility_detector is None:
        global_facility_detector = Florence2FacilityDetector(model_id="microsoft/Florence-2-large", interval_sec=60.0)
    facility_detector = global_facility_detector

    frame_count = 0
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # 影片播放結束：自動尋找該監視器資料夾下的下一支影片播順播放
            current_dir = os.path.dirname(current_video_path) or "video_test"
            dir_videos = find_videos_in_folder(current_dir)
            
            if dir_videos:
                try:
                    curr_idx = dir_videos.index(os.path.normpath(current_video_path))
                    next_idx = (curr_idx + 1) % len(dir_videos)
                except ValueError:
                    next_idx = 0
                next_video = dir_videos[next_idx]
            else:
                next_video = current_video_path

            cap.release()
            current_video_path = next_video
            load_roi_config()
            facility_detector.reset()
            cap = cv2.VideoCapture(current_video_path)
            if not cap.isOpened():
                print(f"[警告] 無法自動切換至下一支影片: {next_video}")
                break
            
            fps_src = cap.get(cv2.CAP_PROP_FPS)
            if fps_src <= 0 or np.isnan(fps_src):
                fps_src = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            continue

        frame_count += 1
        h_frame, w_frame, _ = frame.shape
        video_time_sec = frame_count / fps_src
        time_str = format_timestamp(video_time_sec)

        now_time = time.time()
        calc_fps = 1.0 / (now_time - prev_time) if (now_time - prev_time) > 0 else 30.0
        prev_time = now_time

        # 透過 StrongSORT (BoT-SORT backend) 進行動態動物 (Cat/Dog) 高頻即時追蹤
        results = model.track(
            source=frame,
            conf=0.25,
            iou=0.45,
            imgsz=640,
            tracker=tracker_config,
            persist=True,
            verbose=False
        )

        detected_objects = []
        active_stable_ids = []
        current_positions = {}

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, 'names') else str(cls_id)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                raw_id = int(box.id[0]) if box.id is not None else None
                if cls_name not in ['bowl', 'food_bowl', 'litter_box']:
                    if raw_id is None:
                        raw_id = 99000 + (cls_id * 1000) + (cx % 1000)
                    stable_id = id_smoother.update(raw_id, cx, cy, cls_id, frame_count)
                    active_stable_ids.append(stable_id)
                    current_positions[stable_id] = {'last_pos': (cx, cy), 'cls_id': cls_id}
                else:
                    stable_id = None

                detected_objects.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': cx, 'cy': cy,
                    'stable_id': stable_id,
                    'conf': conf,
                    'cls_id': cls_id,
                    'cls_name': cls_name
                })

        id_smoother.post_frame_update(frame_count, active_stable_ids, current_positions)

        # -------------------------------------------------------------
        # 靜態設施 Florence-2 低頻率 (每 5 秒) 偵測 litter_box 與 bowl
        # -------------------------------------------------------------
        facility_detector.update_facilities(frame, video_time_sec)
        florence_facilities = facility_detector.get_facility_detections()

        if florence_facilities:
            # 移除粗略偵測設施，採用 Florence-2 精確定位框
            detected_objects = [obj for obj in detected_objects if obj['cls_name'] not in ['bowl', 'food_bowl', 'litter_box']]
            detected_objects.extend(florence_facilities)

        # -------------------------------------------------------------
        # ROI 交集計算
        # -------------------------------------------------------------
        current_frame_contacts = set()
        active_roi_hit_ids = set()

        with data_lock:
            current_rois = list(rois)

        # -------------------------------------------------------------
        # ROI / Target 交集計算 (貓咪涵蓋率 >= 15% 或 中心點在 ROI 內)
        # -------------------------------------------------------------
        targets = []
        roi_cat_set = set()

        for r in current_rois:
            targets.append({
                'id': r['id'],
                'category': r['category'],
                'box_name': r['category'],
                'color': r['color'],
                'x1': int(r['x'] * w_frame),
                'y1': int(r['y'] * h_frame),
                'x2': int((r['x'] + r['w']) * w_frame),
                'y2': int((r['y'] + r['h']) * h_frame),
                'is_roi': True,
                'is_florence': False
            })
            roi_cat_set.add(r['category'])

        # 加入 Florence-2 偵測到的設施標記框
        for idx, f_obj in enumerate(florence_facilities):
            b_name = f_obj.get('box_name') or f_obj.get('cls_name', 'facility')
            targets.append({
                'id': f"florence_{idx}_{b_name}",
                'category': f_obj.get('cls_name', b_name),
                'box_name': b_name,
                'color': "#00E676" if "litter" in b_name else "#29B6F6",
                'x1': f_obj['x1'],
                'y1': f_obj['y1'],
                'x2': f_obj['x2'],
                'y2': f_obj['y2'],
                'is_roi': False,
                'is_florence': True
            })

        # 若 Florence-2 尚未就緒且無標記框，才使用 YOLO 備用粗略設施
        if not florence_facilities:
            for obj in detected_objects:
                if obj['cls_name'] in ['bowl', 'litter_box'] and obj['cls_name'] not in roi_cat_set:
                    targets.append({
                        'id': f"det_{obj['cls_name']}",
                        'category': obj['cls_name'],
                        'box_name': obj['cls_name'],
                        'color': "#FF5722" if obj['cls_name'] == 'bowl' else "#29B6F6",
                        'x1': obj['x1'],
                        'y1': obj['y1'],
                        'x2': obj['x2'],
                        'y2': obj['y2'],
                        'is_roi': False,
                        'is_florence': False
                    })

        current_frame_contacts = set()
        active_target_hits = {}

        for target in targets:
            tx1, ty1, tx2, ty2 = target['x1'], target['y1'], target['x2'], target['y2']
            target_area = max(1, (tx2 - tx1) * (ty2 - ty1))

            for obj in detected_objects:
                if obj['stable_id'] is None:
                    continue

                ox1, oy1, ox2, oy2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
                obj_area = max(1, (ox2 - ox1) * (oy2 - oy1))
                
                ix1 = max(ox1, tx1)
                iy1 = max(oy1, ty1)
                ix2 = min(ox2, tx2)
                iy2 = min(oy2, ty2)

                inter_w = max(0, ix2 - ix1)
                inter_h = max(0, iy2 - iy1)
                inter_area = inter_w * inter_h
                union_area = obj_area + target_area - inter_area

                iou = (inter_area / union_area) if union_area > 0 else 0.0
                cat_coverage = (inter_area / obj_area) if obj_area > 0 else 0.0
                center_in_target = (tx1 <= obj['cx'] <= tx2) and (ty1 <= obj['cy'] <= ty2)

                # 接觸判定條件：貓咪 15% 進入 ROI、或中心點在 ROI 內、或 IoU >= 10%
                is_contacting = (cat_coverage >= 0.15) or center_in_target or (iou >= 0.10)

                if is_contacting:
                    contact_key = f"{obj['stable_id']}_{target['id']}"
                    current_frame_contacts.add(contact_key)

                    if contact_key not in active_contacts:
                        active_contacts[contact_key] = {
                            'start_frame': frame_count,
                            'start_sec': video_time_sec,
                            'last_seen_frame': frame_count,
                            'object_id': obj['stable_id'],
                            'target_id': target['id'],
                            'roi_category': target['category'],
                            'box_name': target.get('box_name', target['category']),
                            'is_florence': target.get('is_florence', False),
                            'event_logged': False,
                            'logged_5s': False
                        }
                    else:
                        active_contacts[contact_key]['last_seen_frame'] = frame_count

                    dur_sec = round((frame_count - active_contacts[contact_key]['start_frame']) / fps_src, 1)
                    active_target_hits[target['id']] = {
                        'iou': round(max(iou * 100, cat_coverage * 100), 1),
                        'duration_sec': dur_sec,
                        'is_valid': dur_sec >= 1.0
                    }

                    # 貓咪接觸 Florence-2 設施標記框超過 5 秒：透過 log 紀錄狀態 (包含貓咪 id 與 bounding box 名稱)
                    if active_contacts[contact_key].get('is_florence') and dur_sec >= 5.0:
                        if not active_contacts[contact_key].get('logged_5s'):
                            active_contacts[contact_key]['logged_5s'] = True
                            c_id = obj['stable_id']
                            b_name = active_contacts[contact_key].get('box_name', target['category'])
                            v_name = os.path.basename(current_video_path)
                            log_facility_contact(
                                cat_id=c_id,
                                box_name=b_name,
                                duration_sec=dur_sec,
                                time_str=time_str,
                                video_name=v_name,
                                status="接觸超過 5 秒"
                            )
                            # 同步記錄至 Web 系統 event_logs 以便介面與 CSV 檢視
                            with data_lock:
                                event_logs.append({
                                    'id': next_event_id,
                                    'time_str': time_str,
                                    'timestamp_sec': round(video_time_sec, 1),
                                    'frame_idx': frame_count,
                                    'object_id': f"#{c_id}",
                                    'roi_category': f"[Florence-2] {b_name}",
                                    'event_type': 'STAY_5S',
                                    'duration_sec': dur_sec,
                                    'clip_url': ''
                                })
                                next_event_id += 1

                    # 接觸時立即記錄單一 CONTACT 事件 (針對一般 ROI 或初始接觸)
                    if not active_contacts[contact_key]['event_logged']:
                        active_contacts[contact_key]['event_logged'] = True
                        contact_evt_id = next_event_id
                        active_contacts[contact_key]['event_id'] = contact_evt_id
                        with data_lock:
                            event_logs.append({
                                'id': contact_evt_id,
                                'time_str': time_str,
                                'timestamp_sec': round(video_time_sec, 1),
                                'frame_idx': frame_count,
                                'object_id': f"#{obj['stable_id']}",
                                'roi_category': target['category'],
                                'event_type': 'CONTACT',
                                'duration_sec': dur_sec
                            })
                            next_event_id += 1

                        # 當下即觸發非同步剪輯 CONTACT 前後 10 秒影片片段
                        generate_clip_async(
                            video_path=current_video_path,
                            start_sec=active_contacts[contact_key]['start_sec'],
                            end_sec=video_time_sec,
                            event_ids=[contact_evt_id],
                            object_id=obj['stable_id'],
                            category=target['category']
                        )
                    else:
                        # 動態更新持續接觸秒數
                        evt_id = active_contacts[contact_key].get('event_id')
                        if evt_id is not None:
                            with data_lock:
                                for evt in event_logs:
                                    if evt.get('id') == evt_id:
                                        evt['duration_sec'] = dur_sec

        # 接觸結束清理與生成最終完整片段 (無 EXIT 事件)
        for c_key in list(active_contacts.keys()):
            if c_key not in current_frame_contacts:
                info = active_contacts[c_key]
                if (frame_count - info['last_seen_frame']) > 10:
                    dur_sec = round((info['last_seen_frame'] - info['start_frame']) / fps_src, 1)
                    # 若曾觸發超過 5 秒的 Florence-2 設施接觸，紀錄離開時的最終總停留時間
                    if info.get('is_florence') and info.get('logged_5s'):
                        log_facility_contact(
                            cat_id=info['object_id'],
                            box_name=info.get('box_name', info['roi_category']),
                            duration_sec=dur_sec,
                            time_str=format_timestamp(info['last_seen_frame'] / fps_src),
                            video_name=os.path.basename(current_video_path),
                            status=f"結束接觸 (總停留 {dur_sec} 秒)"
                        )
                    if info['event_logged']:
                        evt_id = info.get('event_id')
                        if evt_id is not None:
                            with data_lock:
                                for evt in event_logs:
                                    if evt.get('id') == evt_id:
                                        evt['duration_sec'] = max(0.1, dur_sec)

                        # 背景非同步觸發剪輯該接觸事件 (進入前 10s 至 離開後 10s) 完整影片片段
                        end_sec = info['last_seen_frame'] / fps_src
                        generate_clip_async(
                            video_path=current_video_path,
                            start_sec=info['start_sec'],
                            end_sec=end_sec,
                            event_ids=[evt_id] if evt_id else [],
                            object_id=info['object_id'],
                            category=info['roi_category']
                        )
                    del active_contacts[c_key]

        # -------------------------------------------------------------
        # 影像繪製與標註 (即時顯示 CONTACT 狀態與計時)
        # -------------------------------------------------------------
        for target in targets:
            tx1, ty1, tx2, ty2 = target['x1'], target['y1'], target['x2'], target['y2']
            color_bgr = hex_to_bgr(target['color'])
            
            hit_info = active_target_hits.get(target['id'])
            is_hit = hit_info is not None

            thickness = 3 if is_hit else 2
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color_bgr, thickness)

            if target.get('is_florence'):
                badge_text = f"[Florence-2] {target.get('box_name', target['category'])}"
            elif target.get('is_roi'):
                badge_text = f"ROI: {target['category']}"
            else:
                badge_text = f"Target: {target['category']}"

            if is_hit:
                badge_text += f" [CONTACT! {hit_info['duration_sec']}s]"

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (tx1, max(0, ty1 - 22)), (tx1 + tw + 10, ty1), color_bgr, -1)
            cv2.putText(frame, badge_text, (tx1 + 5, max(12, ty1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        for obj in detected_objects:
            # 設施框已經在 targets 中繪製並標記 CONTACT 狀態，此處僅繪製貓咪與其他動態追蹤目標
            if obj['cls_name'] in ['bowl', 'food_bowl', 'litter_box'] or obj.get('is_florence'):
                continue
            ox1, oy1, ox2, oy2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
            stable_id = obj['stable_id']
            conf = obj['conf']
            cls_name = obj['cls_name']

            obj_color = hex_to_bgr(COLOR_HEX_LIST[stable_id % len(COLOR_HEX_LIST)]) if stable_id else (0, 255, 0)
            label = f"#{stable_id} {cls_name} ({conf:.2f})" if stable_id else f"{cls_name} ({conf:.2f})"

            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), obj_color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty1 = max(oy1, th + 8)
            cv2.rectangle(frame, (ox1, ty1 - th - 4), (ox1 + tw + 6, ty1 + 2), obj_color, -1)
            cv2.putText(frame, label, (ox1 + 3, ty1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        with data_lock:
            system_status['fps'] = round(calc_fps, 1)
            system_status['frame_count'] = frame_count
            system_status['total_frames'] = total_frames
            system_status['active_objects'] = len(detected_objects)
            system_status['active_contacts'] = len(current_frame_contacts)
            system_status['total_events'] = len(event_logs)
            system_status['current_video'] = os.path.basename(current_video_path)
            system_status['tracker'] = "StrongSORT"

        ret_enc, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret_enc:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_buf.tobytes() + b'\r\n')

    cap.release()

# -------------------------------------------------------------
# Flask 路由與 API 控制
# -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/folders", methods=["GET"])
def get_folders():
    valid_folders = []
    search_base_dirs = [".", "video_test"]
    try:
        for base in search_base_dirs:
            if not os.path.exists(base):
                continue
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if not d.startswith(('.', '_')) and d not in ('runs', 'templates', 'static', 'scratch', '__pycache__')]
                if any(f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) for f in files):
                    norm = os.path.normpath(root).replace("\\", "/")
                    if norm not in valid_folders:
                        valid_folders.append(norm)
    except Exception as e:
        print(f"Error scanning folders: {e}")

    if not valid_folders and os.path.exists("video_test"):
        valid_folders.append("video_test")
    
    current_dir = os.path.dirname(current_video_path).replace("\\", "/") or (valid_folders[0] if valid_folders else "video_test")
    return jsonify({"folders": valid_folders, "current_folder": current_dir})

@app.route("/api/videos", methods=["GET"])
def get_videos():
    folder = request.args.get("folder", "").strip()
    if not folder or not os.path.exists(folder):
        folder = os.path.dirname(current_video_path) or "video_test"
    
    files = []
    if os.path.exists(folder) and os.path.isdir(folder):
        for r, _, filenames in os.walk(folder):
            for f in filenames:
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    rel_path = os.path.relpath(os.path.join(r, f), folder)
                    files.append(rel_path.replace("\\", "/"))
    
    current_rel = os.path.relpath(current_video_path, folder).replace("\\", "/") if current_video_path.startswith(folder) else os.path.basename(current_video_path)
    return jsonify({"folder": folder, "videos": sorted(files), "current": current_rel})

@app.route("/api/select_folder", methods=["POST"])
def select_folder():
    global current_video_path
    data = request.get_json() or {}
    folder = data.get("folder", "").strip()
    if folder and os.path.exists(folder):
        videos = find_videos_in_folder(folder)
        if videos:
            current_video_path = videos[0]
            load_roi_config()
            return jsonify({"status": "success", "folder": folder, "video": os.path.basename(current_video_path)})
    return jsonify({"status": "error", "message": "資料夾不存在或無可用影片"}), 400

@app.route("/api/select_video", methods=["POST"])
def select_video():
    global current_video_path
    data = request.get_json() or {}
    folder = data.get("folder", "video_test").strip()
    filename = data.get("video", "").strip()
    if filename:
        target = os.path.normpath(os.path.join(folder, filename))
        if os.path.exists(target):
            current_video_path = target
            load_roi_config()
            return jsonify({"status": "success", "folder": folder, "video": filename})
    return jsonify({"status": "error", "message": "檔案不存在"}), 400

@app.route("/api/rois", methods=["GET", "POST"])
def manage_rois():
    global rois
    if request.method == "POST":
        data = request.get_json() or {}
        new_rois = data.get("rois", [])
        with data_lock:
            rois = new_rois
        save_roi_config()
        return jsonify({"status": "success", "count": len(rois)})
    else:
        with data_lock:
            return jsonify({"rois": rois})

@app.route("/api/rois/<roi_id>", methods=["DELETE"])
def delete_roi(roi_id):
    global rois
    with data_lock:
        rois = [r for r in rois if r['id'] != roi_id]
    save_roi_config()
    return jsonify({"status": "success", "remaining": len(rois)})



@app.route("/api/events", methods=["GET"])
def get_events():
    with data_lock:
        return jsonify({"events": list(reversed(event_logs))})

@app.route("/api/clear_events", methods=["POST"])
def clear_events():
    global event_logs, active_contacts, next_event_id
    with data_lock:
        event_logs.clear()
        active_contacts.clear()
        next_event_id = 1
    return jsonify({"status": "success"})

@app.route("/api/export_csv", methods=["GET"])
def export_csv():
    with data_lock:
        logs_copy = list(event_logs)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Event ID", "Video Time", "Timestamp (Sec)", "Frame", "Object ID", "ROI Category", "Event Type", "Duration (Sec)", "Clip Video URL"])
    for row in logs_copy:
        writer.writerow([
            row['id'], row['time_str'], row['timestamp_sec'], row['frame_idx'],
            row['object_id'], row['roi_category'],
            row['event_type'], row['duration_sec'],
            row.get('clip_url', '')
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=cctv_roi_events.csv"}
    )

@app.route("/api/status", methods=["GET"])
def get_status():
    with data_lock:
        return jsonify(system_status)

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    
    try:
        subprocess.Popen([sys.executable, "video_download.py"])
        print("[系統] 已在背景成功啟動 video_download.py 影片同步服務")
    except Exception as e:
        print(f"[警告] 自動啟動 video_download.py 失敗: {e}")

    print("=" * 60)
    print("CCTV 物件定位與自訂 ROI 接觸事件紀錄 Web 系統 (StrongSORT 追蹤) 啟動中...")
    print("存取網址: http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
