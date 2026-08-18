import csv
import io
import math
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from threading import Lock

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from ultralytics import YOLO

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

# ROI 列表: [{ id, category, color, x, y, w, h }]
rois = [
    {
        "id": "roi_default_1",
        "category": "飲食區 Food",
        "color": "#FF5722",
        "x": 0.15, "y": 0.20, "w": 0.30, "h": 0.35
    },
    {
        "id": "roi_default_2",
        "category": "休息區 Rest",
        "color": "#4CAF50",
        "x": 0.55, "y": 0.45, "w": 0.35, "h": 0.40
    }
]

# 事件紀錄列表
event_logs = []
active_contacts = {} # key: f"{stable_id}_{roi_id}" -> { start_frame, start_sec, last_seen_frame, roi_category }
next_event_id = 1

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
        os.path.join("runs", "detect", "runs", "detect", "train_yolo11", "weights", "best.pt"),
        os.path.join("runs", "detect", "train_yolo11", "weights", "best.pt"),
        os.path.join("runs", "detect", "train_yolo26n_pet_boxes", "weights", "best.pt"),
        "best.pt"
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

        # 透過 StrongSORT (BoT-SORT backend) 進行追蹤
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
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                raw_id = int(box.id[0]) if box.id is not None else None
                if raw_id is not None:
                    stable_id = id_smoother.update(raw_id, cx, cy, cls_id, frame_count)
                    active_stable_ids.append(stable_id)
                    current_positions[stable_id] = {'last_pos': (cx, cy), 'cls_id': cls_id}
                else:
                    stable_id = None

                detected_objects.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': cx, 'cy': cy,
                    'stable_id': stable_id,
                    'conf': conf
                })

        id_smoother.post_frame_update(frame_count, active_stable_ids, current_positions)

        # -------------------------------------------------------------
        # ROI 交集計算
        # -------------------------------------------------------------
        current_frame_contacts = set()
        active_roi_hit_ids = set()

        with data_lock:
            current_rois = list(rois)

        for roi in current_rois:
            rx1 = int(roi['x'] * w_frame)
            ry1 = int(roi['y'] * h_frame)
            rx2 = int((roi['x'] + roi['w']) * w_frame)
            ry2 = int((roi['y'] + roi['h']) * h_frame)

            for obj in detected_objects:
                if obj['stable_id'] is None:
                    continue

                ox1, oy1, ox2, oy2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
                
                ix1 = max(ox1, rx1)
                iy1 = max(oy1, ry1)
                ix2 = min(ox2, rx2)
                iy2 = min(oy2, ry2)

                inter_w = max(0, ix2 - ix1)
                inter_h = max(0, iy2 - iy1)
                inter_area = inter_w * inter_h
                obj_area = (ox2 - ox1) * (oy2 - oy1)

                if inter_area > 0 and (inter_area / max(1, obj_area)) >= 0.05:
                    contact_key = f"{obj['stable_id']}_{roi['id']}"
                    current_frame_contacts.add(contact_key)
                    active_roi_hit_ids.add(roi['id'])

                    if contact_key not in active_contacts:
                        active_contacts[contact_key] = {
                            'start_frame': frame_count,
                            'start_sec': video_time_sec,
                            'last_seen_frame': frame_count,
                            'object_id': obj['stable_id'],
                            'roi_id': roi['id'],
                            'roi_category': roi['category']
                        }
                        
                        with data_lock:
                            event_logs.append({
                                'id': next_event_id,
                                'time_str': time_str,
                                'timestamp_sec': round(video_time_sec, 1),
                                'frame_idx': frame_count,
                                'object_id': f"#{obj['stable_id']}",
                                'roi_category': roi['category'],
                                'event_type': 'ENTER',
                                'duration_sec': 0.0
                            })
                            next_event_id += 1
                    else:
                        active_contacts[contact_key]['last_seen_frame'] = frame_count

        # 檢查離開事件 (EXIT)
        for c_key in list(active_contacts.keys()):
            if c_key not in current_frame_contacts:
                info = active_contacts[c_key]
                if (frame_count - info['last_seen_frame']) > 10:
                    dur_sec = round((info['last_seen_frame'] - info['start_frame']) / fps_src, 1)
                    with data_lock:
                        event_logs.append({
                            'id': next_event_id,
                            'time_str': time_str,
                            'timestamp_sec': round(video_time_sec, 1),
                            'frame_idx': frame_count,
                            'object_id': f"#{info['object_id']}",
                            'roi_category': info['roi_category'],
                            'event_type': 'EXIT',
                            'duration_sec': max(0.1, dur_sec)
                        })
                        next_event_id += 1
                    del active_contacts[c_key]

        # -------------------------------------------------------------
        # 影像繪製與標註 (僅顯示 Track ID 與 信心度)
        # -------------------------------------------------------------
        for roi in current_rois:
            rx1 = int(roi['x'] * w_frame)
            ry1 = int(roi['y'] * h_frame)
            rx2 = int((roi['x'] + roi['w']) * w_frame)
            ry2 = int((roi['y'] + roi['h']) * h_frame)
            color_bgr = hex_to_bgr(roi['color'])
            is_hit = roi['id'] in active_roi_hit_ids

            thickness = 3 if is_hit else 2
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color_bgr, thickness)

            badge_text = f"ROI: {roi['category']}"
            if is_hit:
                badge_text += " [CONTACT!]"

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (rx1, max(0, ry1 - 22)), (rx1 + tw + 10, ry1), color_bgr, -1)
            cv2.putText(frame, badge_text, (rx1 + 5, max(12, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        for obj in detected_objects:
            ox1, oy1, ox2, oy2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
            stable_id = obj['stable_id']
            conf = obj['conf']

            obj_color = hex_to_bgr(COLOR_HEX_LIST[stable_id % len(COLOR_HEX_LIST)]) if stable_id else (0, 255, 0)
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), obj_color, 2)

            label = f"#{stable_id} ({conf:.2f})" if stable_id else f"({conf:.2f})"
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
        return jsonify({"status": "success", "count": len(rois)})
    else:
        with data_lock:
            return jsonify({"rois": rois})

@app.route("/api/rois/<roi_id>", methods=["DELETE"])
def delete_roi(roi_id):
    global rois
    with data_lock:
        rois = [r for r in rois if r['id'] != roi_id]
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
    writer.writerow(["Event ID", "Video Time", "Timestamp (Sec)", "Frame", "Object ID", "ROI Category", "Event Type", "Duration (Sec)"])
    for row in logs_copy:
        writer.writerow([
            row['id'], row['time_str'], row['timestamp_sec'], row['frame_idx'],
            row['object_id'], row['roi_category'],
            row['event_type'], row['duration_sec']
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
