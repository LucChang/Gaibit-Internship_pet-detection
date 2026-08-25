#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Object Detection - OpenCV Real-time Testing Tool
=====================================================
功能：
1. 使用 OpenCV (`cv2.imshow`) 視窗即時播放與繪製 YOLO 模型的 Object Detection 偵測結果。
2. 顯示類別名稱 (Class Name)、信心度分數 (Confidence Score) 與 FPS 幀率。
3. 支援互動按鍵操作：
   - [Q] 或 [ESC] : 結束退出
   - [SPACE]      : 暫停 / 繼續播放
   - [W] / [S]    : 即時調高 (+) / 調低 (-) 信心門檻 (Conf Threshold)
   - [N]          : 切換到下一支測試影片
   - [R]          : 重頭播放當前影片
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

def find_default_model():
    """自動搜尋最優模型檔 best.pt"""
    candidates = [
        os.path.join("runs", "detect", "runs", "detect", "colab_yolo_p2_highres-2", "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "detect", "colab_yolo_p2_highres", "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "detect", "train_yolo26_small-3", "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "detect", "train_yolo26_small-2", "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "detect", "train_yolo11", "weights", "best.pt"),
        os.path.join("runs", "detect", "train_yolo11", "weights", "best.pt"),
        os.path.join("runs", "detect", "train_yolo26n_pet_boxes", "weights", "best.pt"),
        "yolo26n.pt",
        "best.pt"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    for root, _, files in os.walk("."):
        if "best.pt" in files:
            return os.path.join(root, "best.pt")
    return "yolo26n.pt"

def get_video_sources(source_path):
    """搜尋指定路徑下的所有影片檔案"""
    if os.path.isfile(source_path):
        return [source_path]
    
    videos = []
    search_dir = source_path if os.path.isdir(source_path) else "video_test"
    if os.path.exists(search_dir):
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    videos.append(os.path.normpath(os.path.join(root, file)))
    return sorted(videos)

def generate_colors(num_classes=80):
    """產生不同類別的專屬外框顏色 (RGB)"""
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return [tuple(int(c) for c in color) for color in colors]

def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection Real-time Tester")
    parser.add_argument("--model", type=str, default=None, help="YOLO 模型檔案路徑 (.pt)")
    parser.add_argument("--source", type=str, default="video_test", help="影片檔、資料夾或攝影機 (0代表Webcam)")
    parser.add_argument("--conf", type=float, default=0.25, help="初始信心門檻 (0.01~1.00)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 門檻")
    parser.add_argument("--imgsz", type=int, default=640, help="推論圖像尺寸")
    args = parser.parse_args()

    # 1. 載入模型
    model_path = args.model if args.model else find_default_model()
    print(f"==================================================")
    print(f" Loading YOLO Model: {model_path}")
    print(f"==================================================")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[錯誤] 無法載入模型: {e}")
        sys.exit(1)

    # 類別顏色表
    class_names = model.names if hasattr(model, 'names') else {}
    colors = generate_colors(max(len(class_names), 80))

    # 2. 準備影片來源
    is_webcam = args.source.isdigit()
    if is_webcam:
        video_list = [int(args.source)]
    else:
        video_list = get_video_sources(args.source)
        if not video_list:
            print(f"[警告] 在 '{args.source}' 找不到任何影片檔，嘗試使用預設路徑...")
            video_list = get_video_sources("video_test")
            if not video_list:
                print("[錯誤] 找不到可播放的影片檔案！")
                sys.exit(1)

    video_idx = 0
    conf_thresh = args.conf
    paused = False

    print("\n--- 操作說明 ---")
    print("  [Q] / [ESC]  : 退出程式")
    print("  [SPACE]      : 暫停 / 繼續")
    print("  [W] / [S]    : 提高 / 降低信心門檻 (Conf)")
    print("  [N]          : 播放下一支影片")
    print("  [R]          : 重新播放當前影片\n")

    window_name = "YOLO Object Detection Tester (OpenCV)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while video_idx < len(video_list):
        current_src = video_list[video_idx]
        src_name = f"Webcam #{current_src}" if is_webcam else os.path.basename(current_src)
        print(f"\n▶ 正在播放 [{video_idx + 1}/{len(video_list)}]: {current_src}")

        cap = cv2.VideoCapture(current_src)
        if not cap.isOpened():
            print(f"[錯誤] 無法開啟影片來源: {current_src}")
            video_idx += 1
            continue

        prev_time = time.time()
        fps = 30.0

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print(f"✓ 影片結束: {src_name}")
                    break
            else:
                # 暫停時重用最後一幀
                frame_copy = frame.copy()

            curr_time = time.time()
            if curr_time > prev_time and not paused:
                fps = 1.0 / (curr_time - prev_time)
                prev_time = curr_time

            # 3. 執行推論
            results = model.predict(
                source=frame,
                conf=conf_thresh,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False
            )

            # 4. 在 Frame 上繪製 Bounding Boxes
            annotated_frame = frame.copy()
            det_count = 0

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                det_count = len(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = class_names.get(cls_id, str(cls_id))
                    
                    color = colors[cls_id % len(colors)]

                    # 畫矩形外框
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # 標籤文字 (類別名稱 + 信心度)
                    label_text = f"{cls_name} {conf:.2f}"
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # 標籤底色背景塊
                    cv2.rectangle(
                        annotated_frame,
                        (x1, max(0, y1 - text_h - 8)),
                        (x1 + text_w + 4, max(text_h + 8, y1)),
                        color,
                        -1
                    )
                    # 繪製白色標籤文字
                    cv2.putText(
                        annotated_frame,
                        label_text,
                        (x1 + 2, max(text_h + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

            # 5. 繪製狀態資訊 (HUD Overlay)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (420, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            status_text = f"File: {src_name} ({video_idx + 1}/{len(video_list)})"
            fps_text = f"FPS: {fps:.1f} | Detections: {det_count}"
            conf_text = f"Conf Threshold: {conf_thresh:.2f} (W: +0.05 / S: -0.05)"
            pause_text = "STATUS: PAUSED" if paused else "STATUS: PLAYING"

            cv2.putText(annotated_frame, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, fps_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, conf_text, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, pause_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255) if paused else (200, 200, 200), 1, cv2.LINE_AA)

            # 顯示影像視窗
            cv2.imshow(window_name, annotated_frame)

            # 6. 按鍵響應處理
            key = cv2.waitKey(1 if not paused else 50) & 0xFF

            if key in [ord('q'), 27]:  # Q 或 ESC 鍵
                print("\n[使用者退出程式]")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):      # 空白鍵暫停/繼續
                paused = not paused
            elif key in [ord('w'), ord('W')]:  # W 調高 Conf
                conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
                print(f" -> 信心門檻調高至: {conf_thresh:.2f}")
            elif key in [ord('s'), ord('S')]:  # S 調低 Conf
                conf_thresh = max(0.05, round(conf_thresh - 0.05, 2))
                print(f" -> 信心門檻調低至: {conf_thresh:.2f}")
            elif key in [ord('n'), ord('N')]:  # N 下一支影片
                print(" -> 跳至下一支影片")
                break
            elif key in [ord('r'), ord('R')]:  # R 重新播放
                print(" -> 重頭播放影片")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.release()
        video_idx += 1

    cv2.destroyAllWindows()
    print("\n所有影片測試完畢！")

if __name__ == "__main__":
    main()
