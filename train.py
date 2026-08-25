import os
from ultralytics import YOLO

def main():
    # Load YOLO Small P2-Head Model (4 Detection Heads: P2/P3/P4/P5 for small object detection)
    model = YOLO("yolov8s-p2.yaml")

    # Absolute path to data.yaml
    data_yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Cat and Dog cctv.yolo26", "data.yaml"))
    print(f"Dataset config path: {data_yaml_path}")

    # Optimized Fast GPU Training with P2-Head and RAM Caching
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        cache=True,       # 啟用記憶體快取加速 (零硬碟 I/O 讀取)
        workers=0,        # Windows 系統下消除 PyTorch DataLoader 跨進程開銷
        device=0,
        project="runs/detect",
        name="train_yolo_p2_fast"
    )
    print("Training finished successfully.")

if __name__ == "__main__":
    main()