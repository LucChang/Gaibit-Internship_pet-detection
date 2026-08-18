import os
from ultralytics import YOLO

def main():
    # Load YOLO26 medium model
    model = YOLO("../yolo26s.pt")
    #model = YOLO("yolo11n.pt")

    # Absolute path to data.yaml
    data_yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Cat and Dog cctv.yolo26", "data.yaml"))
    print(f"Dataset config path: {data_yaml_path}")

    # Start training using the GPU (device=0)
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project="runs/detect",
        name="train_yolo26_small"
    )
    print("Training finished successfully.")

if __name__ == "__main__":
    main()