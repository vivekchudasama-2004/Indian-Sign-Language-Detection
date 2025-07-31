from ultralytics import YOLO

# Load local model
model = YOLO("models/yolov8n.pt")  

# Train
results = model.train(
    data="data.yaml",
    epochs=25,
    imgsz=350,
    batch=4,
    device="cpu",  # Force CPU training
    workers=4
)