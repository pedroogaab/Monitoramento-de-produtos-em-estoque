import torch 
from ultralytics import YOLO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

model_path = "detect/weights/best.pt"
model = YOLO(model_path)