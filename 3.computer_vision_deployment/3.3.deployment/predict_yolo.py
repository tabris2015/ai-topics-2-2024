import os
from ultralytics import YOLO

curr_dir = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(curr_dir, f"models/best.pt")

print(f"model directory: {model_dir}")

model = YOLO(model_dir)

results = model("https://cdn.britannica.com/40/75640-050-F894DD85/tiger-Siberian.jpg")

prediction = results[0]

print(prediction)