from ultralytics import YOLO
import os
from roboflow import Roboflow

rf = Roboflow(api_key="6q96lYuYXDGHKvbYFl2t")
project = rf.workspace("weld-detection").project("weld-defect-zjejp")
version = project.version(1)
dataset = version.download("yolov8")

model = YOLO("yolov8n.pt")

results = model.train(data="Weld-Defect-1/data.yaml", epochs=1000)