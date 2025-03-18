import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob

# Load Pretrained YOLOv10 model
model = YOLO("yolov10n.pt")  # Using YOLOv10 nano version

# Dataset Preparation
data_path = "./BCCD/JPEGImages"
image_paths = glob.glob(f"{data_path}/*.jpg")

def augment_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.CropAndPad(percent=(-0.1, 0.1), p=0.5),
        ToTensorV2()
    ])
    augmented = transform(image=image)
    return augmented['image']

# Fine-tune model
model.train(data="./BCCD.yaml", epochs=10, imgsz=640, batch=8)

# Save fine-tuned model
model.save("yolov10_finetuned.pt")

# Inference function
def predict(image_path):
    image = Image.open(image_path)
    results = model(image)
    return results
