import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

def load_model():
    return YOLO("yolov10_finetuned.pt")

def predict(model, image):
    results = model(image)
    return results

def draw_boxes(image, results):
    image = np.array(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            label = box.cls[0].item()
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return Image.fromarray(image)

# Streamlit App
st.title("YOLOv10 Object Detection App")
model = load_model()

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    results = predict(model, image)
    output_image = draw_boxes(image, results)
    st.image(output_image, caption="Detection Result", use_column_width=True)
