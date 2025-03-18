import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# COCO Class Labels (80 Classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Cache the model to load only once
@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    return hub.load(model_url)

model = load_model()

def detect_objects(image):
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)
    output = model(input_tensor)
    return output

# Streamlit UI
st.title("Object Detection Web App")
st.write("Upload an image to detect objects!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Detecting objects...")
    output = detect_objects(image)

    detection_classes = output["detection_classes"].numpy()[0].astype(int)
    detection_scores = output["detection_scores"].numpy()[0]

    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:
            class_id = detection_classes[i]
            class_name = COCO_CLASSES[class_id - 1] if 1 <= class_id <= 80 else "Unknown"
            st.write(f"Detected: {class_name} - Confidence: {detection_scores[i]:.2f}")
