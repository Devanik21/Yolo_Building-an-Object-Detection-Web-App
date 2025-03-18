import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageDraw

# Load EfficientDet Model (Supports 600+ Classes)
@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/google/efficientdet/lite2/detection/1"
    return hub.load(model_url)

model = load_model()

# Load 600+ Class Labels from Open Images Dataset
@st.cache_resource
def load_labels():
    labels_url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
    import pandas as pd
    labels_df = pd.read_csv(labels_url, header=None)
    return dict(zip(labels_df[0], labels_df[1]))

LABELS = load_labels()

def detect_objects(image):
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)
    output = model(input_tensor)
    return output

def draw_boxes(image, output):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size

    detection_classes = output["detection_class_entities"].numpy()[0]
    detection_scores = output["detection_scores"].numpy()[0]
    detection_boxes = output["detection_boxes"].numpy()[0]

    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.4:  # Lower threshold for more objects
            class_id = detection_classes[i].decode("utf-8")
            class_name = LABELS.get(class_id, "Unknown")

            y_min, x_min, y_max, x_max = detection_boxes[i]
            x_min, x_max = int(x_min * width), int(x_max * width)
            y_min, y_max = int(y_min * height), int(y_max * height)

            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            label = f"{class_name} ({detection_scores[i]:.2f})"
            draw.text((x_min, y_min - 10), label, fill="black")

    return image

# Streamlit UI
st.title("🚀 Advanced Object Detection (600+ Classes)")
st.write("Upload an image to detect objects!")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    st.write("🔍 Detecting objects...")
    output = detect_objects(image)

    result_image = draw_boxes(image, output)
    st.image(result_image, caption="🖼️ Detected Objects", use_column_width=True)
