import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# Load Model
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

def draw_boxes(image, output):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    width, height = image.size
    detection_classes = output["detection_classes"].numpy()[0].astype(int)
    detection_scores = output["detection_scores"].numpy()[0]
    detection_boxes = output["detection_boxes"].numpy()[0]

    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:  # Confidence Threshold
            class_id = detection_classes[i]
            class_name = COCO_CLASSES[class_id - 1] if 1 <= class_id <= 80 else "Unknown"

            # Bounding Box Coordinates (Normalized)
            y_min, x_min, y_max, x_max = detection_boxes[i]
            x_min, x_max = int(x_min * width), int(x_max * width)
            y_min, y_max = int(y_min * height), int(y_max * height)

            # Draw Rectangle (Red)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

            # Draw Label (Black Text)
            label = f"{class_name} ({detection_scores[i]:.2f})"
            text_size = draw.textbbox((0, 0), label)  # Get text size
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]

            # Draw text background (White box for better visibility)
            draw.rectangle([x_min, y_min - text_height - 5, x_min + text_width + 5, y_min], fill="white")

            # Draw label text (Black)
            draw.text((x_min + 2, y_min - text_height - 3), label, fill="black")

    return image

# Streamlit UI
st.title("🖼️ Object Detection Web App")
st.write("Upload an image to detect objects with bounding boxes!")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    st.write("🔍 Detecting objects...")
    output = detect_objects(image)

    # Draw bounding boxes and show image
    result_image = draw_boxes(image, output)
    st.image(result_image, caption="🖼️ Detected Objects", use_column_width=True)
