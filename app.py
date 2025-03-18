import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.saved_model.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

model = load_model()

# Function to detect objects
def detect_objects(image):
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Run the model
    detector = model.signatures["serving_default"]
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)
    output = detector(input_tensor)

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

    # Display results
    detection_classes = output["detection_classes"].numpy()[0].astype(int)
    detection_scores = output["detection_scores"].numpy()[0]

    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:  # Confidence threshold
            st.write(f"Detected: {detection_classes[i]} - Confidence: {detection_scores[i]:.2f}")
