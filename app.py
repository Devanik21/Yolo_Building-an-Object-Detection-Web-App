import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Cache the model loading to speed up subsequent runs.
@st.cache_resource
def load_model():
    # Load the SSD MobileNet V2 model from TensorFlow Hub
    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    return hub.load(model_url)

model = load_model()

def detect_objects(image):
    # Convert the PIL image to a NumPy array
    img_array = np.array(image)
    # Expand dimensions to match the expected input shape: [1, height, width, channels]
    img_array = np.expand_dims(img_array, axis=0)
    # Convert to tensor (model expects dtype tf.uint8)
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)
    
    # Run inference
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
    
    # Extract detection classes and scores from model output
    detection_classes = output["detection_classes"].numpy()[0].astype(int)
    detection_scores = output["detection_scores"].numpy()[0]
    
    for i in range(len(detection_scores)):
        # Only display detections with a confidence above 0.5
        if detection_scores[i] > 0.5:
            st.write(f"Detected class: {detection_classes[i]} - Confidence: {detection_scores[i]:.2f}")
