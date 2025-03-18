import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import time
import os

def main():
    # Set page config
    st.set_page_config(
        page_title="Object Detection App",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Object Detection Web App")
    st.sidebar.title("Settings")
    
    # Model selection dropdown
    model_option = st.sidebar.selectbox(
        "Select Model",
        ("YOLO", "SSD MobileNet")
    )
    
    # Confidence threshold slider
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.5, 0.05
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Image or Video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])
    
    # Camera input option
    camera_option = st.sidebar.checkbox("Use Camera")
    
    if camera_option:
        run_camera_detection(model_option, confidence_threshold)
    elif uploaded_file is not None:
        # Check if video or image
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ['jpg', 'jpeg', 'png']:
            process_image(uploaded_file, model_option, confidence_threshold)
        else:
            process_video(uploaded_file, model_option, confidence_threshold)
    else:
        st.info("Upload an image/video file or use camera to detect objects")
        demo_image = Image.open("demo.jpg") if os.path.exists("demo.jpg") else None
        if demo_image:
            st.image(demo_image, caption="Demo Image", use_column_width=True)

def process_image(uploaded_file, model_option, confidence_threshold):
    # Convert uploaded file to opencv format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display original image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_column_width=True)
    
    # Process image based on selected model
    if model_option == "YOLO":
        processed_image, detections = detect_objects_yolo(image, confidence_threshold)
    else:  # SSD MobileNet
        processed_image, detections = detect_objects_ssd(image, confidence_threshold)
    
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Display processed image with detections
    with col2:
        st.subheader("Detected Objects")
        st.image(processed_image_rgb, use_column_width=True)
    
    # Display detection results
    st.subheader("Detection Results")
    if detections:
        for i, (obj_name, confidence) in enumerate(detections):
            st.write(f"{i+1}. {obj_name} - Confidence: {confidence:.2f}")
    else:
        st.write("No objects detected.")

def process_video(uploaded_file, model_option, confidence_threshold):
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Process the video
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Error opening video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video player
    stframe = st.empty()
    st.subheader("Detection Results")
    result_text = st.empty()
    
    detection_results = []
    
    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame based on selected model
        if model_option == "YOLO":
            processed_frame, detections = detect_objects_yolo(frame, confidence_threshold)
        else:  # SSD MobileNet
            processed_frame, detections = detect_objects_ssd(frame, confidence_threshold)
        
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(processed_frame_rgb, use_column_width=True)
        
        # Update detection results
        detection_results = detections
        result_str = ""
        for i, (obj_name, confidence) in enumerate(detection_results):
            result_str += f"{i+1}. {obj_name} - Confidence: {confidence:.2f}\n"
        result_text.text(result_str if result_str else "No objects detected")
        
        # Control the frame rate
        time.sleep(1/fps)
    
    cap.release()
    os.unlink(tfile.name)

def run_camera_detection(model_option, confidence_threshold):
    st.warning("Camera access might not work in deployed web apps due to security restrictions.")
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error opening camera")
        return
    
    # Setup video player
    stframe = st.empty()
    st.subheader("Detection Results")
    result_text = st.empty()
    
    stop_button = st.button("Stop Camera")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame based on selected model
        if model_option == "YOLO":
            processed_frame, detections = detect_objects_yolo(frame, confidence_threshold)
        else:  # SSD MobileNet
            processed_frame, detections = detect_objects_ssd(frame, confidence_threshold)
        
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(processed_frame_rgb, use_column_width=True)
        
        # Update detection results
        result_str = ""
        for i, (obj_name, confidence) in enumerate(detections):
            result_str += f"{i+1}. {obj_name} - Confidence: {confidence:.2f}\n"
        result_text.text(result_str if result_str else "No objects detected")
        
        # Check if stop button is pressed
        if stop_button:
            break
        
        # Rerun the script to check for button press
        time.sleep(0.1)
    
    cap.release()

def detect_objects_yolo(image, confidence_threshold):
    # Load pre-trained YOLO model
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Load class names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Image preprocessing
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    # Forward pass
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    
    # Draw bounding boxes
    output_image = image.copy()
    detections = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green
            
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_image, f"{label} {confidence:.2f}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            detections.append((label, confidence))
    
    return output_image, detections

def detect_objects_ssd(image, confidence_threshold):
    # Load pre-trained SSD MobileNet model
    net = cv2.dnn.readNetFromCaffe("ssd_mobilenet.prototxt", "ssd_mobilenet.caffemodel")
    
    # Load class names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Image preprocessing
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    
    # Forward pass
    net.setInput(blob)
    detections = net.forward()
    
    # Process detections
    output_image = image.copy()
    detection_results = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            
            # Get coordinates
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get label
            if class_id < len(classes):
                label = classes[class_id]
            else:
                label = f"Class {class_id}"
            
            # Add label to image
            cv2.putText(output_image, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detection_results.append((label, confidence))
    
    return output_image, detection_results

if __name__ == "__main__":
    main()
