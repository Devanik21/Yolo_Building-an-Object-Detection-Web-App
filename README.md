# YOLO Building An Object Detection Web App

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/Yolo_Building-an-Object-Detection-Web-App?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/Yolo_Building-an-Object-Detection-Web-App?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Real-time object detection in your browser — YOLO v8 wrapped in a FastAPI backend and Streamlit frontend for zero-setup deployment.

---

**Topics:** `computer-vision` · `deep-learning` · `image-processing` · `object-detection` · `opencv` · `python-web-app` · `real-time-detection` · `ultralytics` · `yolo-algorithm`

## Overview

This project wraps the YOLOv8 object detection model in a clean, deployable web application stack:
a FastAPI backend handles image processing and model inference, while a Streamlit frontend provides
the user interface for image upload, webcam capture, and result visualisation. The combination
delivers near-real-time detection on CPU and GPU-accelerated inference when a CUDA device is available.

YOLOv8 (You Only Look Once version 8) from Ultralytics is the current state of the art in real-time
object detection — achieving 53.9% mAP on COCO at 160 FPS on modern hardware. This application
wraps YOLOv8n (nano), YOLOv8s (small), and YOLOv8m (medium) variants, allowing users to balance
speed and accuracy based on their deployment constraints. All 80 COCO object classes are supported
out of the box, with custom model loading available for fine-tuned variants.

Beyond static image detection, the application supports webcam-based live detection with configurable
confidence threshold and NMS IoU threshold, making it suitable for real-time monitoring use cases.
Detected objects are annotated with bounding boxes, class labels, and confidence scores, and a
detection summary table breaks down counts by class.

---

## Motivation

YOLOv8 is powerful but requires setup: Python environments, model weights, GPU drivers. This project
was built to eliminate that friction — wrapping the full detection pipeline in a web application that
anyone can run with three commands, and that can be deployed to any cloud platform without modification.
It also serves as a production-architecture template for any custom YOLO-based detection system.

---

## Architecture

```
Input: Image Upload / Webcam / Video URL
        │
  Streamlit Frontend
  (file uploader, webcam capture, confidence slider)
        │
  FastAPI /detect endpoint
  ├── Image decode (Pillow / OpenCV)
  ├── YOLOv8 inference (ultralytics)
  │   └── Returns: boxes, classes, confidences
  └── Annotated image + JSON detections
        │
  Streamlit: annotated image display
           + detection summary table
           + class count bar chart
```

---

## Features

### Multi-Modal Input
Accepts image upload (JPG/PNG/WebP), real-time webcam capture via Streamlit's camera input, and URL-based image loading — all processed through the same detection pipeline.

### YOLOv8 Model Variants
Switch between YOLOv8n (fastest, 3.2M params), YOLOv8s (balanced, 11.2M params), and YOLOv8m (most accurate, 25.9M params) via sidebar selector.

### Configurable Detection Thresholds
Confidence threshold (0.1–0.9) and NMS IoU threshold (0.1–0.9) adjustable via sliders, enabling fine-grained control over detection sensitivity and duplicate suppression.

### Custom Model Support
Load any custom-trained YOLOv8 .pt model file via file upload, enabling domain-specific detection (medical imaging, industrial inspection, satellite imagery) without code changes.

### Annotated Output Image
Detected objects annotated with colour-coded bounding boxes per class, class label, and confidence score in a clean, readable font.

### Detection Summary Table
Tabular breakdown of detected objects: class name, count, mean confidence, and bounding box coordinates — exportable as CSV.

### Class Distribution Chart
Horizontal bar chart of detection counts per class for the current image, giving immediate statistical insight into scene composition.

### FastAPI JSON Endpoint
RESTful /detect endpoint accepting multipart form data and returning JSON with full detection metadata — enabling integration with other systems.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **ultralytics (YOLOv8)** | Object detection model | State-of-the-art real-time detection, 80 COCO classes |
| **FastAPI** | Inference API backend | Async REST endpoint for scalable detection requests |
| **Streamlit** | Web frontend | Zero-configuration UI with webcam and file upload support |
| **OpenCV** | Image processing | Frame capture, resize, BGR/RGB conversion |
| **Pillow** | Image I/O | Multi-format image loading and saving |
| **PyTorch** | Deep learning backend | YOLOv8 model inference, CUDA support |
| **pandas** | Results handling | Detection table construction and CSV export |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/Yolo_Building-an-Object-Detection-Web-App.git
cd Yolo_Building-an-Object-Detection-Web-App
python -m venv venv && source venv/bin/activate
pip install ultralytics fastapi uvicorn streamlit opencv-python pillow pandas torch

# Start the FastAPI backend (terminal 1)
uvicorn api:app --reload --port 8000

# Start the Streamlit frontend (terminal 2)
streamlit run app.py
```

---

## Usage

```bash
# API: detect objects in an image
curl -X POST http://localhost:8000/detect \
  -F 'file=@test_image.jpg' \
  -F 'confidence=0.5' \
  -F 'model=yolov8n'

# Batch detect all images in a folder
python batch_detect.py --input ./images/ --output ./results/ --model yolov8s

# Export detection results
python detect.py --image photo.jpg --model yolov8m --save_json detections.json
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `yolov8n.pt` | Default YOLO model variant |
| `CONFIDENCE_THRESHOLD` | `0.5` | Default minimum detection confidence |
| `IOU_THRESHOLD` | `0.45` | Non-maximum suppression IoU threshold |
| `DEVICE` | `auto` | Inference device: auto, cpu, cuda, mps |
| `MAX_IMAGE_SIZE` | `1280` | Maximum input image dimension in pixels |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
Yolo_Building-an-Object-Detection-Web-App/
├── README.md
├── requirements.txt
├── app.py
├── atlas_engine.py
└── ...
```

---

## Roadmap

- [ ] Video file detection with per-frame annotation and output video export
- [ ] Multi-stream webcam support for multi-camera monitoring setups
- [ ] Custom training pipeline: fine-tune YOLOv8 on a user-provided labelled dataset
- [ ] ONNX and TensorRT export for optimised edge deployment
- [ ] Tracking mode: DeepSORT integration for object ID persistence across frames

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

YOLOv8 model weights are downloaded automatically from Ultralytics on first run (~6MB for nano, ~22MB for medium). GPU inference requires CUDA 11.8+ and a compatible NVIDIA GPU. CPU inference is functional but slower — expect 200–800ms per image depending on model size and image resolution.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
