# 🧠 Facial Models & Landmark Predictors

This directory contains (or links to) pretrained deep learning and machine learning models used for face detection, facial landmark prediction, and face embedding generation.

---

## 🔍 Model Overview

| Model | Purpose | Source / Architecture | Speed | Accuracy |
| :--- | :--- | :--- | :---: | :---: |
| **HOG (Default)** | Face Detection | Histogram of Oriented Gradients (Linear SVM via `dlib`) | ⚡ Very Fast | Good for frontal faces |
| **CNN (Optional)** | Face Detection | Maximum-Margin Object Detector (MMOD) via `dlib` | 🐢 Slower (needs GPU) | High for angled/occluded faces |
| **dlib 68-Point Landmark** | Liveness & Align | Shape Predictor 68 (`shape_predictor_68_face_landmarks.dat`) | ⚡ Fast | Critical for blink & EAR anti-spoofing |
| **ResNet 128D Embedding** | Face Recognition | Pretrained ResNet-34 metric learning network (`face_recognition_models`) | ⚡ Fast (CPU/GPU) | 99.38% LFW benchmark accuracy |
| **OpenCV YuNet (2026 Engine)** | Ultra-fast Detection | ONNX lightweight CNN (`face_detection_yunet_2023mar.onnx`) | 🚀 Real-time 60+ FPS | High |

---

## 📥 Downloading Additional Models (Optional)

When enabling advanced liveness verification (Eye Aspect Ratio / Blink Detection) or OpenCV YuNet:
1. **Dlib 68-Point Face Landmark Predictor:**
   * Download from: `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`
   * Extract `shape_predictor_68_face_landmarks.dat` into this directory.
2. **OpenCV YuNet ONNX:**
   * Download from: [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
   * Place `face_detection_yunet_2023mar.onnx` into this directory.
