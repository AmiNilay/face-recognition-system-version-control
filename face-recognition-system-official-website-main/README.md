# 🌐 Face Recognition System — Official Website (v2.1.0)

<p align="center">
  <img src="https://img.shields.io/badge/Version-2.1.0_Enterprise-7928CA?style=for-the-badge" alt="Version 2.1.0">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenCV-YuNet_DNN-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Tests-74%2F74_Passing-success?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-41AD48?style=for-the-badge" alt="License">
</p>

<p align="center">
  <strong>The official responsive web portal for the Face Recognition System (2026 Enterprise Edition).</strong><br>
  Showcases real-time biometric face detection, multi-layered anti-spoofing, Apple Face ID guided enrollment, FastAPI microservices, and RTSP IP camera streaming.
</p>

---

## 🚀 Overview

This repository contains the standalone, high-performance static website for the **Face Recognition System**. Designed with a modern, Apple-inspired playful aesthetic (Fredoka, Baloo 2, Bungee typography with dark/vibrant elements), the website comprehensively presents the entire 2026 biometric platform architecture.

### 💎 Key Highlights Featured on the Website

1. **🛡️ Multi-Layered Biometric Anti-Spoofing:**
   - Dynamic **Eye Aspect Ratio (EAR)** blink validation.
   - **2D Fast Fourier Transform (FFT)** screen moiré pattern defense against phones, tablets, and glossy print attacks.
   - **3D solvePnP Head Pose Estimation** calculating Pitch, Yaw, and Roll Euler angles with 3D nose ray projection.
   - Dynamic challenge-response verification.

2. **📱 Apple Face ID 5-Pose Enrollment Wizard:**
   - 36-segment circular HUD progress ring.
   - Guided multi-angle calibration: Frontal, Left (Yaw -25°), Right (Yaw +25°), Up (Pitch +18°), and Down (Pitch -18°).
   - Computes $L_2$-normalized Fréchet centroid vectors for extreme angle invariance.

3. **⚡ High-Performance $O(\log N)$ Spatial Indexing:**
   - SciPy `cKDTree` Euclidean nearest-neighbor indexing for sub-millisecond recognition against 10,000+ identities.
   - OpenCV YuNet ONNX Deep Neural Network engine executing at **60+ FPS**.

4. **🌐 FastAPI Edge Microservice & WebSockets:**
   - Asynchronous ASGI REST server (`src/api.py`, port 8000) powered by Uvicorn.
   - Interactive OpenAPI **Swagger UI** (`/docs`) and **ReDoc** (`/redoc`).
   - Real-time WebSocket video and event streaming (`/ws/stream`).

5. **📹 RTSP IP Camera Streaming & MOG2 Motion Gating:**
   - Zero-latency frame buffer (`cv2.CAP_PROP_BUFFERSIZE = 1`) with automatic reconnection.
   - OpenCV `createBackgroundSubtractorMOG2` motion gating reducing idle CPU consumption by over **90%**.
   - Asynchronous security webhooks dispatched to IoT access control, Slack, or Discord.

6. **🔒 Cryptographic Hardening & Audit Integrity:**
   - Military-grade AES-256 Fernet biometric template encryption (`.dat`).
   - Tamper-evident chained SHA-256 cryptographic audit logs (`logs/audit_log.json`).
   - Path traversal string sanitization protecting file operations.

7. **🎙️ Offline Speech Synthesizer Voice Engine:**
   - Integrated `AudioAnnouncer` voice synthesizer (pyttsx3) greeting authorized individuals with zero external cloud dependencies.

8. **📊 Executive Attendance Analytics & HTML Reports:**
   - Indian Standard Time (IST) automated timestamping.
   - Apple-styled executive HTML dashboards with responsive tables, badges, and CSV export.

9. **🧪 100% Verified Quality:**
   - 74 / 74 automated unit tests passing across `tests/` in ~2 seconds.

---

## 📂 Website File Structure

```text
face-recognition-system-official-website-main/
├── index.html            # Main responsive landing page (v2.1.0 Enterprise)
├── css/
│   └── style.css         # Custom CSS design system, responsive grid & animations
├── js/
│   └── main.js           # Tab switcher, stats observer, copy-to-clipboard, parallax
├── assets/
│   └── FRC SS.png        # Official GUI application preview screenshot
└── README.md             # Website documentation (this file)
```

---

## 🖥️ Interactive Showcase Tabs in `index.html`

The demo section features 5 interactive tabs:
- **`🖼️ Desktop GUI`**: Real-time video canvas, multi-color confidence boxes, audio greeting feedback, and dark-mode controls (`run_gui.bat`).
- **`📱 Face ID Wizard`**: Apple Face ID 36-segment circular HUD preview with 5-pose angular progress (`run_face_id_wizard.bat`).
- **`🌐 FastAPI Swagger`**: Interactive API catalog with `POST /api/v1/recognize`, `POST /api/v1/enroll/composite`, `GET /api/v1/health`, and `WS /ws/stream` (`run_api.bat`).
- **`📹 RTSP & Motion`**: RTSP IP camera log preview demonstrating zero-latency streaming and MOG2 power-saving motion gating (`run_rtsp_stream.bat`).
- **`💻 Launchers & CLI`**: One-click Windows batch scripts and complete pytest test runner summary.

---

## ⚡ Previewing & Running the Website

### Option 1: Direct File Open
Simply double-click `index.html` or open it directly in any modern browser:
```cmd
start index.html
```

### Option 2: Local HTTP Server (Python)
To test smooth asset loading and WebSocket previews:
```bash
cd face-recognition-system-official-website-main
python -m http.server 3000
```
Open your browser and navigate to `http://localhost:3000`.

---

## 🚀 One-Click Windows Launchers (Main System)

| Launcher Batch File | Description | Port / Target |
| :--- | :--- | :--- |
| `run_gui.bat` | Launches Desktop Dark GUI with Video HUD | Tkinter / OpenCV |
| `run_api.bat` | Starts FastAPI Edge Microservice | `http://localhost:8000` |
| `run_face_id_wizard.bat` | Launches Apple Face ID 5-Pose Enrollment Wizard | HUD GUI (Pygame/Tkinter) |
| `run_rtsp_stream.bat` | Launches RTSP IP Camera Stream with MOG2 Gating | RTSP Streamer |

---

## 👤 Author & Credits

- **Creator & Lead Architect:** [Nilay Naha](https://github.com/AmiNilay)
- **Project Repository:** [face-recognition-system-version-control](https://github.com/AmiNilay/face-recognition-system-version-control)
- **License:** MIT License
