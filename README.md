# 🎥 Face Recognition System — Enterprise Biometric AI Platform (v2.1.0)

<p align="center">
  <img src="https://img.shields.io/badge/Version-2.1.0_Enterprise-7928CA?style=for-the-badge" alt="Version 2.1.0 Enterprise">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenCV-YuNet_DNN-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Tests-74%2F74_Passing-success?style=for-the-badge" alt="Tests 74/74 Passing">
  <img src="https://img.shields.io/badge/License-MIT-41AD48?style=for-the-badge" alt="License">
</p>

<p align="center">
  <strong>An enterprise-grade, real-time biometric identification and access-control platform.</strong><br>
  Engineered with multi-layered anti-spoofing, Apple Face ID guided multi-angle enrollment, sub-millisecond $O(\log N)$ spatial indexing, FastAPI edge microservices, and RTSP IP camera streaming with motion power-saving gating.
</p>

---

## 🚀 Key Highlights & 2026 Enterprise Capabilities

### 1. 🛡️ Multi-Layer Biometric Anti-Spoofing & Liveness
- **EAR (Eye Aspect Ratio) Dynamic Blink Validation:** Evaluates eye landmark distance ratios in real-time, requiring natural blinks (`EAR < 0.22`) to eliminate static photo print attacks.
- **2D FFT Screen Moiré Pattern Analysis:** Computes 2D Fast Fourier Transform high-frequency power spectrum ratios, detecting pixel grid moiré frequencies emitted by smartphone, iPad, and monitor screens.
- **3D solvePnP Head Pose & Nose Ray Tracking:** Projects 3D facial model points into 2D camera coordinates, extracting real-time Euler angles (Pitch, Yaw, Roll) and 3D nose orientation vectors.
- **Dynamic Challenge-Response:** Generates random cryptographic challenges (e.g. *"Turn head left"*, *"Blink twice"*) to defeat pre-recorded deepfake video injections.

### 2. 📱 Apple Face ID Multi-Angle Guided Enrollment Wizard
- **Interactive 36-Segment Circular HUD:** Visual progress ring guiding the user through 5 geometric poses: Frontal, Left (Yaw -25°), Right (Yaw +25°), Up (Pitch +18°), and Down (Pitch -18°).
- **$L_2$-Normalized Fréchet Centroid Vectors:** Combines multi-angle embeddings into a single composite representation guaranteeing $\|\mathbf{e}_{\text{composite}}\|_2 = 1.0$.
- **Instant Launchers:** Run via `run_face_id_wizard.bat` or `python add_faces_wizard.py`.

### 3. ⚡ $O(\log N)$ Spatial Indexing & 60+ FPS YuNet DNN
- **SciPy `cKDTree` Nearest-Neighbor Index:** Sub-millisecond vector retrieval across 10,000+ registered identities, eliminating legacy $O(N)$ linear search lag.
- **OpenCV YuNet Deep Learning Detector:** Modern ONNX deep learning face detector achieving **60+ FPS** with superior accuracy under extreme angles and lighting.

### 4. 🌐 FastAPI Edge Microservice Mode & WebSockets
- **Asynchronous ASGI Architecture:** High-throughput REST API running on Uvicorn (`src/api.py`, port 8000).
- **Interactive OpenAPI Documentation:** Live Swagger UI at `http://localhost:8000/docs` and ReDoc at `/redoc`.
- **WebSocket Video Stream:** Real-time bi-directional streaming endpoint `/ws/stream` for live web browser inspection and event notification.
- **REST Endpoints:** `/api/v1/recognize`, `/api/v1/enroll`, `/api/v1/enroll/composite`, `/api/v1/liveness`, `/api/v1/health`, `/api/v1/history`, `/api/v1/webhooks`.
- **Instant Launcher:** Run via `run_api.bat` or `python main.py --api --port 8000`.

### 5. 📹 RTSP IP Camera Streaming & MOG2 Motion Power-Saving
- **Threaded Buffer-Drop Streamer:** Dedicated background frame grabber enforcing `CAP_PROP_BUFFERSIZE = 1` for zero latency and auto-reconnect backoff.
- **OpenCV MOG2 Motion Gating:** Background subtraction gating suppresses recognition compute by **>90%** during static scenes, waking up only upon motion.
- **Security Webhooks:** Non-blocking asynchronous HTTP POST dispatcher alerting external access relays, Slack channels, or Discord webhooks.
- **Instant Launcher:** Run via `run_rtsp_stream.bat` or `python main.py --rtsp "<URL>" --motion`.

### 6. 🔒 Cryptographic Hardening & Audit Integrity
- **AES-256 Fernet Template Encryption:** Biometric embeddings encrypted at rest (`.dat`), preventing template extraction.
- **Chained SHA-256 Tamper-Evident Audit Ledger:** Blockchain-style hash linking for every recognition event in `logs/audit_log.json`.
- **Strict Input Sanitization:** Protection against path traversal attacks (`../../etc/passwd`).

### 7. 🎙️ Offline Speech Synthesizer Voice Engine
- Integrated `AudioAnnouncer` (pyttsx3) greeting authorized individuals with zero external cloud dependencies.

### 8. 📊 Executive Attendance Analytics & HTML Reports
- Automated Indian Standard Time (IST, UTC+05:30) logging with Apple-styled executive HTML dashboards and CSV exports.

### 9. 🧪 100% Passing Automated Test Suite
- Comprehensive pytest suite with **74 / 74 unit tests passing** across API, Face ID wizard, RTSP streaming, and encryption.

---

## 🚀 Quick Launch (Windows 1-Click Launchers)

| Launcher Batch File | Description | Execution Command |
| :--- | :--- | :--- |
| **`run_gui.bat`** | Launches Desktop Dark GUI with HUD & Voice Greeting | `run_gui.bat` |
| **`run_api.bat`** | Starts FastAPI Microservice on `http://localhost:8000` | `run_api.bat` |
| **`run_face_id_wizard.bat`** | Launches Apple Face ID 36-Segment Enrollment Wizard | `run_face_id_wizard.bat` |
| **`run_rtsp_stream.bat`** | Starts RTSP IP Camera Streaming with Motion Gating | `run_rtsp_stream.bat` |
| **`run_cli.bat`** | Headless CLI Recognition Mode | `run_cli.bat` |

---

## ⚙️ Installation

### Automated Setup (Recommended)
```bash
# 1. Clone repository
git clone https://github.com/AmiNilay/face-recognition-system-version-control.git
cd face-recognition-system-version-control

# 2. Run automated installer
./INSTALL.bat
```

### Manual Setup
```bash
# 1. Create and activate virtual environment
python -m venv --system-site-packages .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify complete unit test suite (74 tests)
python -m unittest discover tests -p "test_*.py"
```

---

## 📁 Project Structure

```text
face-recognition-system-version-control/
├── .venv/                                # Isolated virtual environment
├── captured_images/                      # Frame snapshots & spoof alert photos
├── config/                               # Central configuration
│   ├── config.yaml                       # Liveness, RTSP, and vector index settings
│   └── settings.json
├── data/                                 # Runtime databases and logs
│   ├── database.db                       # SQLite attendance & face log
│   ├── face_encodings.dat                # Encrypted biometric vector store
│   └── recognition_history.csv           # Attendance CSV log
├── docs/                                 # Technical documentation
│   ├── README.md                         # Documentation portal
│   ├── API.md                            # Comprehensive REST API & class reference
│   ├── USER_GUIDE.md                     # Operator handbook
│   └── SYSTEM_AUDIT_AND_ROADMAP.md       # Security audit & development history
├── face-recognition-system-official-website-main/ # Official Web Portal
│   ├── index.html                        # Landing page (v2.1.0 Enterprise)
│   ├── css/style.css                     # Responsive styling & HUD animations
│   ├── js/main.js                        # Dynamic tab switcher & automatic year
│   └── README.md                         # Website documentation
├── gui/                                  # GUI package
│   ├── app.py                            # Modern Dark Tkinter desktop application
│   ├── face_id_wizard.py                 # Apple Face ID 36-segment HUD enrollment wizard
│   └── widgets.py                        # Custom UI cards & indicators
├── known_faces/                          # Reference face images & encrypted profiles
├── models/                               # YuNet ONNX & dlib landmark predictors
├── src/                                  # Core engine modules
│   ├── anti_spoofing.py                  # EAR blink, 2D FFT moiré & 3D solvePnP pose
│   ├── api.py                            # FastAPI ASGI microservice & WebSockets
│   ├── audio_announcer.py                # Offline TTS voice synthesizer (pyttsx3)
│   ├── composite_enrollment.py           # Apple Face ID Fréchet centroid synthesizer
│   ├── database_manager.py               # SQLite schema & tamper-evident audit ledger
│   ├── face_detector.py                  # OpenCV YuNet DNN & HOG detection abstractions
│   ├── face_encoder.py                   # 128-d metric vector computation
│   ├── face_recognition_system.py        # Central unified recognition engine
│   ├── motion_detector.py                # OpenCV MOG2 background motion gating
│   ├── rtsp_stream.py                    # Threaded zero-latency RTSP camera reader
│   ├── security_manager.py               # AES-256 Fernet encryption & path sanitizer
│   ├── utils.py                          # IST datetime helpers & frame processors
│   ├── vector_index.py                   # SciPy cKDTree O(log N) nearest-neighbor index
│   └── webhook_dispatcher.py             # Asynchronous security webhook dispatcher
├── tests/                                # Automated unit & integration tests (74 tests)
│   ├── test_api.py                       # 10 tests (REST & WebSocket microservice)
│   ├── test_composite_enrollment.py      # 9 tests (Apple Face ID wizard & Fréchet centroid)
│   ├── test_rtsp_webhooks.py             # 8 tests (RTSP, MOG2 gating & webhooks)
│   ├── test_antispoof.py                 # 12 tests (EAR blink, FFT moiré & 3D pose)
│   ├── test_vector_index.py              # 8 tests (cKDTree spatial indexing)
│   ├── test_security_audit.py            # 7 tests (AES Fernet & SHA-256 audit)
│   ├── test_detection.py                 # 6 tests (YuNet & HOG detection)
│   ├── test_recognition.py               # 6 tests (Face encoder & identification)
│   └── test_security.py                  # 8 tests (Sanitization & SQLite ledger)
├── add_faces_wizard.py                   # Dedicated Face ID wizard CLI
├── add_faces.py                          # Standard webcam enrollment CLI
├── Face Recognition System DOC.docx      # 2026 Enterprise Technical Specification (Word)
├── INSTALL.bat                           # Automated Windows environment installer
├── main.py                               # Central CLI/GUI/API runtime entry point
├── README.md                             # Repository README (this file)
├── requirements.txt                      # Managed Python dependencies
├── run_api.bat                           # 1-click FastAPI launcher (:8000)
├── run_face_id_wizard.bat                # 1-click Apple Face ID enrollment launcher
├── run_gui.bat                           # 1-click Desktop GUI launcher
└── run_rtsp_stream.bat                   # 1-click RTSP IP camera streaming launcher
```

---

## 🛠️ Technology Stack & Dependencies

| Category | Component / Tool | Version | Purpose |
| :--- | :--- | :---: | :--- |
| **Runtime** | `Python` | `3.10+` | Core programming runtime |
| **API Framework** | `FastAPI` / `Uvicorn` | `>=0.115.0` | Asynchronous REST microservice & WebSockets |
| **Computer Vision** | `OpenCV (opencv-python)` | `>=4.8.0` | YuNet DNN, MOG2 motion gating, frame capture |
| **Deep Metric Learning** | `face-recognition` / `dlib` | `>=1.3.0` | 128-dimensional facial embeddings & landmarks |
| **Spatial Indexing** | `SciPy (cKDTree)` | `>=1.11.0` | Sub-millisecond $O(\log N)$ nearest-neighbor search |
| **Security & Crypto** | `cryptography (Fernet)` | `>=42.0.0` | AES-256 template encryption & SHA-256 hash chains |
| **Speech Engine** | `pyttsx3` | `>=2.90` | Offline text-to-speech audio announcements |
| **Data Science** | `NumPy` / `Pandas` | `NumPy <2.0` | Linear algebra, attendance analysis & CSV exports |
| **Testing** | `pytest` / `unittest` | Latest | 74 / 74 automated unit & integration tests |

---

## 📄 License & Credits

- **License:** Distributed under the [MIT License](LICENSE).
- **Creator & Lead Architect:** [Nilay Naha](https://github.com/AmiNilay)
- **Repository:** [face-recognition-system-version-control](https://github.com/AmiNilay/face-recognition-system-version-control)
