# 🎥 Face Recognition System (2026 Edition)

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-red.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/status-active%20%7C%202026%20modernized-success.svg)](#)
[![Security Audit](https://img.shields.io/badge/security-audit%20v2.0-orange.svg)](docs/SYSTEM_AUDIT_AND_ROADMAP.md)

A comprehensive, real-time face recognition and attendance management system built with Python, OpenCV, and deep metric learning. Features real-time detection, confidence scoring, tamper-evident audit logs, multi-format image support, a modern dark GUI, and a comprehensive 2026-era security roadmap.

---

## ✨ Features

### Core Capabilities
- 🎥 **Real-time Face Detection** – Instant, low-latency detection using HOG and Haar Cascade algorithms.
- 🔍 **Deep Metric Recognition** – 128-dimensional biometric embeddings using pretrained deep ResNet architectures.
- 📊 **Dynamic Confidence Scoring** – Real-time match percentage calculation (0–100%) with color-coded feedback:
  - 🟢 **Green (70–100%)** – High confidence match.
  - 🟡 **Orange (50–70%)** – Moderate confidence match.
  - 🔴 **Red (<50%)** – Low confidence / Unknown face.
- 👥 **Multi-face Tracking** – Detects and identifies multiple faces simultaneously in high-resolution video streams.
- 📁 **Universal Image Ingestion** – Supports JPEG, PNG, WebP, BMP, GIF, and TIFF with automated channel harmonization.

### Advanced Logging & Management
- 💾 **Recognition History** – Real-time event logging with microsecond timestamps and cooldown deduplication.
- 📈 **SQLite Database Integration** – Structured relational storage for identities, attendance logs, and unknown sightings.
- 📤 **Data Export** – 1-click export of attendance and recognition history to standard CSV/Excel format.
- ⚡ **Optimized Video Pipeline** – Frame-skipping buffer management to conserve CPU cycles (processing every 3rd frame).

### 2026 Security Roadmap & Modernization
- 🛡️ **Full Security Audit:** Comprehensive vulnerability catalog (CWE-502, presentation attack/anti-spoofing, biometric privacy) detailed in [docs/SYSTEM_AUDIT_AND_ROADMAP.md](docs/SYSTEM_AUDIT_AND_ROADMAP.md).
- 🔒 **Zero-Cost Modernization:** Eye Aspect Ratio (EAR) blink detection, screen moiré detection, bystander privacy auto-blur, and AES-256 encrypted biometric template storage.

---

## 🚀 Quick Installation (Windows)

### Option A: Easy Automated Install (Recommended)

1. Clone or download the repository:
   ```bash
   git clone https://github.com/AmiNilay/face-recognition-system-version-control.git
   cd face-recognition-system-version-control
   ```

2. Run the automated installer:
   ```cmd
   INSTALL.bat
   ```
   *(Creates a Python 3.10 virtual environment configured with system site-packages and installs all required dependencies from `requirements.txt`).*

3. Launch the application:
   ```cmd
   # For Modern GUI Mode
   run_gui.bat

   # For Fast CLI Mode
   run_cli.bat
   ```

---

### Option B: Manual Installation

#### Prerequisites
* **Python 3.10+** (Python 3.10 64-bit recommended on Windows)
* Webcam or USB video capture device
* 4 GB RAM minimum (8 GB recommended)

#### 1. Setup Virtual Environment
```bash
# Recommended on Windows to inherit pre-compiled dlib wheels:
python -m venv --system-site-packages .venv

# Activate the virtual environment
# On Windows Command Prompt / PowerShell:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

#### 2. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Important Compatibility Note on NumPy:**  
> `requirements.txt` locks `numpy>=1.24.3,<2.0.0` (such as NumPy 1.26.4). NumPy 2.x introduces binary C-API breaking changes with pre-compiled `dlib` and `face-recognition` C-extensions.

---

## 📖 Usage & Operating Modes

### 1. Adding Known Faces

Before running recognition, register at least one known person:

* **Method 1 (GUI):** Run `run_gui.bat`, click **▶ Start Camera**, then click **📸 Capture Face**.
* **Method 2 (CLI Script):** Run `python add_faces.py`, enter the person's name, position face in frame, and press **SPACE**.
* **Method 3 (Manual File Drop):** Drop a clear frontal photo into the [`known_faces/`](known_faces/) directory (e.g., `known_faces/John_Doe.jpg`).

---

### 2. Running Face Recognition

#### GUI Mode
```bash
run_gui.bat
# or
python main.py --gui
```
* **Start / Stop Camera:** Toggle camera capture pipeline.
* **Add / Remove / Refresh:** Manage registered individuals directly from the GUI listbox.
* **View History / Export Logs:** Access historical sightings and export logs to CSV.

#### CLI Mode
```bash
run_cli.bat
# or
python main.py --video 0
```
* Press `Q` to exit.
* Press `S` to save an instant frame screenshot to `captured_images/`.
* Press `C` to register a new face live from the webcam.

---

## 📁 Project Structure

```
face-recognition-system-version-control/
├── .venv/                      # Isolated virtual environment
├── captured_images/            # Captured screenshots & alerts
│   └── README.md
├── config/                     # Configuration files
│   ├── config.yaml
│   └── settings.json
├── data/                       # Local runtime data (generated)
│   ├── database.db             # SQLite attendance & face log
│   ├── face_encodings.pkl      # Stored biometric templates
│   └── recognition_history.csv # CSV export records
├── docs/                       # Technical documentation
│   ├── README.md               # Documentation portal
│   ├── SYSTEM_AUDIT_AND_ROADMAP.md # Security audit & 2026 roadmap
│   ├── USER_GUIDE.md           # End-user handbook
│   └── API.md                  # Developer class reference
├── gui/                        # Modern GUI package
│   ├── __init__.py
│   ├── app.py                  # Main Tkinter application
│   └── widgets.py              # Custom UI components & cards
├── known_faces/                # Reference images for recognition
│   └── README.md
├── models/                     # Deep learning & landmark models
│   └── README.md
├── src/                        # Core algorithmic modules
│   ├── __init__.py
│   ├── database_manager.py     # SQLite schema & log transactions
│   ├── face_detector.py        # HOG & Haar detection abstractions
│   ├── face_encoder.py         # 128-d metric vector computation
│   ├── face_recognition_system.py # Unified recognition engine
│   └── utils.py                # Logging, FPS counters & validators
├── tests/                      # Unit & integration test suites
│   ├── test_detection.py
│   └── test_recognition.py
├── add_faces.py                # Standalone enrollment script
├── build_exe.bat               # PyInstaller executable builder
├── fix_images.py               # Image validation utility
├── INSTALL.bat                 # Automated Windows setup script
├── main.py                     # Primary runtime entry point
├── README.md                   # Project overview & guide
├── requirements.txt            # Managed Python dependencies
├── run_cli.bat                 # 1-click CLI launcher
└── run_gui.bat                 # 1-click GUI launcher
```

---

## 🛠️ Technology Stack & Dependencies

| Library / Tool | Version | Purpose |
| :--- | :---: | :--- |
| **Python** | `3.10+` | Core programming runtime |
| **OpenCV (`opencv-python`)** | `>=4.8.0` | Computer vision, frame capture, and image transformations |
| **`face-recognition`** | `>=1.3.0` | Deep metric facial landmarking and 128-d embedding |
| **`dlib`** | `>=19.22.99` | High-performance C++ machine learning backend |
| **NumPy** | `>=1.24.3,<2.0.0` | High-performance numerical linear algebra |
| **Pillow (PIL)** | `>=10.0.0` | Robust image parsing and color conversion |
| **Pandas** | `>=2.0.3` | Tabular data analysis and CSV log exports |
| **PyYAML** | `>=6.0.1` | YAML configuration file parser |
| **`cryptography`** | `>=42.0.0` | Biometric template encryption at rest |
| **`sqlite3`** | *Standard Library* | Embedded relational logging database |
| **`tkinter`** | *Standard Library* | Desktop user interface framework |

---

## 🐛 Troubleshooting Common Issues

### 1. `ModuleNotFoundError: No module named 'numpy._core._multiarray_umath'`
* **Cause:** Incompatible NumPy 2.x C-API with pre-compiled `dlib` or `cv2` binary wheels.
* **Fix:** Reinstall compatible NumPy: `pip install "numpy>=1.26.0,<2.0.0"`.

### 2. `pip install dlib` fails on Windows
* **Cause:** Missing CMake and Microsoft Visual Studio C++ build tools.
* **Fix:** Use pre-compiled wheels for Python 3.10:
  ```cmd
  pip install https://github.com/sachadee/Dlib/raw/main/dlib-19.22.99-cp310-cp310-win_amd64.whl
  pip install face-recognition
  ```
  Or create your virtual environment with `--system-site-packages` if installed globally.

### 3. Camera fails to open (`Could not open video source`)
* Verify your webcam is not locked by Zoom, Teams, or another background app.
* Try changing the video index in `main.py` or `config/config.yaml`:
  ```cmd
  python main.py --video 1
  ```
* On Windows 10/11, check **Settings → Privacy & Security → Camera** and ensure *"Let desktop apps access your camera"* is enabled.

---

## 📄 License & Credits

* Distributed under the **MIT License**.
* Built using the `face_recognition` library by Adam Geitgey, `dlib` by Davis King, and the OpenCV open-source computer vision team.

---

## 📝 Changelog

### Version 2.0.0 (2026-11-08)
* ⚡ **Dependency Modernization:** Updated `requirements.txt` to latest compatible versions; removed invalid standard library modules (`sqlite3`, `tkinter`).
* 🔒 **NumPy Stability Pin:** Locked `numpy>=1.24.3,<2.0.0` to eliminate fatal C-API ABI mismatch.
* 🛡️ **Security Audit & 2026 Roadmap:** Published comprehensive flaw analysis, anti-spoofing proposals, and mitigation plan in [`docs/SYSTEM_AUDIT_AND_ROADMAP.md`](docs/SYSTEM_AUDIT_AND_ROADMAP.md).
* 📁 **Documentation Complete:** Fully populated README portals across [`docs/`](docs/), [`known_faces/`](known_faces/), [`captured_images/`](captured_images/), and [`models/`](models/).
* 🚀 **Installer Overhaul:** Updated `INSTALL.bat` to safely handle system site-packages and dependency installation without compilation crashes.

---
<p align="center"><b>Maintained by Nilay & Antigravity AI Engineering</b></p>
