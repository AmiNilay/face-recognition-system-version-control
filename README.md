# 🎥 Face Recognition System

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-red.svg)](https://opencv.org/)
![Status](https://img.shields.io/badge/status-active-success.svg)

[![Project Icon](https://i.ibb.co/QtVfbg7/1761922377176.jpg)](https://ibb.co/QtVfbg7)

A comprehensive real-time face recognition system built with Python, featuring advanced detection algorithms, confidence scoring, recognition history tracking, and a modern GUI interface.

![Face Recognition System](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Mac-lightgrey)

---

## ✨ Features

### Core Functionality

- 🎥 **Real-time Face Detection** – Instant detection using HOG algorithm  
- 🔍 **Face Recognition** – 128-dimensional embeddings for accuracy  
- 📊 **Confidence Scoring** – Real-time percentage display (0–100%)  
- 👥 **Multi-face Detection** – Recognizes multiple faces simultaneously  
- 🎨 **Color-coded Feedback**:
  - 🟢 **Green (70–100%)** – High confidence  
  - 🟡 **Orange (50–70%)** – Medium confidence  
  - 🔴 **Red (<50%)** – Low confidence / Unknown  

### Advanced Features

- 💾 **Recognition History** – Auto logging with timestamps  
- 📈 **Database Integration** – SQLite storage  
- 📁 **Multi-format Support** – JPEG, PNG, GIF, TIFF, WebP, BMP  
- 📤 **Export Functionality** – Export logs to CSV  
- ⚡ **Performance Optimized** – Processes every 3rd frame  
- 🔄 **Auto-save** – Configurable cooldown backup  
- 🎯 **Cooldown System** – Prevents duplicate logging  

### User Interface

- 🖥️ **Dual Mode** – GUI + CLI  
- 🌙 **Dark Theme** – Professional dark-mode  
- 📸 **Live Capture** – Add new faces dynamically  
- 👤 **Face Management** – Add/Remove/View faces  
- 📊 **Statistics Dashboard** – Real-time analytics  
- 📅 **History Viewer** – Filtered recognition logs  

---

## 🎬 Demo

### GUI Interface
- Modern dark theme  
- Real-time video feed  
- Live confidence display  
- Easy face management  

### CLI Interface
- Lightweight performance  
- Green = recognized, Red = unknown  
- Keyboard shortcuts for actions  

---

## 🚀 Quick Installation (Windows)

### Easy Install (Recommended)

**1. Download the project:**
```bash
git clone https://github.com/AmiNilay/face-recognition-system-version-control.git
cd face-recognition-system-version-control
````

**2. Run the installer:**

```batch
INSTALL.bat
```

**3. Launch the application:**

```batch
# For GUI mode
run_gui.bat

# For CLI mode
run_cli.bat
```

✅ That’s it!

---

## 💻 Manual Installation

### Prerequisites

* Python 3.10+
* Windows 10/11 (primary), Linux or macOS
* Webcam
* 4GB RAM minimum (8GB recommended)

### Step-by-Step Setup

#### 1. Clone Repository

```bash
git clone https://github.com/AmiNilay/face-recognition-system-version-control.git
cd face-recognition-system-version-control
```

#### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

#### 3. Install Dependencies

```bash
python -m pip install --upgrade pip

pip install opencv-python numpy Pillow pandas PyYAML face-recognition
```

#### 4. Verify Installation

```bash
python test_imports.py
```

**Expected Output:**

```
✅ OpenCV 4.12.0
✅ NumPy 2.2.6
✅ Pillow 12.0.0
✅ Pandas 2.1.3
✅ PyYAML 6.0.1
✅ Face Recognition 1.3.0
✅ All packages installed correctly!
```

---

## 📖 Usage

### Adding Your First Face

#### Method 1: Using Script (Recommended)

```bash
python add_faces.py
```

1. Enter name
2. Position face
3. Press **SPACE** to capture
4. Press **3** to exit

#### Method 2: Manual Addition

* Save a clear photo in `known_faces/PersonName.jpg`
* Restart app

#### Method 3: During Recognition (GUI)

1. Start camera
2. Click “📸 Capture Face”
3. Enter name

---

### Running Face Recognition

#### GUI Mode

```bash
run_gui.bat
# or
python main.py --gui
```

**Controls:**

| Button          | Action            |
| --------------- | ----------------- |
| ▶ Start Camera  | Begin recognition |
| ⏹ Stop Camera   | Stop recognition  |
| 📸 Capture Face | Add new face      |
| ➕ Add           | Add from file     |
| 🗑️ Remove      | Delete face       |
| 🔄 Refresh      | Update list       |

#### CLI Mode

```bash
run_cli.bat
# or
python main.py
```

**Keyboard Shortcuts:**

| Key | Action           |
| --- | ---------------- |
| Q   | Quit             |
| S   | Save screenshot  |
| C   | Capture new face |

---

## 📁 Project Structure

```
face_recognition_project/
├── src/
│   ├── face_recognition_system.py
│   ├── face_detector.py
│   ├── face_encoder.py
│   ├── database_manager.py
│   └── utils.py
│
├── gui/
│   ├── app.py
│   └── widgets.py
│
├── known_faces/
│   └── *.jpg
│
├── data/
│   ├── face_encodings.pkl
│   ├── recognition_history.csv
│   └── database.db
│
├── captured_images/
│   └── capture_*.jpg
│
├── config/
│   ├── config.yaml
│   └── settings.json
│
├── logs/
│   └── app.log
│
├── docs/
│   ├── INSTALLATION.md
│   ├── USER_GUIDE.md
│   └── TROUBLESHOOTING.md
│
├── main.py
├── add_faces.py
├── monitor_confidence.py
├── INSTALL.bat
├── run_gui.bat
├── run_cli.bat
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies

| Technology       | Version  | Purpose            |
| ---------------- | -------- | ------------------ |
| Python           | 3.10+    | Language           |
| OpenCV           | 4.12+    | Vision & video     |
| dlib             | 19.24+   | Recognition engine |
| face_recognition | 1.3.0    | Face API           |
| NumPy            | 2.2+     | Math ops           |
| Pillow           | 12.0+    | Image handling     |
| Pandas           | 2.1+     | Data/CSV           |
| PyYAML           | 6.0+     | Config             |
| Tkinter          | Built-in | GUI                |
| SQLite3          | Built-in | DB                 |

**Algorithms:**

* Face Detection: HOG
* Recognition: 128D embeddings
* Metric: Euclidean distance
* Threshold: 0.6 (adjustable)

---

## 🔧 Configuration

Edit `config/config.yaml`:

```yaml
camera:
  device: 0
  width: 640
  height: 480
  fps: 30

face_recognition:
  tolerance: 0.6
  model: 'hog'
  num_jitters: 1
  process_interval: 3

storage:
  known_faces_dir: 'known_faces'
  captured_images_dir: 'captured_images'
  encodings_file: 'data/face_encodings.pkl'
  database_file: 'data/database.db'
  history_file: 'data/recognition_history.csv'
```

---

## 🐛 Troubleshooting

### 1. `ModuleNotFoundError: No module named 'cv2'`

```bash
pip install opencv-python
```

### 2. dlib installation fails (Windows)

```bash
pip install https://github.com/sachadee/Dlib/raw/main/dlib-19.22.99-cp310-cp310-win_amd64.whl
pip install face-recognition
```

### 3. "Unsupported image type"

* Delete bad images from `known_faces/`
* Re-add faces
* Ensure RGB format

### 4. Camera not opening

```bash
python main.py --video 1
```

Also check:

> Settings → Privacy → Camera → Allow apps

### 5. Low FPS / Lag

* Set `process_interval: 5`
* Use HOG model
* Reduce resolution

### 6. Face not recognized

* Improve lighting
* Face camera directly
* Adjust tolerance (0.65–0.7)

### 7. Virtual environment path issues

```bash
deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Performance

| Component | Minimum           | Recommended        |
| --------- | ----------------- | ------------------ |
| CPU       | Dual-core 2.0 GHz | Quad-core 3.0+ GHz |
| RAM       | 4 GB              | 8 GB               |
| Storage   | 500 MB            | 2 GB               |
| Camera    | 480p              | 720p+              |

**Benchmarks:**

| Config          | FPS   | CPU    | Accuracy |
| --------------- | ----- | ------ | -------- |
| HOG + 3rd frame | 25–30 | 15–25% | 95%+     |
| HOG + 5th frame | 30–35 | 10–15% | 93%+     |
| CNN + 3rd frame | 15–20 | 35–50% | 98%+     |

---

## 📚 Documentation

* [Installation Guide](docs/INSTALLATION.md)
* [User Guide](docs/USER_GUIDE.md)
* [Troubleshooting](docs/TROUBLESHOOTING.md)
* [API Docs](docs/API.md)

---

## 🎯 Feature Highlights

### Recognition History

* Auto logging with confidence
* CSV export
* Date filters
* Summary stats

### Face Management

* Add/remove faces
* Gallery view
* Bulk operations

### Advanced Analytics

* Real-time confidence graphs
* Color-coded indicators
* Average confidence tracking

---

## 🤝 Contributing

1. Fork repo
2. Create branch (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open PR

---

## 📄 License

Licensed under **MIT License** – see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

* **dlib** – Davis King
* **OpenCV** – OpenCV team
* **face_recognition** – Adam Geitgey
* **Python Community** – for amazing libraries

---

## 📞 Contact

**Developer:** Nilay Naha
**GitHub:** [@AmiNilay](https://github.com/AmiNilay)
**Project:** [face-recognition-system-version-control](https://github.com/AmiNilay/face-recognition-system-version-control)

---

## 🌟 Star History

If this project helped you, please give it a ⭐️!

---

### 📝 Changelog

**Version 1.0.0 (2024-10-31)**

* ✅ Initial release
* ✅ Real-time face recognition
* ✅ GUI + CLI
* ✅ Confidence scoring
* ✅ Recognition history
* ✅ SQLite integration
* ✅ Multi-format support
* ✅ Export functionality
* ✅ Batch installers

---

<p align="center"><b>Made with ❤️ using Python</b></p>
<p align="center"><a href="#-face-recognition-system">Back to Top ↑</a></p>
