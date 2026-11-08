# 📖 Face Recognition System: Comprehensive User Guide

Welcome to the **Face Recognition System** user guide. This document explains how to set up, operate, and configure the system in both GUI and CLI modes.

---

## 🎯 Quick Navigation
1. [Prerequisites & Startup](#1-prerequisites--startup)
2. [Operating Modes](#2-operating-modes)
   - [GUI Mode (Recommended)](#gui-mode)
   - [CLI Mode (Fast / Lightweight)](#cli-mode)
3. [Managing Known Faces](#3-managing-known-faces)
4. [Recognition History & Database Logs](#4-recognition-history--database-logs)
5. [Keyboard Shortcuts](#5-keyboard-shortcuts)
6. [Configuration & Tuning](#6-configuration--tuning)

---

## 1. Prerequisites & Startup

* **Camera:** Built-in webcam or USB webcam connected to your machine.
* **Launch Scripts:**
  * For GUI: Double-click `run_gui.bat` or run `python main.py --gui`.
  * For CLI: Double-click `run_cli.bat` or run `python main.py`.

---

## 2. Operating Modes

### GUI Mode
The Graphical User Interface provides a full-featured visual dashboard with real-time feedback:

* **Video Panel (Left):** Displays real-time camera feed with color-coded bounding boxes around detected faces:
  * 🟢 **Green (70%–100%):** High confidence match.
  * 🟡 **Orange (50%–70%):** Medium confidence match.
  * 🔴 **Red (<50%):** Unknown face or low confidence match.
* **Control Buttons (Bottom Left):**
  * **▶ Start Camera:** Activates the video stream.
  * **⏹ Stop Camera:** Suspends capture.
  * **📸 Capture Face:** Takes a screenshot of the detected face to enroll a new identity.
* **Detection Info Panel (Top Right):** Displays live telemetry: number of faces detected, person names, and confidence scores.
* **Known Faces Listbox (Bottom Right):** Displays all registered persons with **➕ Add**, **🗑️ Remove**, and **🔄 Refresh** actions.

### CLI Mode
Lightweight console interface ideal for low-power hardware or headless environments:
* Runs minimal OpenCV windows.
* Streams FPS and recognition stats directly to terminal and HUD.

---

## 3. Managing Known Faces

### Method A: Live Camera Enrollment (GUI)
1. Click **▶ Start Camera**.
2. Center your face in the feed.
3. Click **📸 Capture Face**.
4. Enter the person's name in the popup dialog and press OK.

### Method B: Live Camera Enrollment (CLI)
```bash
python add_faces.py
```
* Enter person's name.
* Position face in box.
* Press **SPACE** to save.

### Method C: Manual Image File Drop
1. Copy a clear frontal photo into the `known_faces/` folder.
2. Name the file with the person's name (e.g. `Elon_Musk.jpg`).
3. Click **🔄 Refresh** in the GUI or restart the app.

---

## 4. Recognition History & Database Logs

The application automatically records recognition events in SQLite (`data/database.db`) and CSV format:

* **Cooldown Deduplication:** Prevents logging the same person multiple times within a 5-second window.
* **Exporting Records:**
  * In the GUI, navigate to **File ➔ Export Recognition History** to save a `.csv` spreadsheet.
* **View History:**
  * Select **View ➔ Recognition History** to review today's timestamps, confidence scores, and IDs.

---

## 5. Keyboard Shortcuts

| Shortcut | Context | Action |
| :---: | :---: | :--- |
| `Q` | CLI Window | Quit application |
| `S` | CLI Window | Save instant screenshot to `captured_images/` |
| `C` | CLI Window | Enroll new face live |
| `SPACE` | `add_faces.py` | Capture reference photo |
| `ESC` | `add_faces.py` | Cancel enrollment |

---

## 6. Configuration & Tuning

Edit `config/config.yaml` to tweak system behavior:

```yaml
camera:
  device: 0          # Switch between cameras (0, 1, 2)
  width: 640
  height: 480
  fps: 30

face_recognition:
  tolerance: 0.6     # 0.5 = stricter match, 0.7 = looser match
  model: 'hog'       # 'hog' (fast CPU) or 'cnn' (GPU)
  process_interval: 3 # Process 1 out of every 3 frames to save CPU

anti_spoofing:
  enabled: true                 # Master toggle for biometric liveness verification
  ear_threshold: 0.22           # Eye Aspect Ratio threshold for blink detection
  consecutive_closed_frames: 2  # Minimum frames below threshold to count as blink
  texture_analysis: true        # Screen moire / digital screen replay detection
  privacy_blur_bystanders: false # Automatic Gaussian blur on unknown / bystander faces
  head_pose: true               # 3D Head pose estimation (solvePnP Euler angles & nose ray)
  challenge_response: false     # Interactive challenge-response mode (turn left/right, nod down)
  challenge_timeout: 5.0        # Timeout in seconds to complete dynamic liveness challenges
```

---

## 7. Advanced Biometric Anti-Spoofing & Liveness

The system features multi-layered anti-spoofing defense against presentation attacks:

1. **Eye Aspect Ratio (EAR) Blink Verification:** Defeats static printed photos by requiring natural eyelid closure cycles.
2. **2D FFT Screen Moiré Analysis:** Defeats digital screen replays (smartphones, iPads, laptops) by detecting sub-pixel LCD/OLED high-frequency lattice patterns.
3. **3D Head Pose Estimation (solvePnP):** Estimates 3D Euler angles (Yaw, Pitch, Roll) by solving the Perspective-n-Point problem against an anthropometric skull model and projects a forward 3D direction ray from the nose tip.
4. **Interactive Challenge-Response Mode:** Issues dynamic motion challenges (e.g. `TURN HEAD LEFT`, `TURN HEAD RIGHT`, `NOD HEAD DOWN`) within a countdown window to defeat deepfake video replay loops.
5. **GDPR Privacy Mode:** Automatically applies 51x51 Gaussian blur to unverified bystanders walking behind enrolled subjects.

---

## 8. Offline Voice Announcements & Audio Feedback

The system includes offline text-to-speech audio feedback (`src/audio_announcer.py`):
* **Greeting on Detection:** Automatically greets recognized personnel (e.g. *"Welcome, Nilay Naha"*).
* **Cooldown Deduplication:** Configurable speech cooldown (default: 15s) prevents repetitive audio loops.
* **Non-Blocking Architecture:** Speech is synthesized in a dedicated background worker thread, ensuring the camera stream maintains full 30 FPS.
* **Toggle:** Can be enabled or disabled via **Settings ➔ Voice Announcements (Text-to-Speech)** in the GUI menu or in `config.yaml`.

---

## 9. Executive Attendance Reporting (Apple Design HTML & CSV)

Generate presentation-grade attendance analytics directly from the GUI or CLI:
* **HTML Executive Report:**
  * Navigate to **File ➔ Generate Executive Attendance Report (HTML)** in the GUI.
  * Creates an Apple-styled dashboard (`data/attendance_report.html`) complete with statistical cards (Present Count, Attendance Rate %, Mean Confidence, Total Sightings), clean attendee table with first/last seen times in IST, and absent personnel highlights.
  * Fully print-ready (`@media print` supported).
* **CSV Export:**
  * Navigate to **File ➔ Export Recognition History (CSV)** to export raw or summarized attendance spreadsheets.

---

## 10. Edge API Microservice Mode (FastAPI & WebSockets)

The Face Recognition System can run as a headless, network-accessible Edge Microservice for smart security cameras, turnstiles, and cloud dashboards:

### Starting the Server
```bash
# Start microservice with default port 8000
python main.py --api --port 8000

# Or launch via the one-click Windows batch script
run_api.bat
```

### Key Capabilities
* **Interactive Documentation:**
  * Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
  * ReDoc UI: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* **REST Biometric Verification:** Send base64 or multipart images to `/api/v1/verify` to receive bounding boxes, match confidence, Euler angles, and anti-spoofing verdicts.
* **Remote Facial Enrollment:** Send portrait photos to `/api/v1/enroll` to remotely register employees. All biometric vectors are encrypted on disk with Fernet AES keys.
* **Live Attendance Queries:** Query `/api/v1/attendance` in IST to receive real-time presence rates and attendee/absentee breakdowns in JSON, CSV, or HTML.
* **Real-time WebSockets:** Connect to `/ws/stream` to stream high-frequency video frames and receive sub-20ms telemetry responses.

---

## 11. Apple Face ID Multi-Angle Enrollment Wizard

The system supports **multi-angle composite facial enrollment** modeled after Apple Face ID. Instead of enrolling from a single photo, the wizard guides the subject through 5 canonical head orientations to synthesize an $L_2$-normalized composite embedding:

### 5-Step Guided Poses
1. **Frontal:** Look directly at the sensor (Yaw: $[-7^\circ, +7^\circ]$, Pitch: $[-7^\circ, +7^\circ]$).
2. **Turn Left:** Slowly rotate head left (Yaw: $\le -12^\circ$).
3. **Turn Right:** Slowly rotate head right (Yaw: $\ge +12^\circ$).
4. **Tilt Up:** Slowly look upward (Pitch: $\le -8^\circ$).
5. **Nod Down:** Slowly look downward (Pitch: $\ge +8^\circ$).

### Three Ways to Run Face ID Enrollment

#### 1. From Desktop GUI
* Click the purple **Face ID Wizard** button in the camera controls panel, or select **File ➔ Face ID Multi-Angle Enrollment Wizard** from the menu.
* Enter the employee's name and position your face within the segmented circular guide.
* The 36 radial tick marks light up Apple Mint Green as each angle is successfully scanned.

#### 2. From Dedicated CLI Launcher
```bash
# Launch interactive console wizard
python add_faces_wizard.py --name "Nilay Naha"

# Or double-click the Windows batch launcher
run_face_id_wizard.bat
```

#### 3. Remotely via REST API
* Submit multiple angle captures to `POST /api/v1/enroll/composite` with image Base64 payloads.
* The server verifies each perspective, computes the $L_2$-normalized Fréchet centroid embedding, and indexes it into the spatial KD-tree.




