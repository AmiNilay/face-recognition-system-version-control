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
```
