"""
Graphical User Interface (GUI) Application Module

This module implements the primary desktop user interface for the Face Recognition System.
Built with Tkinter, it provides:
1. Real-time video rendering with thread-safe UI scheduling.
2. Live face detection telemetry and confidence scoring indicators.
3. Enrolled identity management (live camera enrollment, image import, deletion).
4. Recognition history viewer and database log viewer.
5. Exporting attendance logs to CSV with Indian Standard Time (IST) timestamps.
6. Clean, professional dark theme styling without emojis.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import csv
import logging
import numpy as np
from typing import Optional, List, Dict, Any

# Ensure parent directory is in Python module search path
sys.path.append(str(Path(__file__).parent.parent))

from src.face_recognition_system import FaceRecognitionSystem
from src.database_manager import DatabaseManager
from src.utils import (
    load_config, 
    setup_logging, 
    get_ist_now, 
    format_ist_timestamp, 
    get_ist_date_string, 
    get_ist_filename_timestamp
)

class FaceRecognitionGUI:
    """
    Main Tkinter Desktop Dashboard for Face Recognition & History Tracking.
    """

    def __init__(self):
        """Initialize the GUI application window and system dependencies."""
        self.root = tk.Tk()
        self.root.title("Face Recognition System - Professional Edition")
        self.root.geometry("1200x720")
        self.root.minsize(1000, 650)
        self.root.configure(bg="#2b2b2b")

        # Initialize logging with IST timestamps
        setup_logging()
        self.logger = logging.getLogger("FaceRecognitionSystem")

        # Core operational components
        self.system = FaceRecognitionSystem()
        self.db_manager = DatabaseManager()
        self.config = load_config()

        # Camera streaming state & thread-safe synchronization
        self.is_running: bool = False
        self.video_thread: Optional[threading.Thread] = None
        self.frame_lock: threading.Lock = threading.Lock()
        self.latest_display_frame: Optional[np.ndarray] = None
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_telemetry: Dict[str, Any] = {}
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        # Recognition tracking & cooldown management
        self.recognition_history: List[Dict[str, str]] = []
        self.last_recognized: Dict[str, datetime] = {}
        # Cooldown prevents repeatedly logging the same person within N seconds
        self.recognition_cooldown: int = int(
            self.config.get('ui', {}).get('cooldown_seconds', 5)
        )

        # Local history file persistence
        self.auto_save_enabled: bool = True
        self.history_file: str = self.config.get('storage', {}).get(
            'history_file', 'data/recognition_history.csv'
        )

        # Setup user interface layout and menus
        self.setup_gui()

        # Load enrolled identities from disk
        self.load_faces()

        # Load previous recognition history records
        self.load_history()

    # --------------------------------------------------------------------------
    # 1. GUI LAYOUT SETUP
    # --------------------------------------------------------------------------
    def setup_gui(self):
        """Construct the visual window layout and control panels."""
        self.create_menu()

        # Header Title Bar
        title_frame = tk.Frame(self.root, bg="#1e1e1e", height=60)
        title_frame.pack(fill="x", padx=10, pady=(10, 5))
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="FACE RECOGNITION SYSTEM",
            font=("Arial", 20, "bold"),
            bg="#1e1e1e",
            fg="#00ff00"
        )
        title_label.pack(side="left", padx=20, pady=10)

        # IST Live Clock Subtitle
        self.clock_label = tk.Label(
            title_frame,
            text=f"IST: {format_ist_timestamp()}",
            font=("Arial", 11),
            bg="#1e1e1e",
            fg="#aaaaaa"
        )
        self.clock_label.pack(side="right", padx=20, pady=10)
        self._update_clock()

        # Main Body Container
        main_container = tk.Frame(self.root, bg="#2b2b2b")
        main_container.pack(fill="both", expand=True, padx=10, pady=5)

        # Left Panel: Video Feed & Controls
        left_panel = tk.Frame(main_container, bg="#1e1e1e")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Video Canvas
        video_frame = tk.Frame(left_panel, bg="#000000", relief="sunken", bd=2)
        video_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.video_label = tk.Label(
            video_frame,
            bg="#000000",
            text="Camera Inactive\nClick 'Start Camera' below",
            font=("Arial", 14),
            fg="#666666"
        )
        self.video_label.pack(fill="both", expand=True)

        # Camera Control Action Buttons
        controls_frame = tk.Frame(left_panel, bg="#1e1e1e")
        controls_frame.pack(pady=(0, 10))

        self.start_btn = tk.Button(
            controls_frame,
            text="Start Camera",
            command=self.start_camera,
            bg="#00aa00",
            fg="white",
            font=("Arial", 11, "bold"),
            width=14,
            height=2,
            cursor="hand2",
            relief="raised"
        )
        self.start_btn.grid(row=0, column=0, padx=6)

        self.stop_btn = tk.Button(
            controls_frame,
            text="Stop Camera",
            command=self.stop_camera,
            bg="#aa0000",
            fg="white",
            font=("Arial", 11, "bold"),
            width=14,
            height=2,
            state="disabled",
            cursor="hand2",
            relief="raised"
        )
        self.stop_btn.grid(row=0, column=1, padx=6)

        self.capture_btn = tk.Button(
            controls_frame,
            text="Capture Face",
            command=self.capture_face,
            bg="#0066cc",
            fg="white",
            font=("Arial", 11, "bold"),
            width=14,
            height=2,
            state="disabled",
            cursor="hand2",
            relief="raised"
        )
        self.capture_btn.grid(row=0, column=2, padx=6)

        # Right Panel: Telemetry & Face Management
        right_panel = tk.Frame(main_container, bg="#1e1e1e", width=380)
        right_panel.pack(side="right", fill="both", padx=(5, 0))
        right_panel.pack_propagate(False)

        # Live Detection Telemetry Box
        info_frame = tk.LabelFrame(
            right_panel,
            text="Detection Telemetry",
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Arial", 11, "bold"),
            padx=10,
            pady=8
        )
        info_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.info_text = tk.Text(
            info_frame,
            height=8,
            bg="#2b2b2b",
            fg="#00ff00",
            font=("Consolas", 10),
            wrap="word",
            state="disabled",
            relief="flat",
            padx=5,
            pady=5
        )
        self.info_text.pack(fill="both", expand=True)

        # Enrolled Faces List Panel
        faces_frame = tk.LabelFrame(
            right_panel,
            text="Enrolled Identities",
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Arial", 11, "bold"),
            padx=10,
            pady=8
        )
        faces_frame.pack(fill="both", expand=True, padx=10, pady=5)

        list_container = tk.Frame(faces_frame, bg="#1e1e1e")
        list_container.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")

        self.faces_listbox = tk.Listbox(
            list_container,
            yscrollcommand=scrollbar.set,
            bg="#2b2b2b",
            fg="white",
            font=("Arial", 10),
            selectmode="single",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#00aa00"
        )
        self.faces_listbox.pack(fill="both", expand=True, side="left")
        scrollbar.config(command=self.faces_listbox.yview)

        # Face Management Buttons
        btn_frame = tk.Frame(faces_frame, bg="#1e1e1e")
        btn_frame.pack(pady=8)

        tk.Button(
            btn_frame,
            text="Add Face",
            command=self.add_face_dialog,
            bg="#00aa00",
            fg="white",
            font=("Arial", 9, "bold"),
            width=10,
            cursor="hand2"
        ).grid(row=0, column=0, padx=3)

        tk.Button(
            btn_frame,
            text="Remove Face",
            command=self.remove_face,
            bg="#aa0000",
            fg="white",
            font=("Arial", 9, "bold"),
            width=11,
            cursor="hand2"
        ).grid(row=0, column=1, padx=3)

        tk.Button(
            btn_frame,
            text="Refresh",
            command=self.refresh_faces,
            bg="#0066cc",
            fg="white",
            font=("Arial", 9, "bold"),
            width=10,
            cursor="hand2"
        ).grid(row=0, column=2, padx=3)

        # Bottom Status Bar
        self.status_bar = tk.Label(
            self.root,
            text=" Ready",
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Arial", 9),
            anchor="w",
            relief="sunken",
            padx=5,
            pady=2
        )
        self.status_bar.pack(side="bottom", fill="x")

    def _update_clock(self):
        """Update top-bar clock with current IST time every second."""
        if hasattr(self, 'clock_label') and self.root.winfo_exists():
            self.clock_label.config(text=f"IST: {format_ist_timestamp()}")
            self.root.after(1000, self._update_clock)

    # --------------------------------------------------------------------------
    # 2. MENU BAR SETUP
    # --------------------------------------------------------------------------
    def create_menu(self):
        """Configure the top window application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Enroll Face from Image File", command=self.add_face_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export Recognition History (CSV)", command=self.export_recognition_history)
        file_menu.add_command(label="Export Database Attendance Logs", command=self.export_logs)
        file_menu.add_separator()
        file_menu.add_command(label="Clear Recognition History", command=self.clear_history)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Recognition History", command=self.show_history)
        view_menu.add_command(label="Database History", command=self.show_db_history)
        view_menu.add_command(label="Enrolled Faces Gallery", command=self.show_gallery)
        view_menu.add_separator()
        view_menu.add_command(label="Today's Recognitions (IST)", command=self.show_today_history)

        # Settings Menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        self.auto_save_var = tk.BooleanVar(value=True)
        settings_menu.add_checkbutton(
            label="Auto-save History to Disk",
            variable=self.auto_save_var,
            command=self.toggle_auto_save
        )
        anti_spoof_enabled = bool(getattr(getattr(self.system, 'anti_spoof', None), 'enabled', True))
        self.anti_spoof_var = tk.BooleanVar(value=anti_spoof_enabled)
        settings_menu.add_checkbutton(
            label="Anti-Spoofing (Liveness Verification)",
            variable=self.anti_spoof_var,
            command=self.toggle_anti_spoofing
        )
        privacy_blur_enabled = bool(getattr(getattr(self.system, 'anti_spoof', None), 'privacy_blur_enabled', False))
        self.privacy_blur_var = tk.BooleanVar(value=privacy_blur_enabled)
        settings_menu.add_checkbutton(
            label="Privacy Mode (Blur Unknown Bystanders)",
            variable=self.privacy_blur_var,
            command=self.toggle_privacy_blur
        )
        settings_menu.add_separator()
        settings_menu.add_command(label="Set Recognition Cooldown...", command=self.set_cooldown)

        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About System", command=self.show_about)

    # --------------------------------------------------------------------------
    # 3. CAMERA & THREAD-SAFE VIDEO PIPELINE
    # --------------------------------------------------------------------------
    def _open_camera(self, device_index: int) -> Optional[cv2.VideoCapture]:
        """
        Helper method to open a video capture device reliably across platforms.
        On Windows, attempts cv2.CAP_DSHOW first to avoid MSMF threading deadlocks.
        """
        if sys.platform.startswith("win"):
            try:
                cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    return cap
                cap.release()
            except Exception as dshow_err:
                self.logger.debug(f"DirectShow open attempt: {dshow_err}")

        try:
            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception as cap_err:
            self.logger.error(f"Error opening camera with default backend: {cap_err}")

        return None

    def start_camera(self):
        """Initiate video capture in a dedicated background worker thread."""
        if self.is_running:
            return

        self.is_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.capture_btn.config(state="normal")
        self.update_status("Connecting to camera device...")

        # Spawn video capture loop in background daemon thread
        # Crucial Rule: cv2.VideoCapture is opened, read, and released entirely inside the worker thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

        # Start thread-safe UI update polling loop on the main Tkinter thread
        self.root.after(30, self._render_ui_loop)

    def stop_camera(self):
        """Gracefully signal camera worker thread to stop and restore UI controls."""
        self.is_running = False

        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.capture_btn.config(state="disabled")

        self.video_label.config(image="", text="Camera Inactive\nClick 'Start Camera' below", fg="#666666")
        self.current_photo = None
        with self.frame_lock:
            self.latest_display_frame = None
            self.latest_raw_frame = None

        self.update_status("Camera stream stopped")
        self.logger.info("Camera stopped")

    def _handle_camera_error(self, err_msg: str):
        """Notify user and reset camera controls when camera encounters an unrecoverable error."""
        self.stop_camera()
        messagebox.showerror("Camera Error", err_msg)
        self.update_status(f"Camera Error: {err_msg}")

    def video_loop(self):
        """
        Background worker thread executing video decoding and face recognition.

        Threading Rule:
        1. All OpenCV hardware access (VideoCapture.open, read, release) stays in this thread.
        2. Zero Tkinter/ImageTk widgets or objects are accessed from this thread.
        3. Decoded RGB frames and telemetry dictionaries are placed into a thread-safe buffer
           protected by self.frame_lock for consumption by the main thread.
        """
        import face_recognition

        device_index = int(self.config.get('camera', {}).get('device', 0))
        cap = self._open_camera(device_index)

        if cap is None or not cap.isOpened():
            self.logger.error(f"Could not open camera device {device_index}")
            self.root.after(
                0, 
                self._handle_camera_error, 
                f"Could not access camera device {device_index}!\nPlease verify your camera is connected and not in use."
            )
            return

        # Configure camera capture properties
        try:
            cam_w = int(self.config.get('camera', {}).get('width', 640))
            cam_h = int(self.config.get('camera', {}).get('height', 480))
            cam_fps = int(self.config.get('camera', {}).get('fps', 30))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
            cap.set(cv2.CAP_PROP_FPS, cam_fps)
        except Exception as prop_err:
            self.logger.warning(f"Could not set camera properties: {prop_err}")

        self.logger.info("Camera started")
        self.root.after(0, lambda: self.update_status("Camera stream active"))

        frame_count = 0
        consecutive_read_failures = 0
        process_interval = int(self.config.get('face_recognition', {}).get('process_interval', 3))

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_read_failures += 1
                    if consecutive_read_failures > 15:
                        self.logger.error("Camera read stream lost or disconnected")
                        self.root.after(
                            0, 
                            self._handle_camera_error, 
                            "Camera stream disconnected or stopped responding."
                        )
                        break
                    time.sleep(0.02)
                    continue

                consecutive_read_failures = 0
                frame_count += 1

                if frame_count == 1 or frame_count % 30 == 0:
                    self.logger.debug(f"Camera frame #{frame_count} captured successfully: shape={frame.shape}")

                # Keep untouched raw copy for facial enrollment capture
                raw_frame = frame.copy()

                # Process face recognition every Nth frame to optimize CPU load
                if frame_count % process_interval == 0:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    locations = face_recognition.face_locations(
                        rgb_small_frame,
                        number_of_times_to_upsample=1,
                        model="hog"
                    )
                    self.system.face_locations = locations

                    if locations:
                        self.logger.debug(f"Frame #{frame_count}: Detected {len(locations)} face location(s)")
                        encodings = face_recognition.face_encodings(
                            rgb_small_frame,
                            locations,
                            num_jitters=1
                        )
                        # Extract 68 facial landmarks for Eye Aspect Ratio & blink verification
                        landmarks_list = face_recognition.face_landmarks(rgb_small_frame, locations)

                        self.system.face_encodings = encodings
                        self.system.face_names = []
                        self.system.face_confidences = []
                        self.system.face_liveness = []

                        h_f, w_f = frame.shape[:2]

                        for i, enc in enumerate(encodings):
                            name, conf = self.system.recognize_face(enc)
                            self.system.face_names.append(name)
                            self.system.face_confidences.append(conf)

                            # Crop face region from full frame for screen moire texture analysis
                            top_s, right_s, bottom_s, left_s = locations[i]
                            y1, y2 = max(0, top_s * 4), min(h_f, bottom_s * 4)
                            x1, x2 = max(0, left_s * 4), min(w_f, right_s * 4)
                            face_crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else None

                            lms = landmarks_list[i] if i < len(landmarks_list) else None
                            liveness_res = self.system.anti_spoof.verify_liveness(name, face_crop, lms)
                            self.system.face_liveness.append(liveness_res)

                            self.logger.debug(
                                f"Subject '{name}': {conf:.1f}% match | Liveness: {liveness_res.status_text}"
                            )

                            # Anti-Spoofing Security Gate: Only log attendance when confirmed LIVE
                            if name != "Unknown" and conf >= 50.0 and liveness_res.is_live:
                                self.add_recognition_to_history(name, conf)
                                try:
                                    self.db_manager.log_recognition(name, conf / 100.0)
                                except Exception as log_err:
                                    self.logger.warning(f"Database logging warning: {log_err}")
                    else:
                        self.system.face_encodings = []
                        self.system.face_names = []
                        self.system.face_confidences = []
                        self.system.face_liveness = []

                # Render overlays onto display frame copy
                display_frame = frame.copy()
                self.system.draw_results(display_frame)

                # Prepare frame for Tkinter rendering (BGR -> RGB)
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                # Package safe snapshot for main UI thread
                with self.frame_lock:
                    self.latest_display_frame = frame_rgb
                    self.latest_raw_frame = raw_frame
                    self.latest_telemetry = {
                        'names': list(self.system.face_names),
                        'locations': list(self.system.face_locations),
                        'confidences': list(self.system.face_confidences),
                        'liveness': [r.status_text for r in self.system.face_liveness]
                    }

                # Yield CPU briefly to prevent thread starvation
                time.sleep(0.01)

        except Exception as loop_err:
            self.logger.error(f"Error in video loop: {loop_err}")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            self.logger.info("Camera capture hardware released")

    def _render_ui_loop(self):
        """
        Main-thread consumer updating video canvas and telemetry.
        Runs periodically on the Tkinter main event loop via root.after().
        """
        if not self.is_running:
            return

        frame_rgb = None
        telemetry = None

        with self.frame_lock:
            if self.latest_display_frame is not None:
                frame_rgb = self.latest_display_frame
                self.latest_display_frame = None
                telemetry = self.latest_telemetry

        if frame_rgb is not None:
            try:
                h, w = frame_rgb.shape[:2]
                max_width = 750
                if w > max_width:
                    ratio = max_width / float(w)
                    frame_rgb = cv2.resize(frame_rgb, (max_width, int(h * ratio)))

                pil_image = Image.fromarray(frame_rgb)
                # Create ImageTk.PhotoImage strictly on the MainThread!
                self.current_photo = ImageTk.PhotoImage(image=pil_image)
                self.video_label.config(image=self.current_photo, text="")
            except Exception as render_err:
                self.logger.debug(f"Render warning: {render_err}")

        if telemetry is not None:
            self._update_detection_telemetry(telemetry)

        # Schedule next tick on main event loop
        self.root.after(30, self._render_ui_loop)

    def _update_detection_telemetry(self, telemetry: Dict[str, Any]):
        """Update detection telemetry text panel safely on the main thread."""
        try:
            self.info_text.config(state="normal")
            self.info_text.delete("1.0", tk.END)

            names = telemetry.get('names', [])
            locations = telemetry.get('locations', [])
            confidences = telemetry.get('confidences', [])
            liveness_list = telemetry.get('liveness', [])

            count = len(locations)
            if count == 0:
                self.info_text.insert("1.0", "Status: Monitoring feed...\nNo faces detected in view.")
            else:
                info_lines = [f"Faces in View: {count}", "=" * 32]

                for i, name in enumerate(names):
                    conf = confidences[i] if i < len(confidences) else 0.0
                    live_str = liveness_list[i] if i < len(liveness_list) else ""

                    if name != "Unknown":
                        if conf >= 70.0:
                            level = "[HIGH]  "
                        elif conf >= 50.0:
                            level = "[MEDIUM]"
                        else:
                            level = "[LOW]   "
                        info_lines.append(f"{level} {name} ({conf:.1f}%)")
                    else:
                        info_lines.append("[UNKNOWN] Unregistered Subject")

                    if live_str:
                        info_lines.append(f"  {live_str}")

                # Show average confidence score
                valid_confs = [c for c in confidences if c > 0.0]
                if valid_confs:
                    avg = sum(valid_confs) / float(len(valid_confs))
                    info_lines.append("=" * 32)
                    info_lines.append(f"Mean Confidence: {avg:.1f}%")

                self.info_text.insert("1.0", "\n".join(info_lines))

            self.info_text.config(state="disabled")
        except Exception:
            pass

    def toggle_anti_spoofing(self):
        """Toggle biometric anti-spoofing liveness verification."""
        self.system.anti_spoof.enabled = self.anti_spoof_var.get()
        state = "Enabled" if self.system.anti_spoof.enabled else "Disabled"
        self.update_status(f"Anti-Spoofing: {state}")
        self.logger.info(f"Anti-Spoofing liveness toggled: {state}")

    def toggle_privacy_blur(self):
        """Toggle automated bystander face blurring."""
        self.system.anti_spoof.privacy_blur_enabled = self.privacy_blur_var.get()
        state = "Enabled" if self.system.anti_spoof.privacy_blur_enabled else "Disabled"
        self.update_status(f"Privacy Mode (Bystander Blur): {state}")
        self.logger.info(f"Privacy Mode toggled: {state}")

    def update_detection_info(self):
        """Backward compatibility wrapper for updating detection telemetry."""
        if hasattr(self, 'latest_telemetry'):
            self._update_detection_telemetry(self.latest_telemetry)

    # --------------------------------------------------------------------------
    # 4. IDENTITY MANAGEMENT
    # --------------------------------------------------------------------------
    def load_faces(self):
        """Load enrolled identities into system and update GUI listbox."""
        if not self.system.load_encodings():
            if self.system.load_known_faces():
                self.system.save_encodings()
        self.refresh_faces()

    def refresh_faces(self):
        """Repopulate the enrolled identities listbox."""
        self.faces_listbox.delete(0, tk.END)
        for i, name in enumerate(self.system.known_face_names, 1):
            self.faces_listbox.insert(tk.END, f"  {i}. {name}")

        if not self.system.known_face_names:
            self.faces_listbox.insert(tk.END, "  (No enrolled identities)")

    def capture_face(self):
        """Enroll a face detected in the live camera frame."""
        if not self.is_running:
            messagebox.showwarning("Warning", "Camera is not active. Click 'Start Camera' first.")
            return

        with self.frame_lock:
            frame = self.latest_raw_frame.copy() if self.latest_raw_frame is not None else None

        if frame is None:
            messagebox.showerror("Error", "No camera frame is currently available. Please wait a moment.")
            return

        import face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_frame)

        if len(locations) == 0:
            messagebox.showwarning("No Face Detected", "No face was detected. Please position yourself clearly.")
            return

        if len(locations) > 1:
            messagebox.showwarning("Multiple Faces", "Multiple faces detected. Ensure only one person is in view.")
            return

        name = simpledialog.askstring("Enroll Face", "Enter person's name:")
        if name and name.strip():
            clean_name = name.strip()
            # Enroll into system
            encodings = face_recognition.face_encodings(rgb_frame, locations, num_jitters=1)
            if encodings:
                os.makedirs("known_faces", exist_ok=True)
                safe_filename = clean_name.replace(" ", "_")
                img_path = f"known_faces/{safe_filename}.jpg"

                pil_img = Image.fromarray(rgb_frame)
                pil_img.save(img_path, 'JPEG', quality=95)

                self.system.known_face_encodings.append(encodings[0])
                self.system.known_face_names.append(clean_name)
                self.system.save_encodings()

                self.db_manager.add_person(clean_name, img_path)
                self.refresh_faces()

                messagebox.showinfo("Success", f"Enrolled '{clean_name}' successfully.")
                self.update_status(f"Enrolled: {clean_name}")
                self.logger.info(f"Enrolled face: {clean_name}")
            else:
                messagebox.showerror("Error", "Could not compute biometric encoding.")

    def add_face_dialog(self):
        """Enroll an identity from an image file on disk."""
        file_path = filedialog.askopenfilename(
            title="Select Portrait Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp")]
        )
        if file_path:
            name = simpledialog.askstring("Enroll Identity", "Enter person's name:")
            if name and name.strip():
                clean_name = name.strip()
                try:
                    with Image.open(file_path) as pil_img:
                        rgb_img = pil_img.convert("RGB")
                        arr = np.array(rgb_img, dtype=np.uint8)

                    import face_recognition
                    encodings = face_recognition.face_encodings(arr, num_jitters=1)

                    if encodings:
                        safe_filename = clean_name.replace(" ", "_")
                        target_path = f"known_faces/{safe_filename}.jpg"
                        os.makedirs("known_faces", exist_ok=True)
                        rgb_img.save(target_path, "JPEG", quality=95)

                        self.system.known_face_encodings.append(encodings[0])
                        self.system.known_face_names.append(clean_name)
                        self.system.save_encodings()

                        self.db_manager.add_person(clean_name, target_path)
                        self.refresh_faces()

                        messagebox.showinfo("Success", f"Enrolled '{clean_name}' from file.")
                        self.update_status(f"Enrolled: {clean_name}")
                    else:
                        messagebox.showerror("Error", "No recognizable face found in selected image.")
                except Exception as err:
                    messagebox.showerror("Error", f"Failed to process image: {err}")

    def remove_face(self):
        """Delete selected identity from enrolled faces."""
        selection = self.faces_listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Required", "Please select an identity from the list to remove.")
            return

        selected_text = self.faces_listbox.get(selection[0])
        name = selected_text.split(". ", 1)[1] if ". " in selected_text else selected_text.strip()

        if name == "(No enrolled identities)":
            return

        if messagebox.askyesno("Confirm Deletion", f"Permanently remove '{name}' from enrolled identities?"):
            if self.system.remove_face(name):
                self.db_manager.delete_person(name)
                self.refresh_faces()
                messagebox.showinfo("Success", f"Removed '{name}' from system.")
                self.update_status(f"Removed: {name}")
            else:
                messagebox.showerror("Error", f"Could not remove '{name}'.")

    # --------------------------------------------------------------------------
    # 5. RECOGNITION HISTORY & REPORTING (IST)
    # --------------------------------------------------------------------------
    def add_recognition_to_history(self, name: str, confidence: float):
        """
        Record a recognition event with cooldown deduplication in IST.
        """
        now_ist = get_ist_now()

        # Cooldown check: prevent duplicate logs for same person within cooldown window
        if name in self.last_recognized:
            elapsed = (now_ist - self.last_recognized[name]).total_seconds()
            if elapsed < self.recognition_cooldown:
                return

        self.last_recognized[name] = now_ist

        record = {
            'name': name,
            'date': get_ist_date_string(now_ist),
            'time': now_ist.strftime("%H:%M:%S IST"),
            'timestamp': format_ist_timestamp(now_ist),
            'confidence': f"{confidence:.1f}%"
        }
        self.recognition_history.append(record)

        if self.auto_save_enabled:
            self.save_history()

    def save_history(self):
        """Persist recognition history records to CSV."""
        try:
            Path(self.history_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w', newline='', encoding='utf-8') as f:
                if self.recognition_history:
                    fieldnames = ['name', 'date', 'time', 'timestamp', 'confidence']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.recognition_history)
        except Exception as err:
            self.logger.error(f"Error saving history: {err}")

    def load_history(self):
        """Load historical records from CSV."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self.recognition_history = list(reader)
            except Exception as err:
                self.logger.error(f"Error loading history: {err}")

    def show_history(self):
        """Display recognition history modal dialog."""
        window = tk.Toplevel(self.root)
        window.title("Recognition History (IST)")
        window.geometry("900x550")
        window.configure(bg="#2b2b2b")

        title = tk.Label(
            window,
            text="Recognition History Records (IST)",
            font=("Arial", 16, "bold"),
            bg="#2b2b2b",
            fg="#00ff00"
        )
        title.pack(pady=10)

        # Telemetry Summary
        stats_frame = tk.Frame(window, bg="#1e1e1e")
        stats_frame.pack(fill="x", padx=10, pady=5)
        total = len(self.recognition_history)
        unique = len(set(r['name'] for r in self.recognition_history)) if self.recognition_history else 0

        tk.Label(
            stats_frame,
            text=f"Total Sightings: {total} | Unique Identities: {unique}",
            bg="#1e1e1e",
            fg="white",
            font=("Arial", 10)
        ).pack(pady=6)

        # Table View
        tree_frame = tk.Frame(window, bg="#2b2b2b")
        tree_frame.pack(fill="both", expand=True, padx=10, pady=10)

        vsb = tk.Scrollbar(tree_frame, orient="vertical")
        vsb.pack(side="right", fill="y")
        hsb = tk.Scrollbar(tree_frame, orient="horizontal")
        hsb.pack(side="bottom", fill="x")

        tree = ttk.Treeview(
            tree_frame,
            columns=("Name", "Date", "Time", "Confidence"),
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set
        )
        tree.heading("Name", text="Identity Name")
        tree.heading("Date", text="Date (IST)")
        tree.heading("Time", text="Time (IST)")
        tree.heading("Confidence", text="Match Confidence")

        tree.column("Name", width=220)
        tree.column("Date", width=140)
        tree.column("Time", width=140)
        tree.column("Confidence", width=120)

        for record in reversed(self.recognition_history):
            tree.insert("", tk.END, values=(
                record.get('name', ''),
                record.get('date', ''),
                record.get('time', ''),
                record.get('confidence', '')
            ))

        tree.pack(fill="both", expand=True, side="left")
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        # Action Buttons
        btn_frame = tk.Frame(window, bg="#2b2b2b")
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="Export CSV",
            command=self.export_recognition_history,
            bg="#0066cc",
            fg="white",
            font=("Arial", 10, "bold"),
            width=12
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame,
            text="Clear All",
            command=self.clear_history,
            bg="#aa0000",
            fg="white",
            font=("Arial", 10, "bold"),
            width=12
        ).pack(side="left", padx=5)

    def show_today_history(self):
        """Display recognitions recorded today in Indian Standard Time."""
        today = get_ist_date_string()
        today_records = [r for r in self.recognition_history if r.get('date') == today]

        window = tk.Toplevel(self.root)
        window.title(f"Today's Recognitions - {today} IST")
        window.geometry("800x480")
        window.configure(bg="#2b2b2b")

        tk.Label(
            window,
            text=f"Today's Recognition Events (IST: {today})",
            font=("Arial", 15, "bold"),
            bg="#2b2b2b",
            fg="#00ff00"
        ).pack(pady=10)

        tk.Label(
            window,
            text=f"Total sightings today: {len(today_records)}",
            font=("Arial", 10),
            bg="#2b2b2b",
            fg="white"
        ).pack(pady=(0, 10))

        tree_frame = tk.Frame(window, bg="#2b2b2b")
        tree_frame.pack(fill="both", expand=True, padx=10, pady=5)

        scrollbar = tk.Scrollbar(tree_frame)
        scrollbar.pack(side="right", fill="y")

        tree = ttk.Treeview(
            tree_frame,
            columns=("Name", "Time", "Confidence"),
            show="headings",
            yscrollcommand=scrollbar.set
        )
        tree.heading("Name", text="Identity Name")
        tree.heading("Time", text="Time (IST)")
        tree.heading("Confidence", text="Confidence")

        tree.column("Name", width=250)
        tree.column("Time", width=200)
        tree.column("Confidence", width=150)

        for record in reversed(today_records):
            tree.insert("", tk.END, values=(
                record.get('name', ''),
                record.get('time', ''),
                record.get('confidence', '')
            ))

        tree.pack(fill="both", expand=True, side="left")
        scrollbar.config(command=tree.yview)

    def show_db_history(self):
        """Query and display attendance logs stored in SQLite."""
        window = tk.Toplevel(self.root)
        window.title("Database Attendance History")
        window.geometry("820x480")
        window.configure(bg="#2b2b2b")

        tk.Label(
            window,
            text="Relational Database Attendance Logs (SQLite)",
            font=("Arial", 15, "bold"),
            bg="#2b2b2b",
            fg="#00ff00"
        ).pack(pady=10)

        tree_frame = tk.Frame(window, bg="#2b2b2b")
        tree_frame.pack(fill="both", expand=True, padx=10, pady=5)

        scrollbar = tk.Scrollbar(tree_frame)
        scrollbar.pack(side="right", fill="y")

        tree = ttk.Treeview(
            tree_frame,
            columns=("Name", "Timestamp", "Confidence"),
            show="headings",
            yscrollcommand=scrollbar.set
        )
        tree.heading("Name", text="Identity Name")
        tree.heading("Timestamp", text="Timestamp (IST)")
        tree.heading("Confidence", text="Confidence Score")

        tree.column("Name", width=220)
        tree.column("Timestamp", width=300)
        tree.column("Confidence", width=120)

        history = self.db_manager.get_recognition_history(limit=100)
        for _, row in history.iterrows():
            try:
                conf_val = float(row['confidence']) * 100.0
                conf_str = f"{conf_val:.1f}%"
            except Exception:
                conf_str = str(row['confidence'])

            tree.insert("", tk.END, values=(
                row['name'],
                row['timestamp'],
                conf_str
            ))

        tree.pack(fill="both", expand=True, side="left")
        scrollbar.config(command=tree.yview)

    def show_gallery(self):
        """Display enrolled identity portraits in a scrollable card grid."""
        window = tk.Toplevel(self.root)
        window.title("Enrolled Identities Gallery")
        window.geometry("720x500")
        window.configure(bg="#2b2b2b")

        canvas = tk.Canvas(window, bg="#2b2b2b", highlightthickness=0)
        scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#2b2b2b")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        persons = self.db_manager.get_all_persons()
        if not persons:
            tk.Label(
                scrollable_frame,
                text="No enrolled identities in database.",
                bg="#2b2b2b",
                fg="gray",
                font=("Arial", 12)
            ).pack(pady=40, padx=40)

        # Store photo references to prevent garbage collection
        self._gallery_photos = []

        for i, person in enumerate(persons):
            row = i // 3
            col = i % 3

            card = tk.Frame(scrollable_frame, bg="#1e1e1e", relief="raised", bd=2)
            card.grid(row=row, column=col, padx=12, pady=12)

            img_path = person.get('image_path')
            if img_path and os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        img.thumbnail((140, 140))
                        photo = ImageTk.PhotoImage(img)
                        self._gallery_photos.append(photo)

                    img_lbl = tk.Label(card, image=photo, bg="#1e1e1e")
                    img_lbl.pack(pady=5, padx=5)
                except Exception:
                    pass

            tk.Label(
                card,
                text=person['name'],
                font=("Arial", 10, "bold"),
                bg="#1e1e1e",
                fg="white"
            ).pack()

            date_str = person.get('added_date', '')[:10]
            tk.Label(
                card,
                text=f"Enrolled: {date_str}",
                font=("Arial", 8),
                bg="#1e1e1e",
                fg="#aaaaaa"
            ).pack(pady=(0, 5))

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def export_recognition_history(self):
        """Export session recognition history to CSV with an IST timestamp."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"recognition_history_{get_ist_filename_timestamp()}.csv"
        )
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    if self.recognition_history:
                        fieldnames = ['name', 'date', 'time', 'timestamp', 'confidence']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(self.recognition_history)

                messagebox.showinfo("Export Successful", f"Records exported to:\n{file_path}")
                self.logger.info(f"Exported recognition history to {file_path}")
            except Exception as err:
                messagebox.showerror("Export Error", f"Failed to export history: {err}")

    def export_logs(self):
        """Export SQLite database logs to CSV."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"database_logs_{get_ist_filename_timestamp()}.csv"
        )
        if file_path:
            try:
                self.db_manager.export_to_csv(file_path)
                messagebox.showinfo("Export Successful", f"Database logs exported to:\n{file_path}")
                self.logger.info(f"Exported database logs to {file_path}")
            except Exception as err:
                messagebox.showerror("Export Error", f"Failed to export database logs: {err}")

    def clear_history(self):
        """Clear memory and on-disk recognition history."""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all recognition history?"):
            self.recognition_history.clear()
            self.last_recognized.clear()
            self.save_history()
            messagebox.showinfo("Success", "Recognition history cleared.")
            self.logger.info("Recognition history cleared")

    def toggle_auto_save(self):
        """Toggle auto-save state."""
        self.auto_save_enabled = self.auto_save_var.get()
        state = "enabled" if self.auto_save_enabled else "disabled"
        self.update_status(f"Auto-save {state}")

    def set_cooldown(self):
        """Configure recognition cooldown period in seconds."""
        cooldown = simpledialog.askinteger(
            "Configure Cooldown",
            "Enter cooldown period in seconds (1-60):",
            initialvalue=self.recognition_cooldown,
            minvalue=1,
            maxvalue=60
        )
        if cooldown:
            self.recognition_cooldown = cooldown
            messagebox.showinfo("Success", f"Recognition cooldown set to {cooldown} seconds.")

    def show_about(self):
        """Display about information."""
        about_text = (
            "Face Recognition System v2.0\n"
            "Indian Standard Time (IST) Edition\n\n"
            "Key Capabilities:\n"
            "  [+] Real-time multi-face detection (HOG / Haar)\n"
            "  [+] 128-dimensional deep metric embeddings\n"
            "  [+] Color-coded confidence feedback\n"
            "  [+] Relational SQLite database attendance logging\n"
            "  [+] Thread-safe Tkinter desktop architecture\n"
            "  [+] 2026 Anti-Spoofing and Security Architecture\n\n"
            "Developer: Nilay Naha\n"
            "License: MIT License"
        )
        messagebox.showinfo("About Face Recognition System", about_text)

    def update_status(self, message: str):
        """Update bottom status bar message."""
        self.status_bar.config(text=f"  {message}")

    def on_closing(self):
        """Gracefully release camera and close application window."""
        if messagebox.askyesno("Quit Application", "Are you sure you want to quit?"):
            if self.recognition_history and self.auto_save_enabled:
                self.save_history()
            self.stop_camera()
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=1.0)
            self.logger.info("Application terminated cleanly")
            self.root.destroy()

    def run(self):
        """Launch the Tkinter main event loop."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    """Main function entry point for GUI."""
    app = FaceRecognitionGUI()
    app.run()

if __name__ == "__main__":
    main()
