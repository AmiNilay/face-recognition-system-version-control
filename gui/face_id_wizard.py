"""
Apple Face ID Multi-Angle Enrollment Wizard Modal Dialog
Module: gui/face_id_wizard.py

Provides an Apple-inspired modal dialog for guided multi-angle face enrollment:
- Renders segmented circular progress HUD directly on the live camera stream.
- Dynamic pose status indicators (Frontal, Left, Right, Up, Down).
- Synthesizes L2-normalized composite biometric embeddings.
- Offline audio voice guidance via AudioAnnouncer.

All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code, HUD readouts, and logs. Developer: Nilay Naha.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from typing import Optional, Callable

import numpy as np

from src.composite_enrollment import (
    CompositeEnrollmentSession,
    render_apple_face_id_hud,
    ENROLLMENT_STEPS,
    FaceIDAngle
)
from src.security import SecurityManager
from src.utils import format_ist_timestamp


class FaceIDWizardModal(tk.Toplevel):
    """
    Apple Face ID Multi-Angle Enrollment Modal Window.
    """

    def __init__(
        self,
        parent: tk.Tk,
        system,
        db_manager,
        announcer,
        identity_name: str,
        on_complete: Optional[Callable[[bool, str], None]] = None,
        video_source: int = 0
    ):
        super().__init__(parent)
        self.parent = parent
        self.system = system
        self.db = db_manager
        self.announcer = announcer
        self.identity_name = SecurityManager.sanitize_identity_name(identity_name)
        self.on_complete = on_complete
        self.video_source = video_source

        # Window styling
        self.title("Apple Face ID Enrollment Wizard")
        self.geometry("780x680")
        self.minsize(700, 600)
        self.configure(bg="#161618")
        self.transient(parent)
        self.grab_set()

        # State tracking
        self.session = CompositeEnrollmentSession(identity_name=self.identity_name)
        self.is_running = True
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.thread: Optional[threading.Thread] = None
        self.last_step_announced = -1
        self.success = False

        # Build UI layout
        self.setup_ui()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start camera capture thread
        self.start_stream()

    def setup_ui(self):
        """Construct the Apple-inspired wizard UI."""
        # Header banner
        header = tk.Frame(self, bg="#1c1c1e", height=60)
        header.pack(fill="x", padx=12, pady=(12, 6))
        header.pack_propagate(False)

        title_lbl = tk.Label(
            header,
            text=f"Face ID Setup: {self.identity_name}",
            font=("Segoe UI", 14, "bold"),
            bg="#1c1c1e",
            fg="#f5f5f7"
        )
        title_lbl.pack(side="left", padx=16, pady=8)

        # Pose Indicator Pills Container
        pills_frame = tk.Frame(self, bg="#161618")
        pills_frame.pack(fill="x", padx=16, pady=6)

        self.pill_labels = []
        for step in ENROLLMENT_STEPS:
            lbl = tk.Label(
                pills_frame,
                text=step.title,
                font=("Segoe UI", 9, "bold"),
                bg="#2c2c2e",
                fg="#8e8e93",
                padx=10,
                pady=4
            )
            lbl.pack(side="left", expand=True, fill="x", padx=3)
            self.pill_labels.append(lbl)

        # Video Canvas Frame
        video_container = tk.Frame(self, bg="#0d0d0e", highlightthickness=1, highlightbackground="#2c2c2e")
        video_container.pack(fill="both", expand=True, padx=16, pady=6)

        self.video_label = tk.Label(
            video_container,
            bg="#0d0d0e",
            text="Initializing Face ID sensor...",
            font=("Segoe UI", 12),
            fg="#636366"
        )
        self.video_label.pack(fill="both", expand=True)

        # Bottom Instructions & Progress
        bottom_frame = tk.Frame(self, bg="#1c1c1e")
        bottom_frame.pack(fill="x", padx=16, pady=(6, 12))

        self.instruction_lbl = tk.Label(
            bottom_frame,
            text="Position your face directly in the circular guide",
            font=("Segoe UI", 11, "bold"),
            bg="#1c1c1e",
            fg="#0a84ff"
        )
        self.instruction_lbl.pack(pady=(8, 4))

        self.progress_bar = ttk.Progressbar(bottom_frame, length=600, mode="determinate")
        self.progress_bar.pack(pady=4)

        btn_container = tk.Frame(bottom_frame, bg="#1c1c1e")
        btn_container.pack(pady=6)

        cancel_btn = tk.Button(
            btn_container,
            text="Cancel",
            command=self.on_close,
            bg="#2c2c2e",
            fg="#ff453a",
            activebackground="#3a3a3c",
            activeforeground="#ff453a",
            font=("Segoe UI", 10, "bold"),
            width=14,
            relief="flat",
            bd=0,
            cursor="hand2"
        )
        cancel_btn.pack()

    def start_stream(self):
        """Spawn the background capture and inference worker thread."""
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def _capture_worker(self):
        """Worker thread executing camera capture, landmarks, and Face ID state machine."""
        self.cap = cv2.VideoCapture(self.video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        import face_recognition

        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h_f, w_f = frame.shape[:2]
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            locations = self.system.detector.detect_faces(rgb_small)
            head_pose = None
            face_encoding = None
            face_crop = None

            if locations:
                top_s, right_s, bottom_s, left_s = locations[0]
                y1, y2 = max(0, top_s * 4), min(h_f, bottom_s * 4)
                x1, x2 = max(0, left_s * 4), min(w_f, right_s * 4)
                if y2 > y1 and x2 > x1:
                    face_crop = frame[y1:y2, x1:x2]

                landmarks = face_recognition.face_landmarks(rgb_small, [locations[0]])
                if landmarks:
                    head_pose, _ = self.system.anti_spoof.estimate_head_pose(
                        landmarks[0], frame_shape=(h_f, w_f), scale=4.0
                    )

                encs = self.system.encoder.encode_faces(rgb_small, [locations[0]])
                if encs:
                    face_encoding = encs[0]

            # Evaluate against active pose criteria
            telemetry = self.session.process_frame(
                frame_bgr=frame,
                head_pose=head_pose,
                face_encoding=face_encoding,
                face_crop=face_crop
            )

            # Audio announcement on step progression
            step_idx = self.session.current_step_index
            if step_idx != self.last_step_announced:
                if self.announcer and self.announcer.enabled:
                    if self.session.is_completed:
                        self.announcer.announce_custom("Enrollment complete.")
                    elif self.session.current_step:
                        self.announcer.announce_custom(self.session.current_step.instruction)
                self.last_step_announced = step_idx

            # Render Apple Face ID circular segmented tick ring HUD
            display_frame = frame.copy()
            render_apple_face_id_hud(display_frame, self.session)

            # Post updates to UI thread
            self.after(0, self._update_ui_frame, display_frame, telemetry)

            if self.session.is_completed:
                self._finalize_enrollment()
                break

            time.sleep(0.02)

        if self.cap:
            self.cap.release()

    def _update_ui_frame(self, bgr_frame: np.ndarray, telemetry: dict):
        """Update Tkinter visual elements on the main UI thread."""
        if not self.winfo_exists():
            return

        try:
            # Update video canvas
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_frame.shape[:2]
            max_w = 640
            if w > max_w:
                ratio = max_w / float(w)
                rgb_frame = cv2.resize(rgb_frame, (max_w, int(h * ratio)))

            pil_img = Image.fromarray(rgb_frame)
            self.current_photo = ImageTk.PhotoImage(image=pil_img)
            self.video_label.config(image=self.current_photo, text="")

            # Update instruction and progress bar
            instruction = telemetry.get("instruction", "Follow on-screen instructions")
            progress = telemetry.get("progress", 0.0)
            self.instruction_lbl.config(text=instruction)
            self.progress_bar["value"] = progress

            # Update pose indicator pills
            step_idx = self.session.current_step_index
            for i, pill in enumerate(self.pill_labels):
                if i < step_idx:
                    pill.config(bg="#30d158", fg="#000000")  # Completed: Green
                elif i == step_idx:
                    pill.config(bg="#0a84ff", fg="#ffffff")  # Active: Blue
                else:
                    pill.config(bg="#2c2c2e", fg="#8e8e93")  # Pending: Gray

        except Exception:
            pass

    def _finalize_enrollment(self):
        """Save synthesized composite vector to encrypted storage and database."""
        composite_vec = self.session.composite_embedding
        composite_card = self.session.generate_composite_portrait()

        # Update FaceRecognitionSystem
        self.system.enroll_composite_identity(
            name=self.identity_name,
            composite_embedding=composite_vec,
            reference_portrait=composite_card
        )

        # Update SQLite database
        safe_name = self.identity_name.replace(" ", "_")
        if self.db:
            try:
                self.db.add_person(self.identity_name, f"known_faces/{safe_name}.jpg")
            except Exception:
                pass

        self.success = True
        time.sleep(1.2)
        self.after(0, self._finish_and_close)

    def _finish_and_close(self):
        """Close modal and invoke completion callback."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()
        if self.on_complete:
            self.on_complete(self.success, self.identity_name)

    def on_close(self):
        """Handle user window close / cancellation."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()
        if self.on_complete:
            self.on_complete(False, self.identity_name)
