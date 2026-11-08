"""
Primary Face Recognition System Module

This module coordinates the complete computer vision pipeline:
1. Ingesting and decoding reference photos from the known_faces/ folder.
2. Generating and caching 128-dimensional biometric embeddings.
3. Leveraging modular FaceDetector and FaceEncoder components.
4. Capturing real-time video feeds from camera devices with settings from config.yaml.
5. Downscaling frames for accelerated inference (frame skipping and 0.25 scale).
6. Computing Euclidean face distance and mapping distance to confidence percentages.
7. Drawing bounding overlays with confidence scores and telemetry.
8. Managing facial additions, removals, and frame captures with IST timestamps.
"""

import os
import re
import cv2
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Optional, Dict, Any

from src.utils import get_ist_filename_timestamp, format_ist_timestamp, load_config
from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder
from src.anti_spoofing import AntiSpoofEngine, LivenessResult
from src.security import SecurityManager
from src.vector_index import VectorIndex
from src.motion_detector import MotionDetector
from src.webhook_dispatcher import WebhookDispatcher, SecurityEventType

def sanitize_identity_name(name: str) -> str:
    """
    Sanitize a human identity name to prevent path traversal and filesystem errors.
    Delegates to SecurityManager for centralized cryptographic security policy (SEC-02).
    """
    return SecurityManager.sanitize_identity_name(name)

class FaceRecognitionSystem:
    """
    Core facial recognition controller and video processing engine.
    Integrates FaceDetector and FaceEncoder with configuration-driven tuning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the system with configuration and modular detector/encoder engines.

        Args:
            config (dict, optional): Custom configuration dictionary.
                                     If None, loads settings from config/config.yaml.
        """
        self.config = config if config is not None else load_config()

        # Extract config parameters with robust fallbacks
        recog_cfg = self.config.get('face_recognition', {})
        self.tolerance = float(recog_cfg.get('tolerance', 0.6))
        self.model_type = str(recog_cfg.get('model', 'hog'))
        self.encoding_model = str(recog_cfg.get('encoding_model', 'large'))
        self.process_interval = int(recog_cfg.get('process_interval', 3))

        cam_cfg = self.config.get('camera', {})
        self.cam_width = int(cam_cfg.get('width', 640))
        self.cam_height = int(cam_cfg.get('height', 480))
        self.cam_fps = int(cam_cfg.get('fps', 30))

        # Modular engines
        self.detector = FaceDetector(detection_method=self.model_type)
        self.encoder = FaceEncoder(encoding_model=self.encoding_model)
        self.anti_spoof = AntiSpoofEngine(self.config)
        self.sec = SecurityManager()
        self.vector_index = VectorIndex(use_kdtree=True)

        # Motion detection gating
        motion_cfg = self.config.get('motion_detection', {})
        self.motion_detector = MotionDetector(
            enabled=bool(motion_cfg.get('enabled', False)),
            min_motion_area=int(motion_cfg.get('min_area', 800)),
            var_threshold=float(motion_cfg.get('threshold', 25.0)),
            cooldown_frames=int(motion_cfg.get('cooldown_frames', 20))
        )

        # Webhook alert dispatcher
        webhook_cfg = self.config.get('webhooks', {})
        self.webhook_dispatcher = WebhookDispatcher(
            endpoints=webhook_cfg.get('endpoints', []),
            enabled=bool(webhook_cfg.get('enabled', False)),
            cooldown_seconds=float(webhook_cfg.get('cooldown_seconds', 30.0)),
            include_snapshot=bool(webhook_cfg.get('include_snapshot', True)),
            camera_id=str(self.config.get('camera', {}).get('device', 'Camera_0'))
        )

        # Internal state collections
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.face_locations: List[Tuple[int, int, int, int]] = []
        self.face_encodings: List[np.ndarray] = []
        self.face_names: List[str] = []
        self.face_confidences: List[float] = []
        self.face_liveness: List[LivenessResult] = []

    def load_known_faces(self, images_folder: str = "known_faces") -> bool:
        """
        Load reference portrait images from a directory and compute their 128-d embeddings.
        Supports standard formats: JPEG, PNG, WEBP, BMP, TIFF, GIF.

        Args:
            images_folder (str): Directory containing labeled reference photos.

        Returns:
            bool: True if one or more valid face encodings were loaded; False otherwise.
        """
        folder_path = Path(images_folder)
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Created directory: {images_folder}")
            print("[INFO] Add images in format: Person_Name.jpg")
            return False

        valid_exts = ('.png', '.jpg', '.jpeg', '.gif', '.tiff', '.webp', '.bmp')
        image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(valid_exts)]

        if not image_files:
            print(f"[WARNING] No reference images found in: {images_folder}")
            return False

        print("[INFO] Loading enrolled face images...")
        self.known_face_encodings.clear()
        self.known_face_names.clear()

        for filename in image_files:
            try:
                filepath = os.path.join(images_folder, filename)
                # Load image via PIL to safely handle various color formats and alpha channels
                with Image.open(filepath) as pil_image:
                    rgb_image = pil_image.convert('RGB')
                    image_array = np.array(rgb_image, dtype=np.uint8)

                # Compute 128-d feature encoding
                face_encoding = self.encoder.encode_face(image_array)

                if face_encoding is not None:
                    # Derive clean display name from filename (e.g. 'John_Doe.jpg' -> 'John Doe')
                    raw_stem = os.path.splitext(filename)[0]
                    name = raw_stem.replace('_', ' ').strip()
                    
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                    print(f"[OK] Enrolled: {name}")
                else:
                    print(f"[WARNING] No face detected in: {filename}")
            except Exception as err:
                print(f"[ERROR] Failed to load {filename}: {err}")

        # Sync modular encoder and vector index
        self.encoder.known_encodings = list(self.known_face_encodings)
        self.encoder.known_names = list(self.known_face_names)
        self.vector_index.build(self.known_face_encodings, self.known_face_names)

        print(f"\n[INFO] Total enrolled identities: {len(self.known_face_names)}")
        return len(self.known_face_names) > 0

    def save_encodings(self, filename: Optional[str] = None):
        """
        Serialize enrolled identities and embeddings using AES-authenticated
        encrypted storage without pickle vulnerabilities (SEC-01).

        Args:
            filename (str, optional): Destination cache file path.
        """
        enc_file = filename or self.config.get('storage', {}).get('encodings_file', 'data/face_encodings.dat')
        # Migrate .pkl extension to safe .dat
        if enc_file.endswith('.pkl'):
            enc_file = enc_file[:-4] + '.dat'

        success = self.sec.save_encrypted_encodings(
            enc_file, self.known_face_names, self.known_face_encodings
        )
        if success:
            print(f"[INFO] Face encodings safely encrypted and saved to: {enc_file}")
        else:
            print(f"[ERROR] Failed to save encrypted face encodings to: {enc_file}")

    def load_encodings(self, filename: Optional[str] = None) -> bool:
        """
        Load pre-computed face embeddings from encrypted vector storage.
        Provides safe, automatic migration if legacy .pkl file is detected.

        Args:
            filename (str, optional): Path to stored encodings file.

        Returns:
            bool: True if loaded successfully; False otherwise.
        """
        enc_file = filename or self.config.get('storage', {}).get('encodings_file', 'data/face_encodings.dat')
        dat_file = enc_file[:-4] + '.dat' if enc_file.endswith('.pkl') else enc_file

        # 1. Primary path: Load from secure encrypted archive (.dat)
        if os.path.exists(dat_file):
            names, encodings = self.sec.load_encrypted_encodings(dat_file)
            if names and encodings:
                self.known_face_names = names
                self.known_face_encodings = encodings
                self.encoder.known_names = list(self.known_face_names)
                self.encoder.known_encodings = list(self.known_face_encodings)
                self.vector_index.build(self.known_face_encodings, self.known_face_names)
                print(f"[INFO] Loaded {len(self.known_face_names)} identities from encrypted archive: {dat_file}")
                return True

        # 2. Migration fallback: If legacy .pkl exists, load once and auto-migrate to .dat
        legacy_pkl = "data/face_encodings.pkl"
        if os.path.exists(legacy_pkl):
            try:
                import pickle
                with open(legacy_pkl, 'rb') as f:
                    data = pickle.load(f)
                self.known_face_names = data.get("names", [])
                self.known_face_encodings = data.get("encodings", [])
                self.encoder.known_names = list(self.known_face_names)
                self.encoder.known_encodings = list(self.known_face_encodings)
                self.vector_index.build(self.known_face_encodings, self.known_face_names)
                print(f"[SECURITY MIGRATION] Migrating {len(self.known_face_names)} identities from legacy pickle to encrypted storage...")
                self.save_encodings(dat_file)
                # Safely archive legacy .pkl
                try:
                    os.rename(legacy_pkl, legacy_pkl + ".migrated.bak")
                except Exception:
                    pass
                return True
            except Exception as err:
                print(f"[WARNING] Could not migrate legacy encodings: {err}")
                return False

        return False

    def recognize_face(self, face_encoding: np.ndarray, tolerance: Optional[float] = None) -> Tuple[str, float]:
        """
        Identify a face encoding by calculating Euclidean distances to all known identities
        accelerated via spatial VectorIndex (cKDTree).

        Beginner Concept:
        - Euclidean Distance: Measures distance between two vectors.
          Distance 0.0 means identical. Distance > 0.6 means distinct faces.
        - Confidence Score: Inverted percentage scale:
          Confidence = max(0.0, (1.0 - distance)) * 100

        Args:
            face_encoding (np.ndarray): 128-d vector of newly detected face.
            tolerance (float, optional): Match threshold. Defaults to configured tolerance.

        Returns:
            tuple[str, float]: (matched_person_name, confidence_percentage).
        """
        tol = self.tolerance if tolerance is None else tolerance

        # Keep vector index synchronized if known_face_encodings list was mutated directly
        if len(self.vector_index) != len(self.known_face_encodings):
            self.vector_index.build(self.known_face_encodings, self.known_face_names)

        name, confidence, _ = self.vector_index.query(face_encoding, tolerance=tol)
        return name, confidence

    def run_recognition(self, video_source=0):
        """
        Run the interactive command-line recognition loop on a video stream
        using camera resolution and interval settings from config.yaml.

        Args:
            video_source: Camera index (0) or path to video file.
        """
        video_capture = cv2.VideoCapture(video_source)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        video_capture.set(cv2.CAP_PROP_FPS, self.cam_fps)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not video_capture.isOpened():
            print(f"[ERROR] Could not open video source: {video_source}")
            return

        print("\n[START] Camera stream initiated.")
        print("Keyboard Controls:")
        print("  - 'Q': Quit")
        print("  - 'S': Save screenshot (IST timestamped)")
        print("  - 'C': Enroll new face live\n")

        frame_count = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("[ERROR] Failed to read video frame from camera.")
                break

            frame_count += 1

            # Performance Optimization from config: process 1 of every N frames
            if frame_count % self.process_interval == 0:
                # Downscale frame to 25% (1/4 size) for rapid inference
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Locate faces using modular detector
                self.face_locations = self.detector.detect_faces(rgb_small_frame)

                if self.face_locations:
                    self.face_encodings = self.encoder.encode_faces(rgb_small_frame, self.face_locations)
                    # Extract 68-point landmarks for anti-spoofing and 3D head pose estimation
                    landmarks_list = face_recognition.face_landmarks(rgb_small_frame, self.face_locations)
                    self.face_names = []
                    self.face_confidences = []
                    self.face_liveness = []
                    h_f, w_f = frame.shape[:2]

                    for idx, enc in enumerate(self.face_encodings):
                        name, conf = self.recognize_face(enc)
                        self.face_names.append(name)
                        self.face_confidences.append(conf)

                        top_s, right_s, bottom_s, left_s = self.face_locations[idx]
                        y1, y2 = max(0, top_s * 4), min(h_f, bottom_s * 4)
                        x1, x2 = max(0, left_s * 4), min(w_f, right_s * 4)
                        face_crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else None

                        lms = landmarks_list[idx] if idx < len(landmarks_list) else None
                        liveness_res = self.anti_spoof.verify_liveness(
                            name, face_crop, lms, frame_shape=(h_f, w_f), scale=4.0
                        )
                        self.face_liveness.append(liveness_res)
                else:
                    self.face_encodings = []
                    self.face_names = []
                    self.face_confidences = []
                    self.face_liveness = []

            # Draw visual bounding boxes and HUD stats on full-sized frame
            self.draw_results(frame)
            cv2.imshow("Face Recognition System (CLI Mode) - Press Q to Quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'Q' or ESC
                break
            elif key == ord('s'):
                self.save_frame(frame)
            elif key == ord('c'):
                self.capture_new_face(frame)

        video_capture.release()
        cv2.destroyAllWindows()
        print("\n[STOP] Camera closed.")

    def draw_results(self, frame: np.ndarray):
        """
        Render color-coded bounding boxes, confidence readouts, and liveness HUD onto a frame.

        Color Code:
        - Green (High confidence: >= 70%)
        - Orange (Medium confidence: 50% - 69%)
        - Yellow (Low confidence: < 50%)
        - Red (Unknown / Unregistered)
        """
        # Upscale coordinates by 4 to map back from 0.25 downscaled frame
        scaled_locations = [
            (top * 4, right * 4, bottom * 4, left * 4) 
            for (top, right, bottom, left) in self.face_locations
        ]

        # Apply automated bystander face blur (Privacy Mode) if enabled
        if self.anti_spoof.privacy_blur_enabled:
            self.anti_spoof.apply_privacy_blur(frame, scaled_locations, self.face_names)

        for i, ((top, right, bottom, left), name) in enumerate(zip(scaled_locations, self.face_names)):
            confidence = self.face_confidences[i] if i < len(self.face_confidences) else 0.0

            if name == "Unknown":
                color = (0, 0, 255)       # Red
            elif confidence >= 70.0:
                color = (0, 255, 0)       # Green
            elif confidence >= 50.0:
                color = (0, 165, 255)     # Orange
            else:
                color = (0, 255, 255)     # Yellow

            # Draw bounding rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw label banner
            cv2.rectangle(frame, (left, bottom - 50), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 30), font, 0.6, (255, 255, 255), 1)

            if name != "Unknown":
                conf_text = f"{confidence:.1f}%"
                cv2.putText(frame, conf_text, (left + 6, bottom - 10), font, 0.5, (255, 255, 255), 1)

            # Render liveness indicator banner above top bounding box
            liveness = self.face_liveness[i] if i < len(self.face_liveness) else None
            if liveness:
                # 1. Primary status / challenge banner
                cv2.putText(
                    frame,
                    liveness.status_text,
                    (left, max(top - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    liveness.hud_color,
                    1
                )

                # 2. 3D Head Pose Euler angles (Yaw, Pitch, Roll)
                if liveness.head_pose is not None:
                    yaw_val, pitch_val, roll_val = liveness.head_pose
                    pose_text = f"Y:{yaw_val:+.0f} P:{pitch_val:+.0f} R:{roll_val:+.0f}"
                    cv2.putText(
                        frame,
                        pose_text,
                        (left, max(top - 26, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (200, 255, 200),
                        1
                    )

                # 3. 3D projected nose ray vector
                if liveness.nose_ray is not None:
                    p_start, p_end = liveness.nose_ray
                    cv2.arrowedLine(frame, p_start, p_end, (0, 255, 255), 2, tipLength=0.2)

        # Draw HUD text telemetry in top-left corner
        cv2.putText(frame, f"Faces: {len(self.face_locations)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Enrolled: {len(self.known_face_names)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Average confidence display
        valid_confs = [c for c in self.face_confidences if c > 0.0]
        if valid_confs:
            avg_conf = sum(valid_confs) / float(len(valid_confs))
            cv2.putText(frame, f"Avg Confidence: {avg_conf:.1f}%", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.anti_spoof.enabled:
            hud_mode = "[ACTIVE]"
            if self.anti_spoof.privacy_blur_enabled:
                hud_mode += " [PRIVACY: ON]"
            cv2.putText(frame, f"Anti-Spoof: {hud_mode}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def save_frame(self, frame: np.ndarray):
        """Save a snapshot of the current frame with an IST timestamp."""
        directory = "captured_images"
        os.makedirs(directory, exist_ok=True)
        timestamp = get_ist_filename_timestamp()
        filepath = os.path.join(directory, f"capture_{timestamp}.jpg")
        cv2.imwrite(filepath, frame)
        print(f"[CAPTURE] Snapshot saved: {filepath}")

    def capture_new_face(self, frame: np.ndarray):
        """Enroll a new identity directly from the live video feed."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        temp_locations = self.detector.detect_faces(rgb_frame)

        if not temp_locations:
            print("[WARNING] No face detected in frame. Please reposition.")
            return

        if len(temp_locations) > 1:
            print("[WARNING] Multiple faces detected. Ensure only one person is in view.")
            return

        print("\nEnter name for the detected face: ", end='', flush=True)
        raw_name = input().strip()
        clean_name = sanitize_identity_name(raw_name)

        if not clean_name:
            print("[ERROR] Invalid name. Only alphanumeric characters are allowed.")
            return

        safe_filename = clean_name.replace(" ", "_")
        target_path = os.path.join("known_faces", f"{safe_filename}.jpg")
        os.makedirs("known_faces", exist_ok=True)

        # Save reference portrait
        pil_img = Image.fromarray(rgb_frame.astype(np.uint8))
        pil_img.save(target_path, 'JPEG', quality=95)

        # Compute embedding and enroll
        encodings = self.encoder.encode_faces(rgb_frame, temp_locations)
        if encodings:
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(clean_name)
            self.save_encodings()
            print(f"[SUCCESS] Enrolled {clean_name} into known faces directory.")
        else:
            print("[ERROR] Could not compute face encoding.")

    def add_face_from_frame(self, frame: np.ndarray, name: str) -> bool:
        """
        Add a face from a camera frame programmatically.

        Args:
            frame (np.ndarray): Video frame array in BGR format.
            name (str): Person's name.

        Returns:
            bool: True if enrolled successfully; False otherwise.
        """
        clean_name = sanitize_identity_name(name)
        if not clean_name:
            return False

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = self.encoder.encode_faces(rgb_frame, self.detector.detect_faces(rgb_frame))

        if encodings:
            safe_name = clean_name.replace(" ", "_")
            os.makedirs("known_faces", exist_ok=True)
            filename = f"known_faces/{safe_name}.jpg"

            pil_img = Image.fromarray(rgb_frame.astype(np.uint8))
            pil_img.save(filename, 'JPEG', quality=95)

            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(clean_name)
            self.save_encodings()
            return True
        return False

    def add_face_from_image(self, image_path: str, name: str) -> bool:
        """
        Add a face from an image file on disk programmatically.

        Args:
            image_path (str): Filepath to portrait.
            name (str): Person's name.

        Returns:
            bool: True if enrolled successfully; False otherwise.
        """
        clean_name = sanitize_identity_name(name)
        if not clean_name:
            return False

        try:
            with Image.open(image_path) as pil_img:
                rgb_img = pil_img.convert('RGB')
                arr = np.array(rgb_img, dtype=np.uint8)

            encoding = self.encoder.encode_face(arr)
            if encoding is not None:
                safe_name = clean_name.replace(" ", "_")
                os.makedirs("known_faces", exist_ok=True)
                filename = f"known_faces/{safe_name}.jpg"

                rgb_img.save(filename, 'JPEG', quality=95)

                self.known_face_encodings.append(encoding)
                self.known_face_names.append(clean_name)
                self.save_encodings()
                return True
        except Exception as err:
            print(f"[ERROR] Could not enroll from {image_path}: {err}")
        return False

    def enroll_composite_identity(
        self,
        name: str,
        composite_embedding: np.ndarray,
        reference_portrait: Optional[np.ndarray] = None
    ) -> bool:
        """
        Enroll an identity using a multi-angle composite embedding (Apple Face ID mode).
        
        Args:
            name (str): Person's full name.
            composite_embedding (np.ndarray): 128-d L2-normalized composite embedding.
            reference_portrait (np.ndarray, optional): Reference image in BGR or RGB format.
            
        Returns:
            bool: True if enrolled and saved successfully; False otherwise.
        """
        clean_name = sanitize_identity_name(name)
        if not clean_name or clean_name == "Unknown":
            return False

        if composite_embedding is None or len(composite_embedding) != 128:
            return False

        try:
            safe_name = clean_name.replace(" ", "_")
            os.makedirs("known_faces", exist_ok=True)
            filename = f"known_faces/{safe_name}.jpg"

            # Save reference image if supplied
            if reference_portrait is not None and reference_portrait.size > 0:
                # Ensure RGB format for Pillow saving
                if len(reference_portrait.shape) == 3 and reference_portrait.shape[2] == 3:
                    # Check if BGR; assume OpenCV BGR input
                    rgb_frame = cv2.cvtColor(reference_portrait, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = reference_portrait
                pil_img = Image.fromarray(rgb_frame.astype(np.uint8))
                pil_img.save(filename, 'JPEG', quality=95)

            # Check if identity already exists and update, or append new
            if clean_name in self.known_face_names:
                idx = self.known_face_names.index(clean_name)
                self.known_face_encodings[idx] = np.array(composite_embedding, dtype=np.float64)
            else:
                self.known_face_names.append(clean_name)
                self.known_face_encodings.append(np.array(composite_embedding, dtype=np.float64))

            # Persist encrypted encodings and rebuild spatial KD-tree index
            self.save_encodings()
            return True
        except Exception as err:
            print(f"[ERROR] Failed to enroll composite identity '{name}': {err}")
            return False

    def remove_face(self, name: str) -> bool:
        """
        Safely remove an identity and their reference photos.
        Sanitizes name parameter to prevent path traversal vulnerabilities.

        Args:
            name (str): Person name to delete.

        Returns:
            bool: True if identity was located and removed; False otherwise.
        """
        clean_name = sanitize_identity_name(name)
        if clean_name in self.known_face_names:
            idx = self.known_face_names.index(clean_name)
            del self.known_face_names[idx]
            del self.known_face_encodings[idx]

            # Secure file deletion constrained to known_faces folder
            base_dir = Path("known_faces").resolve()
            safe_filename = clean_name.replace(" ", "_")

            for candidate in [f"{clean_name}.jpg", f"{safe_filename}.jpg"]:
                cand_path = (base_dir / candidate).resolve()
                if cand_path.is_relative_to(base_dir) and cand_path.exists():
                    try:
                        cand_path.unlink()
                    except Exception as err:
                        print(f"[WARNING] Could not delete image file {cand_path}: {err}")

            self.save_encodings()
            self.vector_index.build(self.known_face_encodings, self.known_face_names)
            return True
        return False

    def process_frame(
        self,
        frame: np.ndarray,
        draw_overlays: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Unified end-to-end computer vision processing pipeline for a single frame:
        1. Downscales frame (0.25x) for low-latency inference.
        2. Locates faces via configured FaceDetector (HOG / YuNet / Haar / CNN).
        3. Generates 128-dimensional embedding vectors using FaceEncoder.
        4. Matches identities in O(log N) using spatial VectorIndex (cKDTree).
        5. Computes multi-modal liveness and 3D head pose via AntiSpoofEngine.
        6. Renders color-coded bounding boxes, Euler telemetry, and privacy blur.

        Args:
            frame (np.ndarray): Input video frame (BGR format).
            draw_overlays (bool): Whether to render visual overlays onto frame.

        Returns:
            tuple: (processed_frame, telemetry_dict)
        """
        if frame is None or frame.size == 0:
            return frame, {
                'names': [], 'locations': [], 'confidences': [],
                'liveness': [], 'poses': [], 'challenges': [],
                'num_faces': 0, 'num_enrolled': len(self.known_face_names)
            }

        h_f, w_f = frame.shape[:2]

        # Motion gating optimization: if scene is static and no face is currently tracked, skip heavy inference
        if self.motion_detector.enabled:
            has_motion, motion_area, _ = self.motion_detector.detect_motion(frame)
            if not has_motion and len(self.face_locations) == 0:
                output_frame = frame.copy() if draw_overlays else frame
                if draw_overlays:
                    cv2.putText(
                        output_frame,
                        "Motion: IDLE (Power Save Mode)",
                        (10, h_f - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (100, 200, 100),
                        1,
                        cv2.LINE_AA
                    )
                telemetry = {
                    'names': [], 'locations': [], 'confidences': [],
                    'liveness': [], 'poses': [], 'challenges': [],
                    'num_faces': 0, 'num_enrolled': len(self.known_face_names),
                    'motion_active': False,
                    'savings_ratio': round(self.motion_detector.savings_ratio, 1)
                }
                return output_frame, telemetry

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        self.face_locations = self.detector.detect_faces(rgb_small_frame)
        self.face_encodings = []
        self.face_names = []
        self.face_confidences = []
        self.face_liveness = []

        if self.face_locations:
            import face_recognition
            self.face_encodings = self.encoder.encode_faces(rgb_small_frame, self.face_locations)
            landmarks_list = face_recognition.face_landmarks(rgb_small_frame, self.face_locations)

            for idx, enc in enumerate(self.face_encodings):
                name, conf = self.recognize_face(enc)
                self.face_names.append(name)
                self.face_confidences.append(conf)

                top_s, right_s, bottom_s, left_s = self.face_locations[idx]
                y1, y2 = max(0, top_s * 4), min(h_f, bottom_s * 4)
                x1, x2 = max(0, left_s * 4), min(w_f, right_s * 4)
                face_crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else None

                lms = landmarks_list[idx] if idx < len(landmarks_list) else None
                liveness_res = self.anti_spoof.verify_liveness(
                    name, face_crop, lms, frame_shape=(h_f, w_f), scale=4.0
                )
                self.face_liveness.append(liveness_res)

                # Webhook security alert dispatch
                if self.webhook_dispatcher.enabled:
                    snapshot_img = face_crop if face_crop is not None else frame
                    if name == "Unknown":
                        self.webhook_dispatcher.dispatch(
                            event_type=SecurityEventType.UNKNOWN_FACE,
                            details="Unregistered subject sighted in camera view",
                            subject_name="Unknown",
                            confidence=0.0,
                            frame_bgr=snapshot_img
                        )
                    elif liveness_res and not liveness_res.is_live:
                        self.webhook_dispatcher.dispatch(
                            event_type=SecurityEventType.SPOOF_ATTEMPT,
                            details=f"Presentation attack spoof detected: {liveness_res.spoof_reason or 'Failed Liveness'}",
                            subject_name=name,
                            confidence=float(conf),
                            frame_bgr=snapshot_img
                        )

        output_frame = frame.copy() if draw_overlays else frame
        if draw_overlays:
            self.draw_results(output_frame)

        telemetry = {
            'names': list(self.face_names),
            'locations': list(self.face_locations),
            'confidences': list(self.face_confidences),
            'liveness': list(self.face_liveness),
            'poses': [r.head_pose for r in self.face_liveness],
            'challenges': [r.challenge_prompt for r in self.face_liveness],
            'num_faces': len(self.face_locations),
            'num_enrolled': len(self.known_face_names),
            'motion_active': True,
            'savings_ratio': round(self.motion_detector.savings_ratio, 1)
        }
        return output_frame, telemetry

    def get_known_faces(self) -> List[str]:
        """Return a copy of the list of currently registered identity names."""
        return list(self.known_face_names)
