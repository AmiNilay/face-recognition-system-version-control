"""
Primary Face Recognition System Module

This module coordinates the complete computer vision pipeline:
1. Ingesting and decoding reference photos from the known_faces/ folder.
2. Generating and caching 128-dimensional biometric embeddings.
3. Capturing real-time video feeds from camera devices.
4. Downscaling frames for accelerated inference.
5. Computing Euclidean face distance and mapping distance to confidence percentages.
6. Drawing bounding overlays with confidence scores and telemetry.
7. Managing facial additions, removals, and frame captures with IST timestamps.
"""

import os
import re
import cv2
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Optional
from src.utils import get_ist_filename_timestamp, format_ist_timestamp

def sanitize_identity_name(name: str) -> str:
    """
    Sanitize a human identity name to prevent path traversal and filesystem errors.
    Removes dangerous characters like '..', '/', '\\', and control characters.
    """
    # Allow alphanumeric, spaces, hyphens, and underscores
    clean = re.sub(r'[^a-zA-Z0-9_\- ]', '', name.strip())
    # Collapse multiple spaces or underscores
    clean = re.sub(r'\s+', ' ', clean)
    return clean

class FaceRecognitionSystem:
    """
    Core facial recognition controller and video processing engine.
    """

    def __init__(self):
        """Initialize empty collections for known faces and detection telemetry."""
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.face_locations: List[Tuple[int, int, int, int]] = []
        self.face_encodings: List[np.ndarray] = []
        self.face_names: List[str] = []
        self.face_confidences: List[float] = []

    def load_known_faces(self, images_folder: str = "known_faces") -> bool:
        """
        Load reference portrait images from a directory and compute their 128-d embeddings.
        Supports standard formats: JPEG, PNG, WEBP, BMP, TIFF, GIF.

        Args:
            images_folder (str): Directory containing labeled reference photos.

        Returns:
            bool: True if one or more valid face encodings were loaded; False otherwise.
        """
        import face_recognition

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

                # Generate 128-d encoding (num_jitters=1 provides fast single-pass encoding)
                encodings = face_recognition.face_encodings(image_array, num_jitters=1)

                if encodings:
                    face_encoding = encodings[0]
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

        print(f"\n[INFO] Total enrolled identities: {len(self.known_face_names)}")
        return len(self.known_face_names) > 0

    def save_encodings(self, filename: str = "data/face_encodings.pkl"):
        """
        Serialize enrolled identities and embeddings to disk for faster subsequent loads.

        Args:
            filename (str): Destination cache file path.
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "names": self.known_face_names,
            "encodings": self.known_face_encodings
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"[INFO] Face encodings saved to: {filename}")

    def load_encodings(self, filename: str = "data/face_encodings.pkl") -> bool:
        """
        Load pre-computed face embeddings from cache.

        Args:
            filename (str): Path to stored encodings file.

        Returns:
            bool: True if loaded successfully; False otherwise.
        """
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                self.known_face_names = data.get("names", [])
                self.known_face_encodings = data.get("encodings", [])
                print(f"[INFO] Loaded {len(self.known_face_names)} cached encodings from {filename}")
                return True
            except Exception as err:
                print(f"[WARNING] Could not load cached encodings: {err}")
                return False
        return False

    def recognize_face(self, face_encoding: np.ndarray, tolerance: float = 0.6) -> Tuple[str, float]:
        """
        Identify a face encoding by calculating Euclidean distances to all known identities.

        Beginner Concept:
        - Euclidean Distance: Measures distance between two vectors.
          Distance 0.0 means identical. Distance > 0.6 means distinct faces.
        - Confidence Score: Inverted percentage scale:
          Confidence = max(0.0, (1.0 - distance)) * 100

        Args:
            face_encoding (np.ndarray): 128-d vector of newly detected face.
            tolerance (float): Match threshold. Default 0.6.

        Returns:
            tuple[str, float]: (matched_person_name, confidence_percentage).
        """
        import face_recognition

        name = "Unknown"
        confidence = 0.0

        if self.known_face_encodings:
            # Compute Euclidean distances against all enrolled identities
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_idx = int(np.argmin(distances))
            min_distance = float(distances[best_match_idx])

            # Convert distance into an intuitive confidence percentage
            confidence = max(0.0, (1.0 - min_distance)) * 100.0

            # Stricter verification against tolerance
            if min_distance < tolerance:
                name = self.known_face_names[best_match_idx]
            else:
                name = "Unknown"
                confidence = 0.0

        return name, confidence

    def run_recognition(self, video_source=0):
        """
        Run the interactive command-line recognition loop on a video stream.

        Args:
            video_source: Camera index (0) or path to video file.
        """
        import face_recognition

        video_capture = cv2.VideoCapture(video_source)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
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

            # Performance Optimization:
            # Process recognition only on every 3rd frame to conserve CPU
            if frame_count % 3 == 0:
                # Downscale frame to 25% (1/4 size) for rapid inference
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Locate faces in downscaled frame
                self.face_locations = face_recognition.face_locations(
                    rgb_small_frame, 
                    number_of_times_to_upsample=1, 
                    model="hog"
                )

                if self.face_locations:
                    self.face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, 
                        self.face_locations, 
                        num_jitters=1
                    )
                    self.face_names = []
                    self.face_confidences = []

                    for enc in self.face_encodings:
                        name, conf = self.recognize_face(enc)
                        self.face_names.append(name)
                        self.face_confidences.append(conf)
                else:
                    self.face_encodings = []
                    self.face_names = []
                    self.face_confidences = []

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
        Render color-coded bounding boxes and confidence readouts onto a frame.

        Color Code:
        - Green (High confidence: >= 70%)
        - Orange (Medium confidence: 50% - 69%)
        - Yellow (Low confidence: < 50%)
        - Red (Unknown / Unregistered)
        """
        for i, ((top, right, bottom, left), name) in enumerate(zip(self.face_locations, self.face_names)):
            # Upscale coordinates by 4 to map back from 0.25 downscaled frame
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

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
        import face_recognition

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        temp_locations = face_recognition.face_locations(rgb_frame)

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
        encodings = face_recognition.face_encodings(rgb_frame, temp_locations, num_jitters=1)
        if encodings:
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(clean_name)
            self.save_encodings()
            print(f"[SUCCESS] Enrolled {clean_name} into known faces directory.")
        else:
            print("[ERROR] Could not compute face encoding.")

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
                # Verify cand_path is strictly within base_dir (prevents traversal)
                if cand_path.is_relative_to(base_dir) and cand_path.exists():
                    try:
                        cand_path.unlink()
                    except Exception as err:
                        print(f"[WARNING] Could not delete image file {cand_path}: {err}")

            self.save_encodings()
            return True
        return False

    def get_known_faces(self) -> List[str]:
        """Return a copy of the list of currently registered identity names."""
        return list(self.known_face_names)
