# 🛠️ Face Recognition System: Developer API Reference

This document provides developer reference documentation for the modules in `src/`.

---

## 📦 Architecture Overview

```
src/
├── face_recognition_system.py # Primary unified recognition controller
├── face_detector.py           # Face detection wrappers (HOG / Haar)
├── face_encoder.py            # 128-dimensional biometric embedding engine
├── anti_spoofing.py           # Liveness verification, 3D head pose & challenge-response
├── security.py                # Cryptographic storage & path traversal guards
├── database_manager.py        # SQLite persistence layer with chained SHA-256 audit log
└── utils.py                   # Shared utilities (logging, configs, FPS, IST time)
```

---

## 1. `FaceRecognitionSystem` (`src/face_recognition_system.py`)

Primary controller encapsulating video ingestion, multi-face tracking, and confidence scoring.

### Methods

* `load_known_faces(images_folder="known_faces") -> bool`  
  Loads images from disk, converts color spaces to standard RGB via PIL, and computes 128-d biometric embeddings.
* `save_encodings(filename="data/face_encodings.dat") -> None`  
  Saves computed face encodings using AES-authenticated Fernet encryption (`SecurityManager.save_encrypted_encodings`).
* `load_encodings(filename="data/face_encodings.dat") -> bool`  
  Loads and decrypts cached face encodings.
* `recognize_face(face_encoding: np.ndarray) -> Tuple[str, float]`  
  Matches a single 128-d embedding against known embeddings using Euclidean distance.  
  Returns `(name, confidence_percentage)`.
* `run_recognition(video_source=0) -> None`  
  Spawns OpenCV capture loop with real-time video window, HUD telemetry, and 3D head pose overlays.
* `process_frame(frame: np.ndarray) -> Tuple[np.ndarray, dict]`  
  Processes an individual video frame and returns the annotated frame with telemetry dictionary.

---

## 2. `AntiSpoofEngine` (`src/anti_spoofing.py`)

Multi-layered biometric anti-spoofing engine providing presentation attack defense.

### Key Classes

* `AntiSpoofEngine(config: Optional[dict] = None)`
  * `compute_ear(eye_landmarks: List[Tuple[int, int]]) -> float`: Calculates Eye Aspect Ratio from 6 eyelid coordinates.
  * `analyze_texture_frequency(face_crop: np.ndarray) -> float`: Analyzes 2D FFT magnitude spectrum to detect digital screen moiré patterns.
  * `estimate_head_pose(landmarks, frame_shape, scale=1.0) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple]]`: Solves Perspective-n-Point (`cv2.solvePnP`) to estimate Euler angles (Yaw, Pitch, Roll) and project a 3D nose ray vector.
  * `verify_liveness(subject_id, face_crop, facial_landmarks=None, frame_shape=(480, 640), scale=1.0) -> LivenessResult`: Unified decision matrix returning `LivenessResult` telemetry.
  * `apply_privacy_blur(frame, face_locations, face_names) -> np.ndarray`: Applies Gaussian blur to unregistered bystanders.

* `ChallengeResponseTracker(challenge_timeout=5.0, initial_challenge=None)`
  * `update(yaw, pitch, roll, current_blinks, current_time=None) -> str`: Updates state machine against active motion challenge (`TURN_LEFT`, `TURN_RIGHT`, `NOD_DOWN`, `TILT_UP`, `BLINK`).
  * `reset(challenge=None, current_blinks=0, current_time=None)`: Resets challenge countdown and selects next challenge.
  * `is_verified() -> bool`: Returns True if challenge criteria are satisfied within timeout.

* `BlinkTracker(min_closed_frames=2)`
  * `update(current_ear, ear_threshold=0.22) -> int`: State machine tracking eye closure cycles.

---

## 3. `SecurityManager` (`src/security.py`)

Cryptographic storage and input sanitization layer (CWE-502, CWE-22 mitigations).

### Methods

* `sanitize_identity_name(name: str) -> str`: Enforces strict character allowlist `[A-Za-z0-9_-]` and prevents directory traversal attacks.
* `get_safe_portrait_path(name: str, base_dir="known_faces") -> str`: Resolves canonical path and verifies it resides within `base_dir`.
* `save_encrypted_encodings(filepath, encodings, names)`: Serializes vectors via `np.savez_compressed` and encrypts with Fernet AES.
* `load_encrypted_encodings(filepath) -> Tuple[List[np.ndarray], List[str]]`: Decrypts and deserializes authenticated biometric encodings.

---

## 4. `FaceDetector` (`src/face_detector.py`)

Dedicated detection abstraction supporting both Dlib HOG and OpenCV Haar Cascade methods.

### Methods

* `detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]`  
  Detects faces using HOG/CNN. Returns bounding box coordinates `[(top, right, bottom, left)]`.
* `detect_faces_opencv(image: np.ndarray) -> List[Tuple[int, int, int, int]]`  
  Ultra-fast detection fallback using OpenCV Haar Cascade.
* `extract_face(image: np.ndarray, location: Tuple[int, int, int, int]) -> np.ndarray`  
  Crops and extracts subregion array containing the detected face.

---

## 3. `FaceEncoder` (`src/face_encoder.py`)

Encapsulates 128-dimensional metric learning vector calculations.

### Methods

* `encode_face(image: np.ndarray, face_location=None) -> Optional[np.ndarray]`  
  Generates 128-d float embedding vector for a single face.
* `compare_faces(known: np.ndarray, unknown: np.ndarray, tolerance=0.6) -> bool`  
  Evaluates Euclidean distance against tolerance threshold.

---

## 4. `DatabaseManager` (`src/database_manager.py`)

SQLite relational storage manager.

### Tables
* `persons (id, name, added_date, image_path, encoding_path)`
* `recognition_logs (id, person_id, timestamp, confidence, image_path)`
* `unknown_faces (id, timestamp, image_path, encoding_path)`

### Methods
* `add_person(name: str, image_path: str = None) -> int`
* `log_recognition(person_name: str, confidence: float, image_path: str = None) -> None`
* `get_recognition_history(person_name: str = None, limit: int = 100) -> pd.DataFrame`
* `export_to_csv(output_path: str = "data/recognition_log.csv") -> str`
