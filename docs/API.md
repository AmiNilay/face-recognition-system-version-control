# 🛠️ Face Recognition System: Developer API Reference

This document provides developer reference documentation for the modules in `src/`.

---

## 📦 Architecture Overview

```
src/
├── face_recognition_system.py # Primary unified recognition controller
├── face_detector.py           # Face detection wrappers (HOG, YuNet DNN, Haar, CNN)
├── face_encoder.py            # 128-dimensional biometric embedding engine
├── vector_index.py            # O(log N) spatial nearest-neighbor search (cKDTree)
├── anti_spoofing.py           # Liveness verification, 3D head pose & challenge-response
├── audio_announcer.py         # Asynchronous offline text-to-speech voice synthesizer
├── reporting.py               # Attendance analytics & Apple-style HTML/CSV reporting
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
  Loads and decrypts cached face encodings, automatically rebuilding the spatial `VectorIndex`.
* `recognize_face(face_encoding: np.ndarray, tolerance=None) -> Tuple[str, float]`  
  Matches a single 128-d embedding in $O(\log N)$ time using spatial `VectorIndex`. Returns `(name, confidence_percentage)`.
* `process_frame(frame: np.ndarray, draw_overlays=True) -> Tuple[np.ndarray, dict]`  
  Unified end-to-end processing pipeline for a single video frame. Returns the annotated frame and telemetry dictionary.
* `run_recognition(video_source=0) -> None`  
  Spawns OpenCV capture loop with real-time video window, HUD telemetry, and 3D head pose overlays.

---

## 2. `VectorIndex` (`src/vector_index.py`)

Accelerated spatial index for 128-dimensional biometric embeddings.

### Methods

* `build(encodings: List[np.ndarray], names: List[str]) -> bool`  
  Constructs a spatial `cKDTree` binary partition index over enrolled 128-d vectors.
* `query(query_encoding: np.ndarray, tolerance: float = 0.6) -> Tuple[str, float, float]`  
  Searches for the closest neighbor in $O(\log N)$ time. Returns `(matched_name, confidence_percentage, distance)`.
* `clear() -> None`  
  Resets and empties the index.

---

## 3. `FaceDetector` (`src/face_detector.py`)

Detection engine supporting HOG, YuNet DNN, Haar cascades, and CNN.

### Methods

* `detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]`  
  Dispatches to configured engine (`hog`, `yunet`, `haar`, `cnn`). Returns bounding box coordinates `[(top, right, bottom, left)]`.
* `detect_faces_yunet(image: np.ndarray) -> List[Tuple[int, int, int, int]]`  
  Lightweight 60+ FPS deep learning detection via OpenCV `FaceDetectorYN` ONNX with automatic Haar fallback.
* `detect_faces_opencv(image: np.ndarray) -> List[Tuple[int, int, int, int]]`  
  Zero-dependency detection via OpenCV Haar cascade classifier.
* `extract_face(image: np.ndarray, location: Tuple[int, int, int, int]) -> np.ndarray`  
  Crops and extracts subregion containing the detected face with boundary clamping.

---

## 4. `AntiSpoofEngine` (`src/anti_spoofing.py`)

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

## 5. `SecurityManager` (`src/security.py`)

Cryptographic storage and input sanitization layer (CWE-502, CWE-22 mitigations).

### Methods

* `sanitize_identity_name(name: str) -> str`: Enforces strict character allowlist `[A-Za-z0-9_-]` and prevents directory traversal attacks.
* `get_safe_portrait_path(name: str, base_dir="known_faces") -> str`: Resolves canonical path and verifies it resides within `base_dir`.
* `save_encrypted_encodings(filepath, encodings, names)`: Serializes vectors via `np.savez_compressed` and encrypts with Fernet AES.
* `load_encrypted_encodings(filepath) -> Tuple[List[np.ndarray], List[str]]`: Decrypts and deserializes authenticated biometric encodings.

---

## 6. `FaceEncoder` (`src/face_encoder.py`)

Encapsulates 128-dimensional metric learning vector calculations.

### Methods

* `encode_face(image: np.ndarray, face_location=None) -> Optional[np.ndarray]`  
  Generates 128-d float embedding vector for a single face.
* `encode_faces(image: np.ndarray, face_locations) -> List[np.ndarray]`  
  Generates 128-d embedding vectors for multiple detected faces.
* `compare_faces(known: np.ndarray, unknown: np.ndarray, tolerance=0.6) -> bool`  
  Evaluates Euclidean distance against tolerance threshold.

---

## 7. `DatabaseManager` (`src/database_manager.py`)

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

---

## 8. `AudioAnnouncer` (`src/audio_announcer.py`)

Asynchronous offline speech synthesizer engine for multimodal voice greetings upon verification.

### Methods

* `announce_recognition(name: str, current_time=None) -> bool`: Queues an audio greeting (`"Welcome, {name}"`) if cooldown window has elapsed.
* `announce_custom(message: str) -> bool`: Queues an arbitrary high-priority voice message.
* `set_enabled(enabled: bool) -> None`: Dynamically enables or disables voice announcements.
* `stop() -> None`: Gracefully terminates background audio worker thread.

---

## 9. `AttendanceReporter` (`src/reporting.py`)

Multi-format attendance analytics and executive reporting engine.

### Methods

* `generate_daily_summary(target_date=None) -> Dict[str, Any]`: Computes aggregate attendance statistics (total enrolled, present, absent, rate, first seen, last seen, mean confidence).
* `export_csv(summary, output_path="data/attendance_summary.csv") -> str`: Exports attendance breakdown to a structured CSV file.
* `export_html(summary, output_path="data/attendance_report.html") -> str`: Generates presentation-grade, Apple-styled executive HTML report with responsive cards and printable CSS.

---

## 10. `FastAPI Edge Microservice` (`src/api.py`)

High-performance, async REST and WebSocket microservice for edge devices, mobile clients, and third-party web dashboard integration.

### Execution
```bash
# Start microservice on port 8000
python main.py --api --port 8000

# Or run the Windows batch launcher
run_api.bat
```

* **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`
* **Interactive ReDoc UI:** `http://127.0.0.1:8000/redoc`
* **OpenAPI 3.1 JSON Specification:** `http://127.0.0.1:8000/openapi.json`

### Endpoints

#### 1. `GET /api/v1/health`
Returns service operational status, active computer vision models, anti-spoofing status, and enrolled identity count.
* **Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp_ist": "2026-11-08 23:00:00 IST",
  "enrolled_count": 5,
  "enrolled_identities": ["Alice", "Bob", "Charlie"],
  "detector_model": "hog",
  "encoding_model": "large",
  "anti_spoofing_enabled": true,
  "vector_index_active": true,
  "system_version": "2.1.0"
}
```

#### 2. `POST /api/v1/verify` (JSON) & `POST /api/v1/verify/upload` (Multipart)
Performs 128-d biometric feature extraction, $O(\log N)$ spatial KD-Tree search, EAR blink liveness, 2D FFT moiré screen replay analysis, and 3D head pose estimation.
* **JSON Payload (`POST /api/v1/verify`):**
```json
{
  "image_base64": "<base64-encoded-jpeg-or-png>",
  "auto_log_attendance": true
}
```
* **Multipart Payload (`POST /api/v1/verify/upload`):**
  * `file`: Image file (JPEG, PNG, WEBP)
  * `auto_log_attendance`: `true` or `false`
* **Response (200 OK):**
```json
{
  "success": true,
  "faces_detected": 1,
  "timestamp_ist": "2026-11-08 23:00:05 IST",
  "latency_ms": 18.45,
  "verifications": [
    {
      "name": "Alice",
      "confidence": 96.8,
      "box": { "top": 80, "right": 240, "bottom": 280, "left": 80 },
      "is_live": true,
      "liveness_status": "Verified Human",
      "head_pose": { "yaw": 2.1, "pitch": -4.3, "roll": 0.8 },
      "challenge": null
    }
  ]
}
```

#### 3. `POST /api/v1/enroll` (JSON) & `POST /api/v1/enroll/upload` (Multipart)
Remote biometric enrollment endpoint. Validates single face presence, sanitizes name (preventing path traversal), serializes encrypted `.dat` vector representations with Fernet AES, and saves reference portrait to `known_faces/`.
* **JSON Payload (`POST /api/v1/enroll`):**
```json
{
  "name": "David Miller",
  "image_base64": "<base64-encoded-portrait>"
}
```
* **Response (200 OK):**
```json
{
  "success": true,
  "message": "Identity 'David Miller' successfully enrolled into biometric database.",
  "name": "David Miller",
  "enrolled_at_ist": "2026-11-08 23:01:00 IST",
  "total_enrolled": 6
}
```

#### 4. `POST /api/v1/enroll/composite` (JSON)
Enrolls an identity using a synthesized $L_2$-normalized composite embedding from multiple facial perspectives (Frontal, Turn Left, Turn Right, Tilt Up, Nod Down).
* **JSON Payload (`POST /api/v1/enroll/composite`):**
```json
{
  "name": "Sarah Connor",
  "angles": [
    { "angle": "FRONTAL", "image_base64": "<base64-image>" },
    { "angle": "TURN_LEFT", "image_base64": "<base64-image>" },
    { "angle": "TURN_RIGHT", "image_base64": "<base64-image>" }
  ]
}
```
* **Response (200 OK):**
```json
{
  "success": true,
  "message": "Identity 'Sarah Connor' enrolled with 3-angle composite Face ID embedding.",
  "name": "Sarah Connor",
  "enrolled_at_ist": "2026-11-08 23:20:00 IST",
  "total_enrolled": 7
}
```

#### 5. `GET /api/v1/attendance`
Retrieves daily attendance logs, absentees, and executive metrics in Indian Standard Time (IST).
* **Query Parameters:**
  * `date`: Optional date in `YYYY-MM-DD` format (defaults to current date in IST).
  * `format`: Response format: `json` (default), `csv`, or `html`.
* **Response (200 OK):** Returns structured JSON or downloadable CSV/HTML.

#### 6. `WebSocket /ws/stream`
Low-latency bidirectional WebSocket connection for live camera streams and edge devices.
* **Client sends:** Binary JPEG bytes OR JSON `{"image_base64": "..."}`
* **Server returns:** Real-time JSON telemetry with bounding boxes, match confidences, liveness verdicts, and latency.

---

## 11. `CompositeEnrollment` (`src/composite_enrollment.py`)

Multi-angle biometric enrollment engine inspired by Apple Face ID design principles.

### Key Classes & Functions

* `compute_l2_normalized_composite(embeddings: List[np.ndarray]) -> np.ndarray`  
  Calculates mathematically optimal $L_2$-normalized Fréchet centroid vector:
  $$\mathbf{e}_{\text{composite}} = \frac{\sum \mathbf{e}_i}{\left\|\sum \mathbf{e}_i\right\|_2}$$
  Guarantees $\|\mathbf{e}_{\text{composite}}\|_2 = 1.0$.

* `measure_frame_sharpness(bgr_crop: np.ndarray) -> float`  
  Computes the variance of the Laplacian operator to detect and reject blurry frames during enrollment.

* `CompositeEnrollmentSession(identity_name: str, required_stability_frames=3, min_sharpness=25.0)`  
  State machine tracking user progression through 5 canonical head poses (`FRONTAL`, `TURN_LEFT`, `TURN_RIGHT`, `TILT_UP`, `NOD_DOWN`).
  * `process_frame(frame, head_pose, face_encoding, face_crop) -> dict`: Evaluates live head orientation against active angle criteria.
  * `generate_composite_portrait() -> np.ndarray`: Produces a 3-column reference contact sheet (Left $\rightarrow$ Frontal $\rightarrow$ Right).

* `render_apple_face_id_hud(frame, session, center_xy=None, radius=140) -> np.ndarray`  
  Draws Apple Face ID circular HUD with 36 radial tick marks that dynamically illuminate from dark gray (`#3a3a3c`) to Apple Mint Green (`#30d158`) as each pose is acquired.


