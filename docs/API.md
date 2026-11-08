# 🛠️ Face Recognition System: Developer API Reference

This document provides developer reference documentation for the modules in `src/`.

---

## 📦 Architecture Overview

```
src/
├── face_recognition_system.py # Primary unified recognition controller
├── face_detector.py           # Face detection wrappers (HOG / Haar)
├── face_encoder.py            # 128-dimensional biometric embedding engine
├── database_manager.py        # SQLite persistence layer
└── utils.py                   # Shared utilities (logging, configs, FPS)
```

---

## 1. `FaceRecognitionSystem` (`src/face_recognition_system.py`)

Primary controller encapsulating video ingestion, multi-face tracking, and confidence scoring.

### Methods

* `load_known_faces(images_folder="known_faces") -> bool`  
  Loads images from disk, converts color spaces to standard RGB via PIL, and computes 128-d biometric embeddings.
* `save_encodings(filename="data/face_encodings.pkl") -> None`  
  Saves computed face encodings and corresponding identities.
* `load_encodings(filename="data/face_encodings.pkl") -> bool`  
  Loads cached encodings from disk.
* `recognize_face(face_encoding: np.ndarray) -> Tuple[str, float]`  
  Matches a single 128-d embedding against known embeddings using Euclidean distance.  
  Returns `(name, confidence_percentage)`.
* `run_recognition(video_source=0) -> None`  
  Spawns OpenCV capture loop with real-time video window and HUD telemetry.
* `process_frame(frame: np.ndarray) -> Tuple[np.ndarray, dict]`  
  Processes an individual video frame and returns the annotated frame with telemetry dictionary.

---

## 2. `FaceDetector` (`src/face_detector.py`)

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
