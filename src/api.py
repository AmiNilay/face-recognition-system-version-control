"""
Biometric Face Recognition & Anti-Spoofing Edge API
Module: src/api.py

Enterprise Edge Microservice built with FastAPI, OpenCV, and Dlib:
1. High-Performance REST Endpoints:
   - GET  /api/v1/health: System operational health, models, and enrolled identity counts.
   - POST /api/v1/verify: Biometric verification with EAR blink & 2D FFT moire anti-spoofing.
   - POST /api/v1/enroll: Remote biometric facial enrollment with AES encrypted vector storage.
   - GET  /api/v1/attendance: Real-time IST attendance logs, absentees, and executive metrics.
2. Real-Time Telemetry Streaming:
   - WebSocket /ws/stream: Low-latency frame streaming and JSON telemetry for edge clients.
3. Interactive API Documentation:
   - Swagger UI at /docs
   - ReDoc at /redoc
   - OpenAPI Schema at /openapi.json

All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code, logs, and telemetry. Developer: Nilay Naha.
"""

import os
import io
import time
import base64
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fastapi import (
    FastAPI, APIRouter, File, UploadFile, Form, Query, 
    HTTPException, status, WebSocket, WebSocketDisconnect, Depends
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from src.face_recognition_system import FaceRecognitionSystem
from src.database_manager import DatabaseManager
from src.reporting import AttendanceReporter
from src.security import SecurityManager
from src.composite_enrollment import compute_l2_normalized_composite
from src.utils import format_ist_timestamp, get_ist_date_string, load_config


# ---------------------------------------------------------------------------
# Pydantic Request & Response Data Schemas (V2)
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Coordinates of detected face in video frame."""
    top: int = Field(..., description="Top edge pixel coordinate")
    right: int = Field(..., description="Right edge pixel coordinate")
    bottom: int = Field(..., description="Bottom edge pixel coordinate")
    left: int = Field(..., description="Left edge pixel coordinate")


class HeadPoseAngles(BaseModel):
    """Estimated 3D head pose orientation in degrees."""
    yaw: float = Field(..., description="Horizontal rotation in degrees")
    pitch: float = Field(..., description="Vertical tilt in degrees")
    roll: float = Field(..., description="Lateral tilt in degrees")


class FaceVerificationDetail(BaseModel):
    """Detailed verification result for a single detected face."""
    name: str = Field(..., description="Recognized identity or 'Unknown'")
    confidence: float = Field(..., description="Biometric match confidence percentage (0-100)")
    box: BoundingBox = Field(..., description="Bounding box coordinates")
    is_live: bool = Field(..., description="True if verified as a live human subject")
    liveness_status: str = Field(..., description="Human-readable anti-spoofing verdict")
    head_pose: Optional[HeadPoseAngles] = Field(None, description="Euler head pose angles")
    challenge: Optional[str] = Field(None, description="Current interactive challenge prompt")


class VerifyResponse(BaseModel):
    """Response payload for face verification endpoint."""
    success: bool = Field(True, description="Request execution status")
    faces_detected: int = Field(..., description="Total faces detected in the image")
    timestamp_ist: str = Field(..., description="Processing timestamp in IST")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    verifications: List[FaceVerificationDetail] = Field(..., description="List of recognized faces")


class VerifyRequest(BaseModel):
    """JSON payload for face verification using base64 encoded image."""
    image_base64: str = Field(..., description="Base64-encoded image (JPEG, PNG, or WEBP)")
    auto_log_attendance: bool = Field(True, description="Automatically log recognized faces to database")


class EnrollRequest(BaseModel):
    """JSON payload for enrolling a new face using base64 encoded image."""
    name: str = Field(..., min_length=2, max_length=50, description="Full name of the person")
    image_base64: str = Field(..., description="Base64-encoded portrait image")


class EnrollResponse(BaseModel):
    """Response payload following face enrollment."""
    success: bool = Field(..., description="Enrollment status")
    message: str = Field(..., description="Status summary message")
    name: str = Field(..., description="Sanitized identity name")
    enrolled_at_ist: str = Field(..., description="Enrollment timestamp in IST")
    total_enrolled: int = Field(..., description="Total registered identities after enrollment")


class CompositeAngleInput(BaseModel):
    """Single facial angle snapshot for composite enrollment."""
    angle: str = Field(..., description="Angle label: FRONTAL, TURN_LEFT, TURN_RIGHT, TILT_UP, NOD_DOWN")
    image_base64: str = Field(..., description="Base64-encoded image captured at this angle")


class CompositeEnrollRequest(BaseModel):
    """Payload for multi-angle Face ID composite enrollment."""
    name: str = Field(..., min_length=2, max_length=50, description="Full name of the person")
    angles: List[CompositeAngleInput] = Field(..., min_length=2, description="List of multi-angle images (min 2, rec 5)")


class HealthResponse(BaseModel):
    """System health and operational status telemetry."""
    status: str = Field("healthy", description="Operational service status")
    timestamp_ist: str = Field(..., description="Current system time in IST")
    enrolled_count: int = Field(..., description="Total enrolled identities in vector index")
    enrolled_identities: List[str] = Field(..., description="Registered identity names")
    detector_model: str = Field(..., description="Active face detection backend (hog, cnn, yunet)")
    encoding_model: str = Field(..., description="Facial embedding model (small, large)")
    anti_spoofing_enabled: bool = Field(..., description="Anti-spoofing engine state")
    vector_index_active: bool = Field(..., description="KD-Tree vector indexing state")
    system_version: str = Field("2.1.0", description="API microservice version")


class AttendanceAttendee(BaseModel):
    """Per-person attendance summary metrics."""
    name: str
    first_seen: str
    last_seen: str
    sightings: int
    avg_confidence: float


class AttendanceSummaryResponse(BaseModel):
    """Daily attendance executive summary."""
    date: str
    generated_at_ist: str
    total_enrolled: int
    total_present: int
    total_absent: int
    attendance_rate: float
    overall_avg_confidence: float
    total_sightings: int
    attendees: List[AttendanceAttendee]
    absentees: List[str]


# ---------------------------------------------------------------------------
# Helper Utility Functions for Image Conversion
# ---------------------------------------------------------------------------

def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw byte array into an OpenCV BGR numpy image array.
    
    Args:
        image_bytes: Raw bytes representing JPEG, PNG, or WEBP image.
        
    Returns:
        np.ndarray: Decoded image in OpenCV BGR format.
        
    Raises:
        HTTPException: If the image cannot be decoded.
    """
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image data provided."
        )

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None or img_bgr.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Ensure image is a valid JPEG, PNG, or WEBP file."
        )

    return img_bgr


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64 string (with or without data URI prefix) into an OpenCV BGR image.
    
    Args:
        base64_string: Base64 image string.
        
    Returns:
        np.ndarray: Decoded BGR numpy array.
    """
    cleaned = base64_string.strip()
    # Strip data URL header if present (e.g., 'data:image/jpeg;base64,...')
    if "," in cleaned:
        cleaned = cleaned.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(cleaned)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed base64 string: {str(err)}"
        )

    return decode_image_bytes(raw_bytes)


# ---------------------------------------------------------------------------
# Core FastAPI Application Factory
# ---------------------------------------------------------------------------

def create_app(system: Optional[FaceRecognitionSystem] = None,
               db_manager: Optional[DatabaseManager] = None) -> FastAPI:
    """
    Factory function to construct the FastAPI application.
    Supports dependency injection of core systems for unit testing.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup phase
        if not hasattr(app.state, "system") or app.state.system is None:
            app.state.system = FaceRecognitionSystem()
            # Attempt to load encodings from encrypted store or reference photos
            if not app.state.system.load_encodings():
                if app.state.system.load_known_faces():
                    app.state.system.save_encodings()

        if not hasattr(app.state, "db_manager") or app.state.db_manager is None:
            app.state.db_manager = DatabaseManager()

        yield

        # Shutdown phase
        # Gracefully release any active hardware or thread resources
        pass

    app = FastAPI(
        title="Biometric Face Recognition & Anti-Spoofing Edge API",
        description=(
            "Enterprise Edge Microservice for real-time biometric face recognition, "
            "anti-spoofing liveness verification, and cryptographic attendance tracking.\n\n"
            "Developer: Nilay Naha | Timezone: Indian Standard Time (IST, UTC+05:30)"
        ),
        version="2.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Attach injected systems to state if provided
    app.state.system = system
    app.state.db_manager = db_manager

    # Enable Cross-Origin Resource Sharing (CORS) for web dashboard access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Route Handlers
    # -----------------------------------------------------------------------

    @app.get("/", tags=["General"], summary="Root API Information")
    async def root():
        """Returns API identity, status, and interactive documentation links."""
        return {
            "name": "Biometric Face Recognition & Anti-Spoofing Edge API",
            "version": "2.1.0",
            "status": "online",
            "timezone": "Indian Standard Time (IST, UTC+05:30)",
            "developer": "Nilay Naha",
            "documentation": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json"
            }
        }

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["Diagnostics"],
             summary="System Operational Health Check")
    async def get_health():
        """
        Query system operational status, model configurations, and enrolled identities.
        """
        sys_instance: FaceRecognitionSystem = app.state.system
        if sys_instance is None:
            sys_instance = FaceRecognitionSystem()
            app.state.system = sys_instance

        enrolled = sys_instance.get_known_faces()
        return HealthResponse(
            status="healthy",
            timestamp_ist=format_ist_timestamp(),
            enrolled_count=len(enrolled),
            enrolled_identities=enrolled,
            detector_model=sys_instance.model_type,
            encoding_model=sys_instance.encoding_model,
            anti_spoofing_enabled=sys_instance.anti_spoof.enabled,
            vector_index_active=sys_instance.vector_index.is_fitted(),
            system_version="2.1.0"
        )

    # Internal verification helper shared by JSON and multipart endpoints
    def _execute_verification(bgr_image: np.ndarray, auto_log: bool) -> VerifyResponse:
        start_time = time.perf_counter()
        sys_instance: FaceRecognitionSystem = app.state.system
        db: DatabaseManager = app.state.db_manager

        # Process frame through unified pipeline
        _, telemetry = sys_instance.process_frame(bgr_image, draw_overlays=False)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        names = telemetry.get('names', [])
        locations = telemetry.get('locations', [])
        confidences = telemetry.get('confidences', [])
        liveness_results = telemetry.get('liveness', [])

        verifications: List[FaceVerificationDetail] = []

        for i, name in enumerate(names):
            conf = confidences[i] if i < len(confidences) else 0.0
            # Locations are returned in (top, right, bottom, left) from 0.25 small frame
            # Scale coordinates up by 4 to map back to original frame dimensions
            top_s, right_s, bottom_s, left_s = locations[i]
            box = BoundingBox(
                top=int(top_s * 4),
                right=int(right_s * 4),
                bottom=int(bottom_s * 4),
                left=int(left_s * 4)
            )

            liveness = liveness_results[i] if i < len(liveness_results) else None
            is_live = bool(liveness.is_live) if liveness else False
            status_text = liveness.status_text if liveness else "Liveness Disabled"

            head_pose = None
            challenge = None
            if liveness:
                challenge = liveness.challenge_prompt
                if liveness.head_pose is not None:
                    yaw_val, pitch_val, roll_val = liveness.head_pose
                    head_pose = HeadPoseAngles(
                        yaw=round(float(yaw_val), 1),
                        pitch=round(float(pitch_val), 1),
                        roll=round(float(roll_val), 1)
                    )

            verifications.append(
                FaceVerificationDetail(
                    name=name,
                    confidence=round(float(conf), 2),
                    box=box,
                    is_live=is_live,
                    liveness_status=status_text,
                    head_pose=head_pose,
                    challenge=challenge
                )
            )

            # Auto-log recognized identities to attendance database if requested
            if auto_log and db is not None and name != "Unknown" and is_live:
                try:
                    db.log_recognition(person_name=name, confidence=float(conf))
                except Exception:
                    # Logging failure should not disrupt the real-time API response
                    pass

        return VerifyResponse(
            success=True,
            faces_detected=len(names),
            timestamp_ist=format_ist_timestamp(),
            latency_ms=round(latency_ms, 2),
            verifications=verifications
        )

    @app.post("/api/v1/verify", response_model=VerifyResponse, tags=["Biometrics"],
              summary="Verify Face Identity and Anti-Spoofing Liveness (JSON)")
    async def verify_face_json(payload: VerifyRequest):
        """
        Verify face identity from a base64-encoded image string.
        Performs 128-d biometric feature extraction, KD-Tree search, and anti-spoofing.
        """
        bgr_image = decode_base64_image(payload.image_base64)
        return _execute_verification(bgr_image, auto_log=payload.auto_log_attendance)

    @app.post("/api/v1/verify/upload", response_model=VerifyResponse, tags=["Biometrics"],
              summary="Verify Face Identity from File Upload (Multipart)")
    async def verify_face_upload(
        file: UploadFile = File(..., description="Portrait image file (JPEG, PNG, WEBP)"),
        auto_log_attendance: bool = Form(True, description="Auto-record attendance in database")
    ):
        """
        Verify face identity from an uploaded image file via multipart form.
        """
        raw_bytes = await file.read()
        bgr_image = decode_image_bytes(raw_bytes)
        return _execute_verification(bgr_image, auto_log=auto_log_attendance)

    # Internal enrollment helper shared by JSON and multipart endpoints
    def _execute_enrollment(name: str, bgr_image: np.ndarray) -> EnrollResponse:
        clean_name = SecurityManager.sanitize_identity_name(name)
        if not clean_name or clean_name == "Unknown":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid identity name. Only alphanumeric characters, spaces, and hyphens are permitted. Cannot enroll 'Unknown'."
            )

        sys_instance: FaceRecognitionSystem = app.state.system
        db: DatabaseManager = app.state.db_manager

        rgb_frame = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        temp_locations = sys_instance.detector.detect_faces(rgb_frame)

        if not temp_locations:
            raise HTTPException(
                status_code=422,
                detail="No human face detected in provided image. Please provide a clear frontal portrait."
            )

        if len(temp_locations) > 1:
            raise HTTPException(
                status_code=422,
                detail=f"Multiple faces ({len(temp_locations)}) detected. Enrollment image must contain exactly one face."
            )

        encodings = sys_instance.encoder.encode_faces(rgb_frame, temp_locations)
        if not encodings:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate 128-dimensional biometric embedding."
            )

        # Persist reference portrait in known_faces/
        safe_filename = clean_name.replace(" ", "_")
        known_faces_dir = Path("known_faces")
        known_faces_dir.mkdir(parents=True, exist_ok=True)
        target_path = known_faces_dir / f"{safe_filename}.jpg"

        pil_img = Image.fromarray(rgb_frame.astype(np.uint8))
        pil_img.save(str(target_path), 'JPEG', quality=95)

        # Update in-memory encodings and rebuild spatial index
        sys_instance.known_face_encodings.append(encodings[0])
        sys_instance.known_face_names.append(clean_name)
        sys_instance.save_encodings()

        # Record enrollment in relational database
        if db is not None:
            try:
                db.add_person(clean_name, str(target_path))
            except Exception:
                pass

        return EnrollResponse(
            success=True,
            message=f"Identity '{clean_name}' successfully enrolled into biometric database.",
            name=clean_name,
            enrolled_at_ist=format_ist_timestamp(),
            total_enrolled=len(sys_instance.known_face_names)
        )

    @app.post("/api/v1/enroll", response_model=EnrollResponse, tags=["Biometrics"],
              summary="Enroll New Biometric Identity (JSON)")
    async def enroll_face_json(payload: EnrollRequest):
        """
        Enroll a new identity remotely using a base64 encoded portrait image.
        Encrypts vector embedding with AES Fernet and saves reference photo.
        """
        bgr_image = decode_base64_image(payload.image_base64)
        return _execute_enrollment(payload.name, bgr_image)

    @app.post("/api/v1/enroll/upload", response_model=EnrollResponse, tags=["Biometrics"],
              summary="Enroll New Biometric Identity from File Upload (Multipart)")
    async def enroll_face_upload(
        name: str = Form(..., description="Full name of the person"),
        file: UploadFile = File(..., description="Reference portrait image")
    ):
        """
        Enroll a new identity remotely using a multipart file upload.
        """
        raw_bytes = await file.read()
        bgr_image = decode_image_bytes(raw_bytes)
        return _execute_enrollment(name, bgr_image)

    @app.post("/api/v1/enroll/composite", response_model=EnrollResponse, tags=["Biometrics"],
              summary="Enroll Identity via Apple Face ID Multi-Angle Composite Embeddings")
    async def enroll_face_composite(payload: CompositeEnrollRequest):
        """
        Enroll an identity using multiple face angles (e.g. Frontal, Left, Right, Up, Down).
        Extracts 128-d embeddings for each angle and synthesizes an L2-normalized composite vector.
        """
        clean_name = SecurityManager.sanitize_identity_name(payload.name)
        if not clean_name or clean_name == "Unknown":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid identity name. Only alphanumeric characters are allowed. Cannot enroll 'Unknown'."
            )

        sys_instance: FaceRecognitionSystem = app.state.system
        db: DatabaseManager = app.state.db_manager

        embeddings: List[np.ndarray] = []
        portraits: List[np.ndarray] = []

        for item in payload.angles:
            try:
                bgr_img = decode_base64_image(item.image_base64)
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                locs = sys_instance.detector.detect_faces(rgb_img)
                if locs:
                    encs = sys_instance.encoder.encode_faces(rgb_img, [locs[0]])
                    if encs:
                        embeddings.append(encs[0])
                        portraits.append(bgr_img)
            except Exception:
                continue

        if len(embeddings) < 2:
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient valid facial poses ({len(embeddings)}). At least 2 clear facial angles are required."
            )

        composite_vector = compute_l2_normalized_composite(embeddings)
        reference_portrait = portraits[0] if portraits else None

        # Enroll composite embedding into system and vector index
        enrolled_ok = sys_instance.enroll_composite_identity(
            name=clean_name,
            composite_embedding=composite_vector,
            reference_portrait=reference_portrait
        )

        if not enrolled_ok:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save composite biometric embedding."
            )

        safe_filename = clean_name.replace(" ", "_")
        target_path = f"known_faces/{safe_filename}.jpg"
        if db is not None:
            try:
                db.add_person(clean_name, target_path)
            except Exception:
                pass

        return EnrollResponse(
            success=True,
            message=f"Identity '{clean_name}' enrolled with {len(embeddings)}-angle composite Face ID embedding.",
            name=clean_name,
            enrolled_at_ist=format_ist_timestamp(),
            total_enrolled=len(sys_instance.known_face_names)
        )

    @app.get("/api/v1/attendance", response_model=AttendanceSummaryResponse, tags=["Attendance"],
             summary="Query Daily Attendance Analytics in IST")
    async def get_attendance_summary(
        date: Optional[str] = Query(None, description="Target date in YYYY-MM-DD format (IST). Defaults to today."),
        format: str = Query("json", pattern="^(json|csv|html)$", description="Format: json, csv, or html")
    ):
        """
        Retrieve daily attendance breakdown including total present, absentees,
        arrival times, and average verification confidences in Indian Standard Time.
        """
        db: DatabaseManager = app.state.db_manager
        if db is None:
            db = DatabaseManager()
            app.state.db_manager = db

        reporter = AttendanceReporter(db)
        summary = reporter.generate_daily_summary(target_date=date)

        if format == "csv":
            csv_path = Path("data") / f"attendance_summary_{summary['date']}.csv"
            reporter.export_csv(summary, output_path=str(csv_path))
            if csv_path.exists():
                return Response(
                    content=csv_path.read_text(encoding="utf-8"),
                    media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{csv_path.name}"'}
                )
            raise HTTPException(status_code=500, detail="Failed to generate CSV export.")

        elif format == "html":
            html_content = reporter.export_html(summary)
            return HTMLResponse(content=html_content)

        # Standard JSON response
        attendees = [
            AttendanceAttendee(
                name=a['name'],
                first_seen=a['first_seen'],
                last_seen=a['last_seen'],
                sightings=a['sightings'],
                avg_confidence=a['avg_confidence']
            )
            for a in summary.get('attendees', [])
        ]

        return AttendanceSummaryResponse(
            date=summary['date'],
            generated_at_ist=summary['generated_at'],
            total_enrolled=summary['total_enrolled'],
            total_present=summary['total_present'],
            total_absent=summary['total_absent'],
            attendance_rate=summary['attendance_rate'],
            overall_avg_confidence=summary['overall_avg_confidence'],
            total_sightings=summary['total_sightings'],
            attendees=attendees,
            absentees=summary.get('absentees', [])
        )

    # -----------------------------------------------------------------------
    # WebSocket Real-Time Frame Streaming
    # -----------------------------------------------------------------------

    @app.websocket("/ws/stream")
    async def websocket_stream_endpoint(websocket: WebSocket):
        """
        Real-time bi-directional WebSocket connection for edge streaming.
        Accepts binary JPEG frames or JSON with base64 images;
        Streams back real-time detection bounding boxes, identities, and anti-spoofing verdicts.
        """
        await websocket.accept()
        sys_instance: FaceRecognitionSystem = app.state.system
        db: DatabaseManager = app.state.db_manager

        try:
            while True:
                # Receive frame either as binary bytes or text JSON
                message = await websocket.receive()

                bgr_image = None
                if "bytes" in message and message["bytes"]:
                    bgr_image = decode_image_bytes(message["bytes"])
                elif "text" in message and message["text"]:
                    import json
                    try:
                        data = json.loads(message["text"])
                        if "image_base64" in data:
                            bgr_image = decode_base64_image(data["image_base64"])
                    except Exception:
                        await websocket.send_json({"error": "Invalid JSON frame payload"})
                        continue

                if bgr_image is None:
                    continue

                # Process frame
                start_t = time.perf_counter()
                _, telemetry = sys_instance.process_frame(bgr_image, draw_overlays=False)
                latency = (time.perf_counter() - start_t) * 1000.0

                names = telemetry.get('names', [])
                locations = telemetry.get('locations', [])
                confidences = telemetry.get('confidences', [])
                liveness_list = telemetry.get('liveness', [])

                results = []
                for idx, name in enumerate(names):
                    top_s, right_s, bottom_s, left_s = locations[idx]
                    liv = liveness_list[idx] if idx < len(liveness_list) else None

                    results.append({
                        "name": name,
                        "confidence": round(float(confidences[idx]), 1) if idx < len(confidences) else 0.0,
                        "box": {
                            "top": int(top_s * 4),
                            "right": int(right_s * 4),
                            "bottom": int(bottom_s * 4),
                            "left": int(left_s * 4)
                        },
                        "is_live": bool(liv.is_live) if liv else False,
                        "status": liv.status_text if liv else "N/A"
                    })

                await websocket.send_json({
                    "timestamp": format_ist_timestamp(),
                    "faces_detected": len(names),
                    "latency_ms": round(latency, 2),
                    "verifications": results
                })

        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    return app


# Application singleton for Uvicorn ASGI server
app = create_app()


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Launch the FastAPI Uvicorn ASGI server.
    
    Args:
        host (str): Listening network interface (defaults to 0.0.0.0).
        port (int): Listening TCP port (defaults to 8000).
        reload (bool): Enable auto-reload on code change (development mode).
    """
    import uvicorn
    print("=" * 65)
    print("       BIOMETRIC FACE RECOGNITION EDGE API MICROSERVICE")
    print(f"       Listening on: http://{host}:{port}")
    print(f"       Swagger UI:   http://{host}:{port}/docs")
    print(f"       ReDoc UI:     http://{host}:{port}/redoc")
    print(f"       Timestamp:    {format_ist_timestamp()}")
    print("=" * 65)
    uvicorn.run("src.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start_server(host="127.0.0.1", port=8000)
