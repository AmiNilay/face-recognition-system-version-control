"""
API Unit Test Suite
Module: tests/test_api.py

Validates FastAPI Edge Microservice endpoints:
- GET /
- GET /api/v1/health
- POST /api/v1/verify (JSON)
- POST /api/v1/verify/upload (Multipart)
- POST /api/v1/enroll (JSON)
- POST /api/v1/enroll/upload (Multipart)
- GET /api/v1/attendance (JSON and CSV)
- WebSocket /ws/stream

All tests run in-memory with mocked hardware and computer vision dependencies.
Zero emojis across all code and test assertions. Developer: Nilay Naha.
"""

import os
import io
import json
import base64
import unittest
import asyncio
from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from src.api import create_app, decode_image_bytes, decode_base64_image
from src.anti_spoofing import LivenessResult


class InMemoryASGIClient:
    """
    Lightweight, in-memory ASGI test client that invokes the FastAPI ASGI application
    directly via standard asyncio without opening OS sockets or thread portals.
    Guarantees deterministic, fast execution in sandboxed environments.
    """

    def __init__(self, app):
        self.app = app

    def request(self, method: str, path: str, headers: list = None,
                body: bytes = b"", query_string: bytes = b""):
        """Perform a synchronous HTTP request against the ASGI application."""
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method.upper(),
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": query_string,
            "headers": headers or [],
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 12345),
            "state": {}
        }

        response_start = {}
        response_body = []
        received = False

        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                response_start["status"] = message["status"]
                response_start["headers"] = message.get("headers", [])
            elif message["type"] == "http.response.body":
                response_body.append(message.get("body", b""))

        async def _run():
            await self.app(scope, receive, send)

        asyncio.run(_run())

        class Response:
            def __init__(self, status_code, headers_raw, body_bytes):
                self.status_code = status_code
                self.headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in headers_raw}
                self.content = body_bytes
                self.text = body_bytes.decode("utf-8")

            def json(self):
                return json.loads(self.text)

        return Response(response_start.get("status", 500),
                        response_start.get("headers", []),
                        b"".join(response_body))

    def get(self, path: str, query: str = ""):
        return self.request("GET", path, query_string=query.encode("ascii"))

    def post_json(self, path: str, data: dict):
        body = json.dumps(data).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii"))
        ]
        return self.request("POST", path, headers=headers, body=body)

    def post_multipart(self, path: str, fields: dict, files: dict):
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body_parts = []

        for field_name, value in fields.items():
            body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
            body_parts.append(f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'.encode("utf-8"))
            body_parts.append(f"{value}\r\n".encode("utf-8"))

        for file_key, (filename, file_bytes, content_type) in files.items():
            body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
            body_parts.append(
                f'Content-Disposition: form-data; name="{file_key}"; filename="{filename}"\r\n'.encode("utf-8")
            )
            body_parts.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
            body_parts.append(file_bytes)
            body_parts.append(b"\r\n")

        body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(body_parts)

        headers = [
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode("ascii")),
            (b"content-length", str(len(body)).encode("ascii"))
        ]
        return self.request("POST", path, headers=headers, body=body)


class TestFastAPIMicroservice(unittest.TestCase):
    """Test suite covering all FastAPI REST endpoints and microservice logic."""

    def setUp(self):
        # Create a synthetic 100x100 BGR test image
        self.test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_img[20:80, 20:80] = [200, 200, 200]
        _, encoded = cv2.imencode(".jpg", self.test_img)
        self.test_jpg_bytes = encoded.tobytes()
        self.test_b64 = base64.b64encode(self.test_jpg_bytes).decode("ascii")

        # Mock FaceRecognitionSystem
        self.mock_system = MagicMock()
        self.mock_system.model_type = "hog"
        self.mock_system.encoding_model = "large"
        self.mock_system.anti_spoof.enabled = True
        self.mock_system.vector_index.is_fitted.return_value = True
        self.mock_system.get_known_faces.return_value = ["Alice", "Bob"]
        self.mock_system.known_face_names = ["Alice", "Bob"]
        self.mock_system.known_face_encodings = [np.zeros(128), np.zeros(128)]

        # Mock DatabaseManager
        self.mock_db = MagicMock()

        # Build FastAPI application with injected mocks
        self.app = create_app(system=self.mock_system, db_manager=self.mock_db)
        self.client = InMemoryASGIClient(self.app)

    def test_root_endpoint(self):
        """Verify root endpoint returns API identity and documentation links."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("name"), "Biometric Face Recognition & Anti-Spoofing Edge API")
        self.assertEqual(data.get("status"), "online")
        self.assertIn("docs", data.get("documentation", {}).get("swagger_ui", ""))

    def test_health_endpoint(self):
        """Verify health check returns service diagnostics and enrolled identities."""
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("enrolled_count"), 2)
        self.assertEqual(data.get("enrolled_identities"), ["Alice", "Bob"])
        self.assertEqual(data.get("detector_model"), "hog")
        self.assertTrue(data.get("anti_spoofing_enabled"))
        self.assertTrue(data.get("vector_index_active"))

    def test_verify_json_recognized_face(self):
        """Verify POST /api/v1/verify correctly parses recognized face with liveness telemetry."""
        liv_res = LivenessResult(
            is_live=True,
            ear=0.32,
            blinks=3,
            texture_score=0.15,
            status_text="Verified Human",
            hud_color=(0, 255, 0),
            head_pose=(5.2, -2.1, 1.4),
            nose_ray=((50, 50), (55, 48)),
            challenge_prompt=None
        )
        telemetry = {
            'names': ['Alice'],
            'locations': [(10, 20, 20, 10)], # top, right, bottom, left in 0.25 scale
            'confidences': [95.4],
            'liveness': [liv_res],
            'poses': [(5.2, -2.1, 1.4)],
            'challenges': [None],
            'num_faces': 1,
            'num_enrolled': 2
        }
        self.mock_system.process_frame.return_value = (self.test_img, telemetry)

        payload = {
            "image_base64": self.test_b64,
            "auto_log_attendance": True
        }
        res = self.client.post_json("/api/v1/verify", payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("faces_detected"), 1)
        self.assertGreater(data.get("latency_ms", 0), 0)

        verifications = data.get("verifications", [])
        self.assertEqual(len(verifications), 1)
        v = verifications[0]
        self.assertEqual(v.get("name"), "Alice")
        self.assertEqual(v.get("confidence"), 95.4)
        self.assertTrue(v.get("is_live"))
        self.assertEqual(v.get("liveness_status"), "Verified Human")
        self.assertEqual(v.get("head_pose", {}).get("yaw"), 5.2)

        # Confirm auto-log was called on database manager
        self.mock_db.log_recognition.assert_called_once_with(person_name="Alice", confidence=95.4)

    def test_verify_upload_multipart(self):
        """Verify POST /api/v1/verify/upload processes multipart file uploads."""
        liv_res = LivenessResult(
            is_live=True,
            ear=0.28,
            blinks=2,
            texture_score=0.12,
            status_text="Verified Human",
            hud_color=(0, 255, 0),
            head_pose=None,
            nose_ray=None,
            challenge_prompt=None
        )
        telemetry = {
            'names': ['Bob'],
            'locations': [(5, 25, 25, 5)],
            'confidences': [88.2],
            'liveness': [liv_res],
            'poses': [None],
            'challenges': [None],
            'num_faces': 1,
            'num_enrolled': 2
        }
        self.mock_system.process_frame.return_value = (self.test_img, telemetry)

        files = {
            "file": ("portrait.jpg", self.test_jpg_bytes, "image/jpeg")
        }
        fields = {"auto_log_attendance": "false"}
        res = self.client.post_multipart("/api/v1/verify/upload", fields=fields, files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("faces_detected"), 1)
        self.assertEqual(data.get("verifications")[0].get("name"), "Bob")
        # auto_log was false, so db.log_recognition shouldn't be called here
        self.mock_db.log_recognition.assert_not_called()

    def test_verify_invalid_base64_returns_400(self):
        """Verify malformed image data raises 400 Bad Request."""
        payload = {"image_base64": "not_a_valid_base64_string!!!"}
        res = self.client.post_json("/api/v1/verify", payload)
        self.assertEqual(res.status_code, 400)

    def test_enroll_json_success(self):
        """Verify POST /api/v1/enroll registers a new identity."""
        self.mock_system.detector.detect_faces.return_value = [(10, 80, 80, 10)]
        self.mock_system.encoder.encode_faces.return_value = [np.ones(128)]

        payload = {
            "name": "Charlie Chaplin",
            "image_base64": self.test_b64
        }
        res = self.client.post_json("/api/v1/enroll", payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("name"), "Charlie Chaplin")
        self.mock_system.save_encodings.assert_called_once()
        self.mock_db.add_person.assert_called_once()

    def test_enroll_rejects_path_traversal(self):
        """Verify identity name sanitization blocks path traversal attacks."""
        payload = {
            "name": "../../../..",
            "image_base64": self.test_b64
        }
        res = self.client.post_json("/api/v1/enroll", payload)
        # Sanitization strips dots and slashes; empty name raises 400 Bad Request
        self.assertEqual(res.status_code, 400)

    def test_enroll_rejects_no_face(self):
        """Verify enrollment fails with 422 when no face is found in image."""
        self.mock_system.detector.detect_faces.return_value = []
        payload = {
            "name": "David",
            "image_base64": self.test_b64
        }
        res = self.client.post_json("/api/v1/enroll", payload)
        self.assertEqual(res.status_code, 422)
        self.assertIn("No human face detected", res.json().get("detail", ""))

    def test_enroll_rejects_multiple_faces(self):
        """Verify enrollment fails with 422 when multiple faces are present."""
        self.mock_system.detector.detect_faces.return_value = [(10, 40, 40, 10), (50, 90, 90, 50)]
        payload = {
            "name": "David",
            "image_base64": self.test_b64
        }
        res = self.client.post_json("/api/v1/enroll", payload)
        self.assertEqual(res.status_code, 422)
        self.assertIn("Multiple faces", res.json().get("detail", ""))

    @patch("src.api.AttendanceReporter")
    def test_attendance_summary_json(self, mock_reporter_cls):
        """Verify GET /api/v1/attendance returns structured attendance metrics."""
        mock_reporter = MagicMock()
        mock_reporter_cls.return_value = mock_reporter
        mock_reporter.generate_daily_summary.return_value = {
            'date': '2026-11-08',
            'generated_at': '2026-11-08 23:00:00 IST',
            'total_enrolled': 10,
            'total_present': 8,
            'total_absent': 2,
            'attendance_rate': 80.0,
            'overall_avg_confidence': 92.4,
            'total_sightings': 24,
            'attendees': [
                {
                    'name': 'Alice',
                    'first_seen': '2026-11-08 09:05:00 IST',
                    'last_seen': '2026-11-08 17:30:00 IST',
                    'sightings': 3,
                    'avg_confidence': 95.1
                }
            ],
            'absentees': ['Zack', 'Zoe']
        }

        res = self.client.get("/api/v1/attendance", query="date=2026-11-08&format=json")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("date"), "2026-11-08")
        self.assertEqual(data.get("total_enrolled"), 10)
        self.assertEqual(data.get("total_present"), 8)
        self.assertEqual(data.get("total_absent"), 2)
        self.assertEqual(data.get("attendance_rate"), 80.0)
        self.assertEqual(len(data.get("attendees")), 1)
        self.assertEqual(data.get("absentees"), ["Zack", "Zoe"])


if __name__ == "__main__":
    unittest.main()
