"""
RTSP IP Camera Streaming, Motion Gating, and Webhooks Test Suite
Module: tests/test_rtsp_webhooks.py

Validates:
1. MotionDetector background subtraction and idle compute suppression.
2. WebhookDispatcher event deduplication, rate limiting, and background queue dispatch.
3. RTSPStreamReader zero-latency buffer management and disconnection recovery.
4. FaceRecognitionSystem end-to-end motion gating and security webhook triggering.
5. FastAPI REST endpoints for webhook diagnostics and testing.

All tests run in-memory with zero emojis. Developer: Nilay Naha.
"""

import unittest
import numpy as np
import cv2
import time
from unittest.mock import MagicMock, patch

from src.motion_detector import MotionDetector
from src.webhook_dispatcher import WebhookDispatcher, SecurityEventType
from src.rtsp_stream import RTSPStreamReader
from src.face_recognition_system import FaceRecognitionSystem
from src.api import create_app
from tests.test_api import InMemoryASGIClient


class TestRTSPMotionWebhooks(unittest.TestCase):
    """Test suite covering RTSP streaming, motion detection, and webhook alerts."""

    def test_motion_detector_dynamic_vs_static(self):
        """Verify MotionDetector detects moving objects and settles on static scenes."""
        detector = MotionDetector(enabled=True, min_motion_area=200, cooldown_frames=2)

        # Feed 10 uniform black frames to establish stable background model
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(10):
            detector.detect_motion(blank)

        # Dynamic frame: insert a bright moving white rectangle
        moving_frame = blank.copy()
        moving_frame[50:150, 50:150] = 255
        has_motion, area, boxes = detector.detect_motion(moving_frame)

        self.assertTrue(has_motion)
        self.assertGreater(area, 200)
        self.assertGreater(len(boxes), 0)

        # Cooldown frame 1: blank frame with cooldown active
        has_cooldown, _, _ = detector.detect_motion(blank)
        self.assertTrue(has_cooldown)

        # Cooldown frame 2: blank frame with cooldown active
        has_cooldown_2, _, _ = detector.detect_motion(blank)
        self.assertTrue(has_cooldown_2)

        # Cooldown expired: next blank frame should register NO motion
        no_motion, area_idle, _ = detector.detect_motion(blank)
        self.assertFalse(no_motion)
        self.assertEqual(area_idle, 0.0)
        self.assertGreater(detector.savings_ratio, 0.0)

    def test_motion_detector_disabled_bypass(self):
        """Verify disabled motion detector always reports motion active."""
        detector = MotionDetector(enabled=False)
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        has_motion, _, _ = detector.detect_motion(blank)
        self.assertTrue(has_motion)

    def test_webhook_dispatcher_queue_and_cooldown(self):
        """Verify WebhookDispatcher queues events and suppresses duplicate floods."""
        dispatcher = WebhookDispatcher(
            endpoints=["https://security.example.com/alerts"],
            enabled=True,
            cooldown_seconds=10.0,
            include_snapshot=True
        )

        test_img = np.zeros((100, 100, 3), dtype=np.uint8)

        # First event should successfully dispatch
        ok1 = dispatcher.dispatch(
            event_type=SecurityEventType.UNKNOWN_FACE,
            details="Unregistered subject sighted",
            subject_name="Unknown",
            frame_bgr=test_img
        )
        self.assertTrue(ok1)
        self.assertEqual(dispatcher._queue.qsize(), 1)

        # Immediate second event with same subject should be suppressed by cooldown
        ok2 = dispatcher.dispatch(
            event_type=SecurityEventType.UNKNOWN_FACE,
            details="Unregistered subject sighted",
            subject_name="Unknown",
            frame_bgr=test_img
        )
        self.assertFalse(ok2)
        self.assertEqual(dispatcher.total_suppressed, 1)

        # Different event type or subject should bypass cooldown
        ok3 = dispatcher.dispatch(
            event_type=SecurityEventType.SPOOF_ATTEMPT,
            details="Photo spoof detected",
            subject_name="Bob",
            frame_bgr=test_img
        )
        self.assertTrue(ok3)

        dispatcher.stop()

    def test_webhook_dispatcher_endpoint_management(self):
        """Verify adding and removing target webhook URLs."""
        dispatcher = WebhookDispatcher(enabled=True)
        self.assertEqual(len(dispatcher.endpoints), 0)

        dispatcher.add_endpoint("https://slack.example.com/hooks/123")
        self.assertIn("https://slack.example.com/hooks/123", dispatcher.endpoints)

        # Duplicate addition ignored
        dispatcher.add_endpoint("https://slack.example.com/hooks/123")
        self.assertEqual(len(dispatcher.endpoints), 1)

        dispatcher.remove_endpoint("https://slack.example.com/hooks/123")
        self.assertEqual(len(dispatcher.endpoints), 0)
        dispatcher.stop()

    @patch("src.rtsp_stream.cv2.VideoCapture")
    def test_rtsp_stream_reader_lifecycle(self, mock_cap_cls):
        """Verify RTSPStreamReader initializes, reads latest frame, and handles shutdown."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.ones((240, 320, 3), dtype=np.uint8) * 100
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        reader = RTSPStreamReader(
            stream_url="rtsp://admin:pass@192.168.1.100:554/h264",
            reconnect_interval=1.0
        )
        reader.start()

        # Allow worker thread a brief tick to capture
        time.sleep(0.05)

        ret, frame = reader.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (240, 320, 3))

        stats = reader.get_stats()
        self.assertTrue(stats.get("is_connected"))
        self.assertGreater(stats.get("frames_received"), 0)

        reader.stop()
        self.assertFalse(reader.is_connected())

    def test_face_recognition_system_motion_gating(self):
        """Verify FaceRecognitionSystem process_frame skips deep detection on static scenes."""
        system = FaceRecognitionSystem()
        system.motion_detector.enabled = True

        # Pre-feed 10 blank frames to settle background subtractor
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            system.motion_detector.detect_motion(blank)
        system.motion_detector.cooldown_counter = 0

        # When motion is inactive, process_frame skips heavy inference
        out_frame, telemetry = system.process_frame(blank, draw_overlays=True)
        self.assertFalse(telemetry.get("motion_active"))
        self.assertEqual(telemetry.get("num_faces"), 0)

    @patch("src.face_recognition_system.FaceRecognitionSystem.recognize_face")
    def test_face_recognition_system_webhook_trigger(self, mock_recog):
        """Verify FaceRecognitionSystem triggers webhook alert when an unknown face is seen."""
        system = FaceRecognitionSystem()
        system.webhook_dispatcher.enabled = True
        system.webhook_dispatcher.add_endpoint("https://alerts.example.com")
        mock_recog.return_value = ("Unknown", 0.0)

        # Mock detector to return one face
        system.detector = MagicMock()
        system.detector.detect_faces.return_value = [(10, 80, 80, 10)]
        system.encoder = MagicMock()
        system.encoder.encode_faces.return_value = [np.zeros(128)]

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out_frame, telemetry = system.process_frame(test_frame, draw_overlays=False)

        # Webhook queue should have received the unknown face alert
        self.assertEqual(system.webhook_dispatcher._queue.qsize(), 1)
        system.webhook_dispatcher.stop()

    def test_api_webhook_endpoints(self):
        """Verify FastAPI REST endpoints for webhook diagnostics and registration."""
        mock_system = MagicMock()
        mock_dispatcher = MagicMock()
        mock_dispatcher.enabled = True
        mock_dispatcher.endpoints = ["https://hooks.example.com"]
        mock_dispatcher.cooldown_seconds = 30.0
        mock_dispatcher.total_dispatched = 5
        mock_dispatcher.total_failed = 0
        mock_dispatcher.total_suppressed = 2
        mock_dispatcher.send_test_ping.return_value = {
            "success": True,
            "status_code": 200,
            "message": "Webhook verified successfully"
        }
        mock_system.webhook_dispatcher = mock_dispatcher

        app = create_app(system=mock_system, db_manager=MagicMock())
        client = InMemoryASGIClient(app)

        # 1. GET /api/v1/webhooks
        res_get = client.get("/api/v1/webhooks")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.json()
        self.assertTrue(data_get.get("enabled"))
        self.assertEqual(data_get.get("endpoints"), ["https://hooks.example.com"])
        self.assertEqual(data_get.get("total_dispatched"), 5)

        # 2. POST /api/v1/webhooks/test
        res_test = client.post_json("/api/v1/webhooks/test", {"endpoint_url": "https://hooks.example.com"})
        self.assertEqual(res_test.status_code, 200)
        self.assertTrue(res_test.json().get("success"))

        # 3. POST /api/v1/webhooks/endpoint
        res_reg = client.post_json(
            "/api/v1/webhooks/endpoint",
            {"endpoint_url": "https://new.example.com", "enabled": True}
        )
        self.assertEqual(res_reg.status_code, 200)
        mock_dispatcher.add_endpoint.assert_called_with("https://new.example.com")


if __name__ == "__main__":
    unittest.main()
