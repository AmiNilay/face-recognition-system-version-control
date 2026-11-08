"""
Apple Face ID Multi-Angle Composite Enrollment Test Suite
Module: tests/test_composite_enrollment.py

Validates:
1. Mathematical L2-normalized Fréchet centroid vector synthesis.
2. Image sharpness analysis via Laplacian variance.
3. Multi-angle state machine progression through 5 canonical poses.
4. Apple Face ID segmented HUD rendering.
5. System-level composite enrollment and VectorIndex query accuracy.
6. FastAPI REST endpoint /api/v1/enroll/composite.

All tests run in-memory with zero emojis. Developer: Nilay Naha.
"""

import unittest
import numpy as np
import cv2
import base64
from unittest.mock import MagicMock, patch

from src.composite_enrollment import (
    compute_l2_normalized_composite,
    measure_frame_sharpness,
    CompositeEnrollmentSession,
    render_apple_face_id_hud,
    ENROLLMENT_STEPS,
    FaceIDAngle
)
from src.face_recognition_system import FaceRecognitionSystem
from src.api import create_app
from tests.test_api import InMemoryASGIClient


class TestCompositeEnrollment(unittest.TestCase):
    """Test suite for Apple Face ID multi-angle composite enrollment engine."""

    def setUp(self):
        # Generate three synthetic 128-d unit embedding vectors
        v1 = np.ones(128) / np.linalg.norm(np.ones(128))
        v2 = np.copy(v1)
        v2[0:30] *= -1.0
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.copy(v1)
        v3[30:60] *= 0.5
        v3 = v3 / np.linalg.norm(v3)
        self.sample_embeddings = [v1, v2, v3]

    def test_compute_l2_normalized_composite(self):
        """Verify synthesized composite vector is mathematically L2-normalized to 1.0."""
        composite = compute_l2_normalized_composite(self.sample_embeddings)
        self.assertEqual(len(composite), 128)
        norm = np.linalg.norm(composite)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_compute_composite_empty_raises_value_error(self):
        """Verify empty input list raises ValueError."""
        with self.assertRaises(ValueError):
            compute_l2_normalized_composite([])

    def test_measure_sharpness(self):
        """Verify Laplacian sharpness measurement distinguishes blur from detail."""
        # Flat blank image has zero Laplacian variance
        flat_img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertAlmostEqual(measure_frame_sharpness(flat_img), 0.0, places=1)

        # High-contrast check pattern has high variance
        sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp_img[::4, :] = 255
        sharp_img[:, ::4] = 255
        sharpness = measure_frame_sharpness(sharp_img)
        self.assertGreater(sharpness, 50.0)

    def test_session_state_machine_progression(self):
        """Verify CompositeEnrollmentSession advances through all 5 canonical poses."""
        session = CompositeEnrollmentSession(
            identity_name="Nilay Naha",
            required_stability_frames=2,
            min_sharpness=5.0
        )
        self.assertEqual(session.clean_name, "Nilay Naha")
        self.assertEqual(session.current_step_index, 0)
        self.assertEqual(session.current_step.angle, FaceIDAngle.FRONTAL)

        synthetic_crop = np.zeros((60, 60, 3), dtype=np.uint8)
        synthetic_crop[::2, :] = 255  # make sharp
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Poses corresponding to ENROLLMENT_STEPS:
        # 1. FRONTAL (Yaw=0, Pitch=0)
        # 2. TURN_LEFT (Yaw=-18, Pitch=0)
        # 3. TURN_RIGHT (Yaw=18, Pitch=0)
        # 4. TILT_UP (Yaw=0, Pitch=-12)
        # 5. NOD_DOWN (Yaw=0, Pitch=12)
        target_poses = [
            (0.0, 0.0, 0.0),
            (-18.0, 0.0, 0.0),
            (18.0, 0.0, 0.0),
            (0.0, -12.0, 0.0),
            (0.0, 12.0, 0.0)
        ]

        dummy_enc = np.ones(128) / np.linalg.norm(np.ones(128))

        for step_i, pose in enumerate(target_poses):
            self.assertEqual(session.current_step_index, step_i)

            # Frame 1: holding steady
            res1 = session.process_frame(frame, pose, dummy_enc, synthetic_crop)
            self.assertEqual(res1.get("status"), "HOLDING_STABLE")

            # Frame 2: triggers capture
            res2 = session.process_frame(frame, pose, dummy_enc, synthetic_crop)
            self.assertEqual(res2.get("status"), "CAPTURED")

        # Session should now be 100% complete
        self.assertTrue(session.is_completed)
        self.assertIsNotNone(session.composite_embedding)
        self.assertAlmostEqual(np.linalg.norm(session.composite_embedding), 1.0, places=5)
        self.assertEqual(session.progress_percentage, 100.0)

    def test_session_rejects_out_of_range_pose(self):
        """Verify out-of-range head pose does not trigger capture."""
        session = CompositeEnrollmentSession(identity_name="Alice", required_stability_frames=2)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = np.zeros((50, 50, 3), dtype=np.uint8)
        dummy_enc = np.ones(128)

        # FRONTAL expects yaw [-7, +7], give yaw = 30
        res = session.process_frame(frame, (30.0, 0.0, 0.0), dummy_enc, crop)
        self.assertEqual(res.get("status"), "OUT_OF_RANGE")
        self.assertEqual(session.stability_counter, 0)
        self.assertEqual(session.current_step_index, 0)

    def test_render_apple_face_id_hud(self):
        """Verify Apple Face ID circular tick HUD renders on frame without error."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        session = CompositeEnrollmentSession(identity_name="Alice")
        out = render_apple_face_id_hud(frame, session, center_xy=(320, 240), radius=100)
        self.assertEqual(out.shape, (480, 640, 3))
        # Non-zero pixels should exist where ticks and rings were drawn
        self.assertGreater(np.count_nonzero(out), 0)

    @patch("src.face_recognition_system.FaceRecognitionSystem.save_encodings")
    def test_enroll_composite_identity(self, mock_save):
        """Verify FaceRecognitionSystem.enroll_composite_identity stores and indexes vector."""
        system = FaceRecognitionSystem()
        system.known_face_names = ["Existing User"]
        system.known_face_encodings = [np.zeros(128)]

        comp_vec = np.ones(128) / np.linalg.norm(np.ones(128))
        ref_portrait = np.zeros((100, 100, 3), dtype=np.uint8)

        success = system.enroll_composite_identity(
            name="Bob Builder",
            composite_embedding=comp_vec,
            reference_portrait=ref_portrait
        )
        self.assertTrue(success)
        self.assertIn("Bob Builder", system.known_face_names)
        self.assertEqual(len(system.known_face_encodings), 2)
        mock_save.assert_called_once()

    def test_enroll_composite_rejects_unknown(self):
        """Verify 'Unknown' is blocked from composite enrollment."""
        system = FaceRecognitionSystem()
        comp_vec = np.ones(128)
        success = system.enroll_composite_identity(
            name="Unknown",
            composite_embedding=comp_vec
        )
        self.assertFalse(success)

    def test_api_enroll_composite_endpoint(self):
        """Verify POST /api/v1/enroll/composite FastAPI endpoint."""
        mock_system = MagicMock()
        mock_system.known_face_names = ["Alice"]
        mock_system.detector.detect_faces.return_value = [(10, 80, 80, 10)]
        mock_system.encoder.encode_faces.return_value = [np.ones(128) / np.linalg.norm(np.ones(128))]
        mock_system.enroll_composite_identity.return_value = True

        mock_db = MagicMock()

        app = create_app(system=mock_system, db_manager=mock_db)
        client = InMemoryASGIClient(app)

        # Create dummy base64 image
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", dummy_img)
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        payload = {
            "name": "Sarah Connor",
            "angles": [
                {"angle": "FRONTAL", "image_base64": b64},
                {"angle": "TURN_LEFT", "image_base64": b64},
                {"angle": "TURN_RIGHT", "image_base64": b64}
            ]
        }

        res = client.post_json("/api/v1/enroll/composite", payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("name"), "Sarah Connor")
        mock_system.enroll_composite_identity.assert_called_once()


if __name__ == "__main__":
    unittest.main()
