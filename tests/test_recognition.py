"""Unit tests for FaceRecognitionSystem, FaceEncoder, and DatabaseManager."""

import unittest
import numpy as np
import tempfile
import os
import gc
import re
from unittest.mock import patch, MagicMock
from src.face_encoder import FaceEncoder
from src.face_recognition_system import FaceRecognitionSystem, sanitize_identity_name
from src.database_manager import DatabaseManager
from src.vector_index import VectorIndex
from src.utils import format_ist_timestamp, get_ist_now

class TestFaceRecognition(unittest.TestCase):
    """Test suite for FaceRecognitionSystem & FaceEncoder."""

    def setUp(self):
        """Set up test fixtures."""
        self.encoder = FaceEncoder(encoding_model="large")
        self.system = FaceRecognitionSystem()

    def test_encoder_initialization(self):
        """Verify encoder initializes with empty lists."""
        self.assertEqual(len(self.encoder.known_encodings), 0)
        self.assertEqual(len(self.encoder.known_names), 0)

    def test_compare_identical_encodings(self):
        """Identical embeddings should match with 100% confidence."""
        mock_embedding = np.random.rand(128)
        self.system.known_face_encodings = [mock_embedding]
        self.system.known_face_names = ["Test Person"]

        name, confidence = self.system.recognize_face(mock_embedding)
        self.assertEqual(name, "Test Person")
        self.assertAlmostEqual(confidence, 100.0, places=1)

    def test_unknown_face_distance_threshold(self):
        """Distant vectors should be classified as Unknown."""
        known_embedding = np.ones(128)
        unknown_embedding = -np.ones(128)  # Opposing vector, distance > 0.6
        
        self.system.known_face_encodings = [known_embedding]
        self.system.known_face_names = ["Registered Person"]

        name, confidence = self.system.recognize_face(unknown_embedding)
        self.assertEqual(name, "Unknown")
        self.assertEqual(confidence, 0.0)

    def test_sanitize_identity_name(self):
        """Ensure malicious directory traversal sequences are stripped."""
        unsafe_name = "../../etc/passwd/John..Doe@@##$$"
        cleaned = sanitize_identity_name(unsafe_name)
        self.assertNotIn("..", cleaned)
        self.assertNotIn("/", cleaned)
        self.assertEqual(cleaned, "etcpasswdJohnDoe")

    def test_config_driven_parameters(self):
        """Verify custom configuration dictionaries are respected."""
        custom_cfg = {
            'face_recognition': {
                'tolerance': 0.45,
                'model': 'hog',
                'encoding_model': 'large',
                'process_interval': 5
            },
            'camera': {
                'width': 1280,
                'height': 720,
                'fps': 60
            }
        }
        sys_custom = FaceRecognitionSystem(config=custom_cfg)
        self.assertEqual(sys_custom.tolerance, 0.45)
        self.assertEqual(sys_custom.process_interval, 5)
        self.assertEqual(sys_custom.cam_width, 1280)
        self.assertEqual(sys_custom.cam_height, 720)

    def test_ist_timestamp_format(self):
        """Verify IST timestamps contain IST and valid date pattern."""
        ts = format_ist_timestamp()
        self.assertTrue(ts.endswith("IST"))
        # Match pattern YYYY-MM-DD HH:MM:SS IST
        pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} IST$"
        self.assertRegex(ts, pattern)

    def test_database_manager_operations(self):
        """Test database initialization, logging, and query."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db_path = tmp.name
        tmp.close()

        try:
            db = DatabaseManager(db_path=tmp_db_path)
            person_id = db.add_person("Alice")
            self.assertGreater(person_id, 0)

            # Test duplicate add returns existing ID
            dup_id = db.add_person("Alice")
            self.assertEqual(person_id, dup_id)

            # Log recognition and verify
            db.log_recognition("Alice", 0.95)
            history = db.get_recognition_history(person_name="Alice")
            self.assertEqual(len(history), 1)
            self.assertEqual(history.iloc[0]['name'], "Alice")
            
            # Clean up db reference before removing file
            del db
            gc.collect()
        finally:
            if os.path.exists(tmp_db_path):
                try:
                    os.remove(tmp_db_path)
                except PermissionError:
                    pass

    def test_vector_index_basic_and_edge_cases(self):
        """Verify VectorIndex handling of empty indices, invalid shapes, and queries."""
        v_index = VectorIndex(use_kdtree=True)
        self.assertEqual(len(v_index), 0)

        # Empty index query returns Unknown
        name, conf, dist = v_index.query(np.ones(128))
        self.assertEqual(name, "Unknown")
        self.assertEqual(conf, 0.0)

        # None query returns Unknown
        name, conf, dist = v_index.query(None)
        self.assertEqual(name, "Unknown")

        # Invalid vector length returns Unknown
        name, conf, dist = v_index.query(np.ones(64))
        self.assertEqual(name, "Unknown")

    def test_vector_index_kdtree_search_and_linear_equivalence(self):
        """Verify spatial cKDTree index produces identical matches to linear scan."""
        from src.vector_index import HAS_CKDTREE
        if not HAS_CKDTREE:
            self.skipTest("scipy.spatial.cKDTree is not available in environment")

        np.random.seed(42)
        mock_encodings = [np.random.rand(128) for _ in range(20)]
        mock_names = [f"User_{i}" for i in range(20)]

        # 1. Accelerated cKDTree index
        tree_index = VectorIndex(use_kdtree=True)
        self.assertTrue(tree_index.build(mock_encodings, mock_names))
        self.assertEqual(len(tree_index), 20)

        # 2. Linear scan index (use_kdtree=False)
        linear_index = VectorIndex(use_kdtree=False)
        linear_index.build(mock_encodings, mock_names)

        # Exact query for User_7
        exact_query = mock_encodings[7]
        name_t, conf_t, dist_t = tree_index.query(exact_query)
        name_l, conf_l, dist_l = linear_index.query(exact_query)

        self.assertEqual(name_t, "User_7")
        self.assertEqual(name_l, "User_7")
        self.assertAlmostEqual(conf_t, 100.0, places=2)
        self.assertAlmostEqual(conf_l, 100.0, places=2)
        self.assertAlmostEqual(dist_t, 0.0, places=5)
        self.assertAlmostEqual(dist_l, 0.0, places=5)

        # Noisy query near User_12
        noisy_query = mock_encodings[12] + np.random.normal(0, 0.02, 128)
        name_noisy, conf_noisy, dist_noisy = tree_index.query(noisy_query, tolerance=0.6)
        name_noisy_l, conf_noisy_l, dist_noisy_l = linear_index.query(noisy_query, tolerance=0.6)

        self.assertEqual(name_noisy, "User_12")
        self.assertEqual(name_noisy, name_noisy_l)
        self.assertAlmostEqual(dist_noisy, dist_noisy_l, places=4)

    def test_system_process_frame_end_to_end(self):
        """Verify unified process_frame pipeline produces frame and complete telemetry."""
        test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Mock detector to return 1 face location
        self.system.detector.detect_faces = lambda img: [(10, 80, 80, 10)]
        # Mock encoder to return 1 mock embedding
        self.system.encoder.encode_faces = lambda img, locs: [np.random.rand(128)]
        # Mock landmark extractor
        with patch("face_recognition.face_landmarks", return_value=[{'left_eye': [(0, 0)]*6, 'right_eye': [(0, 0)]*6}]):
            annotated_frame, telemetry = self.system.process_frame(test_frame, draw_overlays=True)

        self.assertEqual(annotated_frame.shape, (240, 320, 3))
        self.assertIn('names', telemetry)
        self.assertIn('locations', telemetry)
        self.assertIn('confidences', telemetry)
        self.assertIn('liveness', telemetry)
        self.assertIn('num_faces', telemetry)
        self.assertEqual(telemetry['num_faces'], 1)


if __name__ == "__main__":
    unittest.main()
