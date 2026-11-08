"""Unit tests for FaceRecognitionSystem, FaceEncoder, and DatabaseManager."""

import unittest
import numpy as np
import tempfile
import os
import gc
import re
from src.face_encoder import FaceEncoder
from src.face_recognition_system import FaceRecognitionSystem, sanitize_identity_name
from src.database_manager import DatabaseManager
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

if __name__ == "__main__":
    unittest.main()
