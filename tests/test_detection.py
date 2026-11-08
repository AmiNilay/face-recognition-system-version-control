"""Unit tests for FaceDetector module."""

import unittest
import numpy as np
from src.face_detector import FaceDetector

class TestFaceDetector(unittest.TestCase):
    """Test suite for FaceDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector(detection_method="hog")

    def test_detector_initialization(self):
        """Verify detector initializes with expected defaults."""
        self.assertEqual(self.detector.detection_method, "hog")
        self.assertIsNotNone(self.detector.face_cascade)

    def test_extract_face_bounds(self):
        """Test extraction of a specific face subregion."""
        # Create a dummy image with known color in a region
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:60, 30:70] = [255, 0, 0]  # Blue box
        
        location = (20, 70, 60, 30)  # top, right, bottom, left
        extracted = self.detector.extract_face(image, location)
        
        self.assertEqual(extracted.shape, (40, 40, 3))
        self.assertTrue(np.all(extracted == [255, 0, 0]))

    def test_extract_face_dimensions(self):
        """Test boundary conditions for face extraction."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        location = (10, 110, 110, 10)
        face_crop = self.detector.extract_face(image, location)
        self.assertEqual(face_crop.shape, (100, 100, 3))

if __name__ == "__main__":
    unittest.main()
