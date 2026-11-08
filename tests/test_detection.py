"""
Unit Tests for FaceDetector Module (tests/test_detection.py)

Tests HOG, YuNet DNN, and Haar cascade face detection abstractions,
subregion extraction, boundary clamping, and annotated bounding boxes.
All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code, HUD readouts, and logs. Developer: Nilay Naha.
"""

import unittest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
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
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:60, 30:70] = [255, 0, 0]  # Blue box
        
        location = (20, 70, 60, 30)  # top, right, bottom, left
        extracted = self.detector.extract_face(image, location)
        
        self.assertEqual(extracted.shape, (40, 40, 3))
        self.assertTrue(np.all(extracted == [255, 0, 0]))

    def test_extract_face_dimensions_and_clamping(self):
        """Test boundary clamping for face extraction outside image dimensions."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Normal extraction
        location = (10, 110, 110, 10)
        face_crop = self.detector.extract_face(image, location)
        self.assertEqual(face_crop.shape, (100, 100, 3))

        # Coordinates extending beyond frame boundaries (negative & overflow)
        out_of_bounds = (-20, 250, 250, -20)
        clamped_crop = self.detector.extract_face(image, out_of_bounds)
        self.assertEqual(clamped_crop.shape, (200, 200, 3))

    def test_detect_faces_empty_or_none(self):
        """Verify detector handles None or empty image frames safely."""
        self.assertEqual(self.detector.detect_faces(None), [])
        self.assertEqual(self.detector.detect_faces(np.array([])), [])

    def test_detect_faces_haar_cascade(self):
        """Verify Haar cascade detector runs without crashing on synthetic image."""
        blank = np.zeros((120, 120, 3), dtype=np.uint8)
        locations = self.detector.detect_faces_opencv(blank)
        self.assertIsInstance(locations, list)

    def test_yunet_detector_fallback_and_execution(self):
        """Verify YuNet detector falls back gracefully to Haar if ONNX is missing."""
        yunet_detector = FaceDetector(
            detection_method="yunet",
            model_path="models/non_existent_model.onnx"
        )
        self.assertEqual(yunet_detector.detection_method, "yunet")
        self.assertIsNone(yunet_detector.yunet)

        # Calling detect_faces should fall back to Haar cascade without raising an exception
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = yunet_detector.detect_faces(test_img)
        self.assertIsInstance(result, list)

    def test_yunet_mock_detection_parsing(self):
        """Verify YuNet bounding box parsing when model inference produces detections."""
        yunet_detector = FaceDetector(detection_method="yunet")
        mock_yunet = MagicMock()
        # Mock detection result: 1 face [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        mock_face = np.array([[30.0, 20.0, 50.0, 60.0, 40, 35, 70, 35, 55, 50, 42, 65, 68, 65, 0.95]])
        mock_yunet.detect.return_value = (None, mock_face)
        yunet_detector.yunet = mock_yunet

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        locations = yunet_detector.detect_faces_yunet(img)
        self.assertEqual(len(locations), 1)
        top, right, bottom, left = locations[0]
        self.assertEqual(top, 20)
        self.assertEqual(right, 80)
        self.assertEqual(bottom, 80)
        self.assertEqual(left, 30)

    def test_draw_face_rectangle(self):
        """Verify drawing labeled bounding boxes with custom and default colors."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Default green for known face
        drawn = self.detector.draw_face_rectangle(img, (10, 90, 90, 10), name="Alice")
        self.assertEqual(drawn.shape, (100, 100, 3))

        # Default red for unknown
        drawn_unknown = self.detector.draw_face_rectangle(img, (10, 90, 90, 10), name="Unknown")
        self.assertEqual(drawn_unknown.shape, (100, 100, 3))


if __name__ == "__main__":
    unittest.main()
