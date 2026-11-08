"""
Face Detection Module
Module: src/face_detector.py

This module abstracts facial detection algorithms across multiple engines:
1. Histogram of Oriented Gradients (HOG) via dlib - Fast CPU-based detection.
2. OpenCV YuNet Deep Neural Network (DNN) - Ultra-lightweight, 2026 real-time 60+ FPS detector.
3. Haar Feature-based Cascade Classifiers via OpenCV - Lightweight zero-dependency fallback.
4. Convolutional Neural Network (CNN) via dlib - Maximum accuracy when GPU is available.

Beginner Concept:
A face detector scans an image and returns bounding box coordinates
(top, right, bottom, left) where a human face is located.

All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code, HUD readouts, and logs. Developer: Nilay Naha.
"""

import os
import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional, Any


class FaceDetector:
    """
    Handles locating faces in image frames using HOG, YuNet DNN, Haar, or CNN.
    """

    def __init__(
        self,
        detection_method: str = "hog",
        model_path: str = "models/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3
    ):
        """
        Initialize the face detector with chosen strategy and fallback cascades.

        Args:
            detection_method (str): 'hog', 'yunet', 'haar', or 'cnn'.
            model_path (str): Path to YuNet ONNX model file.
            score_threshold (float): Confidence threshold for YuNet DNN detection.
            nms_threshold (float): Non-Maximum Suppression threshold for YuNet.
        """
        self.detection_method: str = str(detection_method).lower()
        self.model_path: str = model_path
        self.score_threshold: float = float(score_threshold)
        self.nms_threshold: float = float(nms_threshold)
        self.yunet: Optional[Any] = None

        # Load OpenCV's pre-trained frontal face Haar cascade model as universal fallback
        cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

        # Initialize OpenCV YuNet DNN detector if model exists on disk
        if os.path.exists(self.model_path) and hasattr(cv2, 'FaceDetectorYN'):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    model=self.model_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=5000
                )
            except Exception:
                self.yunet = None

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Unified detection entry point. Dispatches to the configured detection engine.

        Args:
            image (np.ndarray): Image array in RGB or BGR format.

        Returns:
            list[tuple]: List of face bounding boxes in (top, right, bottom, left) format.
        """
        if image is None or image.size == 0:
            return []

        if self.detection_method == "yunet":
            return self.detect_faces_yunet(image)
        elif self.detection_method == "haar":
            return self.detect_faces_opencv(image)
        else:
            # Default to face_recognition (dlib 'hog' or 'cnn')
            try:
                return face_recognition.face_locations(
                    image, 
                    model=self.detection_method
                )
            except Exception:
                # If dlib fails or encounters incompatible frame format, fallback to Haar
                return self.detect_faces_opencv(image)

    def detect_faces_yunet(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV's lightweight deep learning YuNet ONNX model.
        Falls back to OpenCV Haar cascade if the ONNX model is uninitialized.

        Args:
            image (np.ndarray): Image array.

        Returns:
            list[tuple]: List of face bounding boxes in (top, right, bottom, left) format.
        """
        if self.yunet is None:
            # Graceful fallback to Haar cascade
            return self.detect_faces_opencv(image)

        h, w = image.shape[:2]
        self.yunet.setInputSize((w, h))

        # YuNet expects BGR format
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 and image.shape[2] == 3 else image
        _, faces = self.yunet.detect(bgr)

        if faces is None:
            return []

        face_locations = []
        for face in faces:
            # Layout: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, width, height = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            top = max(0, y)
            right = min(w, x + width)
            bottom = min(h, y + height)
            left = max(0, x)
            face_locations.append((top, right, bottom, left))

        return face_locations

    def detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV Haar cascades. Serves as a lightweight, zero-dependency alternative.

        Args:
            image (np.ndarray): Image array.

        Returns:
            list[tuple]: Converted to (top, right, bottom, left) matching standard convention.
        """
        if image is None or image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        face_locations = []
        for (x, y, w, h) in faces:
            top = int(y)
            right = int(x + w)
            bottom = int(y + h)
            left = int(x)
            face_locations.append((top, right, bottom, left))

        return face_locations

    def extract_face(
        self,
        image: np.ndarray, 
        location: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop out the subregion containing the face from the larger image frame.

        Args:
            image (np.ndarray): Original full image frame.
            location (tuple): (top, right, bottom, left) pixel coordinates.

        Returns:
            np.ndarray: Cropped face sub-image.
        """
        top, right, bottom, left = location
        h, w = image.shape[:2]
        y1 = max(0, min(h, top))
        y2 = max(0, min(h, bottom))
        x1 = max(0, min(w, left))
        x2 = max(0, min(w, right))
        return image[y1:y2, x1:x2]

    def draw_face_rectangle(
        self,
        image: np.ndarray, 
        location: Tuple[int, int, int, int],
        name: str = "Unknown",
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Draw a labeled bounding rectangle around a detected face.

        Args:
            image (np.ndarray): Target image to draw upon (BGR format).
            location (tuple): (top, right, bottom, left) face boundaries.
            name (str): Label text displayed beneath the box (e.g. person name).
            color (tuple, optional): (B, G, R) color tuple.

        Returns:
            np.ndarray: The modified image array.
        """
        top, right, bottom, left = location

        if color is None:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw the main bounding box outline
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)

        # Draw filled banner box at the bottom to host readable label text
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        return image
