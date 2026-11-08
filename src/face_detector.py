"""
Face Detection Module

This module abstracts facial detection algorithms. It provides two primary
detection strategies:
1. Histogram of Oriented Gradients (HOG) via dlib - Fast CPU-based detection.
2. Haar Feature-based Cascade Classifiers via OpenCV - Lightweight fallback.

Beginner Concept:
A face detector scans an image and returns the bounding box coordinates
(top, right, bottom, left) where a human face is located.
"""

import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    """
    Handles locating faces in image frames and extracting face regions.
    """

    def __init__(self, detection_method: str = "hog"):
        """
        Initialize the face detector.

        Args:
            detection_method (str): 'hog' for fast CPU detection, or 'cnn'
                                   for deeper GPU-accelerated accuracy.
        """
        self.detection_method = detection_method
        # Load OpenCV's pre-trained frontal face Haar cascade model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an RGB image using the dlib/face_recognition model.

        Args:
            image (np.ndarray): Image array in RGB color space.

        Returns:
            list[tuple]: List of face bounding boxes in (top, right, bottom, left) format.
        """
        return face_recognition.face_locations(
            image, 
            model=self.detection_method
        )

    def detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV Haar cascades. This serves as a lightweight alternative.

        Args:
            image (np.ndarray): Image array in BGR format.

        Returns:
            list[tuple]: Converted to (top, right, bottom, left) matching dlib convention.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detectMultiScale parameters: image, scaleFactor=1.1, minNeighbors=4
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        # Convert OpenCV's (x, y, w, h) output to standard (top, right, bottom, left)
        face_locations = []
        for (x, y, w, h) in faces:
            top = y
            right = x + w
            bottom = y + h
            left = x
            face_locations.append((top, right, bottom, left))

        return face_locations

    def extract_face(self, image: np.ndarray, 
                     location: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop out the subregion containing the face from the larger image frame.

        Args:
            image (np.ndarray): Original full image frame.
            location (tuple): (top, right, bottom, left) pixel coordinates.

        Returns:
            np.ndarray: Cropped face sub-image.
        """
        top, right, bottom, left = location
        return image[top:bottom, left:right]

    def draw_face_rectangle(self, image: np.ndarray, 
                            location: Tuple[int, int, int, int],
                            name: str = "Unknown",
                            color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Draw a labeled bounding rectangle around a detected face.

        Args:
            image (np.ndarray): Target image to draw upon (BGR format).
            location (tuple): (top, right, bottom, left) face boundaries.
            name (str): Label text displayed beneath the box (e.g. person name).
            color (tuple, optional): (B, G, R) color tuple. If None, uses Green for known, Red for unknown.

        Returns:
            np.ndarray: The modified image array.
        """
        top, right, bottom, left = location

        if color is None:
            # Green (0, 255, 0) for recognized identities; Red (0, 0, 255) for unknowns
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw the main bounding box outline
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)

        # Draw filled banner box at the bottom to host readable label text
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        return image
