"""
Face Encoding Module

This module handles computing 128-dimensional facial embedding vectors.

Beginner Concept:
A face encoding is a mathematical fingerprint of a human face composed of 128 numbers.
Pre-trained deep neural networks analyze facial geometry (distance between eyes, nose width,
jawline shape) and represent them as a point in 128-dimensional space.
If two photos show the same person, their 128 numbers will be very close in distance.
"""

import face_recognition
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path

class FaceEncoder:
    """
    Handles converting facial images into 128-dimensional mathematical feature vectors
    and comparing them to identify matches.
    """

    def __init__(self, encoding_model: str = "large"):
        """
        Initialize the face encoder.

        Args:
            encoding_model (str): 'large' uses 68 facial landmarks for higher precision;
                                 'small' uses 5 landmarks for reduced latency.
        """
        self.encoding_model = encoding_model
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []

    def encode_face(self, image: np.ndarray, 
                    face_location: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Compute the 128-dimensional feature vector for a single face.

        Args:
            image (np.ndarray): Image array in RGB format.
            face_location (tuple, optional): (top, right, bottom, left) bounding box.
                                             If omitted, detection runs automatically.

        Returns:
            np.ndarray or None: 128-d float array, or None if no face is detected.
        """
        if face_location:
            encodings = face_recognition.face_encodings(image, [face_location], model=self.encoding_model)
        else:
            encodings = face_recognition.face_encodings(image, model=self.encoding_model)

        return encodings[0] if encodings else None

    def encode_faces(self, image: np.ndarray, 
                     face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Compute feature vectors for multiple faces detected in an image frame.

        Args:
            image (np.ndarray): Image array in RGB format.
            face_locations (list[tuple]): List of bounding boxes.

        Returns:
            list[np.ndarray]: List of 128-d embedding vectors.
        """
        return face_recognition.face_encodings(image, face_locations, model=self.encoding_model)

    def compare_faces(self, known_encoding: np.ndarray, 
                      unknown_encoding: np.ndarray,
                      tolerance: float = 0.6) -> bool:
        """
        Compare two 128-d encodings using Euclidean distance.

        Args:
            known_encoding (np.ndarray): Enrolled reference embedding.
            unknown_encoding (np.ndarray): Newly detected embedding.
            tolerance (float): Maximum Euclidean distance threshold to qualify as a match.
                              Default 0.6 provides optimal balance between false positives and negatives.

        Returns:
            bool: True if Euclidean distance <= tolerance; False otherwise.
        """
        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
        return bool(distance <= tolerance)

    def find_match(self, unknown_encoding: np.ndarray, 
                   tolerance: float = 0.6) -> Optional[str]:
        """
        Find the best matching identity among enrolled faces.

        Args:
            unknown_encoding (np.ndarray): Newly detected embedding vector.
            tolerance (float): Distance threshold.

        Returns:
            str or None: Name of the closest matching identity, or None if no match.
        """
        if not self.known_encodings:
            return None

        face_distances = face_recognition.face_distance(self.known_encodings, unknown_encoding)
        best_match_idx = int(np.argmin(face_distances))

        if face_distances[best_match_idx] <= tolerance:
            return self.known_names[best_match_idx]
        return None
