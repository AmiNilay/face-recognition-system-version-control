import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    """Handle face detection operations"""
    
    def __init__(self, detection_method: str = "hog"):
        """
        Initialize face detector
        Args:
            detection_method: 'hog' (faster) or 'cnn' (more accurate)
        """
        self.detection_method = detection_method
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image
        Returns: List of face locations (top, right, bottom, left)
        """
        face_locations = face_recognition.face_locations(
            image, 
            model=self.detection_method
        )
        return face_locations
    
    def detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV (faster alternative)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        

        face_locations = []
        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))
        
        return face_locations
    
    def extract_face(self, image: np.ndarray, 
                    location: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face region from image"""
        top, right, bottom, left = location
        return image[top:bottom, left:right]
    
    def draw_face_rectangle(self, image: np.ndarray, 
                           location: Tuple[int, int, int, int],
                           name: str = "Unknown",
                           color: Tuple[int, int, int] = None) -> np.ndarray:
        """Draw rectangle around face with name"""
        top, right, bottom, left = location
        

        if color is None:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        

        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        

        cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
        
        return image