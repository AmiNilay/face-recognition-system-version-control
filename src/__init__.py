"""Face Recognition System Package"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .face_recognition_system import FaceRecognitionSystem
from .face_detector import FaceDetector
from .face_encoder import FaceEncoder
from .database_manager import DatabaseManager

__all__ = [
    'FaceRecognitionSystem',
    'FaceDetector', 
    'FaceEncoder',
    'DatabaseManager'
]