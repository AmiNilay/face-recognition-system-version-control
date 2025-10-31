import face_recognition
import numpy as np
import pickle
from typing import List, Dict, Optional
from pathlib import Path

class FaceEncoder:
    """Handle face encoding operations"""
    
    def __init__(self, encoding_model: str = "large"):
        """
        Initialize face encoder
        Args:
            encoding_model: 'large' or 'small'
        """
        self.encoding_model = encoding_model
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
    
    def encode_face(self, image: np.ndarray, 
                   face_location: Optional[tuple] = None) -> Optional[np.ndarray]:
        """
        Encode a single face
        Returns: Face encoding or None if no face found
        """
        if face_location:
            encodings = face_recognition.face_encodings(image, [face_location])
        else:
            encodings = face_recognition.face_encodings(image)
        
        return encodings[0] if encodings else None
    
    def encode_faces(self, image: np.ndarray, 
                    face_locations: List[tuple]) -> List[np.ndarray]:
        """Encode multiple faces"""
        return face_recognition.face_encodings(image, face_locations)
    
    def compare_faces(self, known_encoding: np.ndarray, 
                     unknown_encoding: np.ndarray,
                     tolerance: float = 0.6) -> bool:
        """Compare two face encodings"""
        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
        return distance <= tolerance
    
    def find_match(self, unknown_encoding: np.ndarray, 
                  tolerance: float = 0.6) -> Optional[str]:
        """
        Find matching face in known faces
        Returns: Name of matched person or None
        """
        if not self.known_encodings:
            return None
        
        matches = face_recognition.compare_faces(
            self.known_encodings, 
            unknown_encoding, 
            tolerance
        )
        face_distances = face_recognition.face_distance(
            self.known_encodings, 
            unknown_encoding
        )
        
        if True in matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                return self.known_names[best_match_index]
        
        return None
    
    def add_known_face(self, encoding: np.ndarray, name: str):
        """Add a known face encoding"""
        self.known_encodings.append(encoding)
        self.known_names.append(name)
    
    def save_encodings(self, filepath: Path):
        """Save encodings to file"""
        data = {
            'encodings': self.known_encodings,
            'names': self.known_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_encodings(self, filepath: Path) -> bool:
        """Load encodings from file"""
        if not filepath.exists():
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.known_encodings = data['encodings']
        self.known_names = data['names']
        return True