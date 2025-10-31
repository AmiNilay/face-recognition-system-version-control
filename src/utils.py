"""Utility Functions"""

import os
import cv2
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

def setup_logging(log_file: str = "logs/app.log"):
    """Setup logging configuration"""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Create default config
        default_config = {
            'camera': {
                'device': 0,
                'width': 640,
                'height': 480,
                'fps': 30
            },
            'face_recognition': {
                'tolerance': 0.6,
                'model': 'hog',
                'encoding_model': 'large'
            },
            'storage': {
                'known_faces_dir': 'known_faces',
                'captured_images_dir': 'captured_images',
                'encodings_file': 'data/face_encodings.pkl',
                'database_file': 'data/database.db'
            },
            'ui': {
                'window_name': 'Face Recognition System',
                'show_fps': True,
                'show_confidence': True
            }
        }
        
        # Save default config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_image(image: np.ndarray, directory: str = "captured_images", 
               prefix: str = "capture") -> str:
    """Save image to file with timestamp"""
    # Create directory if not exists
    Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)
    
    # Save image
    cv2.imwrite(filepath, image)
    
    return filepath

def resize_image(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resize image maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    if width > max_width:
        ratio = max_width / width
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return cv2.resize(image, (new_width, new_height))
    
    return image

def calculate_fps(prev_time: float, current_time: float) -> float:
    """Calculate frames per second"""
    if prev_time == 0:
        return 0
    return 1 / (current_time - prev_time)

def draw_fps(image: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS on image"""
    cv2.putText(image, f"FPS: {fps:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    return image

def check_requirements() -> bool:
    """Check if all required packages are installed"""
    required_packages = [
        'cv2',
        'face_recognition',
        'numpy',
        'pandas',
        'yaml',
        'PIL'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def validate_image(image_path: str) -> bool:
    """Validate if file is a valid image"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    path = Path(image_path)
    if not path.exists():
        return False
    
    if path.suffix.lower() not in valid_extensions:
        return False
    
    # Try to read image
    try:
        img = cv2.imread(str(path))
        return img is not None
    except:
        return False

def create_thumbnail(image_path: str, size: tuple = (150, 150)) -> Optional[np.ndarray]:
    """Create thumbnail from image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Calculate aspect ratio
        h, w = img.shape[:2]
        aspect = w / h
        
        if aspect > 1:
            new_w = size[0]
            new_h = int(new_w / aspect)
        else:
            new_h = size[1]
            new_w = int(new_h * aspect)
        
        # Resize image
        thumbnail = cv2.resize(img, (new_w, new_h))
        
        # Create canvas and center image
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        y_offset = (size[1] - new_h) // 2
        x_offset = (size[0] - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = thumbnail
        
        return canvas
    except Exception as e:
        logging.error(f"Error creating thumbnail: {e}")
        return None

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.start_time = None
        self.frame_count = 0
        self.fps_list = []
    
    def start(self):
        """Start monitoring"""
        self.start_time = datetime.now()
        self.frame_count = 0
        self.fps_list = []
    
    def update(self) -> float:
        """Update frame count and return current FPS"""
        self.frame_count += 1
        
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed > 0:
                fps = self.frame_count / elapsed
                self.fps_list.append(fps)
                return fps
        
        return 0
    
    def get_average_fps(self) -> float:
        """Get average FPS"""
        if self.fps_list:
            return sum(self.fps_list) / len(self.fps_list)
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'frame_count': self.frame_count,
            'average_fps': self.get_average_fps(),
            'current_fps': self.fps_list[-1] if self.fps_list else 0,
            'runtime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }