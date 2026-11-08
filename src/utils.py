"""
Utilities and Helper Functions Module

This module provides essential supporting functions used throughout the Face
Recognition System, including:
1. Indian Standard Time (IST) timekeeping and formatting helpers.
2. System logging configuration with IST timestamps.
3. YAML configuration loading with fallback defaults.
4. Image pre-processing (aspect ratio preservation, thumbnailing).
5. Real-time Frames Per Second (FPS) performance monitoring.
6. Dependency verification to ensure all necessary libraries are present.
"""

import os
import cv2
import yaml
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

# ------------------------------------------------------------------------------
# 1. TIMEZONE CONFIGURATION (Indian Standard Time - IST)
# ------------------------------------------------------------------------------
# Indian Standard Time is UTC + 5 hours and 30 minutes.
# We define a timezone instance so all timestamps across database logs,
# file names, and GUI displays are accurate in IST.
IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30), name="IST")

def get_ist_now() -> datetime:
    """
    Get the current date and time in Indian Standard Time (IST).

    Returns:
        datetime: Timezone-aware datetime object set to IST.
    """
    return datetime.now(IST_TIMEZONE)

def format_ist_timestamp(dt: Optional[datetime] = None, 
                         format_str: str = "%Y-%m-%d %H:%M:%S IST") -> str:
    """
    Format a datetime object into a readable IST string.

    Args:
        dt (datetime, optional): Datetime to format. Defaults to current IST time.
        format_str (str): The strftime format pattern. Defaults to 'YYYY-MM-DD HH:MM:SS IST'.

    Returns:
        str: Formatted timestamp string.
    """
    if dt is None:
        dt = get_ist_now()
    elif dt.tzinfo is None:
        # If naive datetime is provided, attach IST timezone
        dt = dt.replace(tzinfo=IST_TIMEZONE)
    else:
        # If another timezone is provided, convert to IST
        dt = dt.astimezone(IST_TIMEZONE)
    return dt.strftime(format_str)

def get_ist_date_string(dt: Optional[datetime] = None) -> str:
    """
    Get today's date in Indian Standard Time formatted as YYYY-MM-DD.

    Returns:
        str: Current IST date string (e.g. '2026-11-08').
    """
    if dt is None:
        dt = get_ist_now()
    return dt.strftime("%Y-%m-%d")

def get_ist_filename_timestamp() -> str:
    """
    Generate a filesystem-safe timestamp in IST for saving images and logs.

    Returns:
        str: Safe string suitable for filenames (e.g. '20261108_201945_IST').
    """
    return get_ist_now().strftime("%Y%m%d_%H%M%S_IST")


# ------------------------------------------------------------------------------
# 2. LOGGING CONFIGURATION
# ------------------------------------------------------------------------------
class ISTFormatter(logging.Formatter):
    """Custom logging formatter that renders timestamps strictly in IST."""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=IST_TIMEZONE)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S IST")

def setup_logging(log_file: str = "logs/app.log") -> logging.Logger:
    """
    Configure application logging to write to both console and a log file.

    Timestamps in logs are formatted in Indian Standard Time (IST).

    Args:
        log_file (str): Filepath where log records will be written.

    Returns:
        logging.Logger: Configured root logger instance.
    """
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("FaceRecognitionSystem")
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        formatter = ISTFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler: persists logs on disk
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler: prints logs to standard output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


# ------------------------------------------------------------------------------
# 3. CONFIGURATION MANAGEMENT
# ------------------------------------------------------------------------------
def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load system settings from a YAML configuration file.

    If the file does not exist or is empty, a standard default configuration
    is created automatically on disk and returned.

    Args:
        config_path (str): Relative or absolute path to config.yaml.

    Returns:
        dict: Parsed configuration dictionary.
    """
    config_file = Path(config_path)
    
    default_config: Dict[str, Any] = {
        'camera': {
            'device': 0,          # Default camera device index (0 = primary webcam)
            'width': 640,         # Video frame width in pixels
            'height': 480,        # Video frame height in pixels
            'fps': 30             # Desired video frames per second
        },
        'face_recognition': {
            'tolerance': 0.6,     # Distance threshold: lower = stricter match, higher = looser
            'model': 'hog',       # 'hog' is CPU-optimized; 'cnn' requires GPU acceleration
            'encoding_model': 'large', # 'large' uses 68 facial landmarks for higher precision
            'process_interval': 3 # Process recognition on 1 of every 3 frames for smooth FPS
        },
        'storage': {
            'known_faces_dir': 'known_faces',
            'captured_images_dir': 'captured_images',
            'encodings_file': 'data/face_encodings.pkl',
            'database_file': 'data/database.db',
            'history_file': 'data/recognition_history.csv'
        },
        'ui': {
            'window_name': 'Face Recognition System',
            'show_fps': True,
            'show_confidence': True,
            'cooldown_seconds': 5
        }
    }

    if not config_file.exists() or config_file.stat().st_size == 0:
        # Create directory and dump defaults if missing
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        return default_config

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
            return loaded if isinstance(loaded, dict) else default_config
    except Exception:
        return default_config


# ------------------------------------------------------------------------------
# 4. IMAGE PROCESSING & STORAGE UTILITIES
# ------------------------------------------------------------------------------
def save_image(image: np.ndarray, directory: str = "captured_images", 
               prefix: str = "capture") -> str:
    """
    Save an image frame to disk with an Indian Standard Time (IST) timestamp.

    Args:
        image (np.ndarray): OpenCV image in BGR or grayscale format.
        directory (str): Destination directory where the image will be stored.
        prefix (str): Filename prefix (e.g. 'capture', 'unknown', 'manual').

    Returns:
        str: Full path of the saved image file.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    timestamp = get_ist_filename_timestamp()
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)
    
    cv2.imwrite(filepath, image)
    return filepath

def resize_image(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """
    Resize an image down to max_width while strictly preserving its aspect ratio.

    Args:
        image (np.ndarray): Original OpenCV image array.
        max_width (int): Maximum allowable width in pixels.

    Returns:
        np.ndarray: Resized image or original image if already within limits.
    """
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / float(width)
        new_width = max_width
        new_height = int(height * ratio)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def create_thumbnail(image_path: str, size: Tuple[int, int] = (150, 150)) -> Optional[np.ndarray]:
    """
    Create a square thumbnail with centered aspect-ratio letterboxing.

    Args:
        image_path (str): Path to the source image file on disk.
        size (tuple): Target (width, height) dimensions.

    Returns:
        np.ndarray or None: Padded square thumbnail image, or None if reading fails.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        target_w, target_h = size
        h, w = img.shape[:2]
        aspect = w / float(h)
        
        if aspect > 1:
            new_w = target_w
            new_h = int(new_w / aspect)
        else:
            new_h = target_h
            new_w = int(new_h * aspect)
            
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create black square canvas and center the resized thumbnail
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
    except Exception as err:
        logging.getLogger("FaceRecognitionSystem").error(f"Thumbnail error for {image_path}: {err}")
        return None

def validate_image(image_path: str) -> bool:
    """
    Check if a given file path exists and is a readable, uncorrupted image.

    Args:
        image_path (str): Filepath to check.

    Returns:
        bool: True if file exists and can be decoded by OpenCV; False otherwise.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    path = Path(image_path)
    
    if not path.is_file() or path.suffix.lower() not in valid_extensions:
        return False
        
    try:
        img = cv2.imread(str(path))
        return img is not None and img.size > 0
    except Exception:
        return False


# ------------------------------------------------------------------------------
# 5. PERFORMANCE MONITORING (FPS TRACKING)
# ------------------------------------------------------------------------------
def calculate_fps(prev_time: float, current_time: float) -> float:
    """
    Calculate instantaneous Frames Per Second between two timestamps.

    Args:
        prev_time (float): Previous frame timestamp (in seconds).
        current_time (float): Current frame timestamp (in seconds).

    Returns:
        float: Frames per second rate. Returns 0 if time difference is negligible.
    """
    delta = current_time - prev_time
    if delta <= 0:
        return 0.0
    return 1.0 / delta

def draw_fps(image: np.ndarray, fps: float) -> np.ndarray:
    """
    Draw a clean, high-contrast FPS counter onto an image frame.

    Args:
        image (np.ndarray): Image array to draw on.
        fps (float): Frames per second to display.

    Returns:
        np.ndarray: Annotated image frame.
    """
    text = f"FPS: {fps:.1f}"
    # Draw dark shadow for contrast against bright scenes
    cv2.putText(image, text, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    # Draw bright green text over shadow
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image

class PerformanceMonitor:
    """
    Monitors video stream throughput, rolling FPS, and total runtime statistics.
    """
    def __init__(self):
        self.start_time: Optional[datetime] = None
        self.frame_count: int = 0
        self.fps_list: list = []

    def start(self):
        """Reset and start monitoring from the current instant."""
        self.start_time = get_ist_now()
        self.frame_count = 0
        self.fps_list = []

    def update(self) -> float:
        """
        Record a newly processed frame and compute average FPS since start.

        Returns:
            float: Current overall frames per second rate.
        """
        self.frame_count += 1
        if self.start_time:
            elapsed = (get_ist_now() - self.start_time).total_seconds()
            if elapsed > 0:
                fps = self.frame_count / elapsed
                self.fps_list.append(fps)
                return fps
        return 0.0

    def get_average_fps(self) -> float:
        """Return the mean FPS across all recorded samples."""
        if self.fps_list:
            return sum(self.fps_list) / float(len(self.fps_list))
        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        Produce a summary dictionary of runtime performance metrics.

        Returns:
            dict: Summary containing frame count, average FPS, and total runtime.
        """
        runtime = (get_ist_now() - self.start_time).total_seconds() if self.start_time else 0.0
        return {
            'frame_count': self.frame_count,
            'average_fps': self.get_average_fps(),
            'current_fps': self.fps_list[-1] if self.fps_list else 0.0,
            'runtime_seconds': runtime
        }


# ------------------------------------------------------------------------------
# 6. SYSTEM DEPENDENCY VERIFICATION
# ------------------------------------------------------------------------------
def check_requirements() -> bool:
    """
    Verify that all foundational computer vision and data libraries are importable.

    Returns:
        bool: True if all packages are ready; False if one or more are missing.
    """
    required_packages = [
        ('cv2', 'OpenCV (opencv-python)'),
        ('face_recognition', 'Face Recognition Library'),
        ('numpy', 'NumPy Numerical Library'),
        ('pandas', 'Pandas Data Analysis Library'),
        ('yaml', 'PyYAML Configuration Parser'),
        ('PIL', 'Pillow Imaging Library')
    ]
    
    missing_packages = []
    for module_name, user_friendly_name in required_packages:
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(user_friendly_name)

    if missing_packages:
        print("[ERROR] The following required packages are missing:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease run: pip install -r requirements.txt")
        return False
        
    return True
