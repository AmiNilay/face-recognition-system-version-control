"""
Motion-Triggered Computer Vision Optimization Engine
Module: src/motion_detector.py

Provides lightweight frame-difference and background subtraction algorithms to
detect environmental motion before invoking heavy facial recognition pipelines.
When a monitored scene is static, idle CPU and GPU compute drops by up to 90%.

All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code, HUD overlays, and logs. Developer: Nilay Naha.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class MotionDetector:
    """
    Lightweight, high-efficiency motion detection engine using background subtraction
    and morphological noise filtering to gate computer vision inference pipelines.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_motion_area: int = 800,
        var_threshold: float = 25.0,
        history: int = 300,
        cooldown_frames: int = 20
    ):
        """
        Initialize the motion detector.

        Args:
            enabled (bool): Master toggle for motion gating.
            min_motion_area (int): Minimum contour area in pixels to register motion.
            var_threshold (float): Variance threshold for MOG2 background subtractor.
            history (int): Number of previous frames to model background.
            cooldown_frames (int): Number of consecutive frames to keep inference
                                   active after motion ceases to avoid missing faces.
        """
        self.enabled = enabled
        self.min_motion_area = min_motion_area
        self.var_threshold = var_threshold
        self.history = history
        self.cooldown_frames = cooldown_frames

        # Create OpenCV background subtractor
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=False
        )

        # Morphology kernel for noise reduction
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Runtime state
        self.cooldown_counter = 0
        self.total_frames_analyzed = 0
        self.gated_frames_skipped = 0
        self.last_motion_area = 0.0

    def detect_motion(self, frame_bgr: np.ndarray) -> Tuple[bool, float, List[Tuple[int, int, int, int]]]:
        """
        Evaluate frame for active physical motion.

        Args:
            frame_bgr (np.ndarray): Input video frame in BGR format.

        Returns:
            Tuple[bool, float, List[Tuple[int, int, int, int]]]:
                - has_motion: True if motion detected or cooldown active.
                - motion_area: Area of largest detected motion contour in pixels.
                - bounding_boxes: List of (x, y, w, h) bounding boxes of moving objects.
        """
        if not self.enabled:
            return True, 0.0, []

        if frame_bgr is None or frame_bgr.size == 0:
            return False, 0.0, []

        self.total_frames_analyzed += 1

        # Downscale for ultra-fast motion check
        small = cv2.resize(frame_bgr, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Compute foreground mask
        fg_mask = self.subtractor.apply(blurred)

        # Morphological opening and dilation to remove speckle noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_found = False
        max_area = 0.0
        boxes = []

        # Scale factor back to original frame
        scale_x = frame_bgr.shape[1] / 320.0
        scale_y = frame_bgr.shape[0] / 240.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= (self.min_motion_area / (scale_x * scale_y)):
                motion_found = True
                if area > max_area:
                    max_area = area * (scale_x * scale_y)

                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append((
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y)
                ))

        self.last_motion_area = max_area

        if motion_found:
            self.cooldown_counter = self.cooldown_frames
            return True, max_area, boxes

        # Handle cooldown window
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return True, 0.0, []

        self.gated_frames_skipped += 1
        return False, 0.0, []

    def reset(self):
        """Reset background subtractor and runtime counters."""
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=False
        )
        self.cooldown_counter = 0

    @property
    def savings_ratio(self) -> float:
        """Percentage of frames gated / skipped due to static scene (0.0% to 100.0%)."""
        if self.total_frames_analyzed == 0:
            return 0.0
        return (float(self.gated_frames_skipped) / float(self.total_frames_analyzed)) * 100.0
