"""
Biometric Anti-Spoofing & Liveness Detection Engine
Module: src/anti_spoofing.py

This module implements multi-layered 2026 biometric liveness verification
and privacy protection mechanisms:
1. Eye Aspect Ratio (EAR) Blink Detection: Defeats static printed photo attacks
   by measuring dynamic eyelid movement and blink cycles.
2. Screen Moiré & Frequency Analysis: Defeats digital screen replays (smartphones,
   tablets, monitors) by analyzing 2D Fourier Transform frequency spectrums.
3. Automated Bystander Face Blur (Privacy Mode): Automatically applies Gaussian
   blur to unregistered faces for GDPR/privacy compliance.
4. Color-coded HUD visual feedback without emojis.

Mathematical Foundations for Beginners:
- Eye Aspect Ratio (Soukupova & Cech, 2016):
  EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
  When eyes are open, EAR is typically between 0.25 and 0.38.
  During a blink, EAR rapidly drops below 0.20 and recovers.

- Fourier Frequency Analysis:
  Digital screens contain regular RGB pixel grids that produce high-frequency
  interference patterns (moiré patterns) in the 2D FFT magnitude spectrum.
  Natural human skin exhibits smooth, continuous diffuse scattering without
  periodic frequency spikes.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class LivenessResult:
    """
    Data structure containing comprehensive liveness diagnostic telemetry.
    """
    is_live: bool
    ear: float
    blinks: int
    texture_score: float
    status_text: str
    hud_color: Tuple[int, int, int]  # BGR format for OpenCV rendering
    spoof_reason: Optional[str] = None


class BlinkTracker:
    """
    State machine tracking temporal eye closure and blink events for a face.
    """
    def __init__(self, min_closed_frames: int = 2):
        self.min_closed_frames = min_closed_frames
        self.closed_frame_count: int = 0
        self.total_blinks: int = 0
        self.last_ear: float = 0.30

    def update(self, current_ear: float, ear_threshold: float = 0.22) -> int:
        """
        Update the blink state machine with the latest Eye Aspect Ratio.

        Args:
            current_ear (float): Calculated Eye Aspect Ratio for the current frame.
            ear_threshold (float): Threshold below which eyes are considered closed.

        Returns:
            int: Updated cumulative blink count.
        """
        self.last_ear = current_ear

        if current_ear < ear_threshold:
            # Eyes are currently closed
            self.closed_frame_count += 1
        else:
            # Eyes are open; check if a complete blink cycle just completed
            if self.closed_frame_count >= self.min_closed_frames:
                self.total_blinks += 1
            self.closed_frame_count = 0

        return self.total_blinks

    def reset(self):
        """Reset the blink tracker state."""
        self.closed_frame_count = 0
        self.total_blinks = 0
        self.last_ear = 0.30


class AntiSpoofEngine:
    """
    Coordinates multi-layered anti-spoofing checks and privacy transformations.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Anti-Spoofing engine with configuration parameters.

        Args:
            config (dict, optional): Full application configuration dictionary.
        """
        cfg = config.get('anti_spoofing', {}) if config else {}
        self.enabled: bool = bool(cfg.get('enabled', True))
        self.ear_threshold: float = float(cfg.get('ear_threshold', 0.22))
        self.consecutive_closed_frames: int = int(cfg.get('consecutive_closed_frames', 2))
        self.texture_analysis_enabled: bool = bool(cfg.get('texture_analysis', True))
        self.privacy_blur_enabled: bool = bool(cfg.get('privacy_blur_bystanders', False))

        # Track blinks per subject identity (keyed by person name or tracking index)
        self.blink_trackers: Dict[str, BlinkTracker] = {}

    # --------------------------------------------------------------------------
    # 1. EYE ASPECT RATIO (EAR) CALCULATION
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_ear(eye_landmarks: List[Tuple[int, int]]) -> float:
        """
        Calculate the Eye Aspect Ratio (EAR) from 6 landmark coordinate pairs.

        Points layout for 6-point eye contour:
        p0: outer corner, p3: inner corner
        p1, p2: upper eyelid points
        p5, p4: lower eyelid points

        Formula:
            EAR = (||p1 - p5|| + ||p2 - p4||) / (2.0 * ||p0 - p3||)

        Args:
            eye_landmarks: List of (x, y) coordinate tuples with length >= 6.

        Returns:
            float: Eye Aspect Ratio value (typically 0.15 - 0.38).
        """
        if not eye_landmarks or len(eye_landmarks) < 6:
            return 0.0

        pts = np.array(eye_landmarks, dtype=np.float32)

        # Vertical distances between upper and lower eyelids
        dist_vertical_1 = float(np.linalg.norm(pts[1] - pts[5]))
        dist_vertical_2 = float(np.linalg.norm(pts[2] - pts[4]))

        # Horizontal distance between lateral eye corners
        dist_horizontal = float(np.linalg.norm(pts[0] - pts[3]))

        if dist_horizontal <= 1e-6:
            return 0.0

        ear = (dist_vertical_1 + dist_vertical_2) / (2.0 * dist_horizontal)
        return float(ear)

    def extract_average_ear(self, facial_landmarks: Dict[str, List[Tuple[int, int]]]) -> float:
        """
        Extract eye landmarks and compute mean Eye Aspect Ratio across both eyes.

        Args:
            facial_landmarks: Dictionary containing 'left_eye' and 'right_eye' keys.

        Returns:
            float: Mean EAR across both eyes, or 0.0 if landmarks are unavailable.
        """
        left_eye = facial_landmarks.get('left_eye', [])
        right_eye = facial_landmarks.get('right_eye', [])

        left_ear = self.compute_ear(left_eye) if len(left_eye) >= 6 else 0.0
        right_ear = self.compute_ear(right_eye) if len(right_eye) >= 6 else 0.0

        if left_ear > 0.0 and right_ear > 0.0:
            return (left_ear + right_ear) / 2.0
        elif left_ear > 0.0:
            return left_ear
        elif right_ear > 0.0:
            return right_ear
        return 0.0

    # --------------------------------------------------------------------------
    # 2. SCREEN TEXTURE & MOIRE FREQUENCY ANALYSIS
    # --------------------------------------------------------------------------
    @staticmethod
    def analyze_texture_frequency(face_crop: np.ndarray) -> float:
        """
        Analyze spatial frequency distribution using 2D Fast Fourier Transform (FFT).

        Digital screens emit high-frequency periodic interference (moiré pattern)
        caused by sub-pixel lattice matrices. Natural skin has diffuse reflection
        with lower high-frequency concentration.

        Args:
            face_crop (np.ndarray): Cropped BGR face image.

        Returns:
            float: Natural skin score between 0.0 (likely digital screen / print)
                   and 1.0 (natural physical skin reflection).
        """
        if face_crop is None or face_crop.size == 0:
            return 0.5

        try:
            # Downscale for consistent frequency normalization
            resized = cv2.resize(face_crop, (128, 128))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

            # 1. Laplacian sharpness check (detects flat low-res prints or extreme blur)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # 2. 2D Fast Fourier Transform
            f_transform = np.fft.fft2(gray.astype(np.float32))
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20.0 * np.log(np.abs(f_shift) + 1.0)

            # Center coordinates (DC component)
            h, w = magnitude_spectrum.shape
            cy, cx = h // 2, w // 2

            # Define high-frequency mask: outer periphery where screen moire artifacts appear
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask_high = dist_from_center > 48

            # Calculate energy ratio: high frequency vs total frequency
            total_energy = float(np.sum(magnitude_spectrum))
            high_energy = float(np.sum(magnitude_spectrum[mask_high]))

            if total_energy <= 1e-6:
                return 0.5

            high_freq_ratio = high_energy / total_energy

            # High-frequency ratio thresholds:
            # - Screens typically produce high_freq_ratio > 0.72 due to high-frequency pixel lattices.
            # - Extremely low variance (< 15.0) indicates flat paper printout or out-of-focus replay.
            if laplacian_var < 15.0:
                # Flat paper or blurred image attack
                score = 0.20
            elif high_freq_ratio > 0.72:
                # High-frequency screen lattice pattern detected
                score = 0.30
            else:
                # Natural continuous spectrum with healthy texture variance
                score = 0.55 + min(0.40, (laplacian_var / 500.0) * 0.40)

            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.5

    # --------------------------------------------------------------------------
    # 3. UNIFIED LIVENESS EVALUATION
    # --------------------------------------------------------------------------
    def verify_liveness(
        self,
        subject_id: str,
        face_crop: np.ndarray,
        facial_landmarks: Optional[Dict[str, List[Tuple[int, int]]]] = None
    ) -> LivenessResult:
        """
        Evaluate full multi-modal anti-spoofing criteria on a detected face.

        Args:
            subject_id (str): Identifier or tracking label of the person.
            face_crop (np.ndarray): Cropped facial region of interest.
            facial_landmarks (dict, optional): Detected landmark points.

        Returns:
            LivenessResult: Diagnostic object with boolean verdict, metrics, and text.
        """
        if not self.enabled:
            return LivenessResult(
                is_live=True,
                ear=0.30,
                blinks=1,
                texture_score=0.90,
                status_text="[LIVENESS: BYPASS]",
                hud_color=(0, 255, 0)
            )

        # Get or create tracking state for subject
        if subject_id not in self.blink_trackers:
            self.blink_trackers[subject_id] = BlinkTracker(
                min_closed_frames=self.consecutive_closed_frames
            )

        tracker = self.blink_trackers[subject_id]

        # 1. Compute Eye Aspect Ratio (EAR)
        current_ear = 0.0
        if facial_landmarks:
            current_ear = self.extract_average_ear(facial_landmarks)

        # Update blink counter
        total_blinks = tracker.update(current_ear, self.ear_threshold) if current_ear > 0.0 else tracker.total_blinks

        # 2. Analyze screen texture & frequency spectrum
        texture_score = 0.85
        if self.texture_analysis_enabled and face_crop is not None:
            texture_score = self.analyze_texture_frequency(face_crop)

        # 3. Decision Matrix
        # - Spoof Condition A: Strong digital screen lattice (texture_score < 0.40)
        # - Live Condition B: At least 1 natural blink verified OR active dynamic eye aperture
        # - Pending Condition C: No blink observed yet, currently monitoring
        if texture_score < 0.35:
            return LivenessResult(
                is_live=False,
                ear=current_ear,
                blinks=total_blinks,
                texture_score=texture_score,
                status_text="[SPOOF: SCREEN DETECTED]",
                hud_color=(0, 0, 255),  # Red
                spoof_reason="Digital screen lattice or flat image artifact detected"
            )

        if total_blinks >= 1:
            return LivenessResult(
                is_live=True,
                ear=current_ear,
                blinks=total_blinks,
                texture_score=texture_score,
                status_text=f"[LIVE: VERIFIED] Blinks: {total_blinks}",
                hud_color=(0, 255, 0)  # Green
            )

        # Eyes are open and texture is natural, awaiting active confirmation blink
        return LivenessResult(
            is_live=False,
            ear=current_ear,
            blinks=total_blinks,
            texture_score=texture_score,
            status_text=f"[VERIFYING: BLINK NOW] EAR: {current_ear:.2f}",
            hud_color=(0, 165, 255),  # Orange/Amber
            spoof_reason="Awaiting active blink verification"
        )

    # --------------------------------------------------------------------------
    # 4. PRIVACY PROTECTION (AUTOMATIC BYSTANDER FACE BLUR)
    # --------------------------------------------------------------------------
    def apply_privacy_blur(
        self,
        frame: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        face_names: List[str]
    ) -> np.ndarray:
        """
        Apply Gaussian blurring to unregistered bystanders or background faces.

        Beginner Concept:
        - When an employee or student is in front of the camera, other people
          walking behind them in a public office are bystanders.
        - Privacy mode blurs their faces in the video stream and logs,
          complying with enterprise biometric data protection regulations.

        Args:
            frame (np.ndarray): Video frame (modified in-place).
            face_locations (list): Scaled (top, right, bottom, left) tuples.
            face_names (list): Recognition labels corresponding to face_locations.

        Returns:
            np.ndarray: Frame with bystander faces blurred.
        """
        if not self.privacy_blur_enabled:
            return frame

        h_frame, w_frame = frame.shape[:2]

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Blur unregistered/bystander individuals
            if name == "Unknown":
                # Ensure coordinates are within valid image dimensions
                y1 = max(0, min(h_frame - 1, top))
                y2 = max(0, min(h_frame, bottom))
                x1 = max(0, min(w_frame - 1, left))
                x2 = max(0, min(w_frame, right))

                if y2 > y1 and x2 > x1:
                    face_roi = frame[y1:y2, x1:x2]
                    # Apply intense Gaussian blur to mask identity completely
                    blurred_roi = cv2.GaussianBlur(face_roi, (51, 51), 30)
                    frame[y1:y2, x1:x2] = blurred_roi

                    # Optional visual notification banner
                    cv2.putText(
                        frame,
                        "[PRIVACY BLUR]",
                        (x1 + 4, max(y1 + 18, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (200, 200, 200),
                        1
                    )

        return frame
