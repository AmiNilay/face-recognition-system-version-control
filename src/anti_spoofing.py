"""
Biometric Anti-Spoofing & Liveness Detection Engine
Module: src/anti_spoofing.py

This module implements multi-layered 2026 biometric liveness verification,
3D head pose estimation, interactive challenge-response, and privacy protection:
1. Eye Aspect Ratio (EAR) Blink Detection: Defeats static printed photo attacks
   by measuring dynamic eyelid movement and blink cycles.
2. Screen Moiré & Frequency Analysis: Defeats digital screen replays (smartphones,
   tablets, monitors) by analyzing 2D Fourier Transform frequency spectrums.
3. 3D Head Pose Estimation (solvePnP): Solves the Perspective-n-Point problem
   mapping 2D facial landmarks onto a canonical 3D anthropometric skull model
   to determine 3D orientation (Yaw, Pitch, Roll) and project a 3D nose ray.
4. Interactive Challenge-Response: Defeats prerecorded deepfake video replay loops
   by issuing randomized temporal challenges (e.g. turn head left/right, nod down)
   with strict timeout expiration.
5. Automated Bystander Face Blur (Privacy Mode): Automatically applies Gaussian
   blur to unregistered faces for GDPR/privacy compliance.
6. Zero emojis across all code, HUD readouts, and logs. Developer: Nilay Naha.

Mathematical Foundations for Beginners:
- Eye Aspect Ratio (Soukupova & Cech, 2016):
  EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
  When eyes are open, EAR is typically between 0.25 and 0.38.
  During a blink, EAR rapidly drops below 0.20 and recovers.

- Fourier Frequency Analysis:
  Digital screens contain regular RGB sub-pixel lattices that produce high-frequency
  interference patterns (moiré patterns) in the 2D FFT magnitude spectrum.
  Natural human skin exhibits smooth, continuous diffuse scattering without
  periodic frequency spikes.

- Perspective-n-Point (PnP) 3D Head Pose:
  Estimates the rotation vector (rvec) and translation vector (tvec) of the head
  relative to the camera coordinate system by minimizing reprojection error between
  known 3D facial landmarks (nose tip, chin, eye corners, mouth corners) and their
  2D camera image coordinates:
  s * [u, v, 1]^T = K * [R | t] * [X, Y, Z, 1]^T
  Euler angles (Yaw, Pitch, Roll) are derived via RQ-decomposition of the rotation matrix.
"""

import time
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

# ------------------------------------------------------------------------------
# DATA STRUCTURES & CONSTANTS
# ------------------------------------------------------------------------------

class ChallengeType:
    """Supported interactive liveness challenge types."""
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    NOD_DOWN = "NOD_DOWN"
    TILT_UP = "TILT_UP"
    BLINK = "BLINK"

CHALLENGE_PROMPTS = {
    ChallengeType.TURN_LEFT: "TURN HEAD LEFT",
    ChallengeType.TURN_RIGHT: "TURN HEAD RIGHT",
    ChallengeType.NOD_DOWN: "NOD HEAD DOWN",
    ChallengeType.TILT_UP: "TILT HEAD UP",
    ChallengeType.BLINK: "BLINK EYES"
}

STATUS_PENDING = "PENDING"
STATUS_PASSED = "PASSED"
STATUS_TIMEOUT = "TIMEOUT"

# Canonical 3D anthropometric face model coordinates in millimeters (OpenCV camera coords: +X right, +Y down, +Z forward)
MODEL_POINTS_6 = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, 330.0, -65.0),         # Chin
    (-225.0, -170.0, -135.0),    # Left eye outer corner (viewer's left)
    (225.0, -170.0, -135.0),     # Right eye outer corner (viewer's right)
    (-150.0, 150.0, -125.0),     # Left mouth corner
    (150.0, 150.0, -125.0)       # Right mouth corner
], dtype=np.float64)


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
    head_pose: Optional[Tuple[float, float, float]] = None  # (yaw, pitch, roll) in degrees
    challenge_status: Optional[str] = None                  # PENDING, PASSED, TIMEOUT
    challenge_prompt: Optional[str] = None                  # e.g. "TURN HEAD LEFT (3.2s)"
    nose_ray: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None  # ((x1, y1), (x2, y2))


# ------------------------------------------------------------------------------
# 1. BLINK TRACKER STATE MACHINE
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# 2. INTERACTIVE CHALLENGE-RESPONSE TRACKER
# ------------------------------------------------------------------------------

class ChallengeResponseTracker:
    """
    State machine managing interactive anti-spoofing challenge-response challenges.
    Defeats video replay attacks by demanding dynamic head movement within a timeout.
    """
    def __init__(
        self,
        challenge_timeout: float = 5.0,
        initial_challenge: Optional[str] = None,
        current_time: Optional[float] = None
    ):
        """
        Initialize the challenge-response state machine.

        Args:
            challenge_timeout (float): Maximum seconds before challenge times out.
            initial_challenge (str, optional): Specific challenge type to begin with.
            current_time (float, optional): Custom timestamp for deterministic testing.
        """
        self.timeout = float(challenge_timeout)
        self.active_challenge: str = initial_challenge or ChallengeType.TURN_LEFT
        self.status: str = STATUS_PENDING
        self.issued_at: float = current_time if current_time is not None else time.time()
        self.base_blinks: int = 0
        self.passed_at: Optional[float] = None
        self.challenge_sequence = [
            ChallengeType.TURN_LEFT,
            ChallengeType.TURN_RIGHT,
            ChallengeType.NOD_DOWN
        ]
        self.seq_index: int = 0

    def reset(
        self,
        challenge: Optional[str] = None,
        current_blinks: int = 0,
        current_time: Optional[float] = None
    ):
        """
        Reset tracker with a new challenge or next challenge in rotation.

        Args:
            challenge (str, optional): Specific challenge type to assign.
            current_blinks (int): Current blink count to set as baseline.
            current_time (float, optional): Custom timestamp for deterministic testing.
        """
        if challenge is not None:
            self.active_challenge = challenge
        else:
            self.seq_index = (self.seq_index + 1) % len(self.challenge_sequence)
            self.active_challenge = self.challenge_sequence[self.seq_index]

        self.status = STATUS_PENDING
        self.issued_at = current_time if current_time is not None else time.time()
        self.base_blinks = current_blinks
        self.passed_at = None

    def update(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        current_blinks: int,
        current_time: Optional[float] = None
    ) -> str:
        """
        Evaluate subject head angles and blinks against the active challenge criteria.

        Args:
            yaw (float): Head yaw angle in degrees (negative = left, positive = right).
            pitch (float): Head pitch angle in degrees (positive = down, negative = up).
            roll (float): Head roll angle in degrees.
            current_blinks (int): Current cumulative blink count.
            current_time (float, optional): Custom timestamp for testing.

        Returns:
            str: Updated status (PENDING, PASSED, or TIMEOUT).
        """
        if self.status == STATUS_PASSED:
            return STATUS_PASSED

        now = current_time if current_time is not None else time.time()
        elapsed = now - self.issued_at

        # Check for timeout expiration
        if elapsed > self.timeout:
            self.status = STATUS_TIMEOUT
            return self.status

        # Evaluate target movement criteria
        passed = False
        if self.active_challenge == ChallengeType.TURN_LEFT and yaw < -12.0:
            passed = True
        elif self.active_challenge == ChallengeType.TURN_RIGHT and yaw > 12.0:
            passed = True
        elif self.active_challenge == ChallengeType.NOD_DOWN and pitch > 10.0:
            passed = True
        elif self.active_challenge == ChallengeType.TILT_UP and pitch < -10.0:
            passed = True
        elif self.active_challenge == ChallengeType.BLINK and current_blinks > self.base_blinks:
            passed = True

        if passed:
            self.status = STATUS_PASSED
            self.passed_at = now

        return self.status

    def get_prompt(self, current_time: Optional[float] = None) -> str:
        """
        Get human-readable prompt string with countdown time.

        Args:
            current_time (float, optional): Custom timestamp for testing.

        Returns:
            str: Formatted instruction text.
        """
        if self.status == STATUS_PASSED:
            return f"{CHALLENGE_PROMPTS.get(self.active_challenge, self.active_challenge)} [PASSED]"
        if self.status == STATUS_TIMEOUT:
            return "[CHALLENGE TIMED OUT]"

        now = current_time if current_time is not None else time.time()
        remaining = max(0.0, self.timeout - (now - self.issued_at))
        prompt_base = CHALLENGE_PROMPTS.get(self.active_challenge, self.active_challenge)
        return f"{prompt_base} ({remaining:.1f}s)"

    def is_verified(self) -> bool:
        """Check if challenge has been successfully passed."""
        return self.status == STATUS_PASSED


# ------------------------------------------------------------------------------
# 3. ANTI-SPOOFING MAIN ENGINE
# ------------------------------------------------------------------------------

class AntiSpoofEngine:
    """
    Coordinates multi-layered anti-spoofing checks, 3D head pose estimation,
    challenge-response verification, and privacy transformations.
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

        # Advanced Phase 3 features
        self.head_pose_enabled: bool = bool(cfg.get('head_pose', True))
        self.challenge_response_enabled: bool = bool(cfg.get('challenge_response', False))
        self.challenge_timeout: float = float(cfg.get('challenge_timeout', 5.0))

        # Track blinks per subject identity
        self.blink_trackers: Dict[str, BlinkTracker] = {}

        # Track interactive challenges per subject identity
        self.challenge_trackers: Dict[str, ChallengeResponseTracker] = {}

    # --------------------------------------------------------------------------
    # EYE ASPECT RATIO (EAR) CALCULATION
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
    # SCREEN TEXTURE & MOIRE FREQUENCY ANALYSIS
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
    # 3D HEAD POSE ESTIMATION (PERSPECTIVE-N-POINT / solvePnP)
    # --------------------------------------------------------------------------
    @staticmethod
    def extract_6_landmarks(
        landmarks: Union[Dict[str, List[Tuple[int, int]]], List[Tuple[int, int]], np.ndarray],
        scale: float = 1.0
    ) -> Optional[np.ndarray]:
        """
        Extract the 6 canonical 2D landmark coordinates required for solvePnP:
        [Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner].

        Args:
            landmarks: Either a dictionary from face_recognition.face_landmarks,
                       a list of 6 points, or a 68-point list.
            scale (float): Coordinate scaling factor (e.g. 4.0 for upscaling from downscaled frames).

        Returns:
            np.ndarray: (6, 2) float64 array of image coordinates, or None if points are missing.
        """
        if landmarks is None:
            return None

        try:
            if isinstance(landmarks, dict):
                # Standard face_recognition.face_landmarks dict
                nose_bridge = landmarks.get('nose_bridge', [])
                chin = landmarks.get('chin', [])
                left_eye = landmarks.get('left_eye', [])
                right_eye = landmarks.get('right_eye', [])
                top_lip = landmarks.get('top_lip', [])

                if not (nose_bridge and chin and left_eye and right_eye and top_lip):
                    return None

                # Nose tip: last point of nose bridge (point 30 in 68-pt model)
                p_nose = nose_bridge[-1]
                # Chin: middle point (point 8 in 68-pt model)
                p_chin = chin[len(chin) // 2] if len(chin) > 8 else chin[-1]
                # Left eye outer corner (point 36)
                p_left_eye = left_eye[0]
                # Right eye outer corner (point 45)
                p_right_eye = right_eye[3] if len(right_eye) > 3 else right_eye[-1]
                # Mouth corners: left (point 48) and right (point 54)
                p_left_mouth = top_lip[0]
                p_right_mouth = top_lip[6] if len(top_lip) > 6 else top_lip[-1]

                pts_6 = [p_nose, p_chin, p_left_eye, p_right_eye, p_left_mouth, p_right_mouth]

            elif isinstance(landmarks, (list, np.ndarray)):
                raw_list = list(landmarks)
                if len(raw_list) == 6:
                    pts_6 = raw_list
                elif len(raw_list) >= 68:
                    # Map from standard 68-point facial landmarks
                    pts_6 = [
                        raw_list[30],  # Nose tip
                        raw_list[8],   # Chin bottom
                        raw_list[36],  # Left eye outer
                        raw_list[45],  # Right eye outer
                        raw_list[48],  # Left mouth corner
                        raw_list[54]   # Right mouth corner
                    ]
                else:
                    return None
            else:
                return None

            pts_array = np.array(pts_6, dtype=np.float64)
            if scale != 1.0:
                pts_array *= float(scale)

            return pts_array

        except Exception:
            return None

    def estimate_head_pose(
        self,
        landmarks: Union[Dict[str, List[Tuple[int, int]]], List[Tuple[int, int]], np.ndarray],
        frame_shape: Tuple[int, int],
        scale: float = 1.0
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Estimate 3D head pose Euler angles (Yaw, Pitch, Roll) and 3D nose projection vector
        using Perspective-n-Point (cv2.solvePnP).

        Args:
            landmarks: Facial landmarks dict or coordinate points array.
            frame_shape (tuple): (height, width) of the video frame.
            scale (float): Scale factor to map coordinates to full frame dimensions.

        Returns:
            tuple: ((yaw, pitch, roll) in degrees, ((x1, y1), (x2, y2)) nose ray endpoints)
                   or (None, None) if estimation fails.
        """
        image_points = self.extract_6_landmarks(landmarks, scale=scale)
        if image_points is None or len(image_points) < 6:
            return None, None

        h, w = frame_shape[:2]
        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)

        # Standard pinhole camera matrix:
        # | f   0   cx |
        # | 0   f   cy |
        # | 0   0   1  |
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_6,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None, None

            # Convert rotation vector to 3x3 rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)

            # Decompose rotation matrix into Euler angles
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch = float(angles[0])
            yaw = float(angles[1])
            roll = float(angles[2])
            pose = (round(yaw, 1), round(pitch, 1), round(roll, 1))

            # Project 3D vector forward from nose tip along the Z axis (500mm ahead)
            nose_end_3d = np.array([[0.0, 0.0, 500.0]], dtype=np.float64)
            nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rvec, tvec, camera_matrix, dist_coeffs)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))
            nose_ray = (p1, p2)

            return pose, nose_ray

        except Exception:
            return None, None

    # --------------------------------------------------------------------------
    # UNIFIED LIVENESS & ANTI-SPOOFING VERIFICATION
    # --------------------------------------------------------------------------
    def verify_liveness(
        self,
        subject_id: str,
        face_crop: Optional[np.ndarray],
        facial_landmarks: Optional[Union[Dict[str, List[Tuple[int, int]]], Any]] = None,
        frame_shape: Tuple[int, int] = (480, 640),
        scale: float = 1.0,
        current_time: Optional[float] = None
    ) -> LivenessResult:
        """
        Evaluate full multi-modal anti-spoofing criteria on a detected face.

        Args:
            subject_id (str): Identifier or tracking label of the person.
            face_crop (np.ndarray, optional): Cropped facial region of interest.
            facial_landmarks (dict, optional): Detected landmark points.
            frame_shape (tuple): Height and width of reference frame.
            scale (float): Scale multiplier from detection crop to reference frame.
            current_time (float, optional): Custom timestamp for deterministic testing.

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

        # 1. Track blink events
        if subject_id not in self.blink_trackers:
            self.blink_trackers[subject_id] = BlinkTracker(
                min_closed_frames=self.consecutive_closed_frames
            )
        b_tracker = self.blink_trackers[subject_id]

        current_ear = 0.0
        if isinstance(facial_landmarks, dict):
            current_ear = self.extract_average_ear(facial_landmarks)

        total_blinks = b_tracker.update(current_ear, self.ear_threshold) if current_ear > 0.0 else b_tracker.total_blinks

        # 2. Analyze screen texture & frequency spectrum
        texture_score = 0.85
        if self.texture_analysis_enabled and face_crop is not None:
            texture_score = self.analyze_texture_frequency(face_crop)

        # 3. 3D Head Pose Estimation
        head_pose = None
        nose_ray = None
        if self.head_pose_enabled and facial_landmarks is not None:
            head_pose, nose_ray = self.estimate_head_pose(
                facial_landmarks, frame_shape=frame_shape, scale=scale
            )

        # 4. Interactive Challenge-Response Evaluation
        challenge_status = None
        challenge_prompt = None
        if self.challenge_response_enabled:
            if subject_id not in self.challenge_trackers:
                self.challenge_trackers[subject_id] = ChallengeResponseTracker(
                    challenge_timeout=self.challenge_timeout,
                    current_time=current_time
                )
                self.challenge_trackers[subject_id].base_blinks = total_blinks

            c_tracker = self.challenge_trackers[subject_id]
            yaw_val = head_pose[0] if head_pose else 0.0
            pitch_val = head_pose[1] if head_pose else 0.0
            roll_val = head_pose[2] if head_pose else 0.0

            challenge_status = c_tracker.update(
                yaw_val, pitch_val, roll_val, total_blinks, current_time=current_time
            )
            challenge_prompt = c_tracker.get_prompt(current_time=current_time)

        # ----------------------------------------------------------------------
        # 5. DECISION ENGINE
        # ----------------------------------------------------------------------

        # Check 1: Screen replay or flat printed attack
        if texture_score < 0.35:
            return LivenessResult(
                is_live=False,
                ear=current_ear,
                blinks=total_blinks,
                texture_score=texture_score,
                status_text="[SPOOF: SCREEN DETECTED]",
                hud_color=(0, 0, 255),  # Red
                spoof_reason="Digital screen lattice or flat image artifact detected",
                head_pose=head_pose,
                challenge_status=challenge_status,
                challenge_prompt=challenge_prompt,
                nose_ray=nose_ray
            )

        # Check 2: Challenge-Response mode active
        if self.challenge_response_enabled and challenge_status is not None:
            if challenge_status == STATUS_PASSED:
                return LivenessResult(
                    is_live=True,
                    ear=current_ear,
                    blinks=total_blinks,
                    texture_score=texture_score,
                    status_text=f"[LIVE: CHALLENGE PASSED] {challenge_prompt}",
                    hud_color=(0, 255, 0),  # Green
                    head_pose=head_pose,
                    challenge_status=challenge_status,
                    challenge_prompt=challenge_prompt,
                    nose_ray=nose_ray
                )
            elif challenge_status == STATUS_TIMEOUT:
                return LivenessResult(
                    is_live=False,
                    ear=current_ear,
                    blinks=total_blinks,
                    texture_score=texture_score,
                    status_text="[SPOOF: CHALLENGE TIMEOUT]",
                    hud_color=(0, 0, 255),  # Red
                    spoof_reason="Failed to respond to dynamic liveness challenge within timeout",
                    head_pose=head_pose,
                    challenge_status=challenge_status,
                    challenge_prompt=challenge_prompt,
                    nose_ray=nose_ray
                )
            else:  # PENDING
                return LivenessResult(
                    is_live=False,
                    ear=current_ear,
                    blinks=total_blinks,
                    texture_score=texture_score,
                    status_text=f"[CHALLENGE] {challenge_prompt}",
                    hud_color=(0, 255, 255),  # Cyan/Yellow
                    spoof_reason="Awaiting active challenge response",
                    head_pose=head_pose,
                    challenge_status=challenge_status,
                    challenge_prompt=challenge_prompt,
                    nose_ray=nose_ray
                )

        # Check 3: Standard Blink-based Liveness Verification
        if total_blinks >= 1:
            return LivenessResult(
                is_live=True,
                ear=current_ear,
                blinks=total_blinks,
                texture_score=texture_score,
                status_text=f"[LIVE: VERIFIED] Blinks: {total_blinks}",
                hud_color=(0, 255, 0),  # Green
                head_pose=head_pose,
                challenge_status=challenge_status,
                challenge_prompt=challenge_prompt,
                nose_ray=nose_ray
            )

        # Default state: eyes open and texture clear, awaiting verification blink
        return LivenessResult(
            is_live=False,
            ear=current_ear,
            blinks=total_blinks,
            texture_score=texture_score,
            status_text=f"[VERIFYING: BLINK NOW] EAR: {current_ear:.2f}",
            hud_color=(0, 165, 255),  # Orange/Amber
            spoof_reason="Awaiting active blink verification",
            head_pose=head_pose,
            challenge_status=challenge_status,
            challenge_prompt=challenge_prompt,
            nose_ray=nose_ray
        )

    # --------------------------------------------------------------------------
    # PRIVACY PROTECTION (AUTOMATIC BYSTANDER FACE BLUR)
    # --------------------------------------------------------------------------
    def apply_privacy_blur(
        self,
        frame: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        face_names: List[str]
    ) -> np.ndarray:
        """
        Apply Gaussian blurring to unregistered bystanders or background faces.

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
            if name == "Unknown":
                y1 = max(0, min(h_frame - 1, top))
                y2 = max(0, min(h_frame, bottom))
                x1 = max(0, min(w_frame - 1, left))
                x2 = max(0, min(w_frame, right))

                if y2 > y1 and x2 > x1:
                    face_roi = frame[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(face_roi, (51, 51), 30)
                    frame[y1:y2, x1:x2] = blurred_roi

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
