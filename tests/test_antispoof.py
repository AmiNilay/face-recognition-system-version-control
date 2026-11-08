"""
Unit Tests for Biometric Anti-Spoofing Engine (tests/test_antispoof.py)

Tests Eye Aspect Ratio (EAR), blink state machine, screen texture analysis,
and automated bystander privacy blurring.
All timestamps follow Indian Standard Time (IST).
Zero emojis.
"""

import unittest
import numpy as np
import cv2
from src.anti_spoofing import (
    AntiSpoofEngine,
    BlinkTracker,
    ChallengeResponseTracker,
    ChallengeType,
    STATUS_PENDING,
    STATUS_PASSED,
    STATUS_TIMEOUT,
    MODEL_POINTS_6,
    LivenessResult
)

class TestAntiSpoofing(unittest.TestCase):
    """Test suite for 2026 Anti-Spoofing and Liveness verification."""

    def setUp(self):
        """Set up AntiSpoofEngine with test configuration."""
        self.config = {
            'anti_spoofing': {
                'enabled': True,
                'ear_threshold': 0.22,
                'consecutive_closed_frames': 2,
                'texture_analysis': True,
                'privacy_blur_bystanders': True
            }
        }
        self.engine = AntiSpoofEngine(self.config)

    def test_compute_ear_open_vs_closed(self):
        """Verify Eye Aspect Ratio calculation on open vs closed eye geometries."""
        # 1. Open eye coordinates: width = 20, vertical distances = 6 and 6
        # EAR = (6 + 6) / (2 * 20) = 12 / 40 = 0.30
        open_eye = [
            (0, 10),   # p0 (outer corner)
            (5, 4),    # p1 (upper lid left)
            (15, 4),   # p2 (upper lid right)
            (20, 10),  # p3 (inner corner)
            (15, 10),  # p4 (lower lid right)
            (5, 10)    # p5 (lower lid left)
        ]
        ear_open = self.engine.compute_ear(open_eye)
        self.assertAlmostEqual(ear_open, 0.30, places=2)

        # 2. Closed eye coordinates: width = 20, vertical distances = 2 and 2
        # EAR = (2 + 2) / (2 * 20) = 4 / 40 = 0.10
        closed_eye = [
            (0, 10),   # p0
            (5, 9),    # p1
            (15, 9),   # p2
            (20, 10),  # p3
            (15, 11),  # p4
            (5, 11)    # p5
        ]
        ear_closed = self.engine.compute_ear(closed_eye)
        self.assertAlmostEqual(ear_closed, 0.10, places=2)
        self.assertLess(ear_closed, self.engine.ear_threshold)

    def test_compute_ear_degenerate_inputs(self):
        """Verify EAR handles degenerate or empty landmark coordinates safely."""
        self.assertEqual(self.engine.compute_ear([]), 0.0)
        self.assertEqual(self.engine.compute_ear([(0, 0), (0, 0)]), 0.0)
        # Identical horizontal corners
        zero_width = [(10, 10)] * 6
        self.assertEqual(self.engine.compute_ear(zero_width), 0.0)

    def test_blink_tracker_state_machine(self):
        """Verify blink state machine transitions and blink counting."""
        tracker = BlinkTracker(min_closed_frames=2)
        self.assertEqual(tracker.total_blinks, 0)

        # Frame 1: Eyes open
        tracker.update(0.32, ear_threshold=0.22)
        self.assertEqual(tracker.total_blinks, 0)

        # Frame 2: Eyes closed (frame 1 of closed)
        tracker.update(0.15, ear_threshold=0.22)
        self.assertEqual(tracker.total_blinks, 0)

        # Frame 3: Eyes closed (frame 2 of closed)
        tracker.update(0.14, ear_threshold=0.22)
        self.assertEqual(tracker.total_blinks, 0)

        # Frame 4: Eyes reopen -> blink detected!
        blinks = tracker.update(0.30, ear_threshold=0.22)
        self.assertEqual(blinks, 1)
        self.assertEqual(tracker.total_blinks, 1)

    def test_screen_texture_frequency_analysis(self):
        """Verify 2D FFT frequency analysis produces normalized texture scores."""
        # 1. Natural gradient image
        gradient = np.zeros((128, 128, 3), dtype=np.uint8)
        for r in range(128):
            gradient[r, :, :] = (r, 128 - r, (r * 2) % 255)
        score_gradient = self.engine.analyze_texture_frequency(gradient)
        self.assertGreaterEqual(score_gradient, 0.0)
        self.assertLessEqual(score_gradient, 1.0)

        # 2. Extreme flat printout (low variance)
        flat = np.full((128, 128, 3), 128, dtype=np.uint8)
        score_flat = self.engine.analyze_texture_frequency(flat)
        self.assertLess(score_flat, 0.40)

    def test_privacy_blur_bystanders(self):
        """Verify automated bystander blurring only alters unregistered faces."""
        frame = np.full((200, 200, 3), 100, dtype=np.uint8)
        # Put high-contrast pattern in two regions
        frame[20:80, 20:80] = 255
        frame[100:160, 100:160] = 255

        locations = [(20, 80, 80, 20), (100, 160, 160, 100)]
        names = ["Registered User", "Unknown"]

        original_registered = frame[20:80, 20:80].copy()
        original_unknown = frame[100:160, 100:160].copy()

        self.engine.apply_privacy_blur(frame, locations, names)

        # Registered face region should remain untouched
        np.testing.assert_array_equal(frame[20:80, 20:80], original_registered)

        # Unknown bystander face region should be blurred (modified)
        self.assertFalse(np.array_equal(frame[100:160, 100:160], original_unknown))

    def test_verify_liveness_decision(self):
        """Verify liveness evaluation states: Pending -> Verified."""
        # Use image with natural contrast and facial features so texture analysis passes
        face_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(face_crop, (50, 50), 30, (200, 200, 200), -1)
        cv2.circle(face_crop, (40, 40), 8, (50, 50, 50), -1)
        cv2.circle(face_crop, (60, 40), 8, (50, 50, 50), -1)
        cv2.line(face_crop, (40, 65), (60, 65), (50, 50, 50), 3)

        open_eye = [(0, 10), (5, 4), (15, 4), (20, 10), (15, 10), (5, 10)]
        landmarks = {'left_eye': open_eye, 'right_eye': open_eye}

        # Step 1: Initial sighting with open eyes -> Awaiting blink confirmation
        res1 = self.engine.verify_liveness("SubjectA", face_crop, landmarks)
        self.assertFalse(res1.is_live)
        self.assertIn("BLINK NOW", res1.status_text)

        # Step 2: Subject blinks (2 frames closed)
        closed_eye = [(0, 10), (5, 9), (15, 9), (20, 10), (15, 11), (5, 11)]
        closed_landmarks = {'left_eye': closed_eye, 'right_eye': closed_eye}
        self.engine.verify_liveness("SubjectA", face_crop, closed_landmarks)
        self.engine.verify_liveness("SubjectA", face_crop, closed_landmarks)

        # Step 3: Eyes reopen -> Verified Live!
        res_live = self.engine.verify_liveness("SubjectA", face_crop, landmarks)
        self.assertTrue(res_live.is_live)
        self.assertIn("LIVE: VERIFIED", res_live.status_text)
        self.assertGreaterEqual(res_live.blinks, 1)

    def test_extract_6_landmarks(self):
        """Verify extraction and scaling of 6 canonical landmarks from dictionaries and arrays."""
        # Test 1: From face_recognition landmarks dictionary
        landmarks_dict = {
            'chin': [(10, 20)] * 17,
            'left_eyebrow': [(15, 25)] * 5,
            'right_eyebrow': [(45, 25)] * 5,
            'nose_bridge': [(30, 25), (30, 30), (30, 35), (30, 40)],
            'nose_tip': [(28, 42), (29, 43), (30, 44), (31, 43), (32, 42)],
            'left_eye': [(20, 25), (22, 23), (26, 23), (28, 25), (26, 27), (22, 27)],
            'right_eye': [(35, 25), (37, 23), (41, 23), (43, 25), (41, 27), (37, 27)],
            'top_lip': [(22, 50), (25, 48), (28, 47), (30, 48), (32, 47), (35, 48), (38, 50)] + [(30, 50)] * 5
        }
        pts = self.engine.extract_6_landmarks(landmarks_dict, scale=1.0)
        self.assertIsNotNone(pts)
        self.assertEqual(pts.shape, (6, 2))
        # Nose tip is last point of nose_bridge: (30, 40)
        np.testing.assert_array_equal(pts[0], [30.0, 40.0])

        # Test 2: Scaling by 4.0
        pts_scaled = self.engine.extract_6_landmarks(landmarks_dict, scale=4.0)
        np.testing.assert_array_equal(pts_scaled[0], [120.0, 160.0])

        # Test 3: Insufficient/None landmarks returns None
        self.assertIsNone(self.engine.extract_6_landmarks(None))
        self.assertIsNone(self.engine.extract_6_landmarks({'chin': []}))

    def test_estimate_head_pose_frontal_and_angles(self):
        """Verify 3D head pose estimation on projected synthetic facial geometry."""
        h, w = 480, 640
        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 1. Frontal face (0 deg rotation) placed at Z = 1000mm
        rvec_zero = np.zeros((3, 1), dtype=np.float64)
        tvec_zero = np.array([[0.0], [0.0], [1000.0]], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(MODEL_POINTS_6, rvec_zero, tvec_zero, camera_matrix, dist_coeffs)
        frontal_pts = img_pts.reshape(-1, 2)

        pose, ray = self.engine.estimate_head_pose(frontal_pts, frame_shape=(h, w), scale=1.0)
        self.assertIsNotNone(pose)
        self.assertIsNotNone(ray)
        yaw, pitch, roll = pose
        self.assertAlmostEqual(yaw, 0.0, delta=1.0)
        self.assertAlmostEqual(pitch, 0.0, delta=1.0)
        self.assertAlmostEqual(roll, 0.0, delta=1.0)

        # Nose ray should point forward: p1 is nose tip, p2 is forward projection
        p1, p2 = ray
        self.assertAlmostEqual(p1[0], center[0], delta=5)
        self.assertAlmostEqual(p1[1], center[1], delta=5)

        # 2. Turned head (+20 deg Yaw around Y axis)
        rvec_yaw = np.array([[0.0], [np.radians(20.0)], [0.0]], dtype=np.float64)
        img_pts_yaw, _ = cv2.projectPoints(MODEL_POINTS_6, rvec_yaw, tvec_zero, camera_matrix, dist_coeffs)
        yaw_pts = img_pts_yaw.reshape(-1, 2)

        pose_yaw, ray_yaw = self.engine.estimate_head_pose(yaw_pts, frame_shape=(h, w), scale=1.0)
        self.assertIsNotNone(pose_yaw)
        self.assertAlmostEqual(pose_yaw[0], 20.0, delta=1.5)

    def test_challenge_response_state_machine(self):
        """Verify dynamic challenge-response state machine transitions and timeouts."""
        tracker = ChallengeResponseTracker(
            challenge_timeout=5.0,
            initial_challenge=ChallengeType.TURN_LEFT
        )
        self.assertEqual(tracker.status, STATUS_PENDING)
        self.assertFalse(tracker.is_verified())

        # Test 1: Subject has not turned yet (yaw = 0.0) -> still PENDING
        status = tracker.update(yaw=0.0, pitch=0.0, roll=0.0, current_blinks=0, current_time=tracker.issued_at + 1.0)
        self.assertEqual(status, STATUS_PENDING)
        self.assertFalse(tracker.is_verified())

        # Test 2: Subject turns left (yaw = -16.0 < -12.0) -> PASSED
        status = tracker.update(yaw=-16.0, pitch=0.0, roll=0.0, current_blinks=0, current_time=tracker.issued_at + 2.0)
        self.assertEqual(status, STATUS_PASSED)
        self.assertTrue(tracker.is_verified())

        # Test 3: Reset with NOD_DOWN challenge
        tracker.reset(challenge=ChallengeType.NOD_DOWN, current_blinks=0, current_time=100.0)
        self.assertEqual(tracker.status, STATUS_PENDING)
        self.assertEqual(tracker.active_challenge, ChallengeType.NOD_DOWN)

        # Test 4: Pitch exceeding +10.0 (nod down) -> PASSED
        status = tracker.update(yaw=0.0, pitch=14.0, roll=0.0, current_blinks=0, current_time=101.0)
        self.assertEqual(status, STATUS_PASSED)
        self.assertTrue(tracker.is_verified())

        # Test 5: Timeout expiration
        tracker.reset(challenge=ChallengeType.TURN_RIGHT, current_blinks=0, current_time=200.0)
        status_timeout = tracker.update(yaw=0.0, pitch=0.0, roll=0.0, current_blinks=0, current_time=206.0)
        self.assertEqual(status_timeout, STATUS_TIMEOUT)
        self.assertFalse(tracker.is_verified())
        self.assertIn("TIMED OUT", tracker.get_prompt(current_time=206.0))

    def test_verify_liveness_with_challenge_response_enabled(self):
        """Verify unified verify_liveness workflow when interactive challenge mode is active."""
        config_with_challenge = {
            'anti_spoofing': {
                'enabled': True,
                'ear_threshold': 0.22,
                'texture_analysis': True,
                'head_pose': True,
                'challenge_response': True,
                'challenge_timeout': 4.0
            }
        }
        engine = AntiSpoofEngine(config_with_challenge)

        # High texture contrast face crop
        face_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(face_crop, (50, 50), 30, (200, 200, 200), -1)
        cv2.circle(face_crop, (40, 40), 8, (50, 50, 50), -1)
        cv2.circle(face_crop, (60, 40), 8, (50, 50, 50), -1)

        # Frontal landmarks
        h, w = 480, 640
        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        img_pts, _ = cv2.projectPoints(
            MODEL_POINTS_6,
            np.zeros((3, 1), dtype=np.float64),
            np.array([[0.0], [0.0], [1000.0]], dtype=np.float64),
            camera_matrix,
            dist_coeffs
        )
        frontal_6 = img_pts.reshape(-1, 2)

        # Step 1: Initial call -> Challenge PENDING, subject has not responded
        t0 = 1000.0
        res1 = engine.verify_liveness("UserX", face_crop, frontal_6, frame_shape=(h, w), current_time=t0)
        self.assertFalse(res1.is_live)
        self.assertIn("CHALLENGE", res1.status_text)
        self.assertEqual(res1.challenge_status, STATUS_PENDING)

        # Step 2: Subject completes challenge (turn head left)
        rvec_left = np.array([[0.0], [np.radians(-18.0)], [0.0]], dtype=np.float64)
        img_pts_left, _ = cv2.projectPoints(
            MODEL_POINTS_6,
            rvec_left,
            np.array([[0.0], [0.0], [1000.0]], dtype=np.float64),
            camera_matrix,
            dist_coeffs
        )
        left_6 = img_pts_left.reshape(-1, 2)

        res2 = engine.verify_liveness("UserX", face_crop, left_6, frame_shape=(h, w), current_time=t0 + 1.5)
        self.assertTrue(res2.is_live)
        self.assertIn("CHALLENGE PASSED", res2.status_text)
        self.assertEqual(res2.challenge_status, STATUS_PASSED)
        self.assertEqual(res2.hud_color, (0, 255, 0))

        # Step 3: Different subject times out
        res_timeout = engine.verify_liveness("UserY", face_crop, frontal_6, frame_shape=(h, w), current_time=t0)
        res_expired = engine.verify_liveness("UserY", face_crop, frontal_6, frame_shape=(h, w), current_time=t0 + 6.0)
        self.assertFalse(res_expired.is_live)
        self.assertEqual(res_expired.challenge_status, STATUS_TIMEOUT)
        self.assertIn("CHALLENGE TIMEOUT", res_expired.status_text)

if __name__ == '__main__':
    unittest.main()
