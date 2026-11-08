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
from src.anti_spoofing import AntiSpoofEngine, BlinkTracker, LivenessResult

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

if __name__ == '__main__':
    unittest.main()
