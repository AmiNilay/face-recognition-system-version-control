"""
Unit Tests for GUI Module (tests/test_gui.py)

Tests GUI state lifecycle, thread-safe buffers, and camera lifecycle.
All timestamps follow Indian Standard Time (IST).
Zero emojis.
"""

import unittest
import numpy as np
import threading
from unittest.mock import MagicMock, patch
from gui.app import FaceRecognitionGUI

class TestFaceRecognitionGUI(unittest.TestCase):
    """Test suite for desktop GUI module."""

    @patch("gui.app.FaceRecognitionSystem")
    @patch("gui.app.DatabaseManager")
    def setUp(self, mock_db, mock_system):
        """Set up GUI instance with mocked heavyweight dependencies."""
        self.mock_sys_instance = MagicMock()
        self.mock_sys_instance.load_encodings.return_value = False
        self.mock_sys_instance.load_known_faces.return_value = False
        self.mock_sys_instance.known_face_names = []
        mock_system.return_value = self.mock_sys_instance

        self.mock_db_instance = MagicMock()
        mock_db.return_value = self.mock_db_instance

        self.app = FaceRecognitionGUI()
        # Withdraw window to prevent UI popup during testing
        self.app.root.withdraw()

    def tearDown(self):
        """Cleanly destroy Tkinter root."""
        try:
            self.app.stop_camera()
            self.app.root.destroy()
        except Exception:
            pass

    def test_gui_initial_state(self):
        """Verify initial UI state has camera stopped and controls enabled/disabled properly."""
        self.assertFalse(self.app.is_running)
        self.assertEqual(str(self.app.start_btn['state']), 'normal')
        self.assertEqual(str(self.app.stop_btn['state']), 'disabled')
        self.assertEqual(str(self.app.capture_btn['state']), 'disabled')

    def test_thread_safe_buffer_and_telemetry(self):
        """Verify thread-safe frame locking and telemetry dispatch."""
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_telemetry = {
            'names': ['Test User'],
            'locations': [(10, 90, 90, 10)],
            'confidences': [85.5]
        }

        with self.app.frame_lock:
            self.app.latest_display_frame = test_frame
            self.app.latest_telemetry = test_telemetry

        # Update telemetry on main thread
        self.app.update_detection_info()
        text_content = self.app.info_text.get("1.0", "end")
        self.assertIn("Test User", text_content)
        self.assertIn("85.5%", text_content)

    def test_camera_error_handler_resets_ui(self):
        """Verify _handle_camera_error stops camera and restores controls without crashing."""
        self.app.is_running = True
        self.app.start_btn.config(state="disabled")
        self.app.stop_btn.config(state="normal")

        with patch("tkinter.messagebox.showerror") as mock_errbox:
            self.app._handle_camera_error("Test device error")
            mock_errbox.assert_called_once()

        self.assertFalse(self.app.is_running)
        self.assertEqual(str(self.app.start_btn['state']), 'normal')
        self.assertEqual(str(self.app.stop_btn['state']), 'disabled')

    def test_anti_spoof_extension_toggles(self):
        """Verify 3D head pose and challenge response menu toggles."""
        mock_as = MagicMock()
        mock_as.head_pose_enabled = True
        mock_as.challenge_response_enabled = False
        self.app.system.anti_spoof = mock_as

        # Toggle head pose off
        self.app.head_pose_var.set(False)
        self.app.toggle_head_pose()
        self.assertFalse(mock_as.head_pose_enabled)

        # Toggle challenge response on
        self.app.challenge_var.set(True)
        self.app.toggle_challenge_response()
        self.assertTrue(mock_as.challenge_response_enabled)

    def test_telemetry_with_pose_and_challenge(self):
        """Verify detection telemetry displays pose angles and challenge tasks."""
        test_telemetry = {
            'names': ['Alice'],
            'locations': [(10, 90, 90, 10)],
            'confidences': [92.0],
            'liveness': ['[CHALLENGE] TURN HEAD LEFT (4.1s)'],
            'poses': [(-15.2, 3.4, 0.5)],
            'challenges': ['TURN HEAD LEFT (4.1s)']
        }
        self.app._update_detection_telemetry(test_telemetry)
        text_content = self.app.info_text.get("1.0", "end")
        self.assertIn("Alice", text_content)
        self.assertIn("Pose: Y:-15 P:+3 R:+0", text_content)
        self.assertIn("Task: TURN HEAD LEFT (4.1s)", text_content)

if __name__ == '__main__':
    unittest.main()
