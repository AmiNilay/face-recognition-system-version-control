"""
Unit Tests for AudioAnnouncer Module (tests/test_audio.py)

Tests asynchronous voice announcement queuing, per-person cooldown deduplication,
custom announcement dispatch, and toggle lifecycle.
All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code, HUD readouts, and logs. Developer: Nilay Naha.
"""

import unittest
import time
from src.audio_announcer import AudioAnnouncer


class TestAudioAnnouncer(unittest.TestCase):
    """Test suite for AudioAnnouncer voice synthesizer engine."""

    def setUp(self):
        """Set up announcer test instance."""
        self.config = {
            'audio': {
                'enabled': True,
                'cooldown_seconds': 5.0,
                'rate': 175
            }
        }
        self.announcer = AudioAnnouncer(self.config)

    def tearDown(self):
        """Cleanly stop worker thread."""
        self.announcer.stop()

    def test_initialization(self):
        """Verify announcer initializes with configured parameters."""
        self.assertTrue(self.announcer.enabled)
        self.assertEqual(self.announcer.cooldown_seconds, 5.0)
        self.assertEqual(self.announcer.speech_rate, 175)

    def test_announce_recognition_and_cooldown(self):
        """Verify announcement queuing on first sighting and suppression within cooldown."""
        t0 = 1000.0

        # Sighting 1: First time seen -> Queued successfully
        queued_first = self.announcer.announce_recognition("Alice", current_time=t0)
        self.assertTrue(queued_first)
        self.assertIn("Alice", self.announcer.last_announced)

        # Sighting 2: 2 seconds later (< 5s cooldown) -> Suppressed
        queued_second = self.announcer.announce_recognition("Alice", current_time=t0 + 2.0)
        self.assertFalse(queued_second)

        # Sighting 3: Different person seen -> Queued independently
        queued_bob = self.announcer.announce_recognition("Bob", current_time=t0 + 2.5)
        self.assertTrue(queued_bob)

        # Sighting 4: 6 seconds later (> 5s cooldown) -> Queued successfully
        queued_third = self.announcer.announce_recognition("Alice", current_time=t0 + 6.0)
        self.assertTrue(queued_third)

    def test_ignore_unknown_or_empty(self):
        """Verify announcer ignores Unknown or empty identity names."""
        self.assertFalse(self.announcer.announce_recognition("Unknown"))
        self.assertFalse(self.announcer.announce_recognition(""))

    def test_announce_custom_message(self):
        """Verify arbitrary text announcements can be queued."""
        self.assertTrue(self.announcer.announce_custom("System ready"))

    def test_toggle_enabled_state(self):
        """Verify disabling announcer suppresses all speech queuing."""
        self.announcer.set_enabled(False)
        self.assertFalse(self.announcer.enabled)
        self.assertFalse(self.announcer.announce_recognition("Charlie"))
        self.assertFalse(self.announcer.announce_custom("Test message"))


if __name__ == "__main__":
    unittest.main()
