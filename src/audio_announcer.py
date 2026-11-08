"""
Offline Voice Announcement & Audio Feedback Engine
Module: src/audio_announcer.py

Provides multimodal audio feedback upon successful facial identification:
1. Greets recognized enrolled personnel (e.g. "Welcome, Nilay").
2. Operates entirely offline with zero cloud dependency.
3. Completely asynchronous via background worker thread and queue, ensuring
   zero latency, frame drops, or freezing on the video processing pipeline.
4. Per-person cooldown deduplication prevents repetitive audio loops.
5. Graceful multi-backend support: Win32 SAPI, pyttsx3, and silent fallback.

All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code, HUD readouts, and logs. Developer: Nilay Naha.
"""

import sys
import time
import queue
import threading
from typing import Dict, Optional, Any


class AudioAnnouncer:
    """
    Thread-safe asynchronous speech synthesizer for biometric verification feedback.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the voice announcement engine.

        Args:
            config (dict, optional): Full application configuration dictionary.
        """
        cfg = config.get('audio', {}) if config else {}
        self.enabled: bool = bool(cfg.get('enabled', False))
        self.cooldown_seconds: float = float(cfg.get('cooldown_seconds', 15.0))
        self.speech_rate: int = int(cfg.get('rate', 170))

        # Timestamp tracking for per-person announcement cooldowns
        self.last_announced: Dict[str, float] = {}

        # Asynchronous speech job queue
        self.speech_queue: queue.Queue = queue.Queue(maxsize=10)
        self.is_running: bool = True

        # Spawn background consumer worker thread
        self.worker_thread = threading.Thread(
            target=self._speech_worker_loop,
            name="AudioAnnouncerWorker",
            daemon=True
        )
        self.worker_thread.start()

    def announce_recognition(self, name: str, current_time: Optional[float] = None) -> bool:
        """
        Queue an announcement for a recognized person if cooldown window has elapsed.

        Args:
            name (str): Identified person name.
            current_time (float, optional): Custom timestamp for deterministic testing.

        Returns:
            bool: True if queued for announcement; False if suppressed by cooldown or disabled.
        """
        if not self.enabled or not name or name == "Unknown":
            return False

        now = current_time if current_time is not None else time.time()
        last_time = self.last_announced.get(name, 0.0)

        if (now - last_time) < self.cooldown_seconds:
            return False

        self.last_announced[name] = now
        phrase = f"Welcome, {name}"

        try:
            self.speech_queue.put_nowait(phrase)
            return True
        except queue.Full:
            return False

    def announce_custom(self, message: str) -> bool:
        """
        Queue an arbitrary high-priority text message for speech output.

        Args:
            message (str): Text message to speak aloud.

        Returns:
            bool: True if queued; False otherwise.
        """
        if not self.enabled or not message:
            return False

        try:
            self.speech_queue.put_nowait(message)
            return True
        except queue.Full:
            return False

    def _speech_worker_loop(self):
        """
        Dedicated background worker consuming and speaking queued messages.
        Ensures camera loops and GUI threads never block on audio I/O.
        """
        # Attempt to initialize Windows COM for SAPI if on Windows platform
        use_win32 = False
        if sys.platform.startswith("win"):
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                use_win32 = True
            except Exception:
                use_win32 = False

        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if use_win32:
                    import pythoncom
                    import win32com.client
                    pythoncom.CoInitialize()
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    speaker.Rate = 1
                    speaker.Speak(text)
                else:
                    # Alternative pyttsx3 backend
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty('rate', self.speech_rate)
                    engine.say(text)
                    engine.runAndWait()
            except Exception:
                # Silently catch audio driver / virtual audio errors on headless systems
                pass
            finally:
                self.speech_queue.task_done()

    def set_enabled(self, enabled: bool):
        """Toggle voice announcement state."""
        self.enabled = bool(enabled)

    def stop(self):
        """Signal background worker thread to terminate cleanly."""
        self.is_running = False
