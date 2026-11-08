"""
Security Webhook Dispatcher Engine
Module: src/webhook_dispatcher.py

Provides asynchronous outbound HTTP webhook notifications for security events:
1. Unknown / Unregistered face alerts with Base64 image snapshot attachments.
2. Anti-spoofing presentation attack detection warnings.
3. Verified VIP / personnel arrival notifications.
4. Camera stream disconnection / reconnection status updates.

All dispatches occur in a background queue worker thread to guarantee zero camera latency.
All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code and logs. Developer: Nilay Naha.
"""

import json
import time
import queue
import base64
import threading
import urllib.request
import urllib.error
from typing import List, Dict, Optional, Any
from pathlib import Path

import cv2
import numpy as np

from src.utils import format_ist_timestamp


class SecurityEventType:
    """Standardized security event types."""
    UNKNOWN_FACE = "UNKNOWN_FACE_DETECTED"
    SPOOF_ATTEMPT = "SPOOF_ATTEMPT_DETECTED"
    ATTENDANCE_VERIFIED = "ATTENDANCE_VERIFIED"
    STREAM_DISCONNECTED = "STREAM_DISCONNECTED"
    STREAM_RECONNECTED = "STREAM_RECONNECTED"


class WebhookDispatcher:
    """
    Non-blocking outbound webhook client with rate-limiting cooldown and background delivery queue.
    """

    def __init__(
        self,
        endpoints: Optional[List[str]] = None,
        enabled: bool = False,
        cooldown_seconds: float = 30.0,
        include_snapshot: bool = True,
        camera_id: str = "Primary_Camera"
    ):
        """
        Initialize the webhook dispatcher.

        Args:
            endpoints (list, optional): List of destination HTTP/HTTPS URLs.
            enabled (bool): Master toggle for webhook dispatching.
            cooldown_seconds (float): Window during which identical events are deduplicated.
            include_snapshot (bool): Attach base64 JPEG thumbnail to payload.
            camera_id (str): Camera or RTSP stream identifier.
        """
        self.endpoints = list(endpoints) if endpoints else []
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.include_snapshot = include_snapshot
        self.camera_id = camera_id

        self._queue: queue.Queue = queue.Queue(maxsize=100)
        self._last_event_time: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._is_running = True

        # Dispatch statistics
        self.total_dispatched = 0
        self.total_failed = 0
        self.total_suppressed = 0

        # Start worker thread
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def add_endpoint(self, url: str) -> bool:
        """Add a target webhook URL."""
        with self._lock:
            if url and url not in self.endpoints:
                self.endpoints.append(url)
                return True
        return False

    def remove_endpoint(self, url: str) -> bool:
        """Remove a target webhook URL."""
        with self._lock:
            if url in self.endpoints:
                self.endpoints.remove(url)
                return True
        return False

    def dispatch(
        self,
        event_type: str,
        details: str,
        subject_name: Optional[str] = None,
        confidence: Optional[float] = None,
        frame_bgr: Optional[np.ndarray] = None
    ) -> bool:
        """
        Enqueue an event for asynchronous dispatch to all registered webhooks.

        Args:
            event_type (str): Type from SecurityEventType.
            details (str): Human-readable event description.
            subject_name (str, optional): Identity name (e.g. 'Unknown' or person name).
            confidence (float, optional): Match confidence percentage.
            frame_bgr (np.ndarray, optional): Snapshot image array.

        Returns:
            bool: True if queued; False if disabled, in cooldown, or queue full.
        """
        if not self.enabled or not self.endpoints:
            return False

        # Cooldown deduplication key: event + subject
        cooldown_key = f"{event_type}:{subject_name or 'generic'}"
        now = time.time()

        with self._lock:
            last_time = self._last_event_time.get(cooldown_key, 0.0)
            if (now - last_time) < self.cooldown_seconds:
                self.total_suppressed += 1
                return False
            self._last_event_time[cooldown_key] = now

        # Encode thumbnail snapshot if requested
        snapshot_b64 = None
        if self.include_snapshot and frame_bgr is not None and frame_bgr.size > 0:
            try:
                # Downscale snapshot to 320px width for fast network transfer
                h, w = frame_bgr.shape[:2]
                thumb_w = min(320, w)
                thumb_h = int(h * (thumb_w / float(w)))
                thumb = cv2.resize(frame_bgr, (thumb_w, thumb_h))
                _, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                snapshot_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            except Exception:
                pass

        payload = {
            "event": event_type,
            "timestamp": format_ist_timestamp(),
            "camera_id": self.camera_id,
            "subject_name": subject_name or "Unknown",
            "confidence": round(float(confidence), 1) if confidence is not None else None,
            "details": details,
            "snapshot_base64": snapshot_b64
        }

        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            self.total_failed += 1
            return False

    def _process_queue(self):
        """Worker thread retrieving payloads and sending HTTP POST requests."""
        while self._is_running:
            try:
                payload = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with self._lock:
                target_urls = list(self.endpoints)

            json_bytes = json.dumps(payload).encode("utf-8")

            for url in target_urls:
                try:
                    req = urllib.request.Request(
                        url,
                        data=json_bytes,
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "Biometric-Face-Recognition-Edge/2.1.0"
                        },
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        if resp.status in (200, 201, 202, 204):
                            self.total_dispatched += 1
                        else:
                            self.total_failed += 1
                except Exception:
                    self.total_failed += 1

            self._queue.task_done()

    def send_test_ping(self, endpoint_url: str) -> Dict[str, Any]:
        """
        Send a synchronous test ping to verify an endpoint.

        Returns:
            dict: { "success": bool, "status_code": int, "message": str }
        """
        test_payload = {
            "event": "WEBHOOK_VERIFICATION_TEST",
            "timestamp": format_ist_timestamp(),
            "camera_id": self.camera_id,
            "details": "Biometric Face Recognition System test ping.",
            "status": "online"
        }
        json_bytes = json.dumps(test_payload).encode("utf-8")

        try:
            req = urllib.request.Request(
                endpoint_url,
                data=json_bytes,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return {
                    "success": True,
                    "status_code": resp.status,
                    "message": f"Webhook verified successfully (HTTP {resp.status})"
                }
        except urllib.error.HTTPError as http_err:
            return {
                "success": False,
                "status_code": http_err.code,
                "message": f"HTTP Error {http_err.code}: {http_err.reason}"
            }
        except Exception as err:
            return {
                "success": False,
                "status_code": 0,
                "message": f"Connection failed: {str(err)}"
            }

    def stop(self):
        """Terminate background worker thread."""
        self._is_running = False
