"""
Unit Tests for Security Engine (tests/test_security.py)

Tests encryption, safe vector storage, path traversal protection,
and SHA-256 chained audit integrity verification.
All timestamps follow Indian Standard Time (IST).
Zero emojis.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.security import SecurityManager, GENESIS_HASH

class TestSecurityManager(unittest.TestCase):
    """Test suite for Phase 2 cryptographic security hardening."""

    def setUp(self):
        """Create temporary key path and SecurityManager instance."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.key_path = os.path.join(self.temp_dir.name, "test.key")
        self.sec = SecurityManager(key_path=self.key_path)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_text_encryption_decryption(self):
        """Verify field-level encryption and authenticated decryption."""
        original = "Nilay Naha - Executive Officer"
        encrypted = self.sec.encrypt_text(original)
        self.assertNotEqual(original, encrypted)
        self.assertIsInstance(encrypted, str)

        decrypted = self.sec.decrypt_text(encrypted)
        self.assertEqual(original, decrypted)

    def test_safe_encrypted_vector_storage(self):
        """Verify vector storage without pickle serialization (SEC-01)."""
        names = ["Alice Wonder", "Bob Builder"]
        encodings = [
            np.random.rand(128).astype(np.float64),
            np.random.rand(128).astype(np.float64)
        ]
        save_path = os.path.join(self.temp_dir.name, "encodings.dat")

        success = self.sec.save_encrypted_encodings(save_path, names, encodings)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(save_path))

        # Ensure file content is encrypted on disk (does not contain plaintext name)
        raw_bytes = Path(save_path).read_bytes()
        self.assertNotIn(b"Alice Wonder", raw_bytes)

        loaded_names, loaded_encodings = self.sec.load_encrypted_encodings(save_path)
        self.assertEqual(names, loaded_names)
        self.assertEqual(len(encodings), len(loaded_encodings))
        np.testing.assert_allclose(encodings[0], loaded_encodings[0], rtol=1e-5)
        np.testing.assert_allclose(encodings[1], loaded_encodings[1], rtol=1e-5)

    def test_path_traversal_sanitization(self):
        """Verify path traversal prevention on malicious identity names."""
        malicious_inputs = [
            ("../../etc/passwd", "etcpasswd"),
            ("..\\..\\Windows\\System32\\cmd.exe", "WindowsSystem32cmdexe"),
            ("Valid Name / Traversal", "Valid Name Traversal"),
            ("   ...LeadingDots...   ", "LeadingDots"),
            ("User\x00NullByte", "UserNullByte")
        ]
        for raw, expected in malicious_inputs:
            sanitized = self.sec.sanitize_identity_name(raw)
            self.assertEqual(sanitized, expected)

    def test_get_safe_portrait_path(self):
        """Verify canonical portrait path is confined to base directory."""
        base = os.path.join(self.temp_dir.name, "portraits")
        os.makedirs(base, exist_ok=True)

        safe_path = self.sec.get_safe_portrait_path(base, "Nilay Naha")
        self.assertTrue(str(safe_path).startswith(str(Path(base).resolve())))
        self.assertEqual(safe_path.name, "Nilay_Naha.jpg")

    def test_chained_hash_audit_log_integrity(self):
        """Verify SHA-256 chained hash integrity verification and tamper detection."""
        # Build valid 3-record chain
        chain = []
        prev = GENESIS_HASH

        entries = [
            ("2026-11-08 09:00:00 IST", "Nilay Naha", 95.5),
            ("2026-11-08 09:05:00 IST", "Alice Wonder", 88.0),
            ("2026-11-08 09:10:00 IST", "Bob Builder", 92.3)
        ]

        for ts, name, conf in entries:
            curr = self.sec.compute_record_hash(prev, ts, name, conf)
            chain.append({
                'prev_hash': prev,
                'current_hash': curr,
                'timestamp': ts,
                'name': name,
                'confidence': conf
            })
            prev = curr

        # Verify pristine chain passes
        is_valid, corrupted_idx = self.sec.verify_chain_integrity(chain)
        self.assertTrue(is_valid)
        self.assertIsNone(corrupted_idx)

        # Tamper with record 1 (change confidence score)
        chain[1]['confidence'] = 99.9
        is_valid_tampered, tampered_idx = self.sec.verify_chain_integrity(chain)
        self.assertFalse(is_valid_tampered)
        self.assertEqual(tampered_idx, 1)

    def test_database_chained_audit_logging_and_tamper_detection(self):
        """Verify DatabaseManager logs with cryptographic hash chain and detects tampering (SEC-05)."""
        from src.database_manager import DatabaseManager
        import sqlite3

        db_file = os.path.join(self.temp_dir.name, "audit_test.db")
        db = DatabaseManager(db_path=db_file)
        db.add_person("Nilay Naha")

        # Log two recognition events
        db.log_recognition("Nilay Naha", 0.95)
        db.log_recognition("Nilay Naha", 0.92)

        is_valid, corrupted_id = db.verify_audit_log_integrity()
        self.assertTrue(is_valid)
        self.assertIsNone(corrupted_id)

        # Tamper directly with the SQLite database row
        with sqlite3.connect(db_file) as conn:
            conn.execute("UPDATE recognition_logs SET confidence = 0.50 WHERE id = 1")
            conn.commit()

        # Integrity verification must immediately detect the compromise!
        is_valid_after, corrupted_id_after = db.verify_audit_log_integrity()
        self.assertFalse(is_valid_after)
        self.assertEqual(corrupted_id_after, 1)

if __name__ == '__main__':
    unittest.main()
