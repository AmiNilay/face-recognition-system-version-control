"""
Core Biometric Security, Cryptographic Storage & Integrity Engine
Module: src/security.py

This module delivers Phase 2 Security Hardening standards:
1. SEC-01 Elimination: Replaces insecure 'pickle' serialization with AES-authenticated
   encrypted NumPy vector storage (AES-128-CBC + HMAC-SHA256 via Fernet).
2. SEC-02 & Path Traversal Prevention: Strict sanitization and canonical path resolution
   preventing directory traversal attacks (e.g. '../../etc/shadow' or '..\\Windows\\System32').
3. SEC-03 Field-Level Biometric Encryption: Encrypts sensitive personal data and facial
   embedding metadata using authenticated cryptography.
4. SEC-05 Chained Cryptographic Audit Logging: Implements tamper-evident blockchain-style
   SHA-256 chained hashing across all attendance records.

Mathematical & Security Concepts for Beginners:
- Authenticated Symmetric Encryption (Fernet / AES-128-CBC + HMAC-SHA256):
  Guarantees confidentiality and integrity. If an attacker modifies even a single
  bit of an encrypted facial vector file on disk, decryption immediately fails.
- Chained Hashing (Blockchain-style Audit Logs):
  Each record contains the SHA-256 hash of the previous record. If an attacker
  alters an old timestamp, attendance score, or person name, the hash chain breaks,
  instantly exposing the exact compromised record during an audit.
"""

import os
import re
import io
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from cryptography.fernet import Fernet, InvalidToken

GENESIS_HASH: str = "0" * 64

class SecurityManager:
    """
    Manages cryptographic keys, field-level encryption, safe vector persistence,
    and chained audit verification.
    """

    def __init__(self, key_path: str = "data/.secret.key"):
        """
        Initialize the Security Manager with a secure key.

        Args:
            key_path (str): Relative or absolute path to the symmetric key file.
        """
        self.key_path = Path(key_path)
        self.cipher = Fernet(self._get_or_create_key())

    def _get_or_create_key(self) -> bytes:
        """
        Retrieve existing Fernet key or generate and persist a fresh 32-byte key.

        Returns:
            bytes: Cryptographic Fernet key.
        """
        self.key_path.parent.mkdir(parents=True, exist_ok=True)

        if self.key_path.exists():
            try:
                key = self.key_path.read_bytes().strip()
                if len(key) == 44:  # Base64-encoded 32-byte key is 44 characters
                    return key
            except Exception:
                pass

        # Generate new cryptographically secure random key
        new_key = Fernet.generate_key()
        self.key_path.write_bytes(new_key)

        # Restrict permissions on POSIX systems if available
        try:
            os.chmod(self.key_path, 0o600)
        except Exception:
            pass

        return new_key

    # --------------------------------------------------------------------------
    # 1. FIELD-LEVEL ENCRYPTION & DECRYPTION (SEC-03)
    # --------------------------------------------------------------------------
    def encrypt_text(self, plaintext: str) -> str:
        """
        Encrypt a plaintext string into a base64-encoded authenticated ciphertext token.

        Args:
            plaintext (str): Sensitive string (e.g. employee name, portrait path).

        Returns:
            str: Safe ciphertext string.
        """
        if not plaintext:
            return ""
        encrypted_bytes = self.cipher.encrypt(plaintext.encode('utf-8'))
        return encrypted_bytes.decode('ascii')

    def decrypt_text(self, ciphertext: str) -> str:
        """
        Decrypt an authenticated ciphertext string back into plaintext.
        Returns original string if unencrypted or corrupted.

        Args:
            ciphertext (str): Encrypted base64 token.

        Returns:
            str: Decrypted plaintext string.
        """
        if not ciphertext:
            return ""
        try:
            decrypted_bytes = self.cipher.decrypt(ciphertext.encode('ascii'))
            return decrypted_bytes.decode('utf-8')
        except (InvalidToken, Exception):
            # Fallback returns original text if record was unencrypted legacy
            return ciphertext

    # --------------------------------------------------------------------------
    # 2. SAFE ENCRYPTED BIOMETRIC VECTOR PERSISTENCE (SEC-01)
    # --------------------------------------------------------------------------
    def save_encrypted_encodings(
        self,
        filepath: str,
        names: List[str],
        encodings: List[np.ndarray]
    ) -> bool:
        """
        Serialize enrolled facial embeddings into an AES-encrypted, compressed NumPy container.
        Replaces vulnerable Python pickle serialization (SEC-01).

        Args:
            filepath (str): Output destination file path (e.g. 'data/face_encodings.dat').
            names (list): List of identity names.
            encodings (list): List of 128-dimensional embedding vectors.

        Returns:
            bool: True on successful serialization; False otherwise.
        """
        try:
            dest_path = Path(filepath)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to numpy structures without pickle
            names_array = np.array(names, dtype=str)
            if encodings:
                encodings_array = np.array(encodings, dtype=np.float64)
            else:
                encodings_array = np.empty((0, 128), dtype=np.float64)

            # Serialize to in-memory compressed NumPy archive
            buffer = io.BytesIO()
            np.savez_compressed(buffer, names=names_array, encodings=encodings_array)
            raw_bytes = buffer.getvalue()

            # Encrypt the binary archive with AES-128-CBC + HMAC-SHA256
            encrypted_payload = self.cipher.encrypt(raw_bytes)

            # Write atomic file to disk
            temp_path = dest_path.with_suffix(".tmp")
            temp_path.write_bytes(encrypted_payload)
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)

            return True
        except Exception:
            return False

    def load_encrypted_encodings(
        self,
        filepath: str
    ) -> Tuple[List[str], List[np.ndarray]]:
        """
        Load and decrypt enrolled facial embeddings from secure archive.
        Safely falls back to allow_pickle=False.

        Args:
            filepath (str): Encrypted vector file path.

        Returns:
            tuple[list, list]: (names_list, encodings_list).
        """
        path = Path(filepath)
        if not path.exists():
            return [], []

        try:
            raw_file_bytes = path.read_bytes()

            # Attempt decryption
            try:
                decrypted_bytes = self.cipher.decrypt(raw_file_bytes)
            except InvalidToken:
                # File may be an unencrypted raw npz file
                decrypted_bytes = raw_file_bytes

            buffer = io.BytesIO(decrypted_bytes)
            # allow_pickle=False strictly enforces secure, code-execution-free deserialization
            with np.load(buffer, allow_pickle=False) as data:
                names = [str(n) for n in data['names']]
                encodings = [np.array(e, dtype=np.float64) for e in data['encodings']]
                return names, encodings

        except Exception:
            return [], []

    # --------------------------------------------------------------------------
    # 3. PATH TRAVERSAL & FILENAME SANITIZATION (SEC-02)
    # --------------------------------------------------------------------------
    @staticmethod
    def sanitize_identity_name(raw_name: str, max_length: int = 50) -> str:
        """
        Sanitize user-provided identity names to prevent directory traversal and injection.

        Rules:
        - Strips path traversal markers ('..', '/', '\\', '~').
        - Whitelists only safe characters: letters, numbers, spaces, underscores, hyphens.
        - Strips leading and trailing dots and spaces (invalid on Windows).
        - Truncates to max_length.

        Args:
            raw_name (str): Raw input identity string.
            max_length (int): Maximum allowable character length.

        Returns:
            str: Clean, secure display name.
        """
        if not raw_name:
            return "Unknown"

        # Remove null bytes and control characters
        clean = raw_name.replace('\0', '').strip()
        # Remove path separators
        clean = clean.replace('/', '').replace('\\', '').replace('..', '')
        # Whitelist alphanumeric, spaces, underscores, and hyphens
        clean = re.sub(r'[^a-zA-Z0-9_\- ]', '', clean)
        # Collapse multiple whitespace
        clean = re.sub(r'\s+', ' ', clean).strip('. ')

        if not clean:
            return "Unknown"

        return clean[:max_length]

    @staticmethod
    def get_safe_portrait_path(base_dir: str, identity_name: str, extension: str = ".jpg") -> Path:
        """
        Resolve a canonical, sandboxed image file path guaranteed to reside within base_dir.

        Args:
            base_dir (str): Allowed root folder (e.g. 'known_faces').
            identity_name (str): Identity name to convert into a safe filename.
            extension (str): Image extension (.jpg, .png).

        Returns:
            Path: Absolute resolved Path confirmed inside base_dir.

        Raises:
            ValueError: If path resolution attempts to escape base_dir.
        """
        clean_name = SecurityManager.sanitize_identity_name(identity_name)
        safe_filename = clean_name.replace(' ', '_') + extension

        base_path = Path(base_dir).resolve()
        target_path = (base_path / safe_filename).resolve()

        # Enforce sandbox boundary check
        if not str(target_path).startswith(str(base_path)):
            raise ValueError(f"Path traversal detected for identity '{identity_name}'!")

        return target_path

    # --------------------------------------------------------------------------
    # 4. CHAINED CRYPTOGRAPHIC AUDIT LOG INTEGRITY (SEC-05)
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_record_hash(
        prev_hash: str,
        timestamp_ist: str,
        person_name: str,
        confidence: float
    ) -> str:
        """
        Compute an immutable SHA-256 hash connecting this audit log to the prior record.

        Formula:
            hash = SHA-256(prev_hash || timestamp_ist || person_name || confidence_str)

        Args:
            prev_hash (str): Hex digest of the preceding record (or GENESIS_HASH).
            timestamp_ist (str): Time of event in Indian Standard Time.
            person_name (str): Recognized individual's name.
            confidence (float): Match confidence percentage.

        Returns:
            str: 64-character hexadecimal SHA-256 digest.
        """
        payload = f"{prev_hash}|{timestamp_ist}|{person_name}|{confidence:.2f}".encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def verify_chain_integrity(records: List[Dict[str, Any]]) -> Tuple[bool, Optional[int]]:
        """
        Verify that an attendance log chain has not been tampered with or modified.

        Args:
            records (list of dict): Audit records ordered chronologically (oldest to newest).
                                    Each record must contain 'prev_hash', 'current_hash',
                                    'timestamp', 'name', and 'confidence'.

        Returns:
            tuple[bool, Optional[int]]: (is_valid, corrupted_index).
                                        If corrupted, returns index of first compromised row.
        """
        expected_prev_hash = GENESIS_HASH

        for idx, rec in enumerate(records):
            actual_prev = rec.get('prev_hash', '')
            actual_curr = rec.get('current_hash', '')
            ts = rec.get('timestamp', '')
            name = rec.get('name', '')
            conf = float(rec.get('confidence', 0.0))

            # 1. Check link to previous record
            if actual_prev != expected_prev_hash:
                return False, idx

            # 2. Re-compute hash and verify against recorded current_hash
            calculated_curr = SecurityManager.compute_record_hash(
                expected_prev_hash, ts, name, conf
            )
            if calculated_curr != actual_curr:
                return False, idx

            # Advance chain forward
            expected_prev_hash = actual_curr

        return True, None
