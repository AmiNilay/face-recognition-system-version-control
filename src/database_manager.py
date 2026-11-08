"""
Database Management Module

This module manages all persistent relational data for the Face Recognition System:
1. Registered Person Directory (names, reference image paths, enrollment date in IST).
2. Recognition Event Logs (person ID, match confidence, timestamp in IST, snapshot path).
3. Unknown Face Logs (snapshots of unregistered persons detected by camera).
4. Exporting attendance history to standardized CSV files.
5. Providing resilient fallback data structures to prevent NumPy/Pandas ABI crashes.
"""

import sqlite3
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from src.utils import get_ist_now, format_ist_timestamp
from src.security import SecurityManager, GENESIS_HASH

class RecordRow(dict):
    """
    Dictionary representing a single query result row.
    Allows accessing fields by dictionary key (e.g. row['name']).
    """
    pass

class RecordDataFrame(list):
    """
    Lightweight, resilient fallback DataFrame compatible with pandas API:
    - Supports iteration via .iterrows() -> (index, row)
    - Supports indexed lookup via .iloc[idx]
    - Supports CSV persistence via .to_csv(filepath)
    This prevents application crashes if Pandas encounters C-extension ABI issues.
    """
    def __init__(self, records: List[Any], columns: List[str]):
        row_dicts = [RecordRow(zip(columns, r)) for r in records]
        super().__init__(row_dicts)
        self.columns = columns

    @property
    def iloc(self):
        """Allows positional index access similar to pandas df.iloc[0]."""
        return self

    def iterrows(self):
        """Yields (index, row) tuples matching pandas DataFrame convention."""
        for idx, row in enumerate(self):
            yield idx, row

    def to_csv(self, output_path: str, index: bool = False):
        """Exports the records to a comma-separated values (CSV) file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
            for row in self:
                writer.writerow([row.get(col, '') for col in self.columns])


class DatabaseManager:
    """
    Manages SQLite database connections, schema migrations, and log operations.
    All recorded timestamps are strictly stored in Indian Standard Time (IST).
    """

    def __init__(self, db_path: str = "data/database.db"):
        """
        Initialize the database manager with cryptographic security services.

        Args:
            db_path (str): Relative or absolute path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        # Ensure the parent directory (e.g. 'data/') exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.sec = SecurityManager()
        self.init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Helper to create and return a SQLite database connection."""
        return sqlite3.connect(str(self.db_path))

    def init_database(self):
        """
        Create necessary database tables if they do not already exist:
        - persons: Stores registered identity records.
        - recognition_logs: Records every verified face recognition event with SHA-256 chained hashes.
        - unknown_faces: Records sightings of unregistered individuals.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table 1: Registered persons
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    added_date TEXT NOT NULL,
                    image_path TEXT,
                    encoding_path TEXT
                )
            ''')

            # Table 2: Recognition event history with cryptographic hash chain (SEC-05)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_path TEXT,
                    prev_hash TEXT,
                    current_hash TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
                )
            ''')

            # Schema Migration: Add prev_hash and current_hash if table existed from earlier versions
            cursor.execute("PRAGMA table_info(recognition_logs)")
            existing_cols = [col[1] for col in cursor.fetchall()]
            if 'prev_hash' not in existing_cols:
                cursor.execute("ALTER TABLE recognition_logs ADD COLUMN prev_hash TEXT")
            if 'current_hash' not in existing_cols:
                cursor.execute("ALTER TABLE recognition_logs ADD COLUMN current_hash TEXT")

            # Table 3: Unidentified face detections
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unknown_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_path TEXT,
                    encoding_path TEXT
                )
            ''')
            conn.commit()

    def add_person(self, name: str, image_path: Optional[str] = None) -> int:
        """
        Register a new person in the database.
        If the person already exists, returns their existing ID.

        Args:
            name (str): Full name or identifier of the person.
            image_path (str, optional): Relative path to their reference photo.

        Returns:
            int: The person's database ID.
        """
        current_time_ist = format_ist_timestamp()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Use parameterized SQL queries to prevent SQL injection vulnerabilities
                cursor.execute('''
                    INSERT INTO persons (name, added_date, image_path)
                    VALUES (?, ?, ?)
                ''', (name, current_time_ist, image_path))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # If name already exists, query and return existing record ID
                cursor.execute('SELECT id FROM persons WHERE name = ?', (name,))
                row = cursor.fetchone()
                return row[0] if row else 0

    def log_recognition(self, person_name: str, confidence: float, 
                        image_path: Optional[str] = None):
        """
        Log a successful face recognition sighting with IST timestamp and
        SHA-256 chained cryptographic integrity hashing (SEC-05).

        Args:
            person_name (str): Name of the identified person.
            confidence (float): Match confidence score as a decimal (e.g. 0.95 for 95%).
            image_path (str, optional): Path to the saved video frame snapshot.
        """
        current_time_ist = format_ist_timestamp()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM persons WHERE name = ?', (person_name,))
            result = cursor.fetchone()

            if result:
                person_id = result[0]

                # Fetch hash of previous log entry to extend the cryptographic audit chain
                cursor.execute('SELECT current_hash FROM recognition_logs ORDER BY id DESC LIMIT 1')
                last_row = cursor.fetchone()
                prev_hash = last_row[0] if (last_row and last_row[0]) else GENESIS_HASH

                # Compute immutable SHA-256 block hash for this event
                conf_pct = float(confidence) * 100.0 if float(confidence) <= 1.0 else float(confidence)
                current_hash = self.sec.compute_record_hash(
                    prev_hash=prev_hash,
                    timestamp_ist=current_time_ist,
                    person_name=person_name,
                    confidence=conf_pct
                )

                cursor.execute('''
                    INSERT INTO recognition_logs (person_id, timestamp, confidence, image_path, prev_hash, current_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (person_id, current_time_ist, confidence, image_path, prev_hash, current_hash))
                conn.commit()

    def verify_audit_log_integrity(self) -> Tuple[bool, Optional[int]]:
        """
        Cryptographically verify the SHA-256 hash chain across all recorded attendance logs.

        Returns:
            tuple[bool, Optional[int]]: (True, None) if completely pristine and untampered.
                                        (False, corrupted_log_id) if tampering is detected.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT r.id, p.name, r.timestamp, r.confidence, r.prev_hash, r.current_hash
                FROM recognition_logs r
                JOIN persons p ON r.person_id = p.id
                ORDER BY r.id ASC
            ''')
            rows = cursor.fetchall()

            records = []
            id_map = []
            for row in rows:
                id_map.append(row[0])
                conf = float(row[3])
                conf_pct = conf * 100.0 if conf <= 1.0 else conf
                records.append({
                    'name': row[1],
                    'timestamp': row[2],
                    'confidence': conf_pct,
                    'prev_hash': row[4] or GENESIS_HASH,
                    'current_hash': row[5] or ''
                })

            is_valid, corrupted_idx = self.sec.verify_chain_integrity(records)
            if not is_valid and corrupted_idx is not None:
                return False, id_map[corrupted_idx]
            return True, None

    def log_unknown_face(self, image_path: str, encoding_path: Optional[str] = None):
        """
        Log an unknown/unrecognized face sighting for security auditing.

        Args:
            image_path (str): Filepath to the captured snapshot of the unknown face.
            encoding_path (str, optional): Path to saved embedding vector file if cached.
        """
        current_time_ist = format_ist_timestamp()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO unknown_faces (timestamp, image_path, encoding_path)
                VALUES (?, ?, ?)
            ''', (current_time_ist, image_path, encoding_path))
            conn.commit()

    def get_recognition_history(self, person_name: Optional[str] = None, 
                                limit: Optional[int] = 100):
        """
        Retrieve historical recognition events ordered from newest to oldest.

        Args:
            person_name (str, optional): If specified, filters records for that person only.
            limit (int, optional): Maximum number of log rows to return. None returns all rows.

        Returns:
            pandas.DataFrame or RecordDataFrame: Tabular log data supporting .iterrows().
        """
        conn = self._get_connection()
        try:
            sql_limit = -1 if limit is None else limit
            columns = ['name', 'timestamp', 'confidence', 'image_path']

            if person_name:
                query = '''
                    SELECT p.name, r.timestamp, r.confidence, r.image_path
                    FROM recognition_logs r
                    JOIN persons p ON r.person_id = p.id
                    WHERE p.name = ?
                    ORDER BY r.id DESC
                    LIMIT ?
                '''
                params = (person_name, sql_limit)
            else:
                query = '''
                    SELECT p.name, r.timestamp, r.confidence, r.image_path
                    FROM recognition_logs r
                    JOIN persons p ON r.person_id = p.id
                    ORDER BY r.id DESC
                    LIMIT ?
                '''
                params = (sql_limit,)

            # Attempt standard pandas query; gracefully fallback to native wrapper if ABI issues occur
            try:
                import pandas as pd
                return pd.read_sql_query(query, conn, params=params)
            except Exception:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return RecordDataFrame(rows, columns)
        finally:
            conn.close()

    def get_all_persons(self) -> List[Dict[str, Any]]:
        """
        Fetch all registered individuals in alphabetical order.

        Returns:
            list[dict]: List of dictionaries containing 'id', 'name', 'added_date', 'image_path'.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, added_date, image_path
                FROM persons
                ORDER BY name ASC
            ''')
            
            persons = []
            for row in cursor.fetchall():
                persons.append({
                    'id': row[0],
                    'name': row[1],
                    'added_date': row[2],
                    'image_path': row[3]
                })
            return persons

    def delete_person(self, name: str):
        """
        Remove a person and their associated recognition history from the database.

        Args:
            name (str): Person name to delete.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM persons WHERE name = ?', (name,))
            result = cursor.fetchone()

            if result:
                person_id = result[0]
                cursor.execute('DELETE FROM recognition_logs WHERE person_id = ?', (person_id,))
                cursor.execute('DELETE FROM persons WHERE id = ?', (person_id,))
                conn.commit()

    def export_to_csv(self, output_path: str = "data/recognition_log.csv") -> str:
        """
        Export all database recognition events to a standard CSV file.

        Args:
            output_path (str): Destination file path for the CSV output.

        Returns:
            str: Absolute or relative path to the saved CSV file.
        """
        history_data = self.get_recognition_history(limit=None)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        history_data.to_csv(output_path, index=False)
        return output_path
