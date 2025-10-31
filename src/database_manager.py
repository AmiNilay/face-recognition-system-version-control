"""Database Management Module"""

import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional

class DatabaseManager:
    """Handle database operations for face recognition logs"""
    
    def __init__(self, db_path: str = "data/database.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                encoding_path TEXT
            )
        ''')
        
        # Create recognition_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                image_path TEXT,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        # Create unknown_faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS unknown_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                encoding_path TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_person(self, name: str, image_path: str = None) -> int:
        """Add new person to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO persons (name, image_path)
                VALUES (?, ?)
            ''', (name, image_path))
            conn.commit()
            person_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            # Person already exists
            cursor.execute('SELECT id FROM persons WHERE name = ?', (name,))
            person_id = cursor.fetchone()[0]
        finally:
            conn.close()
        
        return person_id
    
    def log_recognition(self, person_name: str, confidence: float, 
                       image_path: str = None):
        """Log face recognition event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get person ID
        cursor.execute('SELECT id FROM persons WHERE name = ?', (person_name,))
        result = cursor.fetchone()
        
        if result:
            person_id = result[0]
            cursor.execute('''
                INSERT INTO recognition_logs (person_id, confidence, image_path)
                VALUES (?, ?, ?)
            ''', (person_id, confidence, image_path))
            conn.commit()
        
        conn.close()
    
    def log_unknown_face(self, image_path: str, encoding_path: str = None):
        """Log unknown face detection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO unknown_faces (image_path, encoding_path)
            VALUES (?, ?)
        ''', (image_path, encoding_path))
        
        conn.commit()
        conn.close()
    
    def get_recognition_history(self, person_name: str = None, 
                               limit: int = 100) -> pd.DataFrame:
        """Get recognition history"""
        conn = sqlite3.connect(self.db_path)
        
        if person_name:
            query = '''
                SELECT p.name, r.timestamp, r.confidence, r.image_path
                FROM recognition_logs r
                JOIN persons p ON r.person_id = p.id
                WHERE p.name = ?
                ORDER BY r.timestamp DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(person_name, limit))
        else:
            query = '''
                SELECT p.name, r.timestamp, r.confidence, r.image_path
                FROM recognition_logs r
                JOIN persons p ON r.person_id = p.id
                ORDER BY r.timestamp DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(limit,))
        
        conn.close()
        return df
    
    def get_all_persons(self) -> List[Dict]:
        """Get all registered persons"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, added_date, image_path
            FROM persons
            ORDER BY name
        ''')
        
        persons = []
        for row in cursor.fetchall():
            persons.append({
                'id': row[0],
                'name': row[1],
                'added_date': row[2],
                'image_path': row[3]
            })
        
        conn.close()
        return persons
    
    def delete_person(self, name: str):
        """Delete person from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get person ID
        cursor.execute('SELECT id FROM persons WHERE name = ?', (name,))
        result = cursor.fetchone()
        
        if result:
            person_id = result[0]
            # Delete recognition logs
            cursor.execute('DELETE FROM recognition_logs WHERE person_id = ?', (person_id,))
            # Delete person
            cursor.execute('DELETE FROM persons WHERE id = ?', (person_id,))
            conn.commit()
        
        conn.close()
    
    def export_to_csv(self, output_path: str = "data/recognition_log.csv"):
        """Export recognition logs to CSV"""
        df = self.get_recognition_history(limit=None)
        df.to_csv(output_path, index=False)
        return output_path