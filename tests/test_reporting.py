"""
Unit Tests for AttendanceReporter Module (tests/test_reporting.py)

Tests daily attendance analytics, metrics aggregation, absentee detection,
and multi-format CSV / Apple-style HTML report export.
All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code, HUD readouts, and logs. Developer: Nilay Naha.
"""

import unittest
import tempfile
import os
import pandas as pd
from unittest.mock import MagicMock
from src.reporting import AttendanceReporter


class TestAttendanceReporter(unittest.TestCase):
    """Test suite for AttendanceReporter analytics and export engine."""

    def setUp(self):
        """Set up mock database and reporter instance."""
        self.mock_db = MagicMock()
        # Enrolled personnel: Alice and Bob
        self.mock_db.get_all_persons.return_value = [
            {'id': 1, 'name': 'Alice', 'added_date': '2026-11-08 10:00:00 IST'},
            {'id': 2, 'name': 'Bob', 'added_date': '2026-11-08 10:00:00 IST'}
        ]
        # Recognition history: Alice seen twice on 2026-11-08; Bob not seen
        self.mock_db.get_recognition_history.return_value = pd.DataFrame([
            {
                'name': 'Alice',
                'confidence': 0.95,
                'timestamp': '2026-11-08 10:15:00 IST'
            },
            {
                'name': 'Alice',
                'confidence': 0.91,
                'timestamp': '2026-11-08 14:30:00 IST'
            }
        ])
        self.reporter = AttendanceReporter(self.mock_db)

    def test_generate_daily_summary(self):
        """Verify daily attendance statistical summary calculation."""
        summary = self.reporter.generate_daily_summary("2026-11-08")

        self.assertEqual(summary['date'], "2026-11-08")
        self.assertEqual(summary['total_enrolled'], 2)
        self.assertEqual(summary['total_present'], 1)
        self.assertEqual(summary['total_absent'], 1)
        self.assertEqual(summary['attendance_rate'], 50.0)
        self.assertAlmostEqual(summary['overall_avg_confidence'], 93.0, delta=0.5)

        # Check attendees table
        self.assertEqual(len(summary['attendees']), 1)
        att = summary['attendees'][0]
        self.assertEqual(att['name'], 'Alice')
        self.assertEqual(att['first_seen'], '2026-11-08 10:15:00 IST')
        self.assertEqual(att['last_seen'], '2026-11-08 14:30:00 IST')
        self.assertEqual(att['sightings'], 2)

        # Check absentees
        self.assertIn('Bob', summary['absentees'])

    def test_export_csv(self):
        """Verify CSV export produces well-formatted spreadsheet."""
        summary = self.reporter.generate_daily_summary("2026-11-08")

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "test_report.csv")
            saved_path = self.reporter.export_csv(summary, output_path=csv_path)

            self.assertTrue(os.path.exists(saved_path))
            with open(saved_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn("ATTENDANCE REPORT (IST)", content)
            self.assertIn("Alice", content)
            self.assertIn("Bob", content)
            self.assertIn("ABSENT", content)

    def test_export_html(self):
        """Verify HTML export produces presentation-grade executive report."""
        summary = self.reporter.generate_daily_summary("2026-11-08")

        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = os.path.join(tmp_dir, "test_report.html")
            saved_path = self.reporter.export_html(summary, output_path=html_path)

            self.assertTrue(os.path.exists(saved_path))
            with open(saved_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn("Biometric Attendance Summary", content)
            self.assertIn("Alice", content)
            self.assertIn("Bob", content)
            self.assertIn("PRESENT", content)
            self.assertIn("Attendance Rate", content)
            self.assertIn("50.0%", content)


if __name__ == "__main__":
    unittest.main()
