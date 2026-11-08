"""
Attendance Reporting & Analytics Engine
Module: src/reporting.py

Generates professional, presentation-ready attendance and recognition reports:
1. Multi-format export: Formatted CSV and Apple-inspired executive HTML reports.
2. Analytics & Summaries:
   - Total verifications and unique attendees.
   - Per-person first arrival, last sighting, and session frequency in IST.
   - Absentee tracking: Highlights registered personnel not spotted today.
   - Average confidence percentages.
3. Print-ready layout with responsive executive cards.

All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code, HUD readouts, and logs. Developer: Nilay Naha.
"""

import os
import csv
from typing import Dict, List, Any, Optional
from pathlib import Path
from src.utils import format_ist_timestamp, get_ist_now, get_ist_date_string


class AttendanceReporter:
    """
    Analytics and report generator for biometric facial recognition logs.
    """

    def __init__(self, db_manager):
        """
        Initialize the reporter with database reference.

        Args:
            db_manager: DatabaseManager instance.
        """
        self.db = db_manager

    def generate_daily_summary(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate statistical attendance breakdown for a given date.

        Args:
            target_date (str, optional): Date in 'YYYY-MM-DD' format.
                                         Defaults to current date in IST.

        Returns:
            dict: Summary metrics, attendee details, and absentee list.
        """
        date_key = target_date or get_ist_date_string()
        enrolled_persons = self.db.get_all_persons()
        logs_df = self.db.get_recognition_history(limit=5000)

        # Filter logs for the selected date
        matching_logs = []
        if not logs_df.empty:
            for _, row in logs_df.iterrows():
                ts_str = str(row.get('timestamp', ''))
                if ts_str.startswith(date_key):
                    matching_logs.append(row.to_dict())

        # Aggregate attendee statistics
        attendee_stats: Dict[str, Dict[str, Any]] = {}
        for log in matching_logs:
            name = str(log.get('name', 'Unknown'))
            if name == "Unknown":
                continue

            conf = float(log.get('confidence', 0.0)) * 100.0 if float(log.get('confidence', 0.0)) <= 1.0 else float(log.get('confidence', 0.0))
            ts = str(log.get('timestamp', ''))

            if name not in attendee_stats:
                attendee_stats[name] = {
                    'name': name,
                    'first_seen': ts,
                    'last_seen': ts,
                    'count': 1,
                    'confidences': [conf]
                }
            else:
                entry = attendee_stats[name]
                entry['count'] += 1
                entry['confidences'].append(conf)
                if ts < entry['first_seen']:
                    entry['first_seen'] = ts
                if ts > entry['last_seen']:
                    entry['last_seen'] = ts

        # Process attendees table
        attendees = []
        for name, data in attendee_stats.items():
            mean_conf = sum(data['confidences']) / float(len(data['confidences']))
            attendees.append({
                'name': name,
                'first_seen': data['first_seen'],
                'last_seen': data['last_seen'],
                'sightings': data['count'],
                'avg_confidence': round(mean_conf, 1)
            })

        attendees.sort(key=lambda x: x['name'])

        # Identify enrolled persons who were absent on this date
        enrolled_names = {p['name'] for p in enrolled_persons}
        present_names = set(attendee_stats.keys())
        absentees = sorted(list(enrolled_names - present_names))

        total_enrolled = len(enrolled_names)
        total_present = len(present_names)
        attendance_rate = (total_present / float(total_enrolled) * 100.0) if total_enrolled > 0 else 0.0

        all_confs = [c for a in attendees for c in attendee_stats[a['name']]['confidences']]
        overall_avg_conf = (sum(all_confs) / float(len(all_confs))) if all_confs else 0.0

        return {
            'date': date_key,
            'generated_at': format_ist_timestamp(),
            'total_enrolled': total_enrolled,
            'total_present': total_present,
            'total_absent': len(absentees),
            'attendance_rate': round(attendance_rate, 1),
            'overall_avg_confidence': round(overall_avg_conf, 1),
            'total_sightings': len(matching_logs),
            'attendees': attendees,
            'absentees': absentees
        }

    def export_csv(self, summary: Dict[str, Any], output_path: str = "data/attendance_summary.csv") -> str:
        """
        Export attendance summary to a structured CSV file.

        Args:
            summary (dict): Output from generate_daily_summary().
            output_path (str): Output destination path.

        Returns:
            str: Resolved absolute path to the generated CSV.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ATTENDANCE REPORT (IST)", summary['date']])
            writer.writerow(["Generated At", summary['generated_at']])
            writer.writerow(["Total Enrolled", summary['total_enrolled']])
            writer.writerow(["Total Present", summary['total_present']])
            writer.writerow(["Total Absent", summary['total_absent']])
            writer.writerow(["Attendance Rate", f"{summary['attendance_rate']}%"])
            writer.writerow(["Mean Confidence", f"{summary['overall_avg_confidence']}%"])
            writer.writerow([])
            writer.writerow(["NAME", "FIRST SEEN (IST)", "LAST SEEN (IST)", "SIGHTINGS", "AVG CONFIDENCE"])

            for att in summary['attendees']:
                writer.writerow([
                    att['name'],
                    att['first_seen'],
                    att['last_seen'],
                    att['sightings'],
                    f"{att['avg_confidence']}%"
                ])

            if summary['absentees']:
                writer.writerow([])
                writer.writerow(["ABSENT PERSONNEL"])
                for ab in summary['absentees']:
                    writer.writerow([ab, "ABSENT", "ABSENT", 0, "0.0%"])

        return str(out_file.resolve())

    def export_html(self, summary: Dict[str, Any], output_path: str = "data/attendance_report.html") -> str:
        """
        Export presentation-grade, Apple-inspired executive attendance report in HTML.

        Args:
            summary (dict): Output from generate_daily_summary().
            output_path (str): Output destination path.

        Returns:
            str: Resolved absolute path to the generated HTML file.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Format attendee table rows
        rows_html = ""
        for att in summary['attendees']:
            rows_html += f"""
            <tr>
                <td class="name-cell"><strong>{att['name']}</strong></td>
                <td>{att['first_seen']}</td>
                <td>{att['last_seen']}</td>
                <td><span class="badge badge-neutral">{att['sightings']}</span></td>
                <td><span class="badge badge-success">{att['avg_confidence']}%</span></td>
                <td><span class="status-present">PRESENT</span></td>
            </tr>
            """

        if not summary['attendees']:
            rows_html = '<tr><td colspan="6" class="text-center muted">No attendance records found for this date.</td></tr>'

        # Format absentee badges
        absentees_html = ""
        if summary['absentees']:
            for ab in summary['absentees']:
                absentees_html += f'<span class="badge badge-danger">{ab}</span> '
        else:
            absentees_html = '<span class="muted">All enrolled personnel were accounted for.</span>'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attendance Executive Summary - {summary['date']}</title>
    <style>
        :root {{
            --bg-primary: #121214;
            --bg-surface: #1c1c1e;
            --bg-card: #252528;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --text-primary: #f5f5f7;
            --text-secondary: #a1a1a6;
            --accent-green: #30d158;
            --accent-blue: #0a84ff;
            --accent-orange: #ff9f0a;
            --accent-red: #ff453a;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 40px 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        .header .meta {{
            font-size: 13px;
            color: var(--text-secondary);
            text-align: right;
        }}
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }}
        .card .title {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }}
        .card .value {{
            font-size: 32px;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: var(--text-primary);
        }}
        .card .value.green {{ color: var(--accent-green); }}
        .card .value.blue {{ color: var(--accent-blue); }}
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 14px;
            letter-spacing: -0.3px;
        }}
        .table-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 32px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 14px;
        }}
        th {{
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-subtle);
        }}
        td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-subtle);
        }}
        tr:last-child td {{ border-bottom: none; }}
        .name-cell {{ font-size: 15px; }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-success {{ background: rgba(48, 209, 88, 0.15); color: var(--accent-green); }}
        .badge-neutral {{ background: rgba(255, 255, 255, 0.08); color: var(--text-primary); }}
        .badge-danger {{ background: rgba(255, 69, 58, 0.15); color: var(--accent-red); margin-bottom: 6px; }}
        .status-present {{ color: var(--accent-green); font-weight: 600; font-size: 12px; letter-spacing: 0.5px; }}
        .muted {{ color: var(--text-secondary); }}
        .text-center {{ text-align: center; }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 40px;
        }}
        @media print {{
            body {{ background: white; color: black; padding: 0; }}
            .card, .table-card {{ border: 1px solid #ddd; background: white; color: black; box-shadow: none; }}
            .card .title, th {{ color: #555; }}
            .card .value {{ color: black; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Biometric Attendance Summary</h1>
                <div class="meta" style="text-align: left; margin-top: 4px;">Face Recognition System &bull; Verification Report</div>
            </div>
            <div class="meta">
                <div>Date: <strong>{summary['date']}</strong></div>
                <div>Generated: <strong>{summary['generated_at']}</strong></div>
            </div>
        </div>

        <div class="cards-grid">
            <div class="card">
                <div class="title">Personnel Present</div>
                <div class="value green">{summary['total_present']} <span style="font-size: 16px; color: var(--text-secondary);">/ {summary['total_enrolled']}</span></div>
            </div>
            <div class="card">
                <div class="title">Attendance Rate</div>
                <div class="value blue">{summary['attendance_rate']}%</div>
            </div>
            <div class="card">
                <div class="title">Total Sightings</div>
                <div class="value">{summary['total_sightings']}</div>
            </div>
            <div class="card">
                <div class="title">Mean Confidence</div>
                <div class="value">{summary['overall_avg_confidence']}%</div>
            </div>
        </div>

        <div class="section-title">Verified Personnel</div>
        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>Identity Name</th>
                        <th>First Seen (IST)</th>
                        <th>Last Seen (IST)</th>
                        <th>Sightings</th>
                        <th>Confidence</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>

        <div class="section-title">Unaccounted / Absent Personnel ({summary['total_absent']})</div>
        <div class="table-card" style="padding: 16px;">
            {absentees_html}
        </div>

        <div class="footer">
            Generated autonomously by Face Recognition System &bull; Indian Standard Time (IST) &bull; Developer: Nilay Naha
        </div>
    </div>
</body>
</html>
        """

        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(out_file.resolve())
