"""
Main Application Launcher

Entry point for the Face Recognition System. Provides command-line options
to start the application in GUI or CLI mode.

Usage Examples:
    python main.py --gui       # Launch the modern dark-themed GUI dashboard
    python main.py             # Launch fast command-line / HUD recognition mode
    python main.py --video 1   # Use secondary USB camera device
"""

import sys
import os
from pathlib import Path
import argparse

# Ensure project root directory is present in Python's module search path
sys.path.append(str(Path(__file__).parent))

from src.face_recognition_system import FaceRecognitionSystem
from src.utils import setup_logging, check_requirements, format_ist_timestamp

def main():
    """Main execution orchestrator."""
    parser = argparse.ArgumentParser(
        description='Real-time Face Recognition System (Indian Standard Time - IST Edition)'
    )
    parser.add_argument('--gui', action='store_true', help='Launch Graphical User Interface (GUI)')
    parser.add_argument('--video', type=str, default='0', help='Video source: device index (0) or video file path')
    
    args = parser.parse_args()

    # Configure logging with IST timestamps
    logger = setup_logging()
    logger.info(f"Application starting at {format_ist_timestamp()}")

    # Verify that core libraries are importable
    if not check_requirements():
        print("[ERROR] Dependency verification failed. Please install dependencies from requirements.txt")
        return

    if args.gui:
        # Launch Tkinter Desktop Application
        print("[INFO] Starting Desktop GUI mode...")
        from gui.app import FaceRecognitionGUI
        app = FaceRecognitionGUI()
        app.run()
    else:
        # Launch lightweight CLI mode
        print("=" * 60)
        print("               FACE RECOGNITION SYSTEM")
        print(f"            Started: {format_ist_timestamp()}")
        print("=" * 60)

        system = FaceRecognitionSystem()

        # Attempt to load cached encodings first; otherwise compute from known_faces/
        if not system.load_encodings():
            if system.load_known_faces():
                system.save_encodings()

        if len(system.known_face_names) == 0:
            print("\n[ERROR] No enrolled faces found!")
            print("Please run: python add_faces.py to enroll faces first, or drop photos into known_faces/")
            return

        print(f"\n[OK] Loaded {len(system.known_face_names)} enrolled identities.")
        print("\nStarting video stream...")
        print("Keyboard Controls:")
        print("  - 'Q': Quit")
        print("  - 'S': Save screenshot")
        print("  - 'C': Enroll new face live\n")

        video_source = int(args.video) if args.video.isdigit() else args.video
        system.run_recognition(video_source)

if __name__ == "__main__":
    main()
