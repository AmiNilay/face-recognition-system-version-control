import sys
import os
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.face_recognition_system import FaceRecognitionSystem
from src.utils import setup_logging, check_requirements

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--gui', action='store_true', help='Run with GUI')
    parser.add_argument('--video', type=str, default='0', help='Video source (0 for webcam)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Check requirements
    if not check_requirements():
        print("Missing requirements. Please install required packages.")
        return
    
    if args.gui:
        # Run GUI version
        print("Starting GUI mode...")
        from gui.app import FaceRecognitionGUI
        app = FaceRecognitionGUI()
        app.run()
    else:
        # Run command line version
        print("="*60)
        print("         FACE RECOGNITION SYSTEM")
        print("="*60)
        
        system = FaceRecognitionSystem()
        
        # Load known faces
        if not system.load_encodings():
            if system.load_known_faces():
                system.save_encodings()
        
        if len(system.known_face_names) == 0:
            print("\n❌ No faces loaded!")
            print("Please run: python add_faces.py to add faces first")
            return
        
        print(f"\n✅ Loaded {len(system.known_face_names)} known faces")
        print("\nStarting face recognition...")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("  - Press 'c' to add new face from camera\n")
        
        video_source = int(args.video) if args.video.isdigit() else args.video
        system.run_recognition(video_source)

if __name__ == "__main__":
    main()