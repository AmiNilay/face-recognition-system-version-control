"""
Apple Face ID Multi-Angle Guided Enrollment CLI
Module: add_faces_wizard.py

Interactive wizard guiding the user through 5 canonical head poses
(Frontal, Left, Right, Up, Down) to synthesize an L2-normalized 128-d
composite biometric embedding for robust real-world recognition.

Usage:
    python add_faces_wizard.py
    python add_faces_wizard.py --name "Nilay Naha" --video 0

All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code, HUD readouts, and logs. Developer: Nilay Naha.
"""

import sys
import os
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in module search path
sys.path.append(str(Path(__file__).parent))

from src.face_recognition_system import FaceRecognitionSystem
from src.database_manager import DatabaseManager
from src.audio_announcer import AudioAnnouncer
from src.composite_enrollment import (
    CompositeEnrollmentSession,
    render_apple_face_id_hud,
    ENROLLMENT_STEPS
)
from src.security import SecurityManager
from src.utils import format_ist_timestamp, load_config


def run_face_id_wizard(name: str = "", video_source: int = 0):
    """
    Launch interactive Apple Face ID multi-angle enrollment loop.
    """
    print("=" * 68)
    print("       APPLE FACE ID MULTI-ANGLE COMPOSITE ENROLLMENT WIZARD")
    print(f"       Started: {format_ist_timestamp()}")
    print("=" * 68)

    # 1. Identity Name Acquisition & Sanitization
    while not name or name == "Unknown":
        print("\nEnter full name for new identity: ", end="", flush=True)
        raw_name = input().strip()
        name = SecurityManager.sanitize_identity_name(raw_name)
        if not name or name == "Unknown":
            print("[ERROR] Invalid name. Use alphanumeric characters and spaces only.")

    print(f"\n[INIT] Beginning guided Face ID enrollment for: '{name}'")
    print("[INFO] You will be guided through 5 head poses:")
    for idx, step in enumerate(ENROLLMENT_STEPS, 1):
        print(f"       {idx}. {step.title:<18} ({step.instruction})")
    print("\nPress 'ESC' or 'Q' in camera window to cancel.\n")

    # 2. Initialize Hardware & Engines
    config = load_config()
    system = FaceRecognitionSystem(config=config)
    db = DatabaseManager()
    announcer = AudioAnnouncer(config=config)
    session = CompositeEnrollmentSession(identity_name=name)

    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source {video_source}.")
        return False

    window_name = "Apple Face ID Multi-Angle Enrollment Wizard (Press Q to Cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    import face_recognition

    last_step_announced = -1
    success = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h_f, w_f = frame.shape[:2]
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            locations = system.detector.detect_faces(rgb_small)
            head_pose = None
            face_encoding = None
            face_crop = None

            if locations:
                # Use primary detected face
                top_s, right_s, bottom_s, left_s = locations[0]
                y1, y2 = max(0, top_s * 4), min(h_f, bottom_s * 4)
                x1, x2 = max(0, left_s * 4), min(w_f, right_s * 4)
                if y2 > y1 and x2 > x1:
                    face_crop = frame[y1:y2, x1:x2]

                landmarks_list = face_recognition.face_landmarks(rgb_small, [locations[0]])
                if landmarks_list:
                    lms = landmarks_list[0]
                    head_pose, _ = system.anti_spoof.estimate_head_pose(
                        lms, frame_shape=(h_f, w_f), scale=4.0
                    )

                encs = system.encoder.encode_faces(rgb_small, [locations[0]])
                if encs:
                    face_encoding = encs[0]

            # Process state machine
            res = session.process_frame(
                frame_bgr=frame,
                head_pose=head_pose,
                face_encoding=face_encoding,
                face_crop=face_crop
            )

            # Audio guidance on step advancement
            current_step = session.current_step_index
            if current_step != last_step_announced:
                if session.is_completed:
                    announcer.announce_custom("Enrollment complete. Profile saved.")
                elif session.current_step:
                    announcer.announce_custom(session.current_step.instruction)
                last_step_announced = current_step

            # Render Apple Face ID HUD Overlay
            display_frame = frame.copy()
            render_apple_face_id_hud(display_frame, session)

            # Show window
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[CANCEL] Enrollment cancelled by user.")
                break

            if session.is_completed:
                # Save composite profile
                composite_vec = session.composite_embedding
                composite_card = session.generate_composite_portrait()

                print("\n[SUCCESS] All 5 multi-angle poses successfully captured!")
                print(f"[COMPOSITE] Synthesized L2-normalized 128-d composite vector (Norm: {np.linalg.norm(composite_vec):.4f})")

                system.enroll_composite_identity(
                    name=name,
                    composite_embedding=composite_vec,
                    reference_portrait=composite_card
                )

                safe_name = name.replace(" ", "_")
                db.add_person(name, f"known_faces/{safe_name}.jpg")
                print(f"[SAVED] Identity '{name}' successfully enrolled with multi-angle composite embedding.")

                # Render final congratulatory frame for 2.5 seconds
                congrats_frame = display_frame.copy()
                cv2.putText(
                    congrats_frame,
                    "ENROLLMENT COMPLETE",
                    (w_f // 2 - 170, h_f // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    (88, 209, 48),
                    2,
                    cv2.LINE_AA
                )
                cv2.imshow(window_name, congrats_frame)
                cv2.waitKey(2500)
                success = True
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        announcer.stop()

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Apple Face ID Multi-Angle Composite Enrollment Wizard"
    )
    parser.add_argument("--name", type=str, default="", help="Person's full name to enroll")
    parser.add_argument("--video", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    run_face_id_wizard(name=args.name, video_source=args.video)


if __name__ == "__main__":
    main()
