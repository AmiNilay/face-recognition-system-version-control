"""
Standalone Face Enrollment Script

This interactive command-line utility enables users to register new identities
into the Face Recognition System by capturing a live webcam portrait or reviewing
existing enrolled portraits.

Timestamps and date records follow Indian Standard Time (IST).
"""

import os
import sys
import re
import cv2
import numpy as np
from PIL import Image
from src.utils import get_ist_filename_timestamp, format_ist_timestamp

def sanitize_name(name: str) -> str:
    """Sanitize input name to avoid directory traversal or invalid characters."""
    clean = re.sub(r'[^a-zA-Z0-9_\- ]', '', name.strip())
    return re.sub(r'\s+', ' ', clean)

def capture_face(name: str):
    """
    Open the camera feed, provide an on-screen alignment guide,
    and save the user's portrait when SPACE is pressed.

    Args:
        name (str): Person's full name.
    """
    os.makedirs("known_faces", exist_ok=True)
    clean_name = sanitize_name(name)
    safe_filename = clean_name.replace(" ", "_")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera device.")
        return

    print(f"\n[INFO] Position face inside the target box for: {clean_name}")
    print("[INFO] Press SPACE to capture and save.")
    print("[INFO] Press ESC to cancel.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        # Display on-screen instructions
        cv2.putText(frame, f"Capturing: {clean_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to save | ESC to cancel", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw a centered square targeting box for face alignment
        height, width = frame.shape[:2]
        cv2.rectangle(frame, 
                      (width // 2 - 110, height // 2 - 110),
                      (width // 2 + 110, height // 2 + 110),
                      (255, 0, 0), 2)

        cv2.imshow("Enroll Face - Press SPACE to Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE bar pressed
            target_path = f"known_faces/{safe_filename}.jpg"
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Crop or save full image using PIL
            pil_img = Image.fromarray(rgb_frame)
            pil_img.save(target_path, 'JPEG', quality=95)

            print(f"[OK] Face image saved successfully: {target_path}")

            # Visual confirmation on screen
            cv2.putText(frame, "SAVED!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow("Enroll Face - Press SPACE to Capture", frame)
            cv2.waitKey(1200)
            break

        elif key == 27:  # ESC key pressed
            print("[INFO] Enrollment cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Interactive command-line menu loop."""
    print("=" * 50)
    print("       ENROLL FACE INTO RECOGNITION SYSTEM")
    print(f"            Current IST: {format_ist_timestamp()}")
    print("=" * 50)

    while True:
        print("\n1. Enroll new face via camera")
        print("2. List currently enrolled faces")
        print("3. Exit")

        choice = input("\nEnter selection (1-3): ").strip()

        if choice == "1":
            raw_name = input("Enter person's name: ").strip()
            clean_name = sanitize_name(raw_name)

            if clean_name:
                safe_filename = clean_name.replace(" ", "_")
                existing_path = f"known_faces/{safe_filename}.jpg"

                if os.path.exists(existing_path):
                    overwrite = input(f"[WARNING] '{clean_name}' is already enrolled. Overwrite? (y/n): ")
                    if overwrite.lower() != 'y':
                        continue

                capture_face(clean_name)
            else:
                print("[ERROR] Name cannot be empty or contain only invalid characters.")

        elif choice == "2":
            print("\n[DIRECTORY] Enrolled Identities in known_faces/:")
            if os.path.exists("known_faces"):
                faces = [f.replace('.jpg', '').replace('_', ' ') 
                         for f in os.listdir("known_faces") 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if faces:
                    for i, face in enumerate(sorted(faces), 1):
                        print(f"  {i}. {face}")
                else:
                    print("  (No enrolled faces found)")
            else:
                print("  (known_faces directory not found)")

        elif choice == "3":
            print("[EXIT] Exiting enrollment utility.")
            break
        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
