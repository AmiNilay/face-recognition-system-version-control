import os
import sys
import cv2
import numpy as np

try:
    import cv2
except ImportError:
    print("Installing OpenCV...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2

def capture_face(name):
    """Capture and save face properly"""
    os.makedirs("known_faces", exist_ok=True)
    
    # Replace spaces with underscores in filename
    safe_name = name.replace(" ", "_")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print(f"\n📸 Position your face in the camera for: {name}")
    print("➡️  Press SPACE to capture")
    print("➡️  Press ESC to cancel\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        cv2.putText(frame, f"Capturing: {name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to save", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        height, width = frame.shape[:2]
        cv2.rectangle(frame, 
                     (width//2 - 100, height//2 - 100),
                     (width//2 + 100, height//2 + 100),
                     (255, 0, 0), 2)
        
        cv2.imshow("Add Face - Press SPACE to capture", frame)
        
        key = cv2.waitKey(1)
        
        if key == 32:  # SPACE
            filename = f"known_faces/{safe_name}.jpg"
            
            # Convert to RGB then back to ensure proper format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save using PIL to ensure compatibility
            from PIL import Image
            img = Image.fromarray(rgb_frame)
            img.save(filename, 'JPEG', quality=95)
            
            print(f"✅ Face saved successfully: {filename}")
            
            cv2.putText(frame, "CAPTURED!", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow("Add Face - Press SPACE to capture", frame)
            cv2.waitKey(2000)
            break
            
        elif key == 27:  # ESC
            print("❌ Cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("="*50)
    print("     ADD NEW FACE TO RECOGNITION SYSTEM")
    print("="*50)
    
    while True:
        print("\n1. Add new face")
        print("2. View existing faces")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            name = input("Enter person's name: ").strip()
            
            if name:
                safe_name = name.replace(" ", "_")
                if os.path.exists(f"known_faces/{safe_name}.jpg"):
                    overwrite = input(f"⚠️  {name} already exists. Overwrite? (y/n): ")
                    if overwrite.lower() != 'y':
                        continue
                
                capture_face(name)
            else:
                print("❌ Name cannot be empty!")
                
        elif choice == "2":
            print("\n📁 Existing faces in system:")
            if os.path.exists("known_faces"):
                faces = [f.replace('.jpg', '').replace('_', ' ') 
                        for f in os.listdir("known_faces") if f.endswith('.jpg')]
                if faces:
                    for i, face in enumerate(faces, 1):
                        print(f"  {i}. {face}")
                else:
                    print("  No faces found")
            else:
                print("  No faces directory found")
                
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    try:
        print(f"OpenCV version: {cv2.__version__}")
    except:
        print("OpenCV not properly installed")
    
    main()