import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import pickle
from pathlib import Path
from PIL import Image

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.face_confidences = []  # NEW: Store confidence scores
        
    def load_known_faces(self, images_folder="known_faces"):
        """Load images from folder and encode faces - supports all image formats"""
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
            print(f"Created folder: {images_folder}")
            print("Please add images in format: person_name.jpg")
            return False
        
        image_files = [f for f in os.listdir(images_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.tiff', '.webp', '.bmp'))]
        
        if not image_files:
            print(f"No images found in {images_folder}")
            return False
        
        print("Loading known faces...")
        
        for filename in image_files:
            try:
                image_path = os.path.join(images_folder, filename)
                
                # Load image using PIL (supports all formats)
                pil_image = Image.open(image_path)
                
                # Convert to RGB (handles PNG with alpha, GIF, etc.)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Convert PIL image to numpy array with explicit dtype=np.uint8
                image_array = np.array(pil_image, dtype=np.uint8)
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(image_array, num_jitters=1)
                
                if face_encodings:
                    face_encoding = face_encodings[0]
                    name = os.path.splitext(filename)[0].replace('_', ' ')
                    
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                    print(f"✅ Loaded: {name}")
                else:
                    print(f"⚠️  No face found in {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {str(e)}")
        
        print(f"\n📊 Total loaded: {len(self.known_face_names)} faces")
        return len(self.known_face_names) > 0
    
    def save_encodings(self, filename="face_encodings.pkl"):
        """Save face encodings to file"""
        data = {
            "names": self.known_face_names,
            "encodings": self.known_face_encodings
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Encodings saved to {filename}")
    
    def load_encodings(self, filename="face_encodings.pkl"):
        """Load face encodings from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                self.known_face_names = data["names"]
                self.known_face_encodings = data["encodings"]
                print(f"Loaded {len(self.known_face_names)} face encodings from {filename}")
                return True
            except Exception as e:
                print(f"Error loading encodings: {e}")
                return False
        return False
    
    def recognize_face(self, face_encoding):
        """
        Recognize a single face and return name with confidence score
        Returns: (name, confidence)
        """
        name = "Unknown"
        confidence = 0.0
        
        if len(self.known_face_encodings) > 0:
            # Calculate face distances
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, 
                face_encoding
            )
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            distance = face_distances[best_match_index]
            
            # Convert distance to confidence percentage
            # Distance of 0.0 = 100% confidence, Distance of 1.0 = 0% confidence
            confidence = max(0.0, (1.0 - distance)) * 100
            
            # Check if match is good enough (distance < 0.6 = confidence > 40%)
            if distance < 0.6:
                name = self.known_face_names[best_match_index]
            else:
                name = "Unknown"
                confidence = 0.0
        
        return name, confidence
    
    def run_recognition(self, video_source=0):
        """Run face recognition on video stream - Optimized & Working"""
        video_capture = cv2.VideoCapture(video_source)
        
        # Set camera properties
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not video_capture.isOpened():
            print("Error: Could not open video source")
            return
        
        print("\n🎥 Camera started!")
        print("\n⌨️  Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("  - Press 'c' to capture and add new face\n")
        
        # Process every 3rd frame (good balance)
        frame_count = 0
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Process every 3rd frame for face recognition
            if frame_count % 3 == 0:
                # Resize to 1/4 size for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                
                # Convert BGR to RGB
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces - using HOG model with 1 upsample
                self.face_locations = face_recognition.face_locations(
                    rgb_small_frame, 
                    number_of_times_to_upsample=1,
                    model="hog"
                )
                
                # Get face encodings
                if self.face_locations:
                    self.face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, 
                        self.face_locations,
                        num_jitters=1
                    )
                    
                    # Recognize each face with confidence
                    self.face_names = []
                    self.face_confidences = []
                    
                    for face_encoding in self.face_encodings:
                        name, confidence = self.recognize_face(face_encoding)
                        self.face_names.append(name)
                        self.face_confidences.append(confidence)
                else:
                    self.face_encodings = []
                    self.face_names = []
                    self.face_confidences = []
            
            # Draw results on every frame
            self.draw_results(frame)
            
            # Display frame
            cv2.imshow('Face Recognition System - Press Q to quit', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_frame(frame)
            elif key == ord('c'):
                self.capture_new_face(frame)
        
        video_capture.release()
        cv2.destroyAllWindows()
        print("\n👋 Camera closed")
    
    def draw_results(self, frame):
        """Draw bounding boxes, names, and confidence scores on frame"""
        for i, ((top, right, bottom, left), name) in enumerate(zip(self.face_locations, self.face_names)):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Get confidence for this face
            confidence = self.face_confidences[i] if i < len(self.face_confidences) else 0
            
            # Choose color based on confidence
            if name == "Unknown":
                color = (0, 0, 255)  # Red
            elif confidence >= 70:
                color = (0, 255, 0)  # Green - High confidence
            elif confidence >= 50:
                color = (0, 165, 255)  # Orange - Medium confidence
            else:
                color = (0, 255, 255)  # Yellow - Low confidence
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background (taller to fit confidence)
            cv2.rectangle(frame, (left, bottom - 55), (right, bottom), color, cv2.FILLED)
            
            # Draw name
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 35), 
                       font, 0.6, (255, 255, 255), 1)
            
            # Draw confidence score
            if name != "Unknown":
                confidence_text = f"{confidence:.1f}%"
                cv2.putText(frame, confidence_text, (left + 6, bottom - 10), 
                           font, 0.5, (255, 255, 255), 1)
        
        # Display stats with average confidence
        cv2.putText(frame, f"Faces: {len(self.face_locations)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Known: {len(self.known_face_names)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show average confidence if faces detected
        if self.face_confidences and any(c > 0 for c in self.face_confidences):
            valid_confidences = [c for c in self.face_confidences if c > 0]
            avg_confidence = sum(valid_confidences) / len(valid_confidences)
            cv2.putText(frame, f"Avg Confidence: {avg_confidence:.1f}%", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def save_frame(self, frame):
        """Save current frame to file"""
        os.makedirs("captured_images", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_images/capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"\n📸 Frame saved as {filename}")
    
    def capture_new_face(self, frame):
        """Capture and add new face to known faces"""
        # Detect faces in current frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        temp_locations = face_recognition.face_locations(rgb_frame)
        
        if len(temp_locations) == 0:
            print("\n❌ No face detected in frame")
            return
        
        if len(temp_locations) > 1:
            print("\n⚠️  Multiple faces detected. Please ensure only one face is visible")
            return
        
        print("\n📝 Enter name for the detected face: ", end='', flush=True)
        name = input().strip()
        
        if name:
            # Save image
            safe_name = name.replace(" ", "_")
            filename = f"known_faces/{safe_name}.jpg"
            os.makedirs("known_faces", exist_ok=True)
            
            # Save using PIL
            pil_image = Image.fromarray(rgb_frame.astype(np.uint8))
            pil_image.save(filename, 'JPEG', quality=95)
            
            # Get encoding
            temp_encodings = face_recognition.face_encodings(rgb_frame, temp_locations, num_jitters=1)
            
            if temp_encodings:
                # Add to known faces
                self.known_face_encodings.append(temp_encodings[0])
                self.known_face_names.append(name)
                
                print(f"✅ Added {name} to known faces\n")
                self.save_encodings()
            else:
                print(f"❌ Could not encode face\n")
        else:
            print("❌ Name cannot be empty\n")
    
    def add_face_from_frame(self, frame, name):
        """Add face from a given frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame, num_jitters=1)
        
        if face_encodings:
            safe_name = name.replace(" ", "_")
            filename = f"known_faces/{safe_name}.jpg"
            os.makedirs("known_faces", exist_ok=True)
            
            # Save using PIL
            pil_image = Image.fromarray(rgb_frame.astype(np.uint8))
            pil_image.save(filename, 'JPEG', quality=95)
            
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            self.save_encodings()
            return True
        return False
    
    def add_face_from_image(self, image_path, name):
        """Add face from image file"""
        try:
            # Load using PIL
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image, dtype=np.uint8)
            face_encodings = face_recognition.face_encodings(image_array, num_jitters=1)
            
            if face_encodings:
                safe_name = name.replace(" ", "_")
                filename = f"known_faces/{safe_name}.jpg"
                os.makedirs("known_faces", exist_ok=True)
                
                # Save using PIL
                pil_image.save(filename, 'JPEG', quality=95)
                
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                self.save_encodings()
                return True
        except Exception as e:
            print(f"Error adding face: {e}")
        return False
    
    def remove_face(self, name):
        """Remove a face from known faces"""
        if name in self.known_face_names:
            idx = self.known_face_names.index(name)
            del self.known_face_names[idx]
            del self.known_face_encodings[idx]
            
            # Try both formats
            safe_name = name.replace(" ", "_")
            image_path1 = f"known_faces/{name}.jpg"
            image_path2 = f"known_faces/{safe_name}.jpg"
            
            if os.path.exists(image_path1):
                os.remove(image_path1)
            if os.path.exists(image_path2):
                os.remove(image_path2)
            
            self.save_encodings()
            return True
        return False
    
    def get_known_faces(self):
        """Get list of known face names"""
        return self.known_face_names
    
    def process_frame(self, frame):
        """Process a single frame and return results with confidence"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)
        
        faces_info = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name, confidence = self.recognize_face(face_encoding)
            
            # Determine color based on confidence
            if name == "Unknown":
                color = (0, 0, 255)
            elif confidence >= 70:
                color = (0, 255, 0)
            elif confidence >= 50:
                color = (0, 165, 255)
            else:
                color = (0, 255, 255)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 55), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 35), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            if name != "Unknown":
                confidence_text = f"{confidence:.1f}%"
                cv2.putText(frame, confidence_text, (left + 6, bottom - 10), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            faces_info.append({
                'name': name,
                'confidence': confidence,
                'location': (top, right, bottom, left)
            })
        
        info = {
            'face_count': len(face_locations),
            'faces': faces_info
        }
        
        return frame, info