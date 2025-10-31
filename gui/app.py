"""
GUI Application for Face Recognition System with Recognition History
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import os
import sys
from pathlib import Path
from datetime import datetime
import csv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.face_recognition_system import FaceRecognitionSystem
from src.database_manager import DatabaseManager
from src.utils import load_config, setup_logging
import logging

class FaceRecognitionGUI:
    """Main GUI Application for Face Recognition with History Tracking"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition System v1.0")
        self.root.geometry("1200x700")
        self.root.configure(bg="#2b2b2b")
        
        # Initialize logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize system components
        self.system = FaceRecognitionSystem()
        self.db_manager = DatabaseManager()
        self.config = load_config()
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.video_thread = None
        
        # Recognition history tracking
        self.recognition_history = []
        self.last_recognized = {}  # Track last recognition time for each person
        self.recognition_cooldown = 5  # Seconds between logging same person
        
        # Auto-save history
        self.auto_save_enabled = True
        self.history_file = "data/recognition_history.csv"
        
        # Setup GUI
        self.setup_gui()
        
        # Load known faces
        self.load_faces()
        
        # Load existing history
        self.load_history()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        
        # Create menu bar
        self.create_menu()
        
        # Title bar
        title_frame = tk.Frame(self.root, bg="#1e1e1e", height=70)
        title_frame.pack(fill="x", padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🎥 Face Recognition System",
            font=("Arial", 24, "bold"),
            bg="#1e1e1e",
            fg="#00ff00"
        )
        title_label.pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg="#2b2b2b")
        main_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left panel - Video feed
        left_panel = tk.Frame(main_container, bg="#1e1e1e")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Video display
        video_frame = tk.Frame(left_panel, bg="#000000", relief="sunken", bd=2)
        video_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.video_label = tk.Label(video_frame, bg="#000000", text="Camera Off", 
                                    font=("Arial", 16), fg="gray")
        self.video_label.pack(fill="both", expand=True)
        
        # Video controls
        controls_frame = tk.Frame(left_panel, bg="#1e1e1e")
        controls_frame.pack(pady=10)
        
        self.start_btn = tk.Button(
            controls_frame,
            text="▶ Start Camera",
            command=self.start_camera,
            bg="#00aa00",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            cursor="hand2",
            relief="raised"
        )
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = tk.Button(
            controls_frame,
            text="⏹ Stop Camera",
            command=self.stop_camera,
            bg="#aa0000",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            state="disabled",
            cursor="hand2",
            relief="raised"
        )
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        self.capture_btn = tk.Button(
            controls_frame,
            text="📸 Capture Face",
            command=self.capture_face,
            bg="#0066cc",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            state="disabled",
            cursor="hand2",
            relief="raised"
        )
        self.capture_btn.grid(row=0, column=2, padx=5)
        
        # Right panel - Controls and info
        right_panel = tk.Frame(main_container, bg="#1e1e1e", width=380)
        right_panel.pack(side="right", fill="both", padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Detection info
        info_frame = tk.LabelFrame(
            right_panel,
            text="📊 Detection Info",
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Arial", 12, "bold"),
            relief="groove",
            bd=2
        )
        info_frame.pack(fill="x", padx=10, pady=10)
        
        self.info_text = tk.Text(
            info_frame,
            height=8,
            width=35,
            bg="#2b2b2b",
            fg="white",
            font=("Courier", 10),
            relief="flat"
        )
        self.info_text.pack(fill="x", padx=10, pady=10)
        self.info_text.insert("1.0", "No faces detected")
        self.info_text.config(state="disabled")
        
        # Known faces frame
        faces_frame = tk.LabelFrame(
            right_panel,
            text="👥 Known Faces",
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Arial", 12, "bold"),
            relief="groove",
            bd=2
        )
        faces_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Listbox with scrollbar
        list_container = tk.Frame(faces_frame, bg="#1e1e1e")
        list_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        
        self.faces_listbox = tk.Listbox(
            list_container,
            yscrollcommand=scrollbar.set,
            bg="#2b2b2b",
            fg="white",
            font=("Arial", 11),
            selectmode="single",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#00ff00"
        )
        self.faces_listbox.pack(fill="both", expand=True, side="left")
        scrollbar.config(command=self.faces_listbox.yview)
        
        # Face management buttons
        btn_frame = tk.Frame(faces_frame, bg="#1e1e1e")
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame,
            text="➕ Add",
            command=self.add_face_dialog,
            bg="#00aa00",
            fg="white",
            font=("Arial", 10, "bold"),
            width=10,
            cursor="hand2"
        ).grid(row=0, column=0, padx=3)
        
        tk.Button(
            btn_frame,
            text="🗑️ Remove",
            command=self.remove_face,
            bg="#aa0000",
            fg="white",
            font=("Arial", 10, "bold"),
            width=10,
            cursor="hand2"
        ).grid(row=0, column=1, padx=3)
        
        tk.Button(
            btn_frame,
            text="🔄 Refresh",
            command=self.refresh_faces,
            bg="#0066cc",
            fg="white",
            font=("Arial", 10, "bold"),
            width=10,
            cursor="hand2"
        ).grid(row=0, column=2, padx=3)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Arial", 10),
            anchor="w",
            relief="sunken"
        )
        self.status_bar.pack(side="bottom", fill="x")
        
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Add Face from File", command=self.add_face_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export Recognition History", command=self.export_recognition_history)
        file_menu.add_command(label="Export Database Logs", command=self.export_logs)
        file_menu.add_separator()
        file_menu.add_command(label="Clear History", command=self.clear_history)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Recognition History", command=self.show_history)
        view_menu.add_command(label="Database History", command=self.show_db_history)
        view_menu.add_command(label="Known Faces Gallery", command=self.show_gallery)
        view_menu.add_separator()
        view_menu.add_command(label="Today's Recognitions", command=self.show_today_history)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        
        # Auto-save toggle
        self.auto_save_var = tk.BooleanVar(value=True)
        settings_menu.add_checkbutton(
            label="Auto-save History",
            variable=self.auto_save_var,
            command=self.toggle_auto_save
        )
        settings_menu.add_command(label="Set Recognition Cooldown", command=self.set_cooldown)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def load_faces(self):
        """Load known faces on startup"""
        try:
            if not self.system.load_encodings():
                if self.system.load_known_faces():
                    self.system.save_encodings()
            
            self.refresh_faces()
            count = len(self.system.known_face_names)
            self.update_status(f"Loaded {count} known face(s)")
            self.logger.info(f"Loaded {count} known faces")
        except Exception as e:
            self.logger.error(f"Error loading faces: {e}")
            messagebox.showerror("Error", f"Failed to load faces: {str(e)}")
    
    def load_history(self):
        """Load existing recognition history from CSV"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self.recognition_history = list(reader)
                self.logger.info(f"Loaded {len(self.recognition_history)} history records")
        except Exception as e:
            self.logger.error(f"Error loading history: {e}")
    
    def save_history(self):
        """Save recognition history to CSV"""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            with open(self.history_file, 'w', newline='', encoding='utf-8') as f:
                if self.recognition_history:
                    fieldnames = self.recognition_history[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.recognition_history)
            
            self.logger.info(f"Saved {len(self.recognition_history)} history records")
        except Exception as e:
            self.logger.error(f"Error saving history: {e}")
    
    def add_recognition_to_history(self, name, confidence):
        """Add a recognition event to history with cooldown"""
        try:
            current_time = datetime.now()
            
            # Check cooldown
            if name in self.last_recognized:
                time_diff = (current_time - self.last_recognized[name]).total_seconds()
                if time_diff < self.recognition_cooldown:
                    return  # Skip if within cooldown period
            
            # Add to history
            record = {
                'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'name': name,
                'confidence': f"{float(confidence):.2f}%",
                'date': current_time.strftime("%Y-%m-%d"),
                'time': current_time.strftime("%H:%M:%S")
            }
            
            self.recognition_history.append(record)
            self.last_recognized[name] = current_time
            
            # Auto-save if enabled
            if self.auto_save_enabled:
                self.save_history()
            
            self.logger.info(f"Recognition logged: {name} ({confidence:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Error adding to history: {e}")
            
    def refresh_faces(self):
        """Refresh the faces listbox"""
        self.faces_listbox.delete(0, tk.END)
        
        for i, name in enumerate(self.system.known_face_names, 1):
            self.faces_listbox.insert(tk.END, f"  {i}. {name}")
        
        if len(self.system.known_face_names) == 0:
            self.faces_listbox.insert(tk.END, "  (No faces registered)")
            
    def start_camera(self):
        """Start the camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera!")
                return
            
            self.is_running = True
            
            # Update buttons
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.capture_btn.config(state="normal")
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
            self.update_status("Camera started")
            self.logger.info("Camera started")
            
        except Exception as e:
            self.logger.error(f"Error starting camera: {e}")
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
            
    def stop_camera(self):
        """Stop the camera"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        # Update buttons
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.capture_btn.config(state="disabled")
        
        # Clear video
        self.video_label.config(image="", text="Camera Off", fg="gray")
        
        self.update_status("Camera stopped")
        self.logger.info("Camera stopped")
        
    def video_loop(self):
        """Video capture and processing loop"""
        import face_recognition
        frame_count = 0
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 3rd frame
                if frame_count % 3 == 0:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    self.system.face_locations = face_recognition.face_locations(
                        rgb_small_frame,
                        number_of_times_to_upsample=1,
                        model="hog"
                    )
                    
                    if self.system.face_locations:
                        self.system.face_encodings = face_recognition.face_encodings(
                            rgb_small_frame,
                            self.system.face_locations,
                            num_jitters=1
                        )
                        
                        self.system.face_names = []
                        self.system.face_confidences = []
                        
                        for face_encoding in self.system.face_encodings:
                            name, confidence = self.system.recognize_face(face_encoding)
                            self.system.face_names.append(name)
                            self.system.face_confidences.append(confidence)
                            
                            # Log recognition to history
                            if name != "Unknown" and confidence > 0:
                                self.add_recognition_to_history(name, confidence)
                                
                                # Log to database
                                try:
                                    confidence_value = float(confidence) / 100.0
                                    self.db_manager.log_recognition(name, confidence_value)
                                except Exception as e:
                                    self.logger.warning(f"Could not log to database: {e}")
                    else:
                        self.system.face_encodings = []
                        self.system.face_names = []
                        self.system.face_confidences = []
                
                # Draw results
                self.system.draw_results(frame)
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                height, width = frame_rgb.shape[:2]
                max_width = 750
                if width > max_width:
                    ratio = max_width / width
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Convert to PIL Image
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                
                # Update video display
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo
                
                # Update info
                self.update_detection_info()
                
            except Exception as e:
                self.logger.error(f"Error in video loop: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                
    def update_detection_info(self):
        """Update detection info display with confidence scores"""
        try:
            self.info_text.config(state="normal")
            self.info_text.delete("1.0", tk.END)
            
            if len(self.system.face_locations) == 0:
                self.info_text.insert("1.0", "No faces detected")
            else:
                info = f"Faces Detected: {len(self.system.face_locations)}\n"
                info += "=" * 35 + "\n\n"
                
                # Ensure we have confidence scores
                if not hasattr(self.system, 'face_confidences'):
                    self.system.face_confidences = []
                
                # Match faces with confidences
                for i, name in enumerate(self.system.face_names):
                    # Get confidence safely
                    if i < len(self.system.face_confidences):
                        confidence = self.system.face_confidences[i]
                    else:
                        confidence = 0.0
                    
                    # Ensure confidence is a number
                    try:
                        confidence = float(confidence)
                    except:
                        confidence = 0.0
                    
                    if name != "Unknown":
                        # Color code based on confidence
                        if confidence >= 70:
                            status = "🟢"  # High confidence
                        elif confidence >= 50:
                            status = "🟡"  # Medium confidence
                        else:
                            status = "🟠"  # Low confidence
                        
                        info += f"{status} Face {i+1}: {name}\n"
                        info += f"   Confidence: {confidence:.1f}%\n\n"
                    else:
                        info += f"❌ Face {i+1}: Unknown\n\n"
                
                # Show average confidence
                if self.system.face_confidences:
                    try:
                        valid_confidences = [float(c) for c in self.system.face_confidences if float(c) > 0]
                        if valid_confidences:
                            avg_conf = sum(valid_confidences) / len(valid_confidences)
                            info += "=" * 35 + "\n"
                            info += f"Average Confidence: {avg_conf:.1f}%\n"
                    except:
                        pass
                
                self.info_text.insert("1.0", info)
            
            self.info_text.config(state="disabled")
        except Exception as e:
            self.logger.error(f"Error updating detection info: {e}")
                
    def capture_face(self):
        """Capture face from current frame"""
        if not self.is_running or self.cap is None:
            messagebox.showwarning("Warning", "Camera is not running!")
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Could not capture frame!")
                return
            
            # Detect faces
            import face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            temp_locations = face_recognition.face_locations(rgb_frame)
            
            if len(temp_locations) == 0:
                messagebox.showwarning("No Face", "No face detected in frame!")
                return
            
            if len(temp_locations) > 1:
                messagebox.showwarning("Multiple Faces", 
                    "Multiple faces detected! Please ensure only one face is visible.")
                return
            
            # Ask for name
            name = simpledialog.askstring("Add Face", "Enter person's name:")
            
            if name and name.strip():
                name = name.strip()
                
                # Add face
                success = self.system.add_face_from_frame(frame, name)
                
                if success:
                    self.db_manager.add_person(name, f"known_faces/{name.replace(' ', '_')}.jpg")
                    self.refresh_faces()
                    messagebox.showinfo("Success", f"Face added for {name}!")
                    self.update_status(f"Added face: {name}")
                    self.logger.info(f"Added face: {name}")
                else:
                    messagebox.showerror("Error", "Could not encode face!")
                    
        except Exception as e:
            self.logger.error(f"Error capturing face: {e}")
            messagebox.showerror("Error", f"Failed to capture face: {str(e)}")
            
    def add_face_dialog(self):
        """Add face from image file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
            
            if file_path:
                name = simpledialog.askstring("Add Face", "Enter person's name:")
                
                if name and name.strip():
                    name = name.strip()
                    success = self.system.add_face_from_image(file_path, name)
                    
                    if success:
                        self.db_manager.add_person(name, f"known_faces/{name.replace(' ', '_')}.jpg")
                        self.refresh_faces()
                        messagebox.showinfo("Success", f"Face added for {name}!")
                        self.update_status(f"Added face: {name}")
                        self.logger.info(f"Added face from file: {name}")
                    else:
                        messagebox.showerror("Error", "No face detected in image!")
                        
        except Exception as e:
            self.logger.error(f"Error adding face from file: {e}")
            messagebox.showerror("Error", f"Failed to add face: {str(e)}")
            
    def remove_face(self):
        """Remove selected face"""
        selection = self.faces_listbox.curselection()
        
        if not selection:
            messagebox.showwarning("No Selection", "Please select a face to remove!")
            return
        
        try:
            selected_text = self.faces_listbox.get(selection[0])
            # Extract name (remove number prefix)
            name = selected_text.split(". ", 1)[1] if ". " in selected_text else selected_text.strip()
            
            if messagebox.askyesno("Confirm", f"Remove '{name}' from known faces?"):
                success = self.system.remove_face(name)
                
                if success:
                    self.db_manager.delete_person(name)
                    self.refresh_faces()
                    messagebox.showinfo("Success", f"'{name}' removed!")
                    self.update_status(f"Removed face: {name}")
                    self.logger.info(f"Removed face: {name}")
                else:
                    messagebox.showerror("Error", "Could not remove face!")
                    
        except Exception as e:
            self.logger.error(f"Error removing face: {e}")
            messagebox.showerror("Error", f"Failed to remove face: {str(e)}")
    
    def show_history(self):
        """Show recognition history from CSV"""
        try:
            history_window = tk.Toplevel(self.root)
            history_window.title("Recognition History")
            history_window.geometry("900x600")
            history_window.configure(bg="#2b2b2b")
            
            # Title
            title = tk.Label(
                history_window,
                text="📜 Recognition History",
                font=("Arial", 16, "bold"),
                bg="#2b2b2b",
                fg="#00ff00"
            )
            title.pack(pady=10)
            
            # Stats
            stats_frame = tk.Frame(history_window, bg="#1e1e1e")
            stats_frame.pack(fill="x", padx=10, pady=5)
            
            total_records = len(self.recognition_history)
            unique_people = len(set(r['name'] for r in self.recognition_history)) if self.recognition_history else 0
            
            tk.Label(
                stats_frame,
                text=f"Total Records: {total_records} | Unique People: {unique_people}",
                bg="#1e1e1e",
                fg="white",
                font=("Arial", 11)
            ).pack(pady=5)
            
            # Create frame for treeview
            tree_frame = tk.Frame(history_window, bg="#2b2b2b")
            tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Scrollbars
            vsb = tk.Scrollbar(tree_frame, orient="vertical")
            vsb.pack(side="right", fill="y")
            
            hsb = tk.Scrollbar(tree_frame, orient="horizontal")
            hsb.pack(side="bottom", fill="x")
            
            # Create treeview
            tree = ttk.Treeview(
                tree_frame,
                columns=("Name", "Date", "Time", "Confidence"),
                show="headings",
                yscrollcommand=vsb.set,
                xscrollcommand=hsb.set
            )
            
            tree.heading("Name", text="Name")
            tree.heading("Date", text="Date")
            tree.heading("Time", text="Time")
            tree.heading("Confidence", text="Confidence")
            
            tree.column("Name", width=200)
            tree.column("Date", width=150)
            tree.column("Time", width=150)
            tree.column("Confidence", width=120)
            
            # Add data (most recent first)
            for record in reversed(self.recognition_history):
                tree.insert("", tk.END, values=(
                    record.get('name', ''),
                    record.get('date', ''),
                    record.get('time', ''),
                    record.get('confidence', '')
                ))
            
            tree.pack(fill="both", expand=True, side="left")
            vsb.config(command=tree.yview)
            hsb.config(command=tree.xview)
            
            # Buttons
            btn_frame = tk.Frame(history_window, bg="#2b2b2b")
            btn_frame.pack(pady=10)
            
            tk.Button(
                btn_frame,
                text="📊 Export",
                command=self.export_recognition_history,
                bg="#0066cc",
                fg="white",
                font=("Arial", 10, "bold"),
                width=12
            ).pack(side="left", padx=5)
            
            tk.Button(
                btn_frame,
                text="🔄 Refresh",
                command=lambda: self.show_history(),
                bg="#00aa00",
                fg="white",
                font=("Arial", 10, "bold"),
                width=12
            ).pack(side="left", padx=5)
            
            tk.Button(
                btn_frame,
                text="🗑️ Clear All",
                command=self.clear_history,
                bg="#aa0000",
                fg="white",
                font=("Arial", 10, "bold"),
                width=12
            ).pack(side="left", padx=5)
            
        except Exception as e:
            self.logger.error(f"Error showing history: {e}")
            messagebox.showerror("Error", f"Failed to show history: {str(e)}")
    
    def show_db_history(self):
        """Show database recognition history"""
        try:
            history_window = tk.Toplevel(self.root)
            history_window.title("Database History")
            history_window.geometry("800x500")
            history_window.configure(bg="#2b2b2b")
            
            # Title
            title = tk.Label(
                history_window,
                text="💾 Database History",
                font=("Arial", 16, "bold"),
                bg="#2b2b2b",
                fg="#00ff00"
            )
            title.pack(pady=10)
            
            # Create treeview
            tree_frame = tk.Frame(history_window, bg="#2b2b2b")
            tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            scrollbar = tk.Scrollbar(tree_frame)
            scrollbar.pack(side="right", fill="y")
            
            tree = ttk.Treeview(
                tree_frame,
                columns=("Name", "Time", "Confidence"),
                show="headings",
                yscrollcommand=scrollbar.set
            )
            
            tree.heading("Name", text="Name")
            tree.heading("Time", text="Time")
            tree.heading("Confidence", text="Confidence")
            
            tree.column("Name", width=200)
            tree.column("Time", width=300)
            tree.column("Confidence", width=100)
            
            # Get history
            history = self.db_manager.get_recognition_history(limit=100)
            
            for _, row in history.iterrows():
                tree.insert("", tk.END, values=(
                    row['name'],
                    row['timestamp'],
                    f"{row['confidence']:.1%}"
                ))
            
            tree.pack(fill="both", expand=True, side="left")
            scrollbar.config(command=tree.yview)
            
        except Exception as e:
            self.logger.error(f"Error showing database history: {e}")
            messagebox.showerror("Error", f"Failed to load database history: {str(e)}")
    
    def show_today_history(self):
        """Show today's recognitions"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            today_records = [r for r in self.recognition_history if r.get('date') == today]
            
            # Create window
            history_window = tk.Toplevel(self.root)
            history_window.title("Today's Recognitions")
            history_window.geometry("800x500")
            history_window.configure(bg="#2b2b2b")
            
            # Title
            title = tk.Label(
                history_window,
                text=f"📅 Today's Recognitions ({today})",
                font=("Arial", 16, "bold"),
                bg="#2b2b2b",
                fg="#00ff00"
            )
            title.pack(pady=10)
            
            # Stats
            stats = tk.Label(
                history_window,
                text=f"Total recognitions today: {len(today_records)}",
                font=("Arial", 11),
                bg="#2b2b2b",
                fg="white"
            )
            stats.pack(pady=5)
            
            # Create treeview
            tree_frame = tk.Frame(history_window, bg="#2b2b2b")
            tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            scrollbar = tk.Scrollbar(tree_frame)
            scrollbar.pack(side="right", fill="y")
            
            tree = ttk.Treeview(
                tree_frame,
                columns=("Name", "Time", "Confidence"),
                show="headings",
                yscrollcommand=scrollbar.set
            )
            
            tree.heading("Name", text="Name")
            tree.heading("Time", text="Time")
            tree.heading("Confidence", text="Confidence")
            
            tree.column("Name", width=250)
            tree.column("Time", width=200)
            tree.column("Confidence", width=150)
            
            # Add today's records
            for record in reversed(today_records):
                tree.insert("", tk.END, values=(
                    record.get('name', ''),
                    record.get('time', ''),
                    record.get('confidence', '')
                ))
            
            tree.pack(fill="both", expand=True, side="left")
            scrollbar.config(command=tree.yview)
            
        except Exception as e:
            self.logger.error(f"Error showing today's history: {e}")
            messagebox.showerror("Error", f"Failed to show today's history: {str(e)}")
            
    def show_gallery(self):
        """Show known faces gallery"""
        try:
            gallery_window = tk.Toplevel(self.root)
            gallery_window.title("Known Faces Gallery")
            gallery_window.geometry("700x500")
            gallery_window.configure(bg="#2b2b2b")
            
            canvas = tk.Canvas(gallery_window, bg="#2b2b2b")
            scrollbar = tk.Scrollbar(gallery_window, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg="#2b2b2b")
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Display faces
            persons = self.db_manager.get_all_persons()
            
            for i, person in enumerate(persons):
                row = i // 3
                col = i % 3
                
                frame = tk.Frame(scrollable_frame, bg="#1e1e1e", relief="raised", bd=2)
                frame.grid(row=row, column=col, padx=10, pady=10)
                
                # Display image
                image_path = person['image_path']
                if image_path and os.path.exists(image_path):
                    img = Image.open(image_path)
                    img.thumbnail((150, 150))
                    photo = ImageTk.PhotoImage(img)
                    label = tk.Label(frame, image=photo, bg="#1e1e1e")
                    label.image = photo
                    label.pack(pady=5)
                
                tk.Label(
                    frame,
                    text=person['name'],
                    font=("Arial", 11, "bold"),
                    bg="#1e1e1e",
                    fg="white"
                ).pack()
                
                tk.Label(
                    frame,
                    text=f"Added: {person['added_date'][:10]}",
                    font=("Arial", 8),
                    bg="#1e1e1e",
                    fg="gray"
                ).pack()
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
        except Exception as e:
            self.logger.error(f"Error showing gallery: {e}")
            messagebox.showerror("Error", f"Failed to load gallery: {str(e)}")
    
    def export_recognition_history(self):
        """Export recognition history to CSV"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=f"recognition_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if file_path:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    if self.recognition_history:
                        fieldnames = self.recognition_history[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(self.recognition_history)
                
                messagebox.showinfo("Success", f"History exported to:\n{file_path}")
                self.logger.info(f"Exported recognition history to: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error exporting recognition history: {e}")
            messagebox.showerror("Error", f"Failed to export history: {str(e)}")
            
    def export_logs(self):
        """Export database logs to CSV"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=f"database_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if file_path:
                exported_path = self.db_manager.export_to_csv(file_path)
                messagebox.showinfo("Success", f"Database logs exported to:\n{exported_path}")
                self.logger.info(f"Exported database logs to: {exported_path}")
                
        except Exception as e:
            self.logger.error(f"Error exporting database logs: {e}")
            messagebox.showerror("Error", f"Failed to export logs: {str(e)}")
    
    def clear_history(self):
        """Clear recognition history"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all recognition history?"):
            try:
                self.recognition_history = []
                self.last_recognized = {}
                self.save_history()
                messagebox.showinfo("Success", "Recognition history cleared!")
                self.logger.info("Recognition history cleared")
            except Exception as e:
                self.logger.error(f"Error clearing history: {e}")
                messagebox.showerror("Error", f"Failed to clear history: {str(e)}")
    
    def toggle_auto_save(self):
        """Toggle auto-save feature"""
        self.auto_save_enabled = self.auto_save_var.get()
        status = "enabled" if self.auto_save_enabled else "disabled"
        self.update_status(f"Auto-save {status}")
        self.logger.info(f"Auto-save {status}")
    
    def set_cooldown(self):
        """Set recognition cooldown period"""
        cooldown = simpledialog.askinteger(
            "Set Cooldown",
            "Enter cooldown period in seconds:",
            initialvalue=self.recognition_cooldown,
            minvalue=1,
            maxvalue=60
        )
        
        if cooldown:
            self.recognition_cooldown = cooldown
            messagebox.showinfo("Success", f"Cooldown set to {cooldown} seconds")
            self.logger.info(f"Recognition cooldown set to {cooldown} seconds")
            
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
        "About Face Recognition System",
        "Face Recognition System v2.0\n\n"  
        "Developed by: Nilay Naha\n"  
        "Email: niloynaha2003@gmail.com\n"  
        "Website: nilay-naha-portfolio.vercel.app\n\n"
        
        "Description:\n"
        "Advanced AI-powered face recognition\n"
        "with real-time detection and tracking\n\n"
        
        "Key Features:\n"
        "✓ Real-time face detection\n"
        "✓ 99% accuracy recognition\n"
        "✓ Confidence scoring\n"
        "✓ History tracking & export\n"
        "✓ Database logging\n"
        "✓ Multi-format support\n\n"
        
        "Technologies Used:\n"
        "• Python 3.10+\n"
        "• OpenCV 4.12\n"
        "• dlib 19.24\n"
        "• face_recognition library\n"
        "• Tkinter (GUI)\n"
        "• SQLite Database\n"
        "• NumPy & Pillow\n\n"
        
        "License: MIT\n"
        "© 2025 - 2030"
        )
        
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=f"  {message}")
        
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            # Save history before closing
            if self.recognition_history:
                self.save_history()
            
            self.stop_camera()
            self.logger.info("Application closed")
            self.root.destroy()
            
    def run(self):
        """Run the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    """Main entry point for GUI"""
    app = FaceRecognitionGUI()
    app.run()

if __name__ == "__main__":
    main()