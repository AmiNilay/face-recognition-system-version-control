"""Custom Widgets for Face Recognition GUI"""

import tkinter as tk
from tkinter import ttk

class StatusBar(tk.Frame):
    """Custom status bar widget"""
    
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.label = tk.Label(self, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.label.pack(fill=tk.X)
        
    def set_text(self, text):
        """Set status bar text"""
        self.label.config(text=text)
        
    def clear(self):
        """Clear status bar"""
        self.label.config(text="")

class FaceCard(tk.Frame):
    """Widget to display face information"""
    
    def __init__(self, parent, name, image_path=None):
        tk.Frame.__init__(self, parent, bg="#1e1e1e", relief="raised", bd=2)
        
        self.name = name
        self.image_path = image_path
        
        # Name label
        self.name_label = tk.Label(
            self,
            text=name,
            font=("Arial", 11, "bold"),
            bg="#1e1e1e",
            fg="white"
        )
        self.name_label.pack(pady=5)
        
        # Image display (if provided)
        if image_path:
            try:
                from PIL import Image, ImageTk
                img = Image.open(image_path)
                img.thumbnail((100, 100))
                photo = ImageTk.PhotoImage(img)
                
                img_label = tk.Label(self, image=photo, bg="#1e1e1e")
                img_label.image = photo
                img_label.pack()
            except:
                pass

class VideoDisplay(tk.Label):
    """Custom video display widget"""
    
    def __init__(self, parent, **kwargs):
        tk.Label.__init__(self, parent, **kwargs)
        self.configure(bg="#000000")
        
    def update_frame(self, frame):
        """Update displayed frame"""
        from PIL import Image, ImageTk
        
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        
        self.config(image=photo)
        self.image = photo