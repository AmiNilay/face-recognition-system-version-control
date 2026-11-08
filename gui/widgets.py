"""
Custom UI Widgets Module

Provides reusable, modular Tkinter UI components for the Face Recognition GUI:
- StatusBar: Bottom-docked status indicator.
- FaceCard: Information and thumbnail display card for enrolled individuals.
- VideoDisplay: Dark-canvas video container.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
from pathlib import Path

class StatusBar(tk.Frame):
    """
    Bottom-docked sunken status bar displaying current operational status.
    """
    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, **kwargs)
        self.label = tk.Label(
            self,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg="#1e1e1e",
            fg="#00ff00",
            font=("Arial", 10),
            padx=10,
            pady=3
        )
        self.label.pack(fill=tk.X)

    def set_text(self, text: str):
        """Update status bar message."""
        self.label.config(text=text)

    def clear(self):
        """Clear status bar text."""
        self.label.config(text="")


class FaceCard(tk.Frame):
    """
    Card widget displaying a person's name and their portrait thumbnail.
    """
    def __init__(self, parent: tk.Widget, name: str, image_path: Optional[str] = None, **kwargs):
        super().__init__(parent, bg="#1e1e1e", relief="raised", bd=2, **kwargs)
        self.name = name
        self.image_path = image_path

        # Name title label
        self.name_label = tk.Label(
            self,
            text=name,
            font=("Arial", 11, "bold"),
            bg="#1e1e1e",
            fg="white"
        )
        self.name_label.pack(pady=5)

        # Image thumbnail
        if image_path and Path(image_path).is_file():
            try:
                from PIL import Image, ImageTk
                with Image.open(image_path) as img:
                    img.thumbnail((100, 100))
                    self.photo = ImageTk.PhotoImage(img)

                self.img_label = tk.Label(self, image=self.photo, bg="#1e1e1e")
                self.img_label.pack(padx=5, pady=5)
            except Exception:
                pass


class VideoDisplay(tk.Label):
    """
    Custom video canvas widget pre-configured with black background.
    """
    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, bg="#000000", **kwargs)
