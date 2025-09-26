# gui/ui_components.py
import tkinter as tk

class StatusBar:
    def __init__(self, parent):
        self.frame = tk.Frame(parent, bd=1, relief=tk.SUNKEN, height=20)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.label = tk.Label(
            self.frame,
            text="Draw a digit and click Predict",
            bd=1,
            anchor=tk.W,
            font=("Arial", 10),
            fg="#616161"
        )
        self.label.pack(fill=tk.X)
    
    def set(self, text, status_type="normal"):
        """Set status bar text with appropriate color"""
        self.label.config(text=text)
        
        if status_type == "success":
            self.label.config(fg="#388E3C")
        elif status_type == "warning":
            self.label.config(fg="#F57C00")
        elif status_type == "error":
            self.label.config(fg="#D32F2F")
        else:
            self.label.config(fg="#616161")

def create_status_bar(parent):
    """Create and return a status bar widget"""
    return StatusBar(parent)