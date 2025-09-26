# gui/main.py
import tkinter as tk
from .ui_components import create_status_bar
from .drawing_canvas import DrawingCanvas
from .image_window import ImageRecognitionWindow

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.geometry("350x450")
        self.root.resizable(False, False)
        
        # Create status bar FIRST (critical fix)
        self.status_bar = create_status_bar(self.root)
        
        # Create main UI components AFTER status bar
        self.drawing_canvas = DrawingCanvas(self.root, self)
        
        # Initialize windows
        self.image_window = None
    
    def open_image_window(self):
        """Open separate window for image recognition"""
        if self.image_window is None or not hasattr(self.image_window, 'window') or not self.image_window.window.winfo_exists():
            self.image_window = ImageRecognitionWindow(self.root, self)
        else:
            self.image_window.window.lift()

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()