# gui/image_window.py
import tkinter as tk
from tkinter import filedialog, messagebox, Frame, Canvas, Scrollbar, Text
import os
from PIL import Image, ImageTk
import numpy as np
import cv2
from .prediction import predict_multiple_digits
import time

class ImageRecognitionWindow:
    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        
        # Create new window with enhanced size
        self.window = tk.Toplevel(parent)
        self.window.title("Multi-Digit Recognition - Enhanced")
        self.window.geometry("600x750")  # Larger window for better UI
        self.window.resizable(True, True)
        self.window.minsize(500, 650)
        
        # Configure grid weights for responsive design
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Main content frame with better styling
        main_frame = tk.Frame(self.window, padx=20, pady=20, bg="#f8f9fa")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title header
        title_frame = tk.Frame(main_frame, bg="#f8f9fa")
        title_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        
        title_label = tk.Label(
            title_frame,
            text="🔍 Multi-Digit Recognition",
            font=("Arial", 16, "bold"),
            bg="#f8f9fa",
            fg="#2c3e50"
        )
        title_label.pack()
        
        # Image Display Area - Enhanced
        display_frame = tk.LabelFrame(
            main_frame, 
            text="Image Preview", 
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            fg="#34495e",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=15
        )
        display_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        display_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas for image with better styling
        canvas_frame = tk.Frame(display_frame, bg="#ffffff")
        canvas_frame.grid(row=0, column=0, pady=10)
        
        self.image_display = tk.Canvas(
            canvas_frame, 
            width=350, 
            height=280, 
            bg="#f7f8fa",
            relief=tk.SUNKEN,
            bd=2,
            highlightthickness=0
        )
        self.image_display.pack()
        
        # Enhanced Buttons with better styling
        button_frame = tk.Frame(display_frame, bg="#ffffff")
        button_frame.grid(row=1, column=0, pady=(10, 0))
        
        # Styled buttons with icons
        self.load_btn = tk.Button(
            button_frame, 
            text="📁 Load Image", 
            command=self.load_image, 
            width=15, 
            height=2,
            bg="#3498db", 
            fg="white",
            font=("Arial", 10, "bold"),
            relief=tk.RAISED,
            bd=2,
            cursor="hand2"
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.predict_btn = tk.Button(
            button_frame, 
            text="🔮 Predict", 
            command=self.predict, 
            width=15,
            height=2,
            bg="#27ae60", 
            fg="white",
            font=("Arial", 10, "bold"),
            relief=tk.RAISED,
            bd=2,
            cursor="hand2"
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.toggle_btn = tk.Button(
            button_frame, 
            text="👁️ Toggle View", 
            command=self.toggle_view, 
            width=15,
            height=2,
            bg="#9b59b6", 
            fg="white",
            font=("Arial", 10, "bold"),
            relief=tk.RAISED,
            bd=2,
            cursor="hand2"
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        # ENHANCED RESULTS AREA - Better handling for long sequences
        result_frame = tk.LabelFrame(
            main_frame,
            text="🎯 Recognition Results", 
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            fg="#34495e",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=15
        )
        result_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        result_frame.grid_columnconfigure(0, weight=1)
        
        # IMPROVED DIGIT SEQUENCE DISPLAY
        sequence_label = tk.Label(
            result_frame,
            text="Recognized Sequence:",
            font=("Arial", 11, "bold"),
            bg="#ffffff",
            anchor="w"
        )
        sequence_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # Enhanced scrollable text widget for long sequences
        sequence_container = tk.Frame(result_frame, bg="#ffffff")
        sequence_container.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        sequence_container.grid_columnconfigure(0, weight=1)
        
        # Text widget with scrollbar for unlimited digit display
        text_frame = tk.Frame(sequence_container)
        text_frame.grid(row=0, column=0, sticky="ew")
        text_frame.grid_columnconfigure(0, weight=1)
        
        self.result_text = tk.Text(
            text_frame,
            height=3,
            font=("Consolas", 28, "bold"),  # Monospace font for digits
            bg="#f8f9fa",
            fg="#2c3e50",
            relief=tk.SUNKEN,
            bd=2,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            state=tk.DISABLED,
            cursor="arrow"
        )
        self.result_text.grid(row=0, column=0, sticky="ew")
        
        # Horizontal scrollbar for very long sequences
        h_scrollbar = tk.Scrollbar(
            text_frame, 
            orient=tk.HORIZONTAL, 
            command=self.result_text.xview
        )
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.result_text.config(xscrollcommand=h_scrollbar.set)
        
        # Vertical scrollbar for wrapped text
        v_scrollbar = tk.Scrollbar(
            text_frame, 
            orient=tk.VERTICAL, 
            command=self.result_text.yview
        )
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.result_text.config(yscrollcommand=v_scrollbar.set)
        
        # Info panel with better layout
        info_frame = tk.Frame(result_frame, bg="#ffffff")
        info_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        info_frame.grid_columnconfigure(1, weight=1)
        
        # Confidence display with icon
        confidence_icon = tk.Label(
            info_frame, 
            text="📊", 
            font=("Arial", 14), 
            bg="#ffffff"
        )
        confidence_icon.grid(row=0, column=0, padx=(0, 5), sticky="w")
        
        self.confidence_label = tk.Label(
            info_frame, 
            text="Confidence: N/A", 
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            anchor="w"
        )
        self.confidence_label.grid(row=0, column=1, sticky="w")
        
        # Digit count with icon
        count_icon = tk.Label(
            info_frame, 
            text="🔢", 
            font=("Arial", 14), 
            bg="#ffffff"
        )
        count_icon.grid(row=1, column=0, padx=(0, 5), sticky="w")
        
        self.digit_count_label = tk.Label(
            info_frame,
            text="Digits detected: N/A",
            font=("Arial", 12),
            fg="#7f8c8d",
            bg="#ffffff",
            anchor="w"
        )
        self.digit_count_label.grid(row=1, column=1, sticky="w")
        
        # ENHANCED STATUS AREA
        status_frame = tk.LabelFrame(
            main_frame,
            text="📋 Processing Details", 
            font=("Arial", 12, "bold"),
            bg="#ffffff",
            fg="#34495e",
            relief=tk.RAISED,
            bd=2,
            padx=15,
            pady=15
        )
        status_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 0))
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_rowconfigure(0, weight=1)
        
        # Enhanced status display with scrollable text
        self.status_text = tk.Text(
            status_frame,
            height=8,
            font=("Arial", 10),
            bg="#f8f9fa",
            fg="#2c3e50",
            relief=tk.SUNKEN,
            bd=2,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.status_text.grid(row=0, column=0, sticky="nsew")
        
        # Status scrollbar
        status_scrollbar = tk.Scrollbar(
            status_frame, 
            orient=tk.VERTICAL, 
            command=self.status_text.yview
        )
        status_scrollbar.grid(row=0, column=1, sticky="ns")
        self.status_text.config(yscrollcommand=status_scrollbar.set)
        
        # Configure main frame row weights for expansion
        main_frame.grid_rowconfigure(3, weight=1)
        
        # Initialize image variables
        self.current_image = None
        self.photo_image = None
        self.original_image = None
        self.display_mode = "original"
        self.region_image = None
        
        # Reset UI to default state
        self.reset_ui()
        
        # Add hover effects to buttons
        self.add_button_hover_effects()
    
    def add_button_hover_effects(self):
        """Add hover effects to buttons for better UX"""
        def on_enter(event, color):
            event.widget['bg'] = color
            
        def on_leave(event, color):
            event.widget['bg'] = color
        
        # Load button hover
        self.load_btn.bind("<Enter>", lambda e: on_enter(e, "#2980b9"))
        self.load_btn.bind("<Leave>", lambda e: on_leave(e, "#3498db"))
        
        # Predict button hover
        self.predict_btn.bind("<Enter>", lambda e: on_enter(e, "#229954"))
        self.predict_btn.bind("<Leave>", lambda e: on_leave(e, "#27ae60"))
        
        # Toggle button hover
        self.toggle_btn.bind("<Enter>", lambda e: on_enter(e, "#8e44ad"))
        self.toggle_btn.bind("<Leave>", lambda e: on_leave(e, "#9b59b6"))
    
    def reset_ui(self):
        """Reset UI elements to default state"""
        # Clear result text
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, "? ? ?")
        self.result_text.config(state=tk.DISABLED)
        
        # Reset labels
        self.confidence_label.config(text="Confidence: N/A", fg="#34495e")
        self.digit_count_label.config(text="Digits detected: N/A", fg="#7f8c8d")
        
        # Clear status
        self.update_status("🚀 Ready to recognize digits! Load an image to begin the magic.")
    
    def update_status(self, message):
        """Update status text with better formatting"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        
        # Add timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.status_text.insert(tk.END, formatted_message)
        self.status_text.config(state=tk.DISABLED)
        self.status_text.see(tk.END)  # Auto-scroll to bottom
    
    def update_result_display(self, sequence, confidence_list, color_scheme="normal"):
        """Enhanced result display with color coding and formatting"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        # Format sequence with spaces for better readability
        if len(sequence) > 10:
            # Add spaces every 5 digits for very long sequences
            formatted_sequence = ' '.join([sequence[i:i+5] for i in range(0, len(sequence), 5)])
        else:
            # Add spaces between individual digits for shorter sequences
            formatted_sequence = ' '.join(sequence)
        
        self.result_text.insert(1.0, formatted_sequence)
        
        # Color coding based on confidence
        if color_scheme == "high":
            self.result_text.config(fg="#27ae60", bg="#d5f4e6")
        elif color_scheme == "medium":
            self.result_text.config(fg="#f39c12", bg="#fdf2e3")
        elif color_scheme == "low":
            self.result_text.config(fg="#e74c3c", bg="#fdeaea")
        else:
            self.result_text.config(fg="#2c3e50", bg="#f8f9fa")
        
        self.result_text.config(state=tk.DISABLED)
    
    def load_image(self):
        """Load image into the image window with enhanced feedback"""
        file_path = filedialog.askopenfilename(
            title="Select an Image with Digits",
            filetypes=[
                ("All Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("Bitmap files", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            self.update_status("❌ No image selected. Please choose an image file to continue.")
            return
        
        try:
            # Show loading state
            self.load_btn.config(state=tk.DISABLED, text="📁 Loading...")
            self.update_status("📂 Loading image... Please wait.")
            self.window.update()
            
            # Store original for processing
            self.original_image = Image.open(file_path).convert('L')
            
            # Resize for display
            self.display_image(self.original_image)
            
            filename = os.path.basename(file_path)
            self.update_status(f"✅ Successfully loaded: {filename}\n📏 Image size: {self.original_image.size}\n🎯 Click 'Predict' to recognize digits!")
            
            self.display_mode = "original"
            self.reset_ui()
            
            # Optional auto-predict after loading (commented out to prevent issues)
            # Uncomment the next line if you want automatic prediction after loading
            # self.window.after(1000, self.predict)
            
        except Exception as e:
            error_msg = f"❌ Error loading image: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Image Loading Error", f"Failed to load image:\n{str(e)}")
        finally:
            self.load_btn.config(state=tk.NORMAL, text="📁 Load Image")
    
    def display_image(self, img):
        """Display image in the canvas with improved scaling and centering"""
        try:
            # Enhanced display size
            canvas_w, canvas_h = 350, 280
            w, h = img.size
            
            # Calculate scale to fit image in canvas
            scale = min(canvas_w/w, canvas_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
            
            # Resize with high quality resampling
            img_display = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(img_display)
            
            # Clear and center the image
            self.image_display.delete("all")
            x = (canvas_w - new_w) // 2
            y = (canvas_h - new_h) // 2
            self.image_display.create_image(x, y, anchor=tk.NW, image=self.photo_image)
            
            # Add a subtle border
            self.image_display.create_rectangle(
                x-1, y-1, x+new_w+1, y+new_h+1, 
                outline="#bdc3c7", width=1
            )
            
            self.current_image = img
            
        except Exception as e:
            self.update_status(f"❌ Error displaying image: {str(e)}")
    
    def visualize_digit_regions(self, digit_regions):
        """Enhanced visualization with better bounding boxes and numbering"""
        if not digit_regions or not self.original_image:
            return
        
        try:
            # Create enhanced visualization
            img_np = np.array(self.original_image)
            if len(img_np.shape) == 2:
                display_img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            else:
                display_img = img_np.copy()
            
            # Enhanced bounding box colors and styling
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for i, (x, y, w, h) in enumerate(digit_regions):
                color = colors[i % len(colors)]
                
                # Draw thicker, more visible rectangle
                cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 3)
                
                # Enhanced number labeling
                label = str(i+1)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # Background for label
                cv2.rectangle(display_img, (x, y-25), (x+label_size[0]+10, y), color, -1)
                cv2.putText(display_img, label, (x+5, y-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Convert and store
            self.region_image = Image.fromarray(display_img)
            self.display_image(self.region_image)
            self.display_mode = "regions"
            
        except Exception as e:
            self.update_status(f"❌ Error visualizing regions: {str(e)}")
    
    def toggle_view(self):
        """Enhanced view toggle with better feedback"""
        if not self.original_image:
            self.update_status("⚠️ Please load an image first to toggle between views")
            return
        
        try:
            self.toggle_btn.config(state=tk.DISABLED)
            
            if self.display_mode == "original":
                if hasattr(self, 'region_image') and self.region_image:
                    self.display_image(self.region_image)
                    self.display_mode = "regions"
                    self.update_status("👁️ Switched to region visualization view (showing detected digit boundaries)")
                else:
                    # Generate regions if not available
                    from .prediction import localize_digits
                    digit_regions, _ = localize_digits(self.original_image)
                    if digit_regions:
                        self.visualize_digit_regions(digit_regions)
                        self.update_status("👁️ Generated and switched to region visualization view")
                    else:
                        self.update_status("⚠️ No digit regions detected to visualize. Try predicting first.")
            else:
                self.display_image(self.original_image)
                self.display_mode = "original"
                self.update_status("👁️ Switched back to original image view")
                
        except Exception as e:
            self.update_status(f"❌ Error toggling view: {str(e)}")
        finally:
            self.toggle_btn.config(state=tk.NORMAL)
    
    def predict(self):
        """Enhanced prediction with better UI feedback and unlimited digit support"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            self.update_status("⚠️ Please load an image first to start digit recognition")
            return
        
        # Enhanced loading state
        self.predict_btn.config(state=tk.DISABLED, text="🔮 Processing...")
        self.update_status("🔍 Analyzing image and detecting digits... This may take a moment.")
        
        # Show progress in result area
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, "Processing...")
        self.result_text.config(state=tk.DISABLED)
        
        self.window.update()
        
        try:
            from .model_loader import load_digit_recognition_model
            model = load_digit_recognition_model()
            
            if not model:
                self.update_status("❌ Model loading failed! Please check if the model file exists and is valid.")
                self.predict_btn.config(state=tk.NORMAL, text="🔮 Predict")
                return
            
            start_time = time.time()
            result = predict_multiple_digits(model, self.current_image)
            processing_time = time.time() - start_time
            
            if result["success"]:
                sequence = result["sequence"]
                confidences = result["confidence"]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Determine color scheme based on confidence
                if avg_confidence > 0.85:
                    color_scheme = "high"
                    confidence_icon = "🟢"
                elif avg_confidence > 0.65:
                    color_scheme = "medium"
                    confidence_icon = "🟡"
                else:
                    color_scheme = "low"
                    confidence_icon = "🔴"
                
                # Update result display with enhanced formatting
                self.update_result_display(sequence, confidences, color_scheme)
                
                # Update confidence and count labels
                self.confidence_label.config(
                    text=f"Average Confidence: {avg_confidence:.1%} {confidence_icon}",
                    fg="#27ae60" if avg_confidence > 0.85 else "#f39c12" if avg_confidence > 0.65 else "#e74c3c"
                )
                
                self.digit_count_label.config(
                    text=f"Total Digits: {result['count']} digits detected",
                    fg="#27ae60" if result['count'] > 0 else "#e74c3c"
                )
                
                # Create comprehensive status report
                status_lines = [
                    f"🎉 Recognition Complete!",
                    f"📊 Sequence: {sequence}",
                    f"📈 Digits Found: {result['count']}",
                    f"⏱️ Processing Time: {processing_time:.2f} seconds",
                    f"🎯 Average Confidence: {avg_confidence:.1%}",
                    "",
                    "📋 Individual Digit Analysis:"
                ]
                
                for i, (digit, conf) in enumerate(zip(result["digits"], confidences)):
                    confidence_level = "🟢 High" if conf > 0.85 else "🟡 Medium" if conf > 0.65 else "🔴 Low"
                    status_lines.append(f"   Position {i+1}: '{digit}' - {conf:.1%} ({confidence_level})")
                
                if len(sequence) > 20:
                    status_lines.extend([
                        "",
                        "💡 Long sequence detected! Use the scrollbars above to view all digits.",
                        f"📏 Total length: {len(sequence)} characters"
                    ])
                
                self.update_status("\n".join(status_lines))
                
                # Visualize detected regions
                if result.get("regions"):
                    self.visualize_digit_regions(result["regions"])
                
            else:
                error_msg = result.get('error', 'Unknown error occurred')
                self.update_status(f"❌ Recognition failed: {error_msg}")
                self.reset_ui()
                
        except Exception as e:
            error_msg = f"💥 Unexpected error during prediction: {str(e)}"
            self.update_status(error_msg)
            self.reset_ui()
        finally:
            self.predict_btn.config(state=tk.NORMAL, text="🔮 Predict")