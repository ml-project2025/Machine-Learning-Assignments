
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from predict import DigitPredictor

class DigitRecognitionGUI:
    """
    واجهة رسومية للتعرف على الأرقام والحروف، تدعم الرسم على الكانفاس أو تحميل الصور.
    """
    def __init__(self, model_path='./saved_models/best_digit_model.pth'):
        self.root = tk.Tk()
        self.root.title("Digit & Letter Recognition")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # تحميل النموذج
        try:
            self.predictor = DigitPredictor(model_path)
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        
        # متغيرات
        self.current_image = None
        self.canvas_image = None
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.draw_canvas = None
        self.canvas_width = 280
        self.canvas_height = 280
        
        self.setup_ui()
    
    def setup_ui(self):
        # عنوان
        ttk.Label(self.root, text="Digit & Letter Recognition System", font=('Arial', 20, 'bold')).pack(pady=10)
        
        # أزرار
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Draw Digit/Letter", command=self.open_drawing_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # عرض الصورة
        image_frame = ttk.LabelFrame(self.root, text="Image Display")
        image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        self.image_canvas = tk.Canvas(image_frame, bg='white', height=350)
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # عرض النتائج
        result_frame = ttk.LabelFrame(self.root, text="Prediction Results")
        result_frame.pack(pady=10, padx=20, fill=tk.X)
        self.result_var = tk.StringVar(value="No prediction yet")
        ttk.Label(result_frame, textvariable=self.result_var, font=('Arial', 16, 'bold'), foreground='blue').pack(pady=10)
        
        # شريط الحالة
        self.status_var = tk.StringVar(value="Model ready" if self.model_loaded else "Model loading failed")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
    
    # ---------------------- تحميل الصور ----------------------
    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image",
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    messagebox.showerror("Error", "Could not load the image file")
                    return
                self.display_image(self.current_image)
                self.predict_from_file(file_path)
                self.status_var.set(f"Image loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        if image is None:
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        h, w = image_rgb.shape[:2]
        scale = min(canvas_width/w, canvas_height/h, 1.0)
        resized = cv2.resize(image_rgb, (int(w*scale), int(h*scale)))
        pil_image = Image.fromarray(resized)
        self.canvas_image = ImageTk.PhotoImage(pil_image)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_width//2, canvas_height//2, image=self.canvas_image)
    
    def predict_from_file(self, file_path):
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded")
            return
        try:
            self.status_var.set("Predicting...")
            self.root.update()
            label, confidence, _ = self.predictor.predict_single_digit(file_path)
            self.result_var.set(f"Predicted: {label}\nConfidence: {confidence:.3f}")
            self.status_var.set("Prediction completed")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Prediction failed")
    
    # ---------------------- الرسم على الكانفاس ----------------------
    def open_drawing_window(self):
        drawing_window = tk.Toplevel(self.root)
        drawing_window.title("Draw Digit or Letter")
        drawing_window.geometry("400x550")
        drawing_window.configure(bg='#f0f0f0')
        drawing_window.grab_set()
        
        ttk.Label(drawing_window, text="Draw a single digit (0-9) or letter (A-Z)", font=('Arial', 12)).pack(pady=10)
        
        # أزرار الكانفاس
        button_frame = ttk.Frame(drawing_window)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Clear", command=lambda: self.clear_canvas(self.draw_canvas)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Predict", command=self.predict_from_canvas).pack(side=tk.LEFT, padx=5)
        
        # كانفاس الرسم
        canvas_frame = ttk.LabelFrame(drawing_window, text="Drawing Area")
        canvas_frame.pack(pady=10, padx=20)
        self.draw_canvas = tk.Canvas(canvas_frame, bg='white', width=self.canvas_width, height=self.canvas_height,
                                     cursor='pencil', highlightthickness=2, highlightbackground='black')
        self.draw_canvas.pack(padx=10, pady=10)
        self.draw_canvas.bind('<Button-1>', self.start_drawing)
        self.draw_canvas.bind('<B1-Motion>', self.draw)
        self.draw_canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # عرض نتيجة الرسم
        result_frame = ttk.LabelFrame(drawing_window, text="Prediction Result")
        result_frame.pack(pady=10, padx=20, fill=tk.X)
        self.canvas_result_var = tk.StringVar(value="Draw and click Predict")
        ttk.Label(result_frame, textvariable=self.canvas_result_var, font=('Arial', 14, 'bold'), foreground='green').pack(pady=10)
    
    def start_drawing(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
    
    def draw(self, event):
        if self.drawing:
            self.draw_canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                         width=12, fill='black', capstyle=tk.ROUND, smooth=True, splinesteps=36)
            self.last_x, self.last_y = event.x, event.y
    
    def stop_drawing(self, event):
        self.drawing = False
    
    def clear_canvas(self, canvas):
        if canvas:
            canvas.delete("all")
            self.canvas_result_var.set("Draw and click Predict")
    
    # ---------------------- التنبؤ من الرسم ----------------------
    def predict_from_canvas(self):
        if not self.model_loaded or not self.draw_canvas:
            messagebox.showerror("Error", "Model not loaded or canvas unavailable")
            return
        try:
            image_array = self.canvas_to_numpy()
            if image_array is None:
                self.canvas_result_var.set("Error: Could not process canvas")
                return
            
            label, confidence, probabilities = self.predictor.predict_from_array(image_array)
            self.canvas_result_var.set(f"Predicted: {label}\nConfidence: {confidence:.3f}")
            
            print("Top 3 predictions:")
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            for i, idx in enumerate(top_3_indices):
                print(f"  {i+1}. Label {self.predictor.index_to_label(idx)}: {probabilities[idx]:.3f}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            print(f"Canvas prediction error: {e}")
    
    def canvas_to_numpy(self):
        """تحويل الكانفاس إلى numpy array 28x28"""
        try:
            image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
            draw = ImageDraw.Draw(image)
            for item in self.draw_canvas.find_all():
                coords = self.draw_canvas.coords(item)
                if len(coords) >= 4:
                    for i in range(0, len(coords)-2, 2):
                        draw.line([coords[i], coords[i+1], coords[i+2], coords[i+3]], fill='black', width=12)
            arr = 255 - np.array(image)
            _, arr = cv2.threshold(arr, 50, 255, cv2.THRESH_BINARY)
            coords = np.where(arr > 0)
            if len(coords[0]) == 0:
                return None
            top, left, bottom, right = np.min(coords[0]), np.min(coords[1]), np.max(coords[0]), np.max(coords[1])
            cropped = arr[top:bottom+1, left:right+1]
            size = max(cropped.shape)
            square = np.zeros((size, size), dtype=np.uint8)
            y_offset, x_offset = (size - cropped.shape[0])//2, (size - cropped.shape[1])//2
            square[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
            final_image = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
            final_image = final_image.astype(np.float32)/255.0  # normalization
            final_image = final_image[np.newaxis, np.newaxis, :, :]  # إضافة batch و channel
            return final_image
        except Exception as e:
            print(f"Canvas to numpy conversion error: {e}")
            return None
    
    # ---------------------- مسح كل البيانات ----------------------
    def clear_all(self):
        self.current_image = None
        self.canvas_image = None
        self.image_canvas.delete("all")
        self.result_var.set("No prediction yet")
        self.status_var.set("Data cleared")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DigitRecognitionGUI()
    app.run()



# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox
# import cv2
# import numpy as np
# from PIL import Image, ImageTk, ImageDraw, ImageGrab
# import os
# from predict import ImprovedDigitPredictor

# class EnhancedDigitGUI:
#     """
#     واجهة رسومية محسنة للتعرف على الأرقام
#     """
#     def __init__(self, model_path='./saved_models/best_digit_model.pth'):
#         self.root = tk.Tk()
#         self.root.title("نظام التعرف على الأرقام المتطور")
#         self.root.geometry("900x700")
#         self.root.configure(bg='#f8f9fa')
        
#         # تحميل النموذج
#         try:
#             self.predictor = ImprovedDigitPredictor(model_path)
#             self.model_loaded = True
#         except Exception as e:
#             self.model_loaded = False
#             print(f"خطأ في تحميل النموذج: {e}")
        
#         # متغيرات الواجهة
#         self.current_image = None
#         self.drawing = False
#         self.last_x = None
#         self.last_y = None
#         self.draw_canvas = None
        
#         self.setup_ui()
        
#         # تحديد اللون والخط
#         self.pen_color = "black"
#         self.pen_width = 8
    
#     def setup_ui(self):
#         """إعداد الواجهة الرسومية المحسنة"""
        
#         # إطار العنوان
#         title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
#         title_frame.pack(fill=tk.X, padx=10, pady=5)
#         title_frame.pack_propagate(False)
        
#         title_label = tk.Label(title_frame, text="🔢 نظام التعرف على الأرقام المتطور", 
#                               font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
#         title_label.pack(expand=True)
        
#         # إطار الأزرار الرئيسية
#         main_buttons_frame = tk.Frame(self.root, bg='#ecf0f1', height=60)
#         main_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
#         main_buttons_frame.pack_propagate(False)
        
#         # الأزرار الرئيسية
#         tk.Button(main_buttons_frame, text="📁 تحميل صورة", font=('Arial', 12, 'bold'),
#                  bg='#3498db', fg='white', relief=tk.RAISED, bd=3,
#                  command=self.load_image).pack(side=tk.LEFT, padx=10, pady=10)
        
#         tk.Button(main_buttons_frame, text="✏️ رسم رقم", font=('Arial', 12, 'bold'),
#                  bg='#2ecc71', fg='white', relief=tk.RAISED, bd=3,
#                  command=self.open_drawing_mode).pack(side=tk.LEFT, padx=10, pady=10)
        
#         tk.Button(main_buttons_frame, text="🧹 مسح الكل", font=('Arial', 12, 'bold'),
#                  bg='#e74c3c', fg='white', relief=tk.RAISED, bd=3,
#                  command=self.clear_all).pack(side=tk.LEFT, padx=10, pady=10)
        
#         # إطار المحتوى الرئيسي
#         content_frame = tk.Frame(self.root, bg='#f8f9fa')
#         content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
#         # إطار عرض الصورة (الجانب الأيسر)
#         image_frame = tk.LabelFrame(content_frame, text="📷 الصورة المحملة", 
#                                    font=('Arial', 12, 'bold'), bg='#ffffff', relief=tk.RAISED, bd=2)
#         image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
#         # كانفاس عرض الصورة
#         self.image_canvas = tk.Canvas(image_frame, bg='white', relief=tk.SUNKEN, bd=2)
#         self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
#         # إطار النتائج (الجانب الأيمن)
#         results_frame = tk.LabelFrame(content_frame, text="🎯 النتائج", 
#                                      font=('Arial', 12, 'bold'), bg='#ffffff', relief=tk.RAISED, bd=2)
#         results_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
#         results_frame.configure(width=300)
#         results_frame.pack_propagate(False)
        
#         # منطقة عرض النتائج
#         self.results_text = tk.Text(results_frame, font=('Courier', 11), bg='#f8f9fa', 
#                                    relief=tk.SUNKEN, bd=2, wrap=tk.WORD)
#         self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
#         # إدراج النص الافتراضي
#         self.update_results("🔍 اختر صورة أو ارسم رقماً للبدء...")
        
#         # شريط الحالة
#         status_frame = tk.Frame(self.root, bg='#34495e', height=30)
#         status_frame.pack(fill=tk.X, side=tk.BOTTOM)
#         status_frame.pack_propagate(False)
        
#         self.status_var = tk.StringVar()
#         if self.model_loaded:
#             self.status_var.set("✅ النموذج جاهز للاستخدام")
#         else:
#             self.status_var.set("❌ فشل في تحميل النموذج")
        
#         status_label = tk.Label(status_frame, textvariable=self.status_var, 
#                                fg='white', bg='#34495e', font=('Arial', 10))
#         status_label.pack(side=tk.LEFT, padx=10, pady=5)
    
#     def update_results(self, text):
#         """تحديث منطقة النتائج"""
#         self.results_text.delete(1.0, tk.END)
#         self.results_text.insert(tk.END, text)
    
#     def load_image(self):
#         """تحميل صورة من النظام"""
#         file_path = filedialog.askopenfilename(
#             title="اختر صورة للتعرف على الأرقام",
#             filetypes=[
#                 ("جميع الصور", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
#                 ("PNG files", "*.png"),
#                 ("JPEG files", "*.jpg *.jpeg"),
#                 ("جميع الملفات", "*.*")
#             ]
#         )
        
#         if file_path:
#             try:
#                 # قراءة وعرض الصورة
#                 self.current_image = cv2.imread(file_path)
#                 if self.current_image is not None:
#                     self.display_image(self.current_image)
#                     self.predict_from_file(file_path)
#                     self.status_var.set(f"✅ تم تحميل: {os.path.basename(file_path)}")
#                 else:
#                     messagebox.showerror("خطأ", "لا يمكن قراءة الصورة المحددة")
                    
#             except Exception as e:
#                 messagebox.showerror("خطأ", f"فشل في تحميل الصورة:\n{str(e)}")
    
#     def display_image(self, image):
#         """عرض الصورة في الكانفاس"""
#         if image is None:
#             return
        
#         try:
#             # تحويل من BGR إلى RGB
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
#             # تغيير حجم الصورة لتناسب الكانفاس
#             canvas_width = self.image_canvas.winfo_width()
#             canvas_height = self.image_canvas.winfo_height()
            
#             if canvas_width > 1 and canvas_height > 1:
#                 # حساب نسبة التغيير
#                 h, w = image_rgb.shape[:2]
#                 scale = min(canvas_width/w, canvas_height/h) * 0.9  # هامش 10%
#                 new_width = int(w * scale)
#                 new_height = int(h * scale)
                
#                 # تغيير الحجم
#                 resized_image = cv2.resize(image_rgb, (new_width, new_height))
                
#                 # تحويل إلى PIL وعرض
#                 pil_image = Image.fromarray(resized_image)
#                 self.photo = ImageTk.PhotoImage(pil_image)
                
#                 self.image_canvas.delete("all")
#                 self.image_canvas.create_image(canvas_width//2, canvas_height//2, 
#                                              image=self.photo)
#         except Exception as e:
#             print(f"خطأ في عرض الصورة: {e}")
    
#     def predict_from_file(self, file_path):
#         """التنبؤ من ملف صورة"""
#         if not self.model_loaded:
#             self.update_results("❌ النموذج غير محمل")
#             return
        
#         try:
#             self.status_var.set("🔄 جاري التحليل...")
#             self.root.update()
            
#             # محاولة التنبؤ برقم واحد أولاً
#             single_result = self.predictor.predict_digit(file_path, show_details=False)
            
#             if single_result and single_result['is_confident']:
#                 # عرض نتيجة الرقم الواحد
#                 result_text = f"""📋 نتائج التعرف:

# 🔹 الرقم المكتشف: {single_result['digit']}

# 🔹 بالعربية: {single_result['digit_name']}

# 🔹 مستوى الثقة: {single_result['confidence']:.1%}

# 🔹 عناصر مكتشفة: 1

# ━━━━━━━━━━━━━━━━━━━━━━━━━

# 📊 توزيع الاحتماليات:
# """
                
#                 # إضافة أعلى 3 احتماليات
#                 top3_indices = np.argsort(single_result['probabilities'])[::-1][:3]
#                 for i, idx in enumerate(top3_indices):
#                     prob = single_result['probabilities'][idx]
#                     result_text += f"  {i+1}. الرقم {idx}: {prob:.1%}\n"
                
#                 if single_result['confidence'] > 0.9:
#                     result_text += "\n✅ ثقة عالية جداً"
#                 elif single_result['confidence'] > 0.7:
#                     result_text += "\n✅ ثقة جيدة"
#                 else:
#                     result_text += "\n⚠️ ثقة متوسطة"
                
#                 self.update_results(result_text)
#                 self.status_var.set("✅ تم التعرف بنجاح")
                
#             else:
#                 # محاولة التنبؤ بأرقام متعددة
#                 multiple_results = self.predictor.segment_and_predict_multiple(file_path)
                
#                 if multiple_results:
#                     digits = [str(r['digit']) for r in multiple_results]
#                     number = ''.join(digits)
#                     avg_confidence = np.mean([r['confidence'] for r in multiple_results])
                    
#                     result_text = f"""📋 نتائج التعرف:

# 🔹 الرقم المكتشف: {number}

# 🔹 عدد الخانات: {len(digits)}

# 🔹 متوسط الثقة: {avg_confidence:.1%}

# 🔹 عناصر مكتشفة: {len(multiple_results)}

# ━━━━━━━━━━━━━━━━━━━━━━━━━

# 📊 تفاصيل كل رقم:
# """
                    
#                     for i, result in enumerate(multiple_results):
#                         result_text += f"  {i+1}. {result['digit']} ({result['digit_name']}) - {result['confidence']:.1%}\n"
                    
#                     if avg_confidence > 0.8:
#                         result_text += "\n✅ ثقة عالية"
#                     elif avg_confidence > 0.6:
#                         result_text += "\n✅ ثقة جيدة"
#                     else:
#                         result_text += "\n⚠️ ثقة متوسطة"
                    
#                     self.update_results(result_text)
#                     self.status_var.set("✅ تم التعرف بنجاح")
                    
#                 else:
#                     # لم يتم العثور على أرقام
#                     result_text = """❌ لم يتم التعرف على أي أرقام

# يرجى المحاولة مع صورة أوضح.

# 💡 نصائح للحصول على نتائج أفضل:

# • استخدم صورة عالية الجودة
# • تأكد من وضوح الأرقام
# • استخدم خلفية بيضاء وأرقام سوداء
# • تجنب الضوضاء والتشويش
# • اجعل الأرقام بحجم مناسب
# • تأكد من الإضاءة الجيدة

# 🔄 جرب مرة أخرى"""
                    
#                     self.update_results(result_text)
#                     self.status_var.set("⚠️ لم يتم العثور على أرقام")
                    
#         except Exception as e:
#             error_text = f"❌ خطأ في التحليل:\n\n{str(e)}\n\nيرجى المحاولة مع صورة أخرى."
#             self.update_results(error_text)
#             self.status_var.set("❌ فشل في التحليل")
    
#     def open_drawing_mode(self):
#         """فتح نافذة الرسم"""
#         drawing_window = tk.Toplevel(self.root)
#         drawing_window.title("✏️ ارسم رقماً")
#         drawing_window.geometry("500x600")
#         drawing_window.configure(bg='#ecf0f1')
#         drawing_window.resizable(False, False)
        
#         # إطار العنوان
#         title_frame = tk.Frame(drawing_window, bg='#34495e', height=60)
#         title_frame.pack(fill=tk.X)
#         title_frame.pack_propagate(False)
        
#         title_label = tk.Label(title_frame, text="✏️ ارسم رقماً بالماوس", 
#                               font=('Arial', 16, 'bold'), fg='white', bg='#34495e')
#         title_label.pack(expand=True)
        
#         # إطار أدوات الرسم
#         tools_frame = tk.Frame(drawing_window, bg='#bdc3c7', height=50)
#         tools_frame.pack(fill=tk.X, padx=5, pady=5)
#         tools_frame.pack_propagate(False)
        
#         tk.Button(tools_frame, text="🧹 مسح", font=('Arial', 10, 'bold'),
#                  bg='#e74c3c', fg='white', relief=tk.RAISED,
#                  command=lambda: self.clear_drawing_canvas(draw_canvas)).pack(side=tk.LEFT, padx=5, pady=5)
        
#         tk.Button(tools_frame, text="🔍 تحليل", font=('Arial', 10, 'bold'),
#                  bg='#27ae60', fg='white', relief=tk.RAISED,
#                  command=lambda: self.predict_from_drawing(draw_canvas, result_label)).pack(side=tk.LEFT, padx=5, pady=5)
        
#         # إطار الكانفاس
#         canvas_frame = tk.Frame(drawing_window, bg='#ecf0f1')
#         canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
#         # كانفاس الرسم
#         draw_canvas = tk.Canvas(canvas_frame, bg='white', width=400, height=300, 
#                                cursor='pencil', relief=tk.SUNKEN, bd=3)
#         draw_canvas.pack(pady=10)
        
#         # ربط أحداث الماوس
#         draw_canvas.bind('<Button-1>', lambda e: self.start_drawing(e, draw_canvas))
#         draw_canvas.bind('<B1-Motion>', lambda e: self.draw_on_canvas(e, draw_canvas))
#         draw_canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
#         # إطار النتيجة
#         result_frame = tk.LabelFrame(canvas_frame, text="🎯 النتيجة", 
#                                     font=('Arial', 12, 'bold'), bg='#ffffff')
#         result_frame.pack(fill=tk.X, pady=10)
        
#         result_label = tk.Label(result_frame, text="ارسم رقماً واضغط 'تحليل'", 
#                                font=('Arial', 12), bg='#ffffff', fg='#2c3e50',
#                                wraplength=350, justify=tk.CENTER)
#         result_label.pack(pady=15)
        
#         # تعليمات
#         instructions = tk.Label(canvas_frame, 
#                                text="💡 ارسم رقماً واضحاً بخط كبير\n🖱️ استخدم الماوس للرسم", 
#                                font=('Arial', 10), bg='#ecf0f1', fg='#7f8c8d')
#         instructions.pack(pady=5)
    
#     def start_drawing(self, event, canvas):
#         """بدء الرسم"""
#         self.drawing = True
#         self.last_x = event.x
#         self.last_y = event.y
    
#     def draw_on_canvas(self, event, canvas):
#         """الرسم على الكانفاس"""
#         if self.drawing:
#             canvas.create_line(self.last_x, self.last_y, event.x, event.y, 
#                              width=self.pen_width, fill=self.pen_color, 
#                              capstyle=tk.ROUND, smooth=tk.TRUE)
#             self.last_x = event.x
#             self.last_y = event.y
    
#     def stop_drawing(self, event):
#         """إيقاف الرسم"""
#         self.drawing = False
    
#     def clear_drawing_canvas(self, canvas):
#         """مسح كانفاس الرسم"""
#         canvas.delete("all")
    
#     def predict_from_drawing(self, canvas, result_label):
#         """التنبؤ من الرسم"""
#         if not self.model_loaded:
#             result_label.config(text="❌ النموذج غير محمل")
#             return
        
#         try:
#             # حفظ الكانفاس كصورة
#             canvas.update()
            
#             # الحصول على إحداثيات الكانفاس
#             x = canvas.winfo_rootx()
#             y = canvas.winfo_rooty()
#             x1 = x + canvas.winfo_width()
#             y1 = y + canvas.winfo_height()
            
#             # التقاط الشاشة
#             screenshot = ImageGrab.grab().crop((x, y, x1, y1))
            
#             # تحويل إلى رمادي
#             screenshot = screenshot.convert('L')
            
#             # تحويل إلى numpy array
#             image_array = np.array(screenshot)
            
#             # التنبؤ
#             result = self.predictor.predict_digit(image_array, show_details=False)
            
#             if result and result['confidence'] > 0.3:  # حد أدنى للثقة أقل للرسم اليدوي
#                 result_text = f"""🎯 النتيجة:

# الرقم: {result['digit']} ({result['digit_name']})
# الثقة: {result['confidence']:.1%}

# {('✅ ممتاز!' if result['confidence'] > 0.8 else 
#   '👍 جيد!' if result['confidence'] > 0.6 else 
#   '⚠️ متوسط')}"""
#                 result_label.config(text=result_text, fg='#27ae60')
#             else:
#                 result_label.config(text="❌ لم يتم التعرف على الرقم\n\nجرب رسم رقم أوضح", 
#                                    fg='#e74c3c')
            
#         except Exception as e:
#             result_label.config(text=f"❌ خطأ في التحليل:\n{str(e)}", fg='#e74c3c')
    
#     def clear_all(self):
#         """مسح جميع البيانات"""
#         self.current_image = None
#         self.image_canvas.delete("all")
#         self.update_results("🔍 اختر صورة أو ارسم رقماً للبدء...")
#         self.status_var.set("🧹 تم مسح البيانات")
    
#     def run(self):
#         """تشغيل الواجهة الرسومية"""
#         self.root.mainloop()

# # للتوافق مع الكود السابق
# DigitRecognitionGUI = EnhancedDigitGUI






# import torch.nn as nn
# import torch.nn.functional as F

# class DigitCNN(nn.Module):
#     """
#     شبكة عصبية تلافيفية للتعرف على الأرقام
#     """
#     def __init__(self, num_classes=10):
#         super(DigitCNN, self).__init__()
        
#         # الطبقات التلافيفية الأولى
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
#         # طبقات التجميع
#         self.pool = nn.MaxPool2d(2, 2)
        
#         # طبقات التسوية
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
        
#         # الطبقات المكتملة الاتصال
#         self.fc1 = nn.Linear(128 * 3 * 3, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, num_classes)
        
#         # طبقة التطبيع المجموعي
#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
        
#     def forward(self, x):
#         # الطبقة التلافيفية الأولى
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
#         # الطبقة التلافيفية الثانية
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
#         # الطبقة التلافيفية الثالثة
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
#         # تطبيق Dropout
#         x = self.dropout1(x)
        
#         # تحويل إلى بعد واحد
#         x = x.view(x.size(0), -1)
        
#         # الطبقات المكتملة الاتصال
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
        
#         return x

# class ImprovedDigitCNN(nn.Module):
#     """
#     نسخة محسنة من الشبكة مع ResNet blocks
#     """
#     def __init__(self, num_classes=10):
#         super(ImprovedDigitCNN, self).__init__()
        
#         self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
        
#         self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
        
#         self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
        
#         self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
#         self.bn4 = nn.BatchNorm2d(512)
        
#         self.pool = nn.MaxPool2d(2, 2)
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(512, num_classes)
        
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = F.relu(self.bn4(self.conv4(x)))
        
#         x = self.adaptive_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         x = self.fc(x)
        
#         return x



