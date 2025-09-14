import tkinter as tk
from tkinter import Canvas, Button, Label
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os

# --- دالة لإصلاح مشكلة عرض الأحرف العربية في ويندوز ---
def arabic_text(text):
    try:
        return text.encode('cp1256').decode('utf-8')
    except:
        return text

# --- تحميل النموذج ---
model_path = 'handwriting_model.h5'
if not os.path.exists(model_path):
    print(f"خطأ: ملف النموذج غير موجود: {model_path}")
    exit()

model = tf.keras.models.load_model(model_path)

# --- إعدادات النافذة ---
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 450
CANVAS_SIZE = 280

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("برنامج التعرف على الكتابة اليدوية")

        # Canvas للرسم
        self.canvas = Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack(pady=10)

        # أزرار
        self.btn_clear = Button(self.root, text="مسح", command=self.clear_canvas)
        self.btn_clear.pack(side="left", padx=20, pady=10)

        self.btn_predict = Button(self.root, text="تعرّف", command=self.predict)
        self.btn_predict.pack(side="right", padx=20, pady=10)

        # لعرض النتيجة
        self.label_result = Label(self.root, text="النتيجة: ", font=("Arial", 16))
        self.label_result.pack(pady=10)

        # أحداث الرسم
        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8  # نصف قطر الفرشاة
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.label_result.config(text="النتيجة: ")

    def predict(self):
        # حفظ الرسمة كصورة
        self.canvas.update()
        self.canvas.postscript(file="drawing.ps", colormode="color")

        # تحويلها لصورة PIL
        img = Image.open("drawing.ps")
        img = img.convert("L")  # تدرج رمادي
        img = ImageOps.invert(img)  # عكس الألوان (أبيض→أسود)
        img = img.resize((28, 28))  # نفس حجم بيانات MNIST أو النموذج
        img = np.array(img).astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        # التنبؤ
        prediction = model.predict(img)
        digit = np.argmax(prediction)

        self.label_result.config(text=f"النتيجة: {digit}")

# --- تشغيل البرنامج ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()