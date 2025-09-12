import tkinter as tk
from PIL import Image, ImageDraw
import os
import time

class DigitDrawer:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Drawer 28x28")
        self.canvas_size = 280  # حجم اللوحة الكبير للرسم
        self.canvas = tk.Canvas(master, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        # إنشاء صورة PIL للرسم عليها
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # أزرار التحكم
        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack()
        self.save_btn = tk.Button(self.btn_frame, text="Save", command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.clear_btn = tk.Button(self.btn_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # متابعة حركة الماوس للرسم
        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x, self.last_y = None, None

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # رسم خط على اللوحة
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=15, fill="black", capstyle=tk.ROUND, smooth=True)
            # رسم نفس الخط على صورة PIL
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=15)
        self.last_x, self.last_y = x, y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

    def save_image(self):
        # تحويل الصورة إلى 28x28
        img28 = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        os.makedirs("saved_images", exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"saved_images/digit_{timestamp}.png"
        img28.save(filename)
        print(f"Saved image as {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawer(root)
    root.mainloop()
