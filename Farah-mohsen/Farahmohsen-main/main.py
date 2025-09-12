import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import pytesseract
import cv2

# مسار برنامج Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img_path = None  # لتخزين مسار الصورة المختارة

def open_image():
    global img_path, img_display
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if img_path:
        img = Image.open(img_path)
        img = img.resize((300, 200))  # لتصغير الصورة للعرض فقط
        img_display = ImageTk.PhotoImage(img)
        image_label.config(image=img_display)

def extract_text():
    if img_path:
        # قراءة الصورة باستخدام OpenCV
        img = cv2.imread(img_path)

        if img is None:
            result_label.config(text="⚠️ لم يتم فتح الصورة. تأكد من المسار")
            return

        # تحسين الصورة: تحويل رمادي + إزالة ضوضاء + زيادة التباين
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)  # إزالة الضوضاء
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        try:
            # تخصيص Tesseract لقراءة الأرقام فقط
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(thresh, config=custom_config)
        except pytesseract.TesseractError as e:
            result_label.config(text="⚠️ خطأ في Tesseract:\n" + str(e))
            return

        if text.strip():
            result_label.config(text="الأرقام المستخرجة:\n" + text.strip())
        else:
            result_label.config(text="⚠️ لم يتم العثور على أرقام واضحة في الصورة")

# إنشاء نافذة
root = tk.Tk()
root.title("استخراج الأرقام فقط")

# زر اختيار الصورة
btn_open = Button(root, text="📂 اختر صورة", command=open_image)
btn_open.pack(pady=10)

# مكان عرض الصورة
image_label = Label(root)
image_label.pack()

# زر استخراج النصوص
btn_extract = Button(root, text="🔍 استخراج الأرقام", command=extract_text)
btn_extract.pack(pady=10)

# مكان عرض النتيجة
result_label = Label(root, text="", wraplength=400, justify="left")
result_label.pack()

root.mainloop()