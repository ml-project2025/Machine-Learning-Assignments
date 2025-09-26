import cv2
import pytesseract

# 🔹 ضع هنا مسار تثبيت Tesseract إذا كنت على ويندوز
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🔹 اسم الصورة
image_path = "numbers.png"
img = cv2.imread(image_path)

# تحويل الصورة إلى تدرج رمادي
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# تحسين التباين
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# إعداد Tesseract لاستخراج الأرقام فقط
custom_config = r'--oem 3 --psm 6 outputbase digits'

# قراءة الأرقام من الصورة وطباعتها في الـ Terminal فقط
text = pytesseract.image_to_string(gray, config=custom_config)
print(text.strip())



