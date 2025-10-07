from PIL import Image
import pytesseract

# تأكد من ضبط مسار Tesseract إذا لم يكن في متغير PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # تأكد من تعديل المسار حسب موقع التثبيت

def extract_text_from_image(image_path):
    # فتح الصورة
    img = Image.open(image_path)
    
    # استخدام Pytesseract لاستخراج النص
    text = pytesseract.image_to_string(img)
    
    return text

# استخدام الدالة
image_path = 'D:\sama_H.W\sss.jpg'  # قم بتعديل هذا المسار إلى مسار الصورة الخاصة بك
extracted_text = extract_text_from_image(image_path)

print("النص المستخرج:")
print(extracted_text)