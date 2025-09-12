import re
from pathlib import Path

import cv2
import pytesseract
from num2words import num2words

# === 1- تحديد مسار tesseract.exe لو لم يكن في PATH ===
tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(tess_path).exists():
    pytesseract.pytesseract.tesseract_cmd = tess_path

# === 2- مسار الصورة ===
image_path = "samples/number1.png"
if not Path(image_path).exists():
    raise FileNotFoundError(f"لم أجد الصورة: {image_path}")

# === 3- قراءة الصورة ومعالجتها ===
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# إزالة الضوضاء
gray = cv2.GaussianBlur(gray, (5,5), 0)
# تحويل الصورة لأسود/أبيض لتحسين الدقة
_, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# === 4- OCR مع التركيز على الأرقام ===
config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789٠١٢٣٤٥٦٧٨٩'
text_raw = pytesseract.image_to_string(bw, config=config)

# === 5- استخراج الأرقام فقط ===
matches = re.findall(r'[0-9]+|[٠-٩]+', text_raw)

# تحويل الأرقام العربية-الهندية إلى الغربية
def to_western_digits(s: str) -> str:
    return s.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩','0123456789'))

if not matches:
    print("لم أتعرف على أرقام في الصورة.")
    print("النص الخام من OCR:", repr(text_raw))
else:
    print("الأرقام المستخرجة:", matches)
    for m in matches:
        western = to_western_digits(m)
        n = int(western)
        words_ar = num2words(n, lang='ar')
        print(f"{m} → {words_ar}")

# === 6- حفظ نسخة من الصورة المعالجة (اختياري) ===
Path("outputs").mkdir(exist_ok=True)
cv2.imwrite("outputs/preprocessed.png", bw)
