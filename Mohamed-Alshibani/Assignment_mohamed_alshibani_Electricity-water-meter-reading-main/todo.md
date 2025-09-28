# شرح مفصل لكود البرنامج (main.py)

## نظرة عامة
هذا البرنامج هو نموذج بسيط لقراءة أرقام عدادات الكهرباء والماء من الصور باستخدام تقنية التعرف الضوئي على الحروف (OCR) بمساعدة مكتبة Tesseract. البرنامج يدعم قراءة عدادات فردية أو مجلدات متعددة، ويحفظ النتائج في ملف نصي. كما يدعم ربط العدادات بأسماء المالكين وتسجيل القراءات تاريخياً.

## المتطلبات
- Python 3.x
- مكتبات: opencv-python, pytesseract, numpy, pillow
- برنامج Tesseract OCR مثبت على النظام

## هيكل الملفات
- `main.py`: الملف الرئيسي للبرنامج.
- `meter_owners.json`: ملف JSON يحتوي على أرقام العدادات وأسماء المالكين.
- `readings.json`: ملف JSON يسجل القراءات التاريخية لكل مالك.
- `requirements.txt`: ملف يحتوي على المتطلبات.
- `dataset/`: مجلد يحتوي على الصور التجريبية.

## شرح الكود بالتفصيل

### الاستيرادات (Imports)
```python
import cv2
import pytesseract
import os
import json
from datetime import datetime
```
- `cv2`: مكتبة OpenCV لمعالجة الصور.
- `pytesseract`: مكتبة للتعرف الضوئي على الحروف باستخدام Tesseract.
- `os`: مكتبة للتعامل مع نظام الملفات.
- `json`: لقراءة وكتابة ملفات JSON.
- `datetime`: للحصول على التاريخ والوقت الحاليين.

### تحديد مسار Tesseract
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```
يحدد مسار برنامج Tesseract يدوياً لتجنب مشاكل PATH.

### دالة get_owner(meter_number)
```python
def get_owner(meter_number):
    try:
        with open('meter_owners.json', 'r', encoding='utf-8') as f:
            owners = json.load(f)
        return owners.get(meter_number, "غير معروف")
    except:
        return "غير معروف"
```
- تقرأ ملف `meter_owners.json`.
- ترجع اسم المالك المرتبط برقم العداد، أو "غير معروف" إذا لم يوجد.

### دالة add_reading(owner, reading)
```python
def add_reading(owner, reading):
    try:
        with open('readings.json', 'r', encoding='utf-8') as f:
            readings = json.load(f)
    except:
        readings = {}
    
    if owner not in readings:
        readings[owner] = []
    
    readings[owner].append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reading": reading
    })
    
    with open('readings.json', 'w', encoding='utf-8') as f:
        json.dump(readings, f, ensure_ascii=False, indent=4)
```
- تقرأ ملف `readings.json` أو تنشئه إذا لم يوجد.
- تضيف القراءة الجديدة مع التاريخ للمالك المحدد.
- تحفظ الملف مرة أخرى.

### دالة create_sample_meter_images()
```python
def create_sample_meter_images():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs("dataset", exist_ok=True)
    samples = [
        ("water_sample.jpg", "عداد ماء", "123456"),
        ("electric_sample.jpg", "عداد كهرباء", "654321"),
    ]
    for fname, label, number in samples:
        path = os.path.join("dataset", fname)
        if not os.path.exists(path):
            img = Image.new("RGB", (300, 120), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 32)
            except:
                font = ImageFont.load_default()
            d.text((10, 10), label, font=font, fill=(0, 0, 0))
            d.text((10, 60), number, font=font, fill=(0, 0, 0))
            img.save(path)
```
- تنشئ صور تجريبية لعدادات الماء والكهرباء إذا لم تكن موجودة.
- تستخدم PIL لرسم النص على صورة بيضاء.

### دالة read_meter(img_path)
```python
def read_meter(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"لم يتم العثور على الصورة أو لا يمكن قراءتها: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # إزالة الضوضاء
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    # تنعيم خفيف
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    # تحويل للصورة الثنائية
    _, thresh = cv2.threshold(
        blurred, 127, 255, cv2.THRESH_BINARY_INV
    )
    # توسيع الأرقام لتحسين القراءة
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    numbers = pytesseract.image_to_string(thresh, config=custom_config)
    owner = get_owner(numbers.strip())
    add_reading(owner, numbers.strip())
    return numbers.strip(), owner
```
- تقرأ الصورة وتعالجها لتحسين جودة النص.
- تستخدم OCR لاستخراج الأرقام.
- تحدد المالك وتضيف القراءة.
- ترجع الرقم والمالك.

### الجزء الرئيسي (if __name__ == "__main__")
- ينشئ الصور التجريبية.
- يطلب من المستخدم إدخال مسار الصورة أو المجلد.
- إذا كان مجلد، يعالج جميع الصور فيه.
- إذا كانت صورة واحدة، يعالجها ويعرض النتيجة.
- يحفظ النتائج في ملف نصي.

### دوال مساعدة
- `list_images_in_dataset()`: ترجع قائمة بالصور في مجلد dataset.
- `prompt_for_image_from_dataset()`: تعرض القائمة وتطلب اختيار صورة.

## كيفية التشغيل
1. شغّل `python main.py`.
2. أدخل مسار الصورة أو اضغط Enter للصورة الافتراضية.
3. إذا لم توجد، ستظهر قائمة للاختيار.
4. سيتم عرض الرقم واسم المالك، وحفظ القراءة.

## ملاحظات
- البرنامج يدعم اللغة العربية في الواجهة والملفات.
- يمكن توسيع قاعدة البيانات في `meter_owners.json`.
- لتحسين دقة OCR، جرب تعديل إعدادات معالجة الصور.