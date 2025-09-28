def get_owner(meter_number):
    try:
        with open('meter_owners.json', 'r', encoding='utf-8') as f:
            owners = json.load(f)
        return owners.get(meter_number, "غير معروف")
    except:
        return "غير معروف"

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

import cv2
import pytesseract
import os
import json
from datetime import datetime

# تحديد مسار Tesseract يدوياً إذا لم يكن في PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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



if __name__ == "__main__":
    # إنشاء صور عداد ماء وكهرباء وهمية إذا لم تكن موجودة
    try:
        create_sample_meter_images()
    except Exception as e:
        print(f"تنبيه: تعذر إنشاء صور العدادات التجريبية تلقائياً: {e}")
    img_path = input(
        "أدخل مسار صورة العداد أو مجلد صور (أو Enter لاستخدام صورة عداد ماء): "
    ).strip()
    def list_images_in_dataset():
        dataset_dir = "dataset"
        if not os.path.isdir(dataset_dir):
            return []
        return [f for f in os.listdir(dataset_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    def prompt_for_image_from_dataset():
        images = list_images_in_dataset()
        if not images:
            print("لا توجد صور متوفرة في مجلد dataset.")
            exit(1)
        print("الصور المتوفرة في مجلد dataset:")
        for idx, img in enumerate(images, 1):
            print(f"{idx}. {img}")
        while True:
            try:
                choice = input("اختر رقم الصورة المطلوبة: ").strip()
            except EOFError:
                print("تم إلغاء الإدخال. جرب تشغيل البرنامج من جديد.")
                exit(1)
            if choice.isdigit() and 1 <= int(choice) <= len(images):
                return os.path.join("dataset", images[int(choice)-1])
            else:
                print("اختيار غير صحيح. حاول مرة أخرى.")

    if not img_path:
        # جرب المسار الحالي ثم dataset
        default_img = "صورة عداد ماء.jpg"
        english_default = "water_sample.jpg"
        if os.path.exists(default_img):
            img_path = default_img
        elif os.path.exists(os.path.join("dataset", default_img)):
            img_path = os.path.join("dataset", default_img)
        elif os.path.exists(os.path.join("dataset", english_default)):
            img_path = os.path.join("dataset", english_default)
        else:
            print(f"لم يتم العثور على الصورة الافتراضية '{default_img}' أو '{english_default}' في المجلد الحالي أو مجلد dataset.")
            img_path = prompt_for_image_from_dataset()
    elif not os.path.exists(img_path):
        print(f"لم يتم العثور على الصورة أو المجلد: {img_path}")
        img_path = prompt_for_image_from_dataset()
    if os.path.isdir(img_path):
        print("\nنتائج معالجة جميع الصور في المجلد (مع دعم المجلدات الفرعية):\n")
        results = []
        for root, dirs, files in os.walk(img_path):
            meter_type = os.path.basename(root) if root != img_path else "غير محدد"
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    fpath = os.path.join(root, fname)
                    try:
                        result, owner = read_meter(fpath)
                    except Exception as e:
                        result = f"خطأ: {e}"
                        owner = "غير معروف"
                    results.append((meter_type, fname, result, owner))
                    print(f"{'-'*40}\nنوع العداد: {meter_type}\nالصورة: {fname}\nالرقم المستخرج: {result}\nصاحب العداد: {owner}\n")
        # حفظ النتائج في ملف نصي
        if results:
            with open("نتائج_قراءة_العدادات.txt", "w", encoding="utf-8") as f:
                for meter_type, fname, result, owner in results:
                    f.write(f"نوع العداد: {meter_type}\nالصورة: {fname}\nالرقم المستخرج: {result}\nصاحب العداد: {owner}\n{'-'*40}\n")
            print("تم حفظ النتائج في ملف: نتائج_قراءة_العدادات.txt\n")
    else:
        try:
            result, owner = read_meter(img_path)
            print(f"\n{'-'*40}\nقراءة العداد: {result}\nصاحب العداد: {owner}\n")
        except Exception as e:
            print(f"\nحدث خطأ أثناء قراءة الصورة: {e}\n")
