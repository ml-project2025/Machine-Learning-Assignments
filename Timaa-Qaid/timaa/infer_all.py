import os
import cv2
from digits_ocr import infer
from tensorflow.keras.models import load_model

# مسار المجلد اللي فيه الصور
images_folder = "test_images"
# مسار النموذج
model_path = "mnist_cnn.h5"
# ملف حفظ النتائج
results_file = "results.txt"

# تحميل الموديل
model = load_model(model_path)
print(f"✅ Loaded model: {model_path}")

# البحث عن كل الصور في المجلد
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"🔹 Found {len(image_files)} image(s) in '{images_folder}'")

# فتح ملف النتائج للكتابة
with open(results_file, "w") as f:
    for img_name in image_files:
        img_path = os.path.join(images_folder, img_name)
        try:
            # استدعاء دالة inference من digits_ocr.py
            digit = infer(img_path, model)
            print(f"{img_name}: {digit}")
            f.write(f"{img_name}: {digit}\n")
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            f.write(f"{img_name}: ERROR ({e})\n")

print(f"\n✅ All results saved to '{results_file}'")
