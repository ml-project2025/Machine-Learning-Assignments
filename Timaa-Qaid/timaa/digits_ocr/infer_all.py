import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# مسار الموديل
MODEL_PATH = "mnist_cnn.h5"
# مجلد الصور
IMAGES_FOLDER = "test_images"

# تحميل الموديل
model = load_model(MODEL_PATH)
print("✅ Loaded model:", MODEL_PATH)

# قراءة كل الملفات داخل المجلد
image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("⚠️ لا يوجد صور داخل المجلد:", IMAGES_FOLDER)
    exit()

print(f"🔹 Found {len(image_files)} image(s) in '{IMAGES_FOLDER}'\n")

# دالة لمعالجة الصورة قبل الإدخال للموديل
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # أبعاد (28,28,1)
    img = np.expand_dims(img, axis=0)   # أبعاد (1,28,28,1)
    return img

# تجربة كل الصور وطباعة النتائج
for img_file in image_files:
    path = os.path.join(IMAGES_FOLDER, img_file)
    img_input = preprocess_image(path)
    pred = model.predict(img_input)
    digit = np.argmax(pred)
    print(f"{img_file}: {digit}")
