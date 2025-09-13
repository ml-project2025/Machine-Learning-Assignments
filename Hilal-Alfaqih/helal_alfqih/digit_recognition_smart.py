# digit_recognition_smart.py
# تجربة صورة رقمية مع تحميل أو تدريب تلقائي للنموذج
# الطالب: هلال الفقيه

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# 1️⃣ مسار النموذج في نفس المجلد الذي يوجد فيه الكود
MODEL_NAME = "mnist_cnn_quick_model.h5"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_NAME)

# 2️⃣ تحميل النموذج أو تدريبه إذا لم يكن موجود
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ تم تحميل النموذج المحفوظ!")
else:
    print("⚡ لم يتم العثور على النموذج، سيتم تدريبه سريعًا من البداية...")
    # تحميل بيانات MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:5000] / 255.0   # استخدام جزء صغير للتدريب السريع
    y_train = y_train[:5000]
    x_test = x_test[:1000] / 255.0
    y_test = y_test[:1000]

    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    # بناء نموذج CNN
    model = Sequential([
        Conv2D(32,(3,3),activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128,activation='relu'),
        Dense(10,activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # تدريب سريع
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test,y_test))
    # حفظ النموذج
    model.save(MODEL_PATH)
    print(f"✅ تم حفظ النموذج في: {MODEL_PATH}")

# 3️⃣ فتح نافذة اختيار الصورة
Tk().withdraw()
IMAGE_PATH = askopenfilename(title="اختر صورة الرقم", filetypes=[("PNG files","*.png"),("JPEG files","*.jpg *.jpeg")])
if not IMAGE_PATH:
    raise FileNotFoundError("لم يتم اختيار أي صورة!")

# 4️⃣ تجهيز الصورة
img = Image.open(IMAGE_PATH).convert('L')
img = img.resize((28,28))
img_array = np.array(img)
if img_array.mean() > 127:  # عكس الألوان إذا كانت بيضاء على أسود
    img_array = 255 - img_array
img_array = img_array / 255.0
img_array = img_array.reshape(1,28,28,1)

# 5️⃣ توقع الرقم
prediction = model.predict(img_array)
predicted_label = np.argmax(prediction)

# 6️⃣ عرض الصورة والرقم المتوقع
plt.imshow(img_array.reshape(28,28), cmap='gray')
plt.title(f"الرقم المتوقع: {predicted_label}")
plt.axis('off')
plt.show()
