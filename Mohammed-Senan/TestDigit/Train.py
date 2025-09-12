# -*- coding: utf-8 -*-
"""
مشروع مبسط للتعرف على الأرقام المكتوبة بخط اليد باستخدام مجموعة بيانات MNIST
يشمل:
- تحميل وتجهيز البيانات
- بناء نموذج Dense بسيط
- بناء نموذج CNN
- تدريب وتقييم النموذجين
- حفظ النموذج
- التنبؤ بصورة جديدة (من ملف خارجي)
- رسم منحنيات التدريب (Accuracy / Loss)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

# -------------------------------
# 1. تحميل وتجهيز بيانات MNIST
# -------------------------------
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# تطبيع البيانات (تحويل القيم إلى [0,1])
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# إعادة تشكيل البيانات لشبكة CNN (28x28x1)
x_train_cnn = np.expand_dims(x_train, -1)
x_test_cnn = np.expand_dims(x_test, -1)

# تحويل التسميات إلى one-hot
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# تقسيم مجموعة التدريب (Training / Validation)
x_val = x_train_cnn[-10000:]
y_val = y_train_cat[-10000:]
x_train_cnn = x_train_cnn[:-10000]
y_train_cat = y_train_cat[:-10000]

# -------------------------------
# 2. نموذج Dense بسيط
# -------------------------------
dense_model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

dense_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

print("\n--- تدريب نموذج Dense ---")
history_dense = dense_model.fit(
    x_train_cnn, y_train_cat,
    epochs=5,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=2
)

# -------------------------------
# 3. نموذج CNN
# -------------------------------
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

print("\n--- تدريب نموذج CNN ---")
history_cnn = cnn_model.fit(
    x_train_cnn, y_train_cat,
    epochs=5,
    batch_size=128,
    validation_data=(x_val, y_val),
    verbose=2
)

# -------------------------------
# 4. التقييم على مجموعة الاختبار
# -------------------------------
print("\n--- تقييم نموذج Dense ---")
y_pred_dense = dense_model.predict(x_test_cnn, verbose=0)
print(classification_report(y_test, np.argmax(y_pred_dense, axis=1)))

print("\n--- تقييم نموذج CNN ---")
y_pred_cnn = cnn_model.predict(x_test_cnn, verbose=0)
print(classification_report(y_test, np.argmax(y_pred_cnn, axis=1)))

# -------------------------------
# 5. حفظ النموذج الأفضل (CNN عادةً)
# -------------------------------
cnn_model.save("mnist_cnn_model.h5")
print("\n✅ تم حفظ نموذج CNN في الملف mnist_cnn_model.h5")

# -------------------------------
# 6. رسم منحنيات التدريب
# -------------------------------
def plot_history(history, title):
    """دالة لرسم Accuracy و Loss"""
    plt.figure(figsize=(12, 4))

    # الدقة
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # الخسارة
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

print("\n📊 منحنيات تدريب نموذج Dense")
plot_history(history_dense, "Dense Model")

print("\n📊 منحنيات تدريب نموذج CNN")
plot_history(history_cnn, "CNN Model")

# -------------------------------
# 7. التنبؤ بصورة جديدة
# -------------------------------
def predict_new_image(img_path):
    """تحميل صورة خارجية (28x28 أبيض وأسود) والتنبؤ بها"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # الشكل (1,28,28,1)
    
    prediction = cnn_model.predict(img, verbose=0)
    return np.argmax(prediction)

# مثال: ضع صورة رقم باسم "digit.png" في نفس المجلد
# print("الرقم المتوقع هو:", predict_new_image("digit.png"))

