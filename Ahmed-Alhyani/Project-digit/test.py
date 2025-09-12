"""
مشروع التعرف على الأرقام المكتوبة بخط اليد (MNIST Dataset)
"""

# استيراد المكتبات المطلوبة
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# استيراد مكتبات TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# تعطيل التحذيرات غير الضرورية
import warnings
warnings.filterwarnings('ignore')

print("تم استيراد جميع المكتبات بنجاح!")

# تحميل بيانات MNIST
print("جاري تحميل بيانات MNIST...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"شكل بيانات التدريب: {X_train.shape}")
print(f"شكل بيانات الاختبار: {X_test.shape}")

# عرض بعض الأمثلة من البيانات
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f'التصنيف: {y_train[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png')
plt.show()

# تجهيز البيانات
def prepare_data(X_train, X_test, y_train, y_test):
    """
    تجهيز البيانات للشبكات العصبية:
    - إعادة التشكيل لإضافة قناة اللون (لشبكة CNN)
    - تطبيع القيم بين 0 و 1
    - تحويل التصنيف إلى ترميز one-hot
    """
    # إعادة تشكيل البيانات لإضافة قناة اللون (لشبكة CNN)
    X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # تطبيع القيم بين 0 و 1
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train_cnn = X_train_cnn.astype('float32') / 255
    X_test_cnn = X_test_cnn.astype('float32') / 255

    # تحويل التصنيف إلى ترميز one-hot
    y_train_categorical = to_categorical(y_train, 10)
    y_test_categorical = to_categorical(y_test, 10)

    return X_train, X_test, X_train_cnn, X_test_cnn, y_train_categorical, y_test_categorical

# تجهيز البيانات
X_train, X_test, X_train_cnn, X_test_cnn, y_train_categorical, y_test_categorical = prepare_data(X_train, X_test, y_train, y_test)

# تقسيم البيانات إلى تدريب وتحقق
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train_categorical, test_size=0.1, random_state=42
)

X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
    X_train_cnn, y_train_categorical, test_size=0.1, random_state=42
)

print(f"شكل بيانات التدريب بعد التقسيم: {X_train.shape}")
print(f"شكل بيانات التحقق: {X_val.shape}")

# بناء النموذج الأول: شبكة عصبية كثيفة بسيطة
def build_dense_model():
    """
    بناء نموذج شبكة عصبية كثيفة بسيطة
    """
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # تحويل الصورة 28x28 إلى متجه 784
        Dense(128, activation='relu'),
        Dropout(0.2),  # إسقاط عشوائي لتقليل الإفراط في التخصيص
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')  # طبقة الإخراج مع 10 فئات
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# بناء النموذج الثاني: شبكة عصبية تلافيفية (CNN)
def build_cnn_model():
    """
    بناء نموذج شبكة عصبية تلافيفية (CNN)
    """
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# بناء النماذج
dense_model = build_dense_model()
cnn_model = build_cnn_model()

# عرض بنية النماذج
print("بنية النموذج الكثيف:")
dense_model.summary()

print("\nبنية نموذج CNN:")
cnn_model.summary()

# تدريب النموذج الكثيف
print("تدريب النموذج الكثيف...")
dense_history = dense_model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=1
)

# تدريب نموذج CNN
print("تدريب نموذج CNN...")
cnn_history = cnn_model.fit(
    X_train_cnn, y_train_cnn,
    batch_size=128,
    epochs=10,
    validation_data=(X_val_cnn, y_val_cnn),
    verbose=1
)

# رسم دقة التدريب والتحقق لكلا النموذجين
plt.figure(figsize=(12, 5))

# دقة النموذج الكثيف
plt.subplot(1, 2, 1)
plt.plot(dense_history.history['accuracy'], label='تدريب دقة')
plt.plot(dense_history.history['val_accuracy'], label='تحقق دقة')
plt.title('النموذج الكثيف - الدقة')
plt.xlabel('العصور')
plt.ylabel('الدقة')
plt.legend()

# دقة نموذج CNN
plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['accuracy'], label='تدريب دقة')
plt.plot(cnn_history.history['val_accuracy'], label='تحقق دقة')
plt.title('نموذج CNN - الدقة')
plt.xlabel('العصور')
plt.ylabel('الدقة')
plt.legend()

plt.tight_layout()
plt.savefig('training_accuracy.png')
plt.show()

# تقييم النماذج على بيانات الاختبار
# إعادة تشكيل بيانات الاختبار للنموذج الكثيف
X_test_flat = X_test.reshape(X_test.shape[0], 28, 28)

# تقييم النموذج الكثيف
dense_test_loss, dense_test_accuracy = dense_model.evaluate(X_test_flat, y_test_categorical, verbose=0)
print(f"دقة النموذج الكثيف على الاختبار: {dense_test_accuracy:.4f}")

# تقييم نموذج CNN
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test_cnn, y_test_categorical, verbose=0)
print(f"دقة نموذج CNN على الاختبار: {cnn_test_accuracy:.4f}")

# التنبؤ باستخدام كلا النموذجين
dense_predictions = dense_model.predict(X_test_flat)
cnn_predictions = cnn_model.predict(X_test_cnn)

# تحويل التنبؤات إلى تسميات
dense_pred_labels = np.argmax(dense_predictions, axis=1)
cnn_pred_labels = np.argmax(cnn_predictions, axis=1)

# تقارير التصنيف
print("تقرير التصنيف للنموذج الكثيف:")
print(classification_report(y_test, dense_pred_labels))

print("تقرير التصنيف لنموذج CNN:")
print(classification_report(y_test, cnn_pred_labels))

# حفظ النماذج المدربة
dense_model.save('mnist_dense_model.h5')
cnn_model.save('mnist_cnn_model.h5')
print("تم حفظ النماذج في ملفات mnist_dense_model.h5 و mnist_cnn_model.h5")

# دالة للتنبؤ برقم من صورة جديدة
def predict_digit(image_path, model_type='cnn'):
    """
    التنبؤ برقم من صورة جديدة

    Parameters:
    image_path (str): مسار ملف الصورة
    model_type (str): نوع النموذج المستخدم ('dense' أو 'cnn')

    Returns:
    int: الرقم المتوقع
    """
    # تحميل الصورة
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"error: {image_path}")

    # تغيير حجم الصورة إلى 28x28 بكسل
    image = cv2.resize(image, (28, 28))

    # عكس الألوان إذا كانت الصورة بيضاء على خلفية سوداء
    if np.mean(image) > 127:
        image = 255 - image

    # تطبيع الصورة
    image = image.astype('float32') / 255

    # تحميل النموذج المناسب
    if model_type == 'dense':
        model = load_model('mnist_dense_model.h5')
        image = image.reshape(1, 28, 28)
    else:  # cnn
        model = load_model('mnist_cnn_model.h5')
        image = image.reshape(1, 28, 28, 1)

    # التنبؤ
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    return predicted_digit

# مثال للتنبؤ بصورة (إذا كانت متاحة)
try:
    # يمكنك استبدال 'your_image.png' بمسار صورتك
    # digit = predict_digit('your_image.png')
    # print(f"الرقم المتوقع هو: {digit}")
    pass
except Exception as e:
    print(f"حدث خطأ أثناء التنبؤ: {e}")

# عرض بعض التنبؤات مع الصور
plt.figure(figsize=(15, 10))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    true_label = y_test[i]
    pred_label_cnn = cnn_pred_labels[i]
    color = 'green' if true_label == pred_label_cnn else 'red'
    plt.title(f'الحقيقي: {true_label}\nCNN: {pred_label_cnn}', color=color)
    plt.axis('off')
plt.tight_layout()
plt.savefig('predictions_examples.png')
plt.show()

print("""
النتيجة والتوصية:

نموذج CNN حقق دقة أعلى ({:.4f}) مقارنة بالنموذج الكثيف ({:.4f}) على بيانات الاختبار.

السبب في ذلك يعود إلى:
1. نموذج CNN مصمم خصيصًا لمعالجة الصور حيث يمكنه التعرف على الأنماط والميزات المحلية.
2. استخدام الطبقات التلافيفية позволяет للنموذج تعلم الميزات المكانية بشكل أفضل.
3. نموذج CNN أكثر كفاءة في التعامل مع التباين في كتابة الأرقام.

لذلك، يوصى باستخدام نموذج CNN للتطبيقات العملية للتعرف على الأرقام المكتوبة بخط اليد.
""".format(cnn_test_accuracy, dense_test_accuracy))