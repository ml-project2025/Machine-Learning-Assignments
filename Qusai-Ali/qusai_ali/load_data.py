import tensorflow as tf
import matplotlib.pyplot as plt

# 1. تحميل مجموعة بيانات MNIST
# تحتوي MNIST على صور أرقام مكتوبة بخط اليد من 0 إلى 9
# تنقسم البيانات إلى مجموعتين: للتدريب (training) وللاختبار (testing)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. استكشاف البيانات
print("شكل بيانات التدريب (الصور):", x_train.shape)
print("شكل بيانات التدريب (العناوين):", y_train.shape)
print("شكل بيانات الاختبار (الصور):", x_test.shape)
print("شكل بيانات الاختبار (العناوين):", y_test.shape)

# 3. عرض بعض الصور من البيانات
# لنعرض أول 10 صور من مجموعة التدريب
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray') # عرض الصورة بتدرج الرمادي
    plt.title(f"الرقم: {y_train[i]}")
    plt.axis('off') # إخفاء المحاور
plt.show()

# 4. تجهيز البيانات (Data Preprocessing)
# الشبكات العصبية تفضل أن تكون قيم الإدخال صغيرة، عادة بين 0 و 1.
# الصور الحالية قيم البكسل فيها تتراوح من 0 إلى 255. [5]
# سنقوم بقسمة كل بكسل على 255 لتحويل النطاق إلى ما بين 0 و 1.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# إضافة بُعد للقناة (Channel dimension)
# الشبكات العصبية التلافيفية (CNNs) تتوقع أن يكون للصور بُعد خاص بالقناة (مثل RGB).
# صورنا باللون الرمادي، لذا لديها قناة واحدة فقط.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print("\nشكل البيانات بعد التجهيز:")
print("شكل بيانات التدريب الجديدة:", x_train.shape)
print("شكل بيانات الاختبار الجديدة:", x_test.shape)

# (الكود السابق لتحميل ومعالجة البيانات يجب أن يكون هنا)
# ...

# 5. بناء نموذج الشبكة العصبية التلافيفية (CNN)
print("\nبدء بناء النموذج...")
model = tf.keras.models.Sequential([
    # الطبقة التلافيفية الأولى: 32 فلتر بحجم 3x3
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # طبقة التجميع لتقليل الأبعاد
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # الطبقة التلافيفية الثانية: 64 فلتر
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # طبقة تجميع أخرى
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # طبقة تسطيح لتحويل البيانات من 2D إلى 1D
    tf.keras.layers.Flatten(),
    
    # طبقة كثيفة (متصلة بالكامل) مع 128 عقدة
    tf.keras.layers.Dense(128, activation='relu'),
    
    # طبقة الإخراج: 10 عقد (واحدة لكل رقم من 0-9)
    # نستخدم 'softmax' لأننا نريد احتمالية لكل فئة
    tf.keras.layers.Dense(10, activation='softmax')
])

# 6. تجميع النموذج (Compile)
print("تجميع النموذج...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# عرض ملخص لبنية النموذج
model.summary()

# 7. تدريب النموذج
print("\nبدء تدريب النموذج...")
# سنقوم بتدريب النموذج لـ 5 دورات (epochs)
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("انتهى التدريب!")

# 8. تقييم النموذج
print("\nبدء تقييم النموذج...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nدقة النموذج على بيانات الاختبار: {test_acc*100:.2f}%")

# 9. حفظ النموذج المدرب (خطوة مهمة جدًا)
model.save('handwriting_model.h5')
print("\nتم حفظ النموذج في ملف 'handwriting_model.h5'")