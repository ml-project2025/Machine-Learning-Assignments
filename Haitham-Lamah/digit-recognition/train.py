# تدريب CNN على أرقام عربية (٠..٩) من MADBase
import numpy as np, os
from datasets import load_dataset

# نحاول استيراد Keras المستقل، وإلا نرجع إلى tf.keras
try:
    import keras
except ImportError:
    from tensorflow import keras
from keras import layers

print("تحميل MADBase من HuggingFace...")
ds = load_dataset("MagedSaeed/MADBase")  # 60k train / 10k test :contentReference[oaicite:2]{index=2}

def ds_to_xy(split):
    imgs = [x.convert("L") for x in ds[split]["image"]]  # PIL -> رمادي
    X = np.stack([np.array(im, dtype="float32")/255.0 for im in imgs], axis=0)
    X = np.expand_dims(X, -1)  # (N,28,28,1)
    y = np.array(ds[split]["label"], dtype="int64")      # (N,)
    return X, y

x_train, y_train = ds_to_xy("train")
x_test,  y_test  = ds_to_xy("test")

# نموذج CNN بسيط + طبقات تعزيز بيانات (داخل النموذج)
data_aug = keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomZoom(0.08),
], name="aug")

model = keras.Sequential([
    layers.Input((28,28,1)),
    data_aug,
    layers.Conv2D(32, 3, activation="relu"),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(), layers.Dropout(0.25),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(), layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation="relu"), layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()
model.fit(x_train, y_train,
          validation_split=0.1,
          epochs=8, batch_size=128, shuffle=True)

print("تقييم على الاختبار:")
model.evaluate(x_test, y_test, verbose=1)

model.save("madbase_cnn.keras")
print("تم حفظ النموذج: madbase_cnn.keras")
