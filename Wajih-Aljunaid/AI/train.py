import pathlib
import tensorflow as tf
from tensorflow.keras import layers

ROOT = pathlib.Path(r"C:\pro\AI")
DATA_DIR = ROOT / "data"

BATCH_SIZE = 32
IMG_SIZE = (28,28)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    label_mode="int",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    label_mode="int",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# تحسين الأداء
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# بناء الموديل
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(28,28,1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# التدريب
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

model.save(ROOT / "model_final.keras")
print("✅ تم حفظ الموديل بنجاح")
