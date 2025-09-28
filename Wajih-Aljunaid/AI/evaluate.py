import pathlib, tensorflow as tf

ROOT = pathlib.Path(r"C:\PythonProject")
DATA_DIR = ROOT / "data"

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    label_mode="int",
    color_mode="grayscale",
    image_size=(28,28),
    batch_size=32,
)

model = tf.keras.models.load_model(ROOT / "model_final.keras")
loss, acc = model.evaluate(test_ds, verbose=2)
print("✅ Test Accuracy:", acc)
