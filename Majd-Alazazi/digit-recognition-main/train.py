import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def build_model(input_shape=(28,28,1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    model = build_model()
    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=args.epochs,
              batch_size=args.batch_size)

    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "digit_cnn.keras")
    model.save(out_path)
    print(f"[OK] Model saved to: {out_path}")

if __name__ == "__main__":
    main()
