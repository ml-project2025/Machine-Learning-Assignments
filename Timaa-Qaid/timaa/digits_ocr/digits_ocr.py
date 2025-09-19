#!/usr/bin/env python3
"""
Digit OCR (Arabic-friendly comments)
-----------------------------------

• هذا السكربت يدرّب شبكة عصبية التفافية CNN على قاعدة بيانات MNIST للتعرف على الأرقام (0-9)
  ثم يستخدم OpenCV لتقطيع الأرقام من صورة وإخراج النص المتوقع.

المزايا:
- train: تدريب النموذج وحفظه كملف mnist_cnn.h5
- infer: قراءة صورة فيها أرقام متعددة، قصّ كل رقم، تصنيفه، وترتيبه من اليسار لليمين، وإخراج سلسلة الأرقام.
- يحفظ صورة ناتجة مع المربعات والتوقعات.

الاستخدام:
  python digits_ocr.py --train --epochs 5 --model mnist_cnn.h5
  python digits_ocr.py --infer path/to/image.jpg --model mnist_cnn.h5 --out result.png

المتطلبات:
  pip install tensorflow==2.15.0 opencv-python numpy matplotlib
(لو كان عندك GPU وكودا مثبت، ممكن تستخدم tensorflow-gpu الموافقة لإصدارك.)
"""
import argparse
import os
import sys
import numpy as np
import cv2

# TensorFlow/Keras
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Utils
# -----------------------------

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # normalize to [0,1] and add channel dim
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    return (x_train, y_train), (x_test, y_test)


def train(model_path: str, epochs: int = 5, batch_size: int = 128):
    (x_train, y_train), (x_test, y_test) = load_mnist()

    model = build_cnn_model()
    model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(model_path)
    print(f"\n✅ Saved model to: {model_path}")


# -----------------------------
# Inference helpers
# -----------------------------

def pad_to_square(img: np.ndarray, border_value: int = 0) -> np.ndarray:
    """Pad grayscale image to make it square (keep digit centered)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    if h > w:
        pad = (h - w)
        left = pad // 2
        right = pad - left
        return cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=border_value)
    else:
        pad = (w - h)
        top = pad // 2
        bottom = pad - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=border_value)


def preprocess_digit_roi(roi: np.ndarray) -> np.ndarray:
    """Prepare ROI for MNIST CNN: grayscale->threshold->resize->normalize."""
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # normalize background/foreground: invert if needed so digit is white on black like MNIST
    # Otsu threshold
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small noise
    th = cv2.medianBlur(th, 3)

    # find bounding rect of the digit mass (to tighten crop)
    ys, xs = np.where(th > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    tight = th[y1:y2 + 1, x1:x2 + 1]

    square = pad_to_square(tight, border_value=0)
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    # final float32 normalization
    out = resized.astype("float32") / 255.0
    out = out[..., None]  # add channel
    return out


def sort_contours_left_to_right(cnts):
    # sort by x coordinate of bounding rect
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    cnts_bb = sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][0])
    return [c for c, _ in cnts_bb], [bb for _, bb in cnts_bb]


def extract_digits(image_bgr: np.ndarray):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # adaptive threshold to handle variable lighting
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)

    # morphological opening to remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # find contours
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter by size to skip dust
    min_area = 30  # tweak if needed
    filtered = [c for c in cnts if cv2.contourArea(c) >= min_area]

    if not filtered:
        return []

    sorted_cnts, boxes = sort_contours_left_to_right(filtered)

    rois = []
    for (x, y, w, h), c in zip(boxes, sorted_cnts):
        # slight padding to include stroke
        pad = max(1, int(0.08 * max(w, h)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image_bgr.shape[1], x + w + pad)
        y1 = min(image_bgr.shape[0], y + h + pad)
        roi = image_bgr[y0:y1, x0:x1]
        rois.append({
            "roi": roi,
            "bbox": (x0, y0, x1, y1)
        })
    return rois


def infer(image_path: str, model_path: str, out_path: str = None):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. درّب أولاً بخيار --train")

    model = keras.models.load_model(model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detections = extract_digits(img)
    if not detections:
        print("لم يتم العثور على أرقام في الصورة. جرّب تحسين الإضاءة/التباين أو غيّر عتبات المعالجة.")
        return ""

    preds = []
    annotated = img.copy()

    for det in detections:
        roi = det["roi"]
        bbox = det["bbox"]
        proc = preprocess_digit_roi(roi)
        if proc is None:
            continue
        x = np.expand_dims(proc, axis=0)
        prob = model.predict(x, verbose=0)[0]
        digit = int(np.argmax(prob))
        conf = float(np.max(prob))
        preds.append({"digit": digit, "conf": conf, "bbox": bbox})

    # compose final string
    text = "".join(str(p["digit"]) for p in preds)

    # draw
    for p in preds:
        x0, y0, x1, y1 = p["bbox"]
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{p['digit']}:{p['conf']:.2f}"
        cv2.putText(annotated, label, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if out_path:
        cv2.imwrite(out_path, annotated)
        print(f"📸 Saved annotated image: {out_path}")

    print(f"✅ Detected digits: {text}")
    return text


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train and run a CNN digit OCR on MNIST and real images.")
    parser.add_argument("--train", action="store_true", help="Train the CNN on MNIST and save model.")
    parser.add_argument("--infer", type=str, default=None, help="Path to input image for inference.")
    parser.add_argument("--model", type=str, default="mnist_cnn.h5", help="Path to save/load model.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=128, help="Training batch size.")
    parser.add_argument("--out", type=str, default=None, help="Output annotated image path.")

    args = parser.parse_args()

    if args.train:
        train(args.model, epochs=args.epochs, batch_size=args.batch)

    if args.infer:
        infer(args.infer, args.model, out_path=args.out)

    if (not args.train) and (args.infer is None):
        parser.print_help()


if __name__ == "__main__":
    main()
