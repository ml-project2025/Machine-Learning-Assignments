import argparse, os
import numpy as np
import tensorflow as tf
from PIL import Image
from utils.preprocess import preprocess_image_28x28

def predict_image(model_path, image_path, top_k=3):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train it with train.py first.")
    model = tf.keras.models.load_model(model_path, compile=False)
    img = Image.open(image_path).convert("L")
    x = preprocess_image_28x28(img)  # shape (1, 28, 28, 1)
    probs = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    result = [(int(i), float(probs[i])) for i in top_idx]
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model", default="models/digit_cnn.keras")
    args = parser.parse_args()

    res = predict_image(args.model, args.image, top_k=3)
    print("Top predictions:")
    for i, p in res:
        print(f"  {i}: {p:.3f}")
