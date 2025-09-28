# create_mnist_dataset.py
# ينزّل MNIST ويحوّله لصور PNG داخل:
# C:\PythonProject1\data\{train,val,test}\{0..9}\xxxxx.png

import os
import shutil
import pathlib
from typing import Optional, Tuple
import numpy as np
from PIL import Image

# --- عدّل هذا المسار إذا كان مشروعك في مكان آخر ---
ROOT = pathlib.Path(r"C:\pro\AI")
DATA_DIR = ROOT / "data"

# استخدم كل البيانات افتراضيًا. إن أردت نسخة خفيفة للتجربة الأولى، ضع رقمًا مثل 1000 (صورة لكل split)
MAX_IMAGES_PER_SPLIT: Optional[int] = None  # مثال: 1000

# إن كان المجلد موجودًا مسبقًا: امسحه وأعد إنشاءه (غالبًا أنسب للبداية من الصفر)
CLEAN_OUTPUT_DIR = True

# ---------- لا تعدّل عادةً تحت هذا السطر ----------
def ensure_clean_dir(path: pathlib.Path, clean: bool = False) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def make_split_dirs(base: pathlib.Path) -> None:
    for split in ["train", "val", "test"]:
        for cls in range(10):
            (base / split / str(cls)).mkdir(parents=True, exist_ok=True)

def save_split(
    images: np.ndarray, labels: np.ndarray, out_dir: pathlib.Path, split: str, max_images: Optional[int] = None
) -> Tuple[int, int]:
    """يحفظ صور split ويعيد (عدد_الصور, عدد_الفئات_الفارغة)"""
    # تأكد من النوع والمدى
    if images.dtype != np.uint8:
        images = images.astype(np.uint8)
    total_saved = 0
    empty_classes = 0

    for cls in range(10):
        cls_mask = labels == cls
        cls_imgs = images[cls_mask]
        if cls_imgs.size == 0:
            empty_classes += 1
            continue

        if max_images is not None:
            cls_imgs = cls_imgs[:max_images // 10 if max_images >= 10 else max_images]

        cls_dir = out_dir / split / str(cls)
        count = 0
        for i, img in enumerate(cls_imgs):
            im = Image.fromarray(img, mode="L")
            # اسم ملف ثابت الطول
            name = f"{i:05d}.png"
            im.save(cls_dir / name)
            count += 1
            total_saved += 1

        print(f"[{split}] class {cls}: saved {count} images to {cls_dir}")

    return total_saved, empty_classes

def main():
    print("==> Preparing output folders at:", DATA_DIR)
    ensure_clean_dir(DATA_DIR, clean=CLEAN_OUTPUT_DIR)
    make_split_dirs(DATA_DIR)

    print("==> Downloading MNIST (via Keras)… قد يأخذ دقيقة أول مرة")
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # أشكال: (60000, 28, 28), (10000, 28, 28)

    # تقسيم train إلى train (50k) و val (10k)
    x_tr, y_tr = x_train[:50000], y_train[:50000]
    x_val, y_val = x_train[50000:], y_train[50000:]

    # نتأكد أنها uint8 ومدى 0..255
    x_tr = np.clip(x_tr, 0, 255).astype(np.uint8)
    x_val = np.clip(x_val, 0, 255).astype(np.uint8)
    x_te = np.clip(x_test, 0, 255).astype(np.uint8)

    print(f"Train: {x_tr.shape}, Val: {x_val.shape}, Test: {x_te.shape}")

    # حفظ الصور
    total_tr, _ = save_split(x_tr, y_tr, DATA_DIR, "train", MAX_IMAGES_PER_SPLIT)
    total_val, _ = save_split(x_val, y_val, DATA_DIR, "val", MAX_IMAGES_PER_SPLIT)
    total_te, _ = save_split(x_te, y_test, DATA_DIR, "test", MAX_IMAGES_PER_SPLIT)

    print("\n✅ Done!")
    print(f"Saved counts -> train: {total_tr}, val: {total_val}, test: {total_te}")
    print(f"Folders created under: {DATA_DIR}")
    print(r"Structure: data\train\0..9, data\val\0..9, data\test\0..9")

if __name__ == "__main__":
    main()
