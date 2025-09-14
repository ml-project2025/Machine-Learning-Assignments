
"""
main.py — خطوة 2: تدريب + تنبؤ على الصور
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

from model import train_and_save, load_model
from preprocess import load_and_prepare

DEFAULT_MODEL = str(Path(__file__).resolve().parents[1] / "models" / "knn.joblib")
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

def visualize_and_save(img8: np.ndarray, pred: int, out_path: Path):
    big = cv2.resize(img8, (256, 256), interpolation=cv2.INTER_NEAREST)
    big = (big / 16.0 * 255).astype(np.uint8)
    cv2.putText(big, f"Pred: {pred}", (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255), 2)
    cv2.imwrite(str(out_path), big)

def main():
    parser = argparse.ArgumentParser(description="مشروع عزّام — خطوة 2 (تدريب + تنبؤ)")
    parser.add_argument("--train", action="store_true", help="درّب النموذج واحفظه")
    parser.add_argument("--predict", type=str, help="مسار صورة للتنبؤ")
    parser.add_argument("--invert", action="store_true", help="اقلب ألوان الصورة قبل المعالجة")
    parser.add_argument("--k", type=int, default=3, help="عدد الجيران في KNN (افتراضي 3)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="مسار النموذج")
    args = parser.parse_args()

    if args.train:
        info = train_and_save(args.model, k=args.k)
        print(f"✅ تم التدريب والحفظ في: {args.model}")
        print(f"🎯 الدقة (Accuracy): {info['accuracy']:.4f}")
        print("📊 تقرير التصنيف:\n", info["report"])

    if args.predict:
        model = load_model(args.model)
        flat, img8 = load_and_prepare(args.predict, invert=args.invert)
        pred = int(model.predict(flat)[0])
        print(f"🔢 الرقم المتوقع: {pred}")
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_DIR / f"pred_{Path(args.predict).stem}.png"
        visualize_and_save(img8, pred, out_path)
        print(f"🖼️ تم حفظ صورة الإخراج: {out_path}")

if __name__ == "__main__":
    main()
