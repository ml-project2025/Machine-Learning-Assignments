
import argparse
from pathlib import Path
import cv2

from model_svm import train_and_save, load_model
from preprocess import load_and_prepare

DEFAULT_MODEL = str(Path(__file__).resolve().parents[1] / "models" / "svm.joblib")
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

def visualize_and_save(img8, pred: int, out_path: Path):
    big = cv2.resize(img8, (256, 256), interpolation=cv2.INTER_NEAREST)
    big = (big / 16.0 * 255).astype("uint8")
    cv2.putText(big, f"Pred: {pred}", (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255), 2)
    cv2.imwrite(str(out_path), big)

def main():
    parser = argparse.ArgumentParser(description="مشروع غدير — تدريب/تنبؤ (SVM)")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", type=str)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.train:
        info = train_and_save(args.model, C=args.C, gamma=args.gamma)
        print(f"✅ تم التدريب والحفظ في: {args.model}")
        print(f"🎯 الدقة: {info['accuracy']:.4f}")
        print("📊 تقرير:\n", info["report"])

    if args.predict:
        model = load_model(args.model)
        flat, img8 = load_and_prepare(args.predict, invert=args.invert)
        pred = int(model.predict(flat)[0])
        print(f"🔢 الرقم المتوقع: {pred}")
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_DIR / f"pred_{Path(args.predict).stem}.png"
        visualize_and_save(img8, pred, out_path)
        print(f"🖼️ تم حفظ النتيجة: {out_path}")

if __name__ == "__main__":
    main()
