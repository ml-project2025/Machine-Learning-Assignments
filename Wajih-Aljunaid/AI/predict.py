import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pathlib, numpy as np, tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Rescaling

ROOT = pathlib.Path(r"C:\pro\AI")
MODEL = tf.keras.models.load_model(ROOT / "model_final.keras")

def model_has_rescaling(model) -> bool:
    def walk(layers):
        for l in layers:
            if isinstance(l, Rescaling):
                return True
            if hasattr(l, "layers") and l.layers:
                if walk(l.layers):
                    return True
        return False
    return walk(model.layers)

HAS_RESCALE = model_has_rescaling(MODEL)

def load_image(path, expect_0_255: bool):
    # تجهيز بسيط: رمادي + 28x28، وبعدها نقرّر نُقسّم أو لا
    img = Image.open(str(path)).convert("L").resize((28,28))
    x = np.array(img).astype("float32")
    if not expect_0_255:
        x = x / 255.0
    return x.reshape(1,28,28,1)

if __name__ == "__main__":
    img_path = ROOT / "data" / "test" / "8" / "img.png"   # غيّر المسار للصورة التي تختبرها
    x = load_image(img_path, expect_0_255=HAS_RESCALE)

    # معلومات تشخيصية مفيدة
    print("🧪 يوجد Rescaling داخل الموديل؟", "نعم" if HAS_RESCALE else "لا")
    print("ℹ️  مدى البكسلات المُدخلة للموديل قبل التنبؤ:",
          float(x.min()), "→", float(x.max()))

    probs = MODEL.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    print("🔮 التوقع:", pred)
    print("📊 نسبة الثقة:", round(conf, 4))
    print("\nاحتمالات كل الأرقام:")
    for i, p in enumerate(probs):
        print(f" - الرقم {i}: {p:.4f}")
