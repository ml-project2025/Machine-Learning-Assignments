import streamlit as st
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

from model_svm import train_and_save, load_model
from preprocess import to_8x8

APP_TITLE = "غدير — واجهة قراءة الأرقام (SVM)"
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "svm.joblib"

st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)
st.write("يدعم وضعين: **رقم واحد** (أكبر كونتور) أو **أرقام متعددة** (يسار → يمين).")



with st.sidebar:
    st.header("إعداد النموذج")
    auto_train = st.checkbox("درّب تلقائيًا إذا لم يوجد نموذج", value=True)
    mode = st.radio("وضع القراءة", ["رقم واحد", "أرقام متعددة"], index=0)
    C_val = st.slider("SVM C", 0.1, 20.0, 10.0)
    gamma_val = st.selectbox("gamma", ["scale", "auto"], index=0)
    if st.button("إعادة تدريب SVM الآن"):
        info = train_and_save(str(MODEL_PATH), C=float(C_val), gamma=str(gamma_val))
        st.success(f"تم التدريب. الدقة: {info['accuracy']:.4f}")
        st.text(info["report"])

if not MODEL_PATH.exists():
    if auto_train:
        info = train_and_save(str(MODEL_PATH), C=float(C_val), gamma=str(gamma_val))
        st.info(f"تم تدريب نموذج جديد (لأنه غير موجود). الدقة: {info['accuracy']:.4f}")
    else:
        st.warning("لا يوجد نموذج محفوظ. فعّل التدريب التلقائي أو اضغط إعادة تدريب.")
        st.stop()

model = load_model(str(MODEL_PATH))

def disambiguate_6_9(pred: int, roi_gray: np.ndarray) -> int:
    """تمييز بين 6 و 9 حسب موضع الفتحة"""
    g = cv2.GaussianBlur(roi_gray, (3,3), 0)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 21, 2)
    if np.mean(th) > 127:
        th = 255 - th
    contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return pred
    holes_y = []
    for i, h in enumerate(hierarchy[0]):
        parent = h[3]
        if parent != -1:  # له أب → فتحة
            M = cv2.moments(contours[i])
            if M["m00"] != 0:
                cy = int(M["m01"]/M["m00"])
                holes_y.append(cy)
    if not holes_y:
        return pred
    mean_y = np.mean(holes_y)
    H = roi_gray.shape[0]
    if pred == 6 and mean_y < H/2: return 9
    if pred == 9 and mean_y >= H/2: return 6
    return pred

def segment_largest(gray):
    """أخذ أكبر كونتور فقط"""
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return [], clean
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return [(x,y,w,h)], clean

def segment_all(gray):
    """إرجاع جميع الأرقام (يسار → يمين)"""
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 50: continue
        boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: b[0])
    return boxes, clean



uploaded = st.file_uploader("اختر صورة (PNG/JPG)", type=["png","jpg","jpeg"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("تعذر قراءة الصورة.")
        st.stop()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if mode == "رقم واحد":
        boxes, binary = segment_largest(gray)
    else:
        boxes, binary = segment_all(gray)

    preds = []
    annotated = bgr.copy()

    for (x,y,w,h) in boxes:
        roi = gray[y:y+h, x:x+w]
        invert_needed = np.mean(roi) < 127
        img8 = to_8x8(roi, invert=invert_needed)
        flat = img8.reshape(1, -1)
        pred = int(model.predict(flat)[0])
        if pred in (6,9):
            pred = disambiguate_6_9(pred, roi)
        preds.append(pred)
        cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(annotated, str(pred), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

    st.subheader("النتيجة")
    if len(preds) == 0:
        st.warning("لم أتعرف على أرقام واضحة. جرّب صورة أوضح.")
    else:
        if mode == "رقم واحد":
            st.code(str(preds[0]), language="text")
        else:
            st.code("".join(str(p) for p in preds), language="text")

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="الصورة مع التوقعات")

    with st.expander("عرض الصورة الثنائية (للتشخيص)"):
        st.image(binary, caption="Binary View", clamp=True)
