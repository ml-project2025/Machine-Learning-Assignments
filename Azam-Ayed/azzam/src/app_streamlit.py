
# app_streamlit.py — قراءة "رقم واحد" من الصورة (مع تحسين الدقة والتفريق بين 6 و 9)

import streamlit as st
from pathlib import Path
import numpy as np
import cv2

from model import train_and_save, load_model
from preprocess import to_8x8

APP_TITLE = " قراءة رقم واحد من الصورة"
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "knn.joblib"

st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)
st.write("هذا الإصدار يتعامل مع صورة تحتوي على **رقم واحد فقط**: يلتقط أكبر خانة في الصورة، مع تحسين خاص للتفريق بين 6 و 9.")

with st.sidebar:
    st.header("إعداد النموذج")
    auto_train = st.checkbox("درّب تلقائيًا إذا لم يوجد نموذج", value=True)
    k_val = st.number_input("عدد الجيران (KNN)", min_value=1, max_value=15, value=3, step=1)
    if st.button("إعادة تدريب النموذج الآن"):
        info = train_and_save(str(MODEL_PATH), k=int(k_val))
        st.success(f"تم التدريب. الدقة: {info['accuracy']:.4f}")
        st.text(info["report"])


if not MODEL_PATH.exists():
    if auto_train:
        info = train_and_save(str(MODEL_PATH), k=int(k_val))
        st.info(f"تم تدريب نموذج جديد (لا يوجد نموذج محفوظ). الدقة: {info['accuracy']:.4f}")
    else:
        st.warning("لا يوجد نموذج محفوظ. فعّل التدريب التلقائي أو اضغط (إعادة تدريب).")
        st.stop()

model = load_model(str(MODEL_PATH))


def segment_biggest_digit(gray):
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return [], clean

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:  
        return [], clean

    x, y, w, h = cv2.boundingRect(c)
    return [(x, y, w, h)], clean

def disambiguate_6_9(pred: int, roi_gray: np.ndarray) -> int:
    g = cv2.GaussianBlur(roi_gray, (3,3), 0)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 21, 2)
    if np.mean(th) > 127:
        th = 255 - th

    contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return pred

    holes = []
    for i, h in enumerate(hierarchy[0]):
        parent = h[3]
        if parent != -1:
            M = cv2.moments(contours[i])
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                holes.append((cx, cy))

    if not holes:
        return pred

    h = roi_gray.shape[0]
    mean_y = np.mean([cy for _, cy in holes])

    if pred == 6 and mean_y < h/2:
        return 9
    if pred == 9 and mean_y >= h/2:
        return 6
    return pred

uploaded = st.file_uploader("اختر صورة (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("تعذّر قراءة الصورة.")
        st.stop()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    boxes, binary = segment_biggest_digit(gray)

    if not boxes:
        st.warning("ما قدرت ألقط أي رقم واضح في الصورة. جرّب صورة أوضح/أكبر.")
    else:
        x, y, w, h = boxes[0]
        roi = gray[y:y+h, x:x+w]

        invert_needed = np.mean(roi) < 127
        img8 = to_8x8(roi, invert=invert_needed)
        flat = img8.reshape(1, -1)
        pred = int(model.predict(flat)[0])

        if pred in (6, 9):
            pred = disambiguate_6_9(pred, roi)

        
        try:
            neighbors = model.kneighbors(flat, return_distance=False)
            votes = model._y[neighbors[0]]
            confidence = np.mean(votes == pred)
            st.write(f"الثقة: {confidence:.2f}")
            st.write(f"أصوات الجيران: {votes.tolist()}")
        except:
            pass

        annotated = bgr.copy()
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(annotated, str(pred), (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        st.subheader("النتيجة")
        st.code(str(pred), language="text")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="الصورة مع التوقع")
        with st.expander("عرض الصورة الثنائية (للتشخيص)"):
            st.image(binary, caption="Binary View", clamp=True)
