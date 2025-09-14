import streamlit as st
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

from model_rf import train_and_save, load_model
from preprocess import to_8x8

APP_TITLE = "فرح — واجهة قراءة الأرقام (RandomForest)"
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "rf.joblib"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("يدعم رقم واحد / أرقام متعددة / معالجة عدة صور دفعة واحدة، مع تحسين تمييز 6 و 9.")

with st.sidebar:
    st.header("إعداد النموذج")
    auto_train = st.checkbox("درّب تلقائيًا إذا لم يوجد نموذج", value=True)
    mode = st.radio("وضع القراءة", ["رقم واحد", "أرقام متعددة"], index=0)
    n_estimators = st.slider("RandomForest n_estimators", 50, 400, 200, step=50)
    max_depth = st.selectbox("max_depth", ["None", "8", "12", "16"], index=0)
    md = None if max_depth == "None" else int(max_depth)
    if st.button("إعادة تدريب الآن"):
        info = train_and_save(str(MODEL_PATH), n_estimators=int(n_estimators), max_depth=md)
        st.success(f"تم التدريب. الدقة: {info['accuracy']:.4f}")
        st.text(info["report"])

if not MODEL_PATH.exists():
    if auto_train:
        info = train_and_save(str(MODEL_PATH), n_estimators=int(n_estimators), max_depth=md)
        st.info(f"تم تدريب نموذج جديد (لا يوجد نموذج محفوظ). الدقة: {info['accuracy']:.4f}")
    else:
        st.warning("لا يوجد نموذج محفوظ. فعّل التدريب التلقائي أو اضغط إعادة تدريب.")
        st.stop()

model = load_model(str(MODEL_PATH))

def disambiguate_6_9(pred: int, roi_gray: np.ndarray) -> int:
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
        if parent != -1:
            M = cv2.moments(contours[i])
            if M["m00"] != 0:
                cy = int(M["m01"]/M["m00"])
                holes_y.append(cy)
    if not holes_y:
        return pred
    mean_y = np.mean(holes_y); H = roi_gray.shape[0]
    if pred == 6 and mean_y < H/2: return 9
    if pred == 9 and mean_y >= H/2: return 6
    return pred

def segment_largest(gray):
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

def predict_one_roi(roi_gray):
    invert_needed = np.mean(roi_gray) < 127
    img8 = to_8x8(roi_gray, invert=invert_needed)
    flat = img8.reshape(1, -1)
    pred = int(model.predict(flat)[0])
    if pred in (6,9):
        pred = disambiguate_6_9(pred, roi_gray)
    return pred, img8

def annotate_and_save(bgr, boxes, preds, out_name):
    annotated = bgr.copy()
    for (x,y,w,h), p in zip(boxes, preds):
        cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(annotated, str(p), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
    out_path = OUTPUTS_DIR / out_name
    cv2.imwrite(str(out_path), annotated)
    return out_path

tab1, tab2 = st.tabs([" صورة واحدة", " عدة صور (دفعة)"])

with tab1:
    uploaded = st.file_uploader("اختر صورة (PNG/JPG)", type=["png","jpg","jpeg"], key="single")
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("تعذّر قراءة الصورة."); st.stop()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes, binary = (segment_largest(gray) if mode=="رقم واحد" else segment_all(gray))

        if not boxes:
            st.warning("لم أتعرف على أرقام واضحة.")
        else:
            preds, img8s = [], []
            for (x,y,w,h) in boxes:
                roi = gray[y:y+h, x:x+w]
                p, img8 = predict_one_roi(roi)
                preds.append(p); img8s.append(img8)
            out_img = annotate_and_save(bgr, boxes, preds, f"farah_single_annotated.png")

            st.subheader("النتيجة")
            st.code(("".join(map(str,preds)) if len(preds)>1 else str(preds[0])), language="text")
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="الأصلية")
            st.image(cv2.cvtColor(cv2.imread(str(out_img)), cv2.COLOR_BGR2RGB), caption=f"المعلّمة (حُفظت: {out_img.name})")
            with st.expander("عرض الصورة الثنائية (للتشخيص)"):
                st.image(binary, caption="Binary View", clamp=True)

with tab2:
    uploaded_many = st.file_uploader("اختر عدة صور (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True, key="batch")
    if uploaded_many:
        rows = []
        for upl in uploaded_many:
            name = upl.name
            file_bytes = np.asarray(bytearray(upl.read()), dtype=np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if bgr is None:
                rows.append({"file": name, "pred": "read_error", "digits": ""})
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            boxes, _ = (segment_largest(gray) if mode=="رقم واحد" else segment_all(gray))
            preds = []
            for (x,y,w,h) in boxes:
                roi = gray[y:y+h, x:x+w]
                p, _ = predict_one_roi(roi)
                preds.append(p)
            out_img = annotate_and_save(bgr, boxes, preds, f"farah_batch_{name}.png")
            rows.append({"file": name, "pred": (preds[0] if len(preds)==1 else "".join(map(str,preds))), "annotated": out_img.name})

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        # حفظ CSV
        csv_path = OUTPUTS_DIR / "farah_batch_results.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        st.success(f"تم حفظ النتائج: {csv_path.name}")
