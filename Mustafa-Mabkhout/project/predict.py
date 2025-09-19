import argparse, time, os
import numpy as np, cv2

# جرّب Keras المستقل ثم tf.keras
try:
    import keras
except ImportError:
    from tensorflow import keras

EN_DIGITS = ['0','1','2','3','4','5','6','7','8','9']

def parse_args():
    ap = argparse.ArgumentParser(description="English-Indic digits live multi-digit prediction with dual windows")
    ap.add_argument("--model", default="mnist_cnn.keras", help="مسار نموذج الأرقام الإنجليزية")
    ap.add_argument("--cam", type=int, default=0, help="فهرس الكاميرا")
    ap.add_argument("--invert", action="store_true", default=True, help="ابدأ وميزة العكس مفعّلة")
    ap.add_argument("--roi", type=int, default=220, help="حجم مربع الالتقاط بالبكسل")
    ap.add_argument("--fps", type=int, default=30, help="معدّل الالتقاط المطلوب")
    ap.add_argument("--min-area", type=int, default=80, help="أصغر مساحة للبقعة لتُعتبر رقمًا")
    ap.add_argument("--max-digits", type=int, default=8, help="الحد الأقصى لعدد الأرقام داخل ROI")
    return ap.parse_args()

def resize_and_pad_to_28(img_bin):
    # تعيير قصاصة ثنائية لرقم واحد إلى 28×28 مع حشو حول الرقم
    cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((28,28), dtype=np.uint8)
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    pad = max(4, int(0.15 * max(w, h)))
    x0, y0 = max(x-pad, 0), max(y-pad, 0)
    x1, y1 = min(x+w+pad, img_bin.shape[1]), min(y+h+pad, img_bin.shape[0])
    digit = img_bin[y0:y1, x0:x1]
    h, w = digit.shape
    if h==0 or w==0:
        return np.zeros((28,28), dtype=np.uint8)

    # حافظ على النسبة: قص إلى 20 بكسل في البعد الأكبر ثم وسّط على لوحة 28×28
    if h > w:
        new_h, new_w = 20, max(1, int(w * (20.0/h)))
    else:
        new_w, new_h = 20, max(1, int(h * (20.0/w)))
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28,28), dtype=np.uint8)
    sx, sy = (28-new_w)//2, (28-new_h)//2
    canvas[sy:sy+new_h, sx:sx+new_w] = digit_resized
    return canvas

def preprocess_roi_for_multi(roi_bgr, invert=True):
    # يحضّر صورتين: رمادية + ثنائية (مع أوتسو + اختيار العكس)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if invert:
        th = cv2.bitwise_not(th)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return th, gray

def extract_digit_boxes(th_bin, min_area=80, max_digits=8):
    # يعثر على مكونات متصلة داخل الـROI ويعيد صناديقها مقصوصة
    contours, _ = cv2.findContours(th_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area:  # تجاهل الضجيج الصغير
            continue
        aspect = w / float(h + 1e-6)
        if h < 10 or w < 5:
            continue
        if aspect < 0.15 or aspect > 1.6:
            continue
        crop = th_bin[y:y+h, x:x+w]
        d28 = resize_and_pad_to_28(crop)
        boxes.append((x, y, w, h, d28))
    # يسار -> يمين
    boxes.sort(key=lambda b: b[0])
    return boxes[:max_digits]

def batch_predict(model, boxes):
    if not boxes:
        return [], [], None
    xs = []
    for (_,_,_,_, d28) in boxes:
        x = d28.astype("float32")/255.0
        x = np.expand_dims(x, axis=(0,-1))  # (1,28,28,1)
        xs.append(x)
    X = np.vstack(xs)  # (N,28,28,1)
    probs = model.predict(X, verbose=0)    # (N,10) غالبًا
    probs = np.asarray(probs)
    if probs.ndim == 3:  # (N,1,10) مثلاً
        probs = probs.reshape(probs.shape[0], probs.shape[-1])
    preds = np.argmax(probs, axis=1).astype(int).tolist()
    confs = probs[np.arange(len(preds)), preds].astype(float).tolist()
    return preds, confs, probs

def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"غير موجود: {args.model}")
    model = keras.models.load_model(args.model)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("تعذّر فتح الكاميرا. جرّب --cam=1 أو 2")
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    roi_size = args.roi
    prev_time = time.time()

    # نافذتان
    cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Inverter", cv2.WINDOW_NORMAL)

    # سلايدر للتحكم بالـ invert في نافذة Inverter (0/1)
    initial_invert = 1 if args.invert else 0
    cv2.createTrackbar("Invert", "Inverter", initial_invert, 1, lambda v: None)

    print("[q]=خروج | [i]=تبديل invert | [s]=حفظ ROI والقصاصات")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("تعذّر قراءة إطار."); break

        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        half = roi_size//2
        x0, y0 = max(cx-half,0), max(cy-half,0)
        x1, y1 = min(cx+half,w), min(cy+half,h)
        roi = frame[y0:y1, x0:x1].copy()

        # اجلب حالة السلايدر (العكس)
        invert = (cv2.getTrackbarPos("Invert", "Inverter") == 1)

        # معالجة متعددة الأرقام
        th, gray_roi = preprocess_roi_for_multi(roi, invert=invert)
        boxes = extract_digit_boxes(th, min_area=args.min_area, max_digits=args.max_digits)
        preds, confs, _ = batch_predict(model, boxes)

        # -------- نافذة Prediction --------
        cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
        # ارسم صناديق الأرقام داخل إطار الكاميرا
        for (box, d, cf) in zip(boxes, preds, confs):
            bx, by, bw, bh, _ = box
            X1, Y1 = x0 + bx, y0 + by
            X2, Y2 = X1 + bw, Y1 + bh
            cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 2)
            lbl = f"{EN_DIGITS[d]} ({cf*100:.0f}%)"
            cv2.putText(frame, lbl, (X1, max(0, Y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        fps = 1.0 / max(1e-6, (time.time()-prev_time)); prev_time=time.time()
        ar_string = "".join(EN_DIGITS[d] for d in preds) if preds else "—"

        lines = [
            f"Digits: {ar_string}",
            f"Count: {len(preds)}",
            f"FPS : {fps:4.1f}",
            f"Invert: {'ON' if invert else 'OFF'}",
            "Keys: q/i/s"
        ]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (330, 10 + 22*len(lines)), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        ytxt = 30
        for t in lines:
            cv2.putText(frame, t, (20, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
            ytxt += 22
        cv2.imshow("Prediction", frame)

        # -------- نافذة Inverter --------
        # يسار: ROI رمادي — يمين: بعد threshold(+invert)
        left  = cv2.resize(gray_roi, (280,280), interpolation=cv2.INTER_AREA)
        right = cv2.resize(th,       (280,280), interpolation=cv2.INTER_NEAREST)
        left_bgr  = cv2.cvtColor(left,  cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        panel = cv2.hconcat([left_bgr, right_bgr])

        cv2.putText(panel, "ROI (gray)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(panel, f"Processed (invert={'ON' if invert else 'OFF'})", (300, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Inverter", panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            # بدّل السلايدر أيضًا
            pos = cv2.getTrackbarPos("Invert", "Inverter")
            cv2.setTrackbarPos("Invert", "Inverter", 0 if pos==1 else 1)
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"roi_{ts}.png", roi)
            cv2.imwrite(f"roi_processed_{ts}.png", right)
            # احفظ كل قصاصة 28×28
            for i, (_,_,_,_, d28) in enumerate(boxes):
                cv2.imwrite(f"digit_{ts}_{i}.png", d28)
            print(f"Saved: roi_{ts}.png , roi_processed_{ts}.png , and {len(boxes)} digit crops")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
