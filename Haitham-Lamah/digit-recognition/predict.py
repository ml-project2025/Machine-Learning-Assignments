# predict.py
import argparse, time, os
import numpy as np, cv2

# جرّب Keras المستقل ثم tf.keras
try:
    import keras
except ImportError:
    from tensorflow import keras

# Pillow لعرض يونيكود، و Tkinter للوحة الرسم
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk

AR_DIGITS = ['٠','١','٢','٣','٤','٥','٦','٧','٨','٩']
LATIN_DIGITS = ['0','1','2','3','4','5','6','7','8','9']

# ---------------------- وسائط التشغيل ----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Arabic-Indic digits live multi-digit prediction with drawing pad")
    ap.add_argument("--model", default="madbase_cnn.keras", help="مسار نموذج الأرقام العربية")
    ap.add_argument("--cam", type=int, default=0, help="فهرس الكاميرا")
    ap.add_argument("--invert", action="store_true", default=True, help="ابدأ وميزة العكس مفعّلة")
    ap.add_argument("--roi", type=int, default=220, help="حجم مربع الالتقاط بالبكسل")
    ap.add_argument("--fps", type=int, default=30, help="معدّل الالتقاط المطلوب")
    ap.add_argument("--min-area", type=int, default=80, help="أصغر مساحة للبقعة لتُعتبر رقمًا")
    ap.add_argument("--max-digits", type=int, default=8, help="الحد الأقصى لعدد الأرقام داخل ROI")
    ap.add_argument("--font", type=str, default="", help="مسار ملف خط .ttf لعرض الأرقام العربية")
    ap.add_argument("--ascii-ui", action="store_true", help="اعرض واجهة لاتينية فقط (حل سريع)")
    ap.add_argument("--min-conf-print", type=float, default=0.65, help="حدّ الثقة الأدنى للطباعة في التيرمنال")
    ap.add_argument("--print-interval", type=float, default=0.5, help="مهلة بالثواني بين طبعات التيرمنال")
    return ap.parse_args()

# ---------------------- أدوات عرض يونيكود ----------------------
def load_font(path, size=22):
    if not path or not os.path.exists(path):
        return None
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        return None

def draw_text_unicode(img_bgr, text, org, font, color=(255,255,255), stroke=(0,0,0), stroke_width=2):
    """يرسم Unicode على صورة BGR باستخدام Pillow ثم يرجعها BGR."""
    if font is None:
        return img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    x, y = org
    y = y - font.size // 3  # مواءمة بسيطة للخط الأساسي
    if stroke_width > 0:
        draw.text((x, y), text, font=font, fill=stroke, stroke_width=stroke_width, stroke_fill=stroke)
    draw.text((x, y), text, font=font, fill=color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ---------------------- تجهيز الصور 28x28 ----------------------
def resize_and_pad_to_28(img_bin):
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
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if invert:
        th = cv2.bitwise_not(th)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return th, gray

def extract_digit_boxes(th_bin, min_area=80, max_digits=8):
    contours, _ = cv2.findContours(th_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area:        # تجاهل الضجيج
            continue
        aspect = w / float(h + 1e-6)
        if h < 10 or w < 5:        # قصاصات صغيرة جدًا
            continue
        if aspect < 0.15 or aspect > 1.6:  # غريبة الشكل
            continue
        crop = th_bin[y:y+h, x:x+w]
        d28 = resize_and_pad_to_28(crop)
        boxes.append((x, y, w, h, d28))
    boxes.sort(key=lambda b: b[0])  # يسار→يمين
    return boxes[:max_digits]

def batch_predict(model, boxes):
    if not boxes:
        return [], [], None
    xs = []
    for (_,_,_,_, d28) in boxes:
        x = d28.astype("float32")/255.0
        x = np.expand_dims(x, axis=(0,-1))
        xs.append(x)
    X = np.vstack(xs)
    probs = model.predict(X, verbose=0)
    probs = np.asarray(probs)
    if probs.ndim == 3:
        probs = probs.reshape(probs.shape[0], probs.shape[-1])
    preds = np.argmax(probs, axis=1).astype(int).tolist()
    confs = probs[np.arange(len(preds)), preds].astype(float).tolist()
    return preds, confs, probs

# ---------------------- محولات نصية ----------------------
def latin_to_display(s, use_ascii):
    if use_ascii:
        return s
    return "".join(AR_DIGITS[int(ch)] for ch in s) if s else s

def preds_to_latin(preds):
    return "".join(LATIN_DIGITS[d] for d in preds)

# ---------------------- لوحة الرسم Tkinter ----------------------
def run_drawing_pad(model, use_ascii):
    """
    يفتح نافذة Tkinter بها Canvas للرسم (أسود/أبيض). Enter=Predict, C=Clear, Esc=Back.
    يغلق النافذة ثم يعود للتحكم إلى الحلقة الرئيسية.
    """
    root = tk.Tk()
    root.title("Draw a digit — LMB to draw | Enter=Predict | C=Clear | Esc=Back")

    W = H = 320
    brush = 18  # سمك القلم

    canvas = tk.Canvas(root, width=W, height=H, bg="black", highlightthickness=0)
    canvas.grid(row=0, column=0, columnspan=3, padx=8, pady=8)

    pil_img = Image.new("L", (W, H), 0)     # أسود
    pil_draw = ImageDraw.Draw(pil_img)

    result_var = tk.StringVar(value="Prediction: —")
    lbl = tk.Label(root, textvariable=result_var, font=("Arial", 14))
    lbl.grid(row=1, column=0, columnspan=3, pady=(0,8))

    last = [None, None]

    def to_display_digit(d):
        return (LATIN_DIGITS[d] if use_ascii else AR_DIGITS[d])

    def on_press(evt):
        last[0], last[1] = evt.x, evt.y

    def on_move(evt):
        if last[0] is None: return
        x0, y0 = last
        x1, y1 = evt.x, evt.y
        canvas.create_line(x0, y0, x1, y1, fill="white", width=brush, capstyle=tk.ROUND, smooth=True)
        pil_draw.line((x0, y0, x1, y1), fill=255, width=brush)
        last[0], last[1] = x1, y1

    def on_release(evt):
        last[0], last[1] = None, None

    def clear_pad(event=None):
        canvas.delete("all")
        pil_draw.rectangle((0,0,W,H), fill=0)
        result_var.set("Prediction: —")

    def predict_once(event=None):
        import numpy as np, cv2
        arr = np.array(pil_img)
        _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        d28 = resize_and_pad_to_28(th)
        x = d28.astype("float32")/255.0
        x = np.expand_dims(x, axis=(0,-1))  # (1,28,28,1)

        probs = model.predict(x, verbose=0)
        probs = np.asarray(probs)
        if probs.ndim == 3:
            probs = probs.reshape(probs.shape[0], probs.shape[-1])
        pred = int(np.argmax(probs[0]))
        conf = float(probs[0, pred])

        digit_txt = to_display_digit(pred)
        result_var.set(f"Prediction: {digit_txt}  ({conf*100:.0f}%)")
        print(f"[PAD] {digit_txt}  ({conf*100:.1f}%)", flush=True)

    def close_pad(event=None):
        root.destroy()

    # ربط الفأرة والمفاتيح
    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Return>", predict_once)
    root.bind("<Escape>", close_pad)
    root.bind("<Key-c>", clear_pad)
    root.bind("<Key-C>", clear_pad)

    # أزرار
    btn_predict = tk.Button(root, text="Predict (Enter)", command=predict_once)
    btn_clear   = tk.Button(root, text="Clear (C)", command=clear_pad)
    btn_close   = tk.Button(root, text="Close (Esc)", command=close_pad)
    btn_predict.grid(row=2, column=0, padx=8, pady=(0,8))
    btn_clear.grid(row=2, column=1, padx=8, pady=(0,8))
    btn_close.grid(row=2, column=2, padx=8, pady=(0,8))

    try:
        root.lift()
        root.attributes("-topmost", True)
        root.after(500, lambda: root.attributes("-topmost", False))
    except Exception:
        pass

    root.mainloop()

# ---------------------- الحلقة الرئيسية ----------------------
def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"غير موجود: {args.model}")
    model = keras.models.load_model(args.model)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("تعذّر فتح الكاميرا. جرّب --cam=1 أو 2")
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    font = load_font(args.font, size=22) if args.font else None
    use_ascii = args.ascii_ui or (font is None)

    roi_size = args.roi
    prev_time = time.time()

    # إعدادات الطباعة في التيرمنال
    last_printed = None
    last_print_time = 0.0
    PRINT_COOLDOWN = float(args.print_interval)
    MIN_CONF = float(args.min_conf_print)

    cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Inverter", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Invert", "Inverter", 1 if args.invert else 0, 1, lambda v: None)

    print("[q]=خروج | [i]=تبديل invert | [s]=حفظ | [h]=لوحة الرسم | التيرمنال يطبع عند تغيّر السلسلة")

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

        invert = (cv2.getTrackbarPos("Invert", "Inverter") == 1)

        th, gray_roi = preprocess_roi_for_multi(roi, invert=invert)
        boxes = extract_digit_boxes(th, min_area=args.min_area, max_digits=args.max_digits)
        preds, confs, _ = batch_predict(model, boxes)

        # إطار ROI
        cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)

        # صناديق الأرقام + تسميات
        for (box, d, cf) in zip(boxes, preds, confs):
            bx, by, bw, bh, _ = box
            X1, Y1 = x0 + bx, y0 + by
            X2, Y2 = X1 + bw, Y1 + bh
            cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 2)

            txt_digit = (LATIN_DIGITS[d] if use_ascii else AR_DIGITS[d]) + f" ({cf*100:.0f}%)"
            if use_ascii or font is None:
                cv2.putText(frame, txt_digit, (X1, max(0, Y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            else:
                frame = draw_text_unicode(
                    frame, txt_digit, (X1, max(0, Y1-6)),
                    font=font, color=(255,255,255), stroke=(0,0,0), stroke_width=2
                )

        # FPS
        fps = 1.0 / max(1e-6, (time.time()-prev_time)); prev_time=time.time()

        # السلسلة المتوقعة
        pred_latin_str = preds_to_latin(preds) if preds else ""
        display_pred   = latin_to_display(pred_latin_str, use_ascii)

        # طباعة في التيرمنال بشكل نظيف
        if preds:
            now = time.time()
            if (min(confs) >= MIN_CONF
                and display_pred != last_printed
                and (now - last_print_time) >= PRINT_COOLDOWN):
                print(f"{display_pred}    [" + ", ".join(f"{c*100:.0f}%" for c in confs) + "]", flush=True)
                last_printed = display_pred
                last_print_time = now

        # لوحة معلومات شفافة
        overlay = frame.copy()
        lines = [
            f"Pred : {display_pred if display_pred else '—'}",
            f"Count: {len(preds)}",
            f"FPS : {fps:4.1f}",
            f"Invert: {'ON' if invert else 'OFF'}",
            "Keys: q,i,s,h"
        ]
        cv2.rectangle(overlay, (10, 10), (360, 10 + 26*len(lines)), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        ytxt = 36
        for t in lines:
            if use_ascii or font is None:
                cv2.putText(frame, t, (18, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            else:
                frame = draw_text_unicode(frame, t, (18, ytxt), font=font, color=(255,255,255), stroke=(0,0,0), stroke_width=2)
            ytxt += 26

        # إظهار النوافذ
        cv2.imshow("Prediction", frame)
        left  = cv2.resize(cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR), (280,280), interpolation=cv2.INTER_AREA)
        right = cv2.resize(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR),       (280,280), interpolation=cv2.INTER_NEAREST)
        panel = cv2.hconcat([left, right])
        cv2.putText(panel, "ROI (gray)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(panel, f"Processed (invert={'ON' if invert else 'OFF'})", (300, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Inverter", panel)

        # مفاتيح
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            pos = cv2.getTrackbarPos("Invert", "Inverter")
            cv2.setTrackbarPos("Invert", "Inverter", 0 if pos==1 else 1)
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"roi_{ts}.png", roi)
            cv2.imwrite(f"roi_processed_{ts}.png", right)
            for i, (_,_,_,_, d28) in enumerate(boxes):
                cv2.imwrite(f"digit_{ts}_{i}.png", d28)
            print(f"Saved: roi_{ts}.png , roi_processed_{ts}.png , and {len(boxes)} digit crops")
        elif key == ord('h') or key == ord('H'):
            # أغلق الكاميرا مؤقتًا وافتح لوحة الرسم
            cap.release()
            cv2.destroyAllWindows()
            run_drawing_pad(model, use_ascii)
            # ارجع وافتح الكاميرا والنوافذ
            cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError("تعذّر إعادة فتح الكاميرا بعد لوحة الرسم.")
            cap.set(cv2.CAP_PROP_FPS, args.fps)
            cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Inverter",  cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Invert", "Inverter", 1 if args.invert else 0, 1, lambda v: None)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
