# مشروع التعرف على الأرقام 
# للطالب : مجد عبدالرقيب احمد العزعزي
# IT/ Group B / 2

مشروع تفاعلي للتعرف على الأرقام المكتوبة يدويًا باستخدام **TensorFlow** و **Gradio**.

## مكونات المشروع
- `app.py`: الواجهة النهائية (رفع صورة أو رسم رقم).
- `train.py`: كود تدريب النموذج.
- python train.py --epochs 3 --batch-size 128 
## يعني أن النموذج سيتعلم 3 مرات على كامل البيانات --epochs 3 
## يحدد عدد العينات التي يتم تدريبها مرة واحدة --batch-size 128
- `predict_image.py`: تجربة التنبؤ على صورة مباشرة.
- python predict_image.py --image samples/three.png
- `models/`: يحتوي على النموذج المدرب `digit_cnn.keras`.
- `samples/`: عينات صور للاختبار.

## المتطلبات
```bash
pip install -r requirements.txt
```
**تشغيل يدوي**:
```
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python train.py --epochs 3 --batch-size 128
python app.py
```
**تشغيل سريع**:
1) افتح المجلد `Majd.A.Alazazi`
2) اضغط دبل-كليك على `run.bat`
3) انتظر التثبيت والتدريب، ستفتح واجهة الويب تلقائيًا
