Integrated Digit & Letter Recognition System

عمل الطالب 
عمر سمير مرشد الاغبري

نظام متكامل للتعرف على الأرقام (0-9) والأحرف باستخدام الذكاء الاصطناعي وتقنيات التعلم العميق.

Features / المميزات

التعرف على الأرقام   

التعرف على الأحرف الإنجليزية (A-Z)

واجهة رسومية سهلة الاستخدام (GUI) للرسم والتنبؤ

إمكانية رسم رقم أو حرف في نفس اللوحة

حفظ الصور المرسومة وتحويلها تلقائيًا لمربعات 28x28

عرض دقة التدريب والتحقق ومخطط الخسارة والدقة


 إذا كانت بطاقة الرسوميات قديمة أو غير مدعومة، سيتم استخدام CPU تلقائيًا.

Installation / التثبيت

إنشاء بيئة افتراضية:

python -m venv myenv
source myenv/bin/activate   # على Linux / macOS
myenv\Scripts\activate      # على Windows


تثبيت المتطلبات:

pip install -r requirements.txt


تنزيل البيانات (EMNIST للأحرف وMNIST للأرقام) سيكون تلقائي عند أول تشغيل.

Usage / طريقة التشغيل
1️⃣ تشغيل التدريب الكامل:
python main.py --mode train

2️⃣ تشغيل التنبؤ:
python main.py --mode predict --image-path path_to_your_image.png

3️⃣ فتح الواجهة الرسومية (GUI):
python main.py --mode gui

4️⃣ تدريب النموذج يدويًا:
للأرقام فقط (0-9):
python train.py --model-type improved --epochs 15 --use-mnist-only

للأحرف فقط (A-Z):
python train.py --model-type improved --epochs 15 --use-letters-only

للأرقام والحروف معًا (0-9 + A-Z):
python train.py --model-type improved --epochs 20

5️⃣ استخدام GUI بعد التدريب:
python src/gui.py


يمكنك الآن رسم الأرقام أو الأحرف مباشرة على اللوحة، وحفظها، أو التنبؤ بها.

Saved Models & Images

النماذج المحفوظة: ./saved_models/best_digit_model.pth

الصور المرسومة: ./saved_images/

مخطط التدريب والخسارة: ./saved_models/training_history.png

نتائج التدريب التفصيلية: ./saved_models/training_results.json

Notes / ملاحظات

الصور المرسومة تُحول تلقائيًا إلى مربعات 28x28 قبل التنبؤ.

يمكن رسم أكثر من رقم أو حرف في نفس اللوحة، وسيتم حفظ الصورة الكبيرة أولًا، مع إمكانية فصل كل رقم باستخدام مكتبة OpenCV لاحقًا.

يوصى باستخدام EMNIST split letters لتدريب الأحرف بشكل منفصل، وbyclass لتدريب الأرقام والحروف معًا.