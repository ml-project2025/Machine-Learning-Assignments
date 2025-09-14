
الاسم : عزّام عايض حمادي احمد ناصر
المجموعة : ب


# تكليف عزّام — التعرف على الأرقام داخل الصور (Skeleton)



# مشروع عزّام — التعرف على الأرقام داخل الصور (KNN + Streamlit)

## المميزات
- نموذج KNN (k-nearest neighbors).
- واجهة Streamlit فيها وضعين:
  1. رقم واحد (أكبر كونتور).
  2. أرقام متعددة (من اليسار لليمين).
- تحسين للتفريق بين 6 و 9.

## التشغيل
```bash
pip install -r requirements.txt
python src/main.py --train
streamlit run src/app_streamlit.py



# تدريب النموذج
python src/main.py --train

# التنبؤ برقم من صور الاختبار
python src/main.py --predict tests/sample_0.png
python src/main.py --predict tests/sample_1.png
python src/main.py --predict tests/sample_2.png
```
النتائج تُحفظ في مجلد `outputs/`.
