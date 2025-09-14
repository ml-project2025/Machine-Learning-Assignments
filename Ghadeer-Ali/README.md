الاسم : غدير علي حسين صالح الزراعي
المجموعة : ب


# مشروع  — التعرف على الأرقام داخل الصور (SVM + Streamlit)



## المميزات
- نموذج SVM بدل KNN.
- واجهة Streamlit فيها وضعين:
  1. رقم واحد (أكبر كونتور).
  2. أرقام متعددة (من اليسار لليمين).
- تحسين للتفريق بين 6 و 9.

## التشغيل
```bash
pip install -r requirements.txt
python src/main.py --train
streamlit run src/app_streamlit.py
