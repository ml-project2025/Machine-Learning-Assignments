الاسم : فرح نبيل عتيق الصغير 
المجموعة : ب


# تكليف  — التعرف على الأرقام داخل الصور (RandomForest + Streamlit)

## المميزات
- نموذج RandomForest.
- واجهة Streamlit تدعم:
  - رقم واحد (أكبر كونتور).
  - أرقام متعددة (يسار → يمين).
  - رفع عدة صور ومعالجتها دفعة واحدة وتصدير CSV.
- تحسين للتفريق بين 6 و 9.

## التشغيل
```bash
pip install -r requirements.txt
python src/main.py --train
streamlit run src/app_streamlit.py
