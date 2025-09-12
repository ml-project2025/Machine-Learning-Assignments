#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'yan9yu'

import sys
import numpy as np
import cv2

# قراءة صورة التدريب
im = cv2.imread('../data/train.png')
if im is None:
    print("خطأ: لا يمكن قراءة صورة التدريب")
    sys.exit(1)

im3 = im.copy()
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# إيجاد الـ contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 100), np.float32)
responses = []

print(f"تم العثور على {len(contours)} contours")

# معالجة كل contour
for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)
        
        if h > 28:  # فقط الأرقام الكبيرة
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            
            # عرض الصورة للمستخدم
            cv2.imshow('Digit to Label', roismall)
            cv2.imshow('Original Image', im)
            
            print(f"Contour {i}: اضغط على رقم (0-9) لتسمية هذا الرقم، أو 'q' للخروج")
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif 48 <= key <= 57:  # أرقام 0-9
                digit = key - 48
                responses.append(digit)
                sample = roismall.reshape((1, 100))
                samples = np.append(samples, sample, 0)
                print(f"تم تسمية الرقم: {digit}")
                
                # رسم مستطيل أحمر حول الرقم المسمى
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                print("مفتاح غير صحيح، تم تجاهل هذا الرقم")

cv2.destroyAllWindows()

if len(responses) == 0:
    print("لم يتم تدريب أي أرقام!")
    sys.exit(1)

# تحويل البيانات إلى الشكل المطلوب
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))

print(f"تم تدريب {len(responses)} رقم")
print("training complete")

# حفظ البيانات
samples = np.float32(samples)
responses = np.float32(responses)

cv2.imwrite("../data/train_result.png", im)
np.savetxt('../data/generalsamples.data', samples)
np.savetxt('../data/generalresponses.data', responses)

print("تم حفظ بيانات التدريب بنجاح!")
