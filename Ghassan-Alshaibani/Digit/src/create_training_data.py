#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'yan9yu'

import cv2
import numpy as np

# إنشاء بيانات تدريبية بسيطة للأرقام 0-9
def create_training_data():
    samples = []
    responses = []
    
    # إنشاء نماذج بسيطة للأرقام
    for digit in range(10):
        # إنشاء صورة 10x10 بسيطة لكل رقم
        img = np.zeros((10, 10), np.uint8)
        
        if digit == 0:
            cv2.circle(img, (5, 5), 3, 255, 1)
        elif digit == 1:
            cv2.line(img, (5, 1), (5, 9), 255, 1)
        elif digit == 2:
            cv2.line(img, (1, 2), (8, 2), 255, 1)
            cv2.line(img, (8, 2), (8, 5), 255, 1)
            cv2.line(img, (1, 5), (8, 5), 255, 1)
            cv2.line(img, (1, 5), (1, 8), 255, 1)
            cv2.line(img, (1, 8), (8, 8), 255, 1)
        elif digit == 3:
            cv2.line(img, (1, 2), (8, 2), 255, 1)
            cv2.line(img, (8, 2), (8, 8), 255, 1)
            cv2.line(img, (1, 5), (8, 5), 255, 1)
            cv2.line(img, (1, 8), (8, 8), 255, 1)
        elif digit == 4:
            cv2.line(img, (1, 2), (1, 5), 255, 1)
            cv2.line(img, (1, 5), (8, 5), 255, 1)
            cv2.line(img, (8, 2), (8, 8), 255, 1)
        elif digit == 5:
            cv2.line(img, (1, 2), (8, 2), 255, 1)
            cv2.line(img, (1, 2), (1, 5), 255, 1)
            cv2.line(img, (1, 5), (8, 5), 255, 1)
            cv2.line(img, (8, 5), (8, 8), 255, 1)
            cv2.line(img, (1, 8), (8, 8), 255, 1)
        elif digit == 6:
            cv2.line(img, (1, 2), (8, 2), 255, 1)
            cv2.line(img, (1, 2), (1, 8), 255, 1)
            cv2.line(img, (1, 5), (8, 5), 255, 1)
            cv2.line(img, (8, 5), (8, 8), 255, 1)
            cv2.line(img, (1, 8), (8, 8), 255, 1)
        elif digit == 7:
            cv2.line(img, (1, 2), (8, 2), 255, 1)
            cv2.line(img, (8, 2), (8, 8), 255, 1)
        elif digit == 8:
            cv2.line(img, (1, 2), (8, 2), 255, 1)
            cv2.line(img, (1, 2), (1, 8), 255, 1)
            cv2.line(img, (8, 2), (8, 8), 255, 1)
            cv2.line(img, (1, 5), (8, 5), 255, 1)
            cv2.line(img, (1, 8), (8, 8), 255, 1)
        elif digit == 9:
            cv2.line(img, (1, 2), (8, 2), 255, 1)
            cv2.line(img, (1, 2), (1, 5), 255, 1)
            cv2.line(img, (8, 2), (8, 8), 255, 1)
            cv2.line(img, (1, 5), (8, 5), 255, 1)
            cv2.line(img, (1, 8), (8, 8), 255, 1)
        
        # إضافة بعض التنويعات
        for _ in range(5):  # 5 نسخ من كل رقم
            sample = img.reshape((1, 100))
            samples.append(sample)
            responses.append(digit)
    
    samples_array = np.vstack(samples)
    return samples_array.astype(np.float32), np.array(responses, np.float32)

print("إنشاء بيانات التدريب...")
samples, responses = create_training_data()
responses = responses.reshape((responses.size, 1))

print(f"تم إنشاء {len(samples)} عينة تدريبية")

# حفظ البيانات
np.savetxt('../data/generalsamples.data', samples)
np.savetxt('../data/generalresponses.data', responses)

print("تم حفظ بيانات التدريب بنجاح!")
print("يمكنك الآن تشغيل test.py")
