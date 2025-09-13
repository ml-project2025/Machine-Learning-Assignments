# استيراد المكتبات
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# قراءة البيانات
data = pd.read_csv('grades.csv')

# عرض البيانات
print("Data:\n", data)



# حساب المتوسط لكل طالب
data['Average'] = data[['Math', 'Physics', 'Chemistry']].mean(axis=1)
print("\nData with Average:\n", data)
print("Highest grade in Math:", data['Math'].max())
print("Lowest grade in Physics:", data['Physics'].min())
best_math_student = data.loc[data['Math'].idxmax(), 'Name']
print("Best student in Math:", best_math_student)

# رسم مخطط درجات كل طالب
students = data['Name']
subjects = ['Math', 'Physics', 'Chemistry']
x = np.arange(len(students))  # مواقع الطلاب على المحور X
width = 0.25  # عرض الأعمدة

plt.bar(x - width, data['Math'], width, label='Math')
plt.bar(x, data['Physics'], width, label='Physics')
plt.bar(x + width, data['Chemistry'], width, label='Chemistry')

plt.xticks(x, students)
plt.ylabel('Grades')
plt.title('Students Grades by Subject')
plt.ylim(0, 100)
plt.legend()
plt.show()
