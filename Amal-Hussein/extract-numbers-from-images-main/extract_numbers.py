import easyocr
import cv2
import matplotlib.pyplot as plt
import re

# إنشاء القارئ
reader = easyocr.Reader(['en'])

# تحميل الصورة
image_path = "test_image.png"   # غيّرها لاسم صورتك
img = cv2.imread(image_path)

# استخراج النصوص
results = reader.readtext(img)

# أخذ الأرقام فقط
numbers = []
for (_, text, _) in results:
    nums = re.findall(r"\d+(?:[.,]\d+)?", text)
    numbers.extend(nums)

# ✅ طباعة الأرقام في الـ Terminal
print("الأرقام المستخرجة:")
for num in numbers:
    print(num)

# ✅ عرض الصورة والأرقام (نافذة صغيرة والأرقام أوضح)
fig, axes = plt.subplots(2, 1, figsize=(4, 5), gridspec_kw={"height_ratios": [3, 1]})

# الصورة
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].axis("off")
axes[0].set_title("الصورة الأصلية", fontsize=12)

# الأرقام المستخرجة (بخط كبير وواضح)
axes[1].axis("off")
output_text = "\n".join(numbers)
axes[1].text(0.5, 0.5, output_text, fontsize=20, ha="center", va="center", color="black", fontweight="bold")

plt.tight_layout()
plt.show()
