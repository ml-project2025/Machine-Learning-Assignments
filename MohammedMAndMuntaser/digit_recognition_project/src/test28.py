from PIL import Image
import numpy as np
# فتح الصورة وتحويلها إلى رمادي
img = Image.open("test28.png").convert("L")  # L = grayscale
# تحويلها لمصفوفة numpy
img_array = np.array(img)
print(img_array.shape)  # يجب أن يكون (28, 28)
