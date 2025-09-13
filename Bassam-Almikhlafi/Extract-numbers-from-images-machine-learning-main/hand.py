#pip install easyocr

import easyocr
import re

reader = easyocr.Reader(['ar', 'en'])

results = reader.readtext(
    "images.png",
    detail=0
)

text = " ".join(results)

numbers = re.findall(r'[0-9]+', text)
print("Numbers Extracted:", numbers)

