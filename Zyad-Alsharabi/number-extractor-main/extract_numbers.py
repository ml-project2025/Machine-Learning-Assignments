import cv2
import pytesseract
# اسم الطالب : زياد عبده سعيد ناصر
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_numbers(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    inverted = cv2.bitwise_not(gray)

    _, thresh = cv2.threshold(inverted, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    numbers = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:
            roi = thresh[y:y+h, x:x+w]
            text = pytesseract.image_to_string(
                roi,
                config='--psm 10 -c tessedit_char_whitelist=0123456789'
            ).strip()

            if text.isdigit():
                val = int(text)
                if 1 <= val <= 30:   
                    numbers.append((x, y, val))

    numbers_sorted = sorted(numbers, key=lambda item: (item[1]//50, item[0]))

    return [val for (_, _, val) in numbers_sorted]


if __name__ == "__main__":
    image_path = r"E:\numpy\imagess.jpg"     #ضع مسار الصورة هنا
    result = extract_numbers(image_path)
    print("الأرقام المستخرجة:", result)
