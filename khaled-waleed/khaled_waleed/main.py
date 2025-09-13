import easyocr, re

reader = easyocr.Reader(['en'])

image_paths = [r'D:\pic\number.jpg', r'D:\pic\77.jpg']

for path in image_paths:
    print(f"\nPhoto: {path}")
    
    results = reader.readtext(path, detail=0, allowlist='0123456789')
    all_numbers = ''.join(re.findall(r'\d+', ''.join(results)))
    
    if not all_numbers:
        print("There is no number")
        continue
    
    n = int(len(all_numbers) ** 0.5) + 1
    for i in range(0, len(all_numbers), n):
        print(' '.join(all_numbers[i:i+n]))
        
