# gui/prediction.py
import numpy as np
import cv2
import time
import torch
import os

def localize_digits(img):
    """Find potential digit regions in the image using connected components"""
    # Convert to numpy if PIL image
    if not isinstance(img, np.ndarray):
        img_np = np.array(img)
    else:
        img_np = img.copy()
    
    # AUTO INVERSION (for real-world images)
    if img_np.mean() > 128:  # Light background
        img_np = 255 - img_np
    
    # Threshold to binary
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find connected components
    components = cv2.connectedComponentsWithStats(
        binary, 
        connectivity=8, 
        ltype=cv2.CV_32S
    )
    
    # Extract component stats
    num_components = components[0]
    stats = components[2]  # [x, y, width, height, area]
    
    # Filter components to find potential digits
    digit_regions = []
    for i in range(1, num_components):  # Skip background (index 0)
        x, y, w, h, area = stats[i]
        
        # Filter by aspect ratio (digits are roughly square)
        aspect_ratio = w / float(h)
        if 0.2 < aspect_ratio < 1.5 and area > 50:  # Reasonable digit size
            # Add padding (10% on each side)
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)
            x = max(0, x - pad_w)
            y = max(0, y - pad_h)
            w = min(img_np.shape[1] - x, w + 2 * pad_w)
            h = min(img_np.shape[0] - y, h + 2 * pad_h)
            
            digit_regions.append((x, y, w, h))
    
    # Sort by x position (left to right)
    digit_regions.sort(key=lambda region: region[0])
    
    return digit_regions, img_np

def preprocess_digit_region(digit_img):
    """Preprocess a single digit region for recognition"""
    h, w = digit_img.shape
    if h > w:
        new_h = 20
        new_w = int(w * 20 / h)
        pad_w = (28 - new_w) // 2
        pad_h = 4
    else:
        new_w = 20
        new_h = int(h * 20 / w)
        pad_h = (28 - new_h) // 2
        pad_w = 4
    
    # Resize and pad
    resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    processed = np.pad(
        resized, 
        ((pad_h, pad_h), (pad_w, pad_w)), 
        mode='constant', 
        constant_values=0
    )
    
    return processed

def predict_single_digit(model, digit_img):
    """Predict a single digit from preprocessed image"""
    # Transform for model
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor = transform(digit_img).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

def predict_multiple_digits(model, image):
    """Predict multiple digits in a single image"""
    try:
        start_time = time.time()
        
        # Localize digit regions
        digit_regions, processed_img = localize_digits(image)
        
        if not digit_regions:
            return {
                "success": False,
                "error": "No digits found in image"
            }
        
        results = []
        for i, (x, y, w, h) in enumerate(digit_regions):
            # Extract digit region
            digit_img = processed_img[y:y+h, x:x+w]
            
            # Preprocess for recognition
            processed = preprocess_digit_region(digit_img)
            
            # Predict
            digit, confidence = predict_single_digit(model, processed)
            
            results.append({
                "digit": digit,
                "confidence": confidence,
                "position": (x, y, w, h)
            })
        
        total_time = time.time() - start_time
        return {
            "success": True,
            "digits": [r["digit"] for r in results],
            "confidence": [r["confidence"] for r in results],
            "sequence": ''.join(str(r["digit"]) for r in results),
            "time": total_time,
            "count": len(results),
            "regions": digit_regions
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def predict_digit(model, image):
    """Predict single digit (maintained for backward compatibility)"""
    result = predict_multiple_digits(model, image)
    if result["success"] and result["count"] == 1:
        return {
            "success": True,
            "digit": result["digits"][0],
            "confidence": result["confidence"][0],
            "time": result["time"]
        }
    elif not result["success"]:
        return result
    else:
        # For single-digit interface, return the most confident digit
        max_idx = result["confidence"].index(max(result["confidence"]))
        return {
            "success": True,
            "digit": result["digits"][max_idx],
            "confidence": result["confidence"][max_idx],
            "time": result["time"]
        }