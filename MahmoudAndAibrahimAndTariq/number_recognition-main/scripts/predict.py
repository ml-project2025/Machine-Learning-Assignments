# scripts/predict.py
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("DigitPredictor")

def preprocess_real_image(image_path):
    """Comprehensive preprocessing for real-world digit images"""
    logger.info(f"🔄 Processing: {image_path}")
    
    # 1. Load image (handle various formats)
    try:
        img = Image.open(image_path).convert('L')  # Grayscale
        logger.info("✅ Image loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load image: {str(e)}")
        return None
    
    # 2. Convert to numpy for OpenCV processing
    img_np = np.array(img)
    
    # 3. AUTO INVERSION (critical for real images)
    if img_np.mean() > 128:  # If background is bright (typical real-world)
        logger.info("🔄 Inverting image (black-on-white → white-on-black)")
        img_np = 255 - img_np
    
    # 4. Adaptive thresholding (handles uneven lighting)
    logger.info("🔄 Applying adaptive thresholding...")
    img_np = cv2.adaptiveThreshold(
        img_np, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 2
    )
    
    # 5. Find digit contours
    contours, _ = cv2.findContours(
        img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        logger.error("❌ No digit contours found")
        return None
    
    # 6. Find largest contour (assumed to be the digit)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 7. Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 8. Extract digit region
    digit_region = img_np[y:y+h, x:x+w]
    
    # 9. Add padding (mimics MNIST centering)
    padding = int(0.2 * max(w, h))  # 20% padding
    padded_digit = np.pad(
        digit_region, 
        ((padding, padding), (padding, padding)), 
        mode='constant', 
        constant_values=0
    )
    
    # 10. Resize to 28x28 (preserving aspect ratio)
    h, w = padded_digit.shape
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
    
    resized = cv2.resize(padded_digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    final = np.pad(resized, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    logger.info(f"✅ Preprocessing complete: {final.shape} → 28x28")
    return final

def predict_digit(image_path):
    """Predict digit from real-world image"""
    # --- Model Architecture ---
    class ImprovedCNN(torch.nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3, 1, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, 3, 1, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Dropout2d(0.25),
                torch.nn.Conv2d(32, 64, 3, 1, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, 1, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Dropout2d(0.25),
                torch.nn.Conv2d(64, 128, 3, 1, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Dropout2d(0.25)
            )
            
            # Calculate flattened size
            dummy_input = torch.zeros(1, 1, 28, 28)
            self.flattened_size = self.features(dummy_input).view(1, -1).shape[1]
            
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.flattened_size, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, 10)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # --- Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "digit_cnn_improved.pth")
    
    # --- Load Model ---
    try:
        model = ImprovedCNN(num_classes=10)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        logger.info(f"✅ Model loaded from: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        return None, None
    
    # --- Preprocess Image ---
    processed = preprocess_real_image(image_path)
    if processed is None:
        return None, None
    
    # --- Save processed image for debugging (optional) ---
    debug_dir = os.path.join(PROJECT_ROOT, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, "processed_" + os.path.basename(image_path))
    Image.fromarray(processed).save(debug_path)
    logger.info(f"📸 Saved processed image for debugging: {debug_path}")
    
    # --- Transform for model ---
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor = transform(processed).unsqueeze(0)  # Add batch dimension
    
    # --- Predict ---
    try:
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        digit = predicted.item()
        confidence_val = confidence.item()
        
        # Quality check: reject very low confidence
        if confidence_val < 0.5:
            logger.warning(f"⚠️ Low confidence ({confidence_val:.2%}) - result may be unreliable")
        
        return digit, confidence_val
    
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        return None, None

def main():
    print("\n" + "="*50)
    print("       REAL-WORLD DIGIT RECOGNITION")
    print("  (Handles black-on-white, off-center, noisy images)")
    print("="*50)
    
    while True:
        image_path = input("\nEnter path to digit image (or 'quit' to exit): ").strip()
        if image_path.lower() == 'quit':
            break
            
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue
        
        logger.info(f"\n📌 Processing: {os.path.basename(image_path)}")
        digit, confidence = predict_digit(image_path)
        
        if digit is not None:
            print("\n" + "-"*50)
            print(f"🎯 PREDICTION: {digit}")
            print(f"📊 CONFIDENCE: {confidence:.2%}")
            
            # Detailed feedback
            if confidence > 0.9:
                print("✅ HIGH CONFIDENCE - result is very reliable")
            elif confidence > 0.7:
                print("🟡 MEDIUM CONFIDENCE - result is likely correct")
            else:
                print("🔴 LOW CONFIDENCE - result may be wrong")
                print("💡 Tip: Try taking a clearer photo with better lighting")
            
            # Save to history
            history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_history.txt")
            with open(history_file, "a") as f:
                f.write(f"{os.path.basename(image_path)}\t{digit}\t{confidence:.4f}\n")
        else:
            print("❌ Prediction failed - check logs for details")

if __name__ == "__main__":
    main()