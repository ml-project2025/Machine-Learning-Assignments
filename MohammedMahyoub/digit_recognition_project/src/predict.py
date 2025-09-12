import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from model import DigitCNN, ImprovedDigitCNN

# ---------------------------
# تحويل رقم الفئة إلى رقم أو حرف
# ---------------------------
def index_to_label(idx):
    """
    0-9 -> digits
    10-35 -> letters A-Z
    """
    if idx < 10:
        return str(idx)
    else:
        return chr(65 + idx - 10)  # A-Z (ASCII 65-90)

class DigitPredictor:
    """
    فئة محسنة للتنبؤ بالأرقام والأحرف من الصور
    """
    def __init__(self, model_path, device=None):
        self.device = torch.device('cpu')
        self.model = self.load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if checkpoint['model_type'] == 'ImprovedDigitCNN':
            model = ImprovedDigitCNN()
        else:
            model = DigitCNN()
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((2,2), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        coords = np.where(image > 0)
        if len(coords[0]) > 0:
            top, left = np.min(coords[0]), np.min(coords[1])
            bottom, right = np.max(coords[0]), np.max(coords[1])
            h, w = image.shape
            padding = min(h, w) // 10
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(h, bottom + padding)
            right = min(w, right + padding)
            image = image[top:bottom, left:right]
            h, w = image.shape
            size = max(h, w)
            square_image = np.zeros((size, size), dtype=np.uint8)
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            square_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
            image = square_image
        
        pil_image = Image.fromarray(image)
        tensor_image = self.transform(pil_image)
        return tensor_image.unsqueeze(0)
    
    def predict_single_digit(self, image_path, show_confidence=True):
        input_tensor = self.preprocess_image(image_path)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_idx = predicted.item()
        predicted_label = index_to_label(predicted_idx)
        confidence_score = confidence.item()
        
        if show_confidence:
            print(f"Predicted: {predicted_label}")
            print(f"Confidence: {confidence_score:.3f}")
            probs_np = probabilities.cpu().numpy()[0]
            top_3 = np.argsort(probs_np)[-3:][::-1]
            print("Top 3 predictions:")
            for i, idx in enumerate(top_3):
                print(f"  {i+1}. Label {index_to_label(idx)}: {probs_np[idx]:.3f}")
        
        return predicted_label, confidence_score, probabilities.cpu().numpy()[0]
    
    def segment_digits_from_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image = cv2.GaussianBlur(image, (3, 3), 0)
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_contours = []
        min_area = 100
        max_area = image.shape[0] * image.shape[1] * 0.5
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w
            if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                continue
            if w < 10 or h < 15:
                continue
            digit_contours.append((x, y, w, h, area))
        
        digit_contours.sort(key=lambda x: x[0])
        extracted_digits = []
        
        for i, (x, y, w, h, area) in enumerate(digit_contours):
            padding = max(5, min(w, h) // 10)
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            digit_image = binary[y_start:y_end, x_start:x_end]
            
            if digit_image.size > 0:
                digit_h, digit_w = digit_image.shape
                size = max(digit_h, digit_w)
                square_digit = np.zeros((size, size), dtype=np.uint8)
                y_offset = (size - digit_h) // 2
                x_offset = (size - digit_w) // 2
                square_digit[y_offset:y_offset+digit_h, x_offset:x_offset+digit_w] = digit_image
                
                digit_path = f"temp_digit_{i}_{np.random.randint(1000,9999)}.png"
                cv2.imwrite(digit_path, square_digit)
                extracted_digits.append(digit_path)
        
        return extracted_digits
    
    def predict_multi_digit_number(self, image_path):
        try:
            digit_paths = self.segment_digits_from_image(image_path)
            if not digit_paths:
                print("No digits or letters found in the image")
                return None, []
            
            predicted_labels = []
            confidences = []
            
            for digit_path in digit_paths:
                try:
                    label, confidence, _ = self.predict_single_digit(digit_path, show_confidence=False)
                    predicted_labels.append(label)
                    confidences.append(confidence)
                    if os.path.exists(digit_path):
                        os.remove(digit_path)
                except:
                    if os.path.exists(digit_path):
                        os.remove(digit_path)
            
            if predicted_labels:
                final_label = ''.join(predicted_labels)
                avg_conf = np.mean(confidences)
                print(f"Predicted: {final_label} | Avg Confidence: {avg_conf:.3f}")
                return final_label, confidences
            else:
                return None, []
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, []
    
    def visualize_prediction(self, image_path):
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Could not load image from {image_path}")
            return
        
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")
        
        prediction, _ = self.predict_multi_digit_number(image_path)
        plt.subplot(1,2,2)
        plt.text(0.5, 0.5, f"Prediction: {prediction}", ha='center', va='center', fontsize=20)
        plt.axis('off')
        plt.title("Prediction")
        plt.show()

if __name__ == "__main__":
    predictor = DigitPredictor('./saved_models/best_digit_model.pth')
    predictor.visualize_prediction('test_image.png')
