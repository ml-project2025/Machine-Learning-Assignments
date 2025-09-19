import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os

from model import create_model

def load_model(model_path, device='cpu'):
    """تحميل النموذج المدرب"""
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(checkpoint['model_name'], checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint['model_name'], checkpoint['num_classes']

def preprocess_image(image_path, device='cpu'):
    """معالجة الصورة للإدخال إلى النموذج"""
    # تحميل الصورة وتحويلها إلى تنسيق MNIST (28x28، تدرج رمادي، خلفية سوداء وأرقام بيضاء)
    image = Image.open(image_path).convert('L')
    
    # تغيير حجم الصورة إلى 28x28 بكسل
    image = image.resize((28, 28))
    
    # تحويل الصورة إلى مصفوفة numpy
    image_array = np.array(image)
    
    # عكس الألوان إذا كانت الخلفية بيضاء والأرقام سوداء (كما في MNIST)
    if np.mean(image_array) > 127:  # إذا كانت الخلفية فاتحة
        image_array = 255 - image_array
    
    # تطبيع الصورة
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image_tensor = transform(image_array).unsqueeze(0).to(device)
    return image_tensor, image_array

def predict_image(model, image_tensor, num_classes=10):
    """التنبؤ بالصورة"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item()

def main():
    # إعدادات
    model_path = '../saved_models/cnn_model.pth'
    image_path = '../data/sample_image.png'  # تغيير هذا إلى مسار صورتك
    
    # التأكد من وجود الصورة
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path.")
        return
    
    # تحديد الجهاز
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # تحميل النموذج
    model, model_name, num_classes = load_model(model_path, device)
    print(f"Loaded {model_name} model with {num_classes} classes")
    
    # معالجة الصورة والتنبؤ
    image_tensor, original_image = preprocess_image(image_path, device)
    prediction, confidence = predict_image(model, image_tensor, num_classes)
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    
    # إذا كان النموذج مدرباً على أكثر من 10 فئات (أرقام إضافية)
    if num_classes > 10:
        print(f"Note: Model trained on {num_classes} classes (0-9 and additional symbols)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict digit from image')
    parser.add_argument('--image', type=str, help='Path to the image file')
    args = parser.parse_args()
    
    if args.image:
        main(args.image)
    else:
        main()
