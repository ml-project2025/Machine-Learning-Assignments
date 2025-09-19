import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import io
import base64
import os

from model import create_model

app = Flask(__name__)

# تحميل النموذج مرة واحدة عند بدء التشغيل
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '../saved_models/cnn_model.pth'

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(checkpoint['model_name'], checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    num_classes = checkpoint['num_classes']
    print(f"Model loaded with {num_classes} classes")
else:
    model = None
    num_classes = 10
    print("Warning: Model not found. Please train the model first.")

def preprocess_image(image):
    """معالجة الصورة للإدخال إلى النموذج"""
    # تحويل الصورة إلى تدرج رمادي وتغيير حجمها
    image = image.convert('L').resize((28, 28))
    
    # تحويل الصورة إلى مصفوفة numpy
    image_array = np.array(image)
    
    # عكس الألوان إذا كانت الخلفية بيضاء والأرقام سوداء
    if np.mean(image_array) > 127:
        image_array = 255 - image_array
    
    # تطبيع الصورة
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image_tensor = transform(image_array).unsqueeze(0).to(device)
    return image_tensor, image_array

def predict_image(image_tensor):
    """التنبؤ بالصورة"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not available. Please train the model first.'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # تحميل الصورة
        image = Image.open(io.BytesIO(file.read()))
        
        # معالجة الصورة
        image_tensor, processed_image = preprocess_image(image)
        
        # التنبؤ
        prediction, confidence, all_probabilities = predict_image(image_tensor)
        
        # إنشاء صورة للعرض
        plt.figure(figsize=(6, 6))
        plt.imshow(processed_image, cmap='gray')
        plt.axis('off')
        plt.title(f'Prediction: {prediction}, Confidence: {confidence:.2%}')
        
        # حفظ الصورة في buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # إعداد نتائج الاحتمالات
        class_probabilities = []
        for i, prob in enumerate(all_probabilities):
            class_probabilities.append({
                'class': i,
                'probability': float(prob)
            })
        
        # ترتيب الاحتمالات تنازلياً
        class_probabilities.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(confidence),
            'image': f'data:image/png;base64,{img_str}',
            'probabilities': class_probabilities[:5]  # أفضل 5 توقعات
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    os.makedirs('../saved_models', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
