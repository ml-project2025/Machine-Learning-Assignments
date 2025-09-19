import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
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

def evaluate_on_test_set(model, test_loader, device='cpu'):
    """تقييم النموذج على مجموعة الاختبار"""
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    
    return test_loss, test_accuracy, all_preds, all_labels

def plot_confusion_matrix(all_labels, all_preds, num_classes=10):
    """رسم مصفوفة الالتباس باستخدام matplotlib فقط"""
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # إضافة القيم إلى الخلايا
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('../saved_models/confusion_matrix.png')
    plt.show()
    
    return cm

def main():
    # إعدادات
    batch_size = 64
    model_path = '../saved_models/cnn_model.pth'
    
    # تحديد الجهاز
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # تحميل النموذج
    model, model_name, num_classes = load_model(model_path, device)
    print(f"Loaded {model_name} model with {num_classes} classes")
    
    # تحميل بيانات الاختبار
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # تقييم النموذج
    test_loss, test_accuracy, all_preds, all_labels = evaluate_on_test_set(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # رسم مصفوفة الالتباس
    cm = plot_confusion_matrix(all_labels, all_preds, num_classes)
    
    # تقرير التصنيف
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    
    # عرض بعض الأمثلة
    show_examples(model, test_dataset, device, num_classes)

def show_examples(model, test_dataset, device, num_classes=10, num_examples=10):
    """عرض بعض الأمثلة مع توقعات النموذج"""
    indices = np.random.choice(len(test_dataset), num_examples, replace=False)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            predicted = predicted.item()
        
        plt.subplot(2, 5, i+1)
        plt.imshow(image.cpu().squeeze(), cmap='gray')
        plt.title(f'True: {label}, Pred: {predicted}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../saved_models/examples.png')
    plt.show()

if __name__ == '__main__':
    main()
