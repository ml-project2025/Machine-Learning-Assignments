import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import time
from tqdm import tqdm

from model import create_model

def load_data(batch_size=64, augment=True):
    """تحميل بيانات MNIST مع إمكانية التوسع"""
    transform_list = [transforms.ToTensor(), 
                     transforms.Normalize((0.1307,), (0.3081,))]
    
    if augment:
        transform_list = [transforms.RandomRotation(10),
                         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                         transforms.RandomResizedCrop(28, scale=(0.9, 1.1))] + transform_list
    
    transform = transforms.Compose(transform_list)
    
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001, device='cpu'):
    """تدريب النموذج"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # مرحلة التدريب
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in progress_bar:
            batch_count += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': running_loss / batch_count,
                'Accuracy': 100 * correct / total
            })
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # مرحلة التقييم
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
        scheduler.step()
    
    return train_losses, train_accuracies, test_losses, test_accuracies
def evaluate_model(model, test_loader, criterion, device='cpu'):
    """تقييم النموذج"""
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
    
    return test_loss, test_accuracy

def plot_results(train_losses, train_accuracies, test_losses, test_accuracies):
    """رسم نتائج التدريب"""
    # التأكد من وجود مجلد saved_models
    os.makedirs('../saved_models', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # رسم الخسارة
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(test_losses, label='Test Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # رسم الدقة
    axes[1].plot(train_accuracies, label='Train Accuracy')
    axes[1].plot(test_accuracies, label='Test Accuracy')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # حفظ الصورة
    save_path = '../saved_models/training_results.png'
    plt.savefig(save_path)
    print(f"Training results saved to {save_path}")
    plt.show()

def save_model(model, path, model_name='cnn', num_classes=10):
    """حفظ النموذج"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'num_classes': num_classes
    }, path)
    print(f"Model saved to {path}")

def main():
    # إعدادات التدريب
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.001
    model_name = 'cnn'  # 'cnn' أو 'extended_cnn'
    num_classes = 10
    
    # تحديد الجهاز (GPU إذا متوفر)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # تحميل البيانات
    train_loader, test_loader = load_data(batch_size)
    
    # إنشاء النموذج
    model = create_model(model_name, num_classes)
    print(f"Created {model_name} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # التدريب النتائج
    train_losses, train_accuracies, test_losses, test_accuracies = train_model(
        model, train_loader, test_loader, num_epochs, learning_rate, device
    )
    
    # رسم النتائج
    plot_results(train_losses, train_accuracies, test_losses, test_accuracies)
    
    # حفظ النتائج
    os.makedirs('../saved_models', exist_ok=True)
    model_path = f'../saved_models/{model_name}_model.pth'
    save_model(model, model_path, model_name, num_classes)
    
    print("Training completed!")

if __name__ == '__main__':
    main()
