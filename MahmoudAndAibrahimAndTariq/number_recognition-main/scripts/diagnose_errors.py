# scripts/diagnose_errors.py
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

def diagnose_model_errors():
    # Paths
    MODEL_PATH = r"D:\Python_Projects\number_recognition\number_recognition\models\digit_cnn.pth"
    TEST_DIR = r"D:\Python_Projects\number_recognition\number_recognition\data\processed\test"
    
    # Lightweight dataset class (no external dependencies)
    class NumberDataset:
        def __init__(self, data_dir):
            self.samples = []
            for label in range(10):
                class_dir = os.path.join(data_dir, str(label))
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(class_dir, img_name), label))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            from PIL import Image
            image = Image.open(img_path).convert('L')
            return transforms.ToTensor()(image), label
    
    # Transform (must match training)
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load test data
    test_dataset = NumberDataset(TEST_DIR)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model architecture (must match training)
    class SimpleCNN(torch.nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, 1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Dropout2d(0.25),
                torch.nn.Conv2d(64, 64, 3, 1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Dropout2d(0.25),
            )
            dummy_input = torch.zeros(1, 1, 28, 28)
            self.flattened_size = self.features(dummy_input).view(1, -1).shape[1]
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.flattened_size, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # Load model
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    # Evaluate with error tracking
    all_preds = []
    all_labels = []
    error_samples = []  # Store problematic samples
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            # Track errors
            errors = predicted != target
            if errors.any():
                for i in range(len(errors)):
                    if errors[i]:
                        idx = batch_idx * 64 + i
                        if idx < len(test_dataset):
                            error_samples.append((
                                test_dataset[idx][0],  # Raw image tensor
                                target[i].item(),
                                predicted[i].item()
                            ))
            
            all_preds.extend(predicted.numpy())
            all_labels.extend(target.numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print problematic digit pairs
    print("\n🔍 MOST COMMON ERRORS:")
    error_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                error_pairs.append((i, j, cm[i, j]))
    
    # Sort by most frequent errors
    error_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for true_digit, pred_digit, count in error_pairs[:5]:
        print(f"  • {true_digit} misclassified as {pred_digit}: {count} times")
    
    # Save error examples
    if error_samples:
        print(f"\n📸 Saving {min(10, len(error_samples))} error examples for analysis...")
        plt.figure(figsize=(15, 3))
        
        for i, (img_tensor, true_label, pred_label) in enumerate(error_samples[:10]):
            plt.subplot(2, 5, i+1)
            # Convert tensor to displayable format
            img = img_tensor.squeeze().numpy()
            img = (img * 0.3081 + 0.1307) * 255  # Reverse normalization
            plt.imshow(img, cmap='gray')
            plt.title(f"True: {true_label}, Pred: {pred_label}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('error_examples.png')
        print("✅ Error examples saved to 'error_examples.png'")
    
    return error_pairs[:3]  # Return top 3 problem pairs

if __name__ == "__main__":
    print("🔍 Diagnosing model errors...")
    top_errors = diagnose_model_errors()
    
    if top_errors:
        print("\n💡 Recommended fixes based on errors:")
        for true_digit, pred_digit, _ in top_errors:
            if {true_digit, pred_digit} == {3, 8}:
                print("  • 3/8 confusion: Add rotation and thinning augmentation")
            elif {true_digit, pred_digit} == {5, 6}:
                print("  • 5/6 confusion: Focus on top stroke differentiation")
            elif {true_digit, pred_digit} == {4, 9}:
                print("  • 4/9 confusion: Add closed-top augmentation")
            elif {true_digit, pred_digit} == {1, 7}:
                print("  • 1/7 confusion: Add crossbar augmentation for 7s")
            elif {true_digit, pred_digit} == {0, 6}:
                print("  • 0/6 confusion: Add center dot augmentation for 0s")
    else:
        print("✅ No significant errors found!")