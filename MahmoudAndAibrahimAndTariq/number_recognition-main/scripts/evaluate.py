# scripts/evaluate.py
import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import logging

def evaluate_model():
    # --- Configuration ---
    MODEL_PATH = r"D:\Python_Projects\number_recognition\number_recognition\models\digit_cnn_improved.pth"
    TEST_DIR = r"D:\Python_Projects\number_recognition\number_recognition\data\processed\test"
    OUTPUT_DIR = r"D:\Python_Projects\number_recognition\number_recognition\evaluation_results"
    
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("Evaluator")
    
    # --- Device Setup ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")
    
    # --- Custom Transforms (MUST MATCH TRAINING) ---
    class Focus38Differences:
        def __call__(self, img):
            img[:, 10:18, :] *= 1.5
            return img.clamp(0, 1)

    class EnhanceTwoTail:
        def __call__(self, img):
            h, w = img.shape[1], img.shape[2]
            img[:, h//2:, :w//3] *= 1.4
            return img.clamp(0, 1)

    class Differentiate35:
        def __call__(self, img):
            h = img.shape[1]
            img[:, :h//3, :] *= 1.3
            img[:, h//2:, :] *= 1.2
            return img.clamp(0, 1)
    
    # --- Data Transform (EXACTLY MATCHES TRAINING) ---
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        Focus38Differences(),
        EnhanceTwoTail(),
        Differentiate35(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
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
    
    # --- Load Model ---
    model = ImprovedCNN(num_classes=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    logger.info(f"✅ Loaded model: {MODEL_PATH}")
    
    # --- Dataset Class ---
    class NumberDataset:
        def __init__(self, data_dir):
            self.samples = []
            for label in range(10):
                class_dir = os.path.join(data_dir, str(label))
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((
                                os.path.join(class_dir, img_name),
                                label
                            ))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            from PIL import Image
            image = Image.open(img_path).convert('L')
            return transform(image), label  # IMPORTANT: Use the SAME transform as training
    
    # --- Load Test Data ---
    test_dataset = NumberDataset(TEST_DIR)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    logger.info(f"✅ Loaded test dataset: {len(test_dataset)} samples")
    
    # --- Evaluation ---
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # --- Metrics ---
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    # --- Save Results ---
    # 1. Save accuracy
    with open(os.path.join(OUTPUT_DIR, "accuracy.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Correct: {np.sum(np.array(all_preds) == np.array(all_labels))}\n")
        f.write(f"Total: {len(all_labels)}\n")
    
    # 2. Save classification report
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    
    # 3. Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    
    # --- Final Report ---
    logger.info("\n" + "="*50)
    logger.info("       FINAL EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Total samples: {len(all_labels)}")
    logger.info(f"Correct predictions: {np.sum(np.array(all_preds) == np.array(all_labels))}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("="*50)

if __name__ == "__main__":
    print("🔍 Starting model evaluation...")
    evaluate_model()
    print("✅ Evaluation completed successfully!")