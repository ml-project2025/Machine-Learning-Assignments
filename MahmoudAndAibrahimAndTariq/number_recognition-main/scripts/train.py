# scripts/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# Add project root to path so we can import src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Custom transforms that work directly with PyTorch tensors (NO NUMPY)
class Focus38Differences:
    """Simple tensor-based transform for 3/8 differentiation"""
    def __call__(self, img):
        # Directly modify the tensor (works with float32)
        img[:, 10:18, :] *= 1.5
        return img.clamp(0, 1)

class EnhanceTwoTail:
    """Simple tensor-based transform for 2/0 differentiation"""
    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        img[:, h//2:, :w//3] *= 1.4
        return img.clamp(0, 1)

class Differentiate35:
    """Simple tensor-based transform for 3/5 differentiation"""
    def __call__(self, img):
        h = img.shape[1]
        img[:, :h//3, :] *= 1.3  # Boost top
        img[:, h//2:, :] *= 1.2  # Boost bottom
        return img.clamp(0, 1)

def train_model():
    # --- Configuration ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 128
    EPOCHS = 25
    LR = 0.001
    MOMENTUM = 0.9
    
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "train")
    VAL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "val")
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "digit_cnn_improved.pth")
    
    # Create models directory
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # --- Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "training_improved.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("Trainer")
    logger.info(f"Using device: {DEVICE}")
    
    # --- Data Transforms (CORRECT ORDER) ---
    # ToTensor MUST be first - converts PIL Image to float32 tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to float32 tensor (critical!)
        Focus38Differences(),
        EnhanceTwoTail(),
        Differentiate35(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Works with float32
        transforms.RandomRotation(8, fill=0),
        transforms.RandomAffine(0, shear=(-5, 5), fill=0),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0)
    ])
    
    # --- Load Data ---
    from src.data.dataset import NumberDataset
    train_dataset = NumberDataset(TRAIN_DIR, transform=transform)
    val_dataset = NumberDataset(VAL_DIR, transform=transform)
    
    # Remove pin_memory for CPU training (fixes warning)
    num_workers = 2 if DEVICE.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")
    
    # --- Model Architecture (Improved) ---
    class ImprovedCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(ImprovedCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                
                nn.Conv2d(32, 64, 3, 1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                
                nn.Conv2d(64, 128, 3, 1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25)
            )
            
            # Calculate flattened size
            dummy_input = torch.zeros(1, 1, 28, 28)
            self.flattened_size = self.features(dummy_input).view(1, -1).shape[1]
            
            self.classifier = nn.Sequential(
                nn.Linear(self.flattened_size, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # --- Model, Loss, Optimizer ---
    model = ImprovedCNN(num_classes=10).to(DEVICE)
    
    # Load previous weights (optional)
    try:
        model.load_state_dict(torch.load(
            os.path.join(PROJECT_ROOT, "models", "digit_cnn.pth"),
            map_location=DEVICE
        ), strict=False)
        logger.info("✅ Loaded previous model weights for fine-tuning")
    except:
        logger.info("Intialized new model")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=5e-4)
    
    # LR Scheduler (works with all PyTorch versions)
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
    
    # --- Training Loop ---
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        for data, target in progress_bar:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_acc)
        
        # --- Save Best Model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"✅ Best model saved with accuracy: {val_acc:.2f}%")
        
        # --- Log Results ---
        logger.info(f'Epoch {epoch+1}: '
                   f'Train Loss: {epoch_train_loss:.4f}, '
                   f'Val Loss: {epoch_val_loss:.4f}, '
                   f'Val Acc: {val_acc:.2f}% '
                   f'(Best: {best_val_acc:.2f}%)')
    
    # --- Final Results ---
    logger.info(f"🎉 Training completed!")
    logger.info(f"📊 Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    logger.info(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"💾 Model saved to: {MODEL_SAVE_PATH}")
    
    # --- Plot Training Curves ---
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='green')
    plt.title(f'Training Accuracy (Best: {best_val_acc:.2f}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "logs", "training_curves_improved.png"))
    plt.show()
    
    return model, best_val_acc

if __name__ == "__main__":
    print("🚀 Starting improved training with targeted fixes...")
    model, accuracy = train_model()
    print(f"✅ Training finished! Best accuracy: {accuracy:.2f}%")