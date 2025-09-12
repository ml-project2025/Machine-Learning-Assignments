import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = correct = total = 0
    
    for data, target in tqdm(train_loader, desc='Training'):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = correct = total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total

def plot_results(train_losses, train_accs, test_accs, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(range(1, epochs+1), train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(range(1, epochs+1), train_accs, 'g-', label='Training Accuracy')
    ax2.plot(range(1, epochs+1), test_accs, 'r-', label='Test Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_predictions(model, test_loader, device, num_samples=8):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images[i].cpu().squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        actual, pred = labels[i].item(), predicted[i].item()
        color = 'green' if actual == pred else 'red'
        axes[i].set_title(f'True: {actual}, Pred: {pred}', color=color)
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions (Green=Correct, Red=Wrong)')
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Data
    train_loader, test_loader = get_data_loaders()
    print(f'Training samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}')
    
    # Model
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # Training
    epochs = 10
    train_losses, train_accs, test_accs = [], [], []
    
    print(f'\nStarting training for {epochs} epochs...')
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    print(f'\nTraining completed! Best test accuracy: {max(test_accs):.2f}%')
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': {'train_losses': train_losses, 'train_accs': train_accs, 'test_accs': test_accs}
    }, 'mnist_model.pth')
    print('Model saved as mnist_model.pth')
    
    # Visualizations
    plot_results(train_losses, train_accs, test_accs, epochs)
    show_predictions(model, test_loader, device)
    
    print('\nAll operations completed successfully!')

if __name__ == '__main__':
    main()

