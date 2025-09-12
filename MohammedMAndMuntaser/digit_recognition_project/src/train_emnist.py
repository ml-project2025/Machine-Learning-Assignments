# train_emnist.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
import argparse
import numpy as np

# تعطيل GPU لتفادي مشاكل التوافق
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# -------------------------
# 1️⃣ نموذج Improved CNN
# -------------------------
class ImprovedDigitCNN(nn.Module):
    def __init__(self, num_classes=10):  # افتراضي 10 فئات للأرقام
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# -------------------------
# 2️⃣ DataLoader معدل بدون إنترنت
# -------------------------
class EMNISTDataLoader:
    def __init__(self, batch_size=64, use_mnist_only=True, use_letters_only=False):
        self.batch_size = batch_size
        self.use_mnist_only = use_mnist_only
        self.use_letters_only = use_letters_only
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def get_data_loaders(self):
        """تحميل البيانات بدون الحاجة للإنترنت"""
        try:
            if self.use_mnist_only:
                print("📦 استخدام بيانات MNIST فقط (0-9)")
                return self._load_mnist_only()
            elif self.use_letters_only:
                print("📦 استخدام بيانات الأحرف فقط (A-Z)")
                return self._load_letters_only()
            else:
                print("📦 محاولة تحميل بيانات EMNIST كاملة...")
                return self._load_emnist_with_fallback()
                
        except Exception as e:
            print(f"⚠️ خطأ في تحميل البيانات: {e}")
            print("🔄 استخدام بيانات MNIST كبديل")
            return self._load_mnist_only()

    def _load_mnist_only(self):
        """تحميل MNIST فقط (يعمل بدون إنترنت)"""
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=False,
            transform=self.transform
        )
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=False,
            transform=self.transform
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"✅ تم تحميل MNIST: {len(train_dataset)} عينة تدريب، {len(test_dataset)} عينة اختبار")
        return train_loader, test_loader

    def _load_letters_only(self):
        """تحميل الأحرف فقط من EMNIST"""
        try:
            train_dataset = datasets.EMNIST(
                root='./data',
                split='letters',
                train=True,
                download=False,
                transform=self.transform
            )
            test_dataset = datasets.EMNIST(
                root='./data',
                split='letters',
                train=False,
                download=False,
                transform=self.transform
            )
            
            # تعديل التسميات لتبدأ من 0
            train_dataset.targets = train_dataset.targets - 1
            test_dataset.targets = test_dataset.targets - 1
            
            print(f"✅ تم تحميل الأحرف: {len(train_dataset)} عينة تدريب، {len(test_dataset)} عينة اختبار")
            
        except (RuntimeError, FileNotFoundError):
            print("⚠️ لم يتم العثور على بيانات الأحرف، استخدام MNIST كبديل")
            return self._load_mnist_only()

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader

    def _load_emnist_with_fallback(self):
        """محاولة تحميل EMNIST مع وجود بديل إذا فشل"""
        try:
            train_dataset = datasets.EMNIST(
                root='./data',
                split='byclass',
                train=True,
                download=False,
                transform=self.transform
            )
            test_dataset = datasets.EMNIST(
                root='./data',
                split='byclass',
                train=False,
                download=False,
                transform=self.transform
            )
            
            print(f"✅ تم تحميل EMNIST: {len(train_dataset)} عينة تدريب، {len(test_dataset)} عينة اختبار")
            
        except (RuntimeError, FileNotFoundError):
            print("⚠️ لم يتم العثور على بيانات EMNIST، استخدام MNIST كبديل")
            return self._load_mnist_only()

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader

    def check_data_exists(self):
        """التحقق من وجود البيانات محلياً"""
        mnist_path = './data/MNIST'
        emnist_path = './data/EMNIST'
        
        mnist_exists = os.path.exists(mnist_path)
        emnist_exists = os.path.exists(emnist_path)
        
        print(f"📁 MNIST موجود: {mnist_exists}")
        print(f"📁 EMNIST موجود: {emnist_exists}")
        
        return mnist_exists or emnist_exists

# -------------------------
# 3️⃣ Trainer معدل
# -------------------------
class DigitTrainer:
    def __init__(self, learning_rate=0.001, use_mnist_only=True, use_letters_only=False):
        # استخدام CPU دائماً
        self.device = torch.device("cpu")
        
        # تحديد عدد الفئات
        if use_mnist_only:
            num_classes = 10  # أرقام فقط
        elif use_letters_only:
            num_classes = 26  # أحرف فقط
        else:
            num_classes = 62  # الكل
        
        self.model = ImprovedDigitCNN(num_classes=num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = [], []
        self.use_mnist_only = use_mnist_only
        self.use_letters_only = use_letters_only

        print(f"🚀 استخدام الجهاز: {self.device}")
        if self.use_mnist_only:
            print("🎯 التدريب على الأرقام فقط (0-9)")
        elif self.use_letters_only:
            print("🎯 التدريب على الأحرف فقط (A-Z)")
        else:
            print("🎯 التدريب على الأرقام والحروف (0-9 + A-Z)")

    def train_epoch(self, loader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        progress_bar = tqdm(loader, desc="التدريب", leave=False)
        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
            
        return running_loss / len(loader), 100. * correct / total

    def validate_epoch(self, loader):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for data, target in tqdm(loader, desc="التحقق", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        return running_loss / len(loader), 100. * correct / total

    def train(self, num_epochs=10, save_path='./saved_models'):
        data_loader = EMNISTDataLoader(
            use_mnist_only=self.use_mnist_only,
            use_letters_only=self.use_letters_only
        )
        
        if not data_loader.check_data_exists():
            print("❌ لا توجد بيانات للتدريب!")
            return
            
        train_loader, val_loader = data_loader.get_data_loaders()
        best_val_acc = 0.0
        os.makedirs(save_path, exist_ok=True)

        print(f"\n🎬 بدء التدريب لمدة {num_epochs} حقبات")
        print("=" * 50)

        for epoch in range(num_epochs):
            print(f"\n📍 الحقبة {epoch+1}/{num_epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            self.scheduler.step(val_loss)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"📊 التدريب - الخسارة: {train_loss:.4f}, الدقة: {train_acc:.2f}%")
            print(f"📈 التحقق - الخسارة: {val_loss:.4f}, الدقة: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(save_path)
                print(f"✅ حفظ أفضل نموذج بدقة تحقق: {val_acc:.2f}%")

        print(f"\n🎉 انتهى التدريب! أفضل دقة تحقق: {best_val_acc:.2f}%")
        self.plot_training_history()
        self.save_training_results()

    def save_model(self, save_path):
        # تحديد عدد الفئات
        if self.use_mnist_only:
            num_classes = 10
        elif self.use_letters_only:
            num_classes = 26
        else:
            num_classes = 62
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': 'ImprovedDigitCNN',
            'num_classes': num_classes,
            'accuracy': self.val_accuracies[-1] if self.val_accuracies else 0
        }, os.path.join(save_path, 'best_digit_model.pth'))

    def plot_training_history(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='خسارة التدريب', color='blue', linewidth=2)
        plt.plot(self.val_losses, label='خسارة التحقق', color='red', linewidth=2)
        plt.xlabel('الحقبة')
        plt.ylabel('الخسارة')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='دقة التدريب', color='green', linewidth=2)
        plt.plot(self.val_accuracies, label='دقة التحقق', color='orange', linewidth=2)
        plt.xlabel('الحقبة')
        plt.ylabel('الدقة (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./saved_models/training_history.png', dpi=300, bbox_inches='tight')
        print("📊 تم حفظ مخطط التدريب")

    def save_training_results(self):
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'final_train_accuracy': self.train_accuracies[-1] if self.train_accuracies else 0,
            'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else 0,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'use_mnist_only': self.use_mnist_only,
            'use_letters_only': self.use_letters_only
        }
        
        with open('./saved_models/training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print("📝 تم حفظ نتائج التدريب")

# -------------------------
# 4️⃣ Main معدل
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='تدريب نموذج التعرف على الأرقام والحروف')
    parser.add_argument("--epochs", type=int, default=10, help="عدد الحقبات (افتراضي: 10)")
    parser.add_argument("--lr", type=float, default=0.001, help="معدل التعلم (افتراضي: 0.001)")
    parser.add_argument("--use-mnist-only", action='store_true', help="استخدام MNIST فقط (0-9)")
    parser.add_argument("--use-letters-only", action='store_true', help="استخدام الأحرف فقط (A-Z)")
    
    args = parser.parse_args()

    print("=" * 60)
    print("🤖 برنامج تدريب نموذج التعرف على الأرقام والحروف")
    print("=" * 60)

    trainer = DigitTrainer(
        learning_rate=args.lr,
        use_mnist_only=args.use_mnist_only,
        use_letters_only=args.use_letters_only
    )
    
    trainer.train(num_epochs=args.epochs)