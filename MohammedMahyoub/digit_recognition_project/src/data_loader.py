import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os

class MNISTDataLoader:
    """
    فئة لتحميل وتجهيز بيانات MNIST
    """
    def __init__(self, data_dir='./data', batch_size=64, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # تحويلات البيانات للتدريب
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # تحويلات البيانات للاختبار
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def get_data_loaders(self):
        """
        إرجاع محملات البيانات للتدريب والاختبار
        """
        # تحميل بيانات التدريب
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # تحميل بيانات الاختبار
        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # إنشاء محملات البيانات
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def get_class_weights(self):
        """
        حساب أوزان الفئات للتعامل مع عدم التوازن
        """
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=False
        )
        
        targets = train_dataset.targets.numpy()
        class_counts = np.bincount(targets)
        total_samples = len(targets)
        
        weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)
    







    def _load_letters_only(self):
        """تحميل الأحرف فقط من EMNIST"""
        try:
            train_dataset = datasets.EMNIST(
                root='./data',
                split='letters',  # الأحرف فقط
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
            
            print(f"✅ تم تحميل الأحرف فقط: {len(train_dataset)} عينة تدريب")
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), \
                DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                
        except Exception as e:
            print(f"❌ خطأ في تحميل الأحرف: {e}")
            return self._load_synthetic_data()
        
        
def get_data_loaders(self):
    if self.use_mnist_only:
        return self._load_mnist_only()
    elif self.use_letters_only:
        return self._load_letters_only()
    else:
        return self._load_emnist_with_fallback()




class CustomImageDataset(Dataset):
    """
    فئة مخصصة لتحميل الصور من مجلد
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('L')  # تحويل إلى رمادي
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.images[idx]

