# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class NumberDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Load all images
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
        image = Image.open(img_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label