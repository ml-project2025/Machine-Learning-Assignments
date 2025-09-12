import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitCNN(nn.Module):
    """
    شبكة عصبية تلافيفية للتعرف على الأرقام والحروف
    """
    def __init__(self, num_classes=36):  # تعديل عدد الفئات إلى 36
        super(DigitCNN, self).__init__()
        
        # الطبقات التلافيفية الأولى
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # طبقات التجميع
        self.pool = nn.MaxPool2d(2, 2)
        # طبقات التسوية
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
                # الطبقات المكتملة الاتصال
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)  # تعديل عدد الفئات هنا أيضاً
                # طبقة التطبيع المجموعي
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
    def forward(self, x):
        # الطبقة التلافيفية الأولى
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
                # الطبقة التلافيفية الثانية
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
                # الطبقة التلافيفية الثالثة
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
                # تطبيق Dropout
        x = self.dropout1(x)
                # تحويل إلى بعد واحد
        x = x.view(x.size(0), -1)
                # الطبقات المكتملة الاتصال
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class ImprovedDigitCNN(nn.Module):
    """
    نسخة محسنة من الشبكة مع ResNet blocks للتعرف على الأرقام والحروف
    """
    def __init__(self, num_classes=36):  # تعديل عدد الفئات إلى 36
        super(ImprovedDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)  # تعديل عدد الفئات هنا أيضاً
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

