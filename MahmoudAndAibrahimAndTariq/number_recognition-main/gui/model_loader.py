# gui/model_loader.py
import os
import torch

def load_digit_recognition_model():
    """Load the digit recognition model with robust path handling"""
    try:
        class DigitRecognitionModel(torch.nn.Module):
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
        
        # Find model path (works from any location)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # Try multiple paths
        model_paths = [
            os.path.join(project_root, "models", "digit_cnn_improved.pth"),
            os.path.join("models", "digit_cnn_improved.pth"),
            os.path.join(project_root, "models", "digit_cnn.pth"),
            os.path.join("models", "digit_cnn.pth")
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            raise FileNotFoundError("Model not found at any location")
        
        # Load model
        model = DigitRecognitionModel(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ Model load error: {str(e)}")
        return None