# model_utils.py
import torch
import torch.nn as nn
import torchvision.models as models

from utils.config import DATASET_CONFIG

class LeNet(nn.Module):
    def __init__(self, dataset_name, pretrained=False):
        super(LeNet, self).__init__()
        
        config = DATASET_CONFIG[dataset_name]
        self.conv1 = nn.Conv2d(config["in_channels"], 6, kernel_size=5, padding=2) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) 
        fc1_input_features = 16 * 6 * 6
        
        self.fc1 = nn.Linear(fc1_input_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config['num_classes'])
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.view(-1, 16 * 6 * 6)

        x = self.relu(self.fc1(x))
        x = self.dropout(x) 
        x = self.relu(self.fc2(x))
        x = self.dropout(x) 
        x = self.fc3(x)
        return x

class TailNet(nn.Module):
    def __init__(self, dataset_name, pretrained=False):
        super().__init__()
        
        config = DATASET_CONFIG[dataset_name]
        
        self.conv = nn.Sequential(
            nn.Conv2d(config["in_channels"], 16, 5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['fc_input_size'], 128), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, config['num_classes'])
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self, dataset_name, pretrained=False):
        super().__init__()
        
        config = DATASET_CONFIG[dataset_name]
        in_channels = config["in_channels"]
        num_classes = config["num_classes"]
        
        self.model = models.resnet18(pretrained=pretrained)
        
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)    
    
    
class VGG16(nn.Module):
    def __init__(self, dataset_name, pretrained=False):
        super().__init__()
        
        config = DATASET_CONFIG[dataset_name]
        in_channels = config["in_channels"]
        num_classes = config["num_classes"]
        
        self.model = models.vgg16(pretrained=pretrained)
        if in_channels != 3:
            self.model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)    
    
# ========== Model Factory ==========
def get_models(model_name):
    if model_name == 'lenet':
        return LeNet
    elif model_name == 'tailnet':
        return TailNet   
    elif model_name == 'resnet18':
        return ResNet18
    elif model_name == 'vgg16':
        return VGG16
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from ['tailnet', 'lenet', 'resnet18', 'vgg16']")    