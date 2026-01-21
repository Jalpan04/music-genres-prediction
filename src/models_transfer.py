
import torch
import torch.nn as nn
from torchvision import models

class GenreResNet(nn.Module):
    """
    ResNet18 wrapper for 10-class Audio Classification.
    Adapts 1-channel mel-spectrogram to 3-channel ResNet input.
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(GenreResNet, self).__init__()
        
        # Load ResNet18
        # weights='DEFAULT' is the modern way to specify pretrained
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, 1, n_mels, time)
        # ResNet expects 3 channels. We can repeat the single channel 3 times.
        x = x.repeat(1, 3, 1, 1)
        return self.resnet(x)
