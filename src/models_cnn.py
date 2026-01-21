
import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    """
    4-Layer CNN for Audio Classification on Mel-Spectrograms.
    Input: (Batch, 1, n_mels, time)
    """
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.flatten = nn.Flatten()
        
        # Calculate linear input size dynamically or hardcode if input size is fixed
        # For n_mels=128, duration=3s (~130 frames)
        # 128x130 -> 64x65 -> 32x32 -> 16x16 -> 8x8
        # 256 * 8 * 8 = 16384 (Approximation, usually better to use adaptive pool)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) # Forces 4x4 output
        self.fc_input = 256 * 4 * 4
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
