import torch
import torch.nn as nn
from torchvision import models

class GenreHybridModel(nn.Module):
    """
    Dual-Branch Architecture:
    1. Visual Branch: ResNet18 (pretrained) on Mel-Spectrograms.
    2. Tabular Branch: MLP on pre-extracted features (MFCCs, etc).
    3. Fusion: Concatenation -> FC Head.
    """
    def __init__(self, num_classes=10, tabular_input_dim=50, pretrained=True):
        super(GenreHybridModel, self).__init__()
        
        # --- 1. Visual Branch (ResNet18) ---
        # Load Pretrained ResNet
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Remove the final FC layer to use as feature extractor
        # ResNet18 structure ends with: AdaptiveAvgPool2d -> Linear
        # The output before the Linear layer is 512-dim
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_out_dim = resnet.fc.in_features # 512
        
        # --- 2. Tabular Branch (MLP) ---
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.tabular_out_dim = 64
        
        # --- 3. Fusion Head ---
        fusion_dim = self.cnn_out_dim + self.tabular_out_dim # 512 + 64 = 576
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, spec, tabular):
        """
        spec: (Batch, 1, F, T) Mel-Spectrograms
        tabular: (Batch, D) Tabular features
        """
        # --- Visual Branch ---
        # Adapt 1-channel spec to 3-channel ResNet input
        x_cnn = spec.repeat(1, 3, 1, 1)
        
        x_cnn = self.cnn_backbone(x_cnn) # (B, 512, 1, 1)
        x_cnn = x_cnn.view(x_cnn.size(0), -1) # Flatten -> (B, 512)
        
        # --- Tabular Branch ---
        x_tab = self.tabular_mlp(tabular) # (B, 64)
        
        # --- Fusion ---
        x_combined = torch.cat((x_cnn, x_tab), dim=1) # (B, 576)
        
        # --- Classification ---
        out = self.fusion_head(x_combined)
        return out
