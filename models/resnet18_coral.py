"""Model A: Baseline ResNet18 with CORAL ordinal regression head (Preserved Baseline)."""
import torch
import torch.nn as nn
from torchvision import models


class ResNet18CORAL(nn.Module):
    def __init__(self, num_stages=4, pretrained=True):
        super().__init__()
        self.num_stages = num_stages
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features  # 512

        # CORAL Ordinal Head
        self.fc = nn.Linear(in_features, 1, bias=False)
        # Initialize K-1 thresholds well separated
        self.bias = nn.Parameter(torch.linspace(2.0, -2.0, num_stages - 1))
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        feat = self.backbone(x)
        return self.fc(feat) + self.bias
