"""Model B: ResNet50 Classifier with standard Softmax output for Fruit ID or Ripeness."""
import torch.nn as nn
from torchvision import models


class ResNet50Softmax(nn.Module):
    def __init__(self, num_classes=6, pretrained=True, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features  # 2048

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
