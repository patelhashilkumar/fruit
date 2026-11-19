"""Model C: ResNet50 with CORAL rank-consistent ordinal regression head for ripeness detection."""
import torch
import torch.nn as nn
from torchvision import models


class ResNet50CORAL(nn.Module):
    def __init__(self, num_stages=4, pretrained=True, dropout=0.2):
        super().__init__()
        self.num_stages = num_stages
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features  # 2048

        self.dropout = nn.Dropout(p=dropout)
        # Shared projection vector for scalar ripeness score
        self.fc = nn.Linear(in_features, 1, bias=False)
        # K-1 learned rank thresholds
        self.bias = nn.Parameter(torch.linspace(2.0, -2.0, num_stages - 1))
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        feat = self.dropout(self.backbone(x))
        # logit_k = s + b_k
        return self.fc(feat) + self.bias
