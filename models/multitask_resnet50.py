"""Model D: Multi-Task ResNet50 for Fruit Identification and Ripeness Detection.

Architecture:
  Input (3, 224, 224)
         |
  ResNet50 Backbone (layer1 -> layer2 -> layer3 -> layer4)
         |
  Global Average Pooling (2048 features)
         |
    +----+--------------------------------+
    |                                     |
  Head 1: Fruit ID                      Head 2: Ripeness CORAL
  Linear(2048 -> num_fruits)            Shared w (2048 -> 1) + (num_stages-1) thresholds
  Softmax CrossEntropy                  Rank-consistent Ordinal BCE
"""
import torch
import torch.nn as nn
from torchvision import models


class MultiTaskResNet50(nn.Module):
    def __init__(self, num_fruits=6, num_stages=4, pretrained=True, dropout=0.3):
        super().__init__()
        self.num_fruits = num_fruits
        self.num_stages = num_stages

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)

        # Feature extractor layers
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        in_features = base.fc.in_features  # 2048

        self.dropout = nn.Dropout(p=dropout)

        # HEAD 1: Fruit Identification (Multi-class Softmax)
        self.fruit_head = nn.Linear(in_features, num_fruits)

        # HEAD 2: Ripeness Detection (CORAL Ordinal Head)
        self.ripeness_fc = nn.Linear(in_features, 1, bias=False)
        self.ripeness_bias = nn.Parameter(torch.linspace(2.0, -2.0, num_stages - 1))

    def extract_features(self, x):
        """Pass through convolutional layers up to layer4."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_from_features(self, feat_maps):
        """Pool features and compute predictions."""
        pooled = self.avgpool(feat_maps)
        pooled = torch.flatten(pooled, 1)
        feat = self.dropout(pooled)

        fruit_logits = self.fruit_head(feat)
        ripeness_logits = self.ripeness_fc(feat) + self.ripeness_bias
        return fruit_logits, ripeness_logits

    def forward(self, x):
        feat_maps = self.extract_features(x)
        fruit_logits, ripeness_logits = self.forward_from_features(feat_maps)
        return {
            "fruit_logits": fruit_logits,
            "ripeness_logits": ripeness_logits,
            "features": feat_maps
        }
