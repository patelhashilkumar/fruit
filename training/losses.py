"""Loss functions for Softmax, CORAL ordinal regression, and Multi-Task learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ordinal_targets(y, num_stages):
    """Convert integer grade y (0 .. num_stages-1) into K-1 binary targets.
    Example for K=4:
      0 -> [0, 0, 0]
      1 -> [1, 0, 0]
      2 -> [1, 1, 0]
      3 -> [1, 1, 1]
    """
    k_minus_1 = num_stages - 1
    thresholds = torch.arange(k_minus_1, device=y.device)
    return (y.unsqueeze(1) > thresholds).float()


def coral_loss(logits, targets, num_stages, weights=None, reduction="mean"):
    """Compute Rank-Consistent Ordinal Regression (CORAL) loss using BCEWithLogits.
    
    Args:
        logits: (B, num_stages - 1)
        targets: (B,) integer labels in [0, num_stages - 1]
        num_stages: Total number of ordinal stages
        weights: Optional class weights tensor
    """
    bin_targets = ordinal_targets(targets, num_stages)
    loss = F.binary_cross_entropy_with_logits(logits, bin_targets, reduction="none")
    
    if weights is not None:
        # Weight each sample by target class weight
        sample_weights = weights[targets].unsqueeze(1)
        loss = loss * sample_weights
        
    if reduction == "mean":
        return loss.sum() / max(1, len(targets))
    elif reduction == "sum":
        return loss.sum()
    return loss


def coral_to_probs(logits):
    """Convert CORAL threshold logits into per-stage probability distribution (B, K).
    P(y = 0) = 1 - P(y > 0)
    P(y = k) = P(y > k-1) - P(y > k)
    P(y = K-1) = P(y > K-2)
    """
    p_gt = torch.sigmoid(logits)  # (B, K-1)
    ones = torch.ones_like(p_gt[:, :1])
    zeros = torch.zeros_like(p_gt[:, :1])
    cum = torch.cat([ones, p_gt, zeros], dim=1)  # (B, K+1)
    probs = (cum[:, :-1] - cum[:, 1:]).clamp_min(0.0)
    # Ensure exact sum to 1
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs


def coral_predict(logits):
    """Predict ordinal grade: count of thresholds where probability > 0.5."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


class MultiTaskLoss(nn.Module):
    """Multi-task loss with masking for missing task annotations."""

    def __init__(self, num_fruits=6, num_stages=4, lambda_ripeness=1.0, fruit_weights=None, ripeness_weights=None):
        super().__init__()
        self.num_fruits = num_fruits
        self.num_stages = num_stages
        self.lambda_ripeness = lambda_ripeness
        self.fruit_weights = fruit_weights
        self.ripeness_weights = ripeness_weights

    def forward(self, fruit_logits, ripeness_logits, fruit_targets, ripeness_targets, has_fruit, has_ripeness):
        # Fruit identification loss (CrossEntropy)
        fruit_mask = has_fruit.bool()
        if fruit_mask.sum() > 0:
            loss_fruit = F.cross_entropy(
                fruit_logits[fruit_mask],
                fruit_targets[fruit_mask],
                weight=self.fruit_weights.to(fruit_logits.device) if self.fruit_weights is not None else None
            )
        else:
            loss_fruit = torch.tensor(0.0, device=fruit_logits.device)

        # Ripeness detection loss (CORAL)
        ripeness_mask = has_ripeness.bool()
        if ripeness_mask.sum() > 0:
            loss_ripeness = coral_loss(
                ripeness_logits[ripeness_mask],
                ripeness_targets[ripeness_mask],
                num_stages=self.num_stages,
                weights=self.ripeness_weights.to(ripeness_logits.device) if self.ripeness_weights is not None else None
            )
        else:
            loss_ripeness = torch.tensor(0.0, device=ripeness_logits.device)

        total_loss = loss_fruit + self.lambda_ripeness * loss_ripeness
        return total_loss, loss_fruit, loss_ripeness
