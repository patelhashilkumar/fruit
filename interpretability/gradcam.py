"""Grad-CAM interpretability for ResNet50 and Multi-Task Architectures.

Visualizes regions driving:
1. Fruit Identification (backpropagating fruit class logit).
2. Ripeness Detection (backpropagating shared ordinal ripeness score).
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.augmentations import get_transforms


class ResNet50GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        if target_layer is None:
            # Default to layer4 in ResNet50
            if hasattr(model, "layer4"):
                target_layer = model.layer4
            elif hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
                target_layer = model.backbone.layer4
            else:
                raise AttributeError("Could not identify layer4 in model")

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out
        if out.requires_grad:
            out.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def generate_cam(self, x, task="fruit", target_class=None):
        """Generate Grad-CAM heatmap for specified task.
        
        task: "fruit" (explains fruit class) or "ripeness" (explains ordinal ripeness score).
        """
        x = x.clone().detach().requires_grad_(True)
        out = self.model(x)

        if isinstance(out, dict):
            fruit_logits = out["fruit_logits"]
            ripeness_logits = out["ripeness_logits"]
        else:
            if task == "fruit":
                fruit_logits = out
                ripeness_logits = None
            else:
                fruit_logits = None
                ripeness_logits = out

        self.model.zero_grad()

        if task == "fruit":
            pred_class = target_class if target_class is not None else fruit_logits.argmax(dim=1).item()
            score = fruit_logits[0, pred_class]
        else:
            # Ripeness CORAL score: backpropagate shared projection score s
            if hasattr(self.model, "ripeness_fc"):
                score = self.model.ripeness_fc(F.adaptive_avg_pool2d(self.activations, 1).flatten(1)).sum()
            elif hasattr(self.model, "fc"):
                score = self.model.fc(F.adaptive_avg_pool2d(self.activations, 1).flatten(1)).sum()
            else:
                score = ripeness_logits.sum()
            pred_class = None

        score.backward()

        # Channel pooling weights: alpha_c = mean over spatial dimensions
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * self.activations).sum(dim=1, keepdim=True))

        # Upsample to image size
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()

        # Normalize to [0, 1]
        denom = cam.max() - cam.min()
        cam_norm = (cam - cam.min()) / (denom + 1e-8) if denom > 0 else np.zeros_like(cam)
        return cam_norm, pred_class


def overlay_cam(img_pil, cam, alpha=0.5, colormap="jet"):
    """Overlay CAM heatmap onto original PIL Image."""
    img_resized = img_pil.resize((224, 224))
    img_np = np.asarray(img_resized, dtype=np.float32) / 255.0

    cmap = plt.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]  # RGB

    overlay = (1.0 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)
