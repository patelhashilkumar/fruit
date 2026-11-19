"""Production Inference Pipeline for Fruit Identification & Ripeness Detection.

Provides:
- Image validation & preprocessing.
- Multi-Task ResNet50 inference.
- Calibrated confidence calculation.
- Grad-CAM heatmap generation as base64 PNG data URIs.
- Graceful handling of edge cases (low confidence, invalid inputs).
"""
import io
import os
import sys
import json
import base64

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.multitask_resnet50 import MultiTaskResNet50
from data.augmentations import get_transforms
from training.losses import coral_to_probs, coral_predict
from interpretability.gradcam import ResNet50GradCAM, overlay_cam


class FruitRipenessPredictor:
    def __init__(self, checkpoint_path=None, device=None, stage_config="4stage"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stage_config = stage_config

        # Load label mapping
        mapping_path = os.path.join(ROOT, "dataset", "metadata", "label_mapping.json")
        with open(mapping_path, "r") as f:
            self.labels = json.load(f)

        self.fruit_classes = self.labels["task1_fruit_identification"]["classes"]
        self.num_fruits = len(self.fruit_classes)

        if stage_config == "3stage":
            cfg = self.labels["task2_ripeness_detection"]["config_A_three_stage"]
            self.stage_names = cfg["stages"]
            self.num_stages = 3
        else:
            cfg = self.labels["task2_ripeness_detection"]["config_B_four_stage"]
            self.stage_names = cfg["stages"]
            self.num_stages = 4

        self.model = MultiTaskResNet50(
            num_fruits=self.num_fruits,
            num_stages=self.num_stages,
            pretrained=False
        ).to(self.device).eval()

        if checkpoint_path is None:
            default_ckpt = os.path.join(ROOT, "experiments", "checkpoints", "Exp_F", "best.pt")
            if os.path.exists(default_ckpt):
                checkpoint_path = default_ckpt

        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            print(f"Loaded predictor weights from {checkpoint_path}")
        else:
            print("Warning: Running with uninitialized / default weights.")

        self.transform = get_transforms(train=False, img_size=224)
        self.gradcam = ResNet50GradCAM(self.model)

    def validate_image(self, image_input):
        """Load and validate PIL Image from file path, bytes, or PIL object."""
        if isinstance(image_input, (str, os.PathLike)):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            img = Image.open(image_input)
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError("Unsupported image input type")

        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        if w < 16 or h < 16:
            raise ValueError(f"Image dimensions too small ({w}x{h})")
        return img

    def _cam_to_base64(self, overlay_np):
        buf = io.BytesIO()
        Image.fromarray(overlay_np).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    @torch.no_grad()
    def predict(self, image_input, generate_cam=True, cam_task="ripeness"):
        try:
            img_pil = self.validate_image(image_input)
        except Exception as e:
            return {"error": str(e), "success": False}

        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        out = self.model(tensor)
        fruit_logits = out["fruit_logits"]
        ripeness_logits = out["ripeness_logits"]

        # 1. Fruit Identification Softmax
        fruit_probs = F.softmax(fruit_logits, dim=1)[0].cpu().numpy()
        fruit_pred_id = int(fruit_probs.argmax())
        fruit_conf = float(fruit_probs[fruit_pred_id])
        fruit_type = self.fruit_classes[fruit_pred_id] if fruit_pred_id < len(self.fruit_classes) else "Unknown"

        # Low confidence threshold check
        is_unknown = (fruit_conf < 0.35)

        # 2. Ripeness CORAL Prediction
        ripe_probs = coral_to_probs(ripeness_logits)[0].cpu().numpy()
        ripe_pred_id = int(coral_predict(ripeness_logits)[0].item())
        ripe_conf = float(ripe_probs[ripe_pred_id])
        ripeness_stage = self.stage_names[ripe_pred_id] if ripe_pred_id < len(self.stage_names) else "Unknown"

        # Continuous ripeness score (sum of sigmoid logits)
        sig_p = torch.sigmoid(ripeness_logits)[0].cpu().numpy()
        continuous_score = float(np.sum(sig_p))

        result = {
            "success": True,
            "fruit_type": "Unknown" if is_unknown else fruit_type,
            "fruit_id": fruit_pred_id,
            "fruit_confidence": round(fruit_conf, 4),
            "fruit_probabilities": {self.fruit_classes[i]: round(float(fruit_probs[i]), 4) for i in range(len(self.fruit_classes))},
            "ripeness": ripeness_stage,
            "ripeness_stage_id": ripe_pred_id,
            "ripeness_confidence": round(ripe_conf, 4),
            "ripeness_score": round(continuous_score, 3),
            "threshold_probabilities": [round(float(p), 4) for p in sig_p],
            "stage_probabilities": [round(float(p), 4) for p in ripe_probs],
            "model": "ResNet50-Multitask-CORAL"
        }

        # 3. Optional Grad-CAM
        if generate_cam:
            with torch.enable_grad():
                cam_norm, _ = self.gradcam.generate_cam(tensor, task=cam_task)
                overlay = overlay_cam(img_pil, cam_norm, alpha=0.5)
                result["gradcam"] = self._cam_to_base64(overlay)

        return result


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "crops", "banana_ripe.jpg")
    print(f"Running inference on: {img_path}")
    predictor = FruitRipenessPredictor()
    res = predictor.predict(img_path)
    print(f"Identified Fruit : {res['fruit_type']} ({res['fruit_confidence']*100:.1f}% confidence)")
    print(f"Ripeness Stage   : {res['ripeness']} (Continuous score: {res['ripeness_score']}/3.0)")
    print(f"Milestone Gates  : {res['threshold_probabilities']}")

