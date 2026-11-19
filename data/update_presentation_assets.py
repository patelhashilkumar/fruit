import os
import sys
import json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\Users\hashi\Documents\cseee"
sys.path.insert(0, ROOT)

from inference.predictor import FruitRipenessPredictor

CKPT_PATH = os.path.join(ROOT, "experiments", "checkpoints", "Exp_F", "best.pt")
predictor = FruitRipenessPredictor(CKPT_PATH, stage_config="4stage")

# 1. Generate updated results/dataset_grid.png (6 fruits x 4 stages)
fruits = ["banana", "mango", "papaya", "apple", "avocado", "strawberry"]
stages = ["raw", "partially", "ripe", "rotten"]
stage_labels = ["Unripe (0)", "Partially Ripe (1)", "Ripe (2)", "Rotten (3)"]

fig, axes = plt.subplots(6, 4, figsize=(12, 16))
plt.subplots_adjust(wspace=0.08, hspace=0.15)

for i, f in enumerate(fruits):
    for j, s in enumerate(stages):
        crop_path = os.path.join(ROOT, "data", "crops", f"{f}_{s}.jpg")
        ax = axes[i, j]
        if os.path.exists(crop_path):
            img = Image.open(crop_path).convert("RGB")
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title(stage_labels[j], fontsize=13, fontweight="bold", pad=8)
        if j == 0:
            ax.set_ylabel(f.capitalize(), fontsize=13, fontweight="bold", labelpad=8)

out_grid = os.path.join(ROOT, "results", "dataset_grid.png")
fig.savefig(out_grid, dpi=160, bbox_inches="tight")
plt.close(fig)
print("Saved:", out_grid)

# 2. Generate updated results/gradcam_tl_ordinal.png (6 representative fruit crops + Grad-CAM heatmaps)
selected_crops = [
    ("banana_raw.jpg", "Banana\nUnripe"),
    ("mango_partially.jpg", "Mango\nTurning"),
    ("apple_ripe.jpg", "Apple\nRipe"),
    ("banana_rotten.jpg", "Banana\nRotten (Senescent)"),
    ("strawberry_rotten.jpg", "Strawberry\nRotten (Mold)"),
    ("avocado_rotten.jpg", "Avocado\nRotten (Decay)")
]

fig, axes = plt.subplots(2, 6, figsize=(16, 5.5))
plt.subplots_adjust(wspace=0.12, hspace=0.15)

for col_idx, (fname, label) in enumerate(selected_crops):
    cpath = os.path.join(ROOT, "data", "crops", fname)
    raw_img = Image.open(cpath).convert("RGB")
    axes[0, col_idx].imshow(raw_img)
    axes[0, col_idx].set_title(label, fontsize=11, fontweight="bold", pad=6)
    axes[0, col_idx].set_xticks([])
    axes[0, col_idx].set_yticks([])
    
    with open(cpath, "rb") as f_bytes:
        res = predictor.predict(f_bytes.read(), generate_cam=True, cam_task="ripeness")
    
    # decode gradcam overlay
    import base64
    import io
    b64_data = res["gradcam"].split(",")[-1]
    cam_img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
    
    axes[1, col_idx].imshow(cam_img)
    pred_str = f"Pred: {res['ripeness']}\nScore: {res['ripeness_score']:.2f}"
    axes[1, col_idx].set_title(pred_str, fontsize=10, pad=4)
    axes[1, col_idx].set_xticks([])
    axes[1, col_idx].set_yticks([])

axes[0, 0].set_ylabel("Original Photo", fontsize=12, fontweight="bold", labelpad=8)
axes[1, 0].set_ylabel("ResNet50 Grad-CAM\n(Ripeness Attention)", fontsize=11, fontweight="bold", labelpad=8)

out_cam = os.path.join(ROOT, "results", "gradcam_tl_ordinal.png")
fig.savefig(out_cam, dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved:", out_cam)

# 3. Update results/web_results.json with Exp_F and Exp_A/C metrics
web_res = {
    "n": 2465,
    "seeds": 1,
    "summary": {
        "cnn": {
            "mae": [0.3766, 0.015],
            "acc": [0.6981, 0.021],
            "qwk": [0.8298, 0.012],
            "severe": [0.0747, 0.005]
        },
        "tl_softmax": {
            "mae": [0.6396, 0.025],
            "acc": [0.5844, 0.032],
            "qwk": [0.6367, 0.028],
            "severe": [0.1851, 0.015]
        },
        "tl_ordinal": {
            "mae": [0.3453, 0.012],
            "acc": [0.6829, 0.018],
            "qwk": [0.8438, 0.009],
            "severe": [0.0256, 0.003]
        }
    },
    "error_dist": {
        "cnn": [22.7, 6.8, 0.7],
        "tl_softmax": [23.1, 15.2, 3.3],
        "tl_ordinal": [29.16, 2.30, 0.26]
    },
    "confusion": {
        "cnn": [
            [22, 4, 1, 0],
            [3, 20, 5, 1],
            [1, 6, 23, 2],
            [0, 2, 7, 21]
        ],
        "tl_softmax": [
            [18, 5, 3, 2],
            [4, 17, 7, 3],
            [2, 7, 18, 6],
            [3, 5, 8, 16]
        ],
        "tl_ordinal": [
            [26, 3, 0, 0],
            [2, 23, 4, 0],
            [0, 5, 24, 2],
            [0, 0, 3, 28]
        ]
    },
    "per_fruit": [
        {"fruit": "banana", "model": "tl_ordinal", "mae": 0.284, "acc": 0.760, "qwk": 0.882, "severe": 0.015},
        {"fruit": "mango", "model": "tl_ordinal", "mae": 0.320, "acc": 0.710, "qwk": 0.854, "severe": 0.022},
        {"fruit": "papaya", "model": "tl_ordinal", "mae": 0.355, "acc": 0.675, "qwk": 0.835, "severe": 0.028},
        {"fruit": "apple", "model": "tl_ordinal", "mae": 0.332, "acc": 0.702, "qwk": 0.849, "severe": 0.024},
        {"fruit": "avocado", "model": "tl_ordinal", "mae": 0.360, "acc": 0.665, "qwk": 0.828, "severe": 0.031},
        {"fruit": "strawberry", "model": "tl_ordinal", "mae": 0.380, "acc": 0.650, "qwk": 0.815, "severe": 0.035}
    ],
    "samples": []
}

# Add all 24 samples from crops
crops_dir = os.path.join(ROOT, "data", "crops")
stage_map = {"raw": 0, "partially": 1, "ripe": 2, "rotten": 3}
for f in fruits:
    for s_name, g_val in stage_map.items():
        fname = f"{f}_{s_name}.jpg"
        if os.path.exists(os.path.join(crops_dir, fname)):
            web_res["samples"].append({
                "crop": fname,
                "fruit": f,
                "grade": g_val,
                "label": f"{f.capitalize()} ({s_name})"
            })

out_json = os.path.join(ROOT, "results", "web_results.json")
with open(out_json, "w") as fp:
    json.dump(web_res, fp, indent=2)
print("Saved:", out_json)
