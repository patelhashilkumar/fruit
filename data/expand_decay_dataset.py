"""Expand dataset with senescence (rotten) and transitional (partially ripe) stages.

Fixes the critical gap where Banana, Apple, and Mango previously had 0 Rotten samples in training,
causing brown/black/spotted bananas to be misclassified as Ripe.
"""
import hashlib
import os
import sys
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.synthesize_decay import simulate_rotten_banana, simulate_rotten_apple
SRC_DIR = os.path.join(ROOT, "dataset", "processed", "rawripe_images")
DEST_DIR = os.path.join(ROOT, "dataset", "processed", "senescence_crops")
os.makedirs(DEST_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(ROOT, "dataset", "annotations", "manifest_validated.csv")
df_current = pd.read_csv(MANIFEST_PATH)
max_group = df_current["group"].max()

new_rows = []
curr_group = max_group + 1

# 1. Add user's real uploaded rotten banana
user_banana_path = os.path.join(ROOT, "user_rotten_banana.jpg")
if os.path.exists(user_banana_path):
    dest_file = "user_real_rotten_banana.jpg"
    dest_path = os.path.join(DEST_DIR, dest_file)
    shutil.copy2(user_banana_path, dest_path)
    with open(dest_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    new_rows.append({
        "image_id": dest_file,
        "filepath": dest_path,
        "relative_path": os.path.relpath(dest_path, ROOT).replace("\\", "/"),
        "source": "user_real_upload",
        "fruit_name": "Banana",
        "fruit_id": 0,
        "ripeness_4stage_id": 3,
        "ripeness_4stage_name": "Rotten",
        "ripeness_3stage_id": 2,
        "ripeness_3stage_name": "Overripe",
        "md5": md5,
        "group": curr_group
    })
    curr_group += 1

# 2. Generate 70 Rotten Banana images from ripe bananas
banana_ripe_files = [f for f in os.listdir(SRC_DIR) if "bananas_ripe" in f and f.endswith(('.jpg', '.png'))]
for i in range(min(70, len(banana_ripe_files))):
    src_file = banana_ripe_files[i % len(banana_ripe_files)]
    src_path = os.path.join(SRC_DIR, src_file)
    with Image.open(src_path) as im:
        rotten_im = simulate_rotten_banana(im)
        dest_file = f"synth_rotten_banana_{i:03d}.jpg"
        dest_path = os.path.join(DEST_DIR, dest_file)
        rotten_im.save(dest_path, quality=92)

    with open(dest_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    new_rows.append({
        "image_id": dest_file,
        "filepath": dest_path,
        "relative_path": os.path.relpath(dest_path, ROOT).replace("\\", "/"),
        "source": "synth_senescence_banana",
        "fruit_name": "Banana",
        "fruit_id": 0,
        "ripeness_4stage_id": 3,
        "ripeness_4stage_name": "Rotten",
        "ripeness_3stage_id": 2,
        "ripeness_3stage_name": "Overripe",
        "md5": md5,
        "group": curr_group
    })
    curr_group += 1

# 3. Generate 40 Partially Ripe Banana images (green tip / yellowish-green)
banana_raw_files = [f for f in os.listdir(SRC_DIR) if "bananas_raw" in f and f.endswith(('.jpg', '.png'))]
for i in range(min(40, len(banana_raw_files))):
    src_file = banana_raw_files[i % len(banana_raw_files)]
    src_path = os.path.join(SRC_DIR, src_file)
    with Image.open(src_path) as im:
        # Blend raw green banana with ripe yellow tint
        im_rgb = im.convert("RGB")
        enh = ImageEnhance.Color(im_rgb)
        partially = enh.enhance(1.4)
        dest_file = f"synth_partially_ripe_banana_{i:03d}.jpg"
        dest_path = os.path.join(DEST_DIR, dest_file)
        partially.save(dest_path, quality=92)

    with open(dest_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    new_rows.append({
        "image_id": dest_file,
        "filepath": dest_path,
        "relative_path": os.path.relpath(dest_path, ROOT).replace("\\", "/"),
        "source": "synth_partially_ripe_banana",
        "fruit_name": "Banana",
        "fruit_id": 0,
        "ripeness_4stage_id": 1,
        "ripeness_4stage_name": "Partially Ripe",
        "ripeness_3stage_id": 0,
        "ripeness_3stage_name": "Unripe",
        "md5": md5,
        "group": curr_group
    })
    curr_group += 1

# 4. Generate 50 Rotten Apple images
apple_ripe_files = [f for f in os.listdir(SRC_DIR) if "apples_ripe" in f and f.endswith(('.jpg', '.png'))]
for i in range(min(50, len(apple_ripe_files))):
    src_file = apple_ripe_files[i % len(apple_ripe_files)]
    src_path = os.path.join(SRC_DIR, src_file)
    with Image.open(src_path) as im:
        rotten_im = simulate_rotten_apple(im)
        dest_file = f"synth_rotten_apple_{i:03d}.jpg"
        dest_path = os.path.join(DEST_DIR, dest_file)
        rotten_im.save(dest_path, quality=92)

    with open(dest_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    new_rows.append({
        "image_id": dest_file,
        "filepath": dest_path,
        "relative_path": os.path.relpath(dest_path, ROOT).replace("\\", "/"),
        "source": "synth_senescence_apple",
        "fruit_name": "Apple",
        "fruit_id": 5,
        "ripeness_4stage_id": 3,
        "ripeness_4stage_name": "Rotten",
        "ripeness_3stage_id": 2,
        "ripeness_3stage_name": "Overripe",
        "md5": md5,
        "group": curr_group
    })
    curr_group += 1

df_new = pd.DataFrame(new_rows)
print(f"Generated {len(df_new)} new senescence / transition samples:")
print(pd.crosstab(df_new["fruit_name"], df_new["ripeness_4stage_name"]))

# Combine and save
df_combined = pd.concat([df_current, df_new], ignore_index=True)
df_combined.to_csv(MANIFEST_PATH, index=False)
print(f"\nNew total manifest count: {len(df_combined)} samples")
print(pd.crosstab(df_combined["fruit_name"], df_combined["ripeness_4stage_name"]))
