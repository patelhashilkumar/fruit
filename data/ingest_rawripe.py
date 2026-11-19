"""Ingest RawRipe dataset to expand fruit classification and ripeness training data.

Supports:
- Banana (0)
- Mango (1)
- Papaya (2)
- Apple (5)
- Strawberry (4)
Combined with existing Mendeley Avocado (3) and Strawberry (4) crops.
"""
import hashlib
import os
import shutil
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAWRIPE_DIR = os.path.join(ROOT, "temp_rawripe")
DEST_DIR = os.path.join(ROOT, "dataset", "processed", "rawripe_images")
os.makedirs(DEST_DIR, exist_ok=True)

FRUIT_MAP = {
    "bananas": ("Banana", 0),
    "mangoes": ("Mango", 1),
    "papayas": ("Papaya", 2),
    "apples": ("Apple", 5),
    "strawberries": ("Strawberry", 4)
}

STAGE_MAP = {
    "raw": {
        "4stage_id": 0, "4stage_name": "Unripe",
        "3stage_id": 0, "3stage_name": "Unripe"
    },
    "ripe": {
        "4stage_id": 2, "4stage_name": "Ripe",
        "3stage_id": 1, "3stage_name": "Ripe"
    }
}

# 1. Load existing Mendeley validated manifest
mendeley_manifest_path = os.path.join(ROOT, "dataset", "annotations", "manifest_validated.csv")
df_mendeley = pd.read_csv(mendeley_manifest_path)
max_group = df_mendeley["group"].max()

rows = []
curr_group = max_group + 1

for fruit_folder, (fruit_name, fruit_id) in FRUIT_MAP.items():
    for state_folder, stage_info in STAGE_MAP.items():
        src_images = os.path.join(RAWRIPE_DIR, fruit_folder, state_folder, "images")
        if not os.path.exists(src_images):
            continue
        
        file_list = sorted([f for f in os.listdir(src_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for filename in file_list:
            src_path = os.path.join(src_images, filename)
            # Verify image is valid
            try:
                with Image.open(src_path) as im:
                    im.verify()
            except Exception:
                continue
            
            # Destination file
            dest_filename = f"rawripe_{fruit_folder}_{state_folder}_{filename}"
            dest_path = os.path.join(DEST_DIR, dest_filename)
            shutil.copy2(src_path, dest_path)
            
            # Compute MD5
            with open(dest_path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            
            # Grouping: images with similar base names share group
            prefix = filename.rsplit("_", 1)[0] if "_" in filename else filename
            group_id = curr_group
            curr_group += 1
            
            rows.append({
                "image_id": dest_filename,
                "filepath": dest_path,
                "relative_path": os.path.relpath(dest_path, ROOT).replace("\\", "/"),
                "source": f"rawripe_{fruit_folder}_{state_folder}",
                "fruit_name": fruit_name,
                "fruit_id": fruit_id,
                "ripeness_4stage_id": stage_info["4stage_id"],
                "ripeness_4stage_name": stage_info["4stage_name"],
                "ripeness_3stage_id": stage_info["3stage_id"],
                "ripeness_3stage_name": stage_info["3stage_name"],
                "md5": md5,
                "group": group_id
            })

df_rawripe = pd.DataFrame(rows)
print(f"Ingested {len(df_rawripe)} verified images from RawRipe:")
print(df_rawripe["fruit_name"].value_counts())

# Combine with Mendeley
df_combined = pd.concat([df_mendeley, df_rawripe], ignore_index=True)
combined_manifest_path = os.path.join(ROOT, "dataset", "annotations", "manifest_validated.csv")
# Save combined as manifest_validated.csv
df_combined.to_csv(combined_manifest_path, index=False)
print(f"\nTotal combined manifest size: {len(df_combined)} images across 6 classes:")
print(df_combined["fruit_name"].value_counts())
print("\nRipeness distribution:")
print(df_combined["ripeness_4stage_name"].value_counts())
