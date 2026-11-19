"""Dataset validation and quality audit pipeline.

Verifies:
- Corrupted images and file formats.
- Image dimensions and minimum size requirements.
- Exact duplicates (MD5) and near-duplicates (flip-invariant dHash).
- Class distributions for Fruit Identification and Ripeness Detection.
- Generates annotations/manifest_validated.csv.
"""
import os
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
METADATA_DIR = os.path.join(DATASET_DIR, "metadata")
REPORTS_DIR = os.path.join(ROOT, "reports")

HASH_SIZE = 16
HASH_THRESH = 40  # bits out of 256


def dhash(img, size=HASH_SIZE):
    """256-bit difference hash over the central 60% of the image."""
    W, H = img.size
    crop = img.crop((int(W * 0.2), int(H * 0.2), int(W * 0.8), int(H * 0.8)))
    g = np.asarray(crop.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def validate_image_file(path):
    """Return True if image is valid, uncorrupted, and meets size standards."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            if w < 16 or h < 16:
                return False, f"Image too small ({w}x{h})"
            img = img.convert("RGB")
        return True, "Valid"
    except Exception as e:
        return False, str(e)


def run_validation():
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    mapping_path = os.path.join(METADATA_DIR, "label_mapping.json")
    with open(mapping_path, "r") as f:
        label_cfg = json.load(f)

    fruit_map = label_cfg["task1_fruit_identification"]["fruit_to_id"]
    ripeness_4stage = label_cfg["task2_ripeness_detection"]["config_B_four_stage"]["id_to_stage"]
    ripeness_3stage = label_cfg["task2_ripeness_detection"]["config_A_three_stage"]["stage_to_id"]

    # Source manifest from existing dataset
    legacy_manifest = os.path.join(ROOT, "data", "splits", "manifest.csv")
    if not os.path.exists(legacy_manifest):
        raise FileNotFoundError(f"Manifest not found at {legacy_manifest}")

    df = pd.read_csv(legacy_manifest)
    print(f"Loaded {len(df)} entries from legacy manifest.")

    validated_rows = []
    hashes = []
    corrupted_count = 0
    duplicate_md5_count = 0
    seen_md5 = set()

    for idx, row in df.iterrows():
        crop_name = row["crop"]
        crop_path = os.path.join(PROCESSED_DIR, "mendeley_crops", crop_name)
        if not os.path.exists(crop_path):
            crop_path = os.path.join(ROOT, "data", "crops", crop_name)

        is_valid, reason = validate_image_file(crop_path)
        if not is_valid:
            corrupted_count += 1
            print(f"Skipping corrupted image {crop_name}: {reason}")
            continue

        with open(crop_path, "rb") as f:
            file_md5 = hashlib.md5(f.read()).hexdigest()

        if file_md5 in seen_md5:
            duplicate_md5_count += 1
        seen_md5.add(file_md5)

        fruit_str = row["fruit"].capitalize()
        fruit_id = fruit_map.get(fruit_str, -1)
        grade_4stage = int(row["grade"])
        stage_name_4 = ripeness_4stage[str(grade_4stage)]

        # Map to 3-stage config: 0 Unripe, 1 Partially Ripe -> Unripe(0); 2 Ripe -> Ripe(1); 3 Rotten -> Overripe(2)
        if grade_4stage in (0, 1):
            grade_3stage = 0  # Unripe
        elif grade_4stage == 2:
            grade_3stage = 1  # Ripe
        else:
            grade_3stage = 2  # Overripe

        img = Image.open(crop_path).convert("RGB")
        h0 = dhash(img)
        h1 = dhash(ImageOps.mirror(img))
        hashes.append((h0, h1))

        validated_rows.append({
            "image_id": crop_name,
            "filepath": os.path.abspath(crop_path),
            "relative_path": os.path.relpath(crop_path, ROOT).replace("\\", "/"),
            "source": row["source"],
            "fruit_name": fruit_str,
            "fruit_id": fruit_id,
            "ripeness_4stage_id": grade_4stage,
            "ripeness_4stage_name": stage_name_4,
            "ripeness_3stage_id": grade_3stage,
            "ripeness_3stage_name": ["Unripe", "Ripe", "Overripe"][grade_3stage],
            "md5": file_md5,
            "group": int(row["group"])
        })

    val_df = pd.DataFrame(validated_rows)

    # Near duplicate detection
    H0 = np.stack([h[0] for h in hashes])
    H1 = np.stack([h[1] for h in hashes])
    n = len(val_df)
    near_dup_pairs = 0
    for i in range(n):
        d = np.minimum((H0[i] != H0).sum(1), (H0[i] != H1).sum(1))
        matches = np.where((d <= HASH_THRESH) & (np.arange(n) > i))[0]
        near_dup_pairs += len(matches)

    out_csv = os.path.join(ANNOTATIONS_DIR, "manifest_validated.csv")
    val_df.to_csv(out_csv, index=False)
    print(f"Wrote validated manifest to {out_csv} ({len(val_df)} valid samples)")
    print(f"Integrity stats: Corrupted: {corrupted_count}, Exact MD5 duplicates: {duplicate_md5_count}, Near-duplicate pairs: {near_dup_pairs}")

    return val_df, {
        "total_images": len(val_df),
        "corrupted": corrupted_count,
        "md5_duplicates": duplicate_md5_count,
        "near_duplicate_pairs": near_dup_pairs,
        "fruit_distribution": val_df["fruit_name"].value_counts().to_dict(),
        "ripeness_4stage_distribution": val_df["ripeness_4stage_name"].value_counts().to_dict(),
        "ripeness_3stage_distribution": val_df["ripeness_3stage_name"].value_counts().to_dict(),
        "per_fruit_ripeness": pd.crosstab(val_df["fruit_name"], val_df["ripeness_4stage_name"]).to_dict()
    }


if __name__ == "__main__":
    run_validation()
