"""Leakage-free dataset splitting utility.

Performs grouped, stratified splitting into:
- Train: ~70%
- Validation: ~15%
- Test: ~15%

Guarantees that no images sharing a group ID (same source photo, burst session,
or near-duplicate cluster) span multiple splits.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")


def create_splits(manifest_csv=None, seed=42):
    os.makedirs(SPLITS_DIR, exist_ok=True)
    if manifest_csv is None:
        manifest_csv = os.path.join(ANNOTATIONS_DIR, "manifest_validated.csv")

    df = pd.read_csv(manifest_csv)
    n = len(df)

    # Use a compound target for stratification: fruit_id * 10 + ripeness_4stage_id
    y_strat = df["fruit_id"].values * 10 + df["ripeness_4stage_id"].values
    groups = df["group"].values
    indices = np.arange(n)

    # 7-fold split: ~71.4% train (5 folds), ~14.3% val (1 fold), ~14.3% test (1 fold)
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    fold_assignment = np.zeros(n, dtype=int)
    for fold_idx, (_, test_indices) in enumerate(sgkf.split(indices, y_strat, groups)):
        fold_assignment[test_indices] = fold_idx

    # Assign folds: fold 0 -> test (~14.3%), fold 1 -> val (~14.3%), folds 2..6 -> train (~71.4%)
    split_col = np.empty(n, dtype=object)
    split_col[fold_assignment == 0] = "test"
    split_col[fold_assignment == 1] = "val"
    split_col[fold_assignment >= 2] = "train"

    df["split"] = split_col
    df["fold_7"] = fold_assignment

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    # Verify zero group leakage
    train_groups = set(train_df["group"])
    val_groups = set(val_df["group"])
    test_groups = set(test_df["group"])

    leakage_train_val = train_groups & val_groups
    leakage_train_test = train_groups & test_groups
    leakage_val_test = val_groups & test_groups

    assert not leakage_train_val, f"Leakage detected between train and val: {leakage_train_val}"
    assert not leakage_train_test, f"Leakage detected between train and test: {leakage_train_test}"
    assert not leakage_val_test, f"Leakage detected between val and test: {leakage_val_test}"

    print("=" * 60)
    print("LEAKAGE-FREE SPLIT SUMMARY (70 / 15 / 15 Grouped & Stratified)")
    print("=" * 60)
    print(f"Total samples: {n}")
    print(f"Train split:   {len(train_df)} ({len(train_df)/n*100:.1f}%) | Groups: {len(train_groups)}")
    print(f"Val split:     {len(val_df)} ({len(val_df)/n*100:.1f}%) | Groups: {len(val_groups)}")
    print(f"Test split:    {len(test_df)} ({len(test_df)/n*100:.1f}%) | Groups: {len(test_groups)}")
    print("Group overlap check: ZERO LEAKAGE VERIFIED.")

    train_df.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)
    df.to_csv(os.path.join(SPLITS_DIR, "splits_manifest.csv"), index=False)

    return train_df, val_df, test_df


if __name__ == "__main__":
    create_splits()
