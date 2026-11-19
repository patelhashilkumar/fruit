"""Unified Fruit Identification & Ripeness Dataset for PyTorch.

Supports:
- Multi-task targets: fruit identification and ripeness classification.
- Task masking: handles images with missing fruit or ripeness labels.
- Both 3-stage and 4-stage ripeness configurations.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from data.augmentations import get_transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FruitDataset(Dataset):
    def __init__(self, df_or_csv, train=True, stage_config="4stage", img_size=224):
        if isinstance(df_or_csv, str):
            self.df = pd.read_csv(df_or_csv).reset_index(drop=True)
        else:
            self.df = df_or_csv.reset_index(drop=True)

        self.train = train
        self.stage_config = stage_config
        self.transform = get_transforms(train=train, img_size=img_size)

        # Preload image paths and verify existence
        self.filepaths = []
        for _, row in self.df.iterrows():
            fp = row.get("filepath")
            if not fp or not os.path.exists(fp):
                # Fallback check under data/crops
                crop_name = row.get("image_id", row.get("crop"))
                fp = os.path.join(ROOT, "dataset", "processed", "mendeley_crops", crop_name)
                if not os.path.exists(fp):
                    fp = os.path.join(ROOT, "data", "crops", crop_name)
            self.filepaths.append(fp)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)

        row = self.df.iloc[idx]

        # Fruit ID
        fruit_id = int(row["fruit_id"]) if "fruit_id" in row and not pd.isna(row["fruit_id"]) else -1
        has_fruit = 1.0 if fruit_id >= 0 else 0.0

        # Ripeness ID
        if self.stage_config == "3stage":
            ripeness_id = int(row["ripeness_3stage_id"]) if "ripeness_3stage_id" in row else -1
        else:
            ripeness_id = int(row["ripeness_4stage_id"]) if "ripeness_4stage_id" in row else int(row.get("grade", -1))

        has_ripeness = 1.0 if ripeness_id >= 0 else 0.0

        return {
            "image": tensor,
            "fruit_id": torch.tensor(max(0, fruit_id), dtype=torch.long),
            "ripeness_id": torch.tensor(max(0, ripeness_id), dtype=torch.long),
            "has_fruit": torch.tensor(has_fruit, dtype=torch.float32),
            "has_ripeness": torch.tensor(has_ripeness, dtype=torch.float32),
            "index": idx,
            "filename": os.path.basename(path)
        }


def get_dataloaders(splits_dir=None, batch_size=8, stage_config="4stage", num_workers=0):
    if splits_dir is None:
        splits_dir = os.path.join(ROOT, "dataset", "splits")

    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")

    train_ds = FruitDataset(train_csv, train=True, stage_config=stage_config)
    val_ds = FruitDataset(val_csv, train=False, stage_config=stage_config)
    test_ds = FruitDataset(test_csv, train=False, stage_config=stage_config)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
