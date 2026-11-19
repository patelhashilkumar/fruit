"""Unit tests for dataset loading, transforms, and data structures."""
import os
import unittest
import torch
from data.dataset import FruitDataset, get_dataloaders

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDataset(unittest.TestCase):
    def test_dataset_item_structure(self):
        splits_dir = os.path.join(ROOT, "dataset", "splits")
        train_csv = os.path.join(splits_dir, "train.csv")
        self.assertTrue(os.path.exists(train_csv), f"train.csv missing at {train_csv}")

        ds = FruitDataset(train_csv, train=True, stage_config="4stage")
        self.assertGreater(len(ds), 0)
        item = ds[0]
        self.assertIn("image", item)
        self.assertIn("fruit_id", item)
        self.assertIn("ripeness_id", item)
        self.assertIn("has_fruit", item)
        self.assertIn("has_ripeness", item)

        self.assertEqual(item["image"].shape, (3, 224, 224))
        self.assertIsInstance(item["fruit_id"], torch.Tensor)
        self.assertIsInstance(item["ripeness_id"], torch.Tensor)

    def test_dataloaders(self):
        train_dl, val_dl, test_dl = get_dataloaders(batch_size=4, num_workers=0)
        batch = next(iter(train_dl))
        self.assertEqual(batch["image"].shape, (4, 3, 224, 224))
        self.assertEqual(batch["fruit_id"].shape, (4,))
        self.assertEqual(batch["ripeness_id"].shape, (4,))


if __name__ == "__main__":
    unittest.main()
