"""Unit tests for CORAL target encoding, loss computation, and probability conversions."""
import math
import torch
import unittest
from training.losses import ordinal_targets, coral_loss, coral_to_probs, coral_predict


class TestCORAL(unittest.TestCase):
    def test_target_encoding_4stage(self):
        y = torch.tensor([0, 1, 2, 3])
        targets = ordinal_targets(y, num_stages=4)
        expected = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        self.assertTrue(torch.equal(targets, expected), f"Expected {expected}, got {targets}")

    def test_target_encoding_3stage(self):
        y = torch.tensor([0, 1, 2])
        targets = ordinal_targets(y, num_stages=3)
        expected = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])
        self.assertTrue(torch.equal(targets, expected), f"Expected {expected}, got {targets}")

    def test_probabilities_sum_to_one(self):
        logits = torch.tensor([
            [5.0, 5.0, -5.0],
            [-5.0, -5.0, -5.0],
            [5.0, 5.0, 5.0]
        ])
        probs = coral_to_probs(logits)
        self.assertEqual(probs.shape, (3, 4))
        sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones(3), atol=1e-5), f"Probabilities must sum to 1: {sums}")

    def test_predict_count_matches(self):
        logits = torch.tensor([
            [5.0, 5.0, -5.0],   # >0 yes, >1 yes, >2 no -> grade 2
            [-5.0, -5.0, -5.0], # all no -> grade 0
            [5.0, 5.0, 5.0]     # all yes -> grade 3
        ])
        preds = coral_predict(logits)
        self.assertEqual(preds.tolist(), [2, 0, 3])

    def test_coral_loss_reduction(self):
        logits = torch.zeros((4, 3))
        targets = torch.tensor([0, 1, 2, 3])
        loss = coral_loss(logits, targets, num_stages=4, reduction="mean")
        # For clueless model (logits=0), BCE = ln(2) per threshold, 3 thresholds = 3 * ln(2)
        expected_loss = 3.0 * math.log(2)
        self.assertAlmostEqual(loss.item(), expected_loss, places=4)


if __name__ == "__main__":
    unittest.main()
