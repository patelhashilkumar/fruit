"""Unit tests for model architectures: initialization, forward pass, output shapes, and multi-task loss."""
import torch
import unittest
from models.resnet18_coral import ResNet18CORAL
from models.resnet50_softmax import ResNet50Softmax
from models.resnet50_coral import ResNet50CORAL
from models.multitask_resnet50 import MultiTaskResNet50
from training.losses import MultiTaskLoss


class TestModels(unittest.TestCase):
    def setUp(self):
        self.dummy_input = torch.randn(2, 3, 224, 224)

    def test_resnet18_coral(self):
        model = ResNet18CORAL(num_stages=4, pretrained=False)
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (2, 3), "ResNet18 CORAL 4-stage must output 3 threshold logits")

    def test_resnet50_softmax(self):
        model = ResNet50Softmax(num_classes=6, pretrained=False)
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (2, 6), "ResNet50 Softmax 6 classes must output (B, 6)")

    def test_resnet50_coral(self):
        model = ResNet50CORAL(num_stages=4, pretrained=False)
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (2, 3), "ResNet50 CORAL 4-stage must output (B, 3)")

    def test_multitask_resnet50(self):
        model = MultiTaskResNet50(num_fruits=6, num_stages=4, pretrained=False)
        out = model(self.dummy_input)
        self.assertIn("fruit_logits", out)
        self.assertIn("ripeness_logits", out)
        self.assertIn("features", out)
        self.assertEqual(out["fruit_logits"].shape, (2, 6))
        self.assertEqual(out["ripeness_logits"].shape, (2, 3))
        self.assertEqual(out["features"].shape, (2, 2048, 7, 7))

    def test_multitask_loss_with_masking(self):
        loss_fn = MultiTaskLoss(num_fruits=6, num_stages=4, lambda_ripeness=1.0)
        fruit_logits = torch.randn(2, 6, requires_grad=True)
        ripeness_logits = torch.randn(2, 3, requires_grad=True)
        fruit_targets = torch.tensor([0, 1])
        ripeness_targets = torch.tensor([2, 3])
        has_fruit = torch.tensor([1.0, 0.0])      # Sample 1 missing fruit label
        has_ripeness = torch.tensor([0.0, 1.0])   # Sample 0 missing ripeness label

        tot_loss, f_loss, r_loss = loss_fn(
            fruit_logits, ripeness_logits,
            fruit_targets, ripeness_targets,
            has_fruit, has_ripeness
        )
        self.assertGreater(tot_loss.item(), 0)
        tot_loss.backward()
        self.assertIsNotNone(fruit_logits.grad)
        self.assertIsNotNone(ripeness_logits.grad)


if __name__ == "__main__":
    unittest.main()
