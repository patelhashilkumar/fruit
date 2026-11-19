"""Unit tests for the end-to-end inference pipeline."""
import os
import unittest
from PIL import Image
from inference.predictor import FruitRipenessPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestInference(unittest.TestCase):
    def setUp(self):
        self.predictor = FruitRipenessPredictor(stage_config="4stage")
        # Create a test synthetic fruit image
        self.test_img = Image.new("RGB", (224, 224), color=(255, 100, 50))

    def test_prediction_output_schema(self):
        res = self.predictor.predict(self.test_img, generate_cam=True)
        self.assertTrue(res.get("success"), f"Inference failed: {res}")
        self.assertIn("fruit_type", res)
        self.assertIn("fruit_confidence", res)
        self.assertIn("ripeness", res)
        self.assertIn("ripeness_confidence", res)
        self.assertIn("ripeness_score", res)
        self.assertIn("gradcam", res)
        self.assertTrue(res["gradcam"].startswith("data:image/png;base64,"))
        self.assertEqual(res["model"], "ResNet50-Multitask-CORAL")

    def test_invalid_image(self):
        tiny_img = Image.new("RGB", (8, 8))
        res = self.predictor.predict(tiny_img)
        self.assertFalse(res.get("success"))
        self.assertIn("error", res)


if __name__ == "__main__":
    unittest.main()
