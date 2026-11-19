"""Image transform and augmentation pipelines.

Design constraints:
- Preserves natural color/chromaticity: NEVER alters hue or saturation.
- Lighting variation handled via subtle brightness and contrast adjustments.
- Mild spatial transformations (crop, flip, small rotation) to model natural orientation.
- Deterministic transforms for validation and evaluation.
"""
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train=True, img_size=224):
    """Return torchvision transform pipeline for training or inference/eval."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),  # Lighting only, no hue!
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
