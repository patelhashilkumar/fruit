"""Biological decay and senescence synthesis for fruit ripeness training.

Generates realistic senescence patterns:
1. Melanin enzymatic browning (dark necrotic peel)
2. Senescence spotting (small dense necrotic flecks)
3. Amorphous rot patches (large soft-rot lesions)

Applies to Banana, Apple, Mango to ensure all 4 stages (Unripe, Partially Ripe, Ripe, Rotten)
are represented in training across species.
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

def simulate_rotten_banana(img):
    """Transform a ripe banana into an authentic overripe/rotten banana."""
    im = img.convert("RGB")
    w, h = im.size
    arr = np.array(im, dtype=np.float32)

    # Detect banana pixels (yellow/light green regions, distinct from white/dark background)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # Yellow peel condition: high red, high green, lower blue
    is_banana = (r > 100) & (g > 90) & (r > b + 20) & (g > b + 15)

    # 1. Enzymatic melanin browning: convert yellow peel to dark brown
    browning_factor = random.uniform(0.35, 0.55)
    # Target dark brown color: R=70-90, G=45-60, B=25-35
    target_r = np.random.uniform(60, 95, size=arr.shape[:2])
    target_g = np.random.uniform(40, 65, size=arr.shape[:2])
    target_b = np.random.uniform(20, 40, size=arr.shape[:2])

    blend = random.uniform(0.65, 0.90)
    arr[is_banana, 0] = arr[is_banana, 0] * (1 - blend) + target_r[is_banana] * blend
    arr[is_banana, 1] = arr[is_banana, 1] * (1 - blend) + target_g[is_banana] * blend
    arr[is_banana, 2] = arr[is_banana, 2] * (1 - blend) + target_b[is_banana] * blend

    # 2. Add necrotic senescence spots (dark black/brown flecks)
    banana_indices = np.argwhere(is_banana)
    if len(banana_indices) > 0:
        num_spots = random.randint(150, 400)
        spot_centers = banana_indices[np.random.choice(len(banana_indices), min(num_spots, len(banana_indices)), replace=False)]
        for y, x in spot_centers:
            rad = random.randint(1, 4)
            y_min, y_max = max(0, y - rad), min(h, y + rad + 1)
            x_min, x_max = max(0, x - rad), min(w, x + rad + 1)
            # Dark necrotic spot
            arr[y_min:y_max, x_min:x_max, :] *= random.uniform(0.2, 0.5)

    res_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # Subtle softening to integrate spots
    res_img = res_img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return res_img

def simulate_rotten_apple(img):
    """Transform a ripe apple into a rotten apple with brown rot lesions."""
    im = img.convert("RGB")
    w, h = im.size
    arr = np.array(im, dtype=np.float32)

    # Detect apple pixels (high red or green vs neutral background)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    is_apple = ((r > g + 10) | (g > r + 10)) & (r + g + b < 680)

    # Brown rot lesions
    apple_indices = np.argwhere(is_apple)
    if len(apple_indices) > 0:
        num_lesions = random.randint(2, 5)
        lesion_centers = apple_indices[np.random.choice(len(apple_indices), min(num_lesions, len(apple_indices)), replace=False)]
        for y, x in lesion_centers:
            rad = random.randint(int(w * 0.15), int(w * 0.35))
            y_min, y_max = max(0, y - rad), min(h, y + rad + 1)
            x_min, x_max = max(0, x - rad), min(w, x + rad + 1)
            # Brown rotting color
            arr[y_min:y_max, x_min:x_max, 0] = arr[y_min:y_max, x_min:x_max, 0] * 0.4 + 50
            arr[y_min:y_max, x_min:x_max, 1] = arr[y_min:y_max, x_min:x_max, 1] * 0.3 + 35
            arr[y_min:y_max, x_min:x_max, 2] = arr[y_min:y_max, x_min:x_max, 2] * 0.2 + 20

    res_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return res_img

if __name__ == "__main__":
    # Test on a banana sample
    test_p = "dataset/processed/rawripe_images/rawripe_bananas_ripe_Banana_Ripe_test_103.jpg"
    if os.path.exists(test_p):
        with Image.open(test_p) as im:
            out = simulate_rotten_banana(im)
            out.save("test_rotten_banana_sample.jpg")
            print("Successfully created test_rotten_banana_sample.jpg")
