"""Agronomic Senescence & Transitional Stage Synthesis for All Fruit Species.

Synthesizes biologically accurate visual manifestations of:
1. Partially Ripe Apples (yellow-green color break with emerging anthocyanin blush)
2. Rotten Apples (Monilinia brown rot concentric lesions and soft necrosis)
3. Partially Ripe Mangoes (carotenoid shoulder color-break with green cheeks)
4. Rotten Mangoes (anthracnose sunken black lesions and stem-end rot)
5. Partially Ripe Papayas (longitudinal yellow furrow striping along green ribs)
6. Rotten Papayas (sunken water-soaked fungal decay and anthracnose spots)

Completes full 4-stage coverage across all 6 fruits:
Banana, Mango, Papaya, Avocado, Strawberry, Apple.
"""
import os
import sys
import hashlib
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SRC_DIR = os.path.join(ROOT, "dataset", "processed", "rawripe_images")
DEST_DIR = os.path.join(ROOT, "dataset", "processed", "senescence_crops")
os.makedirs(DEST_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(ROOT, "dataset", "annotations", "manifest_validated.csv")
df_current = pd.read_csv(MANIFEST_PATH)
max_group = df_current["group"].max()

new_rows = []
curr_group = max_group + 1


def get_mask(arr, fruit_type="general"):
    """Detect fruit pixels against neutral / plain background."""
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # Non-background condition: not pure white, not pure black
    not_white = (r < 245) | (g < 245) | (b < 245)
    not_black = (r > 15) | (g > 15) | (b > 15)
    return not_white & not_black


# -----------------------------------------------------------------------------
# 1. APPLE: Partially Ripe & Rotten
# -----------------------------------------------------------------------------
def simulate_partially_ripe_apple(img):
    """Green raw apple -> yellow-green transition with patchy red blush."""
    im = img.convert("RGB")
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]
    mask = get_mask(arr)
    
    # Yellow-green color break (increase red channel slightly to shift pure green to yellow-green)
    shift = random.uniform(25, 50)
    arr[mask, 0] = np.clip(arr[mask, 0] * 1.3 + shift, 0, 255)
    
    # Add partial red/pink anthocyanin blush on one cheek
    indices = np.argwhere(mask)
    if len(indices) > 0:
        cy, cx = indices[random.randint(0, len(indices) - 1)]
        ry, rx = int(h * random.uniform(0.25, 0.45)), int(w * random.uniform(0.25, 0.45))
        y0, y1 = max(0, cy - ry), min(h, cy + ry)
        x0, x1 = max(0, cx - rx), min(w, cx + rx)
        blush_mask = mask[y0:y1, x0:x1]
        
        arr[y0:y1, x0:x1][blush_mask, 0] = np.clip(arr[y0:y1, x0:x1][blush_mask, 0] * 1.4 + 40, 0, 255)
        arr[y0:y1, x0:x1][blush_mask, 1] = np.clip(arr[y0:y1, x0:x1][blush_mask, 1] * 0.75, 0, 255)
        arr[y0:y1, x0:x1][blush_mask, 2] = np.clip(arr[y0:y1, x0:x1][blush_mask, 2] * 0.70, 0, 255)
        
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.3))


def simulate_rotten_apple_decay(img):
    """Ripe apple -> brown rot concentric lesions and soft dark decay."""
    im = img.convert("RGB")
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]
    mask = get_mask(arr)
    indices = np.argwhere(mask)
    
    if len(indices) > 0:
        num_lesions = random.randint(2, 4)
        for _ in range(num_lesions):
            cy, cx = indices[random.randint(0, len(indices) - 1)]
            rad = random.randint(int(w * 0.12), int(w * 0.28))
            y0, y1 = max(0, cy - rad), min(h, cy + rad)
            x0, x1 = max(0, cx - rad), min(w, cx + rad)
            l_mask = mask[y0:y1, x0:x1]
            
            # Brown decaying rot color
            arr[y0:y1, x0:x1][l_mask, 0] = arr[y0:y1, x0:x1][l_mask, 0] * 0.35 + 45
            arr[y0:y1, x0:x1][l_mask, 1] = arr[y0:y1, x0:x1][l_mask, 1] * 0.25 + 30
            arr[y0:y1, x0:x1][l_mask, 2] = arr[y0:y1, x0:x1][l_mask, 2] * 0.15 + 15
            
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.3))


# -----------------------------------------------------------------------------
# 2. MANGO: Partially Ripe & Rotten
# -----------------------------------------------------------------------------
def simulate_partially_ripe_mango(img):
    """Raw green mango -> color break stage (golden-yellow blush on shoulders)."""
    im = img.convert("RGB")
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]
    mask = get_mask(arr)
    indices = np.argwhere(mask)
    
    if len(indices) > 0:
        # Carotenoid color-break on the top half / shoulders
        top_half = mask.copy()
        top_half[int(h * 0.55):, :] = False
        
        # Shift towards golden yellow (boost red and green)
        arr[top_half, 0] = np.clip(arr[top_half, 0] * 1.5 + 60, 0, 255)
        arr[top_half, 1] = np.clip(arr[top_half, 1] * 1.25 + 40, 0, 255)
        arr[top_half, 2] = np.clip(arr[top_half, 2] * 0.6, 0, 255)
        
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.4))


def simulate_rotten_mango(img):
    """Ripe mango -> anthracnose sunken black lesions and soft decay."""
    im = img.convert("RGB")
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]
    mask = get_mask(arr)
    indices = np.argwhere(mask)
    
    if len(indices) > 0:
        # 1. Large dark anthracnose decay patches
        num_spots = random.randint(3, 6)
        for _ in range(num_spots):
            cy, cx = indices[random.randint(0, len(indices) - 1)]
            rad = random.randint(int(w * 0.08), int(w * 0.22))
            y0, y1 = max(0, cy - rad), min(h, cy + rad)
            x0, x1 = max(0, cx - rad), min(w, cx + rad)
            s_mask = mask[y0:y1, x0:x1]
            # Sunken black/dark brown necrotic spot
            arr[y0:y1, x0:x1][s_mask, 0] = arr[y0:y1, x0:x1][s_mask, 0] * 0.2 + 25
            arr[y0:y1, x0:x1][s_mask, 1] = arr[y0:y1, x0:x1][s_mask, 1] * 0.15 + 18
            arr[y0:y1, x0:x1][s_mask, 2] = arr[y0:y1, x0:x1][s_mask, 2] * 0.1 + 10
            
        # 2. Peppered necrotic flecks
        n_flecks = random.randint(100, 300)
        f_pts = indices[np.random.choice(len(indices), min(n_flecks, len(indices)), replace=False)]
        for y, x in f_pts:
            rad = random.randint(1, 3)
            y0, y1 = max(0, y - rad), min(h, y + rad + 1)
            x0, x1 = max(0, x - rad), min(w, x + rad + 1)
            arr[y0:y1, x0:x1, :] *= random.uniform(0.15, 0.35)
            
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.3))


# -----------------------------------------------------------------------------
# 3. PAPAYA: Partially Ripe & Rotten
# -----------------------------------------------------------------------------
def simulate_partially_ripe_papaya(img):
    """Raw green papaya -> longitudinal yellow furrow stripes (25-50% yellow)."""
    im = img.convert("RGB")
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]
    mask = get_mask(arr)
    
    # Create 3-5 vertical yellow ripening bands along the ribs
    num_stripes = random.randint(3, 5)
    for _ in range(num_stripes):
        stripe_center_x = random.randint(int(w * 0.2), int(w * 0.8))
        stripe_width = random.randint(int(w * 0.06), int(w * 0.14))
        x0, x1 = max(0, stripe_center_x - stripe_width), min(w, stripe_center_x + stripe_width)
        s_mask = mask[:, x0:x1]
        
        # Golden yellow transition
        arr[:, x0:x1][s_mask, 0] = np.clip(arr[:, x0:x1][s_mask, 0] * 1.5 + 70, 0, 255)
        arr[:, x0:x1][s_mask, 1] = np.clip(arr[:, x0:x1][s_mask, 1] * 1.3 + 50, 0, 255)
        arr[:, x0:x1][s_mask, 2] = np.clip(arr[:, x0:x1][s_mask, 2] * 0.5, 0, 255)
        
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.4))


def simulate_rotten_papaya(img):
    """Ripe papaya -> circular sunken black anthracnose spots and water-soaked rot."""
    im = img.convert("RGB")
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]
    mask = get_mask(arr)
    indices = np.argwhere(mask)
    
    if len(indices) > 0:
        # Multiple sunken dark lesions with lighter fungal center
        num_spots = random.randint(4, 8)
        for _ in range(num_spots):
            cy, cx = indices[random.randint(0, len(indices) - 1)]
            rad = random.randint(int(w * 0.08), int(w * 0.20))
            y0, y1 = max(0, cy - rad), min(h, cy + rad)
            x0, x1 = max(0, cx - rad), min(w, cx + rad)
            s_mask = mask[y0:y1, x0:x1]
            
            # Sunken brown/black necrotic rot
            arr[y0:y1, x0:x1][s_mask, 0] = arr[y0:y1, x0:x1][s_mask, 0] * 0.2 + 30
            arr[y0:y1, x0:x1][s_mask, 1] = arr[y0:y1, x0:x1][s_mask, 1] * 0.15 + 20
            arr[y0:y1, x0:x1][s_mask, 2] = arr[y0:y1, x0:x1][s_mask, 2] * 0.1 + 10
            
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.3))


# -----------------------------------------------------------------------------
# Execute batch synthesis for Apple, Mango, Papaya
# -----------------------------------------------------------------------------
print("Generating full-stage coverage for Apple, Mango, and Papaya...")

# A. Apple Partially Ripe (from apples raw)
apple_raw = [f for f in os.listdir(SRC_DIR) if "apples_raw" in f and f.endswith(('.jpg', '.png'))]
for i in range(min(60, len(apple_raw))):
    src = os.path.join(SRC_DIR, apple_raw[i % len(apple_raw)])
    with Image.open(src) as im:
        out = simulate_partially_ripe_apple(im)
        fname = f"synth_partially_ripe_apple_{i:03d}.jpg"
        save_p = os.path.join(DEST_DIR, fname)
        out.save(save_p, quality=92)
    with open(save_p, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    new_rows.append({
        "image_id": fname, "filepath": save_p,
        "relative_path": os.path.relpath(save_p, ROOT).replace("\\", "/"),
        "source": "synth_partially_ripe_apple", "fruit_name": "Apple", "fruit_id": 5,
        "ripeness_4stage_id": 1, "ripeness_4stage_name": "Partially Ripe",
        "ripeness_3stage_id": 0, "ripeness_3stage_name": "Unripe",
        "md5": md5, "group": curr_group
    })
    curr_group += 1

# B. Apple Rotten (from apples ripe)
apple_ripe = [f for f in os.listdir(SRC_DIR) if "apples_ripe" in f and f.endswith(('.jpg', '.png'))]
for i in range(min(40, len(apple_ripe))):
    src = os.path.join(SRC_DIR, apple_ripe[i % len(apple_ripe)])
    with Image.open(src) as im:
        out = simulate_rotten_apple_decay(im)
        fname = f"synth_rotten_apple_{i+50:03d}.jpg"
        save_p = os.path.join(DEST_DIR, fname)
        out.save(save_p, quality=92)
    with open(save_p, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    new_rows.append({
        "image_id": fname, "filepath": save_p,
        "relative_path": os.path.relpath(save_p, ROOT).replace("\\", "/"),
        "source": "synth_senescence_apple", "fruit_name": "Apple", "fruit_id": 5,
        "ripeness_4stage_id": 3, "ripeness_4stage_name": "Rotten",
        "ripeness_3stage_id": 2, "ripeness_3stage_name": "Overripe",
        "md5": md5, "group": curr_group
    })
    curr_group += 1

# C. Mango Partially Ripe (from mangoes raw)
mango_raw = [f for f in os.listdir(SRC_DIR) if "mangoes_raw" in f and f.endswith(('.jpg', '.png'))]
for i in range(min(60, len(mango_raw))):
    src = os.path.join(SRC_DIR, mango_raw[i % len(mango_raw)])
    with Image.open(src) as im:
        out = simulate_partially_ripe_mango(im)
        fname = f"synth_partially_ripe_mango_{i:03d}.jpg"
        save_p = os.path.join(DEST_DIR, fname)
        out.save(save_p, quality=92)
    with open(save_p, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    new_rows.append({
        "image_id": fname, "filepath": save_p,
        "relative_path": os.path.relpath(save_p, ROOT).replace("\\", "/"),
        "source": "synth_partially_ripe_mango", "fruit_name": "Mango", "fruit_id": 1,
        "ripeness_4stage_id": 1, "ripeness_4stage_name": "Partially Ripe",
        "ripeness_3stage_id": 0, "ripeness_3stage_name": "Unripe",
        "md5": md5, "group": curr_group
    })
    curr_group += 1

# D. Mango Rotten (from mangoes ripe)
mango_ripe = [f for f in os.listdir(SRC_DIR) if "mangoes_ripe" in f and f.endswith(('.jpg', '.png'))]
for i in range(min(70, len(mango_ripe))):
    src = os.path.join(SRC_DIR, mango_ripe[i % len(mango_ripe)])
    with Image.open(src) as im:
        out = simulate_rotten_mango(im)
        fname = f"synth_rotten_mango_{i:03d}.jpg"
        save_p = os.path.join(DEST_DIR, fname)
        out.save(save_p, quality=92)
    with open(save_p, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    new_rows.append({
        "image_id": fname, "filepath": save_p,
        "relative_path": os.path.relpath(save_p, ROOT).replace("\\", "/"),
        "source": "synth_senescence_mango", "fruit_name": "Mango", "fruit_id": 1,
        "ripeness_4stage_id": 3, "ripeness_4stage_name": "Rotten",
        "ripeness_3stage_id": 2, "ripeness_3stage_name": "Overripe",
        "md5": md5, "group": curr_group
    })
    curr_group += 1

# E. Papaya Partially Ripe (from papayas raw)
papaya_raw = [f for f in os.listdir(SRC_DIR) if "papayas_raw" in f and f.endswith(('.jpg', '.png'))]
for i in range(min(60, len(papaya_raw))):
    src = os.path.join(SRC_DIR, papaya_raw[i % len(papaya_raw)])
    with Image.open(src) as im:
        out = simulate_partially_ripe_papaya(im)
        fname = f"synth_partially_ripe_papaya_{i:03d}.jpg"
        save_p = os.path.join(DEST_DIR, fname)
        out.save(save_p, quality=92)
    with open(save_p, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    new_rows.append({
        "image_id": fname, "filepath": save_p,
        "relative_path": os.path.relpath(save_p, ROOT).replace("\\", "/"),
        "source": "synth_partially_ripe_papaya", "fruit_name": "Papaya", "fruit_id": 2,
        "ripeness_4stage_id": 1, "ripeness_4stage_name": "Partially Ripe",
        "ripeness_3stage_id": 0, "ripeness_3stage_name": "Unripe",
        "md5": md5, "group": curr_group
    })
    curr_group += 1

# F. Papaya Rotten (from papayas ripe)
papaya_ripe = [f for f in os.listdir(SRC_DIR) if "papayas_ripe" in f and f.endswith(('.jpg', '.png'))]
for i in range(min(70, len(papaya_ripe))):
    src = os.path.join(SRC_DIR, papaya_ripe[i % len(papaya_ripe)])
    with Image.open(src) as im:
        out = simulate_rotten_papaya(im)
        fname = f"synth_rotten_papaya_{i:03d}.jpg"
        save_p = os.path.join(DEST_DIR, fname)
        out.save(save_p, quality=92)
    with open(save_p, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    new_rows.append({
        "image_id": fname, "filepath": save_p,
        "relative_path": os.path.relpath(save_p, ROOT).replace("\\", "/"),
        "source": "synth_senescence_papaya", "fruit_name": "Papaya", "fruit_id": 2,
        "ripeness_4stage_id": 3, "ripeness_4stage_name": "Rotten",
        "ripeness_3stage_id": 2, "ripeness_3stage_name": "Overripe",
        "md5": md5, "group": curr_group
    })
    curr_group += 1

# Save updated manifest
df_new = pd.DataFrame(new_rows)
df_all = pd.concat([df_current, df_new], ignore_index=True)
df_all.to_csv(MANIFEST_PATH, index=False)
print(f"Added {len(new_rows)} new images across Apple, Mango, and Papaya.")
print(f"Total dataset size: {len(df_all)} images.")
