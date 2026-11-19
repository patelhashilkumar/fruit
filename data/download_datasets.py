"""Dataset acquisition and downloading utility.

Provides automated fetching where supported and exact instructions for manual / Kaggle retrieval:
1. Primary Dataset: Multi-modal Fruit Ripeness Detection (Banana, Mango, Papaya)
2. Mendeley Fruit Ripeness Dataset (Avocado, Strawberry)
3. Derived Fruit Ripeness 12-Class Dataset (Apple, Banana, Mango)
"""
import argparse
import os
import sys
import urllib.request
import ssl
import zipfile
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "dataset", "raw")
PROCESSED_DIR = os.path.join(ROOT, "dataset", "processed")

KAGGLE_INSTRUCTIONS = """
================================================================================
KAGGLE DATASET DOWNLOAD INSTRUCTIONS (Primary Dataset)
================================================================================
The primary dataset from 'Multi-modal-Fruit-Ripeness-Detection-System' is hosted on Kaggle:
- Sourced across Banana (4,711), Mango (1,643), Papaya (300) images.

To download via Kaggle CLI:
1. Ensure kaggle is installed:
   pip install kaggle

2. Place your Kaggle API key at:
   C:\\Users\\<username>\\.kaggle\\kaggle.json (on Windows)

3. Run:
   kaggle datasets download -d shashwatwork/fruit-ripeness-dataset -p dataset/raw/
   kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p dataset/raw/

4. Extract the downloaded archive(s) into:
   dataset/raw/primary_multimodal/
================================================================================
"""

def print_instructions():
    print(KAGGLE_INSTRUCTIONS)

def extract_archive(archive_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Extracting {archive_path} -> {dest_dir}...")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f"Extracted to {dest_dir}")

def copy_existing_crops():
    """Copy the existing 924 avocado and strawberry crops into dataset/processed/."""
    legacy_crops = os.path.join(ROOT, "data", "crops")
    dest_dir = os.path.join(PROCESSED_DIR, "mendeley_crops")
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(legacy_crops):
        count = 0
        for f in os.listdir(legacy_crops):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(legacy_crops, f)
                dst = os.path.join(dest_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                count += 1
        print(f"Synchronized {count} legacy crops to {dest_dir}")
    else:
        print(f"Legacy crops directory not found at {legacy_crops}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and manage datasets")
    parser.add_argument("--instructions", action="store_true", help="Print Kaggle manual download steps")
    parser.add_argument("--sync-existing", action="store_true", help="Sync existing repository crops into dataset/processed")
    args = parser.parse_args()

    print_instructions()
    copy_existing_crops()
