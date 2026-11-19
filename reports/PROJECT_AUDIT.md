# Project Audit: Fruit Ripening & Grading System

**Audit Date:** November 2026  
**Auditor:** Senior ML Engineer & Computer Vision Researcher  
**Hardware Profile:** NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM) | AMD Ryzen 9 7740HS | Windows 11 | Python 3.11.9 | PyTorch 2.5.1+cu121  
**Repository:** [AyushDas4890/Fruit-ripening-grading-system](https://github.com/AyushDas4890/Fruit-ripening-grading-system)

---

## 1. Executive Summary & Repository Objectives

The existing repository implements an automated fruit ripeness grading system formulated as an **ordinal regression** problem using a Rank-Consistent Ordinal Regression (CORAL) head on a ResNet18 backbone. The original system specifically targets 4 ordered stages of ripeness (`unripe` < `partially ripe` < `ripe` < `rotten`) across two fruit types (strawberry and avocado). 

The goal of this upgrade is to transform the system into an end-to-end **Fruit Identification and Ripeness Detection System** powered by **ResNet50**, supporting:
1. Multi-class Fruit Identification (initially Banana, Mango, Papaya, expanding to Avocado and Strawberry).
2. Multi-class / Ordinal Ripeness Detection.
3. Prediction confidence calibration.
4. Interpretability via Grad-CAM visualizations for both tasks.
5. Strict constraint adherence: **No quality grading or Grade A/B/C classification.**

---

## 2. In-Depth Audit Checklist

### 2.1 Current Dataset Size & Provenance
* **Dataset Source:** Kamat, P. et al. (2024), *Comprehensive Dataset on Ripening Stages of Strawberries and Avocados*, Mendeley Data, DOI `10.17632/zysvgmxcyz.1` (CC BY 4.0).
* **Raw Extracted Photos:** 758 raw full images (YOLO annotated bounding boxes).
* **Saved Crops in Repo:** 924 processed crops stored under `data/crops/` (all resized $\le 256 \times 256$ pixels, JPEG format).
* **Manifest:** Stored at `data/splits/manifest.csv` with 924 indexed entries mapping crop filename, original image source, YOLO class ID, fruit name, ripeness grade, group ID, and assigned CV fold.

### 2.2 Existing Fruit Categories
* **Avocado:** 576 crops (62.3% of total).
* **Strawberry:** 348 crops (37.7% of total).
* **Limitation:** The existing dataset does not include Banana, Mango, or Papaya.

### 2.3 Existing Ripeness Classes
The system defines $K = 4$ ordered classes:
* `Grade 0 (Unripe)`: 268 crops (156 avocado, 112 strawberry)
* `Grade 1 (Partially Ripe)`: 191 crops (112 avocado, 79 strawberry)
* `Grade 2 (Ripe)`: 117 crops (48 avocado, 69 strawberry)
* `Grade 3 (Rotten)`: 348 crops (260 avocado, 88 strawberry)
* **Class Imbalance Note:** Grade 2 (Ripe) is significantly under-represented, comprising only 12.7% of the total dataset, while Grade 3 comprises 37.7%.

### 2.4 Current Model Architecture
1. **Baseline CNN (`SmallCNN`):** Custom 4-stage convolutional neural network trained from scratch (conv2d $\to$ batchnorm $\to$ relu $\to$ maxpool $\to$ dropout 0.3 $\to$ fc 4).
2. **Transfer Learning Softmax (`tl_softmax`):** Standard ResNet18 backbone pretrained on ImageNet-1K (`ResNet18_Weights.IMAGENET1K_V1`), with the final linear layer replaced by `nn.Linear(512, 4)`.
3. **Transfer Learning CORAL (`tl_ordinal`):** ResNet18 backbone with custom `OrdinalHead`:
   * Shared feature projection vector: $\mathbf{w} \in \mathbb{R}^{512 \times 1}$ calculating a scalar ripeness score $s = \mathbf{w}^\top \mathbf{x}$.
   * Learned rank thresholds: $\mathbf{b} \in \mathbb{R}^{K-1}$ initialized at $[2.0, 0.0, -2.0]$.
   * Threshold logit outputs: $\text{logit}_k = s + b_k$ corresponding to cumulative probabilities $P(y > k) = \sigma(s + b_k)$.
   * Guarantees rank consistency by design (the threshold curves cannot cross because $\mathbf{w}$ is strictly shared).

### 2.5 Existing Accuracy & Experimental Performance
From 5-fold grouped cross-validation across 3 random seeds (pooled $N = 924$ test crops):

| Model Architecture | MAE (Mean ± Std) ↓ | Accuracy (Mean ± Std) ↑ | QWK (Mean ± Std) ↑ | Severe Errors (≥2 grades) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **Small CNN (Scratch, Softmax)** | $0.422 \pm 0.019$ | $66.1 \pm 2.0\%$ | $0.811 \pm 0.005$ | $7.40 \pm 0.31\%$ |
| **ResNet18 (Transfer, Softmax)** | $0.228 \pm 0.045$ | $79.0 \pm 3.8\%$ | $0.910 \pm 0.022$ | $1.37 \pm 0.50\%$ |
| **ResNet18 (Transfer, CORAL)** | **$0.214 \pm 0.030$** | **$79.5 \pm 2.5\%$** | **$0.924 \pm 0.013$** | **$0.79 \pm 0.38\%$** |

Per-fruit breakdowns highlight that the ordinal head halved severe errors compared to softmax (0.79% vs 1.37%) and improved strawberry classification significantly (MAE 0.281 vs 0.309).

### 2.6 Current Data Augmentation
* **Training Pipeline:**
  * `transforms.RandomResizedCrop(224, scale=(0.7, 1.0))`
  * `transforms.RandomHorizontalFlip()`
  * `transforms.RandomVerticalFlip()`
  * `transforms.RandomRotation(20)`
  * `transforms.ColorJitter(brightness=0.15, contrast=0.15)`
  * Normalization: ImageNet $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
  * **Critical Design Choice:** Hue and saturation jitter are intentionally excluded because chromatic signals directly represent fruit ripeness.
* **Validation & Test Pipeline:**
  * Deterministic `transforms.Resize((224, 224))` followed by normalization.

### 2.7 Train / Validation / Test Splits & Leakage Prevention
* **Splitting Scheme:** 5-fold grouped cross-validation.
* **Leakage Vector 1 (Identical Source Photos):** Crops originating from the same photograph share a source ID and are strictly forced into the same fold via `UnionFind`.
* **Leakage Vector 2 (Burst Sequences):** Photo series taken seconds apart with identical fruits are grouped using sequential filename block partitions (`num // 10`).
* **Leakage Vector 3 (Synthetic Duplicates / Pasted Backgrounds):** A difference hash (`dHash`, 256 bits, flip-invariant) is calculated over the central 60% of each crop. Image pairs with Hamming distance $\le 40$ are merged.
* **Stratification:** `StratifiedGroupKFold` optimizes class-mix deviation across folds over 200 seed trials. Zero group overlap is guaranteed between splits.

### 2.8 Duplicate Images
* The dHash pipeline merged duplicate and near-duplicate crops across the dataset into 159 isolated groups.

### 2.9 GPU Memory Requirements
* Previous setup: RTX 3050 Laptop (4GB VRAM) with mixed precision (AMP FP16), batch size 32.
* Host hardware: RTX 4050 Laptop (6.44 GB VRAM).
* ResNet50 has ~25.6M parameters (vs ResNet18's ~11.7M). ResNet50 in FP16 with batch size 8–16 uses ~1.8–3.2 GB VRAM, fitting comfortably within the 6GB limit.

### 2.10 Existing Application Functionality
* **Backend:** `webapp/server.py` with FastAPI serving:
  * `POST /api/predict`: Receives image upload, executes model ensemble and Grad-CAM generation.
  * `GET /api/results`: Delivers cached experiment statistics.
  * Static file mounts for `/results`, `/crops`, and `/static`.
* **Static Client-Side Deployment:** `deploy/build_static.py` compiles a zero-backend web deployment running ONNX Runtime Web in-browser (`infer.js`) powered by `ordinal.onnx` and `softmax.onnx`.
* **Frontend:** Interactive dashboard with SVG visual design, animated ripeness slider, probability distribution bars, Grad-CAM polaroid display, and HTML presentation deck.
* **Identified Gap:** The existing application has no fruit identification classification head or confidence display for fruit type.

---

## 3. Architecture Upgrade Roadmap

```
Existing Pipeline:
Input Crop -> ResNet18 -> CORAL Head (3 logits) -> Ripeness Stage (4 classes)

Target Upgraded Pipeline:
Input Crop
   |
Pretrained ResNet50 Feature Extractor
   |
   +---> Head 1: Fruit Identification (Softmax: Banana, Mango, Papaya, + Avocado, Strawberry)
   |
   +---> Head 2: Ripeness Detection (CORAL Ordinal Head: Unripe < Partially Ripe < Ripe < Rotten)
   |
   +---> Multi-Task Grad-CAM Generator (Heatmaps for Fruit ID & Ripeness)
```

---

## 4. Verification Check
- [x] Repository audit completed without editing legacy production code.
- [x] Baseline architectures catalogued and documented.
- [x] Hardware compatibility confirmed for NVIDIA RTX 4050 6GB.
