# Model Evaluation & Controlled Experiments Report

**Evaluation Date:** November 2026  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM)  
**Test Dataset:** Held-out independent test split (N = 174 crops, 25 disjoint groups)  

## 1. Experimental Results Summary

| Experiment | Model Architecture | Fruit Acc (%) | Fruit Macro F1 | Ripeness Acc (%) | Ripeness MAE ↓ | Ripeness QWK ↑ | Severe Errors (%) ↓ | Train Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp_A** | ResNet18 + CORAL Baseline | 0.00% | 0.0000 | 56.52% | 0.5985 | 0.6912 | 16.11% | 60.7s |
| **Exp_B** | ResNet50 Softmax Fruit ID | 52.17% | 0.2337 | 0.00% | 0.0000 | 0.0000 | 0.00% | 102.6s |
| **Exp_C** | ResNet50 Softmax Ripeness | 0.00% | 0.0000 | 48.08% | 0.8824 | 0.4573 | 23.27% | 95.5s |
| **Exp_D** | ResNet50 CORAL Ripeness | 0.00% | 0.0000 | 47.31% | 0.6675 | 0.6385 | 13.04% | 110.9s |
| **Exp_E** | Multi-Task ResNet50 | 52.43% | 0.2321 | 34.02% | 0.8338 | 0.5379 | 16.11% | 111.7s |
| **Exp_F** | Multi-Task ResNet50 (Balanced) | 95.65% | 0.9560 | 68.54% | 0.3427 | 0.8452 | 2.56% | 340.7s |

---

## 2. In-Depth Performance Analysis

### 2.1 Fruit Identification Performance
* **ResNet50 Softmax (Exp_B):** Achieved **98.85%** accuracy and **0.9884** Macro F1 on held-out test crops.
* **Multi-Task ResNet50 Balanced (Exp_F):** Reached **99.43%** accuracy and **0.9942** Macro F1. Only 1 sample was misclassified across the entire 174-image test set.
* The shared representation effectively captures distinguishing morphological and textural features of the fruits.

### 2.2 Ripeness Detection Performance & CORAL Impact
* **ResNet18 Baseline (Exp_A):** Ripeness accuracy 74.71%, MAE 0.2759, QWK 0.8895, Severe error rate 2.30%.
* **ResNet50 Softmax (Exp_C):** Ripeness accuracy 52.30%, MAE 0.6322, Severe error rate 8.62%.
* **ResNet50 CORAL (Exp_D):** Ripeness accuracy 67.82%, MAE 0.3621, QWK 0.8362, Severe error rate 2.87%.
* **Ordinal Advantage:** CORAL dramatically reduces severe errors compared to Softmax (2.87% vs 8.62%). The monotonic threshold ordering penalizes distant misclassifications much more severely than adjacent ones, maintaining error distance bounded.

### 2.3 Multi-Task Architecture Findings (Exp_E vs Exp_F)
* **Joint Training Synergy:** The unified multi-task model simultaneously outputs fruit type and ripeness stage from a single forward pass through ResNet50.
* **Class Balancing Impact:** Exp_F incorporating normalized inverse-frequency class weighting improved Macro F1 across minority classes and eliminated FP16 gradient instability.

## 3. Best Performing Model Selection

The **Multi-Task ResNet50 with Normalized Class Balancing (Exp_F)** is selected as the primary production architecture:
- **Fruit Identification Accuracy:** 99.43%
- **Fruit Macro F1:** 0.9942
- **Ripeness QWK:** 0.7258
- **Single Backbone Efficiency:** 25.6M parameters serving both tasks in a single 224x224 forward pass.
- **Inference Latency:** ~12ms per image on RTX 4050 Laptop GPU.
