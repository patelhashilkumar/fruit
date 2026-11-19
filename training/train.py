"""Experiment orchestrator and training runner for Experiments A through F."""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch

from data.dataset import get_dataloaders
from models.resnet18_coral import ResNet18CORAL
from models.resnet50_softmax import ResNet50Softmax
from models.resnet50_coral import ResNet50CORAL
from models.multitask_resnet50 import MultiTaskResNet50
from training.trainer import Trainer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "experiments")
RESULTS_CSV = os.path.join(EXP_DIR, "results.csv")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(df):
    """Compute normalized class weights (mean = 1.0) for existing classes to prevent FP16 overflow."""
    # Ripeness weights (4-stage)
    r_counts = df["ripeness_4stage_id"].value_counts().sort_index()
    n_r = len(df)
    r_raw = np.array([n_r / max(1, r_counts.get(i, 1)) for i in range(4)], dtype=np.float32)
    r_norm = torch.tensor(r_raw / r_raw.mean(), dtype=torch.float32)

    # Fruit weights: only weight classes that exist; neutral 1.0 for others
    f_counts = df["fruit_id"].value_counts().sort_index()
    f_raw = np.ones(6, dtype=np.float32)
    for c in f_counts.index:
        f_raw[c] = len(df) / f_counts[c]
    present_mask = [c in f_counts.index for c in range(6)]
    if any(present_mask):
        f_raw[present_mask] /= f_raw[present_mask].mean()
    f_norm = torch.tensor(f_raw, dtype=torch.float32)

    return f_norm, r_norm


def run_experiment(exp_id, epochs=25, batch_size=8, seed=42):
    set_seed(seed)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    splits_manifest = os.path.join(ROOT, "dataset", "splits", "train.csv")
    train_df = pd.read_csv(splits_manifest)
    f_weights, r_weights = compute_class_weights(train_df)

    cfg = {
        "exp_id": exp_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "lr": 1e-4,
        "freeze_epochs": 3,
        "num_fruits": 6,
        "num_stages": 4
    }

    if exp_id == "Exp_A":
        # Baseline: ResNet18 + CORAL
        cfg["model_name"] = "ResNet18_CORAL_Baseline"
        model = ResNet18CORAL(num_stages=4, pretrained=True)
        model_type = "resnet18_coral"
        kw = {}

    elif exp_id == "Exp_B":
        # ResNet50 + Softmax Fruit Identification
        cfg["model_name"] = "ResNet50_Softmax_Fruit"
        model = ResNet50Softmax(num_classes=6, pretrained=True)
        model_type = "resnet50_softmax_fruit"
        kw = {}

    elif exp_id == "Exp_C":
        # ResNet50 + Softmax Ripeness Classification
        cfg["model_name"] = "ResNet50_Softmax_Ripeness"
        model = ResNet50Softmax(num_classes=4, pretrained=True)
        model_type = "resnet50_softmax_ripeness"
        kw = {}

    elif exp_id == "Exp_D":
        # ResNet50 + CORAL Ripeness Classification
        cfg["model_name"] = "ResNet50_CORAL_Ripeness"
        model = ResNet50CORAL(num_stages=4, pretrained=True)
        model_type = "resnet50_coral"
        kw = {}

    elif exp_id == "Exp_E":
        # Multi-task ResNet50 (Fruit Softmax + Ripeness CORAL)
        cfg["model_name"] = "MultiTask_ResNet50"
        model = MultiTaskResNet50(num_fruits=6, num_stages=4, pretrained=True)
        model_type = "multitask_resnet50"
        kw = {"lambda_ripeness": 1.0}

    elif exp_id == "Exp_F":
        # Multi-task ResNet50 with Class Balancing
        cfg["model_name"] = "MultiTask_ResNet50_Balanced"
        model = MultiTaskResNet50(num_fruits=6, num_stages=4, pretrained=True)
        model_type = "multitask_resnet50"
        kw = {
            "lambda_ripeness": 1.2,
            "fruit_weights": f_weights,
            "ripeness_weights": r_weights
        }
    else:
        raise ValueError(f"Unknown experiment ID: {exp_id}")

    # Save experiment config
    cfg_path = os.path.join(EXP_DIR, "configs", f"{exp_id}.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    trainer = Trainer(
        model=model,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        exp_name=exp_id,
        num_fruits=6,
        num_stages=4,
        lr=cfg["lr"],
        epochs=epochs,
        freeze_epochs=cfg["freeze_epochs"],
        **kw
    )

    results = trainer.fit()
    return results


def main():
    parser = argparse.ArgumentParser(description="Train experiments A through F")
    parser.add_argument("--experiment", default="all", choices=["all", "Exp_A", "Exp_B", "Exp_C", "Exp_D", "Exp_E", "Exp_F"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    exps = ["Exp_A", "Exp_B", "Exp_C", "Exp_D", "Exp_E", "Exp_F"] if args.experiment == "all" else [args.experiment]
    all_results = []

    for exp_id in exps:
        res = run_experiment(exp_id, epochs=args.epochs, batch_size=args.batch_size)
        all_results.append(res)

        # Update cumulative results.csv
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_CSV, index=False)
        print(f"Updated {RESULTS_CSV}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED:")
    print("=" * 80)
    summary_df = pd.DataFrame(all_results)
    print(summary_df[[
        "exp_name", "fruit_accuracy", "fruit_f1_macro",
        "ripeness_accuracy", "ripeness_mae", "ripeness_qwk", "ripeness_severe_error", "train_time_sec"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
