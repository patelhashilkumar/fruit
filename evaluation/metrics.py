"""Comprehensive evaluation metrics for Fruit Identification and Ripeness Detection."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score
)


def compute_fruit_metrics(y_true, y_pred, class_names=None):
    """Metrics for Fruit Identification task."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    per_class = {}
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    for cls in unique_classes:
        mask = (y_true == cls)
        name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        per_class[name] = float(accuracy_score(y_true[mask], y_pred[mask])) if mask.sum() > 0 else 0.0

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class
    }


def compute_ripeness_metrics(y_true, y_pred, stage_names=None):
    """Metrics for Ripeness Detection task (including ordinal metrics: MAE, QWK, severe errors)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    diff = np.abs(y_true - y_pred)
    mae = float(np.mean(diff))
    acc = float(accuracy_score(y_true, y_pred))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic")) if len(np.unique(y_true)) > 1 else 1.0

    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    adjacent_error_rate = float(np.mean(diff == 1))
    severe_error_rate = float(np.mean(diff >= 2))

    cm = confusion_matrix(y_true, y_pred)

    per_stage = {}
    unique_stages = np.unique(np.concatenate([y_true, y_pred]))
    for stg in unique_stages:
        mask = (y_true == stg)
        name = stage_names[stg] if stage_names and stg < len(stage_names) else str(stg)
        per_stage[name] = {
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])) if mask.sum() > 0 else 0.0,
            "mae": float(np.mean(diff[mask])) if mask.sum() > 0 else 0.0,
            "count": int(mask.sum())
        }

    return {
        "accuracy": acc,
        "mae": mae,
        "qwk": qwk,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "adjacent_error_rate": adjacent_error_rate,
        "severe_error_rate": severe_error_rate,
        "confusion_matrix": cm.tolist(),
        "per_stage_metrics": per_stage
    }
