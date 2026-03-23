"""
utils.py
========
Shared helpers: seed setting, edge splitting, metric computation.
"""

import random
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_edges(num_edges: int, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Return boolean train / val / test masks over edges."""
    rng     = np.random.default_rng(seed)
    idx     = rng.permutation(num_edges)
    n_train = int(num_edges * train_ratio)
    n_val   = int(num_edges * val_ratio)

    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask   = torch.zeros(num_edges, dtype=torch.bool)
    test_mask  = torch.zeros(num_edges, dtype=torch.bool)

    train_mask[idx[:n_train]]           = True
    val_mask  [idx[n_train:n_train+n_val]] = True
    test_mask [idx[n_train+n_val:]]     = True
    return train_mask, val_mask, test_mask


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "ROC_AUC"  : roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1"       : f1_score(y_true, y_pred, zero_division=0),
    }


def print_metrics(metrics: dict, title="Evaluation Results"):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print(f"{'='*50}\n")


def print_classification_report(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
          target_names=["Legit", "Fraud"], zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Confusion Matrix:")
    print(f"  TN={tn:>7,}  FP={fp:>7,}")
    print(f"  FN={fn:>7,}  TP={tp:>7,}\n")


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    pw    = n_neg / max(n_pos, 1)
    print(f"  pos_weight = {pw:.2f}  "
          f"(neg: {int(n_neg):,}, pos: {int(n_pos):,})")
    return torch.tensor(pw, dtype=torch.float32)
