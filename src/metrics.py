"""Evaluation metrics used across Tasks 4-7.

All functions take ``y_true`` (0/1 labels, 1 == fake) and ``y_score`` (predicted
probability of the fake class).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

__all__ = ["compute_metrics", "equal_error_rate", "roc_points", "confusion", "METRIC_KEYS"]

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "eer"]


def equal_error_rate(y_true, y_score) -> tuple[float, float]:
    """EER = the point where false-accept rate == false-reject rate.

    Returns ``(eer, threshold)``.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thr[idx])


def roc_points(y_true, y_score) -> dict:
    """FPR/TPR arrays + AUC, for plotting ROC curves (Task 8 / verification)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "auc": float(roc_auc_score(y_true, y_score))}


def confusion(y_true, y_score, threshold: float = 0.5) -> np.ndarray:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """All six standard metrics as a flat dict. ``threshold`` applies to the
    thresholded metrics (accuracy/precision/recall/f1); AUC and EER are
    threshold-free."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    both_classes = np.unique(y_true).size > 1
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both_classes else float("nan"),
    }
    if both_classes:
        eer, eer_thr = equal_error_rate(y_true, y_score)
    else:
        eer, eer_thr = float("nan"), float("nan")
    out["eer"] = eer
    out["eer_threshold"] = eer_thr
    return out
