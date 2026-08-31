"""Validation for the metrics module (Task 4)."""

import numpy as np

from src.metrics import compute_metrics, confusion, equal_error_rate, roc_points


def test_perfect_classifier():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.01, 0.2, 0.9, 0.99, 0.1, 0.7])
    m = compute_metrics(y_true, y_score)
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["eer"] == 0.0


def test_inverted_classifier_has_auc_zero():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.1, 0.2])
    assert compute_metrics(y_true, y_score)["roc_auc"] == 0.0


def test_all_metric_keys_present():
    y_true = np.random.randint(0, 2, 50)
    y_score = np.random.rand(50)
    m = compute_metrics(y_true, y_score)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "eer", "eer_threshold"):
        assert k in m and np.isfinite(m[k])


def test_single_class_does_not_crash():
    m = compute_metrics(np.zeros(10, int), np.random.rand(10))
    assert np.isnan(m["roc_auc"]) and np.isnan(m["eer"])


def test_equal_error_rate_symmetric_case():
    # scores symmetric about 0.5 -> EER near 0.5 threshold
    y_true = np.array([0, 1] * 20)
    y_score = np.linspace(0, 1, 40)
    eer, thr = equal_error_rate(y_true, y_score)
    assert 0.0 <= eer <= 1.0 and 0.0 <= thr <= 1.0


def test_confusion_matrix_shape_and_counts():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.9, 0.9, 0.1])
    cm = confusion(y_true, y_score)
    assert cm.shape == (2, 2)
    assert cm.sum() == 4


def test_roc_points_monotone():
    y_true = np.random.randint(0, 2, 100)
    y_score = np.random.rand(100)
    r = roc_points(y_true, y_score)
    assert np.all(np.diff(r["fpr"]) >= 0) and np.all(np.diff(r["tpr"]) >= 0)
    assert 0.0 <= r["auc"] <= 1.0
