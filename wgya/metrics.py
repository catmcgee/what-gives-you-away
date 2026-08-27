"""Evaluation metrics shared by probe and downstream benchmarks."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def expected_calibration_error(y_idx, probs, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    y_idx = np.asarray(y_idx, dtype=int)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_idx
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_idx)
    ece = 0.0
    for lo, hi in pairwise(edges):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += (
                mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
            )
    return float(ece)


def classification_metrics(y_true, probs, classes) -> dict:
    classes = list(classes)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    y_idx = np.asarray([classes.index(str(y)) for y in y_true], dtype=int)
    pred_idx = probs.argmax(axis=1)
    pred = np.asarray([classes[i] for i in pred_idx])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=classes, zero_division=0
    )
    one_hot = np.eye(len(classes), dtype=float)[y_idx]
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(np.mean(f1)),
        "ece_10": expected_calibration_error(y_idx, probs, 10),
        "brier_multiclass": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "confusion": confusion_matrix(y_true, pred, labels=classes).tolist(),
        "per_class": {
            cls: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(classes)
        },
    }
