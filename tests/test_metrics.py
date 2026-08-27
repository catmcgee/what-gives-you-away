import numpy as np

from wgya.metrics import classification_metrics


def test_classification_metrics_perfect():
    classes = ["a", "b", "c"]
    y = ["a", "b", "c", "a"]
    probs = np.asarray(
        [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9], [0.8, 0.1, 0.1]]
    )
    metrics = classification_metrics(y, probs, classes)
    assert metrics["accuracy"] == 1
    assert metrics["balanced_accuracy"] == 1
    assert metrics["macro_f1"] == 1
    assert metrics["brier_multiclass"] > 0
