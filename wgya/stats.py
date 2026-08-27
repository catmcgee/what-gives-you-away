"""Cluster-aware inference utilities for the conversation experiment."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def cluster_bootstrap_mean(values, clusters, n_boot=5000, seed=0):
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    grouped = {cluster: values[clusters == cluster] for cluster in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws[index] = np.concatenate([grouped[cluster] for cluster in sampled]).mean()
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
        "n_observations": len(values),
        "n_clusters": len(unique),
        "cluster_unit": "base message",
    }


def multiway_pigeonhole_ratio(
    numerator, denominator, cluster_dimensions, n_boot=5000, seed=0
):
    """Bootstrap a ratio of means with independent weights per cluster axis."""

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    dimensions = [np.asarray(values) for values in cluster_dimensions]
    uniques = [np.unique(values) for values in dimensions]
    indices = [{value: i for i, value in enumerate(unique)} for unique in uniques]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        cluster_weights = [
            rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
            for unique in uniques
        ]
        weights = np.ones(len(numerator), dtype=float)
        for values, lookup, sampled in zip(dimensions, indices, cluster_weights):
            weights *= np.asarray([sampled[lookup[value]] for value in values])
        denominator_sum = np.sum(weights * denominator)
        if weights.sum() and abs(denominator_sum) > 1e-12:
            draws.append(float(np.sum(weights * numerator) / denominator_sum))
    return {
        "ratio_of_means": float(numerator.mean() / denominator.mean()),
        "ci95": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
        "n_observations": len(numerator),
        "cluster_counts": [len(unique) for unique in uniques],
        "n_valid_bootstrap": len(draws),
    }


def benjamini_hochberg(p_values: dict) -> dict:
    """Return monotone Benjamini-Hochberg q-values."""

    items = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 1.0
    for rank, (key, p_value) in reversed(list(enumerate(items, start=1))):
        running = min(running, float(p_value) * len(items) / rank)
        adjusted[key] = min(1.0, running)
    return adjusted


def cluster_permutation_pvalue(values, clusters, n_perm=10000, seed=0):
    """Two-sided sign-flip test at the independent-cluster level."""

    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(float(value))
    means = np.asarray([np.mean(group) for group in grouped.values()])
    observed = abs(means.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(means)))
    null = abs((signs * means).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (n_perm + 1))
