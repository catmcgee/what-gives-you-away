import numpy as np

from wgya.stats import (
    benjamini_hochberg,
    cluster_bootstrap_mean,
    cluster_permutation_pvalue,
    multiway_pigeonhole_ratio,
)


def test_cluster_bootstrap_is_deterministic_and_counts_clusters():
    first = cluster_bootstrap_mean(
        [1, 2, 10, 11], ["a", "a", "b", "b"], n_boot=200, seed=7
    )
    second = cluster_bootstrap_mean(
        [1, 2, 10, 11], ["a", "a", "b", "b"], n_boot=200, seed=7
    )
    assert first == second
    assert first["n_clusters"] == 2


def test_bh_is_monotone_in_rank():
    q = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.5})
    assert q["a"] <= q["b"] <= q["c"]


def test_multiway_ratio_recovers_constant_ratio():
    denominator = np.arange(1, 13, dtype=float)
    numerator = denominator * 0.4
    result = multiway_pigeonhole_ratio(
        numerator,
        denominator,
        [["a"] * 6 + ["b"] * 6, list(range(6)) * 2],
        n_boot=200,
        seed=0,
    )
    assert abs(result["ratio_of_means"] - 0.4) < 1e-12
    assert all(abs(value - 0.4) < 1e-12 for value in result["ci95"])


def test_cluster_sign_flip_detects_large_consistent_effect():
    p = cluster_permutation_pvalue(
        [2, 3, 4, 5, 6, 7, 8, 9], list("abcdefgh"), n_perm=5000, seed=0
    )
    assert p < 0.02
