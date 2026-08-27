"""Aggregate the context-matched conversation sensitivity experiment."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wgya import config
from wgya.io_utils import newest_result
from wgya.stats import (
    benjamini_hochberg,
    cluster_bootstrap_mean,
    cluster_permutation_pvalue,
    multiway_pigeonhole_ratio,
)

N_BOOT = 5000
N_PERM = 10000
CONDITION_SPECS = {
    "primary_current": [("current_help", 0), ("current_advice", 0)],
    "current_help_suffix0": [("current_help", 0)],
    "current_help_suffix1": [("current_help", 1)],
    "current_help_suffix2": [("current_help", 2)],
    "current_help_suffix3": [("current_help", 3)],
    "current_advice_suffix0": [("current_advice", 0)],
    "one_turn_later": [("one_turn_later", 0)],
    "two_turns_later": [("two_turns_later", 0)],
}


def read_payload(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def movement(seed_record: dict) -> float:
    return float(max(abs(value) for value in seed_record["dlogits"]))


def measurements_for(
    rows: list[dict], conditions: list[tuple[str, int]], attr: str
) -> list[dict]:
    selected = [row for row in rows if (row["view"], row["suffix_index"]) in conditions]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["pair_id"]].append(row)
    if not grouped:
        raise ValueError(f"no rows for {conditions}")
    expected = len(conditions)
    if any(len(group) != expected for group in grouped.values()):
        raise ValueError(f"incomplete condition crossing for {conditions}")

    output = []
    for pair_id, group in grouped.items():
        seed_names = sorted(group[0]["attrs"][attr]["seeds"])
        if any(sorted(row["attrs"][attr]["seeds"]) != seed_names for row in group):
            raise ValueError(f"probe seed mismatch for {pair_id} {attr}")
        seed_movements = {
            seed: float(
                np.mean([movement(row["attrs"][attr]["seeds"][seed]) for row in group])
            )
            for seed in seed_names
        }
        first = group[0]
        output.append(
            {
                "pair_id": pair_id,
                "base_id": first["base_id"],
                "category": first["category"],
                "seed_movements": seed_movements,
                "ensemble_movement": float(np.mean(list(seed_movements.values()))),
                "ensemble_flip_fraction": float(
                    np.mean(
                        [row["attrs"][attr]["ensemble"]["flipped"] for row in group]
                    )
                ),
            }
        )
    return output


def by_base(
    measurements: list[dict], category: str, value_key: str
) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in measurements:
        if row["category"] == category:
            if value_key == "ensemble_movement":
                value = row[value_key]
            else:
                value = row["seed_movements"][value_key]
            grouped[row["base_id"]].append(value)
    return {base: float(np.mean(values)) for base, values in grouped.items()}


def base_differences(
    measurements: list[dict], category: str, value_key: str = "ensemble_movement"
) -> dict[str, float]:
    category_by_base = by_base(measurements, category, value_key)
    control_by_base = by_base(measurements, "control", value_key)
    bases = sorted(set(category_by_base) & set(control_by_base))
    return {base: category_by_base[base] - control_by_base[base] for base in bases}


def cell_summary(measurements: list[dict], category: str) -> dict:
    category_rows = [row for row in measurements if row["category"] == category]
    category_values = [row["ensemble_movement"] for row in category_rows]
    category_clusters = [row["base_id"] for row in category_rows]
    control_values = [
        row["ensemble_movement"] for row in measurements if row["category"] == "control"
    ]
    differences = base_differences(measurements, category)
    bases = sorted(differences)
    values = np.asarray([differences[base] for base in bases], dtype=float)
    paired = cluster_bootstrap_mean(values, bases, n_boot=N_BOOT, seed=config.SEED)
    paired["p_signflip"] = cluster_permutation_pvalue(
        values, bases, n_perm=N_PERM, seed=config.SEED
    )

    seed_names = sorted(category_rows[0]["seed_movements"])
    seed_effects = {}
    for seed in seed_names:
        seed_differences = base_differences(measurements, category, seed)
        seed_effects[seed] = float(np.mean(list(seed_differences.values())))

    return {
        "n": len(category_rows),
        "n_base_clusters": len(set(category_clusters)),
        "category_movement": cluster_bootstrap_mean(
            category_values, category_clusters, n_boot=N_BOOT, seed=config.SEED
        ),
        "control_mean": float(np.mean(control_values)),
        "paired_vs_control": paired,
        "probe_seed_effects": seed_effects,
        "probe_seed_range": [
            float(min(seed_effects.values())),
            float(max(seed_effects.values())),
        ],
        "ensemble_flip_fraction": float(
            np.mean([row["ensemble_flip_fraction"] for row in category_rows])
        ),
    }


def summarize(
    rows: list[dict], name: str, conditions: list[tuple[str, int]], attrs: list[str]
) -> dict:
    categories = sorted(
        {row["category"] for row in rows if row["category"] != "control"}
    )
    matrix = {category: {} for category in categories}
    raw_p = {}
    for attr in attrs:
        measurements = measurements_for(rows, conditions, attr)
        for category in categories:
            cell = cell_summary(measurements, category)
            matrix[category][attr] = cell
            raw_p[f"{category}|{attr}"] = cell["paired_vs_control"]["p_signflip"]
    for key, q_value in benjamini_hochberg(raw_p).items():
        category, attr = key.split("|", 1)
        matrix[category][attr]["paired_vs_control"]["q_bh"] = q_value
    return {
        "name": name,
        "conditions": [
            {"view": view, "suffix_index": suffix_index}
            for view, suffix_index in conditions
        ],
        "matrix": matrix,
    }


def matrix_effects(summary: dict, attrs: list[str]) -> tuple[list[str], np.ndarray]:
    keys, values = [], []
    for category in sorted(summary["matrix"]):
        for attr in attrs:
            keys.append(f"{category}|{attr}")
            values.append(
                summary["matrix"][category][attr]["paired_vs_control"]["mean"]
            )
    return keys, np.asarray(values, dtype=float)


def compare_matrices(reference: dict, candidate: dict, attrs: list[str]) -> dict:
    reference_keys, reference_values = matrix_effects(reference, attrs)
    candidate_keys, candidate_values = matrix_effects(candidate, attrs)
    if reference_keys != candidate_keys:
        raise ValueError("matrix keys do not align")
    rho, p_value = stats.spearmanr(reference_values, candidate_values)
    differences = candidate_values - reference_values
    worst_index = int(np.argmax(np.abs(differences)))
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "mean_absolute_cell_change": float(np.mean(np.abs(differences))),
        "worst_cell": reference_keys[worst_index],
        "worst_cell_change": float(differences[worst_index]),
        "n_cells": len(reference_keys),
    }


def retention_summary(
    rows: list[dict],
    numerator_condition: tuple[str, int],
    denominator_conditions: list[tuple[str, int]],
    attrs: list[str],
) -> dict:
    categories = sorted(
        {row["category"] for row in rows if row["category"] != "control"}
    )
    output = {category: {} for category in categories}
    for attr in attrs:
        numerator_measurements = measurements_for(rows, [numerator_condition], attr)
        denominator_measurements = measurements_for(rows, denominator_conditions, attr)
        for category in categories:
            numerator = base_differences(numerator_measurements, category)
            denominator = base_differences(denominator_measurements, category)
            bases = sorted(set(numerator) & set(denominator))
            num = np.asarray([numerator[base] for base in bases], dtype=float)
            den = np.asarray([denominator[base] for base in bases], dtype=float)
            record = {
                "later_effect": float(num.mean()),
                "current_effect": float(den.mean()),
                "n_base_clusters": len(bases),
            }
            if abs(den.mean()) > 1e-8:
                record["ratio"] = multiway_pigeonhole_ratio(
                    num, den, [bases], n_boot=N_BOOT, seed=config.SEED
                )
            else:
                record["ratio"] = None
            output[category][attr] = record
    return output


def analyze_rows(rows: list[dict]) -> dict:
    """Apply the complete frozen aggregation protocol to result rows."""
    if not rows:
        raise ValueError("cannot analyze an empty result set")
    attrs = sorted(rows[0]["attrs"])
    if attrs != ["age", "education", "gender", "socioeco"]:
        raise ValueError(f"unexpected readouts: {attrs}")

    summaries = {
        name: summarize(rows, name, conditions, attrs)
        for name, conditions in CONDITION_SPECS.items()
    }
    comparison_pairs = {
        "conversation_shells": (
            summaries["current_help_suffix0"],
            summaries["current_advice_suffix0"],
        ),
        "suffix_1_vs_default": (
            summaries["current_help_suffix0"],
            summaries["current_help_suffix1"],
        ),
        "suffix_2_vs_default": (
            summaries["current_help_suffix0"],
            summaries["current_help_suffix2"],
        ),
        "suffix_3_vs_default": (
            summaries["current_help_suffix0"],
            summaries["current_help_suffix3"],
        ),
    }
    comparisons = {
        name: compare_matrices(reference, candidate, attrs)
        for name, (reference, candidate) in comparison_pairs.items()
    }
    retention = {
        "one_turn_later": retention_summary(
            rows,
            ("one_turn_later", 0),
            CONDITION_SPECS["primary_current"],
            attrs,
        ),
        "two_turns_later": retention_summary(
            rows,
            ("two_turns_later", 0),
            CONDITION_SPECS["primary_current"],
            attrs,
        ),
    }
    return {
        "attrs": attrs,
        "primary_estimand": (
            "five-seed ensemble max-|delta logit| averaged over two current-turn "
            "conversation shells, minus matched random control by base"
        ),
        "summaries": summaries,
        "comparisons": comparisons,
        "retention": retention,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    path = args.path or newest_result(
        config.RESULTS_DIR / "llama3b", "deltas_*.json"
    )
    if path is None:
        path = newest_result(
            config.RESULTS_DIR / "llama3b", "deltas_*.json.gz"
        )
    if path is None:
        sys.exit("no conversation delta result found")
    payload = read_payload(path)
    rows = payload["results"]
    results = analyze_rows(rows)
    output_name = path.name.removesuffix(".gz").replace("deltas_", "summary_")
    output = path.with_name(output_name)
    output.write_text(
        json.dumps(
            {
                "config": payload["config"]
                | {
                    "source": path.name,
                    "aggregate_n_boot": N_BOOT,
                    "aggregate_n_perm": N_PERM,
                    "aggregate_seed": config.SEED,
                },
                "written_at": datetime.now(UTC).isoformat(),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Wrote {output}")
    print("\nPrimary paired effects (category - matched control)")
    primary = results["summaries"]["primary_current"]["matrix"]
    print(" " * 14 + "".join(f"{attr:>12s}" for attr in results["attrs"]))
    for category in sorted(primary):
        print(
            f"{category:14s}"
            + "".join(
                f"{primary[category][attr]['paired_vs_control']['mean']:12.3f}"
                for attr in results["attrs"]
            )
        )
    print("\nRobustness correlations")
    for name, comparison in results["comparisons"].items():
        print(f"  {name:28s} rho={comparison['spearman_rho']:.3f}")


if __name__ == "__main__":
    main()
