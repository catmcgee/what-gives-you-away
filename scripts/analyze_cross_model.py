"""Compare conversation-sensitivity summaries without equating probe-logit scales."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

PERSISTENCE_CELLS = (
    ("disclosure", "age"),
    ("price", "socioeco"),
    ("emoji", "gender"),
    ("slang", "age"),
    ("grammar", "education"),
    ("orthography", "education"),
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def model_name(summary: dict) -> str:
    return summary["config"]["model"]


def assess_probe_gate(report: dict, threshold: float = 0.75) -> dict:
    attributes = {}
    for attribute, record in report["attributes"].items():
        scores = [
            float(seed["test"]["balanced_accuracy"])
            for seed in record["seeds"].values()
        ]
        attributes[attribute] = {
            "balanced_accuracy": scores,
            "mean": float(np.mean(scores)),
            "minimum": min(scores),
            "passes": all(score >= threshold for score in scores),
        }
    return {
        "model": report["model"],
        "threshold": threshold,
        "all_attributes_pass": all(record["passes"] for record in attributes.values()),
        "attributes": attributes,
    }


def primary_cells(summary: dict) -> dict[str, dict[str, float]]:
    matrix = summary["results"]["summaries"]["primary_current"]["matrix"]
    return {
        f"{category}|{attribute}": {
            "effect": float(cell["paired_vs_control"]["mean"]),
            "q_bh": float(cell["paired_vs_control"]["q_bh"]),
        }
        for category, attributes in matrix.items()
        for attribute, cell in attributes.items()
    }


def dominant_readouts(summary: dict) -> dict[str, str]:
    matrix = summary["results"]["summaries"]["primary_current"]["matrix"]
    return {
        category: max(
            attributes,
            key=lambda attribute: attributes[attribute]["paired_vs_control"]["mean"],
        )
        for category, attributes in matrix.items()
    }


def persistence_ratios(summary: dict, view: str) -> dict[str, float | None]:
    retention = summary["results"]["retention"][view]
    output = {}
    for category, attribute in PERSISTENCE_CELLS:
        ratio = retention[category][attribute]["ratio"]
        output[f"{category}|{attribute}"] = (
            None if ratio is None else float(ratio["ratio_of_means"])
        )
    return output


def compare_persistence(reference: dict, candidate: dict, view: str) -> dict:
    reference_ratios = persistence_ratios(reference, view)
    candidate_ratios = persistence_ratios(candidate, view)
    keys = [
        key
        for key in reference_ratios
        if reference_ratios[key] is not None and candidate_ratios[key] is not None
    ]
    reference_values = np.asarray([reference_ratios[key] for key in keys])
    candidate_values = np.asarray([candidate_ratios[key] for key in keys])
    return {
        "reference": reference_ratios,
        "candidate": candidate_ratios,
        "n_cells": len(keys),
        "spearman_rho": float(
            stats.spearmanr(reference_values, candidate_values).statistic
        ),
        "mean_absolute_difference": float(
            np.mean(np.abs(reference_values - candidate_values))
        ),
    }


def compare(reference: dict, candidate: dict) -> dict:
    reference_cells = primary_cells(reference)
    candidate_cells = primary_cells(candidate)
    keys = sorted(reference_cells)
    if keys != sorted(candidate_cells):
        raise ValueError("primary cue matrices do not contain the same cells")

    reference_effects = np.asarray(
        [reference_cells[key]["effect"] for key in keys], dtype=float
    )
    candidate_effects = np.asarray(
        [candidate_cells[key]["effect"] for key in keys], dtype=float
    )
    rho = stats.spearmanr(reference_effects, candidate_effects).statistic
    reference_significant = [key for key in keys if reference_cells[key]["q_bh"] < 0.05]
    sign_agreement = [
        key
        for key in reference_significant
        if np.sign(reference_cells[key]["effect"])
        == np.sign(candidate_cells[key]["effect"])
    ]
    corrected_overlap = [
        key for key in reference_significant if candidate_cells[key]["q_bh"] < 0.05
    ]

    reference_dominant = dominant_readouts(reference)
    candidate_dominant = dominant_readouts(candidate)
    matched_cues = [
        category
        for category in sorted(reference_dominant)
        if reference_dominant[category] == candidate_dominant[category]
    ]
    return {
        "reference_model": model_name(reference),
        "candidate_model": model_name(candidate),
        "n_cells": len(keys),
        "spearman_rho": float(rho),
        "reference_significant_cells": len(reference_significant),
        "reference_significant_cell_names": reference_significant,
        "candidate_significant_cells": sum(
            cell["q_bh"] < 0.05 for cell in candidate_cells.values()
        ),
        "direction_agreement": {
            "count": len(sign_agreement),
            "fraction": len(sign_agreement) / len(reference_significant),
            "cells": sign_agreement,
            "disagreements": sorted(set(reference_significant) - set(sign_agreement)),
        },
        "corrected_significance_overlap": {
            "count": len(corrected_overlap),
            "fraction": len(corrected_overlap) / len(reference_significant),
            "cells": corrected_overlap,
            "not_significant": sorted(
                set(reference_significant) - set(corrected_overlap)
            ),
        },
        "dominant_readout_agreement": {
            "count": len(matched_cues),
            "fraction": len(matched_cues) / len(reference_dominant),
            "matched_cues": matched_cues,
            "reference": reference_dominant,
            "candidate": candidate_dominant,
        },
        "persistence": {
            view: compare_persistence(reference, candidate, view)
            for view in ("one_turn_later", "two_turns_later")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=PROJECT_ROOT / "reproducibility" / "cross_model_plan.json",
    )
    parser.add_argument(
        "--training-reports",
        type=Path,
        nargs="+",
        help="ordered reports for the reference and every candidate",
    )
    args = parser.parse_args()

    reference = load_summary(args.reference)
    candidates = [load_summary(path) for path in args.candidates]
    output = {
        "plan": portable_path(args.plan),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "estimand": (
            "Cross-model rank, sign, corrected-significance, dominant-readout, and "
            "retention agreement; raw independently fitted probe-logit magnitudes are "
            "not compared across models."
        ),
        "comparisons": [compare(reference, candidate) for candidate in candidates],
    }
    if args.training_reports:
        expected = len(candidates) + 1
        if len(args.training_reports) != expected:
            parser.error(f"--training-reports requires {expected} paths")
        output["probe_gate"] = [
            assess_probe_gate(load_summary(path)) for path in args.training_reports
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {args.output}")
    for record in output["comparisons"]:
        print(
            f"{record['candidate_model']}: rho={record['spearman_rho']:.3f}, "
            f"direction={record['direction_agreement']['count']}/"
            f"{record['reference_significant_cells']}, "
            f"q-overlap={record['corrected_significance_overlap']['count']}/"
            f"{record['reference_significant_cells']}, "
            f"dominant={record['dominant_readout_agreement']['count']}/9"
        )


if __name__ == "__main__":
    main()
