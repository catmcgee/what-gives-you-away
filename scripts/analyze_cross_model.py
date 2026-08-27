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
ATTRIBUTES = ("age", "education", "gender", "socioeco")
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


def within_readout_rank_agreement(
    reference_cells: dict[str, dict[str, float]],
    candidate_cells: dict[str, dict[str, float]],
) -> dict:
    """Compare cue order separately for each readout.

    Probe logits have no common scale across independently fitted readouts. Ranking
    the nine cues within each readout removes that arbitrary between-probe scale.
    """
    reference_ranks = []
    candidate_ranks = []
    by_readout = {}
    for attribute in ATTRIBUTES:
        keys = sorted(
            key for key in reference_cells if key.endswith(f"|{attribute}")
        )
        reference_effects = np.asarray(
            [reference_cells[key]["effect"] for key in keys], dtype=float
        )
        candidate_effects = np.asarray(
            [candidate_cells[key]["effect"] for key in keys], dtype=float
        )
        reference_attribute_ranks = stats.rankdata(reference_effects)
        candidate_attribute_ranks = stats.rankdata(candidate_effects)
        reference_ranks.extend(reference_attribute_ranks)
        candidate_ranks.extend(candidate_attribute_ranks)
        by_readout[attribute] = {
            "n_cues": len(keys),
            "spearman_rho": float(
                stats.spearmanr(
                    reference_attribute_ranks, candidate_attribute_ranks
                ).statistic
            ),
        }
    return {
        "scope": "cue ranks computed separately within each readout",
        "n_cells": len(reference_ranks),
        "pooled_spearman_rho": float(
            stats.spearmanr(reference_ranks, candidate_ranks).statistic
        ),
        "by_readout": by_readout,
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

    rank_agreement = within_readout_rank_agreement(reference_cells, candidate_cells)
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

    return {
        "reference_model": model_name(reference),
        "candidate_model": model_name(candidate),
        "n_cells": len(keys),
        "rank_agreement": rank_agreement,
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
            "Cross-model within-readout cue-rank, sign, corrected-significance, and "
            "retention agreement; raw independently fitted probe-logit magnitudes "
            "are not compared across readouts or models."
        ),
        "analysis_deviation": (
            "The study plan's pooled 36-cell rank and highest-moving-readout "
            "endpoints compare separately scaled probes. They are replaced here "
            "with cue ranks computed within each readout. Cell-level effects, "
            "inference, and retention endpoints are unchanged."
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
            f"{record['candidate_model']}: within-readout rho="
            f"{record['rank_agreement']['pooled_spearman_rho']:.3f}, "
            f"direction={record['direction_agreement']['count']}/"
            f"{record['reference_significant_cells']}, "
            f"q-overlap={record['corrected_significance_overlap']['count']}/"
            f"{record['reference_significant_cells']}"
        )


if __name__ == "__main__":
    main()
