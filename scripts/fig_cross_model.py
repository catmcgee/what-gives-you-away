"""Plot cross-model agreement using cue ranks computed within each readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)

ATTRS = ("age", "gender", "education", "socioeco")
COLORS = {
    "age": "#4C78A8",
    "gender": "#E45756",
    "education": "#59A14F",
    "socioeco": "#D6A22A",
}
MODEL_LABELS = {
    "unsloth/Llama-3.2-3B-Instruct": "Llama 3.2 3B",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "allenai/OLMo-2-0325-32B-Instruct": "OLMo 2 32B",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def cells(summary: dict) -> tuple[list[str], np.ndarray]:
    matrix = summary["results"]["summaries"]["primary_current"]["matrix"]
    keys = sorted(f"{category}|{attr}" for category in matrix for attr in ATTRS)
    effects = np.asarray(
        [
            matrix[category][attr]["paired_vs_control"]["mean"]
            for category, attr in (key.split("|") for key in keys)
        ]
    )
    return keys, effects


def within_readout_ranks(keys: list[str], effects: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(effects, dtype=float)
    for attr in ATTRS:
        selected = np.asarray([key.endswith(f"|{attr}") for key in keys])
        ranks[selected] = rankdata(effects[selected])
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument(
        "--output-stem", type=Path, default=Path("results/fig_cross_model")
    )
    args = parser.parse_args()

    reference = load(args.reference)
    candidates = [load(path) for path in args.candidates]
    reference_keys, reference_effects = cells(reference)
    reference_ranks = within_readout_ranks(reference_keys, reference_effects)
    reference_name = MODEL_LABELS.get(
        reference["config"]["model"], reference["config"]["model"]
    )

    fig, axes = plt.subplots(
        1, len(candidates), figsize=(4.25 * len(candidates), 3.75), squeeze=False
    )
    for index, (ax, candidate) in enumerate(zip(axes[0], candidates, strict=True)):
        candidate_keys, candidate_effects = cells(candidate)
        if candidate_keys != reference_keys:
            raise ValueError("cue matrices do not contain the same cells")
        candidate_ranks = within_readout_ranks(candidate_keys, candidate_effects)
        rho = float(spearmanr(reference_ranks, candidate_ranks).statistic)

        for attr in ATTRS:
            selected = np.asarray([key.endswith(f"|{attr}") for key in reference_keys])
            ax.scatter(
                reference_ranks[selected],
                candidate_ranks[selected],
                s=34,
                color=COLORS[attr],
                edgecolor="white",
                linewidth=0.55,
                alpha=0.9,
                label="socioeconomic" if attr == "socioeco" else attr,
            )

        disagreements = np.argsort(np.abs(candidate_ranks - reference_ranks))[-3:]
        annotation_counts: dict[tuple[float, float], int] = {}
        annotation_offsets = ((4, 4), (4, -10), (4, 14))
        for cell_index in disagreements:
            category, attr = reference_keys[cell_index].split("|")
            point = (
                float(reference_ranks[cell_index]),
                float(candidate_ranks[cell_index]),
            )
            occurrence = annotation_counts.get(point, 0)
            annotation_counts[point] = occurrence + 1
            ax.annotate(
                f"{category}/{attr}",
                point,
                xytext=annotation_offsets[min(occurrence, 2)],
                textcoords="offset points",
                fontsize=6.6,
                color="#383838",
            )

        candidate_name = MODEL_LABELS.get(
            candidate["config"]["model"], candidate["config"]["model"]
        )
        ax.plot([0.5, 9.5], [0.5, 9.5], color="#999999", lw=0.9, ls="--", zorder=0)
        ax.set(
            xlim=(0.5, 9.5),
            ylim=(0.5, 9.5),
            xticks=range(1, 10),
            yticks=range(1, 10),
            xlabel=f"{reference_name} cue rank within readout",
        )
        ax.set_ylabel(f"{candidate_name} cue rank within readout")
        ax.set_title(
            f"({chr(97 + index)}) {reference_name} vs {candidate_name}\n"
            rf"Spearman $\rho$ = {rho:.3f}",
            loc="left",
        )
        ax.grid(color="#E4E4E4", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.3)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(args.output_stem.with_suffix(f".{extension}"))
    plt.close(fig)
    print(f"wrote {args.output_stem}.[pdf,png]")


if __name__ == "__main__":
    main()
