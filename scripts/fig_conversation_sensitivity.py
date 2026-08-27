"""Plot the primary context-matched cue map and short-horizon retention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wgya import config
from wgya.io_utils import newest_result

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)

ATTRS = ["age", "gender", "education", "socioeco"]
ATTR_LABELS = {
    "age": "age",
    "gender": "gender",
    "education": "education",
    "socioeco": "socioecon.",
}
CATEGORIES = [
    "disclosure",
    "affect",
    "grammar",
    "price",
    "emoji",
    "slang",
    "orthography",
    "dialect",
    "formality",
]
CATEGORY_LABELS = {
    "disclosure": "personal context",
    "affect": "affect language",
    "grammar": "grammatical complexity",
    "price": "price language",
    "emoji": "emoji",
    "slang": "slang",
    "orthography": "orthography",
    "dialect": "US-to-UK spelling",
    "formality": "contraction expansion",
}
RETENTION_CELLS = [
    ("disclosure", "age"),
    ("price", "socioeco"),
    ("emoji", "gender"),
    ("slang", "age"),
    ("grammar", "education"),
    ("orthography", "education"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    path = args.path or newest_result(
        config.RESULTS_DIR / "llama3b", "summary_*.json"
    )
    if path is None:
        sys.exit("no context-matched conversation summary found")

    results = json.loads(path.read_text())["results"]
    matrix = results["summaries"]["primary_current"]["matrix"]
    retention = results["retention"]

    effects = np.asarray(
        [
            [matrix[category][attr]["paired_vs_control"]["mean"] for attr in ATTRS]
            for category in CATEGORIES
        ]
    )

    fig = plt.figure(figsize=(9.6, 4.35))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.17], wspace=0.68)

    ax = fig.add_subplot(grid[0, 0])
    vmax = float(np.percentile(effects, 98))
    if effects.min() < 0:
        norm = TwoSlopeNorm(vmin=float(effects.min()), vcenter=0, vmax=vmax)
        image = ax.imshow(effects, cmap="RdBu_r", norm=norm, aspect="auto")
    else:
        image = ax.imshow(effects, cmap="YlOrBr", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ATTRS)), [ATTR_LABELS[attr] for attr in ATTRS])
    ax.set_yticks(
        range(len(CATEGORIES)), [CATEGORY_LABELS[category] for category in CATEGORIES]
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row, category in enumerate(CATEGORIES):
        for col, attr in enumerate(ATTRS):
            cell = matrix[category][attr]
            value = cell["paired_vs_control"]["mean"]
            significant = cell["paired_vs_control"]["q_bh"] < 0.05
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.8,
                fontweight="bold" if significant else "normal",
                color=(
                    "white"
                    if value > 0.58 * vmax or value < 0.58 * effects.min()
                    else "#2b2118"
                ),
            )
    colorbar = fig.colorbar(
        image, ax=ax, shrink=0.82, pad=0.12, orientation="horizontal", aspect=28
    )
    colorbar.set_label("extra max-|delta logit| movement", fontsize=8)
    colorbar.outline.set_visible(False)
    ax.set_title("(a) Current-turn conversation cue map", loc="left")

    ax = fig.add_subplot(grid[0, 1])
    y = np.arange(len(RETENTION_CELLS))
    offsets = {"one_turn_later": -0.11, "two_turns_later": 0.11}
    styles = {
        "one_turn_later": ("one turn later", "#3d5a80", "o"),
        "two_turns_later": ("two turns later", "#b25d35", "s"),
    }
    for lag, (label, color, marker) in styles.items():
        records = [
            retention[lag][category][attr]["ratio"]
            for category, attr in RETENTION_CELLS
        ]
        means = np.asarray([record["ratio_of_means"] for record in records])
        intervals = np.asarray([record["ci95"] for record in records])
        ax.errorbar(
            means,
            y + offsets[lag],
            xerr=np.vstack([means - intervals[:, 0], intervals[:, 1] - means]),
            fmt=marker,
            color=color,
            ecolor=color,
            markersize=4.5,
            elinewidth=1.1,
            capsize=2.5,
            label=label,
        )
    short_categories = {
        "disclosure": "personal context",
        "price": "price language",
        "emoji": "emoji",
        "slang": "slang",
        "grammar": "grammar",
        "orthography": "orthography",
    }
    labels = [
        f"{short_categories[category]} / {ATTR_LABELS[attr]}"
        for category, attr in RETENTION_CELLS
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(1, color="#777777", lw=0.9, ls="--")
    ax.axvline(0, color="#333333", lw=0.7)
    ax.set_xlim(0, 1.38)
    ax.set_xlabel("retained fraction of current-turn effect")
    ax.set_title("(b) What remains after later turns", loc="left")
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    out_dir = config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_conversation_sensitivity.{extension}")
    plt.close(fig)
    print(f"wrote {out_dir}/fig_conversation_sensitivity.[pdf,png] from {path.name}")


if __name__ == "__main__":
    main()
