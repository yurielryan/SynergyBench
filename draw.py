import json
from pathlib import Path

import matplotlib.pyplot as plt

from analysis import classify

INPUT_PATH = Path("results/base_dataset.json")
OUTPUT_PATH = Path("results/interaction_donut.pdf")


def compute_counts():
    with open(INPUT_PATH) as f:
        data = json.load(f)
    counts = {"R": 0, "U1": 0, "U2": 0, "S": 0, "error": 0}
    for sample in data["results"].values():
        counts[classify(sample)] += 1
    return counts


def main():
    counts = compute_counts()

    labels = ["Redundant", "Unique (X1)", "Unique (X2)", "Synergistic", "Error"]
    keys = ["R", "U1", "U2", "S", "error"]
    values = [counts[k] for k in keys]
    total = sum(values)

    # Muted, conference-paper friendly palette (ColorBrewer-inspired)
    colors = ["#8C9BAB", "#B7A99A", "#A7B7A0", "#9A8AA8", "#CFCFCF"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw=dict(aspect="equal"))

    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.2),
    )

    legend_labels = [
        f"{lab}  {v/total*100:.1f}%  (n={v})"
        for lab, v in zip(labels, values)
    ]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        handlelength=1.2,
        handleheight=1.2,
        borderpad=0.0,
        labelspacing=0.8,
    )

    ax.text(0, 0, f"N = {total}", ha="center", va="center",
            fontsize=12, color="#444444")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300)
    plt.savefig(OUTPUT_PATH.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
