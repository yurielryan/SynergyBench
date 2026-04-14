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
    colors = ["#6E8BA8", "#C2A878", "#8FB08C", "#9A7FB0", "#CFCFCF"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.2, 2.8), subplot_kw=dict(aspect="equal"))

    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.34, edgecolor="white", linewidth=1.0),
    )

    legend_labels = [
        f"{lab}   {v/total*100:4.1f}%   n = {v}"
        for lab, v in zip(labels, values)
    ]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=1.1,
        handleheight=1.1,
        borderpad=0.0,
        labelspacing=0.7,
        fontsize=9.5,
    )

    ax.text(0, 0.08, f"{total}", ha="center", va="center",
            fontsize=16, color="#2B2B2B")
    ax.text(0, -0.14, "samples", ha="center", va="center",
            fontsize=8.5, color="#707070", style="italic")

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.02, dpi=1200)
    fig.savefig(OUTPUT_PATH.with_suffix(".png"),
                bbox_inches="tight", pad_inches=0.02, dpi=1200)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
