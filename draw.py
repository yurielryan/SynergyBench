import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import classify

INPUT_PATH = Path("results/base_dataset.json")
OUTPUT_PATH = Path("results/interaction_donut.pdf")
TEXTMOD_OUTPUT_PATH = Path("results/text_modification.pdf")

PALETTE = {
    "R": "#6E8BA8",
    "U1": "#C2A878",
    "U2": "#8FB08C",
    "S": "#9A7FB0",
    "error": "#CFCFCF",
    "base": "#6E8BA8",
}


def _apply_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


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

    colors = [PALETTE[k] for k in keys]

    _apply_rc()

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


def plot_text_modification():
    """Visualize text-modification results: U2 transitions and synergy rate lift."""
    levels = ["None", "Low", "Medium"]  # low reasoning -> high, left to right

    # U2 transitions (from base U2, n=813). R and U1 are 0 across all levels; omitted.
    n_u2 = 813
    stay_u2 = np.array([462, 494, 423])
    to_s    = np.array([161, 171, 170])
    to_err  = np.array([190, 148, 220])

    # Dataset-wide synergy rates (N=2500)
    N = 2500
    s_base_count = 119
    s_mod_counts = np.array([225, 248, 245])  # none, low, med
    s_base_rate = s_base_count / N * 100
    s_mod_rates = s_mod_counts / N * 100

    _apply_rc()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.6, 3.0),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.35},
    )

    # ---------- Panel A: stacked horizontal transitions ----------
    y = np.arange(len(levels))
    pct_u2  = stay_u2 / n_u2 * 100
    pct_s   = to_s    / n_u2 * 100
    pct_err = to_err  / n_u2 * 100

    bar_h = 0.55
    ax1.barh(y, pct_u2, bar_h, color=PALETTE["U2"],
             label="Remains U2", edgecolor="white", linewidth=0.8)
    ax1.barh(y, pct_s, bar_h, left=pct_u2, color=PALETTE["S"],
             label=r"$\rightarrow$ Synergistic", edgecolor="white", linewidth=0.8)
    ax1.barh(y, pct_err, bar_h, left=pct_u2 + pct_s, color=PALETTE["error"],
             label=r"$\rightarrow$ Error", edgecolor="white", linewidth=0.8)

    for i in range(len(levels)):
        ax1.text(pct_u2[i] / 2, y[i], f"{pct_u2[i]:.1f}%",
                 ha="center", va="center", fontsize=8.5, color="white")
        ax1.text(pct_u2[i] + pct_s[i] / 2, y[i], f"{pct_s[i]:.1f}%",
                 ha="center", va="center", fontsize=8.5, color="white")
        ax1.text(pct_u2[i] + pct_s[i] + pct_err[i] / 2, y[i],
                 f"{pct_err[i]:.1f}%",
                 ha="center", va="center", fontsize=8.5, color="#3A3A3A")

    ax1.set_yticks(y)
    ax1.set_yticklabels(levels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of base U2 samples (%)")
    ax1.set_ylabel("Reasoning effort")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(length=3)
    ax1.set_title(f"(a) Outcomes after text modification  (n = {n_u2})",
                  fontsize=10, pad=8, loc="left")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
               ncol=3, frameon=False, fontsize=9, handlelength=1.2,
               handleheight=1.1, columnspacing=1.4)

    # ---------- Panel B: base vs. modified synergy rate ----------
    x = np.arange(len(levels))
    width = 0.36

    ax2.bar(x - width / 2, [s_base_rate] * 3, width,
            color=PALETTE["base"], label="Base",
            edgecolor="white", linewidth=0.8)
    ax2.bar(x + width / 2, s_mod_rates, width,
            color=PALETTE["S"], label="After modification",
            edgecolor="white", linewidth=0.8)

    ymax = float(s_mod_rates.max()) + 3.2
    for i, m in enumerate(s_mod_rates):
        ax2.text(x[i] - width / 2, s_base_rate + 0.15, f"{s_base_rate:.1f}%",
                 ha="center", va="bottom", fontsize=8.5, color="#3A3A3A")
        ax2.text(x[i] + width / 2, m + 0.15, f"{m:.1f}%",
                 ha="center", va="bottom", fontsize=8.5, color="#3A3A3A")
        ax2.annotate(
            f"+{m - s_base_rate:.1f} pp",
            xy=(x[i], max(s_base_rate, m) + 1.0),
            xytext=(x[i], max(s_base_rate, m) + 2.4),
            ha="center", fontsize=8.5, color="#222",
            arrowprops=dict(arrowstyle="-[,widthB=1.3,lengthB=0.3",
                            color="#888", lw=0.6),
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(levels)
    ax2.set_xlabel("Reasoning effort")
    ax2.set_ylabel("Synergistic rate (%)")
    ax2.set_ylim(0, ymax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(length=3)
    ax2.set_title("(b) Dataset-wide synergy rate",
                  fontsize=10, pad=8, loc="left")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
               ncol=2, frameon=False, fontsize=9,
               handlelength=1.2, handleheight=1.1, columnspacing=1.4)

    fig.savefig(TEXTMOD_OUTPUT_PATH, bbox_inches="tight",
                pad_inches=0.03, dpi=300)
    fig.savefig(TEXTMOD_OUTPUT_PATH.with_suffix(".png"),
                bbox_inches="tight", pad_inches=0.03, dpi=300)
    print(f"Saved {TEXTMOD_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
    plot_text_modification()
