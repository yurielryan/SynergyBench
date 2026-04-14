import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import classify

INPUT_PATH = Path("results/base_dataset.json")
OUTPUT_PATH = Path("results/interaction_donut.pdf")
GRID_OUTPUT_PATH = Path("results/interaction_donut_grid.pdf")
TEXTMOD_OUTPUT_PATH = Path("results/text_modification.pdf")
COMBINED_OUTPUT_PATH = Path("results/figure6_combined.pdf")

EVAL_PATHS = [
    ("None",   Path("results/gpt-5.4_none_eval.json")),
    ("Low",    Path("results/gpt-5.4_low_eval.json")),
    ("Medium", Path("results/gpt-5.4_med_eval.json")),
]

LABELS = ["Redundant", "Unique (X1)", "Unique (X2)", "Synergistic", "Error"]
KEYS   = ["R", "U1", "U2", "S", "error"]

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


def _load_results(path):
    with open(path) as f:
        return json.load(f)["results"]


def compute_counts(path=INPUT_PATH):
    results = _load_results(path)
    counts = {k: 0 for k in KEYS}
    for sample in results.values():
        counts[classify(sample)] += 1
    return counts


def compute_counts_merged(eval_path, base_path=INPUT_PATH):
    """For samples classified as U2 in base, substitute the eval sample; else keep base."""
    base = _load_results(base_path)
    evl = _load_results(eval_path)
    counts = {k: 0 for k in KEYS}
    for sid, base_sample in base.items():
        sample = evl[sid] if classify(base_sample) == "U2" and sid in evl else base_sample
        counts[classify(sample)] += 1
    return counts


def _draw_donut(ax, counts, center_label=None, outside_labels=True):
    values = [counts[k] for k in KEYS]
    total = sum(values)
    colors = [PALETTE[k] for k in KEYS]
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.34, edgecolor="white", linewidth=1.0),
    )
    ax.text(0, 0.10, f"{total}", ha="center", va="center",
            fontsize=13, color="#2B2B2B")
    ax.text(0, -0.13, center_label or "samples", ha="center", va="center",
            fontsize=8, color="#707070", style="italic")

    if outside_labels:
        for w, key, v in zip(wedges, KEYS, values):
            if key == "error" or v == 0:
                continue
            pct = v / total * 100
            if pct < 0.8:  # skip to avoid crowding; label would be ambiguous anyway
                continue
            # Bias R toward the lower end of its wedge so it never collides with
            # labels above (U2/S sit on the upper-left in this dataset).
            frac = 0.15 if key == "R" else 0.5
            ang_deg = w.theta1 + frac * (w.theta2 - w.theta1)
            ang = math.radians(ang_deg)
            x, y = 1.14 * math.cos(ang), 1.14 * math.sin(ang)
            ha = "left" if x >= 0 else "right"
            ax.text(x, y, f"{pct:.1f}%", ha=ha, va="center",
                    fontsize=8.5, color="#2B2B2B")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.25, 1.25)
    return wedges, values, total


def main():
    _apply_rc()

    # ---- Standalone donut for the base dataset ----
    counts = compute_counts(INPUT_PATH)
    values = [counts[k] for k in KEYS]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(5.2, 2.8), subplot_kw=dict(aspect="equal"))
    wedges, _, _ = _draw_donut(ax, counts)

    legend_labels = [
        f"{lab}   {v/total*100:4.1f}%   n = {v}"
        for lab, v in zip(LABELS, values)
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

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.02, dpi=1200)
    fig.savefig(OUTPUT_PATH.with_suffix(".png"),
                bbox_inches="tight", pad_inches=0.02, dpi=1200)
    print(f"Saved {OUTPUT_PATH}")

    # ---- Comparison grid: Base + 3 reasoning levels (U2 -> eval, else base) ----
    fig = plt.figure(figsize=(10.4, 3.0))
    _render_donut_grid(fig, base_counts=counts)
    fig.savefig(GRID_OUTPUT_PATH, bbox_inches="tight", pad_inches=0.04, dpi=300)
    fig.savefig(GRID_OUTPUT_PATH.with_suffix(".png"),
                bbox_inches="tight", pad_inches=0.04, dpi=300)
    print(f"Saved {GRID_OUTPUT_PATH}")


def _render_donut_grid(parent, base_counts=None, axes=None, legend_anchor=(0.5, -0.02)):
    """Render the Base + 3-reasoning-level donut grid.

    `parent` is a Figure or SubFigure (used for the shared legend). If `axes`
    is provided, it must be an array of 4 axes (already created, e.g. from a
    shared GridSpec) — otherwise the parent's own subplots are created.
    """
    if base_counts is None:
        base_counts = compute_counts(INPUT_PATH)
    panels = [("Base", base_counts)] + [
        (name, compute_counts_merged(path)) for name, path in EVAL_PATHS
    ]
    if axes is None:
        axes = parent.subplots(
            1, len(panels),
            subplot_kw=dict(aspect="equal"),
            gridspec_kw={"wspace": 0.15},
        )
    else:
        for ax_i in axes:
            ax_i.set_aspect("equal")

    grid_wedges = None
    for ax_i, (name, c) in zip(axes, panels):
        w, _, _ = _draw_donut(ax_i, c)
        grid_wedges = w
        subtitle = "base dataset" if name == "Base" else f"U2 re-eval @ {name.lower()}"
        ax_i.set_title(f"{name}\n({subtitle})", fontsize=10, pad=6)

    legend_labels_simple = [
        f"{lab}  ({k})" if k != "error" else lab
        for lab, k in zip(LABELS, KEYS)
    ]
    parent.legend(
        grid_wedges,
        legend_labels_simple,
        loc="lower center",
        bbox_to_anchor=legend_anchor,
        ncol=len(LABELS),
        frameon=False,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.6,
        fontsize=9.5,
    )


def plot_text_modification():
    """Visualize text-modification results: U2 transitions and synergy rate lift."""
    _apply_rc()
    fig = plt.figure(figsize=(9.6, 3.0))
    axes = fig.subplots(
        1, 2,
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.35},
    )
    _render_text_modification(axes[0], axes[1])

    fig.savefig(TEXTMOD_OUTPUT_PATH, bbox_inches="tight",
                pad_inches=0.03, dpi=300)
    fig.savefig(TEXTMOD_OUTPUT_PATH.with_suffix(".png"),
                bbox_inches="tight", pad_inches=0.03, dpi=300)
    print(f"Saved {TEXTMOD_OUTPUT_PATH}")


def _render_text_modification(ax1, ax2):
    levels = ["None", "Low", "Medium"]  # low reasoning -> high, left to right

    # U2 transitions (from base U2, n=813). R and U1 are 0 across all levels; omitted.
    n_u2 = 813
    stay_u2 = np.array([462, 494, 423])
    to_s    = np.array([161, 171, 170])
    to_err  = np.array([190, 148, 220])

    # Dataset-wide synergy rates, computed from merged data so they match the donut grid:
    # for base-U2 samples, substitute the eval sample; otherwise keep the base sample.
    base_counts = compute_counts(INPUT_PATH)
    N = sum(base_counts.values())
    s_base_count = base_counts["S"]
    merged_by_level = {name: compute_counts_merged(path) for name, path in EVAL_PATHS}
    s_mod_counts = np.array([merged_by_level[name]["S"] for name in levels])
    s_base_rate = s_base_count / N * 100
    s_mod_rates = s_mod_counts / N * 100

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
    ax1.set_title(f"6(a) Outcomes after text modification  (n = {n_u2})",
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
    ax2.set_ylabel("Synergy (%)")
    ax2.set_ylim(0, ymax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(length=3)
    ax2.set_title("6(b) Dataset-wide Synergy",
                  fontsize=10, pad=8, loc="left")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
               ncol=2, frameon=False, fontsize=9,
               handlelength=1.2, handleheight=1.1, columnspacing=1.4)


def plot_combined():
    """Stack the donut grid (top) and text-modification panels (bottom)
    inside a single GridSpec so both rows share identical left/right margins."""
    _apply_rc()
    fig = plt.figure(figsize=(10.4, 7.0))

    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 1.05],
        hspace=0.25,
        left=0.07, right=0.985, top=0.95, bottom=0.16,
    )
    top_gs = outer[0].subgridspec(1, 4, wspace=0.15)
    bot_gs = outer[1].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.30)

    top_axes = [fig.add_subplot(top_gs[0, i]) for i in range(4)]
    bot_axes = [fig.add_subplot(bot_gs[0, i]) for i in range(2)]

    # Use a per-row legend anchor expressed in *figure* coords, aligned to the
    # top row's vertical span so it sits just below the donuts.
    top_bbox = top_gs.get_grid_positions(fig)  # (bottoms, tops, lefts, rights)
    top_bottom = float(top_bbox[0][0])
    top_left = float(top_bbox[2][0])
    top_right = float(top_bbox[3][-1])
    legend_x = 0.5 * (top_left + top_right)
    legend_y = top_bottom - 0.02

    _render_donut_grid(
        fig, axes=top_axes,
        legend_anchor=(legend_x, legend_y),
    )
    # Move to figure-coord legend by re-anchoring after creation
    fig.legends[-1].set_bbox_to_anchor((legend_x, legend_y),
                                       transform=fig.transFigure)

    _render_text_modification(bot_axes[0], bot_axes[1])

    fig.savefig(COMBINED_OUTPUT_PATH,
                bbox_inches="tight", pad_inches=0.04)  # vector PDF
    fig.savefig(COMBINED_OUTPUT_PATH.with_suffix(".svg"),
                bbox_inches="tight", pad_inches=0.04)  # vector SVG
    fig.savefig(COMBINED_OUTPUT_PATH.with_suffix(".png"),
                bbox_inches="tight", pad_inches=0.04, dpi=1200)
    print(f"Saved {COMBINED_OUTPUT_PATH}")


if __name__ == "__main__":
    # main()
    # plot_text_modification()
    plot_combined()
