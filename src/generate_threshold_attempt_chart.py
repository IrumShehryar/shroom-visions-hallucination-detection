"""
Paper-appendix figure + summary table for the post-processing filter
investigation (confidence threshold / hedge-language / negation-sentence),
documented in project memory as attempt_confidence_threshold_filter.

All numbers below are copied from the actual scorer.py runs performed
during that investigation (src/confidence_threshold_sweep.py against
scoring_run/ for the dev set, and src/run_fresh_validation.py against
fresh_validation_raw_cache.json for the held-out set). This script does not
recompute anything -- it only renders the already-established results.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = r"D:\SHROOM\paper_charts"

# --- palette (validated categorical slots: blue=slot1, red=slot8) ---
COLOR_DEV = "#2a78d6"
COLOR_HELDOUT = "#e34948"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

GROUPS = [
    "Baseline\n(no filter)",
    "Confidence threshold\n(mischar ≥ 0.92)",
    "Hedge-language\nsuppression",
    "Negation\n(sentence-level)",
]

DEV_N = 286
HELDOUT_N = 39

DATA = {
    "Cor": {
        "dev": [0.4313, 0.4636, 0.4556, 0.4325],
        "heldout": [0.3252, 0.2757, 0.3026, 0.3048],
    },
    "Cor+Lbl": {
        "dev": [0.3796, 0.4213, 0.4041, 0.3866],
        "heldout": [0.2684, 0.2316, 0.2561, 0.2616],
    },
    "IoU": {
        "dev": [0.4294, 0.4671, 0.4508, 0.4354],
        "heldout": [0.2707, 0.2418, 0.2598, 0.2582],
    },
}


def render():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=SURFACE)
    x = np.arange(len(GROUPS))
    width = 0.34

    for ax, metric in zip(axes, ["Cor", "Cor+Lbl", "IoU"]):
        dev_vals = DATA[metric]["dev"]
        heldout_vals = DATA[metric]["heldout"]

        ax.set_facecolor(SURFACE)
        ax.bar(x - width / 2, dev_vals, width, label=f"Dev (n={DEV_N})",
               color=COLOR_DEV, zorder=3)
        ax.bar(x + width / 2, heldout_vals, width, label=f"Held-out (n={HELDOUT_N})",
               color=COLOR_HELDOUT, zorder=3)

        for xi, v in zip(x - width / 2, dev_vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                     fontsize=8, color=INK_SECONDARY)
        for xi, v in zip(x + width / 2, heldout_vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                     fontsize=8, color=INK_SECONDARY)

        # baseline reference lines (dashed) per series, for at-a-glance delta reading
        ax.axhline(dev_vals[0], color=COLOR_DEV, linewidth=0.8, linestyle=(0, (3, 3)),
                    alpha=0.5, zorder=2)
        ax.axhline(heldout_vals[0], color=COLOR_HELDOUT, linewidth=0.8, linestyle=(0, (3, 3)),
                    alpha=0.5, zorder=2)

        ax.set_title(metric, fontsize=12, color=INK_PRIMARY, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUPS, fontsize=8, color=INK_SECONDARY)
        ax.set_ylim(0, max(max(dev_vals), max(heldout_vals)) * 1.22)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE_AXIS)
        ax.tick_params(axis="y", labelsize=8, colors=INK_MUTED, length=0)
        ax.tick_params(axis="x", length=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        "Every candidate filter improves the dev set and reverses on held-out data",
        fontsize=11, color=INK_PRIMARY, y=1.14, fontweight="normal",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, "threshold_filter_dev_vs_heldout.png")
    pdf_path = os.path.join(OUTPUT_DIR, "threshold_filter_dev_vs_heldout.pdf")
    fig.savefig(png_path, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(pdf_path, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    render()
