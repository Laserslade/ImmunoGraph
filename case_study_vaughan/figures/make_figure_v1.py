"""
Builds Figure V1 from the sampling protocol table and cluster marginal
extraction. No recovery run needed, existing files only.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "reference"))
import vaughan_ensemble_harness as h

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def run():
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.2])

    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    cat_data = [("published quantitative\nmarginal support", 9, 45, "#2c7a3a"),
                ("definition only\nno numeric support", 36, 45, "#8a8a8a"),
                ("initial conditions\npublished support", 0, 7, "#c0392b")]
    x = 0
    for label, num, denom, color in cat_data:
        ax.barh(0, num / denom, left=x, height=0.5, color=color)
        ax.text(x + num / denom / 2, 0, f"{num}/{denom}", ha="center", va="center", color="white", fontweight="bold")
        ax.text(x, -0.55, label, fontsize=8, va="top")
        x += 1.15
    ax.set_xlim(-0.1, x)
    ax.set_ylim(-1.2, 0.6)
    ax.set_title("A. Provenance composition, 45 parameters plus 7 fitted initial conditions")

    ax = fig.add_subplot(gs[1])
    params9 = list(h.CLUSTER_MARGINALS.keys())
    cluster_colors = {"1": "#2c5f8a", "2A": "#e67e22", "2B": "#c0392b"}
    y = 0
    yticks, yticklabels = [], []
    for p in params9:
        clusters = h.CLUSTER_MARGINALS[p]
        for cname in ["1", "2A", "2B"]:
            if cname in clusters:
                lo, hi, avg = clusters[cname]
                ax.plot([lo, hi], [y, y], color=cluster_colors[cname], lw=6, alpha=0.7, solid_capstyle="butt")
                ax.scatter([avg], [y], color="black", s=15, zorder=5)
            else:
                ax.text(0, y, "  unavailable, not imputed", fontsize=7, color="gray", va="center",
                        transform=ax.get_yaxis_transform())
            yticks.append(y)
            yticklabels.append(f"{p} C{cname}")
            y -= 1
        y -= 0.5

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7)
    ax.set_xscale("symlog")
    ax.set_xlabel("parameter value, symlog scale, dot is published cluster average")
    legend_elems = [Line2D([0], [0], color=cluster_colors[c], lw=6, label=f"cluster {c}") for c in ["1", "2A", "2B"]]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right")
    ax.set_title("B. Published cluster specific ranges for the nine partial range parameters")

    plt.suptitle("Figure V1, Vaughan parameter provenance", fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(FIGURES / "vaughan_parameter_provenance.png", bbox_inches="tight")
    print("saved vaughan_parameter_provenance.png")


if __name__ == "__main__":
    run()
