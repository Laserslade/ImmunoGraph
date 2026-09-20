"""
Builds Figure V4 from the directional hub dynamics export.
Run directional_hub_dynamics.py first.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
NAMES = ["M1", "M2", "IL1", "IL12", "IL10", "IL4", "D"]


def run():
    rows = list(csv.DictReader(open(RESULTS / "vaughan_directional_hub_dynamics.csv")))
    times = sorted(set(float(r["snapshot_h"]) for r in rows))

    def series(node, metric):
        return [float(r["top1_frequency_pct"]) for t in times for r in rows
                if r["node"] == node and r["metric"] == metric and float(r["snapshot_h"]) == t]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(times, series("M2", "weighted_in"), "o-", color="#2c5f8a", label="M2 weighted in")
    ax.plot(times, series("M2", "topology_in"), "s--", color="#5dade2", label="M2 topology in")
    ax.set_xlabel("time h")
    ax.set_ylabel("top1 frequency pct")
    ax.set_title("A. M2, persistent receiver")
    ax.legend()
    ax.set_ylim(0, 105)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(times, series("IL12", "topology_out"), "o-", color="#c0392b", label="IL12 topology out")
    ax.plot(times, series("M2", "topology_out"), "s-", color="#2c5f8a", label="M2 topology out")
    ax.set_xlabel("time h")
    ax.set_ylabel("top1 frequency pct")
    ax.set_title("B. IL12 early broadcaster, declining reach")
    ax.legend()
    ax.set_ylim(0, 105)

    ax = fig.add_subplot(gs[1, 0])
    heat = np.array([series(nm, "weighted_in") for nm in NAMES])
    im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels([f"{int(t)}h" for t in times])
    ax.set_yticks(range(len(NAMES)))
    ax.set_yticklabels(NAMES)
    plt.colorbar(im, ax=ax, label="weighted in top1 pct")
    ax.set_title("C. All node weighted in top1 frequency")

    ax = fig.add_subplot(gs[1, 1])
    heat2 = np.array([series(nm, "topology_out") for nm in NAMES])
    im2 = ax.imshow(heat2, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels([f"{int(t)}h" for t in times])
    ax.set_yticks(range(len(NAMES)))
    ax.set_yticklabels(NAMES)
    plt.colorbar(im2, ax=ax, label="topology out top1 pct")
    ax.set_title("D. All node topology out top1 frequency")

    plt.suptitle("Figure V4, receiver and broadcaster reorganization", fontsize=11, y=1.04)
    plt.tight_layout()
    plt.savefig(FIGURES / "vaughan_directional_hub_dynamics.png", bbox_inches="tight")
    print("saved vaughan_directional_hub_dynamics.png")


if __name__ == "__main__":
    run()
