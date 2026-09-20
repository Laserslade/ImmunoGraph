"""
Builds Figure C3 from Gate 3 results and trajectories.
Run gate3_recovery.py first.
"""

import sys
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
sys.path.insert(0, str(ROOT / "reference"))
import chamberland_reference as ref

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def run():
    gate3 = list(csv.DictReader(open(RESULTS / "chamberland_gate3_results.csv")))
    traj = np.load(RESULTS / "chamberland_gate3_trajectories.npz", allow_pickle=True)
    t = traj["t"] / 365
    names = list(traj["state_names"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    y = traj["y_0_0"]
    idx_N, idx_A, idx_Mpro = [names.index(n) for n in ["N", "A", "M_pro"]]
    ax2 = ax.twinx()
    ax.plot(t, y[idx_N] / y[idx_N, 0], color="#2c5f8a", label="N normalized")
    ax2.plot(t, y[idx_A] / max(y[idx_A].max(), 1e-30), color="#c0392b", label="A normalized", linestyle="--")
    ax2.plot(t, y[idx_Mpro] / max(y[idx_Mpro].max(), 1e-30), color="#e67e22", label="M_pro normalized", linestyle=":")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("N over N at age 30")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=7, loc="center left")
    ax.set_title("A. Trajectory direction behavior")

    ax = axes[1]
    idx_ABpo = names.index("ABpo")
    vals = {
        "Woman\nAPOE4-": traj["y_0_0"][idx_ABpo, -1], "Woman\nAPOE4+": traj["y_0_1"][idx_ABpo, -1],
        "Man\nAPOE4-": traj["y_1_0"][idx_ABpo, -1], "Man\nAPOE4+": traj["y_1_1"][idx_ABpo, -1],
    }
    ax.bar(vals.keys(), vals.values(), color=["#2c5f8a", "#c0392b", "#2c5f8a", "#c0392b"])
    ax.set_yscale("log")
    ax.set_ylabel("ABpo at age 80")
    ax.set_title("B. APOE4 effect on amyloid plaque")

    ax = axes[2]
    for label, key, color in [("Woman APOE4-", "y_0_0", "#2c5f8a"), ("Woman APOE4+", "y_0_1", "#c0392b"),
                                ("Man APOE4-", "y_1_0", "#4a8a5f"), ("Man APOE4+", "y_1_1", "#e67e22")]:
        y = traj[key]
        ax.plot(t, 100 * (1 - y[idx_N] / y[idx_N, 0]), label=label, color=color)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("cumulative neuronal loss percent")
    ax.set_title("C. Age dependent neuronal loss")
    ax.legend(fontsize=7)

    total_pass = sum(1 for r in gate3 if r["pass"] == "True")
    plt.suptitle(f"Figure C3, Chamberland published behavior reproduction, {total_pass}/{len(gate3)} instances pass",
                 fontsize=12, y=1.04)
    plt.tight_layout()
    plt.savefig(FIGURES / "chamberland_gate3_published_behavior.png", bbox_inches="tight")
    print("saved chamberland_gate3_published_behavior.png")


if __name__ == "__main__":
    run()
