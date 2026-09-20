"""
Builds Figure C2 from Gate 2 results, trajectories, and the self
perturbation control. Run gate2_recovery.py and gate2_self_perturbation.py first.
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
    gate2 = list(csv.DictReader(open(RESULTS / "chamberland_gate2_results.csv")))
    traj = np.load(RESULTS / "chamberland_gate2_trajectories.npz", allow_pickle=True)
    selfpert = list(csv.DictReader(open(RESULTS / "chamberland_gate2_self_perturbation_control.csv")))

    demographics = ["sex=0_AP=0", "sex=0_AP=1", "sex=1_AP=0", "sex=1_AP=1"]
    demo_labels = ["Woman APOE4-", "Woman APOE4+", "Man APOE4-", "Man APOE4+ exception"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    ax = axes[0, 0]
    colors = ["#2c5f8a", "#4a8a5f", "#8a6a2c", "#c0392b"]
    rng = np.random.default_rng(0)
    for di, demo in enumerate(demographics):
        rows = [r for r in gate2 if r["demographic"] == demo]
        nrmse = [float(r["nrmse"]) for r in rows]
        x = np.full(len(nrmse), di) + rng.uniform(-0.1, 0.1, len(nrmse))
        ax.scatter(x, nrmse, alpha=0.7, color=colors[di], s=25)
    ax.set_yscale("log")
    ax.set_xticks(range(4))
    ax.set_xticklabels(demo_labels, rotation=20, ha="right")
    ax.axhline(1e-4, color="gray", linestyle=":", lw=1, label="tolerance")
    ax.set_ylabel("nRMSE per state")
    ax.set_title("A. Reproduction error by demographic condition")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    demo = "sex=0_AP=0"
    t = traj[f"t_{demo}"]
    Ya = traj[f"authors_{demo}"]
    Yo = traj[f"ours_{demo}"]
    idx_N = ref.STATE_NAMES.index("N")
    ax.plot(t / 365, Ya[idx_N], label="reference implementation", color="black", lw=2)
    ax.plot(t / 365, Yo[idx_N], label="clean room reconstruction", color="#e67e22", lw=1, linestyle="--")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("N, neuron density")
    ax.set_title("B. Representative trajectory overlay, machine precision case")
    ax.legend()

    ax = axes[1, 0]
    demo = "sex=1_AP=1"
    t = traj[f"t_{demo}"]
    Ya = traj[f"authors_{demo}"]
    Yo = traj[f"ours_{demo}"]
    idx_ABpo = ref.STATE_NAMES.index("ABpo")
    ax2 = ax.twinx()
    ax.plot(t / 365, Ya[idx_ABpo], label="reference", color="black", lw=1.5)
    ax.plot(t / 365, Yo[idx_ABpo], label="clean room", color="#c0392b", lw=1, linestyle="--")
    diff = np.abs(Ya[idx_ABpo] - Yo[idx_ABpo])
    ax2.plot(t / 365, diff, color="gray", alpha=0.5, lw=0.8, label="absolute difference")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("ABpo")
    ax2.set_ylabel("absolute difference", color="gray")
    ax.set_title("C. APOE4+ male discrepancy, near equilibrium state")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=7, loc="upper left")

    ax = axes[1, 1]
    sp_names = [r["state_variable"] for r in selfpert]
    sp_vals = [float(r["max_rel_diff_self_perturbation"]) for r in selfpert]
    order = np.argsort(sp_vals)
    ax.barh([sp_names[i] for i in order], [sp_vals[i] for i in order], color="#7a5a9a")
    ax.set_xscale("log")
    ax.set_xlabel("max relative difference, reference code vs itself")
    ax.set_title("D. Original code perturbation experiment")

    plt.suptitle("Figure C2, Chamberland numerical trajectory reproduction", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / "chamberland_gate2_numerical_reproduction.png", bbox_inches="tight")
    print("saved chamberland_gate2_numerical_reproduction.png")


if __name__ == "__main__":
    run()
