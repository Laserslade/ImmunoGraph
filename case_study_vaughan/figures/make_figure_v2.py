"""
Builds Figure V2 from the frozen ensemble manifest and raw sample data.
No recovery run needed, existing frozen files only.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
ENSEMBLE_DIR = ROOT / "frozen_ensemble"
sys.path.insert(0, str(ROOT / "reference"))
import vaughan_reference as vr
import vaughan_ensemble_harness as h

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def run():
    manifest = json.load(open(ENSEMBLE_DIR / "vaughan_exploratory_ensemble_v1_manifest.json"))
    data = np.load(ENSEMBLE_DIR / "vaughan_exploratory_ensemble_v1_raw.npz", allow_pickle=True)
    param_names = list(data["param_names"])
    param_values = data["param_values"]
    acceptance = data["acceptance_status"]
    initial_states = data["initial_states"]

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.text(0.5, 0.9, "Vaughan exploratory ensemble v1", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.78, f"seed={manifest['master_seed']}  n requested={manifest['n_samples']}", ha="center", fontsize=9)
    boxes = [("9 published support\nparameters", "cluster marginal mixture", "#2c7a3a"),
             ("36 definition only\nparameters", "log uniform 0.1x to 10x fixture", "#8a8a8a"),
             ("7 fitted ICs\nunpublished", "log uniform plus D0 mixture", "#c0392b")]
    y = 0.55
    for title, rule, color in boxes:
        ax.add_patch(plt.Rectangle((0.05, y - 0.13), 0.9, 0.16, color=color, alpha=0.15, ec=color))
        ax.text(0.5, y - 0.02, title, ha="center", fontsize=9, fontweight="bold")
        ax.text(0.5, y - 0.11, rule, ha="center", fontsize=7.5)
        y -= 0.24
    ax.set_title("A. Sampling architecture")

    ax = fig.add_subplot(gs[0, 1])
    params9 = list(h.CLUSTER_MARGINALS.keys())
    for i, p in enumerate(params9):
        pidx = param_names.index(p)
        vals = param_values[:, pidx]
        accepted_mask = acceptance == "accepted"
        rng = np.random.default_rng(0)
        ax.scatter(vals[accepted_mask], np.full(accepted_mask.sum(), i) + rng.uniform(-0.15, 0.15, accepted_mask.sum()),
                   s=3, alpha=0.3, color="black")
        for cname, color in [("1", "#2c5f8a"), ("2A", "#e67e22"), ("2B", "#c0392b")]:
            if cname in h.CLUSTER_MARGINALS[p]:
                lo, hi, _ = h.CLUSTER_MARGINALS[p][cname]
                ax.plot([lo, hi], [i, i], color=color, lw=4, alpha=0.6, zorder=0)
    ax.set_yticks(range(len(params9)))
    ax.set_yticklabels(params9, fontsize=8)
    ax.set_xscale("symlog")
    ax.set_title("B. Frozen samples versus permitted published supports")

    ax = fig.add_subplot(gs[1, 0])
    n_acc, n_rej = manifest["n_accepted"], manifest["n_rejected"]
    bars = ax.bar(["accepted", "rejected"], [n_acc, n_rej], color=["#2c7a3a", "#c0392b"])
    ax.bar_label(bars)
    ax.set_title(f"C. Admissibility, {n_acc}/{manifest['n_samples']} accepted, not resampled")

    ax = fig.add_subplot(gs[1, 1])
    d_idx = vr.STATE_NAMES.index("D")
    d0_vals = initial_states[:, d_idx]
    accepted_mask = acceptance == "accepted"
    frac_zero = np.mean(d0_vals[accepted_mask] == 0)
    ax.bar(["D0 = 0", "D0 > 0"], [frac_zero, 1 - frac_zero], color=["#8a8a8a", "#2c5f8a"])
    ax.set_title(f"D. D0 mixture in frozen ensemble, D0=0: {frac_zero * 100:.1f} pct")
    for i, v in enumerate([frac_zero, 1 - frac_zero]):
        ax.text(i, v + 0.01, f"{v * 100:.1f} pct", ha="center")

    plt.suptitle("Figure V2, frozen exploratory ensemble", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / "vaughan_frozen_ensemble.png", bbox_inches="tight")
    print("saved vaughan_frozen_ensemble.png")


if __name__ == "__main__":
    run()
