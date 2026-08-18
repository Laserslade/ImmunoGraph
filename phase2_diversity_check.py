"""
phase2_diversity_check.py

A small standalone check for Phase 2: overlay a handful of saved runs for
two key variables (TNF-alpha and neuron density) to visually confirm the
perturbation harness produces meaningfully different trajectories, not
near-identical ones. This is a sanity check, not a required pipeline
component -- it can be deleted once you've eyeballed it once.

Run:
    python phase2_diversity_check.py
"""

import glob
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DIRS  # noqa: E402
from phase2_perturbation_harness import load_run  # noqa: E402


def main():
    run_paths = sorted(glob.glob(os.path.join(DIRS["sims"], "run_*.npz")))[:8]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for path in run_paths:
        t, y, state_names, overrides = load_run(path)
        ta_idx = state_names.index("Ta")
        n_idx = state_names.index("N")
        label = f"AP={int(overrides['AP'])}"
        axes[0].plot(t, y[ta_idx], alpha=0.7, label=label)
        axes[1].plot(t, y[n_idx], alpha=0.7, label=label)

    axes[0].set_title("TNF-alpha across 8 perturbed runs")
    axes[0].set_xlabel("age (years)")
    axes[1].set_title("Neuron density across 8 perturbed runs")
    axes[1].set_xlabel("age (years)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    save_path = os.path.join(DIRS["results"], "phase2_diversity_check.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved diversity check plot to {save_path}")


if __name__ == "__main__":
    main()
