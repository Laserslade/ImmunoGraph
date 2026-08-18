"""
phase1_base_model.py

Phase 1 of the Neuroimmune Graph Project.

Solves the Chamberland backbone model (models/chamberland_backbone.py)
over a simulated lifespan (age 30 to 80, matching the source paper's
framing) using default parameters and a healthy age-30 initial state,
then plots every state variable so the trajectories can be sanity-checked
against the qualitative behavior described in the source paper:

  - Anti-inflammatory dominance early in life, shifting toward
    pro-inflammatory dominance with age.
  - Astrocyte activation and microglial activation increasing with age.
  - Neuron density slowly declining.

This script has no dependency on any later phase and can be run on its
own once Phase 0 (setup_environment.py) has been run.

Run:
    python phase1_base_model.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DIRS  # noqa: E402
from models.chamberland_backbone import (  # noqa: E402
    STATE_NAMES,
    chamberland_rhs,
    default_initial_state,
)

AGE_START = 30.0
AGE_END = 80.0
N_EVAL_POINTS = 500


def run_simulation(params=None):
    """Solve the backbone model over the default simulated lifespan."""
    y0 = default_initial_state()
    t_eval = np.linspace(AGE_START, AGE_END, N_EVAL_POINTS)

    solution = solve_ivp(
        fun=lambda t, y: chamberland_rhs(t, y, params),
        t_span=(AGE_START, AGE_END),
        y0=y0,
        method="BDF",       # stiff solver, matches the source paper's choice
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    return solution


def plot_trajectories(solution, save_path):
    """Plot every state variable over simulated age, save to save_path."""
    n_states = len(STATE_NAMES)
    n_cols = 4
    n_rows = int(np.ceil(n_states / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, name in enumerate(STATE_NAMES):
        ax = axes[i]
        ax.plot(solution.t, solution.y[i], linewidth=1.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("age (years)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    # Hide any unused subplot axes
    for j in range(n_states, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Phase 1: Chamberland backbone trajectories (default parameters, "
        "placeholder calibration)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150)
    print(f"Saved trajectory plot to {save_path}")


def qualitative_sanity_check(solution):
    """
    Print a short qualitative summary comparing early-life vs late-life
    values for the key inflammatory variables, as an eyeball check against
    the source paper's described qualitative trends (not a statistical
    validation -- just Phase 1's "does this look right" pass).
    """
    idx = {name: i for i, name in enumerate(STATE_NAMES)}
    y = solution.y

    def start_end(name):
        i = idx[name]
        return y[i, 0], y[i, -1]

    print("\nQualitative sanity check (value at age 30 -> value at age 80):")
    for name in ["N", "A", "M_pro", "M_anti", "Ta", "Tb", "I10"]:
        start, end = start_end(name)
        trend = "up" if end > start * 1.05 else ("down" if end < start * 0.95 else "flat")
        print(f"  {name:8s}: {start:8.4f} -> {end:8.4f}   ({trend})")

    print(
        "\nExpected pattern (per source paper, qualitative only): N should "
        "decline slowly; A, M_pro, and Ta should rise with age; M_anti and "
        "Tb typically decline in relative dominance as M_pro/Ta rise."
    )


def main():
    print(f"Solving Chamberland backbone from age {AGE_START} to {AGE_END}...")
    solution = run_simulation()

    save_path = os.path.join(DIRS["results"], "phase1_trajectories.png")
    plot_trajectories(solution, save_path)

    qualitative_sanity_check(solution)

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
