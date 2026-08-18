"""
phase4_local_sensitivity.py

Phase 4 of the Neuroimmune Graph Project.

Computes LOCAL SENSITIVITY along a single saved trajectory of the
composite model (Phase 3). This is the mathematical core of the entire
graph-representation framework -- see project findings log, Section 3.5
-- and is deliberately kept standalone here: no graph object is built
yet (that's Phase 5). This script only has to prove the sensitivity
math itself is sound.

-------------------------------------------------------------------------
WHAT "LOCAL SENSITIVITY" MEANS HERE (and why it's not a graph edge yet)
-------------------------------------------------------------------------
For a state vector y(t) and the model's right-hand-side function
dy/dt = f(y, t), the local sensitivity matrix is the Jacobian:

    J_ij(t) = d(dy_i/dt) / d(y_j)      evaluated at (t, y(t))

J_ij answers: "at this moment, how much does a small change in variable
j's current value change variable i's rate of change?" That is exactly
the mechanistic, structure-respecting notion of "interaction strength"
decided on in Section 3.5 -- computed directly from the known equations,
NOT inferred from observed data the way Granger causality or transfer
entropy would (see Section 3.5 for why that distinction matters here).

Because the state variables have very different scales (e.g. neuron
density ~1.0 vs. IL-1beta ~0.0005), the raw Jacobian entries aren't
directly comparable to each other. So this script also computes the
normalized/elasticity form used throughout the sensitivity-analysis
literature we reviewed (Section 1, Vaughan et al. 2018's own sensitivity
formula S = (dx/dp)*(p/x)), adapted here from parameter-sensitivity to
state-interaction sensitivity:

    S_ij(t) = J_ij(t) * y_j(t) / (dy_i/dt)

S_ij is dimensionless: "a 1% change in variable j produces what percent
change in variable i's rate of change." This normalized matrix is what
Phase 5 will turn into a graph's edge weights.

Numerically, J is computed with central finite differences (not
autodiff) -- deliberately, since it requires no new dependencies and the
composite model's equations are smooth (Michaelis-Menten/Hill forms
throughout, confirmed in Section 3.8), so finite differences are well
behaved.

Run:
    python phase4_local_sensitivity.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DIRS  # noqa: E402
from models.composite_backbone import (  # noqa: E402
    STATE_NAMES,
    composite_rhs,
    default_params,
)
from phase3_composite_model import run_composite_simulation  # noqa: E402

FINITE_DIFF_EPS = 1e-6


def compute_jacobian(t, y, params, eps=FINITE_DIFF_EPS):
    """
    Central-finite-difference Jacobian of composite_rhs at (t, y).

    Returns an (n_states, n_states) array J where J[i, j] = d(dy_i/dt)/dy_j.
    """
    n = len(y)
    J = np.zeros((n, n))

    for j in range(n):
        step = eps * max(abs(y[j]), 1.0)  # scale step to variable's magnitude

        y_plus = y.copy()
        y_plus[j] += step
        y_minus = y.copy()
        y_minus[j] = max(y_minus[j] - step, 0.0)  # keep states non-negative

        actual_step = y_plus[j] - y_minus[j]
        if actual_step <= 0:
            continue  # variable pinned at zero, no meaningful derivative here

        dydt_plus = composite_rhs(t, y_plus, params)
        dydt_minus = composite_rhs(t, y_minus, params)

        J[:, j] = (dydt_plus - dydt_minus) / actual_step

    return J


def compute_dydt_scales(solution, params):
    """
    Compute a robust per-variable normalization scale: the 90th percentile
    of |dy_i/dt| across the ENTIRE trajectory, for each variable i.

    This replaces naive instantaneous normalization (S_ij = J_ij * y_j /
    dydt_i), which is numerically unstable whenever dydt_i passes through
    or near zero -- which happens at ordinary, harmless inflection points
    in a trajectory (a variable's rate of change crossing zero as it
    switches from rising to falling is normal behavior, not a problem).
    Normalizing by a robust trajectory-wide scale instead of the
    instantaneous value avoids this blow-up while still producing
    dimensionless, comparable-magnitude sensitivity values.
    """
    n_states, n_times = solution.y.shape
    all_dydt = np.zeros((n_states, n_times))

    for k in range(n_times):
        all_dydt[:, k] = composite_rhs(solution.t[k], solution.y[:, k], params)

    scales = np.percentile(np.abs(all_dydt), 90, axis=1)
    return scales


def normalize_sensitivity(J, y, scales, min_denominator=1e-8):
    """
    Convert a raw Jacobian into a dimensionless sensitivity form:
        S_ij = J_ij * y_j / scale_i

    where scale_i is a robust, trajectory-wide typical magnitude of
    dy_i/dt (see compute_dydt_scales), NOT the instantaneous dydt_i.
    This is the fix for the inflection-point blow-up described in the
    project findings log, Section 8.4.

    Entries where |y_j| or scale_i are too small to normalize safely are
    set to 0 rather than exploding.
    """
    n = len(y)
    S = np.zeros((n, n))

    for i in range(n):
        if scales[i] < min_denominator:
            continue
        for j in range(n):
            if abs(y[j]) < min_denominator:
                continue
            S[i, j] = J[i, j] * y[j] / scales[i]

    return S


def interpolate_state(solution, age):
    """
    Linearly interpolate the full state vector at an exact age, rather
    than snapping to the nearest of solve_ivp's fixed output grid points.

    FIX (see findings log for the discrepancy this corrects): the
    original version of sensitivity_at_age picked the nearest grid index
    via argmin and used that grid point's own time as `t`. Because the
    solver's 500-point output grid has non-integer spacing (~0.1002 yr
    over the 30-80 age range), "nearest neighbor" silently drifted the
    effective age away from the one actually requested -- e.g. asking
    for age 55.0 could return a graph labeled and computed at age 54.95.
    Harmless to any conclusion already drawn (the drift is under 0.1yr),
    but it meant graphs were not actually located where their filenames
    claimed. Interpolating instead guarantees the returned state
    corresponds to the EXACT requested age.
    """
    y_interp = np.array([
        np.interp(age, solution.t, solution.y[i, :])
        for i in range(solution.y.shape[0])
    ])
    return y_interp


def sensitivity_at_age(solution, age, params, scales):
    """Compute raw Jacobian and normalized sensitivity at the EXACT
    given age (interpolated -- see interpolate_state), not the nearest
    solver output grid point."""
    y = interpolate_state(solution, age)
    t = float(age)

    J = compute_jacobian(t, y, params)
    S = normalize_sensitivity(J, y, scales)

    return t, y, J, S


def top_interactions(S, state_names, k=8):
    """Return the k largest-magnitude off-diagonal entries of S, as a
    list of (source_var, target_var, value) sorted by |value| descending."""
    n = S.shape[0]
    entries = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # skip self-sensitivity, not an interaction
            if S[i, j] != 0.0:
                entries.append((state_names[j], state_names[i], S[i, j]))

    entries.sort(key=lambda e: abs(e[2]), reverse=True)
    return entries[:k]


def plot_sensitivity_heatmap(S, state_names, age, save_path):
    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = np.percentile(np.abs(S), 99) or 1.0
    im = ax.imshow(S, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(state_names)))
    ax.set_yticks(range(len(state_names)))
    ax.set_xticklabels(state_names, rotation=90, fontsize=8)
    ax.set_yticklabels(state_names, fontsize=8)
    ax.set_xlabel("source variable (j)")
    ax.set_ylabel("target variable (i): d(dy_i/dt)/dy_j, normalized")
    ax.set_title(f"Local sensitivity matrix S at age {age:.0f}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="elasticity")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved sensitivity heatmap to {save_path}")


def main():
    print("Solving composite model to get a trajectory to analyze...")
    params = default_params()
    solution = run_composite_simulation(params)

    print("Computing robust per-variable normalization scales "
          "(90th percentile of |dy/dt| across the full trajectory)...")
    scales = compute_dydt_scales(solution, params)

    ages_to_check = [35, 50, 65, 80]

    for age in ages_to_check:
        t, y, J, S = sensitivity_at_age(solution, age, params, scales)

        n_nonzero = np.count_nonzero(S)
        max_abs = np.max(np.abs(S))
        print(f"\n--- Sensitivity at age {t:.1f} ---")
        print(f"  Non-zero entries: {n_nonzero} / {S.size}")
        print(f"  Max |S_ij|: {max_abs:.4f}")

        print("  Top interactions (source -> target : elasticity):")
        for source, target, value in top_interactions(S, STATE_NAMES, k=6):
            print(f"    {source:8s} -> {target:8s} : {value:+.4f}")

    # Save a heatmap for the middle-aged timepoint as the visual check.
    t, y, J, S = sensitivity_at_age(solution, 50, params, scales)
    save_path = os.path.join(DIRS["results"], "phase4_sensitivity_heatmap_age50.png")
    plot_sensitivity_heatmap(S, STATE_NAMES, t, save_path)

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
