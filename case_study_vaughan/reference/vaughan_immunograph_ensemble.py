"""
vaughan_immunograph_ensemble.py

Applies the project's existing, validated graph-construction method
(phase4_local_sensitivity.py + phase5_graph_construction.py from the
original neuroimmune-graph-project) to the FROZEN Vaughan exploratory
ensemble (vaughan_exploratory_ensemble_v1, 2041 accepted samples).

Ported unchanged in method, only retargeted from the 22-node composite
model to Vaughan's 7-state system:
  - finite-difference Jacobian (same eps, same non-negativity clamp)
  - normalization by the 90th-percentile |dy/dt| scale over each
    sample's OWN trajectory (same fix for the inflection-point blow-up
    documented in the original findings log)
  - graph edges = non-zero normalized sensitivity entries, signed

Question being asked (per project decision log): which network
properties survive parameter uncertainty across the 2041 accepted
exploratory parameterizations? NOT "what does Vaughan predict" -- these
are not Vaughan's fitted patients.

Snapshots taken at t = 0, 24, 48, 72, 120 hours (5 points across the
120h window) rather than the full trajectory, to keep ensemble-scale
compute tractable while still covering early/mid/late/resolve phases.
"""

import numpy as np
from scipy.integrate import solve_ivp
import vaughan_reference as vr

FINITE_DIFF_EPS = 1e-6
MIN_ABS_WEIGHT = 1e-9
SNAPSHOT_TIMES = [0, 24, 48, 72, 120]
N = vr.N_STATES
NAMES = vr.STATE_NAMES


def compute_jacobian(t, y, params, eps=FINITE_DIFF_EPS):
    J = np.zeros((N, N))
    for j in range(N):
        step = eps * max(abs(y[j]), 1.0)
        y_plus = y.copy(); y_plus[j] += step
        y_minus = y.copy(); y_minus[j] = max(y_minus[j] - step, 0.0)
        actual_step = y_plus[j] - y_minus[j]
        if actual_step <= 0:
            continue
        dydt_plus = vr.vaughan_rhs(t, y_plus, params)
        dydt_minus = vr.vaughan_rhs(t, y_minus, params)
        J[:, j] = (dydt_plus - dydt_minus) / actual_step
    return J


def compute_dydt_scales(sol, params):
    all_dydt = np.zeros((N, sol.y.shape[1]))
    for k in range(sol.y.shape[1]):
        all_dydt[:, k] = vr.vaughan_rhs(sol.t[k], sol.y[:, k], params)
    return np.percentile(np.abs(all_dydt), 90, axis=1)


def normalize_sensitivity(J, y, scales, min_denominator=1e-8):
    S = np.zeros((N, N))
    for i in range(N):
        if scales[i] < min_denominator:
            continue
        for j in range(N):
            if abs(y[j]) < min_denominator:
                continue
            S[i, j] = J[i, j] * y[j] / scales[i]
    return S


def build_signed_adjacency(S, min_abs_weight=MIN_ABS_WEIGHT):
    """Returns (present, weight) as (N,N) arrays; present[i,j]=1 if edge j->i exists."""
    present = (np.abs(S) > min_abs_weight).astype(int)
    np.fill_diagonal(present, 0)
    return present, S


def run_one_sample(params, y0, t_span=(0, 120)):
    """Integrate, compute Jacobian-graph snapshots at SNAPSHOT_TIMES.
    Returns list of (t, present, weight) or None if integration fails."""
    sol = solve_ivp(vr.vaughan_rhs, t_span, y0, method="BDF", args=(params,),
                     t_eval=np.linspace(*t_span, 241), atol=1e-8, rtol=1e-8)
    if not sol.success:
        return None
    scales = compute_dydt_scales(sol, params)
    snapshots = []
    for t_snap in SNAPSHOT_TIMES:
        y_interp = np.array([np.interp(t_snap, sol.t, sol.y[i, :]) for i in range(N)])
        J = compute_jacobian(t_snap, y_interp, params)
        S = normalize_sensitivity(J, y_interp, scales)
        present, weight = build_signed_adjacency(S)
        snapshots.append((t_snap, present, weight))
    return snapshots
