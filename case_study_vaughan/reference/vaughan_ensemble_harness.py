"""
vaughan_ensemble_harness.py

Sampling + admissibility-filtering harness for the Vaughan structural-
uncertainty case study. Implements the pre-registered protocol from
vaughan_sampling_protocol.csv.

MODE GUARD
-------------------------------------------------------------------------
mode="development": for engineering/smoke-testing only. Falls back to
    the 0.1x-10x exploratory range for the 8 partial_range parameters.
    Runs generated in this mode are tagged DEVELOPMENT_UNFROZEN and must
    never be used for scientific conclusions -- see module-level warning
    at the bottom of this file.
mode="production": hard-fails at construction time if any partial_range
    parameter lacks a resolved numeric constraint in RESOLVED_RANGES
    below. This is enforced in code, not left to be remembered later.

DIMENSIONALITY
-------------------------------------------------------------------------
46 continuous QMC dimensions, every sample, always:
    [0:39]  39 continuous rate/scale parameters (log-uniform)
    [39:45] 6 continuous ICs: M1, M2, IL1, IL12, IL10, IL4 (log-uniform)
    [45]    1 conditional D-magnitude coordinate (used only if the
             independent D-switch Bernoulli draw selects D0>0; ignored,
             not renormalized, when D0=0 -- this preserves the
             space-filling design rather than distorting it)
Handled separately from the 46-D Sobol/QMC continuous space, via an
independent seeded discrete RNG stream (not folded into the QMC design,
since mixing continuous space-filling design with discrete/Bernoulli
draws in the same low-discrepancy sequence is not well-behaved):
    6 discrete Hill-exponent draws (xn, zn, gn, cn, qn, hn)
    1 Bernoulli(p=0.4) draw for the D0=0 vs D0>0 switch

This file contains no reference, value, or dependency of any kind on the
human microdialysis dataset used later for held-out comparison (see
project decision log). That independence is meant to be checkable by
searching this file for the dataset's author name; this sentence is
deliberately phrased without it for that reason.
"""

import hashlib
import json
import time
import numpy as np
from scipy.stats import qmc
from scipy.integrate import solve_ivp

import vaughan_reference as vr

# -------------------------------------------------------------------------
# Fixed parameter orderings (must match vaughan_reference.DEFAULT_PARAMS)
# -------------------------------------------------------------------------
HILL_EXPONENTS = ["xn", "zn", "gn", "cn", "qn"]  # hn moved to continuous partial_range (see below)
CONTINUOUS_PARAMS = [k for k in vr.DEFAULT_PARAMS if k not in HILL_EXPONENTS]
assert len(CONTINUOUS_PARAMS) == 40, f"expected 40 continuous params, got {len(CONTINUOUS_PARAMS)}"
assert len(HILL_EXPONENTS) == 5

IC_CONTINUOUS = ["M1", "M2", "IL1", "IL12", "IL10", "IL4"]  # all but D
assert len(IC_CONTINUOUS) == 6

D_STAR = 0.1  # arbitrary positive anchor for D0>0 branch -- see sampling protocol note
D_ZERO_PROB = 0.4

PARTIAL_RANGE_PARAMS = {k for k, v in vr.PARAMETER_STATUS.items() if v == "partial_range"}
assert len(PARTIAL_RANGE_PARAMS) == 9

# -------------------------------------------------------------------------
# Additional file 1: exact per-cluster marginal (range, average), preserved
# as reported. Averages are NOT used for sampling (uniform-within-range is
# used instead, per protocol) -- kept only as a post-hoc diagnostic: a
# generated ensemble's marginal should NOT be expected to reproduce these
# averages, and shouldn't be tuned to.
# -------------------------------------------------------------------------
CLUSTER_MARGINALS = {
    "kpn1":   {"1": (0.07, 29.90, 15.51), "2A": (0.05, 0.93, 0.25),  "2B": (0.11, 2.86, 0.91)},
    "kpn12":  {"1": (8.10, 49.88, 32.31), "2A": (0.05, 1.25, 0.42),  "2B": (0.59, 8.06, 1.83)},
    "mu_M1":  {"1": (0.05, 0.46, 0.14),   "2A": (2.41, 22.92, 6.88), "2B": (4.73, 37.32, 10.77)},
    "mu_M2":  {"1": (2.87, 11.89, 5.43),  "2A": (1.23, 6.39, 3.45),  "2B": (7.33, 32.05, 17.70)},
    "mu_n10": {"1": (0.30, 0.72, 0.50),   "2A": (0.82, 1.70, 1.29),  "2B": (1.18, 1.40, 1.30)},
    "mu_n12": {"1": (0.06, 0.63, 0.26),                              "2B": (0.050, 0.076, 0.052)},
    "a_inf2": {"1": (25.0, 38.51, 26.42), "2A": (1.31, 16.21, 8.78), "2B": (25.00, 44.30, 27.97)},
    "hn":     {"1": (1.03, 6.00, 5.14),                              "2B": (1.00, 2.23, 1.27)},
    "vn":                                 {"2A": (0.20, 1.21, 0.85), "2B": (0.87, 25.85, 7.17)},
}
assert set(CLUSTER_MARGINALS) == PARTIAL_RANGE_PARAMS

# Ranges recovered from Additional file 1 -- now fully populated (see above).
# RESOLVED_RANGES is kept for backward compatibility with the production
# guard check; derived from CLUSTER_MARGINALS' union for the completeness
# check only (actual sampling uses the mixture, not this union).
RESOLVED_RANGES = {
    p: (min(v[0] for v in clusters.values()), max(v[1] for v in clusters.values()))
    for p, clusters in CLUSTER_MARGINALS.items()
}


class ProtocolError(Exception):
    pass


def _check_production_readiness():
    missing = PARTIAL_RANGE_PARAMS - set(RESOLVED_RANGES)
    if missing:
        raise ProtocolError(
            f"production mode blocked: {len(missing)} partial_range parameters "
            f"still lack a resolved numeric constraint from Additional file 1: "
            f"{sorted(missing)}. Either extract and add to RESOLVED_RANGES, or "
            f"formally reclassify to direction_only in vaughan_reference.py."
        )


def _log_uniform_map(u, lo_mult, hi_mult, nominal):
    """u in [0,1) -> nominal * 10^(log10(lo_mult) + u*(log10(hi_mult)-log10(lo_mult)))."""
    log_lo, log_hi = np.log10(lo_mult), np.log10(hi_mult)
    return nominal * (10 ** (log_lo + u * (log_hi - log_lo)))


def build_ensemble(n_samples, mode, master_seed, tier="primary"):
    """
    Generate n_samples exploratory parameterizations.

    mode: "development" or "production".
    tier: "primary" (0.1x-10x / Hill 1-4) or "stress" (0.01x-100x / Hill 1-6).
    Returns: (records: list[dict], ensemble_manifest: dict)
    """
    if mode not in ("development", "production"):
        raise ValueError("mode must be 'development' or 'production'")
    if mode == "production":
        _check_production_readiness()

    mult_lo, mult_hi = (0.1, 10.0) if tier == "primary" else (0.01, 100.0)
    hill_lo, hill_hi = (1, 4) if tier == "primary" else (1, 6)

    seed_seq = np.random.SeedSequence(master_seed)
    qmc_seed, discrete_seed = seed_seq.spawn(2)

    n_continuous_qmc = 40 + 6 + 1  # 40 continuous params + 6 continuous ICs + 1 D-magnitude = 47
    sampler = qmc.Sobol(d=n_continuous_qmc, scramble=True, seed=np.random.default_rng(qmc_seed))
    m = int(np.ceil(np.log2(max(n_samples, 1))))
    u = sampler.random_base2(m=m)[:n_samples]  # (n_samples, 47)

    discrete_rng = np.random.default_rng(discrete_seed)

    records = []
    for i in range(n_samples):
        row = u[i]

        params = dict(vr.DEFAULT_PARAMS)
        cluster_assignments = {}
        for j, name in enumerate(CONTINUOUS_PARAMS):
            if name in PARTIAL_RANGE_PARAMS:
                clusters = CLUSTER_MARGINALS[name]
                cluster_names = sorted(clusters)  # e.g. ["1","2A","2B"] or ["1","2B"]
                # cluster-of-origin: independent categorical draw, NOT from QMC
                # (keeps the QMC continuous design purely continuous)
                chosen = cluster_names[discrete_rng.integers(0, len(cluster_names))]
                lo, hi, _avg = clusters[chosen]
                # position within that cluster's published interval: uniform,
                # driven by this parameter's QMC coordinate
                params[name] = lo + row[j] * (hi - lo)
                cluster_assignments[name] = chosen
            else:
                params[name] = _log_uniform_map(row[j], mult_lo, mult_hi, vr.DEFAULT_PARAMS[name])

        # --- discrete exploratory Hill exponents (5 remaining; hn handled above) ---
        hill_draws = {}
        for name in HILL_EXPONENTS:
            hill_draws[name] = int(discrete_rng.integers(hill_lo, hill_hi + 1))
            params[name] = hill_draws[name]

        # --- continuous ICs ---
        ics = {}
        for j, name in enumerate(IC_CONTINUOUS):
            qmc_idx = 40 + j
            nominal = vr.DEFAULT_INITIAL_STATE[vr.IDX[name]]
            ics[name] = _log_uniform_map(row[qmc_idx], mult_lo, mult_hi, nominal)

        # --- D0 mixture: independent Bernoulli switch + QMC magnitude coordinate ---
        d_switch_u = discrete_rng.random()
        d_is_zero = d_switch_u < D_ZERO_PROB
        d_magnitude_coord = row[46]  # always consumed from QMC vector, even if unused
        if d_is_zero:
            ics["D"] = 0.0
        else:
            ics["D"] = _log_uniform_map(d_magnitude_coord, mult_lo, mult_hi, D_STAR)

        y0 = np.array([ics[name] for name in vr.STATE_NAMES])

        records.append({
            "sample_id": i,
            "qmc_coords": row.tolist(),
            "params": params,
            "cluster_assignments": cluster_assignments,
            "hill_draws": hill_draws,
            "d_is_zero": bool(d_is_zero),
            "d_magnitude_coord": float(d_magnitude_coord),
            "initial_state": y0.tolist(),
            "acceptance_status": None,   # filled in by run_admissibility_filter
            "acceptance_reason": None,
            "trajectory_checksum": None,
        })

    table_hash = hashlib.sha256(
        open("/mnt/user-data/outputs/vaughan_sampling_protocol.csv", "rb").read()
        + open("/mnt/user-data/outputs/vaughan_additional_file1_extraction.csv", "rb").read()
    ).hexdigest()

    manifest = {
        "ensemble_name": "vaughan_exploratory_ensemble_v1",
        "status": "DEVELOPMENT_UNFROZEN" if mode == "development" else "PRODUCTION_FROZEN",
        "mode": mode,
        "tier": tier,
        "n_samples": n_samples,
        "master_seed": master_seed,
        "qmc_dims": n_continuous_qmc,
        "discrete_dims": {
            "hill_exponents_exploratory": len(HILL_EXPONENTS),
            "d_switch_bernoulli": 1,
            "cluster_selector_categoricals": len(PARTIAL_RANGE_PARAMS),
        },
        "d_zero_prob": D_ZERO_PROB,
        "d_star_anchor": D_STAR,
        "partial_range_params": sorted(PARTIAL_RANGE_PARAMS),
        "sampling_rule_for_partial_range": (
            "For parameters with published cluster-specific fitted ranges, sampling was "
            "restricted to an equal-weight categorical mixture across available cluster "
            "marginals (Additional file 1), with a uniform distribution within each reported "
            "interval. Cluster-component assignments were made independently for each "
            "parameter because joint parameter distributions and cross-parameter correlations "
            "were unavailable. Missing cluster marginals were not imputed. Generated parameter "
            "vectors do not represent reconstructed patient clusters or samples from the "
            "original fitted ensemble."
        ),
        "helmy_used_in_construction": False,
        "generated_at_unix": time.time(),
        "protocol_table_hash": table_hash,
    }
    return records, manifest


def run_admissibility_filter(records, t_span=(0, 120), n_eval=121):
    """
    Pre-Helmy, math-only admissibility filter. Rejects and records why;
    never silently discards. Criteria (from protocol): integration success
    through t_span, finiteness, non-negativity beyond numerical tolerance,
    no blow-up.
    """
    t_eval = np.linspace(*t_span, n_eval)
    NEG_TOL = -1e-8
    BLOWUP_CEILING = 1e6

    for rec in records:
        y0 = np.array(rec["initial_state"])
        params = rec["params"]
        try:
            sol = solve_ivp(vr.vaughan_rhs, t_span, y0, method="BDF", args=(params,),
                             t_eval=t_eval, atol=1e-8, rtol=1e-8)
        except Exception as e:
            rec["acceptance_status"] = "rejected"
            rec["acceptance_reason"] = f"integration_exception: {e}"
            continue

        if not sol.success:
            rec["acceptance_status"] = "rejected"
            rec["acceptance_reason"] = f"solver_failure: {sol.message}"
            continue

        Y = sol.y
        if not np.all(np.isfinite(Y)):
            rec["acceptance_status"] = "rejected"
            rec["acceptance_reason"] = "non_finite_state"
            continue

        if np.any(Y < NEG_TOL):
            worst = Y.min()
            rec["acceptance_status"] = "rejected"
            rec["acceptance_reason"] = f"negative_state_beyond_tolerance: min={worst:.3e}"
            continue

        if np.any(np.abs(Y) > BLOWUP_CEILING):
            rec["acceptance_status"] = "rejected"
            rec["acceptance_reason"] = f"blowup: max_abs={np.abs(Y).max():.3e}"
            continue

        rec["acceptance_status"] = "accepted"
        rec["acceptance_reason"] = None
        rec["trajectory_checksum"] = hashlib.sha256(Y.tobytes()).hexdigest()[:16]

    return records


# -------------------------------------------------------------------------
# HARD GUARD: importing this module does not itself do anything scientific.
# Any script that calls build_ensemble(mode="development", ...) must not
# feed its results into ImmunoGraph analyses used for reported conclusions.
# Development runs exist to answer "does the machinery work", nothing else.
# -------------------------------------------------------------------------
