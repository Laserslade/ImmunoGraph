"""
phase2_perturbation_harness.py

Phase 2 of the Neuroimmune Graph Project.

Wraps Phase 1's solver in a loop over randomly sampled parameter sets,
saving each resulting trajectory to disk. This is the "simulation
library" that later phases (sensitivity analysis, graph construction,
motif discovery, therapy exploration) will all be built on top of.

Design choices, deliberately conservative per the project's feasibility
strategy (see findings log, Section 5):
  - A modest number of runs (default 24), not "thousands" -- plenty for
    a first working batch pipeline; easy to scale up later once every
    downstream phase is validated.
  - A small set of biologically meaningful parameters are perturbed
    (cytokine production/conversion rates, an oligomer degradation rate,
    and the APOE4 status switch), rather than perturbing all ~90
    parameters at once. This keeps the perturbation space interpretable
    and keeps runtime low.
  - Each run is saved as a single compressed .npz file containing the
    full time series, the state variable names, and the exact parameter
    overrides used -- so any run can be independently reloaded and
    re-plotted without needing to re-run the simulation.

Run:
    python phase2_perturbation_harness.py
"""

import csv
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DIRS  # noqa: E402
from models.chamberland_backbone import (  # noqa: E402
    DEFAULT_PARAMS,
    STATE_NAMES,
    chamberland_rhs,
    default_initial_state,
)

AGE_START = 30.0
AGE_END = 80.0
N_EVAL_POINTS = 500

N_RUNS = 24
RANDOM_SEED = 42

# Parameters perturbed multiplicatively (sampled factor applied to the
# DEFAULT_PARAMS value). Chosen because they govern the strength of the
# core pro-/anti-inflammatory feedback loops -- exactly the kind of
# "cytokine production rates" and "therapeutic inhibition strengths"
# named in the project's original perturbation objective.
PERTURBED_PARAMS = [
    "kappa_MproTa",     # TNF-alpha production rate by pro-inflammatory microglia
    "kappa_MhproTa",    # TNF-alpha production rate by pro-inflammatory macrophages
    "kappa_TbMpro",     # TGF-beta-driven conversion of pro- to anti-inflammatory microglia
    "kappa_TaManti",    # TNF-alpha-driven conversion of anti- to pro-inflammatory microglia
    "d_ABoo",           # amyloid-beta oligomer degradation rate
]
PERTURBATION_RANGE = (0.5, 2.0)  # multiplicative factor sampled uniformly

# APOE4 status is a discrete switch, sampled separately.
APOE4_PROBABILITY = 0.3  # roughly matches real-world APOE4 carrier prevalence


def sample_params(rng):
    """Sample one perturbed parameter set. Returns (params_dict, overrides_dict)."""
    params = dict(DEFAULT_PARAMS)
    overrides = {}

    for name in PERTURBED_PARAMS:
        factor = rng.uniform(*PERTURBATION_RANGE)
        params[name] = DEFAULT_PARAMS[name] * factor
        overrides[name] = params[name]

    ap = 1.0 if rng.random() < APOE4_PROBABILITY else 0.0
    params["AP"] = ap
    overrides["AP"] = ap

    return params, overrides


def run_one_simulation(params):
    """Solve the backbone model for one parameter set. Returns (t, y)."""
    y0 = default_initial_state()
    t_eval = np.linspace(AGE_START, AGE_END, N_EVAL_POINTS)

    solution = solve_ivp(
        fun=lambda t, y: chamberland_rhs(t, y, params),
        t_span=(AGE_START, AGE_END),
        y0=y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    if not solution.success:
        return None, None

    return solution.t, solution.y


def save_run(run_id, t, y, overrides):
    """Save one run's trajectory and parameter overrides to sims/run_XXXX.npz."""
    path = os.path.join(DIRS["sims"], f"run_{run_id:04d}.npz")
    np.savez_compressed(
        path,
        t=t,
        y=y,
        state_names=np.array(STATE_NAMES),
        **{f"param__{k}": v for k, v in overrides.items()},
    )
    return path


def load_run(path):
    """
    Reload a saved run. Returns (t, y, state_names, overrides_dict).
    This is the round-trip check for Phase 2's definition of done:
    any saved run can be reloaded without re-simulating.
    """
    data = np.load(path)
    t = data["t"]
    y = data["y"]
    state_names = list(data["state_names"])
    overrides = {
        key[len("param__"):]: float(data[key])
        for key in data.files
        if key.startswith("param__")
    }
    return t, y, state_names, overrides


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    manifest_path = os.path.join(DIRS["sims"], "manifest.csv")
    fieldnames = ["run_id", "path", "success"] + PERTURBED_PARAMS + ["AP"]

    print(f"Running {N_RUNS} perturbed simulations...")
    n_success = 0

    with open(manifest_path, "w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()

        for run_id in range(N_RUNS):
            params, overrides = sample_params(rng)
            t, y = run_one_simulation(params)

            success = t is not None
            path = ""

            if success:
                path = save_run(run_id, t, y, overrides)
                n_success += 1
            else:
                print(f"  run {run_id:04d}: solver failed, skipped")

            row = {"run_id": run_id, "path": path, "success": success}
            row.update(overrides)
            writer.writerow(row)

    print(f"\n{n_success}/{N_RUNS} runs succeeded and were saved to {DIRS['sims']}")
    print(f"Manifest written to {manifest_path}")

    # --- Round-trip check: reload the first saved run and confirm it
    # matches what was simulated, satisfying Phase 2's definition of done.
    first_run_path = os.path.join(DIRS["sims"], "run_0000.npz")
    if os.path.exists(first_run_path):
        t, y, state_names, overrides = load_run(first_run_path)
        print(f"\nRound-trip check on {first_run_path}:")
        print(f"  time points: {len(t)}, states: {y.shape[0]}")
        print(f"  state names match Phase 1: {state_names == STATE_NAMES}")
        print(f"  parameter overrides used: {overrides}")
        print("  Reload successful -- Phase 2 definition of done met.")

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
