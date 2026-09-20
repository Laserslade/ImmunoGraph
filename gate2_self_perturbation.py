"""
Runs the reference implementation against itself with a tiny initial
condition perturbation, to check whether the source model is sensitive.
"""

import sys
import csv
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "reference" / "authors_orig"))

import chamberland_reference as ref
from authors_orig.parameters import Parameters
from authors_orig.InitialConditions import InitialConditions
from authors_orig.equations_SA import ODEsystem_SA

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)


def run():
    p_auth = Parameters(Sex=1, APOE4_status=1)
    y0 = InitialConditions(p_auth, AgeStart=30)
    y0_perturbed = y0 * (1 + 1e-13)

    t_span = [365 * 30, 365 * 80]
    t_eval = np.linspace(*t_span, 501)
    opts = {"atol": 1e-10, "rtol": 1e-10}

    sol_a = solve_ivp(ODEsystem_SA, t_span, y0, method="BDF", args=[p_auth], t_eval=t_eval, **opts)
    sol_b = solve_ivp(ODEsystem_SA, t_span, y0_perturbed, method="BDF", args=[p_auth], t_eval=t_eval, **opts)
    diff = np.abs(sol_a.y - sol_b.y)
    rel = diff / np.maximum(np.abs(sol_a.y), 1e-300)

    rows = [{"state_variable": name, "max_rel_diff_self_perturbation": rel[i].max()}
            for i, name in enumerate(ref.STATE_NAMES)]

    with open(OUT_DIR / "chamberland_gate2_self_perturbation_control.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["state_variable", "max_rel_diff_self_perturbation"])
        w.writeheader()
        w.writerows(rows)

    np.savez_compressed(OUT_DIR / "chamberland_gate2_self_perturbation_trajectories.npz",
                         t=sol_a.t, baseline=sol_a.y, perturbed=sol_b.y, state_names=ref.STATE_NAMES)

    for r in rows:
        print(r["state_variable"], r["max_rel_diff_self_perturbation"])


if __name__ == "__main__":
    run()
