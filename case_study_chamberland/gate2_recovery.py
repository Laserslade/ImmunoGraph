"""
Gate 2 numerical equivalence recovery. Integrates the reference
implementation and the clean room reconstruction, compares trajectories.
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

AGE_START, AGE_END = 30, 80
T_SPAN = [365 * AGE_START, 365 * AGE_END]
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], 501)
SOLVER_OPTS = {"atol": 1e-10, "rtol": 1e-10}
STRINGENT_TOL = 1e-4
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)


def run():
    canonical = []
    trajectories = {}

    for sex in [0, 1]:
        for ap in [0, 1]:
            label = f"sex={sex}_AP={ap}"
            p_auth = Parameters(Sex=sex, APOE4_status=ap)
            y0 = InitialConditions(p_auth, AgeStart=AGE_START)
            p_ours = ref.default_params(sex, ap)

            sol_auth = solve_ivp(ODEsystem_SA, T_SPAN, y0, method="BDF", args=[p_auth],
                                  t_eval=T_EVAL, **SOLVER_OPTS)
            sol_ours = solve_ivp(ref.chamberland_rhs, T_SPAN, y0, method="BDF", args=(p_ours,),
                                  t_eval=T_EVAL, **SOLVER_OPTS)
            assert sol_auth.success and sol_ours.success

            trajectories[f"t_{label}"] = sol_auth.t
            trajectories[f"authors_{label}"] = sol_auth.y
            trajectories[f"ours_{label}"] = sol_ours.y

            Ya, Yo = sol_auth.y, sol_ours.y
            for i, name in enumerate(ref.STATE_NAMES):
                a, o = Ya[i], Yo[i]
                abs_err = np.abs(a - o)
                max_abs_err = abs_err.max()
                scale = np.maximum(np.abs(a), np.abs(o))
                scale = np.where(scale < 1e-300, 1.0, scale)
                max_rel_err = (abs_err / scale).max()
                rng_a = a.max() - a.min()
                rng_a = rng_a if rng_a > 1e-300 else max(abs(a.max()), 1.0)
                nrmse = np.sqrt(np.mean((a - o) ** 2)) / rng_a
                status = "OK" if (max_rel_err < STRINGENT_TOL or nrmse < STRINGENT_TOL) else "MISMATCH"
                canonical.append({"demographic": label, "state_variable": name, "max_abs_error": max_abs_err,
                                   "max_rel_error": max_rel_err, "nrmse": nrmse, "status": status})

    with open(OUT_DIR / "chamberland_gate2_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["demographic", "state_variable", "max_abs_error", "max_rel_error",
                                           "nrmse", "status"])
        w.writeheader()
        w.writerows(canonical)

    np.savez_compressed(OUT_DIR / "chamberland_gate2_trajectories.npz", **trajectories)

    n_mismatch = sum(1 for r in canonical if r["status"] == "MISMATCH")
    print(f"{len(canonical)} state by condition comparisons, {n_mismatch} mismatch")


if __name__ == "__main__":
    run()
