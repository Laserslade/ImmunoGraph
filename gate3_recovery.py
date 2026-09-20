"""
Gate 3 recovery. Checks the three preregistered published behavior
criteria against the reference implementation.
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
    t_span = [365 * 30, 365 * 80]
    t_eval = np.linspace(*t_span, 201)
    opts = {"atol": 1e-10, "rtol": 1e-10}

    results = {}
    for sex in [0, 1]:
        for ap in [0, 1]:
            p_auth = Parameters(Sex=sex, APOE4_status=ap)
            y0 = InitialConditions(p_auth, AgeStart=30)
            sol = solve_ivp(ODEsystem_SA, t_span, y0, method="BDF", args=[p_auth], t_eval=t_eval, **opts)
            results[(sex, ap)] = sol.y

    rows = []

    # Criterion A, trajectory direction behavior.
    for sex in [0, 1]:
        for ap in [0, 1]:
            y = results[(sex, ap)]
            n_decline = 100 * (1 - y[8, -1] / y[8, 0])
            a_rise = y[9, -1] - y[9, 0]
            mpro_rise = y[11, -1] - y[11, 0]
            mhpro_rise = y[13, -1] - y[13, 0]
            ta_rise = y[17, -1] - y[17, 0]
            ok = n_decline > 0 and a_rise > 0 and mpro_rise > 0 and mhpro_rise > 0 and ta_rise > 0
            rows.append({"criterion": "A_trajectory_direction", "demographic": f"sex={sex}_AP={ap}",
                         "published_behavior": "N declines; A, Mpro, Mhpro, Ta rise with age",
                         "reconstructed_behavior": f"N_decline={n_decline:.2f}pct",
                         "quantitative_comparison": "direction only", "pass": ok})

    # Criterion B, APOE4 effect on amyloid plaque at age 80.
    for sex in [0, 1]:
        ap_minus, ap_plus = results[(sex, 0)][3, -1], results[(sex, 1)][3, -1]
        ratio = ap_plus / ap_minus
        ok = ratio > 1
        rows.append({"criterion": "B_APOE4_effect", "demographic": f"sex={sex}",
                     "published_behavior": "APOE4 carriers show higher amyloid plaque",
                     "reconstructed_behavior": f"ratio_plus_over_minus={ratio:.2f}",
                     "quantitative_comparison": "direction check", "pass": ok})

    # Criterion C, age dependent neuronal loss.
    for sex in [0, 1]:
        for ap in [0, 1]:
            y = results[(sex, ap)]
            n_decline = 100 * (1 - y[8, -1] / y[8, 0])
            monotonic = bool(np.all(np.diff(y[8]) <= 1e-12))
            ok = 0 < n_decline < 50 and monotonic
            rows.append({"criterion": "C_neuronal_loss", "demographic": f"sex={sex}_AP={ap}",
                         "published_behavior": "moderate monotonic neuronal decline",
                         "reconstructed_behavior": f"N_decline={n_decline:.2f}pct, monotonic={monotonic}",
                         "quantitative_comparison": "0-50pct bound and monotonicity", "pass": ok})

    with open(OUT_DIR / "chamberland_gate3_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["criterion", "demographic", "published_behavior",
                                           "reconstructed_behavior", "quantitative_comparison", "pass"])
        w.writeheader()
        w.writerows(rows)

    np.savez_compressed(OUT_DIR / "chamberland_gate3_trajectories.npz", t=t_eval, state_names=ref.STATE_NAMES,
                         **{f"y_{s}_{a}": results[(s, a)] for s in [0, 1] for a in [0, 1]})

    total_pass = sum(1 for r in rows if r["pass"])
    print(f"{total_pass}/{len(rows)} criterion instances pass")


if __name__ == "__main__":
    run()
