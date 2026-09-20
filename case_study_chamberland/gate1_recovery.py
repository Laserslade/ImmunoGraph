"""
Gate 1 static equivalence recovery. Produces the canonical 480-check
export and a separate diagnostic per-state RHS export.
"""

import sys
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "reference" / "authors_orig"))

import chamberland_reference as ref
from authors_orig.parameters import Parameters
from authors_orig.InitialConditions import InitialConditions
from authors_orig.equations_SA import ODEsystem_SA

RTOL = 1e-9
SEXES = [0, 1]
APS = [0, 1]
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)


def run():
    canonical = []
    diagnostic_rhs = []

    # Parameter checks, 76 parameters times 4 demographic configurations.
    for sex in SEXES:
        for ap in APS:
            p_auth = Parameters(Sex=sex, APOE4_status=ap)
            p_ours = ref.default_params(sex, ap)
            auth_attrs = {k: v for k, v in vars(p_auth).items()
                          if isinstance(v, (int, float)) and not k.startswith("_")}
            common = sorted(set(auth_attrs) & set(p_ours))
            for key in common:
                a, o = auth_attrs[key], p_ours[key]
                abs_err = abs(a - o)
                rel_err = abs_err / max(abs(a), 1e-300)
                ok = bool(np.isclose(a, o, rtol=RTOL, atol=0))
                canonical.append({"check_category": "parameter", "demographic": f"sex={sex}_AP={ap}",
                                   "variable": key, "reference_value": a, "reconstructed_value": o,
                                   "abs_error": abs_err, "rel_error": rel_err, "pass": ok})

    # Initial condition checks, 19 states times 4 configurations.
    for sex in SEXES:
        for ap in APS:
            p_auth = Parameters(Sex=sex, APOE4_status=ap)
            p_ours = ref.default_params(sex, ap)
            y0_auth = InitialConditions(p_auth, AgeStart=30)
            y0_ours = ref.default_initial_state(p_ours, age_start_years=30)
            for i, name in enumerate(ref.STATE_NAMES):
                a, o = y0_auth[i], y0_ours[i]
                abs_err = abs(a - o)
                rel_err = abs_err / max(abs(a), 1e-300)
                ok = bool(np.isclose(a, o, rtol=RTOL, atol=1e-30))
                canonical.append({"check_category": "initial_condition", "demographic": f"sex={sex}_AP={ap}",
                                   "variable": name, "reference_value": a, "reconstructed_value": o,
                                   "abs_error": abs_err, "rel_error": rel_err, "pass": ok})

    # RHS checks. 100 canonical vector-level checks, plus 1900 diagnostic
    # per-state rows. The diagnostic rows are not additional validation units.
    rng = np.random.default_rng(0)
    for sex in SEXES:
        for ap in APS:
            p_auth = Parameters(Sex=sex, APOE4_status=ap)
            p_ours = ref.default_params(sex, ap)
            y0 = InitialConditions(p_auth, AgeStart=30)
            for trial in range(25):
                factors = np.exp(rng.uniform(np.log(0.5), np.log(1.5), size=len(y0)))
                y = y0 * factors
                y[8] = max(y[8], 1e-6)
                t_days = rng.uniform(30 * 365, 80 * 365)
                dydt_auth = ODEsystem_SA(t_days, y, p_auth)
                dydt_ours = ref.chamberland_rhs(t_days, y, p_ours)
                vector_ok = bool(np.allclose(dydt_auth, dydt_ours, rtol=1e-8, atol=1e-30))
                canonical.append({"check_category": "rhs_random_sample", "demographic": f"sex={sex}_AP={ap}_trial={trial}",
                                   "variable": "ALL_19_STATES_VECTOR", "reference_value": "", "reconstructed_value": "",
                                   "abs_error": float(np.max(np.abs(dydt_auth - dydt_ours))), "rel_error": "", "pass": vector_ok})
                for i, name in enumerate(ref.STATE_NAMES):
                    a, o = dydt_auth[i], dydt_ours[i]
                    abs_err = abs(a - o)
                    rel_err = abs_err / max(abs(a), 1e-300)
                    ok = bool(np.isclose(a, o, rtol=1e-8, atol=1e-30))
                    diagnostic_rhs.append({"trial_id": f"sex={sex}_AP={ap}_trial={trial}", "state_variable": name,
                                            "reference_dydt": a, "reconstructed_dydt": o, "abs_error": abs_err,
                                            "rel_error": rel_err, "pass": ok})

    with open(OUT_DIR / "chamberland_gate1_validation_checks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check_category", "demographic", "variable", "reference_value",
                                           "reconstructed_value", "abs_error", "rel_error", "pass"])
        w.writeheader()
        w.writerows(canonical)

    with open(OUT_DIR / "chamberland_gate1_rhs_state_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial_id", "state_variable", "reference_dydt", "reconstructed_dydt",
                                           "abs_error", "rel_error", "pass"])
        w.writeheader()
        w.writerows(diagnostic_rhs)

    n_pass = sum(r["pass"] for r in canonical)
    print(f"canonical: {n_pass}/{len(canonical)} pass")
    print(f"diagnostic rhs: {sum(r['pass'] for r in diagnostic_rhs)}/{len(diagnostic_rhs)} pass")


if __name__ == "__main__":
    run()
