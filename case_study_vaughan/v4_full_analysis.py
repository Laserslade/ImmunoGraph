"""
V4 published parameter consistency analysis, complete 45 by 8 test set
with a single joint Benjamini Hochberg correction across all tests.
"""

import sys
from pathlib import Path
import csv
import numpy as np
from scipy.stats import spearmanr, hypergeom

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
import vaughan_reference as vr

RESULTS = ROOT / "results"
ENSEMBLE_DIR = ROOT / "frozen_ensemble"


def run():
    data = np.load(ENSEMBLE_DIR / "vaughan_exploratory_ensemble_v1_raw.npz", allow_pickle=True)
    param_names = list(data["param_names"])
    param_values = data["param_values"]
    acceptance = data["acceptance_status"]
    accepted_idx = np.where(acceptance == "accepted")[0]
    params_accepted = param_values[accepted_idx]

    outcomes = dict(np.load(RESULTS / "v4_outcomes.npz"))
    outcome_names = list(outcomes.keys())

    results = []
    for pi, pname in enumerate(param_names):
        pvals = params_accepted[:, pi]
        for oname in outcome_names:
            rho, p = spearmanr(pvals, outcomes[oname])
            results.append({"parameter": pname, "provenance": vr.PARAMETER_STATUS[pname],
                             "outcome": oname, "rho": rho, "raw_p": p})

    pvals_all = np.array([r["raw_p"] for r in results])
    order = np.argsort(pvals_all)
    m = len(pvals_all)
    ranked_p = pvals_all[order]
    bh_crit = (np.arange(1, m + 1) / m) * 0.05
    passed = ranked_p <= bh_crit
    sig_mask_sorted = np.zeros(m, dtype=bool)
    if passed.any():
        k_max = np.max(np.where(passed)[0])
        sig_mask_sorted[:k_max + 1] = True
    sig_mask = np.zeros(m, dtype=bool)
    sig_mask[order] = sig_mask_sorted

    q_vals = np.zeros(m)
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        q = ranked_p[rank] * m / (rank + 1)
        running_min = min(running_min, q)
        q_vals[idx] = running_min

    for i, r in enumerate(results):
        r["bh_q"] = q_vals[i]
        r["significant"] = bool(sig_mask[i])

    with open(RESULTS / "vaughan_v4_all_360_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["parameter", "provenance", "outcome", "rho", "raw_p", "bh_q", "significant"])
        w.writeheader()
        w.writerows(results)

    n_sig = sum(r["significant"] for r in results)
    n_partial_tests = sum(1 for r in results if r["provenance"] == "partial_range")
    n_partial_sig = sum(1 for r in results if r["provenance"] == "partial_range" and r["significant"])
    pval_enrich = hypergeom.sf(n_partial_sig - 1, len(results), n_partial_tests, n_sig)

    print(f"total {len(results)}, significant {n_sig}")
    print(f"published support {n_partial_sig}/{n_partial_tests}")
    print(f"hypergeometric p = {pval_enrich:.4f}")


if __name__ == "__main__":
    run()
