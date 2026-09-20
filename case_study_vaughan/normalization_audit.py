"""
Normalization stability audit, recovered against the frozen ensemble.
Requires the graph cache built by build_graph_cache.py to exist first.
"""

import sys
from pathlib import Path
import csv
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
import vaughan_reference as vr

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
FLOOR = 1e-8


def run():
    d = np.load(RESULTS / "denominators.npy")
    cache = np.load(RESULTS / "graph_cache.npz", allow_pickle=True)
    names = vr.STATE_NAMES
    times = list(cache["times"])

    rows_denom, rows_conc, rows_assoc = [], [], []

    for ti, t in enumerate(times):
        weight = cache[f"weight_t{ti}"]
        in_degree = np.abs(weight).sum(axis=2)

        for i, nm in enumerate(names):
            vals = d[:, i]
            rows_denom.append({"snapshot_h": t, "node": nm, "d_min": vals.min(),
                                "d_p1": np.percentile(vals, 1), "d_p10": np.percentile(vals, 10),
                                "d_median": np.median(vals), "d_p90": np.percentile(vals, 90), "d_max": vals.max(),
                                "frac_at_floor": np.mean(vals <= FLOOR * 1.0001),
                                "frac_within_10x_floor": np.mean(vals <= 10 * FLOOR),
                                "frac_within_100x_floor": np.mean(vals <= 100 * FLOOR)})
            rho, p = spearmanr(np.log10(d[:, i]), in_degree[:, i])
            rows_assoc.append({"snapshot_h": t, "node": nm, "spearman_rho_logd_vs_indegree": rho, "p_value": p})

        total_per_sample = in_degree.sum(axis=1)
        order = np.argsort(-total_per_sample)
        cum = np.cumsum(total_per_sample[order]) / total_per_sample.sum()
        n_total = len(order)
        n50 = int(np.searchsorted(cum, 0.5)) + 1
        n90 = int(np.searchsorted(cum, 0.9)) + 1
        rows_conc.append({"snapshot_h": t, "n_for_50pct_mass": n50, "pct_for_50pct_mass": n50 / n_total * 100,
                           "n_for_90pct_mass": n90, "pct_for_90pct_mass": n90 / n_total * 100})

    with open(RESULTS / "normalization_audit_denominators.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_denom[0].keys()))
        w.writeheader()
        w.writerows(rows_denom)
    with open(RESULTS / "normalization_audit_concentration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_conc[0].keys()))
        w.writeheader()
        w.writerows(rows_conc)
    with open(RESULTS / "normalization_audit_association.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_assoc[0].keys()))
        w.writeheader()
        w.writerows(rows_assoc)

    print("normalization audit exports written")


if __name__ == "__main__":
    run()
