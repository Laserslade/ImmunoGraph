"""
Recovers the complete node by snapshot directional hub statistics using
the corrected weighted and topology in and out degree functions.
"""

import sys
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
import vaughan_reference as vr
import vaughan_degree_utils as du

RESULTS = ROOT / "results"


def run():
    names = vr.STATE_NAMES
    cache = np.load(RESULTS / "graph_cache.npz", allow_pickle=True)
    times = list(cache["times"])

    rows = []
    for ti, t in enumerate(times):
        w = cache[f"weight_t{ti}"]
        pres = cache[f"presence_t{ti}"]
        wi = du.weighted_in_degree(w)
        wo = du.weighted_out_degree(w)
        topo_i = du.topology_in_degree(pres)
        topo_o = du.topology_out_degree(pres)
        for metric_name, mat in [("weighted_in", wi), ("weighted_out", wo),
                                  ("topology_in", topo_i), ("topology_out", topo_o)]:
            top1 = du.top1_frequency(mat, len(names))
            for i, nm in enumerate(names):
                rows.append({"snapshot_h": t, "node": nm, "metric": metric_name,
                             "top1_frequency_pct": top1[i] * 100, "mean_value": mat[:, i].mean(),
                             "median_value": np.median(mat[:, i])})

    with open(RESULTS / "vaughan_directional_hub_dynamics.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)

    print(f"wrote {len(rows)} directional hub rows")


if __name__ == "__main__":
    run()
