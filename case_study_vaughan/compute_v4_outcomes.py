"""
Computes the eight preregistered V4 outcomes from the graph cache.
Run build_graph_cache.py first.
"""

import sys
from pathlib import Path
from itertools import permutations
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
import vaughan_reference as vr
import vaughan_degree_utils as du

RESULTS = ROOT / "results"


def cyc3_batch(pres_batch):
    n_samples, n = pres_batch.shape[0], pres_batch.shape[1]
    idxs = list(permutations(range(n), 3))
    counts = np.zeros(n_samples)
    for s in range(n_samples):
        P = pres_batch[s]
        c = 0
        for a, b, cc in idxs:
            if P[b, a] and P[cc, b] and P[a, cc]:
                c += 1
        counts[s] = c
    return counts


def run():
    names = vr.STATE_NAMES
    cache = np.load(RESULTS / "graph_cache.npz", allow_pickle=True)
    m2, il12 = names.index("M2"), names.index("IL12")
    w0, w1, w2, w3, w4 = [cache[f"weight_t{i}"] for i in range(5)]
    p0, p1, p2, p3, p4 = [cache[f"presence_t{i}"] for i in range(5)]

    def flat(w):
        return w.reshape(w.shape[0], -1)

    a, b = flat(w0), flat(w1)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den[den == 0] = 1e-12
    o1 = 1 - num / den

    rank_in_t0 = du.within_sample_rank(du.weighted_in_degree(w0))[:, m2]
    rank_in_t1 = du.within_sample_rank(du.weighted_in_degree(w1))[:, m2]
    o2_in = (rank_in_t0 - rank_in_t1).astype(float)
    rank_out_t0 = du.within_sample_rank(du.weighted_out_degree(w0))[:, m2]
    rank_out_t1 = du.within_sample_rank(du.weighted_out_degree(w1))[:, m2]
    o2_out = (rank_out_t0 - rank_out_t1).astype(float)

    topo_in_t0, topo_in_t4 = du.topology_in_degree(p0)[:, m2], du.topology_in_degree(p4)[:, m2]
    o3_in = (topo_in_t4 - topo_in_t0).astype(float)
    topo_out_m2_t0, topo_out_m2_t4 = du.topology_out_degree(p0)[:, m2], du.topology_out_degree(p4)[:, m2]
    o3_out_m2 = (topo_out_m2_t4 - topo_out_m2_t0).astype(float)
    topo_out_il12_t0, topo_out_il12_t4 = du.topology_out_degree(p0)[:, il12], du.topology_out_degree(p4)[:, il12]
    o3_out_il12 = (topo_out_il12_t0 - topo_out_il12_t4).astype(float)

    canonical_edges = [("IL4", "IL10"), ("M2", "D"), ("M2", "IL10"), ("IL10", "IL4"),
                        ("M2", "IL4"), ("IL4", "M2"), ("IL10", "M2"), ("IL10", "D")]
    sign_lookup = {}
    for src, tgt in canonical_edges:
        j, i = names.index(src), names.index(tgt)
        vals = w2[:, i, j]
        present = np.abs(vals) > 1e-9
        sign_lookup[(src, tgt)] = np.sign(vals[present].sum())
    o4 = np.zeros(w2.shape[0])
    for src, tgt in canonical_edges:
        j, i = names.index(src), names.index(tgt)
        maj = sign_lookup[(src, tgt)]
        ok = (np.abs(w2[:, i, j]) > 1e-9) & (np.sign(w2[:, i, j]) == maj)
        o4 += ok.astype(float)
    o4 /= len(canonical_edges)

    o5 = cyc3_batch(p4) - cyc3_batch(p0)

    outcomes = {"O1_reorg": o1, "O2_IN_M2_rank": o2_in, "O2_OUT_M2_rank": o2_out,
                "O3_IN_M2_topo": o3_in, "O3_OUT_M2_topo": o3_out_m2, "O3_OUT_IL12_topo": o3_out_il12,
                "O4_edge_persist": o4, "O5_motif_change": o5}
    np.savez(RESULTS / "v4_outcomes.npz", **outcomes)
    print("wrote 8 outcome arrays")


if __name__ == "__main__":
    run()
