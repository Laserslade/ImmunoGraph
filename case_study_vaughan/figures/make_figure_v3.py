"""
Builds Figure V3 from the normalization audit result files.
Run normalization_audit.py first.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def run():
    denom = list(csv.DictReader(open(RESULTS / "normalization_audit_denominators.csv")))
    assoc = list(csv.DictReader(open(RESULTS / "normalization_audit_association.csv")))
    cache = np.load(RESULTS / "graph_cache.npz", allow_pickle=True)

    names = sorted(set(r["node"] for r in denom), key=lambda n: ["M1", "M2", "IL1", "IL12", "IL10", "IL4", "D"].index(n))
    times = sorted(set(float(r["snapshot_h"]) for r in denom))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    heat = np.zeros((len(names), len(times)))
    for r in denom:
        i = names.index(r["node"])
        j = times.index(float(r["snapshot_h"]))
        heat[i, j] = float(r["frac_within_100x_floor"]) * 100
    im = ax.imshow(heat, aspect="auto", cmap="Reds", vmin=0, vmax=20)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels([f"{int(t)}h" for t in times])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    plt.colorbar(im, ax=ax, label="pct samples within 100x floor")
    for i in range(len(names)):
        for j in range(len(times)):
            ax.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if heat[i, j] > 10 else "black")
    ax.set_title("A. Denominator floor proximity by node and snapshot")

    ax = axes[0, 1]
    for ti, t in enumerate(times):
        w = cache[f"weight_t{ti}"]
        in_deg = np.abs(w).sum(axis=2).sum(axis=1)
        order = np.argsort(-in_deg)
        cum = np.cumsum(in_deg[order]) / in_deg.sum()
        ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum * 100, label=f"t={int(t)}h", alpha=0.8)
    ax.axhline(50, color="gray", ls=":")
    ax.axhline(90, color="gray", ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("pct of samples ranked by total weighted degree")
    ax.set_ylabel("cumulative pct of weighted degree mass")
    ax.set_title("B. Weighted degree mass concentration, all five snapshots")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    heat2 = np.zeros((len(names), len(times)))
    for r in assoc:
        i = names.index(r["node"])
        j = times.index(float(r["snapshot_h"]))
        heat2[i, j] = float(r["spearman_rho_logd_vs_indegree"])
    im2 = ax.imshow(heat2, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels([f"{int(t)}h" for t in times])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    plt.colorbar(im2, ax=ax, label="spearman rho")
    for i in range(len(names)):
        for j in range(len(times)):
            ax.text(j, i, f"{heat2[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("C. Association, log denominator versus weighted in degree")

    ax = axes[1, 1]
    m1_i, m2_i = names.index("M1"), names.index("M2")
    means_m1, medians_m1, means_m2, medians_m2 = [], [], [], []
    for ti, t in enumerate(times):
        w = cache[f"weight_t{ti}"]
        in_deg = np.abs(w).sum(axis=2)
        means_m1.append(in_deg[:, m1_i].mean())
        medians_m1.append(np.median(in_deg[:, m1_i]))
        means_m2.append(in_deg[:, m2_i].mean())
        medians_m2.append(np.median(in_deg[:, m2_i]))
    ax.plot(times, means_m1, "o-", color="#c0392b", label="M1 mean")
    ax.plot(times, medians_m1, "o--", color="#e67e22", label="M1 median")
    ax.plot(times, means_m2, "s-", color="#2c5f8a", label="M2 mean")
    ax.plot(times, medians_m2, "s--", color="#5dade2", label="M2 median")
    ax.set_yscale("log")
    ax.set_xlabel("time h")
    ax.set_ylabel("weighted in degree")
    ax.set_title("D. Mean versus median weighted in degree, M1 and M2")
    ax.legend(fontsize=7)

    plt.suptitle("Figure V3, normalization stability and failure mode", fontsize=11, y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES / "vaughan_normalization_audit.png", bbox_inches="tight")
    print("saved vaughan_normalization_audit.png")


if __name__ == "__main__":
    run()
