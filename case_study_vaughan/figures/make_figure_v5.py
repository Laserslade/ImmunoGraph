"""
Builds Figure V5 from the complete 360 test V4 result file.
Run v4_full_analysis.py first.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 8, "figure.dpi": 150})


def run():
    results = list(csv.DictReader(open(RESULTS / "vaughan_v4_all_360_tests.csv")))
    param_order = sorted(set(r["parameter"] for r in results),
                          key=lambda p: (0 if next(r["provenance"] for r in results if r["parameter"] == p) == "partial_range" else 1, p))
    outcome_order = sorted(set(r["outcome"] for r in results))

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1])

    ax = fig.add_subplot(gs[0, :])
    heat = np.zeros((len(param_order), len(outcome_order)))
    sig = np.zeros((len(param_order), len(outcome_order)), dtype=bool)
    for r in results:
        i = param_order.index(r["parameter"])
        j = outcome_order.index(r["outcome"])
        heat[i, j] = float(r["rho"])
        sig[i, j] = r["significant"] == "True"
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-0.45, vmax=0.45)
    ax.set_xticks(range(len(outcome_order)))
    ax.set_xticklabels(outcome_order, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(param_order)))
    ax.set_yticklabels(param_order, fontsize=7)
    n_partial = sum(1 for p in param_order if next(r["provenance"] for r in results if r["parameter"] == p) == "partial_range")
    ax.axhline(n_partial - 0.5, color="black", lw=2)
    for i in range(len(param_order)):
        for j in range(len(outcome_order)):
            if sig[i, j]:
                ax.plot(j, i, marker="*", color="black", markersize=4)
    plt.colorbar(im, ax=ax, label="spearman rho", fraction=0.025, pad=0.02)
    fig.text(0.045, 0.885, "published support, 9", rotation=90, va="center", fontsize=9, fontweight="bold")
    fig.text(0.045, 0.68, "definition only, 36", rotation=90, va="center", fontsize=9)
    ax.set_title("A. Complete 45 by 8 spearman correlation map, stars are significant at joint BH FDR q under 0.05")

    ax = fig.add_subplot(gs[1, 0])
    n_sig = sum(1 for r in results if r["significant"] == "True")
    n_partial_tests = sum(1 for r in results if r["provenance"] == "partial_range")
    n_partial_sig = sum(1 for r in results if r["provenance"] == "partial_range" and r["significant"] == "True")
    n_def_tests = len(results) - n_partial_tests
    n_def_sig = n_sig - n_partial_sig
    rates = [n_partial_sig / n_partial_tests * 100, n_def_sig / n_def_tests * 100]
    bars = ax.bar(["published support", "definition only"], rates, color=["#2c7a3a", "#8a8a8a"])
    ax.bar_label(bars, fmt="%.1f pct")
    ax.set_ylabel("pct tests significant")
    ax.set_title("B. Enrichment")
    ax.set_ylim(0, 50)

    ax = fig.add_subplot(gs[1, 1])
    data = np.load(ROOT / "frozen_ensemble" / "vaughan_exploratory_ensemble_v1_raw.npz", allow_pickle=True)
    pnames = list(data["param_names"])
    pvals = data["param_values"]
    acc = data["acceptance_status"]
    accepted_idx = np.where(acc == "accepted")[0]
    outcomes = dict(np.load(RESULTS / "v4_outcomes.npz"))
    examples = [("mu_n12", "O3_OUT_IL12_topo", "#2c7a3a"), ("mu_M1", "O1_reorg", "#2c5f8a"), ("xn", "O3_IN_M2_topo", "#c0392b")]
    for k, (pname, oname, color) in enumerate(examples):
        pv = pvals[accepted_idx, pnames.index(pname)]
        ov = outcomes[oname]
        rp, ro = rankdata(pv) / len(pv), rankdata(ov) / len(ov)
        ax.scatter(rp + k * 1.15, ro, s=2, alpha=0.12, color=color)
        ax.text(k * 1.15 + 0.5, 1.08, f"{pname} vs {oname}", ha="center", fontsize=6.5, color=color)
    ax.set_xticks([])
    ax.set_ylabel("outcome percentile rank")
    ax.set_title("C. Representative rank relationships")
    ax.set_ylim(0, 1.2)

    plt.suptitle("Figure V5, V4 published parameter consistency, complete 360 test statistics", fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(FIGURES / "vaughan_v4_parameter_consistency.png", bbox_inches="tight")
    print("saved vaughan_v4_parameter_consistency.png")


if __name__ == "__main__":
    run()
