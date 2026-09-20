"""
Builds Figure V6 from the V5 result files. Run v5_helmy_comparison.py first.
Panel D uses an explicit per patient per cytokine lookup, not string slicing,
to avoid mislabeling which values are censored.
"""

from pathlib import Path
import csv
from collections import Counter
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
    helmy = list(csv.DictReader(open(RESULTS / "vaughan_v5_helmy_results.csv")))
    orderings = list(csv.DictReader(open(RESULTS / "vaughan_v5_ensemble_orderings.csv")))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0, 0]
    cyto_colors = {"IL1β": "#2c5f8a", "IL12p70": "#e67e22", "IL10": "#c0392b"}
    y = 0
    yticks, yticklabels = [], []
    for cyto in ["IL1β", "IL12p70", "IL10"]:
        rows = [r for r in helmy if r["cytokine"] == cyto]
        vals = [float(r["peak_hours"]) for r in rows]
        cens = [r["left_censored"] == "True" for r in rows]
        for v, c in zip(vals, cens):
            ax.scatter([v], [y], color=cyto_colors[cyto], marker="<" if c else "o", s=40, alpha=0.7)
            y -= 1
        med = np.median(vals)
        ax.axvline(med, color=cyto_colors[cyto], linestyle="--", alpha=0.4)
        yticks.append(y + len(vals) / 2)
        yticklabels.append(f"{cyto}\nn={len(vals)} median={med:.0f}h")
        y -= 1.5
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("peak time hours post injury, triangle is left censored")
    ax.set_title("A. Helmy patient level cytokine peak timing")

    ax = axes[0, 1]
    counts = Counter(r["ordering"] for r in orderings)
    n_total = len(orderings)
    order_labels = sorted(counts, key=lambda k: -counts[k])
    freqs = [counts[k] / n_total * 100 for k in order_labels]
    colors = ["#2c7a3a" if lbl == "IL1<IL12<IL10" else "#8a8a8a" for lbl in order_labels]
    bars = ax.bar(range(len(order_labels)), freqs, color=colors)
    ax.set_xticks(range(len(order_labels)))
    ax.set_xticklabels([l.replace("<", " < ") for l in order_labels], rotation=30, ha="right", fontsize=7.5)
    ax.axhline(100 / 6, color="red", linestyle=":", label="uniform order reference")
    ax.bar_label(bars, fmt="%.1f pct", fontsize=7)
    ax.set_ylabel("pct of accepted ensemble")
    ax.set_title("B. Ensemble predicted orderings, all six possible")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    taus = [float(r["kendall_tau"]) for r in orderings]
    tau_counts = Counter(taus)
    tau_vals = sorted(tau_counts)
    ax.bar([str(round(t, 2)) for t in tau_vals], [tau_counts[t] / n_total * 100 for t in tau_vals], color="#2c5f8a")
    ax.set_ylabel("pct of ensemble")
    ax.set_xlabel("kendall tau versus Helmy median ordering")
    ax.set_title(f"C. Kendall tau distribution, mean = {np.mean(taus):.3f}")

    ax = axes[1, 1]
    ax.axis("off")
    paired = {
        3: {"IL1β": (30, True), "IL12p70": (30, True), "IL10": (108, False)},
        11: {"IL1β": (138, False), "IL12p70": (126, False), "IL10": (144, False)},
        12: {"IL1β": (20, True), "IL12p70": (134, False), "IL10": (134, False)},
    }
    ax.text(0.5, 0.98, "D. Same patient paired robustness check", ha="center", fontsize=10, fontweight="bold")
    y = 0.85
    for pid, vals in paired.items():
        order = sorted(vals, key=lambda k: vals[k][0])
        order_str = " < ".join(f"{k}({vals[k][0]}{'*' if vals[k][1] else ''})" for k in order)
        a, b, c = vals["IL1β"][0], vals["IL12p70"][0], vals["IL10"][0]
        if a <= b < c and a < b:
            cls, color = "consistent", "#2c7a3a"
        elif a <= b == c:
            cls, color = "tied IL12 equals IL10", "#b8860b"
        elif a == b < c:
            cls, color = "tied IL1 equals IL12", "#b8860b"
        else:
            cls, color = "reversed", "#c0392b"
        ax.text(0.5, y, f"patient {pid}: {order_str}  [{cls}]", ha="center", fontsize=9.5, color=color)
        y -= 0.15
    ax.text(0.5, y - 0.15, "star marks a left censored value, upper bound on true peak", ha="center", fontsize=7,
            style="italic", color="gray")

    plt.suptitle("Figure V6, held out human temporal ordering consistency", fontsize=11, y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES / "vaughan_v5_helmy_validation.png", bbox_inches="tight")
    print("saved vaughan_v5_helmy_validation.png")


if __name__ == "__main__":
    run()
