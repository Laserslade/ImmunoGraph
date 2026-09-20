"""
Builds Figure C1 from the Gate 1 canonical and diagnostic result files.
Run gate1_recovery.py first.
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
    canon = list(csv.DictReader(open(RESULTS / "chamberland_gate1_validation_checks.csv")))
    rhs_diag = list(csv.DictReader(open(RESULTS / "chamberland_gate1_rhs_state_errors.csv")))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    param_rows = [r for r in canon if r["check_category"] == "parameter"]
    ref_vals = np.array([abs(float(r["reference_value"])) for r in param_rows])
    rec_vals = np.array([abs(float(r["reconstructed_value"])) for r in param_rows])
    mask = (ref_vals > 0) & (rec_vals > 0)
    ax = axes[0, 0]
    ax.scatter(ref_vals[mask], rec_vals[mask], s=8, alpha=0.5, color="#2c5f8a")
    lims = [min(ref_vals[mask].min(), rec_vals[mask].min()), max(ref_vals[mask].max(), rec_vals[mask].max())]
    ax.plot(lims, lims, "r--", lw=1, label="identity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reference implementation value")
    ax.set_ylabel("clean room reconstruction value")
    ax.set_title(f"A. Parameter equivalence (n={len(param_rows)})")
    ax.legend()

    ic_rows = [r for r in canon if r["check_category"] == "initial_condition"]
    ref_vals = np.array([abs(float(r["reference_value"])) for r in ic_rows])
    rec_vals = np.array([abs(float(r["reconstructed_value"])) for r in ic_rows])
    mask = (ref_vals > 0) & (rec_vals > 0)
    ax = axes[0, 1]
    ax.scatter(ref_vals[mask], rec_vals[mask], s=14, alpha=0.6, color="#8a4a2c")
    lims = [min(ref_vals[mask].min(), rec_vals[mask].min()), max(ref_vals[mask].max(), rec_vals[mask].max())]
    ax.plot(lims, lims, "r--", lw=1, label="identity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reference implementation IC value")
    ax.set_ylabel("clean room reconstruction IC value")
    ax.set_title(f"B. Initial condition equivalence (n={len(ic_rows)})")
    ax.legend()

    rel_errs = np.array([float(r["rel_error"]) if r["rel_error"] not in ("", "0") else 1e-300 for r in rhs_diag])
    rel_errs = np.clip(rel_errs, 1e-16, None)
    ax = axes[1, 0]
    ax.hist(np.log10(rel_errs), bins=40, color="#4a8a5f", edgecolor="white")
    ax.axvline(np.log10(1e-8), color="r", linestyle="--", label="tolerance")
    ax.set_xlabel("log10 relative error")
    ax.set_ylabel("count")
    ax.set_title(f"C. RHS equivalence, state level errors (n={len(rhs_diag)})")
    ax.legend()

    ax = axes[1, 1]
    ax.axis("off")
    n_pass = sum(1 for r in canon if r["pass"] == "True")
    n_total = len(canon)
    ax.text(0.5, 0.85, f"{n_pass} / {n_total}", ha="center", fontsize=32, fontweight="bold", color="#2c7a3a")
    ax.text(0.5, 0.68, "prespecified validation checks passed", ha="center", fontsize=12)
    ax.set_title("D. Verification summary")

    plt.suptitle("Figure C1, Chamberland clean room reconstruction, static equivalence", fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(FIGURES / "chamberland_gate1_fidelity.png", bbox_inches="tight")
    print("saved chamberland_gate1_fidelity.png")


if __name__ == "__main__":
    run()
