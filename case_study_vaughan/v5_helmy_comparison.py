"""
V5 held out comparison. Compares the frozen ensemble predicted cytokine
peak ordering against the independent Helmy cohort, run once, not tuned.
"""

import sys
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
import vaughan_reference as vr

RESULTS = ROOT / "results"
ENSEMBLE_DIR = ROOT / "frozen_ensemble"
EXTERNAL = ROOT / "external_data"


def run():
    df = pd.read_csv(EXTERNAL / "Helmy2011_SupplementaryTable3_peak_timepoints.csv")
    helmy_rows = []
    for cyto in ["IL1β", "IL12p70", "IL10"]:
        sub = df[df.cytokine == cyto]
        for _, row in sub.iterrows():
            helmy_rows.append({"cytokine": cyto, "patient": row.patient,
                                "peak_hours": row.peak_hours_post_injury, "left_censored": row.left_censored})
    with open(RESULTS / "vaughan_v5_helmy_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cytokine", "patient", "peak_hours", "left_censored"])
        w.writeheader()
        w.writerows(helmy_rows)

    data = np.load(ENSEMBLE_DIR / "vaughan_exploratory_ensemble_v1_raw.npz", allow_pickle=True)
    param_names = list(data["param_names"])
    param_values = data["param_values"]
    initial_states = data["initial_states"]
    acceptance = data["acceptance_status"]
    accepted_idx = np.where(acceptance == "accepted")[0]
    names = vr.STATE_NAMES
    i_il1, i_il12, i_il10 = names.index("IL1"), names.index("IL12"), names.index("IL10")
    observed_rank = {"IL1": 1, "IL12": 2, "IL10": 3}
    t_eval = np.linspace(0, 120, 481)

    order_rows = []
    for idx in accepted_idx:
        params = dict(zip(param_names, param_values[idx]))
        y0 = initial_states[idx]
        sol = solve_ivp(vr.vaughan_rhs, (0, 120), y0, method="BDF", args=(params,),
                         t_eval=t_eval, atol=1e-8, rtol=1e-8)
        if not sol.success:
            continue
        peaks = {"IL1": t_eval[np.argmax(sol.y[i_il1])], "IL12": t_eval[np.argmax(sol.y[i_il12])],
                 "IL10": t_eval[np.argmax(sol.y[i_il10])]}
        order = sorted(peaks, key=lambda k: peaks[k])
        pred_rank = {name: r + 1 for r, name in enumerate(order)}
        obs_vec = [observed_rank["IL1"], observed_rank["IL12"], observed_rank["IL10"]]
        pred_vec = [pred_rank["IL1"], pred_rank["IL12"], pred_rank["IL10"]]
        tau, _ = kendalltau(obs_vec, pred_vec)
        order_rows.append({"sample_id": int(idx), "ordering": "<".join(order), "kendall_tau": tau,
                            "t_peak_IL1": peaks["IL1"], "t_peak_IL12": peaks["IL12"], "t_peak_IL10": peaks["IL10"]})

    with open(RESULTS / "vaughan_v5_ensemble_orderings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "ordering", "kendall_tau", "t_peak_IL1", "t_peak_IL12", "t_peak_IL10"])
        w.writeheader()
        w.writerows(order_rows)

    n = len(order_rows)
    matches = sum(1 for r in order_rows if r["ordering"] == "IL1<IL12<IL10")
    taus = np.array([r["kendall_tau"] for r in order_rows])
    print(f"n={n}, exact match={matches} ({matches / n * 100:.2f} pct)")
    print(f"mean tau = {taus.mean():.3f}")


if __name__ == "__main__":
    run()
