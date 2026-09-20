"""
Builds the graph snapshot cache and denominator cache from the frozen
Vaughan ensemble. Downstream scripts in this folder depend on this output.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
import vaughan_reference as vr
import vaughan_immunograph_ensemble as vie

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
ENSEMBLE_DIR = ROOT / "frozen_ensemble"


def run():
    data = np.load(ENSEMBLE_DIR / "vaughan_exploratory_ensemble_v1_raw.npz", allow_pickle=True)
    param_names = list(data["param_names"])
    param_values = data["param_values"]
    initial_states = data["initial_states"]
    acceptance = data["acceptance_status"]
    accepted_idx = np.where(acceptance == "accepted")[0]

    n_states = vr.N_STATES
    times = vie.SNAPSHOT_TIMES
    n_t = len(times)

    all_presence = {ti: [] for ti in range(n_t)}
    all_weight = {ti: [] for ti in range(n_t)}
    scales_all = np.zeros((len(accepted_idx), n_states))

    for k, idx in enumerate(accepted_idx):
        params = dict(zip(param_names, param_values[idx]))
        y0 = initial_states[idx]
        snaps = vie.run_one_sample(params, y0)
        if snaps is None:
            continue
        for ti, (t, present, weight) in enumerate(snaps):
            all_presence[ti].append(present)
            all_weight[ti].append(weight)

        sol = solve_ivp(vr.vaughan_rhs, (0, 120), y0, method="BDF", args=(params,),
                         t_eval=np.linspace(0, 120, 241), atol=1e-8, rtol=1e-8)
        scales_all[k] = vie.compute_dydt_scales(sol, params)

    np.savez_compressed(RESULTS / "graph_cache.npz",
                         **{f"presence_t{ti}": np.array(all_presence[ti]) for ti in range(n_t)},
                         **{f"weight_t{ti}": np.array(all_weight[ti]) for ti in range(n_t)},
                         names=vr.STATE_NAMES, times=times)

    d = np.maximum(scales_all, 1e-8)
    np.save(RESULTS / "denominators.npy", d)
    print(f"cached graphs and denominators for {len(all_presence[0])} accepted samples")


if __name__ == "__main__":
    run()
