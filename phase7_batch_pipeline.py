"""
phase7_batch_pipeline.py

Phase 7 of the Neuroimmune Graph Project.

Runs the Phase 6 graph-sequence construction across all 24 saved
perturbation runs (Phase 2's simulation library), not just the single
default-parameter trajectory Phase 5/6 used. This is what turns "one
sequence of graphs" into an ensemble -- the thing Phase 8 (motif
discovery) and Phase 9 (therapy comparison) actually need, since both
require comparing graph structure ACROSS parameter perturbations.

-------------------------------------------------------------------------
IMPORTANT CAVEAT THIS SCRIPT WORKS AROUND (read before changing anything)
-------------------------------------------------------------------------
The 24 runs saved in sims/*.npz (Phase 2) are trajectories of the PURE
19-state Chamberland backbone -- they predate the TBI cytokine module
added in Phase 3. But every sensitivity/graph tool built since Phase 4
(compute_jacobian, sensitivity_at_age, build_graph) operates on the
22-state COMPOSITE model (composite_rhs / STATE_NAMES from
models/composite_backbone.py), which is what the rest of the project
(and the framework-evolution goals in findings log Section 11 -- e.g.
"amyloid toxicity and inflammatory cytokine signaling" as a stated
finding) actually needs.

So this script does NOT reload the saved 19-state .npz trajectories
directly. Instead, for each of the 24 runs it:
  1. Reads that run's parameter overrides from sims/manifest.csv
     (kappa_MproTa, kappa_MhproTa, kappa_TbMpro, kappa_TaManti, d_ABoo,
     AP -- all 6 are confirmed to exist as keys in the composite param
     dict, since they're Chamberland-side parameters the TBI module
     doesn't touch).
  2. RE-RUNS the composite model (Phase 3's run_composite_simulation)
     with those same overrides applied on top of composite
     default_params(), producing a fresh 22-state trajectory.
  3. Runs Phase 6's graph-sequence procedure on that composite
     trajectory.

This means each "run" in this batch is the same 24 sampled parameter
sets from Phase 2, but now expressed through the full composite model
instead of the Chamberland-only one. Re-running is cheap (each
composite solve is <1s) so this isn't a performance concern -- it's a
correctness one, and worth an explicit note in the findings log so
this doesn't get "fixed" later by naively loading the old .npz files
and silently dropping the TBI cytokine states.

-------------------------------------------------------------------------
OUTPUTS
-------------------------------------------------------------------------
- graphs/batch/run_XXXX/graph_ageYY.Y.graphml -- per-run, per-age graphs
  (same format as Phase 6's single-run sequence).
- graphs/batch/run_XXXX/manifest.csv -- per-run manifest (same columns
  as Phase 6's sequence_manifest.csv).
- graphs/batch_manifest.csv -- ONE flat manifest across all runs and all
  ages, with the per-run parameter overrides joined in as extra columns.
  This is the file Phase 8/9/10 should read from directly: it lets you
  ask things like "does top_hub == M_pro correlate with high AP (APOE4
  status)" or "which parameter perturbations produce the earliest hub
  transition" without touching any GraphML file at all.

Run:
    python phase7_batch_pipeline.py
"""

import csv
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DIRS  # noqa: E402
from models.composite_backbone import STATE_NAMES, default_params  # noqa: E402
from phase1_base_model import AGE_END, AGE_START  # noqa: E402
from phase3_composite_model import run_composite_simulation  # noqa: E402
from phase4_local_sensitivity import compute_dydt_scales, sensitivity_at_age  # noqa: E402
from phase5_graph_construction import build_graph  # noqa: E402
from phase6_graph_sequence import (  # noqa: E402
    AGE_STEP,
    build_sequence,
    weighted_in_degree_hub,
)

MANIFEST_PATH = os.path.join(DIRS["sims"], "manifest.csv")
OVERRIDE_PARAM_NAMES = [
    "kappa_MproTa", "kappa_MhproTa", "kappa_TbMpro",
    "kappa_TaManti", "d_ABoo", "AP",
]


def load_run_overrides(manifest_path=MANIFEST_PATH):
    """Read sims/manifest.csv and return a list of dicts, one per run,
    with run_id and the 6 perturbed parameter values as floats.
    Skips any run marked unsuccessful (none currently are, but this is
    the correct guard if the perturbation library is ever regenerated
    with a wider, riskier parameter range)."""
    runs = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["success"] != "True":
                continue
            overrides = {name: float(row[name]) for name in OVERRIDE_PARAM_NAMES}
            runs.append({"run_id": int(row["run_id"]), "overrides": overrides})
    return runs


def run_composite_with_overrides(overrides):
    """Build composite default_params(), apply this run's overrides on
    top, and solve the composite model -- see module docstring for why
    this re-solves rather than reloading the saved 19-state .npz."""
    params = default_params()
    params.update(overrides)
    solution = run_composite_simulation(params)
    return params, solution


def process_run(run_id, overrides, ages, batch_dir):
    """Run the full graph-sequence procedure for one perturbation run.
    Returns the list of per-age summary dicts (same shape as Phase 6's
    manifest rows), with the run's parameter overrides merged in."""
    params, solution = run_composite_with_overrides(overrides)
    scales = compute_dydt_scales(solution, params)
    sequence = build_sequence(solution, params, scales, ages)

    run_dir = os.path.join(batch_dir, f"run_{run_id:04d}")
    os.makedirs(run_dir, exist_ok=True)

    rows = []
    for age, G in sequence:
        filename = f"graph_age{age:05.1f}.graphml".replace(" ", "0")
        nx.write_graphml(G, os.path.join(run_dir, filename))

        hub, hub_weight = weighted_in_degree_hub(G)
        row = {
            "run_id": run_id,
            "age": age,
            "filename": filename,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "density": nx.density(G),
            "top_hub": hub,
            "top_hub_weighted_in_degree": hub_weight,
        }
        row.update({f"param__{k}": v for k, v in overrides.items()})
        rows.append(row)

    # Per-run manifest, same shape as Phase 6's single-run manifest.
    with open(os.path.join(run_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    print("Loading perturbation run parameters from sims/manifest.csv...")
    runs = load_run_overrides()
    print(f"Found {len(runs)} successful runs to process.")

    ages = np.arange(AGE_START, AGE_END + AGE_STEP / 2, AGE_STEP)
    batch_dir = os.path.join(DIRS["graphs"], "batch")
    os.makedirs(batch_dir, exist_ok=True)

    all_rows = []
    for run in runs:
        run_id = run["run_id"]
        print(f"  Processing run {run_id:2d} "
              f"(AP={run['overrides']['AP']:.0f}, "
              f"d_ABoo={run['overrides']['d_ABoo']:.3f})...")
        rows = process_run(run_id, run["overrides"], ages, batch_dir)
        all_rows.extend(rows)

    batch_manifest_path = os.path.join(DIRS["graphs"], "batch_manifest.csv")
    with open(batch_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} graph summaries "
          f"({len(runs)} runs x {len(ages)} ages) to {batch_manifest_path}")
    print(f"Per-run graphs and manifests saved under {batch_dir}/run_XXXX/")

    # Quick sanity check: how often does each node show up as the top
    # hub, across the whole ensemble? A useful first ensemble-level
    # signal, and a preview of what Phase 10's module-concentration
    # finding (Section 11, Finding #2) will need to compute properly.
    hub_counts = {}
    for row in all_rows:
        hub_counts[row["top_hub"]] = hub_counts.get(row["top_hub"], 0) + 1

    print("\nTop-hub frequency across the full ensemble "
          f"({len(all_rows)} graphs):")
    for hub, count in sorted(hub_counts.items(), key=lambda kv: -kv[1]):
        pct = 100 * count / len(all_rows)
        print(f"  {hub:10s} {count:4d}  ({pct:4.1f}%)")

    print("\nPhase 7 complete.")


if __name__ == "__main__":
    main()
