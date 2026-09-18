"""
phase9_therapy_exploration.py

Phase 9 of the Neuroimmune Graph Project: therapy-combination exploration.

Reuses every piece of infrastructure built in Phases 3-8 -- this phase
adds no new machinery, only new *parameter regimes* to run through the
existing pipeline. That was the whole point of building it this way
(findings log, Section 5): once graph construction, sequencing, and
module detection exist, "explore an intervention" is just "solve the
composite model under different parameters and read off the same
statistics."

-------------------------------------------------------------------------
WHAT COUNTS AS A "THERAPY" HERE
-------------------------------------------------------------------------
Each scenario is a constant-parameter regime applied across the full
simulated lifespan (age 30-80) -- i.e. "what if this pathway were
pharmacologically dampened/boosted from adulthood onward," not a
time-varying dose applied at a specific intervention age. Modeling
intervention *timing* (e.g. "start treatment at age 60") would require
piecewise-in-time parameters, which the composite model doesn't support
yet -- flagged here as an explicit scope boundary, not an oversight (see
findings log, Section 5's original therapy-exploration scope: "timing"
and "sequencing" were always the more expensive follow-on, not the MVP).

Four scenarios, chosen to map directly onto real drug mechanism classes
and the composite model's own known dominant pathway (see Phase 8):

  - baseline        : composite_backbone.default_params(), unchanged.
  - TNF_inhibition   : kappa_MproTa and kappa_MhproTa (TNF-alpha
                       production by pro-inflammatory microglia/
                       macrophages) reduced to 30% of default -- the
                       mechanism class of real anti-TNF biologics
                       (e.g. infliximab, etanercept).
  - IL10_boost       : kappa_MantiI10 and kappa_MhantiI10 (IL-10
                       production by anti-inflammatory microglia/
                       macrophages) increased to 300% of default -- an
                       anti-inflammatory-cytokine-boosting strategy.
  - combo            : both of the above together -- directly tests the
                       project's central synergy question (Q3 in the
                       original proposal): does combining a suppressive
                       and a boosting intervention shift the system
                       further than either alone?

Each scenario is additionally run under both AP=0 and AP=1 (APOE4
status), since Phase 7 found APOE4 status already associates with a
different hub-transition profile -- worth checking whether a therapy's
effect size depends on genetic background, not just whether it "works"
in one demographic regime.

-------------------------------------------------------------------------
HOW "SHIFTED THE SYSTEM" IS MEASURED (Phase 9's definition of done)
-------------------------------------------------------------------------
For each scenario, at age 80 (end of simulated lifespan, where disease
burden is most developed):
  1. Top hub (weighted in-degree) -- same metric as Phases 6-8.
  2. Dominant module (module containing the top hub) -- reuses Phase 8's
     detect_modules/module_influence_share directly (imported, not
     reimplemented).
  3. Distance from baseline -- Euclidean distance between this
     scenario's flattened signed weight vector and the baseline
     scenario's, in the SAME AP group (comparing AP=1 treated to AP=0
     baseline would conflate "genetic background changed" with "therapy
     worked"; comparing within AP group isolates the therapy's effect).

This directly supports statements of the form "this combination shifted
the system toward pattern X instead of Y" -- X and Y being named by
their dominant-module membership and top hub, with a distance number
quantifying how far the shift went.

Run:
    python phase9_therapy_exploration.py
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DIRS  # noqa: E402
from models.composite_backbone import STATE_NAMES, default_params  # noqa: E402
from phase1_base_model import AGE_END, AGE_START  # noqa: E402
from phase3_composite_model import run_composite_simulation  # noqa: E402
from phase4_local_sensitivity import compute_dydt_scales, sensitivity_at_age  # noqa: E402
from phase5_graph_construction import build_graph  # noqa: E402
from phase6_graph_sequence import AGE_STEP, build_sequence, weighted_in_degree_hub  # noqa: E402
from phase8_motif_discovery import detect_modules, graph_to_feature_vector, module_influence_share  # noqa: E402

SCENARIOS = [
    {
        "name": "baseline",
        "description": "No intervention -- default composite parameters.",
        "overrides": {},
    },
    {
        "name": "TNF_inhibition",
        "description": "TNF-alpha production reduced to 30% of default "
                        "(mechanism class: anti-TNF biologics).",
        "overrides": {"kappa_MproTa": 0.3, "kappa_MhproTa": 0.3},
    },
    {
        "name": "IL10_boost",
        "description": "IL-10 production increased to 300% of default "
                        "(anti-inflammatory-cytokine-boosting strategy).",
        "overrides": {"kappa_MantiI10": 3.0, "kappa_MhantiI10": 3.0},
    },
    {
        "name": "combo",
        "description": "TNF inhibition + IL-10 boost together.",
        "overrides": {
            "kappa_MproTa": 0.3, "kappa_MhproTa": 0.3,
            "kappa_MantiI10": 3.0, "kappa_MhantiI10": 3.0,
        },
    },
]

AP_VALUES = [0.0, 1.0]
EVAL_AGE = 80.0  # end-of-lifespan snapshot used for the headline comparison


def build_scenario_params(overrides, ap):
    """Apply multiplicative overrides on top of composite default_params(),
    plus the APOE4 status switch."""
    base = default_params()
    params = dict(base)
    for name, factor in overrides.items():
        params[name] = base[name] * factor
    params["AP"] = ap
    return params


def run_scenario(name, overrides, ap, ages):
    """Solve the composite model for one scenario, build its full graph
    sequence, and save it -- same storage pattern as Phase 6/7."""
    params = build_scenario_params(overrides, ap)
    solution = run_composite_simulation(params)
    scales = compute_dydt_scales(solution, params)
    sequence = build_sequence(solution, params, scales, ages)

    run_dir = os.path.join(DIRS["graphs"], "therapy", f"{name}_AP{int(ap)}")
    os.makedirs(run_dir, exist_ok=True)

    rows = []
    graphs_by_age = {}
    for age, G in sequence:
        filename = f"graph_age{age:05.1f}.graphml"
        nx.write_graphml(G, os.path.join(run_dir, filename))
        graphs_by_age[round(age, 1)] = G

        hub, hub_weight = weighted_in_degree_hub(G)
        rows.append({
            "scenario": name, "AP": ap, "age": age,
            "n_edges": G.number_of_edges(), "top_hub": hub,
            "top_hub_weighted_in_degree": hub_weight,
        })

    return rows, graphs_by_age


def dominant_module_at(G):
    """Top hub + the module (from Phase 8's detector) that contains it,
    plus that module's influence share -- the same 'dominant module'
    definition used throughout Phase 8."""
    hub, _ = weighted_in_degree_hub(G)
    total_abs_weight = sum(abs(d["weight"]) for _, _, d in G.edges(data=True))
    modules = detect_modules(G)
    for module in modules:
        if hub in module:
            share, pos_fraction, _ = module_influence_share(G, module, total_abs_weight)
            return hub, sorted(module), share
    return hub, [], 0.0


def main():
    ages = np.arange(AGE_START, AGE_END + AGE_STEP / 2, AGE_STEP)

    print(f"Running {len(SCENARIOS)} scenarios x {len(AP_VALUES)} AP groups "
          f"= {len(SCENARIOS) * len(AP_VALUES)} full lifespan simulations...")

    all_rows = []
    end_state_graphs = {}  # (scenario, ap) -> graph at EVAL_AGE

    for scenario in SCENARIOS:
        for ap in AP_VALUES:
            print(f"  {scenario['name']:15s} AP={int(ap)} ...")
            rows, graphs_by_age = run_scenario(
                scenario["name"], scenario["overrides"], ap, ages
            )
            all_rows.extend(rows)
            end_state_graphs[(scenario["name"], ap)] = graphs_by_age[round(EVAL_AGE, 1)]

    manifest_path = os.path.join(DIRS["graphs"], "therapy_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {manifest_path}")

    # --- Headline comparison: dominant module + hub + distance from
    # baseline, at age 80, per AP group. This is Phase 9's definition of
    # done: a stated "shifted toward X instead of Y" comparison.
    print(f"\n=== Age {EVAL_AGE:.0f} comparison, by APOE4 status ===")

    summary_rows = []
    for ap in AP_VALUES:
        baseline_G = end_state_graphs[("baseline", ap)]
        baseline_vec = graph_to_feature_vector(baseline_G)
        baseline_hub, baseline_module, baseline_share = dominant_module_at(baseline_G)

        print(f"\n--- AP={int(ap)} ---")
        print(f"  baseline: top hub = {baseline_hub}, dominant module = "
              f"{baseline_module} ({baseline_share:.1%} influence share)")

        for scenario in SCENARIOS:
            G = end_state_graphs[(scenario["name"], ap)]
            hub, module, share = dominant_module_at(G)
            vec = graph_to_feature_vector(G)
            distance = float(np.linalg.norm(vec - baseline_vec))

            print(f"  {scenario['name']:15s}: top hub = {hub:8s}, dominant module = "
                  f"{module}, ({share:.1%} share), distance from baseline = {distance:.3f}")

            summary_rows.append({
                "AP": ap, "scenario": scenario["name"],
                "top_hub": hub, "dominant_module": ";".join(module),
                "influence_share": share, "distance_from_baseline": distance,
            })

    summary_path = os.path.join(DIRS["graphs"], "therapy_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved headline summary to {summary_path}")

    plot_summary(summary_rows)

    print("\nPhase 9 complete.")


def plot_summary(summary_rows):
    """Bar chart: distance-from-baseline per scenario, grouped by AP
    status -- the single figure that answers 'did this intervention
    shift the system, and by how much, and does that depend on APOE4
    status.'"""
    scenarios = [s["name"] for s in SCENARIOS if s["name"] != "baseline"]
    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, ap in enumerate(AP_VALUES):
        distances = [
            next(r["distance_from_baseline"] for r in summary_rows
                 if r["scenario"] == s and r["AP"] == ap)
            for s in scenarios
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, distances, width, label=f"AP={int(ap)}")
        for bar, dist in zip(bars, distances):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{dist:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(f"distance from baseline graph (age {EVAL_AGE:.0f})")
    ax.set_title("Phase 9: how far each intervention shifts the network state,\n"
                  "relative to its own AP-matched baseline")
    ax.legend(title="APOE4 status")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    save_path = os.path.join(DIRS["results"], "phase9_therapy_distances.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved summary plot to {save_path}")


if __name__ == "__main__":
    main()
