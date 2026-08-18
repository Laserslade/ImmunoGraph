"""
phase6_graph_sequence.py

Phase 6 of the Neuroimmune Graph Project.

Extends Phase 5 from a single graph at one age to a SEQUENCE of graphs
across the full simulated lifespan (age 30-80) -- this is the "dynamic"
half of "dynamic functional interaction graphs" (README, top line).

Design choices made here, and why:

- One graph per integer age (51 graphs, ages 30..80 inclusive) rather
  than one per Phase 1/3's 500 dense solver-output timepoints. The
  Jacobian/sensitivity computation is cheap per point, but 500 GraphML
  files is unnecessary clutter for what a yearly-resolution biological
  narrative needs; if finer resolution turns out to matter later (e.g.
  for pinpointing a transition to within a fraction of a year), this
  step size is a one-line change (AGE_STEP below).

- Every graph is saved individually to graphs/sequence/ AND a manifest
  CSV (graphs/sequence_manifest.csv) records, per age: filename, node
  count, edge count, density, and the top-1 hub node by weighted
  in-degree. The manifest exists specifically so Phase 8/9/10 (motif
  discovery, longitudinal hub-shift findings, transition detection --
  see findings log, Section 11) can scan the sequence's summary
  statistics WITHOUT reloading and re-parsing 51 GraphML files just to
  answer "did the hub change between age 42 and 48."  This mirrors the
  sims/manifest.csv pattern already used for the 24 perturbation runs
  (Phase 2) -- the project's established convention for "one manifest
  row per saved artifact."

- Edge weight sign is preserved in each saved graph (as in Phase 5).
  "Weighted in-degree" for hub-ranking purposes here sums |weight| of
  incoming edges (a suppressive and a driving edge both count as
  "influence over" the target), not signed sum, which would let
  positive and negative edges cancel out and hide an actively-suppressed
  hub.

Run:
    python phase6_graph_sequence.py
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

AGE_STEP = 1.0  # yearly resolution; see module docstring for why


def weighted_in_degree_hub(G):
    """Return (hub_node, weighted_in_degree) using sum of |weight| over
    incoming edges, so a suppressive edge counts as influence too (see
    module docstring for why signed sum would be misleading here)."""
    in_weight = {n: 0.0 for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        in_weight[v] += abs(data["weight"])

    hub = max(in_weight, key=in_weight.get)
    return hub, in_weight[hub]


def build_sequence(solution, params, scales, ages):
    """Build one graph per age in `ages`. Returns a list of
    (age, G) tuples in age order."""
    sequence = []
    for age in ages:
        t, y, J, S = sensitivity_at_age(solution, age, params, scales)
        G = build_graph(S, STATE_NAMES)
        sequence.append((t, G))
    return sequence


def save_sequence(sequence, seq_dir, manifest_path):
    """Save each graph to its own GraphML file in seq_dir, and write a
    summary manifest CSV (age, filename, nodes, edges, density,
    top_hub, top_hub_weighted_in_degree) so downstream phases can scan
    the sequence's shape without reloading every file."""
    os.makedirs(seq_dir, exist_ok=True)

    rows = []
    for age, G in sequence:
        filename = f"graph_age{age:05.1f}.graphml".replace(" ", "0")
        path = os.path.join(seq_dir, filename)
        nx.write_graphml(G, path)

        hub, hub_weight = weighted_in_degree_hub(G)
        rows.append({
            "age": age,
            "filename": filename,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "density": nx.density(G),
            "top_hub": hub,
            "top_hub_weighted_in_degree": hub_weight,
        })

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} graphs to {seq_dir}")
    print(f"Saved sequence manifest to {manifest_path}")
    return rows


def plot_sequence_summary(rows, save_path):
    """Plot edge count and top-hub identity over age, as the Phase 6
    'definition of done' visual check: does the sequence actually show
    dynamics (changing edge count / changing hub), or is it flat (which
    would suggest a bug, since a genuinely static graph across 50
    simulated years would be biologically implausible for this model)."""
    ages = [r["age"] for r in rows]
    n_edges = [r["n_edges"] for r in rows]
    hubs = [r["top_hub"] for r in rows]

    unique_hubs = sorted(set(hubs))
    hub_to_y = {h: i for i, h in enumerate(unique_hubs)}
    hub_y = [hub_to_y[h] for h in hubs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(ages, n_edges, color="#2c3e50", linewidth=2)
    ax1.set_ylabel("edges in graph")
    ax1.set_title("Phase 6: graph sequence summary across simulated lifespan")
    ax1.grid(alpha=0.3)

    ax2.step(ages, hub_y, where="mid", color="#c0392b", linewidth=2)
    ax2.set_yticks(range(len(unique_hubs)))
    ax2.set_yticklabels(unique_hubs, fontsize=9)
    ax2.set_xlabel("age")
    ax2.set_ylabel("top hub node\n(by weighted in-degree)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved sequence summary plot to {save_path}")


def main():
    print(f"Solving composite model and building a graph sequence "
          f"(age {AGE_START:.0f}-{AGE_END:.0f}, step {AGE_STEP} yr)...")
    params = default_params()
    solution = run_composite_simulation(params)
    scales = compute_dydt_scales(solution, params)

    ages = np.arange(AGE_START, AGE_END + AGE_STEP / 2, AGE_STEP)
    sequence = build_sequence(solution, params, scales, ages)

    seq_dir = os.path.join(DIRS["graphs"], "sequence")
    manifest_path = os.path.join(DIRS["graphs"], "sequence_manifest.csv")
    rows = save_sequence(sequence, seq_dir, manifest_path)

    plot_path = os.path.join(DIRS["results"], "phase6_sequence_summary.png")
    plot_sequence_summary(rows, plot_path)

    # Quick eyeball check: did the hub change at all across the lifespan,
    # and if so, when? (Full transition-detection logic is Phase 10's
    # job -- see findings log, Section 11 -- this is just a sanity print.)
    hub_changes = []
    prev_hub = None
    for r in rows:
        if r["top_hub"] != prev_hub:
            hub_changes.append((r["age"], r["top_hub"]))
            prev_hub = r["top_hub"]

    print("\nTop-hub identity over the sequence (age at each change):")
    for age, hub in hub_changes:
        print(f"  age {age:5.1f}: {hub}")

    print(f"\nBuilt {len(rows)} graphs across the trajectory.")
    print("\nPhase 6 complete.")


if __name__ == "__main__":
    main()
