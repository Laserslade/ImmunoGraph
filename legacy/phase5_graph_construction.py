"""
phase5_graph_construction.py

Phase 5 of the Neuroimmune Graph Project.

Turns ONE sensitivity matrix (Phase 4) into an actual graph object:
nodes = state variables, directed edges = mechanistic sensitivity
(source variable -> target variable it influences), edge weight = the
normalized sensitivity value. This is the first concrete instance of
the project's core representation (see findings log, Section 3.5).

Two outputs are produced, deliberately kept separate:
  1. The FULL graph (every non-zero sensitivity entry) is saved to disk
     in GraphML format -- a standard, tool-agnostic format any later
     phase (or any other software) can reload without depending on
     this project's own code.
  2. A VISUALIZATION uses only the top-K edges by magnitude, because
     drawing all ~110 non-zero edges on 22 nodes produces an unreadable
     hairball. The full graph (saved separately) is not thresholded --
     only the picture is.

Run:
    python phase5_graph_construction.py
"""

import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DIRS  # noqa: E402
from models.composite_backbone import STATE_NAMES, default_params  # noqa: E402
from phase3_composite_model import run_composite_simulation  # noqa: E402
from phase4_local_sensitivity import compute_dydt_scales, sensitivity_at_age  # noqa: E402

VISUALIZATION_TOP_K = 25  # keep the picture readable; the saved graph keeps everything


def build_graph(S, state_names, min_abs_weight=1e-9):
    """
    Build a directed, weighted networkx graph from a sensitivity matrix.

    An edge j -> i exists whenever S[i, j] is non-zero (above
    min_abs_weight in magnitude), meaning "variable j mechanistically
    influences variable i's rate of change." Edge weight is S[i, j]
    itself, so sign is preserved (positive = j drives i up, negative =
    j suppresses i).
    """
    G = nx.DiGraph()
    G.add_nodes_from(state_names)

    n = len(state_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            weight = S[i, j]
            if abs(weight) > min_abs_weight:
                # edge direction: source is the variable being read (j),
                # target is the variable whose rate of change it affects (i)
                G.add_edge(state_names[j], state_names[i], weight=float(weight))

    return G


def save_graph(G, path):
    """Save the full graph to GraphML -- a standard, tool-agnostic format
    (any graph library, not just networkx, can read this)."""
    nx.write_graphml(G, path)
    print(f"Saved full graph ({G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges) to {path}")


def plot_graph(G, age, save_path, top_k=VISUALIZATION_TOP_K):
    """Visualize only the top-K edges by |weight|, to keep the picture
    readable. The saved GraphML file (see save_graph) is NOT thresholded
    -- this function only affects what gets drawn."""
    edges_by_weight = sorted(
        G.edges(data=True), key=lambda e: abs(e[2]["weight"]), reverse=True
    )
    top_edges = edges_by_weight[:top_k]

    G_vis = nx.DiGraph()
    G_vis.add_nodes_from(G.nodes())
    for u, v, data in top_edges:
        G_vis.add_edge(u, v, weight=data["weight"])

    # Only show nodes that actually participate in the top-K edges,
    # so isolated nodes don't clutter the picture.
    connected_nodes = set()
    for u, v in G_vis.edges():
        connected_nodes.add(u)
        connected_nodes.add(v)
    G_vis = G_vis.subgraph(connected_nodes).copy()

    pos = nx.spring_layout(G_vis, seed=42, k=1.2)

    weights = [G_vis[u][v]["weight"] for u, v in G_vis.edges()]
    max_abs_w = max(abs(w) for w in weights) if weights else 1.0
    edge_colors = ["#c0392b" if w > 0 else "#2980b9" for w in weights]
    edge_widths = [1 + 3 * abs(w) / max_abs_w for w in weights]

    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_nodes(G_vis, pos, node_color="#f5f5f5",
                            edgecolors="#333333", node_size=1400, ax=ax)
    nx.draw_networkx_labels(G_vis, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(
        G_vis, pos, edge_color=edge_colors, width=edge_widths,
        arrowsize=15, connectionstyle="arc3,rad=0.08", ax=ax,
    )

    ax.set_title(
        f"Top {top_k} sensitivity-weighted interactions at age {age:.0f}\n"
        f"(red = positive influence, blue = suppressive; full graph has "
        f"{G.number_of_edges()} edges, saved separately)",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved graph visualization ({len(top_edges)} edges shown) to {save_path}")


def main():
    print("Solving composite model and computing sensitivity at age 50...")
    params = default_params()
    solution = run_composite_simulation(params)
    scales = compute_dydt_scales(solution, params)

    age = 50
    t, y, J, S = sensitivity_at_age(solution, age, params, scales)

    G = build_graph(S, STATE_NAMES)
    print(f"\nBuilt graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(density {nx.density(G):.3f})")

    graphml_path = os.path.join(DIRS["graphs"], f"sensitivity_graph_age{int(age)}.graphml")
    save_graph(G, graphml_path)

    plot_path = os.path.join(DIRS["results"], f"phase5_graph_age{int(age)}.png")
    plot_graph(G, t, plot_path)

    # Quick reload check -- confirms the saved GraphML round-trips cleanly,
    # so it can be loaded later without re-running the pipeline.
    G_reloaded = nx.read_graphml(graphml_path)
    matches = (G_reloaded.number_of_nodes() == G.number_of_nodes()
               and G_reloaded.number_of_edges() == G.number_of_edges())
    print(f"\nReload check: {'PASSED' if matches else 'FAILED'} "
          f"({G_reloaded.number_of_nodes()} nodes, {G_reloaded.number_of_edges()} edges)")

    print("\nPhase 5 complete.")


if __name__ == "__main__":
    main()
