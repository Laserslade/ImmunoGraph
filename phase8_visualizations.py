"""
phase8_visualizations.py

Visualization companion to phase8_motif_discovery.py. Not a new pipeline
phase on its own -- it re-derives the same module partition and cluster
centroids Phase 8 already computed, and draws them, so the findings in
graphs/module_manifest.csv and graphs/state_cluster_manifest.csv (and
findings log Section 8.8) are visible as pictures, not just CSV rows.

Produces three figures in results/:
  1. phase8_dominant_module_age50.png -- the actual age-50 default graph,
     nodes colored by which module they landed in (Part A's partition),
     dominant module (the one containing the top hub) highlighted.
  2. phase8_motif_states.png -- the two ensemble-level "motif" graphs
     (Part B's cluster centroids), drawn side by side using each
     centroid's top edges.
  3. phase8_cluster_timeline.png -- cluster label vs age, for the
     default run and all 24 batch runs (colored by AP), showing the
     age~50-54 transition split by APOE4 status documented in findings
     log Section 8.8.

Run:
    python phase8_visualizations.py
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
from phase3_composite_model import run_composite_simulation  # noqa: E402
from phase4_local_sensitivity import compute_dydt_scales, sensitivity_at_age  # noqa: E402
from phase5_graph_construction import build_graph  # noqa: E402
from phase8_motif_discovery import (  # noqa: E402
    N,
    NODE_INDEX,
    build_feature_matrix,
    choose_k_by_silhouette,
    detect_modules,
    list_all_graphs,
)
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

MODULE_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#34495e", "#e67e22",
]


def plot_dominant_module(save_path):
    """Rebuild the age-50 default graph and its module partition
    (exactly as Phase 8 Part A computed it), draw all nodes colored by
    module, with the dominant module's intra-module edges emphasized."""
    params = default_params()
    solution = run_composite_simulation(params)
    scales = compute_dydt_scales(solution, params)
    t, y, J, S = sensitivity_at_age(solution, 50, params, scales)
    G = build_graph(S, STATE_NAMES)

    modules = detect_modules(G)

    in_weight = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        in_weight[v] += abs(d["weight"])
    top_hub = max(in_weight, key=in_weight.get)
    dominant_module = next(m for m in modules if top_hub in m)

    node_color = {}
    node_module_id = {}
    for i, module in enumerate(modules):
        color = MODULE_COLORS[i % len(MODULE_COLORS)]
        for n in module:
            node_color[n] = color
            node_module_id[n] = i

    pos = nx.spring_layout(G, seed=7, k=1.1)

    fig, ax = plt.subplots(figsize=(12, 10))

    # draw all edges lightly first
    for u, v, data in G.edges(data=True):
        same_module = node_module_id.get(u) == node_module_id.get(v)
        is_dominant = same_module and u in dominant_module
        color = "#c0392b" if (is_dominant and data["weight"] > 0) else \
                "#2980b9" if (is_dominant and data["weight"] < 0) else "#dddddd"
        width = 2.5 if is_dominant else 0.6
        alpha = 0.9 if is_dominant else 0.35
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=color, width=width,
            alpha=alpha, arrowsize=10 if is_dominant else 6,
            connectionstyle="arc3,rad=0.08", ax=ax,
        )

    for module in modules:
        nodelist = list(module)
        is_dominant = module == dominant_module
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist,
            node_color=[node_color[n] for n in nodelist],
            edgecolors="#222222" if is_dominant else "#888888",
            linewidths=2.5 if is_dominant else 1.0,
            node_size=1600 if is_dominant else 1000,
            alpha=1.0 if is_dominant else 0.75,
            ax=ax,
        )

    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    dom_share = None
    total_abs = sum(abs(d["weight"]) for _, _, d in G.edges(data=True))
    intra = sum(abs(d["weight"]) for u, v, d in G.edges(data=True)
                if u in dominant_module and v in dominant_module)
    dom_share = intra / total_abs if total_abs else 0.0

    ax.set_title(
        f"Age-50 sensitivity graph, partitioned into {len(modules)} modules\n"
        f"Dominant module (bold outline, thick edges): "
        f"{{{', '.join(sorted(dominant_module))}}} -- "
        f"{100*dom_share:.1f}% of total network influence, "
        f"top hub = {top_hub}",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def plot_motif_states(save_path):
    """Redo Part B's clustering (fast -- k-means on 1275x462 is cheap)
    and draw both cluster centroids as small motif graphs, side by
    side, each showing its top-8 edges by |weight|."""
    graph_records = list_all_graphs()
    X_raw = build_feature_matrix(graph_records)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    best_k, _ = choose_k_by_silhouette(X)
    km = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    labels = km.fit_predict(X)
    centroids_raw = scaler.inverse_transform(km.cluster_centers_)

    fig, axes = plt.subplots(1, best_k, figsize=(7 * best_k, 7))
    if best_k == 1:
        axes = [axes]

    for cluster_id, ax in enumerate(axes):
        n_members = int((labels == cluster_id).sum())
        vec = centroids_raw[cluster_id]
        M = np.zeros((N, N))
        mask = ~np.eye(N, dtype=bool)
        M[mask] = vec

        edge_list = [
            (STATE_NAMES[j], STATE_NAMES[i], M[i, j])
            for i in range(N) for j in range(N) if i != j
        ]
        edge_list.sort(key=lambda e: -abs(e[2]))
        top_edges = edge_list[:8]

        Gm = nx.DiGraph()
        for src, tgt, w in top_edges:
            Gm.add_edge(src, tgt, weight=w)

        pos = nx.spring_layout(Gm, seed=3, k=1.3)
        weights = [Gm[u][v]["weight"] for u, v in Gm.edges()]
        max_abs = max(abs(w) for w in weights)
        colors = ["#c0392b" if w > 0 else "#2980b9" for w in weights]
        widths = [1.5 + 4 * abs(w) / max_abs for w in weights]

        nx.draw_networkx_nodes(Gm, pos, node_color="#f5f5f5",
                                edgecolors="#333333", node_size=1800, ax=ax)
        nx.draw_networkx_labels(Gm, pos, font_size=10, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(
            Gm, pos, edge_color=colors, width=widths, arrowsize=18,
            connectionstyle="arc3,rad=0.1", ax=ax,
        )
        ax.set_title(f"State {cluster_id}  ({n_members}/{len(labels)} graphs, "
                     f"{100*n_members/len(labels):.0f}%)", fontsize=12)
        ax.axis("off")

    fig.suptitle("Recurring network states (motifs) across the full ensemble\n"
                  "top-8 centroid edges shown per state; red = positive, blue = suppressive",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")

    return graph_records, labels


def plot_cluster_timeline(graph_records, labels, save_path):
    """Heatmap: rows = default run + 24 batch runs (sorted by AP status),
    columns = age, color = cluster state. Chosen over overlapping line
    plots because most runs share cluster 1 for ages 30-50 and cluster 0
    (or 1, if non-switching) afterward -- as line plots those overlap
    almost exactly and are hard to read (see findings log Section 8.8
    for why: cluster 1 -> 0 switch only happens in 17/24 runs, all in
    the same age~50-54 window, so pre-switch every run's line coincides).
    A heatmap makes the per-run switch age and the AP split visible at
    a glance instead."""
    import csv as csv_mod

    batch_manifest_path = os.path.join(DIRS["graphs"], "batch_manifest.csv")
    ap_by_run = {}
    with open(batch_manifest_path) as f:
        for row in csv_mod.DictReader(f):
            run_id = f"run_{int(row['run_id']):04d}"
            ap_by_run[run_id] = float(row["param__AP"])

    by_run = {}
    for rec, label in zip(graph_records, labels):
        by_run.setdefault(rec["run_id"], []).append((rec["age"], label))
    for run_id in by_run:
        by_run[run_id].sort()

    # Common age grid: use the default run's ages (batch runs share
    # effectively the same solver-snapped ages, off by rounding only).
    ages = [a for a, _ in by_run["default"]]
    n_ages = len(ages)

    # Row order: default first, then batch runs sorted by AP (1 then 0)
    # so the AP split is visually grouped, not scattered.
    batch_run_ids = sorted(
        (rid for rid in by_run if rid != "default"),
        key=lambda rid: (-ap_by_run[rid], rid),
    )
    row_order = ["default"] + batch_run_ids

    matrix = np.zeros((len(row_order), n_ages))
    for i, rid in enumerate(row_order):
        seq = by_run[rid]
        # sequences should all have n_ages points; guard just in case
        for j in range(min(n_ages, len(seq))):
            matrix[i, j] = seq[j][1]

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=0, vmax=1,
                    extent=[ages[0], ages[-1], len(row_order), 0])

    row_labels = []
    for rid in row_order:
        if rid == "default":
            row_labels.append("default")
        else:
            ap = ap_by_run[rid]
            row_labels.append(f"{rid} (AP={'1' if ap == 1.0 else '0'})")

    ax.set_yticks(np.arange(len(row_order)) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("age")
    ax.set_title(
        "Network-state cluster over age -- default run + all 24 perturbation runs\n"
        "(blue = state 0, red = state 1; rows grouped by APOE4 status, "
        "see findings log Section 8.8)"
    )
    ax.axhline(y=1, color="black", linewidth=1.5)  # separates default from batch

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["state 0", "state 1"])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def main():
    dominant_path = os.path.join(DIRS["results"], "phase8_dominant_module_age50.png")
    plot_dominant_module(dominant_path)

    motifs_path = os.path.join(DIRS["results"], "phase8_motif_states.png")
    graph_records, labels = plot_motif_states(motifs_path)

    timeline_path = os.path.join(DIRS["results"], "phase8_cluster_timeline.png")
    plot_cluster_timeline(graph_records, labels, timeline_path)

    print("\nPhase 8 visualizations complete.")


if __name__ == "__main__":
    main()
