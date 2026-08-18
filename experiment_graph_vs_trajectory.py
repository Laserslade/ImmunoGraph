"""
Experiment: does the sensitivity-weighted graph representation reveal
structure that raw trajectory analysis does not?

The paper's central claim is that converting ODE trajectories into
sensitivity-weighted graphs exposes organization not visible from
trajectories alone. This script tests that claim directly with three
baselines, each built from the same underlying simulations but using
only raw state trajectories, no sensitivity, no graph.

Test A, change-point detection: flag transitions in the default run
using Euclidean distance between raw state vectors at consecutive ages
(same adaptive-threshold method as the graph-based version), and
compare the flagged ages against the graph-based transition ages.

Test B, state clustering: cluster (run, age) pairs using raw,
flattened state vectors instead of graph feature vectors, with the
same k-means-plus-silhouette procedure used for the graph-based
clustering. Compare cluster assignments against the graph-based
clusters (adjusted Rand index) and against how well each recovers the
known APOE4 split, with a permutation test to establish significance.

Test C, module discovery: build an alternative graph using cross-run
Pearson correlation between state variables at a fixed age (a natural,
cheap alternative to sensitivity that still produces a graph), run the
same module-detection procedure, and compare the resulting dominant
module against the sensitivity-based dominant module.

Run:
    python experiment_graph_vs_trajectory.py
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DIRS  # noqa: E402
from models.composite_backbone import STATE_NAMES, default_params  # noqa: E402
from phase1_base_model import AGE_END, AGE_START  # noqa: E402
from phase3_composite_model import run_composite_simulation  # noqa: E402
from phase4_local_sensitivity import interpolate_state  # noqa: E402
from phase6_graph_sequence import AGE_STEP  # noqa: E402
from phase7_batch_pipeline import load_run_overrides, run_composite_with_overrides  # noqa: E402

N = len(STATE_NAMES)
AGES = np.arange(AGE_START, AGE_END + AGE_STEP / 2, AGE_STEP)
N_PERMUTATIONS = 10000


def collect_raw_trajectories():
    """Solve the default run plus all 24 batch runs, and return raw
    state vectors at every age for every run. Returns a dict:
    (run_id, age) -> state vector, plus a parallel dict of AP status."""
    trajectories = {}
    ap_status = {}

    print("Solving default trajectory...")
    default_params_ = default_params()
    default_solution = run_composite_simulation(default_params_)
    for age in AGES:
        trajectories[("default", round(age, 1))] = interpolate_state(default_solution, age)
    ap_status["default"] = default_params_["AP"]

    print("Solving 24 batch trajectories...")
    runs = load_run_overrides()
    for run in runs:
        run_id = f"run_{run['run_id']:04d}"
        params, solution = run_composite_with_overrides(run["overrides"])
        for age in AGES:
            trajectories[(run_id, round(age, 1))] = interpolate_state(solution, age)
        ap_status[run_id] = run["overrides"]["AP"]

    return trajectories, ap_status


# --- Test A: change-point detection, raw trajectories vs graphs ---

def trajectory_change_points(trajectories, run_id, sd_multiplier=1.5):
    """Same adaptive-threshold change-point method as the graph-based
    version (phase10), applied to raw state vectors instead of graph
    feature vectors."""
    vectors = [trajectories[(run_id, round(age, 1))] for age in AGES]

    distances = [np.nan]
    for i in range(1, len(vectors)):
        distances.append(float(np.linalg.norm(vectors[i] - vectors[i - 1])))

    valid = np.array(distances[1:])
    threshold = valid.mean() + sd_multiplier * valid.std()
    is_transition = [False] + [d > threshold for d in distances[1:]]

    flagged_ages = [a for a, f in zip(AGES, is_transition) if f]
    return distances, threshold, flagged_ages


def load_graph_transition_ages():
    """Read the default run's graph-based transition ages, computed in
    Phase 10, from the saved manifest."""
    manifest_path = os.path.join(DIRS["graphs"], "transition_manifest.csv")
    if not os.path.exists(manifest_path):
        return None
    flagged = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["run"] == "default" and row["is_transition"] == "True":
                flagged.append(float(row["age"]))
    return flagged


def run_test_a(trajectories):
    print("\n=== Test A: change-point detection ===")
    traj_distances, traj_threshold, traj_flagged = trajectory_change_points(
        trajectories, "default"
    )
    print(f"Raw-trajectory method flags: "
          f"{[f'{a:.0f}' for a in traj_flagged]}")

    graph_flagged = load_graph_transition_ages()
    if graph_flagged is not None:
        print(f"Graph-based method flags:    "
              f"{[f'{a:.0f}' for a in graph_flagged]}")
    else:
        print("Graph-based transition manifest not found.")

    return traj_distances, traj_threshold, traj_flagged, graph_flagged


def plot_test_a(traj_distances, traj_threshold, graph_flagged, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(AGES, traj_distances, color="#2c3e50", linewidth=1.5,
            marker="o", markersize=3, label="raw trajectory distance")
    ax.axhline(traj_threshold, color="#7f8c8d", linestyle="--",
               label=f"trajectory adaptive threshold ({traj_threshold:.2f})")

    if graph_flagged:
        for age in graph_flagged:
            ax.axvline(age, color="#c0392b", alpha=0.4, linewidth=2)
        ax.axvline(graph_flagged[0], color="#c0392b", alpha=0.4, linewidth=2,
                   label="graph-based transition (Phase 10)")

    ax.set_xlabel("age")
    ax.set_ylabel("distance from previous age")
    ax.set_title("Test A: raw-trajectory change-point signal vs "
                 "graph-based transitions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved Test A plot to {save_path}")


# --- Test B: state clustering, raw trajectories vs graphs ---

def build_trajectory_feature_matrix(trajectories):
    keys = sorted(trajectories.keys(), key=lambda k: (k[0], k[1]))
    X = np.array([trajectories[k] for k in keys])
    return keys, X


def load_graph_cluster_labels():
    """Read the graph-based state cluster assignments saved by Phase 8."""
    manifest_path = os.path.join(DIRS["graphs"], "state_cluster_manifest.csv")
    if not os.path.exists(manifest_path):
        return None
    labels = {}
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            run_id = row["run_id"] if row["source"] == "batch" else "default"
            labels[(run_id, round(float(row["age"]), 1))] = int(row["cluster"])
    return labels


def permutation_test_ap_association(label_map, keys, ap_status, n_permutations=N_PERMUTATIONS, seed=0):
    """
    Permutation test for the association between cluster labels and
    APOE4 status. Permutation is done at the RUN level, not the row
    level: AP status is constant across all 51 ages within a run, so
    shuffling AP labels per-row would break that block structure and
    produce an invalid, overly liberal null distribution. Instead, the
    AP labels attached to each of the 25 run IDs are shuffled as a
    block, preserving the same number of AP=0/AP=1 runs and the same
    within-run repetition structure as the real data.

    Returns (observed_ari, p_value, null_distribution).
    """
    rng = np.random.default_rng(seed)

    run_ids = sorted(set(k[0] for k in keys))
    true_ap_by_run = {r: ap_status[r] for r in run_ids}

    cluster_labels = [label_map[k] for k in keys]
    observed_ap_labels = [true_ap_by_run[k[0]] for k in keys]
    observed_ari = adjusted_rand_score(observed_ap_labels, cluster_labels)

    ap_values = list(true_ap_by_run.values())
    null_scores = np.zeros(n_permutations)

    for i in range(n_permutations):
        shuffled = rng.permutation(ap_values)
        shuffled_ap_by_run = dict(zip(run_ids, shuffled))
        permuted_ap_labels = [shuffled_ap_by_run[k[0]] for k in keys]
        null_scores[i] = adjusted_rand_score(permuted_ap_labels, cluster_labels)

    p_value = float(np.mean(null_scores >= observed_ari))
    return observed_ari, p_value, null_scores


def plot_permutation_test(graph_null, graph_observed, graph_p,
                           traj_null, traj_observed, traj_p, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, null_scores, observed, p, title in [
        (axes[0], graph_null, graph_observed, graph_p, "graph-based clusters"),
        (axes[1], traj_null, traj_observed, traj_p, "raw-trajectory clusters"),
    ]:
        ax.hist(null_scores, bins=50, color="#95a5a6", alpha=0.8,
                label="null (permuted APOE4 labels)")
        ax.axvline(observed, color="#c0392b", linewidth=2,
                   label=f"observed ARI = {observed:.3f}")
        ax.set_title(f"{title}\np = {p:.4f}")
        ax.set_xlabel("adjusted Rand index vs APOE4")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(f"count (of {N_PERMUTATIONS:,} permutations)")
    fig.suptitle("Permutation test: cluster/APOE4 association vs null distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved permutation test plot to {save_path}")


def run_test_b(trajectories, ap_status):
    print("\n=== Test B: state clustering ===")
    keys, X_raw = build_trajectory_feature_matrix(trajectories)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print("Sweeping k=2..10 on raw trajectory features (silhouette score)...")
    scores = {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    for k, s in scores.items():
        marker = "  <-- chosen" if k == best_k else ""
        print(f"  k={k:2d}: silhouette={s:.4f}{marker}")

    km = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    traj_labels = km.fit_predict(X)
    traj_label_map = {k: int(l) for k, l in zip(keys, traj_labels)}

    graph_label_map = load_graph_cluster_labels()
    if graph_label_map is None:
        print("Graph-based cluster manifest not found.")
        return traj_label_map, None, best_k

    common_keys = [k for k in keys if k in graph_label_map]
    traj_common = [traj_label_map[k] for k in common_keys]
    graph_common = [graph_label_map[k] for k in common_keys]

    ari = adjusted_rand_score(graph_common, traj_common)
    print(f"\nAdjusted Rand Index between raw-trajectory clusters and "
          f"graph-based clusters: {ari:.4f}")
    print("(0 = agreement no better than chance, 1 = identical partitions)")

    print(f"\nRunning permutation test ({N_PERMUTATIONS:,} permutations, "
          f"shuffled at the run level)...")

    graph_observed, graph_p, graph_null = permutation_test_ap_association(
        graph_label_map, common_keys, ap_status
    )
    traj_observed, traj_p, traj_null = permutation_test_ap_association(
        traj_label_map, common_keys, ap_status
    )

    print(f"\nGraph-based clusters vs APOE4:")
    print(f"  observed ARI = {graph_observed:.4f}")
    print(f"  null distribution mean = {graph_null.mean():.4f}, "
          f"std = {graph_null.std():.4f}")
    print(f"  p-value (fraction of {N_PERMUTATIONS:,} permutations with "
          f"ARI >= observed) = {graph_p:.4f}")

    print(f"\nRaw-trajectory clusters vs APOE4:")
    print(f"  observed ARI = {traj_observed:.4f}")
    print(f"  null distribution mean = {traj_null.mean():.4f}, "
          f"std = {traj_null.std():.4f}")
    print(f"  p-value = {traj_p:.4f}")

    plot_path = os.path.join(DIRS["results"], "experiment_test_b_permutation.png")
    plot_permutation_test(graph_null, graph_observed, graph_p,
                           traj_null, traj_observed, traj_p, plot_path)

    return (traj_label_map, graph_label_map, best_k, ari,
            graph_observed, graph_p, traj_observed, traj_p)


# --- Test C: module discovery, correlation graph vs sensitivity graph ---

def build_correlation_graph(trajectories, age, run_ids):
    """Build a graph at a fixed age using cross-run Pearson correlation
    between each pair of state variables, as a natural, cheap
    alternative to sensitivity-based edges."""
    X = np.array([trajectories[(r, round(age, 1))] for r in run_ids])  # (n_runs, N)

    corr = np.corrcoef(X.T)  # (N, N)
    corr = np.nan_to_num(corr, nan=0.0)

    G = nx.Graph()
    G.add_nodes_from(STATE_NAMES)
    for i in range(N):
        for j in range(i + 1, N):
            w = abs(corr[i, j])
            if w > 1e-6:
                G.add_edge(STATE_NAMES[i], STATE_NAMES[j], weight=w)
    return G


def load_sensitivity_dominant_module(age=50):
    """Read the sensitivity-based dominant module at a given age from
    the Phase 8 module manifest, for the default run."""
    manifest_path = os.path.join(DIRS["graphs"], "module_manifest.csv")
    if not os.path.exists(manifest_path):
        return None, None
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if (row["source"] == "default" and abs(float(row["age"]) - age) < 0.5
                    and row["contains_top_hub"] == "True"):
                return set(row["module_members"].split(";")), float(row["influence_share"])
    return None, None


def run_test_c(trajectories, ap_status, age=50):
    print(f"\n=== Test C: module discovery at age {age} ===")
    run_ids = sorted(set(k[0] for k in trajectories.keys()))

    G_corr = build_correlation_graph(trajectories, age, run_ids)
    total_weight = sum(d["weight"] for _, _, d in G_corr.edges(data=True))

    communities = nx.algorithms.community.greedy_modularity_communities(
        G_corr, weight="weight"
    )

    print(f"Correlation graph: {G_corr.number_of_nodes()} nodes, "
          f"{G_corr.number_of_edges()} edges, {len(communities)} modules found")

    m_pro_module = None
    for c in communities:
        if "M_pro" in c:
            m_pro_module = c
            break

    intra_weight = sum(
        d["weight"] for u, v, d in G_corr.edges(data=True)
        if u in m_pro_module and v in m_pro_module
    )
    m_pro_share = intra_weight / total_weight if total_weight > 0 else 0.0

    print(f"Correlation-graph module containing M_pro: "
          f"{sorted(m_pro_module)}")
    print(f"  influence share: {m_pro_share:.1%}")

    sens_module, sens_share = load_sensitivity_dominant_module(age)
    if sens_module is not None:
        print(f"\nSensitivity-graph dominant module (default run, age {age}): "
              f"{sorted(sens_module)}")
        print(f"  influence share: {sens_share:.1%}")

        overlap = m_pro_module & sens_module
        print(f"\nOverlap between correlation-graph module and "
              f"sensitivity-graph module: {sorted(overlap)} "
              f"({len(overlap)} of {len(sens_module)} sensitivity-module "
              f"members)")
    else:
        print("Sensitivity-based module manifest not found.")

    return m_pro_module, m_pro_share, sens_module, sens_share


def main():
    trajectories, ap_status = collect_raw_trajectories()

    traj_distances, traj_threshold, traj_flagged, graph_flagged = run_test_a(trajectories)
    plot_path = os.path.join(DIRS["results"], "experiment_test_a_change_points.png")
    plot_test_a(traj_distances, traj_threshold, graph_flagged, plot_path)

    result_b = run_test_b(trajectories, ap_status)

    result_c = run_test_c(trajectories, ap_status, age=50)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
