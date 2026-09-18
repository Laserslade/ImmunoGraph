"""
V5: Dynamic graph state discovery.
Tests whether graph derived features recover known dynamical regimes.
"""

import os
import csv
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v5_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_NODES = 8
NODE_NAMES = [f"N{i}" for i in range(N_NODES)]
FD_EPS = 1e-5
T_SPAN = (0.0, 60.0)
SAMPLE_TIMES = np.arange(1.0, 60.0, 1.0)
THRESHOLD = 0.05
K_REGIMES = 4
N_PERMUTATIONS = 2000

PULSE_CENTER = 15.0
PULSE_WIDTH = 3.0
PULSE_AMPLITUDE = 3.5
INJECTION_NODES = [0, 3]


def build_weight_matrix():
    # Same wiring as V3, single condition, mutual inhibition pair intact.
    w = np.zeros((N_NODES, N_NODES))
    w[1, 0] = 1.2
    w[2, 0] = 0.9
    w[2, 1] = 1.0
    w[4, 3] = 1.1
    w[5, 4] = 1.0
    w[3, 5] = 0.95
    w[7, 6] = -1.3
    w[6, 7] = -0.8
    w[3, 2] = 0.6
    w[6, 5] = 0.5
    w[0, 7] = -0.4
    return w


def pulse_input(t):
    u = np.zeros(N_NODES)
    bump = PULSE_AMPLITUDE * np.exp(-0.5 * ((t - PULSE_CENTER) / PULSE_WIDTH) ** 2)
    for node in INJECTION_NODES:
        u[node] = bump
    return u


def make_rhs(w, decay):
    def rhs(t, y):
        return -decay * y + w @ np.tanh(y) + pulse_input(t)
    return rhs


def make_autonomous_rhs(w, decay):
    # Same internal dynamics, without the exogenous pulse term.
    # Used only for the state Jacobian, since the pulse has no x dependence.
    def rhs(y):
        return -decay * y + w @ np.tanh(y)
    return rhs


def numerical_jacobian(rhs, y, eps=FD_EPS):
    n = len(y)
    jac = np.zeros((n, n))
    for j in range(n):
        y_plus = y.copy()
        y_minus = y.copy()
        y_plus[j] += eps
        y_minus[j] -= eps
        jac[:, j] = (rhs(y_plus) - rhs(y_minus)) / (2 * eps)
    return jac


def build_edge_mask(jac):
    abs_vals = np.abs(jac)
    np.fill_diagonal(abs_vals, 0.0)
    cutoff = THRESHOLD * np.max(abs_vals) if np.max(abs_vals) > 0 else 0.0
    return abs_vals > cutoff


def count_motifs(mask):
    feedforward = 0
    feedback = 0
    mutual_inhibition_edges = 0

    for a, b, c in itertools.combinations(range(N_NODES), 3):
        for x, y, z in itertools.permutations([a, b, c]):
            if mask[y, x] and mask[z, x] and mask[z, y] and not mask[x, y] and not mask[x, z] and not mask[y, z]:
                feedforward += 1
                break
        for x, y, z in [(a, b, c), (b, c, a), (c, a, b)]:
            if mask[y, x] and mask[z, y] and mask[x, z]:
                feedback += 1
                break

    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            if mask[i, j] and mask[j, i]:
                mutual_inhibition_edges += 1

    return feedforward, feedback, mutual_inhibition_edges


def graph_snapshot_features(jac):
    mask = build_edge_mask(jac)
    n_edges = int(mask.sum())
    density = n_edges / (N_NODES * (N_NODES - 1))

    abs_weighted = np.abs(jac) * mask
    in_degree = abs_weighted.sum(axis=1)
    out_degree = abs_weighted.sum(axis=0)
    mean_in_degree = float(np.mean(in_degree))
    mean_out_degree = float(np.mean(out_degree))

    pos_edges = int(((jac > 0) & mask).sum())
    neg_edges = int(((jac < 0) & mask).sum())
    signed_balance = (pos_edges - neg_edges) / n_edges if n_edges > 0 else 0.0

    undirected = nx.Graph()
    undirected.add_nodes_from(NODE_NAMES)
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            w = 0.0
            if mask[i, j]:
                w += abs(jac[i, j])
            if mask[j, i]:
                w += abs(jac[j, i])
            if w > 0:
                undirected.add_edge(NODE_NAMES[i], NODE_NAMES[j], weight=w)

    try:
        centrality = nx.eigenvector_centrality(undirected, weight="weight", max_iter=1000)
        max_centrality = max(centrality.values())
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError, ZeroDivisionError):
        max_centrality = 0.0

    communities = list(nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight"))
    n_communities = len(communities)
    if n_edges > 0 and n_communities > 1:
        modularity = nx.algorithms.community.quality.modularity(undirected, communities, weight="weight")
    else:
        modularity = 0.0

    sym_matrix = np.abs(jac) * mask
    sym_matrix = (sym_matrix + sym_matrix.T) / 2
    eigenvalues = np.linalg.eigvalsh(sym_matrix)
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    laplacian = np.diag(sym_matrix.sum(axis=1)) - sym_matrix
    lap_eigs = np.sort(np.linalg.eigvalsh(laplacian))
    algebraic_connectivity = float(lap_eigs[1]) if len(lap_eigs) > 1 else 0.0

    weights = np.abs(jac)[mask]
    if weights.sum() > 0:
        probs = weights / weights.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        entropy = 0.0

    n_components = nx.number_connected_components(undirected)

    feedforward, feedback, mutual_inhibition_edges = count_motifs(mask)

    return {
        "density": density,
        "mean_in_degree": mean_in_degree,
        "mean_out_degree": mean_out_degree,
        "signed_balance": signed_balance,
        "max_centrality": max_centrality,
        "n_communities": float(n_communities),
        "modularity": modularity,
        "spectral_radius": spectral_radius,
        "algebraic_connectivity": algebraic_connectivity,
        "entropy": entropy,
        "n_components": float(n_components),
        "feedforward": float(feedforward),
        "feedback": float(feedback),
        "mutual_inhibition": float(mutual_inhibition_edges),
    }


def build_representations(sol, autonomous_rhs):
    graph_rows = []
    traj_rows = []

    for t in SAMPLE_TIMES:
        state = sol.sol(t)
        traj_rows.append(state)

        jac = numerical_jacobian(autonomous_rhs, state)
        graph_rows.append(graph_snapshot_features(jac))

    traj_matrix = np.array(traj_rows)
    graph_matrix = np.array([[row[k] for k in row] for row in graph_rows])
    feature_names = list(graph_rows[0].keys())
    return traj_matrix, graph_matrix, feature_names


def true_regime_labels(traj_matrix):
    baseline_mean = traj_matrix[SAMPLE_TIMES < 10].mean(axis=0)
    deviation = np.linalg.norm(traj_matrix - baseline_mean, axis=1)
    peak_idx = int(np.argmax(deviation))
    max_dev = deviation[peak_idx]

    labels = np.zeros(len(SAMPLE_TIMES), dtype=int)
    for i, t in enumerate(SAMPLE_TIMES):
        frac = deviation[i] / max_dev if max_dev > 0 else 0.0
        if i <= peak_idx:
            if frac < 0.1:
                labels[i] = 0
            elif frac < 0.9:
                labels[i] = 1
            else:
                labels[i] = 2
        else:
            if frac >= 0.9:
                labels[i] = 2
            else:
                labels[i] = 3

    return labels, deviation


def cluster_and_score(matrix, true_labels, repr_name):
    scaled = StandardScaler().fit_transform(matrix)
    clusters = KMeans(n_clusters=K_REGIMES, random_state=0, n_init=10).fit_predict(scaled)
    observed_ari = adjusted_rand_score(true_labels, clusters)

    rng = np.random.default_rng(0)
    null_scores = np.zeros(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        shuffled = rng.permutation(true_labels)
        null_scores[i] = adjusted_rand_score(shuffled, clusters)
    p_value = float(np.mean(null_scores >= observed_ari))

    silhouettes = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(scaled)
        silhouettes[k] = silhouette_score(scaled, km)

    return {
        "representation": repr_name,
        "clusters": clusters,
        "observed_ari": observed_ari,
        "null_mean": float(np.mean(null_scores)),
        "p_value": p_value,
        "silhouettes": silhouettes,
    }, null_scores


def save_summary_csv(traj_result, graph_result):
    path = os.path.join(OUTPUT_DIR, "v5_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["representation", "metric", "value"])
        for result in [traj_result, graph_result]:
            writer.writerow([result["representation"], "observed_ari", f"{result['observed_ari']:.4f}"])
            writer.writerow([result["representation"], "p_value", f"{result['p_value']:.4f}"])
            for k, score in result["silhouettes"].items():
                writer.writerow([result["representation"], f"silhouette_k{k}", f"{score:.4f}"])
    return path


def plot_state_timeline(deviation, true_labels, traj_clusters, graph_clusters):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    regime_names = ["baseline", "activation", "peak", "recovery"]
    colors = plt.cm.tab10(np.linspace(0, 1, K_REGIMES))

    axes[0].plot(SAMPLE_TIMES, deviation, color="black")
    for regime in range(K_REGIMES):
        mask = true_labels == regime
        axes[0].fill_between(SAMPLE_TIMES, 0, deviation.max() * 1.1, where=mask,
                              color=colors[regime], alpha=0.2, label=regime_names[regime])
    axes[0].set_title("deviation from baseline, with true regime shading")
    axes[0].legend(fontsize=8, ncol=4)

    axes[1].scatter(SAMPLE_TIMES, graph_clusters, c=[colors[c] for c in graph_clusters])
    axes[1].set_title("graph representation, discovered clusters")
    axes[1].set_yticks(range(K_REGIMES))

    axes[2].scatter(SAMPLE_TIMES, traj_clusters, c=[colors[c] for c in traj_clusters])
    axes[2].set_title("trajectory representation, discovered clusters")
    axes[2].set_yticks(range(K_REGIMES))
    axes[2].set_xlabel("time")

    fig.suptitle("V5: dynamical regime discovery over time")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v5_state_timeline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_silhouette_comparison(traj_result, graph_result):
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = sorted(traj_result["silhouettes"].keys())

    ax.plot(ks, [traj_result["silhouettes"][k] for k in ks], marker="o", label="trajectory")
    ax.plot(ks, [graph_result["silhouettes"][k] for k in ks], marker="o", label="graph")
    ax.axvline(K_REGIMES, color="gray", linestyle="--", label="k used for ARI test")
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("silhouette score")
    ax.set_title("V5: silhouette score by cluster count")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v5_silhouette_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ari_permutation(traj_result, traj_null, graph_result, graph_null):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, result, null in zip(axes, [traj_result, graph_result], [traj_null, graph_null]):
        ax.hist(null, bins=40, color="gray", alpha=0.7, label="null distribution")
        ax.axvline(result["observed_ari"], color="red", linewidth=2, label="observed ARI")
        ax.set_title(f"{result['representation']} (p={result['p_value']:.4f})")
        ax.set_xlabel("adjusted rand index")

    axes[0].legend(fontsize=8)
    fig.suptitle("V5: discovered regimes vs true regimes, permutation test")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v5_ari_permutation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_heatmap(graph_matrix, feature_names):
    scaled = StandardScaler().fit_transform(graph_matrix).T
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(scaled, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                    extent=[SAMPLE_TIMES[0], SAMPLE_TIMES[-1], len(feature_names), 0])
    ax.set_yticks(np.arange(len(feature_names)) + 0.5)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("time")
    ax.set_title("V5: graph feature values over time, z scored")
    fig.colorbar(im, ax=ax, label="z score")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v5_feature_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    w = build_weight_matrix()
    decay = np.ones(N_NODES)
    rhs = make_rhs(w, decay)
    autonomous_rhs = make_autonomous_rhs(w, decay)

    y0 = np.zeros(N_NODES)
    sol = solve_ivp(rhs, T_SPAN, y0, dense_output=True, max_step=0.5)

    traj_matrix, graph_matrix, feature_names = build_representations(sol, autonomous_rhs)
    true_labels, deviation = true_regime_labels(traj_matrix)

    print("True regime counts:", {name: int((true_labels == i).sum())
                                   for i, name in enumerate(["baseline", "activation", "peak", "recovery"])})

    traj_result, traj_null = cluster_and_score(traj_matrix, true_labels, "trajectory")
    graph_result, graph_null = cluster_and_score(graph_matrix, true_labels, "graph")

    print(f"\ntrajectory: observed_ari={traj_result['observed_ari']:.4f} p={traj_result['p_value']:.4f}")
    print(f"graph:      observed_ari={graph_result['observed_ari']:.4f} p={graph_result['p_value']:.4f}")

    csv_path = save_summary_csv(traj_result, graph_result)
    timeline_plot = plot_state_timeline(deviation, true_labels, traj_result["clusters"], graph_result["clusters"])
    silhouette_plot = plot_silhouette_comparison(traj_result, graph_result)
    permutation_plot = plot_ari_permutation(traj_result, traj_null, graph_result, graph_null)
    heatmap_plot = plot_feature_heatmap(graph_matrix, feature_names)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved timeline plot to {timeline_plot}")
    print(f"Saved silhouette plot to {silhouette_plot}")
    print(f"Saved permutation plot to {permutation_plot}")
    print(f"Saved feature heatmap to {heatmap_plot}")


if __name__ == "__main__":
    main()
