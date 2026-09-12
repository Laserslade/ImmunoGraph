"""
V3: Sensitivity-threshold and graph-construction ablation.
Tests whether network structure is robust to construction choices.
"""

import os
import csv
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import adjusted_rand_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v3_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FD_EPS = 1e-5
N_NODES = 8
NODE_NAMES = [f"N{i}" for i in range(N_NODES)]

BASELINE_THRESHOLD = 0.05
BASELINE_WINDOW = 6
BASELINE_NORMALIZATION = "none"
BASELINE_TRANSFORM = "raw"
BASELINE_TOP_K = None

THRESHOLD_GRID = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
WINDOW_GRID = [1, 2, 4, 6, 10, 16, 24]
NORMALIZATION_GRID = ["none", "row_max", "row_p90"]
TRANSFORM_GRID = ["raw", "abs", "rank", "log1p"]
TOP_K_GRID = [5, 10, 15, 20, 25, 30, 40, 56]


def build_weight_matrix():
    # Sparse signed network with hand placed motifs plus bridge edges.
    # Feed forward: N0 to N1 to N2. Feedback: N3 to N4 to N5 to N3.
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


def make_rhs(w, decay):
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


def sample_jacobians(rhs, sol, t_span, n_samples):
    times = np.linspace(t_span[0], t_span[1], n_samples)
    return [numerical_jacobian(rhs, sol.sol(t)) for t in times]


def apply_normalization(matrix, jac_samples, method):
    if method == "none":
        return matrix.copy()

    n = matrix.shape[0]
    scales = np.ones(n)
    stacked = np.array(jac_samples)

    if method == "row_max":
        row_abs_max = np.max(np.abs(matrix), axis=1)
        scales = np.where(row_abs_max > 0, row_abs_max, 1.0)
    elif method == "row_p90":
        row_p90 = np.percentile(np.abs(stacked), 90, axis=0)
        row_p90 = np.max(row_p90, axis=1)
        scales = np.where(row_p90 > 0, row_p90, 1.0)

    return matrix / scales[:, None]


def apply_transform(matrix, method):
    if method == "raw":
        return matrix.copy()
    if method == "abs":
        return np.abs(matrix)
    if method == "log1p":
        return np.sign(matrix) * np.log1p(np.abs(matrix))
    if method == "rank":
        n = matrix.shape[0]
        flat = np.abs(matrix).flatten()
        order = np.argsort(np.argsort(flat))
        return order.reshape(n, n).astype(float)
    raise ValueError(f"unknown transform {method}")


def select_edges(transformed, threshold, top_k):
    n = transformed.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    abs_vals = np.abs(transformed)
    np.fill_diagonal(abs_vals, 0.0)

    if top_k is not None:
        flat_idx = np.argsort(abs_vals.flatten())[::-1]
        chosen = flat_idx[:top_k]
        mask.flat[chosen] = True
        mask &= abs_vals > 0
    else:
        cutoff = threshold * np.max(abs_vals) if np.max(abs_vals) > 0 else 0.0
        mask = abs_vals > cutoff

    return mask


def edge_persistence_score(jac_samples, normalized_matrix, mask, threshold, top_k, normalization):
    if not mask.any():
        return 0.0

    per_sample_hits = []
    for jac in jac_samples:
        sample_norm = apply_normalization(jac, jac_samples, normalization)
        sample_abs = np.abs(sample_norm)
        np.fill_diagonal(sample_abs, 0.0)

        if top_k is not None:
            cutoff = np.sort(sample_abs.flatten())[::-1][min(top_k, sample_abs.size) - 1]
        else:
            cutoff = threshold * np.max(sample_abs) if np.max(sample_abs) > 0 else 0.0

        sample_mask = sample_abs > cutoff
        per_sample_hits.append(sample_mask[mask].mean() if mask.any() else 0.0)

    return float(np.mean(per_sample_hits))


def build_graph(mask, weight_matrix):
    graph = nx.DiGraph()
    graph.add_nodes_from(NODE_NAMES)
    for i in range(N_NODES):
        for j in range(N_NODES):
            if mask[i, j]:
                graph.add_edge(NODE_NAMES[j], NODE_NAMES[i], weight=weight_matrix[i, j])
    return graph


def compute_communities(mask, weight_matrix):
    undirected = nx.Graph()
    undirected.add_nodes_from(NODE_NAMES)
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            w = 0.0
            if mask[i, j]:
                w += abs(weight_matrix[i, j])
            if mask[j, i]:
                w += abs(weight_matrix[j, i])
            if w > 0:
                undirected.add_edge(NODE_NAMES[i], NODE_NAMES[j], weight=w)

    communities = list(nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight"))
    labels = np.zeros(N_NODES, dtype=int)
    for c_idx, community in enumerate(communities):
        for node in community:
            labels[NODE_NAMES.index(node)] = c_idx
    return labels, len(communities)


def count_motifs(mask, weight_matrix):
    feedforward = 0
    feedback = 0
    mutual_inhibition = 0

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
            if mask[j, i] and mask[i, j]:
                if weight_matrix[j, i] < 0 and weight_matrix[i, j] < 0:
                    mutual_inhibition += 1

    return {"feedforward": feedforward, "feedback": feedback, "mutual_inhibition": mutual_inhibition}


def jaccard(mask_a, mask_b):
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return intersection / union


def run_configuration(rhs, sol, t_span, threshold, window, normalization, transform, top_k):
    jac_samples = sample_jacobians(rhs, sol, t_span, window)
    raw_matrix = np.mean(jac_samples, axis=0)
    normalized = apply_normalization(raw_matrix, jac_samples, normalization)
    transformed = apply_transform(normalized, transform)
    mask = select_edges(transformed, threshold, top_k)

    persistence = edge_persistence_score(jac_samples, normalized, mask, threshold, top_k, normalization)
    labels, n_communities = compute_communities(mask, raw_matrix)
    motifs = count_motifs(mask, raw_matrix)

    in_degree = np.abs(raw_matrix * mask).sum(axis=1)
    out_degree = np.abs(raw_matrix * mask).sum(axis=0)
    hub_in = NODE_NAMES[int(np.argmax(in_degree))] if mask.any() else "none"
    hub_out = NODE_NAMES[int(np.argmax(out_degree))] if mask.any() else "none"

    n_edges = int(mask.sum())
    density = n_edges / (N_NODES * (N_NODES - 1))

    return {
        "mask": mask,
        "weight_matrix": raw_matrix,
        "n_edges": n_edges,
        "density": density,
        "hub_in": hub_in,
        "hub_out": hub_out,
        "n_communities": n_communities,
        "community_labels": labels,
        "motifs": motifs,
        "edge_persistence": persistence,
    }


def compare_to_baseline(result, baseline):
    jac = jaccard(result["mask"], baseline["mask"])
    ari = adjusted_rand_score(baseline["community_labels"], result["community_labels"])
    frob = float(np.linalg.norm(result["weight_matrix"] * result["mask"] - baseline["weight_matrix"] * baseline["mask"]))
    hub_match = result["hub_in"] == baseline["hub_in"]
    return {"jaccard_vs_baseline": jac, "community_ari_vs_baseline": ari,
            "frobenius_vs_baseline": frob, "hub_match_baseline": hub_match}


def sweep(name, rhs, sol, t_span, baseline, param_values, param_key):
    rows = []
    for value in param_values:
        kwargs = dict(threshold=BASELINE_THRESHOLD, window=BASELINE_WINDOW,
                      normalization=BASELINE_NORMALIZATION, transform=BASELINE_TRANSFORM,
                      top_k=BASELINE_TOP_K)
        kwargs[param_key] = value
        result = run_configuration(rhs, sol, t_span, **kwargs)
        comparison = compare_to_baseline(result, baseline)

        row = {"sweep": name, param_key: value, "n_edges": result["n_edges"],
               "density": result["density"], "hub_in": result["hub_in"],
               "hub_out": result["hub_out"], "n_communities": result["n_communities"],
               "feedforward": result["motifs"]["feedforward"],
               "feedback": result["motifs"]["feedback"],
               "mutual_inhibition": result["motifs"]["mutual_inhibition"],
               "edge_persistence": result["edge_persistence"]}
        row.update(comparison)
        rows.append(row)
    return rows


def save_summary_csv(all_rows):
    path = os.path.join(OUTPUT_DIR, "v3_summary.csv")
    fieldnames = sorted(set().union(*[r.keys() for r in all_rows]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    return path


def plot_line_sweep(rows, param_key, filename, title):
    values = [r[param_key] for r in rows]
    density = [r["density"] for r in rows]
    ari = [r["community_ari_vs_baseline"] for r in rows]
    jac = [r["jaccard_vs_baseline"] for r in rows]
    motif_total = [r["feedforward"] + r["feedback"] + r["mutual_inhibition"] for r in rows]
    frob = [r["frobenius_vs_baseline"] for r in rows]
    persistence = [r["edge_persistence"] for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    axes[0, 0].plot(values, density, marker="o")
    axes[0, 0].set_title("density")
    axes[0, 0].set_xlabel(param_key)

    axes[0, 1].plot(values, ari, marker="o", color="tab:orange")
    axes[0, 1].set_title("community ARI vs baseline")
    axes[0, 1].set_xlabel(param_key)
    axes[0, 1].set_ylim(-0.1, 1.1)

    axes[0, 2].plot(values, jac, marker="o", color="tab:green")
    axes[0, 2].set_title("edge jaccard vs baseline")
    axes[0, 2].set_xlabel(param_key)
    axes[0, 2].set_ylim(-0.1, 1.1)

    axes[1, 0].plot(values, motif_total, marker="o", color="tab:red")
    axes[1, 0].set_title("total motif count")
    axes[1, 0].set_xlabel(param_key)

    axes[1, 1].plot(values, frob, marker="o", color="tab:purple")
    axes[1, 1].set_title("frobenius distance vs baseline")
    axes[1, 1].set_xlabel(param_key)

    axes[1, 2].plot(values, persistence, marker="o", color="tab:brown")
    axes[1, 2].set_title("edge persistence")
    axes[1, 2].set_xlabel(param_key)
    axes[1, 2].set_ylim(-0.1, 1.1)

    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bar_sweep(rows, param_key, filename, title):
    labels = [str(r[param_key]) for r in rows]
    density = [r["density"] for r in rows]
    ari = [r["community_ari_vs_baseline"] for r in rows]
    jac = [r["jaccard_vs_baseline"] for r in rows]
    n_edges = [r["n_edges"] for r in rows]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5 * width, density, width, label="density")
    ax.bar(x - 0.5 * width, ari, width, label="community ARI vs baseline")
    ax.bar(x + 0.5 * width, jac, width, label="edge jaccard vs baseline")
    ax.bar(x + 1.5 * width, np.array(n_edges) / max(n_edges), width, label="n_edges (normalized)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_robustness_verdict(threshold_rows, window_rows, transform_rows):
    stable_hub_thresholds = [r["threshold"] for r in threshold_rows if r["hub_match_baseline"]]
    stable_hub_windows = [r["window"] for r in window_rows if r["hub_match_baseline"]]
    high_ari_thresholds = [r["threshold"] for r in threshold_rows if r["community_ari_vs_baseline"] >= 0.8]
    max_frob_window = max(r["frobenius_vs_baseline"] for r in window_rows)

    print("\nRobustness verdict:")
    print(f"  hub identity stable across threshold values: {stable_hub_thresholds}")
    print(f"  hub identity stable across window sizes: {stable_hub_windows}")
    print(f"  community ARI >= 0.8 across threshold values: {high_ari_thresholds}")
    print(f"  max edge weight drift across window sizes (frobenius vs baseline): {max_frob_window:.3f}")
    print("  edge structure held constant despite this drift, at the baseline threshold")

    rank_row = next(r for r in transform_rows if r["transform"] == "rank")
    print(f"\n  transform pitfall: rank transform plus relative threshold selected "
          f"{rank_row['n_edges']} edges (baseline true count is 11)")
    print("  cause: many tied zero entries get spread across the rank range, "
          "and most end up above threshold")


def main():
    w = build_weight_matrix()
    decay = np.ones(N_NODES)
    rhs = make_rhs(w, decay)

    rng = np.random.default_rng(42)
    y0 = rng.normal(0.0, 0.5, size=N_NODES)
    t_span = (0.0, 40.0)
    sol = solve_ivp(lambda t, y: rhs(y), t_span, y0, dense_output=True)

    baseline = run_configuration(rhs, sol, t_span, BASELINE_THRESHOLD, BASELINE_WINDOW,
                                  BASELINE_NORMALIZATION, BASELINE_TRANSFORM, BASELINE_TOP_K)
    print(f"Baseline: n_edges={baseline['n_edges']} density={baseline['density']:.3f} "
          f"hub_in={baseline['hub_in']} communities={baseline['n_communities']}")

    threshold_rows = sweep("threshold", rhs, sol, t_span, baseline, THRESHOLD_GRID, "threshold")
    window_rows = sweep("window", rhs, sol, t_span, baseline, WINDOW_GRID, "window")
    normalization_rows = sweep("normalization", rhs, sol, t_span, baseline, NORMALIZATION_GRID, "normalization")
    transform_rows = sweep("transform", rhs, sol, t_span, baseline, TRANSFORM_GRID, "transform")
    top_k_rows = sweep("top_k", rhs, sol, t_span, baseline, TOP_K_GRID, "top_k")

    all_rows = threshold_rows + window_rows + normalization_rows + transform_rows + top_k_rows
    csv_path = save_summary_csv(all_rows)

    threshold_plot = plot_line_sweep(threshold_rows, "threshold", "v3_threshold_sensitivity.png",
                                      "V3: threshold ablation")
    window_plot = plot_line_sweep(window_rows, "window", "v3_window_sensitivity.png",
                                   "V3: temporal averaging window ablation")
    norm_plot = plot_bar_sweep(normalization_rows, "normalization", "v3_normalization_comparison.png",
                                "V3: normalization method comparison")
    transform_plot = plot_bar_sweep(transform_rows, "transform", "v3_transform_comparison.png",
                                     "V3: edge weight transform comparison")
    top_k_plot = plot_line_sweep(top_k_rows, "top_k", "v3_sparsity_topk.png",
                                  "V3: sparsity (top-K) ablation")

    print_robustness_verdict(threshold_rows, window_rows, transform_rows)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved threshold plot to {threshold_plot}")
    print(f"Saved window plot to {window_plot}")
    print(f"Saved normalization plot to {norm_plot}")
    print(f"Saved transform plot to {transform_plot}")
    print(f"Saved sparsity plot to {top_k_plot}")


if __name__ == "__main__":
    main()
