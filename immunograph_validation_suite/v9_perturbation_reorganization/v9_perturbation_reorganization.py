"""
V9: Perturbation induced network reorganization.
Compares a baseline graph against a perturbed graph, and separates
pure magnitude change from genuine architectural reorganization.
"""

import os
import csv
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v9_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_NODES = 8
NODE_NAMES = [f"N{i}" for i in range(N_NODES)]
FD_EPS = 1e-5
THRESHOLD = 0.05
T_SPAN = (0.0, 60.0)
SAMPLE_TIMES = np.arange(1.0, 60.0, 1.0)
PULSE_CENTER = 15.0
PULSE_WIDTH = 2.0
INJECTION_NODES = [0, 3]

SEVERITY_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
REPORT_SEVERITIES = [1.5, 5.0]
N_ENSEMBLE = 40
ENSEMBLE_AMPLITUDE = 3.5


def base_weight_matrix():
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


def make_pulse(amplitude):
    def pulse(t):
        u = np.zeros(N_NODES)
        bump = amplitude * np.exp(-0.5 * ((t - PULSE_CENTER) / PULSE_WIDTH) ** 2)
        for node in INJECTION_NODES:
            u[node] = bump
        return u
    return pulse


def make_rhs(w, decay, pulse):
    def rhs(t, y):
        return -decay * y + w @ np.tanh(y) + pulse(t)
    return rhs


def make_autonomous_rhs(w, decay):
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


def count_motifs(jac, mask):
    coherent_ffl = incoherent_ffl = 0
    positive_feedback = negative_feedback = 0

    for a, b, c in itertools.combinations(range(N_NODES), 3):
        for x, y, z in itertools.permutations([a, b, c]):
            if mask[y, x] and mask[z, x] and mask[z, y] and not mask[x, y] and not mask[x, z] and not mask[y, z]:
                direct_sign = np.sign(jac[z, x])
                indirect_sign = np.sign(jac[y, x]) * np.sign(jac[z, y])
                if direct_sign == indirect_sign:
                    coherent_ffl += 1
                else:
                    incoherent_ffl += 1
                break
        for x, y, z in [(a, b, c), (b, c, a), (c, a, b)]:
            if mask[y, x] and mask[z, y] and mask[x, z]:
                net_sign = np.sign(jac[y, x]) * np.sign(jac[z, y]) * np.sign(jac[x, z])
                if net_sign > 0:
                    positive_feedback += 1
                else:
                    negative_feedback += 1
                break

    mutual_activation = mutual_inhibition = mixed_pair = 0
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            if mask[i, j] and mask[j, i]:
                if jac[i, j] > 0 and jac[j, i] > 0:
                    mutual_activation += 1
                elif jac[i, j] < 0 and jac[j, i] < 0:
                    mutual_inhibition += 1
                else:
                    mixed_pair += 1

    return {"coherent_ffl": coherent_ffl, "incoherent_ffl": incoherent_ffl,
            "positive_feedback": positive_feedback, "negative_feedback": negative_feedback,
            "mutual_activation": mutual_activation, "mutual_inhibition": mutual_inhibition,
            "mixed_pair": mixed_pair}


def graph_report(jac):
    mask = build_edge_mask(jac)
    n_edges = int(mask.sum())

    abs_weighted = np.abs(jac) * mask
    in_degree = abs_weighted.sum(axis=1)
    hub = NODE_NAMES[int(np.argmax(in_degree))] if mask.any() else "none"

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

    communities = list(nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight"))
    n_communities = len(communities)
    if n_edges > 0 and n_communities > 1:
        modularity = nx.algorithms.community.quality.modularity(undirected, communities, weight="weight")
    else:
        modularity = 0.0

    weights = np.abs(jac)[mask]
    if weights.sum() > 0:
        probs = weights / weights.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        entropy = 0.0

    motifs = count_motifs(jac, mask)

    return {"jac": jac, "mask": mask, "n_edges": n_edges, "hub": hub,
            "n_communities": n_communities, "modularity": modularity,
            "entropy": entropy, "motifs": motifs}


def get_baseline_graph():
    w = base_weight_matrix()
    decay = np.ones(N_NODES)
    autonomous_rhs = make_autonomous_rhs(w, decay)
    jac = numerical_jacobian(autonomous_rhs, np.zeros(N_NODES))
    return graph_report(jac)


def get_perturbed_graph(amplitude, w=None):
    if w is None:
        w = base_weight_matrix()
    decay = np.ones(N_NODES)
    pulse = make_pulse(amplitude)
    rhs = make_rhs(w, decay, pulse)
    autonomous_rhs = make_autonomous_rhs(w, decay)

    sol = solve_ivp(rhs, T_SPAN, np.zeros(N_NODES), dense_output=True, max_step=0.5)
    states = [sol.sol(t) for t in SAMPLE_TIMES]
    baseline_mean = states[0]
    deviation = [np.linalg.norm(s - baseline_mean) for s in states]
    peak_idx = int(np.argmax(deviation))
    peak_state = states[peak_idx]

    jac = numerical_jacobian(autonomous_rhs, peak_state)
    return graph_report(jac)


def reorganization_distance(report_a, report_b):
    jac_a, jac_b = report_a["jac"], report_b["jac"]
    d_raw = float(np.linalg.norm(jac_b - jac_a))

    norm_a = np.linalg.norm(jac_a)
    norm_b = np.linalg.norm(jac_b)
    shape_a = jac_a / norm_a if norm_a > 0 else jac_a
    shape_b = jac_b / norm_b if norm_b > 0 else jac_b
    d_shape = float(np.linalg.norm(shape_b - shape_a))

    mask_a, mask_b = report_a["mask"], report_b["mask"]
    gained = int(np.logical_and(mask_b, np.logical_not(mask_a)).sum())
    lost = int(np.logical_and(mask_a, np.logical_not(mask_b)).sum())

    return {"d_raw": d_raw, "d_shape": d_shape,
            "shape_fraction": d_shape / d_raw if d_raw > 0 else 0.0,
            "edges_gained": gained, "edges_lost": lost,
            "hub_changed": report_a["hub"] != report_b["hub"],
            "modularity_change": report_b["modularity"] - report_a["modularity"],
            "entropy_change": report_b["entropy"] - report_a["entropy"],
            "community_change": report_b["n_communities"] - report_a["n_communities"]}


def run_severity_sweep(baseline_report):
    rows = []
    for amplitude in SEVERITY_GRID:
        perturbed_report = get_perturbed_graph(amplitude)
        distance_result = reorganization_distance(baseline_report, perturbed_report)
        distance_result["amplitude"] = amplitude
        rows.append(distance_result)
    return rows


def run_ensemble(baseline_report):
    rng = np.random.default_rng(0)
    results = []
    for _ in range(N_ENSEMBLE):
        w = base_weight_matrix()
        mask = w != 0.0
        jitter = rng.normal(0.0, 0.08, size=w.shape)
        w_jittered = w + jitter * mask
        perturbed_report = get_perturbed_graph(ENSEMBLE_AMPLITUDE, w=w_jittered)
        distance_result = reorganization_distance(baseline_report, perturbed_report)
        results.append(distance_result)
    return results


def save_summary_csv(sweep_rows, ensemble_results, report_snapshots):
    path = os.path.join(OUTPUT_DIR, "v9_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "key", "value"])

        for row in sweep_rows:
            for key in ["d_raw", "d_shape", "shape_fraction", "edges_gained", "edges_lost", "hub_changed"]:
                writer.writerow([f"severity_{row['amplitude']}", key, row[key]])

        d_raw_values = [r["d_raw"] for r in ensemble_results]
        shape_fraction_values = [r["shape_fraction"] for r in ensemble_results]
        writer.writerow(["ensemble", "d_raw_mean", f"{np.mean(d_raw_values):.4f}"])
        writer.writerow(["ensemble", "d_raw_std", f"{np.std(d_raw_values):.4f}"])
        writer.writerow(["ensemble", "shape_fraction_mean", f"{np.mean(shape_fraction_values):.4f}"])
        writer.writerow(["ensemble", "shape_fraction_std", f"{np.std(shape_fraction_values):.4f}"])

        for label, report in report_snapshots.items():
            writer.writerow([label, "hub", report["hub"]])
            writer.writerow([label, "n_edges", report["n_edges"]])
            writer.writerow([label, "n_communities", report["n_communities"]])
            writer.writerow([label, "modularity", f"{report['modularity']:.4f}"])
            writer.writerow([label, "entropy", f"{report['entropy']:.4f}"])
            for motif, count in report["motifs"].items():
                writer.writerow([label, f"motif_{motif}", count])
    return path


def plot_severity_sweep(sweep_rows):
    amplitudes = [r["amplitude"] for r in sweep_rows]
    d_raw = [r["d_raw"] for r in sweep_rows]
    d_shape = [r["d_shape"] for r in sweep_rows]
    shape_fraction = [r["shape_fraction"] for r in sweep_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(amplitudes, d_raw, marker="o", label="raw distance (magnitude included)")
    axes[0].plot(amplitudes, d_shape, marker="o", label="shape distance (magnitude removed)")
    axes[0].set_xlabel("pulse amplitude")
    axes[0].set_ylabel("graph distance")
    axes[0].set_title("V9: reorganization distance vs perturbation severity")
    axes[0].legend(fontsize=8)

    axes[1].plot(amplitudes, shape_fraction, marker="o", color="tab:red")
    axes[1].set_xlabel("pulse amplitude")
    axes[1].set_ylabel("shape distance / raw distance")
    axes[1].set_title("V9: fraction of change that is genuine reorganization")
    axes[1].set_ylim(0, max(shape_fraction) * 1.2 if shape_fraction else 1)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v9_severity_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_structural_breakdown(baseline_report, report_snapshots):
    labels = list(report_snapshots.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    entropy_vals = [baseline_report["entropy"]] + [report_snapshots[l]["entropy"] for l in labels]
    modularity_vals = [baseline_report["modularity"]] + [report_snapshots[l]["modularity"] for l in labels]
    edges_vals = [baseline_report["n_edges"]] + [report_snapshots[l]["n_edges"] for l in labels]
    all_labels = ["baseline"] + labels

    axes[0, 0].bar(all_labels, entropy_vals, color="tab:purple", alpha=0.8)
    axes[0, 0].set_title("network entropy")
    axes[0, 0].tick_params(axis="x", rotation=20)

    axes[0, 1].bar(all_labels, modularity_vals, color="tab:orange", alpha=0.8)
    axes[0, 1].set_title("modularity")
    axes[0, 1].tick_params(axis="x", rotation=20)

    axes[1, 0].bar(all_labels, edges_vals, color="tab:blue", alpha=0.8)
    axes[1, 0].set_title("edge count")
    axes[1, 0].tick_params(axis="x", rotation=20)

    motif_keys = ["coherent_ffl", "positive_feedback"]
    x = np.arange(len(motif_keys))
    width = 0.25
    for i, label in enumerate(all_labels):
        report = baseline_report if label == "baseline" else report_snapshots[label]
        values = [report["motifs"][k] for k in motif_keys]
        axes[1, 1].bar(x + (i - 1) * width, values, width, label=label)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(motif_keys, rotation=15)
    axes[1, 1].set_title("motif counts")
    axes[1, 1].legend(fontsize=7)

    fig.suptitle("V9: structural breakdown, baseline vs perturbed at two severities")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v9_structural_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ensemble_distribution(ensemble_results):
    d_raw_values = [r["d_raw"] for r in ensemble_results]
    shape_fraction_values = [r["shape_fraction"] for r in ensemble_results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(d_raw_values, bins=15, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("R, raw reorganization distance")
    axes[0].set_title("V9: R across 40 jittered networks, fixed severity")

    axes[1].hist(shape_fraction_values, bins=15, color="tab:red", alpha=0.8)
    axes[1].set_xlabel("shape distance / raw distance")
    axes[1].set_title("V9: reorganization fraction across the ensemble")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v9_ensemble_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    baseline_report = get_baseline_graph()
    print(f"Baseline: hub={baseline_report['hub']} n_edges={baseline_report['n_edges']} "
          f"entropy={baseline_report['entropy']:.3f}")

    report_snapshots = {}
    for amplitude in REPORT_SEVERITIES:
        report_snapshots[f"amplitude_{amplitude}"] = get_perturbed_graph(amplitude)

    for label, report in report_snapshots.items():
        distance_result = reorganization_distance(baseline_report, report)
        print(f"\n{label}: hub={report['hub']} n_edges={report['n_edges']} "
              f"entropy={report['entropy']:.3f}")
        print(f"  R (raw distance)={distance_result['d_raw']:.4f}, "
              f"shape distance={distance_result['d_shape']:.4f}, "
              f"shape fraction={distance_result['shape_fraction']:.3f}")
        print(f"  edges gained={distance_result['edges_gained']}, "
              f"edges lost={distance_result['edges_lost']}, "
              f"hub changed={distance_result['hub_changed']}")

    sweep_rows = run_severity_sweep(baseline_report)
    print("\nSeverity sweep:")
    for row in sweep_rows:
        print(f"  amplitude={row['amplitude']:.1f} R={row['d_raw']:.4f} "
              f"shape_fraction={row['shape_fraction']:.3f}")

    ensemble_results = run_ensemble(baseline_report)
    d_raw_values = [r["d_raw"] for r in ensemble_results]
    shape_fraction_values = [r["shape_fraction"] for r in ensemble_results]
    print(f"\nEnsemble (n={N_ENSEMBLE}, amplitude={ENSEMBLE_AMPLITUDE}): "
          f"R mean={np.mean(d_raw_values):.4f} std={np.std(d_raw_values):.4f}, "
          f"shape fraction mean={np.mean(shape_fraction_values):.3f} "
          f"std={np.std(shape_fraction_values):.3f}")

    csv_path = save_summary_csv(sweep_rows, ensemble_results, report_snapshots)
    sweep_plot = plot_severity_sweep(sweep_rows)
    breakdown_plot = plot_structural_breakdown(baseline_report, report_snapshots)
    ensemble_plot = plot_ensemble_distribution(ensemble_results)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved severity sweep plot to {sweep_plot}")
    print(f"Saved structural breakdown plot to {breakdown_plot}")
    print(f"Saved ensemble distribution plot to {ensemble_plot}")


if __name__ == "__main__":
    main()
