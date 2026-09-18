"""
V10: Computational intervention analysis.
Tests whether network level measurements distinguish interventions
that look similar at the biomarker level.
"""

import os
import csv
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v10_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_NODES = 8
NODE_NAMES = [f"N{i}" for i in range(N_NODES)]
FD_EPS = 1e-5
THRESHOLD = 0.05
T_SPAN = (0.0, 60.0)
SAMPLE_TIMES = np.arange(0.0, 60.0, 0.5)
PULSE_CENTER = 15.0
PULSE_WIDTH = 2.0
PULSE_AMPLITUDE = 5.0
INJECTION_NODES = [0, 3]
INTERVENTION_ONSET = 20.0
EVAL_TIME = 40.0

MISTARGET_NODES = [6, 7]

DAMP_SYMPTOMATIC = 0.195
DAMP_TARGETED = 3.0
DAMP_MISTARGETED = 5.0


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


def injury_pulse(t):
    u = np.zeros(N_NODES)
    bump = PULSE_AMPLITUDE * np.exp(-0.5 * ((t - PULSE_CENTER) / PULSE_WIDTH) ** 2)
    for node in INJECTION_NODES:
        u[node] = bump
    return u


def make_intervention(strategy):
    def none_intervention(t, y):
        return np.zeros(N_NODES)

    def symptomatic(t, y):
        if t < INTERVENTION_ONSET:
            return np.zeros(N_NODES)
        return -DAMP_SYMPTOMATIC * y

    def targeted(t, y):
        if t < INTERVENTION_ONSET:
            return np.zeros(N_NODES)
        u = np.zeros(N_NODES)
        for node in INJECTION_NODES:
            u[node] = -DAMP_TARGETED * y[node]
        return u

    def mistargeted(t, y):
        if t < INTERVENTION_ONSET:
            return np.zeros(N_NODES)
        u = np.zeros(N_NODES)
        for node in MISTARGET_NODES:
            u[node] = -DAMP_MISTARGETED * y[node]
        return u

    return {"none": none_intervention, "symptomatic": symptomatic,
            "targeted": targeted, "mistargeted": mistargeted}[strategy]


def make_rhs(w, decay, intervention):
    def rhs(t, y):
        return -decay * y + w @ np.tanh(y) + injury_pulse(t) + intervention(t, y)
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


def count_key_motifs(jac, mask):
    coherent_ffl = 0
    positive_feedback = 0
    for a, b, c in itertools.combinations(range(N_NODES), 3):
        for x, y, z in itertools.permutations([a, b, c]):
            if mask[y, x] and mask[z, x] and mask[z, y] and not mask[x, y] and not mask[x, z] and not mask[y, z]:
                direct_sign = np.sign(jac[z, x])
                indirect_sign = np.sign(jac[y, x]) * np.sign(jac[z, y])
                if direct_sign == indirect_sign:
                    coherent_ffl += 1
                break
        for x, y, z in [(a, b, c), (b, c, a), (c, a, b)]:
            if mask[y, x] and mask[z, y] and mask[x, z]:
                net_sign = np.sign(jac[y, x]) * np.sign(jac[z, y]) * np.sign(jac[x, z])
                if net_sign > 0:
                    positive_feedback += 1
                break
    return coherent_ffl, positive_feedback


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
    if n_edges > 0 and len(communities) > 1:
        modularity = nx.algorithms.community.quality.modularity(undirected, communities, weight="weight")
    else:
        modularity = 0.0

    weights = np.abs(jac)[mask]
    entropy = float(-np.sum((weights / weights.sum()) * np.log(weights / weights.sum() + 1e-12))) \
        if weights.sum() > 0 else 0.0

    coherent_ffl, positive_feedback = count_key_motifs(jac, mask)

    return {"jac": jac, "mask": mask, "n_edges": n_edges, "hub": hub,
            "modularity": modularity, "entropy": entropy,
            "coherent_ffl": coherent_ffl, "positive_feedback": positive_feedback}


def run_condition(strategy):
    w = base_weight_matrix()
    decay = np.ones(N_NODES)
    intervention = make_intervention(strategy)
    rhs = make_rhs(w, decay, intervention)
    autonomous_rhs = make_autonomous_rhs(w, decay)

    sol = solve_ivp(rhs, T_SPAN, np.zeros(N_NODES), dense_output=True, max_step=0.25)
    states = np.array([sol.sol(t) for t in SAMPLE_TIMES])
    deviation = np.linalg.norm(states, axis=1)

    peak_idx = int(np.argmax(deviation))
    peak_deviation = float(deviation[peak_idx])
    peak_time = float(SAMPLE_TIMES[peak_idx])

    half_peak = 0.5 * peak_deviation
    recovery_time = None
    for i in range(peak_idx, len(deviation)):
        if deviation[i] < half_peak:
            recovery_time = float(SAMPLE_TIMES[i] - peak_time)
            break

    post_injury_mask = SAMPLE_TIMES >= PULSE_CENTER
    integrated_response = float(trapezoid(deviation[post_injury_mask], SAMPLE_TIMES[post_injury_mask]))

    eval_state = sol.sol(EVAL_TIME)
    deviation_at_eval = float(np.linalg.norm(eval_state))

    eval_jac = numerical_jacobian(autonomous_rhs, eval_state)
    eval_report = graph_report(eval_jac)

    return {
        "strategy": strategy, "peak_deviation": peak_deviation, "peak_time": peak_time,
        "recovery_time": recovery_time, "integrated_response": integrated_response,
        "deviation_at_eval": deviation_at_eval, "eval_report": eval_report,
        "times": SAMPLE_TIMES, "deviation_series": deviation,
    }


def compare_to_baseline(eval_report, baseline_report):
    d_raw = float(np.linalg.norm(eval_report["jac"] - baseline_report["jac"]))
    gained = int(np.logical_and(eval_report["mask"], np.logical_not(baseline_report["mask"])).sum())
    lost = int(np.logical_and(baseline_report["mask"], np.logical_not(eval_report["mask"])).sum())
    return {
        "graph_distance": d_raw,
        "hub_restored": eval_report["hub"] == baseline_report["hub"],
        "coherent_ffl_restored": eval_report["coherent_ffl"] >= baseline_report["coherent_ffl"],
        "positive_feedback_restored": eval_report["positive_feedback"] >= baseline_report["positive_feedback"],
        "modularity_gap": abs(eval_report["modularity"] - baseline_report["modularity"]),
        "entropy_gap": abs(eval_report["entropy"] - baseline_report["entropy"]),
        "edges_gained": gained, "edges_lost": lost,
    }


def save_summary_csv(results, comparisons, baseline_report):
    path = os.path.join(OUTPUT_DIR, "v10_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "metric", "value"])
        writer.writerow(["baseline", "hub", baseline_report["hub"]])
        writer.writerow(["baseline", "modularity", f"{baseline_report['modularity']:.4f}"])
        writer.writerow(["baseline", "entropy", f"{baseline_report['entropy']:.4f}"])
        writer.writerow(["baseline", "coherent_ffl", baseline_report["coherent_ffl"]])
        writer.writerow(["baseline", "positive_feedback", baseline_report["positive_feedback"]])

        for result, comparison in zip(results, comparisons):
            strategy = result["strategy"]
            for key in ["peak_deviation", "recovery_time", "integrated_response", "deviation_at_eval"]:
                writer.writerow([strategy, key, result[key]])
            for key in ["graph_distance", "hub_restored", "coherent_ffl_restored",
                        "positive_feedback_restored", "modularity_gap", "entropy_gap",
                        "edges_gained", "edges_lost"]:
                writer.writerow([strategy, key, comparison[key]])
    return path


def plot_deviation_trajectories(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    for result in results:
        ax.plot(result["times"], result["deviation_series"], label=result["strategy"])
    ax.axvline(PULSE_CENTER, color="gray", linestyle="--", alpha=0.6, label="injury")
    ax.axvline(INTERVENTION_ONSET, color="black", linestyle=":", alpha=0.6, label="intervention onset")
    ax.axvline(EVAL_TIME, color="red", linestyle=":", alpha=0.6, label="evaluation time")
    ax.set_xlabel("time")
    ax.set_ylabel("deviation from baseline")
    ax.set_title("V10: biomarker trajectories under each intervention")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v10_deviation_trajectories.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_conventional_vs_network(results, comparisons):
    strategies = [r["strategy"] for r in results]
    deviation_at_eval = [r["deviation_at_eval"] for r in results]
    graph_distance = [c["graph_distance"] for c in comparisons]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    x = np.arange(len(strategies))
    axes[0].bar(x, deviation_at_eval, color="tab:blue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies, rotation=15)
    axes[0].set_title("conventional outcome:\ndeviation from baseline at evaluation time", fontsize=10)

    axes[1].bar(x, graph_distance, color="tab:orange", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies, rotation=15)
    axes[1].set_title("network outcome:\ngraph distance from baseline at evaluation time", fontsize=10)

    fig.suptitle("V10: same biomarker snapshot, different network state")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v10_conventional_vs_network.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_restoration_scorecard(comparisons, strategies):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(strategies))

    modularity_gaps = [c["modularity_gap"] for c in comparisons]
    entropy_gaps = [c["entropy_gap"] for c in comparisons]

    axes[0].bar(x, modularity_gaps, color="tab:green", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies, rotation=15)
    axes[0].set_title("modularity gap from baseline")

    axes[1].bar(x, entropy_gaps, color="tab:red", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies, rotation=15)
    axes[1].set_title("entropy gap from baseline")

    fig.suptitle("V10: continuous structural gaps by intervention\n"
                  "(binary hub and motif restoration flags were True for every "
                  "strategy at this evaluation time, so they are not shown)")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v10_restoration_scorecard.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    baseline_w = base_weight_matrix()
    baseline_autonomous_rhs = make_autonomous_rhs(baseline_w, np.ones(N_NODES))
    baseline_jac = numerical_jacobian(baseline_autonomous_rhs, np.zeros(N_NODES))
    baseline_report = graph_report(baseline_jac)
    print(f"Baseline: hub={baseline_report['hub']} modularity={baseline_report['modularity']:.3f} "
          f"entropy={baseline_report['entropy']:.3f} coherent_ffl={baseline_report['coherent_ffl']} "
          f"positive_feedback={baseline_report['positive_feedback']}")

    strategies = ["none", "symptomatic", "targeted", "mistargeted"]
    results = [run_condition(s) for s in strategies]
    comparisons = [compare_to_baseline(r["eval_report"], baseline_report) for r in results]

    print("\nConventional outcomes:")
    for result in results:
        print(f"  {result['strategy']:12s} peak={result['peak_deviation']:.3f} "
              f"recovery_time={result['recovery_time']} "
              f"integrated={result['integrated_response']:.2f} "
              f"deviation_at_eval={result['deviation_at_eval']:.4f}")

    print("\nNetwork outcomes at evaluation time:")
    for result, comparison in zip(results, comparisons):
        print(f"  {result['strategy']:12s} graph_distance={comparison['graph_distance']:.4f} "
              f"hub_restored={comparison['hub_restored']} "
              f"ffl_restored={comparison['coherent_ffl_restored']} "
              f"pf_restored={comparison['positive_feedback_restored']} "
              f"edges_lost={comparison['edges_lost']}")

    csv_path = save_summary_csv(results, comparisons, baseline_report)
    trajectory_plot = plot_deviation_trajectories(results)
    comparison_plot = plot_conventional_vs_network(results, comparisons)
    scorecard_plot = plot_restoration_scorecard(comparisons, strategies)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved trajectory plot to {trajectory_plot}")
    print(f"Saved conventional vs network plot to {comparison_plot}")
    print(f"Saved restoration scorecard to {scorecard_plot}")


if __name__ == "__main__":
    main()
