"""
V6: Temporal network reorganization.
Tests whether topology change coincides with, precedes, or follows
changes in the underlying trajectory.
"""

import os
import csv
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v6_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_NODES = 8
NODE_NAMES = [f"N{i}" for i in range(N_NODES)]
FD_EPS = 1e-5
T_SPAN = (0.0, 80.0)
DT = 0.5
SAMPLE_TIMES = np.arange(1.0, 80.0, DT)
THRESHOLD = 0.05
MAX_LAG_STEPS = 20

PULSES = [
    {"center": 15.0, "width": 2.0, "amplitude": 1.5, "nodes": [0, 3]},
    {"center": 45.0, "width": 1.5, "amplitude": 4.0, "nodes": [0, 3]},
]


def build_weight_matrix():
    # Same wiring as V3 and V5, single condition.
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
    for pulse in PULSES:
        bump = pulse["amplitude"] * np.exp(-0.5 * ((t - pulse["center"]) / pulse["width"]) ** 2)
        for node in pulse["nodes"]:
            u[node] += bump
    return u


def make_rhs(w, decay):
    def rhs(t, y):
        return -decay * y + w @ np.tanh(y) + pulse_input(t)
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


def count_motifs(mask):
    feedforward = 0
    feedback = 0
    for a, b, c in itertools.combinations(range(N_NODES), 3):
        for x, y, z in itertools.permutations([a, b, c]):
            if mask[y, x] and mask[z, x] and mask[z, y] and not mask[x, y] and not mask[x, z] and not mask[y, z]:
                feedforward += 1
                break
        for x, y, z in [(a, b, c), (b, c, a), (c, a, b)]:
            if mask[y, x] and mask[z, y] and mask[x, z]:
                feedback += 1
                break
    return feedforward, feedback


def snapshot_stats(jac, mask):
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

    weights = np.abs(jac)[mask]
    if weights.sum() > 0:
        probs = weights / weights.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        entropy = 0.0

    feedforward, feedback = count_motifs(mask)

    return {"hub": hub, "n_communities": n_communities, "entropy": entropy,
            "feedforward": feedforward, "feedback": feedback, "n_edges": int(mask.sum())}


def build_series(sol, autonomous_rhs):
    states = [sol.sol(t) for t in SAMPLE_TIMES]
    jacs = [numerical_jacobian(autonomous_rhs, s) for s in states]
    masks = [build_edge_mask(j) for j in jacs]
    stats = [snapshot_stats(j, m) for j, m in zip(jacs, masks)]

    graph_distance = [np.linalg.norm(jacs[i + 1] - jacs[i]) for i in range(len(jacs) - 1)]
    jaccard_distance = []
    for i in range(len(masks) - 1):
        union = np.logical_or(masks[i], masks[i + 1]).sum()
        inter = np.logical_and(masks[i], masks[i + 1]).sum()
        jaccard_distance.append(1.0 - (inter / union if union > 0 else 1.0))

    traj_distance = [np.linalg.norm(states[i + 1] - states[i]) for i in range(len(states) - 1)]

    baseline_mean = np.mean(states[:15], axis=0)
    deviation = [np.linalg.norm(s - baseline_mean) for s in states]

    return {
        "states": states, "jacs": jacs, "masks": masks, "stats": stats,
        "graph_distance": np.array(graph_distance), "jaccard_distance": np.array(jaccard_distance),
        "traj_distance": np.array(traj_distance), "deviation": np.array(deviation),
        "mid_times": SAMPLE_TIMES[1:],
    }


def find_reorganization_points(distance_series, times, n_std=2.0):
    threshold = np.mean(distance_series) + n_std * np.std(distance_series)
    flagged = times[distance_series > threshold]
    return threshold, flagged


def find_transitions(values, times):
    transitions = []
    for i in range(1, len(values)):
        if values[i] != values[i - 1]:
            transitions.append((times[i], values[i - 1], values[i]))
    return transitions


def cross_correlation(a, b, max_lag):
    a = (a - np.mean(a)) / np.std(a)
    b = (b - np.mean(b)) / np.std(b)
    lags = range(-max_lag, max_lag + 1)
    correlations = []
    for lag in lags:
        if lag < 0:
            correlations.append(np.corrcoef(a[-lag:], b[:len(b) + lag])[0, 1])
        elif lag > 0:
            correlations.append(np.corrcoef(a[:len(a) - lag], b[lag:])[0, 1])
        else:
            correlations.append(np.corrcoef(a, b)[0, 1])
    return list(lags), correlations


def nearest_peak_offset(distance_series, times, event_time, search_window=8.0):
    mask = (times > event_time - search_window) & (times < event_time + search_window)
    local_times = times[mask]
    local_series = distance_series[mask]
    peak_time = local_times[int(np.argmax(local_series))]
    return peak_time - event_time


def save_summary_csv(series, reorg_threshold, reorg_points, hub_transitions,
                      community_transitions, lags, correlations, event_offsets):
    path = os.path.join(OUTPUT_DIR, "v6_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "key", "value"])
        writer.writerow(["reorganization", "threshold", f"{reorg_threshold:.4f}"])
        writer.writerow(["reorganization", "n_flagged_points", len(reorg_points)])
        for t in reorg_points:
            writer.writerow(["reorganization", "flagged_time", f"{t:.1f}"])
        for t, old, new in hub_transitions:
            writer.writerow(["hub_transition", f"time_{t:.1f}", f"{old}_to_{new}"])
        for t, old, new in community_transitions:
            writer.writerow(["community_transition", f"time_{t:.1f}", f"{old}_to_{new}"])

        best_lag_idx = int(np.argmax(correlations))
        writer.writerow(["lead_lag", "best_lag_steps", lags[best_lag_idx]])
        writer.writerow(["lead_lag", "best_lag_time", f"{lags[best_lag_idx] * DT:.2f}"])
        writer.writerow(["lead_lag", "best_correlation", f"{correlations[best_lag_idx]:.4f}"])

        for i, offsets in enumerate(event_offsets):
            writer.writerow([f"pulse_{i+1}", "graph_offset", f"{offsets['graph']:.2f}"])
            writer.writerow([f"pulse_{i+1}", "trajectory_offset", f"{offsets['traj']:.2f}"])
    return path


def plot_reorganization_timeline(series, reorg_threshold, reorg_points):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(SAMPLE_TIMES, series["deviation"], color="black")
    for pulse in PULSES:
        axes[0].axvline(pulse["center"], color="gray", linestyle="--", alpha=0.6)
    axes[0].set_title("trajectory deviation from baseline")

    axes[1].plot(series["mid_times"], series["graph_distance"], color="tab:orange")
    axes[1].axhline(reorg_threshold, color="red", linestyle=":", label="reorganization threshold")
    for t in reorg_points:
        axes[1].axvline(t, color="red", alpha=0.15)
    axes[1].set_title("graph distance D(G_t, G_t+1), flagged reorganization points in red")
    axes[1].legend(fontsize=8)

    hubs = [s["hub"] for s in series["stats"]]
    unique_hubs = sorted(set(hubs))
    hub_idx = [unique_hubs.index(h) for h in hubs]
    axes[2].scatter(SAMPLE_TIMES, hub_idx, s=10, color="tab:green")
    axes[2].set_yticks(range(len(unique_hubs)))
    axes[2].set_yticklabels(unique_hubs)
    axes[2].set_title("top hub identity over time")

    entropy = [s["entropy"] for s in series["stats"]]
    axes[3].plot(SAMPLE_TIMES, entropy, color="tab:purple", label="entropy")
    ax2 = axes[3].twinx()
    feedback_counts = [s["feedback"] for s in series["stats"]]
    ax2.plot(SAMPLE_TIMES, feedback_counts, color="tab:brown", label="feedback motifs", alpha=0.6)
    axes[3].set_title("network entropy and feedback motif count")
    axes[3].set_xlabel("time")
    axes[3].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("V6: temporal network reorganization")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v6_reorganization_timeline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lead_lag(lags, correlations):
    fig, ax = plt.subplots(figsize=(7, 5))
    times = [lag * DT for lag in lags]
    ax.plot(times, correlations, marker="o", markersize=3)
    best_idx = int(np.argmax(correlations))
    ax.axvline(times[best_idx], color="red", linestyle="--",
               label=f"best lag = {times[best_idx]:.1f}")
    ax.axvline(0.0, color="gray", linestyle=":")
    ax.set_xlabel("lag (positive: graph distance shifted forward in time)")
    ax.set_ylabel("correlation with trajectory distance")
    ax.set_title("V6: graph distance vs trajectory distance, cross correlation")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v6_lead_lag_correlation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_event_offsets(event_offsets):
    labels = [f"pulse {i+1}\n(t={p['center']})" for i, p in enumerate(PULSES)]
    graph_offsets = [o["graph"] for o in event_offsets]
    traj_offsets = [o["traj"] for o in event_offsets]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, graph_offsets, width, label="graph distance peak offset")
    ax.bar(x + width / 2, traj_offsets, width, label="trajectory distance peak offset")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("time offset from pulse center")
    ax.set_title("V6: per event timing, negative means the signal peaked early")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v6_event_offsets.png")
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

    series = build_series(sol, autonomous_rhs)

    reorg_threshold, reorg_points = find_reorganization_points(series["graph_distance"], series["mid_times"])
    print(f"Reorganization threshold: {reorg_threshold:.4f}")
    print(f"Flagged reorganization times: {[round(t, 1) for t in reorg_points]}")

    hubs = [s["hub"] for s in series["stats"]]
    hub_transitions = find_transitions(hubs, SAMPLE_TIMES)
    print(f"\nHub transitions: {hub_transitions}")

    communities = [s["n_communities"] for s in series["stats"]]
    community_transitions = find_transitions(communities, SAMPLE_TIMES)
    print(f"Community count transitions: {community_transitions}")

    lags, correlations = cross_correlation(series["graph_distance"], series["traj_distance"], MAX_LAG_STEPS)
    best_idx = int(np.argmax(correlations))
    print(f"\nBest lag: {lags[best_idx] * DT:.2f} time units, "
          f"correlation={correlations[best_idx]:.4f}")

    event_offsets = []
    for pulse in PULSES:
        graph_offset = nearest_peak_offset(series["graph_distance"], series["mid_times"], pulse["center"])
        traj_offset = nearest_peak_offset(series["traj_distance"], series["mid_times"], pulse["center"])
        event_offsets.append({"center": pulse["center"], "graph": graph_offset, "traj": traj_offset})
        print(f"Pulse at t={pulse['center']}: graph peak offset={graph_offset:.2f}, "
              f"trajectory peak offset={traj_offset:.2f}")

    csv_path = save_summary_csv(series, reorg_threshold, reorg_points, hub_transitions,
                                 community_transitions, lags, correlations, event_offsets)
    timeline_plot = plot_reorganization_timeline(series, reorg_threshold, reorg_points)
    lead_lag_plot = plot_lead_lag(lags, correlations)
    offsets_plot = plot_event_offsets(event_offsets)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved timeline plot to {timeline_plot}")
    print(f"Saved lead lag plot to {lead_lag_plot}")
    print(f"Saved event offsets plot to {offsets_plot}")


if __name__ == "__main__":
    main()
