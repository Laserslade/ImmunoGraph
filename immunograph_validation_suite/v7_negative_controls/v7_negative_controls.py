"""
V7: Negative control experiments.
Tests whether recovered structure exceeds what randomized or
signal free controls produce by chance.
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
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v7_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_NODES = 8
NODE_NAMES = [f"N{i}" for i in range(N_NODES)]
FD_EPS = 1e-5
N_TRIALS = 5000
OFF_DIAG_PAIRS = [(i, j) for i in range(N_NODES) for j in range(N_NODES) if i != j]

REAL_VALUES = [1.2, 0.9, 1.0, 1.1, 1.0, 0.95, -1.3, -0.8, 0.6, 0.5, -0.4]
N_EDGES = len(REAL_VALUES)


def base_weight_matrix():
    # Same wiring used across V3, V5, V6.
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

    mutual_inhibition = 0
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            if mask[i, j] and mask[j, i]:
                mutual_inhibition += 1
    return feedforward, feedback, mutual_inhibition


def count_motifs_signed(weight_matrix):
    mask = weight_matrix != 0.0
    feedforward, feedback, _ = count_motifs(mask)
    mutual_inhibition = 0
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            if mask[i, j] and mask[j, i]:
                if weight_matrix[i, j] < 0 and weight_matrix[j, i] < 0:
                    mutual_inhibition += 1
    return feedforward, feedback, mutual_inhibition


def real_motif_profile():
    return count_motifs_signed(base_weight_matrix())


def test1_random_topology_and_sign(rng):
    hits = 0
    profiles = []
    for _ in range(N_TRIALS):
        positions = rng.choice(len(OFF_DIAG_PAIRS), size=N_EDGES, replace=False)
        w = np.zeros((N_NODES, N_NODES))
        for idx in positions:
            i, j = OFF_DIAG_PAIRS[idx]
            sign = rng.choice([-1.0, 1.0])
            w[i, j] = sign
        ff, fb, mi = count_motifs_signed(w)
        profiles.append((ff, fb, mi))
        if ff >= 1 and fb >= 1 and mi >= 1:
            hits += 1
    return hits / N_TRIALS, profiles


def test2_value_position_permutation(rng):
    hits = 0
    profiles = []
    for _ in range(N_TRIALS):
        positions = rng.choice(len(OFF_DIAG_PAIRS), size=N_EDGES, replace=False)
        values = rng.permutation(REAL_VALUES)
        w = np.zeros((N_NODES, N_NODES))
        for idx, val in zip(positions, values):
            i, j = OFF_DIAG_PAIRS[idx]
            w[i, j] = val
        ff, fb, mi = count_motifs_signed(w)
        profiles.append((ff, fb, mi))
        if ff >= 1 and fb >= 1 and mi >= 1:
            hits += 1
    return hits / N_TRIALS, profiles


PULSES = [
    {"center": 15.0, "width": 2.0, "amplitude": 1.5, "nodes": [0, 3]},
    {"center": 45.0, "width": 1.5, "amplitude": 4.0, "nodes": [0, 3]},
]
T_SPAN = (0.0, 80.0)
DT = 0.5
SAMPLE_TIMES = np.arange(1.0, 80.0, DT)
THRESHOLD = 0.05
EVENT_WINDOW = 3.0


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


def graph_distance_series():
    w = base_weight_matrix()
    decay = np.ones(N_NODES)
    rhs = make_rhs(w, decay)
    autonomous_rhs = make_autonomous_rhs(w, decay)
    sol = solve_ivp(rhs, T_SPAN, np.zeros(N_NODES), dense_output=True, max_step=0.5)

    jacs = [numerical_jacobian(autonomous_rhs, sol.sol(t)) for t in SAMPLE_TIMES]
    distance = np.array([np.linalg.norm(jacs[i + 1] - jacs[i]) for i in range(len(jacs) - 1)])
    mid_times = SAMPLE_TIMES[1:]
    return distance, mid_times


def hit_fraction(distance, times, threshold):
    flagged = times[distance > threshold]
    if len(flagged) == 0:
        return 0.0, flagged
    near_event = np.zeros(len(flagged), dtype=bool)
    for pulse in PULSES:
        near_event |= np.abs(flagged - pulse["center"]) <= EVENT_WINDOW
    return float(np.mean(near_event)), flagged


def test3_temporal_shuffle(rng):
    distance, times = graph_distance_series()
    threshold = np.mean(distance) + 2.0 * np.std(distance)
    real_fraction, real_flagged = hit_fraction(distance, times, threshold)

    null_fractions = np.zeros(N_TRIALS // 5)
    for i in range(len(null_fractions)):
        shuffled_times = rng.permutation(times)
        frac, _ = hit_fraction(distance, shuffled_times, threshold)
        null_fractions[i] = frac

    p_value = float(np.mean(null_fractions >= real_fraction))
    return real_fraction, null_fractions, p_value, len(real_flagged)


def graph_snapshot_features(jac, mask):
    abs_weighted = np.abs(jac) * mask
    in_degree = abs_weighted.sum(axis=1)
    out_degree = abs_weighted.sum(axis=0)
    density = mask.sum() / (N_NODES * (N_NODES - 1))

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

    return [density, float(np.mean(in_degree)), float(np.mean(out_degree)),
            float(n_communities), entropy]


def test4_no_signal_control(rng):
    w = base_weight_matrix()
    decay = np.ones(N_NODES)
    autonomous_rhs = make_autonomous_rhs(w, decay)
    y0 = rng.normal(0.0, 1.5, size=N_NODES)
    sol = solve_ivp(lambda t, y: autonomous_rhs(y), T_SPAN, y0, dense_output=True, max_step=0.5)

    features = []
    for t in SAMPLE_TIMES:
        state = sol.sol(t)
        jac = numerical_jacobian(autonomous_rhs, state)
        abs_vals = np.abs(jac)
        np.fill_diagonal(abs_vals, 0.0)
        cutoff = THRESHOLD * np.max(abs_vals) if np.max(abs_vals) > 0 else 0.0
        mask = abs_vals > cutoff
        features.append(graph_snapshot_features(jac, mask))

    feature_matrix = np.array(features)
    scaled = StandardScaler().fit_transform(feature_matrix)
    clusters = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(scaled)
    score = silhouette_score(scaled, clusters)
    return score, sol


def save_summary_csv(real_profile, test1_p, test2_p, test3_result, test4_score):
    path = os.path.join(OUTPUT_DIR, "v7_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "metric", "value"])
        writer.writerow(["real_network", "feedforward", real_profile[0]])
        writer.writerow(["real_network", "feedback", real_profile[1]])
        writer.writerow(["real_network", "mutual_inhibition", real_profile[2]])
        writer.writerow(["test1_random_topology_sign", "joint_hit_p_value", f"{test1_p:.5f}"])
        writer.writerow(["test2_value_position_permutation", "joint_hit_p_value", f"{test2_p:.5f}"])
        writer.writerow(["test3_temporal_shuffle", "real_hit_fraction", f"{test3_result[0]:.4f}"])
        writer.writerow(["test3_temporal_shuffle", "n_flagged_points", test3_result[3]])
        writer.writerow(["test3_temporal_shuffle", "p_value", f"{test3_result[2]:.5f}"])
        writer.writerow(["test4_no_signal_control", "silhouette_k4", f"{test4_score:.4f}"])
        writer.writerow(["test4_no_signal_control", "reference_real_pulse_silhouette_k4_from_v5", "0.7976"])
    return path


def plot_motif_null_distributions(profiles1, profiles2, real_profile):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    labels = ["feedforward", "feedback", "mutual_inhibition"]

    for ax, profiles, title in zip(axes, [profiles1, profiles2],
                                    ["random topology and sign", "value position permutation"]):
        joint_counts = [sum(1 for p in profiles if p[k] >= 1) / len(profiles) for k in range(3)]
        x = np.arange(3)
        ax.bar(x, joint_counts, color="gray", alpha=0.8, label="fraction of trials with motif present")
        ax.axhline(1.0, color="red", linestyle="--", label="real network (always present)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)

    axes[0].legend(fontsize=8)
    fig.suptitle("V7: motif occurrence rate, real network vs randomized controls")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v7_motif_null_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_temporal_shuffle(real_fraction, null_fractions, p_value):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(null_fractions, bins=30, color="gray", alpha=0.8, label="shuffled time null")
    ax.axvline(real_fraction, color="red", linewidth=2, label="real (unshuffled) alignment")
    ax.set_xlabel("fraction of flagged points near a true pulse")
    ax.set_title(f"V7: temporal order shuffle control (p={p_value:.4f})")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v7_temporal_shuffle_control.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_no_signal_trajectory(sol, score):
    times = np.linspace(0, 80, 200)
    states = np.array([sol.sol(t) for t in times])

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(N_NODES):
        ax.plot(times, states[:, i], label=NODE_NAMES[i], alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("state value")
    ax.set_title(f"V7: no signal control trajectory, k=4 silhouette={score:.3f}")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v7_no_signal_trajectory.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    rng = np.random.default_rng(0)

    real_profile = real_motif_profile()
    print(f"Real network motif profile: feedforward={real_profile[0]} "
          f"feedback={real_profile[1]} mutual_inhibition={real_profile[2]}")

    test1_p, profiles1 = test1_random_topology_and_sign(rng)
    print(f"\nTest 1 (random topology and sign): joint hit p-value = {test1_p:.5f}")

    test2_p, profiles2 = test2_value_position_permutation(rng)
    print(f"Test 2 (value position permutation): joint hit p-value = {test2_p:.5f}")

    test3_result = test3_temporal_shuffle(rng)
    print(f"\nTest 3 (temporal order shuffle): real alignment = {test3_result[0]:.4f}, "
          f"n_flagged={test3_result[3]}, p-value = {test3_result[2]:.5f}")

    test4_score, sol = test4_no_signal_control(rng)
    print(f"\nTest 4 (no signal control): k=4 silhouette = {test4_score:.4f} "
          f"(reference, real pulse driven system from V5: 0.7976)")

    csv_path = save_summary_csv(real_profile, test1_p, test2_p, test3_result, test4_score)
    motif_plot = plot_motif_null_distributions(profiles1, profiles2, real_profile)
    shuffle_plot = plot_temporal_shuffle(test3_result[0], test3_result[1], test3_result[2])
    trajectory_plot = plot_no_signal_trajectory(sol, test4_score)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved motif null plot to {motif_plot}")
    print(f"Saved temporal shuffle plot to {shuffle_plot}")
    print(f"Saved no signal trajectory plot to {trajectory_plot}")


if __name__ == "__main__":
    main()
