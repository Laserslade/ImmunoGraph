"""
V12: Synthetic ground truth benchmark.
Tests recovery of topology and its known temporal behavior on four
canonical circuits: chain, feedback loop, toggle switch, feed forward.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v12_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FD_EPS = 1e-5
THRESHOLD = 0.05
SAMPLE_COUNT = 40


def numerical_jacobian(rhs_state_only, y, eps=FD_EPS):
    n = len(y)
    jac = np.zeros((n, n))
    for j in range(n):
        y_plus = y.copy()
        y_minus = y.copy()
        y_plus[j] += eps
        y_minus[j] -= eps
        jac[:, j] = (rhs_state_only(y_plus) - rhs_state_only(y_minus)) / (2 * eps)
    return jac


def build_edge_mask(jac):
    abs_vals = np.abs(jac)
    np.fill_diagonal(abs_vals, 0.0)
    cutoff = THRESHOLD * np.max(abs_vals) if np.max(abs_vals) > 0 else 0.0
    return abs_vals > cutoff


def edge_metrics(true_matrix, jac, mask):
    n = true_matrix.shape[0]
    tp = fp = fn = sign_matches = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            true_edge = true_matrix[i, j] != 0.0
            recovered_edge = mask[i, j]
            if true_edge and recovered_edge:
                tp += 1
                if np.sign(true_matrix[i, j]) == np.sign(jac[i, j]):
                    sign_matches += 1
            elif recovered_edge and not true_edge:
                fp += 1
            elif true_edge and not recovered_edge:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    sign_accuracy = sign_matches / tp if tp > 0 else 1.0
    return {"precision": precision, "recall": recall, "f1": f1, "sign_accuracy": sign_accuracy}


def chain_system():
    n = 3
    names = ["A", "B", "C"]
    true_matrix = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0, 1.0, -1.0],
    ])

    def input_a(t):
        return 3.0 * np.exp(-0.5 * ((t - 5.0) / 1.0) ** 2)

    def rhs(t, y):
        return true_matrix @ y + np.array([input_a(t), 0.0, 0.0])

    def rhs_state_only(y):
        return true_matrix @ y

    t_span = (0.0, 40.0)
    y0 = np.zeros(n)
    return {"name": "chain", "n": n, "names": names, "true_matrix": true_matrix,
            "rhs": rhs, "rhs_state_only": rhs_state_only, "t_span": t_span, "y0": y0}


def feedback_system():
    n = 3
    names = ["A", "B", "C"]
    true_matrix = np.array([
        [-1.0, 0.0, 0.8],
        [0.9, -1.0, 0.0],
        [0.0, 0.85, -1.0],
    ])

    def input_a(t):
        return 2.0 * np.exp(-0.5 * ((t - 5.0) / 1.0) ** 2)

    def rhs(t, y):
        return true_matrix @ y + np.array([input_a(t), 0.0, 0.0])

    def rhs_state_only(y):
        return true_matrix @ y

    t_span = (0.0, 40.0)
    y0 = np.zeros(n)
    return {"name": "feedback", "n": n, "names": names, "true_matrix": true_matrix,
            "rhs": rhs, "rhs_state_only": rhs_state_only, "t_span": t_span, "y0": y0}


def toggle_system():
    n = 2
    names = ["A", "B"]
    k = 1.5

    def rhs_state_only(y):
        a, b = y
        return np.array([-a - k * np.tanh(b), -b - k * np.tanh(a)])

    def input_flip(t):
        # A strong transient push on B partway through, to force a flip.
        return 4.0 * np.exp(-0.5 * ((t - 25.0) / 1.0) ** 2)

    def rhs(t, y):
        base = rhs_state_only(y)
        return base + np.array([0.0, input_flip(t)])

    true_matrix = np.array([[-1.0, -k], [-k, -1.0]])
    t_span = (0.0, 45.0)
    y0 = np.array([1.0, -1.0])
    return {"name": "toggle", "n": n, "names": names, "true_matrix": true_matrix,
            "rhs": rhs, "rhs_state_only": rhs_state_only, "t_span": t_span, "y0": y0}


def feedforward_system():
    n = 3
    names = ["A", "B", "C"]
    true_matrix = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.7, 1.0, -1.0],
    ])

    def input_a(t):
        return 3.0 * np.exp(-0.5 * ((t - 5.0) / 1.0) ** 2)

    def rhs(t, y):
        return true_matrix @ y + np.array([input_a(t), 0.0, 0.0])

    def rhs_state_only(y):
        return true_matrix @ y

    t_span = (0.0, 40.0)
    y0 = np.zeros(n)
    return {"name": "feedforward", "n": n, "names": names, "true_matrix": true_matrix,
            "rhs": rhs, "rhs_state_only": rhs_state_only, "t_span": t_span, "y0": y0}


def run_system(system):
    sol = solve_ivp(system["rhs"], system["t_span"], system["y0"], dense_output=True, max_step=0.25)
    times = np.linspace(system["t_span"][0] + 0.5, system["t_span"][1] - 0.5, SAMPLE_COUNT)
    states = np.array([sol.sol(t) for t in times])

    jacs = [numerical_jacobian(system["rhs_state_only"], s) for s in states]
    masks = [build_edge_mask(j) for j in jacs]
    metrics = [edge_metrics(system["true_matrix"], j, m) for j, m in zip(jacs, masks)]

    return {"times": times, "states": states, "jacs": jacs, "masks": masks, "metrics": metrics}


def chain_temporal_check(system, run_result):
    states = run_result["states"]
    times = run_result["times"]
    peak_times = [times[int(np.argmax(states[:, i]))] for i in range(system["n"])]
    ordered_correctly = peak_times[0] <= peak_times[1] <= peak_times[2]
    return {"peak_times": dict(zip(system["names"], peak_times)), "correct_order": ordered_correctly}


def feedback_temporal_check(system, run_result):
    from itertools import permutations

    def has_cycle(mask):
        n = system["n"]
        for x, y, z in permutations(range(n)):
            if mask[y, x] and mask[z, y] and mask[x, z]:
                return True
        return False

    persistent = [has_cycle(m) for m in run_result["masks"]]
    fraction_persistent = float(np.mean(persistent))
    return {"fraction_time_cycle_present": fraction_persistent}


def toggle_temporal_check(system, run_result):
    states = run_result["states"]
    times = run_result["times"]
    dominant = ["A" if s[0] > s[1] else "B" for s in states]

    sign_consistent = []
    for jac, mask in zip(run_result["jacs"], run_result["masks"]):
        both_present = mask[0, 1] and mask[1, 0]
        both_negative = jac[0, 1] < 0 and jac[1, 0] < 0
        sign_consistent.append(bool(both_present and both_negative))

    flip_detected = dominant[0] == "A" and dominant[-1] == "B"
    flip_time = None
    for i in range(1, len(dominant)):
        if dominant[i] != dominant[i - 1]:
            flip_time = times[i]
            break

    return {"dominant_sequence": dominant, "sign_consistency_fraction": float(np.mean(sign_consistent)),
            "flip_detected": flip_detected, "flip_time": flip_time}


def feedforward_temporal_check(system, run_result):
    states = run_result["states"]
    times = run_result["times"]
    a_peak_time = times[int(np.argmax(states[:, 0]))]
    b_peak_time = times[int(np.argmax(states[:, 1]))]
    c_peak_time = times[int(np.argmax(states[:, 2]))]
    return {"a_peak_time": a_peak_time, "b_peak_time": b_peak_time, "c_peak_time": c_peak_time,
            "b_lags_a": b_peak_time >= a_peak_time, "c_lags_a": c_peak_time >= a_peak_time,
            "b_vs_c_gap": float(c_peak_time - b_peak_time)}


SYSTEMS = {
    "chain": (chain_system, chain_temporal_check),
    "feedback": (feedback_system, feedback_temporal_check),
    "toggle": (toggle_system, toggle_temporal_check),
    "feedforward": (feedforward_system, feedforward_temporal_check),
}


def save_summary_csv(all_results):
    path = os.path.join(OUTPUT_DIR, "v12_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "metric", "value"])
        for name, (run_result, temporal_result) in all_results.items():
            avg_metrics = {
                key: float(np.mean([m[key] for m in run_result["metrics"]]))
                for key in ["precision", "recall", "f1", "sign_accuracy"]
            }
            for key, value in avg_metrics.items():
                writer.writerow([name, f"topology_{key}_mean", f"{value:.4f}"])
            for key, value in temporal_result.items():
                writer.writerow([name, key, value])
    return path


def plot_topology_recovery(all_results):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (name, (run_result, _)) in zip(axes, all_results.items()):
        times = run_result["times"]
        precision = [m["precision"] for m in run_result["metrics"]]
        recall = [m["recall"] for m in run_result["metrics"]]
        f1 = [m["f1"] for m in run_result["metrics"]]
        ax.plot(times, precision, label="precision")
        ax.plot(times, recall, label="recall")
        ax.plot(times, f1, label="f1")
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(name)
        ax.set_xlabel("time")

    axes[0].legend(fontsize=7)
    fig.suptitle("V12: topology recovery accuracy over time, four canonical circuits")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v12_topology_recovery.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_trajectories(all_results):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    systems = {name: fn() for name, (fn, _) in SYSTEMS.items()}
    for ax, (name, (run_result, _)) in zip(axes, all_results.items()):
        states = run_result["states"]
        times = run_result["times"]
        for i, node_name in enumerate(systems[name]["names"]):
            ax.plot(times, states[:, i], label=node_name)
        ax.set_title(name)
        ax.set_xlabel("time")
        ax.legend(fontsize=7)

    fig.suptitle("V12: state trajectories, four canonical circuits")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v12_trajectories.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_toggle_detail(toggle_run, toggle_temporal):
    states = toggle_run["states"]
    times = toggle_run["times"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(times, states[:, 0], label="A")
    axes[0].plot(times, states[:, 1], label="B")
    if toggle_temporal["flip_time"] is not None:
        axes[0].axvline(toggle_temporal["flip_time"], color="red", linestyle="--", label="detected flip")
    axes[0].set_title("toggle switch state trajectories")
    axes[0].set_xlabel("time")
    axes[0].legend(fontsize=8)

    sign_consistency = [1 if (j[0, 1] < 0 and j[1, 0] < 0) else 0 for j in toggle_run["jacs"]]
    axes[1].plot(times, sign_consistency, marker="o", markersize=3)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_title("mutual inhibition sign consistency over time")
    axes[1].set_xlabel("time")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v12_toggle_detail.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    all_results = {}
    for name, (system_fn, temporal_fn) in SYSTEMS.items():
        system = system_fn()
        run_result = run_system(system)
        temporal_result = temporal_fn(system, run_result)
        all_results[name] = (run_result, temporal_result)

        avg_f1 = np.mean([m["f1"] for m in run_result["metrics"]])
        avg_sign_acc = np.mean([m["sign_accuracy"] for m in run_result["metrics"]])
        print(f"{name}: mean topology F1={avg_f1:.4f}, mean sign accuracy={avg_sign_acc:.4f}")
        print(f"  temporal check: {temporal_result}")

    csv_path = save_summary_csv(all_results)
    topology_plot = plot_topology_recovery(all_results)
    trajectory_plot = plot_trajectories(all_results)
    toggle_plot = plot_toggle_detail(*all_results["toggle"])

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved topology recovery plot to {topology_plot}")
    print(f"Saved trajectory plot to {trajectory_plot}")
    print(f"Saved toggle detail plot to {toggle_plot}")


if __name__ == "__main__":
    main()
