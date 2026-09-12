"""
V15: Computational scaling analysis.
Measures how simulation, sensitivity, graph construction, and motif
counting time scale with problem size, and tracks memory use.
"""

import os
import csv
import time
import tracemalloc
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v15_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FD_EPS = 1e-5
THRESHOLD = 0.05
NODE_COUNTS = [4, 8, 16, 32, 64]
MOTIF_TIME_BUDGET_SECONDS = 5.0
TIME_SAMPLE_COUNTS = [5, 10, 20, 40, 80, 160, 320]
MC_RUN_COUNTS = [5, 10, 20, 40, 80, 160]
STORED_GRAPH_COUNTS = [10, 50, 100, 500, 1000]
N_REPS = 3


def build_random_network(n_nodes, density=0.2, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros((n_nodes, n_nodes))
    n_edges = int(density * n_nodes * (n_nodes - 1))
    off_diag_pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
    chosen = rng.choice(len(off_diag_pairs), size=min(n_edges, len(off_diag_pairs)), replace=False)
    for idx in chosen:
        i, j = off_diag_pairs[idx]
        w[i, j] = rng.uniform(0.3, 1.3) * rng.choice([-1.0, 1.0])
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


def build_edge_mask(jac):
    abs_vals = np.abs(jac)
    np.fill_diagonal(abs_vals, 0.0)
    cutoff = THRESHOLD * np.max(abs_vals) if np.max(abs_vals) > 0 else 0.0
    return abs_vals > cutoff


def count_motifs(mask):
    n = mask.shape[0]
    feedforward = 0
    feedback = 0
    for a, b, c in itertools.combinations(range(n), 3):
        for x, y, z in itertools.permutations([a, b, c]):
            if mask[y, x] and mask[z, x] and mask[z, y] and not mask[x, y] and not mask[x, z] and not mask[y, z]:
                feedforward += 1
                break
        for x, y, z in [(a, b, c), (b, c, a), (c, a, b)]:
            if mask[y, x] and mask[z, y] and mask[x, z]:
                feedback += 1
                break
    return feedforward, feedback


def time_call(fn, reps=N_REPS):
    times = []
    result = None
    for _ in range(reps):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), result


def fit_power_law(x_values, y_values):
    log_x = np.log(x_values)
    log_y = np.log(y_values)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    return slope, intercept


def run_node_scaling():
    rows = []
    motif_budget_exceeded = False

    for n_nodes in NODE_COUNTS:
        w = build_random_network(n_nodes, seed=n_nodes)
        decay = np.ones(n_nodes)
        rhs = make_rhs(w, decay)
        y0 = np.random.default_rng(1).normal(0.0, 0.5, size=n_nodes)

        ode_time, sol = time_call(lambda: solve_ivp(lambda t, y: rhs(y), (0.0, 20.0), y0, max_step=0.5))
        state = sol.sol(10.0) if sol.sol is not None else y0

        jac_time, jac = time_call(lambda: numerical_jacobian(rhs, state))
        mask_time, mask = time_call(lambda: build_edge_mask(jac))

        if not motif_budget_exceeded:
            motif_time, _ = time_call(lambda: count_motifs(mask), reps=1)
            if motif_time > MOTIF_TIME_BUDGET_SECONDS:
                motif_budget_exceeded = True
        else:
            motif_time = None

        rows.append({"n_nodes": n_nodes, "ode_time": ode_time, "jacobian_time": jac_time,
                     "mask_time": mask_time, "motif_time": motif_time})
        print(f"n_nodes={n_nodes}: ode={ode_time:.5f}s jacobian={jac_time:.5f}s "
              f"mask={mask_time:.5f}s motif={motif_time}")

    return rows


def run_time_sample_scaling(n_nodes=16):
    w = build_random_network(n_nodes, seed=42)
    decay = np.ones(n_nodes)
    rhs = make_rhs(w, decay)
    y0 = np.random.default_rng(2).normal(0.0, 0.5, size=n_nodes)
    _, sol = time_call(lambda: solve_ivp(lambda t, y: rhs(y), (0.0, 60.0), y0, dense_output=True, max_step=0.5), reps=1)

    rows = []
    for n_samples in TIME_SAMPLE_COUNTS:
        times = np.linspace(1.0, 59.0, n_samples)

        def compute_all():
            return [numerical_jacobian(rhs, sol.sol(t)) for t in times]

        elapsed, _ = time_call(compute_all)
        rows.append({"n_samples": n_samples, "total_time": elapsed})
        print(f"n_samples={n_samples}: total_jacobian_sequence_time={elapsed:.5f}s")

    return rows


def run_mc_scaling(n_nodes=8):
    rows = []
    for n_runs in MC_RUN_COUNTS:
        def run_batch():
            for i in range(n_runs):
                w = build_random_network(n_nodes, seed=i)
                decay = np.ones(n_nodes)
                rhs = make_rhs(w, decay)
                y0 = np.random.default_rng(i).normal(0.0, 0.5, size=n_nodes)
                sol = solve_ivp(lambda t, y: rhs(y), (0.0, 20.0), y0, max_step=0.5)
                jac = numerical_jacobian(rhs, sol.y[:, -1])
                build_edge_mask(jac)

        elapsed, _ = time_call(run_batch, reps=1)
        rows.append({"n_runs": n_runs, "total_time": elapsed, "time_per_run": elapsed / n_runs})
        print(f"n_runs={n_runs}: total_time={elapsed:.4f}s per_run={elapsed / n_runs:.5f}s")

    return rows


def run_memory_scaling():
    rows = []
    for n_nodes in NODE_COUNTS:
        tracemalloc.start()
        w = build_random_network(n_nodes, seed=n_nodes)
        decay = np.ones(n_nodes)
        rhs = make_rhs(w, decay)
        y0 = np.random.default_rng(1).normal(0.0, 0.5, size=n_nodes)
        jac = numerical_jacobian(rhs, y0)
        mask = build_edge_mask(jac)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({"n_nodes": n_nodes, "peak_memory_kb": peak / 1024})

    for n_stored in STORED_GRAPH_COUNTS:
        tracemalloc.start()
        n_nodes = 8
        graphs = []
        for i in range(n_stored):
            w = build_random_network(n_nodes, seed=i)
            decay = np.ones(n_nodes)
            rhs = make_rhs(w, decay)
            y0 = np.random.default_rng(i).normal(0.0, 0.5, size=n_nodes)
            jac = numerical_jacobian(rhs, y0)
            graphs.append(jac)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({"n_stored_graphs": n_stored, "peak_memory_kb": peak / 1024})

    return rows


def save_summary_csv(node_rows, time_sample_rows, mc_rows, memory_rows):
    path = os.path.join(OUTPUT_DIR, "v15_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "key", "value"])
        for row in node_rows:
            for key in ["ode_time", "jacobian_time", "mask_time", "motif_time"]:
                writer.writerow([f"node_scaling_n{row['n_nodes']}", key, row[key]])
        for row in time_sample_rows:
            writer.writerow([f"time_sample_scaling_n{row['n_samples']}", "total_time", row["total_time"]])
        for row in mc_rows:
            writer.writerow([f"mc_scaling_n{row['n_runs']}", "total_time", row["total_time"]])
            writer.writerow([f"mc_scaling_n{row['n_runs']}", "time_per_run", row["time_per_run"]])
        for row in memory_rows:
            key = f"n_nodes_{row['n_nodes']}" if "n_nodes" in row else f"n_stored_{row['n_stored_graphs']}"
            writer.writerow(["memory_scaling", key, row["peak_memory_kb"]])
    return path


def plot_node_scaling(node_rows):
    n_values = [r["n_nodes"] for r in node_rows]
    fig, ax = plt.subplots(figsize=(8, 6))

    for key, label in [("ode_time", "ODE simulation"), ("jacobian_time", "sensitivity (Jacobian)"),
                        ("mask_time", "graph construction")]:
        values = [r[key] for r in node_rows]
        ax.plot(n_values, values, marker="o", label=label)

    motif_n = [r["n_nodes"] for r in node_rows if r["motif_time"] is not None]
    motif_t = [r["motif_time"] for r in node_rows if r["motif_time"] is not None]
    if motif_t:
        ax.plot(motif_n, motif_t, marker="o", label="motif counting")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of nodes")
    ax.set_ylabel("time (seconds)")
    ax.set_title("V15: computation time versus number of state variables")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v15_node_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_time_sample_and_mc_scaling(time_sample_rows, mc_rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    n_samples = [r["n_samples"] for r in time_sample_rows]
    total_times = [r["total_time"] for r in time_sample_rows]
    axes[0].plot(n_samples, total_times, marker="o")
    axes[0].set_xlabel("number of time samples")
    axes[0].set_ylabel("total time (seconds)")
    axes[0].set_title("time series sensitivity, n_nodes=16")

    n_runs = [r["n_runs"] for r in mc_rows]
    total_mc_times = [r["total_time"] for r in mc_rows]
    axes[1].plot(n_runs, total_mc_times, marker="o", color="tab:orange")
    axes[1].set_xlabel("number of monte carlo runs")
    axes[1].set_ylabel("total time (seconds)")
    axes[1].set_title("batch pipeline, n_nodes=8")

    fig.suptitle("V15: linear scaling checks")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v15_time_sample_and_mc_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_memory_scaling(memory_rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    node_mem_rows = [r for r in memory_rows if "n_nodes" in r]
    stored_mem_rows = [r for r in memory_rows if "n_stored_graphs" in r]

    axes[0].plot([r["n_nodes"] for r in node_mem_rows], [r["peak_memory_kb"] for r in node_mem_rows], marker="o")
    axes[0].set_xlabel("number of nodes")
    axes[0].set_ylabel("peak memory (KB)")
    axes[0].set_title("memory versus node count, single graph")

    axes[1].plot([r["n_stored_graphs"] for r in stored_mem_rows],
                 [r["peak_memory_kb"] for r in stored_mem_rows], marker="o", color="tab:green")
    axes[1].set_xlabel("number of stored graphs")
    axes[1].set_ylabel("peak memory (KB)")
    axes[1].set_title("memory versus stored graph count, n_nodes=8")

    fig.suptitle("V15: memory scaling")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v15_memory_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    print("Node count scaling:")
    node_rows = run_node_scaling()

    print("\nTime sample scaling:")
    time_sample_rows = run_time_sample_scaling()

    print("\nMonte Carlo run scaling:")
    mc_rows = run_mc_scaling()

    print("\nMemory scaling:")
    memory_rows = run_memory_scaling()

    valid_node_rows = [r for r in node_rows if r["motif_time"] is not None]
    if len(valid_node_rows) >= 2:
        n_vals = [r["n_nodes"] for r in valid_node_rows]
        for key in ["ode_time", "jacobian_time", "mask_time"]:
            slope, _ = fit_power_law(n_vals, [r[key] for r in valid_node_rows])
            print(f"\nEmpirical scaling exponent for {key}: N^{slope:.2f}")
        motif_slope, _ = fit_power_law(n_vals, [r["motif_time"] for r in valid_node_rows])
        print(f"Empirical scaling exponent for motif_time: N^{motif_slope:.2f}")

    csv_path = save_summary_csv(node_rows, time_sample_rows, mc_rows, memory_rows)
    node_plot = plot_node_scaling(node_rows)
    ts_mc_plot = plot_time_sample_and_mc_scaling(time_sample_rows, mc_rows)
    memory_plot = plot_memory_scaling(memory_rows)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved node scaling plot to {node_plot}")
    print(f"Saved time sample and mc scaling plot to {ts_mc_plot}")
    print(f"Saved memory scaling plot to {memory_plot}")


if __name__ == "__main__":
    main()
