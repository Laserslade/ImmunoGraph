"""
V2: Graph Construction Validation.
Tests whether sensitivity-derived graphs recover known ground truth interactions.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v2_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REL_THRESHOLD = 0.05
N_TIME_SAMPLES = 6
FD_EPS = 1e-5


def build_systems():
    # Each system is linear and time invariant, so its Jacobian
    # equals true_matrix everywhere. This isolates the graph
    # construction step from confounds of time varying dynamics.
    systems = []

    systems.append({
        "name": "positive_negative_asymmetric",
        "vars": ["x1", "x2"],
        "true_matrix": np.array([
            [-1.0, 4.0],
            [-0.2, -1.0],
        ]),
        "y0": np.array([1.0, 1.0]),
        "tags": ["positive", "negative", "asymmetric", "strong", "weak"],
    })

    systems.append({
        "name": "feedforward",
        "vars": ["A", "B", "C"],
        "true_matrix": np.array([
            [-1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.8, 1.2, -1.0],
        ]),
        "y0": np.array([1.0, 0.5, 0.2]),
        "tags": ["feed-forward", "positive"],
    })

    systems.append({
        "name": "feedback_loop",
        "vars": ["A", "B", "C"],
        "true_matrix": np.array([
            [-1.0, 0.0, 0.9],
            [1.1, -1.0, 0.0],
            [0.0, 0.7, -1.0],
        ]),
        "y0": np.array([1.0, 0.3, 0.6]),
        "tags": ["feedback", "positive"],
    })

    systems.append({
        "name": "mutual_inhibition",
        "vars": ["x1", "x2"],
        "true_matrix": np.array([
            [-1.0, -1.5],
            [-0.9, -1.0],
        ]),
        "y0": np.array([1.0, 0.8]),
        "tags": ["mutual_inhibition", "negative"],
    })

    systems.append({
        "name": "indirect_chain",
        "vars": ["A", "B", "C"],
        "true_matrix": np.array([
            [-1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, -1.0],
        ]),
        "y0": np.array([1.0, 0.4, 0.1]),
        "tags": ["indirect", "positive"],
    })

    return systems


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


def recover_sensitivity(true_matrix, y0, t_span=(0.0, 10.0)):
    n = len(y0)
    rhs = lambda y: true_matrix @ y
    sol = solve_ivp(lambda t, y: rhs(y), t_span, y0, dense_output=True)
    sample_times = np.linspace(t_span[0], t_span[1], N_TIME_SAMPLES)

    jac_samples = []
    for t in sample_times:
        y_t = sol.sol(t)
        jac_samples.append(numerical_jacobian(rhs, y_t))
    return np.mean(jac_samples, axis=0)


def compute_metrics(true_matrix, recovered_matrix):
    n = true_matrix.shape[0]
    threshold = REL_THRESHOLD * np.max(np.abs(recovered_matrix - np.diag(np.diag(recovered_matrix))))

    tp = fp = fn = 0
    sign_matches = 0
    true_vals = []
    recovered_vals = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            true_edge = true_matrix[i, j] != 0.0
            recovered_edge = abs(recovered_matrix[i, j]) > threshold

            if true_edge and recovered_edge:
                tp += 1
                if np.sign(true_matrix[i, j]) == np.sign(recovered_matrix[i, j]):
                    sign_matches += 1
                true_vals.append(true_matrix[i, j])
                recovered_vals.append(recovered_matrix[i, j])
            elif recovered_edge and not true_edge:
                fp += 1
            elif true_edge and not recovered_edge:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    sign_accuracy = sign_matches / tp if tp > 0 else 0.0

    correlation = np.nan
    if len(true_vals) >= 2:
        correlation, _ = pearsonr(true_vals, recovered_vals)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sign_accuracy": sign_accuracy,
        "correlation": correlation,
        "true_vals": true_vals,
        "recovered_vals": recovered_vals,
    }


def save_summary_csv(results):
    path = os.path.join(OUTPUT_DIR, "v2_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "precision", "recall", "f1", "sign_accuracy", "correlation", "tags"])
        for r in results:
            writer.writerow([
                r["name"], f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                f"{r['f1']:.4f}", f"{r['sign_accuracy']:.4f}",
                f"{r['correlation']:.4f}" if not np.isnan(r["correlation"]) else "nan",
                ";".join(r["tags"]),
            ])
    return path


def plot_metrics_summary(results):
    names = [r["name"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    f1 = [r["f1"] for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="precision")
    ax.bar(x, recall, width, label="recall")
    ax.bar(x + width, f1, width, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score")
    ax.set_title("V2: edge recovery metrics by system")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v2_metrics_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_adjacency_comparison(systems, recovered_matrices):
    n_systems = len(systems)
    fig, axes = plt.subplots(n_systems, 2, figsize=(7, 3.3 * n_systems))

    for idx, (system, recovered) in enumerate(zip(systems, recovered_matrices)):
        true_matrix = system["true_matrix"]
        vmax = max(np.max(np.abs(true_matrix)), np.max(np.abs(recovered)))

        ax_true = axes[idx, 0]
        ax_rec = axes[idx, 1]

        ax_true.imshow(true_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_true.set_title(f"{system['name']} (true)", fontsize=9)
        ax_true.set_xticks(range(len(system["vars"])))
        ax_true.set_yticks(range(len(system["vars"])))
        ax_true.set_xticklabels(system["vars"], fontsize=8)
        ax_true.set_yticklabels(system["vars"], fontsize=8)

        ax_rec.imshow(recovered, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_rec.set_title(f"(recovered)", fontsize=9)
        ax_rec.set_xticks(range(len(system["vars"])))
        ax_rec.set_yticks(range(len(system["vars"])))
        ax_rec.set_xticklabels(system["vars"], fontsize=8)
        ax_rec.set_yticklabels(system["vars"], fontsize=8)

    fig.tight_layout(h_pad=2.0)
    path = os.path.join(OUTPUT_DIR, "v2_adjacency_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_magnitude_correlation(results):
    fig, ax = plt.subplots(figsize=(6, 6))
    markers = ["o", "s", "^", "D", "v"]

    for i, r in enumerate(results):
        if len(r["true_vals"]) == 0:
            continue
        ax.scatter(
            r["true_vals"], r["recovered_vals"],
            label=r["name"], marker=markers[i % len(markers)], s=60,
        )

    all_vals = [v for r in results for v in r["true_vals"]]
    if all_vals:
        lims = [min(all_vals) - 0.5, max(all_vals) + 0.5]
        ax.plot(lims, lims, linestyle="--", color="gray")

    ax.set_xlabel("true coefficient")
    ax.set_ylabel("recovered sensitivity")
    ax.set_title("V2: true vs recovered edge strength")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v2_magnitude_correlation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    systems = build_systems()
    results = []
    recovered_matrices = []

    for system in systems:
        recovered = recover_sensitivity(system["true_matrix"], system["y0"])
        recovered_matrices.append(recovered)

        metrics = compute_metrics(system["true_matrix"], recovered)
        metrics["name"] = system["name"]
        metrics["tags"] = system["tags"]
        results.append(metrics)

        print(f"{system['name']}: precision={metrics['precision']:.3f} "
              f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
              f"sign_acc={metrics['sign_accuracy']:.3f} "
              f"corr={metrics['correlation']:.3f}")

    csv_path = save_summary_csv(results)
    metrics_plot_path = plot_metrics_summary(results)
    adjacency_plot_path = plot_adjacency_comparison(systems, recovered_matrices)
    correlation_plot_path = plot_magnitude_correlation(results)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved metrics plot to {metrics_plot_path}")
    print(f"Saved adjacency comparison plot to {adjacency_plot_path}")
    print(f"Saved magnitude correlation plot to {correlation_plot_path}")


if __name__ == "__main__":
    main()
