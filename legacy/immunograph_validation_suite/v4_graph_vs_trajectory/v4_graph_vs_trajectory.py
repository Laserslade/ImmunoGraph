"""
V4: Dynamic graph versus trajectory representation.
Tests whether sensitivity graphs carry information trajectories do not.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v4_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_NODES = 8
FD_EPS = 1e-5
N_PER_CONDITION = 30
TRAJ_SAMPLES = 10
GRAPH_SAMPLES = 6
T_SPAN = (0.0, 40.0)
N_PERMUTATIONS = 2000


def base_edges():
    # Shared wiring across both conditions, excluding the pair that
    # differs. Same feed forward and feedback structure as V3.
    edges = {
        (1, 0): 1.2, (2, 0): 0.9, (2, 1): 1.0,
        (4, 3): 1.1, (5, 4): 1.0, (3, 5): 0.95,
        (3, 2): 0.6, (6, 5): 0.5, (0, 7): -0.4,
    }
    return edges


def build_weight_matrix(condition, rng):
    w = np.zeros((N_NODES, N_NODES))
    for (i, j), val in base_edges().items():
        w[i, j] = val

    if condition == "A":
        w[7, 6] = -1.3
        w[6, 7] = -0.8
    else:
        w[7, 6] = 1.3
        w[6, 7] = 0.8

    jitter = rng.normal(0.0, 0.05, size=w.shape)
    mask = w != 0.0
    w = w + jitter * mask
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


def simulate_run(condition, seed):
    rng = np.random.default_rng(seed)
    w = build_weight_matrix(condition, rng)
    decay = np.ones(N_NODES)
    rhs = make_rhs(w, decay)
    y0 = rng.normal(0.0, 0.5, size=N_NODES)
    sol = solve_ivp(lambda t, y: rhs(y), T_SPAN, y0, dense_output=True)
    return rhs, sol


def trajectory_features(sol):
    times = np.linspace(T_SPAN[0], T_SPAN[1], TRAJ_SAMPLES)
    return np.concatenate([sol.sol(t) for t in times])


def graph_features(rhs, sol):
    times = np.linspace(T_SPAN[0], T_SPAN[1], GRAPH_SAMPLES)
    jacs = [numerical_jacobian(rhs, sol.sol(t)) for t in times]
    avg = np.mean(jacs, axis=0)
    off_diag = avg[~np.eye(N_NODES, dtype=bool)]
    return off_diag


def build_dataset():
    traj_X, graph_X, labels = [], [], []
    seed = 1000
    for condition in ["A", "B"]:
        for _ in range(N_PER_CONDITION):
            rhs, sol = simulate_run(condition, seed)
            traj_X.append(trajectory_features(sol))
            graph_X.append(graph_features(rhs, sol))
            labels.append(0 if condition == "A" else 1)
            seed += 1
    return np.array(traj_X), np.array(graph_X), np.array(labels)


def evaluate_classifiers(X, y, repr_name):
    rows = []
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=0),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for model_name, model in models.items():
        preds = cross_val_predict(model, X, y, cv=cv, method="predict")
        probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

        rows.append({
            "representation": repr_name,
            "model": model_name,
            "balanced_accuracy": balanced_accuracy_score(y, preds),
            "macro_f1": f1_score(y, preds, average="macro"),
            "auroc": roc_auc_score(y, probs),
        })
    return rows


def evaluate_clustering(X, y, repr_name):
    scaled = StandardScaler().fit_transform(X)
    clusters = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(scaled)
    observed_ari = adjusted_rand_score(y, clusters)

    rng = np.random.default_rng(0)
    null_scores = np.zeros(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        shuffled = rng.permutation(y)
        null_scores[i] = adjusted_rand_score(shuffled, clusters)

    p_value = float(np.mean(null_scores >= observed_ari))
    return {
        "representation": repr_name,
        "observed_ari": observed_ari,
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
        "p_value": p_value,
    }, null_scores


def build_switch_run():
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    w_a = build_weight_matrix("A", rng_a)
    w_b = build_weight_matrix("B", rng_b)
    decay = np.ones(N_NODES)
    rhs_a = make_rhs(w_a, decay)
    rhs_b = make_rhs(w_b, decay)

    switch_time = 20.0
    y0 = np.random.default_rng(7).normal(0.0, 0.5, size=N_NODES)
    sol_first = solve_ivp(lambda t, y: rhs_a(y), (0.0, switch_time), y0, dense_output=True)
    y_switch = sol_first.sol(switch_time)
    sol_second = solve_ivp(lambda t, y: rhs_b(y), (switch_time, T_SPAN[1]), y_switch, dense_output=True)

    def combined_state(t):
        return sol_first.sol(t) if t < switch_time else sol_second.sol(t)

    def combined_rhs(t):
        return rhs_a if t < switch_time else rhs_b

    return combined_state, combined_rhs, switch_time


def transition_detection_series():
    combined_state, combined_rhs, switch_time = build_switch_run()
    times = np.arange(1.0, T_SPAN[1], 1.0)

    states = [combined_state(t) for t in times]
    jacs = [numerical_jacobian(combined_rhs(t), combined_state(t)) for t in times]

    traj_distance = [np.linalg.norm(states[i + 1] - states[i]) for i in range(len(states) - 1)]
    graph_distance = [np.linalg.norm(jacs[i + 1] - jacs[i]) for i in range(len(jacs) - 1)]
    mid_times = times[1:]

    traj_detected = mid_times[int(np.argmax(traj_distance))]
    graph_detected = mid_times[int(np.argmax(graph_distance))]

    return {
        "times": mid_times,
        "traj_distance": traj_distance,
        "graph_distance": graph_distance,
        "switch_time": switch_time,
        "traj_detected": traj_detected,
        "graph_detected": graph_detected,
        "traj_error": abs(traj_detected - switch_time),
        "graph_error": abs(graph_detected - switch_time),
    }


def save_summary_csv(classification_rows, clustering_rows, transition_result):
    path = os.path.join(OUTPUT_DIR, "v4_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "representation", "model_or_metric", "value"])

        for row in classification_rows:
            for metric in ["balanced_accuracy", "macro_f1", "auroc"]:
                writer.writerow(["classification", row["representation"],
                                  f"{row['model']}_{metric}", f"{row[metric]:.4f}"])

        for row in clustering_rows:
            writer.writerow(["clustering", row["representation"], "observed_ari", f"{row['observed_ari']:.4f}"])
            writer.writerow(["clustering", row["representation"], "p_value", f"{row['p_value']:.4f}"])

        writer.writerow(["transition", "trajectory", "detected_time", f"{transition_result['traj_detected']:.1f}"])
        writer.writerow(["transition", "trajectory", "error", f"{transition_result['traj_error']:.1f}"])
        writer.writerow(["transition", "graph", "detected_time", f"{transition_result['graph_detected']:.1f}"])
        writer.writerow(["transition", "graph", "error", f"{transition_result['graph_error']:.1f}"])
    return path


def plot_classification_comparison(rows):
    reprs = sorted(set(r["representation"] for r in rows))
    models = sorted(set(r["model"] for r in rows))
    metrics = ["balanced_accuracy", "macro_f1", "auroc"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.5))
    x = np.arange(len(reprs))
    width = 0.35

    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(models):
            values = [next(r[metric] for r in rows if r["representation"] == rep and r["model"] == model)
                      for rep in reprs]
            ax.bar(x + (i - 0.5) * width, values, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(reprs)
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)

    axes[0].legend(fontsize=8)
    fig.suptitle("V4: condition classification, trajectory vs graph representation")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v4_classification_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pca_projection(traj_X, graph_X, y):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, X, title in zip(axes, [traj_X, graph_X], ["trajectory representation", "graph representation"]):
        scaled = StandardScaler().fit_transform(X)
        proj = PCA(n_components=2).fit_transform(scaled)
        for label, color, name in [(0, "tab:blue", "condition A"), (1, "tab:orange", "condition B")]:
            mask = y == label
            ax.scatter(proj[mask, 0], proj[mask, 1], color=color, label=name, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    axes[0].legend(fontsize=8)
    fig.suptitle("V4: PCA projection by representation")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v4_pca_projection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_clustering_permutation(clustering_rows, null_scores_dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, row in zip(axes, clustering_rows):
        null_scores = null_scores_dict[row["representation"]]
        ax.hist(null_scores, bins=40, color="gray", alpha=0.7, label="null distribution")
        ax.axvline(row["observed_ari"], color="red", linewidth=2, label="observed ARI")
        ax.set_title(f"{row['representation']} (p={row['p_value']:.4f})")
        ax.set_xlabel("adjusted rand index")

    axes[0].legend(fontsize=8)
    fig.suptitle("V4: clustering vs condition label, permutation test")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v4_clustering_permutation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_transition_detection(result):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(result["times"], result["traj_distance"], label="trajectory distance", color="tab:blue")
    ax.plot(result["times"], result["graph_distance"], label="graph distance", color="tab:orange")
    ax.axvline(result["switch_time"], color="black", linestyle="--", label="true switch time")
    ax.axvline(result["traj_detected"], color="tab:blue", linestyle=":", label="trajectory detected")
    ax.axvline(result["graph_detected"], color="tab:orange", linestyle=":", label="graph detected")
    ax.set_xlabel("time")
    ax.set_ylabel("consecutive-step distance")
    ax.set_title("V4: transition detection after a mid-run structural switch")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v4_transition_detection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    traj_X, graph_X, y = build_dataset()

    classification_rows = evaluate_classifiers(traj_X, y, "trajectory") + \
        evaluate_classifiers(graph_X, y, "graph")

    for row in classification_rows:
        print(f"{row['representation']:10s} {row['model']:20s} "
              f"bal_acc={row['balanced_accuracy']:.3f} "
              f"macro_f1={row['macro_f1']:.3f} auroc={row['auroc']:.3f}")

    traj_cluster, traj_null = evaluate_clustering(traj_X, y, "trajectory")
    graph_cluster, graph_null = evaluate_clustering(graph_X, y, "graph")
    clustering_rows = [traj_cluster, graph_cluster]
    null_scores_dict = {"trajectory": traj_null, "graph": graph_null}

    for row in clustering_rows:
        print(f"\n{row['representation']} clustering: observed_ari={row['observed_ari']:.4f} "
              f"null_mean={row['null_mean']:.4f} p_value={row['p_value']:.4f}")

    transition_result = transition_detection_series()
    print(f"\nTransition detection: true switch at t={transition_result['switch_time']}")
    print(f"  trajectory detected t={transition_result['traj_detected']:.1f} "
          f"(error {transition_result['traj_error']:.1f})")
    print(f"  graph detected t={transition_result['graph_detected']:.1f} "
          f"(error {transition_result['graph_error']:.1f})")

    csv_path = save_summary_csv(classification_rows, clustering_rows, transition_result)
    classification_plot = plot_classification_comparison(classification_rows)
    pca_plot = plot_pca_projection(traj_X, graph_X, y)
    permutation_plot = plot_clustering_permutation(clustering_rows, null_scores_dict)
    transition_plot = plot_transition_detection(transition_result)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved classification plot to {classification_plot}")
    print(f"Saved pca plot to {pca_plot}")
    print(f"Saved permutation plot to {permutation_plot}")
    print(f"Saved transition plot to {transition_plot}")


if __name__ == "__main__":
    main()
