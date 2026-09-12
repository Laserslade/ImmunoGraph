"""
V11: Representation ablation.
Tests which components of the graph representation, temporality,
sign, weighting, or topology, are necessary for the observed effect.
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v11_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_NODES = 8
FD_EPS = 1e-5
N_PER_CONDITION = 30
TRAJ_SAMPLES = 10
GRAPH_SAMPLES = 6
STATIC_SNAPSHOT_TIME = 20.0
T_SPAN = (0.0, 40.0)
N_PERMUTATIONS = 2000
THRESHOLD = 0.05

REPRESENTATION_ORDER = [
    "raw_trajectory", "static_graph", "dynamic_unweighted",
    "dynamic_weighted_unsigned", "signed_dynamic", "signed_weighted_temporal",
]


def base_edges():
    return {
        (1, 0): 1.2, (2, 0): 0.9, (2, 1): 1.0,
        (4, 3): 1.1, (5, 4): 1.0, (3, 5): 0.95,
        (3, 2): 0.6, (6, 5): 0.5, (0, 7): -0.4,
    }


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
    return w + jitter * mask


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


def off_diag(matrix):
    return matrix[~np.eye(N_NODES, dtype=bool)]


def extract_representations(rhs, sol):
    times = np.linspace(T_SPAN[0], T_SPAN[1], TRAJ_SAMPLES)
    raw_trajectory = np.concatenate([sol.sol(t) for t in times])

    static_jac = numerical_jacobian(rhs, sol.sol(STATIC_SNAPSHOT_TIME))
    static_graph = off_diag(static_jac)

    graph_times = np.linspace(T_SPAN[0], T_SPAN[1], GRAPH_SAMPLES)
    jacs = [numerical_jacobian(rhs, sol.sol(t)) for t in graph_times]
    avg_jac = np.mean(jacs, axis=0)
    avg_off = off_diag(avg_jac)

    cutoff = THRESHOLD * np.max(np.abs(avg_off)) if np.max(np.abs(avg_off)) > 0 else 0.0
    mask = np.abs(avg_off) > cutoff

    dynamic_unweighted = mask.astype(float)
    dynamic_weighted_unsigned = np.abs(avg_off) * mask
    signed_dynamic = np.sign(avg_off) * mask
    signed_weighted_temporal = avg_off

    return {
        "raw_trajectory": raw_trajectory,
        "static_graph": static_graph,
        "dynamic_unweighted": dynamic_unweighted,
        "dynamic_weighted_unsigned": dynamic_weighted_unsigned,
        "signed_dynamic": signed_dynamic,
        "signed_weighted_temporal": signed_weighted_temporal,
    }


def build_dataset():
    data = {name: [] for name in REPRESENTATION_ORDER}
    labels = []
    seed = 2000
    for condition in ["A", "B"]:
        for _ in range(N_PER_CONDITION):
            rhs, sol = simulate_run(condition, seed)
            reps = extract_representations(rhs, sol)
            for name in REPRESENTATION_ORDER:
                data[name].append(reps[name])
            labels.append(0 if condition == "A" else 1)
            seed += 1
    return {name: np.array(vals) for name, vals in data.items()}, np.array(labels)


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
            "representation": repr_name, "model": model_name,
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
    null_scores = np.array([adjusted_rand_score(rng.permutation(y), clusters) for _ in range(N_PERMUTATIONS)])
    p_value = float(np.mean(null_scores >= observed_ari))

    return {"representation": repr_name, "observed_ari": observed_ari, "p_value": p_value}


def save_summary_csv(classification_rows, clustering_rows):
    path = os.path.join(OUTPUT_DIR, "v11_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["representation", "metric", "value"])
        for row in classification_rows:
            for metric in ["balanced_accuracy", "macro_f1", "auroc"]:
                writer.writerow([row["representation"], f"{row['model']}_{metric}", f"{row[metric]:.4f}"])
        for row in clustering_rows:
            writer.writerow([row["representation"], "clustering_ari", f"{row['observed_ari']:.4f}"])
            writer.writerow([row["representation"], "clustering_p_value", f"{row['p_value']:.4f}"])
    return path


def plot_classification_ladder(classification_rows):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["balanced_accuracy", "macro_f1", "auroc"]
    models = ["logistic_regression", "random_forest"]

    x = np.arange(len(REPRESENTATION_ORDER))
    width = 0.35

    for ax, metric in zip(axes, metrics):
        for i, model in enumerate(models):
            values = [next(r[metric] for r in classification_rows
                           if r["representation"] == rep and r["model"] == model)
                      for rep in REPRESENTATION_ORDER]
            ax.bar(x + (i - 0.5) * width, values, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(REPRESENTATION_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)

    axes[0].legend(fontsize=8)
    fig.suptitle("V11: condition classification across the representation ladder")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v11_classification_ladder.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_clustering_ladder(clustering_rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(REPRESENTATION_ORDER))
    values = [next(r["observed_ari"] for r in clustering_rows if r["representation"] == rep)
              for rep in REPRESENTATION_ORDER]
    p_values = [next(r["p_value"] for r in clustering_rows if r["representation"] == rep)
                for rep in REPRESENTATION_ORDER]

    bars = ax.bar(x, values, color="tab:blue", alpha=0.8)
    for bar, p_val in zip(bars, p_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"p={p_val:.3f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(REPRESENTATION_ORDER, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("adjusted rand index vs true condition")
    ax.set_title("V11: unsupervised clustering across the representation ladder")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v11_clustering_ladder.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ladder_summary(classification_rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(REPRESENTATION_ORDER))
    lr_values = [next(r["balanced_accuracy"] for r in classification_rows
                       if r["representation"] == rep and r["model"] == "logistic_regression")
                 for rep in REPRESENTATION_ORDER]

    ax.plot(x, lr_values, marker="o", markersize=10, linewidth=2, color="tab:purple")
    ax.axhline(0.5, color="gray", linestyle="--", label="chance level")
    ax.set_xticks(x)
    ax.set_xticklabels(REPRESENTATION_ORDER, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("linear classifier balanced accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("V11: linear separability as components are added back one at a time")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v11_ladder_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    data, y = build_dataset()

    classification_rows = []
    for name in REPRESENTATION_ORDER:
        classification_rows += evaluate_classifiers(data[name], y, name)

    print("Classification results:")
    for row in classification_rows:
        print(f"  {row['representation']:26s} {row['model']:20s} "
              f"bal_acc={row['balanced_accuracy']:.3f} macro_f1={row['macro_f1']:.3f} "
              f"auroc={row['auroc']:.3f}")

    clustering_rows = [evaluate_clustering(data[name], y, name) for name in REPRESENTATION_ORDER]
    print("\nClustering results:")
    for row in clustering_rows:
        print(f"  {row['representation']:26s} ari={row['observed_ari']:.4f} p={row['p_value']:.4f}")

    csv_path = save_summary_csv(classification_rows, clustering_rows)
    ladder_plot = plot_classification_ladder(classification_rows)
    clustering_plot = plot_clustering_ladder(clustering_rows)
    summary_plot = plot_ladder_summary(classification_rows)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved classification ladder plot to {ladder_plot}")
    print(f"Saved clustering ladder plot to {clustering_plot}")
    print(f"Saved ladder summary plot to {summary_plot}")


if __name__ == "__main__":
    main()
