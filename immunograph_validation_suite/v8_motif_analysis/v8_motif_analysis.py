"""
V8: Network motif analysis.
Tests whether specific signed motifs are associated with specific
dynamical regimes, across an ensemble of perturbed simulations.
"""

import os
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import kruskal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v8_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_NODES = 8
NODE_NAMES = [f"N{i}" for i in range(N_NODES)]
FD_EPS = 1e-5
THRESHOLD = 0.05
T_SPAN = (0.0, 60.0)
SAMPLE_TIMES = np.arange(1.0, 60.0, 1.0)
N_RUNS = 40
REGIME_NAMES = ["baseline", "activation", "peak", "recovery"]
HUB_RATIO_THRESHOLD = 2.5


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


def jittered_weight_matrix(rng, jitter_std=0.08):
    w = base_weight_matrix()
    mask = w != 0.0
    jitter = rng.normal(0.0, jitter_std, size=w.shape)
    return w + jitter * mask


def make_pulse_input(center, width, amplitude, nodes):
    def pulse(t):
        u = np.zeros(N_NODES)
        bump = amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)
        for node in nodes:
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


def classify_motifs(jac, mask):
    coherent_ffl = 0
    incoherent_ffl = 0
    positive_feedback = 0
    negative_feedback = 0

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

    mutual_activation = 0
    mutual_inhibition = 0
    mixed_pair = 0
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            if mask[i, j] and mask[j, i]:
                if jac[i, j] > 0 and jac[j, i] > 0:
                    mutual_activation += 1
                elif jac[i, j] < 0 and jac[j, i] < 0:
                    mutual_inhibition += 1
                else:
                    mixed_pair += 1

    abs_weighted = np.abs(jac) * mask
    degree = abs_weighted.sum(axis=0) + abs_weighted.sum(axis=1)
    max_degree = float(np.max(degree))
    median_degree = float(np.median(degree))
    others_mean = float(np.mean(np.sort(degree)[:-1])) if len(degree) > 1 else 0.0
    centralization = (max_degree - others_mean) / max_degree if max_degree > 0 else 0.0
    hub_and_spoke = max_degree > HUB_RATIO_THRESHOLD * median_degree if median_degree > 0 else False

    return {
        "coherent_ffl": coherent_ffl, "incoherent_ffl": incoherent_ffl,
        "positive_feedback": positive_feedback, "negative_feedback": negative_feedback,
        "mutual_activation": mutual_activation, "mutual_inhibition": mutual_inhibition,
        "mixed_pair": mixed_pair, "hub_centralization": centralization,
        "hub_and_spoke": int(hub_and_spoke),
    }


def true_regime_labels(traj_matrix, sample_times):
    baseline_mean = traj_matrix[sample_times < 10].mean(axis=0)
    deviation = np.linalg.norm(traj_matrix - baseline_mean, axis=1)
    peak_idx = int(np.argmax(deviation))
    max_dev = deviation[peak_idx]

    labels = np.zeros(len(sample_times), dtype=int)
    for i in range(len(sample_times)):
        frac = deviation[i] / max_dev if max_dev > 0 else 0.0
        if i <= peak_idx:
            if frac < 0.1:
                labels[i] = 0
            elif frac < 0.9:
                labels[i] = 1
            else:
                labels[i] = 2
        else:
            if frac >= 0.9:
                labels[i] = 2
            else:
                labels[i] = 3
    return labels


def run_ensemble():
    rng = np.random.default_rng(0)
    rows = []

    for run_idx in range(N_RUNS):
        w = jittered_weight_matrix(rng)
        decay = np.ones(N_NODES)
        amplitude = rng.uniform(2.0, 5.0)
        pulse = make_pulse_input(center=15.0, width=2.0, amplitude=amplitude, nodes=[0, 3])
        rhs = make_rhs(w, decay, pulse)
        autonomous_rhs = make_autonomous_rhs(w, decay)

        sol = solve_ivp(rhs, T_SPAN, np.zeros(N_NODES), dense_output=True, max_step=0.5)
        traj_matrix = np.array([sol.sol(t) for t in SAMPLE_TIMES])
        labels = true_regime_labels(traj_matrix, SAMPLE_TIMES)

        for i, t in enumerate(SAMPLE_TIMES):
            jac = numerical_jacobian(autonomous_rhs, traj_matrix[i])
            mask = build_edge_mask(jac)
            motif_row = classify_motifs(jac, mask)
            motif_row["run"] = run_idx
            motif_row["time"] = t
            motif_row["regime"] = REGIME_NAMES[labels[i]]
            rows.append(motif_row)

    return rows


def save_summary_csv(rows, test_results):
    path = os.path.join(OUTPUT_DIR, "v8_motif_snapshots.csv")
    fieldnames = ["run", "time", "regime", "coherent_ffl", "incoherent_ffl",
                  "positive_feedback", "negative_feedback", "mutual_activation",
                  "mutual_inhibition", "mixed_pair", "hub_centralization", "hub_and_spoke"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    stats_path = os.path.join(OUTPUT_DIR, "v8_regime_association_tests.csv")
    with open(stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["motif", "kruskal_h", "p_value"])
        for motif, (h_stat, p_val) in test_results.items():
            if np.isnan(p_val):
                writer.writerow([motif, "constant", "not_applicable"])
            else:
                writer.writerow([motif, f"{h_stat:.4f}", f"{p_val:.5f}"])
    return path, stats_path


def run_statistical_tests(rows, motif_keys):
    results = {}
    for motif in motif_keys:
        groups = [[r[motif] for r in rows if r["regime"] == regime] for regime in REGIME_NAMES]
        all_values = [v for group in groups for v in group]
        if len(set(all_values)) <= 1:
            results[motif] = (float("nan"), float("nan"))
            continue
        h_stat, p_val = kruskal(*groups)
        results[motif] = (h_stat, p_val)
    return results


def plot_motif_by_regime(rows, motif_keys, test_results):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, motif in zip(axes, motif_keys):
        means = []
        stds = []
        for regime in REGIME_NAMES:
            values = [r[motif] for r in rows if r["regime"] == regime]
            means.append(np.mean(values))
            stds.append(np.std(values))
        x = np.arange(len(REGIME_NAMES))
        ax.bar(x, means, yerr=stds, capsize=3, color="tab:blue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(REGIME_NAMES, rotation=20, fontsize=8)
        p_val = test_results[motif][1]
        if np.isnan(p_val):
            ax.set_title(f"{motif}\n(constant, not testable)", fontsize=9)
        else:
            ax.set_title(f"{motif}\n(kruskal p={p_val:.4f})", fontsize=9)

    fig.suptitle("V8: motif occurrence by dynamical regime, mean plus std across the ensemble")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v8_motif_by_regime.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hub_centralization(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [[r["hub_centralization"] for r in rows if r["regime"] == regime] for regime in REGIME_NAMES]
    ax.boxplot(data, tick_labels=REGIME_NAMES)
    ax.set_ylabel("hub centralization score")
    ax.set_title("V8: hub and spoke centralization by regime")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v8_hub_centralization.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_motif_prevalence_across_ensemble(rows, motif_keys):
    fig, ax = plt.subplots(figsize=(9, 5))
    prevalence = []
    for motif in motif_keys:
        values = [r[motif] for r in rows]
        prevalence.append(np.mean(np.array(values) > 0))

    x = np.arange(len(motif_keys))
    ax.bar(x, prevalence, color="tab:green", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(motif_keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("fraction of snapshots with motif present")
    ax.set_title("V8: motif prevalence across the full ensemble, all runs and times")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v8_motif_prevalence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    rows = run_ensemble()
    print(f"Collected {len(rows)} snapshots across {N_RUNS} runs")

    regime_counts = {regime: sum(1 for r in rows if r["regime"] == regime) for regime in REGIME_NAMES}
    print(f"Regime counts across ensemble: {regime_counts}")

    motif_keys = ["coherent_ffl", "incoherent_ffl", "positive_feedback", "negative_feedback",
                  "mutual_activation", "mutual_inhibition", "mixed_pair", "hub_centralization"]

    test_results = run_statistical_tests(rows, motif_keys)
    print("\nKruskal-Wallis test across regimes, per motif type:")
    for motif, (h_stat, p_val) in test_results.items():
        if np.isnan(p_val):
            print(f"  {motif:20s} constant across entire ensemble, test not applicable")
        else:
            print(f"  {motif:20s} H={h_stat:.3f} p={p_val:.5f}")

    csv_path, stats_path = save_summary_csv(rows, test_results)
    by_regime_plot = plot_motif_by_regime(rows, motif_keys, test_results)
    hub_plot = plot_hub_centralization(rows)
    prevalence_plot = plot_motif_prevalence_across_ensemble(rows, motif_keys)

    print(f"\nSaved snapshot csv to {csv_path}")
    print(f"Saved statistical test csv to {stats_path}")
    print(f"Saved motif by regime plot to {by_regime_plot}")
    print(f"Saved hub centralization plot to {hub_plot}")
    print(f"Saved motif prevalence plot to {prevalence_plot}")


if __name__ == "__main__":
    main()
