"""
V14: Intervention robustness under uncertainty.
Repeats the V10 interventions under an ensemble of parameter,
efficacy, and injury severity uncertainty, and tests rank stability.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v14_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_NODES = 8
NODE_NAMES = [f"N{i}" for i in range(N_NODES)]
FD_EPS = 1e-5
T_SPAN = (0.0, 60.0)
PULSE_CENTER = 15.0
PULSE_WIDTH = 2.0
INJECTION_NODES = [0, 3]
INTERVENTION_ONSET = 20.0
EVAL_TIME = 40.0
N_ENSEMBLE = 50

MISTARGET_NODES = [6, 7]
CENTRAL_AMPLITUDE = 5.0
CENTRAL_DAMP = {"symptomatic": 0.195, "targeted": 3.0, "mistargeted": 5.0}
STRATEGIES = ["symptomatic", "targeted", "mistargeted"]

WEIGHT_JITTER_STD = 0.08
DECAY_JITTER_STD = 0.05
AMPLITUDE_JITTER_STD = 0.5
EFFICACY_JITTER_STD = 0.3


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


def make_intervention(strategy, damp):
    def none_intervention(t, y):
        return np.zeros(N_NODES)

    def symptomatic(t, y):
        if t < INTERVENTION_ONSET:
            return np.zeros(N_NODES)
        return -damp * y

    def targeted(t, y):
        if t < INTERVENTION_ONSET:
            return np.zeros(N_NODES)
        u = np.zeros(N_NODES)
        for node in INJECTION_NODES:
            u[node] = -damp * y[node]
        return u

    def mistargeted(t, y):
        if t < INTERVENTION_ONSET:
            return np.zeros(N_NODES)
        u = np.zeros(N_NODES)
        for node in MISTARGET_NODES:
            u[node] = -damp * y[node]
        return u

    return {"none": none_intervention, "symptomatic": symptomatic,
            "targeted": targeted, "mistargeted": mistargeted}[strategy]


def make_rhs(w, decay, pulse, intervention):
    def rhs(t, y):
        return -decay * y + w @ np.tanh(y) + pulse(t) + intervention(t, y)
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


def eval_state_for(w, decay, pulse, intervention):
    rhs = make_rhs(w, decay, pulse, intervention)
    sol = solve_ivp(rhs, T_SPAN, np.zeros(N_NODES), dense_output=True, max_step=0.25)
    return sol.sol(EVAL_TIME)


def run_one_draw(rng):
    w_mask = base_weight_matrix() != 0.0
    w = base_weight_matrix() + rng.normal(0.0, WEIGHT_JITTER_STD, size=(N_NODES, N_NODES)) * w_mask
    decay = np.ones(N_NODES) + rng.normal(0.0, DECAY_JITTER_STD, size=N_NODES)
    amplitude = max(0.5, rng.normal(CENTRAL_AMPLITUDE, AMPLITUDE_JITTER_STD))
    pulse = make_pulse(amplitude)
    autonomous_rhs = make_autonomous_rhs(w, decay)

    baseline_jac = numerical_jacobian(autonomous_rhs, np.zeros(N_NODES))

    no_intervention = make_intervention("none", 0.0)
    untreated_state = eval_state_for(w, decay, pulse, no_intervention)
    untreated_jac = numerical_jacobian(autonomous_rhs, untreated_state)
    deviation_untreated = float(np.linalg.norm(untreated_state))
    graph_distance_untreated = float(np.linalg.norm(untreated_jac - baseline_jac))

    results = {}
    for strategy in STRATEGIES:
        efficacy_factor = max(0.05, rng.normal(1.0, EFFICACY_JITTER_STD))
        damp = CENTRAL_DAMP[strategy] * efficacy_factor
        intervention = make_intervention(strategy, damp)

        state = eval_state_for(w, decay, pulse, intervention)
        jac = numerical_jacobian(autonomous_rhs, state)
        deviation = float(np.linalg.norm(state))
        graph_distance = float(np.linalg.norm(jac - baseline_jac))

        biomarker_restoration = (deviation_untreated - deviation) / deviation_untreated
        network_restoration = (graph_distance_untreated - graph_distance) / graph_distance_untreated

        results[strategy] = {"biomarker_restoration": biomarker_restoration,
                              "network_restoration": network_restoration,
                              "efficacy_factor": efficacy_factor}

    return results


def run_ensemble():
    rng = np.random.default_rng(0)
    all_draws = []
    for _ in range(N_ENSEMBLE):
        all_draws.append(run_one_draw(rng))
    return all_draws


def compute_rank_fractions(all_draws, metric):
    rank_counts = {s: {1: 0, 2: 0, 3: 0} for s in STRATEGIES}
    for draw in all_draws:
        ordered = sorted(STRATEGIES, key=lambda s: draw[s][metric], reverse=True)
        for rank, strategy in enumerate(ordered, start=1):
            rank_counts[strategy][rank] += 1
    fractions = {s: {r: count / len(all_draws) for r, count in ranks.items()}
                 for s, ranks in rank_counts.items()}
    return fractions


def save_summary_csv(all_draws, rank_fractions_biomarker, rank_fractions_network):
    path = os.path.join(OUTPUT_DIR, "v14_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "metric", "value"])
        for strategy in STRATEGIES:
            bio_vals = [d[strategy]["biomarker_restoration"] for d in all_draws]
            net_vals = [d[strategy]["network_restoration"] for d in all_draws]
            writer.writerow([strategy, "biomarker_restoration_mean", f"{np.mean(bio_vals):.4f}"])
            writer.writerow([strategy, "biomarker_restoration_std", f"{np.std(bio_vals):.4f}"])
            writer.writerow([strategy, "network_restoration_mean", f"{np.mean(net_vals):.4f}"])
            writer.writerow([strategy, "network_restoration_std", f"{np.std(net_vals):.4f}"])
            for rank in [1, 2, 3]:
                writer.writerow([strategy, f"biomarker_rank{rank}_fraction",
                                  f"{rank_fractions_biomarker[strategy][rank]:.3f}"])
                writer.writerow([strategy, f"network_rank{rank}_fraction",
                                  f"{rank_fractions_network[strategy][rank]:.3f}"])
    return path


def plot_outcome_distributions(all_draws):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, title in zip(axes, ["biomarker_restoration", "network_restoration"],
                                  ["biomarker restoration", "network restoration"]):
        data = [[d[s][metric] for d in all_draws] for s in STRATEGIES]
        ax.boxplot(data, tick_labels=STRATEGIES)
        ax.set_title(title)
        ax.set_ylabel("restoration score")

    fig.suptitle(f"V14: outcome distributions across {N_ENSEMBLE} uncertainty draws")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v14_outcome_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rank_stability(rank_fractions_biomarker, rank_fractions_network):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["tab:green", "tab:orange", "tab:red"]

    for ax, fractions, title in zip(axes, [rank_fractions_biomarker, rank_fractions_network],
                                     ["biomarker restoration rank", "network restoration rank"]):
        bottom = np.zeros(len(STRATEGIES))
        for rank, color in zip([1, 2, 3], colors):
            values = [fractions[s][rank] for s in STRATEGIES]
            ax.bar(STRATEGIES, values, bottom=bottom, color=color, label=f"rank {rank}")
            bottom += np.array(values)
        ax.set_title(title)
        ax.set_ylabel("fraction of draws")
        ax.set_ylim(0, 1.05)

    axes[0].legend(fontsize=8)
    fig.suptitle("V14: rank stability across the uncertainty ensemble")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v14_rank_stability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_paired_scatter(all_draws):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [("symptomatic", "targeted"), ("symptomatic", "mistargeted"), ("targeted", "mistargeted")]

    for ax, (s1, s2) in zip(axes, pairs):
        x = [d[s1]["network_restoration"] for d in all_draws]
        y = [d[s2]["network_restoration"] for d in all_draws]
        ax.scatter(x, y, alpha=0.7)
        lims = [min(x + y) - 0.05, max(x + y) + 0.05]
        ax.plot(lims, lims, linestyle="--", color="gray")
        ax.set_xlabel(f"{s1} network restoration")
        ax.set_ylabel(f"{s2} network restoration")
        wins = sum(1 for a, b in zip(x, y) if b > a)
        ax.set_title(f"{s2} beats {s1} in {wins}/{len(x)} draws")

    fig.suptitle("V14: paired per draw comparison of network restoration")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "v14_paired_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    all_draws = run_ensemble()

    print("Outcome summary across ensemble:")
    for strategy in STRATEGIES:
        bio_vals = [d[strategy]["biomarker_restoration"] for d in all_draws]
        net_vals = [d[strategy]["network_restoration"] for d in all_draws]
        print(f"  {strategy:12s} biomarker={np.mean(bio_vals):.3f}+/-{np.std(bio_vals):.3f} "
              f"network={np.mean(net_vals):.3f}+/-{np.std(net_vals):.3f}")

    rank_fractions_biomarker = compute_rank_fractions(all_draws, "biomarker_restoration")
    rank_fractions_network = compute_rank_fractions(all_draws, "network_restoration")

    print("\nNetwork restoration rank fractions (1=best):")
    for strategy in STRATEGIES:
        print(f"  {strategy:12s} {rank_fractions_network[strategy]}")

    print("\nBiomarker restoration rank fractions (1=best):")
    for strategy in STRATEGIES:
        print(f"  {strategy:12s} {rank_fractions_biomarker[strategy]}")

    csv_path = save_summary_csv(all_draws, rank_fractions_biomarker, rank_fractions_network)
    dist_plot = plot_outcome_distributions(all_draws)
    rank_plot = plot_rank_stability(rank_fractions_biomarker, rank_fractions_network)
    paired_plot = plot_paired_scatter(all_draws)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved outcome distributions plot to {dist_plot}")
    print(f"Saved rank stability plot to {rank_plot}")
    print(f"Saved paired scatter plot to {paired_plot}")


if __name__ == "__main__":
    main()
