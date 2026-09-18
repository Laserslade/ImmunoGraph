"""
V13: Network level therapeutic restoration.
Maps the relationship between biomarker restoration and network
restoration across a broad grid of interventions.
"""

import os
import csv
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "v13_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_NODES = 8
NODE_NAMES = [f"N{i}" for i in range(N_NODES)]
FD_EPS = 1e-5
THRESHOLD = 0.05
T_SPAN = (0.0, 60.0)
PULSE_CENTER = 15.0
PULSE_WIDTH = 2.0
PULSE_AMPLITUDE = 5.0
INJECTION_NODES = [0, 3]
INTERVENTION_ONSET = 20.0
EVAL_TIME = 40.0
STABILITY_CHECK_TIME = 35.0

BREADTH_GROUPS = {
    "narrow_targeted": [0, 3],
    "medium_targeted": [0, 1, 2, 3],
    "medium_mistargeted": [4, 5, 6, 7],
    "broad_symptomatic": list(range(N_NODES)),
}
STRENGTH_GRID = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0]


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


def make_intervention(nodes, strength):
    def intervention(t, y):
        if t < INTERVENTION_ONSET:
            return np.zeros(N_NODES)
        u = np.zeros(N_NODES)
        for node in nodes:
            u[node] = -strength * y[node]
        return u
    return intervention


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
    abs_weighted = np.abs(jac) * mask
    in_degree = abs_weighted.sum(axis=1)
    hub = NODE_NAMES[int(np.argmax(in_degree))] if mask.any() else "none"
    coherent_ffl, positive_feedback = count_key_motifs(jac, mask)
    return {"jac": jac, "mask": mask, "hub": hub,
            "coherent_ffl": coherent_ffl, "positive_feedback": positive_feedback}


def run_condition(nodes, strength, w, decay, autonomous_rhs):
    intervention = make_intervention(nodes, strength)
    rhs = make_rhs(w, decay, intervention)
    sol = solve_ivp(rhs, T_SPAN, np.zeros(N_NODES), dense_output=True, max_step=0.25)

    eval_state = sol.sol(EVAL_TIME)
    stability_state = sol.sol(STABILITY_CHECK_TIME)
    deviation_at_eval = float(np.linalg.norm(eval_state))
    stability_gap = float(np.linalg.norm(eval_state - stability_state))

    eval_jac = numerical_jacobian(autonomous_rhs, eval_state)
    eval_report = graph_report(eval_jac)

    return {"deviation_at_eval": deviation_at_eval, "stability_gap": stability_gap,
            "eval_report": eval_report}


def graph_distance(report_a, report_b):
    return float(np.linalg.norm(report_a["jac"] - report_b["jac"]))


def main():
    w = base_weight_matrix()
    decay = np.ones(N_NODES)
    autonomous_rhs = make_autonomous_rhs(w, decay)

    baseline_jac = numerical_jacobian(autonomous_rhs, np.zeros(N_NODES))
    baseline_report = graph_report(baseline_jac)

    none_result = run_condition([], 0.0, w, decay, autonomous_rhs)
    deviation_untreated = none_result["deviation_at_eval"]
    graph_distance_untreated = graph_distance(none_result["eval_report"], baseline_report)
    print(f"Untreated: deviation={deviation_untreated:.4f} graph_distance={graph_distance_untreated:.4f}")

    rows = []
    for breadth_name, nodes in BREADTH_GROUPS.items():
        for strength in STRENGTH_GRID:
            result = run_condition(nodes, strength, w, decay, autonomous_rhs)
            gd = graph_distance(result["eval_report"], baseline_report)

            biomarker_restoration = (deviation_untreated - result["deviation_at_eval"]) / deviation_untreated
            network_restoration = (graph_distance_untreated - gd) / graph_distance_untreated

            hub_restored = result["eval_report"]["hub"] == baseline_report["hub"]
            ffl_restored = result["eval_report"]["coherent_ffl"] >= baseline_report["coherent_ffl"]
            pf_restored = result["eval_report"]["positive_feedback"] >= baseline_report["positive_feedback"]

            rows.append({
                "breadth": breadth_name, "strength": strength,
                "deviation_at_eval": result["deviation_at_eval"],
                "graph_distance": gd,
                "biomarker_restoration": biomarker_restoration,
                "network_restoration": network_restoration,
                "stability_gap": result["stability_gap"],
                "hub_restored": hub_restored, "ffl_restored": ffl_restored,
                "pf_restored": pf_restored,
            })

    biomarker_vals = [r["biomarker_restoration"] for r in rows]
    network_vals = [r["network_restoration"] for r in rows]
    corr, p_value = pearsonr(biomarker_vals, network_vals)
    print(f"\nCorrelation between biomarker restoration and network restoration: "
          f"r={corr:.4f}, p={p_value:.5f}")

    gaps = [(r, r["biomarker_restoration"] - r["network_restoration"]) for r in rows]
    gaps.sort(key=lambda item: item[1])
    most_network_lagging = gaps[-1]
    most_biomarker_lagging = gaps[0]
    print(f"\nLargest biomarker-ahead-of-network gap: breadth={most_network_lagging[0]['breadth']} "
          f"strength={most_network_lagging[0]['strength']} gap={most_network_lagging[1]:.3f}")
    print(f"Largest network-ahead-of-biomarker gap: breadth={most_biomarker_lagging[0]['breadth']} "
          f"strength={most_biomarker_lagging[0]['strength']} gap={most_biomarker_lagging[1]:.3f}")

    csv_path = save_summary_csv(rows, corr, p_value)
    scatter_plot = plot_restoration_scatter(rows, corr)
    gap_plot = plot_restoration_gap(rows)
    stability_plot = plot_stability_vs_restoration(rows)

    print(f"\nSaved summary csv to {csv_path}")
    print(f"Saved restoration scatter plot to {scatter_plot}")
    print(f"Saved restoration gap plot to {gap_plot}")
    print(f"Saved stability plot to {stability_plot}")


def save_summary_csv(rows, corr, p_value):
    path = os.path.join(OUTPUT_DIR, "v13_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    corr_path = os.path.join(OUTPUT_DIR, "v13_correlation.csv")
    with open(corr_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["pearson_r", f"{corr:.4f}"])
        writer.writerow(["p_value", f"{p_value:.5f}"])
    return path


def plot_restoration_scatter(rows, corr):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"narrow_targeted": "tab:blue", "medium_targeted": "tab:orange",
              "medium_mistargeted": "tab:green", "broad_symptomatic": "tab:red"}

    for breadth_name in BREADTH_GROUPS:
        subset = [r for r in rows if r["breadth"] == breadth_name]
        ax.scatter([r["biomarker_restoration"] for r in subset],
                   [r["network_restoration"] for r in subset],
                   label=breadth_name, color=colors[breadth_name], alpha=0.8, s=60)

    lims = [-0.2, 1.1]
    ax.plot(lims, lims, linestyle="--", color="gray", label="equal restoration")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("biomarker restoration")
    ax.set_ylabel("network restoration")
    ax.set_title(f"V13: restoration space across the intervention grid (r={corr:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v13_restoration_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_restoration_gap(rows):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"narrow_targeted": "tab:blue", "medium_targeted": "tab:orange",
              "medium_mistargeted": "tab:green", "broad_symptomatic": "tab:red"}

    for breadth_name in BREADTH_GROUPS:
        subset = sorted([r for r in rows if r["breadth"] == breadth_name], key=lambda r: r["strength"])
        gaps = [r["biomarker_restoration"] - r["network_restoration"] for r in subset]
        strengths = [r["strength"] for r in subset]
        ax.plot(strengths, gaps, marker="o", label=breadth_name, color=colors[breadth_name])

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("intervention strength")
    ax.set_ylabel("biomarker restoration minus network restoration")
    ax.set_title("V13: restoration gap by intervention strength and targeting breadth")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v13_restoration_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_stability_vs_restoration(rows):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"narrow_targeted": "tab:blue", "medium_targeted": "tab:orange",
              "medium_mistargeted": "tab:green", "broad_symptomatic": "tab:red"}

    for breadth_name in BREADTH_GROUPS:
        subset = [r for r in rows if r["breadth"] == breadth_name]
        ax.scatter([r["stability_gap"] for r in subset],
                   [r["network_restoration"] for r in subset],
                   label=breadth_name, color=colors[breadth_name], alpha=0.8, s=60)

    ax.set_xlabel("dynamical stability gap (change from t=35 to t=40)")
    ax.set_ylabel("network restoration")
    ax.set_title("V13: dynamical stability versus network restoration")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "v13_stability_vs_restoration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
