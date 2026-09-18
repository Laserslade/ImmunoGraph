"""
phase8_motif_discovery.py

Phase 8 of the Neuroimmune Graph Project.

"Motif discovery" here means two genuinely different things, kept as
two separate sub-analyses rather than conflated into one:

  PART A -- MODULE DETECTION (node-level).
  Partition each graph's 22 nodes into communities via modularity
  maximization, then measure what fraction of the graph's total
  network influence is concentrated within each module. This is the
  direct numerical prerequisite for findings log Section 11's Finding
  #2 ("the dominant inflammatory module accounts for X% of total
  network influence").

  PART B -- MOTIF / STATE DISCOVERY (graph-level).
  Treat each graph (a full signed weight matrix) as a single point in
  feature space, and cluster ACROSS graphs to find recurring "network
  states" -- i.e. motifs that show up again and again across different
  ages and different parameter perturbations, rather than being
  age-50-or-run-3-specific. This borrows directly from dynamic
  functional connectivity (dFC) methodology in neuroscience, where
  k-means clustering of windowed connectivity matrices is a standard,
  citable way to find recurring "brain states" (e.g. Allen et al. 2014,
  "Tracking Whole-Brain Connectivity Dynamics in the Resting State").
  The composite model's sensitivity matrices are directly analogous to
  those windowed connectivity matrices, so the same method applies.

-------------------------------------------------------------------------
PART A -- METHOD DETAILS AND WHY
-------------------------------------------------------------------------
Modularity-based community detection (Clauset-Newman-Moore greedy
modularity, via networkx's `greedy_modularity_communities`) is defined
for undirected graphs with non-negative edge weights. Two decisions
were needed to make the composite model's graphs fit that requirement,
and both are recorded here rather than applied silently:

  1. SIGN: this project's edge weights are signed (positive = drives up,
     negative = suppresses -- see Phase 5). Modularity maximization
     interprets edge weight as "strength of connection," which isn't
     well-defined for a negative number. Using |weight| as connection
     strength is the standard workaround (a strong suppressive
     relationship is still a strong functional coupling, just directed
     oppositely) -- this project does the same. Sign is NOT discarded,
     though: after a module is found, this script separately reports
     what fraction of ITS intra-module edges are positive vs negative,
     so "this module is tightly coupled" and "this module is mostly
     self-reinforcing vs mostly self-suppressing" remain distinguishable
     downstream (relevant for Phase 10's biological narration).

  2. DIRECTION: the sensitivity graph is directed (j -> i). Converted to
     undirected by summing |weight| across both directions between each
     node pair (so a pair with strong mutual influence in both
     directions scores higher than a pair with a strong edge in only
     one direction, which is the intended behavior for "how tightly
     coupled are these two nodes").

Module influence share is defined as:
    intra-module |weight| sum / total graph |weight| sum
using ONLY edges with both endpoints inside the module (not edges
leaving the module), so it answers "how much of the graph's total
influence is contained within this module" specifically.

-------------------------------------------------------------------------
PART B -- METHOD DETAILS AND WHY
-------------------------------------------------------------------------
Feature vector per graph: the full signed 22x22 weight matrix,
off-diagonal entries only (462 features), in the FIXED order given by
STATE_NAMES -- fixed regardless of which edges are actually present in
a given graph (absent edge = 0), so every graph's vector is directly
comparable. Built straight from each graph's own edge dict rather than
reparsing GraphML files node-by-node in file order (GraphML doesn't
guarantee node order round-trips, and this project reloads its own
saved graphs a lot -- indexing by STATE_NAMES rather than file order
avoids a subtle correctness bug here).

Features are z-scored (per-feature, across the full ensemble) before
clustering, so no single high-magnitude edge dominates the distance
metric. K-means with k selected by silhouette score, swept over
k=2..10, with a fixed random_state for reproducibility (results should
not depend on which run of this script produced them).

Every graph the project has built so far is included: the 51-graph
default-parameter sequence (Phase 6) AND all 1224 batch graphs
(Phase 7) -- 1275 total. Including the default sequence specifically
lets this script check its cluster assignments against the hub
transitions Phase 6 already found by eye (age ~31-32, ~36-37, ~54-55,
~64-65) as an independent consistency check: do the discovered network
"states" actually change at those same ages? If yes, that's the first
real evidence the graph-level clustering is finding something
biologically genuine rather than just noise. If not, that's a genuine
discrepancy to report, not paper over.

Run:
    python phase8_motif_discovery.py
"""

import csv
import os
import sys
from itertools import combinations

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DIRS  # noqa: E402
from models.composite_backbone import STATE_NAMES  # noqa: E402

N = len(STATE_NAMES)
NODE_INDEX = {name: i for i, name in enumerate(STATE_NAMES)}

SEQUENCE_MANIFEST = os.path.join(DIRS["graphs"], "sequence_manifest.csv")
SEQUENCE_DIR = os.path.join(DIRS["graphs"], "sequence")
BATCH_MANIFEST = os.path.join(DIRS["graphs"], "batch_manifest.csv")
BATCH_DIR = os.path.join(DIRS["graphs"], "batch")

# Inflammatory-module hypothesis carried over from earlier project planning
# (findings log Section 11) -- tested explicitly in Part A below rather
# than just assumed.
HYPOTHESIZED_INFLAMMATORY_MODULE = {"M_pro", "ABo", "IL12", "IL1"}


def list_all_graphs():
    """Return a list of dicts describing every graph built so far:
    {source, run_id, age, path}. source is 'default' (Phase 6) or
    'batch' (Phase 7)."""
    graphs = []

    with open(SEQUENCE_MANIFEST) as f:
        for row in csv.DictReader(f):
            graphs.append({
                "source": "default",
                "run_id": "default",
                "age": float(row["age"]),
                "path": os.path.join(SEQUENCE_DIR, row["filename"]),
            })

    with open(BATCH_MANIFEST) as f:
        for row in csv.DictReader(f):
            run_id = f"run_{int(row['run_id']):04d}"
            graphs.append({
                "source": "batch",
                "run_id": run_id,
                "age": float(row["age"]),
                "path": os.path.join(BATCH_DIR, run_id, row["filename"]),
            })

    return graphs


# ---------------------------------------------------------------------
# PART A: module detection
# ---------------------------------------------------------------------

def detect_modules(G):
    """Partition G's nodes into communities using greedy modularity
    maximization on the undirected, |weight|-summed conversion of G
    (see module docstring for why). Returns a list of frozensets of
    node names."""
    G_undirected = nx.Graph()
    G_undirected.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        w = abs(data["weight"])
        if G_undirected.has_edge(u, v):
            G_undirected[u][v]["weight"] += w
        else:
            G_undirected.add_edge(u, v, weight=w)

    communities = nx.algorithms.community.greedy_modularity_communities(
        G_undirected, weight="weight"
    )
    return [frozenset(c) for c in communities]


def module_influence_share(G, module, total_abs_weight):
    """Fraction of the graph's total |weight| contained in edges with
    BOTH endpoints inside `module`. Also returns the fraction of those
    intra-module edges that are positive (sign composition), and counts."""
    intra_abs = 0.0
    intra_pos = 0
    intra_total = 0
    for u, v, data in G.edges(data=True):
        if u in module and v in module:
            w = data["weight"]
            intra_abs += abs(w)
            intra_total += 1
            if w > 0:
                intra_pos += 1

    share = intra_abs / total_abs_weight if total_abs_weight > 0 else 0.0
    pos_fraction = intra_pos / intra_total if intra_total > 0 else None
    return share, pos_fraction, intra_total


def run_module_detection(graph_records):
    """Run Part A across every graph, write graphs/module_manifest.csv,
    and return the rows (also used for the hypothesis test below)."""
    rows = []
    for rec in graph_records:
        G = nx.read_graphml(rec["path"])
        total_abs_weight = sum(abs(d["weight"]) for _, _, d in G.edges(data=True))
        modules = detect_modules(G)

        # identify current top hub (weighted in-degree) to flag which
        # module is "dominant" -- reuses Phase 6's definition exactly.
        in_weight = {n: 0.0 for n in G.nodes()}
        for u, v, d in G.edges(data=True):
            in_weight[v] += abs(d["weight"])
        top_hub = max(in_weight, key=in_weight.get)

        for module in modules:
            share, pos_fraction, intra_edges = module_influence_share(
                G, module, total_abs_weight
            )
            rows.append({
                "source": rec["source"],
                "run_id": rec["run_id"],
                "age": rec["age"],
                "module_members": ";".join(sorted(module)),
                "module_size": len(module),
                "intra_module_edges": intra_edges,
                "influence_share": share,
                "intra_module_positive_fraction": pos_fraction,
                "contains_top_hub": top_hub in module,
                "top_hub": top_hub,
                "matches_hypothesized_inflammatory_module":
                    module == frozenset(HYPOTHESIZED_INFLAMMATORY_MODULE),
                "hypothesized_module_subset_of_this_module":
                    HYPOTHESIZED_INFLAMMATORY_MODULE.issubset(module),
            })

    manifest_path = os.path.join(DIRS["graphs"], "module_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Part A: wrote {len(rows)} module records "
          f"(across {len(graph_records)} graphs) to {manifest_path}")
    return rows


def summarize_dominant_module(module_rows):
    """Print summary stats for the module containing each graph's top
    hub -- the direct answer to 'how concentrated is network influence
    in the dominant module.'"""
    dominant = [r for r in module_rows if r["contains_top_hub"]]
    shares = np.array([r["influence_share"] for r in dominant])

    print(f"\nDominant module (module containing that graph's top hub), "
          f"across {len(dominant)} graphs:")
    print(f"  influence share: mean={shares.mean():.3f}  "
          f"median={np.median(shares):.3f}  "
          f"min={shares.min():.3f}  max={shares.max():.3f}")

    sizes = np.array([r["module_size"] for r in dominant])
    print(f"  module size: mean={sizes.mean():.1f} nodes  "
          f"min={sizes.min()}  max={sizes.max()}")

    exact_matches = sum(r["matches_hypothesized_inflammatory_module"] for r in dominant)
    subset_matches = sum(r["hypothesized_module_subset_of_this_module"] for r in dominant)
    print(f"\nHypothesis test -- {{M_pro, ABo, IL12, IL1}} as 'the inflammatory module':")
    print(f"  exact match to dominant module: {exact_matches}/{len(dominant)} graphs "
          f"({100*exact_matches/len(dominant):.1f}%)")
    print(f"  fully contained within dominant module: {subset_matches}/{len(dominant)} graphs "
          f"({100*subset_matches/len(dominant):.1f}%)")


# ---------------------------------------------------------------------
# PART B: motif / state discovery via graph-level clustering
# ---------------------------------------------------------------------

def graph_to_feature_vector(G):
    """Flatten G's signed weight matrix into a fixed-order 462-dim
    vector (22x22 off-diagonal), indexed by STATE_NAMES regardless of
    file/edge order -- see module docstring for why this matters."""
    M = np.zeros((N, N))
    for u, v, data in G.edges(data=True):
        M[NODE_INDEX[v], NODE_INDEX[u]] = data["weight"]  # [target, source]

    mask = ~np.eye(N, dtype=bool)
    return M[mask]


def build_feature_matrix(graph_records):
    features = []
    for rec in graph_records:
        G = nx.read_graphml(rec["path"])
        features.append(graph_to_feature_vector(G))
    return np.array(features)


def choose_k_by_silhouette(X, k_range=range(2, 11), random_state=0):
    """Sweep k, return (best_k, scores_dict). Fixed random_state so this
    is reproducible run to run."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def run_state_clustering(graph_records):
    print("\nPart B: building feature matrix for all graphs...")
    X_raw = build_feature_matrix(graph_records)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print("Sweeping k=2..10 for state clustering (silhouette score)...")
    best_k, scores = choose_k_by_silhouette(X)
    for k, s in scores.items():
        marker = "  <-- chosen" if k == best_k else ""
        print(f"  k={k:2d}: silhouette={s:.4f}{marker}")

    km = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    labels = km.fit_predict(X)

    rows = []
    for rec, label in zip(graph_records, labels):
        rows.append({
            "source": rec["source"],
            "run_id": rec["run_id"],
            "age": rec["age"],
            "cluster": int(label),
        })

    manifest_path = os.path.join(DIRS["graphs"], "state_cluster_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} cluster assignments to {manifest_path}")

    # Characterize each cluster's motif signature: centroid weight
    # matrix -> top-5 edges by |weight|, and which node is centroid's
    # top hub.
    centroids_raw = scaler.inverse_transform(km.cluster_centers_)
    print(f"\nDiscovered {best_k} recurring network states (motifs):")
    for cluster_id in range(best_k):
        n_members = int((labels == cluster_id).sum())
        vec = centroids_raw[cluster_id]
        M = np.zeros((N, N))
        mask = ~np.eye(N, dtype=bool)
        M[mask] = vec

        # top-5 edges by |weight| in the centroid
        edge_weights = [
            (STATE_NAMES[j], STATE_NAMES[i], M[i, j])
            for i in range(N) for j in range(N) if i != j
        ]
        edge_weights.sort(key=lambda e: -abs(e[2]))
        top_edges = edge_weights[:5]

        in_weight = np.abs(M).sum(axis=1)
        top_hub = STATE_NAMES[int(np.argmax(in_weight))]

        print(f"\n  Cluster {cluster_id} -- {n_members} graphs, centroid top hub: {top_hub}")
        for src, tgt, w in top_edges:
            sign = "+" if w > 0 else "-"
            print(f"      {src:8s} -> {tgt:8s}  {sign}{abs(w):.3f}")

    return rows


def check_against_known_transitions(cluster_rows):
    """Consistency check: does the default-trajectory sequence's cluster
    label actually change at the hub-transition ages Phase 6 already
    found by eye (~31-32, ~36-37, ~54-55, ~64-65)? Reports agreement or
    disagreement plainly -- this is a genuine check, not a formality."""
    default_rows = sorted(
        (r for r in cluster_rows if r["source"] == "default"),
        key=lambda r: r["age"],
    )
    print("\nConsistency check: default-trajectory cluster label vs age "
          "(compare to Phase 6's hub transitions at ~31-32, ~36-37, ~54-55, ~64-65):")
    prev_cluster = None
    for r in default_rows:
        if r["cluster"] != prev_cluster:
            print(f"  age {r['age']:5.1f}: cluster {r['cluster']}")
            prev_cluster = r["cluster"]


def main():
    print("Listing all graphs built so far (Phase 6 default sequence + Phase 7 batch)...")
    graph_records = list_all_graphs()
    print(f"Found {len(graph_records)} graphs total.")

    module_rows = run_module_detection(graph_records)
    summarize_dominant_module(module_rows)

    cluster_rows = run_state_clustering(graph_records)
    check_against_known_transitions(cluster_rows)

    print("\nPhase 8 complete.")


if __name__ == "__main__":
    main()
