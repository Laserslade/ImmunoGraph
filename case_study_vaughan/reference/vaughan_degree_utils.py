"""
vaughan_degree_utils.py

Explicit, unambiguous degree computations for the (n_samples, i, j)
weight/presence arrays produced by vaughan_immunograph_ensemble.py,
where entry [s, i, j] means "edge j -> i in sample s" (i = target,
j = source, matching build_signed_adjacency's present[i,j] convention).

This file exists because a prior version of this analysis used
`array.sum(axis=1)` and called the result "in_degree" when it is
actually weighted/topological OUT-degree (summing over the target
axis for a fixed source). That mislabeling was caught by a synthetic
axis sanity check before any V4 correlation was interpreted (see
project decision log) and is fixed here, permanently, by banning the
ambiguous name: every function below has "in" or "out" in its name and
nothing in this codebase should compute a degree without going through
one of these.

Axis convention, stated once: array shape is (n_samples, i, j).
  - sum over j (axis=2), keep i -> quantity indexed by TARGET -> IN-degree
  - sum over i (axis=1), keep j -> quantity indexed by SOURCE -> OUT-degree
"""

import numpy as np


def weighted_in_degree(weight_stack):
    """(n_samples, i, j) signed weights -> (n_samples, n_nodes) in-degree,
    indexed by target node i. Sums |weight| over source j."""
    return np.abs(weight_stack).sum(axis=2)


def weighted_out_degree(weight_stack):
    """(n_samples, i, j) signed weights -> (n_samples, n_nodes) out-degree,
    indexed by source node j. Sums |weight| over target i."""
    return np.abs(weight_stack).sum(axis=1)


def topology_in_degree(presence_stack):
    """(n_samples, i, j) 0/1 presence -> in-degree count, indexed by target i."""
    return presence_stack.sum(axis=2)


def topology_out_degree(presence_stack):
    """(n_samples, i, j) 0/1 presence -> out-degree count, indexed by source j."""
    return presence_stack.sum(axis=1)


def within_sample_rank(degree_matrix):
    """(n_samples, n_nodes) degree values -> (n_samples, n_nodes) rank,
    1 = highest degree in that sample (i.e. most hub-like), n_nodes = lowest."""
    order = np.argsort(-degree_matrix, axis=1)
    rank = np.zeros_like(degree_matrix, dtype=int)
    n_samples, n_nodes = degree_matrix.shape
    for s in range(n_samples):
        for r, node_idx in enumerate(order[s]):
            rank[s, node_idx] = r + 1
    return rank


def top1_frequency(degree_matrix, n_nodes):
    """Fraction of samples in which each node is rank-1 (the top hub)."""
    top1 = np.argmax(degree_matrix, axis=1)
    return np.bincount(top1, minlength=n_nodes) / degree_matrix.shape[0]
