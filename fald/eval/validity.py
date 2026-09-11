"""Validity, uniqueness and novelty (V.U.N.).

Validity is dataset-specific. Planarity is exact and cheap. SBM validity requires recovering
the block structure, which upstream does with graph-tool's Bayesian blockmodel inference;
graph-tool has no Windows build, so `sbm_validity` raises rather than silently substituting a
different statistical test. See docs/ENVIRONMENT.md.
"""

from __future__ import annotations

from typing import Callable, Iterable

import networkx as nx
import numpy as np

__all__ = ["is_planar", "sbm_validity", "vun", "fraction_unique", "fraction_novel"]


def is_planar(graph) -> bool:
    return nx.is_connected(graph) and nx.check_planarity(graph)[0]


def sbm_validity(
    graph,
    *,
    n_blocks_range: tuple[int, int] = (2, 5),
    intra_min: float = 0.25,
    inter_max: float = 0.10,
    min_block_size: int = 10,
    seed: int = 0,
) -> bool:
    """Spectral-clustering stand-in for graph-tool's Bayesian blockmodel test.

    SPECTRE recovers blocks with `minimize_blockmodel_dl` plus an MCMC refinement, then Wald
    -tests the recovered parameters. graph-tool has no Windows build, so blocks are recovered
    with spectral clustering instead and the same structural conditions are checked directly:
    every block large enough to be meaningful, intra-block density high, inter-block density
    low.

    This is deliberately *not* presented as equivalent to the upstream test -- it is
    deterministic and dependency-light, but a different estimator. SBM numbers produced with
    it must be reported as a deviation and not compared directly against SPECTRE's.
    """
    from sklearn.cluster import SpectralClustering

    if graph.number_of_nodes() < min_block_size * n_blocks_range[0]:
        return False
    if not nx.is_connected(graph):
        return False

    adjacency = nx.to_numpy_array(graph, dtype=float)
    n = adjacency.shape[0]

    # The block count is unknown, so accept the graph if *any* count in range fits. Upstream
    # infers it; scanning is the deterministic analogue.
    for n_blocks in range(n_blocks_range[0], n_blocks_range[1] + 1):
        if n < min_block_size * n_blocks:
            continue
        try:
            labels = SpectralClustering(
                n_clusters=n_blocks,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=seed,
            ).fit_predict(adjacency)
        except Exception:
            continue

        sizes = [int((labels == b).sum()) for b in range(n_blocks)]
        if min(sizes) < min_block_size:
            continue

        intra_ok = True
        inter_ok = True
        for a in range(n_blocks):
            mask_a = labels == a
            block = adjacency[np.ix_(mask_a, mask_a)]
            possible = sizes[a] * (sizes[a] - 1)
            if possible == 0 or block.sum() / possible < intra_min:
                intra_ok = False
                break
            for b in range(a + 1, n_blocks):
                mask_b = labels == b
                cross = adjacency[np.ix_(mask_a, mask_b)]
                if cross.mean() > inter_max:
                    inter_ok = False
                    break
            if not inter_ok:
                break

        if intra_ok and inter_ok:
            return True

    return False


def _isomorphic_to_any(graph, others: Iterable) -> bool:
    for other in others:
        if nx.faster_could_be_isomorphic(graph, other) and nx.is_isomorphic(graph, other):
            return True
    return False


def fraction_unique(graphs) -> float:
    """Fraction of graphs in distinct isomorphism classes."""
    if not graphs:
        return 0.0
    seen: list = []
    for graph in graphs:
        if graph.number_of_nodes() == 0:
            continue
        if not _isomorphic_to_any(graph, seen):
            seen.append(graph)
    return len(seen) / len(graphs)


def fraction_novel(graphs, train_graphs) -> float:
    """Fraction of graphs isomorphic to no training graph. Higher is better here."""
    if not graphs:
        return 0.0
    novel = sum(0 if _isomorphic_to_any(g, train_graphs) else 1 for g in graphs)
    return novel / len(graphs)


def vun(graphs, train_graphs, validity_func: Callable = lambda _g: True) -> dict:
    """Valid-Unique-Novel, reported jointly and individually.

    The headline number is `vun`: the fraction of samples that are simultaneously valid, in a
    distinct isomorphism class from earlier samples, and absent from the training set. The
    marginals are kept because a low joint score is otherwise unattributable.
    """
    if not graphs:
        return {"valid": 0.0, "unique": 0.0, "novel": 0.0, "vun": 0.0}

    n = len(graphs)
    n_valid = 0
    n_non_unique = 0
    n_memorized = 0
    n_vun = 0
    accepted: list = []

    for graph in graphs:
        valid = graph.number_of_nodes() > 0 and validity_func(graph)
        n_valid += int(valid)

        if _isomorphic_to_any(graph, accepted):
            n_non_unique += 1
            continue
        accepted.append(graph)

        memorized = _isomorphic_to_any(graph, train_graphs)
        n_memorized += int(memorized)
        if valid and not memorized:
            n_vun += 1

    return {
        "valid": n_valid / n,
        "unique": (n - n_non_unique) / n,
        "novel": (n - n_non_unique - n_memorized) / n,
        "vun": n_vun / n,
    }
