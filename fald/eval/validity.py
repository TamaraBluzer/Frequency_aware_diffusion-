"""Validity, uniqueness and novelty (V.U.N.).

Validity is dataset-specific. Planarity is exact and cheap. SBM validity requires recovering
the block structure, which upstream does with graph-tool's Bayesian blockmodel inference;
graph-tool has no Windows build, so `sbm_validity` raises rather than silently substituting a
different statistical test. See docs/ENVIRONMENT.md.
"""

from __future__ import annotations

from typing import Callable, Iterable

import networkx as nx

__all__ = ["is_planar", "sbm_validity", "vun", "fraction_unique", "fraction_novel"]


def is_planar(graph) -> bool:
    return nx.is_connected(graph) and nx.check_planarity(graph)[0]


def sbm_validity(graph) -> bool:
    raise NotImplementedError(
        "SBM validity needs graph-tool's minimize_blockmodel_dl, which does not build on "
        "Windows. See docs/ENVIRONMENT.md for the planned spectral-clustering replacement."
    )


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
