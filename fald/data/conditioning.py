"""Batch construction for oracle spectral side conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
import torch

from .spectral import cached_eigendecompositions, select_band

__all__ = ["ConditionTensors", "build_condition_tensors"]


@dataclass(frozen=True)
class ConditionTensors:
    pair: torch.Tensor
    eigenvalues: torch.Tensor
    present: torch.Tensor


def _normalize_pair_channel(channel: np.ndarray) -> np.ndarray:
    """Unit-RMS scaling prevents low/high bands differing only by input magnitude."""
    rms = float(np.sqrt(np.mean(np.square(channel))))
    return channel / max(rms, 1e-8)


def build_condition_tensors(
    graphs: Sequence[nx.Graph],
    *,
    band: str,
    k: int,
    seed: int = 0,
    cache_tag: str = "condition",
) -> ConditionTensors:
    """Return padded pair/global conditions for a graph collection.

    The pair channel is `U diag(lambda) U^T`, which is invariant to eigenvector
    signs and to rotations inside an exactly repeated eigenspace.  Eigenvalues
    are mapped from `[0, 2]` to `[-1, 1]`.  Pair channels are scaled per graph to
    unit RMS so the low/high comparison is not an activation-scale comparison.
    """
    if band not in {"none", "low", "high", "random"}:
        raise ValueError(
            "Tier-0 currently supports none/low/high/random; geometry-matched controls "
            "must be added before running the full sweep"
        )
    if k < 1:
        raise ValueError("k must be >= 1; use band='none' for an absent condition")

    n_max = max(graph.number_of_nodes() for graph in graphs)
    pair = np.zeros((len(graphs), n_max, n_max), dtype=np.float32)
    eigenvalues = np.zeros((len(graphs), k), dtype=np.float32)
    present = np.zeros((len(graphs), 1), dtype=np.float32)

    if band == "none":
        return ConditionTensors(
            pair=torch.from_numpy(pair),
            eigenvalues=torch.from_numpy(eigenvalues),
            present=torch.from_numpy(present),
        )

    all_values, all_vectors = cached_eigendecompositions(graphs, tag=cache_tag)
    for index, (graph, values, vectors) in enumerate(
        zip(graphs, all_values, all_vectors)
    ):
        rng = np.random.default_rng(seed + index)
        condition = select_band(values, vectors, band, k, rng=rng)
        n = graph.number_of_nodes()
        pair[index, :n, :n] = _normalize_pair_channel(
            condition.rank_one_pair_channel()
        ).astype(np.float32)
        eigenvalues[index] = (condition.eigvals - 1.0).astype(np.float32)
        present[index, 0] = 1.0

    return ConditionTensors(
        pair=torch.from_numpy(pair),
        eigenvalues=torch.from_numpy(eigenvalues),
        present=torch.from_numpy(present),
    )
