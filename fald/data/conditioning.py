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


def _derangement(size: int, seed: int) -> np.ndarray:
    """A permutation with no fixed point, so no graph receives its own spectrum."""
    if size < 2:
        raise ValueError("the shuffled-donor control needs at least two graphs")
    rng = np.random.default_rng(seed)
    while True:
        candidate = rng.permutation(size)
        if not np.any(candidate == np.arange(size)):
            return candidate


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
    if band not in {"none", "low", "high", "random", "gaussian", "shuffled"}:
        raise ValueError(
            f"unsupported band {band!r}; expected none/low/high/random/gaussian/shuffled"
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

    # The shuffled-donor control pairs each graph with another graph's spectrum. It is a real
    # low band with real eigenvalue statistics, just not *this* graph's, which separates
    # "low-frequency structure helps" from "the matching condition helps". A derangement is
    # used so no graph can be handed its own spectrum back.
    donor_band = band
    if band == "shuffled":
        donor_band = "low"
        donors = _derangement(len(graphs), seed)
    else:
        donors = np.arange(len(graphs))

    for index, graph in enumerate(graphs):
        donor = int(donors[index])
        values, vectors = all_values[donor], all_vectors[donor]
        rng = np.random.default_rng(seed + index)
        condition = select_band(values, vectors, donor_band, k, rng=rng)
        n = graph.number_of_nodes()
        channel = _normalize_pair_channel(condition.rank_one_pair_channel())
        # A donor with a different node count is cropped or zero-padded to the target's size.
        # Equal-n datasets (Planar, SBM) never hit this; it keeps the control well-defined.
        extent = min(n, channel.shape[0])
        pair[index, :extent, :extent] = channel[:extent, :extent].astype(np.float32)
        eigenvalues[index] = (condition.eigvals - 1.0).astype(np.float32)
        present[index, 0] = 1.0

    return ConditionTensors(
        pair=torch.from_numpy(pair),
        eigenvalues=torch.from_numpy(eigenvalues),
        present=torch.from_numpy(present),
    )
