"""Dense padded batching for variable-size graphs.

The augmented-edge transformer operates on a dense `(n, n, d)` pair tensor, so graphs are padded
to the batch maximum and carried with a node mask. Planar is uniform at n=64, but SBM spans
49-187 nodes, so padding is unavoidable there and the mask has to be respected everywhere: an
unmasked pad row contributes to attention, to LayerNorm statistics, and to the loss.
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["DenseGraphDataset", "collate_dense", "graphs_to_dense", "pair_mask_from_node_mask"]


def graphs_to_dense(graphs: Sequence[nx.Graph]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return adjacency `(B, n_max, n_max)` float and node mask `(B, n_max)` bool."""
    sizes = [g.number_of_nodes() for g in graphs]
    n_max = max(sizes)
    adjacency = torch.zeros(len(graphs), n_max, n_max)
    node_mask = torch.zeros(len(graphs), n_max, dtype=torch.bool)

    for i, graph in enumerate(graphs):
        n = graph.number_of_nodes()
        dense = nx.to_numpy_array(graph, dtype=np.float32)
        adjacency[i, :n, :n] = torch.from_numpy((dense > 0).astype(np.float32))
        node_mask[i, :n] = True

    return adjacency, node_mask


def pair_mask_from_node_mask(node_mask: torch.Tensor, include_diagonal: bool = False) -> torch.Tensor:
    """`(B, n, n)` mask of node pairs where both endpoints are real nodes."""
    pair = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    if not include_diagonal:
        n = node_mask.shape[1]
        eye = torch.eye(n, dtype=torch.bool, device=node_mask.device)
        pair = pair & ~eye.unsqueeze(0)
    return pair


class DenseGraphDataset(Dataset):
    """Holds networkx graphs; densification happens in the collate so padding is per batch."""

    def __init__(self, graphs: Sequence[nx.Graph]):
        self.graphs = list(graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> nx.Graph:
        return self.graphs[idx]


def collate_dense(batch: Sequence[nx.Graph]) -> tuple[torch.Tensor, torch.Tensor]:
    return graphs_to_dense(batch)
