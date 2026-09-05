"""Planar / SBM / community datasets, loaded straight from SPECTRE's released tensors.

We download and split the SPECTRE `.pt` files ourselves rather than going through DiGress's
`SpectreGraphDataset` for two reasons:

  * DiGress's `process()` appends each graph to `data_list` twice, so its processed splits are
    duplicated (its "40 test graphs" are reported as 80). MMD is invariant to duplicating a
    sample, so their published numbers are unaffected, but it doubles descriptor cost and makes
    set sizes misleading.
  * The evaluation harness should not depend on the gitignored third_party/ checkout.

The split is reproduced exactly: 200 graphs, 20% test, 80% of the remainder train, seeded with
`torch.Generator().manual_seed(0)`, matching DiGress so our numbers stay comparable.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np
import torch

__all__ = ["DATASETS", "load_splits", "raw_file"]

DATASETS = {
    "planar": ("planar_64_200.pt", 200),
    "sbm": ("sbm_200.pt", 200),
    "comm20": ("community_12_21_100.pt", 100),
}

_BASE_URL = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_file(name: str) -> Path:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(DATASETS)}")
    filename, _ = DATASETS[name]
    target = _repo_root() / "data" / "raw" / filename
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{_BASE_URL}/{filename}", target)
    return target


def _to_networkx(adj: torch.Tensor) -> nx.Graph:
    dense = adj.cpu().numpy()
    return nx.from_numpy_array((dense > 0).astype(np.uint8))


def load_splits(name: str = "planar") -> dict[str, list[nx.Graph]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} as networkx graphs."""
    _, num_graphs = DATASETS[name]
    payload = torch.load(raw_file(name), weights_only=False)
    adjs = payload[0]

    generator = torch.Generator()
    generator.manual_seed(0)
    test_len = int(round(num_graphs * 0.2))
    train_len = int(round((num_graphs - test_len) * 0.8))
    val_len = num_graphs - train_len - test_len
    indices = torch.randperm(num_graphs, generator=generator)

    bounds = {
        "train": indices[:train_len],
        "val": indices[train_len:train_len + val_len],
        "test": indices[train_len + val_len:],
    }
    return {
        split: [_to_networkx(adjs[int(i)]) for i in idx]
        for split, idx in bounds.items()
    }
