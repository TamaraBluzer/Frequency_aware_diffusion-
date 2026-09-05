"""Normalized Laplacian, eigendecomposition, band selection, and an on-disk cache.

Design decisions inherited from WORKPLAN.md §2 (G3, G4, G6) and §3.1:

  * `L_norm = I - D^-1/2 A D^-1/2` throughout, never the combinatorial `L = D - A`. The
    combinatorial Laplacian mixes degree (a local property) into the low-frequency components
    and its eigenvalues are not comparable across graph sizes; `L_norm` is bounded in [0, 2].
  * *All* trivial eigenpairs are dropped, not just the first. A graph with `c` connected
    components has `c` zero eigenvalues whose eigenvectors are component indicators, carrying
    no frequency information. Dropping exactly one (as SPECTRE's `eigvals[1:]` does) leaves
    `c-1` constant directions inside the band: at `k=2` on a 2-component graph that is half the
    conditioning budget. This follows DiGress's `get_eigenvalues_features`, which offsets the
    band by `n_connected_components`.
  * Every band produces a `(k,)` eigenvalue vector and an `(n, k)` eigenvector matrix, so all
    arms are dimension-matched by construction and the only difference between them is *which*
    eigenpairs are selected. That is what makes the frequency claim falsifiable rather than a
    statement about parameter count.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
from scipy.linalg import eigh

from ..paths import data_dir

__all__ = [
    "BANDS",
    "TRIVIAL_EIGVAL_TOL",
    "SpectralCondition",
    "normalized_laplacian",
    "eigendecomposition",
    "count_trivial_eigenpairs",
    "select_band",
    "cluster_condition",
    "cached_eigendecompositions",
]

BANDS = ("low", "high", "random", "gaussian", "cluster", "none")

# Zero eigenvalues of L_norm come out at ~1e-15 in float64. A connected graph's lambda_2 can be
# genuinely small but not this small, so the threshold separates "exactly zero" from "small".
TRIVIAL_EIGVAL_TOL = 1e-8


@dataclass(frozen=True)
class SpectralCondition:
    """The conditioning signal C_k for one graph.

    `eigvals` is `(k,)` and `eigvecs` is `(n, k)`. Both are empty for the `none` arm. `indices`
    records which eigenpairs were selected, which the `random` arm needs for reproducibility
    and every arm needs for debugging.
    """

    eigvals: np.ndarray
    eigvecs: np.ndarray
    indices: np.ndarray
    band: str
    k: int
    n_components: int = 1

    @property
    def n_nodes(self) -> int:
        return self.eigvecs.shape[0]

    def rank_one_pair_channel(self) -> np.ndarray:
        """`U_k diag(lambda_k) U_k^T`, the pair-token channel from PLAN.md Stage 6."""
        if self.k == 0:
            raise ValueError("the 'none' arm has no pair channel")
        return self.eigvecs @ np.diag(self.eigvals) @ self.eigvecs.T


def normalized_laplacian(graph: nx.Graph) -> np.ndarray:
    """`I - D^-1/2 A D^-1/2`, with isolated nodes contributing a zero row rather than a NaN."""
    adjacency = nx.to_numpy_array(graph, dtype=float)
    degrees = adjacency.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    normalized = adjacency * inv_sqrt[:, None] * inv_sqrt[None, :]
    return np.eye(len(degrees)) - normalized


def eigendecomposition(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Ascending eigenvalues and matching eigenvectors of `L_norm`.

    `scipy.linalg.eigh` is used rather than a sparse solver: `n <= 200` here, we need the whole
    spectrum for the `high` band anyway, and a dense symmetric solve is both faster and more
    numerically trustworthy at this size.
    """
    eigvals, eigvecs = eigh(normalized_laplacian(graph))
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]


def count_trivial_eigenpairs(eigvals: np.ndarray, tol: float = TRIVIAL_EIGVAL_TOL) -> int:
    """Number of zero eigenvalues of `L_norm`, which equals the connected-component count."""
    return int(np.sum(eigvals < tol))


def select_band(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    band: str,
    k: int,
    rng: np.random.Generator | None = None,
    n_trivial: int | None = None,
) -> SpectralCondition:
    """Pick the `k` eigenpairs for one arm.

    The first `n_trivial` eigenpairs are never eligible. That count defaults to the number of
    zero eigenvalues (equivalently, connected components) rather than a hardcoded 1, so a
    disconnected graph does not spend part of its band on constant component indicators. Pass
    `n_trivial` explicitly to override, e.g. with a structurally computed component count.
    """
    if band not in BANDS:
        raise ValueError(f"unknown band {band!r}; expected one of {BANDS}")
    n = eigvecs.shape[0]
    n_trivial = count_trivial_eigenpairs(eigvals) if n_trivial is None else n_trivial
    n_trivial = max(n_trivial, 1)

    if band == "none" or k == 0:
        return SpectralCondition(
            eigvals=np.zeros(0),
            eigvecs=np.zeros((n, 0)),
            indices=np.zeros(0, dtype=int),
            band="none",
            k=0,
            n_components=n_trivial,
        )

    if band == "gaussian":
        # Identical shape to a real band, zero structural information. The strictest control:
        # if low-frequency does not beat this, the result is about capacity, not frequency.
        if rng is None:
            raise ValueError("the 'gaussian' arm requires an rng")
        return SpectralCondition(
            eigvals=rng.standard_normal(k),
            eigvecs=rng.standard_normal((n, k)),
            indices=np.full(k, -1),
            band=band,
            k=k,
            n_components=n_trivial,
        )

    eligible = np.arange(n_trivial, n)
    if k > len(eligible):
        raise ValueError(
            f"k={k} exceeds the {len(eligible)} non-trivial eigenpairs of an n={n} graph "
            f"with {n_trivial} connected component(s)"
        )

    if band == "low":
        indices = eligible[:k]
    elif band == "high":
        indices = eligible[-k:]
    elif band == "random":
        if rng is None:
            raise ValueError("the 'random' arm requires an rng")
        indices = np.sort(rng.choice(eligible, size=k, replace=False))
    else:
        raise ValueError(f"band {band!r} is not an eigenpair band; use cluster_condition")

    return SpectralCondition(
        eigvals=eigvals[indices],
        eigvecs=eigvecs[:, indices],
        indices=indices,
        band=band,
        k=k,
        n_components=n_trivial,
    )


def cluster_condition(graph: nx.Graph, n_clusters: int, seed: int = 0) -> np.ndarray:
    """DualDiff-style hard partition, returned as an `(n, n_clusters)` one-hot matrix.

    This is a *baseline arm*, not a frequency band: it collapses the spectrum into a single
    integer `K`, which is exactly the contrast we want to draw. Spectral clustering is used
    (rather than K-means on coordinates) because these graphs have no node features.
    """
    from sklearn.cluster import SpectralClustering

    adjacency = nx.to_numpy_array(graph, dtype=float)
    labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
    ).fit_predict(adjacency)
    one_hot = np.zeros((graph.number_of_nodes(), n_clusters))
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def _cache_key(graphs: Sequence[nx.Graph], tag: str) -> str:
    """Hash graph structure, not object identity, so the cache survives reloading."""
    digest = hashlib.sha256(tag.encode())
    for graph in graphs:
        digest.update(str(graph.number_of_nodes()).encode())
        digest.update(nx.weisfeiler_lehman_graph_hash(graph).encode())
    return digest.hexdigest()[:16]


def cached_eigendecompositions(
    graphs: Sequence[nx.Graph],
    tag: str,
    cache_dir: Path | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Eigendecompose a list of graphs, memoized on disk.

    The spectrum of a training graph never changes, so recomputing it every epoch is a large
    silent cost (WORKPLAN.md §9). Keyed by graph structure so a changed split misses the cache
    instead of silently returning the wrong spectra.
    """
    cache_dir = cache_dir or (data_dir() / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"eig_{tag}_{_cache_key(graphs, tag)}.npz"

    if path.exists():
        payload = np.load(path)
        count = int(payload["count"])
        return (
            [payload[f"vals_{i}"] for i in range(count)],
            [payload[f"vecs_{i}"] for i in range(count)],
        )

    all_vals, all_vecs = [], []
    for graph in graphs:
        vals, vecs = eigendecomposition(graph)
        all_vals.append(vals)
        all_vecs.append(vecs)

    payload = {"count": len(graphs)}
    for i, (vals, vecs) in enumerate(zip(all_vals, all_vecs)):
        payload[f"vals_{i}"] = vals
        payload[f"vecs_{i}"] = vecs
    np.savez_compressed(path, **payload)
    return all_vals, all_vecs
