"""Per-graph descriptors that the MMD metrics compare.

Each function turns one networkx graph into a fixed-meaning vector. The binning and support
choices are copied from DiGress/SPECTRE deliberately: change them and our numbers stop being
comparable to published tables.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.linalg import eigh, eigvalsh

from . import orca

__all__ = [
    "degree_histogram",
    "clustering_histogram",
    "laplacian_spectrum",
    "eigen_decomposition",
    "wavelet_descriptor",
    "orbit_descriptor",
    "wavelet_filter_bank",
    "WAVELET_N_FILTERS",
]

CLUSTERING_BINS = 100
WAVELET_N_FILTERS = 12
WAVELET_HIST_BINS = 100


def degree_histogram(graph) -> np.ndarray:
    return np.array(nx.degree_histogram(graph), dtype=float)


def clustering_histogram(graph, bins: int = CLUSTERING_BINS) -> np.ndarray:
    coeffs = list(nx.clustering(graph).values())
    hist, _ = np.histogram(coeffs, bins=bins, range=(0.0, 1.0), density=False)
    return hist.astype(float)


def laplacian_spectrum(graph, n_eigvals: int = -1) -> np.ndarray:
    """Eigenvalues of the normalized Laplacian, in [0, 2]."""
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(graph).todense())
    except Exception:
        # Matches upstream: an empty or degenerate graph contributes a zero spectrum rather
        # than removing the graph from the sample and biasing the comparison.
        eigs = np.zeros(graph.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1:n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    return spectral_pmf.astype(float) / len(eigs) if len(eigs) else spectral_pmf.astype(float)


def eigen_decomposition(graph) -> tuple[np.ndarray, np.ndarray]:
    """Full eigendecomposition of the normalized Laplacian, as (eigenvalues, eigenvectors)."""
    laplacian = nx.normalized_laplacian_matrix(graph).todense()
    eigvals, eigvecs = eigh(laplacian)
    return np.asarray(eigvals), np.asarray(eigvecs)


def wavelet_filter_bank(n_filters: int = WAVELET_N_FILTERS):
    """SPECTRE's ab-spline wavelet bank, defined on the fixed spectral range [0, 2].

    The bank is built against a dummy graph whose lmax is pinned to 2 so that every graph in
    the dataset shares one filter bank; the normalized Laplacian guarantees lmax <= 2.
    """
    import pygsp as pg

    class _DummyNormalizedGraph:
        lmax = 2

    filters = pg.filters.Abspline(_DummyNormalizedGraph, n_filters)
    bound = float(np.max(filters.evaluate(np.arange(0, 2, 0.01))))
    return filters, bound


def wavelet_descriptor(eigvals, eigvecs, filters, bound: float) -> np.ndarray:
    """Histogrammed energies of each wavelet operator, flattened over the 12 scales."""
    responses = filters.evaluate(eigvals)
    operators = np.array([eigvecs @ np.diag(response) @ eigvecs.T for response in responses])
    energies = np.sum(operators ** 2, axis=2)
    hist = np.array([
        np.histogram(row, range=[0, bound], bins=WAVELET_HIST_BINS)[0] for row in energies
    ])
    return hist.flatten().astype(float)


def orbit_descriptor(graph) -> np.ndarray:
    """Per-node-averaged counts of the 15 four-node orbits. Not a histogram."""
    counts = orca.node_orbit_counts(graph)
    return np.sum(counts, axis=0) / graph.number_of_nodes()
