"""SPECTRE's Wavelet MMD: 12 ab-spline graph wavelet kernels via PyGSP.

PLAN.md Stage 2: "Add SPECTRE's Wavelet MMD (12 ab-spline kernels via PyGSP), which
DiGress omits." This gives us a spectral-similarity signal at multiple scales, distinct
from the single eigenvalue-histogram `spectral_mmd` in mmd.py.

If PyGSP is unavailable (e.g. not yet installed locally), `wavelet_mmd` returns None so
the rest of the harness keeps working -- mirrors the graceful-degradation convention in
orbit.py.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.eval.mmd import mmd_from_histograms, _normalize_histogram

NUM_SCALES = 12


def _wavelet_coefficients(graph: nx.Graph, num_scales: int = NUM_SCALES) -> np.ndarray | None:
    """Per-node ab-spline wavelet coefficient energies, summed into one histogram."""
    try:
        import pygsp
    except ImportError:
        return None

    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        return None

    adjacency = nx.to_numpy_array(graph)
    g = pygsp.graphs.Graph(adjacency)
    g.compute_fourier_basis()

    filter_bank = pygsp.filters.Abspline(g, Nf=num_scales)
    # Coefficients: (n_nodes, num_scales) from a per-node delta signal, i.e. the
    # wavelet basis itself evaluated at every node.
    signals = np.eye(g.N)
    coeffs = filter_bank.filter(signals, method="exact")
    if coeffs.ndim == 3:
        # pygsp returns (n_nodes, n_signals, Nf); we used one-hot signals so
        # the diagonal over (node, signal) gives each node's own response.
        energies = np.linalg.norm(coeffs[np.arange(g.N), np.arange(g.N), :], axis=0)
    else:
        energies = np.linalg.norm(coeffs, axis=0)
    return energies


def wavelet_histogram(graph: nx.Graph, num_bins: int = 100, num_scales: int = NUM_SCALES) -> np.ndarray | None:
    energies = _wavelet_coefficients(graph, num_scales=num_scales)
    if energies is None:
        return None
    hist, _ = np.histogram(energies, bins=num_bins)
    return _normalize_histogram(hist)


def wavelet_mmd(
    graphs_a: list[nx.Graph], graphs_b: list[nx.Graph], sigma: float = 1.0, num_bins: int = 100
) -> float | None:
    """Wavelet-energy-histogram MMD, or None if PyGSP is unavailable / fails on any graph."""
    hists_a, hists_b = [], []
    for g in graphs_a:
        h = wavelet_histogram(g, num_bins=num_bins)
        if h is None:
            return None
        hists_a.append(h)
    for g in graphs_b:
        h = wavelet_histogram(g, num_bins=num_bins)
        if h is None:
            return None
        hists_b.append(h)
    return mmd_from_histograms(hists_a, hists_b, sigma=sigma)
