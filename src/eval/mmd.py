"""Core MMD suite: degree, clustering, spectral histograms, total-variation Gaussian kernel.

Decoupled from any training framework (PLAN.md Stage 2): every function here takes and
returns plain `networkx.Graph` objects / numpy arrays, so it can score any list of graphs
regardless of which model produced them.

Kernel convention follows GRAN/SPECTRE: total-variation Gaussian kernel on histograms,
not Gaussian-EMD (WORKPLAN.md section 5.1).
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def _pad_histograms(hists: list[np.ndarray]) -> np.ndarray:
    """Stack variable-length histograms into one array, zero-padding the short ones."""
    max_len = max((len(h) for h in hists), default=0)
    max_len = max(max_len, 1)
    padded = np.zeros((len(hists), max_len), dtype=np.float64)
    for i, h in enumerate(hists):
        if len(h) > 0:
            padded[i, : len(h)] = h
    return padded


def _normalize_histogram(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return counts.astype(np.float64)
    return counts.astype(np.float64) / total


def gaussian_tv_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Gaussian kernel over total-variation distance between two histograms."""
    tv_dist = 0.5 * np.abs(x - y).sum()
    return float(np.exp(-(tv_dist**2) / (2 * sigma**2)))


def mmd_from_histograms(hists_a: list[np.ndarray], hists_b: list[np.ndarray], sigma: float = 1.0) -> float:
    """Squared MMD between two sets of (already-normalized) histograms, TV-Gaussian kernel.

    Standard unbiased-ish plug-in estimator:
        MMD^2 = mean_i,i' k(a_i,a_i') + mean_j,j' k(b_j,b_j') - 2 mean_i,j k(a_i,b_j)
    """
    a = _pad_histograms(hists_a)
    b = _pad_histograms(hists_b)
    max_len = max(a.shape[1], b.shape[1])
    if a.shape[1] < max_len:
        a = np.pad(a, ((0, 0), (0, max_len - a.shape[1])))
    if b.shape[1] < max_len:
        b = np.pad(b, ((0, 0), (0, max_len - b.shape[1])))

    def kernel_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        tv = 0.5 * np.abs(x[:, None, :] - y[None, :, :]).sum(axis=-1)
        return np.exp(-(tv**2) / (2 * sigma**2))

    kaa = kernel_matrix(a, a).mean()
    kbb = kernel_matrix(b, b).mean()
    kab = kernel_matrix(a, b).mean()
    return float(kaa + kbb - 2 * kab)


def degree_histogram(graph: nx.Graph, max_degree: int | None = None) -> np.ndarray:
    degrees = [d for _, d in graph.degree()]
    if not degrees:
        return np.zeros(1)
    max_d = max_degree if max_degree is not None else max(degrees)
    counts = np.bincount(degrees, minlength=max_d + 1).astype(np.float64)
    return _normalize_histogram(counts)


def clustering_histogram(graph: nx.Graph, num_bins: int = 100) -> np.ndarray:
    values = list(nx.clustering(graph).values())
    if not values:
        return np.zeros(num_bins)
    counts, _ = np.histogram(values, bins=num_bins, range=(0.0, 1.0))
    return _normalize_histogram(counts)


def spectral_histogram(graph: nx.Graph, num_bins: int = 200) -> np.ndarray:
    """Normalized-Laplacian eigenvalue histogram, eigenvalues in [0, 2]."""
    if graph.number_of_nodes() == 0:
        return np.zeros(num_bins)
    laplacian = nx.normalized_laplacian_matrix(graph).toarray()
    eigenvalues = np.linalg.eigvalsh(laplacian)
    counts, _ = np.histogram(eigenvalues, bins=num_bins, range=(0.0, 2.0))
    return _normalize_histogram(counts)


def degree_mmd(graphs_a: list[nx.Graph], graphs_b: list[nx.Graph], sigma: float = 1.0) -> float:
    max_degree = 0
    for g in graphs_a + graphs_b:
        degs = [d for _, d in g.degree()]
        if degs:
            max_degree = max(max_degree, max(degs))
    hists_a = [degree_histogram(g, max_degree) for g in graphs_a]
    hists_b = [degree_histogram(g, max_degree) for g in graphs_b]
    return mmd_from_histograms(hists_a, hists_b, sigma=sigma)


def clustering_mmd(graphs_a: list[nx.Graph], graphs_b: list[nx.Graph], sigma: float = 1.0, num_bins: int = 100) -> float:
    hists_a = [clustering_histogram(g, num_bins) for g in graphs_a]
    hists_b = [clustering_histogram(g, num_bins) for g in graphs_b]
    return mmd_from_histograms(hists_a, hists_b, sigma=sigma)


def spectral_mmd(graphs_a: list[nx.Graph], graphs_b: list[nx.Graph], sigma: float = 1.0, num_bins: int = 200) -> float:
    hists_a = [spectral_histogram(g, num_bins) for g in graphs_a]
    hists_b = [spectral_histogram(g, num_bins) for g in graphs_b]
    return mmd_from_histograms(hists_a, hists_b, sigma=sigma)


def compute_core_mmds(
    generated: list[nx.Graph], reference: list[nx.Graph], sigma: float = 1.0
) -> dict[str, float]:
    """Deg/Clus/Spec MMD between a generated set and a reference set."""
    return {
        "degree": degree_mmd(generated, reference, sigma=sigma),
        "clustering": clustering_mmd(generated, reference, sigma=sigma),
        "spectral": spectral_mmd(generated, reference, sigma=sigma),
    }


def compute_ratio(mmds: dict[str, float], train_self_mmds: dict[str, float], eps: float = 1e-8) -> float:
    """Ratio summary metric: mean of (our MMD / training-set self-similarity MMD).

    Every ratio term uses the training-set MMD row as the denominator floor
    (WORKPLAN.md section 5.4) -- without it the MMD numbers are meaningless.
    """
    shared_keys = [k for k in mmds if k in train_self_mmds]
    if not shared_keys:
        raise ValueError("No overlapping metric keys between mmds and train_self_mmds")
    ratios = [mmds[k] / max(train_self_mmds[k], eps) for k in shared_keys]
    return float(np.mean(ratios))
