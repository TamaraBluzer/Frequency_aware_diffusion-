"""MMD estimator and kernels.

Vendored from DiGress, which took it from GRAN, which took it from GraphRNN. Kept
byte-compatible in behaviour so our numbers stay comparable to SPECTRE Table 1, with two
deliberate departures:

  * The EMD kernels are dropped. They need `pyemd`, which has no Windows wheel past py3.9,
    and every number DiGress and SPECTRE actually report uses `gaussian_tv`.
  * `disc` is single-threaded. The upstream ThreadPoolExecutor buys nothing under the GIL for
    these small numpy reductions and makes results order-dependent when a worker raises.
"""

from __future__ import annotations

import numpy as np

__all__ = ["gaussian_tv", "gaussian", "disc", "compute_mmd"]


def _pad_to_common_support(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < len(y):
        x = np.hstack((x, np.zeros(len(y) - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, np.zeros(len(x) - len(y))))
    return x, y


def gaussian_tv(x, y, sigma: float = 1.0) -> float:
    """Gaussian kernel on total-variation distance. The kernel used for all reported MMDs."""
    x, y = _pad_to_common_support(x, y)
    dist = np.abs(x - y).sum() / 2.0
    return float(np.exp(-dist * dist / (2 * sigma * sigma)))


def gaussian(x, y, sigma: float = 1.0) -> float:
    """Gaussian kernel on L2 distance. Used for orbit counts, which are not histograms."""
    x, y = _pad_to_common_support(x, y)
    dist = np.linalg.norm(x - y, 2)
    return float(np.exp(-dist * dist / (2 * sigma * sigma)))


def disc(samples1, samples2, kernel, **kwargs) -> float:
    """Mean pairwise kernel value between two sample sets."""
    if len(samples1) == 0 or len(samples2) == 0:
        return 1e6
    total = 0.0
    for s1 in samples1:
        for s2 in samples2:
            total += kernel(s1, s2, **kwargs)
    return total / (len(samples1) * len(samples2))


def compute_mmd(samples1, samples2, kernel, is_hist: bool = True, **kwargs) -> float:
    """Squared MMD between two sets of graph descriptors.

    With is_hist=True each descriptor is normalized to a pmf first, which is what makes the
    degree/clustering/spectral/wavelet numbers invariant to graph size. Orbit counts pass
    is_hist=False because they are already per-node averages, not histograms.
    """
    if is_hist:
        samples1 = [s / (np.sum(s) + 1e-6) for s in samples1]
        samples2 = [s / (np.sum(s) + 1e-6) for s in samples2]
    return (
        disc(samples1, samples1, kernel, **kwargs)
        + disc(samples2, samples2, kernel, **kwargs)
        - 2 * disc(samples1, samples2, kernel, **kwargs)
    )
