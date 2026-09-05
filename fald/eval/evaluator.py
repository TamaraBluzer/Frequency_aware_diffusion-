"""The evaluator.

Scores any list of networkx graphs against any reference list. Deliberately free of
PyTorch Lightning, Hydra, wandb and DiGress imports so that it can be called from a training
loop, a notebook, or a calibration script without dragging a model along.

Five MMD metrics, matching SPECTRE/DiGress kernels and sigmas:

    degree     gaussian_tv, sigma=1.0    histogram
    clustering gaussian_tv, sigma=0.1    histogram, 100 bins on [0, 1]
    spectral   gaussian_tv, sigma=1.0    normalized-Laplacian eigenvalue histogram
    wavelet    gaussian_tv, sigma=1.0    12 ab-spline scales (DiGress vendors but disables this)
    orbit      gaussian_tv, sigma=30.0   per-node 4-orbit counts, not a histogram
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import descriptors, orca, validity
from .mmd import compute_mmd, gaussian_tv

__all__ = ["MMD_SPECS", "GraphEvaluator", "EvalResult"]


@dataclass(frozen=True)
class _MMDSpec:
    name: str
    sigma: float
    is_hist: bool


MMD_SPECS = (
    _MMDSpec("degree", 1.0, True),
    _MMDSpec("clustering", 0.1, True),
    _MMDSpec("spectral", 1.0, True),
    _MMDSpec("wavelet", 1.0, True),
    _MMDSpec("orbit", 30.0, False),
)

# Ratio is a quotient of MMDs, and the denominator is a near-zero self-similarity value.
# Clamping keeps a degenerate reference row from producing a meaningless astronomical ratio.
_RATIO_FLOOR = 1e-8


@dataclass
class EvalResult:
    mmd: dict = field(default_factory=dict)
    vun: dict = field(default_factory=dict)
    ratio: float | None = None
    ratio_per_metric: dict = field(default_factory=dict)
    n_generated: int = 0
    n_reference: int = 0

    def as_flat_dict(self) -> dict:
        out = {f"mmd/{k}": v for k, v in self.mmd.items()}
        out.update({f"vun/{k}": v for k, v in self.vun.items()})
        out.update({f"ratio/{k}": v for k, v in self.ratio_per_metric.items()})
        if self.ratio is not None:
            out["ratio"] = self.ratio
        out["n_generated"] = self.n_generated
        out["n_reference"] = self.n_reference
        return out


class GraphEvaluator:
    """Precomputes reference descriptors once, then scores many candidate sets cheaply.

    Descriptor extraction dominates the cost (the wavelet descriptor builds 12 dense n x n
    operators per graph), so the reference set is described exactly once per evaluator.
    """

    def __init__(
        self,
        reference_graphs: Sequence,
        train_graphs: Sequence | None = None,
        validity_func: Callable | None = None,
        metrics: Sequence[str] | None = None,
    ):
        self.reference_graphs = list(reference_graphs)
        self.train_graphs = list(train_graphs) if train_graphs is not None else None
        self.validity_func = validity_func

        requested = set(metrics) if metrics is not None else {s.name for s in MMD_SPECS}
        if "orbit" in requested and not orca.is_available():
            raise orca.OrcaUnavailable(
                "orbit MMD requested but the ORCA binary is missing. Build it with "
                "scripts/setup_digress.sh, or pass metrics without 'orbit'."
            )
        self.specs = tuple(s for s in MMD_SPECS if s.name in requested)

        if "wavelet" in requested:
            self._filters, self._bound = descriptors.wavelet_filter_bank()
        else:
            self._filters = self._bound = None

        self._reference_descriptors = self._describe(self.reference_graphs)

    def _describe(self, graphs: Sequence) -> dict:
        graphs = [g for g in graphs if g.number_of_nodes() > 0]
        out: dict = {}
        names = {s.name for s in self.specs}

        if "degree" in names:
            out["degree"] = [descriptors.degree_histogram(g) for g in graphs]
        if "clustering" in names:
            out["clustering"] = [descriptors.clustering_histogram(g) for g in graphs]
        if "spectral" in names:
            out["spectral"] = [descriptors.laplacian_spectrum(g) for g in graphs]
        if "orbit" in names:
            out["orbit"] = [descriptors.orbit_descriptor(g) for g in graphs]
        if "wavelet" in names:
            wavelet = []
            for g in graphs:
                eigvals, eigvecs = descriptors.eigen_decomposition(g)
                wavelet.append(
                    descriptors.wavelet_descriptor(eigvals, eigvecs, self._filters, self._bound)
                )
            out["wavelet"] = wavelet
        return out

    def mmd_against_reference(self, graphs: Sequence) -> dict:
        candidate = self._describe(graphs)
        return {
            spec.name: compute_mmd(
                self._reference_descriptors[spec.name],
                candidate[spec.name],
                kernel=gaussian_tv,
                is_hist=spec.is_hist,
                sigma=spec.sigma,
            )
            for spec in self.specs
        }

    def self_similarity(self, graphs: Sequence) -> dict:
        """MMD of a held-out real set against the reference. The floor every metric is read against.

        Without this row an MMD number is uninterpretable: it says nothing about whether 0.01
        is close or far unless you know what two samples of *real* graphs score.
        """
        return self.mmd_against_reference(graphs)

    def evaluate(self, graphs: Sequence, baseline_mmd: dict | None = None) -> EvalResult:
        graphs = list(graphs)
        result = EvalResult(
            n_generated=len(graphs),
            n_reference=len(self.reference_graphs),
        )
        result.mmd = self.mmd_against_reference(graphs)

        if self.train_graphs is not None:
            result.vun = validity.vun(
                graphs,
                self.train_graphs,
                self.validity_func if self.validity_func is not None else (lambda _g: True),
            )

        if baseline_mmd is not None:
            per_metric = {
                name: value / max(baseline_mmd[name], _RATIO_FLOOR)
                for name, value in result.mmd.items()
                if name in baseline_mmd
            }
            result.ratio_per_metric = per_metric
            if per_metric:
                result.ratio = float(np.mean(list(per_metric.values())))

        return result
