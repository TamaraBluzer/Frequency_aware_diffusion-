"""Top-level scoring entrypoint: runs the full MMD/VUN suite on any list of graphs.

PLAN.md Stage 2 gate: MMD calibration passes (train-vs-train near zero, train-vs-ER
large) and the Planar training-set row lands in the same ballpark as SPECTRE Table 1
(Deg ~1e-4, Clus ~3e-2, Spec ~5e-3).
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from src.eval.mmd import compute_core_mmds, compute_ratio
from src.eval.orbit import orbit_mmd
from src.eval.validity import compute_vun, is_valid_planar, is_valid_sbm
from src.eval.wavelet import wavelet_mmd

VALIDITY_FNS = {
    "planar": is_valid_planar,
    "sbm": is_valid_sbm,
}


def score_graphs(
    generated: list[nx.Graph],
    reference: list[nx.Graph],
    training: list[nx.Graph] | None = None,
    dataset: str = "planar",
    orca_binary: str | Path | None = None,
    sigma: float = 1.0,
) -> dict[str, float | None]:
    """Score `generated` against `reference` (the set MMD is computed against).

    `training` defaults to `reference` and is used for novelty / V.U.N. -- pass it
    explicitly when `reference` is a held-out split distinct from the training set.
    """
    training = training if training is not None else reference

    metrics: dict[str, float | None] = dict(compute_core_mmds(generated, reference, sigma=sigma))
    metrics["orbit"] = orbit_mmd(generated, reference, orca_binary=orca_binary)
    metrics["wavelet"] = wavelet_mmd(generated, reference, sigma=sigma)

    validity_fn = VALIDITY_FNS[dataset]
    vun = compute_vun(generated, training, validity_fn)
    metrics.update({f"vun_{k}": v for k, v in vun.items()})

    return metrics


def compute_training_self_similarity(
    training_split_a: list[nx.Graph], training_split_b: list[nx.Graph], sigma: float = 1.0
) -> dict[str, float]:
    """The self-similarity floor: MMD between two disjoint splits of the training set.

    Every results table must include this row (WORKPLAN.md section 5.4); without it,
    an MMD number in isolation is meaningless.
    """
    return compute_core_mmds(training_split_a, training_split_b, sigma=sigma)


def mmd_calibration_report(
    train_split_a: list[nx.Graph],
    train_split_b: list[nx.Graph],
    er_graphs: list[nx.Graph],
    sigma: float = 1.0,
) -> dict[str, dict[str, float]]:
    """PLAN.md Stage 2 gate data: train-vs-train (near zero) and train-vs-ER (large)."""
    return {
        "train_vs_train": compute_core_mmds(train_split_a, train_split_b, sigma=sigma),
        "train_vs_er": compute_core_mmds(train_split_a, er_graphs, sigma=sigma),
    }


def summarize_with_ratio(
    metrics: dict[str, float | None], train_self_mmds: dict[str, float]
) -> dict[str, float | None]:
    """Attach the Ratio headline scalar (mean of MMD / train-self-MMD across shared keys)."""
    numeric_mmds = {
        k: v for k, v in metrics.items() if v is not None and k in train_self_mmds
    }
    out = dict(metrics)
    out["ratio"] = compute_ratio(numeric_mmds, train_self_mmds) if numeric_mmds else None
    return out
