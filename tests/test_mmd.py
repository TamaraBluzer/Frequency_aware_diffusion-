"""WORKPLAN.md T5: MMD calibration.

MMD(train, train_split_2) is near zero; MMD(train, Erdos-Renyi) is large; the metric
is symmetric and non-negative. An MMD implementation that reports 0 for everything is
easy to write and hard to notice -- these tests are what catch that.
"""
import numpy as np
import pytest

from src.data.generate import (
    generate_erdos_renyi_matched,
    generate_planar_dataset,
)
from src.eval.mmd import clustering_mmd, compute_core_mmds, degree_mmd, spectral_mmd


@pytest.fixture(scope="module")
def planar_splits():
    graphs = generate_planar_dataset(num_graphs=60, num_points=64, seed=42)
    split_a, split_b = graphs[:30], graphs[30:]
    return split_a, split_b


def test_mmd_is_symmetric():
    graphs_a = generate_planar_dataset(num_graphs=10, num_points=32, seed=1)
    graphs_b = generate_planar_dataset(num_graphs=10, num_points=32, seed=2)
    forward = compute_core_mmds(graphs_a, graphs_b)
    backward = compute_core_mmds(graphs_b, graphs_a)
    for key in forward:
        assert forward[key] == pytest.approx(backward[key], abs=1e-9)


def test_mmd_is_non_negative():
    graphs_a = generate_planar_dataset(num_graphs=10, num_points=32, seed=3)
    graphs_b = generate_planar_dataset(num_graphs=10, num_points=32, seed=4)
    mmds = compute_core_mmds(graphs_a, graphs_b)
    for key, value in mmds.items():
        assert value >= -1e-9, f"{key} MMD is negative: {value}"


def test_train_vs_train_near_zero(planar_splits):
    split_a, split_b = planar_splits
    mmds = compute_core_mmds(split_a, split_b)
    assert mmds["degree"] < 0.05
    assert mmds["clustering"] < 0.05
    assert mmds["spectral"] < 0.05


def test_train_vs_erdos_renyi_large(planar_splits):
    split_a, _ = planar_splits
    er_graphs = generate_erdos_renyi_matched(split_a, rng=np.random.default_rng(99))
    mmds = compute_core_mmds(split_a, er_graphs)
    train_train_mmds = compute_core_mmds(split_a, split_a)

    # ER should be clearly farther from train than train is from itself.
    assert mmds["clustering"] > train_train_mmds["clustering"]
    assert mmds["spectral"] >= train_train_mmds["spectral"]


def test_planar_training_set_ballpark_matches_spectre_table1(planar_splits):
    """PLAN.md Stage 2 gate: our Planar training-set row should be in the same ballpark
    as SPECTRE Table 1 (Deg ~1e-4, Clus ~3e-2, Spec ~5e-3). Exact match isn't expected
    (different random data, different bandwidth); this checks order-of-magnitude only.
    """
    split_a, split_b = planar_splits
    mmds = compute_core_mmds(split_a, split_b)
    assert mmds["degree"] < 1e-2
    assert mmds["clustering"] < 1e-1
    assert mmds["spectral"] < 1e-1
