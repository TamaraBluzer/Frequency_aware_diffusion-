import numpy as np
import pytest

from src.data.generate import (
    generate_erdos_renyi_matched,
    generate_planar_dataset,
)
from src.eval.mmd import compute_ratio
from src.eval.report import (
    compute_training_self_similarity,
    mmd_calibration_report,
    score_graphs,
    summarize_with_ratio,
)


def test_compute_ratio_is_one_when_equal():
    mmds = {"degree": 0.01, "clustering": 0.02}
    ratio = compute_ratio(mmds, mmds)
    assert ratio == pytest.approx(1.0)


def test_compute_ratio_requires_overlapping_keys():
    with pytest.raises(ValueError):
        compute_ratio({"foo": 0.1}, {"bar": 0.1})


def test_mmd_calibration_report_shape():
    train_a = generate_planar_dataset(num_graphs=20, num_points=32, seed=10)
    train_b = generate_planar_dataset(num_graphs=20, num_points=32, seed=11)
    er = generate_erdos_renyi_matched(train_a, rng=np.random.default_rng(10))

    report = mmd_calibration_report(train_a, train_b, er)
    assert "train_vs_train" in report
    assert "train_vs_er" in report
    for key in ("degree", "clustering", "spectral"):
        assert key in report["train_vs_train"]
        assert key in report["train_vs_er"]


def test_score_graphs_includes_vun_and_optional_rows():
    generated = generate_planar_dataset(num_graphs=10, num_points=32, seed=20)
    reference = generate_planar_dataset(num_graphs=10, num_points=32, seed=21)

    metrics = score_graphs(generated, reference, dataset="planar")
    assert "degree" in metrics
    assert "clustering" in metrics
    assert "spectral" in metrics
    assert "vun_valid" in metrics
    assert "vun_unique" in metrics
    assert "vun_novel" in metrics
    assert "vun_vun" in metrics
    # orbit/wavelet may legitimately be None if ORCA/PyGSP aren't installed.
    assert "orbit" in metrics
    assert "wavelet" in metrics


def test_summarize_with_ratio_attaches_headline_scalar():
    train_a = generate_planar_dataset(num_graphs=15, num_points=32, seed=30)
    train_b = generate_planar_dataset(num_graphs=15, num_points=32, seed=31)
    train_self_mmds = compute_training_self_similarity(train_a, train_b)

    generated = generate_planar_dataset(num_graphs=15, num_points=32, seed=32)
    metrics = score_graphs(generated, train_b, dataset="planar")
    summarized = summarize_with_ratio(metrics, train_self_mmds)

    assert "ratio" in summarized
    assert summarized["ratio"] is not None
    assert summarized["ratio"] >= 0
