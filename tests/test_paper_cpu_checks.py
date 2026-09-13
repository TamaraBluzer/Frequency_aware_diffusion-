from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import paper_cpu_checks as checks


def test_seed_spread_uses_sample_standard_deviation():
    result = checks.stats([116.27, 146.79, 162.67])
    assert result["mean"] == pytest.approx(141.91)
    assert result["sample_sd"] == pytest.approx(23.58178958433816)
    with pytest.raises(ValueError):
        checks.stats([1.0])


@pytest.mark.parametrize("n", [4, 7, 64])
def test_snapshot_preserves_node_order_and_edges(n):
    graphs = [nx.path_graph(n), nx.empty_graph(n), nx.complete_graph(n)]
    packed = checks.pack_graphs(graphs)
    restored = checks.unpack_graphs(packed)
    np.testing.assert_array_equal(checks.upper_triangle_matrix(graphs), checks.upper_triangle_matrix(restored))


def test_snapshot_rejects_invalid_length():
    with pytest.raises(ValueError):
        checks.unpack_graphs({"n_nodes": 64, "edges": ["00"]})


def test_retrieval_does_not_count_ties_as_unique_matches():
    graphs = checks.pack_graphs([nx.path_graph(4), nx.path_graph(4)])
    snapshot = {"reference": graphs, "arms": {
        f"{band}_seed{s}": {"band": band, "seed": s, "graphs": graphs}
        for band in checks.BANDS for s in range(3)}}
    audit = checks.copying_audit(snapshot)
    assert all(row["unique_top1_count"] == 0 for row in audit["arms"].values())
    assert all(row["tied_top1_rows"] == 2 for row in audit["arms"].values())


def test_pilot_component_means_recover_aggregate():
    for row in checks.numeric_summary()["pilot"].values():
        assert np.mean([v["mean"] for v in row["components"].values()]) == pytest.approx(row["ratio"]["mean"])


def test_committed_snapshot_reproduces_copying_and_validity_report():
    snapshot = json.loads(checks.SNAPSHOT.read_text(encoding="utf-8"))
    report = json.loads(checks.REPORT.read_text(encoding="utf-8"))
    assert checks.digest(checks.SNAPSHOT) == report["sample_snapshot_sha256"]
    assert checks.copying_audit(snapshot) == report["copying"]
    assert report["copying"]["aggregates"]["low"]["unique_top1_count"] == 96
    assert report["copying"]["aggregates"]["high"]["unique_top1_count"] == 96


def table_rows(label):
    tex = (REPO / "paper" / "main.tex").read_text(encoding="utf-8")
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", tex, flags=re.S)
    table = next(t for t in tables if rf"\label{{{label}}}" in t)
    return {line.split("&")[0].strip(): line for line in table.splitlines() if "&" in line}


def mean_sd(row, digits=1, leading_zero=True):
    values = [f"{row['mean']:.{digits}f}", f"{row['sample_sd']:.{digits}f}"]
    if not leading_zero:
        values = [value.removeprefix("0") for value in values]
    return r" $\pm$ ".join(values)


def test_manuscript_pilot_and_component_tables_match_sources():
    pilot = checks.numeric_summary()["pilot"]
    rows = table_rows("tab:k8")
    for band, row in pilot.items():
        line = rows[rf"\texttt{{{band}}}"]
        assert mean_sd(row["ratio"]) in line
        assert mean_sd(row["validation_loss"], 3, False) in line
    rows = table_rows("tab:components")
    labels = ["Degree", "Clustering", "Spectrum", "Wavelet", "Orbit"]
    for metric, label in zip(checks.METRICS, labels):
        for band in ("none", "low"):
            assert mean_sd(pilot[band]["components"][metric]) in rows[label]


def test_manuscript_confirmation_table_matches_sample_sd():
    numeric = checks.numeric_summary()["seedcheck"]
    rows = table_rows("tab:spikes")
    for band, k in (("none", 2), ("low", 8), ("high", 32)):
        key = rf"\texttt{{{band}}}" + (rf", $k{{=}}{k}$" if band != "none" else "")
        assert mean_sd(numeric[f"{band}_k{k}"]["ratio"]) in rows[key]


def test_recovery_comparisons_use_seed_zero_scores():
    tex = (REPO / "paper" / "main.tex").read_text(encoding="utf-8")
    comparison = tex.split(r"\paragraph{Reading the two axes.}")[1].split(r"\begin{figure*}")[0]
    sweep = checks.load_result("k_sweep_planar.json")["rows"]
    probe = checks.load_result("condition_leakage_planar.json")["rows"]
    for band, k in (("low", 8), ("low", 16), ("high", 16)):
        score = next(r["ratio"] for r in sweep if r["band"] == band and r["k"] == k and r["seed"] == 0)
        recovery = next(r["edge_recovery"] for r in probe if r["band"] == band and r["k"] == k)
        assert f"{score:.1f}" in comparison
        assert f"{100 * recovery:.1f}\\%" in comparison
    assert "single-seed scores" in comparison
    assert "not the low-$k{=}8$ three-seed mean" in comparison


def test_confirmation_caption_identifies_separate_execution_batch():
    tex = (REPO / "paper" / "main.tex").read_text(encoding="utf-8")
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", tex, flags=re.S)
    table = next(t for t in tables if r"\label{tab:spikes}" in t)
    caption = table.split(r"\caption{")[1].split(r"\label{")[0]
    numeric = checks.numeric_summary()
    assert f"{numeric['pilot']['low']['ratio']['mean']:.1f}" in caption
    assert f"{numeric['seedcheck']['low_k8']['ratio']['mean']:.1f}" in caption
    assert "separate set of runs" in caption


def test_manuscript_leakage_table_matches_corrected_probe():
    rows = table_rows("tab:leakage")
    probe = checks.load_result("condition_leakage_planar.json")
    for band in ("low", "high", "random", "gaussian", "shuffled"):
        cells = rows[rf"\texttt{{{band}}}"].split("&")[1:]
        for k, cell in zip((2, 4, 8, 16, 32), cells):
            value = next(r["edge_recovery"] for r in probe["rows"] if r["band"] == band and r["k"] == k)
            assert f"{100 * value:.1f}" in cell


def test_manuscript_copying_table_matches_snapshot():
    report = json.loads(checks.REPORT.read_text(encoding="utf-8"))
    rows = table_rows("tab:copying")
    for band, row in report["copying"]["aggregates"].items():
        line = rows[rf"\texttt{{{band}}}"]
        assert mean_sd(row["matched_f1"], 3, False) in line
        assert f"{row['unique_top1_count']}/{row['n_samples']}" in line
    assert report["numeric"] == checks.numeric_summary()


def test_manuscript_downstream_table_matches_classifier_runs():
    rows = table_rows("tab:downstream")
    data = checks.load_result("stage9_downstream.json")
    assert data["cond_band"] == "high" and data["cond_k"] == 32
    assert data["n_seeds"] == 5
    arms = ("real_only", "real_plus_unconditioned", "real_plus_conditioned")
    for size in data["train_sizes"]:
        results = data["results"][f"n{size}"]
        expected = [f"{100 * np.mean(results[arm]['accuracies']):.1f}" for arm in arms]
        assert re.findall(r"\d+\.\d+", rows[str(size)]) == expected
    tex = (REPO / "paper" / "main.tex").read_text(encoding="utf-8")
    section = tex.split(r"\label{sec:downstream}")[1].split(r"\section{")[0]
    for arm in ("real_plus_random", "real_only"):
        mean = np.mean(data["results"]["n80"][arm]["accuracies"])
        assert f"{100 * mean:.1f}\\%" in section


def test_node_count_baseline_reproduces_without_training_diffusion():
    report = json.loads(checks.REPORT.read_text(encoding="utf-8"))
    assert report["node_count_baseline"] == checks.node_count_baseline()


def test_sweep_uses_sample_sd_and_readable_axes(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    import make_paper_figures as figures

    captured = []
    monkeypatch.setattr(figures, "_save", lambda fig, out, name: captured.append(fig))
    figures.fig_sweep(tmp_path)
    fig = captured[0]
    try:
        assert fig.get_figwidth() == pytest.approx(6.3)
        assert fig.get_figheight() >= 2.6
        left, right = fig.axes
        assert left.xaxis.label.get_fontsize() >= 9
        assert right.xaxis.label.get_fontsize() >= 9
        segments = [segment for collection in left.collections if hasattr(collection, "get_segments")
                    for segment in collection.get_segments()]
        expected = checks.numeric_summary()["seedcheck"]["low_k8"]["ratio"]
        assert any(np.allclose(segment[:, 1], [expected["mean"] - expected["sample_sd"],
                                             expected["mean"] + expected["sample_sd"]]) for segment in segments)
    finally:
        plt.close(fig)
