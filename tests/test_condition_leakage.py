"""Regression tests pinning the scope of the orientation fix in the leakage probe.

`scripts/condition_leakage.py` used to claim it resolved the pair channel's sign per graph by
keeping "whichever orientation recovers more edges". It did not: the loop compared
`graph.number_of_edges()`, which equals the requested edge count `m` for *both* orientations by
construction, so the strict `>` always kept the first iteration and every committed number was
computed at orientation -1.0 alone.

These tests prove three things:

  * the selection now actually discriminates, because it scores against the true edge set,
  * the shipped `band` rule fixes one sign per (band, k) cell by majority vote, agrees with
    the per-graph `max` bound exactly wherever a cell is unanimous, and can never exceed it,
  * and the fix changed nothing else -- forcing orientation -1.0 reproduces the pre-fix
    numbers bit-for-bit, on both datasets, for all 25 (band, k) rows.

The pre-fix numbers are frozen in `results/condition_leakage_*_single_orientation.json`.

The last group covers the censored distance-to-planarity summary, whose whole job is to
refuse to report a median that right-censoring has taken over.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def _load_probe():
    """Import scripts/condition_leakage.py by path -- `scripts/` is not a package."""
    path = REPO_ROOT / "scripts" / "condition_leakage.py"
    spec = importlib.util.spec_from_file_location("condition_leakage_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def _dataset_available(name: str) -> bool:
    """True when the raw split is already on disk. `raw_file` would otherwise download it."""
    from fald.data.spectre import DATASETS
    from fald.paths import data_dir

    if name not in DATASETS:
        return False
    return (data_dir() / "raw" / DATASETS[name][0]).exists()


def _rows(path: Path) -> dict[tuple[str, int], dict]:
    payload = json.loads(path.read_text())
    return {(row["band"], row["k"]): row for row in payload["rows"]}


# --------------------------------------------------------------------------------------
# The selection itself
# --------------------------------------------------------------------------------------


def _positive_edge_channel() -> tuple[np.ndarray, set[frozenset]]:
    """A pair channel whose true edges are its *most positive* entries.

    This is the case the old code could not see: it is exactly the shape of the `low` band's
    channel, and it scored 0.0% for every committed run.
    """
    n = 6
    true = {frozenset((0, 1)), frozenset((2, 3)), frozenset((4, 5))}
    channel = np.full((n, n), -1.0)
    np.fill_diagonal(channel, 0.0)
    for edge in true:
        i, j = sorted(edge)
        channel[i, j] = channel[j, i] = 10.0
    return channel, true


def test_max_rule_finds_the_positive_orientation():
    channel, true = _positive_edge_channel()

    result = probe.reconstruct_from_condition(channel, true, rule="max")

    assert result.recovery_pos == pytest.approx(1.0)
    assert result.recovery_neg == pytest.approx(0.0)
    assert result.recovery_max == pytest.approx(1.0)
    assert result.orientation == probe.POSITIVE
    assert result.recovery == pytest.approx(1.0)
    assert set(map(frozenset, result.graph.edges())) == true


def test_forced_orientations_bracket_the_max_rule():
    channel, true = _positive_edge_channel()

    neg = probe.reconstruct_from_condition(channel, true, rule="neg")
    pos = probe.reconstruct_from_condition(channel, true, rule="pos")

    assert neg.recovery == pytest.approx(0.0)
    assert pos.recovery == pytest.approx(1.0)
    # Both orientations always report the same *number* of edges -- which is precisely why the
    # old `hits = graph.number_of_edges()` comparison could never discriminate between them.
    assert neg.graph.number_of_edges() == pos.graph.number_of_edges() == len(true)


def test_negative_band_still_selects_the_negative_orientation():
    """A `high`-style channel, where edges sit at the negative end, must be unaffected."""
    n = 6
    true = {frozenset((0, 1)), frozenset((2, 3)), frozenset((4, 5))}
    channel = np.full((n, n), 1.0)
    np.fill_diagonal(channel, 0.0)
    for edge in true:
        i, j = sorted(edge)
        channel[i, j] = channel[j, i] = -10.0

    result = probe.reconstruct_from_condition(channel, true, rule="max")
    assert result.orientation == probe.NEGATIVE
    assert result.recovery_neg == pytest.approx(1.0)
    assert result.recovery == pytest.approx(1.0)


def test_ties_keep_the_negative_orientation():
    """Deterministic tie-break, so the max rule degrades to the old behaviour on a tie."""
    n = 4
    true = {frozenset((0, 1))}
    channel = np.zeros((n, n))

    result = probe.reconstruct_from_condition(channel, true, rule="max")
    assert result.recovery_neg == result.recovery_pos
    assert result.orientation == probe.NEGATIVE


def test_unknown_rule_is_rejected():
    channel, true = _positive_edge_channel()
    with pytest.raises(ValueError):
        probe.reconstruct_from_condition(channel, true, rule="whichever")


def test_band_rule_is_not_resolvable_per_graph():
    """`band` is a cell-level rule, so a single graph must refuse it rather than guess."""
    channel, true = _positive_edge_channel()
    with pytest.raises(ValueError, match="band"):
        probe.reconstruct_from_condition(channel, true, rule="band")


# --------------------------------------------------------------------------------------
# The cell-level majority vote
# --------------------------------------------------------------------------------------


def _cell(pos_count: int, total: int) -> list:
    """A cell of `total` reconstructions, `pos_count` of which prefer orientation +1."""
    positive, true = _positive_edge_channel()
    negative = -positive
    built = []
    for index in range(total):
        channel = positive if index < pos_count else negative
        built.append(probe.reconstruct_from_condition(channel, true, rule="max"))
    return built


def test_band_vote_follows_the_majority():
    assert probe.band_orientation(_cell(7, 10)) == probe.POSITIVE
    assert probe.band_orientation(_cell(3, 10)) == probe.NEGATIVE
    assert probe.band_orientation(_cell(10, 10)) == probe.POSITIVE
    assert probe.band_orientation(_cell(0, 10)) == probe.NEGATIVE


def test_band_vote_breaks_ties_negative():
    """A split cell keeps -1.0, matching the per-graph tie-break and the pre-fix behaviour."""
    assert probe.band_orientation(_cell(5, 10)) == probe.NEGATIVE
    assert probe.band_orientation([]) == probe.NEGATIVE


@pytest.mark.parametrize("dataset,band,k", [("planar", "low", 8), ("planar", "high", 32)])
def test_band_and_max_agree_on_unanimous_cells(dataset, band, k):
    """low and high vote unanimously, so the cheap rule costs nothing against the bound."""
    if not _dataset_available(dataset):
        pytest.skip(f"raw {dataset} split not on disk; would require a download")

    from fald.data.spectre import load_splits

    graphs = load_splits(dataset)["val"]
    row = probe.evaluate_band(graphs, band, k, seed=0, cache_tag=f"leakage_{dataset}_val")

    assert row.orientation_pos_frac in (0.0, 1.0)
    assert row.edge_recovery_band == pytest.approx(row.edge_recovery_max)
    assert row.edge_recovery == pytest.approx(row.edge_recovery_band)


def test_band_rule_never_exceeds_the_max_rule():
    """The cell vote is a restriction of the per-graph argmax, so it cannot score higher."""
    if not _dataset_available("planar"):
        pytest.skip("raw planar split not on disk; would require a download")

    from fald.data.spectre import load_splits

    graphs = load_splits("planar")["val"]
    for band in ("gaussian", "shuffled", "random"):
        row = probe.evaluate_band(graphs, band, 8, seed=0, cache_tag="leakage_planar_val")
        assert row.edge_recovery_band <= row.edge_recovery_max + 1e-12, band


# --------------------------------------------------------------------------------------
# Censored distance to planarity
# --------------------------------------------------------------------------------------


def test_planar_distance_median_ignores_nothing_when_uncensored():
    summary = probe.summarize_planar_distance([0, 2, 4, 6, 8])
    assert summary["n_censored"] == 0
    assert summary["median"] == pytest.approx(4.0)
    assert summary["measured_values"] == [0, 2, 4, 6, 8]


def test_planar_distance_median_survives_minority_censoring():
    """Censored values sort above every measured one, so a minority cannot move the median."""
    summary = probe.summarize_planar_distance([1, 2, 3, None, None])
    assert summary["n_censored"] == 2
    assert summary["median"] == pytest.approx(3.0)


def test_planar_distance_median_is_withheld_when_censoring_dominates():
    """Half or more censored: the median is undefined and must not be quoted."""
    for values in ([1, 2, None, None], [1, None, None, None], [None] * 4):
        summary = probe.summarize_planar_distance(values)
        assert summary["median"] is None, values
        assert summary["censored_frac"] >= 0.5

    # The exact shape of the defect this replaces: 29 of 32 censored, three survivors whose
    # median (4) was being reported as the cell's median.
    summary = probe.summarize_planar_distance([0, 4, 11] + [None] * 29)
    assert summary["median"] is None
    assert summary["n_censored"] == 29
    assert summary["measured_values"] == [0, 4, 11]


def test_planar_distance_euler_shortcut_agrees_with_the_greedy_search():
    """The Euler early-out must return exactly what the greedy search would have returned."""
    rb = probe._report_breakdown()

    dense = nx.complete_graph(10)  # 45 edges against a 3n-6 = 24 bound: hopeless
    assert probe.planar_distance(dense) is None
    assert rb.distance_to_planar(dense) is None  # same answer, the slow way

    grid = nx.grid_2d_graph(4, 4)
    assert probe.planar_distance(nx.convert_node_labels_to_integers(grid)) == 0

    near = nx.complete_graph(5)  # K5: one removal away from planar, under the Euler bound
    assert probe.planar_distance(near) == rb.distance_to_planar(near) == 1


# --------------------------------------------------------------------------------------
# Scope of the fix, against the frozen pre-fix numbers
# --------------------------------------------------------------------------------------

_FIELDS = [
    "edge_recovery",
    "edge_recovery_std",
    "chance_recovery",
    "lift_over_chance",
    "connected_frac",
    "planar_frac",
    "valid_frac",
]


@pytest.mark.parametrize("dataset", ["planar", "sbm"])
def test_committed_neg_column_matches_the_frozen_single_orientation_numbers(dataset):
    """Cheap, data-free half of the proof: read the two committed files and compare."""
    corrected = _rows(RESULTS / f"condition_leakage_{dataset}.json")
    frozen = _rows(RESULTS / f"condition_leakage_{dataset}_single_orientation.json")

    assert set(corrected) == set(frozen)
    for key, old in frozen.items():
        new = corrected[key]
        # We ship the per-band rule: one orientation per (band, k) by majority vote, so the
        # true edge set is never used to pick a sign per graph and the noise arms keep a clean
        # floor at the analytic chance rate.
        assert new["orientation_rule"] == "band", key
        # The headline is whichever single orientation that vote selected...
        assert new["edge_recovery"] == pytest.approx(
            new["edge_recovery_neg"]
        ) or new["edge_recovery"] == pytest.approx(new["edge_recovery_pos"]), key
        # ...which can never beat the per-graph max, and equals it exactly when the band's sign
        # is unanimous (`low` and `high` at every k on both datasets).
        assert new["edge_recovery"] <= new["edge_recovery_max"] + 1e-12, key
        if new["orientation_pos_frac"] in (0.0, 1.0):
            assert new["edge_recovery"] == pytest.approx(new["edge_recovery_max"]), key
        # ...and the old headline must survive untouched as the neg column.
        assert new["edge_recovery_neg"] == pytest.approx(old["edge_recovery"], abs=1e-12), key


@pytest.mark.parametrize("dataset", ["planar", "sbm"])
def test_committed_results_carry_censored_planar_distances(dataset):
    """The committed sweeps were run with --distance-to-planar; keep it that way.

    The paper cites distance-to-planarity measured on these reconstructions, so the numbers
    have to ship with the file rather than being recomputed ad hoc.
    """
    payload = json.loads((RESULTS / f"condition_leakage_{dataset}.json").read_text())
    assert payload["distance_to_planar"] is True

    for row in payload["rows"]:
        key = (row["band"], row["k"])
        distance = row["planar_distance"]
        assert distance is not None, key
        assert distance["n_graphs"] == row["n_graphs"], key
        assert len(distance["per_graph"]) == row["n_graphs"], key
        assert distance["n_censored"] == sum(1 for v in distance["per_graph"] if v is None), key
        # The rule this summary exists to enforce: never quote a median that censoring owns.
        if distance["censored_frac"] >= 0.5:
            assert distance["median"] is None, key
        # A graph counted planar must be zero removals from planar, and vice versa.
        zeros = sum(1 for v in distance["per_graph"] if v == 0)
        assert zeros == pytest.approx(row["planar_frac"] * row["n_graphs"]), key


@pytest.mark.parametrize("dataset", ["planar", "sbm"])
def test_forcing_orientation_neg_reproduces_the_frozen_numbers(dataset):
    """Expensive half: actually recompute the sweep and compare every reported field."""
    if not _dataset_available(dataset):
        pytest.skip(f"raw {dataset} split not on disk; would require a download")

    from fald.data.spectre import load_splits

    graphs = load_splits(dataset)["val"]
    frozen = _rows(RESULTS / f"condition_leakage_{dataset}_single_orientation.json")
    cache_tag = f"leakage_{dataset}_val"

    for (band, k), old in frozen.items():
        row = probe.evaluate_band(graphs, band, k, seed=0, cache_tag=cache_tag, rule="neg")
        for field in _FIELDS:
            assert getattr(row, field) == pytest.approx(old[field], abs=1e-12), (band, k, field)
        assert row.n_graphs == old["n_graphs"]


def test_full_spectrum_recovers_everything():
    """The paper's sanity claim: all non-trivial eigenpairs reconstruct L_norm exactly."""
    if not _dataset_available("planar"):
        pytest.skip("raw planar split not on disk; would require a download")

    from fald.data.spectre import load_splits

    graphs = load_splits("planar")["val"]
    row = probe.evaluate_band(graphs, "low", None, seed=0, cache_tag="leakage_planar_val")

    assert row.k == -1
    assert row.edge_recovery_max == pytest.approx(1.0)
    assert row.edge_recovery == pytest.approx(1.0)
    # Recovering every edge means the reconstruction *is* the target graph, so its structure
    # must match the split's own: Planar val is 32/32 connected and 32/32 planar.
    assert row.connected_frac == pytest.approx(1.0)
    assert row.planar_frac == pytest.approx(1.0)
    assert all(nx.is_connected(g) and nx.check_planarity(g)[0] for g in graphs)
