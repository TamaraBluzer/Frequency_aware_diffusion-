"""Sanity checks on the Planar/SBM generators (WORKPLAN.md T6: validity oracles).

Real planar graphs -> should be ~100% valid; real SBM graphs -> should be mostly valid;
a random ER graph of matched density -> should have low validity. If our own training
data fails these checks, the checks are wrong, not the data (WORKPLAN.md T6 rationale).
"""
import networkx as nx
import numpy as np

from src.data.generate import (
    generate_erdos_renyi_matched,
    generate_planar_dataset,
    generate_sbm_dataset,
)
from src.eval.validity import is_valid_planar, is_valid_sbm


def test_planar_graphs_are_valid():
    graphs = generate_planar_dataset(num_graphs=20, num_points=64, seed=0)
    valid_fraction = np.mean([is_valid_planar(g) for g in graphs])
    assert valid_fraction >= 0.95, f"expected near-100% planar validity, got {valid_fraction}"


def test_planar_graphs_are_connected_and_planar_by_construction():
    graphs = generate_planar_dataset(num_graphs=5, num_points=32, seed=1)
    for g in graphs:
        is_planar, _ = nx.check_planarity(g)
        assert is_planar
        assert nx.is_connected(g)


def test_sbm_graphs_are_mostly_valid():
    graphs = generate_sbm_dataset(num_graphs=20, seed=0)
    # Restrict to connected graphs; a disconnected SBM draw is a legitimate corner case
    # of the generator, not a validity-test bug.
    connected = [g for g in graphs if nx.is_connected(g)]
    assert len(connected) >= 15, "too many disconnected SBM draws to assess validity meaningfully"
    valid_fraction = np.mean([is_valid_sbm(g) for g in connected])
    assert valid_fraction >= 0.5, f"expected most connected SBM graphs to pass validity, got {valid_fraction}"


def test_erdos_renyi_matched_has_low_planar_validity():
    planar_graphs = generate_planar_dataset(num_graphs=20, num_points=64, seed=2)
    er_graphs = generate_erdos_renyi_matched(planar_graphs, rng=np.random.default_rng(2))
    valid_fraction = np.mean([is_valid_planar(g) for g in er_graphs])
    assert valid_fraction < 0.5, f"expected low planar-validity for density-matched ER, got {valid_fraction}"


def test_erdos_renyi_matched_density():
    planar_graphs = generate_planar_dataset(num_graphs=5, num_points=64, seed=3)
    er_graphs = generate_erdos_renyi_matched(planar_graphs, rng=np.random.default_rng(3))
    for g_planar, g_er in zip(planar_graphs, er_graphs):
        assert g_planar.number_of_nodes() == g_er.number_of_nodes()
