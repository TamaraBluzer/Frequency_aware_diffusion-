"""Planar and SBM graph generators, SPECTRE recipe (see WORKPLAN.md section 4).

These are pure NetworkX/NumPy/SciPy, no GPU needed, so the Stage 2 evaluation
harness can be developed and gated locally without depending on the Colab
DiGress training runs from Stage 1.

Note: WORKPLAN.md flags a real typo in SPECTRE's appendix, which states
inter-community probability 0.3 and intra-community probability 0.05 (inverted,
would produce anti-communities). We use the corrected convention:
p_intra=0.3, p_inter=0.05.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay


def generate_planar_graph(num_points: int = 64, rng: np.random.Generator | None = None) -> nx.Graph:
    """One planar graph: Delaunay triangulation of uniform random points in the unit square."""
    rng = rng or np.random.default_rng()
    points = rng.uniform(0.0, 1.0, size=(num_points, 2))
    tri = Delaunay(points)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_points))
    for simplex in tri.simplices:
        for i in range(3):
            u, v = int(simplex[i]), int(simplex[(i + 1) % 3])
            graph.add_edge(u, v)
    return graph


def generate_planar_dataset(
    num_graphs: int = 200, num_points: int = 64, seed: int = 0
) -> list[nx.Graph]:
    rng = np.random.default_rng(seed)
    return [generate_planar_graph(num_points, rng) for _ in range(num_graphs)]


def generate_sbm_graph(
    rng: np.random.Generator | None = None,
    min_communities: int = 2,
    max_communities: int = 5,
    min_community_size: int = 20,
    max_community_size: int = 40,
    p_intra: float = 0.3,
    p_inter: float = 0.05,
) -> nx.Graph:
    """One SBM graph: 2-5 communities (uniform), 20-40 nodes/community (uniform).

    p_intra=0.3, p_inter=0.05 (corrected orientation, see module docstring).
    """
    rng = rng or np.random.default_rng()
    num_communities = int(rng.integers(min_communities, max_communities + 1))
    sizes = [
        int(rng.integers(min_community_size, max_community_size + 1))
        for _ in range(num_communities)
    ]

    probs = np.full((num_communities, num_communities), p_inter)
    np.fill_diagonal(probs, p_intra)

    graph = nx.stochastic_block_model(sizes, probs.tolist(), seed=int(rng.integers(0, 2**31 - 1)))
    # Drop the SPECTRE-internal "block" bookkeeping attribute; keep it as a plain graph
    # but retain community labels for the downstream community-count task (Stage 9 / WORKPLAN G5).
    communities = graph.graph.get("partition")
    if communities is not None:
        for community_id, node_set in enumerate(communities):
            for node in node_set:
                graph.nodes[node]["community"] = community_id
    return graph


def generate_sbm_dataset(num_graphs: int = 200, seed: int = 0) -> list[nx.Graph]:
    rng = np.random.default_rng(seed)
    return [generate_sbm_graph(rng) for _ in range(num_graphs)]


def generate_erdos_renyi_matched(
    graphs: list[nx.Graph], rng: np.random.Generator | None = None
) -> list[nx.Graph]:
    """Density-matched Erdos-Renyi graphs, one per input graph, same n and edge count.

    Used as a negative control for MMD calibration (Stage 2 gate) and as the
    baseline Stage 5's diffusion model must beat.
    """
    rng = rng or np.random.default_rng()
    er_graphs = []
    for graph in graphs:
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        max_edges = n * (n - 1) / 2
        p = min(1.0, m / max_edges) if max_edges > 0 else 0.0
        er_graphs.append(
            nx.gnp_random_graph(n, p, seed=int(rng.integers(0, 2**31 - 1)))
        )
    return er_graphs
