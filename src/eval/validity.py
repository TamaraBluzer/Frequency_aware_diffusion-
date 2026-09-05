"""Validity, uniqueness, novelty, and V.U.N. (WORKPLAN.md section 5.2).

SBM validity uses the pure-NumPy DiGress-style test rather than graph-tool, which is
painful to install (WORKPLAN.md risk R4): we approximate community recovery by
thresholding intra/inter block edge density after a greedy modularity partition, and
check it is consistent with the generating regime (p_intra > p_inter, each community
reasonably sized). This is *not* numerically comparable to SPECTRE's graph-tool +
Wald-test number; state that explicitly wherever both are reported.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities


def is_valid_planar(graph: nx.Graph) -> bool:
    """Planar valid = networkx.check_planarity AND connected (WORKPLAN.md 5.2)."""
    if graph.number_of_nodes() == 0:
        return False
    is_planar, _ = nx.check_planarity(graph)
    return bool(is_planar and nx.is_connected(graph))


def is_valid_sbm(
    graph: nx.Graph,
    min_communities: int = 2,
    max_communities: int = 5,
    min_community_size: int = 10,
    density_margin: float = 1.5,
) -> bool:
    """Pure-NumPy approximate SBM validity test (DiGress-style, see module docstring).

    Recovers communities via greedy modularity, then checks:
      - community count within the generating range,
      - each community at least `min_community_size` nodes,
      - intra-community density exceeds inter-community density by `density_margin`x,
        which is the qualitative signature of an SBM (as opposed to e.g. an ER graph).
    """
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        return False

    communities = list(greedy_modularity_communities(graph))
    if not (min_communities <= len(communities) <= max_communities):
        return False
    if any(len(c) < min_community_size for c in communities):
        return False

    node_to_comm = {}
    for comm_id, nodes in enumerate(communities):
        for node in nodes:
            node_to_comm[node] = comm_id

    intra_actual = inter_actual = 0
    comm_sizes = [len(c) for c in communities]
    intra_possible = sum(size * (size - 1) / 2 for size in comm_sizes)
    total_possible = graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2
    inter_possible = total_possible - intra_possible

    for u, v in graph.edges():
        if node_to_comm[u] == node_to_comm[v]:
            intra_actual += 1
        else:
            inter_actual += 1

    intra_density = intra_actual / intra_possible if intra_possible > 0 else 0.0
    inter_density = inter_actual / inter_possible if inter_possible > 0 else 0.0

    if inter_density <= 0:
        return intra_density > 0
    return intra_density >= density_margin * inter_density


def graph_hash(graph: nx.Graph) -> str:
    """Weisfeiler-Lehman hash, a cheap isomorphism-class proxy for dedup/novelty checks."""
    return nx.weisfeiler_lehman_graph_hash(graph)


def compute_uniqueness(graphs: list[nx.Graph]) -> float:
    """Fraction of graphs in distinct isomorphism classes (via WL hash + VF2 tie-break)."""
    if not graphs:
        return 0.0
    hashes = [graph_hash(g) for g in graphs]
    buckets: dict[str, list[int]] = {}
    for i, h in enumerate(hashes):
        buckets.setdefault(h, []).append(i)

    distinct_classes = 0
    for indices in buckets.values():
        # Within a WL-hash bucket, confirm true isomorphism classes via VF2.
        representatives: list[int] = []
        for idx in indices:
            if all(not nx.is_isomorphic(graphs[idx], graphs[r]) for r in representatives):
                representatives.append(idx)
        distinct_classes += len(representatives)

    return distinct_classes / len(graphs)


def compute_novelty(generated: list[nx.Graph], training: list[nx.Graph]) -> float:
    """Fraction of generated graphs whose isomorphism class is absent from training."""
    if not generated:
        return 0.0
    training_hashes = {graph_hash(g) for g in training}
    train_by_hash: dict[str, list[nx.Graph]] = {}
    for g in training:
        train_by_hash.setdefault(graph_hash(g), []).append(g)

    novel_count = 0
    for g in generated:
        h = graph_hash(g)
        if h not in training_hashes:
            novel_count += 1
            continue
        if not any(nx.is_isomorphic(g, t) for t in train_by_hash[h]):
            novel_count += 1
    return novel_count / len(generated)


def compute_vun(
    generated: list[nx.Graph],
    training: list[nx.Graph],
    validity_fn,
) -> dict[str, float]:
    """Valid, Unique, Novel, and combined V.U.N. fraction (WORKPLAN.md section 5.2).

    `validity_fn` is `is_valid_planar` or `is_valid_sbm`, injected so this function
    stays dataset-agnostic.
    """
    if not generated:
        return {"valid": 0.0, "unique": 0.0, "novel": 0.0, "vun": 0.0}

    valid_flags = [validity_fn(g) for g in generated]
    valid_fraction = float(np.mean(valid_flags))

    valid_graphs = [g for g, v in zip(generated, valid_flags) if v]
    unique_fraction = compute_uniqueness(valid_graphs) if valid_graphs else 0.0
    novelty_fraction = compute_novelty(valid_graphs, training) if valid_graphs else 0.0

    # V.U.N.: fraction simultaneously valid, unique, and novel, out of all generated graphs.
    if valid_graphs:
        train_by_hash: dict[str, list[nx.Graph]] = {}
        for g in training:
            train_by_hash.setdefault(graph_hash(g), []).append(g)

        seen_representatives: list[nx.Graph] = []
        vun_count = 0
        for g in valid_graphs:
            h = graph_hash(g)
            is_novel = h not in train_by_hash or not any(
                nx.is_isomorphic(g, t) for t in train_by_hash[h]
            )
            is_unique_so_far = not any(
                nx.is_isomorphic(g, rep) for rep in seen_representatives
            )
            if is_novel and is_unique_so_far:
                vun_count += 1
            if is_unique_so_far:
                seen_representatives.append(g)
        vun_fraction = vun_count / len(generated)
    else:
        vun_fraction = 0.0

    return {
        "valid": valid_fraction,
        "unique": unique_fraction,
        "novel": novelty_fraction,
        "vun": vun_fraction,
    }
