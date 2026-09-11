"""Model-free edge recovery from the spectral condition alone.

Table 1 reports low > high > random on Planar. That ordering is only evidence about
*frequency* if the three conditions carry comparable information about the target graph.
This script measures the information directly: it reconstructs edges from the condition
tensor with no model, no training, and no learned parameters.

The scored quantity is `U_k diag(lambda_k) U_k^T`, which is exactly the pair channel
`build_condition_tensors` hands the denoiser -- not the raw eigenvectors. A probe on
anything else would measure a signal the model never receives.

Two privileged inputs are used deliberately, and both inflate recovery:

  * the true edge count `m`, used to threshold the scored pairs, and
  * the true node count, implicit in the condition's shape.

That is the point. The probe is an upper bound on what a perfect decoder could extract
from the side channel, so a *low* number is informative (the band cannot leak much) while
a high number means the band hands over the answer. Recovery is compared against a
degree-preserving random baseline, since dense graphs score well by chance alone.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

import networkx as nx
import numpy as np

from fald.data.spectral import cached_eigendecompositions, select_band
from fald.data.spectre import load_splits
from fald.eval.validity import is_planar
from fald.paths import results_dir


@dataclass
class LeakageRow:
    band: str
    k: int
    edge_recovery: float
    edge_recovery_std: float
    chance_recovery: float
    lift_over_chance: float
    connected_frac: float
    planar_frac: float
    valid_frac: float
    n_graphs: int


def _upper_triangle_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def reconstruct_from_condition(
    pair_channel: np.ndarray, n_edges: int
) -> nx.Graph:
    """Keep the `n_edges` highest-scoring off-diagonal pairs.

    The pair channel approximates `-A` up to scale on the off-diagonal for a low band (the
    normalized Laplacian's off-diagonal is `-1/sqrt(d_i d_j)` where an edge exists), so the
    score is negated before ranking. Sign is resolved empirically per graph rather than
    assumed: whichever orientation recovers more edges is used, which again biases the
    probe upward and keeps it an upper bound.
    """
    n = pair_channel.shape[0]
    rows, cols = _upper_triangle_indices(n)
    scores = pair_channel[rows, cols]

    best_graph = None
    best_hits = -1
    for orientation in (-1.0, 1.0):
        ranked = np.argsort(orientation * scores)[::-1][:n_edges]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(zip(rows[ranked], cols[ranked]))
        hits = graph.number_of_edges()
        if hits > best_hits:
            best_hits = hits
            best_graph = graph
    assert best_graph is not None
    return best_graph


def _chance_recovery(graph: nx.Graph) -> float:
    """Expected recovery when `m` pairs are drawn uniformly from the `n(n-1)/2` candidates."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    total_pairs = n * (n - 1) // 2
    return m / total_pairs if total_pairs else 0.0


def evaluate_band(
    graphs: list[nx.Graph],
    band: str,
    k: int,
    seed: int,
    cache_tag: str,
) -> LeakageRow:
    all_values, all_vectors = cached_eigendecompositions(graphs, tag=cache_tag)

    recoveries: list[float] = []
    chances: list[float] = []
    connected: list[bool] = []
    planar: list[bool] = []
    valid: list[bool] = []

    for index, (graph, values, vectors) in enumerate(
        zip(graphs, all_values, all_vectors)
    ):
        rng = np.random.default_rng(seed + index)
        condition = select_band(values, vectors, band, k, rng=rng)
        pair_channel = condition.rank_one_pair_channel()

        true_edges = set(map(frozenset, graph.edges()))
        rebuilt = reconstruct_from_condition(pair_channel, len(true_edges))
        rebuilt_edges = set(map(frozenset, rebuilt.edges()))

        recoveries.append(len(true_edges & rebuilt_edges) / max(len(true_edges), 1))
        chances.append(_chance_recovery(graph))
        connected.append(nx.is_connected(rebuilt))
        planar.append(nx.check_planarity(rebuilt)[0])
        valid.append(is_planar(rebuilt))

    mean_recovery = float(np.mean(recoveries))
    mean_chance = float(np.mean(chances))
    return LeakageRow(
        band=band,
        k=k,
        edge_recovery=mean_recovery,
        edge_recovery_std=float(np.std(recoveries)),
        chance_recovery=mean_chance,
        lift_over_chance=mean_recovery / mean_chance if mean_chance else float("nan"),
        connected_frac=float(np.mean(connected)),
        planar_frac=float(np.mean(planar)),
        valid_frac=float(np.mean(valid)),
        n_graphs=len(graphs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="planar")
    parser.add_argument("--split", default="val")
    parser.add_argument("--bands", nargs="+", default=["low", "high", "random", "gaussian"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    splits = load_splits(args.dataset)
    graphs = splits[args.split]
    cache_tag = f"leakage_{args.dataset}_{args.split}"

    rows: list[LeakageRow] = []
    for band in args.bands:
        for k in args.k_values:
            max_k = min(g.number_of_nodes() for g in graphs) - 1
            if k > max_k:
                continue
            row = evaluate_band(graphs, band, k, args.seed, cache_tag)
            rows.append(row)
            print(
                f"{band:>9} k={k:<3} recovery={row.edge_recovery:6.1%} "
                f"(chance {row.chance_recovery:5.1%}, lift {row.lift_over_chance:5.2f}x)  "
                f"connected={row.connected_frac:5.1%} planar={row.planar_frac:5.1%} "
                f"valid={row.valid_frac:5.1%}"
            )

    payload = {
        "experiment": "condition_leakage",
        "description": (
            "Model-free edge recovery from the spectral condition. Uses the true edge count "
            "to threshold, so values are an upper bound on extractable information."
        ),
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "n_graphs": len(graphs),
        "rows": [asdict(row) for row in rows],
    }
    output = args.output or results_dir() / f"condition_leakage_{args.dataset}.json"
    with open(output, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
