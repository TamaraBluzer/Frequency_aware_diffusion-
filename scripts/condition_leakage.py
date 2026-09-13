"""Model-free edge recovery from the spectral condition alone.

Table 1 reports low > high > random on Planar. That ordering is only evidence about
*frequency* if the three conditions carry comparable information about the target graph.
This script measures the information directly: it reconstructs edges from the condition
tensor with no model, no training, and no learned parameters.

The scored quantity is `U_k diag(lambda_k) U_k^T`, which is exactly the pair channel
`build_condition_tensors` hands the denoiser -- not the raw eigenvectors. A probe on
anything else would measure a signal the model never receives.

What the probe actually does
----------------------------
For one graph it ranks the `n(n-1)/2` off-diagonal entries of the pair channel and keeps the
`m` extreme ones, where `m` is the target's true edge count. Which end is "extreme" is a
sign convention, and it is not the same for every band: a low band's pair channel tracks the
normalized Laplacian's off-diagonal (`-1/sqrt(d_i d_j)` on an edge), so edges are the *most
negative* entries, while other bands put edges at the other end. The probe therefore builds
*both* reconstructions per graph -- the `m` most negative entries (orientation `-1.0`) and
the `m` most positive (orientation `+1.0`) -- and scores each against the true edge set.

How the sign is resolved (`--orientation-rule`)
-----------------------------------------------
Empirically the sign is a property of the *band*, not of the graph: `low` prefers `+1.0` and
`high` prefers `-1.0` unanimously across all 32 graphs, at every k, on both datasets
(`orientation_pos_frac` is exactly 1.0 and 0.0 respectively). So the sign can be fixed per
cell instead of per graph:

  * ``band`` (default, the reported rule) -- each (band, k) cell votes, the majority sign is
    applied to every graph in the cell, and each graph's recovery is measured under that one
    fixed sign. Exactly **one bit** of privileged information per cell.
  * ``max`` -- each graph keeps whichever sign suits it: `1` bit per *graph*. This is a real
    upper bound but it has a selection bias, because taking a max over two noisy estimates
    lifts even a pure-noise condition above chance (the `gaussian` arm rises from ~1.0x to
    ~1.1x under it). Kept as the documented conservative bound.
  * ``neg`` / ``pos`` -- force one sign globally. `neg` reproduces the pre-fix numbers.

`band` and `max` agree to the digit on `low` and `high`, because those cells are unanimous;
they differ only on the arms where the sign really is a coin flip, which is precisely where
`max`'s bias lives. Every row reports all four means (`edge_recovery_neg`,
`edge_recovery_pos`, `edge_recovery_max`, `edge_recovery_band`) whichever rule is active, so
the choice is auditable without rerunning anything.

Privileged inputs, all of which inflate recovery:

  * the true edge count `m`, used to threshold the scored pairs,
  * the true node count, implicit in the condition's shape, and
  * the true edge set, used to resolve the sign -- one bit per cell under `band`, one bit
    per graph under `max`.

What a number does and does not mean
------------------------------------
A *high* score is conclusive: the band demonstrably hands over the answer, since a decoder
this crude already extracts it. A *low* score is much weaker evidence. It bounds only this
decoder family -- rank the pair channel, threshold at `m`, under one of two global signs --
and a band could still encode the edge set in a form this rule cannot read (a nonlinear,
per-node, or degree-normalized decoding, for instance). Earlier versions of this docstring
claimed a low score was "strong evidence a band cannot leak"; that inference was never
licensed by the measurement and is not made here. Recovery is compared against a
degree-preserving random baseline, since dense graphs score well by chance alone.

Provenance of the committed numbers
-----------------------------------
The orientation comparison above was broken in every result committed before this fix. The
selection loop compared `graph.number_of_edges()`, which equals the requested `m` for both
orientations by construction, so the strict `>` always kept the first iteration and the
probe silently ran at orientation `-1.0` only. Every previously committed
`condition_leakage_*.json` number is therefore single-orientation, and the frozen copies in
`results/condition_leakage_*_single_orientation.json` preserve them. They are reproducible
from this script with `--orientation-rule neg`, which is what
`tests/test_condition_leakage.py` asserts. The correction matters most for the `low` band,
whose edges sit at the *positive* end of its pair channel: low k=16 on Planar moves from
0.0% to 78.7%.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np

from fald.data.conditioning import _derangement
from fald.data.spectral import (
    cached_eigendecompositions,
    count_trivial_eigenpairs,
    select_band,
)
from fald.data.spectre import load_splits
from fald.eval.validity import is_planar
from fald.paths import results_dir

# How the pair channel's sign is resolved. "band" is the primary rule and the default; the
# others are diagnostics, and "neg" reproduces every number committed before the orientation
# bug was fixed. See the module docstring for why "band" is preferred over "max".
ORIENTATION_RULES = ("band", "max", "neg", "pos")

# Rules that a single graph can resolve on its own. "band" is a property of a whole
# (band, k) cell, so it is resolved in `evaluate_band`, not in `reconstruct_from_condition`.
PER_GRAPH_RULES = ("max", "neg", "pos")

NEGATIVE, POSITIVE = -1.0, 1.0

# Greedy-search cap for the distance to planarity, matching scripts/report_breakdown.py so
# the two report the same estimator. Defined once: `planar_distance` and
# `summarize_planar_distance` must agree, or the censoring count describes a different cap
# than the search used.
MAX_REMOVALS = 12


@dataclass
class LeakageRow:
    band: str
    k: int
    # `edge_recovery` is the headline number and follows `orientation_rule`: under the default
    # "band" rule it is exactly `edge_recovery_band`. The name is load-bearing:
    # scripts/make_paper_figures.py and scripts/analyze_sweep.py both read it.
    edge_recovery: float
    # All four rules' means are reported on every row regardless of which one is active, so a
    # reader can check the headline against the alternatives without rerunning anything.
    edge_recovery_neg: float
    edge_recovery_pos: float
    edge_recovery_max: float
    edge_recovery_band: float
    edge_recovery_std: float
    chance_recovery: float
    lift_over_chance: float
    connected_frac: float
    planar_frac: float
    valid_frac: float
    # Fraction of graphs where orientation +1 strictly beat orientation -1, i.e. where the
    # per-graph max is not the old single-orientation answer. 0.0 or 1.0 means the band has a
    # consistent sign; anything between means the sign is graph-dependent.
    orientation_pos_frac: float
    # The sign the majority vote picked for this cell, which the "band" rule applies to every
    # graph. +1.0 or -1.0.
    band_orientation: float
    n_graphs: int
    orientation_rule: str = "band"
    # Graded distance to planarity of the selected reconstructions, or None when the sweep was
    # run without --distance-to-planar. See `summarize_planar_distance` for the shape, and read
    # `median` together with `n_censored`: it is null whenever censoring reaches half the cell.
    planar_distance: dict | None = None


@dataclass
class Reconstruction:
    """One graph's reconstruction under *both* sign orientations.

    Both are kept because the cell-level "band" rule cannot be resolved until every graph in
    the cell has been scored, and the connectivity and planarity fractions must then be
    measured on whichever reconstruction that vote selects.
    """

    graphs: dict[float, nx.Graph]
    recoveries: dict[float, float]
    orientation: float  # the sign the per-graph rule picked

    @property
    def graph(self) -> nx.Graph:
        return self.graphs[self.orientation]

    @property
    def recovery(self) -> float:
        return self.recoveries[self.orientation]

    @property
    def recovery_neg(self) -> float:
        return self.recoveries[NEGATIVE]

    @property
    def recovery_pos(self) -> float:
        return self.recoveries[POSITIVE]

    @property
    def recovery_max(self) -> float:
        return max(self.recovery_neg, self.recovery_pos)

    def at(self, orientation: float) -> tuple[nx.Graph, float]:
        """The reconstruction and recovery at one fixed sign, ignoring the per-graph rule."""
        return self.graphs[orientation], self.recoveries[orientation]


def _upper_triangle_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def _threshold_pairs(
    rows: np.ndarray,
    cols: np.ndarray,
    scores: np.ndarray,
    orientation: float,
    n_edges: int,
    n_nodes: int,
) -> nx.Graph:
    """Keep the `n_edges` pairs most extreme in the direction `orientation`."""
    ranked = np.argsort(orientation * scores)[::-1][:n_edges]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(zip(rows[ranked], cols[ranked]))
    return graph


def _recovery(true_edges: set[frozenset], graph: nx.Graph) -> float:
    rebuilt = set(map(frozenset, graph.edges()))
    return len(true_edges & rebuilt) / max(len(true_edges), 1)


def reconstruct_from_condition(
    pair_channel: np.ndarray,
    true_edges: set[frozenset],
    rule: str = "max",
) -> Reconstruction:
    """Keep the `m` highest-scoring off-diagonal pairs, with the sign resolved per graph.

    `m` is `len(true_edges)`. Both orientations are always built and scored *against the true
    edge set*, so `recovery_neg` and `recovery_pos` are always available; `rule` decides only
    which one this graph reports as `orientation` (and therefore as `graph` and `recovery`):

      * ``"max"``  -- whichever orientation recovers more of the true edges, ties to `-1.0`.
                      The conservative upper bound: it spends one bit of privileged
                      information *per graph*.
      * ``"neg"``  -- force orientation `-1.0`, reproducing the pre-fix committed numbers.
      * ``"pos"``  -- force orientation `+1.0`.

    The primary ``"band"`` rule is deliberately absent here: it is a property of a whole
    (band, k) cell and is resolved by `evaluate_band`, which votes across graphs and then
    calls `Reconstruction.at` with the winning sign.

    Scoring against the true edge set is the whole point of the signature: the previous
    version compared `graph.number_of_edges()` between the two orientations, which is `m`
    either way, so no selection ever happened.
    """
    if rule not in PER_GRAPH_RULES:
        raise ValueError(
            f"unknown per-graph orientation rule {rule!r}; expected one of {PER_GRAPH_RULES}. "
            "The 'band' rule is resolved per (band, k) cell by evaluate_band."
        )

    n = pair_channel.shape[0]
    rows, cols = _upper_triangle_indices(n)
    scores = pair_channel[rows, cols]
    n_edges = len(true_edges)

    built = {
        orientation: _threshold_pairs(rows, cols, scores, orientation, n_edges, n)
        for orientation in (NEGATIVE, POSITIVE)
    }
    recoveries = {
        orientation: _recovery(true_edges, graph) for orientation, graph in built.items()
    }

    if rule == "neg":
        chosen = NEGATIVE
    elif rule == "pos":
        chosen = POSITIVE
    else:
        chosen = POSITIVE if recoveries[POSITIVE] > recoveries[NEGATIVE] else NEGATIVE

    return Reconstruction(graphs=built, recoveries=recoveries, orientation=chosen)


def band_orientation(reconstructions: list[Reconstruction]) -> float:
    """The sign a majority of graphs in one (band, k) cell prefer. Ties keep `-1.0`.

    This is the whole of the privileged information the "band" rule spends: one bit per cell,
    rather than one bit per graph. It is well defined because the sign turns out to be a
    property of the band and not of the graph -- `low` votes `+1.0` unanimously and `high`
    votes `-1.0` unanimously, on every k and both datasets.
    """
    if not reconstructions:
        return NEGATIVE
    votes = sum(1 for r in reconstructions if r.recovery_pos > r.recovery_neg)
    return POSITIVE if votes * 2 > len(reconstructions) else NEGATIVE


@lru_cache(maxsize=1)
def _report_breakdown():
    """Load scripts/report_breakdown.py by path -- `scripts/` is not a package.

    Imported lazily and reused rather than reimplemented: the greedy planarity distance must
    be the *same* estimator the rest of the paper reports, or the numbers are not comparable.
    """
    import importlib.util
    import sys

    path = Path(__file__).resolve().parent / "report_breakdown.py"
    spec = importlib.util.spec_from_file_location("fald_report_breakdown", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def planar_distance(graph: nx.Graph, max_removals: int = MAX_REMOVALS) -> int | None:
    """`report_breakdown.distance_to_planar` with an exact early-out for hopeless graphs.

    Euler's bound says a simple planar graph on `n >= 3` nodes has at most `3n - 6` edges. A
    graph with `m - (3n - 6) > max_removals` therefore still exceeds the bound after *any*
    `max_removals` deletions and cannot be made planar, so the greedy search would spend 12
    rounds of `check_planarity` to return None. Skipping it is a shortcut, not an
    approximation -- the answer is identical.

    This is what makes the SBM sweep tractable: every SBM reconstruction overshoots the bound
    by 26 edges or more, so the whole dataset resolves without a search. Planar sits just
    *under* the bound (m - (3n - 6) is -12 to -5), so it is searched for real.
    """
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    if n >= 3 and m - (3 * n - 6) > max_removals:
        return None
    return _report_breakdown().distance_to_planar(graph, max_removals=max_removals)


def summarize_planar_distance(values: list[int | None], max_removals: int = MAX_REMOVALS) -> dict:
    """Censoring-aware summary of one cell's distances.

    `distance_to_planar` returns None for a graph still non-planar after `max_removals`
    deletions, which is right-censoring: the true distance is known only to exceed the cap.
    A median over the *measured* subset silently drops those and reads far too low -- the
    defect in the currently reported "high k=32 median 4", which is a median of three
    survivors out of 32.

    Censored values sort above every measured one, so the median of the full cell is exact
    whenever fewer than half of it is censored, and undefined otherwise. `median` is null in
    that second case, and `n_censored` is the number to quote instead: "> max_removals in N
    of n_graphs cases".
    """
    n = len(values)
    censored = sum(1 for value in values if value is None)
    ranked = sorted(float("inf") if value is None else float(value) for value in values)
    median = float(np.median(ranked)) if n else float("nan")
    return {
        "max_removals": max_removals,
        "n_graphs": n,
        "n_censored": censored,
        "censored_frac": censored / n if n else 0.0,
        "median": median if np.isfinite(median) else None,
        "measured_values": sorted(value for value in values if value is not None),
        "per_graph": list(values),
    }


def _chance_recovery(graph: nx.Graph) -> float:
    """Expected recovery when `m` pairs are drawn uniformly from the `n(n-1)/2` candidates."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    total_pairs = n * (n - 1) // 2
    return m / total_pairs if total_pairs else 0.0


def _condition_for(
    all_values: list[np.ndarray],
    all_vectors: list[np.ndarray],
    donor: int,
    band: str,
    k: int | None,
    rng: np.random.Generator,
):
    """`k=None` means "every non-trivial eigenpair of this graph" (the full-spectrum probe)."""
    values, vectors = all_values[donor], all_vectors[donor]
    if k is None:
        n_trivial = max(count_trivial_eigenpairs(values), 1)
        k = int(vectors.shape[0]) - n_trivial
    return select_band(values, vectors, band, k, rng=rng)


def evaluate_band(
    graphs: list[nx.Graph],
    band: str,
    k: int | None,
    seed: int,
    cache_tag: str,
    rule: str = "band",
    with_distance: bool = False,
) -> LeakageRow:
    """Score one (band, k) cell.

    Two passes, because the primary "band" rule is not resolvable graph by graph: every graph
    is reconstructed under both signs first, the cell then votes on a single sign, and the
    reported recovery, connectivity and planarity all come from that one orientation.

    `with_distance` adds the graded distance to planarity of the selected reconstructions. It
    is off by default because it costs ~50s per Planar cell (SBM resolves instantly via the
    Euler bound in `planar_distance`).
    """
    if rule not in ORIENTATION_RULES:
        raise ValueError(f"unknown orientation rule {rule!r}; expected one of {ORIENTATION_RULES}")

    all_values, all_vectors = cached_eigendecompositions(graphs, tag=cache_tag)

    # 'shuffled' is a donor assignment rather than an eigenpair rule: each graph gets another
    # graph's real low band. Mirrors build_condition_tensors so the probe scores what the model
    # would actually receive.
    reported_band = band
    if band == "shuffled":
        donors = _derangement(len(graphs), seed)
        band = "low"
    else:
        donors = np.arange(len(graphs))

    # --- pass 1: reconstruct every graph under both orientations ------------------------
    rebuilt: list[Reconstruction] = []
    chances: list[float] = []
    for index, graph in enumerate(graphs):
        donor = int(donors[index])
        rng = np.random.default_rng(seed + index)
        condition = _condition_for(all_values, all_vectors, donor, band, k, rng)
        pair_channel = condition.rank_one_pair_channel()

        true_edges = set(map(frozenset, graph.edges()))
        # "max" here only fills in the per-graph `orientation`; the active rule is applied below.
        rebuilt.append(reconstruct_from_condition(pair_channel, true_edges, rule="max"))
        chances.append(_chance_recovery(graph))

    neg = [r.recovery_neg for r in rebuilt]
    pos = [r.recovery_pos for r in rebuilt]
    voted = band_orientation(rebuilt)

    # --- pass 2: apply the active rule --------------------------------------------------
    if rule == "band":
        orientations = [voted] * len(rebuilt)
    elif rule == "neg":
        orientations = [NEGATIVE] * len(rebuilt)
    elif rule == "pos":
        orientations = [POSITIVE] * len(rebuilt)
    else:  # "max": each graph keeps the sign that suits it
        orientations = [r.orientation for r in rebuilt]

    selected: list[float] = []
    connected: list[bool] = []
    planar: list[bool] = []
    valid: list[bool] = []
    distances: list[int | None] = []
    for reconstruction, orientation in zip(rebuilt, orientations):
        graph, recovery = reconstruction.at(orientation)
        selected.append(recovery)
        # Structure is measured on the reconstruction the rule actually selected. At the two
        # orientations the bands do not merely flip a sign -- they produce different graphs --
        # so these fractions are not transferable between rules.
        connected.append(nx.is_connected(graph))
        planar.append(nx.check_planarity(graph)[0])
        valid.append(is_planar(graph))
        if with_distance:
            distances.append(planar_distance(graph, max_removals=MAX_REMOVALS))

    mean_selected = float(np.mean(selected))
    mean_chance = float(np.mean(chances))
    return LeakageRow(
        band=reported_band,
        k=-1 if k is None else k,
        edge_recovery=mean_selected,
        edge_recovery_neg=float(np.mean(neg)),
        edge_recovery_pos=float(np.mean(pos)),
        edge_recovery_max=float(np.mean(np.maximum(neg, pos))),
        edge_recovery_band=float(np.mean(pos if voted == POSITIVE else neg)),
        edge_recovery_std=float(np.std(selected)),
        chance_recovery=mean_chance,
        lift_over_chance=mean_selected / mean_chance if mean_chance else float("nan"),
        connected_frac=float(np.mean(connected)),
        planar_frac=float(np.mean(planar)),
        valid_frac=float(np.mean(valid)),
        orientation_pos_frac=float(np.mean([r.recovery_pos > r.recovery_neg for r in rebuilt])),
        band_orientation=voted,
        n_graphs=len(graphs),
        orientation_rule=rule,
        planar_distance=(
            summarize_planar_distance(distances, max_removals=MAX_REMOVALS)
            if with_distance else None
        ),
    )


def _print_row(row: LeakageRow) -> None:
    label = "full" if row.k < 0 else f"k={row.k}"
    sign = "+" if row.band_orientation > 0 else "-"
    print(
        f"{row.band:>9} {label:<7} recovery={row.edge_recovery:6.1%} "
        f"(neg {row.edge_recovery_neg:6.1%} / pos {row.edge_recovery_pos:6.1%} / "
        f"max {row.edge_recovery_max:6.1%}, vote {sign}1 at {row.orientation_pos_frac:4.0%} pos, "
        f"chance {row.chance_recovery:5.1%}, lift {row.lift_over_chance:5.2f}x)  "
        f"connected={row.connected_frac:5.1%} planar={row.planar_frac:5.1%} "
        f"valid={row.valid_frac:5.1%}"
    )
    distance = row.planar_distance
    if distance is not None:
        cap, censored, total = distance["max_removals"], distance["n_censored"], distance["n_graphs"]
        median = distance["median"]
        verdict = (
            f"median {median:g}" if median is not None
            else f"median n/a (censoring dominates)"
        )
        measured = ", ".join(str(v) for v in distance["measured_values"]) or "none"
        print(
            f"{'':>9} {'':<7} distance-to-planar: >{cap} in {censored}/{total}, "
            f"{verdict}; measured: [{measured}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="planar")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--bands", nargs="+", default=["low", "high", "random", "gaussian", "shuffled"]
    )
    # nargs="*" so `--k-values` with no values is legal: that is how the full-spectrum
    # sanity check is run on its own, without any fixed-k rows.
    parser.add_argument("--k-values", nargs="*", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument(
        "--orientation-rule",
        default="band",
        choices=ORIENTATION_RULES,
        help=(
            "how the pair channel's sign is resolved. 'band' (default) votes once per "
            "(band, k) cell and applies that sign to every graph; 'max' picks per graph and "
            "is the conservative upper bound; 'neg' forces the single orientation every "
            "pre-fix result was silently computed at."
        ),
    )
    parser.add_argument(
        "--distance-to-planar",
        action="store_true",
        help=(
            "also measure the greedy distance to planarity of each selected reconstruction, "
            "reusing scripts/report_breakdown.py. Costs ~50s per Planar cell; SBM is free "
            "because every reconstruction fails Euler's bound outright. Off by default; the "
            "committed results files were produced with it on."
        ),
    )
    parser.add_argument(
        "--full-spectrum",
        action="store_true",
        help=(
            "also probe every non-trivial eigenpair of each graph (reported as k=-1). "
            "U diag(lambda) U^T is then exactly L_norm, so recovery must be 100%%."
        ),
    )
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
            row = evaluate_band(
                graphs, band, k, args.seed, cache_tag,
                rule=args.orientation_rule, with_distance=args.distance_to_planar,
            )
            rows.append(row)
            _print_row(row)
        if args.full_spectrum and band in ("low", "high", "random"):
            # low/high/random all collapse to the same selection when k covers the whole
            # non-trivial spectrum, so this is one sanity check, not three.
            row = evaluate_band(
                graphs, band, None, args.seed, cache_tag,
                rule=args.orientation_rule, with_distance=args.distance_to_planar,
            )
            rows.append(row)
            _print_row(row)

    payload = {
        "experiment": "condition_leakage",
        "description": (
            "Model-free edge recovery from the spectral condition. Uses the true edge count to "
            "threshold the scored pairs and the true edge set to resolve the pair channel's "
            "sign, so values upper-bound what this rank-and-threshold decoder can extract. "
            "'edge_recovery_neg'/'edge_recovery_pos' are the two fixed-sign scores, "
            "'edge_recovery_max' picks the better sign per graph (an upper bound, but biased "
            "upward by the selection itself), and 'edge_recovery_band' fixes one sign per "
            "(band, k) cell by majority vote -- one bit of privilege per cell instead of per "
            "graph, which is why it is the default and the reported rule. 'edge_recovery' "
            "follows 'orientation_rule' and equals 'edge_recovery_band' under the default. "
            "'band_orientation' is the sign the cell voted for and 'orientation_pos_frac' the "
            "share of graphs preferring +1. connected/planar/valid fractions are measured on "
            "the reconstruction the active rule selected. Rows with k=-1 use every non-trivial "
            "eigenpair. Results committed before 2026-09 were single-orientation ('neg') "
            "because of a bug in the orientation selection; see "
            "results/condition_leakage_*_single_orientation.json."
        ),
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "orientation_rule": args.orientation_rule,
        "distance_to_planar": args.distance_to_planar,
        "n_graphs": len(graphs),
        "rows": [asdict(row) for row in rows],
    }
    output = args.output or results_dir() / f"condition_leakage_{args.dataset}.json"
    with open(output, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
