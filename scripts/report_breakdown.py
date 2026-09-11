"""Per-metric decomposition, validity diagnosis, and cross-stage comparability audit.

Three reporting gaps this closes, all answerable from saved reports without retraining:

  1. The headline Ratio is an unweighted mean over metrics whose reference floors differ by
     four orders of magnitude, so one metric can carry the result. This prints the per-metric
     contribution and a geometric mean alongside it.

  2. `is_planar` is `connected AND planar`, so a 0% validity row cannot say which half failed.
     This separates them and adds the 3n-6 planar edge bound.

  3. Reports average whichever metrics were available at run time -- ORCA is appended only
     `if orca.is_available()`. A run without ORCA averages four metrics instead of five and is
     silently on a different scale. This flags any comparison that crosses that boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

from fald.paths import results_dir

FULL_METRICS = ("degree", "clustering", "spectral", "wavelet", "orbit")


def _per_metric_ratios(evaluation: dict) -> dict[str, float]:
    return {
        key.split("/", 1)[1]: value
        for key, value in evaluation.items()
        if key.startswith("ratio/")
    }


def decompose(report: dict) -> dict | None:
    """Split one report's Ratio into per-metric contributions."""
    evaluation = report.get("evaluation")
    if not evaluation:
        return None
    ratios = _per_metric_ratios(evaluation)
    if not ratios:
        return None

    arithmetic = float(np.mean(list(ratios.values())))
    geometric = float(np.exp(np.mean(np.log(np.maximum(list(ratios.values()), 1e-12)))))
    total = sum(ratios.values())
    return {
        "metrics_used": sorted(ratios),
        "n_metrics": len(ratios),
        "has_orbit": "orbit" in ratios,
        "per_metric": ratios,
        "share_of_total": {k: v / total for k, v in ratios.items()},
        "arithmetic_mean": arithmetic,
        "geometric_mean": geometric,
        "reported_ratio": evaluation.get("ratio"),
    }


def improvement_decomposition(
    baseline: dict, treated: dict
) -> dict | None:
    """Attribute a Ratio improvement to individual metrics.

    The headline "-51.7%" is a change in a mean, so each metric's contribution is its own
    change divided by the total change. A metric that moves the wrong way gets a negative
    share, which is why the shares can exceed 100%.
    """
    base = _per_metric_ratios(baseline.get("evaluation", {}))
    treat = _per_metric_ratios(treated.get("evaluation", {}))
    shared = sorted(set(base) & set(treat))
    if not shared:
        return None

    deltas = {metric: treat[metric] - base[metric] for metric in shared}
    total_delta = sum(deltas.values())
    return {
        "metrics": shared,
        "delta_per_metric": deltas,
        "contribution_share": {
            metric: (delta / total_delta if total_delta else float("nan"))
            for metric, delta in deltas.items()
        },
        "baseline_mean": float(np.mean([base[m] for m in shared])),
        "treated_mean": float(np.mean([treat[m] for m in shared])),
        "relative_change": (
            float(np.mean([treat[m] for m in shared]) / np.mean([base[m] for m in shared]) - 1)
        ),
    }


def distance_to_planar(graph: nx.Graph, max_removals: int = 12) -> int | None:
    """Greedy upper bound on the edges that must be removed to make `graph` planar.

    Planarity is 0/1, which cannot distinguish a sample two edges away from one that is
    hopeless -- and on this dataset almost every sample fails, so the boolean carries little
    signal. Each round removes the edge whose deletion most reduces the Kuratowski subgraph,
    approximated by removing an edge of the reported violating subgraph.

    Returns None when the graph is still non-planar after `max_removals`, so a caller can
    treat "far" separately from a number.
    """
    if nx.check_planarity(graph)[0]:
        return 0

    working = graph.copy()
    for removed in range(1, max_removals + 1):
        is_planar, counterexample = nx.check_planarity(working, counterexample=True)
        if is_planar:
            return removed - 1
        edges = list(counterexample.edges())
        if not edges:
            return None
        # Remove the highest-degree edge of the violating subgraph: it participates in the
        # most alternative Kuratowski subgraphs, so it is the greedy best single deletion.
        degrees = dict(counterexample.degree())
        working.remove_edge(*max(edges, key=lambda e: degrees[e[0]] + degrees[e[1]]))
        if nx.check_planarity(working)[0]:
            return removed
    return None


def diagnose_validity(graphs: list[nx.Graph], *, with_distance: bool = False) -> dict:
    """Separate the two failure modes `is_planar` collapses into one boolean.

    `with_distance` adds the graded distance-to-planarity, which costs ~40s per 32 samples on
    graphs far from planar -- the usual case for generated output. Off by default so it does
    not tax every sweep run; turn it on when comparing how close arms get.
    """
    connected, planar, both, over_bound = [], [], [], []
    components, edge_ratios = [], []

    for graph in graphs:
        n, m = graph.number_of_nodes(), graph.number_of_edges()
        is_conn = nx.is_connected(graph) if n else False
        is_plan = nx.check_planarity(graph)[0]
        bound = max(3 * n - 6, 1)
        connected.append(is_conn)
        planar.append(is_plan)
        both.append(is_conn and is_plan)
        over_bound.append(m > bound)
        components.append(nx.number_connected_components(graph))
        edge_ratios.append(m / bound)

    # Graded distance, because the boolean saturates at 0% and hides real progress.
    distances = [distance_to_planar(graph) for graph in graphs] if with_distance else []
    measured = [d for d in distances if d is not None]

    return {
        "n_graphs": len(graphs),
        "connected_frac": float(np.mean(connected)),
        "planar_frac": float(np.mean(planar)),
        "valid_frac": float(np.mean(both)),
        "exceeds_3n_minus_6_frac": float(np.mean(over_bound)),
        "mean_components": float(np.mean(components)),
        "mean_edges_over_bound": float(np.mean(edge_ratios)),
        "mean_distance_to_planar": float(np.mean(measured)) if measured else None,
        "median_distance_to_planar": float(np.median(measured)) if measured else None,
        "beyond_search_frac": (
            float(np.mean([d is None for d in distances])) if with_distance else None
        ),
    }


def audit_comparability(reports: dict[str, dict]) -> dict:
    """Flag reports whose Ratio is not on a common scale."""
    scales: dict[str, list[str]] = {}
    for name, report in reports.items():
        decomposed = decompose(report)
        if decomposed is None:
            continue
        key = "5-metric (with orbit)" if decomposed["has_orbit"] else "4-metric (no orbit)"
        scales.setdefault(key, []).append(name)

    return {
        "scales": scales,
        "comparable": len(scales) <= 1,
        "note": (
            "Ratios from different metric sets are not comparable: orbit's reference floor is "
            "~1e-4, so including it shifts the mean by roughly 2-3x."
        ),
    }


def _load(path: Path) -> dict | None:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.results_dir) if args.results_dir else results_dir()
    reports = {}
    for path in sorted(root.glob("*.json")):
        report = _load(path)
        if report and "evaluation" in report:
            reports[path.stem] = report

    payload: dict = {
        "experiment": "report_breakdown",
        "n_reports": len(reports),
        "comparability_audit": audit_comparability(reports),
        "decompositions": {},
    }
    for name, report in reports.items():
        decomposed = decompose(report)
        if decomposed:
            payload["decompositions"][name] = decomposed

    baseline_key = "adjacency_diffusion_planar_none_k8"
    treated_key = "adjacency_diffusion_planar_low_k8"
    if baseline_key in reports and treated_key in reports:
        payload["headline_improvement"] = improvement_decomposition(
            reports[baseline_key], reports[treated_key]
        )

    output = Path(args.output) if args.output else root / "report_breakdown.json"
    with open(output, "w") as handle:
        json.dump(payload, handle, indent=2)

    audit = payload["comparability_audit"]
    print("Metric-set scales present in results/:")
    for scale, names in audit["scales"].items():
        print(f"  {scale}: {len(names)} report(s)")
    if not audit["comparable"]:
        print("  !! Ratios span different metric sets and are NOT directly comparable.")

    if "headline_improvement" in payload:
        improvement = payload["headline_improvement"]
        print(f"\nHeadline improvement ({baseline_key} -> {treated_key}):")
        print(f"  relative change: {improvement['relative_change']:+.1%}")
        for metric, share in sorted(
            improvement["contribution_share"].items(), key=lambda kv: -abs(kv[1])
        ):
            delta = improvement["delta_per_metric"][metric]
            print(f"    {metric:<11} delta={delta:+10.2f}  share={share:+7.1%}")

    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
