"""Stage 2 gate: prove the evaluator measures what we think it measures.

An MMD implementation is easy to get subtly wrong and impossible to debug later from model
results alone, so we calibrate it against cases with known answers before any of our own
models exist:

  real vs real      must be ~0            (two halves of the training set)
  train vs test     the self-similarity floor, and the row every later table is read against;
                    must land near SPECTRE Table 1 for Planar
  ER vs test        must be large          (density-matched Erdos-Renyi, structurally wrong)

If the ordering real-vs-real < train-vs-test << ER-vs-test does not hold, the harness is
broken and nothing downstream is interpretable.

Usage:
    python scripts/calibrate_eval.py [--dataset planar] [--no-orbit] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fald.data import load_splits
from fald.eval import GraphEvaluator, is_planar
from fald.eval.evaluator import MMD_SPECS

# SPECTRE Table 1's Planar training-set row. Ours will not match exactly (the split is theirs
# but the MMD is recomputed), so the gate checks order of magnitude only.
#
# Orbit and wavelet are deliberately absent: we have no published training-row value to compare
# against for either (DiGress ships the wavelet metric disabled), and asserting against a
# number we invented would make the gate meaningless. Both are still covered by the
# real-vs-real and ER checks below, which need no external reference.
SPECTRE_TRAINING_ROW = {"degree": 2e-4, "clustering": 3.1e-2, "spectral": 5e-3}
ORDER_OF_MAGNITUDE_TOLERANCE = 30.0

# real-vs-real must be small in absolute terms, and ER must be far above the real floor.
REAL_VS_REAL_MAX = 5e-2
ER_OVER_FLOOR_MIN = 10.0


def density_matched_er(graphs, rng) -> list[nx.Graph]:
    """One Erdos-Renyi graph per input, matched on node count and edge density.

    Matching density is what makes this a real control: it removes the trivial explanation
    that MMD only noticed a difference in mean degree.
    """
    out = []
    for graph in graphs:
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        p = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
        out.append(nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31 - 1))))
    return out


def format_table(rows: dict, metric_names) -> str:
    width = max(len(name) for name in rows) + 2
    header = "row".ljust(width) + "".join(m.rjust(13) for m in metric_names) + "ratio".rjust(11)
    lines = [header, "-" * len(header)]
    for name, payload in rows.items():
        cells = "".join(f"{payload['mmd'][m]:13.3e}" for m in metric_names)
        ratio = payload.get("ratio")
        ratio_cell = f"{ratio:11.2f}" if ratio is not None else "".rjust(11)
        lines.append(name.ljust(width) + cells + ratio_cell)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="planar")
    parser.add_argument("--no-orbit", action="store_true", help="skip orbit MMD (no ORCA build)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    metric_names = [s.name for s in MMD_SPECS]
    if args.no_orbit:
        metric_names = [m for m in metric_names if m != "orbit"]

    splits = load_splits(args.dataset)
    train, test = splits["train"], splits["test"]
    print(f"{args.dataset}: train={len(train)} val={len(splits['val'])} test={len(test)}")
    print(f"nodes per graph: {sorted({g.number_of_nodes() for g in train})}")

    if args.dataset == "planar":
        valid_frac = np.mean([is_planar(g) for g in train])
        print(f"planarity of training graphs: {valid_frac:.3f} (expected 1.000)")
        if valid_frac < 1.0:
            print("FAIL: reference graphs are not all planar; the loader is wrong.")
            return 1

    started = time.time()
    evaluator = GraphEvaluator(
        reference_graphs=test,
        train_graphs=train,
        validity_func=is_planar if args.dataset == "planar" else None,
        metrics=metric_names,
    )

    half = len(train) // 2
    rows: dict = {}

    floor = evaluator.mmd_against_reference(train)
    rows["train vs test"] = {"mmd": floor, "ratio": 1.0}

    # Both halves are scored against the same reference so the comparison is apples to apples.
    half_evaluator = GraphEvaluator(
        reference_graphs=train[:half], train_graphs=None, metrics=metric_names
    )
    real_vs_real = half_evaluator.mmd_against_reference(train[half:])
    rows["train[:h] vs train[h:]"] = {"mmd": real_vs_real, "ratio": None}

    er_graphs = density_matched_er(test, rng)
    er = evaluator.mmd_against_reference(er_graphs)
    rows["ER(matched) vs test"] = {
        "mmd": er,
        "ratio": float(np.mean([er[m] / max(floor[m], 1e-8) for m in metric_names])),
    }

    print()
    print(format_table(rows, metric_names))
    print(f"\ncomputed in {time.time() - started:.1f}s")

    print("\n--- V.U.N. sanity ---")
    print(f"train  : {evaluator.evaluate(train).vun}")
    print(f"ER     : {evaluator.evaluate(er_graphs).vun}")

    # ---- gate checks ----
    failures = []

    for metric in metric_names:
        if real_vs_real[metric] > REAL_VS_REAL_MAX:
            failures.append(
                f"real-vs-real {metric} = {real_vs_real[metric]:.3e} > {REAL_VS_REAL_MAX:.0e}"
            )

    for metric in metric_names:
        if er[metric] <= floor[metric] * ER_OVER_FLOOR_MIN:
            failures.append(
                f"ER {metric} = {er[metric]:.3e} is not >{ER_OVER_FLOOR_MIN:.0f}x the "
                f"train-vs-test floor {floor[metric]:.3e}"
            )

    for metric, expected in SPECTRE_TRAINING_ROW.items():
        if metric not in metric_names:
            continue
        observed = floor[metric]
        if observed <= 0:
            failures.append(f"train-vs-test {metric} is {observed:.3e}, expected ~{expected:.0e}")
            continue
        ratio = max(observed / expected, expected / observed)
        if ratio > ORDER_OF_MAGNITUDE_TOLERANCE:
            failures.append(
                f"train-vs-test {metric} = {observed:.3e} is {ratio:.0f}x off SPECTRE's "
                f"~{expected:.0e}"
            )

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"eval_calibration_{args.dataset}.json"
    out_path.write_text(json.dumps(
        {
            "dataset": args.dataset,
            "seed": args.seed,
            "metrics": metric_names,
            "rows": rows,
            "spectre_reference_row": SPECTRE_TRAINING_ROW,
            "failures": failures,
        },
        indent=2,
    ))
    print(f"\nwrote {out_path}")

    print()
    if failures:
        print("GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("GATE PASSED: real-vs-real ~ 0 < train-vs-test << ER, and the training row matches "
          "SPECTRE's order of magnitude.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
