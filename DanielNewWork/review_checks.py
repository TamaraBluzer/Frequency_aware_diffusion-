"""Three CPU-only checks that settle points raised in review of paper/main.tex.

    python DanielNewWork/review_checks.py

Writes DanielNewWork/review_checks.json. Reads committed results/*.json only --
no checkpoints, no samples, no model. Each check exists because the paper makes
a claim its own artifacts contradict.

1. monotonicity  -- the abstract and 4.3 say recovery "rises smoothly with k in
   both bands". The Planar low band falls at k=32.
2. sweep_effects -- the abstract, 4.2 and 5 say 23 of 25 cells are baseline. The
   only seed-repeated arm gives a seed-level sd; measured against it, four cells
   separate from baseline and every control cell does not.
3. label_leakage -- 4.6 says an unconditioned sample is "statistically
   independent of the label it is handed". Its node count is inherited from the
   template, and node count alone carries most of the label.

Check 2 compares single-seed sweep cells against a sd borrowed from the `none`
arm, so it ranks cells by plausibility rather than testing them. It is reported
against both that sd and the wider `low` k=8 sd for exactly that reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = Path(__file__).resolve().parent / "review_checks.json"

BANDS = ["low", "high", "random", "gaussian", "shuffled"]
CUTOFFS = [2, 4, 8, 16, 32]
CONTROLS = ["random", "gaussian", "shuffled"]

# generate_sbm_with_label draws one size per community from this inclusive range.
COMMUNITY_SIZE_RANGE = (20, 40)
COMMUNITY_COUNTS = [2, 3, 4, 5]


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def check_monotonicity() -> dict:
    out = {}
    for dataset in ["planar", "sbm"]:
        rows = _load(f"condition_leakage_{dataset}")["rows"]
        for band in ["low", "high"]:
            series = sorted(
                (r["k"], r["edge_recovery"] * 100.0)
                for r in rows
                if r["band"] == band and r["k"] > 0
            )
            values = [v for _, v in series]
            drops = [
                {"from_k": series[i][0], "to_k": series[i + 1][0], "delta": round(values[i + 1] - values[i], 2)}
                for i in range(len(values) - 1)
                if values[i + 1] < values[i]
            ]
            out[f"{dataset}_{band}"] = {
                "k": [k for k, _ in series],
                "recovery_pct": [round(v, 2) for v in values],
                "monotonic_nondecreasing": not drops,
                "drops": drops,
                "peak_at_k": series[int(np.argmax(values))][0],
            }
    return out


def check_sweep_effects() -> dict:
    seedcheck = _load("k_sweep_planar_seedcheck")["rows"]
    none = [r["ratio"] for r in seedcheck if r["band"] == "none"]
    low8 = [r["ratio"] for r in seedcheck if r["band"] == "low" and r["k"] == 8]

    baseline = float(np.mean(none))
    sd_none = float(np.std(none, ddof=1))
    sd_low = float(np.std(low8, ddof=1))

    sweep = {(r["band"], r["k"]): r["ratio"] for r in _load("k_sweep_planar")["rows"]}

    cells = []
    for band in BANDS:
        for k in CUTOFFS:
            ratio = sweep.get((band, k))
            if ratio is None:
                continue
            gap = baseline - ratio
            cells.append(
                {
                    "band": band,
                    "k": k,
                    "ratio": ratio,
                    "gap_from_baseline": round(gap, 2),
                    "gap_in_none_sd": round(gap / sd_none, 2),
                    "gap_in_low_sd": round(gap / sd_low, 2),
                    "is_control": band in CONTROLS,
                }
            )

    separated = [c for c in cells if abs(c["gap_in_none_sd"]) > 3.0]
    controls = [c for c in cells if c["is_control"]]
    return {
        "baseline_mean_ratio": round(baseline, 2),
        "baseline_seeds": none,
        "sd_none_seeds": round(sd_none, 2),
        "sd_low_k8_seeds": round(sd_low, 2),
        "n_cells": len(cells),
        "n_separated_beyond_3_none_sd": len(separated),
        "separated_cells": sorted(separated, key=lambda c: -abs(c["gap_in_none_sd"])),
        "max_abs_control_gap_in_none_sd": round(
            max(abs(c["gap_in_none_sd"]) for c in controls), 2
        ),
        "n_controls": len(controls),
        "cells": cells,
    }


def check_label_leakage(n_draws: int = 400_000, seed: int = 0) -> dict:
    """Bayes-optimal accuracy for community count given node count alone.

    Node count is the one label-relevant quantity an unconditioned sample keeps,
    since it is copied from the template it inherits its label from.
    """
    rng = np.random.default_rng(seed)
    lo, hi = COMMUNITY_SIZE_RANGE
    max_n = hi * max(COMMUNITY_COUNTS) + 1

    likelihood = {}
    stats = {}
    for count in COMMUNITY_COUNTS:
        totals = rng.integers(lo, hi + 1, size=(n_draws, count)).sum(axis=1)
        hist = np.bincount(totals, minlength=max_n).astype(float) / n_draws
        likelihood[count] = hist
        stats[count] = {
            "n_mean": round(float(totals.mean()), 1),
            "n_min": int(totals.min()),
            "n_max": int(totals.max()),
        }

    # Uniform prior over counts, so the MAP rule is argmax of the likelihood.
    correct = total = 0.0
    for count in COMMUNITY_COUNTS:
        for n_val in range(max_n):
            p = likelihood[count][n_val]
            if p == 0.0:
                continue
            total += p
            if max(COMMUNITY_COUNTS, key=lambda c: likelihood[c][n_val]) == count:
                correct += p

    return {
        "node_count_only_accuracy": round(correct / total, 4),
        "chance_accuracy": round(1.0 / len(COMMUNITY_COUNTS), 4),
        "community_size_range": list(COMMUNITY_SIZE_RANGE),
        "node_count_by_community_count": stats,
        "n_draws": n_draws,
    }


def main() -> int:
    payload = {
        "purpose": "CPU-only checks of claims in paper/main.tex that its own artifacts contradict.",
        "source": "results/condition_leakage_{planar,sbm}.json, results/k_sweep_planar{,_seedcheck}.json, scripts/downstream_classifier.py node-count law",
        "monotonicity": check_monotonicity(),
        "sweep_effects": check_sweep_effects(),
        "label_leakage": check_label_leakage(),
    }
    OUT.write_text(json.dumps(payload, indent=1))

    mono = payload["monotonicity"]["planar_low"]
    sweep = payload["sweep_effects"]
    leak = payload["label_leakage"]

    print(f"wrote {OUT}\n")
    print("1. monotonicity (Planar low band)")
    print(f"   recovery: {mono['recovery_pct']}  monotonic={mono['monotonic_nondecreasing']}  peak at k={mono['peak_at_k']}")
    print(f"   drops: {mono['drops']}\n")
    print("2. sweep effects")
    print(f"   baseline {sweep['baseline_mean_ratio']}, seed sd(none) {sweep['sd_none_seeds']}, sd(low k=8) {sweep['sd_low_k8_seeds']}")
    print(f"   {sweep['n_separated_beyond_3_none_sd']} of {sweep['n_cells']} cells beyond 3 none-sd:")
    for c in sweep["separated_cells"]:
        print(f"     {c['band']:9s} k={c['k']:<3d} ratio {c['ratio']:7.1f}  {c['gap_in_none_sd']:+6.1f} none-sd  {c['gap_in_low_sd']:+5.1f} low-sd")
    print(f"   all {sweep['n_controls']} control cells within {sweep['max_abs_control_gap_in_none_sd']} none-sd\n")
    print("3. label leakage (SBM downstream task)")
    print(f"   community count from node count alone: {leak['node_count_only_accuracy']:.1%} (chance {leak['chance_accuracy']:.0%})")
    for count, s in leak["node_count_by_community_count"].items():
        print(f"     {count} communities -> n mean {s['n_mean']}, range {s['n_min']}-{s['n_max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
