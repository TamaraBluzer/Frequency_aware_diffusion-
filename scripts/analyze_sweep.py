"""Join the k-sweep against measured leakage and plot both views.

The question is whether "frequency band" explains anything that "how much of the target the
condition leaks" does not. Two panels answer it:

  * Ratio vs k, one line per band -- the frequency view.
  * Ratio vs measured leakage, one point per (band, k) -- the information view.

If every arm collapses onto a single curve in the second panel, the frequency axis is a proxy
for information content and the paper's framing has to change. If the bands stay separated at
equal leakage, frequency is carrying something of its own.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fald.paths import results_dir


def load_leakage(dataset: str) -> dict[tuple[str, int], float]:
    path = results_dir() / f"condition_leakage_{dataset}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return {(row["band"], row["k"]): row["edge_recovery"] for row in payload["rows"]}


def aggregate(rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Average over seeds for each (band, k)."""
    grouped = defaultdict(list)
    for row in rows:
        if row.get("failed") or row.get("skipped") or row.get("ratio") is None:
            continue
        grouped[(row["band"], row["k"])].append(row)

    aggregated = {}
    for key, group in grouped.items():
        ratios = [item["ratio"] for item in group]
        validities = [item.get("validity") or 0.0 for item in group]
        losses = [
            item["validation_loss"] for item in group if item.get("validation_loss") is not None
        ]
        aggregated[key] = {
            "ratio_mean": float(np.mean(ratios)),
            "ratio_std": float(np.std(ratios)),
            "validity_mean": float(np.mean(validities)),
            "validation_loss_mean": float(np.mean(losses)) if losses else None,
            "n_seeds": len(group),
        }
    return aggregated


def plot(aggregated: dict, leakage: dict, dataset: str, output: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure")
        return None

    bands = sorted({band for band, _ in aggregated})
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    for band in bands:
        points = sorted(
            ((k, value) for (b, k), value in aggregated.items() if b == band),
            key=lambda item: item[0],
        )
        if not points:
            continue
        ks = [k for k, _ in points]
        means = [value["ratio_mean"] for _, value in points]
        errors = [value["ratio_std"] for _, value in points]
        left.errorbar(ks, means, yerr=errors, marker="o", capsize=3, label=band)

        xs = [leakage.get((band, k)) for k, _ in points]
        pairs = [(x, y) for x, y in zip(xs, means) if x is not None]
        if pairs:
            right.scatter(*zip(*pairs), label=band, s=55)

    left.set_xscale("log", base=2)
    left.set_xlabel("k (eigenpairs)")
    left.set_ylabel("Ratio (lower is better)")
    left.set_title("Frequency view: Ratio vs k")
    left.legend(fontsize=8)
    left.grid(alpha=0.3)

    right.set_xlabel("measured edge recovery from condition alone")
    right.set_ylabel("Ratio (lower is better)")
    right.set_title("Information view: Ratio vs leakage")
    right.legend(fontsize=8)
    right.grid(alpha=0.3)

    figure.suptitle(f"{dataset}: does frequency explain anything leakage does not?")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="planar")
    parser.add_argument("--sweep", default=None)
    args = parser.parse_args()

    sweep_path = Path(args.sweep) if args.sweep else results_dir() / f"k_sweep_{args.dataset}.json"
    if not sweep_path.is_file():
        raise SystemExit(f"no sweep results at {sweep_path}; run scripts/run_k_sweep.py first")

    payload = json.loads(sweep_path.read_text())
    aggregated = aggregate(payload["rows"])
    leakage = load_leakage(args.dataset)

    print(f"{'band':<10}{'k':>4}{'ratio':>10}{'std':>8}{'valid':>8}{'leakage':>9}")
    for (band, k), value in sorted(aggregated.items()):
        recovered = leakage.get((band, k))
        print(
            f"{band:<10}{k:>4}{value['ratio_mean']:>10.2f}{value['ratio_std']:>8.2f}"
            f"{value['validity_mean']:>8.1%}"
            f"{'  n/a' if recovered is None else f'{recovered:>8.1%}'}"
        )

    figure_path = plot(
        aggregated, leakage, args.dataset, results_dir() / "figures" / f"k_sweep_{args.dataset}.png"
    )

    summary_path = results_dir() / f"k_sweep_{args.dataset}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "orca_available": payload.get("orca_available"),
                "aggregated": {f"{b}_k{k}": v for (b, k), v in aggregated.items()},
                "leakage": {f"{b}_k{k}": v for (b, k), v in leakage.items()},
            },
            indent=2,
        )
    )
    print(f"\nwrote {summary_path}")
    if figure_path:
        print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
