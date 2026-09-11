"""Frequency cutoff gradient: the experiment the proposal called key.

Runs `train_adjacency_diffusion.py` across bands x k x seeds and collects the reports into
one file. The proposal specified k in {2,4,8,16,32}; the paper reported a single k, which is
the gap this closes.

Two arms exist here that the original design did not have:

  * `gaussian` -- pure noise, shape-matched. Establishes what "a condition that carries
    nothing" scores, which is the floor any real band must beat.
  * `shuffled` -- another graph's real low band. Separates "low-frequency structure helps"
    from "the *matching* condition helps". If shuffled matches low, the model is exploiting
    band statistics rather than the target's own spectrum.

Pair the output with `results/condition_leakage_<dataset>.json`: plotting Ratio against
measured leakage tests whether the frequency axis explains anything beyond information
content. Points falling on one curve would mean it does not.

ORCA must be importable or every Ratio is on a 4-metric scale and not comparable to Table 1
(see docs/METRIC_SCALES.md). The runner refuses to start otherwise unless forced.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fald.eval import orca
from fald.paths import results_dir

TRAINER = Path(__file__).resolve().parent / "train_adjacency_diffusion.py"


def run_one(
    dataset: str,
    band: str,
    k: int,
    seed: int,
    epochs: int,
    extra: list[str],
    dry_run: bool,
) -> dict:
    command = [
        sys.executable,
        str(TRAINER),
        "--dataset", dataset,
        "--band", band,
        "--k", str(k),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--allow-gate-failure",
        *extra,
    ]
    if dry_run:
        return {"command": " ".join(command), "skipped": True}

    started = time.time()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = (time.time() - started) / 60.0

    if completed.returncode != 0:
        return {
            "band": band, "k": k, "seed": seed, "failed": True,
            "stderr": completed.stderr[-2000:], "minutes": elapsed,
        }

    # The trainer names its report by dataset/band/k/seed; re-read it rather than parsing stdout.
    suffix = "" if seed == 0 else f"_seed{seed}"
    report_path = results_dir() / f"adjacency_diffusion_{dataset}_{band}_k{k}{suffix}.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    evaluation = report.get("evaluation") or {}

    return {
        "band": band,
        "k": k,
        "seed": seed,
        "ratio": evaluation.get("ratio"),
        "er_ratio": report.get("er_ratio"),
        "validation_loss": (report.get("best") or {}).get("validation_loss"),
        "validity": evaluation.get("vun/valid"),
        "validity_diagnosis": report.get("validity_diagnosis"),
        "metrics": report.get("metrics"),
        "ratio_per_metric": {
            key.split("/", 1)[1]: value
            for key, value in evaluation.items()
            if key.startswith("ratio/")
        },
        "minutes": elapsed,
        "report_path": str(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="planar")
    parser.add_argument(
        "--bands", nargs="+",
        default=["low", "high", "random", "gaussian", "shuffled", "none"],
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing reports with the same band/k/seed names.",
    )
    parser.add_argument(
        "--allow-missing-orca", action="store_true",
        help="Run without ORCA. Ratios will be 4-metric and NOT comparable to Table 1.",
    )
    args, extra = parser.parse_known_args()

    # A sweep writes one report per (band, k, seed), and those names collide with the pilot
    # reports already in results/. Losing a published number to a sweep rerun is not
    # recoverable from the report itself, so refuse rather than overwrite.
    if not args.dry_run and not args.overwrite:
        collisions = []
        for band in args.bands:
            for k in args.k_values:
                for seed in args.seeds:
                    suffix = "" if seed == 0 else f"_seed{seed}"
                    candidate = (
                        results_dir()
                        / f"adjacency_diffusion_{args.dataset}_{band}_k{k}{suffix}.json"
                    )
                    if candidate.is_file():
                        collisions.append(candidate.name)
        if collisions:
            print(
                f"{len(collisions)} existing report(s) would be overwritten, including "
                f"{', '.join(collisions[:3])}. Move them aside, or pass --overwrite if you "
                "intend to replace them.",
                file=sys.stderr,
            )
            return 3

    if not orca.is_available() and not args.allow_missing_orca and not args.dry_run:
        print(
            "ORCA is not importable, so Ratio would average 4 metrics instead of 5 and would "
            "not be comparable to Table 1 (see docs/METRIC_SCALES.md).\n"
            "Build ORCA, or pass --allow-missing-orca to accept an incomparable scale.",
            file=sys.stderr,
        )
        return 2

    # 'none' has no band, so it is run once per seed rather than once per (k, seed).
    jobs: list[tuple[str, int, int]] = []
    for band, seed in itertools.product(args.bands, args.seeds):
        if band == "none":
            jobs.append((band, args.k_values[0], seed))
        else:
            jobs.extend((band, k, seed) for k in args.k_values)

    print(f"{len(jobs)} runs: bands={args.bands} k={args.k_values} seeds={args.seeds}")

    rows = []
    for index, (band, k, seed) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {band} k={k} seed={seed}", flush=True)
        row = run_one(args.dataset, band, k, seed, args.epochs, extra, args.dry_run)
        rows.append(row)
        if row.get("failed"):
            print(f"  FAILED: {row['stderr'][-300:]}", flush=True)
        elif not row.get("skipped"):
            ratio = row.get("ratio")
            print(
                f"  ratio={ratio if ratio is None else round(ratio, 2)} "
                f"valid={row.get('validity')} ({row['minutes']:.1f} min)",
                flush=True,
            )

    payload = {
        "experiment": "k_sweep",
        "dataset": args.dataset,
        "bands": args.bands,
        "k_values": args.k_values,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "orca_available": orca.is_available(),
        "rows": rows,
    }
    output = Path(args.output) if args.output else results_dir() / f"k_sweep_{args.dataset}.json"
    output.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
