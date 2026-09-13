"""Run a sweep on Colab so that no completed run can be lost.

Three failures cost this project every SBM sweep result so far, and each one is defended
against here:

  1. **The VM is recycled mid-sweep.** Results written to /content vanish. Every report is
     copied to Drive the moment its run finishes, not at the end of the sweep.
  2. **The copy silently fails.** The previous attempt used
     `cp ... 2>/dev/null; echo "saved"`, where `;` runs the echo unconditionally and
     `2>/dev/null` hides the error, so it reported success while copying nothing. Here every
     copy is verified by reading the bytes back from Drive, and the script aborts if that
     check fails.
  3. **The working directory is gone.** `%cd` fails, every later command runs from the wrong
     place, and nothing says so. Preflight asserts the repo, the trainer, Drive and a
     round-trip write before a single GPU minute is spent.

Already-finished runs are skipped by looking in Drive, so re-running after a disconnect
resumes instead of starting over.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def die(message: str) -> None:
    print(f"\nPREFLIGHT FAILED: {message}", flush=True)
    raise SystemExit(2)


def preflight(repo: Path, drive: Path, trainer: Path) -> None:
    """Fail loudly before any training, rather than silently after it."""
    print("=== preflight ===", flush=True)

    if not repo.is_dir():
        die(f"repo not found at {repo} (VM recycled? re-run the setup cell)")
    print(f"  repo            {repo}", flush=True)

    if not trainer.is_file():
        die(f"trainer not found at {trainer}")
    print(f"  trainer         {trainer}", flush=True)

    gdrive_root = Path("/gdrive/MyDrive")
    if not gdrive_root.is_dir():
        die("/gdrive/MyDrive is not mounted (run drive.mount('/gdrive') first)")

    drive.mkdir(parents=True, exist_ok=True)
    if not drive.is_dir():
        die(f"could not create {drive}")
    print(f"  drive           {drive}", flush=True)

    # A round-trip write is the only proof Drive is actually writable: mkdir can succeed on a
    # stale mount that later drops every write.
    probe = drive / ".write_probe.json"
    payload = {"probe": time.time()}
    probe.write_text(json.dumps(payload))
    if json.loads(probe.read_text()) != payload:
        die(f"write probe to {probe} did not read back identical")
    probe.unlink()
    print("  write probe     round-trip OK", flush=True)

    try:
        from fald.eval import orca
    except Exception as exc:  # noqa: BLE001 - report any import failure verbatim
        die(f"cannot import fald.eval.orca: {exc}")
    if not orca.is_available():
        die(
            "ORCA is not available, so Ratio would average 4 metrics instead of 5 and would "
            "not be comparable to the Planar sweep (see docs/METRIC_SCALES.md)"
        )
    print("  ORCA            available (5-metric Ratio)", flush=True)
    print("=== preflight passed ===\n", flush=True)


def save_verified(source: Path, drive: Path) -> Path:
    """Copy one report to Drive and prove it arrived, or raise.

    Compares parsed JSON rather than byte length: a truncated or partially-flushed file can
    match on size and still be unreadable.
    """
    if not source.is_file():
        raise FileNotFoundError(f"expected report at {source}, which does not exist")

    original = json.loads(source.read_text())
    target = drive / source.name
    shutil.copyfile(source, target)

    if not target.is_file():
        raise IOError(f"copy to {target} produced no file")
    if json.loads(target.read_text()) != original:
        raise IOError(f"copy to {target} does not match the source after read-back")
    return target


def run_one(
    trainer: Path, dataset: str, band: str, k: int, seed: int, epochs: int, subdir: str
) -> subprocess.CompletedProcess:
    command = [
        sys.executable, str(trainer),
        "--dataset", dataset,
        "--band", band,
        "--k", str(k),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--allow-gate-failure",
        "--results-subdir", subdir,
    ]
    return subprocess.run(command, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="sbm")
    parser.add_argument("--repo", default="/content/fald")
    parser.add_argument("--drive", default=None, help="default: /gdrive/MyDrive/fald_results/<dataset>_sweep")
    parser.add_argument("--bands", nargs="+", default=["low", "high", "random", "gaussian", "shuffled", "none"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--subdir", default="k_sweep")
    parser.add_argument("--preflight-only", action="store_true", help="Check everything, train nothing.")
    args = parser.parse_args()

    repo = Path(args.repo)
    drive = Path(args.drive) if args.drive else Path(f"/gdrive/MyDrive/fald_results/{args.dataset}_sweep")
    trainer = repo / "scripts" / "train_adjacency_diffusion.py"

    preflight(repo, drive, trainer)
    if args.preflight_only:
        print("preflight-only: nothing trained.", flush=True)
        return 0

    jobs: list[tuple[str, int, int]] = []
    for band in args.bands:
        for seed in args.seeds:
            if band == "none":
                jobs.append((band, args.k_values[0], seed))
            else:
                jobs.extend((band, k, seed) for k in args.k_values)

    print(f"{len(jobs)} runs planned, saving each to {drive}\n", flush=True)
    completed, skipped, failed = 0, 0, 0

    for index, (band, k, seed) in enumerate(jobs, start=1):
        suffix = "" if seed == 0 else f"_seed{seed}"
        name = f"adjacency_diffusion_{args.dataset}_{band}_k{k}{suffix}.json"
        label = f"[{index}/{len(jobs)}] {band} k={k} seed={seed}"

        # Resume: a report already in Drive means this run finished in an earlier session.
        if (drive / name).is_file():
            print(f"{label} -- already in Drive, skipping", flush=True)
            skipped += 1
            continue

        print(f"{label} -- training", flush=True)
        started = time.time()
        result = run_one(trainer, args.dataset, band, k, seed, args.epochs, args.subdir)
        minutes = (time.time() - started) / 60.0

        if result.returncode != 0:
            print(f"{label} -- FAILED after {minutes:.1f} min", flush=True)
            print(result.stderr[-1500:], flush=True)
            failed += 1
            continue

        for line in result.stdout.splitlines():
            if "Ratio:" in line or "V.U.N" in line or "diagnosis:" in line:
                print(f"    {line.strip()}", flush=True)

        try:
            target = save_verified(repo / "results" / args.subdir / name, drive)
        except (FileNotFoundError, IOError, json.JSONDecodeError) as exc:
            # A save failure means every later run would be lost too, so stop now.
            print(f"\nSAVE FAILED for {name}: {exc}", flush=True)
            print("Aborting: later runs would be lost the same way.", flush=True)
            return 3

        print(f"    saved+verified -> {target}  ({minutes:.1f} min)", flush=True)
        completed += 1

    print(f"\n=== {completed} trained, {skipped} skipped, {failed} failed ===", flush=True)
    print(f"All verified reports are in {drive}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
