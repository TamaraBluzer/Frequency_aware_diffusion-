"""Run the S4.3 copying audit on every saved sample file, not just one arm.

    python DanielNewWork/audit_copying_all_arms.py

Writes DanielNewWork/copying_all_arms.json.

Why this exists: `results/*.pt` is gitignored (.gitignore line 31), so no sample
file is in the repository and each machine holds its own copy. The paper treats
the audit as covering 1 arm of 12 because only one file was present where it
ran. Twelve are present here -- 4 bands x 3 seeds at k=8 -- so the audit can be
run across all of them, which is what this script does.

It reuses the committed `upper_triangle_matrix` and `overlap_matrix` from
scripts/bootstrap_samples.py rather than reimplementing them, so the arithmetic
is the paper's own and only the coverage differs.

Two caveats that belong with any use of the output:

  1. These numbers come from this machine's sample files. They do not reproduce
     results/bootstrap_low_k8.json, which was computed on a different copy of
     `low k=8 seed 0` (see README). The conclusion replicates; the digits do not.
  2. Every file here is k=8. The k=16-vs-k=8 comparison the paper wants is still
     not answerable -- those samples were never saved.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from bootstrap_samples import overlap_matrix, upper_triangle_matrix
from fald.data.spectre import load_splits

BANDS = ("low", "high", "none", "random")
SEEDS = {"": 0, "_seed1": 1, "_seed2": 2}
OUT = Path(__file__).resolve().parent / "copying_all_arms.json"


def load(path: Path) -> list[nx.Graph]:
    """Identical to `load_samples` in scripts/bootstrap_samples.py."""
    return [nx.from_numpy_array(np.asarray(a).astype(np.uint8)) for a in torch.load(path, weights_only=False)]


def audit(gen: list[nx.Graph], ref_edges: np.ndarray) -> dict:
    gen_edges = upper_triangle_matrix(gen)
    scores = overlap_matrix(gen_edges, ref_edges)
    n = len(gen)
    eye = np.eye(n, dtype=bool)
    row = {"n_samples": n, "chance_top1": 1.0 / n}
    for metric in ("f1", "jaccard", "intersection"):
        matrix = scores[metric]
        row[metric] = {
            "matched_mean": float(matrix[eye].mean()),
            "mismatched_mean": float(matrix[~eye].mean()),
            "difference_mean": float(matrix[eye].mean() - matrix[~eye].mean()),
        }
    ranks = [int((scores["f1"][i] > scores["f1"][i, i]).sum()) + 1 for i in range(n)]
    row["retrieval"] = {
        "top1_count": sum(r == 1 for r in ranks),
        "top1_frac": sum(r == 1 for r in ranks) / n,
        "mean_rank": float(np.mean(ranks)),
    }
    row["connected_frac"] = float(np.mean([nx.is_connected(g) for g in gen]))
    row["mean_edges"] = float(np.mean([g.number_of_edges() for g in gen]))
    return row


def main() -> int:
    ref = load_splits("planar")["val"]
    ref_edges = upper_triangle_matrix(ref)

    arms: dict[str, dict] = {}
    for band in BANDS:
        for suffix, seed in SEEDS.items():
            path = REPO / "results" / f"adjacency_diffusion_planar_{band}_k8{suffix}_samples.pt"
            if not path.exists():
                continue
            row = audit(load(path), ref_edges)
            row.update({"band": band, "k": 8, "seed": seed, "samples_file": path.name})
            arms[f"{band}_k8_seed{seed}"] = row

    payload = {
        "experiment": "copying_audit_all_saved_arms",
        "purpose": (
            "Extend the S4.3 conditioning-graph reconstruction audit from 1 arm to every "
            "saved sample file on this machine (4 bands x 3 seeds, k=8, Planar val split)."
        ),
        "method": (
            "Reuses upper_triangle_matrix and overlap_matrix from scripts/bootstrap_samples.py. "
            "Sample i is paired with validation graph i, per that script's documented pairing."
        ),
        "reproducibility_warning": (
            "results/*.pt is gitignored, so these files are not in the repository and this "
            "audit is not reproducible from a clean clone. These numbers do not match "
            "results/bootstrap_low_k8.json, which ran on a different copy of low k=8 seed 0: "
            "matched F1 0.622 here against 0.653 there, connected_frac 0.688 against 0.469. "
            "The copying conclusion replicates; the specific digits do not."
        ),
        "coverage_still_missing": (
            "Every saved file is k=8. Whether k=16 reconstructs less than k=8 -- the "
            "comparison that would settle the section -- remains unanswerable."
        ),
        "dataset": "planar",
        "split": "val",
        "n_reference": len(ref),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": arms,
    }
    OUT.write_text(json.dumps(payload, indent=1))

    print(f"wrote {OUT}\n")
    header = f"{'arm':22s} {'matched':>8s} {'mismat':>7s} {'gap':>7s} {'top-1':>7s} {'conn':>6s}"
    print(header)
    print("-" * len(header))
    for name, row in arms.items():
        print(
            f"{name:22s} {row['f1']['matched_mean']:8.3f} {row['f1']['mismatched_mean']:7.3f} "
            f"{row['f1']['difference_mean']:7.3f} {row['retrieval']['top1_count']:4d}/{row['n_samples']:<2d} "
            f"{row['connected_frac']:6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
