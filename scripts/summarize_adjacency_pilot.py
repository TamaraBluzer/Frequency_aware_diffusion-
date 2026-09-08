"""Summarize the multi-seed Tier-0 frequency pilot with paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fald.data import load_splits
from fald.eval import GraphEvaluator
from fald.eval.mmd import compute_mmd, gaussian_tv
from fald.paths import results_dir


ARMS = ("none", "low", "high", "random")


def _run_stem(dataset: str, arm: str, k: int, seed: int) -> str:
    suffix = "" if seed == 0 else f"_seed{seed}"
    return f"adjacency_diffusion_{dataset}_{arm}_k{k}{suffix}"


def _load_graphs(path: Path) -> list[nx.Graph]:
    matrices = torch.load(path, weights_only=False)
    return [nx.from_numpy_array(np.asarray(matrix, dtype=np.uint8)) for matrix in matrices]


def _ratio_from_descriptors(evaluator, descriptors, indices, floor) -> float:
    ratios = []
    for spec in evaluator.specs:
        candidate = [descriptors[spec.name][index] for index in indices]
        mmd = compute_mmd(
            evaluator._reference_descriptors[spec.name],
            candidate,
            kernel=gaussian_tv,
            is_hist=spec.is_hist,
            sigma=spec.sigma,
        )
        ratios.append(mmd / max(floor[spec.name], 1e-8))
    return float(np.mean(ratios))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="planar", choices=["planar"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1234)
    args = parser.parse_args()

    reports = {}
    graphs = {}
    for arm in ARMS:
        reports[arm] = {}
        graphs[arm] = {}
        for seed in args.seeds:
            stem = _run_stem(args.dataset, arm, args.k, seed)
            report_path = results_dir() / f"{stem}.json"
            sample_path = results_dir() / f"{stem}_samples.pt"
            if not report_path.exists() or not sample_path.exists():
                raise FileNotFoundError(
                    f"missing run artifacts for {arm} seed {seed}: "
                    f"{report_path.name}, {sample_path.name}"
                )
            reports[arm][seed] = json.loads(report_path.read_text())
            graphs[arm][seed] = _load_graphs(sample_path)

    splits = load_splits(args.dataset)
    evaluator = GraphEvaluator(splits["val"], metrics=(
        "degree",
        "clustering",
        "spectral",
        "wavelet",
        "orbit",
    ))
    floor = evaluator.self_similarity(splits["train"])
    descriptors = {
        arm: {
            seed: evaluator._describe(graphs[arm][seed])
            for seed in args.seeds
        }
        for arm in ARMS
    }

    summary = {"dataset": args.dataset, "k": args.k, "seeds": args.seeds, "arms": {}}
    print("arm       Ratio mean ± std    validation loss mean ± std    validity")
    for arm in ARMS:
        ratios = np.array(
            [reports[arm][seed]["evaluation"]["ratio"] for seed in args.seeds]
        )
        losses = np.array(
            [reports[arm][seed]["best"]["validation_loss"] for seed in args.seeds]
        )
        validity = np.array(
            [reports[arm][seed]["evaluation"]["vun/valid"] for seed in args.seeds]
        )
        summary["arms"][arm] = {
            "ratio_mean": float(ratios.mean()),
            "ratio_std": float(ratios.std(ddof=1)),
            "validation_loss_mean": float(losses.mean()),
            "validation_loss_std": float(losses.std(ddof=1)),
            "validity_mean": float(validity.mean()),
        }
        print(
            f"{arm:<9} {ratios.mean():8.2f} ± {ratios.std(ddof=1):6.2f}"
            f"        {losses.mean():.5f} ± {losses.std(ddof=1):.5f}"
            f"              {validity.mean():.3f}"
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons = {}
    for comparator in ("none", "high", "random"):
        differences = []
        for _ in range(args.bootstraps):
            selected_seeds = rng.choice(args.seeds, size=len(args.seeds), replace=True)
            seed_differences = []
            for seed in selected_seeds:
                n_samples = len(graphs["low"][int(seed)])
                indices = rng.integers(0, n_samples, size=n_samples)
                low_ratio = _ratio_from_descriptors(
                    evaluator,
                    descriptors["low"][int(seed)],
                    indices,
                    floor,
                )
                comparator_ratio = _ratio_from_descriptors(
                    evaluator,
                    descriptors[comparator][int(seed)],
                    indices,
                    floor,
                )
                seed_differences.append(low_ratio - comparator_ratio)
            differences.append(float(np.mean(seed_differences)))

        interval = np.percentile(differences, [2.5, 50.0, 97.5])
        comparisons[f"low_minus_{comparator}"] = {
            "ci_95": [float(interval[0]), float(interval[2])],
            "median": float(interval[1]),
            "probability_low_better": float(np.mean(np.asarray(differences) < 0)),
        }
        print(
            f"low - {comparator}: median {interval[1]:.2f}, "
            f"95% CI [{interval[0]:.2f}, {interval[2]:.2f}]"
        )

    summary["paired_hierarchical_bootstrap"] = comparisons
    summary["bootstraps"] = args.bootstraps
    output = results_dir() / f"frequency_pilot_{args.dataset}_k{args.k}_summary.json"
    output.write_text(json.dumps(summary, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
