"""T9: verify that a trained model changes output when only the condition changes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fald.data import build_condition_tensors, load_splits
from fald.data.dense import graphs_to_dense, pair_mask_from_node_mask
from fald.models import AdjacencyDiffusion, AdjacencyDiffusionConfig
from fald.models.adjacency_diffusion import adjacency_to_diffusion_state
from fald.paths import checkpoints_dir, results_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="planar", choices=["planar", "sbm"])
    parser.add_argument("--band", default="low", choices=["low", "high", "random"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    stem = f"adjacency_diffusion_{args.dataset}_{args.band}_k{args.k}{suffix}"
    checkpoint_path = checkpoints_dir() / f"{stem}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = AdjacencyDiffusionConfig(**checkpoint["config"])
    model = AdjacencyDiffusion(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    graphs = load_splits(args.dataset)["val"][: args.batch_size]
    adjacency, node_mask = graphs_to_dense(graphs)
    condition = build_condition_tensors(
        graphs,
        band=args.band,
        k=args.k,
        seed=args.seed + 10_000,
        cache_tag=f"{args.dataset}_val_sensitivity",
    )
    adjacency = adjacency.to(device)
    node_mask = node_mask.to(device)
    pair = condition.pair.to(device)
    eigenvalues = condition.eigenvalues.to(device)
    present = condition.present.to(device)
    absent = torch.zeros_like(present)

    clean = adjacency_to_diffusion_state(adjacency, node_mask)
    t = torch.full(
        (len(graphs),),
        cfg.timesteps // 2,
        dtype=torch.long,
        device=device,
    )
    noisy, _ = model.q_sample(clean, t, node_mask)
    with torch.no_grad():
        predicted_conditioned = model.predict_clean(
            noisy,
            t,
            node_mask,
            pair,
            eigenvalues,
            present,
        )
        predicted_unconditioned = model.predict_clean(
            noisy,
            t,
            node_mask,
            torch.zeros_like(pair),
            torch.zeros_like(eigenvalues),
            absent,
        )

    upper = pair_mask_from_node_mask(node_mask) & torch.triu(
        torch.ones_like(adjacency, dtype=torch.bool), diagonal=1
    )
    prediction_delta = (
        predicted_conditioned - predicted_unconditioned
    ).abs()[upper].mean().item()

    conditioned_generator = torch.Generator(device=device).manual_seed(7_919)
    unconditioned_generator = torch.Generator(device=device).manual_seed(7_919)
    conditioned_samples = model.sample(
        node_mask,
        pair,
        eigenvalues,
        present,
        generator=conditioned_generator,
    )
    unconditioned_samples = model.sample(
        node_mask,
        torch.zeros_like(pair),
        torch.zeros_like(eigenvalues),
        absent,
        generator=unconditioned_generator,
    )
    edit_fraction = (
        conditioned_samples != unconditioned_samples
    )[upper].float().mean().item()

    report = {
        "dataset": args.dataset,
        "band": args.band,
        "k": args.k,
        "seed": args.seed,
        "n_graphs": len(graphs),
        "mean_absolute_prediction_delta": prediction_delta,
        "same_noise_edge_edit_fraction": edit_fraction,
        "passed": prediction_delta > 1e-3 and edit_fraction > 1e-3,
    }
    output = results_dir() / f"{stem}_condition_sensitivity.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
