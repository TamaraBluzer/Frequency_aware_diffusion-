"""End-to-end sampling: spectral prior -> adjacency diffusion.

Stage 8 of PLAN.md. Uses the learned spectral prior to generate (eigenvalues,
eigenvectors), then feeds them as conditioning to the adjacency diffusion model
to produce graphs without any oracle spectral inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_digress_dir = Path(__file__).resolve().parents[1] / "third_party" / "digress"
if _digress_dir.is_dir() and str(_digress_dir) not in sys.path:
    sys.path.insert(0, str(_digress_dir))

from fald.data import load_splits
from fald.data.dense import graphs_to_dense
from fald.eval import GraphEvaluator
from fald.eval import orca
from fald.eval.validity import is_planar
from fald.models import AdjacencyDiffusion, AdjacencyDiffusionConfig
from fald.models.spectral_prior import SpectralPrior, SpectralPriorConfig
from fald.paths import checkpoints_dir, results_dir


def _to_graphs(adjacency: torch.Tensor, node_mask: torch.Tensor) -> list[nx.Graph]:
    graphs = []
    for matrix, mask in zip(adjacency.cpu(), node_mask.cpu()):
        n = int(mask.sum().item())
        graphs.append(nx.from_numpy_array(matrix[:n, :n].numpy().astype(np.uint8)))
    return graphs


def _density_matched_er(graphs, seed: int) -> list[nx.Graph]:
    rng = np.random.default_rng(seed)
    controls = []
    for graph in graphs:
        n = graph.number_of_nodes()
        p = 2.0 * graph.number_of_edges() / max(n * (n - 1), 1)
        controls.append(
            nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31 - 1)))
        )
    return controls


def _build_pair_channel(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    n_nodes: torch.Tensor,
    n_max: int,
) -> torch.Tensor:
    """Build U diag(lambda) U^T pair condition from sampled spectra."""
    batch = eigenvalues.shape[0]
    pair = torch.zeros(batch, n_max, n_max, dtype=eigenvalues.dtype, device=eigenvalues.device)
    for i in range(batch):
        n = int(n_nodes[i].item())
        U = eigenvectors[i, :n, :]
        lam = eigenvalues[i] + 1.0  # undo the [-1,1] -> [0,2] normalisation
        channel = U @ torch.diag(lam) @ U.T
        rms = channel.square().mean().sqrt().clamp(min=1e-8)
        pair[i, :n, :n] = channel / rms
    return pair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="planar", choices=["planar", "sbm"])
    parser.add_argument("--band", default="low", choices=["low", "high", "random"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--prior-checkpoint", type=str, default=None)
    parser.add_argument("--diffusion-checkpoint", type=str, default=None)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Load eval data
    splits = load_splits(args.dataset)
    train_graphs = splits["train"]
    eval_graphs = splits["val"]

    adjacency, node_mask = graphs_to_dense(eval_graphs)
    n_max = node_mask.shape[1]
    n_nodes = node_mask.sum(dim=1).long()

    # Load spectral prior
    prior_path = args.prior_checkpoint
    if prior_path is None:
        prior_path = str(
            checkpoints_dir()
            / f"spectral_prior_{args.dataset}_{args.band}_k{args.k}.pt"
        )
    prior_ckpt = torch.load(prior_path, map_location=device, weights_only=False)
    prior_cfg = SpectralPriorConfig(**prior_ckpt["config"])
    prior = SpectralPrior(prior_cfg).to(device)
    prior.load_state_dict(prior_ckpt["model"])
    print(f"loaded spectral prior from {prior_path}")

    # Load adjacency diffusion model
    diff_path = args.diffusion_checkpoint
    if diff_path is None:
        diff_path = str(
            checkpoints_dir()
            / f"adjacency_diffusion_{args.dataset}_{args.band}_k{args.k}.pt"
        )
    diff_ckpt = torch.load(diff_path, map_location=device, weights_only=False)
    diff_cfg = AdjacencyDiffusionConfig(**diff_ckpt["config"])
    diffusion = AdjacencyDiffusion(diff_cfg).to(device)
    diffusion.load_state_dict(diff_ckpt["model"])
    print(f"loaded adjacency diffusion from {diff_path}")

    # Sample spectra from the prior
    print("sampling spectra from prior...")
    started = time.time()
    gen_eigenvalues, gen_eigenvectors = prior.sample(n_nodes.to(device))
    prior_time = time.time() - started
    print(f"spectral prior sampling: {prior_time:.1f}s")

    # Build pair condition from sampled spectra
    pair_condition = _build_pair_channel(
        gen_eigenvalues, gen_eigenvectors, n_nodes.to(device), n_max,
    )
    eigenvalue_condition = gen_eigenvalues
    condition_present = torch.ones(len(eval_graphs), 1, dtype=torch.float32, device=device)

    # Sample graphs from the adjacency diffusion model
    print("sampling graphs from adjacency diffusion...")
    started = time.time()
    sampled_adjacency = diffusion.sample(
        node_mask.to(device),
        pair_condition,
        eigenvalue_condition,
        condition_present,
        guidance_scale=args.guidance_scale,
    )
    diffusion_time = time.time() - started
    print(f"adjacency diffusion sampling: {diffusion_time:.1f}s")

    generated = _to_graphs(sampled_adjacency, node_mask)
    densities = [nx.density(g) for g in generated]
    print(f"sampled={len(generated)} density={np.mean(densities):.4f}")

    # Evaluate
    metrics_list = ["degree", "clustering", "spectral", "wavelet"]
    if orca.is_available():
        metrics_list.append("orbit")
    evaluator = GraphEvaluator(
        eval_graphs,
        train_graphs=train_graphs,
        validity_func=is_planar if args.dataset == "planar" else None,
        metrics=metrics_list,
    )
    floor = evaluator.self_similarity(train_graphs)
    evaluation = evaluator.evaluate(generated, baseline_mmd=floor).as_flat_dict()
    er_graphs = _density_matched_er(eval_graphs, args.seed)
    er_result = evaluator.evaluate(er_graphs, baseline_mmd=floor)

    print(f"\nRatio: model={evaluation['ratio']:.2f} ER={er_result.ratio:.2f}")
    if args.dataset == "planar":
        print(
            f"V.U.N.: valid={evaluation['vun/valid']:.3f} "
            f"unique={evaluation['vun/unique']:.3f} "
            f"novel={evaluation['vun/novel']:.3f} "
            f"joint={evaluation['vun/vun']:.3f}"
        )

    # Save report
    results_dir().mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "end_to_end",
        "dataset": args.dataset,
        "band": args.band,
        "k": args.k,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "prior_checkpoint": prior_path,
        "diffusion_checkpoint": diff_path,
        "prior_config": asdict(prior_cfg),
        "diffusion_config": asdict(diff_cfg),
        "evaluation": evaluation,
        # Without this, a report's Ratio cannot be placed on a scale: orbit is appended only
        # when ORCA is present, and its absence shifts the mean by ~2.5x. Stage 8's reports
        # predate this field, which is why their 161.67 ER could not be attributed directly.
        "metrics": metrics_list,
        "orca_available": orca.is_available(),
        "er_ratio": er_result.ratio,
        "prior_sampling_seconds": prior_time,
        "diffusion_sampling_seconds": diffusion_time,
    }
    report_path = results_dir() / f"end_to_end_{args.dataset}_{args.band}_k{args.k}_gs{args.guidance_scale}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {report_path}")

    gate_passed = (
        evaluation.get("ratio") is not None
        and er_result.ratio is not None
        and evaluation["ratio"] < er_result.ratio
    )
    if gate_passed:
        print("GATE PASSED: end-to-end generation beats density-matched ER on Ratio")
    else:
        print("GATE FAILED: end-to-end generation did not beat ER on Ratio")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
