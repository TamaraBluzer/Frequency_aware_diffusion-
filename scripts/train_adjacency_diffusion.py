"""Train and evaluate Tier-0 adjacency diffusion.

The model diffuses the dense adjacency directly.  Spectral eigenpairs are an
optional oracle side condition; they are never the diffusion state itself.
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

from fald.data import build_condition_tensors, load_splits
from fald.data.dense import graphs_to_dense
from fald.eval import GraphEvaluator
from fald.eval import orca
from fald.eval.validity import is_planar
from fald.models import (
    AdjacencyDiffusion,
    AdjacencyDiffusionConfig,
    DiscreteAdjacencyDiffusion,
)
from fald.paths import checkpoints_dir, results_dir


def _conditioned_tensors(graphs, band: str, k: int, seed: int, tag: str):
    adjacency, node_mask = graphs_to_dense(graphs)
    condition = build_condition_tensors(
        graphs,
        band=band,
        k=k,
        seed=seed,
        cache_tag=tag,
    )
    return TensorDataset(
        adjacency,
        node_mask,
        condition.pair,
        condition.eigenvalues,
        condition.present,
    )


def _positive_weight(adjacency: torch.Tensor, node_mask: torch.Tensor) -> float:
    n = adjacency.shape[1]
    valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    valid &= torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    positive = adjacency[valid].sum().item()
    negative = valid.sum().item() - positive
    return max(1.0, negative / max(positive, 1.0))


def _move_batch(batch, device):
    return tuple(value.to(device) for value in batch)


@torch.no_grad()
def _validation_loss(model, loader, device, positive_weight) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        adjacency, node_mask, pair, eigenvalues, present = _move_batch(batch, device)
        loss = model.training_loss(
            adjacency,
            node_mask,
            pair,
            eigenvalues,
            present,
            positive_weight,
        )
        total += loss.item() * len(adjacency)
    return total / max(len(loader.dataset), 1)


def _to_graphs(adjacency: torch.Tensor, node_mask: torch.Tensor) -> list[nx.Graph]:
    graphs = []
    for matrix, mask in zip(adjacency.cpu(), node_mask.cpu()):
        n = int(mask.sum().item())
        graphs.append(nx.from_numpy_array(matrix[:n, :n].numpy().astype(np.uint8)))
    return graphs


@torch.no_grad()
def _sample_loader(model, loader, device, guidance_scale) -> list[nx.Graph]:
    generated = []
    for batch in loader:
        _, node_mask, pair, eigenvalues, present = _move_batch(batch, device)
        sampled = model.sample(
            node_mask,
            pair,
            eigenvalues,
            present,
            guidance_scale=guidance_scale,
        )
        generated.extend(_to_graphs(sampled, node_mask))
    return generated


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="planar", choices=["planar", "sbm"])
    parser.add_argument("--process", default="continuous", choices=["continuous", "discrete"])
    parser.add_argument("--band", default="none", choices=["none", "low", "high", "random"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dx", type=int, default=128)
    parser.add_argument("--de", type=int, default=64)
    parser.add_argument("--dy", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--cycle-features", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-full-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--allow-gate-failure", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    splits = load_splits(args.dataset)
    train_graphs = splits["train"][: args.limit_train]
    eval_graphs = splits[args.eval_split][: args.limit_eval]
    train_data = _conditioned_tensors(
        train_graphs,
        args.band,
        args.k,
        args.seed,
        f"{args.dataset}_train",
    )
    eval_data = _conditioned_tensors(
        eval_graphs,
        args.band,
        args.k,
        args.seed + 10_000,
        f"{args.dataset}_{args.eval_split}",
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

    cfg = AdjacencyDiffusionConfig(
        timesteps=args.timesteps,
        condition_k=args.k,
        n_layers=args.n_layers,
        dx=args.dx,
        de=args.de,
        dy=args.dy,
        n_head=args.n_head,
        use_cycle_features=args.cycle_features,
    )
    class_balance_weight = _positive_weight(train_data.tensors[0], train_data.tensors[1])
    edge_marginal = 1.0 / (1.0 + class_balance_weight)
    model = (
        AdjacencyDiffusion(cfg)
        if args.process == "continuous"
        else DiscreteAdjacencyDiffusion(cfg, edge_marginal=edge_marginal)
    ).to(device)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    positive_weight = args.positive_weight
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"{args.dataset}: train={len(train_graphs)} {args.eval_split}={len(eval_graphs)} "
        f"process={args.process} band={args.band} k={args.k}"
    )
    print(
        f"model={n_parameters / 1e6:.2f}M timesteps={cfg.timesteps} "
        f"positive_weight={positive_weight:.2f} "
        f"(class-balance value would be {class_balance_weight:.2f}) amp={use_amp}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = {"epoch": 0, "validation_loss": float("inf")}
    history = []
    seed_suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    model_prefix = (
        "adjacency_diffusion"
        if args.process == "continuous"
        else "discrete_adjacency_diffusion"
    )
    if args.cycle_features:
        model_prefix += "_cycles"
    checkpoint_path = (
        checkpoints_dir()
        / f"{model_prefix}_{args.dataset}_{args.band}_k{args.k}{seed_suffix}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for batch in train_loader:
                adjacency, node_mask, pair, eigenvalues, present = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                    loss = model.training_loss(
                        adjacency,
                        node_mask,
                        pair,
                        eigenvalues,
                        present,
                        positive_weight,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(adjacency)
            scheduler.step()

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                validation_loss = _validation_loss(
                    model,
                    eval_loader,
                    device,
                    positive_weight,
                )
                train_loss = loss_sum / len(train_loader.dataset)
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                    }
                )
                print(
                    f"epoch {epoch:4d} train={train_loss:.5f} "
                    f"{args.eval_split}={validation_loss:.5f}"
                )
                if validation_loss < best["validation_loss"]:
                    best = {"epoch": epoch, "validation_loss": validation_loss}
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": asdict(cfg),
                            "args": vars(args),
                            "best": best,
                        },
                        checkpoint_path,
                    )
    elif not checkpoint_path.exists():
        raise FileNotFoundError(f"no checkpoint to evaluate at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not args.eval_only:
        checkpoint["history"] = history
        checkpoint["training_minutes"] = (time.time() - started) / 60.0
        torch.save(checkpoint, checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    best = checkpoint["best"]
    history = checkpoint.get("history", history)
    generated = _sample_loader(model, eval_loader, device, args.guidance_scale)
    elapsed_minutes = (time.time() - started) / 60.0
    training_minutes = checkpoint.get("training_minutes", elapsed_minutes)
    densities = [nx.density(graph) for graph in generated]
    print(
        f"sampled={len(generated)} density={np.mean(densities):.4f} "
        f"elapsed={elapsed_minutes:.1f} min"
    )

    evaluation = None
    er_ratio = None
    ratio_gate_passed = None
    validity_gate_passed = None
    gate_passed = None
    if not args.no_full_eval:
        metrics = ["degree", "clustering", "spectral", "wavelet"]
        if orca.is_available():
            metrics.append("orbit")
        else:
            print("ORCA unavailable: evaluating with degree/clustering/spectral/wavelet")
        evaluator = GraphEvaluator(
            eval_graphs,
            train_graphs=train_graphs,
            validity_func=is_planar if args.dataset == "planar" else None,
            metrics=metrics,
        )
        floor = evaluator.self_similarity(train_graphs)
        evaluation = evaluator.evaluate(generated, baseline_mmd=floor).as_flat_dict()
        er_graphs = _density_matched_er(eval_graphs, args.seed)
        er_result = evaluator.evaluate(er_graphs, baseline_mmd=floor)
        er_ratio = er_result.ratio
        ratio_gate_passed = (
            evaluation.get("ratio") is not None
            and er_ratio is not None
            and evaluation["ratio"] < er_ratio
        )
        validity_gate_passed = (
            evaluation.get("vun/valid", 0.0) > 0.0
            if args.process == "discrete" and args.dataset == "planar"
            else None
        )
        gate_passed = ratio_gate_passed and validity_gate_passed is not False
        print(f"Ratio: model={evaluation['ratio']:.2f} ER={er_ratio:.2f}")
        if args.dataset == "planar":
            print(
                f"V.U.N.: valid={evaluation['vun/valid']:.3f} "
                f"unique={evaluation['vun/unique']:.3f} "
                f"novel={evaluation['vun/novel']:.3f} "
                f"joint={evaluation['vun/vun']:.3f}"
            )

    report = {
        "dataset": args.dataset,
        "process": args.process,
        "band": args.band,
        "k": args.k,
        "seed": args.seed,
        "config": asdict(cfg),
        "n_parameters": n_parameters,
        "positive_weight": positive_weight,
        "class_balance_weight": class_balance_weight,
        "edge_marginal": edge_marginal,
        "best": best,
        "history": history,
        "generated_density_mean": float(np.mean(densities)),
        "generated_density_std": float(np.std(densities)),
        "evaluation": evaluation,
        "er_ratio": er_ratio,
        "ratio_gate_passed": ratio_gate_passed,
        "validity_gate_passed": validity_gate_passed,
        "gate_passed": gate_passed,
        "training_minutes": training_minutes,
        "evaluation_minutes": elapsed_minutes,
    }
    results_dir().mkdir(parents=True, exist_ok=True)
    run_stem = f"{model_prefix}_{args.dataset}_{args.band}_k{args.k}{seed_suffix}"
    samples_path = results_dir() / f"{run_stem}_samples.pt"
    torch.save(
        [nx.to_numpy_array(graph, dtype=np.uint8) for graph in generated],
        samples_path,
    )
    report["samples_path"] = str(samples_path)
    report_path = results_dir() / f"{run_stem}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {report_path}")

    if gate_passed is False:
        failed_checks = []
        if ratio_gate_passed is False:
            failed_checks.append("generated graphs did not beat density-matched ER on Ratio")
        if validity_gate_passed is False:
            failed_checks.append("discrete generation has zero Planar validity")
        print(f"GATE FAILED: {'; '.join(failed_checks)}")
        return 0 if args.allow_gate_failure else 1
    if gate_passed:
        print("GATE PASSED: generated graphs beat density-matched ER on Ratio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
