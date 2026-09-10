"""Train Stage 8 spectral prior: diffusion over (n, lambda_k, U_k).

Learns to generate spectral features so the adjacency diffusion model can
sample end-to-end without oracle inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fald.data import load_splits
from fald.data.spectral import cached_eigendecompositions, select_band
from fald.models.spectral_prior import SpectralPrior, SpectralPriorConfig
from fald.paths import checkpoints_dir, results_dir


def _build_spectral_dataset(
    graphs,
    band: str,
    k: int,
    n_max: int,
    seed: int,
    cache_tag: str,
) -> TensorDataset:
    """Extract eigenvalues and eigenvectors for each graph, padded to n_max."""
    all_vals, all_vecs = cached_eigendecompositions(graphs, tag=cache_tag)

    eigenvalues = np.zeros((len(graphs), k), dtype=np.float32)
    eigenvectors = np.zeros((len(graphs), n_max, k), dtype=np.float32)
    n_nodes = np.zeros((len(graphs),), dtype=np.int64)

    for i, (graph, vals, vecs) in enumerate(zip(graphs, all_vals, all_vecs)):
        rng = np.random.default_rng(seed + i)
        cond = select_band(vals, vecs, band, k, rng=rng)
        n = graph.number_of_nodes()
        n_nodes[i] = n
        eigenvalues[i] = (cond.eigvals - 1.0).astype(np.float32)
        eigenvectors[i, :n, :] = cond.eigvecs.astype(np.float32)

    return TensorDataset(
        torch.from_numpy(eigenvalues),
        torch.from_numpy(eigenvectors),
        torch.from_numpy(n_nodes),
    )


def _move(batch, device):
    return tuple(v.to(device) for v in batch)


@torch.no_grad()
def _val_loss(model, loader, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        eigenvalues, eigenvectors, n_nodes = _move(batch, device)
        loss = model.training_loss(eigenvalues, eigenvectors, n_nodes)
        total += loss.item() * len(eigenvalues)
        count += len(eigenvalues)
    return total / max(count, 1)


@torch.no_grad()
def _evaluate_samples(
    model,
    ref_eigenvalues: torch.Tensor,
    ref_eigenvectors: torch.Tensor,
    ref_n_nodes: torch.Tensor,
    device,
) -> dict:
    """Generate samples and compute quality metrics against reference spectra."""
    model.eval()
    n_nodes = ref_n_nodes.to(device)
    gen_eigenvalues, gen_eigenvectors = model.sample(n_nodes)

    ref_ev = ref_eigenvalues.to(device)
    gen_ev = gen_eigenvalues

    # Eigenvalue distribution comparison (per-component MSE)
    ev_mse = (gen_ev - ref_ev).square().mean().item()

    # Sort eigenvalues for distribution comparison
    ref_sorted = ref_ev.sort(dim=-1).values
    gen_sorted = gen_ev.sort(dim=-1).values
    ev_sorted_mse = (gen_sorted - ref_sorted).square().mean().item()

    # Orthogonality check on generated eigenvectors
    k = model.cfg.k
    ortho_errors = []
    for i in range(len(n_nodes)):
        n = int(n_nodes[i].item())
        U = gen_eigenvectors[i, :n, :]
        gram = U.T @ U
        eye = torch.eye(k, device=U.device, dtype=U.dtype)
        ortho_errors.append((gram - eye).square().sum().item())
    mean_ortho_error = float(np.mean(ortho_errors))

    # Eigenvalue range check (should be in [-1, 1] after normalisation)
    ev_min = gen_ev.min().item()
    ev_max = gen_ev.max().item()

    # Reconstruct pair channels and compare
    pair_mse_values = []
    for i in range(min(len(n_nodes), 50)):
        n = int(n_nodes[i].item())
        ref_U = ref_eigenvectors[i, :n, :].to(device)
        ref_lam = ref_eigenvalues[i].to(device)
        gen_U = gen_eigenvectors[i, :n, :]
        gen_lam = gen_eigenvalues[i]
        ref_pair = ref_U @ torch.diag(ref_lam + 1.0) @ ref_U.T
        gen_pair = gen_U @ torch.diag(gen_lam + 1.0) @ gen_U.T
        pair_mse_values.append((ref_pair - gen_pair).square().mean().item())
    pair_mse = float(np.mean(pair_mse_values))

    return {
        "eigenvalue_mse": ev_mse,
        "eigenvalue_sorted_mse": ev_sorted_mse,
        "ortho_error": mean_ortho_error,
        "pair_reconstruction_mse": pair_mse,
        "eigenvalue_range": [ev_min, ev_max],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="planar", choices=["planar", "sbm"])
    parser.add_argument("--band", default="low", choices=["low", "high", "random"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-hidden-layers", type=int, default=4)
    parser.add_argument("--ortho-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    splits = load_splits(args.dataset)
    train_graphs = splits["train"]
    val_graphs = splits["val"]
    n_max = max(
        max(g.number_of_nodes() for g in train_graphs),
        max(g.number_of_nodes() for g in val_graphs),
    )

    train_data = _build_spectral_dataset(
        train_graphs, args.band, args.k, n_max, args.seed, f"{args.dataset}_train",
    )
    val_data = _build_spectral_dataset(
        val_graphs, args.band, args.k, n_max, args.seed + 10_000, f"{args.dataset}_val",
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    cfg = SpectralPriorConfig(
        n_max=n_max,
        k=args.k,
        timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        ortho_penalty_weight=args.ortho_weight,
    )
    model = SpectralPrior(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"{args.dataset}: train={len(train_graphs)} val={len(val_graphs)} n_max={n_max}")
    print(f"band={args.band} k={args.k} state_dim={cfg.state_dim}")
    print(f"model={n_params/1e6:.2f}M hidden={args.hidden_dim} layers={args.n_hidden_layers}")
    print(f"timesteps={cfg.timesteps} ortho_weight={args.ortho_weight} amp={use_amp}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs, T_mult=1,
    )
    best = {"epoch": 0, "val_loss": float("inf")}
    history = []
    checkpoint_path = (
        checkpoints_dir()
        / f"spectral_prior_{args.dataset}_{args.band}_k{args.k}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            eigenvalues, eigenvectors, n_nodes = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                loss = model.training_loss(eigenvalues, eigenvectors, n_nodes)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(eigenvalues)
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            val_loss = _val_loss(model, val_loader, device)
            train_loss = loss_sum / len(train_loader.dataset)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
            print(f"epoch {epoch:4d} train={train_loss:.5f} val={val_loss:.5f}")
            if val_loss < best["val_loss"]:
                best = {"epoch": epoch, "val_loss": val_loss}
                torch.save({
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "args": vars(args),
                    "best": best,
                }, checkpoint_path)

    elapsed = (time.time() - started) / 60.0

    # Load best and evaluate
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ckpt["history"] = history
    ckpt["training_minutes"] = elapsed
    torch.save(ckpt, checkpoint_path)

    # Evaluate sample quality
    val_eigenvalues = val_data.tensors[0]
    val_eigenvectors = val_data.tensors[1]
    val_n_nodes = val_data.tensors[2]
    eval_metrics = _evaluate_samples(
        model, val_eigenvalues, val_eigenvectors, val_n_nodes, device,
    )

    print(f"\n--- Evaluation ---")
    print(f"eigenvalue MSE: {eval_metrics['eigenvalue_mse']:.6f}")
    print(f"eigenvalue sorted MSE: {eval_metrics['eigenvalue_sorted_mse']:.6f}")
    print(f"ortho error (post-QR): {eval_metrics['ortho_error']:.6f}")
    print(f"pair reconstruction MSE: {eval_metrics['pair_reconstruction_mse']:.6f}")
    print(f"eigenvalue range: {eval_metrics['eigenvalue_range']}")
    print(f"training time: {elapsed:.1f} min")
    print(f"best val_loss: {best['val_loss']:.5f} at epoch {best['epoch']}")

    # Save report
    results_dir().mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "band": args.band,
        "k": args.k,
        "config": asdict(cfg),
        "n_parameters": n_params,
        "best": best,
        "history": history,
        "evaluation": eval_metrics,
        "training_minutes": elapsed,
    }
    report_path = results_dir() / f"spectral_prior_{args.dataset}_{args.band}_k{args.k}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
