"""Stage 4 gate: train the graph autoencoder and verify it can reconstruct.

The autoencoder is the ceiling for everything downstream. If it cannot rebuild a graph from its
own latent, no diffusion result in that latent is interpretable, so this runs before any
generative model exists.

Gate (PLAN.md Stage 4 + WORKPLAN 3.2):
    edge accuracy > 99% and F1 >= 0.98 on the held-out split.

Accuracy alone is close to meaningless here - Planar at n=64 is ~9% positive pairs, so an
all-zeros predictor already scores ~91%. F1 is the binding constraint.

Also runs T2 from WORKPLAN 8: permutation equivariance of the encoder,
`f(P A P^T) == P f(A) P^T`. Without it the whole method is invalid.

Usage:
    python scripts/train_autoencoder.py --dataset planar
    python scripts/train_autoencoder.py --dataset sbm --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fald.data import load_splits
from fald.data.dense import DenseGraphDataset, collate_dense, graphs_to_dense
from fald.models.autoencoder import AutoencoderConfig, GraphAutoencoder
from fald.paths import checkpoints_dir, results_dir

ACCURACY_GATE = 0.99
F1_GATE = 0.98
PERM_TOL = 1e-4


def evaluate(model, loader, device, pos_weight) -> dict:
    model.eval()
    totals = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    loss_sum, true_pos_pairs, n_pairs, n_edges = 0.0, 0, 0, 0
    correct = 0
    tp = fp = fn = 0

    with torch.no_grad():
        for adjacency, node_mask in loader:
            adjacency, node_mask = adjacency.to(device), node_mask.to(device)
            logits, _, _ = model(adjacency, node_mask, jitter=False)
            loss_sum += model.loss(logits, adjacency, node_mask, pos_weight).item() * len(adjacency)

            # Accumulate confusion counts across batches rather than averaging per-batch F1,
            # which would over-weight small batches.
            from fald.data.dense import pair_mask_from_node_mask
            pair_mask = pair_mask_from_node_mask(node_mask)
            triu = torch.triu(torch.ones_like(pair_mask), diagonal=1).bool()
            mask = pair_mask & triu
            predicted = (logits > 0) & mask
            target = (adjacency > 0.5) & mask
            tp += (predicted & target).sum().item()
            fp += (predicted & ~target & mask).sum().item()
            fn += (~predicted & target).sum().item()
            correct += ((predicted == target) & mask).sum().item()
            n_pairs += mask.sum().item()
            n_edges += target.sum().item()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "loss": loss_sum / max(len(loader.dataset), 1),
        "accuracy": correct / n_pairs if n_pairs else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pairs": n_pairs,
        "n_edges": n_edges,
        "positive_rate": n_edges / n_pairs if n_pairs else 0.0,
    }


def check_permutation_equivariance(model, graphs, device, failures: list) -> dict:
    """T2: permuting the input must permute the latents identically."""
    model.eval()
    adjacency, node_mask = graphs_to_dense(graphs[:2])
    adjacency, node_mask = adjacency.to(device), node_mask.to(device)

    generator = torch.Generator(device="cpu").manual_seed(0)
    n = node_mask.shape[1]
    perm = torch.randperm(n, generator=generator).to(device)

    with torch.no_grad():
        logits, Z, W = model(adjacency, node_mask, jitter=False)
        permuted_adj = adjacency[:, perm][:, :, perm]
        permuted_mask = node_mask[:, perm]
        logits_p, Z_p, W_p = model(permuted_adj, permuted_mask, jitter=False)

        dZ = (Z_p - Z[:, perm]).abs().max().item()
        dW = (W_p - W[:, perm][:, :, perm]).abs().max().item()
        dL = (logits_p - logits[:, perm][:, :, perm]).abs().max().item()

    worst = max(dZ, dW, dL)
    if worst > PERM_TOL:
        failures.append(
            f"T2: encoder is not permutation equivariant (max deviation {worst:.2e} > {PERM_TOL:.0e}; "
            f"Z {dZ:.2e}, W {dW:.2e}, logits {dL:.2e})"
        )
    return {"delta_Z": dZ, "delta_W": dW, "delta_logits": dL}


@torch.no_grad()
def check_latent_quality(model, loader, device, failures: list) -> dict:
    """Is the latent actually diffusable, or a lookup table that happens to pass the gate?

    Three measurements, all cheap and all interpretable without a diffusion model:

    Latent statistics
        Does the latent look like something Gaussian diffusion could plausibly model.

    Noise robustness
        How far a sample can drift off the encoder manifold before reconstruction collapses.
        This is literally the slack Stage 5 gets to work with.

    Degeneracy
        For a per-pair latent, Cohen's d between the edge and non-edge clouds detects the
        two-point binary code (measured at 13302 on the `pair` decoder). That statistic is
        meaningless for a node latent, where the analogous failure is a rank collapse - all
        nodes mapping to the same point, or the latent using far fewer directions than `d_v`.
        Effective rank via the participation ratio of the singular values covers that case.
    """
    from fald.data.dense import pair_mask_from_node_mask

    model.eval()
    node_only = model.cfg.latent_is_node_only
    z_all, w_edge, w_non = [], [], []

    for adjacency, node_mask in loader:
        adjacency, node_mask = adjacency.to(device), node_mask.to(device)
        _, Z, W = model(adjacency, node_mask, jitter=False)
        pair_mask = pair_mask_from_node_mask(node_mask)
        is_edge = (adjacency > 0.5) & pair_mask
        z_all.append(Z[node_mask])
        if not node_only:
            w_edge.append(W[is_edge])
            w_non.append(W[pair_mask & ~is_edge])

    Z_cat = torch.cat(z_all)
    report = {
        "latent": "Z (node-only)" if node_only else "W (per-pair)",
        "Z_mean": Z_cat.mean().item(),
        "Z_std": Z_cat.std().item(),
        "Z_absmax": Z_cat.abs().max().item(),
    }

    # Participation ratio of the singular values: ~d_v means all directions carry signal, ~1
    # means the node latents collapsed onto a line.
    centred = Z_cat - Z_cat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centred.float())
    energy = singular ** 2
    effective_rank = (energy.sum() ** 2 / (energy ** 2).sum()).item()
    report["Z_effective_rank"] = effective_rank
    report["Z_dim"] = Z_cat.shape[1]

    separation = None
    if not node_only:
        W_edge, W_non = torch.cat(w_edge), torch.cat(w_non)
        pooled_std = torch.sqrt(0.5 * (W_edge.var(dim=0) + W_non.var(dim=0))).clamp(min=1e-6)
        separation = ((W_edge.mean(dim=0) - W_non.mean(dim=0)).abs() / pooled_std).mean().item()
        W_cat = torch.cat([W_edge, W_non])
        report.update({
            "W_mean": W_cat.mean().item(),
            "W_std": W_cat.std().item(),
            "W_absmax": W_cat.abs().max().item(),
            "edge_vs_nonedge_cohens_d": separation,
        })

    robustness = {}
    for sigma in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
        tp = fp = fn = 0
        for adjacency, node_mask in loader:
            adjacency, node_mask = adjacency.to(device), node_mask.to(device)
            _, Z, W = model(adjacency, node_mask, jitter=False)
            if sigma > 0:
                if node_only:
                    Z = Z + torch.randn_like(Z) * sigma
                else:
                    W = W + torch.randn_like(W) * sigma
            logits = model.decode(Z, W, node_mask)
            pair_mask = pair_mask_from_node_mask(node_mask)
            triu = torch.triu(torch.ones_like(pair_mask), diagonal=1).bool()
            mask = pair_mask & triu
            predicted = (logits > 0) & mask
            target = (adjacency > 0.5) & mask
            tp += (predicted & target).sum().item()
            fp += (predicted & ~target & mask).sum().item()
            fn += (~predicted & target).sum().item()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        robustness[f"sigma_{sigma}"] = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    report["f1_under_latent_noise"] = robustness

    if separation is not None and separation > 100.0:
        report["verdict"] = "degenerate-binary-code"
    elif effective_rank < 2.0:
        report["verdict"] = "rank-collapse"
        failures.append(
            f"node latent collapsed to effective rank {effective_rank:.2f} of {Z_cat.shape[1]}"
        )
    else:
        report["verdict"] = "ok"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="planar", choices=["planar", "sbm"])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-v", type=int, default=32)
    parser.add_argument("--d-e", type=int, default=8)
    parser.add_argument("--decoder", default="node_mlp", choices=["node_mlp", "node_dot", "pair"],
                        help="'pair' reproduces the collapsing WORKPLAN 3.2 design, as ablation")
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--node-random-dim", type=int, default=0,
                        help="diagnostic: random symmetry-breaking node channels")
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="up-weight positive pairs in BCE; default auto from edge density")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    failures: list = []

    splits = load_splits(args.dataset)
    train_graphs, eval_graphs = splits["train"], splits[args.eval_split]
    sizes = sorted({g.number_of_nodes() for g in train_graphs})
    print(f"{args.dataset}: train={len(train_graphs)} {args.eval_split}={len(eval_graphs)}")
    print(f"nodes per graph: {sizes[0]}..{sizes[-1]}")

    train_loader = DataLoader(
        DenseGraphDataset(train_graphs), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_dense, drop_last=False,
    )
    eval_loader = DataLoader(
        DenseGraphDataset(eval_graphs), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_dense,
    )

    cfg = AutoencoderConfig(
        n_layers=args.n_layers, d_v=args.d_v, d_e=args.d_e,
        decoder=args.decoder, jitter_sigma=args.jitter,
        node_random_dim=args.node_random_dim,
    )
    model = GraphAutoencoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params, decoder={cfg.decoder} "
          f"d_v={cfg.d_v} d_e={cfg.d_e} layers={cfg.n_layers}")
    if cfg.latent_is_node_only:
        n_pairs = sum(g.number_of_nodes() * (g.number_of_nodes() - 1) // 2 for g in eval_graphs)
        n_latent = sum(g.number_of_nodes() * cfg.d_v for g in eval_graphs)
        print(f"latent budget: {n_latent} numbers must determine {n_pairs} edge decisions "
              f"({n_latent/n_pairs:.2f}x)")

    # Positives are the minority class; without up-weighting, BCE happily under-predicts edges
    # and F1 stalls well below the gate even as accuracy looks fine.
    baseline = evaluate(model, eval_loader, device, None)
    rate = baseline["positive_rate"]
    if args.pos_weight is None:
        # Full inverse-frequency weighting. An earlier version used the square root, which left
        # the majority class dominant enough that the model settled on predicting it everywhere.
        args.pos_weight = max(1.0, (1 - rate) / max(rate, 1e-6))
    bias = model.init_output_bias(rate)
    print(f"edge density {rate:.4f} -> pos_weight {args.pos_weight:.2f}, "
          f"decoder bias initialised to {bias:.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    started = time.time()
    best = {"f1": -1.0}
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, seen = 0.0, 0
        for adjacency, node_mask in train_loader:
            adjacency, node_mask = adjacency.to(device), node_mask.to(device)
            logits, _, _ = model(adjacency, node_mask, jitter=True)
            loss = model.loss(logits, adjacency, node_mask, args.pos_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(adjacency)
            seen += len(adjacency)
        scheduler.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            stats = evaluate(model, eval_loader, device, args.pos_weight)
            history.append({"epoch": epoch, "train_loss": epoch_loss / seen, **stats})
            print(
                f"epoch {epoch:4d}  train {epoch_loss/seen:.4f}  "
                f"{args.eval_split} acc {stats['accuracy']:.5f}  f1 {stats['f1']:.5f}  "
                f"P {stats['precision']:.4f}  R {stats['recall']:.4f}"
            )
            if stats["f1"] > best["f1"]:
                best = {"epoch": epoch, **stats}
                checkpoints_dir().mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "cfg": cfg.__dict__, "metrics": stats},
                    checkpoints_dir() / f"autoencoder_{args.dataset}.pt",
                )

    elapsed = time.time() - started
    print(f"\ntrained in {elapsed/60:.1f} min")
    print(f"best {args.eval_split}: acc {best['accuracy']:.5f}  f1 {best['f1']:.5f} (epoch {best['epoch']})")

    print("\n[T2] permutation equivariance...")
    perm_report = check_permutation_equivariance(model, eval_graphs, device, failures)
    print(f"  max deviation: Z {perm_report['delta_Z']:.2e}  W {perm_report['delta_W']:.2e}  "
          f"logits {perm_report['delta_logits']:.2e}")

    print("\n[latent quality] is this diffusable, or a lookup table?")
    latent_report = check_latent_quality(model, eval_loader, device, failures)
    print(f"  diffused latent: {latent_report['latent']}")
    print(f"  Z: mean {latent_report['Z_mean']:+.3f} std {latent_report['Z_std']:.3f} "
          f"absmax {latent_report['Z_absmax']:.2f}")
    print(f"  Z effective rank: {latent_report['Z_effective_rank']:.2f} of "
          f"{latent_report['Z_dim']}")
    if "edge_vs_nonedge_cohens_d" in latent_report:
        print(f"  W: mean {latent_report['W_mean']:+.3f} std {latent_report['W_std']:.3f} "
              f"absmax {latent_report['W_absmax']:.2f}")
        print(f"  edge vs non-edge separation (Cohen's d): "
              f"{latent_report['edge_vs_nonedge_cohens_d']:.2f}")
    print("  F1 under latent noise:")
    for key, value in latent_report["f1_under_latent_noise"].items():
        print(f"    {key:>12}: {value:.4f}")
    print(f"  verdict: {latent_report['verdict']}")

    if best["accuracy"] <= ACCURACY_GATE:
        failures.append(f"edge accuracy {best['accuracy']:.5f} <= {ACCURACY_GATE}")
    if best["f1"] < F1_GATE:
        failures.append(f"F1 {best['f1']:.5f} < {F1_GATE}")

    report = {
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "config": cfg.__dict__,
        "n_params": n_params,
        "pos_weight": args.pos_weight,
        "epochs": args.epochs,
        "minutes": elapsed / 60,
        "best": best,
        "permutation": perm_report,
        "latent_quality": latent_report,
        "history": history,
        "failures": failures,
    }
    results_dir().mkdir(exist_ok=True)
    out = results_dir() / f"autoencoder_{args.dataset}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")

    print()
    if failures:
        print("GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"GATE PASSED: accuracy {best['accuracy']:.5f} > {ACCURACY_GATE} and "
          f"F1 {best['f1']:.5f} >= {F1_GATE}; encoder is permutation equivariant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
