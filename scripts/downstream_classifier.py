"""Downstream utility experiment: SBM community-count classification.

Stage 9 of PLAN.md. Trains a GNN classifier to predict the number of
communities (2/3/4/5) in SBM graphs. Compares three training regimes:
  (i)   real-only
  (ii)  real + unconditioned diffusion augmentation
  (iii) real + spectrally-conditioned diffusion augmentation

Tests whether spectral conditioning produces augmentation data that
preserves community structure better than unconditioned generation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_digress_dir = Path(__file__).resolve().parents[1] / "third_party" / "digress"
if _digress_dir.is_dir() and str(_digress_dir) not in sys.path:
    sys.path.insert(0, str(_digress_dir))

from fald.data.dense import graphs_to_dense
from fald.paths import results_dir


# ---------------------------------------------------------------------------
# SBM data generation with community-count labels
# ---------------------------------------------------------------------------

def generate_sbm_with_label(
    num_communities: int,
    rng: np.random.Generator,
    min_community_size: int = 20,
    max_community_size: int = 40,
    p_intra: float = 0.3,
    p_inter: float = 0.05,
) -> nx.Graph:
    sizes = [
        int(rng.integers(min_community_size, max_community_size + 1))
        for _ in range(num_communities)
    ]
    probs = np.full((num_communities, num_communities), p_inter)
    np.fill_diagonal(probs, p_intra)
    graph = nx.stochastic_block_model(
        sizes, probs.tolist(), seed=int(rng.integers(0, 2**31 - 1))
    )
    graph.graph["num_communities"] = num_communities
    return graph


def generate_labeled_sbm_dataset(
    num_per_class: int,
    seed: int = 0,
) -> list[nx.Graph]:
    rng = np.random.default_rng(seed)
    graphs = []
    for num_comm in [2, 3, 4, 5]:
        for _ in range(num_per_class):
            graphs.append(generate_sbm_with_label(num_comm, rng))
    rng.shuffle(graphs)
    return graphs


def get_label(graph: nx.Graph) -> int:
    return graph.graph["num_communities"] - 2


# ---------------------------------------------------------------------------
# Dense GIN classifier (no torch_geometric dependency)
# ---------------------------------------------------------------------------

class DenseGINLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), adj: (B, N, N), mask: (B, N)
        neighbor_sum = torch.bmm(adj, x)
        out = (1.0 + self.eps) * x + neighbor_sum
        B, N, D = out.shape
        out = out.reshape(B * N, D)
        out = self.mlp(out)
        out = out.reshape(B, N, D)
        return out * mask.unsqueeze(-1).float()


class DenseGINClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList([
            DenseGINLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Node feature: degree (normalized)
        degree = (adj * mask.unsqueeze(1).float()).sum(dim=2, keepdim=True)
        n_real = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        x = degree / n_real
        x = self.input_proj(x) * mask.unsqueeze(-1).float()

        for layer in self.layers:
            x = layer(x, adj, mask)

        # Mean pooling over real nodes
        x_sum = (x * mask.unsqueeze(-1).float()).sum(dim=1)
        x_mean = x_sum / mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.classifier(x_mean)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_classifier(
    model: DenseGINClassifier,
    train_adj: torch.Tensor,
    train_mask: torch.Tensor,
    train_labels: torch.Tensor,
    val_adj: torch.Tensor,
    val_mask: torch.Tensor,
    val_labels: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    n_train = len(train_labels)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            a = train_adj[idx].to(device)
            m = train_mask[idx].to(device)
            y = train_labels[idx].to(device)

            logits = model(a, m)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            val_acc = evaluate_classifier(model, val_adj, val_mask, val_labels, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_acc": best_val_acc}


@torch.no_grad()
def evaluate_classifier(
    model: DenseGINClassifier,
    adj: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 64,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        a = adj[start:end].to(device)
        m = mask[start:end].to(device)
        y = labels[start:end].to(device)
        logits = model(a, m)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Augmentation graph generation
# ---------------------------------------------------------------------------

def generate_random_sbm_augmentation(
    num_graphs: int,
    seed: int = 0,
) -> list[nx.Graph]:
    """Random SBM graphs as the 'unconditioned' augmentation baseline."""
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(num_graphs):
        num_comm = int(rng.integers(2, 6))
        graphs.append(generate_sbm_with_label(num_comm, rng))
    return graphs


def generate_diffusion_augmentation(
    model,
    template_graphs: list[nx.Graph],
    device: str = "cuda",
    band: str = "none",
    k: int = 32,
    guidance_scale: float = 1.5,
) -> list[nx.Graph]:
    """Sample from a trained adjacency diffusion model."""
    from fald.data import build_condition_tensors
    from fald.data.dense import graphs_to_dense

    adjacency, node_mask = graphs_to_dense(template_graphs)
    condition = build_condition_tensors(
        template_graphs, band=band, k=k, seed=42, cache_tag="augment",
    )

    model.eval()
    sampled = model.sample(
        node_mask.to(device),
        condition.pair.to(device),
        condition.eigenvalues.to(device),
        condition.present.to(device),
        guidance_scale=guidance_scale,
    )

    generated = []
    for matrix, mask in zip(sampled.cpu(), node_mask):
        n = int(mask.sum().item())
        g = nx.from_numpy_array(matrix[:n, :n].numpy().astype(np.uint8))
        # Infer community count from the template for labeling
        generated.append(g)
    return generated


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def prepare_tensors(graphs: list[nx.Graph]):
    adjacency, node_mask = graphs_to_dense(graphs)
    labels = torch.tensor([get_label(g) for g in graphs], dtype=torch.long)
    return adjacency, node_mask, labels


def run_experiment(
    train_sizes: list[int],
    n_seeds: int = 5,
    classifier_epochs: int = 100,
    device: str = "cuda",
    diffusion_model_none=None,
    diffusion_model_cond=None,
    cond_band: str = "high",
    cond_k: int = 32,
    guidance_scale: float = 1.5,
) -> dict:
    results = {}

    # Generate test/val data (fixed across all runs)
    test_graphs = generate_labeled_sbm_dataset(num_per_class=25, seed=9999)
    val_graphs = generate_labeled_sbm_dataset(num_per_class=15, seed=8888)
    test_adj, test_mask, test_labels = prepare_tensors(test_graphs)
    val_adj, val_mask, val_labels = prepare_tensors(val_graphs)

    for n_real in train_sizes:
        results[f"n{n_real}"] = {}
        for arm_name in ["real_only", "real_plus_random", "real_plus_unconditioned", "real_plus_conditioned"]:
            accuracies = []

            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Generate real training data
                n_per_class = max(1, n_real // 4)
                train_real = generate_labeled_sbm_dataset(
                    num_per_class=n_per_class, seed=seed * 1000,
                )

                if arm_name == "real_only":
                    train_graphs = train_real
                elif arm_name == "real_plus_random":
                    aug = generate_random_sbm_augmentation(
                        num_graphs=len(train_real), seed=seed * 1000 + 500,
                    )
                    train_graphs = train_real + aug
                elif arm_name == "real_plus_unconditioned":
                    if diffusion_model_none is None:
                        continue
                    aug = generate_diffusion_augmentation(
                        diffusion_model_none,
                        train_real,
                        device=device,
                        band="none",
                        k=cond_k,
                    )
                    for g, ref in zip(aug, train_real):
                        g.graph["num_communities"] = ref.graph["num_communities"]
                    train_graphs = train_real + aug
                elif arm_name == "real_plus_conditioned":
                    if diffusion_model_cond is None:
                        continue
                    aug = generate_diffusion_augmentation(
                        diffusion_model_cond,
                        train_real,
                        device=device,
                        band=cond_band,
                        k=cond_k,
                        guidance_scale=guidance_scale,
                    )
                    for g, ref in zip(aug, train_real):
                        g.graph["num_communities"] = ref.graph["num_communities"]
                    train_graphs = train_real + aug

                train_adj, train_mask, train_labels = prepare_tensors(train_graphs)

                model = DenseGINClassifier(
                    num_classes=4, hidden_dim=64, num_layers=3, dropout=0.3,
                )
                train_classifier(
                    model,
                    train_adj, train_mask, train_labels,
                    val_adj, val_mask, val_labels,
                    epochs=classifier_epochs,
                    device=device,
                )
                acc = evaluate_classifier(model, test_adj, test_mask, test_labels, device)
                accuracies.append(acc)
                print(f"  n={n_real} {arm_name} seed={seed}: test_acc={acc:.4f}")

            if accuracies:
                results[f"n{n_real}"][arm_name] = {
                    "accuracies": accuracies,
                    "mean": float(np.mean(accuracies)),
                    "std": float(np.std(accuracies)),
                    "stderr": float(np.std(accuracies) / np.sqrt(len(accuracies))),
                }
                print(
                    f"  n={n_real} {arm_name}: "
                    f"{np.mean(accuracies):.4f} ± {np.std(accuracies) / np.sqrt(len(accuracies)):.4f}"
                )

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[20, 40, 80])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--diffusion-checkpoint-none", type=str, default=None,
                        help="Checkpoint for unconditioned diffusion model")
    parser.add_argument("--diffusion-checkpoint-cond", type=str, default=None,
                        help="Checkpoint for conditioned diffusion model")
    parser.add_argument("--cond-band", default="high", choices=["low", "high"])
    parser.add_argument("--cond-k", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load diffusion models if checkpoints provided
    diffusion_none = None
    diffusion_cond = None

    if args.diffusion_checkpoint_none:
        from fald.models import AdjacencyDiffusion, AdjacencyDiffusionConfig
        from dataclasses import asdict
        ckpt = torch.load(args.diffusion_checkpoint_none, map_location=device, weights_only=False)
        cfg = AdjacencyDiffusionConfig(**ckpt["config"])
        diffusion_none = AdjacencyDiffusion(cfg).to(device)
        diffusion_none.load_state_dict(ckpt["model"])
        print(f"Loaded unconditioned diffusion from {args.diffusion_checkpoint_none}")

    if args.diffusion_checkpoint_cond:
        from fald.models import AdjacencyDiffusion, AdjacencyDiffusionConfig
        ckpt = torch.load(args.diffusion_checkpoint_cond, map_location=device, weights_only=False)
        cfg = AdjacencyDiffusionConfig(**ckpt["config"])
        diffusion_cond = AdjacencyDiffusion(cfg).to(device)
        diffusion_cond.load_state_dict(ckpt["model"])
        print(f"Loaded conditioned diffusion from {args.diffusion_checkpoint_cond}")

    print(f"\n{'='*60}")
    print("Stage 9: Downstream SBM Community-Count Classification")
    print(f"{'='*60}")
    print(f"Train sizes: {args.train_sizes}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Classifier epochs: {args.classifier_epochs}")
    print(f"Arms: real_only, real_plus_random"
          + (", real_plus_unconditioned" if diffusion_none else "")
          + (", real_plus_conditioned" if diffusion_cond else ""))
    print()

    started = time.time()
    results = run_experiment(
        train_sizes=args.train_sizes,
        n_seeds=args.n_seeds,
        classifier_epochs=args.classifier_epochs,
        device=str(device),
        diffusion_model_none=diffusion_none,
        diffusion_model_cond=diffusion_cond,
        cond_band=args.cond_band,
        cond_k=args.cond_k,
        guidance_scale=args.guidance_scale,
    )
    elapsed = time.time() - started

    # Summary table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for n_key in sorted(results.keys()):
        print(f"\n  {n_key}:")
        for arm, stats in sorted(results[n_key].items()):
            print(f"    {arm:30s} {stats['mean']:.4f} ± {stats['stderr']:.4f}")

    # Save results
    results_dir().mkdir(parents=True, exist_ok=True)
    report = {
        "stage": 9,
        "experiment": "downstream_sbm_community_count",
        "description": "GNN classifier accuracy on 4-way SBM community-count task",
        "train_sizes": args.train_sizes,
        "n_seeds": args.n_seeds,
        "classifier_epochs": args.classifier_epochs,
        "cond_band": args.cond_band,
        "cond_k": args.cond_k,
        "guidance_scale": args.guidance_scale,
        "has_diffusion_unconditioned": diffusion_none is not None,
        "has_diffusion_conditioned": diffusion_cond is not None,
        "results": results,
        "elapsed_minutes": elapsed / 60.0,
    }
    report_path = results_dir() / "stage9_downstream.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")
    print(f"Elapsed: {elapsed / 60:.1f} minutes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
