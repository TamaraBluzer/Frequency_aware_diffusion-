"""Graph autoencoder.

Two decoder families, selected by `AutoencoderConfig.decoder`:

`node_mlp` / `node_dot` (default)
    Latent is **node-only**, `Z` in `R^(n x d_v)`; each edge is predicted from its two endpoint
    latents. This is LGD's task (iii) and matches what their code actually does
    (`edge_decoding: dot`).

`pair`
    Latent includes the per-pair tensor `W` in `R^(n x n x d_e)` and each edge is read from its
    own slot `W_ij`. This is LGD task (ii), and it is what WORKPLAN 3.2 originally specified.
    **It collapses**, and is kept only as a measurable ablation.

Why `pair` collapses. Each pair gets a private lane `A_ij -> W_ij -> A_hat_ij` that never mixes
with any other pair, there are as many pair slots out as pairs in, and the transformer's
residual connections pass the input edge indicator straight through. Copying the bit is
therefore a global optimum reachable by the shortest gradient path. Measured on Planar: perfect
reconstruction with a Cohen's d of 13302 between the edge and non-edge latent clouds (a "large"
effect is 0.8) and within-class spread of ~1.6e-4 - a two-point binary code, not a manifold, so
Gaussian diffusion in it would have nowhere to land.

Why node-only cannot collapse. `W_ij` served exactly one pair, so storing that pair's answer was
free. `z_i` must serve all `n-1` pairs involving node `i`, and 63 independent bits do not fit in
`d_v` numbers. On Planar, 64x16 = 1024 numbers must determine 2016 edges, so a lookup table is
arithmetically unavailable at any setting of the weights. The bottleneck is structural rather
than a penalty to be tuned.

Deviation from LGD worth stating in the report: LGD keeps `W` in the latent *and* trains it, via
tasks (iv)/(v) which reconstruct node and endpoint features from edge representations. Our graphs
are topology-only with constant node features, so those tasks are vacuous and `W` would receive
no signal that is not copyable. An unsupervised tensor is worse than no tensor, so we drop it.

Backbone is DiGress's `XEyTransformerLayer`, reused rather than reimplemented: an
already-debugged permutation-equivariant block that updates node, pair and global features
jointly. Note that pair features still exist *inside* the encoder - they are what make the
architecture expressive - they are simply no longer read out as latent.

Regularization is LayerNorm on the encoder outputs with **no KL term**. LGD applies KL-reg or
VQ-reg to stop latents having "arbitrarily high variance", which is the opposite of the failure
above, so it is not the relevant lever here.

Note on evaluation: edge-level *accuracy* is a weak metric. Planar at n=64 is ~9% positive
pairs, so predicting "no edge" everywhere already scores ~91%. F1 is the binding constraint,
which is why the gate requires both.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dense import pair_mask_from_node_mask

__all__ = ["AutoencoderConfig", "GraphAutoencoder", "ReconstructionMetrics"]


def _import_digress_layer():
    """DiGress lives in the gitignored work dir, so fail with an actionable message."""
    try:
        from src.models.transformer_model import XEyTransformerLayer
    except ImportError as exc:
        raise ImportError(
            "Could not import DiGress's XEyTransformerLayer. Run scripts/setup_digress.sh and "
            "`pip install -e $FALD_WORK_DIR/third_party/digress`."
        ) from exc
    return XEyTransformerLayer


DECODERS = ("node_mlp", "node_dot", "pair")


@dataclass
class AutoencoderConfig:
    d_v: int = 32
    d_e: int = 8
    n_layers: int = 4
    dx: int = 128
    de: int = 64
    dy: int = 64
    n_head: int = 8
    dim_ffX: int = 256
    dim_ffE: int = 128
    dim_ffy: int = 128
    dropout: float = 0.0
    decoder: str = "node_mlp"
    decoder_hidden: int = 128
    jitter_sigma: float = 0.05
    # Symmetry-breaking node inputs. A permutation-equivariant encoder must give structurally
    # similar nodes similar latents, so on featureless near-regular graphs (planar: degrees
    # 3-10) every node latent collapses to one point and a node-only decoder can only emit a
    # constant. Resampled every forward pass, so the model stays equivariant in distribution.
    # Diagnostic only: random IDs carry no structural information and cannot be reproduced at
    # sampling time, so this is not a usable production setting.
    node_random_dim: int = 0

    @property
    def latent_is_node_only(self) -> bool:
        return self.decoder.startswith("node")


@dataclass
class ReconstructionMetrics:
    loss: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    n_pairs: int = 0

    def as_dict(self) -> dict:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "n_pairs": self.n_pairs,
        }


class GraphEncoder(nn.Module):
    """`A -> (Z, W)`. Inputs are derived only from topology; there are no node features."""

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        layer_cls = _import_digress_layer()
        self.cfg = cfg

        # Node input is [1, degree/n]: constant plus a permutation-equivariant local statistic,
        # optionally plus random symmetry-breaking channels (see AutoencoderConfig).
        self.in_X = nn.Linear(2 + cfg.node_random_dim, cfg.dx)
        # Pair input is the one-hot of edge/no-edge, matching DiGress's E encoding.
        self.in_E = nn.Linear(2, cfg.de)
        self.in_y = nn.Linear(1, cfg.dy)

        self.layers = nn.ModuleList([
            layer_cls(
                dx=cfg.dx, de=cfg.de, dy=cfg.dy, n_head=cfg.n_head,
                dim_ffX=cfg.dim_ffX, dim_ffE=cfg.dim_ffE, dim_ffy=cfg.dim_ffy,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])

        self.out_Z = nn.Linear(cfg.dx, cfg.d_v)
        self.out_W = nn.Linear(cfg.de, cfg.d_e)
        self.norm_Z = nn.LayerNorm(cfg.d_v)
        self.norm_W = nn.LayerNorm(cfg.d_e)

    def forward(self, adjacency: torch.Tensor, node_mask: torch.Tensor):
        batch, n = node_mask.shape
        float_mask = node_mask.to(adjacency.dtype)
        pair_mask = pair_mask_from_node_mask(node_mask, include_diagonal=True)
        pair_float = pair_mask.to(adjacency.dtype)

        n_real = float_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        degree = (adjacency * pair_float).sum(dim=2) / n_real
        X = torch.stack([float_mask, degree], dim=-1)
        if self.cfg.node_random_dim > 0:
            noise = torch.randn(
                batch, n, self.cfg.node_random_dim, device=X.device, dtype=X.dtype
            )
            X = torch.cat([X, noise * float_mask.unsqueeze(-1)], dim=-1)
        E = torch.stack([1.0 - adjacency, adjacency], dim=-1) * pair_float.unsqueeze(-1)
        y = (n_real / n)

        X = self.in_X(X) * float_mask.unsqueeze(-1)
        E = self.in_E(E) * pair_float.unsqueeze(-1)
        y = self.in_y(y)

        for layer in self.layers:
            X, E, y = layer(X, E, y, node_mask)
            X = X * float_mask.unsqueeze(-1)
            E = E * pair_float.unsqueeze(-1)

        Z = self.norm_Z(self.out_Z(X)) * float_mask.unsqueeze(-1)
        W = self.norm_W(self.out_W(E)) * pair_float.unsqueeze(-1)
        return Z, W


class PairDecoder(nn.Module):
    """`W -> logits`. Symmetrised before decoding, diagonal zeroed."""

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_e, cfg.decoder_hidden),
            nn.SiLU(),
            nn.Linear(cfg.decoder_hidden, 1),
        )

    def forward(self, W: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        # Symmetrising the latent rather than the logits guarantees an undirected prediction
        # exactly, instead of leaving it to the decoder to learn.
        symmetric = 0.5 * (W + W.transpose(1, 2))
        logits = self.net(symmetric).squeeze(-1)

        n = node_mask.shape[1]
        eye = torch.eye(n, dtype=torch.bool, device=logits.device).unsqueeze(0)
        return logits.masked_fill(eye, 0.0)


class NodePairDecoder(nn.Module):
    """`(z_i, z_j) -> logit_ij`. LGD task (iii); the only non-vacuous cross-task for us.

    Both parameterizations are symmetric in `(i, j)` by construction, so undirectedness is exact
    rather than something the decoder has to learn:

    `node_dot`  `<z_i, z_j>` plus a learned scale and bias. LGD's `edge_decoding: dot`. Cheap,
                but the resulting logit matrix has rank at most `d_v`, which may be too weak for
                planar triangulations - hence the ablation.
    `node_mlp`  `MLP([z_i + z_j, z_i * z_j])`. Symmetric because both sum and elementwise
                product are, and strictly more expressive than the dot product.
    """

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.mode = cfg.decoder
        if self.mode == "node_dot":
            self.scale = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))
            self.project = nn.Linear(cfg.d_v, cfg.d_v)
        else:
            self.net = nn.Sequential(
                nn.Linear(2 * cfg.d_v, cfg.decoder_hidden),
                nn.SiLU(),
                nn.Linear(cfg.decoder_hidden, cfg.decoder_hidden),
                nn.SiLU(),
                nn.Linear(cfg.decoder_hidden, 1),
            )

    def forward(self, Z: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        if self.mode == "node_dot":
            projected = self.project(Z)
            logits = torch.einsum("bid,bjd->bij", projected, projected) * self.scale + self.bias
        else:
            pairwise_sum = Z.unsqueeze(2) + Z.unsqueeze(1)
            pairwise_prod = Z.unsqueeze(2) * Z.unsqueeze(1)
            logits = self.net(torch.cat([pairwise_sum, pairwise_prod], dim=-1)).squeeze(-1)

        n = node_mask.shape[1]
        eye = torch.eye(n, dtype=torch.bool, device=logits.device).unsqueeze(0)
        return logits.masked_fill(eye, 0.0)


class GraphAutoencoder(nn.Module):
    def __init__(self, cfg: AutoencoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or AutoencoderConfig()
        if self.cfg.decoder not in DECODERS:
            raise ValueError(f"unknown decoder {self.cfg.decoder!r}; expected one of {DECODERS}")
        self.encoder = GraphEncoder(self.cfg)
        self.decoder = (
            NodePairDecoder(self.cfg) if self.cfg.latent_is_node_only else PairDecoder(self.cfg)
        )

    @torch.no_grad()
    def init_output_bias(self, edge_density: float) -> float:
        """Start the decoder at the base rate so gradients go to the input-dependent part.

        Edges are ~9% of pairs. With a zero-initialised bias the fastest early loss reduction is
        simply to push the bias toward the base rate, and the model reaches that flat region
        within ~50 steps and stalls there with vanishing gradients (measured: |grad| 1.3 -> 1e-5,
        F1 0). Initialising the bias to logit(density) hands it that solution for free.
        """
        logit = float(torch.logit(torch.tensor(edge_density).clamp(1e-4, 1 - 1e-4)))
        if isinstance(self.decoder, NodePairDecoder) and self.decoder.mode == "node_dot":
            self.decoder.bias.fill_(logit)
        else:
            final = [m for m in self.decoder.net if isinstance(m, nn.Linear)][-1]
            final.bias.fill_(logit)
            # Small final weights keep the initial output close to the constant while leaving a
            # usable gradient signal, rather than starting from large random logits.
            final.weight.mul_(0.1)
        return logit

    def decode(self, Z: torch.Tensor, W: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        latent = Z if self.cfg.latent_is_node_only else W
        return self.decoder(latent, node_mask)

    def forward(self, adjacency: torch.Tensor, node_mask: torch.Tensor, jitter: bool = False):
        Z, W = self.encoder(adjacency, node_mask)
        if jitter and self.cfg.jitter_sigma > 0:
            # Jitter only what is actually diffused. Perturbing an unused tensor is a no-op that
            # looks like regularization.
            noise = torch.randn_like(Z) * self.cfg.jitter_sigma
            if self.cfg.latent_is_node_only:
                Z = Z + noise
            else:
                Z = Z + noise
                W = W + torch.randn_like(W) * self.cfg.jitter_sigma
        logits = self.decode(Z, W, node_mask)
        return logits, Z, W

    @staticmethod
    def loss(
        logits: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        pos_weight: float | None = None,
    ) -> torch.Tensor:
        """Masked BCE over the strict upper triangle.

        Scoring only `i < j` avoids counting every undirected edge twice, which would otherwise
        halve the effective weight of the diagonal-free mask and make the reported loss
        incomparable to the per-pair metrics.
        """
        pair_mask = pair_mask_from_node_mask(node_mask)
        triu = torch.triu(torch.ones_like(pair_mask), diagonal=1).bool()
        mask = pair_mask & triu

        weight = None
        if pos_weight is not None:
            weight = torch.full_like(logits, 1.0)
            weight = torch.where(adjacency > 0.5, weight * pos_weight, weight)

        per_pair = F.binary_cross_entropy_with_logits(
            logits, adjacency, weight=weight, reduction="none"
        )
        return (per_pair * mask).sum() / mask.sum().clamp(min=1)

    @staticmethod
    @torch.no_grad()
    def metrics(logits: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor) -> dict:
        pair_mask = pair_mask_from_node_mask(node_mask)
        triu = torch.triu(torch.ones_like(pair_mask), diagonal=1).bool()
        mask = pair_mask & triu

        predicted = (logits > 0) & mask
        target = (adjacency > 0.5) & mask

        true_pos = (predicted & target).sum().item()
        false_pos = (predicted & ~target & mask).sum().item()
        false_neg = (~predicted & target).sum().item()
        total = mask.sum().item()
        correct = ((predicted == target) & mask).sum().item()

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "accuracy": correct / total if total else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_pairs": total,
            "n_edges": true_pos + false_neg,
        }
