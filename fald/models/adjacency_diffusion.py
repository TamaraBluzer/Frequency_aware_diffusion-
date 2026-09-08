"""Continuous diffusion directly on dense graph adjacency.

This is the Tier-0 model: the diffusion state is the graph itself rather than an
autoencoder latent.  It keeps the scientific intervention narrow--every arm
diffuses the same adjacency tensor, and only the explicit spectral condition
changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dense import pair_mask_from_node_mask

__all__ = [
    "AdjacencyDiffusionConfig",
    "AdjacencyDenoiser",
    "AdjacencyDiffusion",
    "adjacency_to_diffusion_state",
    "diffusion_state_to_adjacency",
    "sample_symmetric_noise",
]


def _import_digress_layer():
    import importlib, importlib.util, sys, types
    from ..paths import third_party_dir
    digress_src = third_party_dir() / "digress" / "src"
    target = digress_src / "models" / "transformer_model.py"
    if not target.is_file():
        raise ImportError(
            "Could not find DiGress transformer_model.py at " + str(target)
        )
    # Temporarily override sys.modules['src'] so that 'from src import X'
    # inside transformer_model.py resolves to DiGress's src, not any other.
    old_src = sys.modules.get('src')
    src_pkg = types.ModuleType('src')
    src_pkg.__path__ = [str(digress_src)]
    src_pkg.__file__ = str(digress_src / '__init__.py')
    sys.modules['src'] = src_pkg
    try:
        spec = importlib.util.spec_from_file_location(
            "src.models.transformer_model", str(target),
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod.XEyTransformerLayer
    finally:
        if old_src is not None:
            sys.modules['src'] = old_src
        elif 'src' in sys.modules:
            del sys.modules['src']


def _masked_pair(
    values: torch.Tensor,
    node_mask: torch.Tensor,
    include_diagonal: bool = False,
) -> torch.Tensor:
    pair_mask = pair_mask_from_node_mask(node_mask, include_diagonal=include_diagonal)
    return values * pair_mask.to(values.dtype)


def adjacency_to_diffusion_state(
    adjacency: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Map binary adjacency to `{-1, +1}` on valid off-diagonal pairs."""
    state = adjacency.mul(2.0).sub(1.0)
    return _masked_pair(state, node_mask)


def diffusion_state_to_adjacency(
    state: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Threshold a symmetric diffusion state into a simple undirected graph."""
    adjacency = (state > 0).to(state.dtype)
    upper = torch.triu(adjacency, diagonal=1)
    adjacency = upper + upper.transpose(1, 2)
    return _masked_pair(adjacency, node_mask)


def sample_symmetric_noise(
    shape: torch.Size | tuple[int, int, int],
    node_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Independent standard-normal noise for each undirected node pair."""
    raw = torch.randn(
        shape,
        generator=generator,
        dtype=dtype or torch.float32,
        device=device or node_mask.device,
    )
    upper = torch.triu(raw, diagonal=1)
    noise = upper + upper.transpose(1, 2)
    return _masked_pair(noise, node_mask)


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return t
    denominator = max(half - 1, 1)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / denominator
    )
    angles = t * 1_000.0 * frequencies.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


def _cycle_features(
    adjacency: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DiGress-style node/global counts for cycles of length three through six."""
    adjacency = adjacency * pair_mask_from_node_mask(node_mask).to(adjacency.dtype)
    degree = adjacency.sum(dim=-1)
    a2 = adjacency @ adjacency
    a3 = a2 @ adjacency
    a4 = a3 @ adjacency
    a5 = a4 @ adjacency
    a6 = a5 @ adjacency

    diagonal_a2 = torch.diagonal(a2, dim1=-2, dim2=-1)
    diagonal_a3 = torch.diagonal(a3, dim1=-2, dim2=-1)
    diagonal_a4 = torch.diagonal(a4, dim1=-2, dim2=-1)
    diagonal_a5 = torch.diagonal(a5, dim1=-2, dim2=-1)

    cycle3_node = diagonal_a3 / 2.0
    cycle4_node = (
        diagonal_a4
        - degree * (degree - 1.0)
        - (adjacency @ degree.unsqueeze(-1)).squeeze(-1)
    ) / 2.0
    cycle5_node = (
        diagonal_a5
        - 2.0 * diagonal_a3 * degree
        - (adjacency @ diagonal_a3.unsqueeze(-1)).squeeze(-1)
        + diagonal_a3
    ) / 2.0

    trace = lambda matrix: torch.diagonal(matrix, dim1=-2, dim2=-1).sum(dim=-1)
    cycle6_global = (
        trace(a6)
        - 3.0 * trace(a3.square())
        + 9.0 * torch.sum(adjacency * a2.square(), dim=(-2, -1))
        - 6.0 * torch.sum(diagonal_a2 * diagonal_a4, dim=-1)
        + 6.0 * trace(a4)
        - 4.0 * trace(a3)
        + 4.0 * torch.sum(diagonal_a2.pow(3), dim=-1)
        + 3.0 * torch.sum(a3, dim=(-2, -1))
        - 12.0 * torch.sum(diagonal_a2.square(), dim=-1)
        + 4.0 * trace(a2)
    ) / 12.0

    node = torch.stack([cycle3_node, cycle4_node, cycle5_node], dim=-1)
    node = (node / 10.0).clamp(min=0.0, max=1.0)
    node = node * node_mask.unsqueeze(-1).to(node.dtype)
    global_features = torch.stack(
        [
            cycle3_node.sum(dim=-1) / 3.0,
            cycle4_node.sum(dim=-1) / 4.0,
            cycle5_node.sum(dim=-1) / 5.0,
            cycle6_global,
        ],
        dim=-1,
    )
    global_features = (global_features / 10.0).clamp(min=0.0, max=1.0)
    return node, global_features


@dataclass
class AdjacencyDiffusionConfig:
    timesteps: int = 200
    condition_k: int = 8
    condition_drop_probability: float = 0.1
    time_dim: int = 64
    n_layers: int = 4
    dx: int = 128
    de: int = 64
    dy: int = 128
    n_head: int = 8
    dim_ffX: int = 256
    dim_ffE: int = 128
    dim_ffy: int = 256
    dropout: float = 0.0
    cosine_offset: float = 0.008
    use_cycle_features: bool = False


class AdjacencyDenoiser(nn.Module):
    """Permutation-equivariant `x_t -> x_0` predictor with an optional condition."""

    def __init__(self, cfg: AdjacencyDiffusionConfig):
        super().__init__()
        layer_cls = _import_digress_layer()
        self.cfg = cfg

        self.in_X = nn.Linear(2 + (3 if cfg.use_cycle_features else 0), cfg.dx)
        self.in_E = nn.Linear(3, cfg.de)
        self.in_y = nn.Linear(
            cfg.time_dim + cfg.condition_k + 1 + (4 if cfg.use_cycle_features else 0),
            cfg.dy,
        )

        self.layers = nn.ModuleList(
            [
                layer_cls(
                    dx=cfg.dx,
                    de=cfg.de,
                    dy=cfg.dy,
                    n_head=cfg.n_head,
                    dim_ffX=cfg.dim_ffX,
                    dim_ffE=cfg.dim_ffE,
                    dim_ffy=cfg.dim_ffy,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.out_E = nn.Sequential(
            nn.Linear(cfg.de, cfg.de),
            nn.SiLU(),
            nn.Linear(cfg.de, 1),
        )

    def forward(
        self,
        noisy_adjacency: torch.Tensor,
        time: torch.Tensor,
        node_mask: torch.Tensor,
        pair_condition: torch.Tensor | None = None,
        eigenvalue_condition: torch.Tensor | None = None,
        condition_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, n = node_mask.shape
        dtype = noisy_adjacency.dtype
        device = noisy_adjacency.device
        pair_mask = pair_mask_from_node_mask(node_mask, include_diagonal=True)
        pair_float = pair_mask.to(dtype)
        node_float = node_mask.to(dtype)

        if pair_condition is None:
            pair_condition = torch.zeros_like(noisy_adjacency)
        if eigenvalue_condition is None:
            eigenvalue_condition = torch.zeros(
                batch, self.cfg.condition_k, dtype=dtype, device=device
            )
        if condition_present is None:
            condition_present = torch.zeros(batch, 1, dtype=dtype, device=device)
        condition_present = condition_present.to(dtype)

        n_real = node_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        noisy_degree = (noisy_adjacency * pair_float).sum(dim=2) / n_real
        X = torch.stack([node_float, noisy_degree], dim=-1)
        global_cycles = None
        if self.cfg.use_cycle_features:
            discrete_adjacency = (noisy_adjacency > 0).to(dtype)
            node_cycles, global_cycles = _cycle_features(discrete_adjacency, node_mask)
            X = torch.cat([X, node_cycles], dim=-1)

        present_on_pairs = condition_present[:, None, :].expand(batch, n, n)
        E = torch.stack(
            [noisy_adjacency, pair_condition, present_on_pairs],
            dim=-1,
        )
        E = E * pair_float.unsqueeze(-1)

        time_embedding = _sinusoidal_time_embedding(time.reshape(batch, 1), self.cfg.time_dim)
        y = torch.cat([time_embedding, eigenvalue_condition, condition_present], dim=-1)
        if global_cycles is not None:
            y = torch.cat([y, global_cycles], dim=-1)

        X = self.in_X(X) * node_float.unsqueeze(-1)
        E = self.in_E(E) * pair_float.unsqueeze(-1)
        y = self.in_y(y)

        for layer in self.layers:
            X, E, y = layer(X, E, y, node_mask)
            X = X * node_float.unsqueeze(-1)
            E = E * pair_float.unsqueeze(-1)

        prediction = self.out_E(E).squeeze(-1)
        prediction = 0.5 * (prediction + prediction.transpose(1, 2))
        return _masked_pair(prediction, node_mask)


class AdjacencyDiffusion(nn.Module):
    """Cosine-schedule DDPM with direct `x_0` prediction."""

    def __init__(self, cfg: AdjacencyDiffusionConfig | None = None):
        super().__init__()
        self.cfg = cfg or AdjacencyDiffusionConfig()
        self.denoiser = AdjacencyDenoiser(self.cfg)

        steps = torch.arange(self.cfg.timesteps + 1, dtype=torch.float64)
        phase = (steps / self.cfg.timesteps + self.cfg.cosine_offset)
        phase = phase / (1.0 + self.cfg.cosine_offset)
        target_alpha_bar = torch.cos(phase * math.pi / 2.0).square()
        target_alpha_bar = target_alpha_bar / target_alpha_bar[0]

        betas = 1.0 - target_alpha_bar[1:] / target_alpha_bar[:-1]
        betas = betas.clamp(min=1e-8, max=0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cat(
            [torch.ones(1, dtype=torch.float64), torch.cumprod(alphas, dim=0)]
        )

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bars", alpha_bars.float())

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        extracted = values.gather(0, t)
        return extracted.reshape(t.shape[0], *([1] * (ndim - 1)))

    def q_sample(
        self,
        clean_state: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = sample_symmetric_noise(
                clean_state.shape,
                node_mask,
                dtype=clean_state.dtype,
                device=clean_state.device,
            )
        alpha_bar = self._extract(self.alpha_bars, t, clean_state.ndim)
        noisy = alpha_bar.sqrt() * clean_state + (1.0 - alpha_bar).sqrt() * noise
        return _masked_pair(noisy, node_mask), noise

    def _drop_condition(
        self,
        pair_condition: torch.Tensor,
        eigenvalue_condition: torch.Tensor,
        condition_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.cfg.condition_drop_probability <= 0:
            return pair_condition, eigenvalue_condition, condition_present
        keep = (
            torch.rand_like(condition_present) >= self.cfg.condition_drop_probability
        ).to(condition_present.dtype)
        present = condition_present * keep
        return (
            pair_condition * present[:, None, :],
            eigenvalue_condition * present,
            present,
        )

    def training_loss(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        pair_condition: torch.Tensor | None = None,
        eigenvalue_condition: torch.Tensor | None = None,
        condition_present: torch.Tensor | None = None,
        positive_weight: float = 1.0,
    ) -> torch.Tensor:
        batch = adjacency.shape[0]
        clean_state = adjacency_to_diffusion_state(adjacency, node_mask)
        t = torch.randint(1, self.cfg.timesteps + 1, (batch,), device=adjacency.device)
        noisy_state, _ = self.q_sample(clean_state, t, node_mask)

        if pair_condition is None:
            pair_condition = torch.zeros_like(adjacency)
        if eigenvalue_condition is None:
            eigenvalue_condition = torch.zeros(
                batch,
                self.cfg.condition_k,
                dtype=adjacency.dtype,
                device=adjacency.device,
            )
        if condition_present is None:
            condition_present = torch.zeros(
                batch, 1, dtype=adjacency.dtype, device=adjacency.device
            )
        pair_condition, eigenvalue_condition, condition_present = self._drop_condition(
            pair_condition,
            eigenvalue_condition,
            condition_present,
        )

        prediction = self.denoiser(
            noisy_state,
            t.to(adjacency.dtype).unsqueeze(1) / self.cfg.timesteps,
            node_mask,
            pair_condition,
            eigenvalue_condition,
            condition_present,
        )

        pair_mask = pair_mask_from_node_mask(node_mask)
        upper_mask = pair_mask & torch.triu(
            torch.ones_like(pair_mask), diagonal=1
        ).bool()
        weights = torch.where(
            adjacency > 0.5,
            torch.as_tensor(positive_weight, dtype=adjacency.dtype, device=adjacency.device),
            torch.ones((), dtype=adjacency.dtype, device=adjacency.device),
        )
        per_pair = (prediction - clean_state).square() * weights
        return (per_pair * upper_mask).sum() / upper_mask.sum().clamp(min=1)

    def predict_clean(
        self,
        noisy_state: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor,
        pair_condition: torch.Tensor,
        eigenvalue_condition: torch.Tensor,
        condition_present: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        normalized_t = t.to(noisy_state.dtype).unsqueeze(1) / self.cfg.timesteps
        conditional = self.denoiser(
            noisy_state,
            normalized_t,
            node_mask,
            pair_condition,
            eigenvalue_condition,
            condition_present,
        )
        if guidance_scale == 1.0 or not torch.any(condition_present > 0):
            return conditional

        unconditional = self.denoiser(
            noisy_state,
            normalized_t,
            node_mask,
            torch.zeros_like(pair_condition),
            torch.zeros_like(eigenvalue_condition),
            torch.zeros_like(condition_present),
        )
        return unconditional + guidance_scale * (conditional - unconditional)

    @torch.no_grad()
    def sample(
        self,
        node_mask: torch.Tensor,
        pair_condition: torch.Tensor | None = None,
        eigenvalue_condition: torch.Tensor | None = None,
        condition_present: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self.eval()
        batch, n = node_mask.shape
        dtype = next(self.parameters()).dtype
        device = node_mask.device

        if pair_condition is None:
            pair_condition = torch.zeros(batch, n, n, dtype=dtype, device=device)
        if eigenvalue_condition is None:
            eigenvalue_condition = torch.zeros(
                batch, self.cfg.condition_k, dtype=dtype, device=device
            )
        if condition_present is None:
            condition_present = torch.zeros(batch, 1, dtype=dtype, device=device)

        state = sample_symmetric_noise(
            (batch, n, n),
            node_mask,
            generator=generator,
            dtype=dtype,
            device=device,
        )

        for step in range(self.cfg.timesteps, 0, -1):
            t = torch.full((batch,), step, dtype=torch.long, device=device)
            predicted_clean = self.predict_clean(
                state,
                t,
                node_mask,
                pair_condition,
                eigenvalue_condition,
                condition_present,
                guidance_scale,
            ).clamp(-1.0, 1.0)

            beta_t = self.betas[step - 1]
            alpha_t = self.alphas[step - 1]
            alpha_bar_t = self.alpha_bars[step]
            alpha_bar_previous = self.alpha_bars[step - 1]

            coefficient_clean = (
                beta_t * alpha_bar_previous.sqrt() / (1.0 - alpha_bar_t)
            )
            coefficient_state = (
                alpha_t.sqrt() * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t)
            )
            state = coefficient_clean * predicted_clean + coefficient_state * state

            if step > 1:
                posterior_variance = (
                    beta_t * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t)
                )
                state = state + posterior_variance.sqrt() * sample_symmetric_noise(
                    state.shape,
                    node_mask,
                    generator=generator,
                    dtype=state.dtype,
                    device=state.device,
                )
            state = _masked_pair(state, node_mask)

        return diffusion_state_to_adjacency(state, node_mask)
