"""Categorical diffusion directly on binary graph adjacency."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dense import pair_mask_from_node_mask
from .adjacency_diffusion import AdjacencyDenoiser, AdjacencyDiffusionConfig

__all__ = ["DiscreteAdjacencyDiffusion", "sample_symmetric_bernoulli"]


def sample_symmetric_bernoulli(
    edge_probability: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample each valid undirected pair exactly once."""
    random = torch.rand(
        edge_probability.shape,
        dtype=edge_probability.dtype,
        device=edge_probability.device,
        generator=generator,
    )
    upper = torch.triu((random < edge_probability).to(edge_probability.dtype), diagonal=1)
    adjacency = upper + upper.transpose(1, 2)
    pair_mask = pair_mask_from_node_mask(node_mask)
    return adjacency * pair_mask.to(adjacency.dtype)


class DiscreteAdjacencyDiffusion(nn.Module):
    """Binary D3PM with a data-marginal transition kernel and `x_0` prediction."""

    def __init__(
        self,
        cfg: AdjacencyDiffusionConfig | None = None,
        *,
        edge_marginal: float,
    ):
        super().__init__()
        self.cfg = cfg or AdjacencyDiffusionConfig()
        if not 0.0 < edge_marginal < 1.0:
            raise ValueError("edge_marginal must be strictly between zero and one")
        self.denoiser = AdjacencyDenoiser(self.cfg)
        self.register_buffer(
            "marginal",
            torch.tensor([1.0 - edge_marginal, edge_marginal], dtype=torch.float32),
        )

        steps = torch.arange(self.cfg.timesteps + 1, dtype=torch.float64)
        phase = (steps / self.cfg.timesteps + self.cfg.cosine_offset)
        phase = phase / (1.0 + self.cfg.cosine_offset)
        target_alpha_bar = torch.cos(phase * math.pi / 2.0).square()
        target_alpha_bar = target_alpha_bar / target_alpha_bar[0]
        betas = (1.0 - target_alpha_bar[1:] / target_alpha_bar[:-1]).clamp(
            min=1e-8,
            max=0.999,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cat(
            [torch.ones(1, dtype=torch.float64), torch.cumprod(alphas, dim=0)]
        )
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bars", alpha_bars.float())

    def _transition(self, alpha: torch.Tensor) -> torch.Tensor:
        """`Q = alpha I + (1-alpha) 1 m^T` for a batch of alpha values."""
        batch = alpha.shape[0]
        identity = torch.eye(2, dtype=alpha.dtype, device=alpha.device)
        stationary = self.marginal.to(alpha).reshape(1, 1, 2).expand(batch, 2, 2)
        return (
            alpha.reshape(batch, 1, 1) * identity.unsqueeze(0)
            + (1.0 - alpha).reshape(batch, 1, 1) * stationary
        )

    def q_sample(
        self,
        adjacency: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars.gather(0, t).reshape(-1, 1, 1)
        edge_probability = (
            alpha_bar * adjacency
            + (1.0 - alpha_bar) * self.marginal[1].to(adjacency)
        )
        return sample_symmetric_bernoulli(
            edge_probability,
            node_mask,
            generator=generator,
        )

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

    def _condition_defaults(
        self,
        adjacency: torch.Tensor,
        pair_condition: torch.Tensor | None,
        eigenvalue_condition: torch.Tensor | None,
        condition_present: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = adjacency.shape[0]
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
                batch,
                1,
                dtype=adjacency.dtype,
                device=adjacency.device,
            )
        return pair_condition, eigenvalue_condition, condition_present

    def training_loss(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        pair_condition: torch.Tensor | None = None,
        eigenvalue_condition: torch.Tensor | None = None,
        condition_present: torch.Tensor | None = None,
        positive_weight: float = 1.0,
    ) -> torch.Tensor:
        if positive_weight != 1.0:
            raise ValueError("discrete diffusion uses unweighted categorical cross-entropy")
        batch = adjacency.shape[0]
        t = torch.randint(1, self.cfg.timesteps + 1, (batch,), device=adjacency.device)
        noisy_adjacency = self.q_sample(adjacency, t, node_mask)
        pair_condition, eigenvalue_condition, condition_present = self._condition_defaults(
            adjacency,
            pair_condition,
            eigenvalue_condition,
            condition_present,
        )
        pair_condition, eigenvalue_condition, condition_present = self._drop_condition(
            pair_condition,
            eigenvalue_condition,
            condition_present,
        )
        logits = self.denoiser(
            noisy_adjacency.mul(2.0).sub(1.0),
            t.to(adjacency.dtype).unsqueeze(1) / self.cfg.timesteps,
            node_mask,
            pair_condition,
            eigenvalue_condition,
            condition_present,
        )

        pair_mask = pair_mask_from_node_mask(node_mask)
        upper_mask = pair_mask & torch.triu(
            torch.ones_like(pair_mask),
            diagonal=1,
        ).bool()
        loss = F.binary_cross_entropy_with_logits(logits, adjacency, reduction="none")
        return loss[upper_mask].mean()

    def predict_clean_probability(
        self,
        noisy_adjacency: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor,
        pair_condition: torch.Tensor,
        eigenvalue_condition: torch.Tensor,
        condition_present: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        normalized_t = t.to(noisy_adjacency.dtype).unsqueeze(1) / self.cfg.timesteps
        centered = noisy_adjacency.mul(2.0).sub(1.0)
        conditional_logits = self.denoiser(
            centered,
            normalized_t,
            node_mask,
            pair_condition,
            eigenvalue_condition,
            condition_present,
        )
        if guidance_scale != 1.0 and torch.any(condition_present > 0):
            unconditional_logits = self.denoiser(
                centered,
                normalized_t,
                node_mask,
                torch.zeros_like(pair_condition),
                torch.zeros_like(eigenvalue_condition),
                torch.zeros_like(condition_present),
            )
            conditional_logits = unconditional_logits + guidance_scale * (
                conditional_logits - unconditional_logits
            )
        return conditional_logits.sigmoid()

    def reverse_probabilities(
        self,
        clean_edge_probability: torch.Tensor,
        current_adjacency: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute `p_theta(x_{t-1} | x_t)` by marginalizing predicted `x_0`."""
        batch, n, _ = current_adjacency.shape
        clean_distribution = torch.stack(
            [1.0 - clean_edge_probability, clean_edge_probability],
            dim=-1,
        ).reshape(batch, n * n, 2)
        current = F.one_hot(
            current_adjacency.long(),
            num_classes=2,
        ).to(clean_distribution.dtype).reshape(batch, n * n, 2)

        alpha_t = self.alphas.gather(0, t - 1)
        alpha_bar_previous = self.alpha_bars.gather(0, t - 1)
        alpha_bar_t = self.alpha_bars.gather(0, t)
        transition = self._transition(alpha_t)
        cumulative_previous = self._transition(alpha_bar_previous)
        cumulative_t = self._transition(alpha_bar_t)

        left = torch.einsum("bnk,bjk->bnj", current, transition)
        denominator = torch.einsum("bik,bnk->bni", cumulative_t, current)
        posterior_given_clean = (
            left.unsqueeze(2)
            * cumulative_previous.unsqueeze(1)
            / denominator.clamp(min=1e-12).unsqueeze(-1)
        )
        reverse = (clean_distribution.unsqueeze(-1) * posterior_given_clean).sum(dim=2)
        reverse = reverse / reverse.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return reverse.reshape(batch, n, n, 2)

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
        template = torch.zeros(batch, n, n, dtype=dtype, device=device)
        pair_condition, eigenvalue_condition, condition_present = self._condition_defaults(
            template,
            pair_condition,
            eigenvalue_condition,
            condition_present,
        )
        limit_probability = torch.full_like(template, float(self.marginal[1]))
        adjacency = sample_symmetric_bernoulli(
            limit_probability,
            node_mask,
            generator=generator,
        )

        for step in range(self.cfg.timesteps, 0, -1):
            t = torch.full((batch,), step, dtype=torch.long, device=device)
            clean_probability = self.predict_clean_probability(
                adjacency,
                t,
                node_mask,
                pair_condition,
                eigenvalue_condition,
                condition_present,
                guidance_scale,
            )
            reverse = self.reverse_probabilities(clean_probability, adjacency, t)
            adjacency = sample_symmetric_bernoulli(
                reverse[..., 1],
                node_mask,
                generator=generator,
            )
        return adjacency
