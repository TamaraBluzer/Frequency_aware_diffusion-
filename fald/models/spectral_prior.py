"""Learned spectral prior: small diffusion model over (n, lambda_k, U_k).

Stage 8 of PLAN.md. Generates eigenvalues and eigenvectors of the normalized
graph Laplacian so the adjacency diffusion model can sample end-to-end without
oracle spectral inputs.

The diffusion state is a flat vector concatenating normalised eigenvalues and
flattened eigenvectors. An orthogonality penalty on U^T U - I is added during
training, and QR retraction is applied at the final sampling step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SpectralPriorConfig",
    "SpectralPriorDenoiser",
    "SpectralPrior",
]


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return t
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    angles = t * 1_000.0 * frequencies.unsqueeze(0)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


@dataclass
class SpectralPriorConfig:
    n_max: int = 64
    k: int = 8
    timesteps: int = 100
    time_dim: int = 64
    hidden_dim: int = 512
    n_hidden_layers: int = 4
    dropout: float = 0.0
    condition_drop_probability: float = 0.1
    cosine_offset: float = 0.008
    ortho_penalty_weight: float = 0.1

    @property
    def state_dim(self) -> int:
        return self.k + self.n_max * self.k


class SpectralPriorDenoiser(nn.Module):
    """MLP x_0-predictor conditioned on timestep and graph size n."""

    def __init__(self, cfg: SpectralPriorConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.state_dim + cfg.time_dim + 1
        self.input_proj = nn.Linear(input_dim, cfg.hidden_dim)
        layers = []
        for _ in range(cfg.n_hidden_layers):
            layers.extend([
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
            ])
        self.trunk = nn.Sequential(*layers)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.state_dim)

    def forward(
        self,
        noisy_state: torch.Tensor,
        time: torch.Tensor,
        n_nodes: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = _sinusoidal_embedding(
            time.reshape(-1, 1), self.cfg.time_dim
        )
        n_cond = (n_nodes.float() / self.cfg.n_max).reshape(-1, 1)
        x = torch.cat([noisy_state, t_emb, n_cond], dim=-1)
        x = self.input_proj(x)
        x = x + self.trunk(x)
        return self.output_proj(x)


class SpectralPrior(nn.Module):
    """Cosine-schedule DDPM over spectral features with orthogonality penalty."""

    def __init__(self, cfg: SpectralPriorConfig | None = None):
        super().__init__()
        self.cfg = cfg or SpectralPriorConfig()
        self.denoiser = SpectralPriorDenoiser(self.cfg)

        steps = torch.arange(self.cfg.timesteps + 1, dtype=torch.float64)
        phase = (steps / self.cfg.timesteps + self.cfg.cosine_offset)
        phase = phase / (1.0 + self.cfg.cosine_offset)
        alpha_bar = torch.cos(phase * math.pi / 2.0).square()
        alpha_bar = alpha_bar / alpha_bar[0]

        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        betas = betas.clamp(min=1e-8, max=0.999)
        alphas = 1.0 - betas
        alpha_bars = torch.cat(
            [torch.ones(1, dtype=torch.float64), torch.cumprod(alphas, dim=0)]
        )
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bars", alpha_bars.float())

    def _pack_state(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        """Pack (batch, k) eigenvalues and (batch, n_max, k) eigenvectors into flat state."""
        batch = eigenvalues.shape[0]
        return torch.cat([
            eigenvalues,
            eigenvectors.reshape(batch, -1),
        ], dim=-1)

    def _unpack_state(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack flat state into eigenvalues (batch, k) and eigenvectors (batch, n_max, k)."""
        k = self.cfg.k
        eigenvalues = state[:, :k]
        eigenvectors = state[:, k:].reshape(-1, self.cfg.n_max, k)
        return eigenvalues, eigenvectors

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        extracted = values.gather(0, t)
        return extracted.reshape(t.shape[0], *([1] * (ndim - 1)))

    def q_sample(
        self,
        clean: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean)
        ab = self._extract(self.alpha_bars, t, clean.ndim)
        noisy = ab.sqrt() * clean + (1.0 - ab).sqrt() * noise
        return noisy, noise

    def orthogonality_loss(
        self,
        eigenvectors: torch.Tensor,
        n_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """||U_n^T U_n - I_k||_F^2, masked to the actual n nodes per graph."""
        k = self.cfg.k
        batch = eigenvectors.shape[0]
        total = torch.tensor(0.0, device=eigenvectors.device)
        for i in range(batch):
            n = int(n_nodes[i].item())
            U = eigenvectors[i, :n, :]  # (n, k)
            gram = U.T @ U  # (k, k)
            eye = torch.eye(k, device=U.device, dtype=U.dtype)
            total = total + (gram - eye).square().sum()
        return total / batch

    def training_loss(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        n_nodes: torch.Tensor,
    ) -> torch.Tensor:
        batch = eigenvalues.shape[0]
        clean = self._pack_state(eigenvalues, eigenvectors)
        t = torch.randint(1, self.cfg.timesteps + 1, (batch,), device=clean.device)
        noisy, _ = self.q_sample(clean, t)

        predicted = self.denoiser(
            noisy,
            t.to(clean.dtype).unsqueeze(1) / self.cfg.timesteps,
            n_nodes,
        )

        mse = (predicted - clean).square().mean()

        pred_eigenvalues, pred_eigenvectors = self._unpack_state(predicted)
        ortho = self.orthogonality_loss(pred_eigenvectors, n_nodes)

        return mse + self.cfg.ortho_penalty_weight * ortho

    def predict_clean(
        self,
        noisy: torch.Tensor,
        t: torch.Tensor,
        n_nodes: torch.Tensor,
    ) -> torch.Tensor:
        normalized_t = t.to(noisy.dtype).unsqueeze(1) / self.cfg.timesteps
        return self.denoiser(noisy, normalized_t, n_nodes)

    @staticmethod
    def _qr_retraction(
        eigenvectors: torch.Tensor,
        n_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Project eigenvector block onto the Stiefel manifold via QR."""
        result = eigenvectors.clone()
        for i in range(eigenvectors.shape[0]):
            n = int(n_nodes[i].item())
            k = eigenvectors.shape[2]
            U = eigenvectors[i, :n, :]  # (n, k)
            if n >= k:
                Q, _ = torch.linalg.qr(U)
                result[i, :n, :] = Q[:, :k]
            result[i, n:, :] = 0.0
        return result

    @torch.no_grad()
    def sample(
        self,
        n_nodes: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample spectral features. Returns (eigenvalues, eigenvectors).

        eigenvalues: (batch, k) in [-1, 1] (normalised from [0, 2])
        eigenvectors: (batch, n_max, k) with QR-retracted orthonormal columns
        """
        self.eval()
        batch = n_nodes.shape[0]
        device = n_nodes.device
        dtype = next(self.parameters()).dtype

        state = torch.randn(
            batch, self.cfg.state_dim,
            generator=generator, dtype=dtype, device=device,
        )

        for step in range(self.cfg.timesteps, 0, -1):
            t = torch.full((batch,), step, dtype=torch.long, device=device)
            predicted_clean = self.predict_clean(state, t, n_nodes)
            predicted_clean = predicted_clean.clamp(-3.0, 3.0)

            beta_t = self.betas[step - 1]
            alpha_t = self.alphas[step - 1]
            alpha_bar_t = self.alpha_bars[step]
            alpha_bar_prev = self.alpha_bars[step - 1]

            coeff_clean = beta_t * alpha_bar_prev.sqrt() / (1.0 - alpha_bar_t)
            coeff_state = alpha_t.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            state = coeff_clean * predicted_clean + coeff_state * state

            if step > 1:
                posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                state = state + posterior_var.sqrt() * torch.randn_like(state)

        eigenvalues, eigenvectors = self._unpack_state(state)
        eigenvalues = eigenvalues.clamp(-1.0, 1.0)

        # Zero out eigenvector rows beyond actual node count
        for i in range(batch):
            n = int(n_nodes[i].item())
            eigenvectors[i, n:, :] = 0.0

        eigenvectors = self._qr_retraction(eigenvectors, n_nodes)

        return eigenvalues, eigenvectors
