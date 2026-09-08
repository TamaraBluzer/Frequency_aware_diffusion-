"""Sign-invariant eigenvector encoder.

`u_i` and `-u_i` are the same eigenvector, and for repeated eigenvalues any rotation within the
eigenspace is equally valid. SBM graphs have near-degenerate eigenvalues by construction, so
this is not a corner case (WORKPLAN.md G3). SPECTRE canonicalizes signs by forcing the
max-magnitude entry positive, which is discontinuous and breaks under multiplicity.

SignNet instead makes the encoder invariant by construction: `psi = rho([phi(u_j) + phi(-u_j)]_j)`.
Because `phi(u) + phi(-u)` is symmetric in the sign of `u`, flipping any column leaves the
output bitwise-comparably unchanged, with no branch and no discontinuity.

`phi` acts on `(u_j[i], lambda_j)` per node per eigenvector, so the encoder is permutation
equivariant: it never sees a node index. Concatenating over `j` rather than summing is
deliberate — the frequency index is meaningful and ordered, and collapsing it would discard the
very structure this project is measuring.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SignNet", "random_sign_flip"]


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int) -> nn.Sequential:
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if n_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class SignNet(nn.Module):
    """Encode `(U_k, lambda_k)` into a per-node vector, invariant to per-column sign flips."""

    def __init__(
        self,
        k: int,
        hidden: int = 64,
        out_dim: int = 64,
        phi_layers: int = 2,
        rho_layers: int = 2,
        use_eigvals: bool = True,
    ):
        super().__init__()
        if k < 1:
            raise ValueError("SignNet needs k >= 1; the 'none' arm should not construct one")
        self.k = k
        self.hidden = hidden
        self.use_eigvals = use_eigvals
        self.phi = _mlp(2 if use_eigvals else 1, hidden, hidden, phi_layers)
        self.rho = _mlp(k * hidden, hidden, out_dim, rho_layers)

    def forward(
        self,
        eigvecs: torch.Tensor,
        eigvals: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """eigvecs `(B, n, k)`, eigvals `(B, k)`, node_mask `(B, n)` -> `(B, n, out_dim)`."""
        if eigvecs.dim() != 3:
            raise ValueError(f"expected eigvecs of shape (B, n, k), got {tuple(eigvecs.shape)}")
        batch, n, k = eigvecs.shape
        if k != self.k:
            raise ValueError(f"this SignNet was built for k={self.k}, got k={k}")

        u = eigvecs.unsqueeze(-1)
        if self.use_eigvals:
            lam = eigvals[:, None, :, None].expand(batch, n, k, 1)
            positive = torch.cat([u, lam], dim=-1)
            negative = torch.cat([-u, lam], dim=-1)
        else:
            positive, negative = u, -u

        # The sum is what buys invariance: swapping the two terms leaves it unchanged.
        per_eigenvector = self.phi(positive) + self.phi(negative)
        psi = self.rho(per_eigenvector.reshape(batch, n, k * self.hidden))

        if node_mask is not None:
            psi = psi * node_mask.unsqueeze(-1).to(psi.dtype)
        return psi


def random_sign_flip(eigvecs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Flip each eigenvector's sign independently at random.

    Secondary defence alongside SignNet: even though the encoder is invariant by construction,
    augmenting keeps any *downstream* component from quietly learning a sign convention.
    """
    batch, _, k = eigvecs.shape
    signs = torch.randint(
        0, 2, (batch, 1, k), generator=generator, device=eigvecs.device, dtype=eigvecs.dtype
    )
    return eigvecs * (signs * 2 - 1)
