from __future__ import annotations

import unittest

import torch

from fald.models.adjacency_diffusion import (
    _cycle_features,
    AdjacencyDiffusion,
    AdjacencyDiffusionConfig,
    adjacency_to_diffusion_state,
    diffusion_state_to_adjacency,
    sample_symmetric_noise,
)
from fald.models.discrete_adjacency_diffusion import DiscreteAdjacencyDiffusion


def _small_config(**overrides) -> AdjacencyDiffusionConfig:
    values = {
        "timesteps": 20,
        "condition_k": 2,
        "time_dim": 16,
        "n_layers": 1,
        "dx": 32,
        "de": 16,
        "dy": 32,
        "n_head": 4,
        "dim_ffX": 64,
        "dim_ffE": 32,
        "dim_ffy": 64,
    }
    values.update(overrides)
    return AdjacencyDiffusionConfig(**values)


class AdjacencyDiffusionTests(unittest.TestCase):
    def test_cycle_features_on_triangle(self):
        adjacency = torch.tensor(
            [[[0, 1, 1], [1, 0, 1], [1, 1, 0]]],
            dtype=torch.float32,
        )
        node_mask = torch.ones(1, 3, dtype=torch.bool)
        node, global_features = _cycle_features(adjacency, node_mask)

        self.assertTrue(torch.allclose(node[0, :, 0], torch.full((3,), 0.1)))
        self.assertTrue(torch.allclose(node[0, :, 1:], torch.zeros(3, 2)))
        self.assertAlmostEqual(float(global_features[0, 0]), 0.1, places=6)
        self.assertTrue(
            torch.allclose(global_features[0, 1:], torch.zeros(3), atol=1e-6)
        )

    def test_adjacency_state_round_trip_and_noise_symmetry(self):
        adjacency = torch.tensor(
            [[[0, 1, 0], [1, 0, 1], [0, 1, 0]]],
            dtype=torch.float32,
        )
        node_mask = torch.ones(1, 3, dtype=torch.bool)

        state = adjacency_to_diffusion_state(adjacency, node_mask)
        reconstructed = diffusion_state_to_adjacency(state, node_mask)
        noise = sample_symmetric_noise(state.shape, node_mask)

        self.assertTrue(torch.equal(adjacency, reconstructed))
        self.assertTrue(torch.equal(noise, noise.transpose(1, 2)))
        self.assertEqual(
            torch.count_nonzero(torch.diagonal(noise, dim1=1, dim2=2)).item(),
            0,
        )

    def test_forward_process_has_exact_zero_endpoint_and_gaussian_terminal(self):
        torch.manual_seed(0)
        diffusion = AdjacencyDiffusion(_small_config(timesteps=100))
        batch, n = 256, 12
        node_mask = torch.ones(batch, n, dtype=torch.bool)
        adjacency = torch.randint(0, 2, (batch, n, n)).float()
        adjacency = torch.triu(adjacency, diagonal=1)
        adjacency = adjacency + adjacency.transpose(1, 2)
        clean = adjacency_to_diffusion_state(adjacency, node_mask)

        at_zero, _ = diffusion.q_sample(
            clean,
            torch.zeros(batch, dtype=torch.long),
            node_mask,
        )
        at_terminal, _ = diffusion.q_sample(
            clean,
            torch.full((batch,), diffusion.cfg.timesteps, dtype=torch.long),
            node_mask,
        )
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        terminal_values = at_terminal[:, upper]

        self.assertTrue(torch.equal(at_zero, clean))
        self.assertLess(abs(float(terminal_values.mean())), 0.05)
        self.assertGreater(float(terminal_values.std()), 0.95)
        self.assertLess(float(terminal_values.std()), 1.05)

    def test_denoiser_is_permutation_equivariant_with_condition(self):
        torch.manual_seed(0)
        model = AdjacencyDiffusion(_small_config()).eval()
        batch, n = 2, 7
        node_mask = torch.ones(batch, n, dtype=torch.bool)
        noisy = sample_symmetric_noise((batch, n, n), node_mask)
        pair_condition = sample_symmetric_noise((batch, n, n), node_mask)
        eigenvalues = torch.randn(batch, model.cfg.condition_k)
        present = torch.ones(batch, 1)
        time = torch.full((batch, 1), 0.5)
        permutation = torch.randperm(n)

        with torch.no_grad():
            original = model.denoiser(
                noisy,
                time,
                node_mask,
                pair_condition,
                eigenvalues,
                present,
            )
            permuted = model.denoiser(
                noisy[:, permutation][:, :, permutation],
                time,
                node_mask[:, permutation],
                pair_condition[:, permutation][:, :, permutation],
                eigenvalues,
                present,
            )

        expected = original[:, permutation][:, :, permutation]
        self.assertTrue(torch.allclose(permuted, expected, atol=1e-5, rtol=1e-5))

    def test_discrete_forward_endpoints_match_data_and_marginal(self):
        torch.manual_seed(0)
        edge_marginal = 0.12
        diffusion = DiscreteAdjacencyDiffusion(
            _small_config(timesteps=100),
            edge_marginal=edge_marginal,
        )
        batch, n = 256, 12
        node_mask = torch.ones(batch, n, dtype=torch.bool)
        adjacency = torch.randint(0, 2, (batch, n, n)).float()
        adjacency = torch.triu(adjacency, diagonal=1)
        adjacency = adjacency + adjacency.transpose(1, 2)

        at_zero = diffusion.q_sample(
            adjacency,
            torch.zeros(batch, dtype=torch.long),
            node_mask,
        )
        at_terminal = diffusion.q_sample(
            adjacency,
            torch.full((batch,), diffusion.cfg.timesteps, dtype=torch.long),
            node_mask,
        )
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)

        self.assertTrue(torch.equal(at_zero, adjacency))
        self.assertLess(
            abs(float(at_terminal[:, upper].mean()) - edge_marginal),
            0.015,
        )

    def test_discrete_reverse_matches_enumerated_posterior(self):
        diffusion = DiscreteAdjacencyDiffusion(
            _small_config(timesteps=20),
            edge_marginal=0.2,
        )
        clean_edge_probability = torch.tensor(
            [[[0.0, 0.7], [0.7, 0.0]]],
            dtype=torch.float32,
        )
        current = torch.tensor(
            [[[0.0, 1.0], [1.0, 0.0]]],
            dtype=torch.float32,
        )
        t = torch.tensor([10], dtype=torch.long)
        reverse = diffusion.reverse_probabilities(
            clean_edge_probability,
            current,
            t,
        )

        transition = diffusion._transition(diffusion.alphas[t - 1])[0]
        cumulative_previous = diffusion._transition(diffusion.alpha_bars[t - 1])[0]
        cumulative_t = diffusion._transition(diffusion.alpha_bars[t])[0]
        p_clean = torch.tensor([0.3, 0.7])
        observed = 1
        enumerated = torch.zeros(2)
        for previous in range(2):
            for clean in range(2):
                enumerated[previous] += (
                    p_clean[clean]
                    * transition[previous, observed]
                    * cumulative_previous[clean, previous]
                    / cumulative_t[clean, observed]
                )
        enumerated = enumerated / enumerated.sum()

        self.assertTrue(
            torch.allclose(reverse[0, 0, 1], enumerated, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                reverse.sum(dim=-1),
                torch.ones_like(reverse[..., 0]),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
