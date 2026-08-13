from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .critics import mlp

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


class GaussianChunkPolicy(nn.Module):
    """Tanh-squashed diagonal Gaussian over a flattened action chunk."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 7,
        action_horizon: int = 8,
        hidden_dims: Sequence[int] = (288, 288, 288),
        activation_name: str = "mish",
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.flat_action_dim = action_dim * action_horizon
        self.mean_network = mlp(
            observation_dim, hidden_dims, self.flat_action_dim, activation_name
        )
        self.raw_log_std = nn.Parameter(torch.full((self.flat_action_dim,), -1.0))

    @property
    def log_std(self) -> torch.Tensor:
        # Smooth bounded parameterization keeps the scale finite under PPO updates.
        midpoint = (LOG_STD_MAX + LOG_STD_MIN) / 2
        radius = (LOG_STD_MAX - LOG_STD_MIN) / 2
        return midpoint + radius * torch.tanh(self.raw_log_std)

    def distribution(self, observations: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_network(observations)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    @staticmethod
    def _squash_log_prob(
        distribution: torch.distributions.Normal,
        pre_tanh: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(torch.clamp(1.0 - actions.square(), min=1e-6))
        return (distribution.log_prob(pre_tanh) - correction).sum(dim=-1)

    def sample_with_log_prob(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(pre_tanh)
        log_prob = self._squash_log_prob(distribution, pre_tanh, actions)
        return actions.view(-1, self.action_horizon, self.action_dim), log_prob

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.sample_with_log_prob(observations, deterministic)[0]

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        flattened = actions.reshape(-1, self.flat_action_dim).clamp(-1 + 1e-6, 1 - 1e-6)
        pre_tanh = torch.atanh(flattened)
        return self._squash_log_prob(self.distribution(observations), pre_tanh, flattened)

    def entropy(self, observations: torch.Tensor) -> torch.Tensor:
        # Base entropy is a stable exploration diagnostic; the transformed entropy
        # has no simple analytic form.
        return self.distribution(observations).entropy().sum(dim=-1)

    def bc_loss(self, observations: torch.Tensor, target_chunks: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(observations, target_chunks.clamp(-0.999, 0.999)).mean()


def closest_hidden_width(
    observation_dim: int,
    target_parameters: int,
    layers: int = 3,
    action_dim: int = 56,
) -> int:
    """Find a uniform Gaussian width nearest a target actor parameter count."""
    best_width, best_gap = 16, math.inf
    for width in range(16, 1025, 8):
        network = GaussianChunkPolicy(
            observation_dim,
            action_dim=action_dim // 8,
            action_horizon=8,
            hidden_dims=(width,) * layers,
        )
        count = sum(parameter.numel() for parameter in network.parameters())
        gap = abs(count - target_parameters)
        if gap < best_gap:
            best_width, best_gap = width, gap
    return best_width

