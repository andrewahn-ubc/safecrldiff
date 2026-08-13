from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .critics import mlp


def sinusoidal_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    exponent = -math.log(10_000.0) * torch.arange(
        half, device=timesteps.device, dtype=torch.float32
    ) / max(half - 1, 1)
    angles = timesteps.float().unsqueeze(-1) * exponent.exp().unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dimension % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


@dataclass
class DiffusionTransition:
    latent: torch.Tensor
    next_latent: torch.Tensor
    timestep: torch.Tensor
    next_timestep: int
    log_prob: torch.Tensor | None = None

    def detached(self) -> DiffusionTransition:
        return DiffusionTransition(
            self.latent.detach(),
            self.next_latent.detach(),
            self.timestep.detach(),
            self.next_timestep,
            self.log_prob.detach() if self.log_prob is not None else None,
        )


class DiffusionPolicy(nn.Module):
    """Low-dimensional conditional diffusion actor with DPPO transition likelihoods."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 7,
        action_horizon: int = 8,
        denoising_steps: int = 100,
        sampling_steps: int = 5,
        hidden_dims: Sequence[int] = (256, 256, 256),
        time_embedding_dim: int = 64,
        activation_name: str = "mish",
        min_sampling_std: float = 0.05,
    ):
        super().__init__()
        if sampling_steps > denoising_steps:
            raise ValueError("sampling_steps cannot exceed denoising_steps")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.flat_action_dim = action_dim * action_horizon
        self.denoising_steps = denoising_steps
        self.sampling_steps = sampling_steps
        self.time_embedding_dim = time_embedding_dim
        self.min_sampling_std = min_sampling_std
        self.denoiser = mlp(
            observation_dim + self.flat_action_dim + time_embedding_dim,
            hidden_dims,
            self.flat_action_dim,
            activation_name,
        )
        betas = torch.linspace(1e-4, 0.02, denoising_steps)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

    def predict_noise(
        self, observations: torch.Tensor, noisy_actions: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        embedded_time = sinusoidal_embedding(timesteps, self.time_embedding_dim)
        return self.denoiser(torch.cat([observations, noisy_actions, embedded_time], dim=-1))

    def bc_loss(self, observations: torch.Tensor, target_chunks: torch.Tensor) -> torch.Tensor:
        target = target_chunks.reshape(-1, self.flat_action_dim)
        timesteps = torch.randint(
            0, self.denoising_steps, (len(target),), device=target.device
        )
        noise = torch.randn_like(target)
        alpha_bar = self.alpha_bars[timesteps].unsqueeze(-1)
        noisy = alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise
        prediction = self.predict_noise(observations, noisy, timesteps)
        return torch.nn.functional.mse_loss(prediction, noise)

    def _schedule(self) -> list[tuple[int, int]]:
        selected = torch.linspace(
            self.denoising_steps - 1, 0, self.sampling_steps, dtype=torch.long
        ).tolist()
        timesteps = [int(value) for value in selected]
        return [(current, timesteps[index + 1] if index + 1 < len(timesteps) else -1) for index, current in enumerate(timesteps)]

    def _transition_parameters(
        self,
        observations: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_now = self.alpha_bars[timestep].unsqueeze(-1)
        predicted_noise = self.predict_noise(observations, latent, timestep)
        predicted_clean = (latent - (1.0 - alpha_now).sqrt() * predicted_noise) / alpha_now.sqrt()
        if next_timestep < 0:
            mean = predicted_clean
        else:
            alpha_next = self.alpha_bars[next_timestep]
            mean = alpha_next.sqrt() * predicted_clean + (1.0 - alpha_next).sqrt() * predicted_noise
        beta = self.betas[timestep].sqrt().unsqueeze(-1)
        std = torch.clamp(beta, min=self.min_sampling_std).expand_as(mean)
        return mean, std

    def sample_with_log_prob(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[DiffusionTransition]]:
        latent = torch.randn(
            len(observations),
            self.flat_action_dim,
            device=observations.device,
            dtype=observations.dtype,
        )
        transitions: list[DiffusionTransition] = []
        log_prob = torch.zeros(len(observations), device=observations.device)
        for current, following in self._schedule():
            timestep = torch.full(
                (len(observations),), current, device=observations.device, dtype=torch.long
            )
            mean, std = self._transition_parameters(observations, latent, timestep, following)
            next_latent = mean if deterministic else mean + std * torch.randn_like(mean)
            distribution = torch.distributions.Normal(mean, std)
            element_log_prob = distribution.log_prob(next_latent)
            log_prob = log_prob + element_log_prob.sum(dim=-1)
            transitions.append(
                DiffusionTransition(
                    latent, next_latent, timestep, following, element_log_prob
                ).detached()
            )
            latent = next_latent
        actions = torch.tanh(latent).view(-1, self.action_horizon, self.action_dim)
        return actions, log_prob, transitions

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.sample_with_log_prob(observations, deterministic)[0]

    def log_prob(
        self, observations: torch.Tensor, transitions: Sequence[DiffusionTransition]
    ) -> torch.Tensor:
        return self.transition_log_probs(observations, transitions).sum(dim=(-1, -2))

    def transition_log_probs(
        self, observations: torch.Tensor, transitions: Sequence[DiffusionTransition]
    ) -> torch.Tensor:
        if len(transitions) != self.sampling_steps:
            raise ValueError("transition trace does not match the sampling schedule")
        result: list[torch.Tensor] = []
        for transition in transitions:
            mean, std = self._transition_parameters(
                observations,
                transition.latent,
                transition.timestep,
                transition.next_timestep,
            )
            result.append(
                torch.distributions.Normal(mean, std).log_prob(transition.next_latent)
            )
        return torch.stack(result, dim=1)

    def entropy(self, observations: torch.Tensor) -> torch.Tensor:
        # Sum of transition entropies evaluated along one detached policy sample.
        _, _, transitions = self.sample_with_log_prob(observations)
        result = torch.zeros(len(observations), device=observations.device)
        for transition in transitions:
            _, std = self._transition_parameters(
                observations,
                transition.latent,
                transition.timestep,
                transition.next_timestep,
            )
            result = result + torch.distributions.Normal(
                torch.zeros_like(std), std
            ).entropy().sum(dim=-1)
        return result
