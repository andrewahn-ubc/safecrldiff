from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from safe_diffusion_cl_pilots.models.diffusion_policy import DiffusionTransition


@dataclass
class PPOStats:
    policy_loss: float
    critic_loss: float
    approximate_kl: float
    ratio_mean: float
    gradient_norm: float


def normalized_advantages(advantages: torch.Tensor) -> torch.Tensor:
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def clipped_policy_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_ratio = torch.clamp(new_log_prob - old_log_prob, -20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.minimum(unclipped, clipped).mean(), ratio


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) != len(rewards) + 1:
        raise ValueError("values must include one bootstrap element")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carried = 0.0
    for index in reversed(range(len(rewards))):
        active = 1.0 - float(terminated[index])
        delta = rewards[index] + gamma * values[index + 1] * active - values[index]
        carried = delta + gamma * gae_lambda * active * carried
        advantages[index] = carried
    return advantages, advantages + values[:-1]


def save_checkpoint(
    path: Path,
    actor: nn.Module,
    critic: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    **metadata: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict() if critic is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "optimizers": {
            name: value.state_dict() for name, value in (optimizers or {}).items()
        },
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    actor: nn.Module,
    critic: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    actor.load_state_dict(payload["actor"])
    if critic is not None and payload.get("critic") is not None:
        critic.load_state_dict(payload["critic"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    for name, value in (optimizers or {}).items():
        if name in payload.get("optimizers", {}):
            value.load_state_dict(payload["optimizers"][name])
    return dict(payload.get("metadata", {}))


def stack_diffusion_traces(
    traces: list[list[DiffusionTransition]],
) -> list[DiffusionTransition]:
    if not traces:
        raise ValueError("cannot stack an empty transition trace")
    count = len(traces[0])
    if any(len(trace) != count for trace in traces):
        raise ValueError("diffusion traces use different sampling schedules")
    return [
        DiffusionTransition(
            latent=torch.cat([trace[index].latent for trace in traces]),
            next_latent=torch.cat([trace[index].next_latent for trace in traces]),
            timestep=torch.cat([trace[index].timestep for trace in traces]),
            next_timestep=traces[0][index].next_timestep,
            log_prob=torch.cat([trace[index].log_prob for trace in traces])
            if traces[0][index].log_prob is not None
            else None,
        )
        for index in range(count)
    ]
