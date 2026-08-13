from __future__ import annotations

import torch

from safe_diffusion_cl_pilots.models.critics import ValueCritic
from safe_diffusion_cl_pilots.models.gaussian_chunk_policy import GaussianChunkPolicy

from .common import PPOStats, clipped_policy_loss, normalized_advantages


def gaussian_ppo_update(
    actor: GaussianChunkPolicy,
    critic: ValueCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_ratio: float = 0.2,
    gradient_clip: float = 5.0,
) -> PPOStats:
    advantages = normalized_advantages(advantages)
    new_log_probs = actor.log_prob(observations, actions)
    actor_loss, ratios = clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip_ratio)
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip)
    actor_optimizer.step()
    critic_loss = torch.nn.functional.mse_loss(critic(observations), returns)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), gradient_clip)
    critic_optimizer.step()
    approximate_kl = ((ratios - 1.0) - (new_log_probs - old_log_probs)).mean()
    return PPOStats(
        policy_loss=float(actor_loss.detach()),
        critic_loss=float(critic_loss.detach()),
        approximate_kl=float(approximate_kl.detach()),
        ratio_mean=float(ratios.mean().detach()),
        gradient_norm=float(gradient_norm),
    )

