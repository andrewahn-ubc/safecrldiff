from __future__ import annotations

from collections.abc import Sequence

import torch

from safe_diffusion_cl_pilots.models.critics import ValueCritic
from safe_diffusion_cl_pilots.models.diffusion_policy import (
    DiffusionPolicy,
    DiffusionTransition,
)
from safe_diffusion_cl_pilots.models.dppo_adapter import transition_ppo_loss

from .common import PPOStats, normalized_advantages


def diffusion_dppo_update(
    actor: DiffusionPolicy,
    critic: ValueCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    transitions: Sequence[DiffusionTransition],
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_ratio: float = 0.2,
    gradient_clip: float = 5.0,
) -> PPOStats:
    """Weight-level DPPO over the sampled denoising transition likelihoods."""
    advantages = normalized_advantages(advantages)
    new_transition_log_probs = actor.transition_log_probs(observations, transitions)
    if any(transition.log_prob is None for transition in transitions):
        raise ValueError("diffusion rollout trace lacks old transition log probabilities")
    old_transition_log_probs = torch.stack(
        [transition.log_prob for transition in transitions if transition.log_prob is not None],
        dim=1,
    )
    old_joint = old_transition_log_probs.sum(dim=(-1, -2))
    if not torch.allclose(old_joint, old_log_probs, atol=1e-4, rtol=1e-4):
        raise ValueError("stored joint and transition diffusion log probabilities disagree")
    actor_loss, ratios = transition_ppo_loss(
        new_transition_log_probs,
        old_transition_log_probs,
        advantages,
        clip_ratio=clip_ratio,
    )
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip)
    actor_optimizer.step()
    critic_loss = torch.nn.functional.mse_loss(critic(observations), returns)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), gradient_clip)
    critic_optimizer.step()
    new = new_transition_log_probs.clamp(-5.0, 2.0).mean(dim=-1)
    old = old_transition_log_probs.clamp(-5.0, 2.0).mean(dim=-1)
    approximate_kl = ((ratios - 1.0) - (new - old)).mean()
    return PPOStats(
        policy_loss=float(actor_loss.detach()),
        critic_loss=float(critic_loss.detach()),
        approximate_kl=float(approximate_kl.detach()),
        ratio_mean=float(ratios.mean().detach()),
        gradient_norm=float(gradient_norm),
    )
