from __future__ import annotations

import math

import torch

# Auditable mapping to the official checkout installed by scripts/bootstrap.py.
# The project owns environment/data adapters, while this objective preserves the
# official PPODiffusion transition-level likelihood, denoising discount, and
# timestep-dependent clipping conventions without patching the vendor tree.
OFFICIAL_DPPO_SOURCE = "vendor/dppo/model/diffusion/diffusion_ppo.py"
OFFICIAL_ROBOMIMIC_CONFIG_ROOT = "vendor/dppo/cfg/robomimic"


def transition_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    gamma_denoising: float = 0.99,
    clip_ratio: float = 0.2,
    clip_ratio_base: float = 1e-3,
    clip_ratio_rate: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official-style PPO objective over diffusion MDP transitions.

    Inputs have shape [batch, denoising_step, flattened_action]. As in the
    upstream PPODiffusion loss, element log probabilities are clamped and
    averaged over action coordinates before each denoising transition becomes
    a PPO sample.
    """
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.ndim != 3:
        raise ValueError("DPPO transition log probabilities must share shape [B, K, A]")
    denoising_steps = new_log_probs.shape[1]
    new = new_log_probs.clamp(-5.0, 2.0).mean(dim=-1)
    old = old_log_probs.clamp(-5.0, 2.0).mean(dim=-1)
    indices = torch.arange(denoising_steps, device=new.device, dtype=new.dtype)
    discount = gamma_denoising ** (denoising_steps - indices - 1)
    weighted_advantages = advantages[:, None] * discount[None, :]
    if denoising_steps > 1:
        relative = indices / (denoising_steps - 1)
        clips = clip_ratio_base + (clip_ratio - clip_ratio_base) * (
            torch.exp(clip_ratio_rate * relative) - 1
        ) / (math.exp(clip_ratio_rate) - 1)
    else:
        clips = torch.full_like(indices, clip_ratio)
    log_ratio = (new - old).clamp(-20.0, 20.0)
    ratios = log_ratio.exp()
    unclipped = ratios * weighted_advantages
    clipped_ratios = torch.maximum(
        torch.minimum(ratios, 1.0 + clips[None, :]), 1.0 - clips[None, :]
    )
    loss = -torch.minimum(unclipped, clipped_ratios * weighted_advantages).mean()
    return loss, ratios
