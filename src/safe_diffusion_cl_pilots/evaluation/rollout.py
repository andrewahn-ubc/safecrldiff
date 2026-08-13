from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from safe_diffusion_cl_pilots.data.schemas import NormalizationStats
from safe_diffusion_cl_pilots.envs.object_centric_obs import ObjectCentricObservation
from safe_diffusion_cl_pilots.utils.seeding import torch_rng


def _reset(env: Any, seed: int) -> Any:
    value = env.reset(seed=seed)
    return value[0] if isinstance(value, tuple) and len(value) == 2 else value


def rollout_policy(
    env: Any,
    policy: Any,
    environment_seed: int,
    action_seed: int,
    device: str | torch.device = "cpu",
    normalizer: NormalizationStats | None = None,
    extractor: ObjectCentricObservation | None = None,
    max_chunk_steps: int | None = None,
) -> dict[str, Any]:
    extractor = extractor or ObjectCentricObservation()
    observation = _reset(env, environment_seed)
    total_reward = 0.0
    chunk_steps = 0
    final_info: dict[str, Any] = {}
    terminated = truncated = False
    with torch_rng(action_seed, device):
        while not terminated and not truncated:
            vector = extractor.extract(observation) if isinstance(observation, Mapping) else np.asarray(observation)
            if normalizer is not None:
                vector = normalizer.normalize_observations(vector)
            tensor = torch.as_tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy.sample(tensor).squeeze(0).cpu().numpy()
            if normalizer is not None:
                action = normalizer.denormalize_actions(action)
            observation, reward, terminated, truncated, final_info = env.step(action)
            total_reward += float(reward)
            chunk_steps += 1
            if max_chunk_steps is not None and chunk_steps >= max_chunk_steps:
                truncated = True
    return {
        "environment_seed": environment_seed,
        "action_seed": action_seed,
        "return": total_reward,
        "episode_length": int(final_info.get("episode_primitive_steps", chunk_steps * 8)),
        "task_success": bool(final_info.get("task_success", False)),
        "damage_event": bool(final_info.get("damage_event", False)),
        "designated_health_loss": float(final_info.get("designated_fragile_health_loss", 0.0)),
        "total_environment_damage": float(final_info.get("total_environment_damage", 0.0)),
        "cereal_trajectory": final_info.get("cereal_trajectory", []),
        "end_effector_trajectory": final_info.get("end_effector_trajectory", []),
    }


def evaluate_fixed_seeds(
    env_factory: Any,
    policy: Any,
    environment_seeds: Sequence[int],
    action_seed_offset: int,
    **rollout_kwargs: Any,
) -> list[dict[str, Any]]:
    env = env_factory()
    try:
        return [
            rollout_policy(
                env,
                policy,
                seed,
                action_seed_offset + seed,
                **rollout_kwargs,
            )
            for seed in environment_seeds
        ]
    finally:
        env.close()


def summarize_rollouts(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    success = np.asarray([row["task_success"] for row in rows], dtype=bool)
    damage = np.asarray([row["damage_event"] for row in rows], dtype=bool)
    return {
        "task_success": float(success.mean()) if len(rows) else 0.0,
        "safe_success": float(np.mean(success & ~damage)) if len(rows) else 0.0,
        "damage_event_rate": float(damage.mean()) if len(rows) else 0.0,
        "p_damage_given_success": float(damage[success].mean()) if success.any() else 0.0,
        "mean_designated_health_loss": float(np.mean([row["designated_health_loss"] for row in rows])) if rows else 0.0,
        "total_environment_damage": float(np.mean([row["total_environment_damage"] for row in rows])) if rows else 0.0,
        "mean_return": float(np.mean([row["return"] for row in rows])) if rows else 0.0,
        "episode_length": float(np.mean([row["episode_length"] for row in rows])) if rows else 0.0,
    }
