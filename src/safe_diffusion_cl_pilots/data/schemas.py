from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FRAGILE_OBJECTS = ("wine_1", "wine_glass", "wine_2")
OBJECT_ORDER = ("cereal", "wine_1", "wine_glass", "wine_2", "flour_bag")


@dataclass
class EpisodeRecord:
    episode_id: str
    source_file_label: str
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    task_success: bool
    health_by_object: dict[str, np.ndarray]
    damage_by_object: dict[str, float]
    total_environment_damage: float
    termination_reason: str
    initial_scene_manifest: dict[str, Any] = field(default_factory=dict)
    cereal_trajectory: np.ndarray | None = None
    end_effector_trajectory: np.ndarray | None = None

    def validate(self) -> None:
        length = len(self.actions)
        if self.observations.ndim != 2 or self.actions.ndim != 2:
            raise ValueError(f"{self.episode_id}: observations/actions must be rank two")
        if self.actions.shape[1] != 7:
            raise ValueError(f"{self.episode_id}: expected 7-D actions, got {self.actions.shape}")
        if len(self.observations) != length or len(self.rewards) != length:
            raise ValueError(f"{self.episode_id}: trajectory arrays disagree in length")
        if length == 0:
            raise ValueError(f"{self.episode_id}: trajectory is empty")
        arrays = [self.observations, self.actions, self.rewards, *self.health_by_object.values()]
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError(f"{self.episode_id}: non-finite trajectory value")
        if any(len(values) != length for values in self.health_by_object.values()):
            raise ValueError(f"{self.episode_id}: health trajectory length mismatch")

    def metadata(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "source_file_label": self.source_file_label,
            "traj_length": int(len(self.actions)),
            "task_success": bool(self.task_success),
            "damage_by_object": {key: float(value) for key, value in self.damage_by_object.items()},
            "total_environment_damage": float(self.total_environment_damage),
            "termination_reason": self.termination_reason,
            "initial_scene_manifest": self.initial_scene_manifest,
        }


@dataclass(frozen=True)
class NormalizationStats:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    fitted_episode_ids: tuple[str, ...]

    def normalize_observations(self, values: np.ndarray) -> np.ndarray:
        return (values - self.observation_mean) / self.observation_std

    def normalize_actions(self, values: np.ndarray) -> np.ndarray:
        return (values - self.action_mean) / self.action_std

    def denormalize_actions(self, values: np.ndarray) -> np.ndarray:
        return values * self.action_std + self.action_mean

    def to_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            observation_mean=self.observation_mean,
            observation_std=self.observation_std,
            action_mean=self.action_mean,
            action_std=self.action_std,
            fitted_episode_ids=np.asarray(self.fitted_episode_ids),
        )
