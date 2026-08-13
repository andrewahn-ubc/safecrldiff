from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .state_enrichment import enrich_object_state


class _ArrayBox:
    """Small Gym Box-compatible fallback for import-only test environments."""

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape
        self.dtype = np.dtype(np.float32)

    def contains(self, value: Any) -> bool:
        array = np.asarray(value)
        return array.shape == self.shape and bool(
            np.all(array >= self.low) and np.all(array <= self.high)
        )


def _chunk_action_space(env: Any, horizon: int) -> Any:
    low = np.full((horizon, 7), -1.0, dtype=np.float32)
    high = np.full((horizon, 7), 1.0, dtype=np.float32)
    try:
        from gymnasium.spaces import Box

        return Box(low=low, high=high, dtype=np.float32)
    except ImportError:
        return _ArrayBox(low, high)


class ChunkedGymWrapper:
    """Execute an Hx7 action through one environment call per primitive action."""

    def __init__(
        self,
        env: Any,
        horizon: int = 8,
        critical_fragile_object: str = "wine_glass",
        epsilon_damage: float = 1.0,
    ):
        self.env = env
        self.horizon = horizon
        self.critical_fragile_object = critical_fragile_object
        self.epsilon_damage = epsilon_damage
        self.primitive_action_space = getattr(env, "action_space", None)
        self.action_space = _chunk_action_space(env, horizon)
        self.observation_space = getattr(env, "observation_space", None)
        self._last_health: dict[str, float] = {}
        self._episode_losses: dict[str, float] = {}
        self._episode_cereal_trajectory: list[Any] = []
        self._episode_eef_trajectory: list[Any] = []
        self._episode_steps = 0
        self.frame_collector: Callable[[Any], None] | None = None

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        try:
            value = self.env.reset(*args, **kwargs)
        except TypeError:
            seed = kwargs.pop("seed", None)
            if seed is None or args:
                raise
            seed_method = getattr(self.env, "seed", None)
            if callable(seed_method):
                seed_method(seed)
            value = self.env.reset(**kwargs)
        if isinstance(value, tuple) and len(value) == 2:
            self._last_health = self._health(dict(value[1] or {}))
            self._episode_losses = {name: 0.0 for name in self._last_health}
            self._episode_cereal_trajectory = []
            self._episode_eef_trajectory = []
            self._episode_steps = 0
            observation = (
                enrich_object_state(self.env, value[0])
                if isinstance(value[0], Mapping)
                else value[0]
            )
            return observation, value[1]
        self._last_health = {}
        self._episode_losses = {}
        self._episode_cereal_trajectory = []
        self._episode_eef_trajectory = []
        self._episode_steps = 0
        return enrich_object_state(self.env, value) if isinstance(value, Mapping) else value

    @staticmethod
    def _health(info: dict[str, Any]) -> dict[str, float]:
        value = info.get(
            "per_object_health", info.get("health_by_object", info.get("damage_info", {}))
        )
        result: dict[str, float] = {}
        for key, item in value.items() if isinstance(value, dict) else ():
            if isinstance(item, dict):
                item = next(
                    (
                        item[name]
                        for name in ("health", "current_health", "overall_health", "remaining_health")
                        if name in item
                    ),
                    None,
                )
            if item is not None:
                result[str(key)] = float(item)
        return result

    def step(self, action_chunk: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.shape == (self.horizon * 7,):
            chunk = chunk.reshape(self.horizon, 7)
        if chunk.shape != (self.horizon, 7):
            raise ValueError(f"expected {(self.horizon, 7)} or flattened equivalent, got {chunk.shape}")
        chunk = np.clip(chunk, -1.0, 1.0)
        total_reward = 0.0
        terminated = truncated = False
        observation: Any = None
        first_health: dict[str, float] | None = dict(self._last_health) or None
        last_health: dict[str, float] = {}
        cereal_trajectory: list[Any] = []
        eef_trajectory: list[Any] = []
        contacted: set[str] = set()
        success = False
        final_info: dict[str, Any] = {}
        executed = 0
        for primitive_action in chunk:
            outcome = self.env.step(primitive_action)
            if len(outcome) == 5:
                observation, reward, terminated, truncated, info = outcome
            elif len(outcome) == 4:
                observation, reward, done, info = outcome
                terminated, truncated = bool(done), False
            else:
                raise RuntimeError("primitive environment returned an invalid step tuple")
            if isinstance(observation, Mapping):
                observation = enrich_object_state(self.env, observation)
                if "cereal_pos" in observation:
                    self._episode_cereal_trajectory.append(
                        np.asarray(observation["cereal_pos"], dtype=float).tolist()
                    )
                for eef_key in ("robot0_eef_pos", "eef_pos", "end_effector_pos"):
                    if eef_key in observation:
                        self._episode_eef_trajectory.append(
                            np.asarray(observation[eef_key], dtype=float).tolist()
                        )
                        break
            final_info = dict(info or {})
            if self.frame_collector is not None:
                self.frame_collector(self.env)
            current_health = self._health(final_info)
            if first_health is None and current_health:
                initial = final_info.get("initial_per_object_health")
                first_health = (
                    {str(key): float(value) for key, value in initial.items()}
                    if isinstance(initial, dict)
                    else dict(current_health)
                )
            last_health = current_health or last_health
            total_reward += float(reward)
            executed += 1
            self._episode_steps += 1
            success = bool(
                final_info.get("task_success", final_info.get("success", success))
            )
            if "cereal_position" in final_info:
                cereal_trajectory.append(final_info["cereal_position"])
            if "end_effector_position" in final_info:
                eef_trajectory.append(final_info["end_effector_position"])
            contacted.update(map(str, final_info.get("contacted_objects", [])))
            if terminated or truncated:
                break
        first_health = first_health or last_health
        losses = {
            name: max(0.0, first_health.get(name, value) - value)
            for name, value in last_health.items()
        }
        for name, loss in losses.items():
            self._episode_losses[name] = self._episode_losses.get(name, 0.0) + loss
        self._last_health = dict(last_health)
        fragile_loss = self._episode_losses.get(self.critical_fragile_object, 0.0)
        final_info.update(
            {
                "primitive_steps_executed": executed,
                "task_success": success,
                "designated_fragile_health_loss": fragile_loss,
                "damage_event": fragile_loss >= self.epsilon_damage,
                "total_environment_damage": float(sum(self._episode_losses.values())),
                "per_object_health": last_health,
                "per_object_health_loss": losses,
                "episode_per_object_health_loss": dict(self._episode_losses),
                "chunk_total_environment_damage": float(sum(losses.values())),
                "cereal_trajectory": self._episode_cereal_trajectory or cereal_trajectory,
                "end_effector_trajectory": self._episode_eef_trajectory or eef_trajectory,
                "episode_primitive_steps": self._episode_steps,
                "contacted_objects": sorted(contacted),
            }
        )
        return observation, total_reward, bool(terminated), bool(truncated), final_info

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close:
            close()
