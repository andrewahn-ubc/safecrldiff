from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from safe_diffusion_cl_pilots.envs.object_centric_obs import ObjectCentricObservation
from safe_diffusion_cl_pilots.envs.state_enrichment import enrich_object_state

from .schemas import OBJECT_ORDER, EpisodeRecord


class ReplayIntegrityError(RuntimeError):
    """A deterministic released-data or simulator alignment failure."""


@dataclass(frozen=True)
class RawEpisode:
    episode_id: str
    states: np.ndarray
    actions: np.ndarray


def aligned_state_actions(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply OopsieVerse playback convention: states[:-1] pairs with actions[1:]."""
    states = np.asarray(states)
    actions = np.asarray(actions)
    if states.ndim != 2 or actions.ndim != 2:
        raise ReplayIntegrityError("states and actions must both be rank-two arrays")
    if len(states) != len(actions):
        raise ReplayIntegrityError(
            f"expected equal recorded state/action counts, got {len(states)} and {len(actions)}"
        )
    if len(states) < 2 or actions.shape[1] != 7:
        raise ReplayIntegrityError(f"invalid recorded shapes: {states.shape}, {actions.shape}")
    paired_states = states[:-1]
    paired_actions = actions[1:]
    if len(paired_states) != len(paired_actions):
        raise AssertionError("official replay alignment produced unequal lengths")
    if not np.isfinite(paired_states).all() or not np.isfinite(paired_actions).all():
        raise ReplayIntegrityError("recorded state or action contains NaN/Inf")
    return paired_states, paired_actions


class OopsieHDF5:
    """Minimal reader supporting the released RoboCasa `data/demo_*` layout."""

    def __init__(self, path: Path):
        self.path = path

    def _open(self) -> Any:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("h5py is required for OopsieVerse replay") from error
        return h5py.File(self.path, "r")

    def episode_ids(self) -> list[str]:
        with self._open() as archive:
            root = archive["data"] if "data" in archive else archive
            ids = [key for key, value in root.items() if hasattr(value, "keys")]
        return sorted(ids)

    def read(self, episode_id: str) -> RawEpisode:
        with self._open() as archive:
            root = archive["data"] if "data" in archive else archive
            if episode_id not in root:
                raise ReplayIntegrityError(f"missing episode {episode_id} in {self.path}")
            group = root[episode_id]
            if "states" not in group or "actions" not in group:
                raise ReplayIntegrityError(f"{episode_id} lacks states/actions")
            return RawEpisode(
                episode_id=episode_id,
                states=np.asarray(group["states"]),
                actions=np.asarray(group["actions"]),
            )

    def environment_metadata(self) -> dict[str, Any]:
        with self._open() as archive:
            root = archive["data"] if "data" in archive else archive
            raw = root.attrs.get("env_args", archive.attrs.get("env_args", "{}"))
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            return json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (TypeError, ValueError):
            return {"raw_env_args": str(raw)}


def _unwrap(env: Any) -> Any:
    while hasattr(env, "env"):
        env = env.env
    return env


def restore_flattened_state(env: Any, state: np.ndarray) -> None:
    base = _unwrap(env)
    simulator = getattr(base, "sim", None)
    if simulator is None:
        raise ReplayIntegrityError("environment has no MuJoCo simulator handle")
    if hasattr(simulator, "set_state_from_flattened"):
        simulator.set_state_from_flattened(state)
    elif hasattr(simulator, "set_state"):
        simulator.set_state(state)
    else:
        raise ReplayIntegrityError("simulator cannot restore flattened states")
    simulator.forward()


def current_observation(env: Any) -> Mapping[str, Any]:
    base = _unwrap(env)
    public = getattr(base, "get_observations", None)
    if public is not None:
        value = public()
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, Mapping):
            return enrich_object_state(base, value)
    for method_name in ("_get_observations", "_get_observation"):
        method = getattr(base, method_name, None)
        if method is not None:
            try:
                value = method(force_update=True)
            except TypeError:
                value = method()
            if isinstance(value, Mapping):
                return enrich_object_state(base, value)
    raise ReplayIntegrityError("environment cannot expose an observation dictionary")


def task_success(env: Any, info: Mapping[str, Any] | None = None) -> bool:
    if info:
        for key in ("task_success", "success", "is_success"):
            if key in info:
                return bool(info[key])
    base = _unwrap(env)
    for method_name in ("_check_success", "check_success"):
        method = getattr(base, method_name, None)
        if method is not None:
            return bool(method())
    raise ReplayIntegrityError("task-success signal is inaccessible")


def _health_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("health", "current_health", "overall_health", "remaining_health"):
            if key in value:
                return float(value[key])
    if isinstance(value, int | float | np.number):
        return float(value)
    return None


def _health_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for name, details in value.items():
        health = _health_value(details)
        if health is not None:
            result[str(name)] = health
    return result


def object_health(env: Any, info: Mapping[str, Any] | None = None) -> dict[str, float]:
    if info:
        for key in ("per_object_health", "health_by_object", "object_health", "damage_info"):
            value = info.get(key)
            parsed = _health_mapping(value)
            if parsed:
                return parsed
    base = _unwrap(env)
    for owner in (base, getattr(base, "damage_manager", None), getattr(base, "damage_sim", None)):
        if owner is None:
            continue
        for attribute in ("get_object_health", "get_health", "object_health", "health"):
            value = getattr(owner, attribute, None)
            if callable(value):
                value = value()
            if isinstance(value, Mapping):
                parsed = _health_mapping(value)
                if parsed:
                    return parsed
    get_observations = getattr(base, "get_observations", None)
    if callable(get_observations):
        value = get_observations()
        if isinstance(value, tuple) and len(value) == 2:
            parsed = _health_mapping(value[1].get("damage_info", {}))
            if parsed:
                return parsed
    get_objects = getattr(base, "get_damageable_objects", None)
    if callable(get_objects):
        result: dict[str, float] = {}
        for item in get_objects():
            health = _health_value(getattr(item, "damage_info", None))
            if health is None:
                health = _health_value(getattr(item, "health", None))
            if health is not None:
                result[str(item.name)] = health
        if result:
            return result
    raise ReplayIntegrityError("DamageSim per-object health signal is inaccessible")


def scene_manifest(env: Any) -> dict[str, Any]:
    base = _unwrap(env)
    result: dict[str, Any] = {}
    try:
        observation = current_observation(base)
    except ReplayIntegrityError:
        return result
    for name in (*OBJECT_ORDER, "target_mat"):
        for suffix in ("_pos", "_position"):
            key = name + suffix
            if key in observation:
                result[name] = np.asarray(observation[key], dtype=float).tolist()
                break
    return result


def _step(env: Any, action: np.ndarray) -> tuple[float, bool, bool, dict[str, Any]]:
    outcome = env.step(action)
    if len(outcome) == 5:
        _, reward, terminated, truncated, info = outcome
    elif len(outcome) == 4:
        _, reward, done, info = outcome
        terminated, truncated = bool(done), False
    else:
        raise ReplayIntegrityError("environment step did not return Gym/Gymnasium format")
    return float(reward), bool(terminated), bool(truncated), dict(info or {})


def replay_episode(
    env: Any,
    raw: RawEpisode,
    source_file_label: str,
    extractor: ObjectCentricObservation | None = None,
) -> EpisodeRecord:
    extractor = extractor or ObjectCentricObservation()
    states, actions = aligned_state_actions(raw.states, raw.actions)
    env.reset()
    initial_manifest: dict[str, Any] = {}
    observations: list[np.ndarray] = []
    rewards: list[float] = []
    health_values: dict[str, list[float]] = {}
    cereal: list[np.ndarray] = []
    end_effector: list[np.ndarray] = []
    termination_reason = "recorded_horizon"
    last_info: dict[str, Any] = {}
    success = False
    initial_health: dict[str, float] | None = None
    for index, (state, action) in enumerate(zip(states, actions, strict=True)):
        restore_flattened_state(env, state)
        observation_dict = current_observation(env)
        if index == 0:
            initial_manifest = scene_manifest(env)
            initial_health = object_health(env)
        observations.append(extractor.extract(observation_dict))
        cereal.append(extractor.object_position(observation_dict, "cereal"))
        end_effector.append(extractor.end_effector_position(observation_dict))
        reward, terminated, truncated, last_info = _step(env, np.clip(action, -1.0, 1.0))
        rewards.append(reward)
        health = object_health(env, last_info)
        for name, value in health.items():
            health_values.setdefault(name, []).append(value)
        success = success or task_success(env, last_info)
        if terminated or truncated:
            termination_reason = "terminated" if terminated else "truncated"
            # State restoration makes later recorded pairs independent, so retaining
            # them is required to preserve the released trajectory length.
    if not health_values or any(len(values) != len(actions) for values in health_values.values()):
        raise ReplayIntegrityError(f"{raw.episode_id}: health unavailable at every aligned step")
    health_arrays = {name: np.asarray(values, dtype=np.float32) for name, values in health_values.items()}
    initial_health = initial_health or {name: float(values[0]) for name, values in health_arrays.items()}
    damage = {
        name: float(max(0.0, initial_health.get(name, float(values[0])) - values[-1]))
        for name, values in health_arrays.items()
    }
    record = EpisodeRecord(
        episode_id=raw.episode_id,
        source_file_label=source_file_label,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        task_success=success,
        health_by_object=health_arrays,
        damage_by_object=damage,
        total_environment_damage=float(sum(damage.values())),
        termination_reason=termination_reason,
        initial_scene_manifest=initial_manifest,
        cereal_trajectory=np.asarray(cereal, dtype=np.float32),
        end_effector_trajectory=np.asarray(end_effector, dtype=np.float32),
    )
    record.validate()
    return record


def replay_file(
    path: Path,
    source_file_label: str,
    env_factory: Callable[[], Any],
    episode_ids: Sequence[str] | None = None,
) -> tuple[list[EpisodeRecord], list[dict[str, str]]]:
    archive = OopsieHDF5(path)
    selected = archive.episode_ids() if episode_ids is None else list(episode_ids)
    env = env_factory()
    records: list[EpisodeRecord] = []
    rejected: list[dict[str, str]] = []
    try:
        for episode_id in selected:
            try:
                record = replay_episode(env, archive.read(episode_id), source_file_label)
                record.episode_id = f"{source_file_label}:{episode_id}"
                records.append(record)
            except (ReplayIntegrityError, ValueError, FloatingPointError) as error:
                rejected.append({"episode_id": episode_id, "reason": f"{type(error).__name__}: {error}"})
    finally:
        close = getattr(env, "close", None)
        if close:
            close()
    return records, rejected


def iter_raw_episodes(path: Path) -> Iterator[RawEpisode]:
    archive = OopsieHDF5(path)
    for episode_id in archive.episode_ids():
        yield archive.read(episode_id)
