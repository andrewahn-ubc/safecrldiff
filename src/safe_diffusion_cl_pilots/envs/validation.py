from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from safe_diffusion_cl_pilots.data.replay_oopsie_hdf5 import current_observation
from safe_diffusion_cl_pilots.data.schemas import OBJECT_ORDER

from .object_centric_obs import ObjectCentricObservation
from .state_enrichment import enrich_object_state


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_canonical(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    return repr(value)


def _nonplacement_config(env: Any, object_name: str) -> dict[str, Any]:
    """Return object identity / damage config with only placement fields removed."""
    configurations = env._get_obj_cfgs()
    for config in configurations:
        name = config.get("name", config.get("obj_name", config.get("object_name")))
        if name != object_name:
            continue
        result = copy.deepcopy(dict(config))
        for key in ("placement", "placement_initializer", "placement_config"):
            result.pop(key, None)
        for key in (
            "pos",
            "position",
            "size",
            "box",
            "sample_region_kwargs",
            "x_range",
            "y_range",
        ):
            result.pop(key, None)
        return result
    raise KeyError(f"missing object configuration for {object_name}")


def _reset(env: Any, seed: int) -> Mapping[str, Any]:
    try:
        result = env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()
    if isinstance(result, tuple):
        result = result[0]
    return enrich_object_state(env, result) if isinstance(result, Mapping) else current_observation(env)


def validate_context_pair(
    factory_a: Callable[..., Any],
    factory_b: Callable[..., Any],
    critical_fragile_object: str,
    reset_count: int = 100,
) -> dict[str, Any]:
    extractor = ObjectCentricObservation()
    env_a, env_b = factory_a(), factory_b()
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        for index in range(reset_count):
            obs_a = _reset(env_a, index)
            obs_b = _reset(env_b, index)
            vector_a, vector_b = extractor.extract(obs_a), extractor.extract(obs_b)
            checks["finite_resets"] = bool(np.isfinite(vector_a).all() and np.isfinite(vector_b).all())
            checks["same_observation_shape"] = vector_a.shape == vector_b.shape
            positions_a = {name: extractor.object_position(obs_a, name) for name in OBJECT_ORDER}
            positions_b = {name: extractor.object_position(obs_b, name) for name in OBJECT_ORDER}
            if index == 0:
                checks["cereal_pose_fixed"] = np.allclose(
                    positions_a["cereal"], positions_b["cereal"], atol=1e-4
                )
                mat_a = np.asarray(obs_a.get("target_mat_pos", obs_a.get("mat_pos")))
                mat_b = np.asarray(obs_b.get("target_mat_pos", obs_b.get("mat_pos")))
                checks["mat_pose_fixed"] = np.allclose(mat_a, mat_b, atol=1e-4)
                checks["critical_placement_changed"] = not np.allclose(
                    positions_a[critical_fragile_object],
                    positions_b[critical_fragile_object],
                    atol=1e-3,
                )
                intended = {critical_fragile_object, "flour_bag"}
                checks["only_intended_bystanders_changed"] = all(
                    np.allclose(positions_a[name], positions_b[name], atol=1e-4)
                    for name in OBJECT_ORDER
                    if name not in intended
                )
            for positions in (positions_a, positions_b):
                distances = [
                    np.linalg.norm(positions[left] - positions[right])
                    for left_index, left in enumerate(OBJECT_ORDER)
                    for right in OBJECT_ORDER[left_index + 1 :]
                ]
                checks["no_initial_overlap"] = checks.get("no_initial_overlap", True) and min(distances) > 0.01
                checks["objects_on_finite_table_region"] = checks.get(
                    "objects_on_finite_table_region", True
                ) and all(np.isfinite(value).all() and np.max(np.abs(value)) < 5.0 for value in positions.values())
        action_a, action_b = getattr(env_a, "action_spec", None), getattr(env_b, "action_spec", None)
        if callable(action_a):
            action_a, action_b = action_a(), action_b()
        checks["same_action_space"] = np.array_equal(np.asarray(action_a), np.asarray(action_b))
        checks["same_horizon"] = getattr(env_a, "horizon", None) == getattr(env_b, "horizon", None)
        checks["same_control_frequency"] = getattr(env_a, "control_freq", None) == getattr(
            env_b, "control_freq", None
        )
        checks["same_reward_function"] = type(env_a).reward is not None and (
            getattr(type(env_a), "reward", None).__code__.co_code
            == getattr(type(env_b), "reward", None).__code__.co_code
        )
        checks["damage_identity_unchanged"] = _canonical(
            _nonplacement_config(env_a, critical_fragile_object)
        ) == _canonical(_nonplacement_config(env_b, critical_fragile_object))
    except Exception as error:  # report all simulator/API drift through the gate
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        env_a.close()
        env_b.close()
    return {"passed": not errors and all(checks.values()), "checks": checks, "errors": errors}


def write_reset_image(env_factory: Callable[..., Any], path: Path, seed: int = 0) -> None:
    env = env_factory(has_offscreen_renderer=True, use_camera_obs=False)
    try:
        _reset(env, seed)
        frame = env.render()
        if frame is None and hasattr(env, "sim"):
            frame = env.sim.render(height=512, width=512, camera_name="agentview")
        array = np.asarray(frame)
        if array.ndim != 3:
            raise RuntimeError("environment did not return an RGB reset image")
        from matplotlib import image as mpimage

        path.parent.mkdir(parents=True, exist_ok=True)
        mpimage.imsave(path, array)
    finally:
        env.close()
