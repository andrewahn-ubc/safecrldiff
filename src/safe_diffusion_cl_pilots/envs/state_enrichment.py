from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from safe_diffusion_cl_pilots.data.schemas import OBJECT_ORDER


def _body_velocity(simulator: Any, body_name: str, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    data = simulator.data
    if hasattr(data, "get_body_xvelp") and hasattr(data, "get_body_xvelr"):
        return np.asarray(data.get_body_xvelp(body_name)), np.asarray(data.get_body_xvelr(body_name))
    spatial = np.asarray(data.cvel[body_id])
    # MuJoCo spatial velocities use rotational then translational components.
    return spatial[3:6], spatial[0:3]


def enrich_object_state(env: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
    """Add the predeclared low-dimensional fields directly from MuJoCo state."""
    result = dict(observation)
    simulator = getattr(env, "sim", None)
    body_ids = getattr(env, "obj_body_id", {})
    if simulator is None:
        return result
    for name in OBJECT_ORDER:
        if name not in body_ids:
            continue
        body_id = int(body_ids[name])
        body_name = simulator.model.body_id2name(body_id)
        result[f"{name}_pos"] = np.asarray(simulator.data.body_xpos[body_id]).copy()
        result[f"{name}_quat"] = np.asarray(simulator.data.body_xquat[body_id]).copy()
        linear, angular = _body_velocity(simulator, body_name, body_id)
        result[f"{name}_linvel"] = linear.copy()
        result[f"{name}_angvel"] = angular.copy()
    get_mat_position = getattr(env, "_get_mat_pos", None)
    if callable(get_mat_position):
        result["target_mat_pos"] = np.asarray(get_mat_position()).copy()
    try:
        mat_id = simulator.model.body_name2id("table_mat_main")
        result["target_mat_quat"] = np.asarray(simulator.data.body_xquat[mat_id]).copy()
    except Exception:
        result.setdefault("target_mat_quat", np.asarray([1.0, 0.0, 0.0, 0.0]))
    return result

