from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from safe_diffusion_cl_pilots.data.schemas import OBJECT_ORDER

PROHIBITED_TOKENS = ("health", "damage", "contact_force", "source_file", "context_id")


def _first(mapping: Mapping[str, Any], aliases: Sequence[str], description: str) -> np.ndarray:
    for alias in aliases:
        if alias in mapping:
            result = np.asarray(mapping[alias], dtype=np.float32).reshape(-1)
            if not np.isfinite(result).all():
                raise ValueError(f"non-finite {description} in observation key {alias}")
            return result
    raise KeyError(f"missing {description}; tried {', '.join(aliases)}")


@dataclass(frozen=True)
class ObjectCentricObservation:
    object_order: tuple[str, ...] = OBJECT_ORDER
    robot_dof: int = 7

    def _robot(self, observation: Mapping[str, Any], stem: str) -> np.ndarray:
        aliases = {
            "qpos": ("robot0_joint_pos", "robot_joint_pos", "joint_positions", "qpos"),
            "qvel": ("robot0_joint_vel", "robot_joint_vel", "joint_velocities", "qvel"),
            "eef_pos": ("robot0_eef_pos", "eef_pos", "end_effector_pos"),
            "eef_quat": ("robot0_eef_quat", "eef_quat", "end_effector_quat"),
            "gripper": ("robot0_gripper_qpos", "gripper_qpos", "gripper_state"),
        }
        value = _first(observation, aliases[stem], stem)
        expected = {"qpos": self.robot_dof, "qvel": self.robot_dof, "eef_pos": 3, "eef_quat": 4}
        if stem in expected and value.size < expected[stem]:
            raise ValueError(f"{stem} has {value.size} values; expected at least {expected[stem]}")
        if stem in ("qpos", "qvel"):
            return value[: self.robot_dof]
        if stem == "gripper":
            return np.asarray([value.mean()], dtype=np.float32)
        return value[: expected[stem]]

    def _object(self, observation: Mapping[str, Any], name: str, field: str) -> np.ndarray:
        aliases_by_field = {
            "pos": (f"{name}_pos", f"{name}_position"),
            "quat": (f"{name}_quat", f"{name}_orientation"),
            "linvel": (f"{name}_linvel", f"{name}_linear_velocity", f"{name}_velp"),
            "angvel": (f"{name}_angvel", f"{name}_angular_velocity", f"{name}_velr"),
        }
        value = _first(observation, aliases_by_field[field], f"{name} {field}")
        expected = 4 if field == "quat" else 3
        if value.size < expected:
            raise ValueError(f"{name} {field} has {value.size} values; expected {expected}")
        return value[:expected]

    def object_position(self, observation: Mapping[str, Any], name: str) -> np.ndarray:
        return self._object(observation, name, "pos")

    def end_effector_position(self, observation: Mapping[str, Any]) -> np.ndarray:
        return self._robot(observation, "eef_pos")

    def extract(self, observation: Mapping[str, Any]) -> np.ndarray:
        for key in observation:
            lowered = key.lower()
            if any(token in lowered for token in PROHIBITED_TOKENS):
                # Prohibited simulator values may exist in the source mapping. They
                # are deliberately ignored; the output schema below is an allowlist.
                continue
        eef = self._robot(observation, "eef_pos")
        pieces = [
            self._robot(observation, "qpos"),
            self._robot(observation, "qvel"),
            eef,
            self._robot(observation, "eef_quat"),
            self._robot(observation, "gripper"),
        ]
        for name in self.object_order:
            position = self._object(observation, name, "pos")
            pieces.extend(
                [
                    position,
                    self._object(observation, name, "quat"),
                    self._object(observation, name, "linvel"),
                    self._object(observation, name, "angvel"),
                    position - eef,
                ]
            )
        pieces.extend(
            [
                _first(observation, ("target_mat_pos", "mat_pos", "target_pos"), "target mat pos")[:3],
                _first(
                    observation,
                    ("target_mat_quat", "mat_quat", "target_orientation"),
                    "target mat orientation",
                )[:4],
            ]
        )
        vector = np.concatenate(pieces).astype(np.float32)
        if vector.shape != (self.dimension,):
            raise ValueError(f"observation schema produced {vector.shape}, expected {(self.dimension,)}")
        return vector

    @property
    def dimension(self) -> int:
        robot = 2 * self.robot_dof + 3 + 4 + 1
        per_object = 3 + 4 + 3 + 3 + 3
        return robot + len(self.object_order) * per_object + 3 + 4

    @property
    def feature_names(self) -> list[str]:
        names = [*(f"robot_joint_pos_{i}" for i in range(self.robot_dof))]
        names += [*(f"robot_joint_vel_{i}" for i in range(self.robot_dof))]
        names += [*(f"eef_pos_{axis}" for axis in "xyz")]
        names += [*(f"eef_quat_{i}" for i in range(4)), "gripper_state"]
        for name in self.object_order:
            names += [*(f"{name}_pos_{axis}" for axis in "xyz")]
            names += [*(f"{name}_quat_{i}" for i in range(4))]
            names += [*(f"{name}_linvel_{axis}" for axis in "xyz")]
            names += [*(f"{name}_angvel_{axis}" for axis in "xyz")]
            names += [*(f"{name}_relative_to_eef_{axis}" for axis in "xyz")]
        names += [*(f"target_mat_pos_{axis}" for axis in "xyz")]
        names += [*(f"target_mat_quat_{i}" for i in range(4))]
        return names

    def assert_no_leakage(self) -> None:
        offending = [
            name for name in self.feature_names if any(token in name.lower() for token in PROHIBITED_TOKENS)
        ]
        if offending:
            raise AssertionError(f"prohibited observation features: {offending}")

