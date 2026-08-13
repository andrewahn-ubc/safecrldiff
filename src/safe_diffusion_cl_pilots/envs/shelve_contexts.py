from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .state_enrichment import enrich_object_state

try:
    from oopsiebench.envs.robocasa.shelve_item import DamageableShelveItem as _BaseShelveItem
except ImportError:
    class _BaseShelveItem:  # type: ignore[no-redef]
        """Import-safe placeholder; construction reports the missing simulator."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "OopsieVerse is not installed. Run scripts/bootstrap.py on the Narval login node."
            )


PLACEMENT_KEYS = ("placement", "placement_initializer", "placement_config")
PLACEMENT_SWAP_FIELDS = (
    "pos",
    "position",
    "size",
    "box",
    "sample_region_kwargs",
    "x_range",
    "y_range",
)


def _name(config: Mapping[str, Any]) -> str:
    for key in ("name", "obj_name", "object_name"):
        if key in config:
            return str(config[key])
    raise KeyError(f"object configuration has no name: {config.keys()}")


def swap_object_placements(
    configurations: Sequence[Mapping[str, Any]],
    fragile_name: str,
    flour_name: str = "flour_bag",
) -> list[dict[str, Any]]:
    """Deep-copy configs and swap only complete placement specifications."""
    result = copy.deepcopy(list(configurations))
    by_name = {_name(config): config for config in result}
    if fragile_name not in by_name or flour_name not in by_name:
        raise KeyError(f"cannot swap {fragile_name} with {flour_name}; found {sorted(by_name)}")
    fragile = by_name[fragile_name]
    flour = by_name[flour_name]
    present_keys = [key for key in PLACEMENT_KEYS if key in fragile or key in flour]
    if present_keys:
        for key in present_keys:
            if key not in fragile or key not in flour:
                raise ValueError(f"placement key {key} is not shared by both objects")
            fragile_placement, flour_placement = fragile[key], flour[key]
            if not isinstance(fragile_placement, dict) or not isinstance(flour_placement, dict):
                raise ValueError(f"placement key {key} must contain mappings")
            fields = [
                field
                for field in PLACEMENT_SWAP_FIELDS
                if field in fragile_placement or field in flour_placement
            ]
            if not fields:
                raise ValueError(f"placement mapping {key} has no position or sampling-box fields")
            for field in fields:
                if field not in fragile_placement or field not in flour_placement:
                    raise ValueError(f"placement field {field} is not shared by both objects")
                fragile_placement[field], flour_placement[field] = (
                    copy.deepcopy(flour_placement[field]),
                    copy.deepcopy(fragile_placement[field]),
                )
    else:
        coordinate_keys = PLACEMENT_SWAP_FIELDS
        keys = [key for key in coordinate_keys if key in fragile or key in flour]
        if not keys:
            raise ValueError("object configurations expose no recognized placement data")
        for key in keys:
            if key not in fragile or key not in flour:
                raise ValueError(f"placement field {key} is not shared by both objects")
            fragile[key], flour[key] = copy.deepcopy(flour[key]), copy.deepcopy(fragile[key])
    return result


class _LowDimStateMixin:
    def _get_observations(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        observation = super()._get_observations(*args, **kwargs)
        return enrich_object_state(self, observation)


class DamageableShelveItemContextA(_LowDimStateMixin, _BaseShelveItem):
    """Unmodified canonical OopsieBench Shelve Item context."""


class DamageableShelveItemContextB(_LowDimStateMixin, _BaseShelveItem):
    """Context with the selected fragile placement swapped with the flour bag."""

    def __init__(self, *args: Any, critical_fragile_object: str = "wine_glass", **kwargs: Any):
        self.critical_fragile_object = critical_fragile_object
        super().__init__(*args, **kwargs)

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        parent = super()._get_obj_cfgs()
        return swap_object_placements(parent, self.critical_fragile_object)


def make_context(
    context: str,
    critical_fragile_object: str = "wine_glass",
    **kwargs: Any,
) -> Any:
    common = {
        "robots": "Panda",
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        **kwargs,
    }
    if context.upper() == "A":
        return DamageableShelveItemContextA(**common)
    if context.upper() == "B":
        return DamageableShelveItemContextB(
            critical_fragile_object=critical_fragile_object, **common
        )
    raise ValueError("context must be A or B")


ENVIRONMENT_METADATA_KEYS = (
    "robots",
    "controller_configs",
    "gripper_types",
    "initialization_noise",
    "use_object_obs",
    "reward_scale",
    "reward_shaping",
    "control_freq",
    "horizon",
    "ignore_done",
    "hard_reset",
)


def environment_kwargs_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    source = metadata.get("env_kwargs", metadata)
    if not isinstance(source, Mapping):
        return {}
    return {key: copy.deepcopy(source[key]) for key in ENVIRONMENT_METADATA_KEYS if key in source}


def make_context_from_metadata(
    metadata: Mapping[str, Any],
    context: str = "A",
    critical_fragile_object: str = "wine_glass",
    **overrides: Any,
) -> Any:
    kwargs = environment_kwargs_from_metadata(metadata)
    kwargs.update(overrides)
    return make_context(context, critical_fragile_object, **kwargs)


def placements_differ(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]], name: str) -> bool:
    return not np.allclose(np.asarray(a[name]), np.asarray(b[name]), atol=1e-5, rtol=0.0)
