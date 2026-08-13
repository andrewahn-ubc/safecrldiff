from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def project_config(project_root: Path, name: str) -> dict[str, Any]:
    return load_yaml(project_root / "configs" / f"{name}.yaml")

