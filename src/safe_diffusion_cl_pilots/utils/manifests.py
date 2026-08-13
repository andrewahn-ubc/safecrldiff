from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from .logging import write_json


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_success(directory: Path, payload: dict[str, Any] | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "_SUCCESS"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload or {}, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def git_sha(repository: Path) -> str | None:
    if not (repository / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_versions(project_root: Path) -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        check=False,
        text=True,
    ).stdout.splitlines()
    repositories = {
        "oopsieverse": {
            "url": "https://github.com/UT-Austin-RobIn/oopsieverse.git",
            "sha": git_sha(project_root / "vendor" / "oopsieverse"),
        },
        "dppo": {
            "url": "https://github.com/irom-princeton/dppo.git",
            "sha": git_sha(project_root / "vendor" / "dppo"),
        },
        "robosuite": {
            "url": "https://github.com/ARISE-Initiative/robosuite.git",
            "sha": git_sha(project_root / "vendor" / "robosuite"),
        },
        "robocasa": {
            "url": "https://github.com/robocasa/robocasa.git",
            "sha": git_sha(project_root / "vendor" / "robocasa"),
        },
    }
    return {
        "repositories": repositories,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cuda": getattr(__import__("torch").version, "cuda", None),
        "packages": {
            name: package_version(name)
            for name in ["torch", "mujoco", "robocasa", "robosuite", "oopsieverse"]
        },
        "pip_freeze": freeze,
        "slurm": {key: value for key, value in os.environ.items() if key.startswith("SLURM_")},
    }


def write_source_versions(project_root: Path, run_root: Path) -> Path:
    path = run_root / "artifacts" / "source_versions.json"
    current = source_versions(project_root)
    try:
        existing = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}
    current["repositories"] = {**existing.get("repositories", {}), **current["repositories"]}
    for key in (
        "inspected_documentation",
        "dataset",
        "working_tree_patches",
        "dppo_adapter",
    ):
        if key in existing:
            current[key] = existing[key]
    write_json(path, current)
    return path
