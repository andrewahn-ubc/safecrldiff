from __future__ import annotations

import ast
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = ROOT / "requirements.lock.txt"
BOOTSTRAP = ROOT / "scripts" / "bootstrap.py"


def _locked_requirements() -> list[Requirement]:
    return [
        Requirement(line.strip())
        for line in LOCK_FILE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_every_lock_entry_is_exact_and_unique() -> None:
    requirements = _locked_requirements()
    names = [canonicalize_name(requirement.name) for requirement in requirements]
    assert len(names) == len(set(names))
    for requirement in requirements:
        specs = list(requirement.specifier)
        assert len(specs) == 1
        assert specs[0].operator == "=="
        assert not specs[0].version.endswith(".*")


def test_bootstrap_pins_installed_vendor_repositories() -> None:
    tree = ast.parse(BOOTSTRAP.read_text())
    repositories = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REPOSITORIES" for target in node.targets)
    )
    for _, requested, installed in repositories.values():
        if installed:
            assert requested is not None
            assert len(requested) >= 7


def test_direct_native_wheels_are_hash_pinned() -> None:
    tree = ast.parse(BOOTSTRAP.read_text())
    wheels = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MANYLINUX_WHEELS" for target in node.targets)
    )
    assert {
        "pyarrow",
        "opencv-python-headless",
        "mujoco",
        "lxml",
        "llvmlite",
        "numba",
    } == set(wheels)
    for version, url in wheels.values():
        assert version
        assert url.startswith("https://files.pythonhosted.org/")
        assert ".whl#sha256=" in url
        assert len(url.rsplit("#sha256=", 1)[1]) == 64
