from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = ROOT / "requirements.lock.txt"
BOOTSTRAP = ROOT / "scripts" / "bootstrap.py"
LAUNCHER = ROOT / "run_pilots.sh"


def _bootstrap_module():
    specification = importlib.util.spec_from_file_location("safe_pilots_bootstrap", BOOTSTRAP)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


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
        "glfw",
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


def test_mujoco_backends_match_node_capabilities() -> None:
    assert "export MUJOCO_GL=disable" in LAUNCHER.read_text()
    for name in ("pilot_minus1_cpu.sbatch", "pilot0_cpu.sbatch", "aggregate_cpu.sbatch"):
        script = (ROOT / "slurm" / name).read_text()
        assert "export MUJOCO_GL=disable" in script
        assert "export PYOPENGL_PLATFORM=egl" not in script
    gpu_script = (ROOT / "slurm" / "pilots12_gpu_array.sbatch").read_text()
    assert "export MUJOCO_GL=egl" in gpu_script
    assert "export PYOPENGL_PLATFORM=egl" in gpu_script


def test_robocasa_compatibility_patch_is_exact_and_idempotent(tmp_path: Path) -> None:
    robocasa = tmp_path / "robocasa"
    package = robocasa / "robocasa"
    package.mkdir(parents=True)
    setup_py = robocasa / "setup.py"
    init_py = package / "__init__.py"
    setup_py.write_text(
        'deps = ["numpy==2.2.5", "numba==0.61.2", "scipy==1.15.3"]\n'
    )
    init_py.write_text(
        'assert numpy.__version__ in ["2.2.5"], "numpy version must be 2.2.5"\n'
    )
    bootstrap = _bootstrap_module()
    bootstrap.patch_robocasa_compatibility(robocasa)
    bootstrap.patch_robocasa_compatibility(robocasa)
    assert "numpy==1.26.4" in setup_py.read_text()
    assert "numba>=0.61.2" in setup_py.read_text()
    assert "scipy==1.15.1" in setup_py.read_text()
    assert '"1.26.4"' in init_py.read_text()
    assert "numpy version must be 1.26.4" in init_py.read_text()

    init_py.write_text('assert numpy.__version__ == "unexpected"\n')
    with pytest.raises(RuntimeError, match="audited compatibility patch"):
        bootstrap.patch_robocasa_compatibility(robocasa)
