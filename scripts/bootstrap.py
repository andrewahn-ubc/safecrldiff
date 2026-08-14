from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

REPOSITORIES = {
    "oopsieverse": (
        "https://github.com/UT-Austin-RobIn/oopsieverse.git",
        "151efcee2200e3ec1ad76524a5961aef15ce5f28",
        True,
    ),
    "dppo": (
        "https://github.com/irom-princeton/dppo.git",
        "cc7234ad7ff39a8f32de3af903606723a16f0648",
        True,
    ),
    "robosuite": ("https://github.com/ARISE-Initiative/robosuite.git", "aaa8b9b", True),
    "robocasa": ("https://github.com/robocasa/robocasa.git", "97a4060", True),
    "dsrl_reference": ("https://github.com/ajwagen/dsrl.git", None, False),
    "robocasa_diffusion_policy_reference": (
        "https://github.com/robocasa-benchmark/diffusion_policy.git",
        None,
        False,
    ),
}

# Narval advertises module-sentinel packages for PyArrow and OpenCV, and its
# Gentoo Python does not advertise standard manylinux tags. Install the exact
# public CPython/Linux wheels into the venv for the native packages that are
# absent from its wheelhouse. Pip verifies every PyPI-published digest.
MANYLINUX_WHEELS = {
    "glfw": (
        "2.10.0",
        "https://files.pythonhosted.org/packages/e2/fa/"
        "b035636cd82198b97b51a93efe9cfc4343d6b15cefbd336a3f2be871d848/"
        "glfw-2.10.0-py2.py27.py3.py30.py31.py32.py33.py34.py35.py36.py37."
        "py38.py39.py310.py311.py312.py313.py314-none-manylinux2014_x86_64.whl"
        "#sha256=91d36b3582a766512eff8e3b5dcc2d3ffcbf10b7cf448551085a08a10f1b8244",
    ),
    "pyarrow": (
        "17.0.0",
        "https://files.pythonhosted.org/packages/18/4c/"
        "3db637d7578f683b0a8fb8999b436bdbedd6e3517bd4f90c70853cf3ad20/"
        "pyarrow-17.0.0-cp310-cp310-manylinux_2_17_x86_64."
        "manylinux2014_x86_64.whl"
        "#sha256=75c06d4624c0ad6674364bb46ef38c3132768139ddec1c56582dbac54f2663e2",
    ),
    "opencv-python-headless": (
        "4.10.0.84",
        "https://files.pythonhosted.org/packages/d1/09/"
        "248f86a404567303cdf120e4a301f389b68e3b18e5c0cc428de327da609c/"
        "opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64."
        "manylinux2014_x86_64.whl"
        "#sha256=377d08a7e48a1405b5e84afcbe4798464ce7ee17081c1c23619c8b398ff18295",
    ),
    "mujoco": (
        "3.3.1",
        "https://files.pythonhosted.org/packages/ff/e3/"
        "b971670848fccdfee38efbfb802cc76e3d5fc2ad3bc03d42f77b79bcdaf7/"
        "mujoco-3.3.1-cp310-cp310-manylinux_2_17_x86_64."
        "manylinux2014_x86_64.whl"
        "#sha256=cfa05a01b2bded5d6a32c8ea97ef9e4995c1791a9d7a485112ff74fc5aee2e9d",
    ),
    "lxml": (
        "5.3.2",
        "https://files.pythonhosted.org/packages/44/cb/"
        "c974078e015990f83d13ef00dac347d74b1d62c2e6ec6e8eeb40ec9a1f1a/"
        "lxml-5.3.2-cp310-cp310-manylinux_2_17_x86_64."
        "manylinux2014_x86_64.whl"
        "#sha256=c9780de781a0d62a7c3680d07963db3048b919fc9e3726d9cfd97296a65ffce1",
    ),
    "llvmlite": (
        "0.45.1",
        "https://files.pythonhosted.org/packages/a6/7b/"
        "6d7585998a5991fa74dc925aae57913ba8c7c2efff909de9d34cc1cd3c27/"
        "llvmlite-0.45.1-cp310-cp310-manylinux2014_x86_64."
        "manylinux_2_17_x86_64.whl"
        "#sha256=f2d47f34e4029e6df3395de34cc1c66440a8d72712993a6e6168db228686711b",
    ),
    "numba": (
        "0.62.1",
        "https://files.pythonhosted.org/packages/a1/13/"
        "9a27bcd0baeea236116070c7df458414336f25e9dd5a872b066cf36b74bf/"
        "numba-0.62.1-cp310-cp310-manylinux2014_x86_64."
        "manylinux_2_17_x86_64.whl"
        "#sha256=14432af305ea68627a084cd702124fd5d0c1f5b8a413b05f4e14757202d1cf6c",
    ),
}


def command(
    *arguments: str,
    cwd: Path | None = None,
    input_text: str | None = None,
) -> str:
    print("+", " ".join(arguments), flush=True)
    return subprocess.check_output(
        arguments,
        cwd=cwd,
        input=input_text,
        text=True,
    ).strip()


def clone_or_resolve(vendor: Path, name: str, url: str, requested: str | None) -> str:
    destination = vendor / name
    newly_cloned = False
    if not destination.exists():
        command("git", "clone", "--filter=blob:none", url, str(destination))
        newly_cloned = True
    elif not (destination / ".git").exists():
        raise RuntimeError(f"{destination} exists but is not a complete Git checkout")
    sha = command("git", "rev-parse", "HEAD", cwd=destination)
    if requested and not sha.startswith(requested):
        status = command("git", "status", "--porcelain", cwd=destination)
        if status and not newly_cloned:
            raise RuntimeError(
                f"{name} is at {sha}, expected {requested}, and has local changes; "
                "refusing to overwrite them"
            )
        command("git", "checkout", "--detach", requested, cwd=destination)
        sha = command("git", "rev-parse", "HEAD", cwd=destination)
    if requested and not sha.startswith(requested):
        raise RuntimeError(f"{name} is at {sha}, expected {requested}")
    return sha


def remote_sha(url: str) -> str:
    output = command("git", "ls-remote", url, "HEAD")
    return output.split()[0]


def pip(*arguments: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *arguments])


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_versions(package: str) -> set[str]:
    normalized = re.sub(r"[-_.]+", "-", package).lower()
    return {
        distribution.version
        for distribution in importlib.metadata.distributions()
        if re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower() == normalized
    }


def install_manylinux_wheels() -> None:
    destination = sysconfig.get_path("purelib")
    for package, (version, wheel) in MANYLINUX_WHEELS.items():
        actual = installed_version(package)
        if actual == version:
            continue
        # A failed earlier bootstrap can leave a Compute Canada build of the
        # same native package behind. Remove it before the targeted install so
        # old dist-info and shared objects cannot shadow the pinned wheel.
        if actual is not None:
            pip("uninstall", "--yes", package)
        pip(
            "install",
            "--no-deps",
            "--only-binary=:all:",
            "--platform=manylinux2014_x86_64",
            "--target",
            destination,
            "--upgrade",
            wheel,
        )


def verify_lock_file(lock_file: Path) -> None:
    seen: set[str] = set()
    problems = []
    for line_number, raw_line in enumerate(lock_file.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*\])?"
            r"==([^;\s]+)",
            line,
        )
        if match is None or match.group(2).endswith(".*"):
            problems.append(f"line {line_number}: requirement is not exactly pinned: {line}")
            continue
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if name in seen:
            problems.append(f"line {line_number}: duplicate requirement {match.group(1)}")
        seen.add(name)
    if problems:
        raise RuntimeError("invalid requirements.lock.txt:\n" + "\n".join(problems))


def verify_pinned_versions(lock_file: Path) -> None:
    from packaging.requirements import Requirement

    requirements = []
    for raw_line in lock_file.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            requirements.append(Requirement(line))
    requirements.extend(
        Requirement(f"{package}=={version}")
        for package, (version, _) in MANYLINUX_WHEELS.items()
    )
    mismatches = []
    for requirement in requirements:
        actual = installed_versions(requirement.name)
        if not actual or any(
            not requirement.specifier.contains(version, prereleases=True) for version in actual
        ):
            mismatches.append(
                f"{requirement.name}: expected {requirement.specifier}, found {sorted(actual)}"
            )
    if mismatches:
        raise RuntimeError("pinned dependency verification failed:\n" + "\n".join(mismatches))


def install_assets(vendor: Path) -> None:
    robocasa_root = vendor / "robocasa"
    assets_root = robocasa_root / "robocasa" / "models" / "assets"
    asset_groups = {
        "tex": (
            assets_root / "textures" / "wood" / "dark_wood_planks_2.png",
            assets_root / "textures" / "tiles" / "diamond_marble_parquet.png",
        ),
        "tex_generative": (
            assets_root / "generative_textures" / "cabinet" / "tex001.png",
        ),
        "fixtures_lw": (
            assets_root / "fixtures" / "ovens" / "Oven043" / "model.xml",
            assets_root / "fixtures" / "fridges" / "Refrigerator066" / "model.xml",
        ),
        "objs_objaverse": (
            assets_root / "objects" / "objaverse" / "cereal" / "cereal_2" / "model.xml",
            assets_root / "objects" / "objaverse" / "wine" / "wine_2" / "model.xml",
            assets_root / "objects" / "objaverse" / "wine" / "wine_5" / "model.xml",
        ),
        "objs_aigen": (
            assets_root
            / "objects"
            / "aigen_objs"
            / "wine_glass"
            / "wine_glass_1"
            / "model.xml",
        ),
        "objs_lw": (
            assets_root
            / "objects"
            / "lightwheel"
            / "flour_bag"
            / "FlourBag002"
            / "model.xml",
        ),
    }

    def complete(paths: tuple[Path, ...]) -> bool:
        return all(path.is_file() and path.stat().st_size > 0 for path in paths)

    missing = [
        name
        for name, paths in asset_groups.items()
        if not complete(paths)
    ]
    if missing:
        command(
            sys.executable,
            "robocasa/scripts/download_kitchen_assets.py",
            "--type",
            *missing,
            cwd=robocasa_root,
            input_text="y\n",
        )
    incomplete = [
        name
        for name, paths in asset_groups.items()
        if not complete(paths)
    ]
    if incomplete:
        raise RuntimeError(f"RoboCasa asset download is incomplete: {incomplete}")
    macros = robocasa_root / "robocasa" / "macros.py"
    private_macros = robocasa_root / "robocasa" / "macros_private.py"
    if not private_macros.exists():
        shutil.copyfile(macros, private_macros)


def optional_binary_tool(package: str) -> bool:
    command_line = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        package,
    ]
    print("+", " ".join(command_line), flush=True)
    return subprocess.run(command_line, check=False).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--skip-assets", action="store_true")
    arguments = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"Python 3.10 is required, running {sys.version.split()[0]}")
    project_root = arguments.project_root.resolve()
    run_root = arguments.run_root.resolve()
    vendor = project_root / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, dict[str, Any]] = {}
    for name, (url, requested, installed) in REPOSITORIES.items():
        sha = clone_or_resolve(vendor, name, url, requested) if installed else remote_sha(url)
        resolved[name] = {"url": url, "sha": sha, "installed": installed}
    # Upgrade packaging tools from wheels and forbid source fallback for the
    # large third-party runtime stack. This prevents login-node Rust/C/C++ builds.
    pip(
        "install",
        "--only-binary=:all:",
        "pip==24.3.1",
        "setuptools==75.1.0",
        "wheel==0.44.0",
    )
    verify_lock_file(project_root / "requirements.lock.txt")
    pip(
        "install",
        "--only-binary=:all:",
        "--no-deps",
        "--requirement",
        str(project_root / "requirements.lock.txt"),
    )
    install_manylinux_wheels()
    verify_pinned_versions(project_root / "requirements.lock.txt")
    editable_options = ("--no-deps", "--no-build-isolation")
    pip("install", *editable_options, "--editable", str(project_root))
    pip("install", *editable_options, "--editable", str(vendor / "robosuite"))
    # The OopsieVerse installer itself applies this compatibility relaxation. Do
    # the same inside the untracked vendor checkout while retaining its source SHA.
    setup_py = vendor / "robocasa" / "setup.py"
    if setup_py.exists():
        text = setup_py.read_text()
        setup_py.write_text(text.replace('"numba==0.61.2"', '"numba>=0.61.2"'))
    pip("install", *editable_options, "--editable", str(vendor / "robocasa"))
    pip("install", *editable_options, "--editable", str(vendor / "oopsieverse"))
    pip("install", *editable_options, "--editable", str(vendor / "dppo"))
    command(
        sys.executable,
        "-c",
        "import cv2, mujoco, numba, pyarrow; from lxml import etree; "
        "from model.diffusion.diffusion_ppo import PPODiffusion; "
        "from oopsiebench.envs.robocasa.shelve_item import DamageableShelveItem; "
        "assert cv2.__version__ == '4.10.0'; "
        "assert mujoco.__version__ == '3.3.1'; "
        "assert pyarrow.__version__ == '17.0.0'; "
        "assert etree.fromstring(b'<ok/>').tag == 'ok'; "
        "assert numba.njit(lambda value: value + 1)(1) == 2",
    )
    ruff_installed = optional_binary_tool("ruff==0.6.9")
    tooling_artifact = run_root / "artifacts" / "bootstrap_tooling.json"
    tooling_artifact.parent.mkdir(parents=True, exist_ok=True)
    tooling_artifact.write_text(
        json.dumps(
            {
                "pytest": "8.3.3",
                "ruff": "0.6.9" if ruff_installed else None,
                "ruff_source_build_forbidden": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if not arguments.skip_assets:
        install_assets(vendor)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "safe_diffusion_cl_pilots.data.download",
            "--destination",
            str(run_root / "data"),
        ],
        cwd=project_root,
    )
    from safe_diffusion_cl_pilots.utils.manifests import source_versions

    manifest = source_versions(project_root)
    manifest["repositories"] = resolved
    manifest["inspected_documentation"] = [
        "https://robin-lab.cs.utexas.edu/oopsieverse/documentation/",
        "https://docs.alliancecan.ca/wiki/Narval/en",
    ]
    manifest["dataset"] = json.loads(
        (run_root / "data" / "demo_download_manifest.json").read_text()
    )
    manifest["working_tree_patches"] = {
        "robocasa": "numba==0.61.2 relaxed to numba>=0.61.2, matching OopsieVerse installer"
    }
    dppo_sources = [
        vendor / "dppo" / "model" / "diffusion" / "diffusion_ppo.py",
        vendor / "dppo" / "cfg" / "robomimic",
    ]
    if not all(path.exists() for path in dppo_sources):
        raise RuntimeError(f"official DPPO adapter sources are missing: {dppo_sources}")
    manifest["dppo_adapter"] = {
        "ppo_objective": "vendor/dppo/model/diffusion/diffusion_ppo.py",
        "starting_config": "vendor/dppo/cfg/robomimic",
        "local_adapter": "src/safe_diffusion_cl_pilots/models/dppo_adapter.py",
    }
    artifact = run_root / "artifacts" / "source_versions.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(artifact)
    resolved_requirements = project_root / "requirements.resolved.txt"
    with resolved_requirements.open("w") as stream:
        subprocess.check_call([sys.executable, "-m", "pip", "freeze"], stdout=stream)
    shutil.copyfile(resolved_requirements, run_root / "artifacts" / "requirements.resolved.txt")
    shutil.copyfile(resolved_requirements, run_root / "artifacts" / "requirements.lock.txt")


if __name__ == "__main__":
    main()
