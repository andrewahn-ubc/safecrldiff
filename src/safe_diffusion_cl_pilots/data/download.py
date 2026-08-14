from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safe_diffusion_cl_pilots.utils.logging import write_json
from safe_diffusion_cl_pilots.utils.manifests import sha256_file

DEMO_REPOSITORY = "ut-robin-lab/oopsieverse-demos"
EXPECTED_EPISODES = {
    "robocasa/teleop/shelve_item_safe.hdf5": 45,
    "robocasa/teleop/shelve_item_unsafe.hdf5": 45,
}
DEMO_FILES = tuple(EXPECTED_EPISODES)


def _demo_episode_count(path: Path) -> int | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        import h5py

        with h5py.File(path, "r") as stream:
            if "data" not in stream:
                return None
            episodes = stream["data"]
            valid = [
                name
                for name, group in episodes.items()
                if hasattr(group, "keys") and "states" in group and "actions" in group
            ]
            return len(valid)
    except (ImportError, OSError):
        return None


def _valid_demo(path: Path, *, minimum_episodes: int = 1) -> bool:
    count = _demo_episode_count(path)
    return count is not None and count >= minimum_episodes


def download_required_demos(
    destination: Path,
    revision: str | None = None,
    source_directory: Path | None = None,
) -> dict[str, dict[str, str | int]]:
    if revision is not None and source_directory is not None:
        raise ValueError("revision and source_directory are mutually exclusive")
    manifest: dict[str, dict[str, str | int]] = {}
    manifest_path = destination / "demo_download_manifest.json"
    status_path = destination / "demo_dataset_status.json"
    try:
        previous = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        previous = {}
    local_manifest = bool(previous) and all(
        isinstance(details, dict)
        and details.get("revision") == "local-verified-override"
        and not str(details.get("source", "")).startswith("hf://")
        for details in previous.values()
    )
    known_revisions = {
        str(details["revision"])
        for details in previous.values()
        if isinstance(details, dict)
        and details.get("revision")
        and str(details.get("source", "")).startswith("hf://")
    }
    if len(known_revisions) > 1:
        raise RuntimeError(f"demonstration manifest mixes revisions: {known_revisions}")
    resolved_revision = revision or (next(iter(known_revisions)) if known_revisions else None)
    for remote_path in DEMO_FILES:
        target = destination / remote_path
        if source_directory is not None:
            source = source_directory / remote_path
            if not source.is_file():
                raise RuntimeError(f"--demo-source-dir is missing {remote_path}: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != target.resolve():
                temporary = target.with_suffix(target.suffix + ".partial")
                shutil.copyfile(source, temporary)
                temporary.replace(target)
            count = _demo_episode_count(target)
            manifest[remote_path] = {
                "episode_count": count or 0,
                "path": str(target),
                "revision": "local-verified-override",
                "sha256": sha256_file(target),
                "source": str(source.resolve()),
            }
            continue
        if local_manifest and revision is None:
            details = previous.get(remote_path, {})
            recorded_hash = details.get("sha256")
            if not target.is_file() or not recorded_hash:
                raise RuntimeError(
                    f"locally supplied demo is missing: {target}; rerun with --demo-source-dir"
                )
            actual_hash = sha256_file(target)
            if actual_hash != recorded_hash:
                raise RuntimeError(
                    f"locally supplied demo hash changed: {target}; rerun with --demo-source-dir"
                )
            count = _demo_episode_count(target)
            manifest[remote_path] = {
                "episode_count": count or 0,
                "path": str(target),
                "revision": "local-verified-override",
                "sha256": actual_hash,
                "source": str(details["source"]),
            }
            continue
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("huggingface-hub is required to download demonstrations") from error
        recorded_hash = previous.get(remote_path, {}).get("sha256")
        recorded_revision = previous.get(remote_path, {}).get("revision")
        # A file without a complete manifest may be an interrupted copy. Never
        # bless it merely because it is nonempty; redownload and then validate
        # the actual HDF5 structure.
        if target.exists() and (not recorded_hash or not recorded_revision):
            target.unlink()
        if target.exists() and revision is not None and recorded_revision != revision:
            target.unlink()
        if target.exists() and recorded_hash and sha256_file(target) != recorded_hash:
            target.unlink()
        if target.exists() and not _valid_demo(target):
            target.unlink()
        if not target.exists():
            cached = Path(
                hf_hub_download(
                    repo_id=DEMO_REPOSITORY,
                    repo_type="dataset",
                    filename=remote_path,
                    revision=resolved_revision,
                )
            )
            if "snapshots" in cached.parts:
                snapshot_index = cached.parts.index("snapshots")
                resolved_revision = cached.parts[snapshot_index + 1]
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            shutil.copyfile(cached, temporary)
            temporary.replace(target)
        count = _demo_episode_count(target)
        if count is None:
            raise RuntimeError(f"downloaded file is not a structurally valid HDF5 demo: {target}")
        if resolved_revision is None:
            from huggingface_hub import HfApi

            resolved_revision = HfApi().dataset_info(DEMO_REPOSITORY).sha
        manifest[remote_path] = {
            "episode_count": count,
            "sha256": sha256_file(target),
            "path": str(target),
            "revision": resolved_revision,
            "source": f"hf://datasets/{DEMO_REPOSITORY}/{remote_path}",
        }
    write_json(manifest_path, manifest)
    observed = {name: int(details["episode_count"]) for name, details in manifest.items()}
    mismatches = {
        name: {"expected": expected, "observed": observed.get(name, 0)}
        for name, expected in EXPECTED_EPISODES.items()
        if observed.get(name) != expected
    }
    status = {
        "expected_episode_counts": EXPECTED_EPISODES,
        "observed_episode_counts": observed,
        "reason_code": "OFFICIAL_DATASET_EPISODE_COUNT_MISMATCH" if mismatches else None,
        "revision": (
            "local-verified-override"
            if source_directory is not None or local_manifest
            else resolved_revision
        ),
        "status": "BLOCKED" if mismatches else "READY",
    }
    write_json(status_path, status)
    if mismatches:
        raise RuntimeError(
            "OFFICIAL_DATASET_EPISODE_COUNT_MISMATCH: "
            f"the specification requires 45+45 whole episodes, but found {mismatches}. "
            f"Details: {status_path}. Do not duplicate or split trajectories; obtain the "
            "complete official files and pass their parent directory with --demo-source-dir."
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--revision")
    source.add_argument("--source-directory", type=Path)
    arguments = parser.parse_args()
    download_required_demos(
        arguments.destination,
        arguments.revision,
        arguments.source_directory,
    )


if __name__ == "__main__":
    main()
