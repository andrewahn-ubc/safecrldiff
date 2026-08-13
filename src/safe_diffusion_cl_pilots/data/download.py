from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safe_diffusion_cl_pilots.utils.logging import write_json
from safe_diffusion_cl_pilots.utils.manifests import sha256_file

DEMO_REPOSITORY = "ut-robin-lab/oopsieverse-demos"
DEMO_FILES = (
    "robocasa/teleop/shelve_item_safe.hdf5",
    "robocasa/teleop/shelve_item_unsafe.hdf5",
)


def _valid_demo(path: Path, *, minimum_episodes: int = 1) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        import h5py

        with h5py.File(path, "r") as stream:
            return "data" in stream and len(stream["data"]) >= minimum_episodes
    except (ImportError, OSError):
        return False


def download_required_demos(destination: Path, revision: str | None = None) -> dict[str, dict[str, str]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface-hub is required to download demonstrations") from error
    manifest: dict[str, dict[str, str]] = {}
    manifest_path = destination / "demo_download_manifest.json"
    try:
        previous = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        previous = {}
    known_revisions = {
        str(details["revision"])
        for details in previous.values()
        if isinstance(details, dict) and details.get("revision")
    }
    if len(known_revisions) > 1:
        raise RuntimeError(f"demonstration manifest mixes revisions: {known_revisions}")
    resolved_revision = revision or (next(iter(known_revisions)) if known_revisions else None)
    for remote_path in DEMO_FILES:
        target = destination / remote_path
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
        if target.exists() and not _valid_demo(target, minimum_episodes=10):
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
        if not _valid_demo(target, minimum_episodes=10):
            raise RuntimeError(f"downloaded file is not a valid HDF5 demo with 10 episodes: {target}")
        if resolved_revision is None:
            from huggingface_hub import HfApi

            resolved_revision = HfApi().dataset_info(DEMO_REPOSITORY).sha
        manifest[remote_path] = {
            "sha256": sha256_file(target),
            "path": str(target),
            "revision": resolved_revision,
        }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--revision")
    arguments = parser.parse_args()
    download_required_demos(arguments.destination, arguments.revision)


if __name__ == "__main__":
    main()
