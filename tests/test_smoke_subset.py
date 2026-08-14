import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from safe_diffusion_cl_pilots.data.download import (
    EXPECTED_EPISODES,
    _valid_demo,
    download_required_demos,
)
from safe_diffusion_cl_pilots.data.select_smoke_subset import (
    expanded_smoke_subset,
    select_smoke_subset,
)


def test_demo_validation_rejects_corrupt_hdf5(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.hdf5"
    corrupt.write_bytes(b"not hdf5")
    assert not _valid_demo(corrupt)


def test_official_demo_contract_requires_45_episodes_per_source() -> None:
    assert EXPECTED_EPISODES == {
        "robocasa/teleop/shelve_item_safe.hdf5": 45,
        "robocasa/teleop/shelve_item_unsafe.hdf5": 45,
    }


def test_local_demo_override_is_validated_and_reused(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    for relative_path in EXPECTED_EPISODES:
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.encode())

    episodes = {
        f"demo_{index}": {"actions": object(), "states": object()}
        for index in range(45)
    }
    fake_h5py = SimpleNamespace(File=lambda *_args, **_kwargs: nullcontext({"data": episodes}))
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)

    first = download_required_demos(destination, source_directory=source)
    second = download_required_demos(destination)

    assert first == second
    status = json.loads((destination / "demo_dataset_status.json").read_text())
    assert status["status"] == "READY"
    assert status["observed_episode_counts"] == EXPECTED_EPISODES


def test_incomplete_demo_override_writes_blocker_status(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    for relative_path in EXPECTED_EPISODES:
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.encode())

    episodes = {"demo_0": {"actions": object(), "states": object()}}
    fake_h5py = SimpleNamespace(File=lambda *_args, **_kwargs: nullcontext({"data": episodes}))
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)

    destination = tmp_path / "destination"
    with pytest.raises(RuntimeError, match="OFFICIAL_DATASET_EPISODE_COUNT_MISMATCH"):
        download_required_demos(destination, source_directory=source)

    status = json.loads((destination / "demo_dataset_status.json").read_text())
    assert status["status"] == "BLOCKED"
    assert set(status["observed_episode_counts"].values()) == {1}


def test_smoke_subset_is_stable_and_source_dependent():
    episodes = [f"demo_{index}" for index in range(45)]
    first = select_smoke_subset("safe.hdf5", episodes, 10, seed=0)
    assert first == select_smoke_subset("safe.hdf5", reversed(episodes), 10, seed=0)
    assert first != select_smoke_subset("unsafe.hdf5", episodes, 10, seed=0)
    assert len(first) == len(set(first)) == 10


def test_expansion_is_cumulative():
    candidates = expanded_smoke_subset("safe.hdf5", map(str, range(45)))
    assert [len(value) for value in candidates] == [10, 15, 20]
    assert candidates[0] == candidates[1][:10]
    assert candidates[1] == candidates[2][:15]
