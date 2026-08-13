from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from .schemas import EpisodeRecord, NormalizationStats


def split_episode_ids(
    episode_ids: Sequence[str], seed: int, train_fraction: float = 0.8
) -> tuple[list[str], list[str]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    ordered = sorted(set(episode_ids))
    generator = np.random.default_rng(seed)
    generator.shuffle(ordered)
    boundary = max(1, min(len(ordered) - 1, int(round(train_fraction * len(ordered)))))
    return sorted(ordered[:boundary]), sorted(ordered[boundary:])


def fit_normalization(records: Sequence[EpisodeRecord]) -> NormalizationStats:
    if not records:
        raise ValueError("normalization requires at least one A-safe training episode")
    for record in records:
        record.validate()
    observations = np.concatenate([record.observations for record in records], axis=0)
    actions = np.concatenate([record.actions for record in records], axis=0)
    obs_std = observations.std(axis=0)
    if np.max(np.abs(actions)) > 1.0001:
        raise ValueError("released actions are not normalized to the environment's [-1, 1] bounds")
    return NormalizationStats(
        observation_mean=observations.mean(axis=0).astype(np.float32),
        observation_std=np.maximum(obs_std, 1e-6).astype(np.float32),
        # The environment action is already normalized. Keeping the action
        # transform exactly identity preserves bounds and avoids seed-dependent
        # rescaling while still freezing it alongside observation statistics.
        action_mean=np.zeros(actions.shape[1], dtype=np.float32),
        action_std=np.ones(actions.shape[1], dtype=np.float32),
        fitted_episode_ids=tuple(sorted(record.episode_id for record in records)),
    )


def build_action_chunks(actions: np.ndarray, horizon: int = 8) -> np.ndarray:
    """Pad only within an episode by repeating its final action."""
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
        raise ValueError("actions must have shape [T, 7] with T > 0")
    indices = np.minimum(
        np.arange(len(actions))[:, None] + np.arange(horizon)[None, :], len(actions) - 1
    )
    return actions[indices]


def iter_episode_chunks(
    records: Iterable[EpisodeRecord], horizon: int = 8
) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    for record in records:
        record.validate()
        chunks = build_action_chunks(record.actions, horizon)
        for observation, chunk in zip(record.observations, chunks, strict=True):
            yield record.episode_id, observation, chunk


def save_dppo_dataset(
    records: Sequence[EpisodeRecord],
    npz_path: Path,
    metadata_path: Path,
    normalization: NormalizationStats | None = None,
) -> None:
    if not records:
        raise ValueError("cannot save an empty dataset")
    for record in records:
        record.validate()
    states = np.concatenate([record.observations for record in records]).astype(np.float32)
    actions = np.concatenate([record.actions for record in records]).astype(np.float32)
    if normalization is not None:
        states = normalization.normalize_observations(states).astype(np.float32)
        actions = normalization.normalize_actions(actions).astype(np.float32)
    lengths = np.asarray([len(record.actions) for record in records], dtype=np.int64)
    if int(lengths.sum()) != len(states) or len(states) != len(actions):
        raise AssertionError("flattened dataset does not preserve trajectory lengths")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, states=states, actions=actions, traj_lengths=lengths)
    temporary.replace(npz_path)
    with metadata_path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record.metadata(), sort_keys=True) + "\n")


class ChunkDataset:
    """Torch-compatible action-chunk view that never crosses episode boundaries."""

    def __init__(self, path: str | Path, horizon: int = 8):
        archive = np.load(path)
        self.states = np.asarray(archive["states"], dtype=np.float32)
        self.actions = np.asarray(archive["actions"], dtype=np.float32)
        self.traj_lengths = np.asarray(archive["traj_lengths"], dtype=np.int64)
        if self.traj_lengths.sum() != len(self.states) or len(self.states) != len(self.actions):
            raise ValueError("invalid DPPO dataset lengths")
        self.horizon = horizon
        self._episode_start = np.repeat(np.cumsum(np.r_[0, self.traj_lengths[:-1]]), self.traj_lengths)
        self._episode_end = np.repeat(np.cumsum(self.traj_lengths), self.traj_lengths)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        end = int(self._episode_end[index])
        indices = np.minimum(np.arange(index, index + self.horizon), end - 1)
        return self.states[index], self.actions[indices]


def load_normalization(path: Path) -> NormalizationStats:
    archive = np.load(path)
    return NormalizationStats(
        observation_mean=np.asarray(archive["observation_mean"], dtype=np.float32),
        observation_std=np.asarray(archive["observation_std"], dtype=np.float32),
        action_mean=np.asarray(archive["action_mean"], dtype=np.float32),
        action_std=np.asarray(archive["action_std"], dtype=np.float32),
        fitted_episode_ids=tuple(str(item) for item in archive["fitted_episode_ids"]),
    )
