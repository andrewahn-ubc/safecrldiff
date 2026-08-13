from __future__ import annotations

import hashlib
from collections.abc import Iterable


def stable_episode_score(source_filename: str, episode_id: str, seed: int = 0) -> str:
    material = f"{source_filename}\0{episode_id}\0{seed}".encode()
    return hashlib.sha256(material).hexdigest()


def select_smoke_subset(
    source_filename: str,
    episode_ids: Iterable[str],
    count: int = 10,
    seed: int = 0,
) -> list[str]:
    unique = sorted(set(episode_ids))
    if len(unique) < count:
        raise ValueError(f"{source_filename} has {len(unique)} episodes; {count} requested")
    return sorted(unique, key=lambda item: (stable_episode_score(source_filename, item, seed), item))[
        :count
    ]


def expanded_smoke_subset(
    source_filename: str,
    episode_ids: Iterable[str],
    initial_count: int = 10,
    expansion: int = 5,
    maximum: int = 20,
    seed: int = 0,
) -> list[list[str]]:
    """Return deterministic cumulative candidates for automatic damage-signal expansion."""
    unique = list(set(episode_ids))
    ordered = select_smoke_subset(source_filename, unique, min(maximum, len(unique)), seed)
    limits = range(initial_count, min(maximum, len(ordered)) + 1, expansion)
    candidates = [ordered[:limit] for limit in limits]
    if candidates and len(candidates[-1]) < min(maximum, len(ordered)):
        candidates.append(ordered[:maximum])
    return candidates
