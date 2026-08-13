from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


class SequentialVectorEnv:
    """Small dependency-free vector adapter used by CPU smoke tests."""

    def __init__(self, factories: Sequence[Callable[[], Any]]):
        self.envs = [factory() for factory in factories]

    def reset(self, seeds: Sequence[int] | None = None) -> list[Any]:
        seeds = seeds or [None] * len(self.envs)  # type: ignore[list-item]
        return [env.reset(seed=seed) for env, seed in zip(self.envs, seeds, strict=True)]

    def step(self, actions: Sequence[Any]) -> list[Any]:
        return [env.step(action) for env, action in zip(self.envs, actions, strict=True)]

    def close(self) -> None:
        for env in self.envs:
            env.close()
