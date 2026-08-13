from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch without mutating simulator configuration."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@contextmanager
def torch_rng(seed: int, device: str | torch.device = "cpu") -> Iterator[None]:
    devices: list[int] = []
    resolved = torch.device(device)
    if resolved.type == "cuda":
        devices = [resolved.index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def evaluation_seeds(base_seed: int, context: str, count: int = 30) -> list[int]:
    offset = 10_000 if context.upper() == "B" else 0
    sequence = np.random.SeedSequence([base_seed, offset, 0x0A51E])
    return [int(value) for value in sequence.generate_state(count, dtype=np.uint32)]

