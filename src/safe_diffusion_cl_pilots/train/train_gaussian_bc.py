from __future__ import annotations

import copy

import numpy as np
import torch

from safe_diffusion_cl_pilots.data.build_lowdim_dataset import ChunkDataset
from safe_diffusion_cl_pilots.models.gaussian_chunk_policy import GaussianChunkPolicy
from safe_diffusion_cl_pilots.train.train_diffusion_bc import _batch


def train_gaussian_bc(
    model: GaussianChunkPolicy,
    train_dataset: ChunkDataset,
    validation_dataset: ChunkDataset | None = None,
    steps: int = 50_000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    gradient_clip: float = 5.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    resolved = torch.device(device)
    model.to(resolved).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)
    metrics: list[dict[str, float]] = []
    best_validation = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for step in range(1, steps + 1):
        indices = generator.integers(0, len(train_dataset), size=batch_size)
        observations, actions = _batch(train_dataset, indices, resolved)
        optimizer.zero_grad(set_to_none=True)
        loss = model.bc_loss(observations, actions)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Gaussian BC loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        row = {"step": float(step), "train_loss": float(loss), "gradient_norm": float(gradient_norm)}
        if validation_dataset is not None and (step == steps or step % min(500, steps) == 0):
            with torch.no_grad():
                indices = np.arange(min(len(validation_dataset), 512))
                val_obs, val_actions = _batch(validation_dataset, indices, resolved)
                validation = float(model.bc_loss(val_obs, val_actions))
            row["validation_loss"] = validation
            if validation < best_validation - 1e-6:
                best_validation, stale = validation, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= 20:
                metrics.append(row)
                break
        metrics.append(row)
    if validation_dataset is not None:
        model.load_state_dict(best_state)
    return metrics
