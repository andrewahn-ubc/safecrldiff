from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    if not frames:
        raise ValueError("cannot write a video without frames")
    try:
        import imageio.v3 as iio
    except ImportError as error:
        raise RuntimeError("install imageio[ffmpeg] for optional diagnostic videos") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.asarray(frames), fps=fps)


def deterministic_video_indices(rows: list[dict[str, Any]], maximum: int = 5) -> list[int]:
    ordered = sorted(
        range(len(rows)), key=lambda index: rows[index].get("environment_seed", index)
    )
    categories = (
        lambda row: row.get("task_success", False) and not row.get("damage_event", False),
        lambda row: row.get("task_success", False) and row.get("damage_event", False),
        lambda row: not row.get("task_success", False),
    )
    selected: list[int] = []
    for predicate in categories:
        match = next((index for index in ordered if predicate(rows[index])), None)
        if match is not None and match not in selected:
            selected.append(match)
    selected.extend(index for index in ordered if index not in selected)
    return selected[:maximum]
