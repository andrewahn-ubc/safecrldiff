from __future__ import annotations

from typing import Any

import numpy as np


def resample_xy(trajectory: np.ndarray, points: int = 32) -> np.ndarray:
    values = np.asarray(trajectory, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) < 2:
        raise ValueError("route trajectory must contain at least two XY points")
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.column_stack([np.interp(target, source, values[:, axis]) for axis in range(2)])


def canonical_route(trajectory: np.ndarray, target_xy: np.ndarray, points: int = 32) -> np.ndarray:
    route = resample_xy(trajectory, points)
    start = route[0].copy()
    route -= start
    target = np.asarray(target_xy, dtype=float)[:2] - start
    angle = -np.arctan2(target[1], target[0])
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = max(np.linalg.norm(target), 1e-6)
    return route @ rotation.T / scale


def discover_route_modes(
    trajectories: list[np.ndarray],
    target_xy: np.ndarray,
    points: int = 32,
    distance_threshold: float = 0.05,
) -> dict[str, Any]:
    if len(trajectories) < 3:
        return {"valid_mode_count": 0, "entropy": 0.0, "dominant_fraction": 1.0, "labels": []}
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    routes = np.stack([canonical_route(route, target_xy, points) for route in trajectories])
    flattened = routes.reshape(len(routes), -1)
    best_labels = np.zeros(len(routes), dtype=int)
    best_score = -1.0
    for clusters in range(2, min(5, len(routes) - 1) + 1):
        labels = AgglomerativeClustering(n_clusters=clusters).fit_predict(flattened)
        score = silhouette_score(flattened, labels)
        if score > best_score:
            best_score, best_labels = score, labels
    counts = np.bincount(best_labels)
    minimum = max(3, int(np.ceil(0.10 * len(routes))))
    eligible = [index for index, count in enumerate(counts) if count >= minimum]
    centroids = {index: routes[best_labels == index].mean(axis=0) for index in eligible}
    valid: list[int] = []
    for index in eligible:
        distances = [
            float(np.mean(np.linalg.norm(centroids[index] - centroids[other], axis=1)))
            for other in eligible
            if other != index
        ]
        if not distances or min(distances) >= distance_threshold:
            valid.append(index)
    valid_counts = np.asarray([counts[index] for index in valid], dtype=float)
    probabilities = valid_counts / valid_counts.sum() if valid_counts.sum() else np.asarray([])
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    within_variance = {
        str(index): float(np.var(routes[best_labels == index], axis=0).mean()) for index in valid
    }
    return {
        "valid_mode_count": len(valid),
        "valid_labels": valid,
        "entropy": entropy,
        "dominant_fraction": float(probabilities.max()) if len(probabilities) else 1.0,
        "within_mode_variance": within_variance,
        "labels": best_labels.tolist(),
        "silhouette": best_score,
        "distance_threshold": distance_threshold,
    }
