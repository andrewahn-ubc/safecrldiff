from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safe_diffusion_cl_pilots.envs.object_centric_obs import ObjectCentricObservation
from safe_diffusion_cl_pilots.evaluation.metrics import pilot2_gate
from safe_diffusion_cl_pilots.evaluation.route_modes import discover_route_modes
from safe_diffusion_cl_pilots.models.critics import parameter_count
from safe_diffusion_cl_pilots.models.diffusion_policy import DiffusionPolicy
from safe_diffusion_cl_pilots.models.gaussian_chunk_policy import GaussianChunkPolicy
from safe_diffusion_cl_pilots.utils.logging import write_json
from safe_diffusion_cl_pilots.utils.manifests import atomic_success

from .common import Stage, stage_parser
from .pilot1 import _scientific_summary
from .training import run_policy_pipeline


def _wait_for_seed0_extension(run_root: Path, seed: int) -> bool:
    """Resolve the cross-array extension decision without racing seed 0."""
    if seed == 0:
        summary = json.loads(
            (run_root / "results" / "seed_0" / "pilot1" / "diffusion" / "summary.json").read_text()
        )
        return bool(summary.get("extended", False))
    manifest = run_root / "artifacts" / "adaptive_extension.json"
    seed0_dir = run_root / "results" / "seed_0" / "pilot1" / "diffusion"
    timeout = float(os.environ.get("SAFE_PILOTS_EXTENSION_WAIT_SECONDS", "36000"))
    deadline = time.monotonic() + timeout
    while True:
        if manifest.exists():
            return bool(json.loads(manifest.read_text()).get("extend_other_seeds", False))
        if (seed0_dir / "_SUCCESS").exists():
            return False
        status_path = seed0_dir / "status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("status") == "FAILED":
            raise RuntimeError("seed-0 diffusion failed before resolving adaptive extension")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for seed-0 adaptive-extension decision")
        time.sleep(30)


def _route_summary(path: Path, run_root: Path) -> dict[str, Any]:
    if not path.exists():
        return {"valid_mode_count": 0, "entropy": 0.0, "dominant_fraction": 1.0, "labels": []}
    frame = pd.read_parquet(path)
    if frame.empty:
        return {"valid_mode_count": 0, "entropy": 0.0, "dominant_fraction": 1.0, "labels": []}
    final_step = frame["checkpoint_steps"].max()
    selected = frame[
        (frame["context"] == "A")
        & (frame["checkpoint_steps"] == final_step)
        & frame["task_success"]
        & ~frame["damage_event"]
    ]
    valid_rows = [row for _, row in selected.iterrows() if len(row["cereal_trajectory"]) >= 2]
    trajectories = [np.asarray(row["cereal_trajectory"], dtype=float) for row in valid_rows]
    manifest = json.loads((run_root / "artifacts" / "context_manifest.json").read_text())
    target = np.asarray(manifest["contexts"]["A"]["target_mat"], dtype=float)
    result = discover_route_modes(trajectories, target)
    valid_labels = set(result.get("valid_labels", []))
    assigned_labels = result.get("labels", [])
    result["mode_environment_seeds"] = {
        str(label): [
            int(row["environment_seed"])
            for row, assigned in zip(valid_rows, assigned_labels, strict=True)
            if assigned == label
        ]
        for label in sorted(valid_labels)
    } if len(assigned_labels) == len(valid_rows) else {}
    return result


def run(project_root: Path, run_root: Path, seed: int) -> None:
    stage = Stage(f"seed_{seed}/pilot2/gaussian", project_root, run_root)
    try:
        diffusion_dir = run_root / "results" / f"seed_{seed}" / "pilot1" / "diffusion"
        diffusion_summary_path = diffusion_dir / "summary.json"
        if not diffusion_summary_path.exists():
            raise RuntimeError("Pilot 1 diffusion summary is required before Gaussian comparator")
        diffusion_summary = json.loads(diffusion_summary_path.read_text())
        extension_applies = _wait_for_seed0_extension(run_root, seed)
        if extension_applies and not diffusion_summary.get("extended", False):
            prior_walltime = float(diffusion_summary.get("walltime_seconds", 0.0))
            diffusion_raw = run_policy_pipeline(
                "diffusion", project_root, run_root, seed, diffusion_dir, 300_000
            )
            diffusion_summary = _scientific_summary(diffusion_raw)
            diffusion_summary["walltime_seconds"] = (
                float(diffusion_summary.get("walltime_seconds", 0.0)) + prior_walltime
            )
            write_json(diffusion_summary_path, diffusion_summary)
        maximum = 300_000 if extension_applies else 150_000
        raw = run_policy_pipeline(
            "gaussian", project_root, run_root, seed, stage.result_dir, maximum
        )
        gaussian_summary = _scientific_summary(raw)
        write_json(stage.result_dir / "summary.json", gaussian_summary)
        for config_name in ("common", "gaussian_bc", "gaussian_ppo"):
            source = project_root / "configs" / f"{config_name}.yaml"
            target = stage.result_dir / ("config.yaml" if config_name == "common" else f"{config_name}.yaml")
            shutil.copyfile(source, target)
        comparison_dir = stage.result_dir.parent / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        if not gaussian_summary.get("skipped") and not gaussian_summary.get("skipped_rl"):
            diffusion_route = _route_summary(diffusion_dir / "rollout_metrics.parquet", run_root)
            gaussian_route = _route_summary(stage.result_dir / "rollout_metrics.parquet", run_root)
            observation_dim = ObjectCentricObservation().dimension
            diffusion_parameters = parameter_count(DiffusionPolicy(observation_dim))
            gaussian_parameters = parameter_count(GaussianChunkPolicy(observation_dim))
            relative_parameter_gap = abs(diffusion_parameters - gaussian_parameters) / diffusion_parameters
            if relative_parameter_gap > 0.20:
                raise RuntimeError(f"actor capacity mismatch is {relative_parameter_gap:.1%}, above 20%")
            diffusion_metrics = {
                "a_safe_success": diffusion_summary["a_post"]["safe_success"],
                "b_post_success": diffusion_summary["b_post"],
                "steps_to_b_success": float(diffusion_summary["primitive_steps"]),
                "route_modes": float(diffusion_route["valid_mode_count"]),
                "route_entropy": float(diffusion_route["entropy"]),
            }
            gaussian_metrics = {
                "a_safe_success": gaussian_summary["a_post"]["safe_success"],
                "b_post_success": gaussian_summary["b_post"],
                "steps_to_b_success": float(gaussian_summary["primitive_steps"]),
                "route_modes": float(gaussian_route["valid_mode_count"]),
                "route_entropy": float(gaussian_route["entropy"]),
            }
            gate = pilot2_gate(diffusion_metrics, gaussian_metrics)
            comparison = {
                "diffusion": diffusion_metrics,
                "gaussian": gaussian_metrics,
                "gate": gate,
                "actor_parameters": {
                    "diffusion": diffusion_parameters,
                    "gaussian": gaussian_parameters,
                    "relative_gap": relative_parameter_gap,
                },
                "route_modes": {"diffusion": diffusion_route, "gaussian": gaussian_route},
                "efficiency": {
                    "diffusion": {
                        "walltime_seconds": diffusion_summary["walltime_seconds"],
                        "environment_steps": diffusion_summary["primitive_steps"],
                    },
                    "gaussian": {
                        "walltime_seconds": gaussian_summary["walltime_seconds"],
                        "environment_steps": gaussian_summary["primitive_steps"],
                    },
                },
                "walltime_ratio_diffusion_over_gaussian": diffusion_summary["walltime_seconds"] / max(gaussian_summary["walltime_seconds"], 1e-9),
            }
            write_json(comparison_dir / "diffusion_vs_gaussian.json", comparison)
            pd.DataFrame(
                [
                    {"policy": "diffusion", **diffusion_route},
                    {"policy": "gaussian", **gaussian_route},
                ]
            ).drop(columns=["labels"], errors="ignore").to_parquet(
                comparison_dir / "route_modes.parquet", index=False
            )
        atomic_success(stage.result_dir, {"seed": seed, "family": "gaussian"})
    except Exception as error:
        stage.fail(error)
        raise


def main() -> None:
    parser = stage_parser("Pilot 2 matched Gaussian PPO comparator", seed=True)
    arguments = parser.parse_args()
    run(arguments.project_root, arguments.run_root, arguments.seed)


if __name__ == "__main__":
    main()
