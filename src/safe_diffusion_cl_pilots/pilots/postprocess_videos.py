from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from safe_diffusion_cl_pilots.data.build_lowdim_dataset import load_normalization
from safe_diffusion_cl_pilots.envs.object_centric_obs import ObjectCentricObservation
from safe_diffusion_cl_pilots.envs.shelve_contexts import make_context
from safe_diffusion_cl_pilots.envs.validation import write_reset_image
from safe_diffusion_cl_pilots.evaluation.videos import (
    deterministic_video_indices,
    write_video,
)
from safe_diffusion_cl_pilots.train.common import load_checkpoint
from safe_diffusion_cl_pilots.utils.logging import write_json
from safe_diffusion_cl_pilots.utils.seeding import torch_rng

from .training import chunk_env, policy_and_critic, read_gate


def _write_context_reset_images(gate: dict[str, Any], run_root: Path) -> None:
    manifest: dict[str, Any] = {"status": "COMPLETED", "images": [], "errors": []}
    environment_kwargs = dict(gate.get("environment_kwargs", {}))
    critical = str(gate["critical_fragile_object"])
    for context in ("A", "B"):
        destination = run_root / "artifacts" / f"context_reset_{context}.png"

        def factory(*, selected: str = context, **kwargs: Any) -> Any:
            return make_context(
                selected,
                critical_fragile_object=critical,
                **environment_kwargs,
                **kwargs,
            )

        try:
            write_reset_image(factory, destination)
            manifest["images"].append(str(destination.relative_to(run_root)))
        except Exception as error:
            manifest["status"] = "PARTIAL"
            manifest["errors"].append(f"Context {context}: {type(error).__name__}: {error}")
    write_json(run_root / "artifacts" / "context_reset_images.json", manifest)


def _capture(env: Any) -> np.ndarray:
    base = env.env
    simulator = getattr(base, "sim", None)
    if simulator is None:
        raise RuntimeError("environment has no simulator renderer")
    frame = simulator.render(height=360, width=480, camera_name="agentview")
    return np.flipud(np.asarray(frame, dtype=np.uint8))


def _render_episode(
    actor: Any,
    context: str,
    environment_seed: int,
    action_seed: int,
    gate: dict[str, Any],
    normalizer: Any,
    device: torch.device,
    destination: Path,
) -> None:
    environment_kwargs = {
        **gate.get("environment_kwargs", {}),
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": False,
    }
    env = chunk_env(
        context,
        gate["critical_fragile_object"],
        float(gate["epsilon_damage"]),
        environment_kwargs,
    )
    frames: list[np.ndarray] = []
    extractor = ObjectCentricObservation()
    try:
        observation = env.reset(seed=environment_seed)
        if isinstance(observation, tuple):
            observation = observation[0]
        frames.append(_capture(env))
        env.frame_collector = lambda _: frames.append(_capture(env))
        terminated = truncated = False
        with torch_rng(action_seed, device):
            while not terminated and not truncated:
                if not isinstance(observation, Mapping):
                    raise RuntimeError("video rollout did not expose object-centric state")
                vector = normalizer.normalize_observations(extractor.extract(observation))
                tensor = torch.as_tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = actor.sample(tensor).squeeze(0).cpu().numpy()
                observation, _, terminated, truncated, _ = env.step(
                    normalizer.denormalize_actions(action)
                )
        write_video(destination, frames)
    finally:
        env.close()


def _rows_for(
    frame: pd.DataFrame,
    context: str,
    checkpoint: int,
    maximum: int = 5,
    preferred_environment_seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    selected = frame[
        (frame["context"] == context) & (frame["checkpoint_steps"] == checkpoint)
    ].to_dict("records")
    preferred = []
    preferred_seeds: set[int] = set()
    for seed in preferred_environment_seeds or []:
        row = next((value for value in selected if int(value["environment_seed"]) == seed), None)
        if row is not None and seed not in preferred_seeds:
            preferred.append(row)
            preferred_seeds.add(seed)
    fallback = [selected[index] for index in deterministic_video_indices(selected, maximum)]
    combined = [
        *preferred,
        *(row for row in fallback if int(row["environment_seed"]) not in preferred_seeds),
    ]
    return combined[:maximum]


def run(run_root: Path, maximum_per_group: int = 5) -> None:
    gate = read_gate(run_root)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU video postprocessing cannot access CUDA")
    _write_context_reset_images(gate, run_root)
    device = torch.device("cuda")
    normalizer = load_normalization(run_root / "data" / "processed" / "seed_0" / "normalization.npz")
    output_manifest: dict[str, Any] = {"seed": 0, "maximum_per_group": maximum_per_group}
    comparison_path = (
        run_root / "results" / "seed_0" / "pilot2" / "comparison" / "diffusion_vs_gaussian.json"
    )
    comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else {}
    for family, result_dir in (
        ("diffusion", run_root / "results" / "seed_0" / "pilot1" / "diffusion"),
        ("gaussian", run_root / "results" / "seed_0" / "pilot2" / "gaussian"),
    ):
        family_manifest: dict[str, Any] = {"videos": [], "errors": []}
        output_manifest[family] = family_manifest
        metrics_path = result_dir / "rollout_metrics.parquet"
        summary_path = result_dir / "summary.json"
        if not metrics_path.exists() or not summary_path.exists():
            family_manifest["errors"].append("complete rollout metrics and summary are unavailable")
            continue
        metrics = pd.read_parquet(metrics_path)
        summary = json.loads(summary_path.read_text())
        if summary.get("skipped") or summary.get("skipped_rl"):
            family_manifest["errors"].append("policy pipeline was skipped")
            continue
        final_step = max(int(value) for value in summary["evaluations"])
        checkpoints = {
            "A_pre": ("A", 0, result_dir / "bc_checkpoint.pt", 300_000),
            "B_post": (
                "B",
                final_step,
                result_dir / "checkpoints" / f"step_{final_step}.pt",
                400_000,
            ),
            "A_post": (
                "A",
                final_step,
                result_dir / "checkpoints" / f"step_{final_step}.pt",
                300_000,
            ),
        }
        actor, _ = policy_and_critic(family, ObjectCentricObservation().dimension, device)
        for label, (context, checkpoint, checkpoint_path, action_offset) in checkpoints.items():
            if not checkpoint_path.exists():
                family_manifest["errors"].append(f"missing checkpoint for {label}: {checkpoint_path}")
                continue
            load_checkpoint(checkpoint_path, actor, map_location=device)
            actor.eval()
            route_seeds: list[int] = []
            if label == "A_post":
                modes = comparison.get("route_modes", {}).get(family, {}).get(
                    "mode_environment_seeds", {}
                )
                route_seeds = [int(values[0]) for values in modes.values() if values]
            for row in _rows_for(
                metrics, context, checkpoint, maximum_per_group, route_seeds
            ):
                environment_seed = int(row["environment_seed"])
                destination = result_dir / "videos" / (
                    f"{label}_env{environment_seed}_success{int(row['task_success'])}_"
                    f"damage{int(row['damage_event'])}.mp4"
                )
                try:
                    _render_episode(
                        actor,
                        context,
                        environment_seed,
                        action_offset + environment_seed,
                        gate,
                        normalizer,
                        device,
                        destination,
                    )
                    family_manifest["videos"].append(str(destination.relative_to(run_root)))
                except Exception as error:
                    family_manifest["errors"].append(
                        f"{label} environment seed {environment_seed}: {type(error).__name__}: {error}"
                    )
    write_json(run_root / "artifacts" / "diagnostic_videos.json", output_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render seed-0 diagnostic rollouts")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--maximum-per-group", type=int, default=5)
    arguments = parser.parse_args()
    if not 1 <= arguments.maximum_per_group <= 5:
        raise ValueError("--maximum-per-group must lie between 1 and 5")
    run(arguments.run_root, arguments.maximum_per_group)


if __name__ == "__main__":
    main()
