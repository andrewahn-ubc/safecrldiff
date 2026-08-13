from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from safe_diffusion_cl_pilots.data.build_lowdim_dataset import ChunkDataset, save_dppo_dataset
from safe_diffusion_cl_pilots.data.replay_oopsie_hdf5 import OopsieHDF5, replay_file
from safe_diffusion_cl_pilots.data.select_smoke_subset import expanded_smoke_subset
from safe_diffusion_cl_pilots.envs.chunked_gym_wrapper import ChunkedGymWrapper
from safe_diffusion_cl_pilots.envs.object_centric_obs import ObjectCentricObservation
from safe_diffusion_cl_pilots.envs.shelve_contexts import make_context, make_context_from_metadata
from safe_diffusion_cl_pilots.models.critics import ValueCritic
from safe_diffusion_cl_pilots.models.diffusion_policy import DiffusionPolicy
from safe_diffusion_cl_pilots.models.gaussian_chunk_policy import GaussianChunkPolicy
from safe_diffusion_cl_pilots.train.common import save_checkpoint, stack_diffusion_traces
from safe_diffusion_cl_pilots.train.finetune_diffusion_dppo import diffusion_dppo_update
from safe_diffusion_cl_pilots.train.finetune_gaussian_ppo import gaussian_ppo_update
from safe_diffusion_cl_pilots.train.train_diffusion_bc import train_diffusion_bc
from safe_diffusion_cl_pilots.train.train_gaussian_bc import train_gaussian_bc
from safe_diffusion_cl_pilots.utils.config import project_config
from safe_diffusion_cl_pilots.utils.logging import write_json
from safe_diffusion_cl_pilots.utils.manifests import write_source_versions
from safe_diffusion_cl_pilots.utils.seeding import seed_everything

from .common import Stage, stage_parser


def _demo_paths(run_root: Path) -> dict[str, Path]:
    base = run_root / "data" / "robocasa" / "teleop"
    return {"safe": base / "shelve_item_safe.hdf5", "unsafe": base / "shelve_item_unsafe.hdf5"}


def _reset_observation(env: Any) -> Any:
    value = env.reset()
    return value[0] if isinstance(value, tuple) else value


def _collect_update(
    policy: DiffusionPolicy | GaussianChunkPolicy,
    observation_dim: int,
    kind: str,
    steps: int,
) -> dict[str, float]:
    extractor = ObjectCentricObservation()
    env = ChunkedGymWrapper(make_context("A"), horizon=8)
    critic = ValueCritic(observation_dim)
    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    observations: list[torch.Tensor] = []
    rewards: list[float] = []
    old_log_probs: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    traces: list[Any] = []
    try:
        observation = _reset_observation(env)
        for _ in range(steps):
            vector = torch.as_tensor(extractor.extract(observation)).unsqueeze(0)
            if kind == "diffusion":
                action, log_prob, trace = policy.sample_with_log_prob(vector)
                traces.append(trace)
            else:
                action, log_prob = policy.sample_with_log_prob(vector)
                actions.append(action.detach())
            observation, reward, terminated, truncated, _ = env.step(action.squeeze(0).detach().numpy())
            observations.append(vector)
            rewards.append(float(reward))
            old_log_probs.append(log_prob.detach())
            if terminated or truncated:
                observation = _reset_observation(env)
    finally:
        env.close()
    obs_batch = torch.cat(observations)
    old_batch = torch.cat(old_log_probs)
    advantages = torch.as_tensor(rewards, dtype=torch.float32)
    returns = advantages.clone()
    if kind == "diffusion":
        stats = diffusion_dppo_update(
            policy,
            critic,
            actor_optimizer,
            critic_optimizer,
            obs_batch,
            stack_diffusion_traces(traces),
            old_batch,
            advantages,
            returns,
        )
    else:
        stats = gaussian_ppo_update(
            policy,
            critic,
            actor_optimizer,
            critic_optimizer,
            obs_batch,
            torch.cat(actions),
            old_batch,
            advantages,
            returns,
        )
    return stats.__dict__


def _reload_reproducible(policy: Any, checkpoint: Path, constructor: Any, observation_dim: int) -> bool:
    reloaded = constructor(observation_dim)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(payload["actor"])
    observation = torch.zeros(2, observation_dim)
    torch.manual_seed(829)
    first = policy.sample(observation)
    torch.manual_seed(829)
    second = reloaded.sample(observation)
    return bool(torch.equal(first, second) and torch.isfinite(second).all())


def run(project_root: Path, run_root: Path) -> None:
    stage = Stage("pilot_minus1", project_root, run_root)
    config = project_config(project_root, "pilot_minus1")
    seed_everything(0)
    try:
        paths = _demo_paths(run_root)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"required demonstrations missing: {missing}")
        archives = {label: OopsieHDF5(path) for label, path in paths.items()}
        environment_metadata = {
            label: archive.environment_metadata() for label, archive in archives.items()
        }
        episode_ids = {label: archive.episode_ids() for label, archive in archives.items()}
        if sum(map(len, episode_ids.values())) != 90 or any(len(ids) != 45 for ids in episode_ids.values()):
            raise RuntimeError(f"expected released 45+45 episodes, found { {key: len(value) for key, value in episode_ids.items()} }")
        stage.mark("HDF5 structure and 45+45 episode counts")
        candidates = {
            label: expanded_smoke_subset(
                paths[label].name,
                ids,
                int(os.environ.get("SAFE_PILOTS_SMOKE_EPISODES_PER_FILE", config["episodes_per_file"])),
                config["expansion_per_file"],
                config["maximum_episodes_per_file"],
                seed=0,
            )
            for label, ids in episode_ids.items()
        }
        records: list[Any] = []
        rejections: list[dict[str, str]] = []
        selected: dict[str, list[str]] = {label: [] for label in paths}
        maximum_rounds = max(len(value) for value in candidates.values())
        for round_index in range(maximum_rounds):
            for label, path in paths.items():
                desired = candidates[label][min(round_index, len(candidates[label]) - 1)]
                new_ids = [episode_id for episode_id in desired if episode_id not in selected[label]]
                if new_ids:
                    added, rejected = replay_file(
                        path,
                        label,
                        lambda metadata=environment_metadata[label]: make_context_from_metadata(metadata),
                        new_ids,
                    )
                    records.extend(added)
                    rejections.extend({"source": label, **row} for row in rejected)
                    selected[label].extend(new_ids)
            if any(record.total_environment_damage > 0 for record in records):
                break
        write_json(run_root / "artifacts" / "pilot_minus1_episode_ids.json", selected)
        if not any(record.total_environment_damage > 0 for record in records):
            raise RuntimeError("DAMAGE_SIGNAL_NOT_OBSERVED after deterministic expansion")
        attempted = sum(map(len, selected.values()))
        if len(records) / attempted < config["minimum_replay_fraction"]:
            raise RuntimeError(f"only {len(records)}/{attempted} episodes replayed; rejections={rejections}")
        stage.mark("production replay and DamageSim signal validation")
        result = stage.result_dir
        save_dppo_dataset(records, result / "mini_dataset.npz", result / "mini_dataset_metadata.jsonl")
        dataset = ChunkDataset(result / "mini_dataset.npz", horizon=8)
        observation_dim = dataset.states.shape[1]
        ObjectCentricObservation().assert_no_leakage()
        diffusion = DiffusionPolicy(observation_dim)
        gaussian = GaussianChunkPolicy(observation_dim)
        diffusion_metrics = train_diffusion_bc(diffusion, dataset, steps=config["bc_optimizer_steps"], batch_size=min(64, len(dataset)))
        gaussian_metrics = train_gaussian_bc(gaussian, dataset, steps=config["bc_optimizer_steps"], batch_size=min(64, len(dataset)))
        if not all(np.isfinite(row["train_loss"]) for row in diffusion_metrics + gaussian_metrics):
            raise FloatingPointError("non-finite behavior-cloning smoke metrics")
        stage.mark("actual loaders and 20-step BC updates")
        diffusion_update = _collect_update(diffusion, observation_dim, "diffusion", config["rollout_chunk_steps"])
        gaussian_update = _collect_update(gaussian, observation_dim, "gaussian", config["rollout_chunk_steps"])
        diffusion_checkpoint = result / "diffusion_smoke_checkpoint.pt"
        gaussian_checkpoint = result / "gaussian_smoke_checkpoint.pt"
        save_checkpoint(diffusion_checkpoint, diffusion)
        save_checkpoint(gaussian_checkpoint, gaussian)
        reload_checks = {
            "diffusion": _reload_reproducible(diffusion, diffusion_checkpoint, DiffusionPolicy, observation_dim),
            "gaussian": _reload_reproducible(gaussian, gaussian_checkpoint, GaussianChunkPolicy, observation_dim),
        }
        if not all(reload_checks.values()):
            raise AssertionError(f"checkpoint reload mismatch: {reload_checks}")
        numerical = {
            "diffusion_bc_final": diffusion_metrics[-1],
            "gaussian_bc_final": gaussian_metrics[-1],
            "diffusion_dppo_update": diffusion_update,
            "gaussian_ppo_update": gaussian_update,
            "checkpoint_reload": reload_checks,
        }
        write_json(result / "numerical_checks.json", numerical)
        rows = [record.metadata() for record in records] + [dict(row, rejected=True) for row in rejections]
        pd.DataFrame(rows).to_parquet(result / "replay_manifest.parquet", index=False)
        write_source_versions(project_root, run_root)
        status = {"status": "PASS", "attempted": attempted, "replayed": len(records), "rejected": rejections}
        result.joinpath("report.md").write_text(
            "# Pilot -1: PASS\n\n"
            f"Automatically replayed {len(records)}/{attempted} selected episodes. "
            "DamageSim changed on at least one trajectory; both production policy classes completed "
            "BC, rollout, weight-level PPO/DPPO, and checkpoint-reload checks.\n"
        )
        stage.succeed(status)
    except Exception as error:
        stage.fail(error, stop_pipeline=True)
        raise


def main() -> None:
    parser = stage_parser("Automated end-to-end Pilot -1")
    arguments = parser.parse_args()
    run(arguments.project_root, arguments.run_root)


if __name__ == "__main__":
    main()
