from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from safe_diffusion_cl_pilots.data.build_lowdim_dataset import (
    ChunkDataset,
    load_normalization,
)
from safe_diffusion_cl_pilots.envs.chunked_gym_wrapper import ChunkedGymWrapper
from safe_diffusion_cl_pilots.envs.object_centric_obs import ObjectCentricObservation
from safe_diffusion_cl_pilots.envs.shelve_contexts import make_context
from safe_diffusion_cl_pilots.evaluation.rollout import evaluate_fixed_seeds, summarize_rollouts
from safe_diffusion_cl_pilots.models.critics import ValueCritic, parameter_count
from safe_diffusion_cl_pilots.models.diffusion_policy import DiffusionPolicy
from safe_diffusion_cl_pilots.models.gaussian_chunk_policy import GaussianChunkPolicy
from safe_diffusion_cl_pilots.train.common import (
    generalized_advantage_estimate,
    load_checkpoint,
    save_checkpoint,
    stack_diffusion_traces,
)
from safe_diffusion_cl_pilots.train.finetune_diffusion_dppo import diffusion_dppo_update
from safe_diffusion_cl_pilots.train.finetune_gaussian_ppo import gaussian_ppo_update
from safe_diffusion_cl_pilots.train.train_diffusion_bc import train_diffusion_bc
from safe_diffusion_cl_pilots.train.train_gaussian_bc import train_gaussian_bc
from safe_diffusion_cl_pilots.utils.logging import write_json
from safe_diffusion_cl_pilots.utils.seeding import evaluation_seeds, seed_everything


def read_gate(run_root: Path) -> dict[str, Any]:
    path = run_root / "results" / "pilot0" / "gate.json"
    if not path.exists():
        raise RuntimeError("Pilot 0 gate.json is missing")
    return json.loads(path.read_text())


def write_eval_seeds(run_root: Path, seed: int) -> tuple[list[int], list[int]]:
    artifact_dir = run_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    a = evaluation_seeds(seed, "A", 30)
    b = evaluation_seeds(seed, "B", 30)
    write_json(artifact_dir / f"eval_seeds_A_seed_{seed}.json", a)
    write_json(artifact_dir / f"eval_seeds_B_seed_{seed}.json", b)
    if seed == 0:
        write_json(artifact_dir / "eval_seeds_A.json", a)
        write_json(artifact_dir / "eval_seeds_B.json", b)
    return a, b


def policy_and_critic(
    family: str, observation_dim: int, device: torch.device
) -> tuple[Any, ValueCritic]:
    actor: Any
    if family == "diffusion":
        actor = DiffusionPolicy(observation_dim)
    elif family == "gaussian":
        actor = GaussianChunkPolicy(observation_dim)
    else:
        raise ValueError(f"unknown policy family: {family}")
    return actor.to(device), ValueCritic(observation_dim).to(device)


def chunk_env(
    context: str,
    critical: str,
    epsilon: float,
    environment_kwargs: dict[str, Any] | None = None,
) -> ChunkedGymWrapper:
    return ChunkedGymWrapper(
        make_context(
            context,
            critical_fragile_object=critical,
            **(environment_kwargs or {}),
        ),
        horizon=8,
        critical_fragile_object=critical,
        epsilon_damage=epsilon,
    )


def evaluate(
    actor: Any,
    context: str,
    critical: str,
    epsilon: float,
    seeds: list[int],
    action_offset: int,
    normalizer: Any,
    device: torch.device,
    environment_kwargs: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = evaluate_fixed_seeds(
        lambda: chunk_env(context, critical, epsilon, environment_kwargs),
        actor,
        seeds,
        action_offset,
        device=device,
        normalizer=normalizer,
    )
    return rows, summarize_rollouts(rows)


def _validation_loss(actor: Any, dataset: ChunkDataset, family: str, device: torch.device) -> float:
    indices = np.arange(min(len(dataset), 512))
    examples = [dataset[int(index)] for index in indices]
    observations = torch.as_tensor(np.stack([value[0] for value in examples]), device=device)
    actions = torch.as_tensor(np.stack([value[1] for value in examples]), device=device)
    with torch.no_grad():
        # Diffusion validation uses a fixed RNG so only model/learning-rate changes vary it.
        torch.manual_seed(91_731)
        return float(actor.bc_loss(observations, actions))


def train_bc_with_retry(
    family: str,
    actor: Any,
    train_data: ChunkDataset,
    validation_data: ChunkDataset,
    device: torch.device,
    seed: int,
    result_dir: Path,
    critical: str,
    epsilon: float,
    eval_seeds_a: list[int],
    normalizer: Any,
    environment_kwargs: dict[str, Any],
) -> tuple[Any, list[dict[str, float]], dict[str, float], bool]:
    checkpoint = result_dir / "bc_checkpoint.pt"
    metrics: list[dict[str, float]] = []
    if checkpoint.exists():
        load_checkpoint(checkpoint, actor, map_location=device)
    else:
        # Project configs set 50k. SAFE_PILOTS_BC_STEPS is intentionally absent;
        # scientific budgets are fixed rather than environment-variable tunable.
        if family == "diffusion":
            metrics = train_diffusion_bc(
                actor,
                train_data,
                validation_data,
                steps=50_000,
                seed=seed,
                device=device,
            )
        else:
            metrics = train_gaussian_bc(
                actor,
                train_data,
                validation_data,
                steps=50_000,
                seed=seed,
                device=device,
            )
        save_checkpoint(checkpoint, actor, family=family, seed=seed, bc_steps=50_000)
    _, competence = evaluate(
        actor,
        "A",
        critical,
        epsilon,
        eval_seeds_a,
        seed * 1_000_000 + 100_000,
        normalizer,
        device,
        environment_kwargs,
    )
    passed = (
        competence["task_success"] >= 0.60
        and competence["safe_success"] >= 0.50
        and competence["p_damage_given_success"] <= 0.20
    )
    if passed:
        return actor, metrics, competence, False
    initial_state = copy.deepcopy(actor.state_dict())
    initial_loss = _validation_loss(actor, validation_data, family, device)
    retry, _ = policy_and_critic(family, train_data.states.shape[1], device)
    if family == "diffusion":
        retry_metrics = train_diffusion_bc(
            retry,
            train_data,
            validation_data,
            steps=100_000,
            learning_rate=1e-4,
            seed=seed,
            device=device,
        )
    else:
        retry_metrics = train_gaussian_bc(
            retry,
            train_data,
            validation_data,
            steps=100_000,
            learning_rate=1e-4,
            seed=seed,
            device=device,
        )
    retry_loss = _validation_loss(retry, validation_data, family, device)
    if retry_loss <= initial_loss:
        actor = retry
        metrics.extend({**row, "retry": 1.0} for row in retry_metrics)
    else:
        actor.load_state_dict(initial_state)
    _, competence = evaluate(
        actor,
        "A",
        critical,
        epsilon,
        eval_seeds_a,
        seed * 1_000_000 + 100_000,
        normalizer,
        device,
        environment_kwargs,
    )
    passed = (
        competence["task_success"] >= 0.60
        and competence["safe_success"] >= 0.50
        and competence["p_damage_given_success"] <= 0.20
    )
    save_checkpoint(
        checkpoint,
        actor,
        family=family,
        seed=seed,
        bc_steps=100_000,
        retry=True,
        validation_selected=True,
    )
    return actor, metrics, competence, True


def _extract_vector(observation: Any, extractor: ObjectCentricObservation, normalizer: Any) -> np.ndarray:
    if isinstance(observation, tuple):
        observation = observation[0]
    vector = extractor.extract(observation)
    return normalizer.normalize_observations(vector).astype(np.float32)


def online_finetune(
    family: str,
    actor: Any,
    critic: ValueCritic,
    critical: str,
    epsilon: float,
    normalizer: Any,
    device: torch.device,
    seed: int,
    output_dir: Path,
    eval_seeds_a: list[int],
    eval_seeds_b: list[int],
    environment_kwargs: dict[str, Any],
    maximum_steps: int = 150_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    extractor = ObjectCentricObservation()
    env = chunk_env("B", critical, epsilon, environment_kwargs)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    evaluation_steps = [0, 10_000, 25_000, 50_000, 100_000, 150_000]
    if maximum_steps > 150_000:
        evaluation_steps.append(300_000)
    rollout_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    primitive_steps = 0
    resumed = False
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    rollout_partial = output_dir / "rollout_metrics.partial.parquet"
    training_partial = output_dir / "training_metrics.partial.parquet"
    if progress_path.exists():
        summaries = json.loads(progress_path.read_text()).get("evaluations", {})
    if rollout_partial.exists():
        rollout_rows = pd.read_parquet(rollout_partial).to_dict("records")
    if training_partial.exists():
        training_rows = pd.read_parquet(training_partial).to_dict("records")
    existing = sorted(
        checkpoints_dir.glob("step_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if existing:
        metadata = load_checkpoint(
            existing[-1],
            actor,
            critic,
            actor_optimizer,
            optimizers={"critic": critic_optimizer},
            map_location=device,
        )
        primitive_steps = int(metadata.get("primitive_steps", 0))
        resumed = primitive_steps > 0

    def run_evaluation(step: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for context, seeds, offset in (
            ("A", eval_seeds_a, 300_000),
            ("B", eval_seeds_b, 400_000),
        ):
            rows, summary = evaluate(
                actor,
                context,
                critical,
                epsilon,
                seeds,
                seed * 1_000_000 + offset,
                normalizer,
                device,
                environment_kwargs,
            )
            result[context] = summary
            rollout_rows.extend(
                {
                    **row,
                    "context": context,
                    "checkpoint_steps": step,
                    "policy": family,
                    "training_seed": seed,
                }
                for row in rows
            )
        summaries[str(step)] = result
        write_json(progress_path, {"primitive_steps": step, "evaluations": summaries})
        pd.DataFrame(rollout_rows).to_parquet(rollout_partial, index=False)
        return result

    if primitive_steps == 0:
        run_evaluation(0)
    observation = env.reset(seed=seed)
    target_index = next(
        (index for index, value in enumerate(evaluation_steps) if value > primitive_steps),
        len(evaluation_steps),
    )
    consecutive_b_competent = 0
    # Preserve the stopping state when a 150k run is resumed for the single
    # predeclared 300k extension. The last pre-extension checkpoint can be the
    # first of the two required consecutive competent evaluations.
    for completed_step in sorted(
        (int(value) for value in summaries if int(value) <= primitive_steps),
        reverse=True,
    ):
        if summaries[str(completed_step)]["B"]["task_success"] < 0.70:
            break
        consecutive_b_competent += 1
    started = time.monotonic()
    try:
        while target_index < len(evaluation_steps) and primitive_steps < maximum_steps:
            target = min(evaluation_steps[target_index], maximum_steps)
            batch_obs: list[torch.Tensor] = []
            batch_actions: list[torch.Tensor] = []
            batch_log_probs: list[torch.Tensor] = []
            batch_rewards: list[float] = []
            batch_terminated: list[bool] = []
            batch_values: list[float] = []
            batch_traces: list[Any] = []
            while len(batch_obs) < 256 and primitive_steps < target:
                vector = _extract_vector(observation, extractor, normalizer)
                obs_tensor = torch.as_tensor(vector, device=device).unsqueeze(0)
                with torch.no_grad():
                    value = float(critic(obs_tensor))
                    if family == "diffusion":
                        action, log_prob, trace = actor.sample_with_log_prob(obs_tensor)
                        batch_traces.append(trace)
                    else:
                        action, log_prob = actor.sample_with_log_prob(obs_tensor)
                        batch_actions.append(action.detach())
                environment_action = normalizer.denormalize_actions(
                    action.squeeze(0).detach().cpu().numpy()
                )
                observation, reward, terminated, truncated, info = env.step(environment_action)
                executed = int(info["primitive_steps_executed"])
                primitive_steps += executed
                batch_obs.append(obs_tensor)
                batch_log_probs.append(log_prob.detach())
                batch_rewards.append(float(reward))
                batch_terminated.append(bool(terminated or truncated))
                batch_values.append(value)
                if terminated or truncated:
                    observation = env.reset()
            final_vector = _extract_vector(observation, extractor, normalizer)
            with torch.no_grad():
                bootstrap = float(critic(torch.as_tensor(final_vector, device=device).unsqueeze(0)))
            advantages_np, returns_np = generalized_advantage_estimate(
                np.asarray(batch_rewards, dtype=np.float32),
                np.asarray([*batch_values, bootstrap], dtype=np.float32),
                np.asarray(batch_terminated, dtype=bool),
            )
            obs_batch = torch.cat(batch_obs)
            old_batch = torch.cat(batch_log_probs)
            advantages = torch.as_tensor(advantages_np, device=device)
            returns = torch.as_tensor(returns_np, device=device)
            for epoch in range(10):
                if family == "diffusion":
                    stats = diffusion_dppo_update(
                        actor,
                        critic,
                        actor_optimizer,
                        critic_optimizer,
                        obs_batch,
                        stack_diffusion_traces(batch_traces),
                        old_batch,
                        advantages,
                        returns,
                    )
                else:
                    stats = gaussian_ppo_update(
                        actor,
                        critic,
                        actor_optimizer,
                        critic_optimizer,
                        obs_batch,
                        torch.cat(batch_actions),
                        old_batch,
                        advantages,
                        returns,
                    )
                training_rows.append(
                    {
                        **stats.__dict__,
                        "primitive_steps": primitive_steps,
                        "ppo_epoch": epoch,
                        "walltime_seconds": time.monotonic() - started,
                        "mean_rollout_reward": float(np.mean(batch_rewards)),
                    }
                )
            if primitive_steps >= target:
                evaluation = run_evaluation(target)
                save_checkpoint(
                    checkpoints_dir / f"step_{target}.pt",
                    actor,
                    critic,
                    actor_optimizer,
                    optimizers={"critic": critic_optimizer},
                    primitive_steps=target,
                    family=family,
                    seed=seed,
                )
                pd.DataFrame(training_rows).to_parquet(training_partial, index=False)
                consecutive_b_competent = (
                    consecutive_b_competent + 1
                    if evaluation["B"]["task_success"] >= 0.70
                    else 0
                )
                target_index += 1
                if consecutive_b_competent >= 2:
                    break
                if target == 150_000 and maximum_steps > 150_000:
                    b_zero = summaries.get("0", {}).get("B", {}).get("task_success", 0.0)
                    b_100 = summaries.get("100000", {}).get("B", {}).get("task_success", 0.0)
                    b_150 = summaries.get("150000", {}).get("B", {}).get("task_success", 0.0)
                    extension_trigger = (
                        b_150 - b_zero >= 0.15
                        and b_150 - b_100 >= 0.0
                    )
                    if not extension_trigger:
                        break
    finally:
        env.close()
    summary = {
        "family": family,
        "seed": seed,
        "actor_parameters": parameter_count(actor),
        "critic_parameters": parameter_count(critic),
        "primitive_steps": primitive_steps,
        "resumed": resumed,
        "evaluations": summaries,
        "b_competence_reached": consecutive_b_competent >= 2,
        "walltime_seconds": time.monotonic() - started,
    }
    return rollout_rows, training_rows, summary


def run_policy_pipeline(
    family: str,
    project_root: Path,
    run_root: Path,
    seed: int,
    output_dir: Path,
    maximum_steps: int = 150_000,
) -> dict[str, Any]:
    seed_everything(seed)
    gate = read_gate(run_root)
    if not gate["proceed_gpu"]:
        result = {"skipped": True, "reason": "Pilot 0 gate is red", "gate": gate}
        write_json(output_dir / "summary.json", result)
        return result
    critical = gate["critical_fragile_object"]
    epsilon = float(gate["epsilon_damage"])
    environment_kwargs = dict(gate.get("environment_kwargs", {}))
    processed = run_root / "data" / "processed" / f"seed_{seed}"
    train_data = ChunkDataset(processed / "train.npz")
    validation_data = ChunkDataset(processed / "validation.npz")
    normalizer = load_normalization(processed / "normalization.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, critic = policy_and_critic(family, train_data.states.shape[1], device)
    eval_a, eval_b = write_eval_seeds(run_root, seed)
    actor, bc_metrics, competence, retried = train_bc_with_retry(
        family,
        actor,
        train_data,
        validation_data,
        device,
        seed,
        output_dir,
        critical,
        epsilon,
        eval_a,
        normalizer,
        environment_kwargs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if bc_metrics:
        pd.DataFrame(bc_metrics).to_parquet(output_dir / "bc_training_metrics.parquet", index=False)
    competence_passed = (
        competence["task_success"] >= 0.60
        and competence["safe_success"] >= 0.50
        and competence["p_damage_given_success"] <= 0.20
    )
    if not competence_passed:
        failure = {
            "skipped_rl": True,
            "reason": "BC_COMPETENCE_GATE_FAILED_AFTER_RETRY",
            "bc_competence": competence,
            "retry_used": retried,
        }
        write_json(output_dir / "failure.json", failure)
        write_json(output_dir / "summary.json", failure)
        return failure
    rollout_rows, training_rows, summary = online_finetune(
        family,
        actor,
        critic,
        critical,
        epsilon,
        normalizer,
        device,
        seed,
        output_dir,
        eval_a,
        eval_b,
        environment_kwargs,
        maximum_steps,
    )
    summary["bc_competence"] = competence
    summary["bc_retry_used"] = retried
    pd.DataFrame(rollout_rows).to_parquet(output_dir / "rollout_metrics.parquet", index=False)
    pd.DataFrame(training_rows).to_parquet(output_dir / "training_metrics.parquet", index=False)
    write_json(output_dir / "summary.json", summary)
    return summary
