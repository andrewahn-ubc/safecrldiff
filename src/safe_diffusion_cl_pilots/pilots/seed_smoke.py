from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safe_diffusion_cl_pilots.envs.object_centric_obs import ObjectCentricObservation
from safe_diffusion_cl_pilots.models.diffusion_policy import DiffusionPolicy
from safe_diffusion_cl_pilots.models.gaussian_chunk_policy import GaussianChunkPolicy
from safe_diffusion_cl_pilots.utils.logging import write_json
from safe_diffusion_cl_pilots.utils.manifests import atomic_success
from safe_diffusion_cl_pilots.utils.seeding import seed_everything

from .training import chunk_env, read_gate


def _reset(env: Any, seed: int | None = None) -> Any:
    value = env.reset(seed=seed) if seed is not None else env.reset()
    return value[0] if isinstance(value, tuple) else value


def _exercise(
    actor: Any,
    env: Any,
    extractor: ObjectCentricObservation,
    device: torch.device,
    primitive_target: int,
    seed: int,
) -> dict[str, Any]:
    observation = _reset(env, seed)
    primitive_steps = 0
    chunks = 0
    while primitive_steps < primitive_target:
        vector = extractor.extract(observation)
        tensor = torch.as_tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = actor.sample_with_log_prob(tensor)[:2]
        if not torch.isfinite(action).all() or not torch.isfinite(log_prob).all():
            raise FloatingPointError("non-finite policy output in seed smoke test")
        observation, reward, terminated, truncated, info = env.step(
            action.squeeze(0).cpu().numpy()
        )
        if not np.isfinite(reward):
            raise FloatingPointError("non-finite environment reward in seed smoke test")
        primitive_steps += int(info["primitive_steps_executed"])
        chunks += 1
        if terminated or truncated:
            observation = _reset(env)
    return {"primitive_steps": primitive_steps, "action_chunks": chunks}


def run(run_root: Path, seed: int, primitive_steps_per_policy: int = 200) -> None:
    output = run_root / "results" / f"seed_{seed}" / "smoke"
    if (output / "_SUCCESS").exists():
        return
    gate = read_gate(run_root)
    if not gate["proceed_gpu"]:
        raise RuntimeError("seed smoke must not run when Pilot 0 is red")
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"A100 job cannot access CUDA (torch={torch.__version__}, "
            f"torch.version.cuda={torch.version.cuda}, devices={torch.cuda.device_count()})"
        )
    device = torch.device("cuda")
    extractor = ObjectCentricObservation()
    results: dict[str, Any] = {}
    for index, actor_class in enumerate((DiffusionPolicy, GaussianChunkPolicy)):
        family = "diffusion" if actor_class is DiffusionPolicy else "gaussian"
        actor = None
        env = chunk_env(
            "A",
            gate["critical_fragile_object"],
            float(gate["epsilon_damage"]),
            gate.get("environment_kwargs", {}),
        )
        try:
            actor = actor_class(extractor.dimension).to(device).eval()
            results[family] = _exercise(
                actor,
                env,
                extractor,
                device,
                primitive_steps_per_policy,
                seed * 10 + index,
            )
        finally:
            env.close()
            if actor is not None:
                del actor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_json(output / "status.json", {"status": "PASS", "seed": seed, **results})
    atomic_success(output, {"seed": seed})


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-seed model/environment GPU smoke test")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--primitive-steps-per-policy", type=int, default=200)
    arguments = parser.parse_args()
    run(arguments.run_root, arguments.seed, arguments.primitive_steps_per_policy)


if __name__ == "__main__":
    main()
