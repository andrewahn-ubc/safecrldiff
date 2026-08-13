from __future__ import annotations

import argparse

import numpy as np
import torch

from safe_diffusion_cl_pilots.envs.chunked_gym_wrapper import ChunkedGymWrapper
from safe_diffusion_cl_pilots.envs.object_centric_obs import ObjectCentricObservation
from safe_diffusion_cl_pilots.envs.shelve_contexts import make_context
from safe_diffusion_cl_pilots.models.diffusion_policy import DiffusionPolicy
from safe_diffusion_cl_pilots.models.gaussian_chunk_policy import GaussianChunkPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critical", default="wine_glass")
    arguments = parser.parse_args()
    extractor = ObjectCentricObservation()
    extractor.assert_no_leakage()
    for context in ("A", "B"):
        raw = make_context(context, critical_fragile_object=arguments.critical)
        env = ChunkedGymWrapper(raw, critical_fragile_object=arguments.critical)
        try:
            reset = env.reset(seed=0)
            observation = reset[0] if isinstance(reset, tuple) else reset
            vector = extractor.extract(observation)
            for _ in range(100):
                observation, _, terminated, truncated, _ = raw.step(
                    np.random.default_rng(0).uniform(-1, 1, size=7)
                )
                if terminated or truncated:
                    reset = raw.reset()
                    observation = reset[0] if isinstance(reset, tuple) else reset
            env.reset(seed=1)
            env.step(np.zeros((8, 7), dtype=np.float32))
        finally:
            env.close()
    for policy in (DiffusionPolicy(len(vector)), GaussianChunkPolicy(len(vector))):
        observation_tensor = torch.as_tensor(vector).unsqueeze(0)
        if isinstance(policy, DiffusionPolicy):
            action, log_prob, trace = policy.sample_with_log_prob(observation_tensor)
            recomputed = policy.log_prob(observation_tensor, trace)
        else:
            action, log_prob = policy.sample_with_log_prob(observation_tensor)
            recomputed = policy.log_prob(observation_tensor, action)
        assert torch.isfinite(action).all() and torch.isfinite(log_prob).all()
        assert torch.allclose(log_prob, recomputed, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    main()

