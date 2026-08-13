import io

import torch

from safe_diffusion_cl_pilots.models.critics import ValueCritic, parameter_count
from safe_diffusion_cl_pilots.models.diffusion_policy import DiffusionPolicy
from safe_diffusion_cl_pilots.models.gaussian_chunk_policy import GaussianChunkPolicy
from safe_diffusion_cl_pilots.train.common import clipped_policy_loss
from safe_diffusion_cl_pilots.train.finetune_diffusion_dppo import diffusion_dppo_update


def test_policy_samples_and_log_probabilities_are_finite_and_bounded():
    observations = torch.randn(4, 109)
    diffusion = DiffusionPolicy(109)
    actions, stored, trace = diffusion.sample_with_log_prob(observations)
    recomputed = diffusion.log_prob(observations, trace)
    assert torch.isfinite(actions).all() and torch.max(torch.abs(actions)) <= 1
    torch.testing.assert_close(stored, recomputed)
    gaussian = GaussianChunkPolicy(109)
    actions, stored = gaussian.sample_with_log_prob(observations)
    recomputed = gaussian.log_prob(observations, actions)
    assert torch.isfinite(actions).all() and torch.max(torch.abs(actions)) <= 1
    torch.testing.assert_close(stored, recomputed, atol=2e-4, rtol=2e-4)


def test_ppo_ratio_is_one_before_update_and_gradients_are_nonzero():
    observations = torch.randn(4, 109)
    policy = GaussianChunkPolicy(109)
    actions, old = policy.sample_with_log_prob(observations)
    new = policy.log_prob(observations, actions.detach())
    loss, ratio = clipped_policy_loss(new, old.detach(), torch.tensor([-1.0, -0.5, 0.5, 1.0]))
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=2e-4, rtol=2e-4)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in policy.parameters())


def test_dppo_transition_ratio_is_one_before_update_and_updates_denoiser():
    observations = torch.randn(4, 109)
    actor = DiffusionPolicy(109)
    critic = ValueCritic(109)
    _, old, transitions = actor.sample_with_log_prob(observations)
    stats = diffusion_dppo_update(
        actor,
        critic,
        torch.optim.Adam(actor.parameters(), lr=1e-4),
        torch.optim.Adam(critic.parameters(), lr=1e-4),
        observations,
        transitions,
        old.detach(),
        torch.tensor([-1.0, -0.5, 0.5, 1.0]),
        torch.randn(4),
    )
    assert abs(stats.ratio_mean - 1.0) < 1e-5
    assert stats.gradient_norm > 0


def test_serialization_preserves_samples_under_fixed_rng():
    observations = torch.randn(2, 109)
    for constructor in (DiffusionPolicy, GaussianChunkPolicy):
        first = constructor(109)
        stream = io.BytesIO()
        torch.save(first.state_dict(), stream)
        stream.seek(0)
        second = constructor(109)
        second.load_state_dict(torch.load(stream, weights_only=True))
        torch.manual_seed(123)
        action_a = first.sample(observations)
        torch.manual_seed(123)
        action_b = second.sample(observations)
        torch.testing.assert_close(action_a, action_b, atol=0, rtol=0)


def test_actor_capacity_is_within_twenty_percent():
    diffusion = parameter_count(DiffusionPolicy(109))
    gaussian = parameter_count(GaussianChunkPolicy(109))
    assert abs(diffusion - gaussian) / diffusion <= 0.20
