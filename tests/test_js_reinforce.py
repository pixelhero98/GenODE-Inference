import numpy as np
import pytest
import torch

from genode.latent_clock.adapters.sana import _prepare_scheduler
from genode.latent_clock.js_reinforce import (
    MIN_INTERVAL,
    DirichletSchedulePolicy,
    JSArchitecture,
    james_stein_baseline,
    materialize_js,
    reinforce_loss,
    sample_intervals,
)


def test_empirical_baseline_matches_paper_equations():
    rewards = torch.tensor([[0.0, 2.0], [1.0, 3.0], [10.0, 12.0]], requires_grad=True)
    baseline, stats = james_stein_baseline(rewards)
    alpha = 8 / (8 + 39.68)
    expected = (1 - alpha) * rewards.detach().double().flip(1) + alpha * (28 - rewards.detach().double()) / 5
    torch.testing.assert_close(baseline, expected)
    assert stats["within_variance"] == 8
    assert stats["between_variance"] == pytest.approx(39.68)
    assert not baseline.requires_grad


def test_baseline_constant_rewards_and_scalar_units():
    constant = torch.full((4, 2), 7.0)
    baseline, stats = james_stein_baseline(constant)
    torch.testing.assert_close(baseline, constant.double())
    assert stats["shrinkage"] == 0
    rewards = torch.tensor([[0.0, 2.0], [1.0, 3.0], [10.0, 12.0]])
    base, original = james_stein_baseline(rewards)
    scaled, changed = james_stein_baseline(3 * rewards + 4)
    torch.testing.assert_close(scaled, 3 * base + 4)
    assert changed["shrinkage"] == pytest.approx(original["shrinkage"])
    with pytest.raises(ValueError, match="three contexts"):
        james_stein_baseline(torch.ones(2, 2))


@pytest.mark.parametrize("nfe", [4, 6, 8, 10])
def test_skipped_interval_mapping_and_actual_euler_calls(nfe):
    action = np.arange(1, nfe + 2, dtype=float)
    action /= action.sum()
    clock = materialize_js(action)
    mass = (1 - (nfe + 1) * MIN_INTERVAL) * action + MIN_INTERVAL
    np.testing.assert_allclose(clock.nodes[:-1], np.cumsum(mass)[:-1], rtol=0, atol=1e-15)
    assert clock.nodes[0] > 0 and clock.nodes[-1] == 1

    class Scheduler:
        class config:
            num_train_timesteps = 1000

        def set_timesteps(self, steps, device):
            self.num_inference_steps = steps

    scheduler = Scheduler()
    times = _prepare_scheduler(scheduler, clock, "cpu")
    calls, value = 0, 0.0
    for i, _ in enumerate(times):
        calls += 1
        value += float(scheduler.sigmas[i + 1] - scheduler.sigmas[i])
    assert calls == nfe == scheduler.num_inference_steps
    assert value == pytest.approx(-float(scheduler.sigmas[0]), abs=1e-7)
    assert torch.all(torch.diff(scheduler.sigmas) < 0)


def test_extreme_intervals_keep_solver_grid_without_redrawing():
    raw = np.full(11, np.finfo(float).tiny)
    raw[0] = 1.0
    schedule = materialize_js(raw)
    assert schedule.raw_intervals == tuple(raw)
    assert np.all(np.diff(np.asarray(schedule.nodes, dtype=np.float32)) > 0)
    with pytest.raises(ValueError, match="positive finite"):
        materialize_js([0, 0.5, 0.5])


def test_joint_sampling_replay_and_generation_rng_isolation():
    concentration = torch.full((3, 5), 0.7)
    state = torch.random.get_rng_state().clone()
    first = sample_intervals(concentration, seed=41, rollouts=2)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(first, sample_intervals(concentration, seed=41, rollouts=2))
    assert not torch.equal(first, sample_intervals(concentration, seed=42, rollouts=2))
    assert first.shape == (3, 2, 5)
    torch.testing.assert_close(first.sum(-1), torch.ones(3, 2, dtype=torch.double))
    assert not torch.equal(first[:, 0], first[:, 1])


def test_native_noise_text_conditioning_and_finite_policy_gradients():
    torch.manual_seed(9)
    architecture = JSArchitecture(2, 6, 4, pooled_dim=3, blocks=2, conv_width=8, attention_heads=2)
    policy = DirichletSchedulePolicy(architecture)
    noise = torch.randn(3, 2, 8, 8, requires_grad=True)
    text = torch.randn(3, 5, 6, requires_grad=True)
    pooled = torch.randn(3, 3, requires_grad=True)
    mask = torch.tensor([[False, False, False, False, True]] * 3)
    alpha = policy(noise, text, pooled, mask)
    assert alpha.shape == (3, 5) and (alpha >= 1e-3).all()
    assert not torch.allclose(alpha, policy(noise + 1, text, pooled, mask))
    assert not torch.allclose(alpha, policy(noise, text + 1, pooled, mask))
    altered_padding = text.detach().clone()
    altered_padding[:, -1] = 1e5
    torch.testing.assert_close(alpha, policy(noise, altered_padding, pooled, mask))
    actions = sample_intervals(alpha, seed=17, rollouts=2)
    loss, _ = reinforce_loss(alpha, actions, torch.tensor([[1.0, 2], [-1.0, 3], [2.0, 0]]))
    loss.backward()
    gradients = [parameter.grad for parameter in policy.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    assert noise.grad is text.grad is pooled.grad is None
