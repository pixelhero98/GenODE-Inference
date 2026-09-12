"""Contracts needed for fair ReFlow comparisons, independent of large assets."""

import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from genode.image_comparators.fid import (
    FeatureMoments,
    edm_uint8,
    evaluate_moments,
    fid_distance,
    seeded_noise,
)
from genode.image_comparators.reflow import (
    RF_END,
    RF_START,
    ReflowVelocity,
    _upstream_imports,
    euler_sample,
    load_reflow,
    reflow_grid,
    restore_ema,
    validate_executed_model_times,
    validate_learned_grids,
)


def test_moments_count_partial_batch_and_merge_stably():
    features = np.random.default_rng(7).normal(size=(11, 3)) + 1e6
    moments = FeatureMoments()
    for batch in (features[:4], features[4:8], features[8:]):
        moments.update(batch)
    mean, covariance = moments.statistics()
    assert moments.count == 11
    np.testing.assert_allclose(mean, features.mean(0), rtol=0, atol=1e-9)
    np.testing.assert_allclose(covariance, np.cov(features, rowvar=False), rtol=1e-9, atol=1e-9)


def test_fid_known_gaussians_and_rejects_invalid_inputs():
    assert fid_distance([0, 0], np.eye(2), [1, 2], np.diag([4, 9])) == pytest.approx(10)
    assert fid_distance([0, 0], np.eye(2), [0, 0], np.eye(2)) == pytest.approx(0)
    with pytest.raises(ValueError, match="non-finite"):
        fid_distance([np.nan], [[1]], [0], [[1]])
    with pytest.raises(ValueError, match="symmetric"):
        fid_distance([0, 0], [[1, 1], [0, 1]], [0, 0], np.eye(2))


def test_exact_count_evaluation_and_seed_replay_across_batch_sizes():
    class Detector:
        def __call__(self, images, *, return_features):
            assert return_features and images.dtype == torch.uint8
            return images.float().mean(dim=(2, 3))

    seeds = list(range(11))
    rng = torch.get_rng_state().clone()
    one = evaluate_moments(lambda x: x, Detector(), seeds, batch_size=4, device="cpu")
    two = evaluate_moments(lambda x: x, Detector(), seeds, batch_size=7, device="cpu")
    assert torch.equal(rng, torch.get_rng_state())
    assert one.count == two.count == 11
    for left, right in zip(one.statistics(), two.statistics(), strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    noise = seeded_noise(seeds, device="cpu")
    assert torch.equal(noise[6], seeded_noise([6], device="cpu")[0])


def test_detector_conversion_uses_edm_clamp_and_truncation():
    values = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32)[None, None, None].expand(1, 3, 1, 5)
    assert edm_uint8(values)[0, 0, 0].tolist() == [0, 0, 127, 255, 255]


@pytest.mark.parametrize("nfe", [4, 6, 8, 10])
def test_reflow_euler_exact_nfe_and_model_time_convention(nfe):
    class ConstantField(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.times = []

        def forward(self, x, time):
            self.times.append(time.clone())
            return torch.ones_like(x)

    network = ConstantField()
    velocity = ReflowVelocity(network)
    initial = torch.zeros(2, 3, 2, 2)
    progress = torch.linspace(0, 1, nfe + 1, dtype=torch.float64)
    result = euler_sample(velocity, initial, progress)
    assert velocity.calls == nfe
    assert torch.allclose(result, torch.full_like(result, RF_END - RF_START))
    grid = reflow_grid(progress, device="cpu")
    assert torch.equal(torch.stack(network.times)[:, 0], grid[:-1] * 1000)
    assert torch.equal(velocity.bezier(initial, torch.tensor(99), grid[0]), torch.ones_like(initial))


def test_grid_rejects_float32_collapse():
    with pytest.raises(ValueError, match="collapses"):
        reflow_grid([0, 0.5, 0.5 + 1e-12, 1], device="cpu")


def test_reflow_rejects_unverified_pickle_before_upstream_import(tmp_path):
    model = tmp_path / "ImageGeneration/models/ncsnpp.py"
    model.parent.mkdir(parents=True)
    model.write_text("raise RuntimeError('must not import')")
    checkpoint = tmp_path / "reflow_1.pth"
    checkpoint.write_bytes(b"not the official checkpoint")
    with pytest.raises(ValueError, match="verified official"):
        load_reflow(tmp_path, checkpoint, {}, device="cpu")


def test_learned_and_transformed_solver_times_reject_zero_width_calls():
    grid = reflow_grid([0, 0.25, 0.5, 0.75, 1], device="cpu")
    validate_learned_grids(grid, grid, 4)
    validate_executed_model_times((grid[:-1] * 1000).tolist(), 4)
    collapsed = grid.clone()
    collapsed[2] = collapsed[1]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_learned_grids(grid, collapsed, 4)
    with pytest.raises(ValueError, match="endpoints"):
        validate_learned_grids(grid + 0.01, grid, 4)
    with pytest.raises(ValueError, match="distinct ordered"):
        validate_executed_model_times([0.1, 200, 200, 999], 4)


def test_ema_preserves_buffers_and_rejects_missing_shadows():
    net = torch.nn.BatchNorm1d(2)
    state = {"model": net.state_dict(), "ema": {"shadow_params": [torch.full((2,), 3.0), torch.full((2,), 4.0)]}}
    state["model"]["running_mean"] = torch.ones(2)
    restore_ema(net, state)
    assert torch.equal(net.weight, torch.full((2,), 3.0))
    assert torch.equal(net.bias, torch.full((2,), 4.0))
    assert torch.equal(net.running_mean, torch.ones(2))
    state["ema"]["shadow_params"].pop()
    with pytest.raises(ValueError, match="tensor count"):
        restore_ema(net, state)


def test_two_time_baselines_allow_tied_model_times_but_not_integration_steps():
    grid = reflow_grid([0, 0.25, 0.5, 0.75, 1], device="cpu")
    times = grid.clone()
    times[2] = times[1]
    validate_learned_grids(grid, times, 4, allow_repeated_model_times=True)
    validate_executed_model_times(times[:-1] * 1000, 4, allow_repeated_model_times=True)

    class StateField(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.states = []

        def forward(self, state, time):
            self.states.append(state.clone())
            return state + 1

    network = StateField()
    velocity = ReflowVelocity(network)
    state = torch.zeros(1, 3, 2, 2)
    for left, right, time in zip(grid[:-1], grid[1:], times[:-1], strict=True):
        state = state + (right - left) * velocity(state, time)
    assert velocity.calls == 4 and torch.isfinite(state).all()
    assert not torch.equal(network.states[1], network.states[2])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_learned_grids(times, grid, 4, allow_repeated_model_times=True)
    times[2] = times[1] - 0.01
    with pytest.raises(ValueError, match="nondecreasing"):
        validate_learned_grids(grid, times, 4, allow_repeated_model_times=True)
    with pytest.raises(ValueError, match="nondecreasing"):
        validate_executed_model_times(times[:-1] * 1000, 4, allow_repeated_model_times=True)


def test_upstream_import_scope_restores_other_methods_after_failure(tmp_path, monkeypatch):
    existing = ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", existing)
    previous_path = sys.path.copy()
    with pytest.raises(RuntimeError, match="import failed"), _upstream_imports(tmp_path):
        assert "models" not in sys.modules
        sys.modules["models"] = ModuleType("models")
        sys.modules["models.synthetic_child"] = ModuleType("models.synthetic_child")
        raise RuntimeError("import failed")
    assert sys.modules["models"] is existing
    assert "models.synthetic_child" not in sys.modules
    assert sys.path == previous_path
