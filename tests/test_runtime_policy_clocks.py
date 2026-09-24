"""Pure runtime-adapter units with synthetic tensors and a stub sampler; no model fitting."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from genode.evaluation.molecule_metrics import _sample_molecule_ar_rollout
from genode.evaluation.otflow_evaluation_support import evaluate_forecast_schedule
from genode.gico.clocks import materialize, verify_measurement_clock
from genode.models.conditioning import ConditioningCache
from genode.models.config import OTFlowConfig


class NativeBackbone(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.histories = []

    def precompute(self, hist, *, cond=None):
        self.histories.append(hist.clone())
        summary = hist.mean(dim=(1, 2))[:, None].repeat(1, self.cfg.model.hidden_dim) * self.weight
        return ConditioningCache(ctx_tokens=hist, summary=summary)


class StubSampler(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.cfg = OTFlowConfig()
        self.cfg.model.hidden_dim = 4
        self.cfg.train.device = torch.device("cpu")
        self.backbone = NativeBackbone(self.cfg)
        self.width = width
        self.grids = []
        self.draws = []
        self.fail = False

    def sample_future(self, hist, *, steps, solver):
        self.grids.append(tuple(self.cfg.sample.time_grid))
        if self.fail:
            raise RuntimeError("injected sampler failure")
        draw = torch.rand((1, 1, self.width))
        self.draws.append(draw.clone())
        return draw


class RecordingPolicy:
    artifact_sha256 = "fixture-policy"
    policy_kind = "stochastic"

    def __init__(self, task):
        self.metadata = {"task": task, "backbone": "fixture-backbone"}
        self.calls = []

    def density(self, context, solver, nfe, *, seed, request_id):
        self.calls.append((np.asarray(context).copy(), solver, nfe, seed, request_id))
        mass = np.full(64, 1 / 128)
        mass[:32] = 3 / 128
        return mass if request_id.endswith("0") else mass[::-1].copy()


class MoleculeFixture:
    data = SimpleNamespace(atom_count=3)
    stats = SimpleNamespace(context_mean=np.zeros(9), context_std=np.ones(9))

    def context_features_from_history_coords(self, history):
        return history[-2:].reshape(2, 9).astype(np.float32)

    def denormalize_target(self, value):
        return value


@pytest.mark.parametrize("fail", [False, True])
def test_molecular_member_clock_sampled_once_reused_and_restored(fail):
    model = StubSampler(9)
    policy = RecordingPolicy("molecule_3d_set1")
    history = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    model.fail = fail
    clock_records = []
    before = {key: value.clone() for key, value in model.state_dict().items()}
    kwargs = {
        "model": model,
        "ds": MoleculeFixture(),
        "history_coords": history,
        "rollout_steps": 3,
        "nfe": 2,
        "solver": "heun",
        "device": torch.device("cpu"),
        "seed": 77,
        "policy": policy,
        "clock_seed": 5,
        "clock_request_id": "member:0",
        "clock_records": clock_records,
    }
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            _sample_molecule_ar_rollout(**kwargs)
    else:
        result = _sample_molecule_ar_rollout(**kwargs)
        assert result.shape == (3, 3, 3)
        assert not np.array_equal(result[0], result[-1])
    assert clock_records == [
        {
            "density_mass": np.r_[np.full(32, 3 / 128), np.full(32, 1 / 128)].tolist(),
            "time_grid": list(materialize(np.r_[np.full(32, 3 / 128), np.full(32, 1 / 128)], "heun", 4)),
            "clock_seed": 5,
            "clock_request_id": "member:0",
            "generation_seed": 77,
            "policy_sha256": "fixture-policy",
            "policy_kind": "stochastic",
            "solver": "heun",
            "nfe_per_horizon": 4,
            "trajectory_nfe": 12,
        }
    ]
    assert len(policy.calls) == 1
    assert policy.calls[0][1:] == ("heun", 4, 5, "member:0")
    assert model.grids == [tuple(clock_records[0]["time_grid"])] * (1 if fail else 3)
    assert len(model.backbone.histories) == 1
    np.testing.assert_array_equal(model.backbone.histories[0][0].numpy(), history.reshape(2, 9))
    assert tuple(model.cfg.sample.time_grid) == ()
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert all(param.grad is None for param in model.parameters())


class ForecastFixture:
    horizon = 1

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.tensor([[1.0], [3.0]]), torch.tensor([4.0]), {"series_idx": 0, "target_t": 2}

    def denormalize_block(self, value, index):
        return value

    def mase_denom(self, index):
        return 1.0


def test_forecast_members_share_frozen_observed_context_and_separate_clock_requests():
    model = StubSampler(1)
    policy = RecordingPolicy("traffic_hourly")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    kwargs = {
        "model": model,
        "ds": ForecastFixture(),
        "cfg": model.cfg,
        "solver_name": "heun",
        "runtime_nfe": 2,
        "time_grid": (0, 0.5, 1),
        "num_eval_samples": 2,
        "seed": 31,
        "logical_seed": 8,
        "target_nfe": 4,
        "scheduler_key": "policy",
        "dataset_key": "traffic_hourly",
        "split_phase": "validation_tuning",
        "checkpoint_id": "fixture-backbone",
        "example_indices": [0],
        "return_per_example_rows": True,
    }
    result = evaluate_forecast_schedule(**kwargs, policy=policy, clock_seed=12)
    assert len(policy.calls) == 2
    np.testing.assert_array_equal(policy.calls[0][0], [4.0] * 4)
    np.testing.assert_array_equal(policy.calls[0][0], policy.calls[1][0])
    assert [call[1:4] for call in policy.calls] == [("heun", 4, 12)] * 2
    assert policy.calls[0][-1].endswith(":8:member:0")
    assert policy.calls[1][-1].endswith(":8:member:1")
    assert len(model.backbone.histories) == 1
    row = result["per_example_rows"][0]
    assert model.grids[0] != model.grids[1]
    assert row["sample_time_grids"] == [list(grid) for grid in model.grids]
    for clock in row["sample_clocks"]:
        verify_measurement_clock({**clock, "solver": "heun", "nfe": 4, "schedule_key": "policy"})
    assert tuple(model.cfg.sample.time_grid) == ()
    candidate_draws = [value.clone() for value in model.draws]
    model.draws.clear()
    evaluate_forecast_schedule(**kwargs)
    assert all(torch.equal(left, right) for left, right in zip(candidate_draws, model.draws, strict=True))
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert all(param.grad is None for param in model.parameters())


def test_forecast_policy_rejects_unpaired_batch_and_checkpoint_scope():
    model, policy = StubSampler(1), RecordingPolicy("traffic_hourly")
    kwargs = {
        "model": model,
        "ds": ForecastFixture(),
        "cfg": model.cfg,
        "solver_name": "euler",
        "runtime_nfe": 2,
        "time_grid": (0, 0.5, 1),
        "num_eval_samples": 2,
        "seed": 1,
        "policy": policy,
        "dataset_key": "traffic_hourly",
        "checkpoint_id": "fixture-backbone",
    }
    with pytest.raises(ValueError, match="batch_size=1"):
        evaluate_forecast_schedule(**kwargs, batch_size=2)
    kwargs["checkpoint_id"] = "different-backbone"
    with pytest.raises(ValueError, match="task/backbone"):
        evaluate_forecast_schedule(**kwargs)
    assert not policy.calls
