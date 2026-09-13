import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from genode.gico.conditioning import Conditioning
from genode.latent_clock.adapters.ipndm import IPNDMAdapter, VariableStepIPNDM
from genode.latent_clock.clocks import Clock
from genode.solver_protocol import normalize_solver_nfe_fields, solver_effective_order, solver_runtime_name


class Schedule:
    T, eps = 3.0, 0.01

    def marginal_alpha(self, t):
        return torch.rsqrt(1 + t.square())

    def marginal_std(self, t):
        return t * self.marginal_alpha(t)

    def prior_transformation(self, x):
        return x


def test_coefficients_use_nonlinear_sigma_over_alpha_not_time_intervals():
    class VPSchedule(Schedule):
        def marginal_alpha(self, t):
            return torch.exp(-t / 2)

        def marginal_std(self, t):
            return torch.sqrt(-torch.expm1(-t))

    schedule = VPSchedule()
    times = torch.tensor([3.0, 2.6, 1.2, 0.1], dtype=torch.float64)
    rho = schedule.marginal_std(times) / schedule.marginal_alpha(times)

    def field(x, t, *args):
        return (schedule.marginal_std(t) / schedule.marginal_alpha(t)).reshape_as(x)

    result = VariableStepIPNDM(schedule).sample_simple(field, torch.zeros(1, dtype=torch.float64), times, times)
    first = (rho[1] - rho[0]) * rho[0]
    exact = first + (rho[-1].square() - rho[1].square()) / 2
    torch.testing.assert_close(result / schedule.marginal_alpha(times[-1]), exact.reshape_as(result))


@pytest.mark.parametrize("grid", [[3, 2, 1], [3, 2, 1.5], [3, 2.5, 1], [3, 2.9, 1.8, 0.01]])
def test_affine_field_integrates_exactly_after_common_euler_start(grid):
    schedule = Schedule()
    times = torch.tensor(grid, dtype=torch.float64)
    calls = []

    def field(x, t, condition, uncondition):
        calls.append(t.clone())
        return torch.ones_like(x) * t[:, None]

    result = VariableStepIPNDM(schedule).sample_simple(field, torch.zeros(2, 1, dtype=torch.float64), times, times)
    first = (times[1] - times[0]) * times[0]
    exact_y = first + (times[-1].square() - times[1].square()) / 2
    torch.testing.assert_close(result / schedule.marginal_alpha(times[-1]), exact_y.expand(2, 1))
    assert len(calls) == len(grid) - 1
    torch.testing.assert_close(torch.stack(calls)[:, 0], times[:-1])


def test_constant_field_preserves_dtype_and_one_step_semantics():
    times = torch.tensor([3.0, 0.01])
    x = torch.ones(1, 2)
    schedule = Schedule()
    result = VariableStepIPNDM(schedule).sample_simple(lambda x, *args: torch.ones_like(x), x, times, times)
    expected_y = x / schedule.marginal_alpha(times[0]) + times[1] - times[0]
    assert result.dtype == x.dtype
    torch.testing.assert_close(result / schedule.marginal_alpha(times[-1]), expected_y)


@pytest.mark.parametrize("grid", [[3, 2, 2], [3, 1, 2], [3, float("nan"), 1], [3], [4, 1], [3, 0]])
def test_invalid_grid_rejected_before_model_evaluation(grid):
    times = torch.tensor(grid, dtype=torch.float32)
    calls = []
    with pytest.raises(ValueError):
        VariableStepIPNDM(Schedule()).sample_simple(lambda *args: calls.append(1), torch.zeros(1, 1), times, times)
    assert not calls


def test_separate_evaluation_grid_is_rejected():
    solver = VariableStepIPNDM(Schedule())
    with pytest.raises(ValueError, match="identical"):
        solver.sample_simple(None, torch.zeros(1, 1), torch.tensor([3.0, 1.0]), torch.tensor([3.0, 2.0]))


def test_solver_conditioning_and_artifact_keys_remain_distinct():
    assert solver_effective_order("ipndm_v") == 2
    assert normalize_solver_nfe_fields("ipndm_v", 6, realized_nfe=6).macro_steps == 6
    with pytest.raises(ValueError, match="external"):
        solver_runtime_name("ipndm_v")
    rows = [{"split": "train", "context_id": "p", "solver": "ipndm", "nfe": 4}]
    conditioning = Conditioning.fit(rows, {"p": np.zeros(2)})
    with pytest.raises(ValueError):
        conditioning.transform(np.zeros(2), "ipndm_v", 4)


def test_adapter_reports_variable_solver_and_freezes_model_with_exact_nfe():
    schedule = Schedule()
    schedule.T, schedule.eps = 1.0, 0.001
    model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    adapter = IPNDMAdapter(
        model_fn=lambda x, *args: model(x),
        decoder=lambda x: x,
        solver=VariableStepIPNDM(schedule),
        solver_key="ipndm_v",
        noise_schedule=schedule,
        context_encoder=lambda _: (None, None, np.zeros(1)),
        latent_factory=lambda seed, _: torch.randn(1, 1, generator=torch.Generator().manual_seed(seed)),
        backbone_revision="frozen",
    )
    context = adapter.encode_context("prompt", "A cube")
    clock = Clock("test", 4, (0.0, 0.1, 0.4, 0.8, 1.0), "diagnostic")
    with patch("torch.cuda.synchronize"):
        image, trace = adapter.sample(noise_seed=17, context=context, clock=clock)
        replay, _ = adapter.sample(noise_seed=17, context=context, clock=clock)
    torch.testing.assert_close(image, replay, atol=0, rtol=0)
    assert trace.solver_key == "ipndm_v" and trace.backbone_forwards == 4 and trace.cfg_sample_equivalents == 8
    assert not image.requires_grad
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert all(p.grad is None for p in model.parameters())


def test_adapter_rejects_mislabeled_solver():
    with pytest.raises(ValueError, match="identity"):
        IPNDMAdapter(
            model_fn=None,
            decoder=None,
            solver=VariableStepIPNDM(Schedule()),
            solver_key="ipndm",
            noise_schedule=Schedule(),
            context_encoder=None,
            latent_factory=None,
            backbone_revision="weights",
        )


@pytest.mark.parametrize("configured,expected", [(None, "ipndm_v"), ("ipndm", "ipndm"), ("ipndm_v", "ipndm_v")])
def test_runtime_solver_dispatch(tmp_path, configured, expected):
    from genode.latent_clock.runtime import load_runtime

    assets = tmp_path / "assets.json"
    assets.write_text("{}")
    config = {"backbone": "sd15", "source": str(tmp_path), "source_revision": "revision", "asset_manifest": str(assets)}
    if configured is not None:
        config["solver"] = configured
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(config))
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("genode.latent_clock.runtime._source_revision", return_value="revision"),
        patch("genode.latent_clock.runtime._load_sd15", return_value=SimpleNamespace(metadata={})) as loader,
    ):
        load_runtime(path)
    assert loader.call_args.kwargs["solver_key"] == expected


def test_official_ld3_rejects_variable_solver_before_fitting_or_inference(tmp_path):
    from genode.latent_clock.collection import collect
    from genode.latent_clock.ld3 import fit_ld3

    runtime = SimpleNamespace(metadata={"backbone": "sd15"}, adapter=SimpleNamespace(solver_key="ipndm_v"))
    config = tmp_path / "runtime.json"
    config.write_text("{}")
    with (
        patch("genode.latent_clock.ld3.load_runtime", return_value=runtime),
        pytest.raises(ValueError, match="explicit"),
    ):
        fit_ld3(runtime_config=str(config), manifest_path="unused", nfe=4, seed=1, output=str(tmp_path / "fit"))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"checkpoint": None, "method": "ld3"}))
    with (
        patch("genode.latent_clock.collection.load_runtime", return_value=runtime),
        pytest.raises(ValueError, match="explicit"),
    ):
        collect(runtime_config=str(config), plan_path=str(plan), output=str(tmp_path / "images"))
