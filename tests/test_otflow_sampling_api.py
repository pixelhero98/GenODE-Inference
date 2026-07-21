from __future__ import annotations

import torch

from genode.models.config import OTFlowConfig
from genode.models.otflow_model import OTFlow
from genode.solver_protocol import FlowDiagnostics


def _config(*, rollout_mode: str = "autoregressive") -> OTFlowConfig:
    return OTFlowConfig(
        device=torch.device("cpu"),
        levels=2,
        history_len=4,
        hidden_dim=16,
        dropout=0.0,
        ctx_heads=4,
        ctx_layers=1,
        fu_net_layers=1,
        fu_net_heads=4,
        rollout_mode=rollout_mode,
        future_block_len=2 if rollout_mode == "non_ar" else 1,
        use_minibatch_ot=False,
        use_amp=False,
    )


def _install_constant_field(model: OTFlow) -> None:
    def constant_field(x, t, hist, cond=None, conditioning_cache=None):
        del t, hist, cond, conditioning_cache
        return torch.ones_like(x)

    model.v_forward = constant_field  # type: ignore[method-assign]


def test_solve_with_diagnostics_returns_typed_batch_major_traces() -> None:
    cfg = _config()
    model = OTFlow(cfg).eval()
    _install_constant_field(model)
    hist = torch.zeros(2, cfg.history_len, cfg.context_dim)
    initial_state = torch.zeros(2, cfg.sample_state_dim)

    diagnostics = model.solve_with_diagnostics(
        initial_state,
        hist,
        solver_key="heun",
        target_nfe=4,
        time_grid=(0.0, 0.25, 1.0),
        include_local_error=True,
    )

    assert isinstance(diagnostics, FlowDiagnostics)
    assert diagnostics.trajectory.solver_key == "heun"
    assert diagnostics.trajectory.target_nfe == 4
    assert diagnostics.trajectory.macro_steps == 2
    assert diagnostics.trajectory.realized_nfe == 4
    assert diagnostics.trajectory.states.shape == (2, 3, cfg.sample_state_dim)
    assert diagnostics.disagreement.shape == (2, 2)
    assert diagnostics.velocity_norm.shape == (2, 2)
    assert diagnostics.ema_velocity_norm.shape == (2, 2)
    assert diagnostics.residual_norm.shape == (2, 2)
    assert diagnostics.local_error.shape == (2, 2)
    assert diagnostics.field_evals_by_step.shape == (2, 2)
    assert torch.equal(diagnostics.field_evals_by_step, torch.full((2, 2), 2.0))
    assert diagnostics.mean_field_evals_per_step == 2.0
    assert diagnostics.mean_total_field_evals_per_rollout == 4.0
    assert torch.allclose(diagnostics.trajectory.final_state, torch.ones_like(initial_state))


def test_sampling_diagnostics_wrappers_match_active_output_shapes() -> None:
    autoregressive_cfg = _config()
    autoregressive = OTFlow(autoregressive_cfg).eval()
    _install_constant_field(autoregressive)
    autoregressive_hist = torch.zeros(
        2,
        autoregressive_cfg.history_len,
        autoregressive_cfg.context_dim,
    )

    torch.manual_seed(17)
    sample, sample_diagnostics = autoregressive.sample_with_diagnostics(
        autoregressive_hist,
        solver_key="euler",
        target_nfe=2,
        time_grid=(0.0, 0.5, 1.0),
    )
    torch.manual_seed(17)
    expected_sample = autoregressive.sample(
        autoregressive_hist,
        solver_key="euler",
        target_nfe=2,
        time_grid=(0.0, 0.5, 1.0),
    )

    assert sample.shape == (2, autoregressive_cfg.snapshot_dim)
    assert torch.equal(sample, expected_sample)
    assert torch.equal(sample, sample_diagnostics.trajectory.final_state)

    block_cfg = _config(rollout_mode="non_ar")
    block_model = OTFlow(block_cfg).eval()
    _install_constant_field(block_model)
    block_hist = torch.zeros(2, block_cfg.history_len, block_cfg.context_dim)

    torch.manual_seed(23)
    future, future_diagnostics = block_model.sample_future_with_diagnostics(
        block_hist,
        solver_key="midpoint_rk2",
        target_nfe=4,
        time_grid=(0.0, 0.5, 1.0),
    )
    torch.manual_seed(23)
    expected_future = block_model.sample_future(
        block_hist,
        solver_key="midpoint_rk2",
        target_nfe=4,
        time_grid=(0.0, 0.5, 1.0),
    )

    assert future.shape == (2, 2, block_cfg.snapshot_dim)
    assert torch.equal(future, expected_future)
    assert torch.equal(
        future.reshape(2, -1),
        future_diagnostics.trajectory.final_state,
    )
