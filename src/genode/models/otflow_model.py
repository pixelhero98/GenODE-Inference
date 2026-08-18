from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from genode.models.config import OTFlowConfig
from genode.models.rectified_flow import RectifiedFlow
from genode.solver_protocol import normalize_solver_key

OTFLOW_TRACE_FIELDS: tuple[str, ...] = (
    "solver",
    "steps",
    "step_index",
    "time",
    "time_grid",
    "disagreement",
    "velocity_norm",
    "ema_velocity_norm",
    "residual_norm",
    "hybrid_signal",
    "u_disagreement",
    "u_residual_norm",
    "u_hybrid_signal",
    "variance_scaled_signal",
    "top_book_disagreement",
    "top_book_residual_norm",
    "top_book_hybrid_signal",
    "oracle_local_error",
    "field_evals_by_step",
    "mean_field_evals_per_step",
    "mean_total_field_evals_per_rollout",
)
_TRACE_EMA_DECAY = 0.9


def _solve_linear_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Solve a square linear assignment problem with the Hungarian algorithm."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError(f"Expected a square cost matrix, got shape={tuple(cost.shape)}")

    matrix = cost.detach().to(device="cpu", dtype=torch.float64).tolist()
    n = len(matrix)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            row = matrix[i0 - 1]
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = torch.empty(n, dtype=torch.long)
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment.to(device=cost.device)


class OTFlow(RectifiedFlow):
    def __init__(self, cfg: OTFlowConfig):
        super().__init__(cfg)

    @torch.no_grad()
    def _match_minibatch_ot(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        cond: torch.Tensor | None,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if x.shape[0] <= 1 or not bool(self.cfg.fm.use_minibatch_ot):
            identity = torch.arange(x.shape[0], device=x.device)
            zero_cost = x.new_tensor(0.0)
            return x, hist, cond, zero_cost, identity

        cost = torch.cdist(z, x, p=2).pow(2)
        perm = _solve_linear_assignment(cost)
        matched_x = x.index_select(0, perm)
        matched_hist = hist.index_select(0, perm)
        matched_cond = None if cond is None else cond.index_select(0, perm)
        matched_cost = cost[torch.arange(cost.shape[0], device=cost.device), perm].mean()
        return matched_x, matched_hist, matched_cond, matched_cost, perm

    def _guided_field(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        hist: torch.Tensor,
        *,
        cond: torch.Tensor | None,
        guidance: float,
    ) -> torch.Tensor:
        if guidance == 1.0 or cond is None:
            return self.v_forward(x, t, hist, cond=cond)
        v_cond = self.v_forward(x, t, hist, cond=cond)
        v_uncond = self.v_forward(x, t, hist, cond=None)
        return v_uncond + guidance * (v_cond - v_uncond)

    def _prediction_horizon(self) -> int:
        return int(max(1, int(getattr(self.cfg, "prediction_horizon", 1))))

    def _sample_state_dim(self) -> int:
        return int(getattr(self.cfg, "sample_state_dim", int(self.cfg.state_dim)))

    def _snapshot_dim(self) -> int:
        return int(getattr(self.cfg, "snapshot_dim", int(self.cfg.state_dim)))

    def _is_non_autoregressive(self) -> bool:
        return self._prediction_horizon() > 1

    def _future_training_target(
        self,
        tgt: torch.Tensor,
        fut: torch.Tensor | None,
    ) -> torch.Tensor:
        horizon = self._prediction_horizon()
        if horizon <= 1:
            return tgt
        if fut is None:
            raise ValueError("Non-autoregressive OTFlow requires dataset batches with future trajectories.")
        required_future = max(0, horizon - 1)
        if int(fut.shape[1]) < required_future:
            raise ValueError(
                f"Non-autoregressive OTFlow requires at least {required_future} future steps, "
                f"but got fut.shape[1]={int(fut.shape[1])}."
            )
        block = torch.cat([tgt[:, None, :], fut[:, :required_future, :]], dim=1)
        return block.reshape(tgt.shape[0], -1)

    def _reshape_sample_block(self, x: torch.Tensor) -> torch.Tensor:
        horizon = self._prediction_horizon()
        if horizon <= 1:
            return x[:, None, :]
        return x.reshape(x.shape[0], horizon, self._snapshot_dim())

    def loss(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        fut: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del meta
        if self._is_non_autoregressive():
            x = self._future_training_target(x, fut)

        batch_size = x.shape[0]
        z = torch.randn_like(x)
        x_target, hist_target, cond_target, ot_cost, _ = self._match_minibatch_ot(
            x=x,
            hist=hist,
            cond=cond,
            z=z,
        )
        t = torch.rand(batch_size, 1, device=x.device, dtype=x.dtype)
        x_t = (1.0 - t) * z + t * x_target
        v_target = x_target - z

        v_hat = self.v_forward(x_t, t, hist_target, cond=cond_target)
        loss = F.mse_loss(v_hat, v_target)
        logs = {
            "mean": float(loss.detach().cpu()),
            "ot_cost": float(ot_cost.detach().cpu()),
            "ot_used": float(bool(self.cfg.fm.use_minibatch_ot and batch_size > 1)),
            "loss": float(loss.detach().cpu()),
        }
        return loss, logs

    def _resolve_solver_name(self, solver: str | None) -> str:
        configured = getattr(self.cfg.sample, "solver", "euler") if solver is None else solver
        return normalize_solver_key(str(configured))

    def _resolved_time_grid(self, n_steps: int) -> tuple[float, ...]:
        raw_grid = tuple(float(x) for x in getattr(self.cfg.sample, "time_grid", ()) or ())
        if len(raw_grid) == 0:
            return tuple(float(i) / float(n_steps) for i in range(int(n_steps) + 1))
        if len(raw_grid) != int(n_steps) + 1:
            raise ValueError(
                f"sample.time_grid must have length n_steps + 1 ({int(n_steps) + 1}), got {len(raw_grid)}."
            )
        if abs(float(raw_grid[0])) > 1e-8 or abs(float(raw_grid[-1]) - 1.0) > 1e-8:
            raise ValueError("sample.time_grid must start at 0.0 and end at 1.0.")
        for left, right in zip(raw_grid, raw_grid[1:], strict=False):
            if float(right) <= float(left):
                raise ValueError("sample.time_grid must be strictly increasing.")
        return raw_grid

    def _top_of_book_feature_weights(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_dim = self._snapshot_dim()
        weights = torch.ones(base_dim, device=device, dtype=dtype)
        levels = int(self.cfg.data.levels)
        if int(weights.numel()) < 2:
            return weights

        weights[0] = 2.0
        weights[1] = 2.0
        ask_gap_start = 2
        bid_gap_start = ask_gap_start + max(0, levels - 1)
        size_start = bid_gap_start + max(0, levels - 1)

        for depth in range(max(0, levels - 1)):
            decay = 1.0 / float(depth + 1)
            idx = ask_gap_start + depth
            if idx < weights.numel():
                weights[idx] = 1.5 * decay
            idx = bid_gap_start + depth
            if idx < weights.numel():
                weights[idx] = 1.5 * decay

        for depth in range(levels):
            decay = 1.0 / float(depth + 1)
            idx = size_start + depth
            if idx < weights.numel():
                weights[idx] = 2.0 * decay
            idx = size_start + levels + depth
            if idx < weights.numel():
                weights[idx] = 2.0 * decay
        if self._prediction_horizon() <= 1:
            return weights
        return weights.repeat(self._prediction_horizon())

    def _oracle_local_error_proxy(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        *,
        hist: torch.Tensor,
        cond: torch.Tensor | None,
        guidance: float,
        dt: float,
        t_cur: float,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        x_euler = x + dt * v
        x_half = x + 0.5 * dt * v
        t_mid = torch.full((batch_size, 1), t_cur + 0.5 * dt, device=x.device)
        v_mid = self._guided_field(x_half, t_mid, hist, cond=cond, guidance=guidance)
        x_two_half = x_half + 0.5 * dt * v_mid
        return torch.sqrt((x_euler - x_two_half).reshape(batch_size, -1).square().sum(dim=-1) + 1e-12)

    def _sample_impl(
        self,
        hist: torch.Tensor,
        *,
        cond: torch.Tensor | None,
        steps: int | None,
        cfg_scale: float | None,
        solver: str | None,
        record_trace: bool,
        oracle_local_error: bool,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        batch_size = hist.shape[0]
        state_dim = self._sample_state_dim()
        x = torch.randn(batch_size, state_dim, device=hist.device)

        default_steps = int(self.cfg.sample.steps)
        n_steps = int(max(1, default_steps if steps is None else steps))

        default_cfg_scale = float(self.cfg.sample.cfg_scale)
        guidance = float(default_cfg_scale if cfg_scale is None else cfg_scale)
        solver_name = self._resolve_solver_name(solver)
        time_grid = self._resolved_time_grid(n_steps)
        prev_dpm_v: torch.Tensor | None = None
        prev_dpm_dt: float | None = None
        ema_v: torch.Tensor | None = None
        ema_v_sq: torch.Tensor | None = None
        ema_u: torch.Tensor | None = None
        top_book_weights = self._top_of_book_feature_weights(device=hist.device, dtype=x.dtype)[None, :]

        if record_trace:
            trace_disagreement = []
            trace_velocity_norm = []
            trace_ema_velocity_norm = []
            trace_residual_norm = []
            trace_hybrid_signal = []
            trace_u_disagreement = []
            trace_u_residual_norm = []
            trace_u_hybrid_signal = []
            trace_variance_scaled_signal = []
            trace_top_book_disagreement = []
            trace_top_book_residual_norm = []
            trace_top_book_hybrid_signal = []
            trace_oracle_error = []
            trace_field_evals = []
            trace_time = []
        else:
            trace_disagreement = trace_velocity_norm = None
            trace_ema_velocity_norm = None
            trace_residual_norm = trace_hybrid_signal = trace_u_disagreement = None
            trace_u_residual_norm = trace_u_hybrid_signal = trace_variance_scaled_signal = None
            trace_top_book_disagreement = trace_top_book_residual_norm = trace_top_book_hybrid_signal = None
            trace_oracle_error = trace_field_evals = None
            trace_time = None

        for i in range(n_steps):
            t_cur = float(time_grid[i])
            t_next = float(time_grid[i + 1])
            dt = float(t_next - t_cur)
            t = torch.full((batch_size, 1), t_cur, device=hist.device, dtype=x.dtype)
            v = self._guided_field(x, t, hist, cond=cond, guidance=guidance)
            v_flat = v.reshape(batch_size, -1)
            vel_norm = torch.sqrt(v_flat.square().sum(dim=-1) + 1e-12)

            if ema_v is None:
                ema_v = v_flat.detach().clone()
            if ema_v_sq is None:
                ema_v_sq = v_flat.detach().square().clone()
            ema_vel_norm = torch.sqrt(ema_v.square().sum(dim=-1) + 1e-12)
            cos = F.cosine_similarity(v_flat, ema_v, dim=-1, eps=1e-8).clamp(-1.0, 1.0)
            disagreement = 1.0 - cos
            residual_flat = v_flat - ema_v
            residual_norm = torch.sqrt(residual_flat.square().sum(dim=-1) + 1e-12)
            hybrid_signal = residual_norm * disagreement
            feature_var = torch.clamp(ema_v_sq - ema_v.square(), min=0.0)
            variance_scale = torch.sqrt(feature_var + 1e-6)
            scaled_v_flat = v_flat / variance_scale
            scaled_ema_flat = ema_v / variance_scale
            scaled_cos = F.cosine_similarity(scaled_v_flat, scaled_ema_flat, dim=-1, eps=1e-8).clamp(-1.0, 1.0)
            variance_scaled_disagreement = 1.0 - scaled_cos
            variance_scaled_residual_flat = residual_flat / variance_scale
            variance_scaled_residual_norm = torch.sqrt(variance_scaled_residual_flat.square().sum(dim=-1) + 1e-12)
            variance_scaled_signal = variance_scaled_residual_norm * variance_scaled_disagreement
            weighted_v_flat = v_flat * top_book_weights
            weighted_ema_flat = ema_v * top_book_weights
            weighted_cos = F.cosine_similarity(weighted_v_flat, weighted_ema_flat, dim=-1, eps=1e-8).clamp(-1.0, 1.0)
            top_book_disagreement = 1.0 - weighted_cos
            top_book_residual_flat = weighted_v_flat - weighted_ema_flat
            top_book_residual_norm = torch.sqrt(top_book_residual_flat.square().sum(dim=-1) + 1e-12)
            top_book_hybrid_signal = top_book_residual_norm * top_book_disagreement
            tail_cur = max(1e-12, 1.0 - t_cur)
            u_flat = (x + tail_cur * v).reshape(batch_size, -1)
            if ema_u is None:
                ema_u = u_flat.detach().clone()
            u_cos = F.cosine_similarity(u_flat, ema_u, dim=-1, eps=1e-8).clamp(-1.0, 1.0)
            u_disagreement = 1.0 - u_cos
            u_residual_flat = u_flat - ema_u
            u_residual_norm = torch.sqrt(u_residual_flat.square().sum(dim=-1) + 1e-12)
            u_hybrid_signal = u_residual_norm * u_disagreement

            oracle_error = torch.zeros(batch_size, device=hist.device, dtype=x.dtype)
            field_evals = torch.ones(batch_size, device=hist.device, dtype=x.dtype)

            if oracle_local_error:
                oracle_error = self._oracle_local_error_proxy(
                    x,
                    v,
                    hist=hist,
                    cond=cond,
                    guidance=guidance,
                    dt=dt,
                    t_cur=t_cur,
                )

            if solver_name == "heun":
                x_pred = x + dt * v
                t_next_tensor = torch.full((batch_size, 1), t_next, device=hist.device)
                v_next = self._guided_field(x_pred, t_next_tensor, hist, cond=cond, guidance=guidance)
                x = x + dt * 0.5 * (v + v_next)
                field_evals = torch.full_like(field_evals, 2.0)
            elif solver_name == "midpoint_rk2":
                x_mid = x + 0.5 * dt * v
                t_mid_tensor = torch.full((batch_size, 1), t_cur + 0.5 * dt, device=hist.device)
                v_mid = self._guided_field(x_mid, t_mid_tensor, hist, cond=cond, guidance=guidance)
                x = x + dt * v_mid
                field_evals = torch.full_like(field_evals, 2.0)
            elif solver_name == "euler":
                x = x + dt * v
            elif solver_name == "dpmpp2m":
                if prev_dpm_v is None or prev_dpm_dt is None:
                    x = x + dt * v
                else:
                    ratio = float(dt) / max(float(prev_dpm_dt), 1e-12)
                    x = x + dt * ((1.0 + 0.5 * ratio) * v - 0.5 * ratio * prev_dpm_v)
                prev_dpm_v = v
                prev_dpm_dt = dt
            else:
                raise ValueError(f"Unhandled sample solver={solver_name}")

            ema_beta = _TRACE_EMA_DECAY
            ema_v = ema_beta * ema_v + (1.0 - ema_beta) * v_flat.detach()
            ema_v_sq = ema_beta * ema_v_sq + (1.0 - ema_beta) * v_flat.detach().square()
            ema_u = ema_beta * ema_u + (1.0 - ema_beta) * u_flat.detach()

            if record_trace:
                trace_disagreement.append(disagreement.detach().cpu())
                trace_velocity_norm.append(vel_norm.detach().cpu())
                trace_ema_velocity_norm.append(ema_vel_norm.detach().cpu())
                trace_residual_norm.append(residual_norm.detach().cpu())
                trace_hybrid_signal.append(hybrid_signal.detach().cpu())
                trace_u_disagreement.append(u_disagreement.detach().cpu())
                trace_u_residual_norm.append(u_residual_norm.detach().cpu())
                trace_u_hybrid_signal.append(u_hybrid_signal.detach().cpu())
                trace_variance_scaled_signal.append(variance_scaled_signal.detach().cpu())
                trace_top_book_disagreement.append(top_book_disagreement.detach().cpu())
                trace_top_book_residual_norm.append(top_book_residual_norm.detach().cpu())
                trace_top_book_hybrid_signal.append(top_book_hybrid_signal.detach().cpu())
                trace_oracle_error.append(oracle_error.detach().cpu())
                trace_field_evals.append(field_evals.detach().cpu())
                trace_time.append(float(t_cur))

        trace: dict[str, Any] | None = None
        if record_trace:
            disagreement_t = torch.stack(trace_disagreement, dim=1)
            velocity_norm_t = torch.stack(trace_velocity_norm, dim=1)
            ema_velocity_norm_t = torch.stack(trace_ema_velocity_norm, dim=1)
            residual_norm_t = torch.stack(trace_residual_norm, dim=1)
            hybrid_signal_t = torch.stack(trace_hybrid_signal, dim=1)
            u_disagreement_t = torch.stack(trace_u_disagreement, dim=1)
            u_residual_norm_t = torch.stack(trace_u_residual_norm, dim=1)
            u_hybrid_signal_t = torch.stack(trace_u_hybrid_signal, dim=1)
            variance_scaled_signal_t = torch.stack(trace_variance_scaled_signal, dim=1)
            top_book_disagreement_t = torch.stack(trace_top_book_disagreement, dim=1)
            top_book_residual_norm_t = torch.stack(trace_top_book_residual_norm, dim=1)
            top_book_hybrid_signal_t = torch.stack(trace_top_book_hybrid_signal, dim=1)
            oracle_error_t = torch.stack(trace_oracle_error, dim=1)
            field_evals_t = torch.stack(trace_field_evals, dim=1)
            trace = {
                "solver": solver_name,
                "steps": int(n_steps),
                "step_index": torch.arange(n_steps, dtype=torch.long),
                "time": torch.tensor(trace_time, dtype=disagreement_t.dtype),
                "time_grid": torch.tensor(time_grid, dtype=disagreement_t.dtype),
                "disagreement": disagreement_t,
                "velocity_norm": velocity_norm_t,
                "ema_velocity_norm": ema_velocity_norm_t,
                "residual_norm": residual_norm_t,
                "hybrid_signal": hybrid_signal_t,
                "u_disagreement": u_disagreement_t,
                "u_residual_norm": u_residual_norm_t,
                "u_hybrid_signal": u_hybrid_signal_t,
                "variance_scaled_signal": variance_scaled_signal_t,
                "top_book_disagreement": top_book_disagreement_t,
                "top_book_residual_norm": top_book_residual_norm_t,
                "top_book_hybrid_signal": top_book_hybrid_signal_t,
                "oracle_local_error": oracle_error_t,
                "field_evals_by_step": field_evals_t,
                "mean_field_evals_per_step": float(field_evals_t.mean().item()),
                "mean_total_field_evals_per_rollout": float(field_evals_t.sum(dim=1).mean().item()),
            }
        return x, trace

    @torch.no_grad()
    def sample_trace(
        self,
        hist: torch.Tensor,
        cond: torch.Tensor | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        solver: str | None = None,
        oracle_local_error: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample a single next state and return per-solver-step trace statistics."""
        if self._is_non_autoregressive():
            raise RuntimeError("Non-autoregressive OTFlow uses sample_future_trace(...), not sample_trace(...).")
        x, trace = self._sample_impl(
            hist,
            cond=cond,
            steps=steps,
            cfg_scale=cfg_scale,
            solver=solver,
            record_trace=True,
            oracle_local_error=oracle_local_error,
        )
        if trace is None:
            raise RuntimeError("Trace recording completed without trace statistics.")
        return x, trace

    @torch.no_grad()
    def sample(
        self,
        hist: torch.Tensor,
        cond: torch.Tensor | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        solver: str | None = None,
    ) -> torch.Tensor:
        """Sampler with optional classifier-free guidance."""
        if self._is_non_autoregressive():
            raise RuntimeError("Non-autoregressive OTFlow uses sample_future(...), not sample(...).")
        x, _ = self._sample_impl(
            hist,
            cond=cond,
            steps=steps,
            cfg_scale=cfg_scale,
            solver=solver,
            record_trace=False,
            oracle_local_error=False,
        )
        return x

    @torch.no_grad()
    def sample_future_trace(
        self,
        hist: torch.Tensor,
        cond: torch.Tensor | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        solver: str | None = None,
        oracle_local_error: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x, trace = self._sample_impl(
            hist,
            cond=cond,
            steps=steps,
            cfg_scale=cfg_scale,
            solver=solver,
            record_trace=True,
            oracle_local_error=oracle_local_error,
        )
        if trace is None:
            raise RuntimeError("Trace recording completed without trace statistics.")
        return self._reshape_sample_block(x), trace

    @torch.no_grad()
    def sample_future(
        self,
        hist: torch.Tensor,
        cond: torch.Tensor | None = None,
        steps: int | None = None,
        cfg_scale: float | None = None,
        solver: str | None = None,
    ) -> torch.Tensor:
        x, _ = self._sample_impl(
            hist,
            cond=cond,
            steps=steps,
            cfg_scale=cfg_scale,
            solver=solver,
            record_trace=False,
            oracle_local_error=False,
        )
        return self._reshape_sample_block(x)


__all__ = ["OTFLOW_TRACE_FIELDS", "OTFlow", "_solve_linear_assignment"]
