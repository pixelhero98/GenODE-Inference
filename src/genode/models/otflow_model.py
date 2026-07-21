from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from genode.models.conditioning import ConditioningCache
from genode.models.rectified_flow import RectifiedFlow
from genode.solver_protocol import (
    FlowDiagnostics,
    FlowTrajectory,
    normalize_solver_nfe_fields,
    solver_eval_multiplier,
)


DIAGNOSTIC_EMA_DECAY = 0.9


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
    @torch.no_grad()
    def _match_minibatch_ot(
        self,
        x: torch.Tensor,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor],
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
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
        hist: Optional[torch.Tensor],
        *,
        conditioning_cache: ConditioningCache,
        unconditional_cache: Optional[ConditioningCache],
        cond: Optional[torch.Tensor],
        guidance: float,
    ) -> torch.Tensor:
        if guidance == 1.0 or conditioning_cache.cond_emb is None:
            return self.v_forward(
                x,
                t,
                hist,
                cond=cond,
                conditioning_cache=conditioning_cache,
            )

        v_cond = self.v_forward(
            x,
            t,
            hist,
            cond=cond,
            conditioning_cache=conditioning_cache,
        )
        if unconditional_cache is None:
            raise RuntimeError("Classifier-free guidance requires an unconditional conditioning cache.")
        v_uncond = self.v_forward(
            x,
            t,
            hist,
            cond=None,
            conditioning_cache=unconditional_cache,
        )
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
        fut: Optional[torch.Tensor],
    ) -> torch.Tensor:
        horizon = self._prediction_horizon()
        if horizon <= 1:
            return tgt
        if fut is None:
            raise ValueError("Non-autoregressive OTFlow requires dataset batches with future trajectories.")
        required_future = horizon - 1
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
        fut: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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

    def _prepare_conditioning_cache(
        self,
        hist: Optional[torch.Tensor],
        cond: Optional[torch.Tensor],
        conditioning_cache: Optional[ConditioningCache],
        *,
        batch_size: int,
    ) -> ConditioningCache:
        cache_was_provided = conditioning_cache is not None
        if cache_was_provided and hist is not None:
            raise ValueError("Provide exactly one of hist or conditioning_cache.")
        if conditioning_cache is None:
            if hist is None:
                raise ValueError("hist is required when conditioning_cache is not provided.")
            conditioning_cache = self.backbone.precompute(hist, cond=cond)

        if int(conditioning_cache.ctx_tokens.shape[0]) != batch_size:
            raise ValueError(
                "conditioning_cache batch size does not match initial_state: "
                f"{int(conditioning_cache.ctx_tokens.shape[0])} != {batch_size}."
            )
        if cond is not None and int(cond.shape[0]) != batch_size:
            raise ValueError(
                f"cond batch size does not match initial_state: {int(cond.shape[0])} != {batch_size}."
            )
        if cond is not None and cache_was_provided:
            cond_emb = self.backbone.conditioner.embed_cond(cond)
            conditioning_cache = ConditioningCache(
                ctx_tokens=conditioning_cache.ctx_tokens,
                ctx_summary=conditioning_cache.ctx_summary,
                summary=conditioning_cache.summary,
                cond_emb=cond_emb,
            )
        return conditioning_cache

    @staticmethod
    def _validate_time_grid(
        time_grid: Sequence[float] | torch.Tensor,
        *,
        initial_state: torch.Tensor,
        macro_steps: int,
    ) -> torch.Tensor:
        if isinstance(time_grid, torch.Tensor):
            grid = time_grid.to(device=initial_state.device, dtype=initial_state.dtype)
        else:
            grid = torch.as_tensor(time_grid, device=initial_state.device, dtype=initial_state.dtype)
        if grid.ndim != 1:
            raise ValueError(f"time_grid must be one-dimensional, got shape={tuple(grid.shape)}.")
        expected_length = int(macro_steps) + 1
        if int(grid.numel()) != expected_length:
            raise ValueError(
                f"time_grid must contain macro_steps + 1 values ({expected_length}), "
                f"got {int(grid.numel())}."
            )
        if not bool(torch.isfinite(grid).all()):
            raise ValueError("time_grid must contain only finite values.")
        tolerance = 1e-6
        if abs(float(grid[0].item())) > tolerance or abs(float(grid[-1].item()) - 1.0) > tolerance:
            raise ValueError("time_grid must start at 0.0 and end at 1.0.")
        if not bool(torch.all(torch.diff(grid) > 0)):
            raise ValueError("time_grid must be strictly increasing.")
        return grid

    @torch.no_grad()
    def solve(
        self,
        initial_state: torch.Tensor,
        hist: Optional[torch.Tensor] = None,
        conditioning_cache: Optional[ConditioningCache] = None,
        cond: Optional[torch.Tensor] = None,
        *,
        solver_key: str,
        target_nfe: int,
        time_grid: Sequence[float] | torch.Tensor,
        return_trajectory: bool = False,
    ) -> torch.Tensor | FlowTrajectory:
        """Integrate an explicit initial state over ``time_grid``.

        The solver key and target NFE are validated by the shared solver
        protocol. Two-evaluation solvers therefore require half as many macro
        intervals as their target NFE.
        """
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("initial_state must be a torch.Tensor.")
        if initial_state.ndim != 2:
            raise ValueError(
                "initial_state must have shape [batch, state_dim], "
                f"got {tuple(initial_state.shape)}."
            )
        expected_state_dim = self._sample_state_dim()
        if int(initial_state.shape[1]) != expected_state_dim:
            raise ValueError(
                f"initial_state has state_dim={int(initial_state.shape[1])}; "
                f"expected {expected_state_dim}."
            )
        if not initial_state.is_floating_point():
            raise TypeError("initial_state must use a floating-point dtype.")

        nfe = normalize_solver_nfe_fields(solver_key, target_nfe, source="OTFlow.solve")
        grid = self._validate_time_grid(
            time_grid,
            initial_state=initial_state,
            macro_steps=nfe.macro_steps,
        )
        batch_size = int(initial_state.shape[0])
        if hist is not None and int(hist.shape[0]) != batch_size:
            raise ValueError(
                f"hist batch size does not match initial_state: {int(hist.shape[0])} != {batch_size}."
            )
        cache = self._prepare_conditioning_cache(
            hist,
            cond,
            conditioning_cache,
            batch_size=batch_size,
        )
        guidance = float(self.cfg.sample.cfg_scale)
        if not torch.isfinite(torch.tensor(guidance)):
            raise ValueError(f"sample.cfg_scale must be finite, got {guidance!r}.")
        unconditional_cache = None
        if guidance != 1.0 and cache.cond_emb is not None:
            unconditional_cache = ConditioningCache(
                ctx_tokens=cache.ctx_tokens,
                ctx_summary=cache.ctx_summary,
                summary=cache.summary,
                cond_emb=None,
            )

        x = initial_state
        states = [x] if return_trajectory else None
        previous_velocity: Optional[torch.Tensor] = None
        previous_dt: Optional[float] = None

        def field(x_state: torch.Tensor, t_value: float) -> torch.Tensor:
            t = torch.full(
                (batch_size, 1),
                float(t_value),
                device=x_state.device,
                dtype=x_state.dtype,
            )
            velocity = self._guided_field(
                x_state,
                t,
                None,
                conditioning_cache=cache,
                unconditional_cache=unconditional_cache,
                cond=None,
                guidance=guidance,
            )
            if velocity.shape != x_state.shape:
                raise ValueError(
                    f"Velocity field returned shape={tuple(velocity.shape)} for "
                    f"state shape={tuple(x_state.shape)}."
                )
            return velocity

        for step_index in range(nfe.macro_steps):
            t_current = float(grid[step_index].item())
            t_next = float(grid[step_index + 1].item())
            dt = t_next - t_current
            velocity = field(x, t_current)

            if nfe.solver_key == "euler":
                x = x + dt * velocity
            elif nfe.solver_key == "heun":
                predicted = x + dt * velocity
                next_velocity = field(predicted, t_next)
                x = x + 0.5 * dt * (velocity + next_velocity)
            elif nfe.solver_key == "midpoint_rk2":
                midpoint = x + 0.5 * dt * velocity
                midpoint_velocity = field(midpoint, t_current + 0.5 * dt)
                x = x + dt * midpoint_velocity
            elif nfe.solver_key == "dpmpp2m":
                if previous_velocity is None or previous_dt is None:
                    x = x + dt * velocity
                else:
                    step_ratio = dt / previous_dt
                    x = x + dt * (
                        (1.0 + 0.5 * step_ratio) * velocity
                        - 0.5 * step_ratio * previous_velocity
                    )
                previous_velocity = velocity
                previous_dt = dt
            else:  # pragma: no cover - normalize_solver_nfe_fields is authoritative.
                raise AssertionError(f"Unhandled solver_key={nfe.solver_key!r}.")

            if states is not None:
                states.append(x)

        if states is None:
            return x
        stacked_states = torch.stack(states, dim=1)
        return FlowTrajectory(
            initial_state=stacked_states[:, 0],
            time_grid=grid,
            states=stacked_states,
            final_state=stacked_states[:, -1],
            solver_key=nfe.solver_key,
            target_nfe=nfe.target_nfe,
            macro_steps=nfe.macro_steps,
            realized_nfe=nfe.realized_nfe,
        )

    def _sample_initial_state(self, hist: torch.Tensor) -> torch.Tensor:
        if hist.ndim < 1:
            raise ValueError(f"hist must include a batch dimension, got shape={tuple(hist.shape)}.")
        return torch.randn(
            int(hist.shape[0]),
            self._sample_state_dim(),
            device=hist.device,
            dtype=hist.dtype,
        )

    @torch.no_grad()
    def solve_with_diagnostics(
        self,
        initial_state: torch.Tensor,
        hist: Optional[torch.Tensor] = None,
        conditioning_cache: Optional[ConditioningCache] = None,
        cond: Optional[torch.Tensor] = None,
        *,
        solver_key: str,
        target_nfe: int,
        time_grid: Sequence[float] | torch.Tensor,
        include_local_error: bool = False,
    ) -> FlowDiagnostics:
        """Solve once and report fixed-schedule field diagnostics.

        Diagnostics are intentionally separate from ``solve`` so ordinary
        sampling does not pay for the additional field evaluations.
        """

        batch_size = int(initial_state.shape[0])
        cache = self._prepare_conditioning_cache(
            hist,
            cond,
            conditioning_cache,
            batch_size=batch_size,
        )
        trajectory = self.solve(
            initial_state,
            conditioning_cache=cache,
            solver_key=solver_key,
            target_nfe=target_nfe,
            time_grid=time_grid,
            return_trajectory=True,
        )
        assert isinstance(trajectory, FlowTrajectory)
        guidance = float(self.cfg.sample.cfg_scale)
        unconditional_cache = None
        if guidance != 1.0 and cache.cond_emb is not None:
            unconditional_cache = ConditioningCache(
                ctx_tokens=cache.ctx_tokens,
                ctx_summary=cache.ctx_summary,
                summary=cache.summary,
                cond_emb=None,
            )
        ema_velocity: Optional[torch.Tensor] = None
        disagreement_rows: list[torch.Tensor] = []
        velocity_norm_rows: list[torch.Tensor] = []
        ema_velocity_norm_rows: list[torch.Tensor] = []
        residual_norm_rows: list[torch.Tensor] = []
        local_error_rows: list[torch.Tensor] = []
        field_eval_rows: list[torch.Tensor] = []

        for step_index in range(trajectory.macro_steps):
            state = trajectory.states[:, step_index]
            t_current = float(trajectory.time_grid[step_index].item())
            t_next = float(trajectory.time_grid[step_index + 1].item())
            dt = t_next - t_current
            t = torch.full(
                (batch_size, 1),
                t_current,
                device=state.device,
                dtype=state.dtype,
            )
            velocity = self._guided_field(
                state,
                t,
                None,
                conditioning_cache=cache,
                unconditional_cache=unconditional_cache,
                cond=None,
                guidance=guidance,
            )
            flat_velocity = velocity.reshape(batch_size, -1)
            if ema_velocity is None:
                ema_velocity = flat_velocity.detach().clone()
            residual = flat_velocity - ema_velocity
            cosine = F.cosine_similarity(
                flat_velocity,
                ema_velocity,
                dim=-1,
                eps=1e-8,
            ).clamp(-1.0, 1.0)
            disagreement_rows.append(1.0 - cosine)
            velocity_norm_rows.append(
                torch.sqrt(flat_velocity.square().sum(dim=-1) + 1e-12)
            )
            ema_velocity_norm_rows.append(
                torch.sqrt(ema_velocity.square().sum(dim=-1) + 1e-12)
            )
            residual_norm_rows.append(
                torch.sqrt(residual.square().sum(dim=-1) + 1e-12)
            )
            if include_local_error:
                half_state = state + 0.5 * dt * velocity
                t_midpoint = torch.full(
                    (batch_size, 1),
                    t_current + 0.5 * dt,
                    device=state.device,
                    dtype=state.dtype,
                )
                midpoint_velocity = self._guided_field(
                    half_state,
                    t_midpoint,
                    None,
                    conditioning_cache=cache,
                    unconditional_cache=unconditional_cache,
                    cond=None,
                    guidance=guidance,
                )
                euler_state = state + dt * velocity
                two_half_state = half_state + 0.5 * dt * midpoint_velocity
                local_error = torch.sqrt(
                    (euler_state - two_half_state)
                    .reshape(batch_size, -1)
                    .square()
                    .sum(dim=-1)
                    + 1e-12
                )
            else:
                local_error = torch.zeros(
                    batch_size,
                    device=state.device,
                    dtype=state.dtype,
                )
            local_error_rows.append(local_error)
            field_eval_rows.append(
                torch.full(
                    (batch_size,),
                    float(solver_eval_multiplier(trajectory.solver_key)),
                    device=state.device,
                    dtype=state.dtype,
                )
            )
            ema_velocity = (
                DIAGNOSTIC_EMA_DECAY * ema_velocity
                + (1.0 - DIAGNOSTIC_EMA_DECAY) * flat_velocity.detach()
            )

        disagreement = torch.stack(disagreement_rows, dim=1)
        field_evals = torch.stack(field_eval_rows, dim=1)
        return FlowDiagnostics(
            trajectory=trajectory,
            disagreement=disagreement,
            velocity_norm=torch.stack(velocity_norm_rows, dim=1),
            ema_velocity_norm=torch.stack(ema_velocity_norm_rows, dim=1),
            residual_norm=torch.stack(residual_norm_rows, dim=1),
            local_error=torch.stack(local_error_rows, dim=1),
            field_evals_by_step=field_evals,
            mean_field_evals_per_step=float(field_evals.mean().item()),
            mean_total_field_evals_per_rollout=float(
                field_evals.sum(dim=1).mean().item()
            ),
        )

    @torch.no_grad()
    def sample_with_diagnostics(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        *,
        solver_key: str,
        target_nfe: int,
        time_grid: Sequence[float] | torch.Tensor,
        include_local_error: bool = False,
    ) -> tuple[torch.Tensor, FlowDiagnostics]:
        if self._is_non_autoregressive():
            raise RuntimeError(
                "Non-autoregressive OTFlow uses sample_future_with_diagnostics(...)."
            )
        diagnostics = self.solve_with_diagnostics(
            self._sample_initial_state(hist),
            hist,
            cond=cond,
            solver_key=solver_key,
            target_nfe=target_nfe,
            time_grid=time_grid,
            include_local_error=include_local_error,
        )
        return diagnostics.trajectory.final_state, diagnostics

    @torch.no_grad()
    def sample_future_with_diagnostics(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        *,
        solver_key: str,
        target_nfe: int,
        time_grid: Sequence[float] | torch.Tensor,
        include_local_error: bool = False,
    ) -> tuple[torch.Tensor, FlowDiagnostics]:
        diagnostics = self.solve_with_diagnostics(
            self._sample_initial_state(hist),
            hist,
            cond=cond,
            solver_key=solver_key,
            target_nfe=target_nfe,
            time_grid=time_grid,
            include_local_error=include_local_error,
        )
        return self._reshape_sample_block(diagnostics.trajectory.final_state), diagnostics

    @torch.no_grad()
    def sample(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        *,
        solver_key: str,
        target_nfe: int,
        time_grid: Sequence[float] | torch.Tensor,
    ) -> torch.Tensor:
        """Draw an initial state and return one autoregressive sample."""
        if self._is_non_autoregressive():
            raise RuntimeError("Non-autoregressive OTFlow uses sample_future(...), not sample(...).")
        result = self.solve(
            self._sample_initial_state(hist),
            hist,
            cond=cond,
            solver_key=solver_key,
            target_nfe=target_nfe,
            time_grid=time_grid,
        )
        assert isinstance(result, torch.Tensor)
        return result

    @torch.no_grad()
    def sample_future(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        *,
        solver_key: str,
        target_nfe: int,
        time_grid: Sequence[float] | torch.Tensor,
    ) -> torch.Tensor:
        """Draw an initial state and return a future block."""
        result = self.solve(
            self._sample_initial_state(hist),
            hist,
            cond=cond,
            solver_key=solver_key,
            target_nfe=target_nfe,
            time_grid=time_grid,
        )
        assert isinstance(result, torch.Tensor)
        return self._reshape_sample_block(result)


__all__ = ["OTFlow"]
