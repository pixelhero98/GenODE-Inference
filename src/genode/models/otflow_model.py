from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from genode.models.config import OTFlowConfig
from genode.models.rectified_flow import RectifiedFlow


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
        self._last_sample_stats: Dict[str, Any] = {}

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
        hist: torch.Tensor,
        *,
        cond: Optional[torch.Tensor],
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
        fut: Optional[torch.Tensor],
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

    def _resolve_solver_name(self, solver: Optional[str]) -> str:
        solver_name = str(getattr(self.cfg.sample, "solver", "euler") if solver is None else solver).lower().strip()
        if solver_name in {"dpmpp_2m", "dpm++", "dpm++2m"}:
            solver_name = "dpmpp2m"
        if solver_name in {"rk2", "heun_rk2", "heun/rk2"}:
            solver_name = "heun"
        if solver_name in {"midpoint-rk2", "rk2_midpoint", "midpoint rk2"}:
            solver_name = "midpoint_rk2"
        if solver_name in {"dormand_prince", "dormand-prince", "dormand_prince_45", "dopri5"}:
            solver_name = "dopri5"
        if solver_name in {"dopri5_adapt", "dopri5-adaptive", "dormand_prince_adaptive", "dormand-prince-adaptive"}:
            solver_name = "dopri5_adaptive"
        if solver_name in {"runge_kutta45", "runge-kutta45", "rk_45"}:
            solver_name = "rk45"
        if solver_name in {"rk45_adapt", "rk45-adaptive", "runge_kutta45_adaptive", "runge-kutta45-adaptive"}:
            solver_name = "rk45_adaptive"
        if solver_name not in {
            "euler",
            "dpmpp2m",
            "dopri5",
            "dopri5_adaptive",
            "rk45",
            "rk45_adaptive",
            "euler_adaptive",
            "heun",
            "midpoint_rk2",
            "euler_refine_half",
            "euler_refine_heun",
        }:
            raise ValueError(f"Unknown sample solver={solver_name}")
        return solver_name

    def _adaptive_sampler_settings(self) -> Dict[str, Any]:
        noise_mode = str(getattr(self.cfg.sample, "adaptive_noise_mode", "orthogonal")).lower().strip()
        if noise_mode not in {"orthogonal", "isotropic", "none"}:
            raise ValueError(f"Unknown adaptive_noise_mode={noise_mode}")
        trigger_mode = str(getattr(self.cfg.sample, "adaptive_trigger_mode", "adaptive")).lower().strip()
        if trigger_mode not in {"adaptive", "always", "none"}:
            raise ValueError(f"Unknown adaptive_trigger_mode={trigger_mode}")
        return {
            "beta": float(getattr(self.cfg.sample, "adaptive_beta", 0.9)),
            "tau": float(getattr(self.cfg.sample, "adaptive_tau", 0.15)),
            "kappa": float(getattr(self.cfg.sample, "adaptive_kappa", 12.0)),
            "gamma_max": float(getattr(self.cfg.sample, "adaptive_gamma_max", 0.05)),
            "cooldown_steps": int(getattr(self.cfg.sample, "adaptive_cooldown_steps", 0)),
            "noise_mode": noise_mode,
            "trigger_mode": trigger_mode,
            "disable_noise_frac": float(getattr(self.cfg.sample, "adaptive_disable_noise_frac", 0.1)),
        }

    def _adaptive_rk_settings(self) -> Dict[str, Any]:
        rtol = float(getattr(self.cfg.sample, "adaptive_rtol", 1e-3))
        atol = float(getattr(self.cfg.sample, "adaptive_atol", 1e-6))
        safety = float(getattr(self.cfg.sample, "adaptive_safety", 0.9))
        min_step = float(getattr(self.cfg.sample, "adaptive_min_step", 1e-5))
        max_nfe = int(getattr(self.cfg.sample, "adaptive_max_nfe", 512))
        if rtol <= 0.0 or not math.isfinite(rtol):
            raise ValueError(f"adaptive_rtol must be positive and finite, got {rtol}")
        if atol <= 0.0 or not math.isfinite(atol):
            raise ValueError(f"adaptive_atol must be positive and finite, got {atol}")
        if safety <= 0.0 or not math.isfinite(safety):
            raise ValueError(f"adaptive_safety must be positive and finite, got {safety}")
        if min_step <= 0.0 or not math.isfinite(min_step):
            raise ValueError(f"adaptive_min_step must be positive and finite, got {min_step}")
        if max_nfe <= 0:
            raise ValueError(f"adaptive_max_nfe must be positive, got {max_nfe}")
        return {
            "rtol": rtol,
            "atol": atol,
            "safety": safety,
            "min_step": min_step,
            "max_nfe": max_nfe,
            "min_factor": 0.2,
            "max_factor": 5.0,
        }

    def _refine_sampler_settings(self) -> Dict[str, Any]:
        step_mu = tuple(float(x) for x in getattr(self.cfg.sample, "refine_step_mu", ()) or ())
        step_sigma = tuple(float(x) for x in getattr(self.cfg.sample, "refine_step_sigma", ()) or ())
        step_threshold = tuple(float(x) for x in getattr(self.cfg.sample, "refine_step_threshold", ()) or ())
        selected_steps = tuple(int(x) for x in getattr(self.cfg.sample, "refine_selected_steps", ()) or ())
        return {
            "beta": float(getattr(self.cfg.sample, "refine_beta", 0.9)),
            "trigger_mode": str(getattr(self.cfg.sample, "refine_trigger_mode", "zscore")),
            "threshold_z": float(getattr(self.cfg.sample, "refine_threshold_z", 1.5)),
            "threshold_raw": float(getattr(self.cfg.sample, "refine_threshold_raw", 0.0)),
            "step_mu": step_mu,
            "step_sigma": step_sigma,
            "step_threshold": step_threshold,
            "selected_steps": selected_steps,
            "fixed_last_k": int(getattr(self.cfg.sample, "refine_fixed_last_k", 0)),
            "sigma_eps": float(getattr(self.cfg.sample, "refine_sigma_eps", 1e-6)),
            "disallow_final_step": bool(getattr(self.cfg.sample, "refine_disallow_final_step", True)),
        }

    @staticmethod
    def _step_stat_value(values: Tuple[float, ...], step_idx: int, default: float) -> float:
        if len(values) == 0:
            return float(default)
        clamped_idx = min(max(int(step_idx), 0), len(values) - 1)
        return float(values[clamped_idx])

    def _resolved_time_grid(self, n_steps: int) -> Tuple[float, ...]:
        raw_grid = tuple(float(x) for x in getattr(self.cfg.sample, "time_grid", ()) or ())
        if len(raw_grid) == 0:
            return tuple(float(i) / float(n_steps) for i in range(int(n_steps) + 1))
        if len(raw_grid) != int(n_steps) + 1:
            raise ValueError(
                f"sample.time_grid must have length n_steps + 1 ({int(n_steps) + 1}), got {len(raw_grid)}."
            )
        if abs(float(raw_grid[0])) > 1e-8 or abs(float(raw_grid[-1]) - 1.0) > 1e-8:
            raise ValueError("sample.time_grid must start at 0.0 and end at 1.0.")
        for left, right in zip(raw_grid, raw_grid[1:]):
            if float(right) <= float(left):
                raise ValueError("sample.time_grid must be strictly increasing.")
        return raw_grid

    def _normalized_disagreement(
        self,
        disagreement: torch.Tensor,
        *,
        step_idx: int,
        refine_cfg: Mapping[str, Any],
    ) -> torch.Tensor:
        mu = self._step_stat_value(tuple(refine_cfg["step_mu"]), int(step_idx), 0.0)
        sigma = self._step_stat_value(tuple(refine_cfg["step_sigma"]), int(step_idx), 1.0)
        sigma = max(float(sigma), float(refine_cfg["sigma_eps"]))
        return (disagreement - disagreement.new_tensor(mu)) / disagreement.new_tensor(sigma)

    def _refine_trigger_state(
        self,
        disagreement: torch.Tensor,
        normalized_disagreement: torch.Tensor,
        *,
        step_idx: int,
        n_steps: int,
        refine_cfg: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        disallow_final = bool(refine_cfg["disallow_final_step"]) and int(step_idx) == (int(n_steps) - 1)
        trigger_mode = str(refine_cfg.get("trigger_mode", "zscore"))
        dtype = disagreement.dtype
        if trigger_mode == "selected_steps":
            if disallow_final:
                is_selected = False
            else:
                is_selected = int(step_idx) in {int(x) for x in tuple(refine_cfg.get("selected_steps", ()))}
            trigger_eligible = torch.full_like(disagreement, bool(is_selected), dtype=torch.bool)
            fired = trigger_eligible.clone()
            trigger_strength = fired.to(dtype=dtype)
            return trigger_eligible, fired, trigger_strength
        if trigger_mode == "fixed_last_k":
            fixed_last_k = max(int(refine_cfg.get("fixed_last_k", 0)), 0)
            if disallow_final or fixed_last_k <= 0:
                trigger_eligible = torch.zeros_like(disagreement, dtype=torch.bool)
            else:
                first_actionable = max(0, int(n_steps) - 1 - fixed_last_k)
                is_selected = first_actionable <= int(step_idx) < (int(n_steps) - 1)
                trigger_eligible = torch.full_like(disagreement, bool(is_selected), dtype=torch.bool)
            fired = trigger_eligible.clone()
            trigger_strength = fired.to(dtype=dtype)
            return trigger_eligible, fired, trigger_strength

        trigger_eligible = torch.full_like(disagreement, not disallow_final, dtype=torch.bool)
        if trigger_mode == "raw_step":
            threshold = self._step_stat_value(tuple(refine_cfg.get("step_threshold", ())), int(step_idx), float(refine_cfg.get("threshold_raw", 0.0)))
            fired = trigger_eligible & (disagreement > float(threshold))
        else:
            fired = trigger_eligible & (normalized_disagreement > float(refine_cfg["threshold_z"]))
        trigger_strength = torch.where(fired, torch.ones_like(disagreement, dtype=dtype), torch.zeros_like(disagreement, dtype=dtype))
        return trigger_eligible, fired, trigger_strength

    @staticmethod
    def _adaptive_time_gate(t_cur: float, disable_noise_frac: float) -> float:
        t_val = float(t_cur)
        if not (disable_noise_frac < t_val < 1.0 - disable_noise_frac):
            return 0.0
        return float(4.0 * t_val * (1.0 - t_val))

    @staticmethod
    def _project_noise_orthogonal(
        noise: torch.Tensor,
        velocity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = velocity.shape[0]
        noise_flat = noise.reshape(batch_size, -1)
        vel_flat = velocity.reshape(batch_size, -1)
        denom = vel_flat.square().sum(dim=-1, keepdim=True)
        valid = denom.squeeze(-1) >= 1e-8
        safe_denom = torch.clamp(denom, min=1e-8)
        proj = (noise_flat * vel_flat).sum(dim=-1, keepdim=True) / safe_denom
        orth_flat = noise_flat - proj * vel_flat
        orth_flat = torch.where(valid[:, None], orth_flat, torch.zeros_like(orth_flat))
        return orth_flat.view_as(noise), valid

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
        cond: Optional[torch.Tensor],
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

    @staticmethod
    def _adaptive_error_norm(
        x_old: torch.Tensor,
        x_new: torch.Tensor,
        x_low_order: torch.Tensor,
        *,
        rtol: float,
        atol: float,
    ) -> float:
        scale = float(atol) + float(rtol) * torch.maximum(torch.abs(x_old), torch.abs(x_new))
        err = (x_new - x_low_order) / torch.clamp(scale, min=1e-12)
        per_sample = torch.sqrt(err.reshape(err.shape[0], -1).square().mean(dim=-1) + 1e-12)
        return float(torch.max(per_sample).detach().cpu().item())

    @staticmethod
    def _adaptive_step_factor(error_norm: float, *, settings: Mapping[str, Any]) -> float:
        if not math.isfinite(float(error_norm)) or float(error_norm) <= 0.0:
            raw = float(settings["max_factor"])
        else:
            raw = float(settings["safety"]) * float(error_norm) ** (-0.2)
        return float(min(max(raw, float(settings["min_factor"])), float(settings["max_factor"])))

    @staticmethod
    def _rk45_embedded_step(
        x: torch.Tensor,
        t_cur: float,
        dt: float,
        field_fn: Any,
        first_eval: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        k1 = field_fn(x, t_cur) if first_eval is None else first_eval
        k2 = field_fn(x + dt * ((1.0 / 4.0) * k1), t_cur + dt * (1.0 / 4.0))
        k3 = field_fn(
            x + dt * ((3.0 / 32.0) * k1 + (9.0 / 32.0) * k2),
            t_cur + dt * (3.0 / 8.0),
        )
        k4 = field_fn(
            x
            + dt
            * (
                (1932.0 / 2197.0) * k1
                + (-7200.0 / 2197.0) * k2
                + (7296.0 / 2197.0) * k3
            ),
            t_cur + dt * (12.0 / 13.0),
        )
        k5 = field_fn(
            x
            + dt
            * (
                (439.0 / 216.0) * k1
                + (-8.0) * k2
                + (3680.0 / 513.0) * k3
                + (-845.0 / 4104.0) * k4
            ),
            t_cur + dt,
        )
        k6 = field_fn(
            x
            + dt
            * (
                (-8.0 / 27.0) * k1
                + 2.0 * k2
                + (-3544.0 / 2565.0) * k3
                + (1859.0 / 4104.0) * k4
                + (-11.0 / 40.0) * k5
            ),
            t_cur + dt * 0.5,
        )
        high = x + dt * (
            (16.0 / 135.0) * k1
            + (6656.0 / 12825.0) * k3
            + (28561.0 / 56430.0) * k4
            + (-9.0 / 50.0) * k5
            + (2.0 / 55.0) * k6
        )
        low = x + dt * (
            (25.0 / 216.0) * k1
            + (1408.0 / 2565.0) * k3
            + (2197.0 / 4104.0) * k4
            + (-1.0 / 5.0) * k5
        )
        return high, low, 6 if first_eval is None else 5

    @staticmethod
    def _dopri5_embedded_step(
        x: torch.Tensor,
        t_cur: float,
        dt: float,
        field_fn: Any,
        first_eval: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        k1 = field_fn(x, t_cur) if first_eval is None else first_eval
        k2 = field_fn(x + dt * ((1.0 / 5.0) * k1), t_cur + dt * (1.0 / 5.0))
        k3 = field_fn(
            x + dt * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2),
            t_cur + dt * (3.0 / 10.0),
        )
        k4 = field_fn(
            x + dt * ((44.0 / 45.0) * k1 + (-56.0 / 15.0) * k2 + (32.0 / 9.0) * k3),
            t_cur + dt * (4.0 / 5.0),
        )
        k5 = field_fn(
            x
            + dt
            * (
                (19372.0 / 6561.0) * k1
                + (-25360.0 / 2187.0) * k2
                + (64448.0 / 6561.0) * k3
                + (-212.0 / 729.0) * k4
            ),
            t_cur + dt * (8.0 / 9.0),
        )
        k6 = field_fn(
            x
            + dt
            * (
                (9017.0 / 3168.0) * k1
                + (-355.0 / 33.0) * k2
                + (46732.0 / 5247.0) * k3
                + (49.0 / 176.0) * k4
                + (-5103.0 / 18656.0) * k5
            ),
            t_cur + dt,
        )
        high = x + dt * (
            (35.0 / 384.0) * k1
            + (500.0 / 1113.0) * k3
            + (125.0 / 192.0) * k4
            + (-2187.0 / 6784.0) * k5
            + (11.0 / 84.0) * k6
        )
        k7 = field_fn(high, t_cur + dt)
        low = x + dt * (
            (5179.0 / 57600.0) * k1
            + (7571.0 / 16695.0) * k3
            + (393.0 / 640.0) * k4
            + (-92097.0 / 339200.0) * k5
            + (187.0 / 2100.0) * k6
            + (1.0 / 40.0) * k7
        )
        return high, low, 7 if first_eval is None else 6

    def _sample_impl(
        self,
        hist: torch.Tensor,
        *,
        cond: Optional[torch.Tensor],
        steps: Optional[int],
        cfg_scale: Optional[float],
        solver: Optional[str],
        record_trace: bool,
        oracle_local_error: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        batch_size = hist.shape[0]
        state_dim = self._sample_state_dim()
        x = torch.randn(batch_size, state_dim, device=hist.device)

        configured_steps = int(self.cfg.sample.steps)
        n_steps = int(max(1, configured_steps if steps is None else steps))

        configured_cfg_scale = float(self.cfg.sample.cfg_scale)
        guidance = float(configured_cfg_scale if cfg_scale is None else cfg_scale)
        solver_name = self._resolve_solver_name(solver)
        adaptive_cfg = self._adaptive_sampler_settings()
        refine_cfg = self._refine_sampler_settings()
        time_grid = self._resolved_time_grid(n_steps)
        prev_dpm_v: Optional[torch.Tensor] = None
        prev_dpm_dt: Optional[float] = None
        ema_v: Optional[torch.Tensor] = None
        ema_v_sq: Optional[torch.Tensor] = None
        ema_u: Optional[torch.Tensor] = None
        cooldown = torch.zeros(batch_size, device=hist.device, dtype=torch.long)
        top_book_weights = self._top_of_book_feature_weights(device=hist.device, dtype=x.dtype)[None, :]

        if record_trace:
            trace_disagreement = []
            trace_normalized_disagreement = []
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
            trace_gamma = []
            trace_trigger = []
            trace_fired = []
            trace_noise_norm = []
            trace_noise_inner = []
            trace_oracle_error = []
            trace_field_evals = []
            trace_time = []
            trace_gate = []
            trace_trigger_eligible = []
        else:
            trace_disagreement = trace_normalized_disagreement = trace_velocity_norm = None
            trace_ema_velocity_norm = None
            trace_residual_norm = trace_hybrid_signal = trace_u_disagreement = None
            trace_u_residual_norm = trace_u_hybrid_signal = trace_variance_scaled_signal = None
            trace_top_book_disagreement = trace_top_book_residual_norm = trace_top_book_hybrid_signal = None
            trace_gamma = None
            trace_trigger = trace_fired = trace_noise_norm = None
            trace_noise_inner = trace_oracle_error = trace_field_evals = None
            trace_time = trace_gate = None
            trace_trigger_eligible = None

        def _guided_field_at(x_state: torch.Tensor, t_scalar: float) -> torch.Tensor:
            t_tensor = torch.full((batch_size, 1), float(t_scalar), device=hist.device, dtype=x.dtype)
            return self._guided_field(x_state, t_tensor, hist, cond=cond, guidance=guidance)

        if solver_name in {"rk45_adaptive", "dopri5_adaptive"}:
            rk_cfg = self._adaptive_rk_settings()
            total_evals = 0
            accepted_steps = 0
            rejected_steps = 0
            hit_max_nfe = False
            t_cur = 0.0
            dt = 1.0 / float(max(int(n_steps), 1))
            min_step = float(rk_cfg["min_step"])
            max_nfe = int(rk_cfg["max_nfe"])
            trial_times: List[float] = []
            trial_dts: List[float] = []
            trial_errors: List[float] = []
            trial_accepted: List[bool] = []
            trial_evals: List[int] = []
            accepted_time_grid: List[float] = [0.0]
            accepted_start_times: List[float] = []
            accepted_dts: List[float] = []
            accepted_errors: List[float] = []
            accepted_evals: List[int] = []
            step_fn = self._rk45_embedded_step if solver_name == "rk45_adaptive" else self._dopri5_embedded_step
            first_eval: Optional[torch.Tensor] = None

            while t_cur < 1.0 - 1e-12:
                dt = min(float(dt), 1.0 - float(t_cur))
                if dt <= min_step:
                    dt = min(min_step, 1.0 - float(t_cur))
                x_high, x_low, evals = step_fn(x, float(t_cur), float(dt), _guided_field_at, first_eval=first_eval)
                total_evals += int(evals)
                error_norm = self._adaptive_error_norm(
                    x,
                    x_high,
                    x_low,
                    rtol=float(rk_cfg["rtol"]),
                    atol=float(rk_cfg["atol"]),
                )
                force_accept = bool(dt <= min_step * (1.0 + 1e-9) or total_evals >= max_nfe)
                accepted = bool(error_norm <= 1.0 or force_accept)
                trial_times.append(float(t_cur))
                trial_dts.append(float(dt))
                trial_errors.append(float(error_norm))
                trial_accepted.append(bool(accepted))
                trial_evals.append(int(evals))
                factor = self._adaptive_step_factor(float(error_norm), settings=rk_cfg)
                if accepted:
                    accepted_start_times.append(float(t_cur))
                    accepted_dts.append(float(dt))
                    accepted_errors.append(float(error_norm))
                    accepted_evals.append(int(evals))
                    x = x_high
                    t_cur = min(1.0, float(t_cur) + float(dt))
                    accepted_time_grid.append(float(t_cur))
                    accepted_steps += 1
                    first_eval = None
                    dt = min(1.0 - float(t_cur), float(dt) * factor) if t_cur < 1.0 else float(dt)
                else:
                    rejected_steps += 1
                    first_eval = None
                    dt = max(min_step, float(dt) * factor)
                if total_evals >= max_nfe and t_cur < 1.0 - 1e-12:
                    hit_max_nfe = True
                    remaining_dt = 1.0 - float(t_cur)
                    x_high, _, evals = step_fn(x, float(t_cur), float(remaining_dt), _guided_field_at, first_eval=None)
                    x = x_high
                    total_evals += int(evals)
                    accepted_steps += 1
                    trial_times.append(float(t_cur))
                    trial_dts.append(float(remaining_dt))
                    trial_errors.append(float("nan"))
                    trial_accepted.append(True)
                    trial_evals.append(int(evals))
                    accepted_start_times.append(float(t_cur))
                    accepted_dts.append(float(remaining_dt))
                    accepted_errors.append(float("nan"))
                    accepted_evals.append(int(evals))
                    t_cur = 1.0
                    accepted_time_grid.append(float(t_cur))
                    break

            self._last_sample_stats = {
                "solver": solver_name,
                "steps": int(n_steps),
                "adaptive_rtol": float(rk_cfg["rtol"]),
                "adaptive_atol": float(rk_cfg["atol"]),
                "adaptive_safety": float(rk_cfg["safety"]),
                "adaptive_min_step": float(rk_cfg["min_step"]),
                "adaptive_max_nfe": int(rk_cfg["max_nfe"]),
                "accepted_steps": int(accepted_steps),
                "rejected_steps": int(rejected_steps),
                "trial_steps": int(len(trial_evals)),
                "total_field_evals": int(total_evals),
                "mean_total_field_evals_per_rollout": float(total_evals),
                "hit_max_nfe": bool(hit_max_nfe),
            }
            if not record_trace:
                return x, None
            evals_t = torch.tensor(accepted_evals, dtype=x.dtype).repeat(batch_size, 1)
            trial_evals_t = torch.tensor(trial_evals, dtype=x.dtype).repeat(batch_size, 1)
            trace = {
                "solver": solver_name,
                "steps": int(len(accepted_evals)),
                "step_index": torch.arange(len(accepted_evals), dtype=torch.long),
                "time": torch.tensor(accepted_start_times, dtype=x.dtype),
                "time_grid": torch.tensor(accepted_time_grid, dtype=x.dtype),
                "dt": torch.tensor(accepted_dts, dtype=x.dtype),
                "adaptive_error_norm": torch.tensor(accepted_errors, dtype=x.dtype),
                "adaptive_accepted": torch.ones(len(accepted_evals), dtype=torch.bool),
                "field_evals_by_step": evals_t,
                "mean_field_evals_per_step": float(evals_t.mean().item()) if evals_t.numel() else 0.0,
                "mean_total_field_evals_per_rollout": float(total_evals),
                "trial_step_index": torch.arange(len(trial_evals), dtype=torch.long),
                "trial_time": torch.tensor(trial_times, dtype=x.dtype),
                "trial_dt": torch.tensor(trial_dts, dtype=x.dtype),
                "trial_adaptive_error_norm": torch.tensor(trial_errors, dtype=x.dtype),
                "trial_accepted": torch.tensor(trial_accepted, dtype=torch.bool),
                "trial_field_evals_by_step": trial_evals_t,
                "accepted_steps": int(accepted_steps),
                "rejected_steps": int(rejected_steps),
                "hit_max_nfe": bool(hit_max_nfe),
                "rtol": float(rk_cfg["rtol"]),
                "atol": float(rk_cfg["atol"]),
            }
            return x, trace

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
            normalized_disagreement = self._normalized_disagreement(disagreement, step_idx=i, refine_cfg=refine_cfg)

            gamma = torch.zeros(batch_size, device=hist.device, dtype=x.dtype)
            trigger_strength = torch.zeros(batch_size, device=hist.device, dtype=x.dtype)
            fired = torch.zeros(batch_size, device=hist.device, dtype=torch.bool)
            noise_norm = torch.zeros(batch_size, device=hist.device, dtype=x.dtype)
            noise_inner = torch.zeros(batch_size, device=hist.device, dtype=x.dtype)
            oracle_error = torch.zeros(batch_size, device=hist.device, dtype=x.dtype)
            field_evals = torch.ones(batch_size, device=hist.device, dtype=x.dtype)
            trigger_eligible = torch.zeros(batch_size, device=hist.device, dtype=torch.bool)
            gate_scalar = 0.0

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

            if solver_name == "euler_adaptive":
                if adaptive_cfg["trigger_mode"] == "none":
                    x = x + dt * v
                else:
                    gate_scalar = self._adaptive_time_gate(t_cur, adaptive_cfg["disable_noise_frac"])
                    gate = torch.full((batch_size,), gate_scalar, device=hist.device, dtype=x.dtype)
                    trigger_eligible = gate > 0.0
                    if adaptive_cfg["trigger_mode"] == "always":
                        trigger_strength = torch.ones(batch_size, device=hist.device, dtype=x.dtype)
                        fired = trigger_eligible
                    else:
                        trigger_strength = torch.sigmoid(adaptive_cfg["kappa"] * (disagreement - adaptive_cfg["tau"]))
                        fired = trigger_eligible & (disagreement > adaptive_cfg["tau"])
                        if adaptive_cfg["cooldown_steps"] > 0:
                            trigger_strength = torch.maximum(trigger_strength, (cooldown > 0).to(dtype=x.dtype))
                            fired = fired | (trigger_eligible & (cooldown > 0))
                    gamma = adaptive_cfg["gamma_max"] * trigger_strength * gate

                    if float(adaptive_cfg["gamma_max"]) > 0.0 and torch.any(gamma > 0):
                        noise = torch.randn_like(x)
                        if adaptive_cfg["noise_mode"] == "orthogonal":
                            noise, valid = self._project_noise_orthogonal(noise, v)
                            gamma = torch.where(valid, gamma, torch.zeros_like(gamma))
                        noise = noise * gamma[:, None]
                        noise_norm = torch.sqrt(noise.reshape(batch_size, -1).square().sum(dim=-1))
                        noise_inner = (noise.reshape(batch_size, -1) * v_flat).sum(dim=-1)
                        x = x + dt * v + math.sqrt(abs(dt)) * noise
                    else:
                        x = x + dt * v

                if adaptive_cfg["trigger_mode"] == "adaptive" and adaptive_cfg["cooldown_steps"] > 0:
                    cooldown = torch.where(
                        disagreement > adaptive_cfg["tau"],
                        torch.full_like(cooldown, adaptive_cfg["cooldown_steps"]),
                        torch.clamp(cooldown - 1, min=0),
                    )
            elif solver_name == "heun":
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
            elif solver_name == "dopri5":
                k1 = v
                k2 = _guided_field_at(x + dt * ((1.0 / 5.0) * k1), t_cur + dt * (1.0 / 5.0))
                k3 = _guided_field_at(
                    x + dt * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2),
                    t_cur + dt * (3.0 / 10.0),
                )
                k4 = _guided_field_at(
                    x + dt * ((44.0 / 45.0) * k1 + (-56.0 / 15.0) * k2 + (32.0 / 9.0) * k3),
                    t_cur + dt * (4.0 / 5.0),
                )
                k5 = _guided_field_at(
                    x
                    + dt
                    * (
                        (19372.0 / 6561.0) * k1
                        + (-25360.0 / 2187.0) * k2
                        + (64448.0 / 6561.0) * k3
                        + (-212.0 / 729.0) * k4
                    ),
                    t_cur + dt * (8.0 / 9.0),
                )
                k6 = _guided_field_at(
                    x
                    + dt
                    * (
                        (9017.0 / 3168.0) * k1
                        + (-355.0 / 33.0) * k2
                        + (46732.0 / 5247.0) * k3
                        + (49.0 / 176.0) * k4
                        + (-5103.0 / 18656.0) * k5
                    ),
                    t_cur + dt,
                )
                x = x + dt * (
                    (35.0 / 384.0) * k1
                    + (500.0 / 1113.0) * k3
                    + (125.0 / 192.0) * k4
                    + (-2187.0 / 6784.0) * k5
                    + (11.0 / 84.0) * k6
                )
                field_evals = torch.full_like(field_evals, 6.0)
            elif solver_name == "rk45":
                k1 = v
                k2 = _guided_field_at(x + dt * ((1.0 / 4.0) * k1), t_cur + dt * (1.0 / 4.0))
                k3 = _guided_field_at(
                    x + dt * ((3.0 / 32.0) * k1 + (9.0 / 32.0) * k2),
                    t_cur + dt * (3.0 / 8.0),
                )
                k4 = _guided_field_at(
                    x
                    + dt
                    * (
                        (1932.0 / 2197.0) * k1
                        + (-7200.0 / 2197.0) * k2
                        + (7296.0 / 2197.0) * k3
                    ),
                    t_cur + dt * (12.0 / 13.0),
                )
                k5 = _guided_field_at(
                    x
                    + dt
                    * (
                        (439.0 / 216.0) * k1
                        + (-8.0) * k2
                        + (3680.0 / 513.0) * k3
                        + (-845.0 / 4104.0) * k4
                    ),
                    t_cur + dt,
                )
                k6 = _guided_field_at(
                    x
                    + dt
                    * (
                        (-8.0 / 27.0) * k1
                        + 2.0 * k2
                        + (-3544.0 / 2565.0) * k3
                        + (1859.0 / 4104.0) * k4
                        + (-11.0 / 40.0) * k5
                    ),
                    t_cur + dt * 0.5,
                )
                x = x + dt * (
                    (16.0 / 135.0) * k1
                    + (6656.0 / 12825.0) * k3
                    + (28561.0 / 56430.0) * k4
                    + (-9.0 / 50.0) * k5
                    + (2.0 / 55.0) * k6
                )
                field_evals = torch.full_like(field_evals, 6.0)
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
            elif solver_name in {"euler_refine_half", "euler_refine_heun"}:
                trigger_eligible, fired, trigger_strength = self._refine_trigger_state(
                    disagreement,
                    normalized_disagreement,
                    step_idx=i,
                    n_steps=n_steps,
                    refine_cfg=refine_cfg,
                )
                x_euler = x + dt * v
                if torch.any(fired):
                    if solver_name == "euler_refine_half":
                        x_mid = x + 0.5 * dt * v
                        t_mid_tensor = torch.full((batch_size, 1), t_cur + 0.5 * dt, device=hist.device)
                        v_mid = self._guided_field(x_mid, t_mid_tensor, hist, cond=cond, guidance=guidance)
                        x_refined = x_mid + 0.5 * dt * v_mid
                    else:
                        x_pred = x + dt * v
                        t_next_tensor = torch.full((batch_size, 1), t_next, device=hist.device)
                        v_next = self._guided_field(x_pred, t_next_tensor, hist, cond=cond, guidance=guidance)
                        x_refined = x + dt * 0.5 * (v + v_next)
                    x = torch.where(fired[:, None], x_refined, x_euler)
                    field_evals = field_evals + fired.to(dtype=x.dtype)
                else:
                    x = x_euler
            else:
                raise ValueError(f"Unhandled sample solver={solver_name}")

            ema_beta = adaptive_cfg["beta"] if solver_name == "euler_adaptive" else refine_cfg["beta"]
            ema_v = ema_beta * ema_v + (1.0 - ema_beta) * v_flat.detach()
            ema_v_sq = ema_beta * ema_v_sq + (1.0 - ema_beta) * v_flat.detach().square()
            ema_u = ema_beta * ema_u + (1.0 - ema_beta) * u_flat.detach()

            if record_trace:
                trace_disagreement.append(disagreement.detach().cpu())
                trace_normalized_disagreement.append(normalized_disagreement.detach().cpu())
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
                trace_gamma.append(gamma.detach().cpu())
                trace_trigger.append(trigger_strength.detach().cpu())
                trace_fired.append(fired.detach().cpu())
                trace_noise_norm.append(noise_norm.detach().cpu())
                trace_noise_inner.append(noise_inner.detach().cpu())
                trace_oracle_error.append(oracle_error.detach().cpu())
                trace_field_evals.append(field_evals.detach().cpu())
                trace_time.append(float(t_cur))
                trace_gate.append(float(gate_scalar))
                trace_trigger_eligible.append(trigger_eligible.detach().cpu())

        trace: Optional[Dict[str, Any]] = None
        if record_trace:
            disagreement_t = torch.stack(trace_disagreement, dim=1)
            normalized_disagreement_t = torch.stack(trace_normalized_disagreement, dim=1)
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
            gamma_t = torch.stack(trace_gamma, dim=1)
            trigger_t = torch.stack(trace_trigger, dim=1)
            fired_t = torch.stack(trace_fired, dim=1)
            noise_norm_t = torch.stack(trace_noise_norm, dim=1)
            noise_inner_t = torch.stack(trace_noise_inner, dim=1)
            oracle_error_t = torch.stack(trace_oracle_error, dim=1)
            field_evals_t = torch.stack(trace_field_evals, dim=1)
            eligible_t = torch.stack(trace_trigger_eligible, dim=1)
            gate_t = torch.tensor(trace_gate, dtype=gamma_t.dtype)
            if solver_name == "euler_adaptive":
                active = (gate_t[None, :] > 0.0).expand_as(fired_t)
            else:
                active = eligible_t
            fire_rate = float(fired_t[active].float().mean().item()) if torch.any(active) else 0.0
            trace = {
                "solver": solver_name,
                "steps": int(n_steps),
                "step_index": torch.arange(n_steps, dtype=torch.long),
                "time": torch.tensor(trace_time, dtype=gamma_t.dtype),
                "time_grid": torch.tensor(time_grid, dtype=gamma_t.dtype),
                "time_gate": gate_t,
                "trigger_eligible": eligible_t,
                "disagreement": disagreement_t,
                "normalized_disagreement": normalized_disagreement_t,
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
                "gamma": gamma_t,
                "trigger_strength": trigger_t,
                "fired": fired_t,
                "triggered": fired_t,
                "noise_norm": noise_norm_t,
                "noise_velocity_inner": noise_inner_t,
                "field_evals_by_step": field_evals_t,
                "fire_rate_active": fire_rate,
                "mean_gamma": float(gamma_t.mean().item()),
                "mean_field_evals_per_step": float(field_evals_t.mean().item()),
                "mean_total_field_evals_per_rollout": float(field_evals_t.sum(dim=1).mean().item()),
                "tau": float(adaptive_cfg["tau"]),
                "threshold_z": float(refine_cfg["threshold_z"]),
                "noise_mode": adaptive_cfg["noise_mode"] if solver_name == "euler_adaptive" else "none",
                "trigger_mode": (
                    adaptive_cfg["trigger_mode"]
                    if solver_name == "euler_adaptive"
                    else refine_cfg["trigger_mode"]
                    if solver_name in {"euler_refine_half", "euler_refine_heun"}
                    else "none"
                ),
                "selected_steps": list(refine_cfg["selected_steps"]) if solver_name in {"euler_refine_half", "euler_refine_heun"} else [],
            }
        return x, trace

    @torch.no_grad()
    def sample_trace(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        solver: Optional[str] = None,
        oracle_local_error: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
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
        assert trace is not None
        return x, trace

    @torch.no_grad()
    def sample(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        solver: Optional[str] = None,
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
        cond: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        solver: Optional[str] = None,
        oracle_local_error: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, trace = self._sample_impl(
            hist,
            cond=cond,
            steps=steps,
            cfg_scale=cfg_scale,
            solver=solver,
            record_trace=True,
            oracle_local_error=oracle_local_error,
        )
        assert trace is not None
        return self._reshape_sample_block(x), trace

    @torch.no_grad()
    def sample_future(
        self,
        hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        solver: Optional[str] = None,
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


__all__ = ["OTFlow"]
