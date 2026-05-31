from __future__ import annotations

import json
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from genode.evaluation.otflow_sampling_support import _apply_sample_overrides, _metric_bundle, _restore_sample_overrides
from genode.schedule_transfer.otflow_schedule_diagnostics import _collect_rollout_diagnostics
from genode.models.otflow_train_val import eval_many_windows

BASELINE_SCHEDULE_KEYS: Tuple[str, ...] = ("uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots")
TRANSFER_SCHEDULE_KEYS: Tuple[str, ...] = ("ays", "gits", "ots")

_AYS_REFERENCE_TIMESTEPS: Tuple[int, ...] = (999, 850, 736, 645, 545, 455, 343, 233, 124, 24, 0)
_GITS_REFERENCE_SIGMAS: Tuple[float, ...] = (80.0, 10.9836, 3.8811, 1.5840, 0.5666, 0.1698, 0.0020)
_OTS_DEFAULT_EPS = 1e-3
_OTS_LINEAR_BETA_0 = 0.1
_OTS_LINEAR_BETA_1 = 20.0

_SCHEDULE_TIME_ALIGNMENT: Dict[str, str] = {
    "uniform": "runtime_uniform",
    "late_power_3": "runtime_late_power_3",
    "flowts_power_sampling": "runtime_flowts_power_sampling",
    "ays": "runtime_ays_ddpm_index_affine",
    "gits": "runtime_gits_sigma_affine",
    "ots": "runtime_ots_vp_time_affine",
}


def _uniform_grid(n_steps: int) -> Tuple[float, ...]:
    n_steps = int(n_steps)
    return tuple(float(idx) / float(n_steps) for idx in range(n_steps + 1))


def _late_power_grid(n_steps: int, power: float) -> Tuple[float, ...]:
    n_steps = int(n_steps)
    power = float(power)
    return tuple(1.0 - (1.0 - float(idx) / float(n_steps)) ** power for idx in range(n_steps + 1))


def _flowts_power_grid(n_steps: int, power: float = 0.03) -> Tuple[float, ...]:
    n_steps = int(n_steps)
    power = float(power)
    return tuple((float(idx) / float(n_steps)) ** power for idx in range(n_steps + 1))


def _ensure_monotone(grid: Sequence[float]) -> Tuple[float, ...]:
    if not grid:
        raise ValueError("Grid must contain at least one point.")
    out: List[float] = [float(grid[0])]
    for value in grid[1:]:
        current = float(value)
        if current <= out[-1]:
            current = min(1.0, out[-1] + 1e-6)
        out.append(current)
    out[0] = 0.0
    out[-1] = 1.0
    return tuple(float(x) for x in out)


def _resample_reference_progression(progression: Sequence[float], n_steps: int) -> Tuple[float, ...]:
    ref = np.asarray(progression, dtype=np.float64)
    if ref.ndim != 1 or ref.size < 2:
        raise ValueError("Reference progression must be one-dimensional with at least two points.")
    src = np.linspace(0.0, 1.0, int(ref.size), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, int(n_steps) + 1, dtype=np.float64)
    return _ensure_monotone(np.interp(dst, src, ref).tolist())


def _normalize_descending_reference(values: Sequence[float]) -> Tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("Expected a one-dimensional reference sequence.")
    span = float(arr[0] - arr[-1])
    if abs(span) < 1e-12:
        raise ValueError("Reference sequence must have non-zero span.")
    progression = (float(arr[0]) - arr) / span
    progression[0] = 0.0
    progression[-1] = 1.0
    return _ensure_monotone(progression.tolist())


def _ays_reference_progression() -> Tuple[float, ...]:
    return _normalize_descending_reference(_AYS_REFERENCE_TIMESTEPS)


def _gits_reference_progression() -> Tuple[float, ...]:
    return _normalize_descending_reference(_GITS_REFERENCE_SIGMAS)


def _scipy_optimizer():
    try:
        from scipy.optimize import LinearConstraint, minimize
    except ImportError as exc:
        raise RuntimeError("OTS schedule construction requires scipy.optimize to match DM-NonUniform.") from exc
    return minimize, LinearConstraint


class _NoiseScheduleVP:
    def __init__(
        self,
        *,
        continuous_beta_0: float = _OTS_LINEAR_BETA_0,
        continuous_beta_1: float = _OTS_LINEAR_BETA_1,
    ):
        self.beta_0 = float(continuous_beta_0)
        self.beta_1 = float(continuous_beta_1)
        self.T = 1.0

    def marginal_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)), min=1e-12))

    def marginal_lambda(self, t: torch.Tensor) -> torch.Tensor:
        alpha = self.marginal_alpha(t)
        sigma = self.marginal_std(t)
        return torch.log(torch.clamp(alpha / sigma, min=1e-12))

    def inverse_lambda(self, lamb: torch.Tensor) -> torch.Tensor:
        lamb = lamb.to(dtype=torch.float64)
        tmp = 2.0 * (self.beta_1 - self.beta_0) * torch.logaddexp(
            -2.0 * lamb,
            torch.zeros((1,), dtype=lamb.dtype, device=lamb.device),
        )
        denom = torch.sqrt(self.beta_0 ** 2 + tmp) + self.beta_0
        return tmp / torch.clamp(denom * (self.beta_1 - self.beta_0), min=1e-12)


class _OtsStepOptim:
    def __init__(self):
        self.ns = _NoiseScheduleVP()
        self.T = 1.0

    def alpha(self, t: Sequence[float]) -> np.ndarray:
        t_t = torch.as_tensor(t, dtype=torch.float64)
        return self.ns.marginal_alpha(t_t).detach().cpu().numpy()

    def sigma(self, t: Sequence[float]) -> np.ndarray:
        alpha = self.alpha(t)
        return np.sqrt(np.clip(1.0 - alpha * alpha, 1e-12, None))

    def lambda_func(self, t: Sequence[float]) -> np.ndarray:
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        return np.log(np.clip(alpha / sigma, 1e-12, None))

    def inverse_lambda(self, lamb: Sequence[float]) -> np.ndarray:
        lamb_t = torch.as_tensor(lamb, dtype=torch.float64)
        return self.ns.inverse_lambda(lamb_t).detach().cpu().numpy()

    def H0(self, h: np.ndarray) -> np.ndarray:
        return np.exp(h) - 1.0

    def H1(self, h: np.ndarray) -> np.ndarray:
        return np.exp(h) * h - self.H0(h)

    def H2(self, h: np.ndarray) -> np.ndarray:
        return np.exp(h) * h * h - 2.0 * self.H1(h)

    def sel_lambdas_lof_obj(self, lambda_vec: Sequence[float], eps: float) -> float:
        lambda_eps = float(self.lambda_func([eps])[0])
        lambda_T = float(self.lambda_func([self.T])[0])
        lambda_vec_ext = np.concatenate(([lambda_T], np.asarray(lambda_vec, dtype=np.float64), [lambda_eps]))
        hv = np.diff(lambda_vec_ext)
        emlv_sq = np.exp(-2.0 * lambda_vec_ext)
        alpha_vec = 1.0 / np.sqrt(1.0 + emlv_sq)
        sigma_vec = 1.0 / np.sqrt(1.0 + np.exp(2.0 * lambda_vec_ext))
        data_err_vec = (sigma_vec ** 2) / np.clip(alpha_vec, 1e-12, None)
        trunc_num = 3
        res = 0.0
        c_vec = np.zeros(len(lambda_vec_ext) - 1, dtype=np.float64)
        elv = np.exp(lambda_vec_ext)
        for s in range(len(lambda_vec_ext) - 1):
            if s in (0, len(lambda_vec_ext) - 2):
                coeff = elv[s + 1] - elv[s]
                res += abs(coeff * data_err_vec[s])
            elif s in (1, len(lambda_vec_ext) - 3):
                n = s - 1
                j0 = -elv[n + 1] * self.H1(hv[n + 1]) / max(hv[n], 1e-12)
                j1 = elv[n + 1] * (self.H1(hv[n + 1]) + hv[n] * self.H0(hv[n + 1])) / max(hv[n], 1e-12)
                if s >= trunc_num:
                    c_vec[n] += data_err_vec[n] * j0
                    c_vec[n + 1] += data_err_vec[n + 1] * j1
                else:
                    res += math.sqrt((data_err_vec[n] * j0) ** 2 + (data_err_vec[n + 1] * j1) ** 2)
            else:
                n = s - 2
                denom0 = max(hv[n] * (hv[n] + hv[n + 1]), 1e-12)
                denom1 = max(hv[n] * hv[n + 1], 1e-12)
                denom2 = max(hv[n + 1] * (hv[n] + hv[n + 1]), 1e-12)
                j0 = elv[n + 2] * (self.H2(hv[n + 2]) + hv[n + 1] * self.H1(hv[n + 2])) / denom0
                j1 = -elv[n + 2] * (self.H2(hv[n + 2]) + (hv[n] + hv[n + 1]) * self.H1(hv[n + 2])) / denom1
                j2 = (
                    elv[n + 2]
                    * (
                        self.H2(hv[n + 2])
                        + (2.0 * hv[n + 1] + hv[n]) * self.H1(hv[n + 2])
                        + hv[n + 1] * (hv[n] + hv[n + 1]) * self.H0(hv[n + 2])
                    )
                    / denom2
                )
                if s >= trunc_num:
                    c_vec[n] += data_err_vec[n] * j0
                    c_vec[n + 1] += data_err_vec[n + 1] * j1
                    c_vec[n + 2] += data_err_vec[n + 2] * j2
                else:
                    res += math.sqrt(
                        (data_err_vec[n] * j0) ** 2
                        + (data_err_vec[n + 1] * j1) ** 2
                        + (data_err_vec[n + 2] * j2) ** 2
                    )
        res += float(np.sum(np.abs(c_vec)))
        return float(res)

    def get_ts_lambdas(
        self,
        n_steps: int,
        eps: float = _OTS_DEFAULT_EPS,
        init_type: str = "unif_t",
    ) -> Tuple[np.ndarray, np.ndarray]:
        minimize, LinearConstraint = _scipy_optimizer()
        n_steps = int(n_steps)
        if n_steps < 1:
            raise ValueError("OTS n_steps must be positive.")
        lambda_eps = float(self.lambda_func([eps])[0])
        lambda_T = float(self.lambda_func([self.T])[0])
        if n_steps == 1:
            lambda_res = np.asarray([lambda_T, lambda_eps], dtype=np.float64)
            t_res = self.inverse_lambda(lambda_res)
            t_res[0] = self.T
            t_res[-1] = eps
            return t_res, lambda_res

        constr_mat = np.zeros((n_steps, max(n_steps - 1, 0)), dtype=np.float64)
        for idx in range(n_steps - 1):
            constr_mat[idx][idx] = 1.0
            constr_mat[idx + 1][idx] = -1.0
        lb_vec = np.zeros(n_steps, dtype=np.float64)
        lb_vec[0] = lambda_T
        lb_vec[-1] = -lambda_eps
        ub_vec = np.full(n_steps, np.inf, dtype=np.float64)
        linear_constraint = LinearConstraint(constr_mat, lb_vec, ub_vec)

        if init_type == "unif":
            lambda_vec_ext = np.linspace(lambda_T, lambda_eps, n_steps + 1, dtype=np.float64)
        elif init_type == "unif_t":
            t_vec = np.linspace(self.T, eps, n_steps + 1, dtype=np.float64)
            lambda_vec_ext = np.asarray(self.lambda_func(t_vec), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported OTS init_type={init_type!r}.")

        lambda_init = np.asarray(lambda_vec_ext[1:-1], dtype=np.float64)
        result = minimize(
            self.sel_lambdas_lof_obj,
            lambda_init,
            method="trust-constr",
            args=(float(eps),),
            constraints=[linear_constraint],
            options={"verbose": 0},
        )
        if not np.all(np.isfinite(result.x)):
            raise RuntimeError("OTS scipy optimizer produced non-finite lambda nodes.")
        if not bool(result.success):
            raise RuntimeError(f"OTS scipy optimizer failed: {result.message}")
        lambda_opt = np.asarray(result.x, dtype=np.float64)
        lambda_res = np.concatenate(([lambda_T], lambda_opt, [lambda_eps]))
        t_res = self.inverse_lambda(lambda_res)
        t_res[0] = self.T
        t_res[-1] = eps
        return t_res, lambda_res


@lru_cache(maxsize=None)
def _ots_reference_progression(n_steps: int) -> Tuple[float, ...]:
    optim = _OtsStepOptim()
    t_res, _ = optim.get_ts_lambdas(int(n_steps), eps=_OTS_DEFAULT_EPS, init_type="unif_t")
    progression = (float(optim.T) - t_res) / max(float(optim.T - _OTS_DEFAULT_EPS), 1e-12)
    progression[0] = 0.0
    progression[-1] = 1.0
    return _ensure_monotone(progression.tolist())


def _catalog_path() -> Path:
    return Path(__file__).resolve().with_name("otflow_external_schedule_catalog.json")


def load_external_schedule_catalog() -> Dict[str, Dict[str, Any]]:
    if not _catalog_path().exists():
        return {}
    payload = json.loads(_catalog_path().read_text(encoding="utf-8"))
    return {str(key).lower(): dict(value) for key, value in payload.items()}


def _validate_positive_n_steps(n_steps: int) -> int:
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}.")
    return n_steps


def build_schedule_grid(schedule_key: str, n_steps: int) -> Optional[Tuple[float, ...]]:
    key = str(schedule_key).strip().lower()
    n_steps = _validate_positive_n_steps(int(n_steps))
    if key == "uniform":
        return _ensure_monotone(_uniform_grid(n_steps))
    if key == "late_power_3":
        return _ensure_monotone(_late_power_grid(n_steps, power=3.0))
    if key == "flowts_power_sampling":
        return _ensure_monotone(_flowts_power_grid(n_steps, power=0.03))
    if key == "ays":
        return _resample_reference_progression(_ays_reference_progression(), n_steps)
    if key == "gits":
        return _resample_reference_progression(_gits_reference_progression(), n_steps)
    if key == "ots":
        return _ots_reference_progression(n_steps)
    return None

def schedule_display_name(schedule_key: str) -> str:
    key = str(schedule_key).strip().lower()
    names = {
        "uniform": "Time-uniform",
        "late_power_3": "Late-power-3",
        "flowts_power_sampling": "FlowTS power sampling",
        "ays": "AYS",
        "gits": "GITS",
        "ots": "OTS",
    }
    return names.get(key, str(schedule_key))


def schedule_time_alignment(schedule_key: str) -> str:
    key = str(schedule_key).strip().lower()
    return _SCHEDULE_TIME_ALIGNMENT.get(key, f"runtime_{key}")


def fixed_schedule_shape_statistics(time_grid: Sequence[float]) -> Dict[str, Optional[float]]:
    grid = np.asarray(time_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        return {"runtime_grid_q25": None, "runtime_grid_q50": None, "runtime_grid_q75": None}
    positions = np.linspace(0.0, 1.0, int(grid.size), dtype=np.float64)
    q25, q50, q75 = np.interp(np.asarray([0.25, 0.50, 0.75], dtype=np.float64), positions, grid)
    return {"runtime_grid_q25": float(q25), "runtime_grid_q50": float(q50), "runtime_grid_q75": float(q75)}


def run_fixed_schedule_variant(
    *,
    model,
    ds,
    cfg,
    eval_horizon: int,
    eval_windows: int,
    grid_spec: Mapping[str, Any],
    chosen_t0s: Sequence[int],
    generation_seed_base: int,
    metrics_seed: int,
    score_main_only: bool,
) -> Dict[str, Any]:
    solver_name = str(grid_spec["solver_name"])
    time_grid = tuple(float(x) for x in grid_spec["time_grid"])
    backup = _apply_sample_overrides(model, cfg, solver=solver_name, time_grid=time_grid)
    try:
        t0 = time.time()
        result = eval_many_windows(
            ds,
            model,
            cfg,
            horizon=int(eval_horizon),
            nfe=int(grid_spec["nfe"]),
            n_windows=int(eval_windows),
            seed=int(metrics_seed),
            horizons_eval=[int(eval_horizon)],
            chosen_t0s=chosen_t0s,
            generation_seed_base=int(generation_seed_base),
            metrics_seed=int(metrics_seed),
            main_metrics_only=bool(score_main_only),
        )
        eval_seconds = float(time.time() - t0)
        diag = _collect_rollout_diagnostics(
            model,
            ds,
            cfg,
            horizon=int(eval_horizon),
            macro_steps=int(grid_spec["nfe"]),
            n_windows=int(eval_windows),
            seed=int(metrics_seed),
            solver=solver_name,
            chosen_t0s=chosen_t0s,
            generation_seed_base=int(generation_seed_base),
        )
    finally:
        _restore_sample_overrides(model, cfg, backup)

    row = {
        "grid_name": str(grid_spec["grid_name"]),
        "grid_kind": str(grid_spec["grid_kind"]),
        "selection_group": str(grid_spec["selection_group"]),
        "comparison_role": None if grid_spec.get("comparison_role") is None else str(grid_spec["comparison_role"]),
        "solver_name": str(solver_name),
        "nfe": int(grid_spec["nfe"]),
        "power": None if grid_spec.get("power") is None else float(grid_spec["power"]),
        "piecewise_early_frac": None
        if grid_spec.get("piecewise_early_frac") is None
        else float(grid_spec["piecewise_early_frac"]),
        "signal_validation_spearman": None
        if grid_spec.get("signal_validation_spearman") is None
        else float(grid_spec["signal_validation_spearman"]),
        "time_grid": [float(x) for x in time_grid],
        "target_total_field_evals": int(grid_spec["nfe"]),
        "solver_override": str(solver_name),
        "eval_seconds": eval_seconds,
        "mean_field_evals_per_step": float(diag["mean_field_evals_per_step"]),
        "mean_total_field_evals_per_rollout": float(diag["mean_total_field_evals_per_rollout"]),
        "trigger_rate": float(diag["trigger_rate"]),
        "diag": diag,
        "evaluation_protocol": {
            "chosen_t0s": [int(t0) for t0 in result["meta"]["chosen_t0s"]],
            "chosen_t0s_hash": str(result["meta"].get("chosen_t0s_hash", "")),
            "stage_counts": dict(result["meta"].get("stage_counts", {}) or {}),
            "eval_horizon": int(result["meta"].get("horizon", eval_horizon)),
            "dataset_kind": str(result["meta"].get("dataset_kind", "")),
            "generation_seed_base": None
            if result["meta"]["generation_seed_base"] is None
            else int(result["meta"]["generation_seed_base"]),
            "metrics_seed": int(result["meta"]["metrics_seed"]),
            "main_metrics_only": bool(result["meta"].get("main_metrics_only", False)),
        },
        "score_main_only": bool(result["meta"].get("main_metrics_only", False)),
    }
    row.update(_metric_bundle(result))
    main_metrics = dict(result.get("cmp", {}).get("main", {}) or {})
    for metric_name in ("stage_mismatch_rate", "stage_classifier_real_macro_f1"):
        metric_payload = main_metrics.get(metric_name, {})
        if isinstance(metric_payload, Mapping) and "mean" in metric_payload:
            row[metric_name] = float(metric_payload["mean"])
    return row


__all__ = [
    "BASELINE_SCHEDULE_KEYS",
    "TRANSFER_SCHEDULE_KEYS",
    "build_schedule_grid",
    "fixed_schedule_shape_statistics",
    "load_external_schedule_catalog",
    "run_fixed_schedule_variant",
    "schedule_display_name",
    "schedule_time_alignment",
]
