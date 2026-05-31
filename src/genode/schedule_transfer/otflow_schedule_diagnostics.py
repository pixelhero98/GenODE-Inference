from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from genode.evaluation.otflow_sampling_support import _choose_valid_windows
from genode.schedule_transfer.otflow_signal_traces import (
    MODEL_SIGNAL_SPECS,
    NATIVE_INFO_GROWTH_ROW_KEY,
    NATIVE_INFO_GROWTH_TRACE_KEY,
    compute_info_growth_hardness_numpy,
    resolved_info_growth_scale,
)
from genode.models.otflow_train_val import (
    _future_time_context_seq,
    _get_dataset_item_by_t,
    _parse_batch,
    _temporary_eval_seed,
    crop_history_window,
    resolve_context_length,
)


def _safe_percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size > 0 else float("nan")


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    if x.size < 2 or np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return {"n": int(x.size), "pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])
    return {"n": int(x.size), "pearson": pearson, "spearman": spearman}


def _step_arrays(rows: Sequence[Mapping[str, Any]], macro_steps: int) -> Dict[str, List[float]]:
    mu = []
    sigma = []
    disagreement_by_step = []
    oracle_by_step = []
    signal_specs = [(row_key, out_key) for row_key, out_key in MODEL_SIGNAL_SPECS if out_key != "disagreement_by_step"]
    if rows and NATIVE_INFO_GROWTH_ROW_KEY in rows[0]:
        signal_specs.append((NATIVE_INFO_GROWTH_ROW_KEY, NATIVE_INFO_GROWTH_TRACE_KEY))
    signal_means: Dict[str, List[float]] = {out_key: [] for _, out_key in signal_specs}
    for step_idx in range(int(macro_steps)):
        d_vals = np.asarray(
            [float(row["disagreement"]) for row in rows if int(row["step_index"]) == step_idx],
            dtype=np.float64,
        )
        e_vals = np.asarray(
            [float(row["oracle_local_error"]) for row in rows if int(row["step_index"]) == step_idx],
            dtype=np.float64,
        )
        mu.append(float(np.mean(d_vals)) if d_vals.size > 0 else 0.0)
        sigma.append(float(np.std(d_vals)) if d_vals.size > 0 else 0.0)
        disagreement_by_step.append(float(np.mean(d_vals)) if d_vals.size > 0 else float("nan"))
        oracle_by_step.append(float(np.mean(e_vals)) if e_vals.size > 0 else float("nan"))
        for row_key, out_key in signal_specs:
            vals = np.asarray(
                [float(row[row_key]) for row in rows if int(row["step_index"]) == step_idx],
                dtype=np.float64,
            )
            signal_means[out_key].append(float(np.mean(vals)) if vals.size > 0 else float("nan"))
    payload = {
        "step_mu": mu,
        "step_sigma": sigma,
        "disagreement_by_step": disagreement_by_step,
        "oracle_local_error_by_step": oracle_by_step,
    }
    payload.update(signal_means)
    return payload


def _prediction_horizon(model) -> int:
    model_cfg = getattr(model, "cfg", None)
    return int(max(1, int(getattr(model_cfg, "prediction_horizon", 1))))


def _sample_eval_trace(
    model,
    hist_t: torch.Tensor,
    *,
    cond_t: Optional[torch.Tensor],
    steps: int,
    solver: str,
    oracle_local_error: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any], int]:
    prediction_horizon = _prediction_horizon(model)
    if prediction_horizon > 1 and hasattr(model, "sample_future_trace"):
        x_block, trace = model.sample_future_trace(
            hist_t,
            cond=cond_t,
            steps=int(steps),
            solver=solver,
            oracle_local_error=oracle_local_error,
        )
        return x_block, trace, int(x_block.shape[1])

    x_next, trace = model.sample_trace(
        hist_t,
        cond=cond_t,
        steps=int(steps),
        solver=solver,
        oracle_local_error=oracle_local_error,
    )
    return x_next[:, None, :], trace, 1


def _append_rollout_context_features(
    block: torch.Tensor,
    *,
    x_hist: torch.Tensor,
    future_context_seq: Optional[torch.Tensor],
    cursor: int,
    take: int,
) -> torch.Tensor:
    target_dim = int(x_hist.shape[-1])
    block_dim = int(block.shape[-1])
    if block_dim == target_dim:
        return block
    if block_dim > target_dim:
        return block[..., :target_dim]

    extra_dim = int(target_dim - block_dim)
    if future_context_seq is None:
        extra = torch.zeros(block.shape[0], int(take), extra_dim, device=block.device, dtype=block.dtype)
    else:
        extra = future_context_seq[:, int(cursor) : int(cursor) + int(take), :].to(
            device=block.device,
            dtype=block.dtype,
        )
        if int(extra.shape[1]) < int(take):
            pad = torch.zeros(
                extra.shape[0],
                int(take) - int(extra.shape[1]),
                extra.shape[2],
                device=extra.device,
                dtype=extra.dtype,
            )
            extra = torch.cat([extra, pad], dim=1)
        if int(extra.shape[-1]) < extra_dim:
            pad = torch.zeros(
                extra.shape[0],
                extra.shape[1],
                extra_dim - int(extra.shape[-1]),
                device=extra.device,
                dtype=extra.dtype,
            )
            extra = torch.cat([extra, pad], dim=-1)
        elif int(extra.shape[-1]) > extra_dim:
            extra = extra[..., :extra_dim]
    return torch.cat([block, extra], dim=-1)


def _collect_calibration(
    model,
    ds_val,
    cfg,
    *,
    horizon: int,
    macro_steps: int,
    n_windows: int,
    seed: int,
    sigma_eps: float,
    solver: str = "euler",
    chosen_t0s: Optional[Sequence[int]] = None,
    generation_seed_base: Optional[int] = None,
) -> Dict[str, Any]:
    if chosen_t0s is None:
        chosen = _choose_valid_windows(ds_val, horizon=horizon, n_windows=n_windows, seed=seed)
    else:
        chosen = np.asarray([int(t0) for t0 in chosen_t0s], dtype=np.int64)
    seed_base = int(seed if generation_seed_base is None else generation_seed_base)
    rows: List[Dict[str, Any]] = []
    reference_time_grid: Optional[np.ndarray] = None

    for window_idx, t0 in enumerate(chosen.tolist()):
        batch = _get_dataset_item_by_t(ds_val, int(t0))
        hist, _, _, cond, _ = _parse_batch(batch)
        hist_t = hist[None, :, :].to(cfg.device).float()
        cond_t = cond[None, :].to(cfg.device).float() if cond is not None else None
        with _temporary_eval_seed(seed_base + int(window_idx)):
            _, trace, _ = _sample_eval_trace(
                model,
                hist_t,
                cond_t=cond_t,
                steps=int(macro_steps),
                solver=str(solver),
                oracle_local_error=True,
            )
        rollout_time_grid = trace["time_grid"].cpu().numpy()
        if reference_time_grid is None:
            reference_time_grid = np.asarray(rollout_time_grid, dtype=np.float64)
        elif not np.allclose(reference_time_grid, rollout_time_grid, atol=1e-8, rtol=1e-8):
            raise ValueError("Calibration trace time grids must match across validation rollouts.")
        step_left_times = reference_time_grid[:-1]
        disagreement = trace["disagreement"][0].cpu().numpy()
        velocity = trace["velocity_norm"][0].cpu().numpy()
        trace_signal_arrays = {
            row_key: trace[row_key][0].cpu().numpy()
            for row_key, _ in MODEL_SIGNAL_SPECS
            if row_key != "disagreement"
        }
        oracle = trace["oracle_local_error"][0].cpu().numpy()
        for step_idx in range(int(macro_steps)):
            row = {
                "window_index": int(window_idx),
                "t0": int(t0),
                "step_index": int(step_idx),
                "time": float(step_left_times[step_idx]),
                "disagreement": float(disagreement[step_idx]),
                "velocity_norm": float(velocity[step_idx]),
                "oracle_local_error": float(oracle[step_idx]),
            }
            for row_key, _ in MODEL_SIGNAL_SPECS:
                if row_key == "disagreement":
                    continue
                row[row_key] = float(trace_signal_arrays[row_key][step_idx])
            rows.append(row)

    residual_values = np.asarray([float(row["residual_norm"]) for row in rows], dtype=np.float64)
    info_growth_scale = resolved_info_growth_scale(residual_values)
    for row in rows:
        row[NATIVE_INFO_GROWTH_ROW_KEY] = float(
            compute_info_growth_hardness_numpy(
                np.asarray([float(row["residual_norm"])], dtype=np.float64),
                np.asarray([float(row["disagreement"])], dtype=np.float64),
                scale=float(info_growth_scale),
            )[0]
        )

    step_stats = _step_arrays(rows, macro_steps)
    for row in rows:
        step_idx = int(row["step_index"])
        sigma = max(float(step_stats["step_sigma"][step_idx]), float(sigma_eps))
        row["normalized_disagreement"] = float(
            (float(row["disagreement"]) - float(step_stats["step_mu"][step_idx])) / sigma
        )

    corr_rows = [row for row in rows if int(row["step_index"]) > 0]
    d_arr = np.asarray([float(row["disagreement"]) for row in corr_rows], dtype=np.float64)
    z_arr = np.asarray([float(row["normalized_disagreement"]) for row in corr_rows], dtype=np.float64)
    e_arr = np.asarray([float(row["oracle_local_error"]) for row in corr_rows], dtype=np.float64)
    signal_specs = list(MODEL_SIGNAL_SPECS) + [(NATIVE_INFO_GROWTH_ROW_KEY, NATIVE_INFO_GROWTH_TRACE_KEY)]
    signal_correlations_vs_oracle = {
        out_key: _safe_corr(np.asarray([float(row[row_key]) for row in corr_rows], dtype=np.float64), e_arr)
        for row_key, out_key in signal_specs
    }
    if reference_time_grid is None:
        reference_time_grid = np.linspace(0.0, 1.0, int(macro_steps) + 1, dtype=np.float64)

    payload = {
        "macro_steps": int(macro_steps),
        "solver": str(solver),
        "n_windows": int(len(chosen)),
        "exclude_step0_for_correlation": True,
        "reference_time_grid": [float(x) for x in reference_time_grid.tolist()],
        "reference_time_alignment": "left_endpoint",
        "info_growth_scale": float(info_growth_scale),
        "rows": rows,
        "step_mu": step_stats["step_mu"],
        "step_sigma": step_stats["step_sigma"],
        "disagreement_by_step": step_stats["disagreement_by_step"],
        "oracle_local_error_by_step": step_stats["oracle_local_error_by_step"],
        "disagreement_stats": {
            "mean": float(np.mean(d_arr)) if d_arr.size > 0 else float("nan"),
            "std": float(np.std(d_arr)) if d_arr.size > 0 else float("nan"),
            "p50": _safe_percentile(d_arr, 0.50),
            "p85": _safe_percentile(d_arr, 0.85),
            "p95": _safe_percentile(d_arr, 0.95),
        },
        "oracle_local_error_stats": {
            "mean": float(np.mean(e_arr)) if e_arr.size > 0 else float("nan"),
            "std": float(np.std(e_arr)) if e_arr.size > 0 else float("nan"),
            "p50": _safe_percentile(e_arr, 0.50),
            "p85": _safe_percentile(e_arr, 0.85),
            "p95": _safe_percentile(e_arr, 0.95),
        },
        "correlation_disagreement_vs_oracle": _safe_corr(d_arr, e_arr),
        "correlation_normalized_disagreement_vs_oracle": _safe_corr(z_arr, e_arr),
        "signal_correlations_vs_oracle": signal_correlations_vs_oracle,
    }
    for _, out_key in signal_specs:
        if out_key == "disagreement_by_step":
            continue
        payload[out_key] = step_stats[out_key]
    return payload


def _collect_rollout_diagnostics(
    model,
    ds,
    cfg,
    *,
    horizon: int,
    macro_steps: int,
    n_windows: int,
    seed: int,
    solver: str,
    chosen_t0s: Optional[Sequence[int]] = None,
    generation_seed_base: Optional[int] = None,
) -> Dict[str, Any]:
    if chosen_t0s is None:
        chosen = _choose_valid_windows(ds, horizon=horizon, n_windows=n_windows, seed=seed)
    else:
        chosen = np.asarray([int(t0) for t0 in chosen_t0s], dtype=np.int64)
    if chosen.ndim != 1 or chosen.size == 0:
        raise ValueError("chosen_t0s must be a non-empty 1D sequence of valid window starts.")
    seed_base = int(seed if generation_seed_base is None else generation_seed_base)
    fired_rows = []
    eligible_rows = []
    field_eval_rows = []
    z_rows = []
    d_rows = []
    sample_total_evals = []

    for window_idx, t0 in enumerate(chosen.tolist()):
        batch = _get_dataset_item_by_t(ds, int(t0))
        hist, _, _, _, _ = _parse_batch(batch)
        hist_t = hist[None, :, :].to(cfg.device).float()
        context_len = resolve_context_length(hist_t.shape[1], horizon=horizon, cfg=cfg)
        cond_seq = None
        if ds.cond is not None:
            cond_seq = torch.from_numpy(ds.cond[int(t0) : int(t0) + int(horizon)]).to(cfg.device).float()[None, :, :]
        future_context_seq = None
        future_context = _future_time_context_seq(ds, int(t0), int(horizon))
        if future_context is not None:
            future_context_seq = future_context.to(cfg.device).float()[None, :, :]

        x_hist = crop_history_window(hist_t, context_len).clone()
        cursor = 0
        while cursor < int(horizon):
            cond_t = cond_seq[:, cursor, :] if cond_seq is not None else None
            call_seed = seed_base + int(window_idx) * int(horizon) + int(cursor)
            with _temporary_eval_seed(call_seed):
                x_block, trace, block_len = _sample_eval_trace(
                    model,
                    x_hist,
                    cond_t=cond_t,
                    steps=int(macro_steps),
                    solver=solver,
                )
            fired_rows.append(trace["fired"].to(dtype=torch.float32).cpu().numpy()[0])
            eligible_rows.append(trace["trigger_eligible"].to(dtype=torch.float32).cpu().numpy()[0])
            field_eval_rows.append(trace["field_evals_by_step"].cpu().numpy()[0])
            z_rows.append(trace["normalized_disagreement"].cpu().numpy()[0])
            d_rows.append(trace["disagreement"].cpu().numpy()[0])
            sample_total_evals.append(float(trace["mean_total_field_evals_per_rollout"]))
            take = min(int(block_len), int(horizon) - int(cursor))
            hist_block = _append_rollout_context_features(
                x_block[:, :take, :],
                x_hist=x_hist,
                future_context_seq=future_context_seq,
                cursor=int(cursor),
                take=int(take),
            )
            x_hist = torch.cat([x_hist, hist_block], dim=1)
            x_hist = crop_history_window(x_hist, context_len)
            cursor += int(take)

    fired = np.asarray(fired_rows, dtype=np.float32)
    eligible = np.asarray(eligible_rows, dtype=np.float32)
    field_evals = np.asarray(field_eval_rows, dtype=np.float32)
    z_vals = np.asarray(z_rows, dtype=np.float32)
    d_vals = np.asarray(d_rows, dtype=np.float32)
    eligible_mask = eligible > 0.5

    trigger_rate = float(fired[eligible_mask].mean()) if np.any(eligible_mask) else 0.0
    return {
        "n_rollout_calls": int(fired.shape[0]),
        "macro_steps": int(macro_steps),
        "trigger_rate": trigger_rate,
        "trigger_by_step": [float(x) for x in fired.mean(axis=0)],
        "eligible_by_step": [float(x) for x in eligible.mean(axis=0)],
        "field_evals_by_step": [float(x) for x in field_evals.mean(axis=0)],
        "disagreement_by_step": [float(x) for x in d_vals.mean(axis=0)],
        "normalized_disagreement_by_step": [float(x) for x in z_vals.mean(axis=0)],
        "mean_field_evals_per_step": float(field_evals.mean()),
        "mean_total_field_evals_per_rollout": float(np.mean(sample_total_evals)),
    }


__all__ = [
    "_append_rollout_context_features",
    "_collect_calibration",
    "_collect_rollout_diagnostics",
    "_sample_eval_trace",
]
