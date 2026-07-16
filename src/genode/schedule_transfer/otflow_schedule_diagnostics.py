from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from genode.evaluation.otflow_sampling_support import _choose_valid_windows
from genode.models.otflow_train_val import (
    _future_time_context_seq,
    _get_dataset_item_by_t,
    _parse_batch,
    _temporary_eval_seed,
    crop_history_window,
    resolve_context_length,
)


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
        raise ValueError(
            "Rollout context expansion requires explicit future_context_seq. "
            "Use a domain-specific rollout path for non-temporal augmented contexts."
        )
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


__all__: list[str] = []
