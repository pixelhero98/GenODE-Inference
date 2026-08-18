from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from genode.canonical_experiment_layout import schedule_family_for_key
from genode.evaluation.otflow_sampling_support import _apply_sample_overrides, _metric_bundle, _restore_sample_overrides
from genode.models.otflow_train_val import eval_many_windows
from genode.schedule_transfer.otflow_schedule_diagnostics import _collect_rollout_diagnostics
from genode.schedule_transfer.reference_clocks import (
    DEFAULT_REFERENCE_CLOCK_KEYS,
    REFERENCE_CLOCK_BASE_KEYS,
    REFERENCE_CLOCK_REVERSED_KEYS,
    build_reference_clock_grid,
    reference_clock_provenance,
    reference_clock_registry,
)
from genode.solver_protocol import solver_eval_multiplier

BASELINE_SCHEDULE_KEYS: tuple[str, ...] = REFERENCE_CLOCK_BASE_KEYS
TRANSFER_SCHEDULE_KEYS: tuple[str, ...] = (
    "ays_sd15_native",
    "ays_sd15_log_sigma",
    "gits_cifar10_native",
    "gits_cifar10_log_sigma",
    "ots_vp_linear_native",
    "ots_vp_linear_log_sigma",
)
EXPERIMENTAL_REVERSED_SCHEDULE_KEYS: tuple[str, ...] = REFERENCE_CLOCK_REVERSED_KEYS
EXPERIMENTAL_AVERAGED_FIXED_SCHEDULE_KEYS: tuple[str, ...] = ()
EXPERIMENTAL_FIXED_SCHEDULE_KEYS: tuple[str, ...] = DEFAULT_REFERENCE_CLOCK_KEYS


def load_external_schedule_catalog() -> dict[str, dict[str, Any]]:
    return {
        key: spec.as_dict()
        for key, spec in reference_clock_registry().items()
        if not key.endswith("_reversed") and spec.application_behavior == "transferred_reference"
    }


def build_schedule_grid(schedule_key: str, n_steps: int) -> tuple[float, ...] | None:
    """Build a fixed reference grid, returning ``None`` only for externally supplied dynamic clocks."""
    key = str(schedule_key).strip().lower()
    try:
        return build_reference_clock_grid(key, n_steps)
    except KeyError:
        return None


def schedule_display_name(schedule_key: str) -> str:
    key = str(schedule_key).strip().lower()
    try:
        return str(reference_clock_provenance(key)["display_name"])
    except KeyError:
        return str(schedule_key)


def schedule_time_alignment(schedule_key: str) -> str:
    key = str(schedule_key).strip().lower()
    try:
        provenance = reference_clock_provenance(key)
    except KeyError:
        return f"runtime_{key}"
    coordinate = str(provenance["coordinate"])
    suffix = "_reversed" if key.endswith("_reversed") else ""
    return f"runtime_{str(provenance['family'])}_{coordinate}{suffix}"


def schedule_density_family(schedule_key: str) -> str:
    key = str(schedule_key).strip().lower()
    try:
        provenance = reference_clock_provenance(key)
    except KeyError:
        return schedule_family_for_key(key)
    return f"{provenance['family']}_{provenance['coordinate']}"


def fixed_schedule_shape_statistics(time_grid: Sequence[float]) -> dict[str, float | None]:
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
) -> dict[str, Any]:
    solver_name = str(grid_spec["solver_name"])
    macro_steps = int(grid_spec["nfe"])
    target_total_field_evals = macro_steps * solver_eval_multiplier(solver_name)
    time_grid = tuple(float(x) for x in grid_spec["time_grid"])
    backup = _apply_sample_overrides(model, cfg, solver=solver_name, time_grid=time_grid)
    try:
        start = time.time()
        result = eval_many_windows(
            ds,
            model,
            cfg,
            horizon=int(eval_horizon),
            nfe=macro_steps,
            n_windows=int(eval_windows),
            seed=int(metrics_seed),
            horizons_eval=[int(eval_horizon)],
            chosen_t0s=chosen_t0s,
            generation_seed_base=int(generation_seed_base),
            metrics_seed=int(metrics_seed),
            main_metrics_only=bool(score_main_only),
        )
        eval_seconds = float(time.time() - start)
        diagnostics = _collect_rollout_diagnostics(
            model,
            ds,
            cfg,
            horizon=int(eval_horizon),
            macro_steps=macro_steps,
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
        "solver_name": solver_name,
        "nfe": macro_steps,
        "power": None if grid_spec.get("power") is None else float(grid_spec["power"]),
        "piecewise_early_frac": None
        if grid_spec.get("piecewise_early_frac") is None
        else float(grid_spec["piecewise_early_frac"]),
        "signal_validation_spearman": None
        if grid_spec.get("signal_validation_spearman") is None
        else float(grid_spec["signal_validation_spearman"]),
        "time_grid": [float(x) for x in time_grid],
        "target_total_field_evals": target_total_field_evals,
        "solver_override": solver_name,
        "eval_seconds": eval_seconds,
        "mean_field_evals_per_step": float(diagnostics["mean_field_evals_per_step"]),
        "mean_total_field_evals_per_rollout": float(diagnostics["mean_total_field_evals_per_rollout"]),
        "diag": diagnostics,
        "evaluation_protocol": {
            "chosen_t0s": [int(t0) for t0 in result["meta"]["chosen_t0s"]],
            "chosen_t0s_hash": str(result["meta"].get("chosen_t0s_hash", "")),
            "eval_horizon": int(result["meta"].get("horizon", eval_horizon)),
            "dataset_kind": str(result["meta"].get("dataset_kind", "")),
            "generation_seed_base": None
            if result["meta"]["generation_seed_base"] is None
            else int(result["meta"]["generation_seed_base"]),
            "metrics_seed": int(result["meta"]["metrics_seed"]),
            "main_metrics_only": bool(result["meta"].get("main_metrics_only", False)),
        },
        "per_window_metric_rows": [dict(item) for item in list(result["meta"].get("per_window_metric_rows", []) or [])],
        "score_main_only": bool(result["meta"].get("main_metrics_only", False)),
    }
    row.update(_metric_bundle(result))
    return row


__all__ = [
    "BASELINE_SCHEDULE_KEYS",
    "EXPERIMENTAL_AVERAGED_FIXED_SCHEDULE_KEYS",
    "EXPERIMENTAL_FIXED_SCHEDULE_KEYS",
    "EXPERIMENTAL_REVERSED_SCHEDULE_KEYS",
    "TRANSFER_SCHEDULE_KEYS",
    "build_schedule_grid",
    "fixed_schedule_shape_statistics",
    "load_external_schedule_catalog",
    "run_fixed_schedule_variant",
    "schedule_density_family",
    "schedule_display_name",
    "schedule_time_alignment",
]
