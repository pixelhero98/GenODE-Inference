from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from genode.data.otflow_experiment_plan import FORECAST_FAMILY


def _row_value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _relative_match_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _row_value(row, "benchmark_family"),
        _row_value(row, "split_phase"),
        _row_value(row, "dataset", "dataset_key"),
        _row_value(row, "backbone_name"),
        _row_value(row, "train_steps"),
        _row_value(row, "train_budget_label"),
        _row_value(row, "checkpoint_id"),
        _row_value(row, "target_nfe"),
        _row_value(row, "solver_key", "solver_name"),
        _row_value(row, "experiment_scope"),
        _row_value(row, "seed"),
    )


def _safe_relative_gain(metric_value: Any, baseline_value: Any) -> float | None:
    try:
        metric = float(metric_value)
        baseline = float(baseline_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(metric) or not math.isfinite(baseline) or baseline <= 0.0:
        return None
    return float(1.0 - metric / baseline)


def _relative_gain_value(row: Mapping[str, Any], relative_key: str) -> float | None:
    value = _row_value(row, relative_key, f"{relative_key}_mean")
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cast):
        return None
    return cast


def _metric_value(row: Mapping[str, Any], metric_key: str) -> Any:
    return _row_value(row, metric_key, f"{metric_key}_mean")


def augment_rows_with_relative_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baseline_rows: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("scheduler_key") == "uniform":
            baseline_rows[_relative_match_key(row)] = row
    enriched: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        baseline = baseline_rows.get(_relative_match_key(row))
        family = str(_row_value(row, "benchmark_family") or "")
        payload["forecast_relative_crps_gain_vs_uniform"] = _relative_gain_value(
            row, "forecast_relative_crps_gain_vs_uniform"
        )
        payload["forecast_relative_mase_gain_vs_uniform"] = _relative_gain_value(
            row, "forecast_relative_mase_gain_vs_uniform"
        )
        if baseline is not None and family == FORECAST_FAMILY:
            if payload["forecast_relative_crps_gain_vs_uniform"] is None:
                payload["forecast_relative_crps_gain_vs_uniform"] = _safe_relative_gain(
                    _metric_value(row, "forecast_crps"), _metric_value(baseline, "forecast_crps")
                )
            if payload["forecast_relative_mase_gain_vs_uniform"] is None:
                payload["forecast_relative_mase_gain_vs_uniform"] = _safe_relative_gain(
                    _metric_value(row, "forecast_mase"), _metric_value(baseline, "forecast_mase")
                )
        enriched.append(payload)
    return enriched
