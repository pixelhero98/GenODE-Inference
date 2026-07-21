from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
from genode.models.otflow_train_val import select_eval_window_starts


PRIMARY_METRICS = (
    "score_main",
    "temporal_tstr_f1",
    "temporal_tstr_f1_applicable",
    "disc_auc",
    "disc_auc_gap",
    "temporal_uw1",
    "temporal_cw1",
)

EXTRA_METRICS = (
    "u_l1",
    "c_l1",
    "spread_specific_error",
    "imbalance_specific_error",
    "ret_vol_acf_error",
    "impact_response_error",
    "efficiency_ms_per_sample",
)

ALL_METRICS = PRIMARY_METRICS + EXTRA_METRICS


def _metric_value(result: Mapping[str, Any], metric: str) -> Any:
    if metric == "score_main":
        return float(result["cmp"]["score_main"]["mean"])
    if metric == "temporal_tstr_f1_applicable":
        return bool(result["cmp"]["main"][metric])
    if metric in PRIMARY_METRICS:
        value = result["cmp"]["main"][metric]["mean"]
        return None if value is None else float(value)
    return float(result["cmp"]["extra"][metric]["mean"])


def _metric_bundle(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {metric: _metric_value(result, metric) for metric in ALL_METRICS}


def _choose_valid_windows(ds, horizon: int, n_windows: int, seed: int) -> np.ndarray:
    return select_eval_window_starts(ds, horizon=int(horizon), n_windows=int(n_windows), seed=int(seed))


__all__ = [
    "ALL_METRICS",
    "EXTRA_METRICS",
    "PRIMARY_METRICS",
]
