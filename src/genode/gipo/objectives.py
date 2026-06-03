from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

MetricRow = Mapping[str, object]
SettingKey = Tuple[str, int]
ScheduleKey = Tuple[str, int, str]
DEFAULT_REWARD_EPS = 1e-12
UNIFORM_SCHEDULE_KEY = "uniform"


def _finite_positive(value: object) -> float:
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"Metric values must be finite and positive, got {value!r}")
    return val


def crps_mase_reward(
    crps: float,
    mase: float,
    *,
    crps_center: float,
    mase_center: float,
    eps: float = DEFAULT_REWARD_EPS,
) -> float:
    """Equal-weight lower-is-better CRPS/MASE utility with no soft regularizers.

    ``crps_center`` and ``mase_center`` are reference metrics for the same
    solver/NFE cell, usually fixed-support baselines used in reporting.
    """
    c = _finite_positive(crps)
    m = _finite_positive(mase)
    c0 = _finite_positive(crps_center)
    m0 = _finite_positive(mase_center)
    e = float(eps)
    if not math.isfinite(e) or e < 0.0:
        raise ValueError(f"eps must be finite and nonnegative, got {eps!r}")
    return float(-0.5 * (math.log((c + e) / (c0 + e)) + math.log((m + e) / (m0 + e))))


def seed_mean_metric_rows(rows: Iterable[MetricRow]) -> List[Dict[str, object]]:
    """Aggregate repeated seed rows before reward construction.

    This removes row-order dependence when multiple seeds exist for one schedule.
    """
    grouped: Dict[ScheduleKey, List[MetricRow]] = defaultdict(list)
    for row in rows:
        key = (str(row["solver_key"]), int(row["target_nfe"]), str(row["scheduler_key"]))
        grouped[key].append(row)
    out: List[Dict[str, object]] = []
    for (solver_key, target_nfe, scheduler_key), group in sorted(grouped.items(), key=lambda item: item[0]):
        crps_values = np.asarray([_finite_positive(row["crps"]) for row in group], dtype=np.float64)
        mase_values = np.asarray([_finite_positive(row["mase"]) for row in group], dtype=np.float64)
        seed_values = []
        for row in group:
            raw_seed = row.get("seed", "")
            if str(raw_seed) == "":
                continue
            try:
                seed_values.append(int(raw_seed))
            except (TypeError, ValueError):
                continue
        seed_values = sorted(set(seed_values))
        first = dict(group[0])
        first.update(
            {
                "solver_key": solver_key,
                "target_nfe": int(target_nfe),
                "scheduler_key": scheduler_key,
                "seed": "seed_mean",
                "crps": float(np.mean(crps_values)),
                "mase": float(np.mean(mase_values)),
                "crps_std": float(np.std(crps_values, ddof=1)) if crps_values.size > 1 else 0.0,
                "mase_std": float(np.std(mase_values, ddof=1)) if mase_values.size > 1 else 0.0,
                "n_seeds": int(len(group)),
                "seed_values": seed_values,
            }
        )
        out.append(first)
    return out


def _maybe_seed_mean_metric_rows(rows: Iterable[MetricRow]) -> List[Dict[str, object]]:
    materialized = [dict(row) for row in rows]
    if materialized and all("n_seeds" in row and "seed_values" in row for row in materialized):
        return materialized
    return seed_mean_metric_rows(materialized)


def best_fixed_references_by_setting(
    rows: Iterable[MetricRow],
    *,
    fixed_schedule_keys: Sequence[str],
) -> Dict[SettingKey, Dict[str, float]]:
    """Return best fixed-baseline CRPS/MASE references for each solver/NFE cell."""
    fixed_keys = {str(key) for key in fixed_schedule_keys}
    grouped: Dict[SettingKey, List[MetricRow]] = defaultdict(list)
    for row in _maybe_seed_mean_metric_rows(rows):
        if str(row["scheduler_key"]) in fixed_keys:
            grouped[(str(row["solver_key"]), int(row["target_nfe"]))].append(row)
    references: Dict[SettingKey, Dict[str, float]] = {}
    for setting, setting_rows in grouped.items():
        if not setting_rows:
            continue
        references[setting] = {
            "crps": min(_finite_positive(row["crps"]) for row in setting_rows),
            "mase": min(_finite_positive(row["mase"]) for row in setting_rows),
            "fixed_schedule_count": int(len(setting_rows)),
        }
    return references


def build_fixed_reference_table(
    rows: Iterable[MetricRow],
    *,
    fixed_schedule_keys: Sequence[str],
    uniform_schedule_key: str = UNIFORM_SCHEDULE_KEY,
) -> Dict[SettingKey, Dict[str, object]]:
    """Build per-setting best-fixed and uniform reference metrics.

    Input rows may be seed-level or already seed-aggregated; output values are
    based on seed means.
    """
    fixed_keys = tuple(str(key) for key in fixed_schedule_keys)
    fixed_key_set = set(fixed_keys)
    uniform_key = str(uniform_schedule_key)
    grouped: Dict[SettingKey, List[MetricRow]] = defaultdict(list)
    for row in _maybe_seed_mean_metric_rows(rows):
        if str(row["scheduler_key"]) in fixed_key_set:
            grouped[(str(row["solver_key"]), int(row["target_nfe"]))].append(row)
    references: Dict[SettingKey, Dict[str, object]] = {}
    for setting, setting_rows in grouped.items():
        if not setting_rows:
            continue
        best_crps_row = min(setting_rows, key=lambda row: _finite_positive(row["crps"]))
        best_mase_row = min(setting_rows, key=lambda row: _finite_positive(row["mase"]))
        uniform_rows = [row for row in setting_rows if str(row["scheduler_key"]) == uniform_key]
        uniform_crps = None
        uniform_mase = None
        if uniform_rows:
            uniform_crps = _finite_positive(uniform_rows[0]["crps"])
            uniform_mase = _finite_positive(uniform_rows[0]["mase"])
        references[setting] = {
            "solver_key": setting[0],
            "target_nfe": int(setting[1]),
            "fixed_schedule_keys": list(fixed_keys),
            "fixed_schedule_count": int(len(setting_rows)),
            "best_fixed_crps": _finite_positive(best_crps_row["crps"]),
            "best_fixed_crps_schedule": str(best_crps_row["scheduler_key"]),
            "best_fixed_mase": _finite_positive(best_mase_row["mase"]),
            "best_fixed_mase_schedule": str(best_mase_row["scheduler_key"]),
            "uniform_crps": uniform_crps,
            "uniform_mase": uniform_mase,
            "uniform_schedule_key": uniform_key if uniform_rows else "",
        }
    return references


def reward_columns_for_row(
    row: MetricRow,
    reference: Mapping[str, object],
    *,
    eps: float = DEFAULT_REWARD_EPS,
) -> Dict[str, float | str | int | None]:
    """Return materialized utility/reference columns for one seed-mean row."""
    crps = _finite_positive(row["crps"])
    mase = _finite_positive(row["mase"])
    best_fixed_crps = _finite_positive(reference["best_fixed_crps"])
    best_fixed_mase = _finite_positive(reference["best_fixed_mase"])
    e = float(eps)
    u_crps_best = float(-math.log((crps + e) / (best_fixed_crps + e)))
    u_mase_best = float(-math.log((mase + e) / (best_fixed_mase + e)))
    out: Dict[str, float | str | int | None] = {
        "best_fixed_crps": float(best_fixed_crps),
        "best_fixed_mase": float(best_fixed_mase),
        "best_fixed_crps_schedule": str(reference.get("best_fixed_crps_schedule", "")),
        "best_fixed_mase_schedule": str(reference.get("best_fixed_mase_schedule", "")),
        "uniform_crps": None,
        "uniform_mase": None,
        "u_crps_best": u_crps_best,
        "u_mase_best": u_mase_best,
        "u_comp_best": float(0.5 * (u_crps_best + u_mase_best)),
        "u_crps_uniform": None,
        "u_mase_uniform": None,
        "u_comp_uniform": None,
        "fixed_schedule_count": int(reference.get("fixed_schedule_count", 0) or 0),
    }
    uniform_crps = reference.get("uniform_crps")
    uniform_mase = reference.get("uniform_mase")
    if uniform_crps is not None and uniform_mase is not None:
        uniform_crps_f = _finite_positive(uniform_crps)
        uniform_mase_f = _finite_positive(uniform_mase)
        u_crps_uniform = float(-math.log((crps + e) / (uniform_crps_f + e)))
        u_mase_uniform = float(-math.log((mase + e) / (uniform_mase_f + e)))
        out.update(
            {
                "uniform_crps": float(uniform_crps_f),
                "uniform_mase": float(uniform_mase_f),
                "u_crps_uniform": u_crps_uniform,
                "u_mase_uniform": u_mase_uniform,
                "u_comp_uniform": float(0.5 * (u_crps_uniform + u_mase_uniform)),
            }
        )
    return out


def attach_reward_columns(
    rows: Iterable[MetricRow],
    *,
    fixed_schedule_keys: Sequence[str],
    eps: float = DEFAULT_REWARD_EPS,
) -> List[Dict[str, object]]:
    """Attach best-fixed and uniform utility columns to seed-mean rows."""
    seed_mean_rows = _maybe_seed_mean_metric_rows(rows)
    references = build_fixed_reference_table(seed_mean_rows, fixed_schedule_keys=fixed_schedule_keys)
    out: List[Dict[str, object]] = []
    for row in seed_mean_rows:
        setting = (str(row["solver_key"]), int(row["target_nfe"]))
        if setting not in references:
            raise ValueError(
                "Reward columns require fixed baseline references "
                f"for setting {setting}; none were available."
            )
        copied = dict(row)
        copied.update(reward_columns_for_row(copied, references[setting], eps=float(eps)))
        out.append(copied)
    return out


def rewards_by_setting(
    rows: Iterable[MetricRow],
    *,
    fixed_schedule_keys: Sequence[str] | None = None,
    eps: float = DEFAULT_REWARD_EPS,
) -> Dict[SettingKey, Dict[str, float]]:
    if fixed_schedule_keys is None:
        from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS

        fixed_schedule_keys = BASELINE_SCHEDULE_KEYS
    grouped: Dict[SettingKey, List[MetricRow]] = defaultdict(list)
    seed_mean_rows = _maybe_seed_mean_metric_rows(rows)
    references = build_fixed_reference_table(seed_mean_rows, fixed_schedule_keys=fixed_schedule_keys)
    for row in seed_mean_rows:
        grouped[(str(row["solver_key"]), int(row["target_nfe"]))].append(row)
    rewards: Dict[SettingKey, Dict[str, float]] = {}
    for setting, setting_rows in grouped.items():
        if setting not in references:
            raise ValueError(
                "Best-fixed-baseline reward scaling requires fixed baseline rows "
                f"for setting {setting}; none were available."
            )
        reference = references[setting]
        rewards[setting] = {
            str(row["scheduler_key"]): crps_mase_reward(
                float(row["crps"]),
                float(row["mase"]),
                crps_center=float(reference["best_fixed_crps"]),
                mase_center=float(reference["best_fixed_mase"]),
                eps=float(eps),
            )
            for row in setting_rows
        }
    return rewards

def best_schedule_by_setting(rows: Iterable[MetricRow]) -> Dict[SettingKey, str]:
    rewards = rewards_by_setting(rows)
    return {
        setting: max(schedule_rewards.items(), key=lambda item: float(item[1]))[0]
        for setting, schedule_rewards in rewards.items()
    }
