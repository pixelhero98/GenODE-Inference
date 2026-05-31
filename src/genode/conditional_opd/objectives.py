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

    ``crps_center`` and ``mase_center`` are reference metrics. In the Train20 OPD
    protocol they are the best fixed-baseline metrics for the same solver/NFE
    cell, not medians over generated candidates.
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

    V4.1 uses best fixed CRPS/MASE as the optimization reference and keeps the
    uniform comparison as a diagnostic. Input rows may be seed-level or already
    seed-aggregated; output values are based on seed means.
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
    """Return materialized V4.1 utility/reference columns for one seed-mean row."""
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
    """Attach V4.1 best-fixed and uniform utility columns to seed-mean rows."""
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



def _normalized_source_weights(
    source_names: Sequence[str],
    source_weights: Mapping[str, float] | None,
) -> Dict[str, float]:
    if not source_names:
        raise ValueError("At least one calibration source is required.")
    if source_weights is None:
        weight = 1.0 / float(len(source_names))
        return {str(name): float(weight) for name in source_names}
    weights: Dict[str, float] = {}
    for name in source_names:
        if str(name) not in source_weights:
            raise ValueError(f"Missing calibration source weight for {name!r}.")
        value = float(source_weights[str(name)])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Calibration source weights must be finite and non-negative, got {name!r}={value!r}.")
        weights[str(name)] = value
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("Calibration source weights must have positive total mass.")
    return {name: float(value / total) for name, value in weights.items()}


def source_balanced_rewards_by_setting(
    source_rows: Mapping[str, Iterable[MetricRow]],
    *,
    fixed_schedule_keys: Sequence[str] | None = None,
    source_weights: Mapping[str, float] | None = None,
    eps: float = DEFAULT_REWARD_EPS,
) -> Dict[SettingKey, Dict[str, float]]:
    """Compute V4.2-F source-balanced calibration rewards.

    Rewards are computed independently inside each calibration source, using that
    source's best fixed-baseline CRPS/MASE reference. The final utility is the
    weighted mean of those source-local rewards. This deliberately avoids
    pooling train and former-validation rows before reward construction.
    """
    materialized: Dict[str, List[Dict[str, object]]] = {
        str(name): seed_mean_metric_rows(rows) for name, rows in source_rows.items()
    }
    materialized = {name: rows for name, rows in materialized.items() if rows}
    source_names = sorted(materialized)
    weights = _normalized_source_weights(source_names, source_weights)
    per_source_rewards: Dict[str, Dict[SettingKey, Dict[str, float]]] = {
        name: rewards_by_setting(rows, fixed_schedule_keys=fixed_schedule_keys, eps=float(eps))
        for name, rows in materialized.items()
    }
    settings = sorted(
        {setting for source_reward in per_source_rewards.values() for setting in source_reward},
        key=lambda item: (item[0], item[1]),
    )
    out: Dict[SettingKey, Dict[str, float]] = {}
    for setting in settings:
        schedule_keys = sorted(
            {key for source_reward in per_source_rewards.values() for key in source_reward.get(setting, {})}
        )
        if not schedule_keys:
            continue
        out[setting] = {}
        for schedule_key in schedule_keys:
            missing = [
                name
                for name in source_names
                if setting not in per_source_rewards[name] or schedule_key not in per_source_rewards[name][setting]
            ]
            if missing:
                raise ValueError(
                    "Source-balanced calibration reward requires every schedule to be labeled in every source; "
                    f"missing schedule={schedule_key!r}, setting={setting!r}, sources={missing}."
                )
            out[setting][schedule_key] = float(
                sum(weights[name] * float(per_source_rewards[name][setting][schedule_key]) for name in source_names)
            )
    return out


def source_balanced_seed_mean_rows(
    source_rows: Mapping[str, Iterable[MetricRow]],
    *,
    source_weights: Mapping[str, float] | None = None,
) -> List[Dict[str, object]]:
    """Return one weighted seed-mean metric row per schedule/cell."""
    materialized: Dict[str, List[Dict[str, object]]] = {
        str(name): seed_mean_metric_rows(rows) for name, rows in source_rows.items()
    }
    materialized = {name: rows for name, rows in materialized.items() if rows}
    source_names = sorted(materialized)
    weights = _normalized_source_weights(source_names, source_weights)
    by_key: Dict[ScheduleKey, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for source_name, rows in materialized.items():
        for row in rows:
            key = (str(row["solver_key"]), int(row["target_nfe"]), str(row["scheduler_key"]))
            by_key[key][source_name] = dict(row)
    out: List[Dict[str, object]] = []
    for (solver_key, target_nfe, scheduler_key), rows_by_source in sorted(by_key.items(), key=lambda item: item[0]):
        missing = [name for name in source_names if name not in rows_by_source]
        if missing:
            raise ValueError(
                "Source-balanced seed means require every schedule to be labeled in every source; "
                f"missing schedule={scheduler_key!r}, setting={(solver_key, target_nfe)!r}, sources={missing}."
            )
        first = dict(rows_by_source[source_names[0]])
        crps = sum(weights[name] * _finite_positive(rows_by_source[name]["crps"]) for name in source_names)
        mase = sum(weights[name] * _finite_positive(rows_by_source[name]["mase"]) for name in source_names)
        seed_values = sorted(
            {
                int(seed)
                for name in source_names
                for seed in list(rows_by_source[name].get("seed_values", []) or [])
                if str(seed) != ""
            }
        )
        first.update(
            {
                "solver_key": solver_key,
                "target_nfe": int(target_nfe),
                "scheduler_key": scheduler_key,
                "seed": "source_balanced_seed_mean",
                "crps": float(crps),
                "mase": float(mase),
                "calibration_sources": list(source_names),
                "calibration_source_weights": {name: float(weights[name]) for name in source_names},
                "calibration_source_count": int(len(source_names)),
                "seed_values": seed_values,
                "n_seeds": int(sum(int(rows_by_source[name].get("n_seeds", 1) or 1) for name in source_names)),
            }
        )
        out.append(first)
    return out


def best_schedule_by_setting(rows: Iterable[MetricRow]) -> Dict[SettingKey, str]:
    rewards = rewards_by_setting(rows)
    return {
        setting: max(schedule_rewards.items(), key=lambda item: float(item[1]))[0]
        for setting, schedule_rewards in rewards.items()
    }
