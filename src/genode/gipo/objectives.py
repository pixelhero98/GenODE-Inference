from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from genode.experiment_layout import (
    SCENARIO_FAMILY_CONDITIONAL_GENERATION,
    SCENARIO_FAMILY_FORECAST,
    SCENARIO_FAMILY_MOLECULE,
    scenario_family_for_key,
)

MetricRow = Mapping[str, object]
SettingKey = Tuple[str, int]
ScheduleKey = Tuple[str, int, str]
DEFAULT_REWARD_EPS = 1e-12
UNIFORM_SCHEDULE_KEY = "uniform"
METRIC_DIRECTION_LOWER = "lower"
METRIC_DIRECTION_HIGHER = "higher"


@dataclass(frozen=True)
class MetricObjectiveSpec:
    metric_key: str
    utility_key: str
    direction: str
    weight: float = 1.0
    applicable_key: str = ""


FORECAST_METRIC_SPECS: Tuple[MetricObjectiveSpec, ...] = (
    MetricObjectiveSpec("forecast_crps", "u_crps_uniform", METRIC_DIRECTION_LOWER, 0.5),
    MetricObjectiveSpec("forecast_mase", "u_mase_uniform", METRIC_DIRECTION_LOWER, 0.5),
)
CONDITIONAL_PRIMARY_LOB_METRIC_SPECS: Tuple[MetricObjectiveSpec, ...] = (
    MetricObjectiveSpec("temporal_uw1", "u_temporal_uw1_uniform", METRIC_DIRECTION_LOWER, 1.0 / 3.0),
    MetricObjectiveSpec("temporal_cw1", "u_temporal_cw1_uniform", METRIC_DIRECTION_LOWER, 1.0 / 3.0),
    MetricObjectiveSpec(
        "temporal_tstr_f1",
        "u_temporal_tstr_f1_uniform",
        METRIC_DIRECTION_HIGHER,
        1.0 / 3.0,
        applicable_key="temporal_tstr_f1_applicable",
    ),
)
CONDITIONAL_PRIMARY_ECG_METRIC_SPECS: Tuple[MetricObjectiveSpec, ...] = (
    MetricObjectiveSpec("temporal_uw1", "u_temporal_uw1_uniform", METRIC_DIRECTION_LOWER, 0.5),
    MetricObjectiveSpec("temporal_cw1", "u_temporal_cw1_uniform", METRIC_DIRECTION_LOWER, 0.5),
)
CONDITIONAL_DIAGNOSTIC_METRIC_SPECS: Tuple[MetricObjectiveSpec, ...] = (
    MetricObjectiveSpec("u_l1", "u_temporal_u_l1_uniform", METRIC_DIRECTION_LOWER, 0.0),
    MetricObjectiveSpec("c_l1", "u_temporal_c_l1_uniform", METRIC_DIRECTION_LOWER, 0.0),
    MetricObjectiveSpec("spread_specific_error", "u_temporal_spread_specific_error_uniform", METRIC_DIRECTION_LOWER, 0.0),
    MetricObjectiveSpec("imbalance_specific_error", "u_temporal_imbalance_specific_error_uniform", METRIC_DIRECTION_LOWER, 0.0),
    MetricObjectiveSpec("ret_vol_acf_error", "u_temporal_ret_vol_acf_error_uniform", METRIC_DIRECTION_LOWER, 0.0),
    MetricObjectiveSpec("impact_response_error", "u_temporal_impact_response_error_uniform", METRIC_DIRECTION_LOWER, 0.0),
)
CONDITIONAL_METRIC_SPECS: Tuple[MetricObjectiveSpec, ...] = (
    *CONDITIONAL_PRIMARY_LOB_METRIC_SPECS,
    *CONDITIONAL_DIAGNOSTIC_METRIC_SPECS,
)
MOLECULE_METRIC_SPECS: Tuple[MetricObjectiveSpec, ...] = (
    MetricObjectiveSpec("molecule_kabsch_rmsd_3d", "u_molecule_kabsch_rmsd_3d_uniform", METRIC_DIRECTION_LOWER, 0.40),
    MetricObjectiveSpec("molecule_ensemble_velocity_norm_w1", "u_molecule_ensemble_velocity_norm_w1_uniform", METRIC_DIRECTION_LOWER, 0.15),
    MetricObjectiveSpec("molecule_ensemble_acceleration_norm_w1", "u_molecule_ensemble_acceleration_norm_w1_uniform", METRIC_DIRECTION_LOWER, 0.15),
    MetricObjectiveSpec("molecule_rollout_velocity_norm_w1", "u_molecule_rollout_velocity_norm_w1_uniform", METRIC_DIRECTION_LOWER, 0.15),
    MetricObjectiveSpec("molecule_rollout_acceleration_norm_w1", "u_molecule_rollout_acceleration_norm_w1_uniform", METRIC_DIRECTION_LOWER, 0.15),
)
OBJECTIVE_SPECS_BY_FAMILY: Dict[str, Tuple[MetricObjectiveSpec, ...]] = {
    SCENARIO_FAMILY_FORECAST: FORECAST_METRIC_SPECS,
    SCENARIO_FAMILY_CONDITIONAL_GENERATION: CONDITIONAL_METRIC_SPECS,
    SCENARIO_FAMILY_MOLECULE: MOLECULE_METRIC_SPECS,
}


def objective_specs_for_family(benchmark_family: str) -> Tuple[MetricObjectiveSpec, ...]:
    family = str(benchmark_family).strip()
    if family not in OBJECTIVE_SPECS_BY_FAMILY:
        raise ValueError(f"Unsupported benchmark_family for GIPO teacher targets: {benchmark_family!r}")
    return OBJECTIVE_SPECS_BY_FAMILY[family]


def teacher_objective_specs_for_family(benchmark_family: str) -> Tuple[MetricObjectiveSpec, ...]:
    family = str(benchmark_family).strip()
    if family == SCENARIO_FAMILY_CONDITIONAL_GENERATION:
        return CONDITIONAL_PRIMARY_LOB_METRIC_SPECS
    return objective_specs_for_family(family)


def teacher_objective_specs_for_scenario(scenario_key: str) -> Tuple[MetricObjectiveSpec, ...]:
    key = str(scenario_key).strip()
    family = scenario_family_for_key(key)
    if family == SCENARIO_FAMILY_CONDITIONAL_GENERATION:
        if key == "long_term_st":
            return CONDITIONAL_PRIMARY_ECG_METRIC_SPECS
        if key in {"cryptos", "lobster_synthetic"}:
            return CONDITIONAL_PRIMARY_LOB_METRIC_SPECS
        raise ValueError(f"Unsupported conditional-generation scenario for GIPO teacher targets: {scenario_key!r}")
    return objective_specs_for_family(family)


def objective_utility_keys_for_family(benchmark_family: str) -> Tuple[str, ...]:
    return tuple(spec.utility_key for spec in objective_specs_for_family(benchmark_family))


def teacher_objective_utility_keys_for_family(benchmark_family: str) -> Tuple[str, ...]:
    return tuple(spec.utility_key for spec in teacher_objective_specs_for_family(benchmark_family))


def teacher_objective_utility_keys_for_scenario(scenario_key: str) -> Tuple[str, ...]:
    return tuple(spec.utility_key for spec in teacher_objective_specs_for_scenario(scenario_key))


def teacher_metric_profile_for_scenario(scenario_key: str) -> Dict[str, object]:
    specs = teacher_objective_specs_for_scenario(str(scenario_key))
    diagnostic_specs: Tuple[MetricObjectiveSpec, ...] = ()
    if scenario_family_for_key(str(scenario_key)) == SCENARIO_FAMILY_CONDITIONAL_GENERATION:
        diagnostic_specs = CONDITIONAL_DIAGNOSTIC_METRIC_SPECS
    return {
        "scenario_key": str(scenario_key),
        "benchmark_family": scenario_family_for_key(str(scenario_key)),
        "target_metric_keys": [spec.metric_key for spec in specs],
        "target_utility_keys": [spec.utility_key for spec in specs],
        "target_weights": {spec.utility_key: float(spec.weight) for spec in specs},
        "diagnostic_metric_keys": [spec.metric_key for spec in diagnostic_specs],
        "diagnostic_utility_keys": [spec.utility_key for spec in diagnostic_specs],
    }


def objective_weight_map_for_keys(target_keys: Sequence[str]) -> Dict[str, float]:
    spec_by_utility = {
        spec.utility_key: spec
        for specs in OBJECTIVE_SPECS_BY_FAMILY.values()
        for spec in specs
    }
    return {str(key): float(spec_by_utility[str(key)].weight) for key in target_keys if str(key) in spec_by_utility}


def _finite_positive(value: object) -> float:
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"Metric values must be finite and positive, got {value!r}")
    return val


def _finite_nonnegative(value: object) -> float:
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"Metric values must be finite and nonnegative, got {value!r}")
    return val


def _optional_finite(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _metric_value(row: Mapping[str, object], spec: MetricObjectiveSpec) -> float | None:
    return _optional_finite(row.get(spec.metric_key))


def _metric_applicable(row: Mapping[str, object], spec: MetricObjectiveSpec) -> bool:
    if not spec.applicable_key:
        return True
    value = row.get(spec.applicable_key)
    if value in (None, ""):
        return False
    return _truthy(value)


def directional_uniform_utility(
    candidate_value: object,
    uniform_value: object,
    *,
    direction: str,
    eps: float = DEFAULT_REWARD_EPS,
) -> float:
    candidate = _finite_nonnegative(candidate_value)
    uniform = _finite_nonnegative(uniform_value)
    e = float(eps)
    if not math.isfinite(e) or e < 0.0:
        raise ValueError(f"eps must be finite and nonnegative, got {eps!r}")
    if str(direction) == METRIC_DIRECTION_LOWER:
        return float(math.log(uniform + e) - math.log(candidate + e))
    if str(direction) == METRIC_DIRECTION_HIGHER:
        return float(math.log(candidate + e) - math.log(uniform + e))
    raise ValueError(f"Unknown metric direction={direction!r}.")


def uniform_anchored_objective_columns(
    row: MetricRow,
    uniform_row: MetricRow,
    specs: Sequence[MetricObjectiveSpec],
    *,
    uniform_scheduler_key: str = UNIFORM_SCHEDULE_KEY,
    eps: float = DEFAULT_REWARD_EPS,
) -> Dict[str, object]:
    schedule_key = str(row.get("scheduler_key", ""))
    is_uniform = schedule_key == str(uniform_scheduler_key)
    utilities: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    metric_directions: Dict[str, str] = {}
    references: Dict[str, float] = {}
    out: Dict[str, object] = {
        "u_comp_uniform": None,
        "reward_metric_count": 0,
        "reward_metric_weights_json": "{}",
        "reward_metric_directions_json": "{}",
    }
    for spec in specs:
        if not _metric_applicable(row, spec) or not _metric_applicable(uniform_row, spec):
            out[spec.utility_key] = None
            continue
        candidate = _metric_value(row, spec)
        reference = _metric_value(uniform_row, spec)
        if candidate is None or reference is None:
            out[spec.utility_key] = None
            continue
        utility = 0.0 if is_uniform else directional_uniform_utility(
            candidate,
            reference,
            direction=spec.direction,
            eps=float(eps),
        )
        utilities[spec.utility_key] = float(utility)
        weights[spec.utility_key] = float(spec.weight)
        metric_directions[spec.metric_key] = str(spec.direction)
        references[f"uniform_{spec.metric_key}"] = float(reference)
        out[spec.utility_key] = float(utility)
        out[f"uniform_{spec.metric_key}"] = float(reference)
    if utilities:
        total_weight = float(sum(weight for weight in weights.values() if math.isfinite(weight) and weight > 0.0))
        if total_weight <= 0.0:
            raise ValueError("At least one applicable reward metric must have a positive weight.")
        normalized = {key: float(weight / total_weight) for key, weight in weights.items() if weight > 0.0}
        out["u_comp_uniform"] = float(sum(float(utilities[key]) * float(normalized[key]) for key in normalized))
        out["reward_metric_count"] = int(len(normalized))
        out["reward_metric_weights_json"] = json_dumps_stable(normalized)
        out["reward_metric_directions_json"] = json_dumps_stable(metric_directions)
    return out


def json_dumps_stable(payload: Mapping[str, object]) -> str:
    import json

    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


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
    u_crps = directional_uniform_utility(crps, crps_center, direction=METRIC_DIRECTION_LOWER, eps=float(eps))
    u_mase = directional_uniform_utility(mase, mase_center, direction=METRIC_DIRECTION_LOWER, eps=float(eps))
    return float(0.5 * (u_crps + u_mase))


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
        crps_values = np.asarray([_finite_positive(row["forecast_crps"]) for row in group], dtype=np.float64)
        mase_values = np.asarray([_finite_positive(row["forecast_mase"]) for row in group], dtype=np.float64)
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
                "forecast_crps": float(np.mean(crps_values)),
                "forecast_mase": float(np.mean(mase_values)),
                "forecast_crps_std": float(np.std(crps_values, ddof=1)) if crps_values.size > 1 else 0.0,
                "forecast_mase_std": float(np.std(mase_values, ddof=1)) if mase_values.size > 1 else 0.0,
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


def build_fixed_reference_table(
    rows: Iterable[MetricRow],
    *,
    fixed_scheduler_keys: Sequence[str],
    uniform_scheduler_key: str = UNIFORM_SCHEDULE_KEY,
) -> Dict[SettingKey, Dict[str, object]]:
    """Build per-setting best-fixed and uniform reference metrics.

    Input rows may be seed-level or already seed-aggregated; output values are
    based on seed means.
    """
    fixed_keys = tuple(str(key) for key in fixed_scheduler_keys)
    fixed_key_set = set(fixed_keys)
    uniform_key = str(uniform_scheduler_key)
    grouped: Dict[SettingKey, List[MetricRow]] = defaultdict(list)
    for row in _maybe_seed_mean_metric_rows(rows):
        if str(row["scheduler_key"]) in fixed_key_set:
            grouped[(str(row["solver_key"]), int(row["target_nfe"]))].append(row)
    references: Dict[SettingKey, Dict[str, object]] = {}
    for setting, setting_rows in grouped.items():
        if not setting_rows:
            continue
        best_crps_row = min(setting_rows, key=lambda row: _finite_positive(row["forecast_crps"]))
        best_mase_row = min(setting_rows, key=lambda row: _finite_positive(row["forecast_mase"]))
        uniform_rows = [row for row in setting_rows if str(row["scheduler_key"]) == uniform_key]
        uniform_crps = None
        uniform_mase = None
        if uniform_rows:
            uniform_crps = _finite_positive(uniform_rows[0]["forecast_crps"])
            uniform_mase = _finite_positive(uniform_rows[0]["forecast_mase"])
        references[setting] = {
            "solver_key": setting[0],
            "target_nfe": int(setting[1]),
            "fixed_scheduler_keys": list(fixed_keys),
            "fixed_scheduler_count": int(len(setting_rows)),
            "best_fixed_crps": _finite_positive(best_crps_row["forecast_crps"]),
            "best_fixed_crps_scheduler_key": str(best_crps_row["scheduler_key"]),
            "best_fixed_mase": _finite_positive(best_mase_row["forecast_mase"]),
            "best_fixed_mase_scheduler_key": str(best_mase_row["scheduler_key"]),
            "uniform_crps": uniform_crps,
            "uniform_mase": uniform_mase,
            "uniform_scheduler_key": uniform_key if uniform_rows else "",
        }
    return references


def reward_columns_for_row(
    row: MetricRow,
    reference: Mapping[str, object],
    *,
    eps: float = DEFAULT_REWARD_EPS,
) -> Dict[str, float | str | int | None]:
    """Return materialized utility/reference columns for one seed-mean row."""
    crps = _finite_positive(row["forecast_crps"])
    mase = _finite_positive(row["forecast_mase"])
    best_fixed_crps = _finite_positive(reference["best_fixed_crps"])
    best_fixed_mase = _finite_positive(reference["best_fixed_mase"])
    e = float(eps)
    u_crps_best = directional_uniform_utility(crps, best_fixed_crps, direction=METRIC_DIRECTION_LOWER, eps=e)
    u_mase_best = directional_uniform_utility(mase, best_fixed_mase, direction=METRIC_DIRECTION_LOWER, eps=e)
    out: Dict[str, float | str | int | None] = {
        "best_fixed_crps": float(best_fixed_crps),
        "best_fixed_mase": float(best_fixed_mase),
        "best_fixed_crps_scheduler_key": str(reference.get("best_fixed_crps_scheduler_key", "")),
        "best_fixed_mase_scheduler_key": str(reference.get("best_fixed_mase_scheduler_key", "")),
        "uniform_crps": None,
        "uniform_mase": None,
        "u_crps_best": u_crps_best,
        "u_mase_best": u_mase_best,
        "u_comp_best": float(0.5 * (u_crps_best + u_mase_best)),
        "u_crps_uniform": None,
        "u_mase_uniform": None,
        "u_comp_uniform": None,
        "fixed_scheduler_count": int(reference.get("fixed_scheduler_count", 0) or 0),
    }
    uniform_crps = reference.get("uniform_crps")
    uniform_mase = reference.get("uniform_mase")
    if uniform_crps is not None and uniform_mase is not None:
        uniform_crps_f = _finite_positive(uniform_crps)
        uniform_mase_f = _finite_positive(uniform_mase)
        generic = uniform_anchored_objective_columns(
            row,
            {
                "scheduler_key": str(reference.get("uniform_scheduler_key", UNIFORM_SCHEDULE_KEY)),
                "forecast_crps": uniform_crps_f,
                "forecast_mase": uniform_mase_f,
            },
            FORECAST_METRIC_SPECS,
            uniform_scheduler_key=str(reference.get("uniform_scheduler_key", UNIFORM_SCHEDULE_KEY)),
            eps=e,
        )
        u_crps_uniform = float(generic["u_crps_uniform"])
        u_mase_uniform = float(generic["u_mase_uniform"])
        out.update(
            {
                "uniform_crps": float(uniform_crps_f),
                "uniform_mase": float(uniform_mase_f),
                "u_crps_uniform": u_crps_uniform,
                "u_mase_uniform": u_mase_uniform,
                "u_comp_uniform": float(generic["u_comp_uniform"]),
                "reward_metric_count": generic.get("reward_metric_count"),
                "reward_metric_weights_json": generic.get("reward_metric_weights_json"),
                "reward_metric_directions_json": generic.get("reward_metric_directions_json"),
            }
        )
    return out


def attach_reward_columns(
    rows: Iterable[MetricRow],
    *,
    fixed_scheduler_keys: Sequence[str],
    eps: float = DEFAULT_REWARD_EPS,
) -> List[Dict[str, object]]:
    """Attach best-fixed and uniform utility columns to seed-mean rows."""
    seed_mean_rows = _maybe_seed_mean_metric_rows(rows)
    references = build_fixed_reference_table(seed_mean_rows, fixed_scheduler_keys=fixed_scheduler_keys)
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
    fixed_scheduler_keys: Sequence[str] | None = None,
    eps: float = DEFAULT_REWARD_EPS,
) -> Dict[SettingKey, Dict[str, float]]:
    if fixed_scheduler_keys is None:
        from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS

        fixed_scheduler_keys = BASELINE_SCHEDULE_KEYS
    grouped: Dict[SettingKey, List[MetricRow]] = defaultdict(list)
    seed_mean_rows = _maybe_seed_mean_metric_rows(rows)
    references = build_fixed_reference_table(seed_mean_rows, fixed_scheduler_keys=fixed_scheduler_keys)
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
                float(row["forecast_crps"]),
                float(row["forecast_mase"]),
                crps_center=float(reference["best_fixed_crps"]),
                mase_center=float(reference["best_fixed_mase"]),
                eps=float(eps),
            )
            for row in setting_rows
        }
    return rewards
