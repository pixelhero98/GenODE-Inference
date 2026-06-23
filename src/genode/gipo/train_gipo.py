from __future__ import annotations

import argparse
import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from genode.canonical_experiment_layout import (
    CANONICAL_CONTEXT_SAMPLE_COUNT,
    CANONICAL_PSEUDO_TARGET_WEIGHT,
    CANONICAL_SEEN_NFES,
    CANONICAL_UNSEEN_NFES,
    SCENARIO_FAMILY_CONDITIONAL_GENERATION,
    SCENARIO_FAMILY_FORECAST,
    SCENARIO_FAMILY_MOLECULE,
    STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
    STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO,
    scenario_family_for_key,
)
from genode.gipo.objectives import (
    objective_utility_keys_for_family,
    teacher_objective_utility_keys_for_family,
    teacher_objective_utility_keys_for_scenario,
)
from genode.gipo.policy import (
    GIPO_PROTOCOL,
    MODEL_PAYLOAD_VERSION,
    SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
    DEFAULT_SUPPORT_SCHEDULE_KEYS,
    DEFAULT_TEACHER_TARGET_TEMPERATURE,
    DEFAULT_STUDENT_TARGET_ELITE_BLEND_ALL_WEIGHT,
    DEFAULT_STUDENT_TARGET_ELITE_FRACTION,
    DEFAULT_STUDENT_TARGET_ELITE_K,
    DEFAULT_STUDENT_TARGET_ELITE_MIN_COUNT,
    DEFAULT_STUDENT_TARGET_MIXTURE_MODE,
    DEFAULT_STUDENT_TEACHER_SCORE_CLIP,
    DEFAULT_STUDENT_TEACHER_SCORE_WEIGHT,
    DEFAULT_STUDENT_TEACHER_SCORE_WARMUP_FRACTION,
    STUDENT_TARGET_PROTOCOL_SOFT_MIXTURE,
    STUDENT_TARGET_MIXTURE_MODES,
    TEACHER_METRIC_MASK_PROTOCOL,
    TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
    TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
    DEFAULT_DENSITY_FAMILY_HOLDOUT_SCHEDULE_KEYS,
    TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET,
    STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE,
    ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
    ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
    DEFAULT_TRANSFORMER_DROPOUT,
    DEFAULT_TRANSFORMER_HEADS,
    DEFAULT_TRANSFORMER_HIDDEN_DIM,
    DEFAULT_TRANSFORMER_LAYERS,
    CONDITIONING_STYLE_ADDITIVE_MLP,
    DensityFeatureNormalizer,
    EmbeddingNormalizer,
    attach_uniform_gipo_rewards,
    build_gipo_student_model,
    build_gipo_teacher_model,
    build_series_index_map,
    checkpoint_scope_from_row,
    context_embedding_id_from_row,
    context_id_from_row,
    context_pair_key,
    density_mass_for_row,
    load_context_embedding_table,
    nfe_sequence_diagnostic_summary,
    read_metric_rows_csv,
    recommended_context_calibration_count,
    sample_context_ids_stratified,
    select_weighted_normalized_regret_checkpoint,
    series_key_from_row,
    split_rows_by_context_holdout,
    split_rows_by_density_family_holdout,
    teacher_rank_pair_diagnostics,
    train_gipo_student,
    train_gipo_teacher,
    normalize_teacher_utility_weights,
    teacher_utility_weights_for_summary,
    validate_gipo_attention_heads,
    validate_gipo_support_schedule_keys,
    validate_student_target_mixture_mode,
    validate_teacher_metric_target_keys,
    validate_teacher_objective_hyperparameters,
)
from genode.gipo.density_representation import density_metadata, reference_grid_hash, uniform_reference_grid
from genode.gipo.preflight import (
    build_gipo_support_preflight_report,
    teacher_metric_target_coverage,
    validate_gipo_support_preflight_report,
)
from genode.gipo.schedule_grids import load_schedule_summary_grids, validate_schedule_grid_coverage
from genode.gipo.models import (
    SETTING_ENCODER_MODE_CONTINUOUS_V3,
    setting_encoder_config_for_rows,
    setting_feature_dim,
    validate_setting_feature_mode,
)
from genode.data.otflow_paths import resolve_project_path
from genode.evaluation.otflow_evaluation_support import FORECAST_FAMILY
from genode.models.otflow_train_val import seed_all
from genode.runtime import resolve_torch_device
from genode.solver_protocol import normalize_solver_key

CANONICAL_DENSITY_BIN_COUNT = 64
CANONICAL_TEACHER_SELECTION_COMPONENT_WEIGHTS: Dict[str, float] = {
    "context": 0.25,
    "density_family": 0.25,
    "unseen_nfe": 0.50,
}
TEACHER_SELECTION_AXIS_ORDER: Tuple[str, ...] = ("context", "density_family", "unseen_nfe")
GIPO_PREPROCESSING_PROTOCOL = "selector_final_normalizer_scopes_v1"
GIPO_TEACHER_TARGET_COVERAGE_PROTOCOL = "strict_per_metric_finite_coverage_v1"
GIPO_TEACHER_SELECTION_WEIGHT_PROTOCOL = "nominal_effective_axis_weights_v1"
CANONICAL_TEACHER_RANK_TEMPERATURE = 0.5
CANONICAL_TEACHER_REGRESSION_WEIGHT = 0.25
CANONICAL_TEACHER_PAIR_MARGIN = 0.0


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _read_metric_rows_csvs(paths_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path_text in _parse_csv(str(paths_text)):
        rows.extend(read_metric_rows_csv(resolve_project_path(path_text)))
    return rows


def _load_context_embedding_tables(paths_text: str) -> Dict[str, np.ndarray]:
    merged: Dict[str, np.ndarray] = {}
    for path_text in _parse_csv(str(paths_text)):
        table = load_context_embedding_table(resolve_project_path(path_text))
        _merge_embedding_tables_guarded(merged, table, label="context_embeddings")
    return merged


def _artifact_input_summary(paths_text: str) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for path_text in _parse_csv(str(paths_text)):
        path = resolve_project_path(path_text)
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
        item: Dict[str, Any] = {"name": path.name, "path_hash": digest, "exists": bool(path.exists())}
        if path.exists() and path.is_file():
            stat = path.stat()
            item.update({"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        summary.append(item)
    return summary


def _parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in _parse_csv(text)]


def _parse_int_csv_or_default(text: Any, default: Sequence[int]) -> List[int]:
    parsed = _parse_int_csv(str(text))
    return parsed if parsed else [int(value) for value in default]


def _parse_float_mapping(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in _parse_csv(text):
        if "=" not in part:
            raise ValueError("teacher_utility_weights entries must be name=value pairs.")
        key, value = part.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("teacher_utility_weights contains an empty metric name.")
        out[key] = float(value)
    return out


def _rows_for_target_nfes(rows: Sequence[Mapping[str, Any]], target_nfes: Sequence[int]) -> List[Dict[str, Any]]:
    allowed = {int(value) for value in target_nfes}
    return [dict(row) for row in rows if int(row["target_nfe"]) in allowed]


def _has_nonempty_value(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, None)
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def _infer_single_benchmark_family(rows: Sequence[Mapping[str, Any]]) -> str:
    families = {str(row.get("benchmark_family", "")).strip() for row in rows if str(row.get("benchmark_family", "")).strip()}
    if not families:
        dataset_families = set()
        for row in rows:
            dataset = str(row.get("dataset", row.get("dataset_key", ""))).strip()
            if not dataset:
                continue
            try:
                dataset_families.add(scenario_family_for_key(dataset))
            except (KeyError, ValueError):
                pass
        families = dataset_families
    if not families:
        if _rows_have_forecast_metrics(rows):
            return SCENARIO_FAMILY_FORECAST
        if any(_has_nonempty_value(row, key) for row in rows for key in objective_utility_keys_for_family(SCENARIO_FAMILY_CONDITIONAL_GENERATION)):
            return SCENARIO_FAMILY_CONDITIONAL_GENERATION
        if any(_has_nonempty_value(row, key) for row in rows for key in objective_utility_keys_for_family(SCENARIO_FAMILY_MOLECULE)):
            return SCENARIO_FAMILY_MOLECULE
        raise ValueError("Cannot infer benchmark_family for automatic GIPO teacher target selection.")
    if len(families) != 1:
        raise ValueError(f"GIPO training rows must contain exactly one benchmark_family; found {sorted(families)}.")
    family = next(iter(families))
    if family not in {SCENARIO_FAMILY_FORECAST, SCENARIO_FAMILY_CONDITIONAL_GENERATION, SCENARIO_FAMILY_MOLECULE}:
        raise ValueError(f"Unsupported benchmark_family for GIPO teacher target selection: {family!r}.")
    return family


def _infer_single_dataset_key(rows: Sequence[Mapping[str, Any]]) -> str:
    datasets = {
        str(row.get("scenario_key") or row.get("dataset") or row.get("dataset_key") or "").strip()
        for row in rows
        if str(row.get("scenario_key") or row.get("dataset") or row.get("dataset_key") or "").strip()
    }
    if len(datasets) > 1:
        raise ValueError(f"GIPO training rows must contain exactly one dataset/scenario key; found {sorted(datasets)}.")
    return next(iter(datasets)) if datasets else ""


def _resolve_teacher_metric_target_keys(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    raw = str(args.teacher_metric_target_keys).strip()
    if not raw or raw.lower() == "auto":
        dataset_key = _infer_single_dataset_key(rows)
        if dataset_key:
            try:
                return teacher_objective_utility_keys_for_scenario(dataset_key)
            except (KeyError, ValueError):
                pass
        return teacher_objective_utility_keys_for_family(_infer_single_benchmark_family(rows))
    return validate_teacher_metric_target_keys(raw)


def _rows_are_forecast_family(rows: Sequence[Mapping[str, Any]]) -> bool:
    families = {str(row.get("benchmark_family", "")).strip() for row in rows if str(row.get("benchmark_family", "")).strip()}
    return not families or families == {FORECAST_FAMILY}


def _rows_have_forecast_metrics(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        _has_nonempty_value(row, "forecast_crps")
        or _has_nonempty_value(row, "forecast_mase")
        or _has_nonempty_value(row, "crps")
        or _has_nonempty_value(row, "mase")
        for row in rows
    )


def _missing_target_value_keys(rows: Sequence[Mapping[str, Any]], target_keys: Sequence[str]) -> List[str]:
    return sorted({key for key in target_keys if all(not _has_nonempty_value(row, key) for row in rows)})


def _needs_forecast_uniform_rewards(rows: Sequence[Mapping[str, Any]], target_keys: Sequence[str]) -> bool:
    reward_keys = {"u_comp_uniform", "u_crps_uniform", "u_mase_uniform"}
    if not any(key in reward_keys for key in target_keys):
        return False
    return bool(_rows_are_forecast_family(rows) and _rows_have_forecast_metrics(rows) and _missing_target_value_keys(rows, target_keys))


def _rows_without_schedule_keys(rows: Sequence[Mapping[str, Any]], schedule_keys: Sequence[str]) -> List[Dict[str, Any]]:
    excluded = {str(value) for value in schedule_keys}
    return [dict(row) for row in rows if str(row["scheduler_key"]) not in excluded]


def _history_by_step(training_summary: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    selection = dict(training_summary.get("teacher_checkpoint_selection", {}))
    out: Dict[int, Dict[str, Any]] = {}
    for entry in list(selection.get("history", []) or []):
        if not bool(entry.get("selection_constraints_passed", True)):
            continue
        out[int(entry["step"])] = dict(entry)
    return out


def _select_weighted_normalized_regret_step(
    *,
    context_density_training: Mapping[str, Any],
    unseen_nfe_training: Mapping[str, Any],
    component_weights: Mapping[str, float],
) -> Dict[str, Any]:
    context_history = _history_by_step(context_density_training)
    unseen_history = _history_by_step(unseen_nfe_training)
    common_steps = sorted(set(context_history) & set(unseen_history))
    if not common_steps:
        raise ValueError("Weighted normalized-regret checkpoint selection found no common checkpoint steps.")
    required_splits = (
        "context_disjoint",
        "density_family_holdout",
        "unseen_nfe_holdout",
    )
    combined_history: List[Dict[str, Any]] = []
    for step in common_steps:
        context_diagnostics = dict(context_history[step].get("diagnostics", {}) or {})
        unseen_diagnostics = dict(unseen_history[step].get("diagnostics", {}) or {})
        diagnostics = {
            "context_disjoint": dict(context_diagnostics.get("context_disjoint", {}) or {}),
            "density_family_holdout": dict(context_diagnostics.get("density_family_holdout", {}) or {}),
            "unseen_nfe_holdout": dict(unseen_diagnostics.get("unseen_nfe_holdout", {}) or {}),
        }
        combined_history.append({"step": int(step), "diagnostics": diagnostics})
    weight_summary = _teacher_selection_weight_summary(
        component_weights,
        active_axes=("context", "density_family", "unseen_nfe"),
    )
    effective_weights = dict(weight_summary["effective_axis_weights"])
    split_weights = {
        "context": float(effective_weights["context"]),
        "density_family": float(effective_weights["density_family"]),
        "unseen_nfe_holdout": float(effective_weights["unseen_nfe"]),
    }
    selection = select_weighted_normalized_regret_checkpoint(
        combined_history,
        required_split_names=required_splits,
        component_weights=split_weights,
    )
    return {
        **selection,
        "selection_component_axis_weights": dict(effective_weights),
        "selection_nominal_axis_weights": dict(weight_summary["nominal_axis_weights"]),
        "selection_effective_axis_weights": dict(effective_weights),
        "selection_inactive_axes": list(weight_summary["inactive_axes"]),
        "uses_unseen_nfe_selection_diagnostics": True,
        "component_histories": {
            "context_density": context_density_training.get("teacher_checkpoint_selection", {}).get("history", []),
            "unseen_nfe": unseen_nfe_training.get("teacher_checkpoint_selection", {}).get("history", []),
        },
    }


def _select_context_density_regret_step(
    *,
    context_density_training: Mapping[str, Any],
    component_weights: Mapping[str, float],
) -> Dict[str, Any]:
    context_history = _history_by_step(context_density_training)
    if not context_history:
        raise ValueError("Context/density checkpoint selection found no eligible checkpoint steps.")
    required_splits = ("context_disjoint", "density_family_holdout")
    weight_summary = _teacher_selection_weight_summary(
        component_weights,
        active_axes=("context", "density_family"),
    )
    effective_weights = dict(weight_summary["effective_axis_weights"])
    split_weights = {
        "context": float(effective_weights["context"]),
        "density_family": float(effective_weights["density_family"]),
    }
    selection = select_weighted_normalized_regret_checkpoint(
        [{"step": int(step), "diagnostics": dict(entry.get("diagnostics", {}) or {})} for step, entry in sorted(context_history.items())],
        required_split_names=required_splits,
        component_weights=split_weights,
    )
    return {
        **selection,
        "selection_component_axis_weights": dict(effective_weights),
        "selection_nominal_axis_weights": dict(weight_summary["nominal_axis_weights"]),
        "selection_effective_axis_weights": dict(effective_weights),
        "selection_inactive_axes": list(weight_summary["inactive_axes"]),
        "uses_unseen_nfe_selection_diagnostics": False,
        "component_histories": {
            "context_density": context_density_training.get("teacher_checkpoint_selection", {}).get("history", []),
            "unseen_nfe": [],
        },
    }


def _observed_support(rows: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    observed = tuple(sorted({str(row["scheduler_key"]) for row in rows}))
    canonical_order = tuple(key for key in DEFAULT_SUPPORT_SCHEDULE_KEYS if key in observed)
    extras = tuple(key for key in observed if key not in canonical_order)
    return validate_gipo_support_schedule_keys(canonical_order + extras)


def _stable_hash(values: Sequence[str]) -> str:
    payload = json.dumps([str(value) for value in values], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_scope_from_row(row: Mapping[str, Any]) -> str:
    return checkpoint_scope_from_row(row, empty_label="unscoped")


def _checkpoint_context_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return (_checkpoint_scope_from_row(row), context_id_from_row(row))


def _scope_seed(base_seed: int, scope: str) -> int:
    digest = hashlib.sha256(str(scope).encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest[:8], 16)


def _row_membership_key(row: Mapping[str, Any]) -> str:
    payload = {
        "dataset": str(row.get("dataset", row.get("dataset_key", ""))),
        "solver_key": normalize_solver_key(str(row.get("solver_key", ""))),
        "target_nfe": int(row["target_nfe"]),
        "context_id": context_id_from_row(row),
        "context_embedding_id": context_embedding_id_from_row(row),
        "seed": str(row.get("seed", "")),
        "checkpoint_id": str(row.get("checkpoint_id", "") or ""),
        "checkpoint_scope": _checkpoint_scope_from_row(row),
        "scheduler_key": str(row["scheduler_key"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _split_membership_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    context_ids = sorted({context_id_from_row(row) for row in rows})
    checkpoint_scopes = sorted({_checkpoint_scope_from_row(row) for row in rows})
    series_keys = sorted({series_key_from_row(row) for row in rows})
    split_phases = sorted({str(row.get("split_phase", row.get("split", ""))) for row in rows})
    return {
        "row_count": int(len(rows)),
        "context_count": int(len(context_ids)),
        "checkpoint_scope_count": int(len(checkpoint_scopes)),
        "series_count": int(len(series_keys)),
        "source_split_phases": split_phases,
        "context_ids": context_ids,
        "context_id_hash": _stable_hash(context_ids),
        "checkpoint_scopes": checkpoint_scopes,
        "checkpoint_scope_hash": _stable_hash(checkpoint_scopes),
        "series_keys": series_keys,
        "series_key_hash": _stable_hash(series_keys),
    }


def _normalizer_fit_scope_summary(scope: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    context_ids = sorted({context_id_from_row(row) for row in rows})
    embedding_ids = sorted(_embedding_ids(rows))
    series_keys = sorted({series_key_from_row(row) for row in rows})
    schedule_keys = sorted({str(row["scheduler_key"]) for row in rows})
    row_keys = sorted(_row_membership_key(row) for row in rows)
    return {
        "scope": str(scope),
        "row_count": int(len(rows)),
        "context_count": int(len(context_ids)),
        "embedding_count": int(len(embedding_ids)),
        "series_count": int(len(series_keys)),
        "schedule_count": int(len(schedule_keys)),
        "context_id_hash": _stable_hash(context_ids),
        "embedding_id_hash": _stable_hash(embedding_ids),
        "series_key_hash": _stable_hash(series_keys),
        "schedule_key_hash": _stable_hash(schedule_keys),
        "row_membership_hash": _stable_hash(row_keys),
        "membership_hash": _stable_hash(row_keys),
    }


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    from genode.gipo.policy import series_key_from_row

    return {
        "row_count": int(len(rows)),
        "context_count": int(len({context_id_from_row(row) for row in rows})),
        "checkpoint_scope_count": int(len({_checkpoint_scope_from_row(row) for row in rows})),
        "series_count": int(len({series_key_from_row(row) for row in rows})),
        "schedule_count": int(len({str(row["scheduler_key"]) for row in rows})),
    }


def _finite_metric_target_present(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, None)
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _teacher_metric_coverage_summary(
    rows: Sequence[Mapping[str, Any]],
    target_keys: Sequence[str],
    *,
    scope: str,
) -> Dict[str, Any]:
    row_count = int(len(rows))
    raw_coverage = teacher_metric_target_coverage(rows, target_keys)
    metrics: Dict[str, Dict[str, Any]] = {}
    for key, item in raw_coverage.items():
        valid_count = int(item["valid_count"])
        metrics[str(key)] = {
            "valid_row_count": valid_count,
            "applicable_row_count": int(item["applicable_count"]),
            "missing_row_count": int(item["missing_count"]),
            "nonfinite_row_count": int(item["nonfinite_count"]),
            "inapplicable_row_count": int(item["inapplicable_count"]),
            "missing_or_invalid_row_count": int(item["missing_count"]) + int(item["nonfinite_count"]),
            "coverage_fraction": float(item["coverage_fraction"]),
        }
    all_target_valid_count = int(
        sum(1 for row in rows if all(_finite_metric_target_present(row, key) for key in target_keys))
    )
    any_target_valid_count = int(
        sum(1 for row in rows if any(_finite_metric_target_present(row, key) for key in target_keys))
    )
    return {
        "scope": str(scope),
        "row_count": row_count,
        "metric_count": int(len(tuple(target_keys))),
        "all_target_valid_row_count": all_target_valid_count,
        "any_target_valid_row_count": any_target_valid_count,
        "metrics": metrics,
    }


def _validate_teacher_metric_coverage(
    coverage: Mapping[str, Any],
    *,
    min_coverage_fraction: float,
    min_valid_rows: int,
) -> None:
    failures: List[str] = []
    for key, item in dict(coverage.get("metrics", {}) or {}).items():
        valid_count = int(dict(item).get("valid_row_count", 0))
        fraction = float(dict(item).get("coverage_fraction", 0.0))
        if valid_count < int(min_valid_rows) or fraction + 1e-12 < float(min_coverage_fraction):
            failures.append(
                f"{key}: valid_rows={valid_count}/{int(coverage.get('row_count', 0))}, "
                f"coverage={fraction:.6g}"
            )
    if failures:
        raise ValueError(
            f"Teacher metric target coverage failed for {coverage.get('scope', 'rows')}: "
            f"min_coverage_fraction={float(min_coverage_fraction):.6g}, "
            f"min_valid_rows={int(min_valid_rows)}, failures={failures}."
        )


def _normalized_teacher_selection_axis_weights(component_weights: Mapping[str, float]) -> Dict[str, float]:
    raw = {axis: max(0.0, float(component_weights.get(axis, 0.0))) for axis in TEACHER_SELECTION_AXIS_ORDER}
    total = float(sum(raw.values()))
    if total <= 0.0:
        raise ValueError("Teacher selection axis weights require a positive total.")
    return {axis: float(value / total) for axis, value in raw.items()}


def _teacher_selection_weight_summary(
    component_weights: Mapping[str, float],
    *,
    active_axes: Sequence[str],
) -> Dict[str, Any]:
    nominal = _normalized_teacher_selection_axis_weights(component_weights)
    active_set = {str(axis) for axis in active_axes if str(axis) in set(TEACHER_SELECTION_AXIS_ORDER)}
    inactive_axes = [axis for axis in TEACHER_SELECTION_AXIS_ORDER if axis not in active_set]
    active_total = float(sum(nominal[axis] for axis in active_set))
    if active_set and active_total <= 0.0:
        raise ValueError("Teacher selection active axes have zero nominal weight.")
    effective = {
        axis: float(nominal[axis] / active_total) if axis in active_set and active_total > 0.0 else 0.0
        for axis in TEACHER_SELECTION_AXIS_ORDER
    }
    return {
        "protocol": GIPO_TEACHER_SELECTION_WEIGHT_PROTOCOL,
        "nominal_axis_weights": nominal,
        "effective_axis_weights": effective,
        "inactive_axes": inactive_axes,
        "active_axes": [axis for axis in TEACHER_SELECTION_AXIS_ORDER if axis in active_set],
    }


def _validate_support_group_counts(rows: Sequence[Mapping[str, Any]], support_schedule_keys: Sequence[str]) -> None:
    support_keys = tuple(str(key) for key in support_schedule_keys)
    if "uniform" not in set(support_keys):
        raise ValueError("GIPO supervision support must include the uniform reward anchor schedule.")
    support_set = set(support_keys)
    grouped: Dict[Tuple[Any, ...], Dict[str, int]] = {}
    for row in rows:
        schedule_key = str(row["scheduler_key"])
        if schedule_key not in support_set:
            raise ValueError(f"Supervision row {schedule_key!r} is outside support_schedule_keys.")
        counts = grouped.setdefault(tuple(context_pair_key(row, pair_on_seed=True)), {key: 0 for key in support_keys})
        counts[schedule_key] = int(counts.get(schedule_key, 0)) + 1
    bad = {
        key: {schedule: count for schedule, count in counts.items() if count != 1}
        for key, counts in grouped.items()
        if any(count != 1 for count in counts.values())
    }
    if bad:
        first_key = next(iter(bad))
        raise ValueError(
            "GIPO supervision requires exactly one row for every support schedule in each "
            f"context/seed/solver/NFE group; first bad group={first_key}, counts={bad[first_key]}."
        )


def _validate_rank_pair_preflight(rank_pair_preflight: Mapping[str, Mapping[str, Any]]) -> None:
    failures: List[str] = []
    for split_name, diagnostics in rank_pair_preflight.items():
        row_count = int(diagnostics.get("row_count", 0) or 0)
        rankable_pair_count = int(diagnostics.get("rankable_pair_count", 0) or 0)
        if row_count > 0 and rankable_pair_count <= 0:
            failures.append(
                f"{split_name}: row_count={row_count}, "
                f"rankable_pair_count={rankable_pair_count}, "
                f"pair_group_count={int(diagnostics.get('pair_group_count', 0) or 0)}, "
                f"singleton_group_count={int(diagnostics.get('singleton_group_count', 0) or 0)}, "
                f"tie_only_group_count={int(diagnostics.get('tie_only_group_count', 0) or 0)}, "
                f"examples={list(diagnostics.get('example_bad_groups', []) or [])[:3]}"
            )
    if failures:
        raise ValueError(
            "GIPO rank_pair_preflight failed before teacher training; "
            "each teacher fit split requires at least one same-context support pair with different scalarized utility. "
            f"Failures: {failures}"
        )


def _validate_unique_schedule_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    counts: Dict[Tuple[Any, ...], int] = {}
    for row in rows:
        key = (*context_pair_key(row, pair_on_seed=True), str(row["scheduler_key"]))
        counts[key] = int(counts.get(key, 0)) + 1
    duplicates = {key: count for key, count in counts.items() if count != 1}
    if duplicates:
        first_key = next(iter(duplicates))
        raise ValueError(
            f"{label} contains duplicate schedule rows for exact "
            "(dataset,solver_key,target_nfe,context_id,seed,checkpoint_scope,scheduler_key) cells; "
            f"first duplicate={first_key}, count={duplicates[first_key]}."
        )


def _validate_context_embedding_checkpoint_scope(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    mismatches: List[str] = []
    prefixed_context_ids: List[str] = []
    for row in rows:
        checkpoint_id = str(row.get("checkpoint_id", "") or "").strip()
        if not checkpoint_id:
            continue
        context_id = context_id_from_row(row)
        if str(context_id).startswith(f"{checkpoint_id}:"):
            prefixed_context_ids.append(str(context_id))
        embedding_id = context_embedding_id_from_row(row)
        if not str(embedding_id).startswith(f"{checkpoint_id}:"):
            mismatches.append(f"{checkpoint_id}->{embedding_id}")
    if mismatches:
        raise ValueError(
            f"{label} requires checkpoint-scoped context_embedding_id values; "
            f"first mismatches: {mismatches[:8]}."
        )
    if prefixed_context_ids:
        raise ValueError(
            f"{label} requires physical context_id values without checkpoint prefixes; "
            f"first prefixed context_ids: {prefixed_context_ids[:8]}."
        )


def _embedding_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {context_embedding_id_from_row(row) for row in rows}


def _checkpoint_id_from_row(row: Mapping[str, Any]) -> str:
    return _checkpoint_scope_from_row(row)


def _sample_context_keys_by_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
    seed: int,
) -> set[Tuple[str, str]]:
    by_checkpoint: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_checkpoint.setdefault(_checkpoint_id_from_row(row), []).append(row)
    selected: set[Tuple[str, str]] = set()
    for checkpoint_idx, checkpoint_id in enumerate(sorted(by_checkpoint)):
        checkpoint_rows = by_checkpoint[checkpoint_id]
        per_checkpoint_count = min(
            int(sample_count),
            len({context_id_from_row(row) for row in checkpoint_rows}),
        )
        context_ids = sample_context_ids_stratified(
            checkpoint_rows,
            sample_count=per_checkpoint_count,
            seed=_scope_seed(int(seed) + 10_000 * int(checkpoint_idx), checkpoint_id),
        )
        selected.update((str(checkpoint_id), str(context_id)) for context_id in context_ids)
    return selected


def _rows_for_context_keys(
    rows: Sequence[Mapping[str, Any]],
    selected: set[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if _checkpoint_context_key(row) in selected
    ]


def _context_sampling_summary(
    rows: Sequence[Mapping[str, Any]],
    selected: set[Tuple[str, str]],
    *,
    sample_count: int,
) -> Dict[str, Any]:
    per_checkpoint: Dict[str, Dict[str, int]] = {}
    available_by_checkpoint: Dict[str, set[str]] = {}
    selected_by_checkpoint: Dict[str, set[str]] = {}
    for row in rows:
        checkpoint_id = _checkpoint_id_from_row(row)
        available_by_checkpoint.setdefault(checkpoint_id, set()).add(context_id_from_row(row))
    for checkpoint_id, context_id in selected:
        selected_by_checkpoint.setdefault(str(checkpoint_id), set()).add(str(context_id))
    for checkpoint_id in sorted(available_by_checkpoint):
        per_checkpoint[checkpoint_id] = {
            "available_contexts": int(len(available_by_checkpoint[checkpoint_id])),
            "selected_contexts": int(len(selected_by_checkpoint.get(checkpoint_id, set()))),
        }
    return {
        "protocol": "checkpoint_maturity_stratified_context_sample_v1",
        "sample_count_per_checkpoint": int(sample_count),
        "checkpoint_count": int(len(available_by_checkpoint)),
        "available_physical_context_count": int(len({context_id for values in available_by_checkpoint.values() for context_id in values})),
        "selected_physical_context_count": int(len({context_id for _checkpoint_id, context_id in selected})),
        "per_checkpoint": per_checkpoint,
    }


def _assert_embedding_overlap_compatible(
    base: Mapping[str, Sequence[float]],
    extra: Mapping[str, Sequence[float]],
    *,
    label: str,
    atol: float = 1e-6,
) -> None:
    overlap = sorted(set(base) & set(extra))
    bad: List[str] = []
    for key in overlap:
        left = np.asarray(base[key], dtype=np.float32)
        right = np.asarray(extra[key], dtype=np.float32)
        if left.shape != right.shape or not np.allclose(left, right, rtol=1e-5, atol=float(atol)):
            bad.append(str(key))
    if bad:
        raise ValueError(
            f"{label} context embeddings collide with different vectors; first incompatible IDs: {bad[:8]}"
        )


def _merge_embedding_tables_guarded(
    base: Dict[str, np.ndarray],
    extra: Mapping[str, Sequence[float]],
    *,
    label: str,
) -> None:
    _assert_embedding_overlap_compatible(base, extra, label=label)
    for key, value in extra.items():
        if str(key) not in base:
            base[str(key)] = np.asarray(value, dtype=np.float32)


def _source_split_phase(row: Mapping[str, Any]) -> str:
    return str(row.get("source_split_phase") or row.get("split_phase", row.get("split", ""))).strip()


def _raw_split_phase(row: Mapping[str, Any]) -> str:
    return str(row.get("split_phase", row.get("split", ""))).strip()


def _validate_train_tuning_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    locked_rows = [
        row
        for row in rows
        if _source_split_phase(row) == "locked_test" or _raw_split_phase(row) == "locked_test"
    ]
    if locked_rows:
        raise ValueError(f"{label} refuses locked_test rows; found {len(locked_rows)} locked-test rows.")
    bad_source_phases = sorted(
        {
            _source_split_phase(row)
            for row in rows
            if _source_split_phase(row) and _source_split_phase(row) != "train_tuning"
        }
    )
    if bad_source_phases:
        raise ValueError(f"{label} requires train_tuning source rows; found {bad_source_phases}.")


def _validate_positive_training_args(args: argparse.Namespace) -> None:
    for name in ("teacher_steps", "student_steps", "teacher_checkpoint_every", "student_checkpoint_every"):
        value = int(getattr(args, name))
        if value <= 0:
            raise ValueError(f"{name} must be positive.")


def _validate_context_split_capacity(
    rows: Sequence[Mapping[str, Any]],
    *,
    holdout_fraction: float,
    label: str,
    minimum_contexts: int = 3,
) -> Dict[str, Any]:
    context_ids = sorted({context_id_from_row(row) for row in rows})
    report = {
        "label": str(label),
        "row_count": int(len(rows)),
        "context_count": int(len(context_ids)),
        "minimum_context_count": int(minimum_contexts),
        "holdout_fraction": float(holdout_fraction),
        "status": "ok",
    }
    if float(holdout_fraction) > 0.0 and len(context_ids) < int(minimum_contexts):
        report["status"] = "insufficient_contexts"
        raise ValueError(
            f"{label} requires at least {minimum_contexts} distinct contexts when holdout_fraction is positive; "
            f"found {len(context_ids)}."
        )
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GIPO continuous-density from per-example fixed/SER rows.")
    parser.add_argument("--rows_csv", required=True, help="Per-example fixed/SER metric rows CSV.")
    parser.add_argument("--context_embeddings_npz", required=True, help="Frozen context embedding sidecar NPZ.")
    parser.add_argument("--schedule_summary_json", default="", help="Comma-separated schedule summaries for non-fixed references such as SER.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--support_schedule_keys", default="", help="Comma-separated fixed/SER supervision keys. Defaults to observed row keys.")
    parser.add_argument("--context_sample_count", type=int, default=CANONICAL_CONTEXT_SAMPLE_COUNT)
    parser.add_argument("--context_holdout_fraction", type=float, default=0.20)
    parser.add_argument(
        "--seen_target_nfe_values",
        default=",".join(str(value) for value in CANONICAL_SEEN_NFES),
        help="Comma-separated seen calibration NFEs expected in --rows_csv.",
    )
    parser.add_argument("--teacher_steps", type=int, default=500)
    parser.add_argument("--teacher_checkpoint_every", type=int, default=100)
    parser.add_argument("--teacher_loss_log_every", type=int, default=0)
    parser.add_argument(
        "--teacher_density_holdout_schedule_keys",
        default=",".join(DEFAULT_DENSITY_FAMILY_HOLDOUT_SCHEDULE_KEYS),
        help="Comma-separated fixed/SER support keys held out from the selector teacher for density-family diagnostics.",
    )
    parser.add_argument(
        "--teacher_unseen_selection_rows_csv",
        default="",
        help="Optional unseen train_tuning rows used only for weighted normalized-regret teacher selection diagnostics.",
    )
    parser.add_argument(
        "--teacher_unseen_selection_context_embeddings_npz",
        default="",
        help="Context embeddings for --teacher_unseen_selection_rows_csv; defaults to --context_embeddings_npz.",
    )
    parser.add_argument(
        "--teacher_unseen_selection_schedule_summary_json",
        default="",
        help="Schedule summaries for unseen selection rows, such as unseen SER references.",
    )
    parser.add_argument(
        "--teacher_unseen_selection_target_nfe_values",
        default=",".join(str(value) for value in CANONICAL_UNSEEN_NFES),
    )
    parser.add_argument(
        "--student_pseudo_rows_csv",
        default="",
        help="Optional unseen-NFE train_tuning rows for down-weighted teacher pseudo-target distillation.",
    )
    parser.add_argument(
        "--student_pseudo_context_embeddings_npz",
        default="",
        help="Context embeddings for --student_pseudo_rows_csv; defaults to --context_embeddings_npz.",
    )
    parser.add_argument(
        "--student_pseudo_schedule_summary_json",
        default="",
        help="Schedule summaries for pseudo rows, such as unseen SER-derived references.",
    )
    parser.add_argument(
        "--pseudo_target_nfe_values",
        default=",".join(str(value) for value in CANONICAL_UNSEEN_NFES),
        help="Comma-separated pseudo-distillation NFEs selected from --student_pseudo_rows_csv.",
    )
    parser.add_argument("--student_pseudo_target_weight", type=float, default=CANONICAL_PSEUDO_TARGET_WEIGHT)
    parser.add_argument("--student_teacher_score_weight", type=float, default=DEFAULT_STUDENT_TEACHER_SCORE_WEIGHT)
    parser.add_argument("--student_teacher_score_warmup_fraction", type=float, default=DEFAULT_STUDENT_TEACHER_SCORE_WARMUP_FRACTION)
    parser.add_argument("--student_teacher_score_include_pseudo", action="store_true", default=False)
    parser.add_argument("--student_target_mixture_mode", choices=STUDENT_TARGET_MIXTURE_MODES, default=DEFAULT_STUDENT_TARGET_MIXTURE_MODE)
    parser.add_argument("--student_target_elite_fraction", type=float, default=DEFAULT_STUDENT_TARGET_ELITE_FRACTION)
    parser.add_argument("--student_target_elite_k", type=int, default=DEFAULT_STUDENT_TARGET_ELITE_K)
    parser.add_argument("--student_target_elite_min_count", type=int, default=DEFAULT_STUDENT_TARGET_ELITE_MIN_COUNT)
    parser.add_argument("--student_target_elite_blend_all_weight", type=float, default=DEFAULT_STUDENT_TARGET_ELITE_BLEND_ALL_WEIGHT)
    parser.add_argument("--student_steps", type=int, default=500)
    parser.add_argument("--student_log_every", type=int, default=0)
    parser.add_argument("--student_checkpoint_every", type=int, default=100)
    parser.add_argument("--student_selection_holdout_fraction", type=float, default=0.10)
    parser.add_argument("--teacher_lr", type=float, default=1e-3)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--transformer_hidden_dim", type=int, default=DEFAULT_TRANSFORMER_HIDDEN_DIM)
    parser.add_argument("--transformer_layers", type=int, default=DEFAULT_TRANSFORMER_LAYERS)
    parser.add_argument("--transformer_heads", type=int, default=DEFAULT_TRANSFORMER_HEADS)
    parser.add_argument("--transformer_dropout", type=float, default=DEFAULT_TRANSFORMER_DROPOUT)
    parser.add_argument("--teacher_temperature", type=float, default=DEFAULT_TEACHER_TARGET_TEMPERATURE)
    parser.add_argument("--teacher_utility_crps_weight", type=float, default=0.5)
    parser.add_argument("--teacher_utility_mase_weight", type=float, default=0.5)
    parser.add_argument(
        "--teacher_metric_target_keys",
        default="auto",
        help="Comma-separated utility columns predicted by the metric-vector teacher, or auto for family defaults.",
    )
    parser.add_argument(
        "--teacher_utility_weights",
        default="",
        help="Optional comma-separated name=value weights for --teacher_metric_target_keys.",
    )
    parser.add_argument(
        "--teacher_metric_min_coverage_fraction",
        type=float,
        default=1.0,
        help="Minimum finite-row coverage required for each requested teacher metric before training.",
    )
    parser.add_argument(
        "--teacher_metric_min_valid_rows",
        type=int,
        default=1,
        help="Minimum finite rows required for each requested teacher metric before training.",
    )
    parser.add_argument("--student_weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def train_gipo(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_positive_training_args(args)
    seed_all(int(args.seed))
    density_bin_count = CANONICAL_DENSITY_BIN_COUNT
    selection_mode = TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET
    student_selection_mode = STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE
    validate_gipo_attention_heads(int(args.transformer_heads))
    conditioning_style = CONDITIONING_STYLE_ADDITIVE_MLP
    requested_setting_mode = SETTING_ENCODER_MODE_CONTINUOUS_V3
    setting_feature_mode = validate_setting_feature_mode(requested_setting_mode)
    rows = _read_metric_rows_csvs(str(args.rows_csv))
    if not rows:
        raise ValueError("rows_csv contains no rows.")
    locked_rows = [
        row
        for row in rows
        if _source_split_phase(row) == "locked_test" or _raw_split_phase(row) == "locked_test"
    ]
    if locked_rows:
        raise ValueError(f"GIPO training refuses locked_test rows in rows_csv; found {len(locked_rows)} locked-test rows.")
    observed_source_phases = sorted({_source_split_phase(row) for row in rows if _source_split_phase(row)})
    if observed_source_phases and observed_source_phases != ["train_tuning"]:
        raise ValueError(f"GIPO training rows_csv must contain only train_tuning source rows; found {observed_source_phases}.")
    _validate_unique_schedule_rows(rows, label="GIPO training rows_csv")
    _validate_context_embedding_checkpoint_scope(rows, label="GIPO training rows_csv")
    validate_teacher_objective_hyperparameters(
        rank_temperature=CANONICAL_TEACHER_RANK_TEMPERATURE,
        regression_weight=CANONICAL_TEACHER_REGRESSION_WEIGHT,
        pair_margin=CANONICAL_TEACHER_PAIR_MARGIN,
    )
    support_keys = (
        validate_gipo_support_schedule_keys(_parse_csv(str(args.support_schedule_keys)))
        if str(args.support_schedule_keys).strip()
        else _observed_support(rows)
    )
    observed_keys = {str(row["scheduler_key"]) for row in rows}
    missing_support_rows = sorted(set(support_keys) - observed_keys)
    if missing_support_rows:
        raise ValueError(f"Supervision schedules must have measured context rows; missing rows for {missing_support_rows}")
    seen_target_nfes = sorted(
        set(
            _parse_int_csv_or_default(
                getattr(args, "seen_target_nfe_values", ",".join(str(value) for value in CANONICAL_SEEN_NFES)),
                CANONICAL_SEEN_NFES,
            )
        )
    )
    pseudo_target_nfes = sorted(
        set(
            _parse_int_csv_or_default(
                getattr(args, "pseudo_target_nfe_values", ",".join(str(value) for value in CANONICAL_UNSEEN_NFES)),
                CANONICAL_UNSEEN_NFES,
            )
        )
    )

    teacher_metric_target_keys = _resolve_teacher_metric_target_keys(args, rows)
    explicit_teacher_weights = _parse_float_mapping(str(args.teacher_utility_weights)) if str(args.teacher_utility_weights).strip() else None
    if explicit_teacher_weights is None and teacher_metric_target_keys == ("u_crps_uniform", "u_mase_uniform"):
        explicit_teacher_weights = {
            "u_crps_uniform": float(args.teacher_utility_crps_weight),
            "u_mase_uniform": float(args.teacher_utility_mase_weight),
        }
    teacher_utility_weights = teacher_utility_weights_for_summary(
        teacher_metric_target_keys,
        normalize_teacher_utility_weights(teacher_metric_target_keys, explicit_teacher_weights),
    )
    teacher_metric_min_coverage_fraction = float(getattr(args, "teacher_metric_min_coverage_fraction", 1.0))
    if (
        not math.isfinite(teacher_metric_min_coverage_fraction)
        or teacher_metric_min_coverage_fraction < 0.0
        or teacher_metric_min_coverage_fraction > 1.0
    ):
        raise ValueError("teacher_metric_min_coverage_fraction must be finite and in [0, 1].")
    teacher_metric_min_valid_rows = int(getattr(args, "teacher_metric_min_valid_rows", 1))
    if teacher_metric_min_valid_rows < 0:
        raise ValueError("teacher_metric_min_valid_rows must be nonnegative.")
    teacher_metric_coverage_scopes: Dict[str, Any] = {}
    if _needs_forecast_uniform_rewards(rows, teacher_metric_target_keys):
        rewarded_rows = attach_uniform_gipo_rewards(
            rows,
            support_schedule_keys=support_keys,
            utility_crps_weight=float(args.teacher_utility_crps_weight),
            utility_mase_weight=float(args.teacher_utility_mase_weight),
            pair_on_seed=True,
        )
    else:
        rewarded_rows = [dict(row) for row in rows]
        for row in rewarded_rows:
            row["context_id"] = context_id_from_row(row)
    _validate_support_group_counts(rewarded_rows, support_keys)
    missing_metric_columns = _missing_target_value_keys(rewarded_rows, teacher_metric_target_keys)
    if len(missing_metric_columns) == len(teacher_metric_target_keys):
        raise ValueError(f"GIPO rows are missing all teacher metric target columns: {missing_metric_columns}")
    primary_metric_coverage = _teacher_metric_coverage_summary(
        rewarded_rows,
        teacher_metric_target_keys,
        scope="rows_csv",
    )
    _validate_teacher_metric_coverage(
        primary_metric_coverage,
        min_coverage_fraction=teacher_metric_min_coverage_fraction,
        min_valid_rows=teacher_metric_min_valid_rows,
    )
    teacher_metric_coverage_scopes["rows_csv"] = primary_metric_coverage
    available_context_ids = sorted({context_id_from_row(row) for row in rewarded_rows})
    sample_count = int(args.context_sample_count)
    if sample_count <= 0:
        sample_count = recommended_context_calibration_count(len(available_context_ids))
    selected_context_keys = _sample_context_keys_by_checkpoint(
        rewarded_rows,
        sample_count=sample_count,
        seed=int(args.seed),
    )
    sampled_rows = _rows_for_context_keys(rewarded_rows, selected_context_keys)
    context_sampling = _context_sampling_summary(
        rewarded_rows,
        selected_context_keys,
        sample_count=sample_count,
    )

    selection_component_weights = dict(CANONICAL_TEACHER_SELECTION_COMPONENT_WEIGHTS)
    expected_weights = {"context": 0.25, "density_family": 0.25, "unseen_nfe": 0.50}
    bad_weights = {
        key: selection_component_weights.get(key)
        for key, expected in expected_weights.items()
        if abs(float(selection_component_weights.get(key, 0.0)) - float(expected)) > 1e-9
    }
    if bad_weights:
        raise ValueError(
            "weighted_normalized_regret requires J_CDN weights "
            "context=0.25,density_family=0.25,unseen_nfe=0.50."
        )
    sampled_target_nfes = sorted({int(row["target_nfe"]) for row in sampled_rows})
    if sampled_target_nfes != seen_target_nfes:
        raise ValueError(
            "weighted_normalized_regret final teacher/student fitting expects seen calibration NFEs "
            f"{seen_target_nfes}; "
            f"found {sampled_target_nfes}."
        )
    unseen_selection_rows: List[Dict[str, Any]] = []
    unseen_context_sampling: Dict[str, Any] = {}
    unseen_selection_target_nfes = _parse_int_csv(str(args.teacher_unseen_selection_target_nfe_values))
    if str(args.teacher_unseen_selection_rows_csv).strip():
        unseen_raw_rows = _read_metric_rows_csvs(str(args.teacher_unseen_selection_rows_csv))
        if not unseen_raw_rows:
            raise ValueError("teacher_unseen_selection_rows_csv contains no rows.")
        _validate_unique_schedule_rows(unseen_raw_rows, label="Teacher unseen selection rows")
        _validate_context_embedding_checkpoint_scope(unseen_raw_rows, label="Teacher unseen selection rows")
        unseen_locked_rows = [
            row
            for row in unseen_raw_rows
            if _source_split_phase(row) == "locked_test" or _raw_split_phase(row) == "locked_test"
        ]
        if unseen_locked_rows:
            raise ValueError(
                "Teacher unseen selection diagnostics refuse locked_test rows; "
                f"found {len(unseen_locked_rows)} locked-test rows."
            )
        bad_unseen_source_phases = sorted(
            {
                _source_split_phase(row)
                for row in unseen_raw_rows
                if _source_split_phase(row) != "train_tuning"
            }
        )
        if bad_unseen_source_phases:
            raise ValueError(
                "Teacher unseen selection diagnostics require train_tuning source rows; "
                f"found {bad_unseen_source_phases}."
            )
        unseen_filtered_rows = _rows_for_target_nfes(unseen_raw_rows, unseen_selection_target_nfes)
        if not unseen_filtered_rows:
            raise ValueError(
                "teacher_unseen_selection_rows_csv has no rows after filtering to "
                f"teacher_unseen_selection_target_nfe_values={unseen_selection_target_nfes}."
            )
        if _needs_forecast_uniform_rewards(unseen_filtered_rows, teacher_metric_target_keys):
            unseen_rewarded_rows = attach_uniform_gipo_rewards(
                unseen_filtered_rows,
                support_schedule_keys=support_keys,
                utility_crps_weight=float(args.teacher_utility_crps_weight),
                utility_mase_weight=float(args.teacher_utility_mase_weight),
                pair_on_seed=True,
            )
        else:
            unseen_rewarded_rows = [dict(row) for row in unseen_filtered_rows]
            for row in unseen_rewarded_rows:
                row["context_id"] = context_id_from_row(row)
        _validate_support_group_counts(unseen_rewarded_rows, support_keys)
        missing_unseen_columns = _missing_target_value_keys(unseen_rewarded_rows, teacher_metric_target_keys)
        if len(missing_unseen_columns) == len(teacher_metric_target_keys):
            raise ValueError(f"Teacher unseen selection rows are missing all metric target columns: {missing_unseen_columns}")
        unseen_metric_coverage = _teacher_metric_coverage_summary(
            unseen_rewarded_rows,
            teacher_metric_target_keys,
            scope="teacher_unseen_selection_rows_csv",
        )
        _validate_teacher_metric_coverage(
            unseen_metric_coverage,
            min_coverage_fraction=teacher_metric_min_coverage_fraction,
            min_valid_rows=teacher_metric_min_valid_rows,
        )
        teacher_metric_coverage_scopes["teacher_unseen_selection_rows_csv"] = unseen_metric_coverage
        unseen_context_keys = _sample_context_keys_by_checkpoint(
            unseen_rewarded_rows,
            sample_count=sample_count,
            seed=int(args.seed) + 404,
        )
        unseen_selection_rows = _rows_for_context_keys(unseen_rewarded_rows, unseen_context_keys)
        unseen_context_sampling = _context_sampling_summary(
            unseen_rewarded_rows,
            unseen_context_keys,
            sample_count=sample_count,
        )
    density_holdout_requested = _parse_csv(str(args.teacher_density_holdout_schedule_keys))
    if "uniform" in {str(key) for key in density_holdout_requested}:
        raise ValueError("density-family holdout must not include uniform; uniform is the reward anchor.")
    support_set = set(support_keys)
    missing_density_holdout_keys = sorted(set(density_holdout_requested) - support_set)
    if missing_density_holdout_keys and tuple(density_holdout_requested) == DEFAULT_DENSITY_FAMILY_HOLDOUT_SCHEDULE_KEYS:
        present_default_keys = [key for key in density_holdout_requested if key in support_set]
        if not present_default_keys:
            raise ValueError(
                "Default teacher_density_holdout_schedule_keys require support schedules "
                f"{list(DEFAULT_DENSITY_FAMILY_HOLDOUT_SCHEDULE_KEYS)}; reduced support "
                f"{list(support_keys)} is missing {missing_density_holdout_keys}. "
                "Pass --teacher_density_holdout_schedule_keys with supported non-uniform keys or include the default rows."
            )
        density_holdout_keys: List[str] = present_default_keys
    elif missing_density_holdout_keys:
        raise ValueError(f"density-family holdout keys must be in support_schedule_keys; unsupported={missing_density_holdout_keys}.")
    else:
        density_holdout_keys = list(density_holdout_requested)

    context_split_preflight = {
        "teacher_context_holdout": _validate_context_split_capacity(
            sampled_rows,
            holdout_fraction=float(args.context_holdout_fraction),
            label="Teacher context holdout",
        )
    }
    context_fit_pool_rows, context_holdout_rows = split_rows_by_context_holdout(
        sampled_rows,
        holdout_fraction=float(args.context_holdout_fraction),
        seed=int(args.seed),
    )
    density_selection_fit_rows, density_holdout_rows, density_holdout_metadata = split_rows_by_density_family_holdout(
        context_fit_pool_rows,
        holdout_schedule_keys=density_holdout_keys,
        support_schedule_keys=support_keys,
    )
    selection_support_schedule_keys = tuple(key for key in support_keys if key not in set(density_holdout_keys))
    context_holdout_diagnostic_rows = _rows_without_schedule_keys(context_holdout_rows, density_holdout_keys)
    if not density_holdout_rows and not bool(args.dry_run):
        raise ValueError("weighted_normalized_regret selection requires non-empty density-family holdout rows.")
    selector_fit_rows = density_selection_fit_rows
    final_fit_rows = [dict(row) for row in sampled_rows]
    if not selector_fit_rows:
        raise ValueError("Teacher selection fitting requires at least one row after context, series, and density-family holdouts.")
    if not final_fit_rows:
        raise ValueError("Final teacher fitting requires at least one eligible non-test calibration row.")
    setting_encoder_config = setting_encoder_config_for_rows(final_fit_rows, mode=setting_feature_mode)

    density_family_diagnostic_rows = [dict(row) for row in density_holdout_rows]
    unseen_nfe_diagnostic_rows = _rows_without_schedule_keys(unseen_selection_rows, density_holdout_keys)
    context_embeddings = _load_context_embedding_tables(str(args.context_embeddings_npz))
    fit_rows = final_fit_rows
    fit_context_ids = sorted({context_id_from_row(row) for row in fit_rows})
    final_embedding_ids = sorted(_embedding_ids(fit_rows))
    selector_embedding_ids = sorted(_embedding_ids(selector_fit_rows))
    final_embedding_normalizer = EmbeddingNormalizer.fit(context_embeddings, final_embedding_ids)
    selector_embedding_normalizer = EmbeddingNormalizer.fit(context_embeddings, selector_embedding_ids)
    final_normalized_embeddings = final_embedding_normalizer.transform_table(context_embeddings)
    selector_normalized_embeddings = selector_embedding_normalizer.transform_table(context_embeddings)
    missing_final_embeddings = sorted(_embedding_ids(sampled_rows) - set(final_normalized_embeddings))
    if missing_final_embeddings:
        raise KeyError(f"Context embeddings are missing sampled contexts: {missing_final_embeddings[:8]}")
    selector_required_rows = [
        *selector_fit_rows,
        *context_holdout_diagnostic_rows,
        *density_family_diagnostic_rows,
    ]
    missing_selector_embeddings = sorted(_embedding_ids(selector_required_rows) - set(selector_normalized_embeddings))
    if missing_selector_embeddings:
        raise KeyError(f"Selector context embeddings are missing sampled contexts: {missing_selector_embeddings[:8]}")
    if unseen_selection_rows:
        unseen_embeddings_path = str(args.teacher_unseen_selection_context_embeddings_npz).strip() or str(args.context_embeddings_npz)
        unseen_raw_embeddings = _load_context_embedding_tables(unseen_embeddings_path)
        missing_unseen_embeddings = sorted(_embedding_ids(unseen_selection_rows) - set(unseen_raw_embeddings))
        if missing_unseen_embeddings:
            raise KeyError(f"Unseen selection context embeddings are missing contexts: {missing_unseen_embeddings[:8]}")
        _merge_embedding_tables_guarded(
            selector_normalized_embeddings,
            selector_embedding_normalizer.transform_table(unseen_raw_embeddings),
            label="teacher_unseen_selection",
        )
    embedding_normalizer = final_embedding_normalizer
    normalized_embeddings = final_normalized_embeddings

    series_index_map = build_series_index_map(fit_rows)
    fit_series_keys = sorted({series_key_from_row(row) for row in fit_rows})
    context_dim = int(next(iter(normalized_embeddings.values())).shape[0])
    setting_dim = int(setting_feature_dim(setting_feature_mode, config=setting_encoder_config))
    reference_time_grid = uniform_reference_grid(density_bin_count)
    schedule_grids = load_schedule_summary_grids(_parse_csv(str(args.schedule_summary_json)))
    unseen_schedule_summary_paths = _parse_csv(str(args.teacher_unseen_selection_schedule_summary_json))
    if unseen_schedule_summary_paths:
        schedule_grids.update(load_schedule_summary_grids(unseen_schedule_summary_paths))
    schedule_grid_preflight = {
        "selector_fit_rows": validate_schedule_grid_coverage(
            selector_fit_rows,
            schedule_grids=schedule_grids,
            reference_time_grid=reference_time_grid,
            label="Selector fit",
        ),
        "final_fit_rows": validate_schedule_grid_coverage(
            fit_rows,
            schedule_grids=schedule_grids,
            reference_time_grid=reference_time_grid,
            label="Final fit",
        ),
        "context_disjoint_diagnostic_rows": validate_schedule_grid_coverage(
            context_holdout_diagnostic_rows,
            schedule_grids=schedule_grids,
            reference_time_grid=reference_time_grid,
            label="Context-disjoint diagnostic",
        ),
        "density_family_diagnostic_rows": validate_schedule_grid_coverage(
            density_family_diagnostic_rows,
            schedule_grids=schedule_grids,
            reference_time_grid=reference_time_grid,
            label="Density-family diagnostic",
        ),
    }
    if unseen_nfe_diagnostic_rows:
        schedule_grid_preflight["unseen_nfe_diagnostic_rows"] = validate_schedule_grid_coverage(
            unseen_nfe_diagnostic_rows,
            schedule_grids=schedule_grids,
            reference_time_grid=reference_time_grid,
            label="Unseen-NFE diagnostic",
        )
    selector_density_normalizer = DensityFeatureNormalizer.fit(
        (
            density_mass_for_row(row, schedule_grids=schedule_grids, reference_time_grid=reference_time_grid)
            for row in selector_fit_rows
        ),
        reference_time_grid=reference_time_grid,
    )
    final_density_normalizer = DensityFeatureNormalizer.fit(
        (
            density_mass_for_row(row, schedule_grids=schedule_grids, reference_time_grid=reference_time_grid)
            for row in fit_rows
        ),
        reference_time_grid=reference_time_grid,
    )
    density_normalizer = final_density_normalizer
    student_teacher_score_weight = float(getattr(args, "student_teacher_score_weight", DEFAULT_STUDENT_TEACHER_SCORE_WEIGHT))
    if not np.isfinite(student_teacher_score_weight) or student_teacher_score_weight < 0.0:
        raise ValueError("student_teacher_score_weight must be finite and nonnegative.")
    student_teacher_score_warmup_fraction = float(
        getattr(args, "student_teacher_score_warmup_fraction", DEFAULT_STUDENT_TEACHER_SCORE_WARMUP_FRACTION)
    )
    if (
        not np.isfinite(student_teacher_score_warmup_fraction)
        or student_teacher_score_warmup_fraction < 0.0
        or student_teacher_score_warmup_fraction >= 1.0
    ):
        raise ValueError("student_teacher_score_warmup_fraction must be finite and in [0, 1).")
    student_teacher_score_include_pseudo = bool(getattr(args, "student_teacher_score_include_pseudo", False))
    student_target_mixture_mode = validate_student_target_mixture_mode(
        str(getattr(args, "student_target_mixture_mode", DEFAULT_STUDENT_TARGET_MIXTURE_MODE))
    )
    student_target_elite_fraction = float(getattr(args, "student_target_elite_fraction", DEFAULT_STUDENT_TARGET_ELITE_FRACTION))
    student_target_elite_k = int(getattr(args, "student_target_elite_k", DEFAULT_STUDENT_TARGET_ELITE_K))
    student_target_elite_min_count = int(getattr(args, "student_target_elite_min_count", DEFAULT_STUDENT_TARGET_ELITE_MIN_COUNT))
    student_target_elite_blend_all_weight = float(
        getattr(args, "student_target_elite_blend_all_weight", DEFAULT_STUDENT_TARGET_ELITE_BLEND_ALL_WEIGHT)
    )
    pseudo_rows: List[Dict[str, Any]] = []
    pseudo_embeddings: Dict[str, Any] | None = None
    pseudo_schedule_grids: Dict[Tuple[str, str, int], Tuple[float, ...]] | None = None
    pseudo_target_weight = float(args.student_pseudo_target_weight)
    if not torch.isfinite(torch.tensor(pseudo_target_weight, dtype=torch.float64)) or pseudo_target_weight < 0.0:
        raise ValueError("student_pseudo_target_weight must be finite and nonnegative.")
    pseudo_source_rows: List[Dict[str, Any]] = []
    pseudo_support_keys: Tuple[str, ...] = ()
    pseudo_context_sampling: Dict[str, Any] = {}
    if str(args.student_pseudo_rows_csv).strip():
        raw_pseudo_rows = _read_metric_rows_csvs(str(args.student_pseudo_rows_csv))
        if not raw_pseudo_rows:
            raise ValueError("student_pseudo_rows_csv contains no rows.")
        _validate_unique_schedule_rows(raw_pseudo_rows, label="Student pseudo distillation rows")
        _validate_context_embedding_checkpoint_scope(raw_pseudo_rows, label="Student pseudo distillation rows")
        _validate_train_tuning_rows(raw_pseudo_rows, label="Student pseudo distillation")
        pseudo_filtered_rows = _rows_for_target_nfes(raw_pseudo_rows, pseudo_target_nfes)
        if not pseudo_filtered_rows:
            raise ValueError(
                "student_pseudo_rows_csv has no rows after filtering to pseudo_target_nfe_values "
                f"{pseudo_target_nfes}."
            )
        pseudo_support_keys = _observed_support(pseudo_filtered_rows)
        missing_pseudo_support = sorted(set(support_keys) - set(pseudo_support_keys))
        extra_pseudo_support = sorted(set(pseudo_support_keys) - set(support_keys))
        if missing_pseudo_support or extra_pseudo_support:
            raise ValueError(
                "Student pseudo distillation rows must contain the same support schedule universe as seen rows; "
                f"missing={missing_pseudo_support}, extra={extra_pseudo_support}."
            )
        if _needs_forecast_uniform_rewards(pseudo_filtered_rows, teacher_metric_target_keys):
            pseudo_source_rows = attach_uniform_gipo_rewards(
                pseudo_filtered_rows,
                support_schedule_keys=support_keys,
                utility_crps_weight=float(args.teacher_utility_crps_weight),
                utility_mase_weight=float(args.teacher_utility_mase_weight),
                pair_on_seed=True,
            )
        else:
            pseudo_source_rows = [dict(row) for row in pseudo_filtered_rows]
            for row in pseudo_source_rows:
                row["context_id"] = context_id_from_row(row)
        _validate_support_group_counts(pseudo_source_rows, support_keys)
        missing_pseudo_columns = _missing_target_value_keys(pseudo_source_rows, teacher_metric_target_keys)
        if len(missing_pseudo_columns) == len(teacher_metric_target_keys):
            raise ValueError(f"Student pseudo distillation rows are missing all metric target columns: {missing_pseudo_columns}")
        pseudo_metric_coverage = _teacher_metric_coverage_summary(
            pseudo_source_rows,
            teacher_metric_target_keys,
            scope="student_pseudo_rows_csv",
        )
        _validate_teacher_metric_coverage(
            pseudo_metric_coverage,
            min_coverage_fraction=teacher_metric_min_coverage_fraction,
            min_valid_rows=teacher_metric_min_valid_rows,
        )
        teacher_metric_coverage_scopes["student_pseudo_rows_csv"] = pseudo_metric_coverage
        pseudo_context_keys = _sample_context_keys_by_checkpoint(
            pseudo_source_rows,
            sample_count=sample_count,
            seed=int(args.seed) + 808,
        )
        pseudo_rows = _rows_for_context_keys(pseudo_source_rows, pseudo_context_keys)
        pseudo_context_sampling = _context_sampling_summary(
            pseudo_source_rows,
            pseudo_context_keys,
            sample_count=sample_count,
        )
        pseudo_embeddings_path = str(args.student_pseudo_context_embeddings_npz).strip() or str(args.context_embeddings_npz)
        raw_pseudo_embeddings = _load_context_embedding_tables(pseudo_embeddings_path)
        missing_pseudo_embeddings = sorted(_embedding_ids(pseudo_rows) - set(raw_pseudo_embeddings))
        if missing_pseudo_embeddings:
            raise KeyError(f"Pseudo distillation context embeddings are missing contexts: {missing_pseudo_embeddings[:8]}")
        _assert_embedding_overlap_compatible(context_embeddings, raw_pseudo_embeddings, label="student_pseudo")
        pseudo_embeddings = embedding_normalizer.transform_table(raw_pseudo_embeddings)
        pseudo_schedule_grids = dict(schedule_grids)
        pseudo_schedule_summary_paths = _parse_csv(str(args.student_pseudo_schedule_summary_json))
        if pseudo_schedule_summary_paths:
            pseudo_schedule_grids.update(load_schedule_summary_grids(pseudo_schedule_summary_paths))
        schedule_grid_preflight["student_pseudo_rows"] = validate_schedule_grid_coverage(
            pseudo_rows,
            schedule_grids=pseudo_schedule_grids,
            reference_time_grid=reference_time_grid,
            label="Pseudo distillation",
        )
    student_training_mode = (
        STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO
        if pseudo_rows and pseudo_target_weight > 0.0
        else STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT
    )
    student_selector_fit_rows = [dict(row) for row in fit_rows]
    student_selector_validation_rows: List[Dict[str, Any]] = []
    if student_selection_mode == STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE:
        context_split_preflight["student_validation_holdout"] = _validate_context_split_capacity(
            fit_rows,
            holdout_fraction=float(args.student_selection_holdout_fraction),
            label="Student validation holdout",
        )
        student_selector_fit_rows, student_selector_validation_rows = split_rows_by_context_holdout(
            fit_rows,
            holdout_fraction=float(args.student_selection_holdout_fraction),
            seed=int(args.seed) + 20_000,
        )
        if not student_selector_fit_rows or not student_selector_validation_rows:
            raise ValueError("student validation CE checkpoint selection requires non-empty selector fit and validation rows.")

    density_meta = density_metadata(reference_time_grid)
    base_transformer_model_config = {
        "hidden_dim": int(args.transformer_hidden_dim),
        "hidden_layers": int(args.transformer_layers),
        "attention_heads": int(args.transformer_heads),
        "dropout": float(args.transformer_dropout),
        "series_conditioning": SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
    }

    def _transformer_model_config_for(normalizer: DensityFeatureNormalizer) -> Dict[str, Any]:
        return {
            **base_transformer_model_config,
            "density_feature_mean": normalizer.mean.astype(float).tolist(),
            "density_feature_std": normalizer.std.astype(float).tolist(),
        }

    transformer_model_config = _transformer_model_config_for(final_density_normalizer)
    teacher_transformer_model_config = {
        **transformer_model_config,
        "conditioning_style": conditioning_style,
        "teacher_metric_targets": list(teacher_metric_target_keys),
        "teacher_metric_target_protocol": TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
        "teacher_metric_mask_protocol": TEACHER_METRIC_MASK_PROTOCOL,
        "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
    }
    selector_teacher_transformer_model_config = {
        **_transformer_model_config_for(selector_density_normalizer),
        "conditioning_style": conditioning_style,
        "teacher_metric_targets": list(teacher_metric_target_keys),
        "teacher_metric_target_protocol": TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
        "teacher_metric_mask_protocol": TEACHER_METRIC_MASK_PROTOCOL,
        "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
    }
    student_transformer_model_config = {
        **transformer_model_config,
        "conditioning_style": conditioning_style,
    }

    def _build_teacher_instance(seed_offset: int = 0, model_config: Mapping[str, Any] | None = None):
        seed_all(int(args.seed) + int(seed_offset))
        return build_gipo_teacher_model(
            architecture=ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
            setting_dim=setting_dim,
            density_dim=int(len(reference_time_grid) - 1),
            context_dim=context_dim,
            num_series=len(series_index_map),
            model_config=model_config or teacher_transformer_model_config,
        )

    def _build_student_instance(seed_offset: int = 0):
        seed_all(int(args.seed) + int(seed_offset))
        return build_gipo_student_model(
            architecture=ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
            setting_dim=setting_dim,
            density_dim=int(len(reference_time_grid) - 1),
            context_dim=context_dim,
            num_series=len(series_index_map),
            model_config=student_transformer_model_config,
        )

    active_selection_axes = []
    if context_holdout_diagnostic_rows:
        active_selection_axes.append("context")
    if density_family_diagnostic_rows:
        active_selection_axes.append("density_family")
    if unseen_nfe_diagnostic_rows:
        active_selection_axes.append("unseen_nfe")
    teacher_selection_weight_metadata = _teacher_selection_weight_summary(
        selection_component_weights,
        active_axes=active_selection_axes,
    )
    rank_pair_preflight = {
        "selector_fit_rows": teacher_rank_pair_diagnostics(
            selector_fit_rows,
            target_keys=teacher_metric_target_keys,
            teacher_utility_weights=teacher_utility_weights,
            pair_margin=CANONICAL_TEACHER_PAIR_MARGIN,
        ),
        "final_fit_rows": teacher_rank_pair_diagnostics(
            final_fit_rows,
            target_keys=teacher_metric_target_keys,
            teacher_utility_weights=teacher_utility_weights,
            pair_margin=CANONICAL_TEACHER_PAIR_MARGIN,
        ),
    }
    _validate_rank_pair_preflight(rank_pair_preflight)
    normalizer_fit_scopes = {
        "protocol": GIPO_PREPROCESSING_PROTOCOL,
        "selector": {
            "embedding": _normalizer_fit_scope_summary("selector_fit_rows", selector_fit_rows),
            "density_feature": _normalizer_fit_scope_summary("selector_fit_rows", selector_fit_rows),
        },
        "final": {
            "embedding": _normalizer_fit_scope_summary("final_fit_rows", final_fit_rows),
            "density_feature": _normalizer_fit_scope_summary("final_fit_rows", final_fit_rows),
        },
    }
    teacher_metric_target_coverage = {
        "protocol": GIPO_TEACHER_TARGET_COVERAGE_PROTOCOL,
        "thresholds": {
            "min_coverage_fraction": float(teacher_metric_min_coverage_fraction),
            "min_valid_rows": int(teacher_metric_min_valid_rows),
        },
        "scopes": teacher_metric_coverage_scopes,
    }
    gipo_protocol_metadata = {
        "protocol_revision": "gipo_checkpoint_aware_cells_physical_context_holdout_v1",
        "preprocessing_protocol": GIPO_PREPROCESSING_PROTOCOL,
        "teacher_target_coverage_protocol": GIPO_TEACHER_TARGET_COVERAGE_PROTOCOL,
        "teacher_selection_weight_protocol": GIPO_TEACHER_SELECTION_WEIGHT_PROTOCOL,
        "model_payload_version": int(MODEL_PAYLOAD_VERSION),
    }

    teacher = _build_teacher_instance(0)
    student = _build_student_instance(10_000)
    teacher_model_config = teacher.model_config()
    student_model_config = student.model_config()

    summary_base: Dict[str, Any] = {
        "artifact": "gipo_training_summary",
        "protocol": GIPO_PROTOCOL,
        "student_policy_type": "continuous_density",
        "student_objective": "teacher_weighted_density_mle_kl_plus_teacher_score"
        if student_teacher_score_weight > 0.0
        else "teacher_weighted_density_mle_kl",
        "student_target_protocol": STUDENT_TARGET_PROTOCOL_SOFT_MIXTURE,
        "student_target_mixture_mode": student_target_mixture_mode,
        "student_target_elite_fraction": float(student_target_elite_fraction),
        "student_target_elite_k": int(student_target_elite_k),
        "student_target_elite_min_count": int(student_target_elite_min_count),
        "student_target_elite_blend_all_weight": float(student_target_elite_blend_all_weight),
        "student_teacher_score_enabled": bool(student_teacher_score_weight > 0.0),
        "student_teacher_score_weight": float(student_teacher_score_weight),
        "student_teacher_score_warmup_fraction": float(student_teacher_score_warmup_fraction),
        "student_teacher_score_schedule_steps": int(args.student_steps),
        "student_teacher_score_clip": float(DEFAULT_STUDENT_TEACHER_SCORE_CLIP),
        "student_teacher_score_protocol": "late_ramped_per_cell_teacher_utility_z_score",
        "student_teacher_score_include_pseudo": bool(student_teacher_score_include_pseudo),
        "student_regularizers": {
            "smooth": False,
            "guard": False,
        },
        "teacher_objective": "pairwise_rank_plus_huber_regression",
        "model_payload_version": MODEL_PAYLOAD_VERSION,
        "teacher_architecture": ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
        "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
        "teacher_model_config": teacher_model_config,
        "student_model_config": student_model_config,
        "teacher_metric_targets": list(teacher_metric_target_keys),
        "teacher_metric_target_protocol": TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
        "teacher_metric_mask_protocol": TEACHER_METRIC_MASK_PROTOCOL,
        "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
        "teacher_utility_weights": teacher_utility_weights,
        "teacher_checkpoint_selection_mode": selection_mode,
        "teacher_selection_axis_weights": dict(teacher_selection_weight_metadata["nominal_axis_weights"]),
        "teacher_selection_nominal_axis_weights": dict(teacher_selection_weight_metadata["nominal_axis_weights"]),
        "teacher_selection_effective_axis_weights": dict(teacher_selection_weight_metadata["effective_axis_weights"]),
        "teacher_selection_inactive_axes": list(teacher_selection_weight_metadata["inactive_axes"]),
        "teacher_selection_active_axes": list(teacher_selection_weight_metadata["active_axes"]),
        "teacher_selection_weight_protocol": GIPO_TEACHER_SELECTION_WEIGHT_PROTOCOL,
        "teacher_metric_target_coverage": teacher_metric_target_coverage,
        "teacher_loss_log_every": int(args.teacher_loss_log_every),
        "teacher_checkpoint_every": int(args.teacher_checkpoint_every),
        "student_checkpoint_selection_mode": student_selection_mode,
        "student_log_every": int(args.student_log_every),
        "student_checkpoint_every": int(args.student_checkpoint_every),
        "student_selection_holdout_fraction": float(args.student_selection_holdout_fraction),
        "conditioning_style": conditioning_style,
        "student_weight_decay": float(args.student_weight_decay),
        "series_conditioning": SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
        "setting_feature_mode": setting_feature_mode,
        "setting_encoder_mode": setting_encoder_config.mode,
        "setting_encoder_config": setting_encoder_config.to_payload(),
        "density_representation": density_meta,
        "normalizer_fit_scopes": normalizer_fit_scopes,
        "gipo_protocol_metadata": gipo_protocol_metadata,
        "context_split_preflight": context_split_preflight,
        "rank_pair_preflight": rank_pair_preflight,
        "schedule_grid_preflight": schedule_grid_preflight,
        "support_schedule_keys": list(support_keys),
        "canonical_seen_nfes": [int(value) for value in CANONICAL_SEEN_NFES],
        "canonical_unseen_nfes": [int(value) for value in CANONICAL_UNSEEN_NFES],
        "seen_target_nfe_values": [int(value) for value in seen_target_nfes],
        "pseudo_target_nfe_values": [int(value) for value in pseudo_target_nfes],
        "context_sample_count": int(sample_count),
        "context_sampling": context_sampling,
        "sampled_context_count": int(context_sampling["selected_physical_context_count"]),
        "sampled_checkpoint_context_count": int(
            sum(item["selected_contexts"] for item in context_sampling["per_checkpoint"].values())
        ),
        "student_training_mode": student_training_mode,
        "student_pseudo_distillation": {
            "enabled": bool(pseudo_rows) and pseudo_target_weight > 0.0,
            "pseudo_target_weight": float(pseudo_target_weight),
            "target_nfes": [int(value) for value in pseudo_target_nfes],
            "raw_csv": _artifact_input_summary(str(args.student_pseudo_rows_csv)),
            "context_embeddings_npz": _artifact_input_summary(str(args.student_pseudo_context_embeddings_npz or args.context_embeddings_npz)),
            "schedule_summary_json": _artifact_input_summary(str(args.student_pseudo_schedule_summary_json)),
            "source_row_count": int(len(pseudo_source_rows)),
            "selected_row_count": int(len(pseudo_rows)),
            "selected_context_count": int(len({context_id_from_row(row) for row in pseudo_rows})),
            "selected_checkpoint_context_count": int(len({_checkpoint_context_key(row) for row in pseudo_rows})),
            "context_sampling": pseudo_context_sampling,
            "support_schedule_keys": list(pseudo_support_keys),
            "required_support_schedule_keys": list(support_keys),
            "student_target_mixture_mode": student_target_mixture_mode,
            "student_teacher_score_include_pseudo": bool(student_teacher_score_include_pseudo),
            "used_for_teacher_fitting": False,
            "used_for_teacher_selection": False,
            "locked_test_used_for_pseudo": False,
        },
        "density_family_holdout": {
            **density_holdout_metadata,
            "density_family_holdout_requested_schedule_keys": list(density_holdout_requested),
            "density_family_holdout_missing_requested_schedule_keys": list(missing_density_holdout_keys),
            "selection_support_schedule_keys": list(selection_support_schedule_keys),
            "density_family_diagnostic_row_count": int(len(density_family_diagnostic_rows)),
        },
        "unseen_nfe_selection": {
            "protocol": "unseen_train_tuning_selection_diagnostic",
            "enabled": bool(unseen_selection_rows),
            "target_nfes": [int(value) for value in unseen_selection_target_nfes],
            "raw_csv": _artifact_input_summary(str(args.teacher_unseen_selection_rows_csv)),
            "context_embeddings_npz": _artifact_input_summary(str(args.teacher_unseen_selection_context_embeddings_npz or args.context_embeddings_npz)),
            "schedule_summary_json": _artifact_input_summary(str(args.teacher_unseen_selection_schedule_summary_json)),
            "source_row_count": int(len(unseen_selection_rows)),
            "selected_checkpoint_context_count": int(len({_checkpoint_context_key(row) for row in unseen_selection_rows})),
            "context_sampling": unseen_context_sampling,
            "diagnostic": _split_counts(unseen_nfe_diagnostic_rows),
            "excluded_density_holdout_schedule_keys": list(density_holdout_keys),
            "used_for_final_fitting": False,
            "locked_test_used_for_selection": False,
        },
        "split_counts": {
            "fit": _split_counts(fit_rows),
            "final_fit": _split_counts(final_fit_rows),
            "selector_fit": _split_counts(selector_fit_rows),
            "context_disjoint": _split_counts(context_holdout_rows),
            "context_disjoint_diagnostic": _split_counts(context_holdout_diagnostic_rows),
            "density_family_holdout": _split_counts(density_holdout_rows),
            "density_family_diagnostic": _split_counts(density_family_diagnostic_rows),
            "unseen_nfe_selection": _split_counts(unseen_selection_rows),
            "unseen_nfe_diagnostic": _split_counts(unseen_nfe_diagnostic_rows),
            "student_pseudo": _split_counts(pseudo_rows),
            "student_selector_fit": _split_counts(student_selector_fit_rows),
            "student_selector_validation": _split_counts(student_selector_validation_rows),
        },
        "split_membership": {
            "fit": _split_membership_summary(fit_rows),
            "final_fit": _split_membership_summary(final_fit_rows),
            "selector_fit": _split_membership_summary(selector_fit_rows),
            "context_disjoint": _split_membership_summary(context_holdout_rows),
            "context_disjoint_diagnostic": _split_membership_summary(context_holdout_diagnostic_rows),
            "density_family_holdout": _split_membership_summary(density_holdout_rows),
            "density_family_diagnostic": _split_membership_summary(density_family_diagnostic_rows),
            "unseen_nfe_selection": _split_membership_summary(unseen_selection_rows),
            "unseen_nfe_diagnostic": _split_membership_summary(unseen_nfe_diagnostic_rows),
            "student_pseudo": _split_membership_summary(pseudo_rows),
            "student_selector_fit": _split_membership_summary(student_selector_fit_rows),
            "student_selector_validation": _split_membership_summary(student_selector_validation_rows),
        },
        "nfe_sequence_diagnostics": {
            "raw_rows": nfe_sequence_diagnostic_summary(rewarded_rows),
            "sampled_rows": nfe_sequence_diagnostic_summary(sampled_rows),
            "fit_rows": nfe_sequence_diagnostic_summary(fit_rows),
            "selector_fit_rows": nfe_sequence_diagnostic_summary(selector_fit_rows),
            "unseen_nfe_selection_rows": nfe_sequence_diagnostic_summary(unseen_selection_rows),
            "unseen_nfe_diagnostic_rows": nfe_sequence_diagnostic_summary(unseen_nfe_diagnostic_rows),
            "student_pseudo_rows": nfe_sequence_diagnostic_summary(pseudo_rows),
            "student_selector_fit_rows": nfe_sequence_diagnostic_summary(student_selector_fit_rows),
            "student_selector_validation_rows": nfe_sequence_diagnostic_summary(student_selector_validation_rows),
        },
        "locked_test_used_for_selection": False,
    }

    out_dir = resolve_project_path(str(args.out_dir))
    if bool(args.dry_run):
        return {**summary_base, "status": "dry_run"}

    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(str(args.device))

    def _train_teacher_pass(
        *,
        seed_offset: int,
        pass_name: str,
        pass_fit_rows: Sequence[Mapping[str, Any]],
        diagnostic_splits: Mapping[str, Sequence[Mapping[str, Any]]],
        diagnostic_candidate_schedule_keys: Mapping[str, Sequence[str]] | None = None,
        steps: int,
    ) -> Dict[str, Any]:
        if not pass_fit_rows:
            raise ValueError(f"{pass_name} teacher pass requires non-empty fit rows.")
        active_diagnostics = {name: [dict(row) for row in split] for name, split in diagnostic_splits.items() if split}
        pass_teacher = _build_teacher_instance(seed_offset, selector_teacher_transformer_model_config)
        return train_gipo_teacher(
            pass_teacher,
            pass_fit_rows,
            context_embeddings=selector_normalized_embeddings,
            series_index_map=series_index_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference_time_grid,
            density_normalizer=selector_density_normalizer,
            steps=int(steps),
            lr=float(args.teacher_lr),
            rank_temperature=CANONICAL_TEACHER_RANK_TEMPERATURE,
            regression_weight=CANONICAL_TEACHER_REGRESSION_WEIGHT,
            pair_margin=CANONICAL_TEACHER_PAIR_MARGIN,
            diagnostic_splits=active_diagnostics,
            diagnostic_candidate_schedule_keys=diagnostic_candidate_schedule_keys,
            teacher_checkpoint_every=int(args.teacher_checkpoint_every),
            teacher_loss_log_every=int(args.teacher_loss_log_every),
            teacher_selection_axis_weights=selection_component_weights,
            seed=int(args.seed) + int(seed_offset),
            allowed_schedule_keys=support_keys,
            setting_feature_mode=setting_feature_mode,
            setting_encoder_config=setting_encoder_config,
            teacher_utility_weights=teacher_utility_weights,
            device=device,
        )

    context_density_diagnostics = {
        "context_disjoint": context_holdout_diagnostic_rows,
        "density_family_holdout": density_family_diagnostic_rows,
    }
    if not context_holdout_diagnostic_rows:
        raise ValueError("weighted_normalized_regret selection requires non-empty context_disjoint diagnostic rows.")
    if not density_family_diagnostic_rows:
        raise ValueError("weighted_normalized_regret selection requires non-empty density_family_holdout diagnostic rows.")
    context_density_training = _train_teacher_pass(
        seed_offset=101,
        pass_name="context_density_selector",
        pass_fit_rows=selector_fit_rows,
        diagnostic_splits=context_density_diagnostics,
        diagnostic_candidate_schedule_keys={
            "context_disjoint": selection_support_schedule_keys,
            "density_family_holdout": density_holdout_keys,
        },
        steps=int(args.teacher_steps),
    )
    unseen_nfe_training: Dict[str, Any] = {}
    if unseen_nfe_diagnostic_rows:
        unseen_nfe_training = _train_teacher_pass(
            seed_offset=102,
            pass_name="unseen_nfe_selector",
            pass_fit_rows=selector_fit_rows,
            diagnostic_splits={"unseen_nfe_holdout": unseen_nfe_diagnostic_rows},
            diagnostic_candidate_schedule_keys={"unseen_nfe_holdout": selection_support_schedule_keys},
            steps=int(args.teacher_steps),
        )
        checkpoint_selection = _select_weighted_normalized_regret_step(
            context_density_training=context_density_training,
            component_weights=selection_component_weights,
            unseen_nfe_training=unseen_nfe_training,
        )
    else:
        checkpoint_selection = _select_context_density_regret_step(
            context_density_training=context_density_training,
            component_weights=selection_component_weights,
        )
    selected_step = int(checkpoint_selection["selected_step"])
    teacher = _build_teacher_instance(0)
    final_teacher_training = train_gipo_teacher(
        teacher,
        fit_rows,
        context_embeddings=normalized_embeddings,
        series_index_map=series_index_map,
        schedule_grids=schedule_grids,
        reference_time_grid=reference_time_grid,
        density_normalizer=density_normalizer,
        steps=selected_step,
        lr=float(args.teacher_lr),
        rank_temperature=CANONICAL_TEACHER_RANK_TEMPERATURE,
        regression_weight=CANONICAL_TEACHER_REGRESSION_WEIGHT,
        pair_margin=CANONICAL_TEACHER_PAIR_MARGIN,
        diagnostic_splits={},
        teacher_checkpoint_every=int(args.teacher_checkpoint_every),
        teacher_loss_log_every=int(args.teacher_loss_log_every),
        final_retrain_mode=True,
        seed=int(args.seed),
        allowed_schedule_keys=support_keys,
        setting_feature_mode=setting_feature_mode,
        setting_encoder_config=setting_encoder_config,
        teacher_utility_weights=teacher_utility_weights,
        device=device,
    )
    final_retrain_metadata = {
        "enabled": True,
        "selected_step": int(selected_step),
        "fit_source": "all_sampled_non_locked_seen_calibration_rows",
        "fit_row_count": int(len(fit_rows)),
        "fit_context_count": int(len({context_id_from_row(row) for row in fit_rows})),
        "fit_schedule_count": int(len({str(row["scheduler_key"]) for row in fit_rows})),
        "unseen_selection_diagnostics_used": bool(unseen_nfe_training),
        "locked_test_used_for_selection": False,
    }
    teacher_training = {
        **final_teacher_training,
        "teacher_checkpoint_selection": checkpoint_selection,
        "teacher_checkpoint_selection_mode": selection_mode,
        "teacher_final_retrain": final_retrain_metadata,
        "final_teacher_retrain": final_retrain_metadata,
        "teacher_selection_passes": {
            "context_density": context_density_training,
            "unseen_nfe": unseen_nfe_training,
        },
    }

    def _train_student_instance(
        student_model: torch.nn.Module,
        pass_rows: Sequence[Mapping[str, Any]],
        *,
        steps: int,
        validation_rows: Sequence[Mapping[str, Any]] | None = None,
        final_retrain_mode: bool = False,
    ) -> Dict[str, Any]:
        return train_gipo_student(
            student_model,
            teacher,
            pass_rows,
            context_embeddings=normalized_embeddings,
            series_index_map=series_index_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference_time_grid,
            density_normalizer=density_normalizer,
            steps=int(steps),
            lr=float(args.student_lr),
            teacher_temperature=float(args.teacher_temperature),
            teacher_utility_weights=teacher_utility_weights,
            setting_feature_mode=setting_feature_mode,
            setting_encoder_config=setting_encoder_config,
            student_weight_decay=float(args.student_weight_decay),
            pseudo_rows=pseudo_rows,
            pseudo_context_embeddings=pseudo_embeddings,
            pseudo_schedule_grids=pseudo_schedule_grids,
            pseudo_target_weight=float(pseudo_target_weight),
            student_teacher_score_weight=float(student_teacher_score_weight),
            student_teacher_score_warmup_fraction=float(student_teacher_score_warmup_fraction),
            student_teacher_score_schedule_steps=int(args.student_steps),
            student_teacher_score_include_pseudo=bool(student_teacher_score_include_pseudo),
            student_target_mixture_mode=student_target_mixture_mode,
            student_target_elite_fraction=float(student_target_elite_fraction),
            student_target_elite_k=int(student_target_elite_k),
            student_target_elite_min_count=int(student_target_elite_min_count),
            student_target_elite_blend_all_weight=float(student_target_elite_blend_all_weight),
            validation_rows=validation_rows,
            validation_context_embeddings=normalized_embeddings,
            student_log_every=int(args.student_log_every),
            student_checkpoint_every=int(args.student_checkpoint_every),
            final_retrain_mode=bool(final_retrain_mode),
            device=device,
        )

    selector_student = _build_student_instance(10_000)
    student_selection_training = _train_student_instance(
        selector_student,
        student_selector_fit_rows,
        steps=int(args.student_steps),
        validation_rows=student_selector_validation_rows,
    )
    student_checkpoint_selection = dict(student_selection_training.get("student_checkpoint_selection", {}) or {})
    selected_student_step = int(student_checkpoint_selection.get("selected_step") or 0)
    if selected_student_step <= 0:
        raise ValueError("student validation CE checkpoint selection did not produce a positive selected step.")
    student = _build_student_instance(10_000)
    student_training = _train_student_instance(
        student,
        fit_rows,
        steps=selected_student_step,
        final_retrain_mode=True,
    )
    student_final_retrain = {
        "enabled": True,
        "performed": True,
        "protocol": "gipo_student_final_retrain",
        "selection_protocol": STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE,
        "selected_step": int(selected_student_step),
        "selector_max_steps": int(args.student_steps),
        "selector_fit_row_count": int(len(student_selector_fit_rows)),
        "selector_fit_context_count": int(len({context_id_from_row(row) for row in student_selector_fit_rows})),
        "selector_validation_row_count": int(len(student_selector_validation_rows)),
        "selector_validation_context_count": int(len({context_id_from_row(row) for row in student_selector_validation_rows})),
        "final_fit_source": "all_eligible_train_tuning_rows_after_teacher_selection",
        "final_fit_row_count": int(len(fit_rows)),
        "final_fit_context_count": int(len({context_id_from_row(row) for row in fit_rows})),
        "student_selection_holdout_fraction": float(args.student_selection_holdout_fraction),
        "locked_test_used_for_selection": False,
    }
    student_training = {
        **student_training,
        "student_checkpoint_selection_mode": STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE,
        "student_checkpoint_selection": student_checkpoint_selection,
        "student_selection_pass": student_selection_training,
        "student_selector_training": student_selection_training,
        "student_validation_split": {
            "protocol": "context_disjoint_student_validation",
            "fit_row_count": int(len(student_selector_fit_rows)),
            "fit_context_count": int(len({context_id_from_row(row) for row in student_selector_fit_rows})),
            "validation_row_count": int(len(student_selector_validation_rows)),
            "validation_context_count": int(len({context_id_from_row(row) for row in student_selector_validation_rows})),
            "holdout_fraction": float(args.student_selection_holdout_fraction),
            "locked_test_used_for_selection": False,
        },
        "student_validation_used_for_selection": True,
        "locked_test_used_for_selection": False,
        "student_final_retrain": student_final_retrain,
    }

    student_objective_settings = {
        "student_objective": student_training.get("student_objective", summary_base["student_objective"]),
        "student_target_protocol": STUDENT_TARGET_PROTOCOL_SOFT_MIXTURE,
        "student_target_mixture_mode": student_target_mixture_mode,
        "student_target_elite_fraction": float(student_target_elite_fraction),
        "student_target_elite_k": int(student_target_elite_k),
        "student_target_elite_min_count": int(student_target_elite_min_count),
        "student_target_elite_blend_all_weight": float(student_target_elite_blend_all_weight),
        "student_teacher_score_enabled": bool(student_teacher_score_weight > 0.0),
        "student_teacher_score_weight": float(student_teacher_score_weight),
        "student_teacher_score_warmup_fraction": float(student_teacher_score_warmup_fraction),
        "student_teacher_score_schedule_steps": int(args.student_steps),
        "student_teacher_score_clip": float(DEFAULT_STUDENT_TEACHER_SCORE_CLIP),
        "student_teacher_score_protocol": "late_ramped_per_cell_teacher_utility_z_score",
        "student_teacher_score_include_pseudo": bool(student_teacher_score_include_pseudo),
        "student_regularizers": {
            "smooth": False,
            "guard": False,
        },
    }

    teacher_path = out_dir / "gipo_teacher.pt"
    student_path = out_dir / "gipo_student.pt"
    torch.save(
        {
            "protocol": GIPO_PROTOCOL,
            "model_payload_version": MODEL_PAYLOAD_VERSION,
            "teacher_architecture": ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
            "teacher_model_config": teacher_model_config,
            "teacher_state": teacher.state_dict(),
            "setting_dim": int(setting_dim),
            "setting_feature_mode": setting_feature_mode,
            "setting_encoder_mode": setting_encoder_config.mode,
            "setting_encoder_config": setting_encoder_config.to_payload(),
            "density_dim": int(len(reference_time_grid) - 1),
            "context_dim": int(context_dim),
            "series_index_map": dict(series_index_map),
            "embedding_normalizer": embedding_normalizer.to_payload(),
            "density_feature_normalizer": density_normalizer.to_payload(),
            "density_representation": density_meta,
            "normalizer_fit_scopes": normalizer_fit_scopes,
            "gipo_protocol_metadata": gipo_protocol_metadata,
            "support_schedule_keys": list(support_keys),
            "seen_target_nfe_values": [int(value) for value in seen_target_nfes],
            "pseudo_target_nfe_values": [int(value) for value in pseudo_target_nfes],
            "teacher_utility_weights": teacher_utility_weights,
            "teacher_metric_target_coverage": teacher_metric_target_coverage,
            "teacher_selection_nominal_axis_weights": dict(teacher_selection_weight_metadata["nominal_axis_weights"]),
            "teacher_selection_effective_axis_weights": dict(teacher_selection_weight_metadata["effective_axis_weights"]),
            "teacher_selection_inactive_axes": list(teacher_selection_weight_metadata["inactive_axes"]),
            "teacher_training": teacher_training,
            "student_objective_settings": student_objective_settings,
            "student_target_summary": student_training.get("student_target_summary", {}),
            "conditioning_style": conditioning_style,
            "teacher_checkpoint_selection_mode": selection_mode,
            "teacher_checkpoint_selection": teacher_training.get("teacher_checkpoint_selection", {}),
            "student_checkpoint_selection_mode": student_selection_mode,
            "student_checkpoint_selection": student_training.get("student_checkpoint_selection", {}),
            "student_training_mode": student_training_mode,
            "student_pseudo_distillation": summary_base["student_pseudo_distillation"],
            "student_final_retrain": student_training.get("student_final_retrain", {}),
            "teacher_final_retrain": teacher_training.get("teacher_final_retrain", {}),
            "final_teacher_retrain": teacher_training.get("final_teacher_retrain", teacher_training.get("teacher_final_retrain", {})),
            "series_conditioning": SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
            "nfe_sequence_diagnostics": summary_base["nfe_sequence_diagnostics"],
            "locked_test_used_for_selection": False,
        },
        teacher_path,
    )
    torch.save(
        {
            "protocol": GIPO_PROTOCOL,
            "model_payload_version": MODEL_PAYLOAD_VERSION,
            "student_policy_type": "continuous_density",
            "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
            "student_model_config": student_model_config,
            "student_objective": student_training.get("student_objective", "teacher_weighted_density_mle_kl"),
            "student_state": student.state_dict(),
            "conditioning_style": conditioning_style,
            "setting_dim": int(setting_dim),
            "setting_feature_mode": setting_feature_mode,
            "setting_encoder_mode": setting_encoder_config.mode,
            "setting_encoder_config": setting_encoder_config.to_payload(),
            "density_dim": int(len(reference_time_grid) - 1),
            "context_dim": int(context_dim),
            "series_index_map": dict(series_index_map),
            "embedding_normalizer": embedding_normalizer.to_payload(),
            "density_feature_normalizer": density_normalizer.to_payload(),
            "density_representation": density_meta,
            "normalizer_fit_scopes": normalizer_fit_scopes,
            "gipo_protocol_metadata": gipo_protocol_metadata,
            "support_schedule_keys": list(support_keys),
            "seen_target_nfe_values": [int(value) for value in seen_target_nfes],
            "pseudo_target_nfe_values": [int(value) for value in pseudo_target_nfes],
            "teacher_utility_weights": teacher_utility_weights,
            "teacher_metric_target_coverage": teacher_metric_target_coverage,
            "teacher_selection_nominal_axis_weights": dict(teacher_selection_weight_metadata["nominal_axis_weights"]),
            "teacher_selection_effective_axis_weights": dict(teacher_selection_weight_metadata["effective_axis_weights"]),
            "teacher_selection_inactive_axes": list(teacher_selection_weight_metadata["inactive_axes"]),
            "teacher_checkpoint": teacher_path.name,
            "teacher_training": teacher_training,
            "student_training": student_training,
            "student_objective_settings": student_objective_settings,
            "student_target_summary": student_training.get("student_target_summary", {}),
            "teacher_checkpoint_selection_mode": selection_mode,
            "teacher_checkpoint_selection": teacher_training.get("teacher_checkpoint_selection", {}),
            "student_checkpoint_selection_mode": student_selection_mode,
            "student_checkpoint_selection": student_training.get("student_checkpoint_selection", {}),
            "student_training_mode": student_training_mode,
            "student_pseudo_distillation": summary_base["student_pseudo_distillation"],
            "student_final_retrain": student_training.get("student_final_retrain", {}),
            "teacher_final_retrain": teacher_training.get("teacher_final_retrain", {}),
            "final_teacher_retrain": teacher_training.get("final_teacher_retrain", teacher_training.get("teacher_final_retrain", {})),
            "series_conditioning": SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
            "nfe_sequence_diagnostics": summary_base["nfe_sequence_diagnostics"],
            "locked_test_used_for_selection": False,
        },
        student_path,
    )

    policy_id_payload = {
        "protocol": GIPO_PROTOCOL,
        "gipo_protocol_metadata": gipo_protocol_metadata,
        "reference_grid_hash": reference_grid_hash(reference_time_grid),
        "support_schedule_keys": list(support_keys),
        "context_sampling": context_sampling,
        "seen_target_nfe_values": [int(value) for value in seen_target_nfes],
        "pseudo_target_nfe_values": [int(value) for value in pseudo_target_nfes],
        "teacher_selected_step": teacher_training.get("teacher_checkpoint_selection", {}).get("selected_step"),
        "student_target_protocol": STUDENT_TARGET_PROTOCOL_SOFT_MIXTURE,
        "student_objective_settings": student_objective_settings,
        "setting_feature_mode": setting_feature_mode,
        "setting_encoder_config": setting_encoder_config.to_payload(),
        "teacher_architecture": ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
        "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
        "teacher_model_config": teacher_model_config,
        "student_model_config": student_model_config,
        "normalizer_fit_scopes": normalizer_fit_scopes,
        "conditioning_style": conditioning_style,
        "teacher_utility_weights": teacher_utility_weights,
        "teacher_metric_target_coverage": teacher_metric_target_coverage,
        "teacher_selection_nominal_axis_weights": dict(teacher_selection_weight_metadata["nominal_axis_weights"]),
        "teacher_selection_effective_axis_weights": dict(teacher_selection_weight_metadata["effective_axis_weights"]),
        "teacher_selection_inactive_axes": list(teacher_selection_weight_metadata["inactive_axes"]),
        "series_conditioning": SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
        "teacher_checkpoint_selection_mode": selection_mode,
        "student_checkpoint_selection_mode": student_selection_mode,
        "student_checkpoint_selection": student_training.get("student_checkpoint_selection", {}),
        "student_training_mode": student_training_mode,
        "student_pseudo_distillation": summary_base["student_pseudo_distillation"],
        "student_final_retrain": student_training.get("student_final_retrain", {}),
        "teacher_final_retrain": teacher_training.get("teacher_final_retrain", {}),
        "final_teacher_retrain": teacher_training.get("final_teacher_retrain", teacher_training.get("teacher_final_retrain", {})),
    }
    policy_id = "gipo_" + hashlib.sha256(
        json.dumps(policy_id_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    summary = {
        **summary_base,
        "status": "completed",
        "policy_id": policy_id,
        "gipo_teacher_checkpoint": teacher_path.name,
        "gipo_student_checkpoint": student_path.name,
        "teacher_training": teacher_training,
        "teacher_checkpoint_selection": teacher_training.get("teacher_checkpoint_selection", {}),
        "student_checkpoint_selection": student_training.get("student_checkpoint_selection", {}),
        "student_final_retrain": student_training.get("student_final_retrain", {}),
        "teacher_final_retrain": teacher_training.get("teacher_final_retrain", {}),
        "final_teacher_retrain": teacher_training.get("final_teacher_retrain", teacher_training.get("teacher_final_retrain", {})),
        "student_objective_settings": student_objective_settings,
        "student_target_summary": student_training.get("student_target_summary", {}),
        "student_training": student_training,
    }
    (out_dir / "gipo_training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    summary = train_gipo(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
