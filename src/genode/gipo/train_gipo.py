from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from genode.canonical_experiment_layout import (
    CANONICAL_CONTEXT_SAMPLE_COUNT,
    CANONICAL_PSEUDO_TARGET_WEIGHT,
    CANONICAL_SEEN_NFES,
    CANONICAL_UNSEEN_NFES,
    STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
    STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO,
)
from genode.gipo.policy import (
    GIPO_PROTOCOL,
    MODEL_PAYLOAD_VERSION,
    SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
    DEFAULT_SUPPORT_SCHEDULE_KEYS,
    DEFAULT_TEACHER_TARGET_TEMPERATURE,
    STUDENT_TARGET_PROTOCOL_SOFT_MIXTURE,
    DEFAULT_DENSITY_FAMILY_HOLDOUT_SCHEDULE_KEYS,
    DEFAULT_TEACHER_SELECTION_COMPONENT_WEIGHTS,
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
    context_id_from_row,
    context_pair_key,
    density_mass_for_row,
    density_family_for_schedule_key,
    load_context_embedding_table,
    nfe_sequence_diagnostic_summary,
    read_metric_rows_csv,
    recommended_context_calibration_count,
    sample_context_ids_stratified,
    select_weighted_normalized_regret_checkpoint,
    series_key_from_row,
    split_rows_by_context_holdout,
    split_rows_by_density_family_holdout,
    train_gipo_student,
    train_gipo_teacher,
    normalize_teacher_utility_weights,
    teacher_utility_weights_for_summary,
    validate_gipo_attention_heads,
    validate_gipo_support_schedule_keys,
    validate_teacher_metric_target_keys,
    validate_teacher_objective_hyperparameters,
)
from genode.gipo.density_representation import density_metadata, reference_grid_hash, uniform_reference_grid
from genode.gipo.density_representation import average_density_masses, density_mass_to_time_grid, grid_to_density_mass
from genode.gipo.models import (
    SETTING_ENCODER_MODE_CONTINUOUS_V3,
    setting_encoder_config_for_rows,
    setting_feature_dim,
    solver_macro_steps,
    validate_setting_feature_mode,
    validate_time_grid,
)
from genode.data.otflow_paths import resolve_project_path
from genode.models.otflow_train_val import seed_all
from genode.runtime import resolve_torch_device

CANONICAL_DENSITY_BIN_COUNT = 64
CANONICAL_TEACHER_SELECTION_COMPONENT_WEIGHTS: Dict[str, float] = {
    "context": 0.25,
    "density_family": 0.25,
    "unseen_nfe": 0.50,
}
CANONICAL_TEACHER_RANK_TEMPERATURE = 0.5
CANONICAL_TEACHER_REGRESSION_WEIGHT = 0.25
CANONICAL_TEACHER_PAIR_MARGIN = 0.0


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in _parse_csv(text)]


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
    weights = {key: float(component_weights.get(key, 0.0)) for key in DEFAULT_TEACHER_SELECTION_COMPONENT_WEIGHTS}
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("Weighted normalized-regret checkpoint selection requires a positive component weight total.")
    weights = {key: value / total for key, value in weights.items()}
    split_weights = {
        "context": float(weights["context"]),
        "density_family": float(weights["density_family"]),
        "unseen_nfe_holdout": float(weights["unseen_nfe"]),
    }
    selection = select_weighted_normalized_regret_checkpoint(
        combined_history,
        required_split_names=required_splits,
        component_weights=split_weights,
    )
    return {
        **selection,
        "selection_component_axis_weights": dict(weights),
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
    adjusted_weights = {
        "context": float(component_weights.get("context", 0.0)),
        "density_family": float(component_weights.get("density_family", 0.0)),
    }
    total = sum(adjusted_weights.values())
    if total <= 0.0:
        raise ValueError("Context/density checkpoint selection requires positive context or density weights.")
    split_weights = {
        "context": adjusted_weights["context"] / total,
        "density_family": adjusted_weights["density_family"] / total,
    }
    selection = select_weighted_normalized_regret_checkpoint(
        [{"step": int(step), "diagnostics": dict(entry.get("diagnostics", {}) or {})} for step, entry in sorted(context_history.items())],
        required_split_names=required_splits,
        component_weights=split_weights,
    )
    return {
        **selection,
        "selection_component_axis_weights": dict(split_weights),
        "uses_unseen_nfe_selection_diagnostics": False,
        "component_histories": {
            "context_density": context_density_training.get("teacher_checkpoint_selection", {}).get("history", []),
            "unseen_nfe": [],
        },
    }


def _load_schedule_summary_grids(paths: Sequence[str]) -> Dict[Tuple[str, str, int], Tuple[float, ...]]:
    grids: Dict[Tuple[str, str, int], Tuple[float, ...]] = {}
    for path_text in paths:
        path = resolve_project_path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Schedule summary not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        schedules = payload.get("schedules")
        if schedules:
            schedule_items = list(schedules)
        else:
            schedule_items = [
                {
                    "scheduler_key": str(payload.get("scheduler_key", payload.get("schedule_key", ""))),
                    "predictions": payload.get("predictions", []) or [],
                }
            ]
        for schedule in schedule_items:
            schedule_key = str(schedule.get("scheduler_key", schedule.get("schedule_key", ""))).strip()
            for item in list(schedule.get("predictions", []) or []):
                solver = str(item["solver_key"])
                target_nfe = int(item["target_nfe"])
                macro_steps = solver_macro_steps(solver, target_nfe)
                base_grid = validate_time_grid(item["time_grid"], macro_steps=macro_steps)
                grids[(schedule_key, solver, target_nfe)] = base_grid
                if schedule_key == "ser_ptg_local_defect_eta005":
                    reversed_grid = validate_time_grid(
                        [1.0 - float(value) for value in reversed(base_grid)],
                        macro_steps=macro_steps,
                    )
                    grids[("ser_ptg_local_defect_eta005_reversed", solver, target_nfe)] = reversed_grid
                    reference = uniform_reference_grid(CANONICAL_DENSITY_BIN_COUNT)
                    base_mass = grid_to_density_mass(base_grid, reference_time_grid=reference, macro_steps=macro_steps)
                    reversed_mass = grid_to_density_mass(reversed_grid, reference_time_grid=reference, macro_steps=macro_steps)
                    averaged_mass = average_density_masses(base_mass, reversed_mass)
                    grids[("ser_ptg_local_defect_eta005_avg_reversed", solver, target_nfe)] = density_mass_to_time_grid(
                        averaged_mass,
                        reference_time_grid=reference,
                        macro_steps=macro_steps,
                    )
    return grids


def _observed_support(rows: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    observed = tuple(sorted({str(row["scheduler_key"]) for row in rows}))
    canonical_order = tuple(key for key in DEFAULT_SUPPORT_SCHEDULE_KEYS if key in observed)
    extras = tuple(key for key in observed if key not in canonical_order)
    return validate_gipo_support_schedule_keys(canonical_order + extras)


def _stable_hash(values: Sequence[str]) -> str:
    payload = json.dumps([str(value) for value in values], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_membership_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    context_ids = sorted({context_id_from_row(row) for row in rows})
    series_keys = sorted({series_key_from_row(row) for row in rows})
    split_phases = sorted({str(row.get("split_phase", row.get("split", ""))) for row in rows})
    return {
        "row_count": int(len(rows)),
        "context_count": int(len(context_ids)),
        "series_count": int(len(series_keys)),
        "source_split_phases": split_phases,
        "context_ids": context_ids,
        "context_id_hash": _stable_hash(context_ids),
        "series_keys": series_keys,
        "series_key_hash": _stable_hash(series_keys),
    }


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    from genode.gipo.policy import series_key_from_row

    return {
        "row_count": int(len(rows)),
        "context_count": int(len({context_id_from_row(row) for row in rows})),
        "series_count": int(len({series_key_from_row(row) for row in rows})),
        "schedule_count": int(len({str(row["scheduler_key"]) for row in rows})),
    }


def _validate_support_group_counts(rows: Sequence[Mapping[str, Any]], support_schedule_keys: Sequence[str]) -> None:
    support_keys = tuple(str(key) for key in support_schedule_keys)
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GIPO continuous-density from per-example fixed/SER rows.")
    parser.add_argument("--rows_csv", required=True, help="Per-example fixed/SER metric rows CSV.")
    parser.add_argument("--context_embeddings_npz", required=True, help="Frozen context embedding sidecar NPZ.")
    parser.add_argument("--schedule_summary_json", default="", help="Comma-separated schedule summaries for non-fixed references such as SER.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--support_schedule_keys", default="", help="Comma-separated fixed/SER supervision keys. Defaults to observed row keys.")
    parser.add_argument("--context_sample_count", type=int, default=CANONICAL_CONTEXT_SAMPLE_COUNT)
    parser.add_argument("--context_holdout_fraction", type=float, default=0.20)
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
    parser.add_argument("--student_pseudo_target_weight", type=float, default=CANONICAL_PSEUDO_TARGET_WEIGHT)
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
        default="u_crps_uniform,u_mase_uniform",
        help="Comma-separated utility columns predicted by the metric-vector teacher.",
    )
    parser.add_argument(
        "--teacher_utility_weights",
        default="",
        help="Optional comma-separated name=value weights for --teacher_metric_target_keys.",
    )
    parser.add_argument("--student_weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def train_gipo(args: argparse.Namespace) -> Dict[str, Any]:
    seed_all(int(args.seed))
    density_bin_count = CANONICAL_DENSITY_BIN_COUNT
    selection_mode = TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET
    student_selection_mode = STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE
    validate_gipo_attention_heads(int(args.transformer_heads))
    conditioning_style = CONDITIONING_STYLE_ADDITIVE_MLP
    requested_setting_mode = SETTING_ENCODER_MODE_CONTINUOUS_V3
    setting_feature_mode = validate_setting_feature_mode(requested_setting_mode)
    rows = read_metric_rows_csv(resolve_project_path(str(args.rows_csv)))
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

    teacher_metric_target_keys = validate_teacher_metric_target_keys(str(args.teacher_metric_target_keys))
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
    forecast_utility_keys = {"u_crps_uniform", "u_mase_uniform"}
    missing_requested_metric_keys = {key for key in teacher_metric_target_keys if any(key not in row for row in rows)}
    if missing_requested_metric_keys & forecast_utility_keys:
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
    missing_metric_columns = sorted({key for key in teacher_metric_target_keys if any(key not in row for row in rewarded_rows)})
    if missing_metric_columns:
        raise ValueError(f"GIPO rows are missing teacher metric target columns: {missing_metric_columns}")
    available_context_ids = sorted({context_id_from_row(row) for row in rewarded_rows})
    sample_count = int(args.context_sample_count)
    if sample_count <= 0:
        sample_count = recommended_context_calibration_count(len(available_context_ids))
    selected_context_ids = set(sample_context_ids_stratified(rewarded_rows, sample_count=sample_count, seed=int(args.seed)))
    sampled_rows = [row for row in rewarded_rows if context_id_from_row(row) in selected_context_ids]

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
    canonical_seen_nfes = list(CANONICAL_SEEN_NFES)
    if sampled_target_nfes != canonical_seen_nfes:
        raise ValueError(
            "weighted_normalized_regret final teacher/student fitting expects seen calibration NFEs "
            f"{canonical_seen_nfes}; "
            f"found {sampled_target_nfes}."
        )
    unseen_selection_rows: List[Dict[str, Any]] = []
    unseen_selection_target_nfes = _parse_int_csv(str(args.teacher_unseen_selection_target_nfe_values))
    if str(args.teacher_unseen_selection_rows_csv).strip():
        unseen_raw_rows = read_metric_rows_csv(resolve_project_path(str(args.teacher_unseen_selection_rows_csv)))
        if not unseen_raw_rows:
            raise ValueError("teacher_unseen_selection_rows_csv contains no rows.")
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
        missing_unseen_metric_keys = {
            key
            for key in teacher_metric_target_keys
            if any(key not in row for row in unseen_filtered_rows)
        }
        if missing_unseen_metric_keys & forecast_utility_keys:
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
        missing_unseen_columns = sorted(
            {key for key in teacher_metric_target_keys if any(key not in row for row in unseen_rewarded_rows)}
        )
        if missing_unseen_columns:
            raise ValueError(f"Teacher unseen selection rows are missing metric target columns: {missing_unseen_columns}")
        unseen_context_ids = set(
            sample_context_ids_stratified(
                unseen_rewarded_rows,
                sample_count=min(sample_count, len({context_id_from_row(row) for row in unseen_rewarded_rows})),
                seed=int(args.seed) + 404,
            )
        )
        unseen_selection_rows = [row for row in unseen_rewarded_rows if context_id_from_row(row) in unseen_context_ids]
    density_holdout_requested = _parse_csv(str(args.teacher_density_holdout_schedule_keys))
    if "uniform" in {str(key) for key in density_holdout_requested}:
        raise ValueError("density-family holdout must not include uniform; uniform is the reward anchor.")
    support_set = set(support_keys)
    missing_density_holdout_keys = sorted(set(density_holdout_requested) - support_set)
    if missing_density_holdout_keys and tuple(density_holdout_requested) == DEFAULT_DENSITY_FAMILY_HOLDOUT_SCHEDULE_KEYS:
        density_holdout_keys: List[str] = []
    elif missing_density_holdout_keys:
        raise ValueError(f"density-family holdout keys must be in support_schedule_keys; unsupported={missing_density_holdout_keys}.")
    else:
        density_holdout_keys = list(density_holdout_requested)

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

    context_embeddings = load_context_embedding_table(resolve_project_path(str(args.context_embeddings_npz)))
    fit_rows = final_fit_rows
    fit_context_ids = sorted({context_id_from_row(row) for row in fit_rows})
    embedding_normalizer = EmbeddingNormalizer.fit(context_embeddings, fit_context_ids)
    normalized_embeddings = embedding_normalizer.transform_table(context_embeddings)
    missing_embeddings = sorted({context_id_from_row(row) for row in sampled_rows} - set(normalized_embeddings))
    if missing_embeddings:
        raise KeyError(f"Context embeddings are missing sampled contexts: {missing_embeddings[:8]}")
    if unseen_selection_rows:
        unseen_embeddings_path = str(args.teacher_unseen_selection_context_embeddings_npz).strip() or str(args.context_embeddings_npz)
        unseen_raw_embeddings = load_context_embedding_table(resolve_project_path(unseen_embeddings_path))
        missing_unseen_embeddings = sorted({context_id_from_row(row) for row in unseen_selection_rows} - set(unseen_raw_embeddings))
        if missing_unseen_embeddings:
            raise KeyError(f"Unseen selection context embeddings are missing contexts: {missing_unseen_embeddings[:8]}")
        normalized_embeddings.update(embedding_normalizer.transform_table(unseen_raw_embeddings))

    series_index_map = build_series_index_map(fit_rows)
    fit_series_keys = sorted({series_key_from_row(row) for row in fit_rows})
    context_dim = int(next(iter(normalized_embeddings.values())).shape[0])
    setting_dim = int(setting_feature_dim(setting_feature_mode, config=setting_encoder_config))
    reference_time_grid = uniform_reference_grid(density_bin_count)
    schedule_grids = _load_schedule_summary_grids(_parse_csv(str(args.schedule_summary_json)))
    unseen_schedule_summary_paths = _parse_csv(str(args.teacher_unseen_selection_schedule_summary_json))
    if unseen_schedule_summary_paths:
        schedule_grids.update(_load_schedule_summary_grids(unseen_schedule_summary_paths))
    density_normalizer = DensityFeatureNormalizer.fit(
        (
            density_mass_for_row(row, schedule_grids=schedule_grids, reference_time_grid=reference_time_grid)
            for row in fit_rows
        ),
        reference_time_grid=reference_time_grid,
    )
    density_family_diagnostic_rows = [dict(row) for row in density_holdout_rows]
    unseen_nfe_diagnostic_rows = _rows_without_schedule_keys(unseen_selection_rows, density_holdout_keys)
    pseudo_rows: List[Dict[str, Any]] = []
    pseudo_embeddings: Dict[str, Any] | None = None
    pseudo_schedule_grids: Dict[Tuple[str, str, int], Tuple[float, ...]] | None = None
    pseudo_target_weight = float(args.student_pseudo_target_weight)
    if not torch.isfinite(torch.tensor(pseudo_target_weight, dtype=torch.float64)) or pseudo_target_weight < 0.0:
        raise ValueError("student_pseudo_target_weight must be finite and nonnegative.")
    pseudo_source_rows: List[Dict[str, Any]] = []
    pseudo_support_keys: Tuple[str, ...] = ()
    if str(args.student_pseudo_rows_csv).strip():
        raw_pseudo_rows = read_metric_rows_csv(resolve_project_path(str(args.student_pseudo_rows_csv)))
        if not raw_pseudo_rows:
            raise ValueError("student_pseudo_rows_csv contains no rows.")
        _validate_train_tuning_rows(raw_pseudo_rows, label="Student pseudo distillation")
        pseudo_filtered_rows = _rows_for_target_nfes(raw_pseudo_rows, CANONICAL_UNSEEN_NFES)
        if not pseudo_filtered_rows:
            raise ValueError(
                "student_pseudo_rows_csv has no rows after filtering to canonical unseen NFEs "
                f"{list(CANONICAL_UNSEEN_NFES)}."
            )
        pseudo_source_rows = [dict(row) for row in pseudo_filtered_rows]
        for row in pseudo_source_rows:
            row["context_id"] = context_id_from_row(row)
        pseudo_support_keys = _observed_support(pseudo_source_rows)
        _validate_support_group_counts(pseudo_source_rows, pseudo_support_keys)
        pseudo_context_ids = set(
            sample_context_ids_stratified(
                pseudo_source_rows,
                sample_count=min(sample_count, len({context_id_from_row(row) for row in pseudo_source_rows})),
                seed=int(args.seed) + 808,
            )
        )
        pseudo_rows = [row for row in pseudo_source_rows if context_id_from_row(row) in pseudo_context_ids]
        pseudo_embeddings_path = str(args.student_pseudo_context_embeddings_npz).strip() or str(args.context_embeddings_npz)
        raw_pseudo_embeddings = load_context_embedding_table(resolve_project_path(pseudo_embeddings_path))
        missing_pseudo_embeddings = sorted({context_id_from_row(row) for row in pseudo_rows} - set(raw_pseudo_embeddings))
        if missing_pseudo_embeddings:
            raise KeyError(f"Pseudo distillation context embeddings are missing contexts: {missing_pseudo_embeddings[:8]}")
        pseudo_embeddings = embedding_normalizer.transform_table(raw_pseudo_embeddings)
        normalized_embeddings.update(pseudo_embeddings)
        pseudo_schedule_grids = dict(schedule_grids)
        pseudo_schedule_summary_paths = _parse_csv(str(args.student_pseudo_schedule_summary_json))
        if pseudo_schedule_summary_paths:
            pseudo_schedule_grids.update(_load_schedule_summary_grids(pseudo_schedule_summary_paths))
        missing_pseudo_grid_rows: List[str] = []
        for row in pseudo_rows:
            try:
                density_mass_for_row(row, schedule_grids=pseudo_schedule_grids, reference_time_grid=reference_time_grid)
            except (KeyError, ValueError):
                missing_pseudo_grid_rows.append(str((row["scheduler_key"], row["solver_key"], int(row["target_nfe"]))))
        if missing_pseudo_grid_rows:
            raise ValueError(f"Pseudo distillation rows are missing schedule grids: {missing_pseudo_grid_rows[:8]}")
    student_training_mode = (
        STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO
        if pseudo_rows and pseudo_target_weight > 0.0
        else STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT
    )
    student_selector_fit_rows = [dict(row) for row in fit_rows]
    student_selector_validation_rows: List[Dict[str, Any]] = []
    if student_selection_mode == STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE:
        student_selector_fit_rows, student_selector_validation_rows = split_rows_by_context_holdout(
            fit_rows,
            holdout_fraction=float(args.student_selection_holdout_fraction),
            seed=int(args.seed) + 20_000,
        )
        if not student_selector_fit_rows or not student_selector_validation_rows:
            raise ValueError("student validation CE checkpoint selection requires non-empty selector fit and validation rows.")

    density_meta = density_metadata(reference_time_grid)
    transformer_model_config = {
        "hidden_dim": int(args.transformer_hidden_dim),
        "hidden_layers": int(args.transformer_layers),
        "attention_heads": int(args.transformer_heads),
        "dropout": float(args.transformer_dropout),
        "series_conditioning": SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
        "density_feature_mean": density_normalizer.mean.astype(float).tolist(),
        "density_feature_std": density_normalizer.std.astype(float).tolist(),
    }
    teacher_transformer_model_config = {
        **transformer_model_config,
        "conditioning_style": conditioning_style,
        "teacher_metric_targets": list(teacher_metric_target_keys),
    }
    student_transformer_model_config = {
        **transformer_model_config,
        "conditioning_style": conditioning_style,
    }
    def _build_teacher_instance(seed_offset: int = 0):
        seed_all(int(args.seed) + int(seed_offset))
        return build_gipo_teacher_model(
            architecture=ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
            setting_dim=setting_dim,
            density_dim=int(len(reference_time_grid) - 1),
            context_dim=context_dim,
            num_series=len(series_index_map),
            model_config=teacher_transformer_model_config,
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

    teacher = _build_teacher_instance(0)
    student = _build_student_instance(10_000)
    teacher_model_config = teacher.model_config()
    student_model_config = student.model_config()

    summary_base: Dict[str, Any] = {
        "artifact": "gipo_training_summary",
        "protocol": GIPO_PROTOCOL,
        "student_policy_type": "continuous_density",
        "student_objective": "teacher_weighted_density_mle_kl",
        "student_target_protocol": STUDENT_TARGET_PROTOCOL_SOFT_MIXTURE,
        "teacher_objective": "pairwise_rank_plus_huber_regression",
        "model_payload_version": MODEL_PAYLOAD_VERSION,
        "teacher_architecture": ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
        "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
        "teacher_model_config": teacher_model_config,
        "student_model_config": student_model_config,
        "teacher_metric_targets": list(teacher_metric_target_keys),
        "teacher_utility_weights": teacher_utility_weights,
        "teacher_checkpoint_selection_mode": selection_mode,
        "teacher_selection_axis_weights": selection_component_weights,
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
        "support_schedule_keys": list(support_keys),
        "canonical_seen_nfes": [int(value) for value in CANONICAL_SEEN_NFES],
        "canonical_unseen_nfes": [int(value) for value in CANONICAL_UNSEEN_NFES],
        "context_sample_count": int(sample_count),
        "sampled_context_count": int(len(selected_context_ids)),
        "student_training_mode": student_training_mode,
        "student_pseudo_distillation": {
            "enabled": bool(pseudo_rows) and pseudo_target_weight > 0.0,
            "pseudo_target_weight": float(pseudo_target_weight),
            "target_nfes": [int(value) for value in CANONICAL_UNSEEN_NFES],
            "raw_csv": str(args.student_pseudo_rows_csv),
            "context_embeddings_npz": str(args.student_pseudo_context_embeddings_npz or args.context_embeddings_npz),
            "schedule_summary_json": str(args.student_pseudo_schedule_summary_json),
            "source_row_count": int(len(pseudo_source_rows)),
            "selected_row_count": int(len(pseudo_rows)),
            "selected_context_count": int(len({context_id_from_row(row) for row in pseudo_rows})),
            "support_schedule_keys": list(pseudo_support_keys),
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
            "raw_csv": str(args.teacher_unseen_selection_rows_csv),
            "context_embeddings_npz": str(args.teacher_unseen_selection_context_embeddings_npz or args.context_embeddings_npz),
            "schedule_summary_json": str(args.teacher_unseen_selection_schedule_summary_json),
            "source_row_count": int(len(unseen_selection_rows)),
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
        pass_teacher = _build_teacher_instance(seed_offset)
        return train_gipo_teacher(
            pass_teacher,
            pass_fit_rows,
            context_embeddings=normalized_embeddings,
            series_index_map=series_index_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference_time_grid,
            density_normalizer=density_normalizer,
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
            "support_schedule_keys": list(support_keys),
            "teacher_utility_weights": teacher_utility_weights,
            "teacher_training": teacher_training,
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
            "support_schedule_keys": list(support_keys),
            "teacher_utility_weights": teacher_utility_weights,
            "teacher_checkpoint": teacher_path.name,
            "teacher_training": teacher_training,
            "student_training": student_training,
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
        "reference_grid_hash": reference_grid_hash(reference_time_grid),
        "support_schedule_keys": list(support_keys),
        "teacher_selected_step": teacher_training.get("teacher_checkpoint_selection", {}).get("selected_step"),
        "student_target_protocol": STUDENT_TARGET_PROTOCOL_SOFT_MIXTURE,
        "setting_feature_mode": setting_feature_mode,
        "setting_encoder_config": setting_encoder_config.to_payload(),
        "teacher_architecture": ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
        "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
        "teacher_model_config": teacher_model_config,
        "student_model_config": student_model_config,
        "conditioning_style": conditioning_style,
        "teacher_utility_weights": teacher_utility_weights,
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
