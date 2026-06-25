from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from genode.canonical_experiment_layout import (
    CANONICAL_CHECKPOINT_STEPS,
    CANONICAL_LAYOUT_VERSION,
    CANONICAL_SEEN_NFES,
    NFE_ROLE_SEEN,
    NFE_ROLES,
    canonical_nfes_for_role,
    density_source_key_for_schedule,
    schedule_family_for_key,
    SCENARIO_FAMILY_MOLECULE,
)
from genode.data.molecule_xyz import (
    MOLECULE_GROUP_DATASET_KEYS,
    default_molecule_group_root,
    load_molecule_group_manifest,
    trainable_molecule_group_members,
)
from genode.schedule_transfer.diffusion_flow_schedules import (
    BASELINE_SCHEDULE_KEYS,
    EXPERIMENTAL_FIXED_SCHEDULE_KEYS,
    TRANSFER_SCHEDULE_KEYS,
    build_schedule_grid,
    fixed_schedule_shape_statistics,
    run_fixed_schedule_variant,
    schedule_display_name,
    schedule_time_alignment,
)
from genode.evaluation.fm_backbone_registry import (
    BACKBONE_NAME_OTFLOW_MOLECULE,
    MOLECULE_FAMILY,
    find_backbone_artifact,
    load_backbone_manifest,
)
from genode.evaluation.molecule_metrics import (
    MOLECULE_CONTEXT_SCHEMA,
    MOLECULE_PRIMARY_METRICS,
    evaluate_molecule_rollout_schedule,
    load_molecule_checkpoint_splits,
    molecule_context_embeddings_for_indices,
)
from genode.evaluation.otflow_sampling_support import _choose_valid_windows
from genode.evaluation.otflow_evaluation_support import (
    ALL_SOLVER_ORDER,
    CONDITIONAL_GENERATION_FAMILY,
    DEFAULT_CONDITIONAL_GENERATION_DATASETS,
    DEFAULT_FORECAST_DATASETS,
    DEFAULT_SHARED_BACKBONE_ROOT,
    DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION,
    DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION,
    FORECAST_FAMILY,
    LOCKED_TEST_PHASE,
    SOLVER_RUNTIME_NAMES,
    TRAIN_TUNING_PHASE,
    TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION,
    TRAIN_TUNING_SAMPLING_MODES,
    UNIFORM_SCHEDULER_KEY,
    VALIDATION_PHASE,
    choose_forecast_example_indices,
    choose_forecast_train_tuning_indices,
    evaluate_forecast_schedule,
    load_conditional_generation_checkpoint_splits,
    load_forecast_checkpoint_splits,
    parse_conditional_generation_datasets,
    parse_csv,
    parse_forecast_datasets,
    parse_int_csv,
    resolved_eval_horizon,
    resolved_eval_windows,
    selection_metric_for_family,
    solver_eval_multiplier,
    solver_experiment_scope,
    solver_macro_steps,
    train_tuning_sampler_key,
    train_tuning_target_example_count,
    validate_execution_preflight,
)
from genode.gipo.objectives import (
    CONDITIONAL_METRIC_SPECS,
    MOLECULE_METRIC_SPECS,
    teacher_metric_profile_for_scenario,
    teacher_objective_specs_for_scenario,
    uniform_anchored_objective_columns,
)
from genode.gipo.models import validate_time_grid
from genode.solver_protocol import normalize_solver_keys, normalize_solver_nfe_fields
from genode.schedule_transfer.otflow_paper_registry import METHOD_KEY
from genode.schedule_transfer.otflow_paper_tables import augment_rows_with_relative_metrics
from genode.data.otflow_paths import (
    default_backbone_manifest_path,
    default_cryptos_data_path,
    default_lobster_synthetic_profile_path,
    default_long_term_st_data_path,
    project_outputs_root,
    project_paper_dataset_root,
    project_root,
    resolve_project_path,
)
from genode.models.otflow_train_val import _get_dataset_item_by_t, _parse_batch, save_json
from genode.runtime import resolve_torch_device
from genode.gipo.policy import GIPO_PROTOCOL, load_context_embedding_table, save_context_embedding_table
from genode.gipo.schedule_hash import schedule_grid_hash

RUNNER_SIGNATURE_VERSION = "diffusion_flow_time_reparameterization_seen_unseen_phase_context_v4"
CONTEXT_REWARD_PROTOCOL_VERSION = "conditional_primary_metric_context_rewards_v2"
DEFAULT_OUT_ROOT = project_outputs_root() / "diffusion_flow_time_reparameterization"
DEFAULT_TARGET_NFE_VALUES: Tuple[int, ...] = CANONICAL_SEEN_NFES
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_SCHEDULES: Tuple[str, ...] = BASELINE_SCHEDULE_KEYS
SCHEDULE_CONTEXT_SELECTION_PROTOCOL = "schedule_evaluation_phase_context_capped_v4"
SUPPORTED_SPLIT_PHASES: Tuple[str, ...] = (LOCKED_TEST_PHASE, VALIDATION_PHASE, TRAIN_TUNING_PHASE)

ROW_RECORD_FIELDS: Tuple[str, ...] = (
    "benchmark_family",
    "canonical_layout_version",
    "scenario_key",
    "scenario_family",
    "nfe_role",
    "checkpoint_step",
    "checkpoint_maturity_label",
    "checkpoint_maturity_index",
    "split_phase",
    "seed",
    "dataset",
    "checkpoint_id",
    "checkpoint_path",
    "backbone_name",
    "member_key",
    "stratum",
    "formula",
    "source_zip_name",
    "train_steps",
    "effective_train_steps",
    "checkpoint_export_protocol",
    "train_budget_label",
    "target_nfe",
    "runtime_nfe",
    "macro_steps",
    "solver_key",
    "solver_name",
    "scheduler_key",
    "scheduler_variant_key",
    "scheduler_variant_name",
    "schedule_name",
    "schedule_family",
    "density_source_key",
    "student_training_mode",
    "row_signature",
    "signal_trace_key",
    "signal_validation_spearman",
    "info_growth_scale",
    "reference_macro_factor",
    "paper_duplicate_count",
    "experiment_scope",
    "selection_metric",
    "selection_metric_value",
    "reference_macro_steps",
    "reference_time_alignment",
    "runtime_grid_q25",
    "runtime_grid_q50",
    "runtime_grid_q75",
    "forecast_crps",
    "forecast_mse",
    "forecast_mase",
    "forecast_mase_scale_kind",
    "forecast_mase_scale_period",
    "score_main",
    "disc_auc",
    "disc_auc_gap",
    "temporal_uw1",
    "temporal_cw1",
    "temporal_tstr_f1",
    "temporal_tstr_f1_applicable",
    "u_l1",
    "c_l1",
    "spread_specific_error",
    "imbalance_specific_error",
    "ret_vol_acf_error",
    "impact_response_error",
    "molecule_kabsch_rmsd_3d",
    "molecule_ensemble_velocity_norm_w1",
    "molecule_ensemble_acceleration_norm_w1",
    "molecule_rollout_velocity_norm_w1",
    "molecule_rollout_acceleration_norm_w1",
    "molecule_coordinate_w1_mean",
    "molecule_pair_distance_w1",
    "forecast_relative_crps_gain_vs_uniform",
    "forecast_relative_mase_gain_vs_uniform",
    "relative_score_gain_vs_uniform",
    "realized_nfe",
    "latency_ms_per_sample",
    "num_eval_samples",
    "eval_examples",
    "eval_windows",
    "eval_horizon",
    "evaluation_protocol_hash",
    "chosen_t0s_hash",
    "chosen_examples_hash",
    "example_selection_protocol",
    "context_sample_count",
    "selected_examples",
    "selected_examples_cap",
    "selected_examples_cap_source",
    "uncapped_candidate_examples",
    "candidate_examples_after_initial_selection",
    "selection_was_capped",
    "global_selected_examples",
    "global_uncapped_candidate_examples",
    "global_candidate_examples_after_initial_selection",
    "global_selection_was_capped",
    "schedule_grid_hash",
    "protocol_hash",
    "row_status",
    "train_tuning_fraction",
    "train_tuning_seed",
    "train_tuning_strata",
    "train_tuning_sampler",
    "train_tuning_sampling_mode",
    "train_tuning_reference_examples",
    "train_tuning_target_examples",
    "train_tuning_uncapped_candidate_examples",
    "train_tuning_train_split_fraction",
    "train_tuning_val_split_fraction",
)

CONTEXT_ROW_FIELDS: Tuple[str, ...] = (
    "benchmark_family",
    "canonical_layout_version",
    "scenario_key",
    "scenario_family",
    "nfe_role",
    "checkpoint_step",
    "checkpoint_maturity_label",
    "checkpoint_maturity_index",
    "parent_row_signature",
    "protocol_hash",
    "dataset",
    "split_phase",
    "seed",
    "logical_seed",
    "evaluation_seed",
    "solver_key",
    "target_nfe",
    "runtime_nfe",
    "macro_steps",
    "realized_nfe",
    "scheduler_key",
    "schedule_family",
    "density_source_key",
    "context_schema",
    "axis_dataset",
    "axis_series",
    "axis_time_bin",
    "axis_record",
    "axis_window",
    "axis_stratum",
    "axis_member",
    "axis_formula",
    "axis_atom_count",
    "axis_trajectory",
    "axis_iso_id",
    "axis_flags",
    "schedule_grid_hash",
    "example_idx",
    "series_id",
    "series_idx",
    "target_t",
    "history_start",
    "history_stop",
    "target_stop",
    "context_id",
    "context_embedding_id",
    "checkpoint_id",
    "effective_train_steps",
    "checkpoint_export_protocol",
    "forecast_crps",
    "forecast_mase",
    "forecast_mse",
    "crps",
    "mase",
    "mse",
    "forecast_mase_scale_kind",
    "forecast_mase_scale_period",
    "score_main",
    "u_score_uniform",
    "u_comp_uniform",
    "u_temporal_cw1_uniform",
    "u_temporal_uw1_uniform",
    "u_temporal_tstr_f1_uniform",
    "u_temporal_u_l1_uniform",
    "u_temporal_c_l1_uniform",
    "u_temporal_spread_specific_error_uniform",
    "u_temporal_imbalance_specific_error_uniform",
    "u_temporal_ret_vol_acf_error_uniform",
    "u_temporal_impact_response_error_uniform",
    "u_molecule_kabsch_rmsd_3d_uniform",
    "u_molecule_ensemble_velocity_norm_w1_uniform",
    "u_molecule_ensemble_acceleration_norm_w1_uniform",
    "u_molecule_rollout_velocity_norm_w1_uniform",
    "u_molecule_rollout_acceleration_norm_w1_uniform",
    "reward_metric_count",
    "reward_metric_weights_json",
    "reward_metric_directions_json",
    "gipo_reward_protocol",
    "reward_anchor_schedule_key",
    "reward_utility_transform",
    "reward_granularity",
    "temporal_uw1",
    "temporal_cw1",
    "temporal_tstr_f1",
    "temporal_tstr_f1_applicable",
    "u_l1",
    "c_l1",
    "spread_specific_error",
    "imbalance_specific_error",
    "ret_vol_acf_error",
    "impact_response_error",
    "molecule_kabsch_rmsd_3d",
    "molecule_ensemble_velocity_norm_w1",
    "molecule_ensemble_acceleration_norm_w1",
    "molecule_rollout_velocity_norm_w1",
    "molecule_rollout_acceleration_norm_w1",
    "molecule_coordinate_w1_mean",
    "molecule_pair_distance_w1",
    "num_eval_samples",
    "eval_horizon",
    "batch_size",
    "sample_seed_start",
    "sample_seed_values_json",
    "chosen_examples_hash",
    "evaluation_protocol_hash",
    "row_signature",
    "train_tuning_fraction",
    "train_tuning_seed",
    "train_tuning_strata",
    "train_tuning_sampler",
)

FORECAST_CONTEXT_ROW_FIELDS = CONTEXT_ROW_FIELDS


def _assert_unique_fields(name: str, fields: Sequence[str]) -> None:
    duplicates = sorted({field for field in fields if fields.count(field) > 1})
    if duplicates:
        raise ValueError(f"{name} contains duplicate fields: {duplicates}")


_assert_unique_fields("ROW_RECORD_FIELDS", ROW_RECORD_FIELDS)
_assert_unique_fields("CONTEXT_ROW_FIELDS", CONTEXT_ROW_FIELDS)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(cast):
        return None
    return cast


def _mean(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray([float(x) for x in values if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _std(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray([float(x) for x in values if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else None
    return float(np.std(arr, ddof=1))


def _safe_relative_gain(value: Any, baseline_value: Any) -> Optional[float]:
    v = _optional_float(value)
    b = _optional_float(baseline_value)
    if v is None or b is None or abs(float(b)) <= 1e-12:
        return None
    return float(1.0 - float(v) / float(b))


def _safe_log_utility_gain(value: Any, baseline_value: Any, *, eps: float = 1e-12) -> Optional[float]:
    v = _optional_float(value)
    b = _optional_float(baseline_value)
    if v is None or b is None:
        return None
    if not (np.isfinite(v) and np.isfinite(b)):
        return None
    if float(v) < 0.0 or float(b) < 0.0:
        return None
    e = float(eps)
    return float(np.log(float(b) + e) - np.log(float(v) + e))


def _write_context_rows_enabled(cli_args: argparse.Namespace) -> bool:
    if bool(getattr(cli_args, "write_forecast_context_rows", False)):
        raise ValueError("--write_forecast_context_rows is retired; use --write_context_rows.")
    return bool(getattr(cli_args, "write_context_rows", False))


def _context_row_csv_name(cli_args: argparse.Namespace) -> str:
    value = str(getattr(cli_args, "context_row_csv_name", "") or "").strip()
    if str(getattr(cli_args, "forecast_context_row_csv_name", "") or "").strip():
        raise ValueError("--forecast_context_row_csv_name is retired; use --context_row_csv_name.")
    if value:
        return value
    return "context_rows.csv"


def _context_embeddings_npz_name(cli_args: argparse.Namespace) -> str:
    value = str(getattr(cli_args, "context_embeddings_npz_name", "") or "").strip()
    if str(getattr(cli_args, "forecast_context_embeddings_npz_name", "") or "").strip():
        raise ValueError("--forecast_context_embeddings_npz_name is retired; use --context_embeddings_npz_name.")
    if value:
        return value
    return "context_embeddings.npz"


def _parse_schedule_names(text: str) -> List[str]:
    names = [name.strip().lower() for name in parse_csv(text)]
    unknown = [name for name in names if name not in EXPERIMENTAL_FIXED_SCHEDULE_KEYS]
    if unknown:
        raise ValueError(f"Unknown fixed diffusion-flow schedules: {unknown}")
    return names


def _parse_summary_schedule_names(text: str) -> List[str]:
    names = [name.strip().lower() for name in parse_csv(text)]
    return names


def _load_schedule_summary_cases(path_text: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    if not str(path_text).strip():
        return cases
    path = resolve_project_path(str(path_text))
    if not path.exists():
        raise FileNotFoundError(f"Schedule summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schedule_items = list(payload.get("schedules") or [])
    if not schedule_items:
        schedule_items = [
            {
                "scheduler_key": str(payload.get("scheduler_key", payload.get("schedule_key", ""))),
                "predictions": payload.get("predictions", []) or [],
            }
        ]
    for schedule in schedule_items:
        scheduler_key = str(schedule.get("scheduler_key", schedule.get("schedule_key", ""))).strip()
        if not scheduler_key:
            continue
        for item in list(schedule.get("predictions", []) or []):
            solver_key = str(item.get("solver_key", "")).strip()
            target_nfe = int(item.get("target_nfe", 0) or 0)
            checkpoint_step = item.get("checkpoint_step", schedule.get("checkpoint_step", payload.get("checkpoint_step", "")))
            if not solver_key or target_nfe <= 0:
                continue
            nfe = normalize_solver_nfe_fields(
                solver_key,
                target_nfe,
                macro_steps=item.get("macro_steps"),
                runtime_nfe=item.get("runtime_nfe"),
                realized_nfe=item.get("realized_nfe"),
                source=f"schedule summary {path}",
            )
            grid = validate_time_grid(item["time_grid"], macro_steps=nfe.macro_steps)
            cases.append(
                {
                    "scheduler_key": scheduler_key,
                    "solver_key": nfe.solver_key,
                    "target_nfe": int(target_nfe),
                    "runtime_nfe": int(nfe.runtime_nfe),
                    "macro_steps": int(nfe.macro_steps),
                    "realized_nfe": int(nfe.realized_nfe),
                    "checkpoint_step": "" if checkpoint_step in (None, "") else int(checkpoint_step),
                    "time_grid": [float(x) for x in grid],
                    "schedule_grid_hash": schedule_grid_hash(grid),
                    "reference_time_alignment": str(
                        item.get("reference_time_alignment", schedule.get("reference_time_alignment", "summary_density_time_grid"))
                    ),
                    "paper_duplicate_count": int(item.get("paper_duplicate_count", 0) or 0),
                    "reference_macro_steps": int(item.get("reference_macro_steps", nfe.macro_steps) or nfe.macro_steps),
                    "schedule_summary_path": _logical_artifact_path(path),
                }
            )
    return cases


def parse_molecule_datasets(text: str) -> Tuple[str, ...]:
    requested = tuple(parse_csv(str(text)))
    if not requested:
        return ()
    allowed = set(MOLECULE_GROUP_DATASET_KEYS)
    unknown = [key for key in requested if key not in allowed]
    if unknown:
        raise ValueError(f"Unknown molecule 3D group datasets: {unknown}; expected one of {sorted(allowed)}.")
    return requested


def _target_nfe_values_for_args(cli_args: argparse.Namespace) -> List[int]:
    role = str(getattr(cli_args, "nfe_role", NFE_ROLE_SEEN) or NFE_ROLE_SEEN)
    canonical = tuple(int(value) for value in canonical_nfes_for_role(role))
    raw = str(getattr(cli_args, "target_nfe_values", "") or "").strip()
    values = parse_int_csv(raw) if raw else list(canonical)
    unknown = [int(value) for value in values if int(value) not in set(canonical)]
    if unknown:
        raise ValueError(
            f"target_nfe_values {unknown} are not canonical for nfe_role={role!r}; "
            f"allowed values are {list(canonical)}."
        )
    if not values:
        raise ValueError("At least one target NFE is required.")
    return [int(value) for value in values]


def _checkpoint_steps_for_args(cli_args: argparse.Namespace) -> List[int]:
    raw = str(getattr(cli_args, "checkpoint_steps", "") or "").strip()
    if raw:
        values = parse_int_csv(raw)
    else:
        fallback = int(getattr(cli_args, "otflow_train_steps", 0) or 0)
        values = [fallback] if fallback > 0 else list(CANONICAL_CHECKPOINT_STEPS)
    allowed = set(int(value) for value in CANONICAL_CHECKPOINT_STEPS)
    unknown = [int(value) for value in values if int(value) not in allowed]
    if unknown:
        raise ValueError(
            f"checkpoint_steps {unknown} are not canonical; allowed values are {list(CANONICAL_CHECKPOINT_STEPS)}."
        )
    if not values:
        raise ValueError("At least one checkpoint step is required.")
    return [int(value) for value in values]


def _checkpoint_maturity_label(step: int) -> str:
    return f"{int(step)}_steps"


def _checkpoint_maturity_index(step: int) -> int:
    steps = tuple(int(value) for value in CANONICAL_CHECKPOINT_STEPS)
    if int(step) not in steps:
        raise ValueError(f"Unknown canonical checkpoint step: {step}")
    return int(steps.index(int(step)))


def _args_for_checkpoint_step(cli_args: argparse.Namespace, checkpoint_step: int) -> argparse.Namespace:
    copied = copy.copy(cli_args)
    copied.otflow_train_steps = int(checkpoint_step)
    copied.steps = int(checkpoint_step)
    return copied


def _path_fingerprint(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_project_path(str(path))
    path_hash = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
    payload: Dict[str, Any] = {"path_hash": str(path_hash), "name": str(resolved.name), "exists": bool(resolved.exists())}
    if resolved.is_file():
        stat = resolved.stat()
        payload.update({"kind": "file", "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    elif resolved.is_dir():
        stat = resolved.stat()
        payload.update({"kind": "dir", "mtime_ns": int(stat.st_mtime_ns)})
    else:
        payload["kind"] = "missing"
    return payload


def _logical_artifact_path(path: str | Path) -> str:
    resolved = resolve_project_path(str(path))
    root = project_root().resolve()
    try:
        return str(resolved.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(resolved.name)


def _data_path_fingerprints(cli_args: argparse.Namespace) -> Dict[str, Any]:
    cryptos_path = str(cli_args.cryptos_path).strip() or default_cryptos_data_path()
    lobster_profile_path = str(getattr(cli_args, "lobster_synthetic_profile_path", "")).strip() or default_lobster_synthetic_profile_path()
    long_term_st_path = str(getattr(cli_args, "long_term_st_path", "")).strip() or default_long_term_st_data_path()
    return {
        "cryptos": _path_fingerprint(cryptos_path),
        "lobster_synthetic": _path_fingerprint(lobster_profile_path),
        "long_term_st": _path_fingerprint(long_term_st_path),
        "dataset_root": _path_fingerprint(str(cli_args.dataset_root)),
        "shared_backbone_root": _path_fingerprint(str(cli_args.shared_backbone_root)),
    }


def _sanitized_cli_args(cli_args: argparse.Namespace) -> Dict[str, Any]:
    path_fields = {
        "out_root",
        "dataset_root",
        "shared_backbone_root",
        "backbone_manifest",
        "molecule_group_root",
        "cryptos_path",
        "lobster_synthetic_profile_path",
        "long_term_st_path",
    }
    payload: Dict[str, Any] = {}
    for key, value in vars(cli_args).items():
        if key in path_fields:
            text = str(value).strip()
            payload[key] = None if not text else _path_fingerprint(text)
        else:
            payload[key] = value
    return payload


def _context_reward_protocol_payload(cli_args: argparse.Namespace) -> Dict[str, Any]:
    conditional_datasets = parse_conditional_generation_datasets(str(cli_args.conditional_generation_datasets))
    return {
        "version": CONTEXT_REWARD_PROTOCOL_VERSION,
        "conditional_generation_reward_granularity": "aggregate_primary_metric_components",
        "conditional_generation_profiles": {
            str(dataset): teacher_metric_profile_for_scenario(str(dataset))
            for dataset in conditional_datasets
        },
        "conditional_diagnostic_metrics_are_teacher_targets": False,
        "conditional_diagnostic_columns": [
            "u_l1",
            "c_l1",
            "spread_specific_error",
            "imbalance_specific_error",
            "ret_vol_acf_error",
            "impact_response_error",
        ],
    }


def _context_sample_cap(cli_args: argparse.Namespace) -> int:
    cap = int(getattr(cli_args, "context_sample_count", 256))
    if cap <= 0:
        raise ValueError(f"--context_sample_count must be positive, got {cap!r}.")
    return int(cap)


def _split_example_cap(cli_args: argparse.Namespace, split_phase: str) -> Tuple[int | None, str]:
    context_cap = _context_sample_cap(cli_args)
    if str(split_phase) == TRAIN_TUNING_PHASE:
        return int(context_cap), "train_tuning_context_sample_count"
    attr = "eval_windows_val" if str(split_phase) == VALIDATION_PHASE else "eval_windows_test"
    explicit = int(getattr(cli_args, attr, 0))
    if explicit < 0:
        raise ValueError(f"--{attr} must be nonnegative, got {explicit!r}.")
    if explicit > 0:
        return int(explicit), attr
    return None, f"{split_phase}_default"


def _selection_group_candidate_count(groups: Sequence[Mapping[str, Any]]) -> int:
    return int(sum(len(group.get("candidate_indices", []) or []) for group in groups))


def _selection_cap_for_groups(groups: Sequence[Mapping[str, Any]], requested_cap: int | None) -> int:
    if requested_cap is not None:
        return int(requested_cap)
    total = _selection_group_candidate_count(groups)
    if total <= 0:
        raise ValueError("Default split evaluation requires at least one candidate example.")
    return int(total)


def _choose_stratified_train_tuning_positions(
    total: int,
    *,
    fraction: float,
    seed: int,
    strata: int,
    dataset: str,
    salt: str,
    max_examples: int | None = None,
) -> Tuple[List[int], int]:
    target_examples = train_tuning_target_example_count(
        int(total),
        fraction=float(fraction),
        sampling_mode=TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION,
        strata=int(strata),
    )
    selected = choose_forecast_train_tuning_indices(
        range(int(total)),
        fraction=float(fraction),
        seed=int(seed),
        strata=int(strata),
        dataset=str(dataset),
        salt=str(salt),
        sampling_mode=TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION,
        max_examples=None if max_examples is None or int(max_examples) <= 0 else int(max_examples),
    )
    return [int(idx) for idx in selected.tolist()], int(target_examples)


def _train_tuning_metadata(
    cli_args: argparse.Namespace,
    *,
    tuning_seed: int,
    target_examples: int,
    uncapped_candidate_examples: int,
    sampling_mode: str = TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION,
    reference_examples: Any = "",
) -> Dict[str, Any]:
    return {
        "train_tuning_fraction": float(cli_args.eval_train_fraction),
        "train_tuning_seed": int(tuning_seed),
        "train_tuning_strata": int(cli_args.train_tuning_strata),
        "train_tuning_sampler": train_tuning_sampler_key(str(sampling_mode)),
        "train_tuning_sampling_mode": str(sampling_mode),
        "train_tuning_reference_examples": reference_examples,
        "train_tuning_target_examples": int(target_examples),
        "train_tuning_uncapped_candidate_examples": int(uncapped_candidate_examples),
        "train_tuning_train_split_fraction": float(cli_args.train_tuning_train_split_fraction),
        "train_tuning_val_split_fraction": float(cli_args.train_tuning_val_split_fraction),
    }


def _cap_context_indices(
    indices: Sequence[int],
    *,
    cap: int,
    seed: int,
    salt: str,
    uncapped_candidate_examples: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    candidate = [int(idx) for idx in indices]
    selected_cap = int(cap)
    if selected_cap <= 0:
        raise ValueError(f"selected_examples_cap must be positive, got {selected_cap!r}.")
    uncapped_count = int(len(candidate) if uncapped_candidate_examples is None else uncapped_candidate_examples)
    if uncapped_count < len(candidate):
        raise ValueError(
            "uncapped_candidate_examples cannot be smaller than the selected candidate list "
            f"({uncapped_count} < {len(candidate)})."
        )
    if len(candidate) <= selected_cap:
        selected = list(candidate)
        was_capped = uncapped_count > len(selected)
    else:
        token = f"{SCHEDULE_CONTEXT_SELECTION_PROTOCOL}|{salt}|{int(seed)}|{len(candidate)}|{selected_cap}"
        local_seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(local_seed)
        keep_positions = sorted(int(pos) for pos in rng.choice(np.arange(len(candidate)), size=selected_cap, replace=False).tolist())
        selected = [candidate[pos] for pos in keep_positions]
        was_capped = True
    return np.asarray(selected, dtype=np.int64), {
        "example_selection_protocol": SCHEDULE_CONTEXT_SELECTION_PROTOCOL,
        "selected_examples": int(len(selected)),
        "selected_examples_cap": int(selected_cap),
        "uncapped_candidate_examples": int(uncapped_count),
        "candidate_examples_after_initial_selection": int(len(candidate)),
        "selection_was_capped": bool(was_capped),
    }


def _cap_context_index_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    seed: int,
    salt: str,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    selected_cap = int(cap)
    if selected_cap <= 0:
        raise ValueError(f"selected_examples_cap must be positive, got {selected_cap!r}.")
    normalized: List[Dict[str, Any]] = []
    flat_candidates: List[Tuple[int, int]] = []
    uncapped_total = 0
    for group_idx, group in enumerate(groups):
        candidate = [int(idx) for idx in group.get("candidate_indices", [])]
        uncapped_count = int(group.get("uncapped_candidate_examples", len(candidate)))
        if uncapped_count < len(candidate):
            raise ValueError(
                "uncapped_candidate_examples cannot be smaller than the selected candidate list "
                f"({uncapped_count} < {len(candidate)})."
            )
        normalized.append({**dict(group), "candidate_indices": candidate, "uncapped_candidate_examples": uncapped_count})
        uncapped_total += int(uncapped_count)
        flat_candidates.extend((int(group_idx), int(pos)) for pos in range(len(candidate)))
    if not flat_candidates:
        raise ValueError("Global context selection requires at least one candidate example.")
    offsets: List[int] = []
    cursor = 0
    for group in normalized:
        offsets.append(int(cursor))
        cursor += len(group["candidate_indices"])
    selected_target = min(int(selected_cap), int(len(flat_candidates)))
    active_groups = [idx for idx, group in enumerate(normalized) if group["candidate_indices"]]
    if len(flat_candidates) <= selected_cap and uncapped_total <= len(flat_candidates):
        kept_positions = list(range(len(flat_candidates)))
        was_capped = False
    else:
        shape = ",".join(str(len(group["candidate_indices"])) for group in normalized)
        token = f"{SCHEDULE_CONTEXT_SELECTION_PROTOCOL}|global|{salt}|{int(seed)}|{shape}|{len(flat_candidates)}|{selected_cap}"
        local_seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(local_seed)
        kept_positions = []
        if selected_target >= len(active_groups):
            for group_idx in active_groups:
                group_size = len(normalized[group_idx]["candidate_indices"])
                token_one = f"{token}|group|{group_idx}|{group_size}"
                candidate_pos = int(hashlib.sha256(token_one.encode("utf-8")).hexdigest()[:16], 16) % int(group_size)
                kept_positions.append(int(offsets[group_idx]) + int(candidate_pos))
        remaining = int(selected_target) - len(kept_positions)
        if remaining > 0:
            kept = set(kept_positions)
            available_positions = [pos for pos in range(len(flat_candidates)) if pos not in kept]
            chosen = rng.choice(np.arange(len(available_positions)), size=int(remaining), replace=False).tolist()
            kept_positions.extend(int(available_positions[int(pos)]) for pos in chosen)
        kept_positions = sorted(kept_positions)
        was_capped = True
    selected_by_group: List[List[int]] = [[] for _ in normalized]
    for flat_pos in kept_positions:
        group_idx, candidate_pos = flat_candidates[int(flat_pos)]
        selected_by_group[group_idx].append(int(normalized[group_idx]["candidate_indices"][candidate_pos]))
    selected_total = int(sum(len(indices) for indices in selected_by_group))
    records: List[Dict[str, Any]] = []
    for group, selected in zip(normalized, selected_by_group):
        record = dict(group.get("selection_record", {}))
        record.update(
            {
                "example_selection_protocol": SCHEDULE_CONTEXT_SELECTION_PROTOCOL,
                "selected_examples": int(len(selected)),
                "selected_examples_cap": int(selected_cap),
                "uncapped_candidate_examples": int(group["uncapped_candidate_examples"]),
                "candidate_examples_after_initial_selection": int(len(group["candidate_indices"])),
                "selection_was_capped": bool(was_capped or int(group["uncapped_candidate_examples"]) > len(selected)),
                "global_selected_examples": int(selected_total),
                "global_uncapped_candidate_examples": int(uncapped_total),
                "global_candidate_examples_after_initial_selection": int(len(flat_candidates)),
                "global_selection_was_capped": bool(was_capped),
            }
        )
        records.append(record)
    return [np.asarray(indices, dtype=np.int64) for indices in selected_by_group], records, {
        "selected_examples": int(selected_total),
        "selected_examples_cap": int(selected_cap),
        "uncapped_candidate_examples": int(uncapped_total),
        "candidate_examples_after_initial_selection": int(len(flat_candidates)),
        "selection_was_capped": bool(was_capped),
    }


def _selection_metadata_row_fields(selection_meta: Mapping[str, Any], *, cap_source: str, context_sample_count: int) -> Dict[str, Any]:
    fields = {
        "example_selection_protocol": str(selection_meta["example_selection_protocol"]),
        "context_sample_count": int(context_sample_count),
        "selected_examples": int(selection_meta["selected_examples"]),
        "selected_examples_cap": int(selection_meta["selected_examples_cap"]),
        "selected_examples_cap_source": str(cap_source),
        "uncapped_candidate_examples": int(selection_meta["uncapped_candidate_examples"]),
        "candidate_examples_after_initial_selection": int(selection_meta["candidate_examples_after_initial_selection"]),
        "selection_was_capped": bool(selection_meta["selection_was_capped"]),
    }
    for key in (
        "global_selected_examples",
        "global_uncapped_candidate_examples",
        "global_candidate_examples_after_initial_selection",
        "global_selection_was_capped",
    ):
        if key in selection_meta:
            fields[key] = bool(selection_meta[key]) if key.endswith("was_capped") else int(selection_meta[key])
    return fields


def _protocol_config_fingerprint(cli_args: argparse.Namespace) -> str:
    payload = {
        "runner_signature": RUNNER_SIGNATURE_VERSION,
        "forecast_datasets": parse_forecast_datasets(str(cli_args.forecast_datasets)),
        "conditional_generation_datasets": parse_conditional_generation_datasets(
            str(cli_args.conditional_generation_datasets)
        ),
        "molecule_datasets": parse_molecule_datasets(str(getattr(cli_args, "molecule_datasets", ""))),
        "seeds": parse_int_csv(str(cli_args.seeds)),
        "canonical_layout_version": CANONICAL_LAYOUT_VERSION,
        "nfe_role": str(getattr(cli_args, "nfe_role", NFE_ROLE_SEEN)),
        "target_nfe_values": _target_nfe_values_for_args(cli_args),
        "checkpoint_steps": _checkpoint_steps_for_args(cli_args),
        "solver_names": list(normalize_solver_keys(str(cli_args.solver_names))),
        "baseline_scheduler_names": _parse_schedule_names(str(cli_args.baseline_scheduler_names)),
        "schedule_summary_json": _path_fingerprint(str(getattr(cli_args, "schedule_summary_json", ""))) if str(getattr(cli_args, "schedule_summary_json", "")).strip() else None,
        "summary_scheduler_names": _parse_summary_schedule_names(str(getattr(cli_args, "summary_scheduler_names", ""))),
        "split_phase": str(cli_args.split_phase),
        "dataset_seed": int(cli_args.dataset_seed),
        "num_eval_samples": int(cli_args.num_eval_samples),
        "molecule_sample_count": int(getattr(cli_args, "molecule_sample_count", 1)),
        "molecule_rollout_steps": int(getattr(cli_args, "molecule_rollout_steps", 16)),
        "molecule_stride_eval": int(getattr(cli_args, "molecule_stride_eval", 1)),
        "forecast_eval_batch_size": int(cli_args.forecast_eval_batch_size),
        "write_context_rows": _write_context_rows_enabled(cli_args),
        "context_embedding_kind": str(getattr(cli_args, "context_embedding_kind", "ctx_summary")),
        "context_sample_count": _context_sample_cap(cli_args),
        "example_selection_protocol": SCHEDULE_CONTEXT_SELECTION_PROTOCOL,
        "eval_horizon": int(cli_args.eval_horizon),
        "eval_windows_val": int(cli_args.eval_windows_val),
        "eval_windows_test": int(cli_args.eval_windows_test),
        "eval_train_fraction": float(cli_args.eval_train_fraction),
        "train_tuning_seed": int(cli_args.train_tuning_seed),
        "train_tuning_strata": int(cli_args.train_tuning_strata),
        "train_tuning_sampling_mode": str(cli_args.train_tuning_sampling_mode),
        "train_tuning_sampler": train_tuning_sampler_key(str(cli_args.train_tuning_sampling_mode)),
        "train_tuning_train_split_fraction": float(cli_args.train_tuning_train_split_fraction),
        "train_tuning_val_split_fraction": float(cli_args.train_tuning_val_split_fraction),
        "calibration_trace_samples": int(cli_args.calibration_trace_samples),
        "dataset_root": _path_fingerprint(str(cli_args.dataset_root)),
        "shared_backbone_root": _path_fingerprint(str(cli_args.shared_backbone_root)),
        "backbone_manifest": _path_fingerprint(str(cli_args.backbone_manifest)) if str(cli_args.backbone_manifest).strip() else None,
        "molecule_group_root": _path_fingerprint(str(getattr(cli_args, "molecule_group_root", default_molecule_group_root()))),
        "data_path_fingerprints": _data_path_fingerprints(cli_args),
        "context_reward_protocol": _context_reward_protocol_payload(cli_args),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _realized_nfe_for_solver(solver_key: str, runtime_nfe: int) -> int:
    return int(runtime_nfe) * int(solver_eval_multiplier(str(solver_key)))


def _row_signature(*, dataset: str, split_phase: str, seed: int, target_nfe: int, solver_key: str, scheduler_key: str, checkpoint_id: str) -> str:
    return "|".join(
        [str(dataset), str(split_phase), str(seed), str(target_nfe), str(solver_key), str(scheduler_key), str(checkpoint_id)]
    )


def _row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("protocol_hash"),
        row.get("benchmark_family"),
        row.get("split_phase"),
        int(row.get("seed", -1)),
        row.get("dataset"),
        int(row.get("target_nfe", -1)),
        row.get("solver_key"),
        row.get("scheduler_key"),
        row.get("row_signature"),
    )


def _write_row_csv(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_RECORD_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ROW_RECORD_FIELDS})


def _write_context_row_csv(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CONTEXT_ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CONTEXT_ROW_FIELDS})


def _load_context_rows(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not csv_path.exists():
        return rows
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            signature = str(row.get("row_signature", "")).strip()
            if signature:
                if signature in rows:
                    raise ValueError(f"Duplicate context row signature in {csv_path}: {signature}")
                rows[signature] = dict(row)
    return rows


def _load_rows(jsonl_path: Path, *, protocol_hash: str) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("protocol_hash", "")) != str(protocol_hash):
                continue
            rows[_row_key(row)] = row
    return rows


def _init_row_recorder(out_root: Path, cli_args: argparse.Namespace) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / str(getattr(cli_args, "row_jsonl_name", "rows.jsonl"))
    csv_path = out_root / str(getattr(cli_args, "row_csv_name", "rows.csv"))
    context_csv_path = out_root / _context_row_csv_name(cli_args)
    context_embeddings_path = out_root / _context_embeddings_npz_name(cli_args)
    protocol_hash = _protocol_config_fingerprint(cli_args)
    run_config_path = out_root / "run_config.json"
    previous_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    can_resume = bool(getattr(cli_args, "resume", True)) and str(previous_config.get("protocol_hash", "")) == protocol_hash
    rows_by_key = _load_rows(jsonl_path, protocol_hash=str(protocol_hash)) if can_resume else {}
    fh = jsonl_path.open("a" if can_resume else "w", encoding="utf-8")
    save_json(
        {
            "runner_signature": RUNNER_SIGNATURE_VERSION,
            "method_key": METHOD_KEY,
            "protocol_hash": protocol_hash,
            "context_reward_protocol": _context_reward_protocol_payload(cli_args),
            "args": _sanitized_cli_args(cli_args),
            "data_path_fingerprints": _data_path_fingerprints(cli_args),
        },
        str(run_config_path),
    )
    if rows_by_key:
        _write_row_csv(csv_path, list(rows_by_key.values()))
    context_rows_by_signature = _load_context_rows(context_csv_path) if can_resume else {}
    existing_context_embeddings = load_context_embedding_table(context_embeddings_path) if can_resume and context_embeddings_path.exists() else {}
    if context_rows_by_signature:
        _write_context_row_csv(context_csv_path, list(context_rows_by_signature.values()))
    return {
        "out_root": out_root,
        "jsonl_path": jsonl_path,
        "csv_path": csv_path,
        "context_csv_path": context_csv_path,
        "context_embeddings_path": context_embeddings_path,
        "fh": fh,
        "rows_by_key": rows_by_key,
        "context_rows_by_signature": context_rows_by_signature,
        "context_embeddings": existing_context_embeddings,
        "context_embedding_metadata": {},
        "context_embedding_coverage": {},
        "protocol_hash": protocol_hash,
    }


def _append_row_record(row_recorder: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    row_dict = dict(row)
    key = _row_key(row_dict)
    row_recorder["rows_by_key"][key] = row_dict
    row_recorder["fh"].write(json.dumps(row_dict, sort_keys=True) + "\n")
    row_recorder["fh"].flush()
    _write_row_csv(Path(row_recorder["csv_path"]), list(row_recorder["rows_by_key"].values()))


def _append_context_records(
    row_recorder: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    context_embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Any],
) -> None:
    if not rows and not context_embeddings:
        return
    rows_by_signature = row_recorder["context_rows_by_signature"]
    for row in rows:
        signature = str(row.get("row_signature", "")).strip()
        if not signature:
            continue
        if signature in rows_by_signature:
            raise ValueError(f"Duplicate context row signature while appending context artifacts: {signature}")
        rows_by_signature[signature] = dict(row)
    existing_embeddings = row_recorder["context_embeddings"]
    for key, value in context_embeddings.items():
        key_text = str(key)
        new_vec = np.asarray(value, dtype=np.float32)
        if key_text in existing_embeddings:
            old_vec = np.asarray(existing_embeddings[key_text], dtype=np.float32)
            if old_vec.shape != new_vec.shape or not np.allclose(old_vec, new_vec, rtol=1e-5, atol=1e-6):
                raise ValueError(f"Context embedding collision for {key_text!r} with different vector/protocol.")
            continue
        existing_embeddings[key_text] = new_vec.astype(float).tolist()
    missing_embeddings = sorted(
        {
            str(row.get("context_embedding_id", "") or "").strip()
            for row in rows
            if str(row.get("context_embedding_id", "") or "").strip()
            and str(row.get("context_embedding_id", "") or "").strip() not in existing_embeddings
        }
    )
    if missing_embeddings:
        raise KeyError(f"Context rows are missing embedding vectors: {missing_embeddings[:8]}")
    coverage_key = "|".join(
        str(metadata.get(field, ""))
        for field in ("benchmark_family", "dataset", "checkpoint_id", "split_phase", "context_schema")
    )
    coverage = row_recorder["context_embedding_coverage"].setdefault(
        coverage_key,
        {
            "benchmark_family": str(metadata.get("benchmark_family", "")),
            "dataset": str(metadata.get("dataset", "")),
            "checkpoint_id": str(metadata.get("checkpoint_id", "")),
            "checkpoint_step": metadata.get("checkpoint_step", ""),
            "split_phase": str(metadata.get("split_phase", "")),
            "context_schema": str(metadata.get("context_schema", "")),
            "row_count": 0,
            "embedding_count": 0,
        },
    )
    coverage["row_count"] = int(coverage.get("row_count", 0)) + int(len(rows))
    coverage["embedding_count"] = int(coverage.get("embedding_count", 0)) + int(len(context_embeddings))
    row_recorder["context_embedding_metadata"] = {
        "coverage": sorted(row_recorder["context_embedding_coverage"].values(), key=lambda item: tuple(str(item.get(field, "")) for field in ("benchmark_family", "dataset", "checkpoint_id", "split_phase", "context_schema"))),
    }
    _write_context_row_csv(Path(row_recorder["context_csv_path"]), list(rows_by_signature.values()))
    if row_recorder["context_embeddings"]:
        save_context_embedding_table(
            Path(row_recorder["context_embeddings_path"]),
            row_recorder["context_embeddings"],
            metadata=row_recorder["context_embedding_metadata"],
        )


def _time_bin_for_target(t0: int, chosen_t0s: Sequence[int]) -> str:
    values = np.asarray([int(x) for x in chosen_t0s], dtype=np.int64)
    if values.size <= 1:
        return "0"
    order = np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(values.size, dtype=np.int64)
    matches = np.where(values == int(t0))[0]
    rank = int(ranks[int(matches[0])]) if matches.size else 0
    return str(min(9, int(np.floor(10.0 * rank / max(1, values.size)))))


def _extract_conditional_context_embeddings(
    *,
    model: Any,
    ds: Any,
    chosen_t0s: Sequence[int],
    device: torch.device,
    context_embedding_kind: str,
) -> Dict[int, List[float]]:
    if not chosen_t0s:
        return {}
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "precompute"):
        raise ValueError("context row export requires model.backbone.precompute(hist).")
    hist_rows = []
    for t0 in chosen_t0s:
        hist, _tgt, _fut, _cond, _meta = _parse_batch(_get_dataset_item_by_t(ds, int(t0)))
        hist_rows.append(hist.float())
    hist_batch = torch.stack(hist_rows, dim=0).to(device).float()
    cache = backbone.precompute(hist_batch)
    if not hasattr(cache, str(context_embedding_kind)):
        raise ValueError(f"Unknown context_embedding_kind={context_embedding_kind!r}.")
    embedding_tensor = getattr(cache, str(context_embedding_kind))
    if not torch.is_tensor(embedding_tensor) or embedding_tensor.ndim != 2:
        raise ValueError(f"Context embedding {context_embedding_kind!r} must be a rank-2 tensor.")
    arr = embedding_tensor.detach().cpu().numpy().astype(np.float32)
    return {int(t0): [float(x) for x in arr[idx].tolist()] for idx, t0 in enumerate(chosen_t0s)}


def _conditional_context_records(
    *,
    benchmark_family: str,
    dataset: str,
    split_phase: str,
    seed: int,
    evaluation_seed: int,
    solver_key: str,
    target_nfe: int,
    runtime_nfe: int,
    scheduler_key: str,
    details: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_step: int,
    nfe_role: str,
    parent_row_signature: str,
    protocol_hash: str,
    cfg: Any,
    eval_horizon: int,
    chosen_t0s: Sequence[int],
    score_main: Any,
    uniform_score_main: Any,
    per_window_metrics_by_t0: Mapping[int, Mapping[str, Any]] | None = None,
    uniform_per_window_metrics_by_t0: Mapping[int, Mapping[str, Any]] | None = None,
    metric_row: Mapping[str, Any] | None = None,
    uniform_metric_row: Mapping[str, Any] | None = None,
    evaluation_protocol_hash: str = "",
    chosen_t0s_hash: str = "",
    train_tuning_context_metadata: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    from genode.gipo.policy import schedule_grid_hash, stable_context_id

    rows: List[Dict[str, Any]] = []
    history_len = int(getattr(cfg, "history_len", 0) or 0)
    aggregate_metrics = dict(metric_row or {})
    uniform_aggregate_metrics = dict(uniform_metric_row or {})
    per_window = {int(key): dict(value) for key, value in dict(per_window_metrics_by_t0 or {}).items()}
    uniform_per_window = {int(key): dict(value) for key, value in dict(uniform_per_window_metrics_by_t0 or {}).items()}
    teacher_specs = teacher_objective_specs_for_scenario(str(dataset))
    for window_idx, t0 in enumerate([int(x) for x in chosen_t0s]):
        metrics_for_row = dict(per_window.get(int(t0), {}) or aggregate_metrics)
        if not metrics_for_row:
            raise ValueError(
                "Conditional context rows require per-window diagnostics or aggregate schedule metrics."
            )
        if scheduler_key == UNIFORM_SCHEDULER_KEY:
            uniform_metrics_for_row = dict(metrics_for_row)
        else:
            uniform_metrics_for_row = dict(uniform_per_window.get(int(t0), {}) or uniform_aggregate_metrics)
        if not uniform_metrics_for_row:
            raise ValueError(
                "Conditional context reward construction requires a matched uniform per-window or aggregate metric row "
                f"for target_t={int(t0)}."
            )
        reward_metric_row = {**metrics_for_row, **aggregate_metrics}
        uniform_reward_metric_row = {**uniform_metrics_for_row, **uniform_aggregate_metrics}
        score_value = metrics_for_row.get("score_main", score_main)
        uniform_score_value = uniform_metrics_for_row.get("score_main", uniform_score_main)
        score_gain = (
            0.0
            if scheduler_key == UNIFORM_SCHEDULER_KEY
            else _safe_log_utility_gain(score_value, uniform_score_value)
        )
        reward_columns = uniform_anchored_objective_columns(
            {**reward_metric_row, "scheduler_key": scheduler_key},
            {**uniform_reward_metric_row, "scheduler_key": UNIFORM_SCHEDULER_KEY},
            teacher_specs,
            uniform_schedule_key=UNIFORM_SCHEDULER_KEY,
        )
        if reward_columns.get("u_comp_uniform") in (None, ""):
            raise ValueError(
                "Conditional context reward construction produced no finite component utility; "
                f"target_t={int(t0)} scheduler={scheduler_key!r}."
            )
        raw_context_id = stable_context_id(
            dataset=str(dataset),
            split_phase=str(split_phase),
            example_idx=int(window_idx),
            series_id=str(dataset),
            series_idx=0,
            target_t=int(t0),
            history_start=int(t0) - int(history_len),
            history_stop=int(t0),
            context_schema="conditional_generation_window",
        )
        context_id = raw_context_id
        context_embedding_id = f"{checkpoint['checkpoint_id']}:{raw_context_id}"
        row_signature_payload = {
            "context_id": str(context_id),
            "context_embedding_id": str(context_embedding_id),
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "seed": int(seed),
            "scheduler_key": str(scheduler_key),
            "solver_key": str(solver_key),
            "target_nfe": int(target_nfe),
        }
        row_signature = hashlib.sha256(
            json.dumps(row_signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "benchmark_family": str(benchmark_family),
                "canonical_layout_version": CANONICAL_LAYOUT_VERSION,
                "scenario_key": str(dataset),
                "scenario_family": str(benchmark_family),
                "nfe_role": str(nfe_role),
                "checkpoint_step": int(checkpoint_step),
                "checkpoint_maturity_label": _checkpoint_maturity_label(int(checkpoint_step)),
                "checkpoint_maturity_index": _checkpoint_maturity_index(int(checkpoint_step)),
                "effective_train_steps": int(checkpoint.get("effective_train_steps", checkpoint_step)),
                "checkpoint_export_protocol": str(checkpoint.get("checkpoint_export_protocol", "")),
                "parent_row_signature": str(parent_row_signature),
                "protocol_hash": str(protocol_hash),
                "dataset": str(dataset),
                "split_phase": str(split_phase),
                "seed": int(seed),
                "logical_seed": int(seed),
                "evaluation_seed": int(evaluation_seed),
                "solver_key": str(solver_key),
                "target_nfe": int(target_nfe),
                "runtime_nfe": int(runtime_nfe),
                "macro_steps": int(runtime_nfe),
                "realized_nfe": _realized_nfe_for_solver(str(solver_key), int(runtime_nfe)),
                "scheduler_key": str(scheduler_key),
                "schedule_family": schedule_family_for_key(str(scheduler_key)),
                "density_source_key": density_source_key_for_schedule(str(scheduler_key)),
                "context_schema": "conditional_generation_window",
                "axis_dataset": str(dataset),
                "axis_series": str(dataset),
                "axis_time_bin": _time_bin_for_target(int(t0), chosen_t0s),
                "axis_record": str(dataset),
                "axis_window": str(t0),
                "axis_stratum": "",
                "axis_member": "",
                "axis_formula": "",
                "axis_atom_count": "",
                "axis_trajectory": "",
                "axis_iso_id": "",
                "axis_flags": "",
                "schedule_grid_hash": schedule_grid_hash(details["time_grid"]),
                "example_idx": int(window_idx),
                "series_id": str(dataset),
                "series_idx": 0,
                "target_t": int(t0),
                "history_start": int(t0) - int(history_len),
                "history_stop": int(t0),
                "target_stop": int(t0) + int(eval_horizon),
                "context_id": str(context_id),
                "context_embedding_id": str(context_embedding_id),
                "checkpoint_id": str(checkpoint["checkpoint_id"]),
                "score_main": score_value,
                "temporal_uw1": reward_metric_row.get("temporal_uw1", ""),
                "temporal_cw1": reward_metric_row.get("temporal_cw1", ""),
                "temporal_tstr_f1": reward_metric_row.get("temporal_tstr_f1", ""),
                "temporal_tstr_f1_applicable": reward_metric_row.get("temporal_tstr_f1_applicable", ""),
                "u_l1": metrics_for_row.get("u_l1", ""),
                "c_l1": metrics_for_row.get("c_l1", ""),
                "spread_specific_error": metrics_for_row.get("spread_specific_error", ""),
                "imbalance_specific_error": metrics_for_row.get("imbalance_specific_error", ""),
                "ret_vol_acf_error": metrics_for_row.get("ret_vol_acf_error", ""),
                "impact_response_error": metrics_for_row.get("impact_response_error", ""),
                "u_score_uniform": score_gain,
                **reward_columns,
                "gipo_reward_protocol": GIPO_PROTOCOL,
                "reward_anchor_schedule_key": UNIFORM_SCHEDULER_KEY,
                "reward_utility_transform": "directional_log_uniform_anchor",
                "reward_granularity": "aggregate_primary_metric_components",
                "num_eval_samples": "",
                "eval_horizon": int(eval_horizon),
                "batch_size": "",
                "sample_seed_start": int(evaluation_seed),
                "sample_seed_values_json": json.dumps([int(evaluation_seed) + int(window_idx)], separators=(",", ":")),
                "chosen_examples_hash": str(chosen_t0s_hash),
                "evaluation_protocol_hash": str(evaluation_protocol_hash),
                "row_signature": str(row_signature),
                **dict(train_tuning_context_metadata or {}),
            }
        )
    return rows


def _existing_complete_row(row_recorder: Mapping[str, Any], row_key: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    row = row_recorder["rows_by_key"].get(row_key)
    if row is not None and str(row.get("row_status")) == "complete":
        return dict(row)
    return None


def _pending_scheduler_cases(
    row_recorder: Mapping[str, Any],
    *,
    benchmark_family: str,
    split_phase: str,
    seed: int,
    dataset: str,
    checkpoint_id: str,
    checkpoint_step: int,
    target_nfe: int,
    solver_key: str,
    scheduler_cases: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    existing: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for case in scheduler_cases:
        scheduler_key = str(case["scheduler_key"])
        case_solver = str(case.get("solver_key", "") or "")
        if case_solver and case_solver != str(solver_key):
            continue
        case_target_nfe = case.get("target_nfe", "")
        if case_target_nfe not in ("", None) and int(case_target_nfe) != int(target_nfe):
            continue
        case_checkpoint_step = case.get("checkpoint_step", "")
        if case_checkpoint_step not in ("", None) and int(case_checkpoint_step) != int(checkpoint_step):
            continue
        signature = _row_signature(dataset=dataset, split_phase=split_phase, seed=seed, target_nfe=target_nfe, solver_key=solver_key, scheduler_key=scheduler_key, checkpoint_id=checkpoint_id)
        key = (row_recorder["protocol_hash"], benchmark_family, split_phase, int(seed), dataset, int(target_nfe), solver_key, scheduler_key, signature)
        row = _existing_complete_row(row_recorder, key)
        if row is None:
            pending.append(dict(case, row_signature=signature))
        else:
            existing.append(row)
    return existing, pending


def _choose_molecule_indices(ds: Any, *, count: int, seed: int) -> List[int]:
    total = int(len(ds))
    if total <= 0:
        raise ValueError("Empty molecule evaluation split.")
    target = min(max(1, int(count)), total)
    indices = np.arange(total, dtype=np.int64)
    if target < total:
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(indices, size=target, replace=False))
    return [int(x) for x in indices.tolist()]


def _molecule_split_for_phase(split_phase: str) -> str:
    if str(split_phase) == TRAIN_TUNING_PHASE:
        return "train"
    if str(split_phase) == VALIDATION_PHASE:
        return "val"
    if str(split_phase) == LOCKED_TEST_PHASE:
        return "test"
    raise ValueError(f"Unsupported molecule split_phase={split_phase!r}.")


def _molecule_member_processed_dir(group_root: Path, dataset: str, member: Mapping[str, Any]) -> Path:
    return group_root / str(dataset) / str(member["processed_dir"])


def _molecule_context_records(
    *,
    dataset: str,
    split_phase: str,
    seed: int,
    evaluation_seed: int,
    solver_key: str,
    target_nfe: int,
    runtime_nfe: int,
    scheduler_key: str,
    details: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_step: int,
    nfe_role: str,
    parent_row_signature: str,
    protocol_hash: str,
    per_context_metrics: Sequence[Mapping[str, Any]],
    uniform_by_context_id: Mapping[str, Mapping[str, Any]] | None,
    rollout_steps: int,
    train_tuning_context_metadata: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    uniform_by_context = dict(uniform_by_context_id or {})
    for metric_row in per_context_metrics:
        raw_context_id = str(metric_row["context_id"])
        context_id = raw_context_id
        context_embedding_id = str(metric_row.get("context_embedding_id") or f"{checkpoint['checkpoint_id']}:{raw_context_id}")
        uniform_metric_row = metric_row if str(scheduler_key) == UNIFORM_SCHEDULER_KEY else uniform_by_context.get(context_id)
        reward_columns = uniform_anchored_objective_columns(
            dict(metric_row, scheduler_key=str(scheduler_key)),
            dict(uniform_metric_row or {}),
            MOLECULE_METRIC_SPECS,
            uniform_schedule_key=UNIFORM_SCHEDULER_KEY,
        ) if uniform_metric_row is not None else {"u_comp_uniform": None, "reward_metric_count": 0, "reward_metric_weights_json": "{}", "reward_metric_directions_json": "{}"}
        row_signature_payload = {
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "context_id": context_id,
            "context_embedding_id": context_embedding_id,
            "scheduler_key": str(scheduler_key),
            "seed": int(seed),
            "solver_key": str(solver_key),
            "target_nfe": int(target_nfe),
        }
        row_signature = hashlib.sha256(
            json.dumps(row_signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "benchmark_family": SCENARIO_FAMILY_MOLECULE,
                "canonical_layout_version": CANONICAL_LAYOUT_VERSION,
                "scenario_key": str(dataset),
                "scenario_family": SCENARIO_FAMILY_MOLECULE,
                "nfe_role": str(nfe_role),
                "checkpoint_step": int(checkpoint_step),
                "checkpoint_maturity_label": _checkpoint_maturity_label(int(checkpoint_step)),
                "checkpoint_maturity_index": _checkpoint_maturity_index(int(checkpoint_step)),
                "effective_train_steps": int(checkpoint.get("effective_train_steps", checkpoint_step)),
                "checkpoint_export_protocol": str(checkpoint.get("checkpoint_export_protocol", "")),
                "parent_row_signature": str(parent_row_signature),
                "protocol_hash": str(protocol_hash),
                "dataset": str(dataset),
                "split_phase": str(split_phase),
                "seed": int(seed),
                "logical_seed": int(seed),
                "evaluation_seed": int(evaluation_seed),
                "solver_key": str(solver_key),
                "target_nfe": int(target_nfe),
                "runtime_nfe": int(runtime_nfe),
                "macro_steps": int(runtime_nfe),
                "realized_nfe": _realized_nfe_for_solver(str(solver_key), int(runtime_nfe)),
                "scheduler_key": str(scheduler_key),
                "schedule_family": schedule_family_for_key(str(scheduler_key)),
                "density_source_key": density_source_key_for_schedule(str(scheduler_key)),
                "context_schema": MOLECULE_CONTEXT_SCHEMA,
                "axis_dataset": str(dataset),
                "axis_series": str(metric_row.get("axis_member", "")),
                "axis_time_bin": _time_bin_for_target(int(metric_row.get("target_t", 0)), [int(row.get("target_t", 0)) for row in per_context_metrics]),
                "axis_record": str(metric_row.get("axis_trajectory", "")),
                "axis_window": str(metric_row.get("axis_window", "")),
                "axis_stratum": str(metric_row.get("axis_stratum", "")),
                "axis_member": str(metric_row.get("axis_member", "")),
                "axis_formula": str(metric_row.get("axis_formula", "")),
                "axis_atom_count": metric_row.get("axis_atom_count", ""),
                "axis_trajectory": str(metric_row.get("axis_trajectory", "")),
                "axis_iso_id": str(metric_row.get("axis_iso_id", "")),
                "axis_flags": str(metric_row.get("axis_flags", "")),
                "schedule_grid_hash": str(details["schedule_grid_hash"]),
                "example_idx": int(metric_row.get("example_idx", 0)),
                "series_id": str(metric_row.get("axis_member", "")),
                "series_idx": "",
                "target_t": int(metric_row.get("target_t", 0)),
                "history_start": int(metric_row.get("history_start", 0)),
                "history_stop": int(metric_row.get("history_stop", 0)),
                "target_stop": int(metric_row.get("target_stop", 0)),
                "context_id": context_id,
                "context_embedding_id": context_embedding_id,
                "checkpoint_id": str(checkpoint["checkpoint_id"]),
                "molecule_kabsch_rmsd_3d": metric_row.get("molecule_kabsch_rmsd_3d"),
                "molecule_ensemble_velocity_norm_w1": metric_row.get("molecule_ensemble_velocity_norm_w1"),
                "molecule_ensemble_acceleration_norm_w1": metric_row.get("molecule_ensemble_acceleration_norm_w1"),
                "molecule_rollout_velocity_norm_w1": metric_row.get("molecule_rollout_velocity_norm_w1"),
                "molecule_rollout_acceleration_norm_w1": metric_row.get("molecule_rollout_acceleration_norm_w1"),
                "molecule_coordinate_w1_mean": metric_row.get("molecule_coordinate_w1_mean"),
                "molecule_pair_distance_w1": metric_row.get("molecule_pair_distance_w1"),
                **reward_columns,
                "gipo_reward_protocol": GIPO_PROTOCOL,
                "reward_anchor_schedule_key": UNIFORM_SCHEDULER_KEY,
                "reward_utility_transform": "directional_log_uniform_anchor",
                "reward_granularity": "context_window_metric_components",
                "num_eval_samples": int(metric_row.get("num_eval_samples", 1) or 1),
                "eval_horizon": int(rollout_steps),
                "batch_size": "",
                "sample_seed_start": int(evaluation_seed),
                "sample_seed_values_json": json.dumps([int(evaluation_seed)], separators=(",", ":")),
                "row_signature": str(row_signature),
                **dict(train_tuning_context_metadata or {}),
            }
        )
    return rows


def _existing_uniform_context_rows(
    row_recorder: Mapping[str, Any],
    *,
    dataset: str,
    split_phase: str,
    seed: int,
    solver_key: str,
    target_nfe: int,
    checkpoint_id: str,
) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for row in row_recorder.get("context_rows_by_signature", {}).values():
        if str(row.get("scheduler_key")) != UNIFORM_SCHEDULER_KEY:
            continue
        if str(row.get("dataset")) != str(dataset) or str(row.get("split_phase")) != str(split_phase):
            continue
        if str(row.get("solver_key")) != str(solver_key) or int(row.get("target_nfe", -1)) != int(target_nfe):
            continue
        if int(row.get("seed", -1)) != int(seed):
            continue
        if str(row.get("checkpoint_id", "")) != str(checkpoint_id):
            continue
        context_id = str(row.get("context_id", "") or "").strip()
        if context_id:
            out[context_id] = dict(row, scheduler_key=UNIFORM_SCHEDULER_KEY)
    return out


def _fixed_schedule_details(scheduler_key: str, runtime_nfe: int) -> Dict[str, Any]:
    fixed_grid = build_schedule_grid(str(scheduler_key), int(runtime_nfe))
    if fixed_grid is None:
        raise ValueError(f"Unable to build fixed grid for scheduler={scheduler_key}")
    grid_hash = schedule_grid_hash(fixed_grid)
    details: Dict[str, Any] = {
        "time_grid": [float(x) for x in fixed_grid],
        "schedule_grid_hash": str(grid_hash),
        "reference_time_alignment": schedule_time_alignment(str(scheduler_key)),
        "paper_duplicate_count": 0,
        "reference_macro_steps": int(runtime_nfe),
    }
    details.update(fixed_schedule_shape_statistics(fixed_grid))
    return details


def _schedule_details_from_case(case: Mapping[str, Any], runtime_nfe: int) -> Dict[str, Any]:
    if "time_grid" not in case:
        return _fixed_schedule_details(str(case["scheduler_key"]), int(runtime_nfe))
    grid = validate_time_grid(case["time_grid"], macro_steps=int(runtime_nfe))
    details: Dict[str, Any] = {
        "time_grid": [float(x) for x in grid],
        "schedule_grid_hash": str(case.get("schedule_grid_hash") or schedule_grid_hash(grid)),
        "reference_time_alignment": str(case.get("reference_time_alignment", "summary_density_time_grid")),
        "paper_duplicate_count": int(case.get("paper_duplicate_count", 0) or 0),
        "macro_steps": int(case.get("macro_steps", runtime_nfe) or runtime_nfe),
        "runtime_nfe": int(case.get("runtime_nfe", runtime_nfe) or runtime_nfe),
        "realized_nfe": int(case.get("realized_nfe", _realized_nfe_for_solver(str(case.get("solver_key", "euler")), int(runtime_nfe)))),
        "reference_macro_steps": int(case.get("reference_macro_steps", runtime_nfe) or runtime_nfe),
    }
    details.update(fixed_schedule_shape_statistics(grid))
    if str(case.get("schedule_summary_path", "")).strip():
        details["schedule_summary_path"] = str(case["schedule_summary_path"])
    return details


def _evaluation_protocol_fields(result_row: Mapping[str, Any], *, eval_horizon: int) -> Dict[str, Any]:
    protocol = dict(result_row.get("evaluation_protocol", {}) or {})
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return {
        "eval_horizon": int(eval_horizon),
        "evaluation_protocol_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "chosen_t0s_hash": str(protocol.get("chosen_t0s_hash", "")),
    }


def _build_row(*, benchmark_family: str, split_phase: str, seed: int, dataset: str, checkpoint: Mapping[str, Any], checkpoint_step: int, nfe_role: str, target_nfe: int, runtime_nfe: int, solver_key: str, scheduler_key: str, details: Mapping[str, Any], metrics: Mapping[str, Any], row_signature: str, protocol_hash: str) -> Dict[str, Any]:
    selection_metric = selection_metric_for_family(str(benchmark_family))
    nfe = normalize_solver_nfe_fields(
        str(solver_key),
        int(target_nfe),
        macro_steps=details.get("macro_steps"),
        runtime_nfe=runtime_nfe,
        realized_nfe=metrics.get("realized_nfe", details.get("realized_nfe")),
        source=f"{scheduler_key} row {solver_key}/{target_nfe}",
    )
    return {
        "benchmark_family": str(benchmark_family),
        "canonical_layout_version": CANONICAL_LAYOUT_VERSION,
        "scenario_key": str(dataset),
        "scenario_family": str(benchmark_family),
        "nfe_role": str(nfe_role),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_maturity_label": _checkpoint_maturity_label(int(checkpoint_step)),
        "checkpoint_maturity_index": _checkpoint_maturity_index(int(checkpoint_step)),
        "split_phase": str(split_phase),
        "seed": int(seed),
        "dataset": str(dataset),
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "checkpoint_path": _logical_artifact_path(str(checkpoint["checkpoint_path"])),
        "backbone_name": str(checkpoint.get("backbone_name", "otflow")),
        "member_key": str(checkpoint.get("member_key", "")),
        "stratum": str(checkpoint.get("stratum", "")),
        "formula": str(checkpoint.get("formula", "")),
        "source_zip_name": str(checkpoint.get("source_zip_name", "")),
        "train_steps": int(checkpoint["train_steps"]),
        "effective_train_steps": int(checkpoint.get("effective_train_steps", checkpoint["train_steps"])),
        "checkpoint_export_protocol": str(checkpoint.get("checkpoint_export_protocol", "")),
        "train_budget_label": str(checkpoint["train_budget_label"]),
        "target_nfe": int(target_nfe),
        "runtime_nfe": int(nfe.runtime_nfe),
        "macro_steps": int(nfe.macro_steps),
        "solver_key": str(solver_key),
        "solver_name": str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
        "scheduler_key": str(scheduler_key),
        "scheduler_variant_key": str(scheduler_key),
        "scheduler_variant_name": schedule_display_name(str(scheduler_key)),
        "schedule_name": schedule_display_name(str(scheduler_key)),
        "schedule_family": schedule_family_for_key(str(scheduler_key)),
        "density_source_key": density_source_key_for_schedule(str(scheduler_key)),
        "student_training_mode": "",
        "row_signature": str(row_signature),
        "signal_trace_key": None,
        "signal_validation_spearman": None,
        "info_growth_scale": None,
        "reference_macro_factor": None,
        "paper_duplicate_count": int(details.get("paper_duplicate_count", 0) or 0),
        "experiment_scope": solver_experiment_scope(str(solver_key)),
        "selection_metric": str(selection_metric),
        "selection_metric_value": metrics.get(selection_metric),
        "reference_macro_steps": int(details.get("reference_macro_steps", nfe.macro_steps)),
        "reference_time_alignment": str(details.get("reference_time_alignment", schedule_time_alignment(str(scheduler_key)))),
        "runtime_grid_q25": details.get("runtime_grid_q25"),
        "runtime_grid_q50": details.get("runtime_grid_q50"),
        "runtime_grid_q75": details.get("runtime_grid_q75"),
        "forecast_crps": metrics.get("forecast_crps"),
        "forecast_mse": metrics.get("forecast_mse"),
        "forecast_mase": metrics.get("forecast_mase"),
        "forecast_mase_scale_kind": metrics.get("forecast_mase_scale_kind"),
        "forecast_mase_scale_period": metrics.get("forecast_mase_scale_period"),
        "score_main": metrics.get("score_main"),
        "disc_auc": metrics.get("disc_auc"),
        "disc_auc_gap": metrics.get("disc_auc_gap"),
        "temporal_uw1": metrics.get("temporal_uw1"),
        "temporal_cw1": metrics.get("temporal_cw1"),
        "temporal_tstr_f1": metrics.get("temporal_tstr_f1"),
        "temporal_tstr_f1_applicable": metrics.get("temporal_tstr_f1_applicable"),
        "u_l1": metrics.get("u_l1"),
        "c_l1": metrics.get("c_l1"),
        "spread_specific_error": metrics.get("spread_specific_error"),
        "imbalance_specific_error": metrics.get("imbalance_specific_error"),
        "ret_vol_acf_error": metrics.get("ret_vol_acf_error"),
        "impact_response_error": metrics.get("impact_response_error"),
        "molecule_kabsch_rmsd_3d": metrics.get("molecule_kabsch_rmsd_3d"),
        "molecule_ensemble_velocity_norm_w1": metrics.get("molecule_ensemble_velocity_norm_w1"),
        "molecule_ensemble_acceleration_norm_w1": metrics.get("molecule_ensemble_acceleration_norm_w1"),
        "molecule_rollout_velocity_norm_w1": metrics.get("molecule_rollout_velocity_norm_w1"),
        "molecule_rollout_acceleration_norm_w1": metrics.get("molecule_rollout_acceleration_norm_w1"),
        "molecule_coordinate_w1_mean": metrics.get("molecule_coordinate_w1_mean"),
        "molecule_pair_distance_w1": metrics.get("molecule_pair_distance_w1"),
        "forecast_relative_crps_gain_vs_uniform": metrics.get("forecast_relative_crps_gain_vs_uniform"),
        "forecast_relative_mase_gain_vs_uniform": metrics.get("forecast_relative_mase_gain_vs_uniform"),
        "relative_score_gain_vs_uniform": metrics.get("relative_score_gain_vs_uniform"),
        "realized_nfe": int(nfe.realized_nfe),
        "latency_ms_per_sample": metrics.get("latency_ms_per_sample", metrics.get("efficiency_ms_per_sample")),
        "num_eval_samples": metrics.get("num_eval_samples"),
        "eval_examples": metrics.get("eval_examples"),
        "eval_windows": metrics.get("eval_windows"),
        "eval_horizon": metrics.get("eval_horizon"),
        "evaluation_protocol_hash": metrics.get("evaluation_protocol_hash"),
        "chosen_t0s_hash": metrics.get("chosen_t0s_hash"),
        "chosen_examples_hash": metrics.get("chosen_examples_hash"),
        "schedule_grid_hash": details.get("schedule_grid_hash"),
        "protocol_hash": str(protocol_hash),
        "row_status": "complete",
        "train_tuning_fraction": "",
        "train_tuning_seed": "",
        "train_tuning_strata": "",
        "train_tuning_sampler": "",
    }


def _scheduler_cases_for_datasets(
    cli_args: argparse.Namespace,
    datasets: Iterable[str],
    *,
    include_summary_cases: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    schedule_names = _parse_schedule_names(str(cli_args.baseline_scheduler_names))
    if UNIFORM_SCHEDULER_KEY in schedule_names:
        schedule_names = [UNIFORM_SCHEDULER_KEY] + [key for key in schedule_names if key != UNIFORM_SCHEDULER_KEY]
    summary_cases = _load_schedule_summary_cases(str(getattr(cli_args, "schedule_summary_json", ""))) if include_summary_cases else []
    requested_summary_names = _parse_summary_schedule_names(str(getattr(cli_args, "summary_scheduler_names", "")))
    if requested_summary_names and include_summary_cases:
        summary_cases = [case for case in summary_cases if str(case.get("scheduler_key")) in set(requested_summary_names)]
        observed_summary_names = {str(case.get("scheduler_key")) for case in summary_cases}
        missing = sorted(set(requested_summary_names) - observed_summary_names)
        if missing:
            raise ValueError(f"Schedule summary is missing requested schedules: {missing}")
    cases = [{"scheduler_key": key} for key in schedule_names] + summary_cases
    if not cases:
        raise ValueError("At least one fixed or summary-derived schedule is required.")
    return {str(dataset): [dict(case) for case in cases] for dataset in datasets}


def _run_forecast_phase(cli_args: argparse.Namespace, *, row_recorder: Mapping[str, Any], split_phase: str, seeds: Sequence[int], scheduler_cases_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    dataset_root = resolve_project_path(str(cli_args.dataset_root))
    shared_backbone_root = resolve_project_path(str(cli_args.shared_backbone_root))
    device = resolve_torch_device(str(cli_args.device))
    dataset_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    datasets = parse_forecast_datasets(str(cli_args.forecast_datasets))
    target_nfes = _target_nfe_values_for_args(cli_args)
    checkpoint_steps = _checkpoint_steps_for_args(cli_args)
    nfe_role = str(getattr(cli_args, "nfe_role", NFE_ROLE_SEEN) or NFE_ROLE_SEEN)
    for dataset_idx, dataset in enumerate(datasets):
        for checkpoint_step in checkpoint_steps:
            step_args = _args_for_checkpoint_step(cli_args, int(checkpoint_step))
            cache_key = (str(dataset), int(checkpoint_step))
            if cache_key not in dataset_cache:
                dataset_cache[cache_key] = load_forecast_checkpoint_splits(
                    cli_args=step_args,
                    dataset_root=dataset_root,
                    shared_backbone_root=shared_backbone_root,
                    dataset=dataset,
                    device=device,
                )
            checkpoint = dataset_cache[cache_key]
            model = checkpoint["model"]
            cfg = checkpoint["cfg"]
            splits = checkpoint["splits"]
            train_tuning_reference_examples = int(len(splits.get("val", [])))
            selected_examples_cap, selected_examples_cap_source = _split_example_cap(cli_args, str(split_phase))
            if str(split_phase) == TRAIN_TUNING_PHASE:
                eval_ds = splits["train"]
            elif str(split_phase) == VALIDATION_PHASE:
                eval_ds = splits["val"]
            else:
                eval_ds = splits["test"]
            selection_groups: List[Dict[str, Any]] = []
            for seed in seeds:
                if str(split_phase) == TRAIN_TUNING_PHASE:
                    assert selected_examples_cap is not None
                    tuning_seed = int(cli_args.train_tuning_seed) + int(seed) + 1_000 * dataset_idx
                    uncapped_candidate_examples = train_tuning_target_example_count(
                        len(eval_ds),
                        fraction=float(cli_args.eval_train_fraction),
                        sampling_mode=str(cli_args.train_tuning_sampling_mode),
                        strata=int(cli_args.train_tuning_strata),
                        reference_examples=int(train_tuning_reference_examples),
                        train_split_fraction=float(cli_args.train_tuning_train_split_fraction),
                        val_split_fraction=float(cli_args.train_tuning_val_split_fraction),
                    )
                    candidate_examples = choose_forecast_train_tuning_indices(
                        eval_ds,
                        fraction=float(cli_args.eval_train_fraction),
                        seed=tuning_seed,
                        strata=int(cli_args.train_tuning_strata),
                        dataset=str(dataset),
                        sampling_mode=str(cli_args.train_tuning_sampling_mode),
                        reference_examples=int(train_tuning_reference_examples),
                        train_split_fraction=float(cli_args.train_tuning_train_split_fraction),
                        val_split_fraction=float(cli_args.train_tuning_val_split_fraction),
                    )
                else:
                    tuning_seed = int(seed) + 1_000 * dataset_idx
                    eval_examples = int(selected_examples_cap) if selected_examples_cap is not None else int(len(eval_ds))
                    candidate_examples = choose_forecast_example_indices(
                        eval_ds,
                        n_examples=int(eval_examples),
                        seed=tuning_seed,
                    )
                    uncapped_candidate_examples = int(len(eval_ds))
                selection_groups.append(
                    {
                        "candidate_indices": [int(idx) for idx in candidate_examples],
                        "uncapped_candidate_examples": int(uncapped_candidate_examples),
                        "selection_record": {
                            "seed": int(seed),
                            "tuning_seed": int(tuning_seed),
                        },
                    }
                )
            selected_groups, selection_records, _global_selection_meta = _cap_context_index_groups(
                selection_groups,
                cap=_selection_cap_for_groups(selection_groups, selected_examples_cap),
                seed=int(cli_args.train_tuning_seed) + 1_000 * dataset_idx + int(checkpoint_step),
                salt=f"forecast|{dataset}|{split_phase}|{checkpoint_step}",
            )
            selections_by_seed: Dict[int, Tuple[np.ndarray, Dict[str, Any], int]] = {}
            for selected, record in zip(selected_groups, selection_records):
                selections_by_seed[int(record["seed"])] = (selected, record, int(record["tuning_seed"]))
            for seed in seeds:
                chosen_examples, selection_meta, tuning_seed = selections_by_seed[int(seed)]
                if len(chosen_examples) == 0:
                    continue
                for target_idx, target_nfe in enumerate(target_nfes):
                    for solver_idx, solver_key in enumerate(normalize_solver_keys(str(cli_args.solver_names))):
                        runtime_nfe = solver_macro_steps(str(solver_key), int(target_nfe))
                        scheduler_cases = list(scheduler_cases_by_dataset[str(dataset)])
                        existing_rows, pending_cases = _pending_scheduler_cases(
                            row_recorder,
                            benchmark_family=FORECAST_FAMILY,
                            split_phase=str(split_phase),
                            seed=int(seed),
                            dataset=str(dataset),
                            checkpoint_id=str(checkpoint["checkpoint_id"]),
                            checkpoint_step=int(checkpoint_step),
                            target_nfe=int(target_nfe),
                            solver_key=str(solver_key),
                            scheduler_cases=scheduler_cases,
                        )
                        rows.extend(existing_rows)
                        cell_uniform_metrics: Optional[Mapping[str, Any]] = None
                        for existing_row in existing_rows:
                            if str(existing_row.get("scheduler_key")) == UNIFORM_SCHEDULER_KEY:
                                cell_uniform_metrics = existing_row
                        for case in pending_cases:
                            scheduler_key = str(case["scheduler_key"])
                            details = _schedule_details_from_case(case, int(runtime_nfe))
                            eval_seed = int(seed) + 100_000 * dataset_idx + 1_000 * target_idx + solver_idx
                            metrics = evaluate_forecast_schedule(
                                model,
                                eval_ds,
                                cfg,
                                solver_name=str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
                                runtime_nfe=int(runtime_nfe),
                                target_nfe=int(target_nfe),
                                time_grid=details["time_grid"],
                                num_eval_samples=int(cli_args.num_eval_samples),
                                seed=int(eval_seed),
                                logical_seed=int(seed),
                                scheduler_key=str(scheduler_key),
                                dataset_key=str(dataset),
                                split_phase=str(split_phase),
                                checkpoint_id=str(checkpoint["checkpoint_id"]),
                                example_indices=chosen_examples,
                                batch_size=int(cli_args.forecast_eval_batch_size),
                                progress_label=f"{split_phase} {dataset} {scheduler_key} step={checkpoint_step} seed={seed} {solver_key}/{target_nfe}",
                                return_per_example_rows=_write_context_rows_enabled(cli_args),
                                return_context_embeddings=_write_context_rows_enabled(cli_args),
                                context_embedding_kind=str(getattr(cli_args, "context_embedding_kind", "ctx_summary")),
                            )
                            if scheduler_key != UNIFORM_SCHEDULER_KEY and cell_uniform_metrics is not None:
                                metrics = dict(metrics)
                                metrics["forecast_relative_crps_gain_vs_uniform"] = _safe_relative_gain(
                                    metrics.get("forecast_crps"),
                                    cell_uniform_metrics.get("forecast_crps"),
                                )
                                metrics["forecast_relative_mase_gain_vs_uniform"] = _safe_relative_gain(
                                    metrics.get("forecast_mase"),
                                    cell_uniform_metrics.get("forecast_mase"),
                                )
                            row = _build_row(
                                benchmark_family=FORECAST_FAMILY,
                                split_phase=str(split_phase),
                                seed=int(seed),
                                dataset=str(dataset),
                                checkpoint=checkpoint,
                                checkpoint_step=int(checkpoint_step),
                                nfe_role=nfe_role,
                                target_nfe=int(target_nfe),
                                runtime_nfe=int(runtime_nfe),
                                solver_key=str(solver_key),
                                scheduler_key=scheduler_key,
                                details=details,
                                metrics=metrics,
                                row_signature=str(case["row_signature"]),
                                protocol_hash=str(row_recorder["protocol_hash"]),
                            )
                            row.update(
                                _selection_metadata_row_fields(
                                    selection_meta,
                                    cap_source=str(selected_examples_cap_source),
                                    context_sample_count=_context_sample_cap(cli_args),
                                )
                            )
                            if str(split_phase) == TRAIN_TUNING_PHASE:
                                row.update(
                                    {
                                        "train_tuning_fraction": float(cli_args.eval_train_fraction),
                                        "train_tuning_seed": int(tuning_seed),
                                        "train_tuning_strata": int(cli_args.train_tuning_strata),
                                        "train_tuning_sampler": train_tuning_sampler_key(str(cli_args.train_tuning_sampling_mode)),
                                        "train_tuning_sampling_mode": str(cli_args.train_tuning_sampling_mode),
                                        "train_tuning_reference_examples": int(train_tuning_reference_examples),
                                        "train_tuning_target_examples": int(selection_meta["uncapped_candidate_examples"]),
                                        "train_tuning_uncapped_candidate_examples": int(selection_meta["uncapped_candidate_examples"]),
                                        "train_tuning_train_split_fraction": float(cli_args.train_tuning_train_split_fraction),
                                        "train_tuning_val_split_fraction": float(cli_args.train_tuning_val_split_fraction),
                                    }
                                )
                            _append_row_record(row_recorder, row)
                            if _write_context_rows_enabled(cli_args):
                                context_rows = []
                                for detail_row in list(metrics.get("per_example_rows", []) or []):
                                    copied_detail = dict(detail_row)
                                    copied_detail.update(
                                        {
                                            "benchmark_family": FORECAST_FAMILY,
                                            "canonical_layout_version": CANONICAL_LAYOUT_VERSION,
                                            "scenario_key": str(dataset),
                                            "scenario_family": FORECAST_FAMILY,
                                            "nfe_role": nfe_role,
                                            "checkpoint_step": int(checkpoint_step),
                                            "checkpoint_maturity_label": _checkpoint_maturity_label(int(checkpoint_step)),
                                            "checkpoint_maturity_index": _checkpoint_maturity_index(int(checkpoint_step)),
                                            "effective_train_steps": int(checkpoint.get("effective_train_steps", checkpoint_step)),
                                            "checkpoint_export_protocol": str(checkpoint.get("checkpoint_export_protocol", "")),
                                            "parent_row_signature": str(case["row_signature"]),
                                            "protocol_hash": str(row_recorder["protocol_hash"]),
                                            "schedule_family": schedule_family_for_key(str(scheduler_key)),
                                            "density_source_key": density_source_key_for_schedule(str(scheduler_key)),
                                        }
                                    )
                                    if str(split_phase) == TRAIN_TUNING_PHASE:
                                        copied_detail.update(
                                            {
                                                "train_tuning_fraction": float(cli_args.eval_train_fraction),
                                                "train_tuning_seed": int(tuning_seed),
                                                "train_tuning_strata": int(cli_args.train_tuning_strata),
                                                "train_tuning_sampler": train_tuning_sampler_key(str(cli_args.train_tuning_sampling_mode)),
                                            }
                                        )
                                    context_rows.append(copied_detail)
                                _append_context_records(
                                    row_recorder,
                                    context_rows,
                                    context_embeddings=dict(metrics.get("context_embeddings", {}) or {}),
                                    metadata={
                                        "benchmark_family": FORECAST_FAMILY,
                                        "checkpoint_id": str(checkpoint["checkpoint_id"]),
                                        "checkpoint_step": int(checkpoint_step),
                                        "dataset": str(dataset),
                                        "split_phase": str(split_phase),
                                        "context_schema": "forecast_window",
                                        "context_embedding_kind": str(getattr(cli_args, "context_embedding_kind", "ctx_summary")),
                                        "history_len": int(getattr(cfg, "history_len", 0)),
                                        "horizon": int(getattr(eval_ds, "horizon", 1)),
                                        "chosen_examples_hash": str(metrics.get("chosen_examples_hash", "")),
                                        "evaluation_protocol_hash": str(metrics.get("evaluation_protocol_hash", "")),
                                    },
                                )
                            rows.append(row)
                            if scheduler_key == UNIFORM_SCHEDULER_KEY:
                                cell_uniform_metrics = row
    return rows


def _run_conditional_generation_phase(cli_args: argparse.Namespace, *, row_recorder: Mapping[str, Any], split_phase: str, seeds: Sequence[int], scheduler_cases_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    shared_backbone_root = resolve_project_path(str(cli_args.shared_backbone_root))
    device = resolve_torch_device(str(cli_args.device))
    dataset_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    datasets = parse_conditional_generation_datasets(str(cli_args.conditional_generation_datasets))
    target_nfes = _target_nfe_values_for_args(cli_args)
    checkpoint_steps = _checkpoint_steps_for_args(cli_args)
    nfe_role = str(getattr(cli_args, "nfe_role", NFE_ROLE_SEEN) or NFE_ROLE_SEEN)
    for dataset_idx, dataset in enumerate(datasets):
        for checkpoint_step in checkpoint_steps:
            step_args = _args_for_checkpoint_step(cli_args, int(checkpoint_step))
            cache_key = (str(dataset), int(checkpoint_step))
            if cache_key not in dataset_cache:
                dataset_cache[cache_key] = load_conditional_generation_checkpoint_splits(
                    cli_args=step_args,
                    shared_backbone_root=shared_backbone_root,
                    dataset=dataset,
                    device=device,
                )
            checkpoint = dataset_cache[cache_key]
            model = checkpoint["model"]
            cfg = checkpoint["cfg"]
            splits = checkpoint["splits"]
            if str(split_phase) == TRAIN_TUNING_PHASE:
                eval_ds = splits["train"]
                split_key = "train"
            elif str(split_phase) == VALIDATION_PHASE:
                eval_ds = splits["val"]
                split_key = "val"
            else:
                eval_ds = splits["test"]
                split_key = "test"
            eval_horizon = resolved_eval_horizon(step_args, str(dataset))
            selected_examples_cap, selected_examples_cap_source = _split_example_cap(cli_args, str(split_phase))
            available_windows = int(len(getattr(eval_ds, "start_indices", [])))
            if str(split_phase) == TRAIN_TUNING_PHASE:
                assert selected_examples_cap is not None
                eval_windows = train_tuning_target_example_count(
                    available_windows,
                    fraction=float(cli_args.eval_train_fraction),
                    sampling_mode=TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION,
                    strata=int(cli_args.train_tuning_strata),
                )
            else:
                requested_attr = "eval_windows_val" if str(split_phase) == VALIDATION_PHASE else "eval_windows_test"
                requested_windows = int(getattr(cli_args, requested_attr, 0))
                if requested_windows < 0:
                    raise ValueError(f"--{requested_attr} must be nonnegative, got {requested_windows!r}.")
                eval_windows = (
                    int(requested_windows)
                    if requested_windows > 0
                    else int(resolved_eval_windows(step_args, str(dataset), split_key))
                )
            selection_groups: List[Dict[str, Any]] = []
            for seed in seeds:
                if str(split_phase) == TRAIN_TUNING_PHASE:
                    selection_seed = int(cli_args.train_tuning_seed) + int(seed) + 1_000 * dataset_idx
                    train_starts = [int(value) for value in getattr(eval_ds, "start_indices", [])]
                    selected_positions, target_examples = _choose_stratified_train_tuning_positions(
                        len(train_starts),
                        fraction=float(cli_args.eval_train_fraction),
                        seed=int(selection_seed),
                        strata=int(cli_args.train_tuning_strata),
                        dataset=str(dataset),
                        salt=f"conditional_train_tuning|{dataset}",
                    )
                    candidate_eval_t0s = [int(train_starts[pos]) for pos in selected_positions]
                    uncapped_candidate_examples = int(target_examples)
                else:
                    selection_seed = int(seed) + 1_000 * dataset_idx
                    candidate_eval_t0s = _choose_valid_windows(
                        eval_ds,
                        horizon=int(eval_horizon),
                        n_windows=int(eval_windows),
                        seed=int(selection_seed),
                    )
                    uncapped_candidate_examples = max(int(available_windows), len(candidate_eval_t0s))
                selection_groups.append(
                    {
                        "candidate_indices": [int(t0) for t0 in candidate_eval_t0s],
                        "uncapped_candidate_examples": int(uncapped_candidate_examples),
                        "selection_record": {
                            "seed": int(seed),
                            "selection_seed": int(selection_seed),
                        },
                    }
                )
            selected_groups, selection_records, _global_selection_meta = _cap_context_index_groups(
                selection_groups,
                cap=_selection_cap_for_groups(selection_groups, selected_examples_cap),
                seed=1_000 * int(dataset_idx) + int(checkpoint_step),
                salt=f"conditional|{dataset}|{split_phase}|{checkpoint_step}",
            )
            selections_by_seed: Dict[int, Tuple[np.ndarray, Dict[str, Any], int]] = {}
            for selected, record in zip(selected_groups, selection_records):
                selections_by_seed[int(record["seed"])] = (selected, record, int(record["selection_seed"]))
            for seed in seeds:
                chosen_eval_t0s, selection_meta, selection_seed = selections_by_seed[int(seed)]
                if len(chosen_eval_t0s) == 0:
                    continue
                for target_idx, target_nfe in enumerate(target_nfes):
                    for solver_idx, solver_key in enumerate(normalize_solver_keys(str(cli_args.solver_names))):
                        runtime_nfe = solver_macro_steps(str(solver_key), int(target_nfe))
                        existing_rows, pending_cases = _pending_scheduler_cases(
                            row_recorder,
                            benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                            split_phase=str(split_phase),
                            seed=int(seed),
                            dataset=str(dataset),
                            checkpoint_id=str(checkpoint["checkpoint_id"]),
                            checkpoint_step=int(checkpoint_step),
                            target_nfe=int(target_nfe),
                            solver_key=str(solver_key),
                            scheduler_cases=list(scheduler_cases_by_dataset[str(dataset)]),
                        )
                        rows.extend(existing_rows)
                        cell_uniform_metrics: Optional[Mapping[str, Any]] = None
                        for existing_row in existing_rows:
                            if str(existing_row.get("scheduler_key")) == UNIFORM_SCHEDULER_KEY:
                                cell_uniform_metrics = existing_row
                        existing_uniform_context_rows = _existing_uniform_context_rows(
                            row_recorder,
                            dataset=str(dataset),
                            split_phase=str(split_phase),
                            seed=int(seed),
                            solver_key=str(solver_key),
                            target_nfe=int(target_nfe),
                            checkpoint_id=str(checkpoint["checkpoint_id"]),
                        )
                        cell_uniform_per_window_metrics_by_t0: Dict[int, Dict[str, Any]] = {
                            int(row["target_t"]): dict(row)
                            for row in existing_uniform_context_rows.values()
                            if str(row.get("context_schema")) == "conditional_generation_window"
                            and row.get("target_t", "") not in ("", None)
                        }
                        conditional_embeddings_by_t0: Dict[int, List[float]] = {}
                        if _write_context_rows_enabled(cli_args):
                            conditional_embeddings_by_t0 = _extract_conditional_context_embeddings(
                                model=model,
                                ds=eval_ds,
                                chosen_t0s=[int(x) for x in chosen_eval_t0s.tolist()],
                                device=device,
                                context_embedding_kind=str(getattr(cli_args, "context_embedding_kind", "ctx_summary")),
                            )
                        for case in pending_cases:
                            scheduler_key = str(case["scheduler_key"])
                            details = _schedule_details_from_case(case, int(runtime_nfe))
                            grid_spec = {
                                "grid_name": scheduler_key,
                                "grid_kind": "fixed_diffusion_flow_time_grid",
                                "selection_group": scheduler_key,
                                "comparison_role": "transferred" if scheduler_key in TRANSFER_SCHEDULE_KEYS else "baseline",
                                "solver_name": str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
                                "nfe": int(runtime_nfe),
                                "time_grid": details["time_grid"],
                            }
                            metrics_seed = int(seed) + 1_000_000 * dataset_idx + 10_000 * target_idx + solver_idx
                            result_row = run_fixed_schedule_variant(
                                model=model,
                                ds=eval_ds,
                                cfg=cfg,
                                eval_horizon=int(eval_horizon),
                                eval_windows=int(len(chosen_eval_t0s)),
                                grid_spec=grid_spec,
                                chosen_t0s=chosen_eval_t0s,
                                generation_seed_base=int(metrics_seed),
                                metrics_seed=int(metrics_seed),
                                score_main_only=False,
                            )
                            per_window_metrics_by_t0 = {
                                int(metric_row["target_t"]): dict(metric_row)
                                for metric_row in list(result_row.get("per_window_metric_rows", []) or [])
                                if "target_t" in metric_row
                            }
                            metrics = {
                                "score_main": result_row.get("score_main"),
                                "temporal_tstr_f1": result_row.get("temporal_tstr_f1"),
                                "temporal_tstr_f1_applicable": result_row.get("temporal_tstr_f1_applicable"),
                                "disc_auc": result_row.get("disc_auc"),
                                "disc_auc_gap": result_row.get("disc_auc_gap"),
                                "temporal_uw1": result_row.get("temporal_uw1"),
                                "temporal_cw1": result_row.get("temporal_cw1"),
                                "u_l1": result_row.get("u_l1"),
                                "c_l1": result_row.get("c_l1"),
                                "spread_specific_error": result_row.get("spread_specific_error"),
                                "imbalance_specific_error": result_row.get("imbalance_specific_error"),
                                "ret_vol_acf_error": result_row.get("ret_vol_acf_error"),
                                "impact_response_error": result_row.get("impact_response_error"),
                                "efficiency_ms_per_sample": result_row.get("efficiency_ms_per_sample"),
                                "eval_windows": int(len(chosen_eval_t0s)),
                                "realized_nfe": _realized_nfe_for_solver(str(solver_key), int(runtime_nfe)),
                                **_evaluation_protocol_fields(result_row, eval_horizon=int(eval_horizon)),
                            }
                            if scheduler_key != UNIFORM_SCHEDULER_KEY and cell_uniform_metrics is not None:
                                metrics["relative_score_gain_vs_uniform"] = _safe_relative_gain(metrics.get("score_main"), cell_uniform_metrics.get("score_main"))
                            row = _build_row(
                                benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                                split_phase=str(split_phase),
                                seed=int(seed),
                                dataset=str(dataset),
                                checkpoint=checkpoint,
                                checkpoint_step=int(checkpoint_step),
                                nfe_role=nfe_role,
                                target_nfe=int(target_nfe),
                                runtime_nfe=int(runtime_nfe),
                                solver_key=str(solver_key),
                                scheduler_key=scheduler_key,
                                details=details,
                                metrics=metrics,
                                row_signature=str(case["row_signature"]),
                                protocol_hash=str(row_recorder["protocol_hash"]),
                            )
                            row.update(
                                _selection_metadata_row_fields(
                                    selection_meta,
                                    cap_source=str(selected_examples_cap_source),
                                    context_sample_count=_context_sample_cap(cli_args),
                                )
                            )
                            if str(split_phase) == TRAIN_TUNING_PHASE:
                                row.update(
                                    _train_tuning_metadata(
                                        cli_args,
                                        tuning_seed=int(selection_seed),
                                        target_examples=int(selection_meta["uncapped_candidate_examples"]),
                                        uncapped_candidate_examples=int(selection_meta["uncapped_candidate_examples"]),
                                    )
                                )
                            _append_row_record(row_recorder, row)
                            if _write_context_rows_enabled(cli_args):
                                uniform_score = row.get("score_main") if scheduler_key == UNIFORM_SCHEDULER_KEY else (
                                    cell_uniform_metrics.get("score_main") if cell_uniform_metrics is not None else None
                                )
                                context_rows = _conditional_context_records(
                                    benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                                    dataset=str(dataset),
                                    split_phase=str(split_phase),
                                    seed=int(seed),
                                    evaluation_seed=int(metrics_seed),
                                    solver_key=str(solver_key),
                                    target_nfe=int(target_nfe),
                                    runtime_nfe=int(runtime_nfe),
                                    scheduler_key=scheduler_key,
                                    details=details,
                                    checkpoint=checkpoint,
                                    checkpoint_step=int(checkpoint_step),
                                    nfe_role=nfe_role,
                                    parent_row_signature=str(case["row_signature"]),
                                    protocol_hash=str(row_recorder["protocol_hash"]),
                                    cfg=cfg,
                                    eval_horizon=int(eval_horizon),
                                    chosen_t0s=[int(x) for x in chosen_eval_t0s.tolist()],
                                    score_main=row.get("score_main"),
                                    uniform_score_main=uniform_score,
                                    per_window_metrics_by_t0=per_window_metrics_by_t0,
                                    uniform_per_window_metrics_by_t0=(
                                        per_window_metrics_by_t0
                                        if scheduler_key == UNIFORM_SCHEDULER_KEY
                                        else cell_uniform_per_window_metrics_by_t0
                                    ),
                                    metric_row=row,
                                    uniform_metric_row=row if scheduler_key == UNIFORM_SCHEDULER_KEY else cell_uniform_metrics,
                                    evaluation_protocol_hash=str(row.get("evaluation_protocol_hash", "")),
                                    chosen_t0s_hash=str(row.get("chosen_t0s_hash", "")),
                                    train_tuning_context_metadata=(
                                        {
                                            "train_tuning_fraction": float(cli_args.eval_train_fraction),
                                            "train_tuning_seed": int(selection_seed),
                                            "train_tuning_strata": int(cli_args.train_tuning_strata),
                                            "train_tuning_sampler": train_tuning_sampler_key(TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION),
                                        }
                                        if str(split_phase) == TRAIN_TUNING_PHASE
                                        else None
                                    ),
                                )
                                context_embeddings: Dict[str, List[float]] = {}
                                for context_row in context_rows:
                                    t0 = int(context_row["target_t"])
                                    if t0 in conditional_embeddings_by_t0:
                                        context_embeddings[str(context_row["context_embedding_id"])] = list(conditional_embeddings_by_t0[t0])
                                _append_context_records(
                                    row_recorder,
                                    context_rows,
                                    context_embeddings=context_embeddings,
                                    metadata={
                                        "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                                        "checkpoint_id": str(checkpoint["checkpoint_id"]),
                                        "checkpoint_step": int(checkpoint_step),
                                        "dataset": str(dataset),
                                        "split_phase": str(split_phase),
                                        "context_schema": "conditional_generation_window",
                                        "context_embedding_kind": str(getattr(cli_args, "context_embedding_kind", "ctx_summary")),
                                        "history_len": int(getattr(cfg, "history_len", 0)),
                                        "horizon": int(eval_horizon),
                                    },
                                )
                            rows.append(row)
                            if scheduler_key == UNIFORM_SCHEDULER_KEY:
                                cell_uniform_metrics = row
                                cell_uniform_per_window_metrics_by_t0 = dict(per_window_metrics_by_t0)
    return rows


def _run_molecule_phase(
    cli_args: argparse.Namespace,
    *,
    row_recorder: Mapping[str, Any],
    split_phase: str,
    seeds: Sequence[int],
    scheduler_cases_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    manifest_path = resolve_project_path(str(cli_args.backbone_manifest))
    if not manifest_path.exists():
        raise FileNotFoundError(f"Molecule schedule evaluation requires backbone manifest: {manifest_path}")
    backbone_manifest = load_backbone_manifest(manifest_path)
    group_root = resolve_project_path(str(getattr(cli_args, "molecule_group_root", default_molecule_group_root())))
    device = resolve_torch_device(str(cli_args.device))
    rows: List[Dict[str, Any]] = []
    datasets = parse_molecule_datasets(str(getattr(cli_args, "molecule_datasets", "")))
    target_nfes = _target_nfe_values_for_args(cli_args)
    checkpoint_steps = _checkpoint_steps_for_args(cli_args)
    nfe_role = str(getattr(cli_args, "nfe_role", NFE_ROLE_SEEN) or NFE_ROLE_SEEN)
    split_key = _molecule_split_for_phase(str(split_phase))
    cache: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for dataset_idx, dataset in enumerate(datasets):
        group_manifest = load_molecule_group_manifest(str(dataset), group_root)
        members = [dict(member) for member in trainable_molecule_group_members(group_manifest)]
        if not members:
            raise ValueError(f"Molecule group {dataset!r} has no trainable fixed-shape members.")
        for checkpoint_step in checkpoint_steps:
            selected_examples_cap, selected_examples_cap_source = _split_example_cap(cli_args, str(split_phase))
            selection_groups: List[Dict[str, Any]] = []
            work_items: List[Dict[str, Any]] = []
            for member_idx, member in enumerate(members):
                member_key = str(member["member_key"])
                stratum = str(member["stratum"])
                processed_dir = _molecule_member_processed_dir(group_root, str(dataset), member)
                artifact = find_backbone_artifact(
                    backbone_manifest,
                    backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
                    benchmark_family=MOLECULE_FAMILY,
                    dataset_key=str(dataset),
                    train_steps=int(checkpoint_step),
                    member_key=member_key,
                    stratum=stratum,
                )
                cache_key = (str(dataset), member_key, stratum, int(checkpoint_step))
                if cache_key not in cache:
                    cache[cache_key] = load_molecule_checkpoint_splits(
                        checkpoint_path=str(artifact["checkpoint_path"]),
                        dataset_key=str(dataset),
                        stratum=stratum,
                        processed_dir=processed_dir,
                        rollout_steps=int(getattr(cli_args, "molecule_rollout_steps", 16)),
                        stride_eval=int(getattr(cli_args, "molecule_stride_eval", 1)),
                        device=device,
                    )
                checkpoint = {
                    **artifact,
                    "checkpoint_id": str(artifact["checkpoint_id"]),
                    "checkpoint_path": str(artifact["checkpoint_path"]),
                    "train_budget_label": str(artifact.get("train_budget_label", _checkpoint_maturity_label(int(checkpoint_step)))),
                }
                loaded = cache[cache_key]
                model = loaded["model"]
                cfg = loaded["cfg"]
                ds = loaded["splits"][split_key]
                for seed in seeds:
                    if str(split_phase) == TRAIN_TUNING_PHASE:
                        assert selected_examples_cap is not None
                        selection_seed = int(cli_args.train_tuning_seed) + int(seed) + 10_000 * dataset_idx + 1_000 * member_idx
                        candidate_indices, target_examples = _choose_stratified_train_tuning_positions(
                            len(ds),
                            fraction=float(cli_args.eval_train_fraction),
                            seed=int(selection_seed),
                            strata=int(cli_args.train_tuning_strata),
                            dataset=f"{dataset}|{member_key}|{stratum}",
                            salt=f"molecule_train_tuning|{dataset}|{member_key}|{stratum}",
                        )
                        uncapped_candidate_examples = int(target_examples)
                    else:
                        selection_seed = int(seed) + 10_000 * dataset_idx + 1_000 * member_idx
                        eval_count = int(selected_examples_cap) if selected_examples_cap is not None else int(len(ds))
                        candidate_indices = _choose_molecule_indices(
                            ds,
                            count=int(eval_count),
                            seed=int(selection_seed),
                        )
                        uncapped_candidate_examples = max(int(len(ds)), len(candidate_indices))
                    selection_groups.append(
                        {
                            "candidate_indices": [int(idx) for idx in candidate_indices],
                            "uncapped_candidate_examples": int(uncapped_candidate_examples),
                            "selection_record": {
                                "seed": int(seed),
                                "selection_seed": int(selection_seed),
                                "member_idx": int(member_idx),
                                "member_key": member_key,
                                "stratum": stratum,
                            },
                        }
                    )
                    work_items.append(
                        {
                            "member": member,
                            "member_idx": int(member_idx),
                            "member_key": member_key,
                            "stratum": stratum,
                            "model": model,
                            "cfg": cfg,
                            "ds": ds,
                            "checkpoint": checkpoint,
                            "seed": int(seed),
                        }
                    )
            selected_groups, selection_records, _global_selection_meta = _cap_context_index_groups(
                selection_groups,
                cap=_selection_cap_for_groups(selection_groups, selected_examples_cap),
                seed=10_000 * int(dataset_idx) + int(checkpoint_step),
                salt=f"molecule|{dataset}|{split_phase}|{checkpoint_step}",
            )
            for selected_indices, selection_meta, work_item in zip(selected_groups, selection_records, work_items):
                indices = [int(idx) for idx in selected_indices.tolist()]
                if not indices:
                    continue
                member = dict(work_item["member"])
                member_idx = int(work_item["member_idx"])
                member_key = str(work_item["member_key"])
                stratum = str(work_item["stratum"])
                model = work_item["model"]
                cfg = work_item["cfg"]
                ds = work_item["ds"]
                checkpoint = work_item["checkpoint"]
                seed = int(work_item["seed"])
                if _write_context_rows_enabled(cli_args):
                    context_embeddings = molecule_context_embeddings_for_indices(
                        model=model,
                        ds=ds,
                        example_indices=indices,
                        checkpoint_id=str(checkpoint["checkpoint_id"]),
                        dataset_key=str(dataset),
                        member_key=member_key,
                        stratum=stratum,
                        split_phase=str(split_phase),
                        device=device,
                        context_embedding_kind=str(getattr(cli_args, "context_embedding_kind", "ctx_summary")),
                    )
                else:
                    context_embeddings = {}
                for target_idx, target_nfe in enumerate(target_nfes):
                    for solver_idx, solver_key in enumerate(normalize_solver_keys(str(cli_args.solver_names))):
                        runtime_nfe = solver_macro_steps(str(solver_key), int(target_nfe))
                        existing_rows, pending_cases = _pending_scheduler_cases(
                            row_recorder,
                            benchmark_family=SCENARIO_FAMILY_MOLECULE,
                            split_phase=str(split_phase),
                            seed=int(seed),
                            dataset=str(dataset),
                            checkpoint_id=str(checkpoint["checkpoint_id"]),
                            checkpoint_step=int(checkpoint_step),
                            target_nfe=int(target_nfe),
                            solver_key=str(solver_key),
                            scheduler_cases=list(scheduler_cases_by_dataset[str(dataset)]),
                        )
                        rows.extend(existing_rows)
                        uniform_row: Optional[Mapping[str, Any]] = next(
                            (row for row in existing_rows if str(row.get("scheduler_key")) == UNIFORM_SCHEDULER_KEY),
                            None,
                        )
                        uniform_context_by_id: Dict[str, Mapping[str, Any]] = _existing_uniform_context_rows(
                            row_recorder,
                            dataset=str(dataset),
                            split_phase=str(split_phase),
                            seed=int(seed),
                            solver_key=str(solver_key),
                            target_nfe=int(target_nfe),
                            checkpoint_id=str(checkpoint["checkpoint_id"]),
                        )
                        for case in pending_cases:
                            scheduler_key = str(case["scheduler_key"])
                            details = _schedule_details_from_case(case, int(runtime_nfe))
                            eval_seed = int(seed) + 1_000_000 * dataset_idx + 100_000 * member_idx + 10_000 * target_idx + solver_idx
                            metrics = evaluate_molecule_rollout_schedule(
                                model=model,
                                ds=ds,
                                cfg=cfg,
                                scheduler_key=scheduler_key,
                                solver_key=str(solver_key),
                                target_nfe=int(target_nfe),
                                runtime_nfe=int(runtime_nfe),
                                time_grid=details["time_grid"],
                                example_indices=indices,
                                sample_count=int(getattr(cli_args, "molecule_sample_count", 1)),
                                rollout_steps=int(getattr(cli_args, "molecule_rollout_steps", 16)),
                                seed=int(eval_seed),
                                split_phase=str(split_phase),
                                checkpoint_id=str(checkpoint["checkpoint_id"]),
                                dataset_key=str(dataset),
                                member_key=member_key,
                                stratum=stratum,
                                formula=str(member.get("formula", "")),
                                source_zip_name=str(member.get("source_zip_name", "")),
                                device=device,
                            )
                            row_metrics = {
                                key: metrics.get(key)
                                for key in (
                                    *MOLECULE_PRIMARY_METRICS,
                                    "molecule_coordinate_w1_mean",
                                    "molecule_pair_distance_w1",
                                    "selection_metric_value",
                                    "num_eval_samples",
                                    "eval_windows",
                                    "realized_nfe",
                                )
                            }
                            row_metrics.update(
                                {
                                    "num_eval_samples": int(getattr(cli_args, "molecule_sample_count", 1)),
                                    "eval_windows": int(metrics.get("eval_windows", len(indices))),
                                    "eval_examples": int(metrics.get("eval_windows", len(indices))),
                                    "eval_horizon": int(getattr(cli_args, "molecule_rollout_steps", 16)),
                                    "realized_nfe": int(metrics.get("realized_nfe", _realized_nfe_for_solver(str(solver_key), int(runtime_nfe)))),
                                }
                            )
                            row = _build_row(
                                benchmark_family=SCENARIO_FAMILY_MOLECULE,
                                split_phase=str(split_phase),
                                seed=int(seed),
                                dataset=str(dataset),
                                checkpoint=checkpoint,
                                checkpoint_step=int(checkpoint_step),
                                nfe_role=nfe_role,
                                target_nfe=int(target_nfe),
                                runtime_nfe=int(runtime_nfe),
                                solver_key=str(solver_key),
                                scheduler_key=scheduler_key,
                                details=details,
                                metrics=row_metrics,
                                row_signature=str(case["row_signature"]),
                                protocol_hash=str(row_recorder["protocol_hash"]),
                            )
                            row.update(
                                {
                                    "member_key": member_key,
                                    "stratum": stratum,
                                    "formula": str(member.get("formula", "")),
                                    "source_zip_name": str(member.get("source_zip_name", "")),
                                }
                            )
                            row.update(
                                _selection_metadata_row_fields(
                                    selection_meta,
                                    cap_source=str(selected_examples_cap_source),
                                    context_sample_count=_context_sample_cap(cli_args),
                                )
                            )
                            if str(split_phase) == TRAIN_TUNING_PHASE:
                                row.update(
                                    _train_tuning_metadata(
                                        cli_args,
                                        tuning_seed=int(selection_meta["selection_seed"]),
                                        target_examples=int(selection_meta["uncapped_candidate_examples"]),
                                        uncapped_candidate_examples=int(selection_meta["uncapped_candidate_examples"]),
                                    )
                                )
                            _append_row_record(row_recorder, row)
                            if scheduler_key == UNIFORM_SCHEDULER_KEY:
                                uniform_row = row
                                uniform_context_by_id = {
                                    str(ctx["context_id"]): dict(ctx, scheduler_key=UNIFORM_SCHEDULER_KEY)
                                    for ctx in list(metrics.get("per_context_rows", []) or [])
                                }
                            if _write_context_rows_enabled(cli_args):
                                context_rows = _molecule_context_records(
                                    dataset=str(dataset),
                                    split_phase=str(split_phase),
                                    seed=int(seed),
                                    evaluation_seed=int(eval_seed),
                                    solver_key=str(solver_key),
                                    target_nfe=int(target_nfe),
                                    runtime_nfe=int(runtime_nfe),
                                    scheduler_key=scheduler_key,
                                    details=details,
                                    checkpoint=checkpoint,
                                    checkpoint_step=int(checkpoint_step),
                                    nfe_role=nfe_role,
                                    parent_row_signature=str(case["row_signature"]),
                                    protocol_hash=str(row_recorder["protocol_hash"]),
                                    per_context_metrics=list(metrics.get("per_context_rows", []) or []),
                                    uniform_by_context_id=uniform_context_by_id if scheduler_key != UNIFORM_SCHEDULER_KEY else {
                                        str(ctx["context_id"]): dict(ctx, scheduler_key=UNIFORM_SCHEDULER_KEY)
                                        for ctx in list(metrics.get("per_context_rows", []) or [])
                                    },
                                    rollout_steps=int(getattr(cli_args, "molecule_rollout_steps", 16)),
                                    train_tuning_context_metadata=(
                                        {
                                            "train_tuning_fraction": float(cli_args.eval_train_fraction),
                                            "train_tuning_seed": int(selection_meta["selection_seed"]),
                                            "train_tuning_strata": int(cli_args.train_tuning_strata),
                                            "train_tuning_sampler": train_tuning_sampler_key(TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION),
                                        }
                                        if str(split_phase) == TRAIN_TUNING_PHASE
                                        else None
                                    ),
                                )
                                _append_context_records(
                                    row_recorder,
                                    context_rows,
                                    context_embeddings=context_embeddings,
                                    metadata={
                                        "benchmark_family": SCENARIO_FAMILY_MOLECULE,
                                        "checkpoint_id": str(checkpoint["checkpoint_id"]),
                                        "checkpoint_step": int(checkpoint_step),
                                        "dataset": str(dataset),
                                        "split_phase": str(split_phase),
                                        "context_schema": MOLECULE_CONTEXT_SCHEMA,
                                        "context_embedding_kind": str(getattr(cli_args, "context_embedding_kind", "ctx_summary")),
                                        "member_key": member_key,
                                        "stratum": stratum,
                                    },
                                )
                            rows.append(row)
    return rows


def _candidate_rows_by_phase(rows: Sequence[Mapping[str, Any]], split_phase: str, solver_names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    solver_filter = None if solver_names is None else {str(x) for x in solver_names}
    out = []
    for row in rows:
        if str(row.get("split_phase")) != str(split_phase):
            continue
        if str(row.get("row_status")) != "complete":
            continue
        if solver_filter is not None and str(row.get("solver_key")) not in solver_filter:
            continue
        out.append(dict(row))
    return out


def _aggregate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("benchmark_family"),
            row.get("scenario_key", row.get("dataset")),
            row.get("nfe_role"),
            row.get("checkpoint_step"),
            row.get("target_nfe"),
            row.get("solver_key"),
            row.get("scheduler_key"),
            row.get("train_budget_label"),
        )
        groups.setdefault(key, []).append(row)
    summaries: List[Dict[str, Any]] = []
    metric_names = (
        "forecast_crps",
        "forecast_mse",
        "forecast_mase",
        "forecast_mase_scale_period",
        "score_main",
        "temporal_tstr_f1",
        "disc_auc",
        "disc_auc_gap",
        "temporal_uw1",
        "temporal_cw1",
        "u_l1",
        "c_l1",
        "spread_specific_error",
        "imbalance_specific_error",
        "ret_vol_acf_error",
        "impact_response_error",
        "molecule_kabsch_rmsd_3d",
        "molecule_ensemble_velocity_norm_w1",
        "molecule_ensemble_acceleration_norm_w1",
        "molecule_rollout_velocity_norm_w1",
        "molecule_rollout_acceleration_norm_w1",
        "molecule_coordinate_w1_mean",
        "molecule_pair_distance_w1",
        "forecast_relative_crps_gain_vs_uniform",
        "forecast_relative_mase_gain_vs_uniform",
        "relative_score_gain_vs_uniform",
        "realized_nfe",
        "latency_ms_per_sample",
    )
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        family, dataset, nfe_role, checkpoint_step, target_nfe, solver_key, scheduler_key, budget = key
        summary: Dict[str, Any] = {
            "benchmark_family": family,
            "canonical_layout_version": CANONICAL_LAYOUT_VERSION,
            "scenario_key": dataset,
            "scenario_family": family,
            "nfe_role": nfe_role,
            "checkpoint_step": int(checkpoint_step),
            "checkpoint_maturity_label": _checkpoint_maturity_label(int(checkpoint_step)),
            "checkpoint_maturity_index": _checkpoint_maturity_index(int(checkpoint_step)),
            "dataset": dataset,
            "target_nfe": int(target_nfe),
            "solver_key": solver_key,
            "scheduler_key": scheduler_key,
            "schedule_name": schedule_display_name(str(scheduler_key)),
            "schedule_family": schedule_family_for_key(str(scheduler_key)),
            "density_source_key": density_source_key_for_schedule(str(scheduler_key)),
            "train_budget_label": budget,
            "n_seeds": int(len(group)),
            "seed_values": sorted(int(row.get("seed", 0)) for row in group),
        }
        for metric in metric_names:
            vals = [_optional_float(row.get(metric)) for row in group]
            vals = [float(v) for v in vals if v is not None]
            summary[f"{metric}_mean"] = _mean(vals)
            summary[f"{metric}_std"] = _std(vals)
        scale_kinds = sorted({str(row.get("forecast_mase_scale_kind")) for row in group if row.get("forecast_mase_scale_kind") not in (None, "")})
        summary["forecast_mase_scale_kind"] = scale_kinds[0] if len(scale_kinds) == 1 else (",".join(scale_kinds) if scale_kinds else "")
        applicability_values = {
            bool(row.get("temporal_tstr_f1_applicable"))
            for row in group
            if row.get("temporal_tstr_f1_applicable") is not None
        }
        summary["temporal_tstr_f1_applicable"] = (
            next(iter(applicability_values)) if len(applicability_values) == 1 else (None if not applicability_values else True)
        )
        summaries.append(summary)
    return summaries


def _aggregate_main_table(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    seed_summaries = _aggregate_seed_rows(rows)
    augmented = augment_rows_with_relative_metrics(seed_summaries)
    return {
        "method_key": METHOD_KEY,
        "row_count": int(len(rows)),
        "summary_row_count": int(len(augmented)),
        "schedule_keys": sorted({str(row.get("scheduler_key")) for row in rows}),
        "baseline_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "experimental_fixed_schedule_keys": list(EXPERIMENTAL_FIXED_SCHEDULE_KEYS),
        "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS),
        "seed_summaries": augmented,
    }


def _prep_summary(cli_args: argparse.Namespace) -> Dict[str, Any]:
    schedules = _parse_schedule_names(str(cli_args.baseline_scheduler_names))
    summary_schedules = _parse_summary_schedule_names(str(getattr(cli_args, "summary_scheduler_names", "")))
    solvers = list(normalize_solver_keys(str(cli_args.solver_names)))
    nfes = _target_nfe_values_for_args(cli_args)
    checkpoint_steps = _checkpoint_steps_for_args(cli_args)
    manifest_path = resolve_project_path(str(cli_args.backbone_manifest)) if str(cli_args.backbone_manifest).strip() else None
    manifest_summary: Dict[str, Any] = {"path": None, "ready_count": None, "missing_count": None}
    if manifest_path is not None:
        resolved = manifest_path
        manifest_summary["path"] = _logical_artifact_path(resolved)
        if resolved.exists():
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            manifest_summary["ready_count"] = int(payload.get("ready_count", 0))
            manifest_summary["missing_count"] = int(payload.get("missing_count", 0))
    return {
        "runner_mode": "diffusion_flow_time_reparameterization",
        "runner_signature": RUNNER_SIGNATURE_VERSION,
        "canonical_layout_version": CANONICAL_LAYOUT_VERSION,
        "method_key": METHOD_KEY,
        "baseline_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "experimental_fixed_schedule_keys": list(EXPERIMENTAL_FIXED_SCHEDULE_KEYS),
        "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS),
        "scheduled_evaluation_keys": schedules,
        "summary_schedule_keys": summary_schedules,
        "solver_names": solvers,
        "nfe_role": str(getattr(cli_args, "nfe_role", NFE_ROLE_SEEN)),
        "target_nfe_values": nfes,
        "checkpoint_steps": checkpoint_steps,
        "forecast_datasets": parse_forecast_datasets(str(cli_args.forecast_datasets)),
        "conditional_generation_datasets": parse_conditional_generation_datasets(
            str(cli_args.conditional_generation_datasets)
        ),
        "molecule_datasets": list(parse_molecule_datasets(str(getattr(cli_args, "molecule_datasets", "")))),
        "split_phase": str(cli_args.split_phase),
        "backbone_manifest": manifest_summary,
        "allow_execute": bool(getattr(cli_args, "allow_execute", False)),
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run diffusion-flow time reparameterization fixed-schedule evaluations.")
    ap.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--dataset_root", type=str, default=str(project_paper_dataset_root()))
    ap.add_argument("--shared_backbone_root", type=str, default=str(DEFAULT_SHARED_BACKBONE_ROOT))
    ap.add_argument("--backbone_manifest", type=str, default=str(default_backbone_manifest_path()))
    ap.add_argument("--checkpoint_steps", type=str, default=",".join(str(x) for x in CANONICAL_CHECKPOINT_STEPS))
    ap.add_argument("--otflow_train_steps", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--forecast_datasets", type=str, default=",".join(DEFAULT_FORECAST_DATASETS))
    ap.add_argument(
        "--conditional_generation_datasets",
        type=str,
        default=",".join(DEFAULT_CONDITIONAL_GENERATION_DATASETS),
    )
    ap.add_argument("--molecule_datasets", type=str, default="")
    ap.add_argument("--molecule_group_root", type=str, default=str(default_molecule_group_root()))
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--lobster_synthetic_profile_path", type=str, default="")
    ap.add_argument("--long_term_st_path", type=str, default="")
    ap.add_argument("--solver_names", type=str, default=",".join(ALL_SOLVER_ORDER))
    ap.add_argument("--nfe_role", type=str, choices=NFE_ROLES, default=NFE_ROLE_SEEN)
    ap.add_argument("--target_nfe_values", type=str, default="")
    ap.add_argument("--baseline_scheduler_names", type=str, default=",".join(DEFAULT_SCHEDULES))
    ap.add_argument("--schedule_summary_json", type=str, default="")
    ap.add_argument("--summary_scheduler_names", type=str, default="")
    ap.add_argument("--seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--split_phase", type=str, choices=SUPPORTED_SPLIT_PHASES, default=LOCKED_TEST_PHASE)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--num_eval_samples", type=int, default=5)
    ap.add_argument("--molecule_sample_count", type=int, default=1)
    ap.add_argument("--molecule_rollout_steps", type=int, default=16)
    ap.add_argument("--molecule_stride_eval", type=int, default=1)
    ap.add_argument("--forecast_eval_batch_size", type=int, default=64)
    ap.add_argument("--write_context_rows", action="store_true", default=False)
    ap.add_argument("--context_row_csv_name", type=str, default="context_rows.csv")
    ap.add_argument("--context_embeddings_npz_name", type=str, default="context_embeddings.npz")
    ap.add_argument(
        "--train_tuning_context_sample_count",
        "--context_sample_count",
        dest="context_sample_count",
        type=int,
        default=256,
        help="Train-tuning context budget for GIPO supervision rows; validation/locked-test use eval-window options.",
    )
    ap.add_argument("--write_forecast_context_rows", action="store_true", default=False)
    ap.add_argument("--forecast_context_row_csv_name", type=str, default="")
    ap.add_argument("--forecast_context_embeddings_npz_name", type=str, default="")
    ap.add_argument("--context_embedding_kind", type=str, choices=("ctx_summary", "summary"), default="ctx_summary")
    ap.add_argument("--calibration_trace_samples", type=int, default=1)
    ap.add_argument("--eval_horizon", type=int, default=0)
    ap.add_argument("--eval_train_fraction", type=float, default=0.20)
    ap.add_argument("--train_tuning_seed", type=int, default=0)
    ap.add_argument("--train_tuning_strata", type=int, default=20)
    ap.add_argument("--train_tuning_sampling_mode", choices=TRAIN_TUNING_SAMPLING_MODES, default=TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION)
    ap.add_argument("--train_tuning_train_split_fraction", type=float, default=DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION)
    ap.add_argument("--train_tuning_val_split_fraction", type=float, default=DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION)
    ap.add_argument("--eval_windows_val", type=int, default=0)
    ap.add_argument("--eval_windows_test", type=int, default=0)
    ap.add_argument("--sigma_eps", type=float, default=1e-6)
    ap.add_argument("--hidden_dim", type=int, default=160)
    ap.add_argument("--fu_net_layers", type=int, default=3)
    ap.add_argument("--fu_net_heads", type=int, default=4)
    ap.add_argument("--rollout_mode", type=str, default="non_ar")
    ap.add_argument("--future_block_len", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--row_jsonl_name", type=str, default="rows.jsonl")
    ap.add_argument("--row_csv_name", type=str, default="rows.csv")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--diagnose_locked_forecast_only", action="store_true", default=False)
    ap.add_argument("--allow_execute", action="store_true", default=False)
    return ap


def run_diffusion_flow_time_reparameterization(cli_args: argparse.Namespace) -> Dict[str, Any]:
    out_root = resolve_project_path(str(cli_args.out_root))
    out_root.mkdir(parents=True, exist_ok=True)
    prep_payload = _prep_summary(cli_args)
    if bool(getattr(cli_args, "diagnose_locked_forecast_only", False)):
        rows = list(
            _load_rows(
                out_root / str(getattr(cli_args, "row_jsonl_name", "rows.jsonl")),
                protocol_hash=_protocol_config_fingerprint(cli_args),
            ).values()
        )
        selected_seeds = set(parse_int_csv(str(cli_args.seeds)))
        locked = [
            row
            for row in _candidate_rows_by_phase(rows, LOCKED_TEST_PHASE)
            if int(row.get("seed", -1)) in selected_seeds
        ]
        payload = {"runner_mode": "diagnose_locked_forecast_only", "row_count": int(len(rows)), "locked_row_count": int(len(locked)), "main_table_summary": _aggregate_main_table(locked)}
        save_json(dict(payload), str(out_root / "combined_summary.json"))
        return payload
    if not bool(cli_args.allow_execute):
        save_json(dict(prep_payload), str(out_root / "combined_summary.json"))
        return dict(prep_payload)

    validate_execution_preflight(cli_args)
    row_recorder = _init_row_recorder(out_root, cli_args)
    active_split_phase = str(cli_args.split_phase)
    selected_seeds = parse_int_csv(str(cli_args.seeds))
    forecast_datasets = parse_forecast_datasets(str(cli_args.forecast_datasets))
    conditional_generation_datasets = parse_conditional_generation_datasets(
        str(cli_args.conditional_generation_datasets)
    )
    molecule_datasets = parse_molecule_datasets(str(getattr(cli_args, "molecule_datasets", "")))
    summary_requested = bool(str(getattr(cli_args, "schedule_summary_json", "")).strip() or str(getattr(cli_args, "summary_scheduler_names", "")).strip())
    scheduler_cases: Dict[str, List[Dict[str, Any]]] = {}
    if forecast_datasets:
        scheduler_cases.update(
            _scheduler_cases_for_datasets(
                cli_args,
                list(forecast_datasets),
                include_summary_cases=True,
            )
        )
    if conditional_generation_datasets:
        scheduler_cases.update(
            _scheduler_cases_for_datasets(
                cli_args,
                list(conditional_generation_datasets),
                include_summary_cases=summary_requested,
            )
        )
    if molecule_datasets:
        scheduler_cases.update(
            _scheduler_cases_for_datasets(
                cli_args,
                list(molecule_datasets),
                include_summary_cases=summary_requested,
            )
        )
    try:
        if forecast_datasets:
            _run_forecast_phase(
                cli_args,
                row_recorder=row_recorder,
                split_phase=active_split_phase,
                seeds=selected_seeds,
                scheduler_cases_by_dataset={dataset: scheduler_cases[dataset] for dataset in forecast_datasets},
            )
        if conditional_generation_datasets:
            _run_conditional_generation_phase(
                cli_args,
                row_recorder=row_recorder,
                split_phase=active_split_phase,
                seeds=selected_seeds,
                scheduler_cases_by_dataset={dataset: scheduler_cases[dataset] for dataset in conditional_generation_datasets},
            )
        if molecule_datasets:
            _run_molecule_phase(
                cli_args,
                row_recorder=row_recorder,
                split_phase=active_split_phase,
                seeds=selected_seeds,
                scheduler_cases_by_dataset={dataset: scheduler_cases[dataset] for dataset in molecule_datasets},
            )
    finally:
        row_recorder["fh"].close()

    selected_seed_set = set(int(seed) for seed in selected_seeds)
    phase_rows = [
        row
        for row in _candidate_rows_by_phase(list(row_recorder["rows_by_key"].values()), active_split_phase)
        if int(row.get("seed", -1)) in selected_seed_set
    ]
    main_table_summary = _aggregate_main_table(phase_rows)
    seed_summaries = main_table_summary.pop("seed_summaries")
    seed_summary_payload = {"split_phase": active_split_phase, "seed_summaries": seed_summaries}
    seed_summary_key = f"{active_split_phase}_seed_summary"
    save_json(dict(seed_summary_payload), str(out_root / f"{active_split_phase}_seed_summary.json"))
    save_json(dict(main_table_summary), str(out_root / "main_table_summary.json"))
    schedule_selection = {
        "method_key": METHOD_KEY,
        "canonical_layout_version": CANONICAL_LAYOUT_VERSION,
        "nfe_role": str(getattr(cli_args, "nfe_role", NFE_ROLE_SEEN)),
        "target_nfe_values": _target_nfe_values_for_args(cli_args),
        "checkpoint_steps": _checkpoint_steps_for_args(cli_args),
        "baseline_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "experimental_fixed_schedule_keys": list(EXPERIMENTAL_FIXED_SCHEDULE_KEYS),
        "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS),
        "scheduled_evaluation_keys": _parse_schedule_names(str(cli_args.baseline_scheduler_names))
        + _parse_summary_schedule_names(str(getattr(cli_args, "summary_scheduler_names", ""))),
        "fixed_schedule_keys": _parse_schedule_names(str(cli_args.baseline_scheduler_names)),
        "summary_schedule_keys": _parse_summary_schedule_names(str(getattr(cli_args, "summary_scheduler_names", ""))),
    }
    save_json(dict(schedule_selection), str(out_root / "schedule_selection_summary.json"))
    combined = {"prep": dict(prep_payload), "schedule_selection_summary": dict(schedule_selection), seed_summary_key: dict(seed_summary_payload), "main_table_summary": dict(main_table_summary)}
    save_json(dict(combined), str(out_root / "combined_summary.json"))
    return combined


def main() -> None:
    run_diffusion_flow_time_reparameterization(build_argparser().parse_args())


if __name__ == "__main__":
    main()
