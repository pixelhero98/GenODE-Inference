from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from genode.checkpoint_validation import validate_strict_integer
from genode.cli import parse_int_csv
from genode.data.otflow_experiment_plan import FORECAST_FAMILY
from genode.data.otflow_monash_datasets import monash_manifest_path
from genode.experiment_layout import (
    LOCKED_TEST_PREVIEW_CONTEXTS,
    REFERENCE_SEEN_NFES,
    TRAIN_TUNING_CONTEXT_SAMPLE_COUNT,
)
from genode.solver_protocol import (
    SUPPORTED_SOLVER_KEYS,
    normalize_solver_key,
    normalize_solver_keys,
    normalize_solver_nfe_fields,
)
from genode.gipo.density_representation import (
    average_density_masses,
    density_mass_to_time_grid,
    grid_to_density_mass,
    uniform_reference_grid,
)
from genode.gipo.models import validate_time_grid
from genode.gipo.ablation_plan import GIPO_POLICY_KEY
from genode.gipo.policy import (
    context_embedding_table_manifest_path,
    load_context_embedding_table,
    save_context_embedding_table,
)
from genode.gipo.schedule_hash import schedule_grid_hash
from genode.gipo.schema import (
    cap_context_indices,
    consistent_metadata_value,
    evaluation_row_signature,
    reject_retired_evaluation_keys,
)
from genode.gipo.objectives import attach_reward_columns, rewards_by_setting, seed_mean_metric_rows
from genode.gipo.ser_ptg_reference import (
    SER_PTG_AVG_REVERSED_SCHEDULE_KEY,
    SER_PTG_REVERSED_SCHEDULE_KEY,
    SER_PTG_SCHEDULE_KEY,
    grid_geometry,
)
from genode.data.otflow_paths import (
    backbone_manifest_path,
    display_project_path,
    project_outputs_root,
    project_dataset_root,
    resolve_project_path,
)
from genode.evaluation.otflow_evaluation_support import (
    DEFAULT_SHARED_BACKBONE_ROOT,
    DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION,
    DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION,
    LOCKED_TEST_PHASE,
    TRAIN_TUNING_PHASE,
    TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION,
    TRAIN_TUNING_SAMPLING_MODES,
    VALIDATION_PHASE,
    choose_forecast_example_indices,
    choose_forecast_train_tuning_indices,
    evaluate_forecast_schedule,
    load_forecast_checkpoint_splits,
    train_tuning_sampler_key,
    train_tuning_target_example_count,
)
from genode.models.otflow_train_val import save_json
from genode.provenance import fingerprint_identity, path_fingerprint
from genode.path_safety import resolve_portable_relative_path
from genode.runtime import ProgressBar, resolve_torch_device
from genode.schedule_transfer.diffusion_flow_schedules import (
    BASELINE_SCHEDULE_KEYS,
    fixed_schedule_shape_statistics,
    schedule_display_name,
)

SELECTED_STUDENT_SCHEDULE_NAME = "GIPO"
EVALUATOR_SIGNATURE_VERSION = "schedule_summary_evaluator_seen_unseen"
SCHEDULE_CONTEXT_SELECTION_PROTOCOL = "schedule_summary_context_selection"
SER_REFERENCE_SCHEDULE_KEYS: Tuple[str, ...] = (
    SER_PTG_SCHEDULE_KEY,
    SER_PTG_REVERSED_SCHEDULE_KEY,
    SER_PTG_AVG_REVERSED_SCHEDULE_KEY,
)


def _filter_rows_to_scheduler_keys(
    rows: Sequence[Mapping[str, Any]],
    scheduler_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed = {str(key) for key in scheduler_keys}
    return [dict(row) for row in rows if str(row.get("scheduler_key", "")) in allowed]


SCHEDULE_ROW_FIELDS: Tuple[str, ...] = (
    "benchmark_family",
    "split_phase",
    "seed",
    "scenario_key",
    "method_key",
    "checkpoint_step",
    "checkpoint_id",
    "checkpoint_path",
    "backbone_name",
    "train_budget_label",
    "target_nfe",
    "macro_steps",
    "solver_key",
    "solver_name",
    "scheduler_key",
    "scheduler_name",
    "gipo_step_budget",
    "mode",
    "teacher_final_retrain",
    "row_signature",
    "selection_metric",
    "selection_metric_value",
    "reference_macro_steps",
    "runtime_grid_q25",
    "runtime_grid_q50",
    "runtime_grid_q75",
    "internal_fraction_after_098",
    "internal_count_after_098",
    "internal_count",
    "min_interval",
    "max_interval",
    "forecast_crps",
    "forecast_mse",
    "forecast_mase",
    "best_fixed_crps",
    "best_fixed_mase",
    "uniform_crps",
    "uniform_mase",
    "u_crps_best",
    "u_mase_best",
    "u_comp_best",
    "u_comp_uniform",
    "realized_nfe",
    "latency_ms_per_sample",
    "num_eval_samples",
    "eval_examples",
    "eval_windows",
    "eval_horizon",
    "evaluation_protocol_hash",
    "chosen_examples_hash",
    "example_selection_protocol",
    "context_sample_count",
    "selected_examples",
    "selected_examples_cap",
    "selected_examples_cap_source",
    "locked_test_mode",
    "locked_test_context_limit",
    "locked_test_context_limit_scope",
    "uncapped_candidate_examples",
    "candidate_examples_after_initial_selection",
    "selection_was_capped",
    "global_selection_was_capped",
    "schedule_grid_hash",
    "time_grid_json",
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
    "candidate_source",
    "active_round",
    "student_seed",
    "perturbation_type",
    "perturbation_params_json",
    "intervals_json",
    "utility",
    "validity_flags_json",
)

CONTEXT_ROW_FIELDS: Tuple[str, ...] = (
    "benchmark_family",
    "parent_row_signature",
    "protocol_hash",
    "scenario_key",
    "split_phase",
    "seed",
    "logical_seed",
    "evaluation_seed",
    "method_key",
    "solver_key",
    "target_nfe",
    "realized_nfe",
    "scheduler_key",
    "gipo_step_budget",
    "mode",
    "teacher_final_retrain",
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
    "context_embedding_kind",
    "checkpoint_step",
    "checkpoint_id",
    "forecast_crps",
    "forecast_mase",
    "forecast_mse",
    "num_eval_samples",
    "eval_horizon",
    "batch_size",
    "sample_seed_start",
    "sample_seed_values_json",
    "chosen_examples_hash",
    "evaluation_protocol_hash",
    "row_signature",
    "locked_test_mode",
    "locked_test_context_limit",
    "locked_test_context_limit_scope",
    "selected_examples_cap_source",
    "selection_was_capped",
    "global_selection_was_capped",
    "train_tuning_fraction",
    "train_tuning_seed",
    "train_tuning_strata",
    "train_tuning_sampler",
)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return float(val) if math.isfinite(val) else None


def _strict_artifact_integer(value: Any, *, label: str, minimum: int = 1) -> int:
    if isinstance(value, str):
        if value != value.strip() or not value or not value.isascii() or not value.isdecimal():
            raise ValueError(f"{label} must be an integer, got {value!r}.")
        value = int(value)
    return validate_strict_integer(value, label=label, minimum=minimum)


def _strict_sha256_digest(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return digest


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = [float(v) for v in (_optional_float(x) for x in values) if v is not None]
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _std(values: Iterable[Any]) -> Optional[float]:
    vals = [float(v) for v in (_optional_float(x) for x in values) if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return float(np.std(np.asarray(vals, dtype=np.float64), ddof=1))


def _safe_gain(value: Any, reference: Any) -> Optional[float]:
    v = _optional_float(value)
    r = _optional_float(reference)
    if v is None or r is None or abs(float(r)) <= 1e-12:
        return None
    return float(1.0 - float(v) / float(r))


def _safe_high_gain(value: Any, reference: Any) -> Optional[float]:
    v = _optional_float(value)
    r = _optional_float(reference)
    if v is None or r is None or abs(float(r)) <= 1e-12:
        return None
    return float(float(v) / float(r) - 1.0)


def _protocol_path_fingerprint(path: str | Path) -> Dict[str, Any]:
    return fingerprint_identity(path_fingerprint(resolve_project_path(str(path))))


def _context_sample_cap(args: argparse.Namespace) -> int:
    cap = int(getattr(args, "context_sample_count", TRAIN_TUNING_CONTEXT_SAMPLE_COUNT))
    if cap <= 0:
        raise ValueError(f"--context_sample_count must be positive, got {cap!r}.")
    return int(cap)


def _split_example_cap(args: argparse.Namespace, split_phase: str) -> Tuple[int | None, str]:
    context_cap = _context_sample_cap(args)
    preview_enabled = bool(getattr(args, "locked_test_preview", False))
    preview_contexts = getattr(args, "locked_test_preview_contexts", None)
    if preview_contexts is not None and not preview_enabled:
        raise ValueError("--locked_test_preview_contexts requires --locked_test_preview.")
    if preview_enabled and str(split_phase) != LOCKED_TEST_PHASE:
        raise ValueError("--locked_test_preview is only valid with --split_phase locked_test.")
    if str(split_phase) == TRAIN_TUNING_PHASE:
        return int(context_cap), "context_sample_count"
    if str(split_phase) == LOCKED_TEST_PHASE:
        if not preview_enabled:
            return None, "locked_test_full"
        context_limit = (
            int(LOCKED_TEST_PREVIEW_CONTEXTS)
            if preview_contexts is None
            else int(preview_contexts)
        )
        if context_limit <= 0:
            raise ValueError("--locked_test_preview_contexts must be positive.")
        return int(context_limit), "locked_test_preview_contexts"
    explicit = int(getattr(args, "eval_windows_val", 0))
    if explicit < 0:
        raise ValueError(f"--eval_windows_val must be nonnegative, got {explicit!r}.")
    if explicit > 0:
        return int(explicit), "eval_windows_val"
    return int(context_cap), "context_sample_count"


def _selection_provenance(
    args: argparse.Namespace,
    *,
    split_phase: str,
    selection_was_capped: bool,
) -> Dict[str, Any]:
    if str(split_phase) != LOCKED_TEST_PHASE:
        return {}
    context_limit, source = _split_example_cap(args, split_phase)
    preview = context_limit is not None
    return {
        "locked_test_mode": "preview" if preview else "full",
        "locked_test_context_limit": int(context_limit) if preview else None,
        "locked_test_context_limit_scope": "per_seed" if preview else "none",
        "selected_examples_cap_source": str(source),
        "selection_was_capped": bool(selection_was_capped),
        "global_selection_was_capped": bool(selection_was_capped),
    }


def _protocol_hash(args: argparse.Namespace) -> str:
    selected_context_limit, selected_context_source = _split_example_cap(
        args,
        str(args.split_phase),
    )
    is_locked_test = str(args.split_phase) == LOCKED_TEST_PHASE
    locked_test_context_limit = selected_context_limit if is_locked_test else None
    locked_test_context_source = selected_context_source if is_locked_test else "none"
    payload = {
        "signature": EVALUATOR_SIGNATURE_VERSION,
        "scenario_key": str(args.scenario_key),
        "split_phase": str(args.split_phase),
        "seeds": parse_int_csv(args.seeds),
        "solver_names": list(normalize_solver_keys(str(args.solver_names))),
        "target_nfe_values": parse_int_csv(args.target_nfe_values),
        "num_eval_samples": int(args.num_eval_samples),
        "forecast_eval_batch_size": int(args.forecast_eval_batch_size),
        "write_context_rows": bool(getattr(args, "write_context_rows", False)),
        "context_embedding_kind": str(getattr(args, "context_embedding_kind", "ctx_summary")),
        "context_sample_count": _context_sample_cap(args),
        "example_selection_protocol": SCHEDULE_CONTEXT_SELECTION_PROTOCOL,
        "eval_train_fraction": float(getattr(args, "eval_train_fraction", 0.20)),
        "train_tuning_seed": int(getattr(args, "train_tuning_seed", 0)),
        "train_tuning_strata": int(getattr(args, "train_tuning_strata", 20)),
        "train_tuning_sampling_mode": str(getattr(args, "train_tuning_sampling_mode", TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION)),
        "train_tuning_sampler": train_tuning_sampler_key(str(getattr(args, "train_tuning_sampling_mode", TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION))),
        "train_tuning_train_split_fraction": float(getattr(args, "train_tuning_train_split_fraction", DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION)),
        "train_tuning_val_split_fraction": float(getattr(args, "train_tuning_val_split_fraction", DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION)),
        "eval_windows_val": int(args.eval_windows_val),
        "locked_test_mode": (
            "preview" if locked_test_context_limit is not None else "full"
        )
        if is_locked_test
        else "not_applicable",
        "locked_test_context_limit": locked_test_context_limit,
        "locked_test_context_limit_source": str(locked_test_context_source),
        "checkpoint_step": int(args.checkpoint_step),
        "scenario_manifest": _protocol_path_fingerprint(
            monash_manifest_path(
                resolve_project_path(str(args.dataset_root)),
                str(args.scenario_key),
            )
        ),
        "backbone_manifest": _protocol_path_fingerprint(str(args.backbone_manifest)) if str(args.backbone_manifest).strip() else None,
        "schedule_summary": _protocol_path_fingerprint(str(args.schedule_summary)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def schedule_display_name_for_key(schedule_key: str) -> str:
    key = str(schedule_key)
    if key == SER_PTG_SCHEDULE_KEY:
        return "SER-PTG local defect eta=0.05"
    if key == SER_PTG_REVERSED_SCHEDULE_KEY:
        return "SER-PTG local defect eta=0.05 reversed"
    if key == SER_PTG_AVG_REVERSED_SCHEDULE_KEY:
        return "SER-PTG local defect eta=0.05 density average"
    if key == GIPO_POLICY_KEY:
        return SELECTED_STUDENT_SCHEDULE_NAME
    return schedule_display_name(key)


def _derived_ser_time_grid(schedule_key: str, base_grid: Sequence[float], *, macro_steps: int) -> Tuple[float, ...]:
    base = validate_time_grid(base_grid, macro_steps=int(macro_steps))
    if str(schedule_key) == SER_PTG_REVERSED_SCHEDULE_KEY:
        return validate_time_grid([1.0 - float(value) for value in reversed(base)], macro_steps=int(macro_steps))
    if str(schedule_key) == SER_PTG_AVG_REVERSED_SCHEDULE_KEY:
        reference = uniform_reference_grid()
        reversed_grid = validate_time_grid([1.0 - float(value) for value in reversed(base)], macro_steps=int(macro_steps))
        base_mass = grid_to_density_mass(base, reference_time_grid=reference, macro_steps=int(macro_steps))
        reversed_mass = grid_to_density_mass(reversed_grid, reference_time_grid=reference, macro_steps=int(macro_steps))
        averaged_mass = average_density_masses(base_mass, reversed_mass)
        return density_mass_to_time_grid(averaged_mass, reference_time_grid=reference, macro_steps=int(macro_steps))
    return base


def _register_prediction(
    predictions: Dict[Tuple[str, str, int], Dict[str, Any]],
    *,
    scheduler_key: str,
    schedule_name: str,
    budget: Any,
    item: Mapping[str, Any],
    time_grid: Sequence[float],
    solver_key: str,
    target_nfe: int,
    macro_steps: int,
    realized_nfe: int,
) -> None:
    key = (str(scheduler_key), str(solver_key), int(target_nfe))
    prediction = dict(item)
    intervals = [float(x) for x in np.diff(np.asarray(time_grid, dtype=np.float64)).tolist()]
    normalized_budget = (
        None
        if budget in (None, "")
        else _strict_artifact_integer(
            budget,
            label="Schedule prediction gipo_step_budget",
            minimum=1,
        )
    )
    prediction.update(
        {
            "scheduler_key": str(scheduler_key),
            "schedule_name": str(schedule_name),
            "gipo_step_budget": normalized_budget,
            "solver_key": str(solver_key),
            "target_nfe": int(target_nfe),
            "macro_steps": int(macro_steps),
            "realized_nfe": int(realized_nfe),
            "time_grid": list(time_grid),
            "schedule_grid_hash": schedule_grid_hash(time_grid),
            "intervals_json": prediction.get("intervals_json", json.dumps(intervals, separators=(",", ":"))),
        }
    )
    if key in predictions:
        existing = predictions[key]
        comparable_fields = (
            "gipo_step_budget",
            "method_key",
            "mode",
            "teacher_final_retrain",
            "checkpoint_step",
            "checkpoint_id",
            "checkpoint_ids",
            "macro_steps",
            "realized_nfe",
            "time_grid",
        )
        conflicts = [field for field in comparable_fields if existing.get(field) != prediction.get(field)]
        if conflicts:
            raise ValueError(f"Conflicting duplicate schedule prediction for {key}: fields={conflicts}.")
        return
    predictions[key] = prediction


def load_schedule_predictions(
    schedule_summary_path: str | Path,
    *,
    scenario_key: str,
    solver_names: Sequence[str] = SUPPORTED_SOLVER_KEYS,
    target_nfe_values: Sequence[int] = REFERENCE_SEEN_NFES,
    require_complete: bool = True,
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    path = resolve_project_path(str(schedule_summary_path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Schedule summary {path} must contain a mapping payload.")
    reject_retired_evaluation_keys(payload, source=f"Schedule summary {path}")
    payload_scenario = str(payload.get("scenario_key", "")).strip()
    if not payload_scenario:
        raise ValueError(f"Schedule summary {path} requires scenario_key.")
    if payload_scenario != str(scenario_key):
        raise ValueError(
            f"Schedule summary scenario_key={payload_scenario!r} does not match requested scenario_key={scenario_key!r}."
        )
    allowed_solvers = {str(name) for name in solver_names}
    allowed_nfes = {int(value) for value in target_nfe_values}
    predictions: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    expected_scheduler_keys: List[str] = []
    schedules = payload.get("schedules")
    if schedules:
        schedule_items = list(schedules)
    else:
        schedule_items = [
            {
                "scheduler_key": str(payload.get("scheduler_key", "")),
                "schedule_name": str(payload.get("schedule_name", SELECTED_STUDENT_SCHEDULE_NAME)),
                "gipo_step_budget": payload.get("gipo_step_budget"),
                "predictions": list(payload.get("predictions", []) or []),
            }
        ]
    for schedule_index, schedule in enumerate(schedule_items):
        if not isinstance(schedule, Mapping):
            raise ValueError(f"Schedule summary {path} schedule {schedule_index} must be a mapping.")
        reject_retired_evaluation_keys(schedule, source=f"Schedule summary {path} schedule {schedule_index}")
        scheduler_key = str(schedule.get("scheduler_key", "")).strip()
        if not scheduler_key:
            raise ValueError("Schedule summary contains a schedule without scheduler_key.")
        expected_scheduler_keys.append(scheduler_key)
        schedule_name = str(schedule.get("schedule_name") or schedule_display_name_for_key(scheduler_key))
        for item_index, item in enumerate(list(schedule.get("predictions", []) or [])):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"Schedule summary {path} schedule {schedule_index} prediction {item_index} must be a mapping."
                )
            reject_retired_evaluation_keys(
                item,
                source=f"Schedule summary {path} schedule {schedule_index} prediction {item_index}",
            )
            metadata_sources = (item, schedule, payload)
            budget = consistent_metadata_value(
                metadata_sources,
                "gipo_step_budget",
                source=f"Schedule summary {path} schedule {schedule_index} prediction {item_index}",
            )
            prediction = dict(item)
            for meta_key in (
                "method_key",
                "mode",
                "teacher_final_retrain",
                "checkpoint_step",
                "checkpoint_id",
                "checkpoint_ids",
            ):
                value = consistent_metadata_value(
                    metadata_sources,
                    meta_key,
                    source=f"Schedule summary {path} schedule {schedule_index} prediction {item_index}",
                )
                if value is not None:
                    prediction[meta_key] = value
            solver_key = normalize_solver_key(str(item.get("solver_key")))
            nfe = normalize_solver_nfe_fields(
                solver_key,
                item.get("target_nfe"),
                macro_steps=item.get("macro_steps"),
                realized_nfe=item.get("realized_nfe"),
                source=f"Schedule {scheduler_key} prediction {item_index}",
            )
            target_nfe = int(nfe.target_nfe)
            if solver_key not in allowed_solvers or target_nfe not in allowed_nfes:
                continue
            time_grid = validate_time_grid(item.get("time_grid", []), macro_steps=nfe.macro_steps)
            prediction["solver_key"] = solver_key
            for meta_key in (
                "method_key",
                "mode",
                "teacher_final_retrain",
                "candidate_source",
                "active_round",
                "student_seed",
                "perturbation_type",
                "perturbation_params_json",
                "utility",
                "validity_flags_json",
            ):
                if meta_key not in prediction and meta_key in schedule:
                    prediction[meta_key] = schedule.get(meta_key)
                if meta_key not in prediction and meta_key in payload:
                    prediction[meta_key] = payload.get(meta_key)
            _register_prediction(
                predictions,
                scheduler_key=scheduler_key,
                schedule_name=schedule_name,
                budget=budget,
                item=prediction,
                time_grid=time_grid,
                solver_key=solver_key,
                target_nfe=int(target_nfe),
                macro_steps=int(nfe.macro_steps),
                realized_nfe=int(nfe.realized_nfe),
            )
            if scheduler_key == SER_PTG_SCHEDULE_KEY:
                for derived_key in (SER_PTG_REVERSED_SCHEDULE_KEY, SER_PTG_AVG_REVERSED_SCHEDULE_KEY):
                    derived_grid = _derived_ser_time_grid(derived_key, time_grid, macro_steps=int(nfe.macro_steps))
                    _register_prediction(
                        predictions,
                        scheduler_key=derived_key,
                        schedule_name=schedule_display_name_for_key(derived_key),
                        budget=budget,
                        item={**prediction, "schedule_derivation": "ser_reference_density_transform"},
                        time_grid=derived_grid,
                        solver_key=solver_key,
                        target_nfe=int(target_nfe),
                        macro_steps=int(nfe.macro_steps),
                        realized_nfe=int(nfe.realized_nfe),
                    )
    if require_complete:
        scheduler_keys = sorted(set(expected_scheduler_keys))
        expected = {
            (scheduler_key, str(solver), int(nfe))
            for scheduler_key in scheduler_keys
            for solver in solver_names
            for nfe in target_nfe_values
        }
        missing = sorted(expected - set(predictions), key=lambda item: (item[0], item[1], item[2]))
        if missing:
            raise ValueError(f"Schedule summary is missing predictions for: {missing[:12]}")
    return predictions


def _row_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("protocol_hash"),
        row.get("split_phase"),
        int(row.get("seed", -1)),
        row.get("scenario_key"),
        int(row.get("target_nfe", -1)),
        row.get("solver_key"),
        row.get("scheduler_key"),
    )


def _load_existing_rows(jsonl_path: Path, *, protocol_hash: str) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open("r", encoding="utf-8") as fh:
        line_number = 0
        line = fh.readline()
        while line:
            line_number += 1
            next_line = fh.readline()
            stripped = line.strip()
            if not stripped:
                line = next_line
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if not next_line and not line.endswith(("\n", "\r")):
                    break
                raise ValueError(
                    f"Invalid JSONL record in {jsonl_path} at line {line_number}."
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL record in {jsonl_path} at line {line_number} must be an object."
                )
            reject_retired_evaluation_keys(row, source=f"Evaluation row in {jsonl_path}")
            if str(row.get("protocol_hash")) != str(protocol_hash):
                raise ValueError(
                    "Existing schedule-evaluation rows use a different protocol; "
                    "choose a new output directory instead of replacing prior results."
                )
            if str(row.get("row_status")) != "complete":
                line = next_line
                continue
            rows[_row_key(row)] = row
            line = next_line
    return rows


def _output_path(
    out_dir: Path,
    value: object,
    *,
    fallback: str,
    suffix: str,
    label: str,
) -> Path:
    name = str(value or fallback)
    path = resolve_portable_relative_path(
        out_dir,
        name,
        label=label,
        reject_links=True,
    )
    if path.suffix.lower() != suffix:
        raise ValueError(f"{label} must end in {suffix!r}.")
    return path


def _validate_distinct_artifact_paths(
    paths: Mapping[str, Path],
    *,
    input_paths: Optional[Mapping[str, Path]] = None,
) -> None:
    names_by_path: Dict[Path, List[str]] = {}
    for name, path in paths.items():
        names_by_path.setdefault(path.resolve(strict=False), []).append(str(name))
    collisions = {
        str(path): names
        for path, names in names_by_path.items()
        if len(names) > 1
    }
    input_names_by_path: Dict[Path, List[str]] = {}
    for name, path in (input_paths or {}).items():
        input_names_by_path.setdefault(path.resolve(strict=False), []).append(str(name))
    input_output_collisions = {
        str(path): {
            "outputs": names_by_path[path],
            "inputs": input_names,
        }
        for path, input_names in input_names_by_path.items()
        if path in names_by_path
    }
    if collisions or input_output_collisions:
        raise ValueError(
            "Schedule-evaluation artifact paths must be pairwise distinct, including "
            "implicit sidecars, and inputs may not collide with outputs: "
            f"output_collisions={collisions}, "
            f"input/output_collisions={input_output_collisions}."
        )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SCHEDULE_ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SCHEDULE_ROW_FIELDS})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            for row in rows:
                fh.write(json.dumps(dict(row), sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _write_context_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CONTEXT_ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CONTEXT_ROW_FIELDS})


def _load_context_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            reject_retired_evaluation_keys(row, source=f"Context evaluation row in {path}")
            signature = str(row.get("row_signature", "")).strip()
            if signature:
                if signature in rows:
                    raise ValueError(f"Duplicate context row signature in {path}: {signature}")
                rows[signature] = dict(row)
    return rows


def _merge_context_embeddings_checked(
    existing: Dict[str, Sequence[float]],
    extra: Mapping[str, Sequence[float]],
) -> None:
    for key, value in extra.items():
        key_text = str(key)
        new_vec = np.asarray(value, dtype=np.float32)
        if key_text in existing:
            old_vec = np.asarray(existing[key_text], dtype=np.float32)
            if old_vec.shape != new_vec.shape or not np.allclose(old_vec, new_vec, rtol=1e-5, atol=1e-6):
                raise ValueError(f"Context embedding collision for {key_text!r} with different vector/protocol.")
            continue
        existing[key_text] = new_vec.astype(float).tolist()


def _identity_field_text(value: Any) -> str:
    """Normalize persisted identity values without discarding numeric zero."""
    return "" if value is None or value == "" else str(value)


def _row_has_complete_context_artifacts(
    row: Mapping[str, Any],
    *,
    context_rows_by_signature: Mapping[str, Mapping[str, Any]],
    context_embeddings: Mapping[str, Any],
    context_embedding_kind: str,
) -> bool:
    if str(row.get("row_status", "")) != "complete":
        return False
    parent_signature = str(row.get("row_signature", "") or "").strip()
    if not parent_signature:
        return False
    selected_examples = row.get("selected_examples")
    if isinstance(selected_examples, bool) or not isinstance(selected_examples, int):
        return False
    if selected_examples <= 0:
        return False
    parent_identity = {
        field: _identity_field_text(row.get(field, ""))
        for field in (
            "benchmark_family",
            "protocol_hash",
            "scenario_key",
            "split_phase",
            "seed",
            "target_nfe",
            "scheduler_key",
            "checkpoint_step",
            "checkpoint_id",
        )
    }
    rows_for_parent = [
        context_row
        for context_row in context_rows_by_signature.values()
        if str(context_row.get("parent_row_signature", "") or "").strip()
        == parent_signature
    ]
    if len(rows_for_parent) != selected_examples:
        return False
    context_ids: set[str] = set()
    embedding_ids: set[str] = set()
    for context_row in rows_for_parent:
        if any(
            _identity_field_text(context_row.get(field, "")) != expected
            for field, expected in parent_identity.items()
        ):
            return False
        try:
            context_solver_key = normalize_solver_key(
                str(context_row.get("solver_key", "") or "")
            )
        except ValueError:
            return False
        if context_solver_key != str(row.get("solver_key", "") or ""):
            return False
        if str(context_row.get("context_embedding_kind", "") or "") != str(
            context_embedding_kind
        ):
            return False
        context_id = str(context_row.get("context_id", "") or "").strip()
        embedding_id = str(
            context_row.get("context_embedding_id", "") or ""
        ).strip()
        if (
            not context_id
            or context_id in context_ids
            or not embedding_id
            or embedding_id in embedding_ids
            or embedding_id not in context_embeddings
        ):
            return False
        context_ids.add(context_id)
        embedding_ids.add(embedding_id)
    return True


def _schedule_row(
    *,
    seed: int,
    scenario_key: str,
    split_phase: str,
    checkpoint: Mapping[str, Any],
    prediction: Mapping[str, Any],
    metrics: Mapping[str, Any],
    protocol_hash: str,
) -> Dict[str, Any]:
    solver_key = str(prediction["solver_key"])
    target_nfe = int(prediction["target_nfe"])
    scheduler_key = str(prediction["scheduler_key"])
    time_grid = [float(x) for x in prediction["time_grid"]]
    shape = fixed_schedule_shape_statistics(time_grid)
    geom = grid_geometry(time_grid)
    nfe = normalize_solver_nfe_fields(
        solver_key,
        target_nfe,
        macro_steps=prediction.get("macro_steps"),
        realized_nfe=metrics.get("realized_nfe", prediction.get("realized_nfe")),
        source=f"{scheduler_key} row {solver_key}/{target_nfe}",
    )
    return {
        "benchmark_family": FORECAST_FAMILY,
        "split_phase": str(split_phase),
        "seed": int(seed),
        "scenario_key": str(scenario_key),
        "method_key": str(prediction.get("method_key") or scheduler_key),
        "checkpoint_step": int(checkpoint["checkpoint_step"]),
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "checkpoint_path": display_project_path(str(checkpoint["checkpoint_path"])),
        "backbone_name": str(checkpoint.get("backbone_name", "otflow")),
        "train_budget_label": str(checkpoint["train_budget_label"]),
        "target_nfe": int(target_nfe),
        "macro_steps": int(nfe.macro_steps),
        "solver_key": solver_key,
        "solver_name": normalize_solver_key(solver_key),
        "scheduler_key": scheduler_key,
        "scheduler_name": str(prediction.get("schedule_name") or schedule_display_name_for_key(scheduler_key)),
        "gipo_step_budget": prediction.get("gipo_step_budget"),
        "mode": prediction.get("mode", ""),
        "teacher_final_retrain": json.dumps(
            prediction.get("teacher_final_retrain", {}),
            sort_keys=True,
            separators=(",", ":"),
        )
        if isinstance(prediction.get("teacher_final_retrain"), Mapping)
        else prediction.get("teacher_final_retrain", ""),
        "row_signature": evaluation_row_signature(
            scenario_key=str(scenario_key),
            split_phase=str(split_phase),
            seed=int(seed),
            target_nfe=int(target_nfe),
            solver_key=solver_key,
            scheduler_key=scheduler_key,
            checkpoint_id=str(checkpoint["checkpoint_id"]),
        ),
        "selection_metric": "forecast_crps",
        "selection_metric_value": metrics.get("forecast_crps"),
        "reference_macro_steps": int(nfe.macro_steps),
        "runtime_grid_q25": shape.get("runtime_grid_q25"),
        "runtime_grid_q50": shape.get("runtime_grid_q50"),
        "runtime_grid_q75": shape.get("runtime_grid_q75"),
        "internal_fraction_after_098": geom.get("internal_fraction_after_098"),
        "internal_count_after_098": geom.get("internal_count_after_098"),
        "internal_count": geom.get("internal_count"),
        "min_interval": geom.get("min_interval"),
        "max_interval": geom.get("max_interval"),
        "forecast_crps": metrics.get("forecast_crps"),
        "forecast_mse": metrics.get("forecast_mse"),
        "forecast_mase": metrics.get("forecast_mase"),
        "forecast_mase_scale_kind": metrics.get("forecast_mase_scale_kind"),
        "forecast_mase_scale_period": metrics.get("forecast_mase_scale_period"),
        "realized_nfe": int(nfe.realized_nfe),
        "latency_ms_per_sample": metrics.get("latency_ms_per_sample"),
        "num_eval_samples": metrics.get("num_eval_samples"),
        "eval_examples": metrics.get("eval_examples"),
        "eval_windows": metrics.get("eval_examples"),
        "eval_horizon": metrics.get("eval_horizon"),
        "evaluation_protocol_hash": metrics.get("evaluation_protocol_hash"),
        "chosen_examples_hash": metrics.get("chosen_examples_hash"),
        "schedule_grid_hash": prediction.get("schedule_grid_hash"),
        "time_grid_json": json.dumps(time_grid, separators=(",", ":")),
        "protocol_hash": str(protocol_hash),
        "row_status": "complete",
        "train_tuning_fraction": "",
        "train_tuning_seed": "",
        "train_tuning_strata": "",
        "train_tuning_sampler": "",
        "candidate_source": prediction.get("candidate_source", ""),
        "active_round": prediction.get("active_round", ""),
        "student_seed": prediction.get("student_seed", ""),
        "perturbation_type": prediction.get("perturbation_type", ""),
        "perturbation_params_json": prediction.get("perturbation_params_json", ""),
        "intervals_json": prediction.get(
            "intervals_json",
            json.dumps([float(x) for x in np.diff(np.asarray(time_grid, dtype=np.float64)).tolist()], separators=(",", ":")),
        ),
        "utility": prediction.get("utility", ""),
        "validity_flags_json": prediction.get("validity_flags_json", ""),
    }


def _load_forecast_rows_csv(
    path: str | Path,
    *,
    scenario_key: str,
    split_phase: Optional[str],
    seeds: Sequence[int],
    solver_names: Sequence[str],
    target_nfe_values: Sequence[int],
    checkpoint_step: int | None = None,
    checkpoint_id: str = "",
) -> List[Dict[str, Any]]:
    resolved = resolve_project_path(str(path))
    seed_set = {int(seed) for seed in seeds}
    solver_set = {str(solver) for solver in solver_names}
    nfe_set = {int(nfe) for nfe in target_nfe_values}
    rows: List[Dict[str, Any]] = []
    mismatched_identities: set[Tuple[str, str]] = set()
    if not resolved.exists():
        return rows
    with resolved.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            reject_retired_evaluation_keys(row, source=f"Evaluation row in {resolved}")
            if str(row.get("benchmark_family", FORECAST_FAMILY)) != FORECAST_FAMILY:
                continue
            if split_phase is not None and str(row.get("split_phase")) != str(split_phase):
                continue
            if str(row.get("scenario_key")) != str(scenario_key):
                continue
            try:
                seed = int(row.get("seed", -1))
                target_nfe = int(row.get("target_nfe", -1))
            except (TypeError, ValueError):
                continue
            try:
                solver_key = normalize_solver_key(str(row.get("solver_key")))
            except ValueError:
                continue
            if seed not in seed_set or target_nfe not in nfe_set or solver_key not in solver_set:
                continue
            nfe = normalize_solver_nfe_fields(
                solver_key,
                target_nfe,
                macro_steps=row.get("macro_steps"),
                realized_nfe=row.get("realized_nfe"),
                source=f"Evaluation row in {resolved}",
            )
            if checkpoint_step is not None or str(checkpoint_id).strip():
                raw_step = row.get("checkpoint_step")
                raw_id = str(row.get("checkpoint_id", "") or "").strip()
                if raw_step in (None, "") or not raw_id:
                    raise ValueError(
                        f"Evaluation rows in {resolved} require checkpoint_step and checkpoint_id."
                    )
                if (
                    checkpoint_step is not None
                    and int(raw_step) != int(checkpoint_step)
                ) or (str(checkpoint_id).strip() and raw_id != str(checkpoint_id)):
                    mismatched_identities.add((str(raw_step), raw_id))
                    continue
            crps = _optional_float(row.get("forecast_crps"))
            mase = _optional_float(row.get("forecast_mase"))
            if crps is None or mase is None:
                continue
            clean = dict(row)
            clean["solver_key"] = solver_key
            clean["seed"] = int(seed)
            clean["target_nfe"] = int(target_nfe)
            clean["macro_steps"] = int(nfe.macro_steps)
            clean["realized_nfe"] = int(nfe.realized_nfe)
            clean["forecast_crps"] = float(crps)
            clean["forecast_mase"] = float(mase)
            for key in ("num_eval_samples", "eval_examples", "eval_horizon"):
                if clean.get(key) not in (None, ""):
                    clean[key] = _strict_artifact_integer(
                        clean[key],
                        label=f"Evaluation row {key}",
                    )
            if clean.get("gipo_step_budget") not in (None, ""):
                clean["gipo_step_budget"] = _strict_artifact_integer(
                    clean["gipo_step_budget"],
                    label="Evaluation row gipo_step_budget",
                    minimum=1,
                )
            for key in ("forecast_mse", "latency_ms_per_sample"):
                value = _optional_float(clean.get(key))
                if value is not None:
                    clean[key] = float(value)
            rows.append(clean)
    if mismatched_identities:
        raise ValueError(
            "Evaluation rows do not match the loaded backbone artifact: "
            f"found={sorted(mismatched_identities)}, "
            f"expected=({checkpoint_step!r}, {str(checkpoint_id)!r})."
        )
    return rows


def _validate_schedule_checkpoint_identity(
    predictions: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    *,
    checkpoint_step: int,
    checkpoint_id: str,
) -> None:
    declared_steps: set[int] = set()
    for prediction in predictions.values():
        if prediction.get("checkpoint_step") in (None, ""):
            raise ValueError("Every schedule prediction requires checkpoint_step.")
        declared_steps.add(
            validate_strict_integer(
                prediction["checkpoint_step"],
                label="Schedule summary checkpoint_step",
                minimum=0,
            )
        )
    if declared_steps != {int(checkpoint_step)}:
        raise ValueError(
            "Schedule summary checkpoint_step does not match the loaded backbone artifact: "
            f"declared={sorted(declared_steps)}, loaded={int(checkpoint_step)}."
        )
    declared_ids: set[str] = set()
    for prediction in predictions.values():
        singular = str(prediction.get("checkpoint_id", "") or "").strip()
        if singular:
            declared_ids.add(singular)
        plural = prediction.get("checkpoint_ids", []) or []
        if isinstance(plural, (str, bytes)):
            plural = [plural]
        declared_ids.update(str(value).strip() for value in plural if str(value).strip())
    if not declared_ids:
        raise ValueError("Every claim-bearing schedule summary requires checkpoint identity.")
    if declared_ids != {str(checkpoint_id)}:
        raise ValueError(
            "Schedule summary checkpoint identity does not match the loaded backbone artifact: "
            f"declared={sorted(declared_ids)}, loaded={str(checkpoint_id)!r}."
        )


def _missing_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    solver_names: Sequence[str],
    target_nfe_values: Sequence[int],
    scheduler_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    observed = {
        (int(row.get("seed", -1)), str(row.get("solver_key")), int(row.get("target_nfe", -1)), str(row.get("scheduler_key")))
        for row in rows
    }
    missing: List[Dict[str, Any]] = []
    for seed in seeds:
        for solver in solver_names:
            for target_nfe in target_nfe_values:
                for schedule in scheduler_keys:
                    key = (int(seed), str(solver), int(target_nfe), str(schedule))
                    if key not in observed:
                        missing.append({"seed": int(seed), "solver_key": str(solver), "target_nfe": int(target_nfe), "scheduler_key": str(schedule)})
    return missing


def _aggregate_schedule_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("solver_key")), int(row.get("target_nfe", -1)), str(row.get("scheduler_key")))
        groups.setdefault(key, []).append(row)
    summaries: List[Dict[str, Any]] = []
    for (solver_key, target_nfe, scheduler_key), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        summary: Dict[str, Any] = {
            "solver_key": solver_key,
            "target_nfe": int(target_nfe),
            "scheduler_key": scheduler_key,
            "schedule_name": schedule_display_name_for_key(scheduler_key),
            "n_seeds": int(len(group)),
            "seed_values": sorted(int(row.get("seed", 0)) for row in group),
        }
        budgets = {int(row["gipo_step_budget"]) for row in group if row.get("gipo_step_budget") not in (None, "")}
        if len(budgets) == 1:
            summary["gipo_step_budget"] = int(next(iter(budgets)))
        for metric in (
            "forecast_crps",
            "forecast_mase",
            "forecast_mse",
            "score_main",
            "temporal_cw1",
            "temporal_uw1",
            "temporal_tstr_f1",
            "molecule_kabsch_rmsd_3d",
            "molecule_ensemble_velocity_norm_w1",
            "molecule_ensemble_acceleration_norm_w1",
            "molecule_rollout_velocity_norm_w1",
            "molecule_rollout_acceleration_norm_w1",
            "u_comp_uniform",
            "latency_ms_per_sample",
            "realized_nfe",
        ):
            values = [row.get(metric) for row in group]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_std"] = _std(values)
        for metric in ("internal_fraction_after_098", "internal_count_after_098", "internal_count", "min_interval", "max_interval"):
            values = [row.get(metric) for row in group]
            summary[f"{metric}_mean"] = _mean(values)
        summaries.append(summary)
    return summaries


def _finite_metric(row: Mapping[str, Any], metric: str) -> float:
    value = _optional_float(row.get(f"{metric}_mean"))
    return float("inf") if value is None else float(value)


def _finite_metric_high(row: Mapping[str, Any], metric: str) -> float:
    value = _optional_float(row.get(f"{metric}_mean"))
    return float("-inf") if value is None else float(value)


def _metric_higher_is_better(metric: str) -> bool:
    return str(metric) in {"u_comp_uniform", "temporal_tstr_f1"}


def _selection_rewards(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]] = (),
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], Dict[str, float]], str, List[str]]:
    _validate_paired_selection_panel(
        candidate_rows=candidate_rows,
        reference_rows=reference_rows,
    )
    aggregated_candidates = seed_mean_metric_rows(candidate_rows)
    aggregated_references = seed_mean_metric_rows(reference_rows)
    if not aggregated_references:
        raise ValueError("Validation selection requires fixed baseline reference rows for paired best-fixed CRPS/MASE utility.")
    annotated = attach_reward_columns([*aggregated_references, *aggregated_candidates], fixed_scheduler_keys=BASELINE_SCHEDULE_KEYS)
    candidate_keys = {(str(row["solver_key"]), int(row["target_nfe"]), str(row["scheduler_key"])) for row in aggregated_candidates}
    annotated_candidates = [
        row
        for row in annotated
        if (str(row["solver_key"]), int(row["target_nfe"]), str(row["scheduler_key"])) in candidate_keys
    ]
    rewards = rewards_by_setting([*aggregated_references, *aggregated_candidates], fixed_scheduler_keys=BASELINE_SCHEDULE_KEYS)
    return annotated_candidates, rewards, "best_fixed_baseline_crps_mase", list(BASELINE_SCHEDULE_KEYS)


def _selection_cell(row: Mapping[str, Any]) -> Tuple[int, str, int]:
    return (
        int(row.get("seed", -1)),
        normalize_solver_key(str(row.get("solver_key", ""))),
        int(row.get("target_nfe", -1)),
    )


def _selection_panel_identity(row: Mapping[str, Any]) -> Tuple[str, int, int, int]:
    chosen_examples_hash = _strict_sha256_digest(
        row.get("chosen_examples_hash"),
        label="Validation selection row chosen_examples_hash",
    )
    # The evaluation-protocol digest intentionally includes the schedule grid and
    # scheduler key, so it cannot be equal across schedules.  Its digest integrity
    # is still required before the schedule-independent panel fields are paired.
    _strict_sha256_digest(
        row.get("evaluation_protocol_hash"),
        label="Validation selection row evaluation_protocol_hash",
    )
    values: List[int] = []
    for field in ("num_eval_samples", "eval_examples", "eval_horizon"):
        values.append(
            _strict_artifact_integer(
                row.get(field),
                label=f"Validation selection row {field}",
                minimum=1,
            )
        )
    return chosen_examples_hash, values[0], values[1], values[2]


def _validate_paired_selection_panel(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require every selected schedule and fixed baseline on one chosen-example/evaluation panel."""

    candidate_keys = sorted(
        {
            str(row.get("scheduler_key", "") or "")
            for row in candidate_rows
            if str(row.get("scheduler_key", "") or "")
        }
    )
    reference_keys = sorted(
        {
            str(row.get("scheduler_key", "") or "")
            for row in reference_rows
            if str(row.get("scheduler_key", "") or "") in BASELINE_SCHEDULE_KEYS
        }
    )
    if not candidate_keys or not reference_keys:
        return
    missing_reference_keys = sorted(set(BASELINE_SCHEDULE_KEYS) - set(reference_keys))
    if missing_reference_keys:
        raise ValueError(
            "Validation selection requires every declared fixed baseline; missing "
            f"scheduler keys: {missing_reference_keys}."
        )

    rows_by_schedule: Dict[str, Dict[Tuple[int, str, int], Mapping[str, Any]]] = {}
    for row in [*candidate_rows, *reference_rows]:
        scheduler_key = str(row.get("scheduler_key", "") or "")
        if scheduler_key not in set(candidate_keys).union(reference_keys):
            continue
        cell = _selection_cell(row)
        schedule_rows = rows_by_schedule.setdefault(scheduler_key, {})
        if cell in schedule_rows:
            raise ValueError(
                "Validation selection contains duplicate rows for "
                f"scheduler={scheduler_key!r}, cell={cell!r}."
            )
        schedule_rows[cell] = row

    expected_cells = set(rows_by_schedule[candidate_keys[0]])
    if not expected_cells:
        raise ValueError("Validation selection requires at least one candidate cell.")
    for scheduler_key in [*candidate_keys, *reference_keys]:
        observed_cells = set(rows_by_schedule.get(scheduler_key, {}))
        if observed_cells != expected_cells:
            missing = sorted(expected_cells - observed_cells)
            extra = sorted(observed_cells - expected_cells)
            raise ValueError(
                "Validation selection requires the same seed/solver/NFE cells for every "
                f"candidate and fixed baseline; scheduler={scheduler_key!r}, "
                f"missing={missing[:8]}, extra={extra[:8]}."
            )

    for cell in sorted(expected_cells):
        identities = {
            _selection_panel_identity(rows_by_schedule[scheduler_key][cell])
            for scheduler_key in [*candidate_keys, *reference_keys]
        }
        if len(identities) != 1:
            raise ValueError(
                "Validation selection candidate and fixed-baseline rows use different "
                f"chosen-example/evaluation panels for cell={cell!r}."
            )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def select_best_validation_schedule(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_rows: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    candidate_rows = [
        dict(row)
        for row in rows
        if str(row.get("scheduler_key", "")) not in set(BASELINE_SCHEDULE_KEYS).union(SER_REFERENCE_SCHEDULE_KEYS)
    ]
    if not candidate_rows:
        raise ValueError("No generated candidate rows were available for validation selection.")
    aggregated, rewards, utility_reference, fixed_reference_scheduler_keys = _selection_rewards(candidate_rows=candidate_rows, reference_rows=reference_rows)
    by_setting: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in aggregated:
        by_setting.setdefault((str(row["solver_key"]), int(row["target_nfe"])), []).append(row)
    scores: Dict[str, List[float]] = {}
    crps_scores: Dict[str, List[float]] = {}
    mase_scores: Dict[str, List[float]] = {}
    worst_metric_scores: Dict[str, List[float]] = {}
    schedule_metadata: Dict[str, Dict[str, Any]] = {}
    per_cell: List[Dict[str, Any]] = []
    for setting, setting_rows in sorted(by_setting.items(), key=lambda item: item[0]):
        for row in setting_rows:
            schedule_key = str(row["scheduler_key"])
            utility = float(rewards[setting][schedule_key])
            u_crps_best = float(row["u_crps_best"])
            u_mase_best = float(row["u_mase_best"])
            worst_metric_utility = min(u_crps_best, u_mase_best)
            scores.setdefault(schedule_key, []).append(float(utility))
            crps_scores.setdefault(schedule_key, []).append(float(u_crps_best))
            mase_scores.setdefault(schedule_key, []).append(float(u_mase_best))
            worst_metric_scores.setdefault(schedule_key, []).append(float(worst_metric_utility))
            schedule_metadata.setdefault(
                schedule_key,
                {
                    "gipo_step_budget": _optional_int(row.get("gipo_step_budget")),
                },
            )
            per_cell.append(
                {
                    "solver_key": setting[0],
                    "target_nfe": int(setting[1]),
                    "scheduler_key": schedule_key,
                    "gipo_step_budget": _optional_int(row.get("gipo_step_budget")),
                    "validation_utility": float(utility),
                    "u_crps_best": float(u_crps_best),
                    "u_mase_best": float(u_mase_best),
                    "u_comp_best": float(row["u_comp_best"]),
                    "u_comp_uniform": row.get("u_comp_uniform"),
                    "worst_metric_utility": float(worst_metric_utility),
                    "best_fixed_crps": float(row["best_fixed_crps"]),
                    "best_fixed_mase": float(row["best_fixed_mase"]),
                    "uniform_crps": row.get("uniform_crps"),
                    "uniform_mase": row.get("uniform_mase"),
                    "forecast_crps": float(row["forecast_crps"]),
                    "forecast_mase": float(row["forecast_mase"]),
                    "n_seeds": int(row.get("n_seeds", 0)),
                }
            )
    table = []
    for schedule_key, values in scores.items():
        metadata = schedule_metadata.get(schedule_key, {})
        budget = _optional_int(metadata.get("gipo_step_budget"))
        table.append(
            {
                "scheduler_key": schedule_key,
                "gipo_step_budget": budget,
                "mean_validation_utility": float(np.mean(np.asarray(values, dtype=np.float64))),
                "mean_u_crps_best": float(np.mean(np.asarray(crps_scores[schedule_key], dtype=np.float64))),
                "mean_u_mase_best": float(np.mean(np.asarray(mase_scores[schedule_key], dtype=np.float64))),
                "mean_min_metric_utility": float(np.mean(np.asarray(worst_metric_scores[schedule_key], dtype=np.float64))),
                "cells": int(len(values)),
            }
        )
    table.sort(
        key=lambda row: (
            -float(row["mean_validation_utility"]),
            -float(row["mean_min_metric_utility"]),
            10**9 if row.get("gipo_step_budget") is None else int(row["gipo_step_budget"]),
            str(row["scheduler_key"]),
        )
    )
    selected = table[0]
    return {
        "selection_split": VALIDATION_PHASE,
        "selection_unit": "generated_schedule_key",
        "utility_reference": utility_reference,
        "fixed_reference_scheduler_keys": fixed_reference_scheduler_keys,
        "selected_scheduler_key": str(selected["scheduler_key"]),
        "gipo_step_budget": _optional_int(selected.get("gipo_step_budget")),
        "tie_break": "mean_validation_utility_then_mean_min_metric_utility_then_smaller_gipo_step_budget_then_scheduler_key",
        "schedule_table": table,
        "per_cell_validation_utilities": per_cell,
    }


def write_selected_schedule_summary(source_summary_path: str | Path, selection: Mapping[str, Any], out_path: str | Path) -> Dict[str, Any]:
    source_path = resolve_project_path(str(source_summary_path))
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Schedule summary {source_path} must contain a mapping payload.")
    reject_retired_evaluation_keys(payload, source=f"Schedule summary {source_path}")
    reject_retired_evaluation_keys(selection, source="Validation schedule selection")
    selected_key = str(selection["selected_scheduler_key"])
    selected_schedule = None
    for schedule in list(payload.get("schedules", []) or []):
        if str(schedule.get("scheduler_key")) == selected_key:
            selected_schedule = dict(schedule)
            break
    if selected_schedule is None:
        raise ValueError(f"Could not find selected schedule {selected_key} in {source_path}.")
    source_predictions = [dict(item) for item in selected_schedule.get("predictions", []) or []]
    metadata_sources: List[Mapping[str, Any]] = [selection, selected_schedule, payload, *source_predictions]
    selected_budget_value = consistent_metadata_value(
        metadata_sources,
        "gipo_step_budget",
        source=f"Selected schedule {selected_key} in {source_path}",
    )
    selected_budget = _optional_int(selected_budget_value)
    shared_metadata: Dict[str, Any] = {
        "method_key": str(
            consistent_metadata_value(
                metadata_sources,
                "method_key",
                source=f"Selected schedule {selected_key} in {source_path}",
            )
            or selected_key
        ),
        "mode": str(
            consistent_metadata_value(
                metadata_sources,
                "mode",
                source=f"Selected schedule {selected_key} in {source_path}",
            )
            or ""
        ),
        "teacher_final_retrain": consistent_metadata_value(
            metadata_sources,
            "teacher_final_retrain",
            source=f"Selected schedule {selected_key} in {source_path}",
        )
        or {},
    }
    for identity_key in ("checkpoint_step", "checkpoint_id", "checkpoint_ids"):
        value = consistent_metadata_value(
            metadata_sources,
            identity_key,
            source=f"Selected schedule {selected_key} in {source_path}",
        )
        if value is not None:
            shared_metadata[identity_key] = value
    predictions = []
    for item in source_predictions:
        copied = {**shared_metadata, **item, "source_scheduler_key": selected_key}
        if selected_budget is not None:
            copied["gipo_step_budget"] = int(selected_budget)
        predictions.append(copied)
    schedule = {
        "scheduler_key": GIPO_POLICY_KEY,
        "schedule_name": SELECTED_STUDENT_SCHEDULE_NAME,
        "comparison_role": "learned_student_selected_by_validation",
        "source_scheduler_key": selected_key,
        **shared_metadata,
        "predictions": predictions,
    }
    if selected_budget is not None:
        schedule["gipo_step_budget"] = int(selected_budget)
    summary = {
        "status": "ready",
        "artifact": "selected_student_schedule_summary",
        "scenario_key": str(payload.get("scenario_key")),
        "selection": dict(selection),
        **shared_metadata,
        "gipo_step_budget": None if selected_budget is None else int(selected_budget),
        "source_scheduler_key": selected_key,
        "baseline_schedule": False,
        "schedules": [schedule],
        "predictions": predictions,
    }
    out = resolve_project_path(str(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, str(out))
    return summary


def build_comparison_summary(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    student_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Sequence[Mapping[str, Any]] = (),
    scenario_key: str,
    benchmark_family: str = FORECAST_FAMILY,
    split_phase: str,
    seeds: Sequence[int],
    solver_names: Sequence[str],
    target_nfe_values: Sequence[int],
) -> Dict[str, Any]:
    family = str(benchmark_family or FORECAST_FAMILY)
    if family != FORECAST_FAMILY:
        if family == "temporal_conditional_generation":
            metric_keys = ("temporal_uw1", "temporal_cw1", "temporal_tstr_f1", "u_comp_uniform")
        elif family == "molecule_3d_coordinate_generation":
            metric_keys = (
                "molecule_kabsch_rmsd_3d",
                "molecule_ensemble_velocity_norm_w1",
                "molecule_ensemble_acceleration_norm_w1",
                "molecule_rollout_velocity_norm_w1",
                "molecule_rollout_acceleration_norm_w1",
                "u_comp_uniform",
            )
        else:
            raise ValueError(f"Unsupported benchmark_family={family!r} for comparison summary.")
        all_rows = [dict(row) for row in baseline_rows] + [dict(row) for row in comparator_rows] + [dict(row) for row in student_rows]
        aggregate_rows = _aggregate_schedule_rows(all_rows)
        by_cell: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for row in aggregate_rows:
            by_cell.setdefault((str(row["solver_key"]), int(row["target_nfe"])), []).append(row)
        student_scheduler_keys = sorted({str(row.get("scheduler_key")) for row in student_rows})
        rankings: List[Dict[str, Any]] = []
        for solver in solver_names:
            for target_nfe in target_nfe_values:
                cell_rows = by_cell.get((str(solver), int(target_nfe)), [])
                uniform = next((row for row in cell_rows if row["scheduler_key"] == "uniform"), None)
                ser_ptg = next((row for row in cell_rows if row["scheduler_key"] == SER_PTG_SCHEDULE_KEY), None)
                baselines = [row for row in cell_rows if row["scheduler_key"] in BASELINE_SCHEDULE_KEYS]
                ranking: Dict[str, Any] = {
                    "benchmark_family": family,
                    "solver_key": str(solver),
                    "target_nfe": int(target_nfe),
                    "student_comparisons": [],
                    "metric_rankings": {},
                }
                for metric in metric_keys:
                    mean_key = f"{metric}_mean"
                    if _metric_higher_is_better(metric):
                        ordered = sorted(cell_rows, key=lambda row: (-_finite_metric_high(row, metric), str(row["scheduler_key"])))
                        best_baseline = max(baselines, key=lambda row: _finite_metric_high(row, metric), default=None)
                    else:
                        ordered = sorted(cell_rows, key=lambda row: (_finite_metric(row, metric), str(row["scheduler_key"])))
                        best_baseline = min(baselines, key=lambda row: _finite_metric(row, metric), default=None)
                    ranking["metric_rankings"][metric] = [row["scheduler_key"] for row in ordered if row.get(mean_key) not in (None, "")]
                    ranking[f"best_baseline_by_{metric}"] = None if best_baseline is None else best_baseline["scheduler_key"]
                for student_row in sorted([row for row in cell_rows if str(row["scheduler_key"]) in student_scheduler_keys], key=lambda row: str(row["scheduler_key"])):
                    comparison: Dict[str, Any] = {"scheduler_key": student_row["scheduler_key"]}
                    for metric in metric_keys:
                        mean_key = f"{metric}_mean"
                        comparison[f"student_{metric}_mean"] = student_row.get(mean_key)
                        if _metric_higher_is_better(metric):
                            comparison[f"student_{metric}_delta_vs_uniform"] = None if uniform is None else (
                                _finite_metric_high(student_row, metric) - _finite_metric_high(uniform, metric)
                            )
                            best_baseline = max(baselines, key=lambda row: _finite_metric_high(row, metric), default=None)
                            comparison[f"student_{metric}_gain_vs_uniform"] = _safe_high_gain(
                                student_row.get(mean_key),
                                None if uniform is None else uniform.get(mean_key),
                            )
                            comparison[f"student_{metric}_gain_vs_best_baseline"] = _safe_high_gain(
                                student_row.get(mean_key),
                                None if best_baseline is None else best_baseline.get(mean_key),
                            )
                            comparison[f"student_{metric}_gain_vs_ser_ptg"] = _safe_high_gain(
                                student_row.get(mean_key),
                                None if ser_ptg is None else ser_ptg.get(mean_key),
                            )
                        else:
                            best_baseline = min(baselines, key=lambda row: _finite_metric(row, metric), default=None)
                            comparison[f"student_{metric}_gain_vs_uniform"] = _safe_gain(
                                student_row.get(mean_key),
                                None if uniform is None else uniform.get(mean_key),
                            )
                            comparison[f"student_{metric}_gain_vs_best_baseline"] = _safe_gain(
                                student_row.get(mean_key),
                                None if best_baseline is None else best_baseline.get(mean_key),
                            )
                            comparison[f"student_{metric}_gain_vs_ser_ptg"] = _safe_gain(
                                student_row.get(mean_key),
                                None if ser_ptg is None else ser_ptg.get(mean_key),
                            )
                    ranking["student_comparisons"].append(comparison)
                rankings.append(ranking)
        student_missing = _missing_cells(
            student_rows,
            seeds=seeds,
            solver_names=solver_names,
            target_nfe_values=target_nfe_values,
            scheduler_keys=student_scheduler_keys,
        )
        baseline_missing = _missing_cells(
            baseline_rows,
            seeds=seeds,
            solver_names=solver_names,
            target_nfe_values=target_nfe_values,
            scheduler_keys=BASELINE_SCHEDULE_KEYS,
        )
        ser_missing = _missing_cells(
            comparator_rows,
            seeds=seeds,
            solver_names=solver_names,
            target_nfe_values=target_nfe_values,
            scheduler_keys=SER_REFERENCE_SCHEDULE_KEYS,
        )
        return {
            "evaluator_signature": EVALUATOR_SIGNATURE_VERSION,
            "benchmark_family": family,
            "scenario_key": str(scenario_key),
            "split_phase": str(split_phase),
            "baseline_scheduler_keys": list(BASELINE_SCHEDULE_KEYS),
            "ser_reference_scheduler_keys": list(SER_REFERENCE_SCHEDULE_KEYS),
            "student_scheduler_key": student_scheduler_keys[0] if len(student_scheduler_keys) == 1 else None,
            "student_scheduler_keys": student_scheduler_keys,
            "seeds": [int(seed) for seed in seeds],
            "solver_names": [str(solver) for solver in solver_names],
            "target_nfe_values": [int(nfe) for nfe in target_nfe_values],
            "expected_baseline_rows": int(
                len(seeds) * len(solver_names) * len(target_nfe_values) * len(BASELINE_SCHEDULE_KEYS)
            ),
            "observed_baseline_rows": int(len(baseline_rows)),
            "missing_baseline_cells": baseline_missing,
            "expected_ser_ptg_rows": int(
                len(seeds) * len(solver_names) * len(target_nfe_values) * len(SER_REFERENCE_SCHEDULE_KEYS)
            ),
            "observed_ser_ptg_rows": int(len(comparator_rows)),
            "missing_ser_ptg_cells": ser_missing,
            "expected_student_rows": int(len(seeds) * len(solver_names) * len(target_nfe_values) * len(student_scheduler_keys)),
            "observed_student_rows": int(len(student_rows)),
            "missing_student_cells": student_missing,
            "metric_keys": list(metric_keys),
            "schedule_summaries": aggregate_rows,
            "cell_rankings": rankings,
        }
    all_rows = [dict(row) for row in baseline_rows] + [dict(row) for row in comparator_rows] + [dict(row) for row in student_rows]
    aggregate_rows = _aggregate_schedule_rows(all_rows)
    by_cell: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in aggregate_rows:
        by_cell.setdefault((str(row["solver_key"]), int(row["target_nfe"])), []).append(row)
    rankings: List[Dict[str, Any]] = []
    student_scheduler_keys = sorted({str(row.get("scheduler_key")) for row in student_rows})
    for solver in solver_names:
        for target_nfe in target_nfe_values:
            cell_rows = by_cell.get((str(solver), int(target_nfe)), [])
            student_rows_for_cell = [row for row in cell_rows if str(row["scheduler_key"]) in student_scheduler_keys]
            student = next((row for row in student_rows_for_cell if row["scheduler_key"] == GIPO_POLICY_KEY), None)
            if student is None and len(student_scheduler_keys) == 1:
                student = next(iter(student_rows_for_cell), None)
            uniform = next((row for row in cell_rows if row["scheduler_key"] == "uniform"), None)
            ser_ptg = next((row for row in cell_rows if row["scheduler_key"] == SER_PTG_SCHEDULE_KEY), None)
            baselines = [row for row in cell_rows if row["scheduler_key"] in BASELINE_SCHEDULE_KEYS]
            best_crps = min(baselines, key=lambda row: _finite_metric(row, "forecast_crps"), default=None)
            best_mase = min(baselines, key=lambda row: _finite_metric(row, "forecast_mase"), default=None)
            ordered_crps = sorted(cell_rows, key=lambda row: (_finite_metric(row, "forecast_crps"), str(row["scheduler_key"])))
            ordered_mase = sorted(cell_rows, key=lambda row: (_finite_metric(row, "forecast_mase"), str(row["scheduler_key"])))
            ranking: Dict[str, Any] = {
                "solver_key": str(solver),
                "target_nfe": int(target_nfe),
                "forecast_crps_ranking": [row["scheduler_key"] for row in ordered_crps],
                "forecast_mase_ranking": [row["scheduler_key"] for row in ordered_mase],
                "best_baseline_by_forecast_crps": None if best_crps is None else best_crps["scheduler_key"],
                "best_baseline_by_forecast_mase": None if best_mase is None else best_mase["scheduler_key"],
                "ser_ptg_forecast_crps_mean": None if ser_ptg is None else ser_ptg.get("forecast_crps_mean"),
                "ser_ptg_forecast_mase_mean": None if ser_ptg is None else ser_ptg.get("forecast_mase_mean"),
                "student_comparisons": [],
            }
            for student_row in sorted(student_rows_for_cell, key=lambda row: str(row["scheduler_key"])):
                ranking["student_comparisons"].append(
                    {
                        "scheduler_key": student_row["scheduler_key"],
                        "student_forecast_crps_mean": student_row.get("forecast_crps_mean"),
                        "student_forecast_mase_mean": student_row.get("forecast_mase_mean"),
                        "student_relative_forecast_crps_gain_vs_uniform": _safe_gain(
                            student_row.get("forecast_crps_mean"),
                            None if uniform is None else uniform.get("forecast_crps_mean"),
                        ),
                        "student_relative_forecast_mase_gain_vs_uniform": _safe_gain(
                            student_row.get("forecast_mase_mean"),
                            None if uniform is None else uniform.get("forecast_mase_mean"),
                        ),
                        "student_relative_forecast_crps_gain_vs_best_baseline": _safe_gain(
                            student_row.get("forecast_crps_mean"),
                            None if best_crps is None else best_crps.get("forecast_crps_mean"),
                        ),
                        "student_relative_forecast_mase_gain_vs_best_baseline": _safe_gain(
                            student_row.get("forecast_mase_mean"),
                            None if best_mase is None else best_mase.get("forecast_mase_mean"),
                        ),
                        "student_relative_forecast_crps_gain_vs_ser_ptg": _safe_gain(
                            student_row.get("forecast_crps_mean"),
                            None if ser_ptg is None else ser_ptg.get("forecast_crps_mean"),
                        ),
                        "student_relative_forecast_mase_gain_vs_ser_ptg": _safe_gain(
                            student_row.get("forecast_mase_mean"),
                            None if ser_ptg is None else ser_ptg.get("forecast_mase_mean"),
                        ),
                        "student_internal_fraction_after_098_mean": student_row.get("internal_fraction_after_098_mean"),
                    }
                )
            if student is not None:
                ranking.update(
                    {
                        "student_forecast_crps_mean": student.get("forecast_crps_mean"),
                        "student_forecast_mase_mean": student.get("forecast_mase_mean"),
                        "student_relative_forecast_crps_gain_vs_uniform": _safe_gain(student.get("forecast_crps_mean"), None if uniform is None else uniform.get("forecast_crps_mean")),
                        "student_relative_forecast_mase_gain_vs_uniform": _safe_gain(student.get("forecast_mase_mean"), None if uniform is None else uniform.get("forecast_mase_mean")),
                        "student_relative_forecast_crps_gain_vs_best_baseline": _safe_gain(student.get("forecast_crps_mean"), None if best_crps is None else best_crps.get("forecast_crps_mean")),
                        "student_relative_forecast_mase_gain_vs_best_baseline": _safe_gain(student.get("forecast_mase_mean"), None if best_mase is None else best_mase.get("forecast_mase_mean")),
                        "student_relative_forecast_crps_gain_vs_ser_ptg": _safe_gain(student.get("forecast_crps_mean"), None if ser_ptg is None else ser_ptg.get("forecast_crps_mean")),
                        "student_relative_forecast_mase_gain_vs_ser_ptg": _safe_gain(student.get("forecast_mase_mean"), None if ser_ptg is None else ser_ptg.get("forecast_mase_mean")),
                        "student_internal_fraction_after_098_mean": student.get("internal_fraction_after_098_mean"),
                    }
                )
            rankings.append(ranking)
    baseline_missing = _missing_cells(
        baseline_rows,
        seeds=seeds,
        solver_names=solver_names,
        target_nfe_values=target_nfe_values,
        scheduler_keys=BASELINE_SCHEDULE_KEYS,
    )
    student_missing = _missing_cells(
        student_rows,
        seeds=seeds,
        solver_names=solver_names,
        target_nfe_values=target_nfe_values,
        scheduler_keys=student_scheduler_keys,
    )
    ser_missing = _missing_cells(
        comparator_rows,
        seeds=seeds,
        solver_names=solver_names,
        target_nfe_values=target_nfe_values,
        scheduler_keys=SER_REFERENCE_SCHEDULE_KEYS,
    )
    return {
        "evaluator_signature": EVALUATOR_SIGNATURE_VERSION,
        "scenario_key": str(scenario_key),
        "split_phase": str(split_phase),
        "baseline_scheduler_keys": list(BASELINE_SCHEDULE_KEYS),
        "ser_ptg_scheduler_key": SER_PTG_SCHEDULE_KEY,
        "ser_reference_scheduler_keys": list(SER_REFERENCE_SCHEDULE_KEYS),
        "ser_ptg_is_baseline": SER_PTG_SCHEDULE_KEY in BASELINE_SCHEDULE_KEYS,
        "student_scheduler_key": student_scheduler_keys[0] if len(student_scheduler_keys) == 1 else None,
        "student_scheduler_keys": student_scheduler_keys,
        "student_is_baseline": False if len(student_scheduler_keys) != 1 else student_scheduler_keys[0] in BASELINE_SCHEDULE_KEYS,
        "student_scheduler_key_is_baseline": {key: key in BASELINE_SCHEDULE_KEYS for key in student_scheduler_keys},
        "seeds": [int(seed) for seed in seeds],
        "solver_names": [str(solver) for solver in solver_names],
        "target_nfe_values": [int(nfe) for nfe in target_nfe_values],
        "expected_baseline_rows": int(len(seeds) * len(solver_names) * len(target_nfe_values) * len(BASELINE_SCHEDULE_KEYS)),
        "observed_baseline_rows": int(len(baseline_rows)),
        "missing_baseline_cells": baseline_missing,
        "expected_ser_ptg_rows": int(
            len(seeds) * len(solver_names) * len(target_nfe_values) * len(SER_REFERENCE_SCHEDULE_KEYS)
        ),
        "observed_ser_ptg_rows": int(len(comparator_rows)),
        "missing_ser_ptg_cells": ser_missing,
        "expected_student_rows": int(len(seeds) * len(solver_names) * len(target_nfe_values) * len(student_scheduler_keys)),
        "observed_student_rows": int(len(student_rows)),
        "missing_student_cells": student_missing,
        "schedule_summaries": aggregate_rows,
        "cell_rankings": rankings,
    }


def evaluate_schedule_summary(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = resolve_project_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_csv(args.seeds)
    solver_names = list(normalize_solver_keys(str(args.solver_names)))
    target_nfes = parse_int_csv(args.target_nfe_values)
    split_phase = str(args.split_phase)
    if split_phase not in {TRAIN_TUNING_PHASE, VALIDATION_PHASE, LOCKED_TEST_PHASE}:
        raise ValueError(f"split_phase must be {TRAIN_TUNING_PHASE!r}, {VALIDATION_PHASE!r}, or {LOCKED_TEST_PHASE!r}.")
    fallback_row_csv_name = {
        TRAIN_TUNING_PHASE: "train_tuning_rows.csv",
        VALIDATION_PHASE: "validation_rows.csv",
        LOCKED_TEST_PHASE: "test_rows.csv",
    }[split_phase]
    fallback_row_jsonl_name = {
        TRAIN_TUNING_PHASE: "train_tuning_rows.jsonl",
        VALIDATION_PHASE: "validation_rows.jsonl",
        LOCKED_TEST_PHASE: "test_rows.jsonl",
    }[split_phase]
    row_csv_name = str(args.row_csv_name or fallback_row_csv_name)
    csv_path = _output_path(
        out_dir,
        row_csv_name,
        fallback=fallback_row_csv_name,
        suffix=".csv",
        label="row CSV output",
    )
    jsonl_path = _output_path(
        out_dir,
        args.row_jsonl_name,
        fallback=fallback_row_jsonl_name,
        suffix=".jsonl",
        label="row JSONL output",
    )
    context_csv_path = _output_path(
        out_dir,
        args.context_row_csv_name,
        fallback=f"context_{row_csv_name}",
        suffix=".csv",
        label="context-row CSV output",
    )
    context_embeddings_path = _output_path(
        out_dir,
        args.context_embeddings_npz_name,
        fallback="context_embeddings.npz",
        suffix=".npz",
        label="context-embedding output",
    )
    summary_path = _output_path(
        out_dir,
        args.summary_output_name,
        fallback=f"{split_phase}_schedule_summary.json",
        suffix=".json",
        label="summary output",
    )
    comparison_path = _output_path(
        out_dir,
        args.comparison_output_name,
        fallback="student_vs_baselines_ser_ptg_summary.json",
        suffix=".json",
        label="comparison output",
    )
    selection_path = out_dir / "validation_schedule_selection.json"
    selected_summary_path = out_dir / "selected_student_schedule_summary.json"
    output_paths: Dict[str, Path] = {
        "row CSV": csv_path,
        "row JSONL": jsonl_path,
        "summary JSON": summary_path,
        "comparison JSON": comparison_path,
    }
    if bool(args.write_context_rows):
        output_paths.update(
            {
                "context-row CSV": context_csv_path,
                "context embeddings": context_embeddings_path,
                "context-embedding manifest": context_embedding_table_manifest_path(
                    context_embeddings_path
                ),
            }
        )
    if bool(args.select_schedule_from_validation):
        output_paths.update(
            {
                "validation selection": selection_path,
                "selected schedule summary": selected_summary_path,
            }
        )
    input_paths = {
        "schedule summary": resolve_project_path(str(args.schedule_summary)),
    }
    for label, value in (
        ("baseline rows", args.baseline_rows),
        ("comparator rows", args.comparator_rows),
        ("selection reference rows", args.selection_reference_rows),
    ):
        if str(value).strip():
            input_paths[label] = resolve_project_path(str(value))
    _validate_distinct_artifact_paths(output_paths, input_paths=input_paths)

    predictions = load_schedule_predictions(
        args.schedule_summary,
        scenario_key=str(args.scenario_key),
        solver_names=solver_names,
        target_nfe_values=target_nfes,
        require_complete=True,
    )
    scheduler_keys = sorted({key[0] for key in predictions})
    protocol_hash = _protocol_hash(args)

    dataset_root = resolve_project_path(str(args.dataset_root))
    shared_backbone_root = resolve_project_path(str(args.shared_backbone_root))
    device = resolve_torch_device(str(args.device))
    checkpoint = load_forecast_checkpoint_splits(
        cli_args=args,
        dataset_root=dataset_root,
        shared_backbone_root=shared_backbone_root,
        dataset=str(args.scenario_key),
        device=device,
    )
    checkpoint_step = int(checkpoint["checkpoint_step"])
    if checkpoint_step != int(args.checkpoint_step):
        raise ValueError(
            f"Loaded checkpoint checkpoint_step={checkpoint_step} does not match --checkpoint_step={int(args.checkpoint_step)}."
        )
    _validate_schedule_checkpoint_identity(
        predictions,
        checkpoint_step=checkpoint_step,
        checkpoint_id=str(checkpoint["checkpoint_id"]),
    )
    comparison_row_args = {
        "scenario_key": str(args.scenario_key),
        "split_phase": split_phase,
        "seeds": seeds,
        "solver_names": solver_names,
        "target_nfe_values": target_nfes,
        "checkpoint_step": checkpoint_step,
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
    }
    selection_reference_rows = (
        _load_forecast_rows_csv(args.selection_reference_rows, **comparison_row_args)
        if str(args.selection_reference_rows).strip()
        else []
    )
    baseline_comparison_rows = (
        _load_forecast_rows_csv(args.baseline_rows, **comparison_row_args)
        if str(args.baseline_rows).strip()
        else []
    )
    comparator_comparison_rows = (
        _load_forecast_rows_csv(args.comparator_rows, **comparison_row_args)
        if str(args.comparator_rows).strip()
        else []
    )
    rows_by_key = _load_existing_rows(jsonl_path, protocol_hash=protocol_hash)
    context_rows_by_signature = _load_context_rows(context_csv_path) if bool(args.write_context_rows) else {}
    original_context_signatures = set(context_rows_by_signature)
    context_embeddings: Dict[str, Sequence[float]] = {}
    context_table_was_invalid = False
    if bool(args.write_context_rows) and context_embeddings_path.exists():
        try:
            context_embeddings = load_context_embedding_table(
                context_embeddings_path,
                expected_context_embedding_kind=str(args.context_embedding_kind),
                require_manifest=True,
                expected_context_rows=list(context_rows_by_signature.values()),
            )
        except ValueError:
            context_embeddings = {}
            context_table_was_invalid = True
    original_embedding_ids = set(context_embeddings)
    if bool(args.write_context_rows):
        rows_by_key = {
            key: row
            for key, row in rows_by_key.items()
            if _row_has_complete_context_artifacts(
                row,
                context_rows_by_signature=context_rows_by_signature,
                context_embeddings=context_embeddings,
                context_embedding_kind=str(args.context_embedding_kind),
            )
        }
        retained_parent_signatures = {
            str(row.get("row_signature", "") or "").strip()
            for row in rows_by_key.values()
        }
        context_rows_by_signature = {
            signature: row
            for signature, row in context_rows_by_signature.items()
            if str(row.get("parent_row_signature", "") or "").strip()
            in retained_parent_signatures
        }
        retained_embedding_ids = {
            str(row.get("context_embedding_id", "") or "").strip()
            for row in context_rows_by_signature.values()
            if str(row.get("context_embedding_id", "") or "").strip()
        }
        context_embeddings = {
            embedding_id: vector
            for embedding_id, vector in context_embeddings.items()
            if embedding_id in retained_embedding_ids
        }
        _write_context_csv(
            context_csv_path,
            list(context_rows_by_signature.values()),
        )
        context_artifacts_changed = (
            context_table_was_invalid
            or set(context_rows_by_signature) != original_context_signatures
            or set(context_embeddings) != original_embedding_ids
        )
        if context_artifacts_changed:
            if context_embeddings:
                save_context_embedding_table(
                    context_embeddings_path,
                    context_embeddings,
                    metadata={"context_embedding_kind": str(args.context_embedding_kind)},
                    context_rows=list(context_rows_by_signature.values()),
                )
            else:
                context_embeddings_path.unlink(missing_ok=True)
                context_embedding_table_manifest_path(context_embeddings_path).unlink(
                    missing_ok=True
                )
    _write_jsonl(jsonl_path, list(rows_by_key.values()))
    _write_csv(csv_path, list(rows_by_key.values()))
    model = checkpoint["model"]
    cfg = checkpoint["cfg"]
    split_key = {TRAIN_TUNING_PHASE: "train", VALIDATION_PHASE: "val", LOCKED_TEST_PHASE: "test"}[split_phase]
    eval_ds = checkpoint["splits"][split_key]
    train_tuning_reference_examples = int(len(checkpoint["splits"].get("val", [])))
    selected_examples_cap, selected_examples_cap_source = _split_example_cap(args, split_phase)
    mode = "a" if rows_by_key else "w"
    total_cells = len(seeds) * len(scheduler_keys) * len(target_nfes) * len(solver_names)
    with jsonl_path.open(mode, encoding="utf-8") as fh, ProgressBar(total_cells, f"{split_phase} inference cells") as progress:
        for seed in seeds:
            if split_phase == TRAIN_TUNING_PHASE:
                tuning_seed = int(args.train_tuning_seed) + int(seed)
                uncapped_candidate_examples = train_tuning_target_example_count(
                    len(eval_ds),
                    fraction=float(args.eval_train_fraction),
                    sampling_mode=str(args.train_tuning_sampling_mode),
                    strata=int(args.train_tuning_strata),
                    reference_examples=int(train_tuning_reference_examples),
                    train_split_fraction=float(args.train_tuning_train_split_fraction),
                    val_split_fraction=float(args.train_tuning_val_split_fraction),
                )
                candidate_examples = choose_forecast_train_tuning_indices(
                    eval_ds,
                    fraction=float(args.eval_train_fraction),
                    seed=int(tuning_seed),
                    strata=int(args.train_tuning_strata),
                    dataset=str(args.scenario_key),
                    sampling_mode=str(args.train_tuning_sampling_mode),
                    reference_examples=int(train_tuning_reference_examples),
                    train_split_fraction=float(args.train_tuning_train_split_fraction),
                    val_split_fraction=float(args.train_tuning_val_split_fraction),
                    max_examples=int(selected_examples_cap),
                )
                chosen_examples, selection_meta = cap_context_indices(
                    candidate_examples,
                    cap=int(selected_examples_cap),
                    seed=int(tuning_seed),
                    salt=f"forecast_summary|{args.scenario_key}|{split_phase}",
                    selection_protocol=SCHEDULE_CONTEXT_SELECTION_PROTOCOL,
                    uncapped_candidate_examples=int(uncapped_candidate_examples),
                )
            else:
                effective_cap = int(selected_examples_cap) if selected_examples_cap is not None else int(len(eval_ds))
                chosen = choose_forecast_example_indices(eval_ds, n_examples=effective_cap, seed=int(seed))
                chosen_examples, selection_meta = cap_context_indices(
                    chosen,
                    cap=int(effective_cap),
                    seed=int(seed),
                    salt=f"forecast_summary|{args.scenario_key}|{split_phase}",
                    selection_protocol=SCHEDULE_CONTEXT_SELECTION_PROTOCOL,
                    uncapped_candidate_examples=int(len(eval_ds)),
                )
            for scheduler_key in scheduler_keys:
                for target_nfe in target_nfes:
                    for solver_key in solver_names:
                        prediction = predictions[(str(scheduler_key), str(solver_key), int(target_nfe))]
                        row_stub = {
                            "protocol_hash": protocol_hash,
                            "split_phase": split_phase,
                            "seed": int(seed),
                            "scenario_key": str(args.scenario_key),
                            "target_nfe": int(target_nfe),
                            "solver_key": str(solver_key),
                            "scheduler_key": str(scheduler_key),
                        }
                        key = _row_key(row_stub)
                        if key in rows_by_key:
                            progress.update()
                            continue
                        eval_seed = int(seed)
                        metrics = evaluate_forecast_schedule(
                            model,
                            eval_ds,
                            cfg,
                            solver_name=normalize_solver_key(solver_key),
                            macro_steps=int(prediction["macro_steps"]),
                            target_nfe=int(target_nfe),
                            time_grid=prediction["time_grid"],
                            num_eval_samples=int(args.num_eval_samples),
                            seed=int(eval_seed),
                            logical_seed=int(seed),
                            scheduler_key=str(scheduler_key),
                            scenario_key=str(args.scenario_key),
                            split_phase=str(split_phase),
                            checkpoint_id=str(checkpoint["checkpoint_id"]),
                            example_indices=chosen_examples,
                            batch_size=int(args.forecast_eval_batch_size),
                            progress_label=f"{split_phase} {scheduler_key} seed={seed} {solver_key}/{target_nfe}",
                            return_per_example_rows=bool(args.write_context_rows),
                            return_context_embeddings=bool(args.write_context_rows),
                            context_embedding_kind=str(args.context_embedding_kind),
                        )
                        row = _schedule_row(
                            seed=int(seed),
                            scenario_key=str(args.scenario_key),
                            split_phase=split_phase,
                            checkpoint=checkpoint,
                            prediction=prediction,
                            metrics=metrics,
                            protocol_hash=protocol_hash,
                        )
                        row.update(
                            {
                                "example_selection_protocol": str(selection_meta["example_selection_protocol"]),
                                "context_sample_count": int(_context_sample_cap(args)),
                                "selected_examples": int(selection_meta["selected_examples"]),
                                "selected_examples_cap": int(selection_meta["selected_examples_cap"]),
                                "selected_examples_cap_source": str(selected_examples_cap_source),
                                "uncapped_candidate_examples": int(selection_meta["uncapped_candidate_examples"]),
                                "candidate_examples_after_initial_selection": int(selection_meta["candidate_examples_after_initial_selection"]),
                                "selection_was_capped": bool(selection_meta["selection_was_capped"]),
                            }
                        )
                        row.update(
                            _selection_provenance(
                                args,
                                split_phase=split_phase,
                                selection_was_capped=bool(selection_meta["selection_was_capped"]),
                            )
                        )
                        if split_phase == TRAIN_TUNING_PHASE:
                            row.update(
                                {
                                    "train_tuning_fraction": float(args.eval_train_fraction),
                                    "train_tuning_seed": int(args.train_tuning_seed) + int(seed),
                                    "train_tuning_strata": int(args.train_tuning_strata),
                                    "train_tuning_sampler": train_tuning_sampler_key(str(args.train_tuning_sampling_mode)),
                                    "train_tuning_sampling_mode": str(args.train_tuning_sampling_mode),
                                    "train_tuning_reference_examples": int(train_tuning_reference_examples),
                                    "train_tuning_target_examples": int(len(chosen_examples)),
                                    "train_tuning_uncapped_candidate_examples": int(selection_meta["uncapped_candidate_examples"]),
                                    "train_tuning_train_split_fraction": float(args.train_tuning_train_split_fraction),
                                    "train_tuning_val_split_fraction": float(args.train_tuning_val_split_fraction),
                                }
                            )
                        if bool(args.write_context_rows):
                            for detail_row in list(metrics.get("per_example_rows", []) or []):
                                copied_detail = dict(detail_row)
                                copied_detail.update(
                                    {
                                        "benchmark_family": FORECAST_FAMILY,
                                        "parent_row_signature": str(row.get("row_signature", "")),
                                        "protocol_hash": str(protocol_hash),
                                        "logical_seed": int(copied_detail.get("logical_seed", seed)),
                                        "evaluation_seed": int(copied_detail.get("evaluation_seed", eval_seed)),
                                        "scenario_key": str(args.scenario_key),
                                        "checkpoint_step": int(checkpoint_step),
                                        "checkpoint_id": str(checkpoint["checkpoint_id"]),
                                        "method_key": str(prediction.get("method_key") or prediction["scheduler_key"]),
                                        "gipo_step_budget": prediction.get("gipo_step_budget"),
                                        "mode": prediction.get("mode", ""),
                                        "teacher_final_retrain": json.dumps(
                                            prediction.get("teacher_final_retrain", {}),
                                            sort_keys=True,
                                            separators=(",", ":"),
                                        )
                                        if isinstance(prediction.get("teacher_final_retrain"), Mapping)
                                        else prediction.get("teacher_final_retrain", ""),
                                    }
                                )
                                copied_detail.update(
                                    _selection_provenance(
                                        args,
                                        split_phase=split_phase,
                                        selection_was_capped=bool(selection_meta["selection_was_capped"]),
                                    )
                                )
                                copied_detail["context_embedding_kind"] = str(
                                    args.context_embedding_kind
                                )
                                if split_phase == TRAIN_TUNING_PHASE:
                                    copied_detail.update(
                                        {
                                            "train_tuning_fraction": float(args.eval_train_fraction),
                                            "train_tuning_seed": int(args.train_tuning_seed) + int(seed),
                                            "train_tuning_strata": int(args.train_tuning_strata),
                                            "train_tuning_sampler": train_tuning_sampler_key(str(args.train_tuning_sampling_mode)),
                                        }
                                    )
                                signature = str(copied_detail["row_signature"])
                                if signature in context_rows_by_signature:
                                    raise ValueError(f"Duplicate context row signature while appending context artifacts: {signature}")
                                context_rows_by_signature[signature] = copied_detail
                            _merge_context_embeddings_checked(
                                context_embeddings,
                                dict(metrics.get("context_embeddings", {}) or {}),
                            )
                            _write_context_csv(context_csv_path, list(context_rows_by_signature.values()))
                            if context_embeddings:
                                save_context_embedding_table(
                                    context_embeddings_path,
                                    context_embeddings,
                                    metadata={
                                        "checkpoint_id": str(checkpoint["checkpoint_id"]),
                                        "scenario_key": str(args.scenario_key),
                                        "split_phase": str(split_phase),
                                        "context_embedding_kind": str(args.context_embedding_kind),
                                        "chosen_examples_hash": str(metrics.get("chosen_examples_hash", "")),
                                        "evaluation_protocol_hash": str(metrics.get("evaluation_protocol_hash", "")),
                                    },
                                    context_rows=list(context_rows_by_signature.values()),
                                )
                            if not _row_has_complete_context_artifacts(
                                row,
                                context_rows_by_signature=context_rows_by_signature,
                                context_embeddings=context_embeddings,
                                context_embedding_kind=str(args.context_embedding_kind),
                            ):
                                raise ValueError(
                                    "Schedule evaluation did not produce a complete, identity-matched "
                                    "context row and embedding set for its parent row."
                                )
                        rows_by_key[_row_key(row)] = row
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                        fh.flush()
                        _write_csv(csv_path, list(rows_by_key.values()))
                        progress.update()
    rows = list(rows_by_key.values())
    summary: Dict[str, Any] = {
        "evaluator_signature": EVALUATOR_SIGNATURE_VERSION,
        "scenario_key": str(args.scenario_key),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "split_phase": split_phase,
        "scheduler_keys": scheduler_keys,
        "scheduler_key_is_section_15_baseline": {key: key in BASELINE_SCHEDULE_KEYS for key in scheduler_keys},
        "seeds": [int(seed) for seed in seeds],
        "solver_names": solver_names,
        "target_nfe_values": [int(nfe) for nfe in target_nfes],
        "expected_rows": int(len(seeds) * len(solver_names) * len(target_nfes) * len(scheduler_keys)),
        "observed_rows": int(len(rows)),
        "missing_cells": _missing_cells(
            rows,
            seeds=seeds,
            solver_names=solver_names,
            target_nfe_values=target_nfes,
            scheduler_keys=scheduler_keys,
        ),
        "row_csv": display_project_path(csv_path),
        "row_jsonl": display_project_path(jsonl_path),
        "context_row_csv": display_project_path(context_csv_path) if bool(args.write_context_rows) else "",
        "context_embeddings_npz": display_project_path(context_embeddings_path) if bool(args.write_context_rows) else "",
        "context_row_count": int(len(context_rows_by_signature)) if bool(args.write_context_rows) else 0,
        "context_embedding_count": int(len(context_embeddings)) if bool(args.write_context_rows) else 0,
        "schedule_summaries": _aggregate_schedule_rows(rows),
    }
    summary.update(
        _selection_provenance(
            args,
            split_phase=split_phase,
            selection_was_capped=any(bool(row.get("selection_was_capped", False)) for row in rows),
        )
    )
    if split_phase == TRAIN_TUNING_PHASE:
        summary["train_tuning"] = {
            "fraction": float(args.eval_train_fraction),
            "seed": int(args.train_tuning_seed),
            "strata": int(args.train_tuning_strata),
            "sampling_mode": str(args.train_tuning_sampling_mode),
            "sampler": train_tuning_sampler_key(str(args.train_tuning_sampling_mode)),
            "reference_split_key": "val",
            "reference_examples": int(train_tuning_reference_examples),
            "max_examples": int(selected_examples_cap),
            "max_examples_source": str(selected_examples_cap_source),
            "train_split_fraction": float(args.train_tuning_train_split_fraction),
            "val_split_fraction": float(args.train_tuning_val_split_fraction),
            "split_key": "train",
        }
    if bool(args.select_schedule_from_validation):
        if split_phase != VALIDATION_PHASE:
            raise ValueError("Validation selection requires --split_phase validation_tuning.")
        selection = select_best_validation_schedule(rows, reference_rows=selection_reference_rows)
        save_json(selection, str(selection_path))
        selected_summary = write_selected_schedule_summary(
            args.schedule_summary,
            selection,
            selected_summary_path,
        )
        summary["selection"] = selection
        summary["selection_json"] = display_project_path(selection_path)
        summary["selected_student_schedule_summary"] = display_project_path(
            selected_summary_path
        )
        summary["selected_summary_schedule_count"] = int(len(selected_summary.get("schedules", [])))
    if str(args.baseline_rows).strip():
        baseline_rows = list(baseline_comparison_rows)
        baseline_rows = _filter_rows_to_scheduler_keys(baseline_rows, BASELINE_SCHEDULE_KEYS)
        comparator_rows: List[Dict[str, Any]] = []
        if str(args.comparator_rows).strip():
            comparator_rows = list(comparator_comparison_rows)
            comparator_rows = _filter_rows_to_scheduler_keys(comparator_rows, SER_REFERENCE_SCHEDULE_KEYS)
        comparison = build_comparison_summary(
            baseline_rows=baseline_rows,
            student_rows=rows,
            comparator_rows=comparator_rows,
            scenario_key=str(args.scenario_key),
            split_phase=split_phase,
            seeds=seeds,
            solver_names=solver_names,
            target_nfe_values=target_nfes,
        )
        save_json(comparison, str(comparison_path))
        summary["comparison_summary"] = comparison
    save_json(summary, str(summary_path))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate forecast schedule-summary grids on train-tuning, validation, or locked test splits.")
    parser.add_argument("--scenario_key", default="traffic_hourly")
    parser.add_argument("--schedule_summary", required=True)
    parser.add_argument("--split_phase", choices=(TRAIN_TUNING_PHASE, VALIDATION_PHASE, LOCKED_TEST_PHASE), required=True)
    parser.add_argument("--out_dir", default=str(project_outputs_root() / "schedule_summary_evaluation"))
    parser.add_argument("--row_csv_name", default="")
    parser.add_argument("--row_jsonl_name", default="")
    parser.add_argument("--summary_output_name", default="")
    parser.add_argument("--comparison_output_name", default="student_vs_baselines_ser_ptg_summary.json")
    parser.add_argument("--baseline_rows", default="")
    parser.add_argument("--comparator_rows", default="")
    parser.add_argument("--selection_reference_rows", default="")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--solver_names", default=",".join(SUPPORTED_SOLVER_KEYS))
    parser.add_argument("--target_nfe_values", default=",".join(str(x) for x in REFERENCE_SEEN_NFES))
    parser.add_argument("--num_eval_samples", type=int, default=5)
    parser.add_argument("--forecast_eval_batch_size", type=int, default=64)
    parser.add_argument("--context_sample_count", type=int, default=TRAIN_TUNING_CONTEXT_SAMPLE_COUNT)
    parser.add_argument("--write_context_rows", action="store_true", default=False)
    parser.add_argument("--context_row_csv_name", default="")
    parser.add_argument("--context_embeddings_npz_name", default="")
    parser.add_argument("--context_embedding_kind", choices=("ctx_summary", "summary"), default="ctx_summary")
    parser.add_argument("--eval_train_fraction", type=float, default=0.20)
    parser.add_argument("--train_tuning_seed", type=int, default=0)
    parser.add_argument("--train_tuning_strata", type=int, default=20)
    parser.add_argument("--train_tuning_sampling_mode", choices=TRAIN_TUNING_SAMPLING_MODES, default=TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION)
    parser.add_argument("--train_tuning_train_split_fraction", type=float, default=DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION)
    parser.add_argument("--train_tuning_val_split_fraction", type=float, default=DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION)
    parser.add_argument("--eval_windows_val", type=int, default=0)
    parser.add_argument(
        "--locked_test_preview",
        action="store_true",
        help="Limit locked-test evaluation for a deterministic result preview; full evaluation is the default.",
    )
    parser.add_argument(
        "--locked_test_preview_contexts",
        type=int,
        default=None,
        help="Per-seed preview limit; requires --locked_test_preview and defaults to 512.",
    )
    parser.add_argument("--dataset_root", default=str(project_dataset_root()))
    parser.add_argument("--shared_backbone_root", default=str(DEFAULT_SHARED_BACKBONE_ROOT))
    parser.add_argument("--backbone_manifest", default=str(backbone_manifest_path()))
    parser.add_argument("--checkpoint_step", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--select_schedule_from_validation", action="store_true", default=False)
    return parser


def main() -> None:
    summary = evaluate_schedule_summary(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
