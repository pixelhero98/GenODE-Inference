from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

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
    validate_execution_preflight,
)
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
from genode.models.otflow_train_val import save_json
from genode.runtime import resolve_torch_device
from genode.gipo.policy import load_context_embedding_table, save_context_embedding_table

RUNNER_SIGNATURE_VERSION = "diffusion_flow_time_reparameterization_v4"
DEFAULT_OUT_ROOT = project_outputs_root() / "diffusion_flow_time_reparameterization"
DEFAULT_TARGET_NFE_VALUES: Tuple[int, ...] = (4, 8, 12)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_SCHEDULES: Tuple[str, ...] = BASELINE_SCHEDULE_KEYS
SUPPORTED_SPLIT_PHASES: Tuple[str, ...] = (LOCKED_TEST_PHASE, VALIDATION_PHASE, TRAIN_TUNING_PHASE)

ROW_RECORD_FIELDS: Tuple[str, ...] = (
    "benchmark_family",
    "split_phase",
    "seed",
    "dataset",
    "checkpoint_id",
    "checkpoint_path",
    "backbone_name",
    "train_steps",
    "train_budget_label",
    "target_nfe",
    "runtime_nfe",
    "solver_key",
    "solver_name",
    "scheduler_key",
    "scheduler_variant_key",
    "scheduler_variant_name",
    "schedule_name",
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
    "crps",
    "mse",
    "mase",
    "score_main",
    "disc_auc",
    "disc_auc_gap",
    "unconditional_w1",
    "conditional_w1",
    "tstr_macro_f1",
    "u_l1",
    "c_l1",
    "spread_specific_error",
    "imbalance_specific_error",
    "ret_vol_acf_error",
    "impact_response_error",
    "stage_mismatch_rate",
    "stage_classifier_real_macro_f1",
    "sleep_signal_mae",
    "sleep_spectral_mae",
    "sleep_stage_mismatch_rate",
    "sleep_stage_classifier_real_macro_f1",
    "relative_crps_gain_vs_uniform",
    "relative_mase_gain_vs_uniform",
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
    "stage_counts_json",
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
    "train_tuning_train_split_fraction",
    "train_tuning_val_split_fraction",
)

FORECAST_CONTEXT_ROW_FIELDS: Tuple[str, ...] = (
    "benchmark_family",
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
    "realized_nfe",
    "scheduler_key",
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
    "crps",
    "mase",
    "mse",
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


def _parse_schedule_names(text: str) -> List[str]:
    names = [name.strip().lower() for name in parse_csv(text)]
    unknown = [name for name in names if name not in EXPERIMENTAL_FIXED_SCHEDULE_KEYS]
    if unknown:
        raise ValueError(f"Unknown fixed diffusion-flow schedules: {unknown}")
    return names


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


def _protocol_config_fingerprint(cli_args: argparse.Namespace) -> str:
    payload = {
        "runner_signature": RUNNER_SIGNATURE_VERSION,
        "forecast_datasets": parse_forecast_datasets(str(cli_args.forecast_datasets)),
        "conditional_generation_datasets": parse_conditional_generation_datasets(
            str(cli_args.conditional_generation_datasets)
        ),
        "seeds": parse_int_csv(str(cli_args.seeds)),
        "target_nfe_values": parse_int_csv(str(cli_args.target_nfe_values)),
        "solver_names": parse_csv(str(cli_args.solver_names)),
        "baseline_scheduler_names": _parse_schedule_names(str(cli_args.baseline_scheduler_names)),
        "split_phase": str(cli_args.split_phase),
        "otflow_train_steps": int(cli_args.otflow_train_steps),
        "dataset_seed": int(cli_args.dataset_seed),
        "num_eval_samples": int(cli_args.num_eval_samples),
        "forecast_eval_batch_size": int(cli_args.forecast_eval_batch_size),
        "write_forecast_context_rows": bool(getattr(cli_args, "write_forecast_context_rows", False)),
        "context_embedding_kind": str(getattr(cli_args, "context_embedding_kind", "ctx_summary")),
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
        "data_path_fingerprints": _data_path_fingerprints(cli_args),
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
        writer = csv.DictWriter(fh, fieldnames=list(FORECAST_CONTEXT_ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FORECAST_CONTEXT_ROW_FIELDS})


def _load_context_rows(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not csv_path.exists():
        return rows
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            signature = str(row.get("row_signature", "")).strip()
            if signature:
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
    context_csv_path = out_root / str(getattr(cli_args, "forecast_context_row_csv_name", "forecast_context_rows.csv"))
    context_embeddings_path = out_root / str(getattr(cli_args, "forecast_context_embeddings_npz_name", "forecast_context_embeddings.npz"))
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
        "protocol_hash": protocol_hash,
    }


def _append_row_record(row_recorder: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    row_dict = dict(row)
    key = _row_key(row_dict)
    row_recorder["rows_by_key"][key] = row_dict
    row_recorder["fh"].write(json.dumps(row_dict, sort_keys=True) + "\n")
    row_recorder["fh"].flush()
    _write_row_csv(Path(row_recorder["csv_path"]), list(row_recorder["rows_by_key"].values()))


def _append_forecast_context_records(
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
        rows_by_signature[signature] = dict(row)
    row_recorder["context_embeddings"].update({str(key): list(value) for key, value in context_embeddings.items()})
    row_recorder["context_embedding_metadata"].update(dict(metadata))
    _write_context_row_csv(Path(row_recorder["context_csv_path"]), list(rows_by_signature.values()))
    if row_recorder["context_embeddings"]:
        save_context_embedding_table(
            Path(row_recorder["context_embeddings_path"]),
            row_recorder["context_embeddings"],
            metadata=row_recorder["context_embedding_metadata"],
        )


def _existing_complete_row(row_recorder: Mapping[str, Any], row_key: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    row = row_recorder["rows_by_key"].get(row_key)
    if row is not None and str(row.get("row_status")) == "complete":
        return dict(row)
    return None


def _pending_scheduler_cases(row_recorder: Mapping[str, Any], *, benchmark_family: str, split_phase: str, seed: int, dataset: str, checkpoint_id: str, target_nfe: int, solver_key: str, scheduler_cases: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    existing: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for case in scheduler_cases:
        scheduler_key = str(case["scheduler_key"])
        signature = _row_signature(dataset=dataset, split_phase=split_phase, seed=seed, target_nfe=target_nfe, solver_key=solver_key, scheduler_key=scheduler_key, checkpoint_id=checkpoint_id)
        key = (row_recorder["protocol_hash"], benchmark_family, split_phase, int(seed), dataset, int(target_nfe), solver_key, scheduler_key, signature)
        row = _existing_complete_row(row_recorder, key)
        if row is None:
            pending.append(dict(case, row_signature=signature))
        else:
            existing.append(row)
    return existing, pending


def _fixed_schedule_details(scheduler_key: str, runtime_nfe: int) -> Dict[str, Any]:
    fixed_grid = build_schedule_grid(str(scheduler_key), int(runtime_nfe))
    if fixed_grid is None:
        raise ValueError(f"Unable to build fixed grid for scheduler={scheduler_key}")
    schedule_grid_hash = hashlib.sha256(
        json.dumps([float(x) for x in fixed_grid], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    details: Dict[str, Any] = {
        "time_grid": [float(x) for x in fixed_grid],
        "schedule_grid_hash": str(schedule_grid_hash),
        "reference_time_alignment": schedule_time_alignment(str(scheduler_key)),
        "paper_duplicate_count": 0,
        "reference_macro_steps": int(runtime_nfe),
    }
    details.update(fixed_schedule_shape_statistics(fixed_grid))
    return details


def _evaluation_protocol_fields(result_row: Mapping[str, Any], *, eval_horizon: int) -> Dict[str, Any]:
    protocol = dict(result_row.get("evaluation_protocol", {}) or {})
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return {
        "eval_horizon": int(eval_horizon),
        "evaluation_protocol_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "chosen_t0s_hash": str(protocol.get("chosen_t0s_hash", "")),
        "stage_counts_json": json.dumps(dict(protocol.get("stage_counts", {}) or {}), sort_keys=True, separators=(",", ":")),
    }


def _build_row(*, benchmark_family: str, split_phase: str, seed: int, dataset: str, checkpoint: Mapping[str, Any], target_nfe: int, runtime_nfe: int, solver_key: str, scheduler_key: str, details: Mapping[str, Any], metrics: Mapping[str, Any], row_signature: str, protocol_hash: str) -> Dict[str, Any]:
    selection_metric = selection_metric_for_family(str(benchmark_family))
    realized_nfe = metrics.get("realized_nfe")
    if realized_nfe is None:
        realized_nfe = _realized_nfe_for_solver(str(solver_key), int(runtime_nfe))
    return {
        "benchmark_family": str(benchmark_family),
        "split_phase": str(split_phase),
        "seed": int(seed),
        "dataset": str(dataset),
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "checkpoint_path": _logical_artifact_path(str(checkpoint["checkpoint_path"])),
        "backbone_name": str(checkpoint.get("backbone_name", "otflow")),
        "train_steps": int(checkpoint["train_steps"]),
        "train_budget_label": str(checkpoint["train_budget_label"]),
        "target_nfe": int(target_nfe),
        "runtime_nfe": int(runtime_nfe),
        "solver_key": str(solver_key),
        "solver_name": str(SOLVER_RUNTIME_NAMES[str(solver_key)]),
        "scheduler_key": str(scheduler_key),
        "scheduler_variant_key": str(scheduler_key),
        "scheduler_variant_name": schedule_display_name(str(scheduler_key)),
        "schedule_name": schedule_display_name(str(scheduler_key)),
        "row_signature": str(row_signature),
        "signal_trace_key": None,
        "signal_validation_spearman": None,
        "info_growth_scale": None,
        "reference_macro_factor": None,
        "paper_duplicate_count": int(details.get("paper_duplicate_count", 0) or 0),
        "experiment_scope": solver_experiment_scope(str(solver_key)),
        "selection_metric": str(selection_metric),
        "selection_metric_value": metrics.get(selection_metric),
        "reference_macro_steps": int(details.get("reference_macro_steps", runtime_nfe)),
        "reference_time_alignment": str(details.get("reference_time_alignment", schedule_time_alignment(str(scheduler_key)))),
        "runtime_grid_q25": details.get("runtime_grid_q25"),
        "runtime_grid_q50": details.get("runtime_grid_q50"),
        "runtime_grid_q75": details.get("runtime_grid_q75"),
        "crps": metrics.get("crps"),
        "mse": metrics.get("mse"),
        "mase": metrics.get("mase"),
        "score_main": metrics.get("score_main"),
        "disc_auc": metrics.get("disc_auc"),
        "disc_auc_gap": metrics.get("disc_auc_gap"),
        "unconditional_w1": metrics.get("unconditional_w1"),
        "conditional_w1": metrics.get("conditional_w1"),
        "tstr_macro_f1": metrics.get("tstr_macro_f1"),
        "u_l1": metrics.get("u_l1"),
        "c_l1": metrics.get("c_l1"),
        "spread_specific_error": metrics.get("spread_specific_error"),
        "imbalance_specific_error": metrics.get("imbalance_specific_error"),
        "ret_vol_acf_error": metrics.get("ret_vol_acf_error"),
        "impact_response_error": metrics.get("impact_response_error"),
        "stage_mismatch_rate": metrics.get("stage_mismatch_rate"),
        "stage_classifier_real_macro_f1": metrics.get("stage_classifier_real_macro_f1"),
        "sleep_signal_mae": metrics.get("sleep_signal_mae"),
        "sleep_spectral_mae": metrics.get("sleep_spectral_mae"),
        "sleep_stage_mismatch_rate": metrics.get("sleep_stage_mismatch_rate"),
        "sleep_stage_classifier_real_macro_f1": metrics.get("sleep_stage_classifier_real_macro_f1"),
        "relative_crps_gain_vs_uniform": metrics.get("relative_crps_gain_vs_uniform"),
        "relative_mase_gain_vs_uniform": metrics.get("relative_mase_gain_vs_uniform"),
        "relative_score_gain_vs_uniform": metrics.get("relative_score_gain_vs_uniform"),
        "realized_nfe": int(realized_nfe),
        "latency_ms_per_sample": metrics.get("latency_ms_per_sample", metrics.get("efficiency_ms_per_sample")),
        "num_eval_samples": metrics.get("num_eval_samples"),
        "eval_examples": metrics.get("eval_examples"),
        "eval_windows": metrics.get("eval_windows"),
        "eval_horizon": metrics.get("eval_horizon"),
        "evaluation_protocol_hash": metrics.get("evaluation_protocol_hash"),
        "chosen_t0s_hash": metrics.get("chosen_t0s_hash"),
        "chosen_examples_hash": metrics.get("chosen_examples_hash"),
        "stage_counts_json": metrics.get("stage_counts_json"),
        "schedule_grid_hash": details.get("schedule_grid_hash"),
        "protocol_hash": str(protocol_hash),
        "row_status": "complete",
        "train_tuning_fraction": "",
        "train_tuning_seed": "",
        "train_tuning_strata": "",
        "train_tuning_sampler": "",
    }


def _scheduler_cases_for_datasets(cli_args: argparse.Namespace, datasets: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    schedule_names = _parse_schedule_names(str(cli_args.baseline_scheduler_names))
    if UNIFORM_SCHEDULER_KEY in schedule_names:
        schedule_names = [UNIFORM_SCHEDULER_KEY] + [key for key in schedule_names if key != UNIFORM_SCHEDULER_KEY]
    return {str(dataset): [{"scheduler_key": key} for key in schedule_names] for dataset in datasets}


def _run_forecast_phase(cli_args: argparse.Namespace, *, row_recorder: Mapping[str, Any], split_phase: str, seeds: Sequence[int], scheduler_cases_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    dataset_root = resolve_project_path(str(cli_args.dataset_root))
    shared_backbone_root = resolve_project_path(str(cli_args.shared_backbone_root))
    device = resolve_torch_device(str(cli_args.device))
    dataset_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    datasets = parse_forecast_datasets(str(cli_args.forecast_datasets))
    for dataset_idx, dataset in enumerate(datasets):
        if dataset not in dataset_cache:
            dataset_cache[dataset] = load_forecast_checkpoint_splits(cli_args=cli_args, dataset_root=dataset_root, shared_backbone_root=shared_backbone_root, dataset=dataset, device=device)
        checkpoint = dataset_cache[dataset]
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        train_tuning_reference_examples = int(len(splits.get("val", [])))
        if str(split_phase) == TRAIN_TUNING_PHASE:
            eval_ds = splits["train"]
            eval_window_count = 0
        elif str(split_phase) == VALIDATION_PHASE:
            eval_ds = splits["val"]
            eval_window_count = int(cli_args.eval_windows_val)
        else:
            eval_ds = splits["test"]
            eval_window_count = int(cli_args.eval_windows_test)
        for seed in seeds:
            if str(split_phase) == TRAIN_TUNING_PHASE:
                chosen_examples = choose_forecast_train_tuning_indices(
                    eval_ds,
                    fraction=float(cli_args.eval_train_fraction),
                    seed=int(cli_args.train_tuning_seed) + int(seed) + 1_000 * dataset_idx,
                    strata=int(cli_args.train_tuning_strata),
                    dataset=str(dataset),
                    sampling_mode=str(cli_args.train_tuning_sampling_mode),
                    reference_examples=int(train_tuning_reference_examples),
                    train_split_fraction=float(cli_args.train_tuning_train_split_fraction),
                    val_split_fraction=float(cli_args.train_tuning_val_split_fraction),
                )
            else:
                chosen_examples = choose_forecast_example_indices(
                    eval_ds,
                    n_examples=int(eval_window_count),
                    seed=int(seed) + 1_000 * dataset_idx,
                )
            for target_idx, target_nfe in enumerate(parse_int_csv(str(cli_args.target_nfe_values))):
                for solver_idx, solver_key in enumerate(parse_csv(str(cli_args.solver_names))):
                    runtime_nfe = solver_macro_steps(str(solver_key), int(target_nfe))
                    scheduler_cases = list(scheduler_cases_by_dataset[str(dataset)])
                    existing_rows, pending_cases = _pending_scheduler_cases(row_recorder, benchmark_family=FORECAST_FAMILY, split_phase=str(split_phase), seed=int(seed), dataset=str(dataset), checkpoint_id=str(checkpoint["checkpoint_id"]), target_nfe=int(target_nfe), solver_key=str(solver_key), scheduler_cases=scheduler_cases)
                    rows.extend(existing_rows)
                    cell_uniform_metrics: Optional[Mapping[str, Any]] = None
                    for existing_row in existing_rows:
                        if str(existing_row.get("scheduler_key")) == UNIFORM_SCHEDULER_KEY:
                            cell_uniform_metrics = existing_row
                    for case in pending_cases:
                        scheduler_key = str(case["scheduler_key"])
                        details = _fixed_schedule_details(scheduler_key, int(runtime_nfe))
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
                            progress_label=f"{split_phase} {dataset} {scheduler_key} seed={seed} {solver_key}/{target_nfe}",
                            return_per_example_rows=bool(getattr(cli_args, "write_forecast_context_rows", False)),
                            return_context_embeddings=bool(getattr(cli_args, "write_forecast_context_rows", False)),
                            context_embedding_kind=str(getattr(cli_args, "context_embedding_kind", "ctx_summary")),
                        )
                        if scheduler_key != UNIFORM_SCHEDULER_KEY and cell_uniform_metrics is not None:
                            metrics = dict(metrics)
                            metrics["relative_crps_gain_vs_uniform"] = _safe_relative_gain(metrics.get("crps"), cell_uniform_metrics.get("crps"))
                            metrics["relative_mase_gain_vs_uniform"] = _safe_relative_gain(metrics.get("mase"), cell_uniform_metrics.get("mase"))
                        row = _build_row(benchmark_family=FORECAST_FAMILY, split_phase=str(split_phase), seed=int(seed), dataset=str(dataset), checkpoint=checkpoint, target_nfe=int(target_nfe), runtime_nfe=int(runtime_nfe), solver_key=str(solver_key), scheduler_key=scheduler_key, details=details, metrics=metrics, row_signature=str(case["row_signature"]), protocol_hash=str(row_recorder["protocol_hash"]))
                        if str(split_phase) == TRAIN_TUNING_PHASE:
                            row.update(
                                {
                                    "train_tuning_fraction": float(cli_args.eval_train_fraction),
                                    "train_tuning_seed": int(cli_args.train_tuning_seed) + int(seed) + 1_000 * dataset_idx,
                                    "train_tuning_strata": int(cli_args.train_tuning_strata),
                                    "train_tuning_sampler": train_tuning_sampler_key(str(cli_args.train_tuning_sampling_mode)),
                                    "train_tuning_sampling_mode": str(cli_args.train_tuning_sampling_mode),
                                    "train_tuning_reference_examples": int(train_tuning_reference_examples),
                                    "train_tuning_target_examples": int(len(chosen_examples)),
                                    "train_tuning_train_split_fraction": float(cli_args.train_tuning_train_split_fraction),
                                    "train_tuning_val_split_fraction": float(cli_args.train_tuning_val_split_fraction),
                                }
                            )
                        _append_row_record(row_recorder, row)
                        if bool(getattr(cli_args, "write_forecast_context_rows", False)):
                            context_rows = []
                            for detail_row in list(metrics.get("per_example_rows", []) or []):
                                copied_detail = dict(detail_row)
                                copied_detail.update(
                                    {
                                        "benchmark_family": FORECAST_FAMILY,
                                        "parent_row_signature": str(case["row_signature"]),
                                        "protocol_hash": str(row_recorder["protocol_hash"]),
                                    }
                                )
                                if str(split_phase) == TRAIN_TUNING_PHASE:
                                    copied_detail.update(
                                        {
                                            "train_tuning_fraction": float(cli_args.eval_train_fraction),
                                            "train_tuning_seed": int(cli_args.train_tuning_seed) + int(seed) + 1_000 * dataset_idx,
                                            "train_tuning_strata": int(cli_args.train_tuning_strata),
                                            "train_tuning_sampler": train_tuning_sampler_key(str(cli_args.train_tuning_sampling_mode)),
                                        }
                                    )
                                context_rows.append(copied_detail)
                            _append_forecast_context_records(
                                row_recorder,
                                context_rows,
                                context_embeddings=dict(metrics.get("context_embeddings", {}) or {}),
                                metadata={
                                    "checkpoint_id": str(checkpoint["checkpoint_id"]),
                                    "dataset": str(dataset),
                                    "split_phase": str(split_phase),
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
    if str(split_phase) == TRAIN_TUNING_PHASE:
        raise ValueError("train_tuning split is only supported for forecast schedule evaluation.")
    shared_backbone_root = resolve_project_path(str(cli_args.shared_backbone_root))
    device = resolve_torch_device(str(cli_args.device))
    dataset_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    datasets = parse_conditional_generation_datasets(str(cli_args.conditional_generation_datasets))
    for dataset_idx, dataset in enumerate(datasets):
        if dataset not in dataset_cache:
            dataset_cache[dataset] = load_conditional_generation_checkpoint_splits(cli_args=cli_args, shared_backbone_root=shared_backbone_root, dataset=dataset, device=device)
        checkpoint = dataset_cache[dataset]
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        eval_ds = splits["val"] if str(split_phase) == VALIDATION_PHASE else splits["test"]
        eval_horizon = resolved_eval_horizon(cli_args, str(dataset))
        eval_windows = resolved_eval_windows(cli_args, str(dataset), "val" if str(split_phase) == VALIDATION_PHASE else "test")
        for seed in seeds:
            chosen_eval_t0s = np.asarray(_choose_valid_windows(eval_ds, horizon=int(eval_horizon), n_windows=int(eval_windows), seed=int(seed) + 1_000 * dataset_idx), dtype=np.int64)
            for target_idx, target_nfe in enumerate(parse_int_csv(str(cli_args.target_nfe_values))):
                for solver_idx, solver_key in enumerate(parse_csv(str(cli_args.solver_names))):
                    runtime_nfe = solver_macro_steps(str(solver_key), int(target_nfe))
                    existing_rows, pending_cases = _pending_scheduler_cases(row_recorder, benchmark_family=CONDITIONAL_GENERATION_FAMILY, split_phase=str(split_phase), seed=int(seed), dataset=str(dataset), checkpoint_id=str(checkpoint["checkpoint_id"]), target_nfe=int(target_nfe), solver_key=str(solver_key), scheduler_cases=list(scheduler_cases_by_dataset[str(dataset)]))
                    rows.extend(existing_rows)
                    cell_uniform_metrics: Optional[Mapping[str, Any]] = None
                    for existing_row in existing_rows:
                        if str(existing_row.get("scheduler_key")) == UNIFORM_SCHEDULER_KEY:
                            cell_uniform_metrics = existing_row
                    for case in pending_cases:
                        scheduler_key = str(case["scheduler_key"])
                        details = _fixed_schedule_details(scheduler_key, int(runtime_nfe))
                        grid_spec = {"grid_name": scheduler_key, "grid_kind": "fixed_diffusion_flow_time_grid", "selection_group": scheduler_key, "comparison_role": "transferred" if scheduler_key in TRANSFER_SCHEDULE_KEYS else "baseline", "solver_name": str(SOLVER_RUNTIME_NAMES[str(solver_key)]), "nfe": int(runtime_nfe), "time_grid": details["time_grid"]}
                        metrics_seed = int(seed) + 1_000_000 * dataset_idx + 10_000 * target_idx + solver_idx
                        result_row = run_fixed_schedule_variant(model=model, ds=eval_ds, cfg=cfg, eval_horizon=int(eval_horizon), eval_windows=int(len(chosen_eval_t0s)), grid_spec=grid_spec, chosen_t0s=chosen_eval_t0s, generation_seed_base=int(metrics_seed), metrics_seed=int(metrics_seed), score_main_only=False)
                        metrics = {
                            "score_main": result_row.get("score_main"),
                            "tstr_macro_f1": result_row.get("tstr_macro_f1"),
                            "disc_auc": result_row.get("disc_auc"),
                            "disc_auc_gap": result_row.get("disc_auc_gap"),
                            "unconditional_w1": result_row.get("unconditional_w1"),
                            "conditional_w1": result_row.get("conditional_w1"),
                            "u_l1": result_row.get("u_l1"),
                            "c_l1": result_row.get("c_l1"),
                            "spread_specific_error": result_row.get("spread_specific_error"),
                            "imbalance_specific_error": result_row.get("imbalance_specific_error"),
                            "ret_vol_acf_error": result_row.get("ret_vol_acf_error"),
                            "impact_response_error": result_row.get("impact_response_error"),
                            "stage_mismatch_rate": result_row.get("stage_mismatch_rate"),
                            "stage_classifier_real_macro_f1": result_row.get("stage_classifier_real_macro_f1"),
                            "sleep_signal_mae": result_row.get("spread_specific_error") if str(dataset) == "sleep_edf" else None,
                            "sleep_spectral_mae": result_row.get("imbalance_specific_error") if str(dataset) == "sleep_edf" else None,
                            "sleep_stage_mismatch_rate": result_row.get("stage_mismatch_rate") if str(dataset) == "sleep_edf" else None,
                            "sleep_stage_classifier_real_macro_f1": result_row.get("stage_classifier_real_macro_f1") if str(dataset) == "sleep_edf" else None,
                            "efficiency_ms_per_sample": result_row.get("efficiency_ms_per_sample"),
                            "eval_windows": int(len(chosen_eval_t0s)),
                            "realized_nfe": _realized_nfe_for_solver(str(solver_key), int(runtime_nfe)),
                            **_evaluation_protocol_fields(result_row, eval_horizon=int(eval_horizon)),
                        }
                        if scheduler_key != UNIFORM_SCHEDULER_KEY and cell_uniform_metrics is not None:
                            metrics["relative_score_gain_vs_uniform"] = _safe_relative_gain(metrics.get("score_main"), cell_uniform_metrics.get("score_main"))
                        row = _build_row(benchmark_family=CONDITIONAL_GENERATION_FAMILY, split_phase=str(split_phase), seed=int(seed), dataset=str(dataset), checkpoint=checkpoint, target_nfe=int(target_nfe), runtime_nfe=int(runtime_nfe), solver_key=str(solver_key), scheduler_key=scheduler_key, details=details, metrics=metrics, row_signature=str(case["row_signature"]), protocol_hash=str(row_recorder["protocol_hash"]))
                        _append_row_record(row_recorder, row)
                        rows.append(row)
                        if scheduler_key == UNIFORM_SCHEDULER_KEY:
                            cell_uniform_metrics = row
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
        key = (row.get("benchmark_family"), row.get("dataset"), row.get("target_nfe"), row.get("solver_key"), row.get("scheduler_key"), row.get("train_budget_label"))
        groups.setdefault(key, []).append(row)
    summaries: List[Dict[str, Any]] = []
    metric_names = (
        "crps",
        "mse",
        "mase",
        "score_main",
        "tstr_macro_f1",
        "disc_auc",
        "disc_auc_gap",
        "unconditional_w1",
        "conditional_w1",
        "u_l1",
        "c_l1",
        "spread_specific_error",
        "imbalance_specific_error",
        "ret_vol_acf_error",
        "impact_response_error",
        "stage_mismatch_rate",
        "stage_classifier_real_macro_f1",
        "sleep_signal_mae",
        "sleep_spectral_mae",
        "sleep_stage_mismatch_rate",
        "sleep_stage_classifier_real_macro_f1",
        "relative_crps_gain_vs_uniform",
        "relative_mase_gain_vs_uniform",
        "relative_score_gain_vs_uniform",
        "realized_nfe",
        "latency_ms_per_sample",
    )
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        family, dataset, target_nfe, solver_key, scheduler_key, budget = key
        summary: Dict[str, Any] = {"benchmark_family": family, "dataset": dataset, "target_nfe": int(target_nfe), "solver_key": solver_key, "scheduler_key": scheduler_key, "schedule_name": schedule_display_name(str(scheduler_key)), "train_budget_label": budget, "n_seeds": int(len(group)), "seed_values": sorted(int(row.get("seed", 0)) for row in group)}
        for metric in metric_names:
            vals = [_optional_float(row.get(metric)) for row in group]
            vals = [float(v) for v in vals if v is not None]
            summary[f"{metric}_mean"] = _mean(vals)
            summary[f"{metric}_std"] = _std(vals)
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
    solvers = parse_csv(str(cli_args.solver_names))
    nfes = parse_int_csv(str(cli_args.target_nfe_values))
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
        "method_key": METHOD_KEY,
        "baseline_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "experimental_fixed_schedule_keys": list(EXPERIMENTAL_FIXED_SCHEDULE_KEYS),
        "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS),
        "scheduled_evaluation_keys": schedules,
        "solver_names": solvers,
        "target_nfe_values": nfes,
        "forecast_datasets": parse_forecast_datasets(str(cli_args.forecast_datasets)),
        "conditional_generation_datasets": parse_conditional_generation_datasets(
            str(cli_args.conditional_generation_datasets)
        ),
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
    ap.add_argument("--otflow_train_steps", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--forecast_datasets", type=str, default=",".join(DEFAULT_FORECAST_DATASETS))
    ap.add_argument(
        "--conditional_generation_datasets",
        type=str,
        default=",".join(DEFAULT_CONDITIONAL_GENERATION_DATASETS),
    )
    ap.add_argument("--cryptos_path", type=str, default="")
    ap.add_argument("--lobster_synthetic_profile_path", type=str, default="")
    ap.add_argument("--long_term_st_path", type=str, default="")
    ap.add_argument("--solver_names", type=str, default=",".join(ALL_SOLVER_ORDER))
    ap.add_argument("--target_nfe_values", type=str, default=",".join(str(x) for x in DEFAULT_TARGET_NFE_VALUES))
    ap.add_argument("--baseline_scheduler_names", type=str, default=",".join(DEFAULT_SCHEDULES))
    ap.add_argument("--seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--split_phase", type=str, choices=SUPPORTED_SPLIT_PHASES, default=LOCKED_TEST_PHASE)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--num_eval_samples", type=int, default=5)
    ap.add_argument("--forecast_eval_batch_size", type=int, default=64)
    ap.add_argument("--write_forecast_context_rows", action="store_true", default=False)
    ap.add_argument("--forecast_context_row_csv_name", type=str, default="forecast_context_rows.csv")
    ap.add_argument("--forecast_context_embeddings_npz_name", type=str, default="forecast_context_embeddings.npz")
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
    scheduler_cases = _scheduler_cases_for_datasets(
        cli_args,
        list(forecast_datasets) + list(conditional_generation_datasets),
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
        "baseline_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "experimental_fixed_schedule_keys": list(EXPERIMENTAL_FIXED_SCHEDULE_KEYS),
        "transfer_schedule_keys": list(TRANSFER_SCHEDULE_KEYS),
        "scheduled_evaluation_keys": _parse_schedule_names(str(cli_args.baseline_scheduler_names)),
    }
    save_json(dict(schedule_selection), str(out_root / "schedule_selection_summary.json"))
    combined = {"prep": dict(prep_payload), "schedule_selection_summary": dict(schedule_selection), seed_summary_key: dict(seed_summary_payload), "main_table_summary": dict(main_table_summary)}
    save_json(dict(combined), str(out_root / "combined_summary.json"))
    return combined


def main() -> None:
    run_diffusion_flow_time_reparameterization(build_argparser().parse_args())


if __name__ == "__main__":
    main()
