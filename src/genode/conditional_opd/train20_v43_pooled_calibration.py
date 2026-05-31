from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from genode.conditional_opd.candidate_pool import _prediction_from_grid
from genode.conditional_opd.evaluate_schedule_summary import (
    SELECTED_STUDENT_SCHEDULE_KEY,
    SELECTED_STUDENT_SCHEDULE_NAME,
    SCHEDULE_ROW_FIELDS,
    _schedule_row,
    load_schedule_predictions,
)
from genode.conditional_opd.models import (
    ScheduleStudentMLP,
    ScheduleTeacherMLP,
    count_parameters,
    grid_to_intervals,
    setting_features,
    solver_macro_steps,
    validate_time_grid,
)
from genode.conditional_opd.objectives import (
    attach_reward_columns,
    build_fixed_reference_table,
    rewards_by_setting,
    seed_mean_metric_rows,
)
from genode.conditional_opd.ser_ptg_reference import SER_PTG_SCHEDULE_KEY, grid_geometry
from genode.conditional_opd.train_conditional_opd import (
    DEFAULT_LATE_BIASED_DEMO_SCHEDULES,
    DEFAULT_LATE_BIASED_DEMO_WEIGHT,
    DEFAULT_SOLVERS,
    DEFAULT_STUDENT_SER_PTG_REGULARIZATION_EPS,
    DEFAULT_TARGET_NFES,
    DEFAULT_TEACHER_DIAGNOSTIC_HOLDOUT_FRACTION,
    DEFAULT_TEACHER_PAIR_MARGIN,
    DEFAULT_TEACHER_PAIRS_PER_CANDIDATE,
    DEFAULT_TEACHER_RANK_TEMPERATURE,
    DEFAULT_TEACHER_REGRESSION_WEIGHT,
    STUDENT_SER_PTG_REGULARIZER_NONE,
    STUDENT_SER_PTG_REGULARIZERS,
    _assert_complete_seed_rows,
    _candidate_table,
    _grid_for_schedule,
    _load_csv_rows,
    _load_schedule_summary_grids,
    _load_schedule_summary_grids_many,
    _parse_csv,
    _parse_int_csv,
    _schedule_keys_from_summary_paths,
    differentiable_teacher_features,
    fit_student_to_reference,
    optimize_student_with_teacher,
    split_teacher_diagnostic_holdout,
    teacher_schedule_weights,
    train_teacher,
)
from genode.data.otflow_paths import (
    default_backbone_manifest_path,
    project_outputs_root,
    project_paper_dataset_root,
    project_root,
    resolve_project_path,
)
from genode.evaluation.otflow_evaluation_support import (
    DEFAULT_SHARED_BACKBONE_ROOT,
    DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION,
    DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION,
    LOCKED_TEST_PHASE,
    SOLVER_RUNTIME_NAMES,
    TRAIN_TUNING_PHASE,
    TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
    TRAIN_TUNING_SAMPLING_MODES,
    choose_forecast_train_tuning_indices,
    evaluate_forecast_schedule,
    load_forecast_checkpoint_splits,
    train_tuning_sampler_key,
)
from genode.models.otflow_train_val import save_json, seed_all
from genode.runtime import ProgressBar, resolve_torch_device
from genode.schedule_transfer.diffusion_flow_schedules import (
    BASELINE_SCHEDULE_KEYS,
    build_schedule_grid,
    schedule_display_name,
)


POOLED_CALIBRATION_PHASE = "pooled_calibration"
DEFAULT_OUT_DIR = project_outputs_root() / "train20_v43_pooled_one_round_calibration"
DEFAULT_V42_F_OUT_DIR = project_outputs_root() / "train20_v42_f_calibration20_fullval_uniform_seedmean"
DEFAULT_BASELINE_TEST_ROWS = project_outputs_root() / "diffusion_flow_time_reparameterization_full_test" / "rows.csv"
DEFAULT_STUDENT_BUDGETS: Tuple[int, ...] = (5, 10, 15, 20, 25)
DEFAULT_CALIBRATION_SEEDS: Tuple[int, ...] = (0, 1)
DEFAULT_STUDENT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_LOCKED_TEST_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_GEOMETRY_MAX_INTERNAL_AFTER_098 = 0.5
DEFAULT_GEOMETRY_MIN_INTERVAL = 1e-4
DEFAULT_GEOMETRY_MAX_INTERVAL = 0.97

POOLED_ROW_FIELDS: Tuple[str, ...] = tuple(
    dict.fromkeys(
        [
            *SCHEDULE_ROW_FIELDS,
            "calibration_origin_counts_json",
            "calibration_origin_hashes_json",
            "calibration_origin_weighting",
            "calibration_pool",
        ]
    )
)


def _csv_text(values: Sequence[Any]) -> str:
    return ",".join(str(value) for value in values)


def _python_module_command(module: str, args: Sequence[str]) -> List[str]:
    return [sys.executable, "-m", module, *list(args)]


def _run_command(command: Sequence[str], *, cwd: Path, allow_execute: bool, commands: List[List[str]]) -> None:
    commands.append(list(command))
    if allow_execute:
        print(f"Running: {' '.join(str(part) for part in command)}", file=sys.stderr, flush=True)
        subprocess.run(list(command), cwd=str(cwd), check=True)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _weighted_mean(metrics: Sequence[Mapping[str, Any]], key: str) -> float:
    total_weight = 0
    weighted = 0.0
    for item in metrics:
        weight = int(item.get("eval_examples", 0) or 0)
        value = float(item[key])
        if not math.isfinite(value):
            raise ValueError(f"Cannot pool non-finite {key}={value!r}.")
        total_weight += int(weight)
        weighted += float(weight) * value
    if total_weight <= 0:
        raise ValueError("Pooled calibration metrics require at least one evaluated example.")
    return float(weighted / float(total_weight))


def combine_origin_metrics(train_metrics: Mapping[str, Any], val_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Combine Train20 and former validation metrics into one pooled row."""
    origins = [
        {"origin": "train20", "metrics": dict(train_metrics)},
        {"origin": "former_val", "metrics": dict(val_metrics)},
    ]
    metrics = [item["metrics"] for item in origins]
    origin_counts = {item["origin"]: int(item["metrics"].get("eval_examples", 0) or 0) for item in origins}
    origin_hashes = {item["origin"]: str(item["metrics"].get("chosen_examples_hash", "")) for item in origins}
    total_examples = int(sum(origin_counts.values()))
    payload = {
        "pool": POOLED_CALIBRATION_PHASE,
        "origin_counts": origin_counts,
        "origin_hashes": origin_hashes,
        "origin_protocol_hashes": {
            item["origin"]: str(item["metrics"].get("evaluation_protocol_hash", "")) for item in origins
        },
    }
    return {
        "crps": _weighted_mean(metrics, "crps"),
        "mse": _weighted_mean(metrics, "mse"),
        "mase": _weighted_mean(metrics, "mase"),
        "latency_ms_per_sample": _weighted_mean(metrics, "latency_ms_per_sample"),
        "eval_examples": int(total_examples),
        "eval_horizon": int(metrics[0].get("eval_horizon", 1)),
        "num_eval_samples": int(metrics[0].get("num_eval_samples", 1)),
        "realized_nfe": int(metrics[0].get("realized_nfe", 0)),
        "chosen_examples_hash": _hash_payload({"origin_hashes": origin_hashes}),
        "evaluation_protocol_hash": _hash_payload(payload),
        "calibration_origin_counts": origin_counts,
        "calibration_origin_hashes": origin_hashes,
    }


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], *, fields: Sequence[str] = POOLED_ROW_FIELDS) -> None:
    out = resolve_project_path(str(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_fixed_schedule_summary(args: argparse.Namespace) -> Dict[str, Any]:
    solvers = tuple(_parse_csv(args.solver_names))
    target_nfes = tuple(_parse_int_csv(args.target_nfe_values))
    max_macro_steps = max(solver_macro_steps(solver, nfe) for solver in solvers for nfe in target_nfes)
    schedules: List[Dict[str, Any]] = []
    with ProgressBar(len(BASELINE_SCHEDULE_KEYS), "Fixed schedule preparation") as progress:
        for schedule_key in BASELINE_SCHEDULE_KEYS:
            predictions: List[Dict[str, Any]] = []
            for solver in solvers:
                for target_nfe in target_nfes:
                    macro_steps = solver_macro_steps(str(solver), int(target_nfe))
                    grid = build_schedule_grid(str(schedule_key), int(macro_steps))
                    if grid is None:
                        raise ValueError(f"No fixed schedule grid for {schedule_key!r}.")
                    source = {
                        "solver_key": str(solver),
                        "target_nfe": int(target_nfe),
                        "runtime_nfe": int(macro_steps),
                        "macro_steps": int(macro_steps),
                        "max_macro_steps": int(max_macro_steps),
                        "candidate_source": "fixed_reference",
                    }
                    predictions.append(
                        _prediction_from_grid(
                            source=source,
                            grid=grid,
                            scheduler_key=str(schedule_key),
                            candidate_source="fixed_reference",
                            active_round=-1,
                            perturbation_type="fixed_baseline",
                            perturbation_params={"schedule_key": str(schedule_key)},
                        )
                    )
            schedules.append(
                {
                    "scheduler_key": str(schedule_key),
                    "schedule_name": schedule_display_name(str(schedule_key)),
                    "comparison_role": "v43_pooled_fixed_reward_reference",
                    "candidate_source": "fixed_reference",
                    "active_round": -1,
                    "predictions": predictions,
                }
            )
            progress.update()
    summary = {
        "status": "ready",
        "artifact": "v43_fixed_reference_schedule_summary",
        "protocol": "v4.3_pooled_one_round_calibration",
        "dataset": str(args.dataset),
        "baseline_schedule": True,
        "fixed_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "schedules": schedules,
    }
    out_path = resolve_project_path(str(args.fixed_schedule_summary))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, str(out_path))
    return {"status": "ready", "fixed_schedule_summary": str(out_path), "schedule_count": int(len(schedules))}


def evaluate_pooled_schedule_summary(args: argparse.Namespace) -> Dict[str, Any]:
    solvers = tuple(_parse_csv(args.solver_names))
    target_nfes = tuple(_parse_int_csv(args.target_nfe_values))
    seeds = tuple(_parse_int_csv(args.calibration_seeds))
    schedule_predictions = load_schedule_predictions(
        args.schedule_summary,
        dataset=str(args.dataset),
        solver_names=solvers,
        target_nfe_values=target_nfes,
        require_complete=True,
    )
    schedule_keys = sorted({key[0] for key in schedule_predictions})
    device = resolve_torch_device(str(args.device))
    dataset_root = resolve_project_path(str(args.dataset_root))
    shared_backbone_root = resolve_project_path(str(args.shared_backbone_root))
    checkpoint = load_forecast_checkpoint_splits(
        cli_args=args,
        dataset_root=dataset_root,
        shared_backbone_root=shared_backbone_root,
        dataset=str(args.dataset),
        device=device,
    )
    model = checkpoint["model"]
    cfg = checkpoint["cfg"]
    splits = checkpoint["splits"]
    train_ds = splits["train"]
    val_ds = splits["val"]
    train_reference_examples = int(len(val_ds))
    rows: List[Dict[str, Any]] = []
    total_cells = len(seeds) * len(schedule_keys) * len(solvers) * len(target_nfes)
    with ProgressBar(total_cells, "Pooled inference cells") as progress:
        for seed in seeds:
            train_indices = choose_forecast_train_tuning_indices(
                train_ds,
                fraction=float(args.eval_train_fraction),
                seed=int(args.train_tuning_seed) + int(seed),
                strata=int(args.train_tuning_strata),
                dataset=str(args.dataset),
                sampling_mode=str(args.train_tuning_sampling_mode),
                reference_examples=train_reference_examples,
                train_split_fraction=float(args.train_tuning_train_split_fraction),
                val_split_fraction=float(args.train_tuning_val_split_fraction),
            )
            val_indices = np.arange(int(len(val_ds)), dtype=np.int64)
            for schedule_key in schedule_keys:
                for solver in solvers:
                    for target_nfe in target_nfes:
                        prediction = schedule_predictions[(schedule_key, str(solver), int(target_nfe))]
                        label_base = f"{schedule_key} seed={seed} {solver}/{target_nfe}"
                        train_metrics = evaluate_forecast_schedule(
                            model,
                            train_ds,
                            cfg,
                            solver_name=SOLVER_RUNTIME_NAMES[str(solver)],
                            runtime_nfe=int(prediction["runtime_nfe"]),
                            time_grid=prediction["time_grid"],
                            num_eval_samples=int(args.num_eval_samples),
                            seed=int(seed),
                            example_indices=train_indices,
                            batch_size=int(args.forecast_eval_batch_size),
                            progress_label=f"{label_base} train20",
                        )
                        val_metrics = evaluate_forecast_schedule(
                            model,
                            val_ds,
                            cfg,
                            solver_name=SOLVER_RUNTIME_NAMES[str(solver)],
                            runtime_nfe=int(prediction["runtime_nfe"]),
                            time_grid=prediction["time_grid"],
                            num_eval_samples=int(args.num_eval_samples),
                            seed=int(seed),
                            example_indices=val_indices,
                            batch_size=int(args.forecast_eval_batch_size),
                            progress_label=f"{label_base} former_val",
                        )
                        pooled = combine_origin_metrics(train_metrics, val_metrics)
                        row = _schedule_row(
                            seed=int(seed),
                            dataset=str(args.dataset),
                            split_phase=POOLED_CALIBRATION_PHASE,
                            checkpoint=checkpoint,
                            prediction=prediction,
                            metrics=pooled,
                            protocol_hash=_hash_payload(
                                {
                                    "phase": POOLED_CALIBRATION_PHASE,
                                    "schedule_summary": str(resolve_project_path(str(args.schedule_summary))),
                                    "seed": int(seed),
                                    "solver": str(solver),
                                    "target_nfe": int(target_nfe),
                                }
                            ),
                        )
                        row.update(
                            {
                                "train_tuning_fraction": float(args.eval_train_fraction),
                                "train_tuning_seed": int(args.train_tuning_seed) + int(seed),
                                "train_tuning_strata": int(args.train_tuning_strata),
                                "train_tuning_sampler": train_tuning_sampler_key(str(args.train_tuning_sampling_mode)),
                                "train_tuning_sampling_mode": str(args.train_tuning_sampling_mode),
                                "train_tuning_reference_examples": int(train_reference_examples),
                                "train_tuning_target_examples": int(len(train_indices)),
                                "train_tuning_train_split_fraction": float(args.train_tuning_train_split_fraction),
                                "train_tuning_val_split_fraction": float(args.train_tuning_val_split_fraction),
                                "calibration_origin_counts_json": json.dumps(pooled["calibration_origin_counts"], sort_keys=True, separators=(",", ":")),
                                "calibration_origin_hashes_json": json.dumps(pooled["calibration_origin_hashes"], sort_keys=True, separators=(",", ":")),
                                "calibration_origin_weighting": "example_count_weighted",
                                "calibration_pool": "train20_plus_full_former_val",
                            }
                        )
                        rows.append(row)
                        progress.update()
    out_dir = resolve_project_path(str(args.out_dir))
    row_csv = out_dir / str(args.row_csv_name)
    _write_csv(row_csv, rows)
    summary = {
        "status": "ready",
        "artifact": "v43_pooled_calibration_rows",
        "protocol": "v4.3_pooled_one_round_calibration",
        "dataset": str(args.dataset),
        "schedule_summary": str(resolve_project_path(str(args.schedule_summary))),
        "row_csv": str(row_csv),
        "split_phase": POOLED_CALIBRATION_PHASE,
        "schedule_keys": schedule_keys,
        "seeds": list(seeds),
        "origin_policy": "Train20 sampled examples plus all former validation examples, re-evaluated together",
        "row_count": int(len(rows)),
    }
    save_json(summary, str(out_dir / f"{Path(str(args.row_csv_name)).stem}_summary.json"))
    return summary


def _clean_pooled_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    solvers: Sequence[str],
    target_nfes: Sequence[int],
    allowed_schedules: Sequence[str],
    seeds: Sequence[int],
    label: str,
) -> List[Dict[str, Any]]:
    solver_set = {str(solver) for solver in solvers}
    nfe_set = {int(nfe) for nfe in target_nfes}
    schedule_set = {str(key) for key in allowed_schedules}
    seed_set = {int(seed) for seed in seeds}
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("split_phase")) != POOLED_CALIBRATION_PHASE:
            continue
        if str(row.get("dataset")) != str(dataset):
            continue
        try:
            seed = int(row.get("seed", -1))
            target_nfe = int(row.get("target_nfe", -1))
            crps = float(row["crps"])
            mase = float(row["mase"])
        except (KeyError, TypeError, ValueError):
            continue
        if seed not in seed_set or target_nfe not in nfe_set:
            continue
        if str(row.get("solver_key")) not in solver_set or str(row.get("scheduler_key")) not in schedule_set:
            continue
        if not (math.isfinite(crps) and math.isfinite(mase) and crps > 0.0 and mase > 0.0):
            continue
        clean = dict(row)
        clean["seed"] = int(seed)
        clean["target_nfe"] = int(target_nfe)
        clean["crps"] = float(crps)
        clean["mase"] = float(mase)
        out.append(clean)
    if not out:
        raise ValueError(f"{label} produced no usable pooled calibration rows.")
    return out


def _prediction_geometry(prediction: Mapping[str, Any]) -> Dict[str, float | None]:
    geometry = prediction.get("grid_geometry")
    if isinstance(geometry, Mapping):
        try:
            return {
                "internal_fraction_after_098": float(geometry["internal_fraction_after_098"]),
                "min_interval": float(geometry["min_interval"]),
                "max_interval": float(geometry["max_interval"]),
            }
        except (KeyError, TypeError, ValueError):
            pass
    try:
        grid = [float(value) for value in prediction.get("time_grid", [])]
    except (TypeError, ValueError):
        grid = []
    if len(grid) < 2:
        return {"internal_fraction_after_098": None, "min_interval": None, "max_interval": None}
    intervals = [b - a for a, b in zip(grid[:-1], grid[1:])]
    internal = grid[1:-1]
    return {
        "internal_fraction_after_098": float(sum(1 for value in internal if value > 0.98) / max(1, len(internal))),
        "min_interval": float(min(intervals)),
        "max_interval": float(max(intervals)),
    }


def geometry_guard_for_schedule(
    schedule: Mapping[str, Any],
    *,
    max_internal_after_098: float = DEFAULT_GEOMETRY_MAX_INTERNAL_AFTER_098,
    min_interval_floor: float = DEFAULT_GEOMETRY_MIN_INTERVAL,
    max_interval_ceiling: float = DEFAULT_GEOMETRY_MAX_INTERVAL,
) -> Dict[str, Any]:
    predictions = list(schedule.get("predictions", []) or [])
    if not predictions:
        return {"passes_geometry_guard": False, "geometry_risk_flags": ["missing_predictions"]}
    late_values: List[float] = []
    min_values: List[float] = []
    max_values: List[float] = []
    missing: List[str] = []
    for prediction in predictions:
        geom = _prediction_geometry(prediction)
        for key, target in (
            ("internal_fraction_after_098", late_values),
            ("min_interval", min_values),
            ("max_interval", max_values),
        ):
            value = geom.get(key)
            if value is None or not math.isfinite(float(value)):
                missing.append(key)
            else:
                target.append(float(value))
    thresholds = {
        "max_internal_fraction_after_098": float(max_internal_after_098),
        "min_interval": float(min_interval_floor),
        "max_interval": float(max_interval_ceiling),
    }
    if missing:
        return {
            "passes_geometry_guard": False,
            "geometry_risk_flags": ["missing_geometry_metrics"],
            "missing_geometry_fields": sorted(set(missing)),
            "thresholds": thresholds,
        }
    flags: List[str] = []
    max_late = max(late_values)
    min_interval = min(min_values)
    max_interval = max(max_values)
    if max_late > float(max_internal_after_098):
        flags.append("late_internal_fraction")
    if min_interval < float(min_interval_floor):
        flags.append("tiny_interval")
    if max_interval > float(max_interval_ceiling):
        flags.append("oversized_interval")
    return {
        "passes_geometry_guard": not flags,
        "geometry_risk_flags": flags,
        "max_internal_fraction_after_098": float(max_late),
        "min_interval": float(min_interval),
        "max_interval": float(max_interval),
        "thresholds": thresholds,
    }


def select_guarded_teacher_utility_schedule(
    schedule_summary: Mapping[str, Any],
    *,
    max_internal_after_098: float = DEFAULT_GEOMETRY_MAX_INTERNAL_AFTER_098,
    min_interval_floor: float = DEFAULT_GEOMETRY_MIN_INTERVAL,
    max_interval_ceiling: float = DEFAULT_GEOMETRY_MAX_INTERVAL,
) -> Dict[str, Any]:
    table: List[Dict[str, Any]] = []
    for schedule in list(schedule_summary.get("schedules", []) or []):
        utilities = [float(item["utility"]) for item in list(schedule.get("predictions", []) or []) if item.get("utility") not in (None, "")]
        if not utilities:
            raise ValueError(f"Schedule {schedule.get('scheduler_key')} is missing teacher utility predictions.")
        score = schedule.get("teacher_predicted_utility_mean")
        score = float(np.mean(np.asarray(utilities, dtype=np.float64))) if score in (None, "") else float(score)
        row = {
            "scheduler_key": str(schedule["scheduler_key"]),
            "opd_step_budget": schedule.get("opd_step_budget"),
            "teacher_predicted_utility_mean": float(score),
            "min_teacher_predicted_utility": float(min(utilities)),
            "selection_score": float(score),
        }
        row.update(
            geometry_guard_for_schedule(
                schedule,
                max_internal_after_098=max_internal_after_098,
                min_interval_floor=min_interval_floor,
                max_interval_ceiling=max_interval_ceiling,
            )
        )
        table.append(row)
    table.sort(
        key=lambda row: (
            float(row["selection_score"]),
            float(row["min_teacher_predicted_utility"]),
            -int(row.get("opd_step_budget") or 0),
            str(row["scheduler_key"]),
        ),
        reverse=True,
    )
    selected = next((row for row in table if bool(row["passes_geometry_guard"])), None)
    if selected is None:
        raise ValueError("No V4.3 seed-mean student schedule passed the hard geometry guard.")
    return {
        "status": "ready",
        "selection_protocol": "v4.3_pooled_teacher_utility_with_hard_geometry_guard",
        "selection_mode": "teacher_utility_after_seed_mean_geometry_guard",
        "selection_unit": "seed_mean_student_budget_schedule",
        "uses_validation_labels_for_selection": False,
        "uses_locked_test_for_selection": False,
        "selection_score_name": "pooled_teacher_predicted_utility_mean",
        "unguarded_top_schedule_key": str(table[0]["scheduler_key"]),
        "selected_schedule_key": str(selected["scheduler_key"]),
        "selected_opd_step_budget": selected.get("opd_step_budget"),
        "selected_score": selected.get("selection_score"),
        "selected_geometry": {
            key: selected.get(key)
            for key in ("passes_geometry_guard", "geometry_risk_flags", "max_internal_fraction_after_098", "min_interval", "max_interval")
        },
        "geometry_guard": selected.get("thresholds", {}),
        "schedule_table": table,
    }


def _teacher_utility_for_grid(
    teacher: ScheduleTeacherMLP,
    *,
    solver: str,
    target_nfe: int,
    grid: Sequence[float],
    max_macro_steps: int,
) -> float:
    macro_steps = solver_macro_steps(str(solver), int(target_nfe))
    values = validate_time_grid(grid, macro_steps=macro_steps)
    padded = torch.zeros((1, int(max_macro_steps)), dtype=torch.float32)
    padded[:, :macro_steps] = torch.tensor(grid_to_intervals(values), dtype=torch.float32)[None, :]
    with torch.no_grad():
        return float(
            teacher(
                differentiable_teacher_features(
                    setting_features(str(solver), int(target_nfe))[None, :],
                    padded,
                    max_macro_steps=max_macro_steps,
                )
            )
            .detach()
            .cpu()
            .item()
        )


def _seed_mean_predictions_from_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    *,
    solvers: Sequence[str],
    target_nfes: Sequence[int],
    setting_dim: int,
    max_macro_steps: int,
    teacher: ScheduleTeacherMLP,
    seed_values: Sequence[int],
    opd_steps: int,
) -> List[Dict[str, Any]]:
    if not states:
        raise ValueError("Seed-mean schedule requires at least one student state.")
    model = ScheduleStudentMLP(setting_dim, max_macro_steps=max_macro_steps, hidden_dim=128, hidden_layers=2)
    predictions: List[Dict[str, Any]] = []
    for solver in solvers:
        for target_nfe in target_nfes:
            macro_steps = solver_macro_steps(str(solver), int(target_nfe))
            feat = setting_features(str(solver), int(target_nfe))[None, :]
            logits: List[torch.Tensor] = []
            for state in states:
                model.load_state_dict(dict(state))
                with torch.no_grad():
                    logits.append(model.interval_logits(feat)[0, :macro_steps].detach().clone())
            mean_logits = torch.stack(logits, dim=0).mean(dim=0)
            intervals = torch.softmax(mean_logits, dim=-1).detach().cpu().numpy().astype(np.float64)
            intervals = np.maximum(intervals, 1e-7)
            intervals = intervals / float(np.sum(intervals))
            grid = np.concatenate([[0.0], np.cumsum(intervals)])
            grid[0] = 0.0
            grid[-1] = 1.0
            values = validate_time_grid([float(x) for x in grid.tolist()], macro_steps=macro_steps)
            utility = _teacher_utility_for_grid(
                teacher,
                solver=str(solver),
                target_nfe=int(target_nfe),
                grid=values,
                max_macro_steps=max_macro_steps,
            )
            predictions.append(
                {
                    "solver_key": str(solver),
                    "target_nfe": int(target_nfe),
                    "runtime_nfe": int(macro_steps),
                    "macro_steps": int(macro_steps),
                    "realized_nfe": int(target_nfe),
                    "time_grid": list(values),
                    "grid_geometry": grid_geometry(values),
                    "max_macro_steps": int(max_macro_steps),
                    "candidate_source": "seed_mean_student",
                    "student_seed": "seed_mean",
                    "student_seed_values": list(int(seed) for seed in seed_values),
                    "opd_steps": int(opd_steps),
                    "opd_step_budget": int(opd_steps),
                    "averaged_representation": "interval_logits",
                    "perturbation_type": "none",
                    "perturbation_params_json": "{}",
                    "intervals_json": json.dumps([float(x) for x in intervals.tolist()], separators=(",", ":")),
                    "utility": float(utility),
                    "validity_flags_json": json.dumps({"finite": True, "monotone": True, "exact_realized_nfe": True}, separators=(",", ":")),
                }
            )
    return predictions


def _write_selected_schedule_summary(
    schedule_summary: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    out_path: str | Path,
) -> Dict[str, Any]:
    selected_key = str(selection["selected_schedule_key"])
    selected = next((dict(item) for item in list(schedule_summary.get("schedules", []) or []) if str(item.get("scheduler_key")) == selected_key), None)
    if selected is None:
        raise ValueError(f"Selected schedule {selected_key!r} was not present in the V4.3 student summary.")
    predictions = []
    for item in list(selected.get("predictions", []) or []):
        copied = dict(item)
        copied["source_scheduler_key"] = selected_key
        predictions.append(copied)
    schedule = {
        "scheduler_key": SELECTED_STUDENT_SCHEDULE_KEY,
        "schedule_name": SELECTED_STUDENT_SCHEDULE_NAME,
        "comparison_role": "learned_student_frozen_v43_pooled_seed_mean",
        "source_scheduler_key": selected_key,
        "uses_validation_labels_for_selection": False,
        "predictions": predictions,
    }
    summary = {
        "status": "ready",
        "artifact": "v43_selected_seed_mean_student_schedule_summary",
        "protocol": "v4.3_pooled_one_round_calibration",
        "dataset": str(schedule_summary.get("dataset")),
        "selection": dict(selection),
        "selected_source_schedule_key": selected_key,
        "baseline_schedule": False,
        "schedules": [schedule],
        "predictions": predictions,
    }
    out = resolve_project_path(str(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, str(out))
    return summary


def train_v43_policy(args: argparse.Namespace) -> Dict[str, Any]:
    seed_all(int(args.seed))
    solvers = tuple(_parse_csv(args.solver_names))
    target_nfes = tuple(_parse_int_csv(args.target_nfe_values))
    calibration_seeds = tuple(_parse_int_csv(args.calibration_seeds))
    student_seeds = tuple(_parse_int_csv(args.student_seeds))
    budgets = tuple(_parse_int_csv(args.student_opd_step_values))
    candidate_summary_paths = tuple(_parse_csv(args.candidate_schedule_summary))
    candidate_keys = _schedule_keys_from_summary_paths(candidate_summary_paths)
    if not candidate_keys:
        raise ValueError("V4.3 teacher training requires one pooled BO candidate schedule summary.")

    out_dir = resolve_project_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_anchor_keys = tuple(BASELINE_SCHEDULE_KEYS) + (SER_PTG_SCHEDULE_KEY,)
    if bool(args.dry_run):
        summary = {
            "status": "dry_run",
            "protocol": "v4.3_pooled_one_round_calibration",
            "dataset": str(args.dataset),
            "split_phase": POOLED_CALIBRATION_PHASE,
            "calibration_seeds": list(calibration_seeds),
            "student_seeds": list(student_seeds),
            "student_initialization": SER_PTG_SCHEDULE_KEY,
            "teacher_anchor_schedule_keys": list(teacher_anchor_keys),
            "reward_reference_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
            "reward_reference_source": "pooled_recomputed_rows_only",
            "uses_validation_selection": False,
            "uses_source_balanced_rewards": False,
            "final_selector": "pooled_teacher_utility_with_hard_geometry_guard",
            "lowest_internal_loss_selector_used": False,
        }
        save_json(summary, str(out_dir / "conditional_opd_v43_summary.json"))
        save_json(summary, str(out_dir / "conditional_opd_summary.json"))
        return summary

    fixed_rows = _clean_pooled_rows(
        _load_csv_rows(args.pooled_fixed_rows),
        dataset=str(args.dataset),
        solvers=solvers,
        target_nfes=target_nfes,
        allowed_schedules=BASELINE_SCHEDULE_KEYS,
        seeds=calibration_seeds,
        label="V4.3 pooled fixed reward references",
    )
    ser_rows = _clean_pooled_rows(
        _load_csv_rows(args.pooled_ser_rows),
        dataset=str(args.dataset),
        solvers=solvers,
        target_nfes=target_nfes,
        allowed_schedules=(SER_PTG_SCHEDULE_KEY,),
        seeds=calibration_seeds,
        label="V4.3 pooled SER-PTG anchor",
    )
    candidate_rows = _clean_pooled_rows(
        _load_csv_rows(args.pooled_candidate_rows),
        dataset=str(args.dataset),
        solvers=solvers,
        target_nfes=target_nfes,
        allowed_schedules=candidate_keys,
        seeds=calibration_seeds,
        label="V4.3 pooled BO candidates",
    )
    _assert_complete_seed_rows(
        fixed_rows,
        solvers=solvers,
        target_nfes=target_nfes,
        schedules=BASELINE_SCHEDULE_KEYS,
        seeds=calibration_seeds,
        label="V4.3 pooled fixed reward references",
    )
    _assert_complete_seed_rows(
        ser_rows,
        solvers=solvers,
        target_nfes=target_nfes,
        schedules=(SER_PTG_SCHEDULE_KEY,),
        seeds=calibration_seeds,
        label="V4.3 pooled SER-PTG anchor",
    )
    _assert_complete_seed_rows(
        candidate_rows,
        solvers=solvers,
        target_nfes=target_nfes,
        schedules=candidate_keys,
        seeds=calibration_seeds,
        label="V4.3 pooled BO candidates",
    )

    max_macro_steps = max(solver_macro_steps(solver, nfe) for solver in solvers for nfe in target_nfes)
    setting_dim = int(setting_features("euler", 4).numel())
    teacher_input_dim = setting_dim + int(max_macro_steps)
    teacher = ScheduleTeacherMLP(teacher_input_dim, hidden_dim=256, hidden_layers=3)
    teacher_param_count = count_parameters(teacher)
    student_param_count = count_parameters(ScheduleStudentMLP(setting_dim, max_macro_steps=max_macro_steps, hidden_dim=128, hidden_layers=2))

    summary_grids = _load_schedule_summary_grids(args.ser_schedule_summary)
    summary_grids.update(_load_schedule_summary_grids_many(candidate_summary_paths))
    all_rows = [*fixed_rows, *ser_rows, *candidate_rows]
    aggregate_rows = seed_mean_metric_rows(all_rows)
    annotated_rows = attach_reward_columns(aggregate_rows, fixed_schedule_keys=BASELINE_SCHEDULE_KEYS)
    reward_reference_table = build_fixed_reference_table(fixed_rows, fixed_schedule_keys=BASELINE_SCHEDULE_KEYS)
    rewards = rewards_by_setting(annotated_rows, fixed_schedule_keys=BASELINE_SCHEDULE_KEYS)
    teacher_training_keys = set(candidate_keys).union(teacher_anchor_keys)
    teacher_rows = [row for row in annotated_rows if str(row["scheduler_key"]) in teacher_training_keys]
    teacher_fit_rows, teacher_diagnostic_rows = split_teacher_diagnostic_holdout(
        teacher_rows,
        fraction=float(args.teacher_diagnostic_holdout_fraction),
        seed=int(args.seed),
        fixed_schedule_keys=teacher_anchor_keys,
        rewards=rewards,
    )
    schedule_weights = teacher_schedule_weights(
        sorted({str(row["scheduler_key"]) for row in teacher_rows}),
        late_biased_demo_schedules=tuple(_parse_csv(args.late_biased_demo_schedules)),
        late_biased_demo_weight=float(args.late_biased_demo_weight),
    )
    teacher_training = train_teacher(
        teacher,
        teacher_fit_rows,
        rewards=rewards,
        summary_grids=summary_grids,
        max_macro_steps=max_macro_steps,
        steps=int(args.teacher_steps),
        lr=float(args.lr),
        schedule_weights=schedule_weights,
        rank_temperature=float(args.teacher_rank_temperature),
        regression_weight=float(args.teacher_regression_weight),
        pairs_per_candidate=int(args.teacher_pairs_per_candidate),
        pair_margin=float(args.teacher_pair_margin),
        diagnostic_rows=teacher_diagnostic_rows,
        diagnostic_top_k=int(args.teacher_diagnostic_top_k),
    )
    teacher_diagnostics = dict(teacher_training["selected_diagnostics"])
    teacher_diagnostics.update(
        {
            "diagnostic_split": POOLED_CALIBRATION_PHASE,
            "uses_validation_labels": False,
            "teacher_holdout_source": "pooled_bo_candidate_rows_only",
            "reference_thresholds_from_v42": {
                "pairwise_accuracy": 0.8214,
                "spearman": 0.7679,
                "top_k_recall": 0.8667,
                "best_fixed_crossing": 0.8306,
            },
        }
    )

    settings = [(str(solver), int(target_nfe)) for solver in solvers for target_nfe in target_nfes]
    ser_targets = [
        (solver, target_nfe, _grid_for_schedule(SER_PTG_SCHEDULE_KEY, solver, target_nfe, summary_grids=summary_grids))
        for solver, target_nfe in settings
    ]
    states_by_budget: Dict[str, List[Mapping[str, torch.Tensor]]] = {str(int(budget)): [] for budget in budgets}
    init_losses_by_seed: Dict[str, List[Dict[str, float]]] = {}
    opd_losses_by_seed_budget: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    checkpoint_payload: Dict[str, Any] = {"budget_student_states": {}}
    for student_seed in student_seeds:
        seed_all(int(student_seed))
        base_student = ScheduleStudentMLP(setting_dim, max_macro_steps=max_macro_steps, hidden_dim=128, hidden_layers=2)
        init_losses_by_seed[str(student_seed)] = fit_student_to_reference(
            base_student,
            ser_targets,
            max_macro_steps=max_macro_steps,
            steps=int(args.student_init_steps),
            lr=float(args.lr),
        )
        base_state = copy.deepcopy(base_student.state_dict())
        opd_losses_by_seed_budget[str(student_seed)] = {}
        for budget in budgets:
            budget_student = ScheduleStudentMLP(setting_dim, max_macro_steps=max_macro_steps, hidden_dim=128, hidden_layers=2)
            budget_student.load_state_dict(base_state)
            losses = optimize_student_with_teacher(
                budget_student,
                teacher,
                settings,
                max_macro_steps=max_macro_steps,
                steps=int(budget),
                lr=float(args.lr),
                ser_ptg_reference_pairs=ser_targets,
                ser_ptg_regularizer=str(args.student_ser_ptg_regularizer),
                ser_ptg_regularization_weight=float(args.student_ser_ptg_regularization_weight),
                ser_ptg_regularization_eps=float(args.student_ser_ptg_regularization_eps),
            )
            states_by_budget[str(int(budget))].append(copy.deepcopy(budget_student.state_dict()))
            opd_losses_by_seed_budget[str(student_seed)][str(int(budget))] = losses
            checkpoint_payload["budget_student_states"].setdefault(str(int(budget)), {})[str(student_seed)] = copy.deepcopy(budget_student.state_dict())

    schedules: List[Dict[str, Any]] = []
    prefix = str(args.student_schedule_key_prefix).strip() or "conditional_opd_student_v43_steps"
    for budget in budgets:
        predictions = _seed_mean_predictions_from_states(
            states_by_budget[str(int(budget))],
            solvers=solvers,
            target_nfes=target_nfes,
            setting_dim=setting_dim,
            max_macro_steps=max_macro_steps,
            teacher=teacher,
            seed_values=student_seeds,
            opd_steps=int(budget),
        )
        schedules.append(
            {
                "scheduler_key": f"{prefix}{int(budget)}",
                "schedule_name": f"V4.3 SER-PTG seed-mean student {int(budget)} updates",
                "comparison_role": "learned_student_v43_pooled_seed_mean_budget",
                "student_initialization": SER_PTG_SCHEDULE_KEY,
                "student_seed_values": list(student_seeds),
                "opd_step_budget": int(budget),
                "teacher_predicted_utility_mean": float(np.mean([float(item["utility"]) for item in predictions])),
                "predictions": predictions,
            }
        )
    schedule_summary = {
        "status": "ready",
        "artifact": "v43_seed_mean_student_schedule_summary",
        "protocol": "v4.3_pooled_one_round_calibration",
        "dataset": str(args.dataset),
        "baseline_schedule": False,
        "student_initialization": SER_PTG_SCHEDULE_KEY,
        "student_seed_values": list(student_seeds),
        "student_opd_step_values": [int(value) for value in budgets],
        "schedules": schedules,
    }
    save_json(schedule_summary, str(out_dir / "student_seed_mean_schedule_summary.json"))
    selection = select_guarded_teacher_utility_schedule(
        schedule_summary,
        max_internal_after_098=float(args.geometry_max_internal_after_098),
        min_interval_floor=float(args.geometry_min_interval),
        max_interval_ceiling=float(args.geometry_max_interval),
    )
    save_json(selection, str(out_dir / "final_selection.json"))
    selected_summary = _write_selected_schedule_summary(
        schedule_summary,
        selection=selection,
        out_path=out_dir / "selected_schedule_summary.json",
    )
    torch.save(
        {
            "teacher_state": teacher.state_dict(),
            "setting_dim": int(setting_dim),
            "teacher_input_dim": int(teacher_input_dim),
            "max_macro_steps": int(max_macro_steps),
            "protocol": "v4.3_pooled_one_round_calibration",
            **checkpoint_payload,
        },
        out_dir / "conditional_opd_v43.pt",
    )
    summary = {
        "status": "ready",
        "protocol": "v4.3_pooled_one_round_calibration",
        "dataset": str(args.dataset),
        "split_phase": POOLED_CALIBRATION_PHASE,
        "calibration_seeds": list(calibration_seeds),
        "student_seeds": list(student_seeds),
        "uses_validation_selection": False,
        "uses_source_balanced_rewards": False,
        "reward_reference_source": "pooled_recomputed_rows_only",
        "reward_reference_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "teacher_anchor_schedule_keys": list(teacher_anchor_keys),
        "teacher_objective": "pooled_rank_first_best_fixed_composite_with_huber_calibration",
        "teacher_selection_protocol": "v4.3_pooled_bo_holdout_teacher_checkpoint",
        "teacher_checkpoint_selection": dict(teacher_training["checkpoint_selection"]),
        "teacher_diagnostics": teacher_diagnostics,
        "teacher_losses": list(teacher_training["losses"]),
        "teacher_parameters": int(teacher_param_count),
        "student_parameters": int(student_param_count),
        "student_initialization": SER_PTG_SCHEDULE_KEY,
        "student_initialization_losses": init_losses_by_seed,
        "student_opd_losses": opd_losses_by_seed_budget,
        "final_selector": "pooled_teacher_utility_with_hard_geometry_guard",
        "lowest_internal_loss_selector_used": False,
        "metric_reward": "pooled_paired_best_fixed_equal_weight_negative_log_crps_mase",
        "seed_aggregation": "mean_by_solver_target_nfe_schedule_before_reward",
        "teacher_supervision_sources": {
            "pooled_fixed_seed_rows": int(len(fixed_rows)),
            "pooled_ser_seed_rows": int(len(ser_rows)),
            "pooled_candidate_seed_rows": int(len(candidate_rows)),
            "aggregate_rows": int(len(aggregate_rows)),
            "teacher_training_rows": int(len(teacher_rows)),
            "teacher_fit_rows": int(len(teacher_fit_rows)),
            "teacher_diagnostic_holdout_rows": int(len(teacher_diagnostic_rows)),
            "candidate_schedule_count": int(len(candidate_keys)),
        },
        "reward_reference_pooled_calibration": [dict(value) for _, value in sorted(reward_reference_table.items(), key=lambda item: item[0])],
        "candidate_table": _candidate_table(annotated_rows, rewards=rewards, teacher_fixed_schedule_keys=teacher_anchor_keys),
        "student_seed_mean_schedule_summary": str(out_dir / "student_seed_mean_schedule_summary.json"),
        "selected_schedule_summary": str(out_dir / "selected_schedule_summary.json"),
        "selected_summary_schedule_count": int(len(selected_summary.get("schedules", []))),
        "final_selection": selection,
    }
    save_json(teacher_diagnostics, str(out_dir / "teacher_diagnostics.json"))
    save_json(summary, str(out_dir / "conditional_opd_v43_summary.json"))
    save_json(summary, str(out_dir / "conditional_opd_summary.json"))
    return summary


def _archive_v42_f_output(path: Path, *, allow_execute: bool, commands: List[List[str]]) -> str:
    resolved = resolve_project_path(str(path))
    if not resolved.exists():
        return ""
    archive_root = project_outputs_root() / "archive" / f"pre_v43_{time.strftime('%Y%m%d_%H%M%S')}"
    target = archive_root / resolved.name
    commands.append(["archive_output_root", str(resolved), str(target)])
    if allow_execute:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved), str(target))
    return str(target)


def run_train20_v43_pooled_calibration(args: argparse.Namespace) -> Dict[str, Any]:
    calibration_seeds = tuple(_parse_int_csv(args.calibration_seeds))
    student_seeds = tuple(_parse_int_csv(args.student_seeds))
    locked_test_seeds = tuple(_parse_int_csv(args.locked_test_seeds))
    if bool(args.strict_v43_protocol):
        if calibration_seeds != DEFAULT_CALIBRATION_SEEDS:
            raise ValueError("Strict V4.3 pooled calibration reward seeds must be exactly 0,1.")
        if student_seeds != DEFAULT_STUDENT_SEEDS:
            raise ValueError("Strict V4.3 student seeds must be exactly 0,1,2.")
        if locked_test_seeds != DEFAULT_LOCKED_TEST_SEEDS:
            raise ValueError("Strict V4.3 locked-test seeds must be exactly 0,1,2.")

    out_dir = resolve_project_path(str(args.out_dir))
    project_dir = resolve_project_path(".")
    allow_execute = bool(args.allow_execute)
    commands: List[List[str]] = []
    archived_v42_f = ""
    if bool(args.archive_v42_f_output):
        archived_v42_f = _archive_v42_f_output(resolve_project_path(str(args.v42_f_out_dir)), allow_execute=allow_execute, commands=commands)
    if allow_execute:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    fixed_summary = out_dir / "fixed_reference_schedule_summary.json"
    ser_dir = out_dir / "ser_ptg_reference"
    ser_summary = ser_dir / "ser_ptg_schedule_summary.json"
    fixed_eval_dir = out_dir / "pooled_fixed_reference"
    ser_eval_dir = out_dir / "pooled_ser_ptg_reference"
    bo_summary = out_dir / "bo_pooled_candidate_schedule_summary.json"
    bo_eval_dir = out_dir / "pooled_bo_candidates"
    neural_dir = out_dir / "neural_policy"

    common = [
        "--dataset",
        str(args.dataset),
        "--solver_names",
        str(args.solver_names),
        "--target_nfe_values",
        str(args.target_nfe_values),
    ]
    tuning = [
        "--eval_train_fraction",
        str(args.eval_train_fraction),
        "--train_tuning_seed",
        str(args.train_tuning_seed),
        "--train_tuning_strata",
        str(args.train_tuning_strata),
        "--train_tuning_sampling_mode",
        str(args.train_tuning_sampling_mode),
        "--train_tuning_train_split_fraction",
        str(args.train_tuning_train_split_fraction),
        "--train_tuning_val_split_fraction",
        str(args.train_tuning_val_split_fraction),
    ]
    eval_common = [
        *common,
        "--calibration_seeds",
        str(args.calibration_seeds),
        "--num_eval_samples",
        str(args.num_eval_samples),
        "--forecast_eval_batch_size",
        str(args.forecast_eval_batch_size),
        "--dataset_root",
        str(args.dataset_root),
        "--shared_backbone_root",
        str(args.shared_backbone_root),
        "--backbone_manifest",
        str(args.backbone_manifest),
        "--otflow_train_steps",
        str(args.otflow_train_steps),
        "--device",
        str(args.device),
        *tuning,
    ]

    _run_command(
        _python_module_command(
            "genode.conditional_opd.train20_v43_pooled_calibration",
            ["--stage", "write-fixed-summary", *common, "--fixed_schedule_summary", str(fixed_summary)],
        ),
        cwd=project_dir,
        allow_execute=allow_execute,
        commands=commands,
    )
    _run_command(
        _python_module_command(
            "genode.conditional_opd.ser_ptg_reference",
            [
                *common,
                "--trace_variant",
                "local_defect",
                "--density_floor_eta",
                str(args.ser_density_floor_eta),
                "--reference_split",
                TRAIN_TUNING_PHASE,
                "--seeds",
                str(args.calibration_seeds),
                "--train_tuning_fraction",
                str(args.eval_train_fraction),
                "--train_tuning_seed",
                str(args.train_tuning_seed),
                "--train_tuning_strata",
                str(args.train_tuning_strata),
                "--train_tuning_sampling_mode",
                str(args.train_tuning_sampling_mode),
                "--train_tuning_train_split_fraction",
                str(args.train_tuning_train_split_fraction),
                "--train_tuning_val_split_fraction",
                str(args.train_tuning_val_split_fraction),
                "--out_dir",
                str(ser_dir),
                "--dataset_root",
                str(args.dataset_root),
                "--shared_backbone_root",
                str(args.shared_backbone_root),
                "--backbone_manifest",
                str(args.backbone_manifest),
                "--otflow_train_steps",
                str(args.otflow_train_steps),
                "--device",
                str(args.device),
            ],
        ),
        cwd=project_dir,
        allow_execute=allow_execute,
        commands=commands,
    )
    for schedule_summary, eval_dir, row_name in (
        (fixed_summary, fixed_eval_dir, "pooled_fixed_rows.csv"),
        (ser_summary, ser_eval_dir, "pooled_ser_rows.csv"),
    ):
        _run_command(
            _python_module_command(
                "genode.conditional_opd.train20_v43_pooled_calibration",
                [
                    "--stage",
                    "evaluate-pooled",
                    *eval_common,
                    "--schedule_summary",
                    str(schedule_summary),
                    "--out_dir",
                    str(eval_dir),
                    "--row_csv_name",
                    row_name,
                ],
            ),
            cwd=project_dir,
            allow_execute=allow_execute,
            commands=commands,
        )
    _run_command(
        _python_module_command(
            "genode.conditional_opd.bo_candidate_pool",
            [
                "--mode",
                "generate",
                "--dataset",
                str(args.dataset),
                "--reference_schedule_summary",
                str(ser_summary),
                "--out_path",
                str(bo_summary),
                "--active_round",
                "0",
                "--seed",
                str(args.train_tuning_seed),
                "--candidate_count",
                str(args.bo_candidate_count),
                "--density_grid_size",
                str(args.density_grid_size),
                "--theta_bound",
                str(args.bo_theta_bound),
                "--sobol_pool",
                str(args.bo_sobol_pool),
            ],
        ),
        cwd=project_dir,
        allow_execute=allow_execute,
        commands=commands,
    )
    _run_command(
        _python_module_command(
            "genode.conditional_opd.train20_v43_pooled_calibration",
            [
                "--stage",
                "evaluate-pooled",
                *eval_common,
                "--schedule_summary",
                str(bo_summary),
                "--out_dir",
                str(bo_eval_dir),
                "--row_csv_name",
                "pooled_bo_rows.csv",
            ],
        ),
        cwd=project_dir,
        allow_execute=allow_execute,
        commands=commands,
    )
    train_args = [
        "--stage",
        "train-policy",
        *common,
        "--calibration_seeds",
        str(args.calibration_seeds),
        "--student_seeds",
        str(args.student_seeds),
        "--pooled_fixed_rows",
        str(fixed_eval_dir / "pooled_fixed_rows.csv"),
        "--pooled_ser_rows",
        str(ser_eval_dir / "pooled_ser_rows.csv"),
        "--pooled_candidate_rows",
        str(bo_eval_dir / "pooled_bo_rows.csv"),
        "--ser_schedule_summary",
        str(ser_summary),
        "--candidate_schedule_summary",
        str(bo_summary),
        "--out_dir",
        str(neural_dir),
        "--teacher_steps",
        str(args.teacher_steps),
        "--teacher_diagnostic_holdout_fraction",
        str(args.teacher_diagnostic_holdout_fraction),
        "--student_init_steps",
        str(args.student_init_steps),
        "--student_opd_step_values",
        str(args.student_opd_step_values),
        "--student_ser_ptg_regularizer",
        str(args.student_ser_ptg_regularizer),
        "--student_ser_ptg_regularization_weight",
        str(args.student_ser_ptg_regularization_weight),
        "--student_ser_ptg_regularization_eps",
        str(args.student_ser_ptg_regularization_eps),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
    ]
    _run_command(
        _python_module_command("genode.conditional_opd.train20_v43_pooled_calibration", train_args),
        cwd=project_dir,
        allow_execute=allow_execute,
        commands=commands,
    )
    if not bool(args.skip_locked_test):
        _run_command(
            _python_module_command(
                "genode.conditional_opd.evaluate_schedule_summary",
                [
                    "--schedule_summary",
                    str(neural_dir / "selected_schedule_summary.json"),
                    "--split_phase",
                    LOCKED_TEST_PHASE,
                    "--out_dir",
                    str(out_dir),
                    "--row_csv_name",
                    "locked_test_rows.csv",
                    "--comparison_output_name",
                    "locked_test_comparison_summary.json",
                    "--baseline_rows",
                    str(args.baseline_test_rows),
                    *common,
                    "--seeds",
                    str(args.locked_test_seeds),
                    "--num_eval_samples",
                    str(args.num_eval_samples),
                    "--eval_windows_test",
                    str(args.eval_windows_test),
                    "--forecast_eval_batch_size",
                    str(args.forecast_eval_batch_size),
                    "--dataset_root",
                    str(args.dataset_root),
                    "--shared_backbone_root",
                    str(args.shared_backbone_root),
                    "--backbone_manifest",
                    str(args.backbone_manifest),
                    "--otflow_train_steps",
                    str(args.otflow_train_steps),
                    "--device",
                    str(args.device),
                ],
            ),
            cwd=project_dir,
            allow_execute=allow_execute,
            commands=commands,
        )
    summary = {
        "status": "ready" if allow_execute else "dry_run",
        "artifact": "train20_v43_pooled_one_round_calibration",
        "protocol": "v4.3_pooled_one_round_calibration",
        "dataset": str(args.dataset),
        "out_dir": str(out_dir),
        "archived_v42_f_output_root": archived_v42_f,
        "calibration_seeds": list(calibration_seeds),
        "student_seeds": list(student_seeds),
        "locked_test_seeds": list(locked_test_seeds),
        "strict_v43_protocol": bool(args.strict_v43_protocol),
        "uses_validation_selection": False,
        "uses_source_balanced_rewards": False,
        "student_initialization": SER_PTG_SCHEDULE_KEY,
        "final_selector": "pooled_teacher_utility_with_hard_geometry_guard",
        "lowest_internal_loss_selector_used": False,
        "bo_rounds": 1,
        "bo_candidate_count": int(args.bo_candidate_count),
        "fixed_schedule_summary": str(fixed_summary),
        "ser_schedule_summary": str(ser_summary),
        "bo_candidate_schedule_summary": str(bo_summary),
        "neural_policy_dir": str(neural_dir),
        "selected_schedule_summary": str(neural_dir / "selected_schedule_summary.json"),
        "locked_test_rows": str(out_dir / "locked_test_rows.csv") if not bool(args.skip_locked_test) else "",
        "locked_test_comparison_summary": str(out_dir / "locked_test_comparison_summary.json") if not bool(args.skip_locked_test) else "",
        "commands": commands,
        "allow_execute": allow_execute,
    }
    save_json(summary, str(out_dir / "train20_v43_pooled_calibration_summary.json"))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V4.3 pooled one-round calibration for Train20 conditional OPD.")
    parser.add_argument("--stage", choices=("run", "write-fixed-summary", "evaluate-pooled", "train-policy"), default="run")
    parser.add_argument("--dataset", default="san_francisco_traffic")
    parser.add_argument("--solver_names", default=",".join(DEFAULT_SOLVERS))
    parser.add_argument("--target_nfe_values", default=",".join(str(value) for value in DEFAULT_TARGET_NFES))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--calibration_seeds", default=_csv_text(DEFAULT_CALIBRATION_SEEDS))
    parser.add_argument("--student_seeds", default=_csv_text(DEFAULT_STUDENT_SEEDS))
    parser.add_argument("--locked_test_seeds", default=_csv_text(DEFAULT_LOCKED_TEST_SEEDS))
    parser.add_argument("--fixed_schedule_summary", default=str(DEFAULT_OUT_DIR / "fixed_reference_schedule_summary.json"))
    parser.add_argument("--schedule_summary", default="")
    parser.add_argument("--row_csv_name", default="pooled_rows.csv")
    parser.add_argument("--pooled_fixed_rows", default="")
    parser.add_argument("--pooled_ser_rows", default="")
    parser.add_argument("--pooled_candidate_rows", default="")
    parser.add_argument("--ser_schedule_summary", default="")
    parser.add_argument("--candidate_schedule_summary", default="")
    parser.add_argument("--baseline_test_rows", default=str(DEFAULT_BASELINE_TEST_ROWS))
    parser.add_argument("--bo_candidate_count", type=int, default=32)
    parser.add_argument("--bo_theta_bound", type=float, default=2.5)
    parser.add_argument("--bo_sobol_pool", type=int, default=512)
    parser.add_argument("--density_grid_size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--ser_density_floor_eta", type=float, default=0.05)
    parser.add_argument("--teacher_steps", type=int, default=500)
    parser.add_argument("--teacher_rank_temperature", type=float, default=DEFAULT_TEACHER_RANK_TEMPERATURE)
    parser.add_argument("--teacher_regression_weight", type=float, default=DEFAULT_TEACHER_REGRESSION_WEIGHT)
    parser.add_argument("--teacher_pairs_per_candidate", type=int, default=DEFAULT_TEACHER_PAIRS_PER_CANDIDATE)
    parser.add_argument("--teacher_pair_margin", type=float, default=DEFAULT_TEACHER_PAIR_MARGIN)
    parser.add_argument("--teacher_diagnostic_top_k", type=int, default=5)
    parser.add_argument("--teacher_diagnostic_holdout_fraction", type=float, default=DEFAULT_TEACHER_DIAGNOSTIC_HOLDOUT_FRACTION)
    parser.add_argument("--student_init_steps", type=int, default=500)
    parser.add_argument("--student_opd_step_values", default=_csv_text(DEFAULT_STUDENT_BUDGETS))
    parser.add_argument("--student_schedule_key_prefix", default="conditional_opd_student_v43_steps")
    parser.add_argument("--student_ser_ptg_regularizer", choices=STUDENT_SER_PTG_REGULARIZERS, default=STUDENT_SER_PTG_REGULARIZER_NONE)
    parser.add_argument("--student_ser_ptg_regularization_weight", type=float, default=0.0)
    parser.add_argument("--student_ser_ptg_regularization_eps", type=float, default=DEFAULT_STUDENT_SER_PTG_REGULARIZATION_EPS)
    parser.add_argument("--late_biased_demo_schedules", default=",".join(DEFAULT_LATE_BIASED_DEMO_SCHEDULES))
    parser.add_argument("--late_biased_demo_weight", type=float, default=DEFAULT_LATE_BIASED_DEMO_WEIGHT)
    parser.add_argument("--geometry_max_internal_after_098", type=float, default=DEFAULT_GEOMETRY_MAX_INTERNAL_AFTER_098)
    parser.add_argument("--geometry_min_interval", type=float, default=DEFAULT_GEOMETRY_MIN_INTERVAL)
    parser.add_argument("--geometry_max_interval", type=float, default=DEFAULT_GEOMETRY_MAX_INTERVAL)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_train_fraction", type=float, default=0.20)
    parser.add_argument("--train_tuning_seed", type=int, default=0)
    parser.add_argument("--train_tuning_strata", type=int, default=20)
    parser.add_argument("--train_tuning_sampling_mode", choices=TRAIN_TUNING_SAMPLING_MODES, default=TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED)
    parser.add_argument("--train_tuning_train_split_fraction", type=float, default=DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION)
    parser.add_argument("--train_tuning_val_split_fraction", type=float, default=DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION)
    parser.add_argument("--num_eval_samples", type=int, default=5)
    parser.add_argument("--forecast_eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_windows_test", type=int, default=0)
    parser.add_argument("--dataset_root", default=str(project_paper_dataset_root()))
    parser.add_argument("--shared_backbone_root", default=str(DEFAULT_SHARED_BACKBONE_ROOT))
    parser.add_argument("--backbone_manifest", default=str(default_backbone_manifest_path()))
    parser.add_argument("--otflow_train_steps", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip_locked_test", action="store_true", default=False)
    parser.add_argument("--archive_v42_f_output", action="store_true", default=False)
    parser.add_argument("--no_archive_v42_f_output", action="store_false", dest="archive_v42_f_output")
    parser.add_argument("--v42_f_out_dir", default=str(DEFAULT_V42_F_OUT_DIR))
    parser.add_argument("--strict_v43_protocol", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--allow_execute", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if str(args.stage) == "write-fixed-summary":
        summary = write_fixed_schedule_summary(args)
    elif str(args.stage) == "evaluate-pooled":
        summary = evaluate_pooled_schedule_summary(args)
    elif str(args.stage) == "train-policy":
        summary = train_v43_policy(args)
    else:
        summary = run_train20_v43_pooled_calibration(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
