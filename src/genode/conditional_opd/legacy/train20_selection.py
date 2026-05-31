from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from genode.conditional_opd.evaluate_schedule_summary import (
    SCHEDULE_ROW_FIELDS,
    select_best_validation_schedule,
    write_selected_schedule_summary,
)
from genode.conditional_opd.ser_ptg_reference import SER_PTG_SCHEDULE_KEY
from genode.data.otflow_paths import project_outputs_root, resolve_project_path
from genode.evaluation.otflow_evaluation_support import (
    DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION,
    DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION,
    LOCKED_TEST_PHASE,
    TRAIN_TUNING_PHASE,
    TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
    TRAIN_TUNING_SAMPLING_MODES,
    VALIDATION_PHASE,
    train_tuning_sampler_key,
)
from genode.models.otflow_train_val import save_json
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS

DEFAULT_GEOMETRY_MAX_INTERNAL_AFTER_098 = 0.5
DEFAULT_GEOMETRY_MIN_INTERVAL = 1e-4
DEFAULT_GEOMETRY_MAX_INTERVAL = 0.97
DEFAULT_TEACHER_FIXED_SCHEDULE_KEYS = ("uniform", "late_power_3")
DEFAULT_BASELINE_VALIDATION_ROWS = project_outputs_root() / "diffusion_flow_time_reparameterization_full_val" / "rows.csv"


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in _parse_csv(text)]


def _parse_teacher_fixed_schedule_keys(text: str) -> List[str]:
    keys = _parse_csv(text)
    if not keys:
        raise ValueError("teacher_fixed_schedule_keys must contain at least one schedule.")
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"teacher_fixed_schedule_keys contains duplicates: {duplicates}")
    unsupported = sorted(set(keys) - set(BASELINE_SCHEDULE_KEYS))
    if unsupported:
        raise ValueError(f"teacher_fixed_schedule_keys must be fixed baselines; unsupported: {unsupported}")
    if "uniform" not in keys:
        raise ValueError("teacher_fixed_schedule_keys must include uniform.")
    return keys


def _read_rows(path: str | Path) -> List[Dict[str, Any]]:
    resolved = resolve_project_path(str(path))
    if not resolved.exists():
        raise FileNotFoundError(f"Rows CSV not found: {resolved}")
    with resolved.open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = resolve_project_path(str(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SCHEDULE_ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SCHEDULE_ROW_FIELDS})


def _validate_baseline_validation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    dataset: str,
    seeds: Sequence[int],
    solver_names: Sequence[str],
    target_nfe_values: Sequence[int],
    require_candidate_hash_match: bool = True,
) -> List[Dict[str, Any]]:
    if not rows:
        raise ValueError("baseline_validation_rows produced no rows; fixed validation references are required for Train20 selection.")
    seed_set = {int(seed) for seed in seeds}
    solver_set = {str(solver) for solver in solver_names}
    nfe_set = {int(nfe) for nfe in target_nfe_values}
    schedule_set = set(BASELINE_SCHEDULE_KEYS)
    observed: Dict[Tuple[int, str, int, str], Mapping[str, Any]] = {}
    invalid: List[Mapping[str, Any]] = []
    duplicates: List[Tuple[int, str, int, str]] = []
    for row in rows:
        try:
            seed = int(row.get("seed", -1))
            target_nfe = int(row.get("target_nfe", -1))
        except (TypeError, ValueError):
            invalid.append(row)
            continue
        solver = str(row.get("solver_key", ""))
        schedule = str(row.get("scheduler_key", ""))
        if (
            str(row.get("dataset", "")) != str(dataset)
            or str(row.get("split_phase", "")) != VALIDATION_PHASE
            or seed not in seed_set
            or solver not in solver_set
            or target_nfe not in nfe_set
            or schedule not in schedule_set
        ):
            invalid.append(row)
            continue
        key = (seed, solver, target_nfe, schedule)
        if key in observed:
            duplicates.append(key)
        observed[key] = row
    if invalid:
        sample = [
            {
                "dataset": row.get("dataset"),
                "split_phase": row.get("split_phase"),
                "seed": row.get("seed"),
                "solver_key": row.get("solver_key"),
                "target_nfe": row.get("target_nfe"),
                "scheduler_key": row.get("scheduler_key"),
            }
            for row in invalid[:5]
        ]
        raise ValueError(f"baseline_validation_rows contains rows outside the requested full validation fixed-grid scope: {sample}")
    if duplicates:
        raise ValueError(f"baseline_validation_rows contains duplicate fixed validation cells: {duplicates[:12]}")
    expected = {
        (int(seed), str(solver), int(target_nfe), str(schedule))
        for seed in seed_set
        for solver in solver_set
        for target_nfe in nfe_set
        for schedule in BASELINE_SCHEDULE_KEYS
    }
    missing = sorted(expected - set(observed), key=lambda item: (item[0], item[1], item[2], item[3]))
    if missing:
        raise ValueError(f"baseline_validation_rows is missing fixed validation cells: {missing[:12]}")
    reference_hashes = {str(row.get("chosen_examples_hash", "")) for row in rows if str(row.get("chosen_examples_hash", "")).strip()}
    candidate_hashes = {
        str(row.get("chosen_examples_hash", ""))
        for row in candidate_rows
        if str(row.get("chosen_examples_hash", "")).strip()
    }
    if not reference_hashes:
        raise ValueError("baseline_validation_rows is missing chosen_examples_hash; paired validation selection requires fixed reference hashes.")
    if require_candidate_hash_match and not candidate_hashes:
        raise ValueError("Validation candidate rows are missing chosen_examples_hash; paired validation selection requires matching examples.")
    if candidate_hashes and reference_hashes != candidate_hashes:
        raise ValueError(
            "baseline_validation_rows chosen_examples_hash does not match validation candidate rows: "
            f"reference={sorted(reference_hashes)}, candidate={sorted(candidate_hashes)}"
        )
    return [dict(row) for row in rows]


def _optional_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def geometry_guard_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_internal_after_098: float = DEFAULT_GEOMETRY_MAX_INTERNAL_AFTER_098,
    min_interval_floor: float = DEFAULT_GEOMETRY_MIN_INTERVAL,
    max_interval_ceiling: float = DEFAULT_GEOMETRY_MAX_INTERVAL,
) -> Dict[str, Any]:
    if not rows:
        return {
            "passes_geometry_guard": False,
            "geometry_risk_flags": ["missing_rows"],
            "max_internal_fraction_after_098": None,
            "min_interval": None,
            "max_interval": None,
            "thresholds": {
                "max_internal_fraction_after_098": float(max_internal_after_098),
                "min_interval": float(min_interval_floor),
                "max_interval": float(max_interval_ceiling),
            },
        }
    missing_fields = []
    late_values = []
    min_values = []
    max_values = []
    for row in rows:
        late = _optional_float(row.get("internal_fraction_after_098"), default=float("nan"))
        min_interval_value = _optional_float(row.get("min_interval"), default=float("nan"))
        max_interval_value = _optional_float(row.get("max_interval"), default=float("nan"))
        if not math.isfinite(late):
            missing_fields.append("internal_fraction_after_098")
        if not math.isfinite(min_interval_value):
            missing_fields.append("min_interval")
        if not math.isfinite(max_interval_value):
            missing_fields.append("max_interval")
        late_values.append(late)
        min_values.append(min_interval_value)
        max_values.append(max_interval_value)
    if missing_fields:
        return {
            "passes_geometry_guard": False,
            "geometry_risk_flags": ["missing_geometry_metrics"],
            "missing_geometry_fields": sorted(set(missing_fields)),
            "max_internal_fraction_after_098": None,
            "min_interval": None,
            "max_interval": None,
            "thresholds": {
                "max_internal_fraction_after_098": float(max_internal_after_098),
                "min_interval": float(min_interval_floor),
                "max_interval": float(max_interval_ceiling),
            },
        }
    max_late = max(late_values)
    min_interval = min(min_values)
    max_interval = max(max_values)
    flags: List[str] = []
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
        "thresholds": {
            "max_internal_fraction_after_098": float(max_internal_after_098),
            "min_interval": float(min_interval_floor),
            "max_interval": float(max_interval_ceiling),
        },
    }


def select_guarded_validation_schedule(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_rows: Sequence[Mapping[str, Any]] = (),
    max_internal_after_098: float = DEFAULT_GEOMETRY_MAX_INTERNAL_AFTER_098,
    min_interval_floor: float = DEFAULT_GEOMETRY_MIN_INTERVAL,
    max_interval_ceiling: float = DEFAULT_GEOMETRY_MAX_INTERVAL,
    allow_risky_selection: bool = False,
) -> Dict[str, Any]:
    base = select_best_validation_schedule(rows, reference_rows=reference_rows)
    fixed_keys = set(BASELINE_SCHEDULE_KEYS).union({SER_PTG_SCHEDULE_KEY})
    candidate_rows = [dict(row) for row in rows if str(row.get("scheduler_key", "")) not in fixed_keys]
    by_schedule: Dict[str, List[Mapping[str, Any]]] = {}
    for row in candidate_rows:
        by_schedule.setdefault(str(row["scheduler_key"]), []).append(row)
    guarded_table = []
    for item in base["schedule_table"]:
        schedule_key = str(item["scheduler_key"])
        geometry = geometry_guard_for_rows(
            by_schedule.get(schedule_key, []),
            max_internal_after_098=float(max_internal_after_098),
            min_interval_floor=float(min_interval_floor),
            max_interval_ceiling=float(max_interval_ceiling),
        )
        guarded = dict(item)
        guarded.update(geometry)
        guarded_table.append(guarded)
    selected = next((row for row in guarded_table if bool(row["passes_geometry_guard"])), None)
    if selected is None:
        if not bool(allow_risky_selection):
            raise ValueError("No validation candidate passed the geometry guard.")
        selected = guarded_table[0]
    selection = dict(base)
    selection["selection_unit"] = "generated_schedule_key_with_geometry_guard"
    selection["unguarded_top_schedule_key"] = str(guarded_table[0]["scheduler_key"])
    selection["selected_schedule_key"] = str(selected["scheduler_key"])
    selection["selected_opd_step_budget"] = selected.get("opd_step_budget")
    selection["geometry_guard"] = {
        "max_internal_fraction_after_098": float(max_internal_after_098),
        "min_interval": float(min_interval_floor),
        "max_interval": float(max_interval_ceiling),
        "allow_risky_selection": bool(allow_risky_selection),
    }
    selection["selected_geometry"] = {
        key: selected[key]
        for key in ("passes_geometry_guard", "geometry_risk_flags", "max_internal_fraction_after_098", "min_interval", "max_interval")
    }
    selection["schedule_table"] = guarded_table
    return selection


def _python_module_command(module: str, args: Sequence[str]) -> List[str]:
    return [sys.executable, "-m", module, *list(args)]


def _run_command(command: Sequence[str], *, cwd: Path, allow_execute: bool, commands: List[List[str]]) -> None:
    commands.append(list(command))
    if allow_execute:
        subprocess.run(list(command), cwd=str(cwd), check=True)


def run_train20_expanded_opd_selection(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = resolve_project_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = resolve_project_path(".")
    allow_execute = bool(args.allow_execute)
    commands: List[List[str]] = []

    baseline_train_dir = out_dir / "baseline_train_tuning"
    ser_reference_dir = out_dir / "ser_ptg_train_tuning_reference"
    active_root = out_dir / "active_rounds"
    final_candidate_dir = out_dir / "final_validation_candidates"
    validation_eval_dir = out_dir / "validation_eval"

    common_eval_args = [
        "--dataset",
        str(args.dataset),
        "--seeds",
        str(args.seeds),
        "--solver_names",
        str(args.solver_names),
        "--target_nfe_values",
        str(args.target_nfe_values),
        "--num_eval_samples",
        str(args.num_eval_samples),
        "--device",
        str(args.device),
    ]
    train_tuning_args = [
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
    teacher_fixed_schedule_keys = _parse_teacher_fixed_schedule_keys(str(args.teacher_fixed_schedule_keys))
    ser_schedule_summary = ser_reference_dir / "ser_ptg_schedule_summary.json"
    preflight_baseline_validation_rows: List[Dict[str, Any]] = []
    if allow_execute:
        preflight_baseline_validation_rows = _validate_baseline_validation_rows(
            _read_rows(args.baseline_validation_rows),
            candidate_rows=(),
            dataset=str(args.dataset),
            seeds=_parse_int_csv(str(args.seeds)),
            solver_names=_parse_csv(str(args.solver_names)),
            target_nfe_values=_parse_int_csv(str(args.target_nfe_values)),
            require_candidate_hash_match=False,
        )

    baseline_train_command_args = [
        "--forecast_datasets",
        str(args.dataset),
        "--conditional_generation_datasets",
        "",
        "--split_phase",
        TRAIN_TUNING_PHASE,
        "--out_root",
        str(baseline_train_dir),
        "--baseline_scheduler_names",
        ",".join(teacher_fixed_schedule_keys),
        *common_eval_args[2:],
        *train_tuning_args,
    ]
    if allow_execute:
        baseline_train_command_args.append("--allow_execute")
    _run_command(
        _python_module_command(
            "genode.evaluation.diffusion_flow_time_reparameterization",
            baseline_train_command_args,
        ),
        cwd=project_root,
        allow_execute=allow_execute,
        commands=commands,
    )
    _run_command(
        _python_module_command(
            "genode.conditional_opd.ser_ptg_reference",
            [
                "--dataset",
                str(args.dataset),
                "--trace_variant",
                "local_defect",
                "--density_floor_eta",
                str(args.ser_density_floor_eta),
                "--reference_split",
                TRAIN_TUNING_PHASE,
                "--seeds",
                str(args.seeds),
                "--solver_names",
                str(args.solver_names),
                "--target_nfe_values",
                str(args.target_nfe_values),
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
                str(ser_reference_dir),
                "--device",
                str(args.device),
            ],
        ),
        cwd=project_root,
        allow_execute=allow_execute,
        commands=commands,
    )
    evaluated_candidate_rows_csvs: List[Path] = []
    selected_candidate_summaries: List[Path] = []
    round_summaries: List[Dict[str, Any]] = []
    for round_idx in range(int(args.active_rounds)):
        round_dir = active_root / f"round_{round_idx:02d}"
        direct_summary_paths: List[Path] = []
        direct_teacher_paths: List[Path] = []
        for student_seed in _parse_int_csv(str(args.direct_student_seeds)):
            seed_dir = round_dir / f"direct_seed_{student_seed}"
            train_args = [
                "--dataset",
                str(args.dataset),
                "--solver_names",
                str(args.solver_names),
                "--target_nfe_values",
                str(args.target_nfe_values),
                "--seeds",
                str(args.seeds),
                "--rows_csv",
                str(baseline_train_dir / "rows.csv"),
                "--reference_schedule_summary",
                str(ser_schedule_summary),
                "--teacher_fixed_schedule_keys",
                ",".join(teacher_fixed_schedule_keys),
                "--required_split_phase",
                TRAIN_TUNING_PHASE,
                "--expected_train_tuning_sampler",
                train_tuning_sampler_key(str(args.train_tuning_sampling_mode)),
                "--expected_train_tuning_fraction",
                str(args.eval_train_fraction),
                "--student_opd_step_values",
                str(args.student_opd_step_values),
                "--teacher_steps",
                str(args.teacher_steps),
                "--student_init_steps",
                str(args.student_init_steps),
                "--late_biased_demo_schedules",
                str(args.late_biased_demo_schedules),
                "--late_biased_demo_weight",
                str(args.late_biased_demo_weight),
                "--student_ser_ptg_regularizer",
                str(args.student_ser_ptg_regularizer),
                "--student_ser_ptg_regularization_weight",
                str(args.student_ser_ptg_regularization_weight),
                "--student_ser_ptg_regularization_eps",
                str(args.student_ser_ptg_regularization_eps),
                "--schedule_key_prefix",
                f"conditional_opd_r{round_idx}_seed{student_seed}_steps",
                "--active_round",
                str(round_idx),
                "--seed",
                str(student_seed),
                "--out_dir",
                str(seed_dir),
            ]
            if evaluated_candidate_rows_csvs:
                train_args.extend(["--candidate_rows_csv", ",".join(str(path) for path in evaluated_candidate_rows_csvs)])
                train_args.extend(["--candidate_schedule_summary", ",".join(str(path) for path in selected_candidate_summaries)])
            if bool(args.include_diagnostic_budgets):
                train_args.append("--include_diagnostic_budgets")
            _run_command(
                _python_module_command("genode.conditional_opd.train_conditional_opd", train_args),
                cwd=project_root,
                allow_execute=allow_execute,
                commands=commands,
            )
            direct_summary_paths.append(seed_dir / "student_budget_schedule_summary.json")
            direct_teacher_paths.append(seed_dir / "conditional_opd.pt")
        pool_summary = round_dir / "candidate_pool_schedule_summary.json"
        selected_pool_summary = round_dir / "selected_train_tuning_candidate_schedule_summary.json"
        _run_command(
            _python_module_command(
                "genode.conditional_opd.candidate_pool",
                [
                    "--source_schedule_summaries",
                    ",".join(str(path) for path in direct_summary_paths),
                    "--teacher_checkpoint_paths",
                    ",".join(str(path) for path in direct_teacher_paths),
                    "--out_path",
                    str(pool_summary),
                    "--selected_out_path",
                    str(selected_pool_summary),
                    "--active_round",
                    str(round_idx),
                    "--seed",
                    str(args.train_tuning_seed + round_idx),
                    "--temperature_values",
                    str(args.temperature_values),
                    "--logit_noise_values",
                    str(args.logit_noise_values),
                    "--dirichlet_student_alpha_values",
                    str(args.dirichlet_student_alpha_values),
                    "--random_dirichlet_alpha_values",
                    str(args.random_dirichlet_alpha_values),
                    "--exploit_count",
                    str(args.round_exploit_count),
                    "--diverse_count",
                    str(args.round_diverse_count),
                    "--random_count",
                    str(args.round_random_count),
                ],
            ),
            cwd=project_root,
            allow_execute=allow_execute,
            commands=commands,
        )
        train_eval_dir = round_dir / "selected_train_tuning_eval"
        _run_command(
            _python_module_command(
                "genode.conditional_opd.evaluate_schedule_summary",
                [
                    "--schedule_summary",
                    str(selected_pool_summary),
                    "--split_phase",
                    TRAIN_TUNING_PHASE,
                    "--out_dir",
                    str(train_eval_dir),
                    *common_eval_args,
                    *train_tuning_args,
                ],
            ),
            cwd=project_root,
            allow_execute=allow_execute,
            commands=commands,
        )
        evaluated_candidate_rows_csvs.append(train_eval_dir / "train_tuning_rows.csv")
        selected_candidate_summaries.append(selected_pool_summary)
        round_summaries.append(
            {
                "active_round": int(round_idx),
                "direct_student_seeds": _parse_int_csv(str(args.direct_student_seeds)),
                "direct_schedule_summaries": [str(path) for path in direct_summary_paths],
                "direct_teacher_checkpoints": [str(path) for path in direct_teacher_paths],
                "candidate_pool_schedule_summary": str(pool_summary),
                "selected_train_tuning_candidate_schedule_summary": str(selected_pool_summary),
                "train_tuning_rows": str(train_eval_dir / "train_tuning_rows.csv"),
            }
        )

    final_candidate_summary = final_candidate_dir / "validation_candidate_schedule_summary.json"
    _run_command(
        _python_module_command(
            "genode.conditional_opd.candidate_pool",
            [
                "--source_schedule_summaries",
                ",".join(str(path) for path in selected_candidate_summaries),
                "--out_path",
                str(final_candidate_dir / "candidate_pool_schedule_summary.json"),
                "--selected_out_path",
                str(final_candidate_summary),
                "--active_round",
                str(int(args.active_rounds)),
                "--seed",
                str(args.train_tuning_seed + 10_000),
                "--temperature_values",
                "",
                "--logit_noise_values",
                "",
                "--dirichlet_student_alpha_values",
                "",
                "--random_dirichlet_alpha_values",
                "",
                "--selection_rows_csv",
                ",".join(str(path) for path in evaluated_candidate_rows_csvs),
                "--fixed_reference_rows_csv",
                str(baseline_train_dir / "rows.csv"),
                "--exploit_count",
                str(args.final_exploit_count),
                "--diverse_count",
                str(args.final_diverse_count),
                "--random_count",
                str(args.final_random_count),
            ],
        ),
        cwd=project_root,
        allow_execute=allow_execute,
        commands=commands,
    )
    _run_command(
        _python_module_command(
            "genode.conditional_opd.evaluate_schedule_summary",
            [
                "--schedule_summary",
                str(final_candidate_summary),
                "--split_phase",
                VALIDATION_PHASE,
                "--out_dir",
                str(validation_eval_dir),
                "--row_csv_name",
                "validation_rows.csv",
                *common_eval_args,
                "--eval_windows_val",
                str(args.eval_windows_val),
            ],
        ),
        cwd=project_root,
        allow_execute=allow_execute,
        commands=commands,
    )

    summary: Dict[str, Any] = {
        "status": "ready" if allow_execute else "dry_run",
        "dataset": str(args.dataset),
        "teacher_metric_split": TRAIN_TUNING_PHASE,
        "teacher_train_fraction": float(args.eval_train_fraction),
        "teacher_train_sampling": train_tuning_sampler_key(str(args.train_tuning_sampling_mode)),
        "train_tuning_strata": int(args.train_tuning_strata),
        "train_tuning_sampling_mode": str(args.train_tuning_sampling_mode),
        "train_tuning_train_split_fraction": float(args.train_tuning_train_split_fraction),
        "train_tuning_val_split_fraction": float(args.train_tuning_val_split_fraction),
        "selection_split": VALIDATION_PHASE,
        "teacher_fixed_schedule_keys": list(teacher_fixed_schedule_keys),
        "final_baseline_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "late_biased_demo_schedules": _parse_csv(str(args.late_biased_demo_schedules)),
        "late_biased_demo_weight": float(args.late_biased_demo_weight),
        "reward_scaling": "teacher_fixed_crps_mase",
        "ser_reference_split": TRAIN_TUNING_PHASE,
        "ser_ptg_usage": (
            "student_initialization_and_regularization"
            if str(args.student_ser_ptg_regularizer) != "none" and float(args.student_ser_ptg_regularization_weight) > 0.0
            else "student_initialization_only"
        ),
        "student_ser_ptg_regularization": {
            "mode": str(args.student_ser_ptg_regularizer),
            "weight": float(args.student_ser_ptg_regularization_weight),
            "eps": float(args.student_ser_ptg_regularization_eps),
            "reference_schedule_key": SER_PTG_SCHEDULE_KEY,
        },
        "baseline_validation_rows": str(args.baseline_validation_rows),
        "active_rounds": int(args.active_rounds),
        "direct_student_seeds": _parse_int_csv(str(args.direct_student_seeds)),
        "student_opd_step_values": _parse_int_csv(str(args.student_opd_step_values)),
        "rounds": round_summaries,
        "final_validation_candidate_summary": str(final_candidate_summary),
        "commands": commands,
        "allow_execute": allow_execute,
    }
    if not allow_execute:
        save_json(summary, str(out_dir / "train20_expanded_opd_selection_dry_run.json"))
        return summary

    validation_rows = _read_rows(validation_eval_dir / "validation_rows.csv")
    _write_rows(out_dir / "validation_rows.csv", validation_rows)
    baseline_validation_rows = _validate_baseline_validation_rows(
        preflight_baseline_validation_rows or _read_rows(args.baseline_validation_rows),
        candidate_rows=validation_rows,
        dataset=str(args.dataset),
        seeds=_parse_int_csv(str(args.seeds)),
        solver_names=_parse_csv(str(args.solver_names)),
        target_nfe_values=_parse_int_csv(str(args.target_nfe_values)),
    )
    selection = select_guarded_validation_schedule(
        validation_rows,
        reference_rows=baseline_validation_rows,
        max_internal_after_098=float(args.geometry_max_internal_after_098),
        min_interval_floor=float(args.geometry_min_interval),
        max_interval_ceiling=float(args.geometry_max_interval),
        allow_risky_selection=bool(args.allow_risky_selection),
    )
    save_json(selection, str(out_dir / "validation_selection.json"))
    selected_summary = write_selected_schedule_summary(
        final_candidate_summary,
        selection,
        out_dir / "selected_schedule_summary.json",
    )
    summary.update(
        {
            "validation_rows": str(out_dir / "validation_rows.csv"),
            "baseline_validation_rows": str(resolve_project_path(str(args.baseline_validation_rows))),
            "baseline_validation_reference_rows": int(len(baseline_validation_rows)),
            "validation_selection": str(out_dir / "validation_selection.json"),
            "selected_schedule_summary": str(out_dir / "selected_schedule_summary.json"),
            "selected_schedule_key": str(selection["selected_schedule_key"]),
            "selected_geometry": selection.get("selected_geometry"),
            "selected_summary_schedule_count": int(len(selected_summary.get("schedules", []))),
        }
    )
    if not bool(args.skip_locked_test):
        locked_args = [
            "--schedule_summary",
            str(out_dir / "selected_schedule_summary.json"),
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
            *common_eval_args,
            "--eval_windows_test",
            str(args.eval_windows_test),
        ]
        if str(args.ser_test_rows).strip():
            locked_args.extend(["--comparator_rows", str(args.ser_test_rows)])
        _run_command(
            _python_module_command("genode.conditional_opd.evaluate_schedule_summary", locked_args),
            cwd=project_root,
            allow_execute=allow_execute,
            commands=commands,
        )
        summary["locked_test_rows"] = str(out_dir / "locked_test_rows.csv")
        summary["locked_test_comparison_summary"] = str(out_dir / "locked_test_comparison_summary.json")
        summary["locked_test_ser_comparator_rows"] = str(args.ser_test_rows) if str(args.ser_test_rows).strip() else ""
    save_json(summary, str(out_dir / "train20_expanded_opd_selection_summary.json"))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Train20 stratified-hash teacher expansion with validation-only selection.")
    parser.add_argument("--dataset", default="san_francisco_traffic")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--solver_names", default="euler,heun,midpoint_rk2,dpmpp2m")
    parser.add_argument("--target_nfe_values", default="4,8,12")
    parser.add_argument("--out_dir", default=str(project_outputs_root() / "legacy_train20_expanded_opd_selection"))
    parser.add_argument("--baseline_validation_rows", default=str(DEFAULT_BASELINE_VALIDATION_ROWS))
    parser.add_argument("--baseline_test_rows", default=str(project_outputs_root() / "diffusion_flow_time_reparameterization_full_test" / "rows.csv"))
    parser.add_argument("--ser_test_rows", default="")
    parser.add_argument("--teacher_steps", type=int, default=500)
    parser.add_argument("--student_init_steps", type=int, default=500)
    parser.add_argument("--active_rounds", type=int, default=2)
    parser.add_argument("--direct_student_seeds", default="0,1,2,3")
    parser.add_argument("--student_opd_step_values", default="5,10,15,20,25")
    parser.add_argument("--include_diagnostic_budgets", action="store_true", default=False)
    parser.add_argument("--temperature_values", default="0.85,1.15")
    parser.add_argument("--logit_noise_values", default="0.05,0.10")
    parser.add_argument("--dirichlet_student_alpha_values", default="100,200")
    parser.add_argument("--random_dirichlet_alpha_values", default="1,2,5")
    parser.add_argument("--round_exploit_count", type=int, default=8)
    parser.add_argument("--round_diverse_count", type=int, default=5)
    parser.add_argument("--round_random_count", type=int, default=3)
    parser.add_argument("--final_exploit_count", type=int, default=6)
    parser.add_argument("--final_diverse_count", type=int, default=4)
    parser.add_argument("--final_random_count", type=int, default=2)
    parser.add_argument("--ser_density_floor_eta", type=float, default=0.05)
    parser.add_argument("--teacher_fixed_schedule_keys", default=",".join(DEFAULT_TEACHER_FIXED_SCHEDULE_KEYS))
    parser.add_argument("--late_biased_demo_schedules", default="late_power_3")
    parser.add_argument("--late_biased_demo_weight", type=float, default=1.0)
    parser.add_argument("--student_ser_ptg_regularizer", choices=("none", "js", "kl"), default="none")
    parser.add_argument("--student_ser_ptg_regularization_weight", type=float, default=0.0)
    parser.add_argument("--student_ser_ptg_regularization_eps", type=float, default=1e-8)
    parser.add_argument("--eval_train_fraction", type=float, default=0.20)
    parser.add_argument("--train_tuning_seed", type=int, default=0)
    parser.add_argument("--train_tuning_strata", type=int, default=20)
    parser.add_argument("--train_tuning_sampling_mode", choices=TRAIN_TUNING_SAMPLING_MODES, default=TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED)
    parser.add_argument("--train_tuning_train_split_fraction", type=float, default=DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION)
    parser.add_argument("--train_tuning_val_split_fraction", type=float, default=DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION)
    parser.add_argument("--num_eval_samples", type=int, default=5)
    parser.add_argument("--eval_windows_val", type=int, default=0)
    parser.add_argument("--eval_windows_test", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--geometry_max_internal_after_098", type=float, default=DEFAULT_GEOMETRY_MAX_INTERNAL_AFTER_098)
    parser.add_argument("--geometry_min_interval", type=float, default=DEFAULT_GEOMETRY_MIN_INTERVAL)
    parser.add_argument("--geometry_max_interval", type=float, default=DEFAULT_GEOMETRY_MAX_INTERVAL)
    parser.add_argument("--allow_risky_selection", action="store_true", default=False)
    parser.add_argument("--skip_locked_test", action="store_true", default=False)
    parser.add_argument("--allow_execute", action="store_true", default=False)
    return parser


def main() -> None:
    summary = run_train20_expanded_opd_selection(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
