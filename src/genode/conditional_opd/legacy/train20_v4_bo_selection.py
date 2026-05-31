from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from genode.conditional_opd.evaluate_schedule_summary import write_selected_schedule_summary
from genode.conditional_opd.train_conditional_opd import DEFAULT_DIRECT_OPD_BUDGETS, DEFAULT_TEACHER_FIXED_SCHEDULE_KEYS, STUDENT_SCHEDULE_PREFIX
from genode.conditional_opd.legacy.train20_selection import (
    DEFAULT_BASELINE_VALIDATION_ROWS,
    DEFAULT_GEOMETRY_MAX_INTERNAL_AFTER_098,
    DEFAULT_GEOMETRY_MAX_INTERVAL,
    DEFAULT_GEOMETRY_MIN_INTERVAL,
    _parse_csv,
    _parse_int_csv,
    _read_rows,
    _validate_baseline_validation_rows,
    _write_rows,
    select_guarded_validation_schedule,
)
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


DEFAULT_OUT_DIR = project_outputs_root() / "train20_v4_bo_neural_composite_2seed_valnorm_paired"


def _python_module_command(module: str, args: Sequence[str]) -> List[str]:
    return [sys.executable, "-m", module, *list(args)]


def _run_command(command: Sequence[str], *, cwd: Path, allow_execute: bool, commands: List[List[str]]) -> None:
    commands.append(list(command))
    if allow_execute:
        subprocess.run(list(command), cwd=str(cwd), check=True)


def _common_eval_args(args: argparse.Namespace, *, seeds: str) -> List[str]:
    return [
        "--dataset",
        str(args.dataset),
        "--seeds",
        str(seeds),
        "--solver_names",
        str(args.solver_names),
        "--target_nfe_values",
        str(args.target_nfe_values),
        "--num_eval_samples",
        str(args.num_eval_samples),
        "--device",
        str(args.device),
    ]


def _train_tuning_args(args: argparse.Namespace) -> List[str]:
    return [
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


def run_train20_v4_bo_selection(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = resolve_project_path(str(args.out_dir))
    project_root = resolve_project_path(".")
    allow_execute = bool(args.allow_execute)
    commands: List[List[str]] = []

    bo_eval_seed_values = _parse_int_csv(str(args.bo_eval_seeds))
    if bo_eval_seed_values != [0, 1]:
        raise ValueError("Train20 v4 BO requires exactly BO eval seeds 0,1.")
    bo_eval_seeds = ",".join(str(x) for x in bo_eval_seed_values)
    selection_seeds = ",".join(str(x) for x in _parse_int_csv(str(args.selection_seeds)))
    locked_test_seeds = ",".join(str(x) for x in _parse_int_csv(str(args.locked_test_seeds)))
    teacher_fixed_schedule_keys = _parse_csv(str(args.teacher_fixed_schedule_keys))
    if teacher_fixed_schedule_keys != list(DEFAULT_TEACHER_FIXED_SCHEDULE_KEYS):
        raise ValueError(
            "Train20 v4 neural BO requires teacher fixed demos "
            f"{','.join(DEFAULT_TEACHER_FIXED_SCHEDULE_KEYS)}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_train_dir = out_dir / "baseline_train_tuning"
    ser_reference_dir = out_dir / "ser_ptg_train_tuning_reference"
    bo_root = out_dir / "bo_rounds"
    neural_policy_dir = out_dir / "neural_policy"
    validation_eval_dir = out_dir / "validation_eval"
    ser_schedule_summary = ser_reference_dir / "ser_ptg_schedule_summary.json"
    student_budget_schedule_summary = neural_policy_dir / "student_budget_schedule_summary.json"
    conditional_opd_summary = neural_policy_dir / "conditional_opd_summary.json"

    preflight_baseline_validation_rows: List[Dict[str, Any]] = []
    if allow_execute:
        preflight_baseline_validation_rows = _validate_baseline_validation_rows(
            _read_rows(args.baseline_validation_rows),
            candidate_rows=(),
            dataset=str(args.dataset),
            seeds=_parse_int_csv(selection_seeds),
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
        ",".join(BASELINE_SCHEDULE_KEYS),
        *_common_eval_args(args, seeds=bo_eval_seeds)[2:],
        *_train_tuning_args(args),
    ]
    if allow_execute:
        baseline_train_command_args.append("--allow_execute")
    _run_command(
        _python_module_command("genode.evaluation.diffusion_flow_time_reparameterization", baseline_train_command_args),
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
                bo_eval_seeds,
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

    observed_rows_csvs: List[Path] = []
    observed_schedule_summaries: List[Path] = []
    round_summaries: List[Dict[str, Any]] = []
    for round_idx in range(int(args.bo_rounds)):
        round_dir = bo_root / f"round_{round_idx:02d}"
        pool_summary = round_dir / "bo_candidate_schedule_summary.json"
        generate_args = [
            "--mode",
            "generate",
            "--reference_schedule_summary",
            str(ser_schedule_summary),
            "--out_path",
            str(pool_summary),
            "--active_round",
            str(round_idx),
            "--seed",
            str(int(args.train_tuning_seed) + round_idx),
            "--candidate_count",
            str(args.bo_candidates_per_round),
            "--density_grid_size",
            str(args.density_grid_size),
            "--theta_bound",
            str(args.bo_theta_bound),
            "--sobol_pool",
            str(args.bo_sobol_pool),
            "--fixed_reference_rows_csv",
            str(baseline_train_dir / "rows.csv"),
        ]
        if observed_rows_csvs:
            generate_args.extend(["--observed_rows_csv", ",".join(str(path) for path in observed_rows_csvs)])
            generate_args.extend(["--observed_schedule_summaries", ",".join(str(path) for path in observed_schedule_summaries)])
        _run_command(
            _python_module_command("genode.conditional_opd.bo_candidate_pool", generate_args),
            cwd=project_root,
            allow_execute=allow_execute,
            commands=commands,
        )
        train_eval_dir = round_dir / "bo_train_tuning_eval"
        _run_command(
            _python_module_command(
                "genode.conditional_opd.evaluate_schedule_summary",
                [
                    "--schedule_summary",
                    str(pool_summary),
                    "--split_phase",
                    TRAIN_TUNING_PHASE,
                    "--out_dir",
                    str(train_eval_dir),
                    *_common_eval_args(args, seeds=bo_eval_seeds),
                    *_train_tuning_args(args),
                ],
            ),
            cwd=project_root,
            allow_execute=allow_execute,
            commands=commands,
        )
        observed_rows_csvs.append(train_eval_dir / "train_tuning_rows.csv")
        observed_schedule_summaries.append(pool_summary)
        round_summaries.append(
            {
                "bo_round": int(round_idx),
                "bo_candidate_schedule_summary": str(pool_summary),
                "bo_train_tuning_rows": str(train_eval_dir / "train_tuning_rows.csv"),
                "expected_bo_train_tuning_rows": int(args.bo_candidates_per_round)
                * len(_parse_csv(str(args.solver_names)))
                * len(_parse_int_csv(str(args.target_nfe_values)))
                * len(_parse_int_csv(bo_eval_seeds)),
            }
        )

    _run_command(
        _python_module_command(
            "genode.conditional_opd.train_conditional_opd",
            [
                "--dataset",
                str(args.dataset),
                "--solver_names",
                str(args.solver_names),
                "--target_nfe_values",
                str(args.target_nfe_values),
                "--seeds",
                bo_eval_seeds,
                "--rows_csv",
                str(baseline_train_dir / "rows.csv"),
                "--reference_schedule_summary",
                str(ser_schedule_summary),
                "--teacher_fixed_schedule_keys",
                ",".join(teacher_fixed_schedule_keys),
                "--reward_reference_schedule_keys",
                ",".join(BASELINE_SCHEDULE_KEYS),
                "--candidate_rows_csv",
                ",".join(str(path) for path in observed_rows_csvs),
                "--candidate_schedule_summary",
                ",".join(str(path) for path in observed_schedule_summaries),
                "--required_split_phase",
                TRAIN_TUNING_PHASE,
                "--expected_train_tuning_sampler",
                train_tuning_sampler_key(str(args.train_tuning_sampling_mode)),
                "--expected_train_tuning_fraction",
                str(args.eval_train_fraction),
                "--out_dir",
                str(neural_policy_dir),
                "--teacher_steps",
                str(args.teacher_steps),
                "--teacher_diagnostic_holdout_fraction",
                str(args.bo_teacher_holdout_fraction),
                "--student_init_steps",
                str(args.student_init_steps),
                "--student_opd_step_values",
                str(args.student_opd_step_values),
                "--schedule_key_prefix",
                str(args.student_schedule_key_prefix),
                "--active_round",
                str(int(args.bo_rounds)),
                "--late_biased_demo_schedules",
                str(args.late_biased_demo_schedules),
                "--late_biased_demo_weight",
                str(args.late_biased_demo_weight),
                "--student_ser_ptg_regularizer",
                "none",
                "--student_ser_ptg_regularization_weight",
                "0.0",
                "--lr",
                str(args.lr),
                "--seed",
                str(args.student_seed),
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
                str(student_budget_schedule_summary),
                "--split_phase",
                VALIDATION_PHASE,
                "--out_dir",
                str(validation_eval_dir),
                "--row_csv_name",
                "validation_rows.csv",
                *_common_eval_args(args, seeds=selection_seeds),
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
        "artifact": "train20_v4_bo_neural_selection",
        "bo_eval_seeds": _parse_int_csv(bo_eval_seeds),
        "selection_seeds": _parse_int_csv(selection_seeds),
        "locked_test_seeds": _parse_int_csv(locked_test_seeds),
        "bo_objective": "paired_best_fixed_baseline_crps_mase_composite",
        "composite_policy": True,
        "canonical_policy": "neural_conditional_student",
        "density_grid_size": int(args.density_grid_size),
        "bo_candidates_per_round": int(args.bo_candidates_per_round),
        "bo_rounds": int(args.bo_rounds),
        "student_opd_step_values": _parse_int_csv(str(args.student_opd_step_values)),
        "train_tuning_sampling_mode": str(args.train_tuning_sampling_mode),
        "teacher_train_fraction": float(args.eval_train_fraction),
        "ser_ptg_usage": "initialization_base_density_and_diagnostic_only",
        "ser_ptg_regularization": {"mode": "none", "weight": 0.0},
        "baseline_train_tuning_schedules": list(BASELINE_SCHEDULE_KEYS),
        "teacher_fixed_schedule_keys": _parse_csv(str(args.teacher_fixed_schedule_keys)),
        "bo_teacher_holdout_fraction": float(args.bo_teacher_holdout_fraction),
        "teacher_selection_protocol": "v4.2_option_a_bo_heldout_teacher_then_guarded_validation_student",
        "reward_reference_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "baseline_validation_rows": str(args.baseline_validation_rows),
        "rounds": round_summaries,
        "neural_policy_dir": str(neural_policy_dir),
        "conditional_opd_summary": str(conditional_opd_summary),
        "student_budget_schedule_summary": str(student_budget_schedule_summary),
        "final_validation_candidate_summary": str(student_budget_schedule_summary),
        "commands": commands,
        "allow_execute": allow_execute,
    }
    if not allow_execute:
        if not bool(args.skip_locked_test):
            _run_command(
                _python_module_command(
                    "genode.conditional_opd.evaluate_schedule_summary",
                    [
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
                        *_common_eval_args(args, seeds=locked_test_seeds),
                        "--eval_windows_test",
                        str(args.eval_windows_test),
                    ],
                ),
                cwd=project_root,
                allow_execute=False,
                commands=commands,
            )
        save_json(summary, str(out_dir / "train20_v4_bo_selection_dry_run.json"))
        return summary

    validation_rows = _read_rows(validation_eval_dir / "validation_rows.csv")
    _write_rows(out_dir / "validation_rows.csv", validation_rows)
    baseline_validation_rows = _validate_baseline_validation_rows(
        preflight_baseline_validation_rows or _read_rows(args.baseline_validation_rows),
        candidate_rows=validation_rows,
        dataset=str(args.dataset),
        seeds=_parse_int_csv(selection_seeds),
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
        student_budget_schedule_summary,
        selection,
        out_dir / "selected_schedule_summary.json",
    )
    summary.update(
        {
            "validation_rows": str(out_dir / "validation_rows.csv"),
            "baseline_validation_reference_rows": int(len(baseline_validation_rows)),
            "validation_selection": str(out_dir / "validation_selection.json"),
            "selected_schedule_summary": str(out_dir / "selected_schedule_summary.json"),
            "selected_schedule_key": str(selection["selected_schedule_key"]),
            "selected_geometry": selection.get("selected_geometry"),
            "selected_summary_schedule_count": int(len(selected_summary.get("schedules", []))),
        }
    )
    if not bool(args.skip_locked_test):
        _run_command(
            _python_module_command(
                "genode.conditional_opd.evaluate_schedule_summary",
                [
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
                    *_common_eval_args(args, seeds=locked_test_seeds),
                    "--eval_windows_test",
                    str(args.eval_windows_test),
                ],
            ),
            cwd=project_root,
            allow_execute=allow_execute,
            commands=commands,
        )
        summary["locked_test_rows"] = str(out_dir / "locked_test_rows.csv")
        summary["locked_test_comparison_summary"] = str(out_dir / "locked_test_comparison_summary.json")
    save_json(summary, str(out_dir / "train20_v4_bo_selection_summary.json"))
    save_json(summary, str(out_dir / "train20_v4_bo_neural_selection_summary.json"))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Train20 v4 BoTorch BO exploration plus neural conditional policy selection.")
    parser.add_argument("--dataset", default="san_francisco_traffic")
    parser.add_argument("--solver_names", default="euler,heun,midpoint_rk2,dpmpp2m")
    parser.add_argument("--target_nfe_values", default="4,8,12")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--bo_eval_seeds", default="0,1")
    parser.add_argument("--selection_seeds", default="0,1,2")
    parser.add_argument("--locked_test_seeds", default="0,1,2")
    parser.add_argument("--baseline_validation_rows", default=str(DEFAULT_BASELINE_VALIDATION_ROWS))
    parser.add_argument("--baseline_test_rows", default=str(project_outputs_root() / "diffusion_flow_time_reparameterization_full_test" / "rows.csv"))
    parser.add_argument("--bo_rounds", type=int, default=2)
    parser.add_argument("--bo_candidates_per_round", type=int, default=16)
    parser.add_argument("--bo_theta_bound", type=float, default=2.5)
    parser.add_argument("--bo_sobol_pool", type=int, default=512)
    parser.add_argument("--density_grid_size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--ser_density_floor_eta", type=float, default=0.05)
    parser.add_argument("--teacher_fixed_schedule_keys", default=",".join(DEFAULT_TEACHER_FIXED_SCHEDULE_KEYS))
    parser.add_argument("--teacher_steps", type=int, default=500)
    parser.add_argument("--bo_teacher_holdout_fraction", type=float, default=0.20)
    parser.add_argument("--student_init_steps", type=int, default=500)
    parser.add_argument("--student_opd_step_values", default=",".join(str(x) for x in DEFAULT_DIRECT_OPD_BUDGETS))
    parser.add_argument("--student_schedule_key_prefix", default=STUDENT_SCHEDULE_PREFIX)
    parser.add_argument("--student_seed", type=int, default=0)
    parser.add_argument("--late_biased_demo_schedules", default="late_power_3")
    parser.add_argument("--late_biased_demo_weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    summary = run_train20_v4_bo_selection(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
