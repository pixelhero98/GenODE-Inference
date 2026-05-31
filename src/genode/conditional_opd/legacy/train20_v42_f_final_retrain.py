from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from genode.conditional_opd.legacy.train20_selection import _parse_csv, _parse_int_csv
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

DEFAULT_OUT_DIR = project_outputs_root() / "train20_v42_f_calibration20_fullval_uniform_seedmean"
DEFAULT_BASELINE_VAL_ROWS = project_outputs_root() / "diffusion_flow_time_reparameterization_full_val" / "rows.csv"
DEFAULT_BASELINE_TEST_ROWS = project_outputs_root() / "diffusion_flow_time_reparameterization_full_test" / "rows.csv"


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


def _archive_existing_root(path: Path, *, allow_execute: bool, commands: List[List[str]]) -> str:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return ""
    archive_root = project_outputs_root() / "archive" / "v42_f_clean_start"
    target = archive_root / resolved.name
    suffix = 0
    while target.exists():
        suffix += 1
        target = archive_root / f"{resolved.name}_{suffix:02d}"
    commands.append(["archive_output_root", str(resolved), str(target)])
    if allow_execute:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved), str(target))
    return str(target)


def run_train20_v42_f_final_retrain(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = resolve_project_path(str(args.out_dir))
    project_root = resolve_project_path(".")
    allow_execute = bool(args.allow_execute)
    commands: List[List[str]] = []

    bo_eval_seed_values = _parse_int_csv(str(args.bo_eval_seeds))
    if bo_eval_seed_values != [0, 1]:
        raise ValueError("V4.2-F keeps BO/train calibration eval seeds exactly 0,1.")
    calibration_val_seed_values = _parse_int_csv(str(args.calibration_val_seeds))
    if calibration_val_seed_values != [0, 1, 2]:
        raise ValueError("V4.2-F uses former-validation calibration seeds exactly 0,1,2.")
    locked_test_seed_values = _parse_int_csv(str(args.locked_test_seeds))
    if locked_test_seed_values != [0, 1, 2]:
        raise ValueError("V4.2-F locked test seeds must be 0,1,2.")
    student_seed_values = _parse_int_csv(str(args.student_seeds))
    if student_seed_values != [0, 1, 2]:
        raise ValueError("V4.2-F student seeds must be the fixed set 0,1,2.")
    if str(args.train_tuning_sampling_mode) != TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED:
        raise ValueError("V4.2-F uses validation_normalized train-tuning sampling.")
    if abs(float(args.eval_train_fraction) - 0.20) > 1e-12:
        raise ValueError("V4.2-F default run uses eval_train_fraction=0.20.")

    archived_existing = ""
    if bool(args.clean_output_root):
        archived_existing = _archive_existing_root(out_dir, allow_execute=allow_execute, commands=commands)
    out_dir.mkdir(parents=True, exist_ok=True)

    bo_eval_seeds = ",".join(str(x) for x in bo_eval_seed_values)
    calibration_val_seeds = ",".join(str(x) for x in calibration_val_seed_values)
    locked_test_seeds = ",".join(str(x) for x in locked_test_seed_values)
    student_seeds = ",".join(str(x) for x in student_seed_values)

    baseline_train_dir = out_dir / "baseline_calibration_train"
    ser_reference_dir = out_dir / "ser_ptg_train_tuning_reference"
    bo_root = out_dir / "bo_rounds"
    neural_policy_dir = out_dir / "neural_policy"
    ser_schedule_summary = ser_reference_dir / "ser_ptg_schedule_summary.json"
    selected_schedule_summary = neural_policy_dir / "selected_schedule_summary.json"

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

    observed_train_rows_csvs: List[Path] = []
    observed_val_rows_csvs: List[Path] = []
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
        if observed_train_rows_csvs:
            generate_args.extend(["--observed_schedule_summaries", ",".join(str(path) for path in observed_schedule_summaries)])
            generate_args.extend(["--observed_train_rows_csv", ",".join(str(path) for path in observed_train_rows_csvs)])
            generate_args.extend(["--observed_validation_rows_csv", ",".join(str(path) for path in observed_val_rows_csvs)])
            generate_args.extend(["--fixed_train_rows_csv", str(baseline_train_dir / "rows.csv")])
            generate_args.extend(["--fixed_validation_rows_csv", str(args.baseline_validation_rows)])
        _run_command(
            _python_module_command("genode.conditional_opd.bo_candidate_pool", generate_args),
            cwd=project_root,
            allow_execute=allow_execute,
            commands=commands,
        )

        train_eval_dir = round_dir / "bo_calibration_train_eval"
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
                    "--row_csv_name",
                    "train_tuning_rows.csv",
                    *_common_eval_args(args, seeds=bo_eval_seeds),
                    *_train_tuning_args(args),
                ],
            ),
            cwd=project_root,
            allow_execute=allow_execute,
            commands=commands,
        )

        val_eval_dir = round_dir / "bo_calibration_val_eval"
        _run_command(
            _python_module_command(
                "genode.conditional_opd.evaluate_schedule_summary",
                [
                    "--schedule_summary",
                    str(pool_summary),
                    "--split_phase",
                    VALIDATION_PHASE,
                    "--out_dir",
                    str(val_eval_dir),
                    "--row_csv_name",
                    "validation_rows.csv",
                    *_common_eval_args(args, seeds=calibration_val_seeds),
                    "--eval_windows_val",
                    str(args.eval_windows_val),
                ],
            ),
            cwd=project_root,
            allow_execute=allow_execute,
            commands=commands,
        )
        observed_train_rows_csvs.append(train_eval_dir / "train_tuning_rows.csv")
        observed_val_rows_csvs.append(val_eval_dir / "validation_rows.csv")
        observed_schedule_summaries.append(pool_summary)
        round_summaries.append(
            {
                "bo_round": int(round_idx),
                "bo_candidate_schedule_summary": str(pool_summary),
                "bo_calibration_train_rows": str(train_eval_dir / "train_tuning_rows.csv"),
                "bo_calibration_val_rows": str(val_eval_dir / "validation_rows.csv"),
            }
        )

    _run_command(
        _python_module_command(
            "genode.conditional_opd.legacy.train_conditional_opd_v42_f",
            [
                "--dataset",
                str(args.dataset),
                "--solver_names",
                str(args.solver_names),
                "--target_nfe_values",
                str(args.target_nfe_values),
                "--train_rows_csv",
                str(baseline_train_dir / "rows.csv"),
                "--validation_rows_csv",
                str(args.baseline_validation_rows),
                "--candidate_train_rows_csv",
                ",".join(str(path) for path in observed_train_rows_csvs),
                "--candidate_validation_rows_csv",
                ",".join(str(path) for path in observed_val_rows_csvs),
                "--candidate_schedule_summary",
                ",".join(str(path) for path in observed_schedule_summaries),
                "--reference_schedule_summary",
                str(ser_schedule_summary),
                "--calibration_train_seeds",
                bo_eval_seeds,
                "--calibration_val_seeds",
                calibration_val_seeds,
                "--teacher_fixed_schedule_keys",
                str(args.teacher_fixed_schedule_keys),
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
                "--student_seeds",
                student_seeds,
                "--student_checkpoint_modes",
                str(args.student_checkpoint_modes),
                "--fixed_epoch_steps",
                str(args.fixed_epoch_steps),
                "--final_checkpoint_mode",
                str(args.final_checkpoint_mode),
                "--lr",
                str(args.lr),
                "--seed",
                str(args.student_base_seed),
            ],
        ),
        cwd=project_root,
        allow_execute=allow_execute,
        commands=commands,
    )

    if not bool(args.skip_locked_test):
        _run_command(
            _python_module_command(
                "genode.conditional_opd.evaluate_schedule_summary",
                [
                    "--schedule_summary",
                    str(selected_schedule_summary),
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

    summary: Dict[str, Any] = {
        "status": "ready" if allow_execute else "dry_run",
        "artifact": "train20_v42_f_final_calibration_retrain",
        "dataset": str(args.dataset),
        "protocol": "v4.2_f_final_calibration_set",
        "archived_existing_output_root": archived_existing,
        "out_dir": str(out_dir),
        "bo_eval_seeds": bo_eval_seed_values,
        "calibration_val_seeds": calibration_val_seed_values,
        "locked_test_seeds": locked_test_seed_values,
        "student_seeds": student_seed_values,
        "eval_train_fraction": float(args.eval_train_fraction),
        "train_tuning_sampling_mode": str(args.train_tuning_sampling_mode),
        "uses_validation_selection": False,
        "student_initialization": "uniform",
        "student_checkpoint_modes": _parse_csv(str(args.student_checkpoint_modes)),
        "final_checkpoint_mode": str(args.final_checkpoint_mode),
        "fixed_epoch_steps": int(args.fixed_epoch_steps),
        "teacher_fixed_schedule_keys": str(args.teacher_fixed_schedule_keys),
        "reward_reference_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "bo_rounds": int(args.bo_rounds),
        "bo_candidates_per_round": int(args.bo_candidates_per_round),
        "density_grid_size": int(args.density_grid_size),
        "rounds": round_summaries,
        "neural_policy_dir": str(neural_policy_dir),
        "selected_schedule_summary": str(selected_schedule_summary),
        "locked_test_rows": str(out_dir / "locked_test_rows.csv") if not bool(args.skip_locked_test) else "",
        "locked_test_comparison_summary": str(out_dir / "locked_test_comparison_summary.json") if not bool(args.skip_locked_test) else "",
        "commands": commands,
        "allow_execute": allow_execute,
    }
    save_json(summary, str(out_dir / "train20_v42_f_final_retrain_summary.json"))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Train20 V4.2-F final calibration-set retrain.")
    parser.add_argument("--dataset", default="san_francisco_traffic")
    parser.add_argument("--solver_names", default="euler,heun,midpoint_rk2,dpmpp2m")
    parser.add_argument("--target_nfe_values", default="4,8,12")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--bo_eval_seeds", default="0,1")
    parser.add_argument("--calibration_val_seeds", default="0,1,2")
    parser.add_argument("--locked_test_seeds", default="0,1,2")
    parser.add_argument("--student_seeds", default="0,1,2")
    parser.add_argument("--baseline_validation_rows", default=str(DEFAULT_BASELINE_VAL_ROWS))
    parser.add_argument("--baseline_test_rows", default=str(DEFAULT_BASELINE_TEST_ROWS))
    parser.add_argument("--bo_rounds", type=int, default=2)
    parser.add_argument("--bo_candidates_per_round", type=int, default=16)
    parser.add_argument("--bo_theta_bound", type=float, default=2.5)
    parser.add_argument("--bo_sobol_pool", type=int, default=512)
    parser.add_argument("--density_grid_size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--ser_density_floor_eta", type=float, default=0.05)
    parser.add_argument("--teacher_fixed_schedule_keys", default="none")
    parser.add_argument("--teacher_steps", type=int, default=500)
    parser.add_argument("--bo_teacher_holdout_fraction", type=float, default=0.20)
    parser.add_argument("--student_init_steps", type=int, default=500)
    parser.add_argument("--student_opd_step_values", default="5,10,15,20,25")
    parser.add_argument("--student_checkpoint_modes", default="fixed_epoch,lowest_internal_loss")
    parser.add_argument("--fixed_epoch_steps", type=int, default=20)
    parser.add_argument("--final_checkpoint_mode", choices=("fixed_epoch", "lowest_internal_loss"), default="lowest_internal_loss")
    parser.add_argument("--student_base_seed", type=int, default=0)
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
    parser.add_argument("--skip_locked_test", action="store_true", default=False)
    parser.add_argument("--clean_output_root", action="store_true", default=True)
    parser.add_argument("--no_clean_output_root", action="store_false", dest="clean_output_root")
    parser.add_argument("--allow_execute", action="store_true", default=False)
    return parser


def main() -> None:
    summary = run_train20_v42_f_final_retrain(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
