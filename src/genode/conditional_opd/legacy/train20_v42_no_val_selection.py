from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from genode.conditional_opd.evaluate_schedule_summary import write_selected_schedule_summary
from genode.conditional_opd.legacy.train20_selection import (
    DEFAULT_GEOMETRY_MAX_INTERNAL_AFTER_098,
    DEFAULT_GEOMETRY_MAX_INTERVAL,
    DEFAULT_GEOMETRY_MIN_INTERVAL,
    _parse_csv,
    _parse_int_csv,
)
from genode.data.otflow_paths import project_outputs_root, resolve_project_path
from genode.evaluation.otflow_evaluation_support import LOCKED_TEST_PHASE
from genode.models.otflow_train_val import save_json


DEFAULT_SOURCE_RUN_DIR = project_outputs_root() / "train20_v42_bo_option_a_neural_composite_2seed_valnorm_paired"
DEFAULT_OUT_DIR = project_outputs_root() / "train20_v42_no_val_neural_composite_2seed_valnorm_paired"


def _python_module_command(module: str, args: Sequence[str]) -> List[str]:
    return [sys.executable, "-m", module, *list(args)]


def _run_command(command: Sequence[str], *, cwd: Path, allow_execute: bool, commands: List[List[str]]) -> None:
    commands.append(list(command))
    if allow_execute:
        subprocess.run(list(command), cwd=str(cwd), check=True)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return float(val) if math.isfinite(val) else None


def _prediction_geometry(prediction: Mapping[str, Any]) -> Dict[str, float | None]:
    grid_geometry = prediction.get("grid_geometry")
    if isinstance(grid_geometry, Mapping):
        late = _optional_float(grid_geometry.get("internal_fraction_after_098"))
        min_interval = _optional_float(grid_geometry.get("min_interval"))
        max_interval = _optional_float(grid_geometry.get("max_interval"))
        if late is not None and min_interval is not None and max_interval is not None:
            return {
                "internal_fraction_after_098": late,
                "min_interval": min_interval,
                "max_interval": max_interval,
            }

    raw_grid = prediction.get("time_grid", [])
    try:
        grid = [float(value) for value in raw_grid]
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
        return {
            "passes_geometry_guard": False,
            "geometry_risk_flags": ["missing_predictions"],
            "max_internal_fraction_after_098": None,
            "min_interval": None,
            "max_interval": None,
            "thresholds": {
                "max_internal_fraction_after_098": float(max_internal_after_098),
                "min_interval": float(min_interval_floor),
                "max_interval": float(max_interval_ceiling),
            },
        }
    late_values: List[float] = []
    min_values: List[float] = []
    max_values: List[float] = []
    missing_fields: List[str] = []
    for prediction in predictions:
        geometry = _prediction_geometry(prediction)
        for key, values in (
            ("internal_fraction_after_098", late_values),
            ("min_interval", min_values),
            ("max_interval", max_values),
        ):
            value = geometry.get(key)
            if value is None or not math.isfinite(float(value)):
                missing_fields.append(key)
            else:
                values.append(float(value))
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


def select_no_validation_student_schedule(
    schedule_summary: Mapping[str, Any],
    *,
    conditional_opd_summary: Mapping[str, Any] | None = None,
    max_internal_after_098: float = DEFAULT_GEOMETRY_MAX_INTERNAL_AFTER_098,
    min_interval_floor: float = DEFAULT_GEOMETRY_MIN_INTERVAL,
    max_interval_ceiling: float = DEFAULT_GEOMETRY_MAX_INTERVAL,
    allow_risky_selection: bool = False,
) -> Dict[str, Any]:
    schedules = [dict(schedule) for schedule in schedule_summary.get("schedules", []) or []]
    if not schedules:
        raise ValueError("No student budget schedules are available for no-validation selection.")

    table: List[Dict[str, Any]] = []
    for schedule in schedules:
        prediction_scores = [
            _optional_float(prediction.get("utility"))
            for prediction in list(schedule.get("predictions", []) or [])
        ]
        finite_scores = [float(value) for value in prediction_scores if value is not None]
        if len(finite_scores) != len(prediction_scores) or not finite_scores:
            raise ValueError(f"Schedule {schedule.get('scheduler_key')} is missing finite teacher utility values.")
        score = _optional_float(schedule.get("teacher_predicted_utility_mean"))
        if score is None:
            score = float(sum(finite_scores) / len(finite_scores))
        min_score = float(min(finite_scores))
        geometry = geometry_guard_for_schedule(
            schedule,
            max_internal_after_098=max_internal_after_098,
            min_interval_floor=min_interval_floor,
            max_interval_ceiling=max_interval_ceiling,
        )
        row = {
            "scheduler_key": str(schedule.get("scheduler_key")),
            "opd_step_budget": schedule.get("opd_step_budget"),
            "teacher_predicted_utility_mean": score,
            "min_teacher_predicted_utility": min_score,
            "selection_score": score,
            "selection_score_name": "teacher_predicted_utility_mean",
        }
        row.update(geometry)
        table.append(row)

    table.sort(
        key=lambda row: (
            float("-inf") if row["selection_score"] is None else float(row["selection_score"]),
            float("-inf") if row["min_teacher_predicted_utility"] is None else float(row["min_teacher_predicted_utility"]),
            -int(row["opd_step_budget"] or 0),
            str(row["scheduler_key"]),
        ),
        reverse=True,
    )
    selected = next((row for row in table if bool(row["passes_geometry_guard"])), None)
    if selected is None:
        if not allow_risky_selection:
            raise ValueError("No student budget schedule passed the no-validation geometry guard.")
        selected = table[0]
    teacher_summary = conditional_opd_summary or {}
    selection = {
        "status": "ready",
        "selection_mode": "no_validation_teacher_surrogate",
        "selection_protocol": "v4.2_option_b_no_validation_student_selection",
        "selection_unit": "student_budget_schedule_key_with_geometry_guard",
        "uses_validation_labels_for_selection": False,
        "validation_usage": "not_used_for_selection",
        "selection_score_name": "teacher_predicted_utility_mean",
        "student_selection_rule": "highest_frozen_teacher_predicted_utility_mean_with_min_utility_tiebreak_and_geometry_guard",
        "teacher_selection_protocol": teacher_summary.get("teacher_selection_protocol", ""),
        "teacher_checkpoint_selection": teacher_summary.get("teacher_checkpoint_selection", {}),
        "teacher_diagnostics": teacher_summary.get("teacher_diagnostics", {}),
        "unguarded_top_schedule_key": str(table[0]["scheduler_key"]),
        "selected_schedule_key": str(selected["scheduler_key"]),
        "selected_opd_step_budget": selected.get("opd_step_budget"),
        "selected_score": selected.get("selection_score"),
        "geometry_guard": {
            "max_internal_fraction_after_098": float(max_internal_after_098),
            "min_interval": float(min_interval_floor),
            "max_interval": float(max_interval_ceiling),
            "allow_risky_selection": bool(allow_risky_selection),
        },
        "selected_geometry": {
            key: selected[key]
            for key in ("passes_geometry_guard", "geometry_risk_flags", "max_internal_fraction_after_098", "min_interval", "max_interval")
        },
        "schedule_table": table,
    }
    return selection


def _load_json(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_project_path(str(path))
    return json.loads(resolved.read_text(encoding="utf-8"))


def run_train20_v42_no_val_selection(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = resolve_project_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = resolve_project_path(".")
    allow_execute = bool(args.allow_execute)
    commands: List[List[str]] = []

    source_run_dir = resolve_project_path(str(args.source_run_dir))
    schedule_summary_path = resolve_project_path(str(args.student_budget_schedule_summary or source_run_dir / "neural_policy" / "student_budget_schedule_summary.json"))
    conditional_summary_path = resolve_project_path(str(args.conditional_opd_summary or source_run_dir / "neural_policy" / "conditional_opd_summary.json"))
    schedule_summary = _load_json(schedule_summary_path)
    conditional_summary = _load_json(conditional_summary_path)
    selection = select_no_validation_student_schedule(
        schedule_summary,
        conditional_opd_summary=conditional_summary,
        max_internal_after_098=float(args.geometry_max_internal_after_098),
        min_interval_floor=float(args.geometry_min_interval),
        max_interval_ceiling=float(args.geometry_max_interval),
        allow_risky_selection=bool(args.allow_risky_selection),
    )
    selection.update(
        {
            "source_run_dir": str(source_run_dir),
            "student_budget_schedule_summary": str(schedule_summary_path),
            "conditional_opd_summary": str(conditional_summary_path),
        }
    )
    save_json(selection, str(out_dir / "no_validation_selection.json"))
    save_json(selection, str(out_dir / "option_b_no_validation_selection_summary.json"))
    selected_summary = write_selected_schedule_summary(
        schedule_summary_path,
        selection,
        out_dir / "selected_schedule_summary.json",
    )
    selected_summary["schedules"][0]["comparison_role"] = "learned_student_selected_without_validation"
    save_json(selected_summary, str(out_dir / "selected_schedule_summary.json"))

    locked_test_seeds = ",".join(str(x) for x in _parse_int_csv(str(args.locked_test_seeds)))
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
                    "--dataset",
                    str(args.dataset),
                    "--seeds",
                    locked_test_seeds,
                    "--solver_names",
                    str(args.solver_names),
                    "--target_nfe_values",
                    str(args.target_nfe_values),
                    "--num_eval_samples",
                    str(args.num_eval_samples),
                    "--device",
                    str(args.device),
                    "--eval_windows_test",
                    str(args.eval_windows_test),
                ],
            ),
            cwd=project_root,
            allow_execute=allow_execute,
            commands=commands,
        )

    summary = {
        "status": "ready" if allow_execute else "dry_run",
        "artifact": "train20_v42_no_validation_selection",
        "dataset": str(args.dataset),
        "source_run_dir": str(source_run_dir),
        "selection_protocol": "v4.2_option_b_no_validation_student_selection",
        "uses_validation_labels_for_selection": False,
        "validation_rows_evaluated": False,
        "student_budget_schedule_summary": str(schedule_summary_path),
        "conditional_opd_summary": str(conditional_summary_path),
        "no_validation_selection": str(out_dir / "no_validation_selection.json"),
        "option_b_no_validation_selection_summary": str(out_dir / "option_b_no_validation_selection_summary.json"),
        "selected_schedule_summary": str(out_dir / "selected_schedule_summary.json"),
        "selected_schedule_key": str(selection["selected_schedule_key"]),
        "selected_opd_step_budget": selection.get("selected_opd_step_budget"),
        "selected_score": selection.get("selected_score"),
        "selected_geometry": selection.get("selected_geometry"),
        "selected_summary_schedule_count": int(len(selected_summary.get("schedules", []))),
        "locked_test_seeds": _parse_int_csv(locked_test_seeds),
        "baseline_test_rows": str(args.baseline_test_rows),
        "commands": commands,
        "allow_execute": allow_execute,
    }
    if not bool(args.skip_locked_test):
        summary["locked_test_rows"] = str(out_dir / "locked_test_rows.csv")
        summary["locked_test_comparison_summary"] = str(out_dir / "locked_test_comparison_summary.json")
    save_json(summary, str(out_dir / "train20_v42_no_val_selection_summary.json"))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Train20 V4.2 no-validation student selection from an existing neural BO policy.")
    parser.add_argument("--source_run_dir", default=str(DEFAULT_SOURCE_RUN_DIR))
    parser.add_argument("--student_budget_schedule_summary", default="")
    parser.add_argument("--conditional_opd_summary", default="")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--baseline_test_rows", default=str(project_outputs_root() / "diffusion_flow_time_reparameterization_full_test" / "rows.csv"))
    parser.add_argument("--dataset", default="san_francisco_traffic")
    parser.add_argument("--solver_names", default="euler,heun,midpoint_rk2,dpmpp2m")
    parser.add_argument("--target_nfe_values", default="4,8,12")
    parser.add_argument("--locked_test_seeds", default="0,1,2")
    parser.add_argument("--num_eval_samples", type=int, default=5)
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
    summary = run_train20_v42_no_val_selection(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
