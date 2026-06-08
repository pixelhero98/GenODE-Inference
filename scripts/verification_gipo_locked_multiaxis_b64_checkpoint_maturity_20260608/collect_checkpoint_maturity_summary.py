from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PANELS = {"seen": [4, 8, 12], "unseen": [6, 10, 14, 16]}
RUNS = {
    "ckpt16k": {
        "train_steps": 16000,
        "run_id": "additive_locked_b64_normregret_ckpt16k",
    },
    "ckpt12k": {
        "train_steps": 12000,
        "run_id": "additive_locked_b64_normregret_ckpt12k",
    },
}
FINAL20_RUN_ID = "additive_locked_b64_normregret_final"
PHYSICAL_SCHEDULES = (
    "uniform",
    "late_power_3",
    "flowts_power_sampling",
    "ays",
    "gits",
    "ots",
    "late_power_3_reversed",
    "flowts_power_sampling_reversed",
    "ays_reversed",
    "gits_reversed",
    "ots_reversed",
)
SER_SCHEDULE = "ser_ptg_local_defect_eta005"
REQUIRED_SELECTION_MODE = "weighted_normalized_regret_v1"
REQUIRED_STUDENT_SELECTION_MODE = "validation_ce_v1"
REQUIRED_DENSITY_BIN_COUNT = 64


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _nested(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _density_bin_count(payload: Mapping[str, Any]) -> int | None:
    value = _nested(payload, "density_representation", "reference_bin_count")
    if value is None:
        value = _nested(payload, "teacher_model_config", "density_dim")
    return None if value is None else int(value)


def _balanced(crps: float, mase: float) -> float:
    return 0.5 * float(crps) + 0.5 * float(mase)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty metric sequence.")
    return float(sum(float(value) for value in values) / len(values))


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]], by: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: {"crps": [], "mase": []})
    for row in rows:
        key = tuple(str(row[name]) for name in by)
        grouped[key]["crps"].append(float(row["crps"]))
        grouped[key]["mase"].append(float(row["mase"]))
    out: list[dict[str, Any]] = []
    for key, values in grouped.items():
        crps = _mean(values["crps"])
        mase = _mean(values["mase"])
        out.append(
            {
                **{name: value for name, value in zip(by, key)},
                "crps": crps,
                "mase": mase,
                "balanced_crps_mase": _balanced(crps, mase),
                "row_count": len(values["crps"]),
            }
        )
    return out


def _panel_student_summary(root: Path, run_id: str, panel: str) -> dict[str, Any]:
    summary = _load_json(root / "locked_reports" / panel / "student" / run_id / "locked_test_gipo_policy_summary.json")
    aggregate_rows = _read_csv(root / "locked_reports" / panel / "student" / run_id / "locked_test_gipo_aggregate_rows.csv")
    crps = float(summary["mean_crps"])
    mase = float(summary["mean_mase"])
    return {
        "summary": summary,
        "aggregate_rows": aggregate_rows,
        "crps": crps,
        "mase": mase,
        "balanced_crps_mase": _balanced(crps, mase),
    }


def _physical_summary(root: Path, panel: str) -> dict[str, Any]:
    fixed_rows = _read_csv(root / "standard_inputs" / panel / "fixed_locked" / "fixed_locked_context_rows.csv")
    physical_rows = [row for row in fixed_rows if str(row.get("scheduler_key")) in set(PHYSICAL_SCHEDULES)]
    schedule_rows = sorted(_aggregate_metric_rows(physical_rows, ["scheduler_key"]), key=lambda row: row["balanced_crps_mase"])
    per_cell = _aggregate_metric_rows(physical_rows, ["solver_key", "target_nfe", "scheduler_key"])
    best_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    for row in per_cell:
        key = (str(row["solver_key"]), str(row["target_nfe"]))
        if key not in best_by_cell or float(row["balanced_crps_mase"]) < float(best_by_cell[key]["balanced_crps_mase"]):
            best_by_cell[key] = row
    ser_rows = _read_csv(root / "standard_inputs" / panel / "ser_locked" / "ser_locked_context_rows.csv")
    ser_summary = _aggregate_metric_rows(ser_rows, ["scheduler_key"])[0]
    return {
        "schedule_rankings": schedule_rows,
        "best_physical": schedule_rows[0],
        "best_physical_by_cell": sorted(best_by_cell.values(), key=lambda row: (str(row["solver_key"]), int(row["target_nfe"]))),
        "ser": ser_summary,
    }


def _validate_artifacts(root: Path, run_key: str, issues: list[str]) -> None:
    summary_path = root / "summary" / "artifact_validation_summary.json"
    if not summary_path.exists():
        issues.append(f"{run_key}: missing artifact validation summary")
        return
    payload = _load_json(summary_path)
    if not bool(payload.get("validation_passed", False)):
        issues.append(f"{run_key}: artifact validation did not pass")
    artifact_issues = payload.get("issues", {})
    if artifact_issues:
        issues.append(f"{run_key}: artifact validation issues present")


def _validate_training(root: Path, run_key: str, run_id: str, train_steps: int, issues: list[str]) -> dict[str, Any]:
    training = _load_json(root / "policy_runs" / run_id / "gipo_training_summary.json")
    if str(training.get("status")) != "completed":
        issues.append(f"{run_key}: training incomplete")
    if str(training.get("gipo_conditioning_style")) != "additive_mlp_v1":
        issues.append(f"{run_key}: non-additive conditioning")
    if bool(training.get("noncanonical_conditioning_allowed")):
        issues.append(f"{run_key}: noncanonical conditioning flag enabled")
    if _density_bin_count(training) != REQUIRED_DENSITY_BIN_COUNT:
        issues.append(f"{run_key}: wrong density bin count")
    if str(training.get("teacher_checkpoint_selection_mode")) != REQUIRED_SELECTION_MODE:
        issues.append(f"{run_key}: wrong teacher selector")
    if str(training.get("student_checkpoint_selection_mode")) != REQUIRED_STUDENT_SELECTION_MODE:
        issues.append(f"{run_key}: wrong student selector")
    if bool(training.get("locked_test_used_for_selection", False)):
        issues.append(f"{run_key}: locked test used for selection")
    if float(_nested(training, "pseudo_distillation", "pseudo_target_weight", default=0.0) or 0.0) != 0.0:
        issues.append(f"{run_key}: pseudo target weight nonzero")
    if not bool(_nested(training, "student_final_retrain", "enabled", default=False)):
        issues.append(f"{run_key}: missing student final retrain")
    if not bool(_nested(training, "teacher_final_retrain", "enabled", default=False)):
        issues.append(f"{run_key}: missing teacher final retrain")
    teacher_selection = dict(training.get("teacher_checkpoint_selection", {}) or {})
    if str(teacher_selection.get("selection_protocol")) != REQUIRED_SELECTION_MODE:
        issues.append(f"{run_key}: wrong teacher selection protocol")
    if int(teacher_selection.get("selected_step") or 0) <= 0:
        issues.append(f"{run_key}: missing positive teacher selected step")
    if teacher_selection.get("selected_weighted_normalized_regret_v1_score") is None:
        issues.append(f"{run_key}: missing weighted normalized regret score")
    if bool(teacher_selection.get("locked_test_used_for_selection", False)):
        issues.append(f"{run_key}: teacher selection used locked test")
    expected_splits = {"context_disjoint", "density_family_holdout", "unseen_nfe_holdout"}
    if not expected_splits <= set((teacher_selection.get("selected_normalized_regret_values") or {}).keys()):
        issues.append(f"{run_key}: teacher selection missing J_CDN normalized regret splits")
    weights = dict(teacher_selection.get("selected_component_weights", {}) or {})
    if not (
        abs(float(weights.get("context_disjoint", 0.0)) - 0.25) <= 1e-6
        and abs(float(weights.get("density_family_holdout", 0.0)) - 0.25) <= 1e-6
        and abs(float(weights.get("unseen_nfe_holdout", 0.0)) - 0.50) <= 1e-6
    ):
        issues.append(f"{run_key}: teacher selection weights are not fixed J_CDN")
    unseen_selection = dict(training.get("unseen_nfe_selection", {}) or {})
    if not bool(unseen_selection.get("enabled", False)):
        issues.append(f"{run_key}: unseen-NFE selection diagnostics disabled")
    if [int(value) for value in unseen_selection.get("target_nfes", [])] != [6, 10, 14, 16]:
        issues.append(f"{run_key}: wrong unseen-NFE selection diagnostics panel")
    if bool(unseen_selection.get("used_for_final_fitting", True)):
        issues.append(f"{run_key}: unseen-NFE diagnostics used for final fitting")
    if bool(unseen_selection.get("locked_test_used_for_selection", False)):
        issues.append(f"{run_key}: unseen-NFE diagnostics used locked test")
    student_selection = dict(training.get("student_checkpoint_selection", {}) or {})
    if str(student_selection.get("selection_protocol")) != REQUIRED_STUDENT_SELECTION_MODE:
        issues.append(f"{run_key}: wrong student selection protocol")
    if str(student_selection.get("selection_metric")) != "validation_ce_loss":
        issues.append(f"{run_key}: wrong student selection metric")
    if int(student_selection.get("selected_step") or 0) <= 0:
        issues.append(f"{run_key}: missing positive student selected step")
    if bool(student_selection.get("locked_test_used_for_selection", False)):
        issues.append(f"{run_key}: student selection used locked test")
    if not bool(_nested(training, "student_final_retrain", "performed", default=False)):
        issues.append(f"{run_key}: student final retrain not performed")
    raw_target_nfes = training.get("nfe_sequence_diagnostics", {}).get("fit_rows", {}).get("target_nfes", [])
    seen_nfes: list[int] = []
    for item in raw_target_nfes:
        if isinstance(item, Mapping):
            value = item.get("target_nfe", -1)
        else:
            value = item
        seen_nfes.append(int(value))
    seen_nfes = sorted(set(seen_nfes))
    if seen_nfes and seen_nfes != [4, 8, 12]:
        issues.append(f"{run_key}: final fit NFE metadata is not [4,8,12]")
    metadata = _load_json(root / "policy_runs" / run_id / "final_retrain_metadata.json")
    if int(metadata.get("otflow_train_steps", train_steps)) != int(train_steps):
        issues.append(f"{run_key}: final metadata train steps mismatch")
    if bool(_nested(metadata, "final_retrain", "locked_test_used_for_selection", default=True)):
        issues.append(f"{run_key}: metadata locked-test selection flag not false")
    return training


def _validate_reports(root: Path, run_key: str, run_id: str, issues: list[str]) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    for panel, expected_nfes in PANELS.items():
        current = _panel_student_summary(root, run_id, panel)
        report = current["summary"]
        if str(report.get("conditioning_style")) != "additive_mlp_v1":
            issues.append(f"{run_key}/{panel}: non-additive report")
        if _density_bin_count(report) != REQUIRED_DENSITY_BIN_COUNT:
            issues.append(f"{run_key}/{panel}: wrong report bin count")
        if int(report.get("missing_cell_count", -1)) != 0:
            issues.append(f"{run_key}/{panel}: missing cells")
        if [int(value) for value in report.get("target_nfe_values", [])] != expected_nfes:
            issues.append(f"{run_key}/{panel}: wrong target NFEs")
        if bool(report.get("locked_test_used_for_selection", False)):
            issues.append(f"{run_key}/{panel}: locked test used for selection")
        oracle_path = root / "locked_reports" / panel / "oracle" / run_id
        if oracle_path.exists():
            issues.append(f"{run_key}/{panel}: teacher-oracle report exists")
        physical = _physical_summary(root, panel)
        panels[panel] = {
            "student": {
                "crps": current["crps"],
                "mase": current["mase"],
                "balanced_crps_mase": current["balanced_crps_mase"],
            },
            "best_physical_11": physical["best_physical"],
            "ser": physical["ser"],
            "delta_student_minus_best_physical": current["balanced_crps_mase"] - float(physical["best_physical"]["balanced_crps_mase"]),
            "delta_student_minus_ser": current["balanced_crps_mase"] - float(physical["ser"]["balanced_crps_mase"]),
            "physical_schedule_rankings": physical["schedule_rankings"],
            "best_physical_by_cell": physical["best_physical_by_cell"],
        }
    return panels


def collect(summary_root: Path, ckpt16_root: Path, ckpt12_root: Path, final20_root: Path) -> dict[str, Any]:
    issues: list[str] = []
    roots = {"ckpt16k": ckpt16_root, "ckpt12k": ckpt12_root}
    runs: dict[str, Any] = {}
    for run_key, spec in RUNS.items():
        root = roots[run_key]
        run_id = str(spec["run_id"])
        _validate_artifacts(root, run_key, issues)
        training = _validate_training(root, run_key, run_id, int(spec["train_steps"]), issues)
        panels = _validate_reports(root, run_key, run_id, issues)
        runs[run_key] = {
            "root": str(root),
            "run_id": run_id,
            "train_steps": int(spec["train_steps"]),
            "selected_teacher_step": _nested(training, "teacher_checkpoint_selection", "selected_step"),
            "selected_student_step": _nested(training, "student_checkpoint_selection", "selected_step"),
            "sampled_context_count": training.get("sampled_context_count"),
            "panels": panels,
        }
    final20: dict[str, Any] = {}
    for panel in PANELS:
        current = _panel_student_summary(final20_root, FINAL20_RUN_ID, panel)
        final20[panel] = {
            "crps": current["crps"],
            "mase": current["mase"],
            "balanced_crps_mase": current["balanced_crps_mase"],
        }
    comparison_rows: list[dict[str, Any]] = []
    for run_key, run_payload in runs.items():
        for panel, panel_payload in run_payload["panels"].items():
            student = panel_payload["student"]
            comparison_rows.append(
                {
                    "run_key": run_key,
                    "panel": panel,
                    "train_steps": run_payload["train_steps"],
                    "run_id": run_payload["run_id"],
                    "student_crps": student["crps"],
                    "student_mase": student["mase"],
                    "student_balanced_crps_mase": student["balanced_crps_mase"],
                    "best_physical_schedule": panel_payload["best_physical_11"]["scheduler_key"],
                    "best_physical_balanced_crps_mase": panel_payload["best_physical_11"]["balanced_crps_mase"],
                    "delta_student_minus_best_physical": panel_payload["delta_student_minus_best_physical"],
                    "ser_balanced_crps_mase": panel_payload["ser"]["balanced_crps_mase"],
                    "delta_student_minus_ser": panel_payload["delta_student_minus_ser"],
                    "final20_balanced_crps_mase": final20[panel]["balanced_crps_mase"],
                    "delta_student_minus_final20": student["balanced_crps_mase"] - final20[panel]["balanced_crps_mase"],
                }
            )
    return {
        "artifact": "gipo_locked_multiaxis_b64_checkpoint_maturity_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_root": str(summary_root),
        "canonical_conditioning_style": "additive_mlp_v1",
        "density_bin_count": REQUIRED_DENSITY_BIN_COUNT,
        "teacher_checkpoint_selection_mode": REQUIRED_SELECTION_MODE,
        "student_checkpoint_selection_mode": REQUIRED_STUDENT_SELECTION_MODE,
        "locked_test_used_for_backbone_maturity_selection": False,
        "maturity_interpretation": "posthoc_locked_characterization_only",
        "final20_root": str(final20_root),
        "final20_run_id": FINAL20_RUN_ID,
        "runs": runs,
        "final20": final20,
        "comparison_rows": comparison_rows,
        "validation_passed": not issues,
        "issues": issues,
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# GIPO 64-Bin Checkpoint Maturity",
        "",
        f"- Validation passed: `{payload['validation_passed']}`",
        f"- Canonical conditioning: `{payload['canonical_conditioning_style']}`",
        f"- Locked test used for maturity selection: `{payload['locked_test_used_for_backbone_maturity_selection']}`",
        "",
        "| Run | Panel | Student Balanced | Best Physical | Delta vs Physical | Delta vs 20k |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in payload["comparison_rows"]:
        lines.append(
            "| {run_key} | {panel} | {student_balanced_crps_mase:.6f} | {best_physical_schedule} "
            "({best_physical_balanced_crps_mase:.6f}) | {delta_student_minus_best_physical:.6f} | "
            "{delta_student_minus_final20:.6f} |".format(**row)
        )
    lines.append("")
    if payload["issues"]:
        lines.extend(["Issues:", "", *[f"- {issue}" for issue in payload["issues"]], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect additive 64-bin GIPO checkpoint maturity summary.")
    parser.add_argument("--summary_root", required=True)
    parser.add_argument("--ckpt16_root", required=True)
    parser.add_argument("--ckpt12_root", required=True)
    parser.add_argument("--final20_root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    summary_root = Path(args.summary_root)
    payload = collect(
        summary_root,
        Path(args.ckpt16_root),
        Path(args.ckpt12_root),
        Path(args.final20_root),
    )
    out_dir = summary_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "gipo_locked_multiaxis_b64_checkpoint_maturity_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(out_dir / "gipo_locked_multiaxis_b64_checkpoint_maturity_summary.md", payload)
    print(json.dumps({"summary": str(summary_path), "validation_passed": payload["validation_passed"]}, sort_keys=True))
    if not payload["validation_passed"]:
        raise SystemExit("GIPO checkpoint maturity collection failed validation.")


if __name__ == "__main__":
    main()
