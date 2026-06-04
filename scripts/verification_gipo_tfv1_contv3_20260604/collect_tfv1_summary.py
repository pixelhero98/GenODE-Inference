from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


RUN_SPECS: tuple[dict[str, str], ...] = (
    {"source": "tfv1", "run_id": "tfv1_contv3_fixed005_b128"},
    {"source": "tfv1", "run_id": "tfv1_contv3_margin_b128"},
    {"source": "tfv1", "run_id": "tfv1_contv3_mase06_margin_b128"},
)
REQUIRED_SELECTION_NFES = (6, 10, 14, 16)
PRIMARY_SELECTION_NFES = (10, 16)
CRPS_MEAN_GUARDRAIL_FLOOR = -0.005
CRPS_PER_NFE_GUARDRAIL_FLOOR = -0.01


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _source_root(spec: Mapping[str, str], args: argparse.Namespace) -> Path:
    return Path(args.root)


def _last_loss(training_summary: Mapping[str, Any]) -> dict[str, Any]:
    losses = list(dict(training_summary.get("student_training", {}) or {}).get("losses", []) or [])
    return dict(losses[-1]) if losses else {}


def _target_summary(training_summary: Mapping[str, Any]) -> dict[str, Any]:
    return dict(dict(training_summary.get("student_training", {}) or {}).get("student_target_summary", {}) or {})


def _gain(new_value: float | None, ref_value: float | None) -> float | None:
    if new_value is None or ref_value is None or float(ref_value) == 0.0:
        return None
    return float((float(ref_value) - float(new_value)) / float(ref_value))


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _comparison_path(summary: Mapping[str, Any]) -> Path | None:
    value = str(summary.get("comparison_summary_path", "") or "").strip()
    return Path(value) if value else None


def _nfe_gain_panel(summary: Mapping[str, Any]) -> dict[str, Any]:
    path = _comparison_path(summary)
    if path is None or not path.exists():
        return {}
    comparison = _read_json(path)
    expected_nfes = [int(value) for value in comparison.get("target_nfe_values", REQUIRED_SELECTION_NFES)]
    expected_solver_count = len(list(comparison.get("solver_names", []) or []))
    missing_cells = (
        list(summary.get("missing_expected_cells", []) or [])
        + list(comparison.get("missing_baseline_cells", []) or [])
        + list(comparison.get("missing_ser_ptg_cells", []) or [])
        + list(comparison.get("missing_student_cells", []) or [])
    )
    by_nfe: dict[int, list[dict[str, Any]]] = {}
    for cell in list(comparison.get("cell_rankings", []) or []):
        if cell.get("student_relative_mase_gain_vs_best_baseline") is None:
            continue
        by_nfe.setdefault(int(cell["target_nfe"]), []).append(dict(cell))
    per_nfe: dict[str, dict[str, Any]] = {}
    for nfe, cells in sorted(by_nfe.items()):
        crps = [float(cell["student_relative_crps_gain_vs_best_baseline"]) for cell in cells if cell.get("student_relative_crps_gain_vs_best_baseline") is not None]
        mase = [float(cell["student_relative_mase_gain_vs_best_baseline"]) for cell in cells if cell.get("student_relative_mase_gain_vs_best_baseline") is not None]
        per_nfe[str(nfe)] = {
            "cell_count": int(len(cells)),
            "mean_crps_gain_vs_best_fixed": _mean(crps),
            "mean_mase_gain_vs_best_fixed": _mean(mase),
            "mase_win_count_vs_best_fixed": int(sum(1 for value in mase if value > 0.0)),
        }
    interpolation = [per_nfe[str(nfe)] for nfe in (6, 10) if str(nfe) in per_nfe]
    extrapolation = [per_nfe[str(nfe)] for nfe in (14, 16) if str(nfe) in per_nfe]
    all_cells = list(per_nfe.values())

    def panel_mean(items: list[dict[str, Any]], key: str) -> float | None:
        values = [float(item[key]) for item in items if item.get(key) is not None]
        return _mean(values)

    nfe10 = per_nfe.get("10", {})
    nfe16 = per_nfe.get("16", {})
    primary_values = [
        float(item["mean_mase_gain_vs_best_fixed"])
        for item in (nfe10, nfe16)
        if item.get("mean_mase_gain_vs_best_fixed") is not None
    ]
    complete_nfes = [
        int(nfe)
        for nfe in expected_nfes
        if str(nfe) in per_nfe and (expected_solver_count <= 0 or int(per_nfe[str(nfe)]["cell_count"]) >= expected_solver_count)
    ]
    all_unseen_crps = panel_mean(all_cells, "mean_crps_gain_vs_best_fixed")
    per_nfe_crps_values = [
        float(item["mean_crps_gain_vs_best_fixed"])
        for item in all_cells
        if item.get("mean_crps_gain_vs_best_fixed") is not None
    ]
    crps_guardrail_passed = (
        all_unseen_crps is not None
        and float(all_unseen_crps) >= CRPS_MEAN_GUARDRAIL_FLOOR
        and (not per_nfe_crps_values or min(per_nfe_crps_values) >= CRPS_PER_NFE_GUARDRAIL_FLOOR)
    )
    return {
        "per_nfe": per_nfe,
        "expected_target_nfes": expected_nfes,
        "expected_solver_count": int(expected_solver_count),
        "complete_target_nfes": complete_nfes,
        "missing_comparison_cell_count": int(len(missing_cells)),
        "coverage_complete": bool(not missing_cells and set(complete_nfes) >= set(REQUIRED_SELECTION_NFES)),
        "interpolation_mase_gain_vs_best_fixed_mean": panel_mean(interpolation, "mean_mase_gain_vs_best_fixed"),
        "extrapolation_mase_gain_vs_best_fixed_mean": panel_mean(extrapolation, "mean_mase_gain_vs_best_fixed"),
        "all_unseen_mase_gain_vs_best_fixed_mean": panel_mean(all_cells, "mean_mase_gain_vs_best_fixed"),
        "all_unseen_crps_gain_vs_best_fixed_mean": all_unseen_crps,
        "crps_guardrail_passed": bool(crps_guardrail_passed),
        "crps_mean_guardrail_floor": CRPS_MEAN_GUARDRAIL_FLOOR,
        "crps_per_nfe_guardrail_floor": CRPS_PER_NFE_GUARDRAIL_FLOOR,
        "nfe10_16_mase_gain_vs_best_fixed_min": min(primary_values) if len(primary_values) == len(PRIMARY_SELECTION_NFES) else None,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    reference_root = Path(args.reference_root)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest = _maybe_json(summary_dir / "tfv1_inputs_manifest.json")

    rows: list[dict[str, Any]] = []
    reference_student_crps = None
    reference_student_mase = None
    for spec in RUN_SPECS:
        run_id = spec["run_id"]
        run_root = _source_root(spec, args)
        training_summary_path = run_root / "policy_runs" / run_id / "gipo_training_summary.json"
        student_summary_path = root / "calibration_unseen_reports" / "student" / run_id / "unseen_validation_gipo_policy_summary.json"
        oracle_summary_path = root / "calibration_unseen_reports" / "oracle" / run_id / "unseen_validation_gipo_teacher_oracle_policy_summary.json"
        if not training_summary_path.exists() or not student_summary_path.exists() or not oracle_summary_path.exists():
            rows.append(
                {
                    "run_id": run_id,
                    "source": spec["source"],
                    "status": "missing",
                    "training_summary": str(training_summary_path),
                    "student_summary": str(student_summary_path),
                    "oracle_summary": str(oracle_summary_path),
                }
            )
            continue
        training = _read_json(training_summary_path)
        student = _read_json(student_summary_path)
        oracle = _read_json(oracle_summary_path)
        student_panel = _nfe_gain_panel(student)
        oracle_panel = _nfe_gain_panel(oracle)
        target = _target_summary(training)
        last_loss = _last_loss(training)
        student_crps = student.get("mean_crps")
        student_mase = student.get("mean_mase")
        oracle_crps = oracle.get("mean_crps")
        oracle_mase = oracle.get("mean_mase")
        if run_id == "tfv1_contv3_fixed005_b128":
            reference_student_crps = float(student_crps) if student_crps is not None else None
            reference_student_mase = float(student_mase) if student_mase is not None else None
        rows.append(
            {
                "run_id": run_id,
                "source": spec["source"],
                "status": "completed",
                "locked_test_used_for_selection": False,
                "student_mean_crps": student_crps,
                "student_mean_mase": student_mase,
                "oracle_mean_crps": oracle_crps,
                "oracle_mean_mase": oracle_mase,
                "oracle_gain_vs_student_crps": _gain(oracle_crps, student_crps),
                "oracle_gain_vs_student_mase": _gain(oracle_mase, student_mase),
                "student_gain_vs_reference_crps": _gain(student_crps, reference_student_crps),
                "student_gain_vs_reference_mase": _gain(student_mase, reference_student_mase),
                "student_target_mode": training.get("student_target_mode", target.get("student_target_mode")),
                "setting_feature_mode": training.get("setting_feature_mode", target.get("setting_feature_mode")),
                "setting_encoder_mode": training.get("setting_encoder_mode", target.get("setting_encoder_mode")),
                "student_nfe_panel": student_panel,
                "oracle_nfe_panel": oracle_panel,
                "selection_primary_nfe10_16_mase_gain_min": student_panel.get("nfe10_16_mase_gain_vs_best_fixed_min"),
                "selection_secondary_unseen_mase_gain_mean": student_panel.get("all_unseen_mase_gain_vs_best_fixed_mean"),
                "selection_crps_guardrail_unseen_gain_mean": student_panel.get("all_unseen_crps_gain_vs_best_fixed_mean"),
                "selection_coverage_complete": bool(student_panel.get("coverage_complete", False)),
                "selection_crps_guardrail_passed": bool(student_panel.get("crps_guardrail_passed", False)),
                "teacher_utility_weights": training.get("teacher_utility_weights", {}),
                "teacher_candidate_ess_p50": target.get("teacher_candidate_ess_p50"),
                "teacher_candidate_max_weight_p95": target.get("teacher_candidate_max_weight_p95"),
                "hard_target_fraction": target.get("hard_target_fraction"),
                "student_nfe_sequence_pair_count": dict(training.get("student_training", {}) or {}).get("student_nfe_sequence_pair_count"),
                "target_nfe_sequence_js_mean": target.get("nfe_sequence_js_mean"),
                "student_kl_ce_last": last_loss.get("student_kl_ce_loss"),
                "student_nfe_smoothness_loss_last": last_loss.get("student_nfe_smoothness_loss"),
                "student_entropy_last": last_loss.get("student_entropy"),
                "training_summary": str(training_summary_path),
                "student_summary": str(student_summary_path),
                "oracle_summary": str(oracle_summary_path),
            }
        )

    completed = [row for row in rows if row.get("status") == "completed"]
    best_student_crps = min(completed, key=lambda row: float(row["student_mean_crps"])) if completed else None
    best_student_mase = min(completed, key=lambda row: float(row["student_mean_mase"])) if completed else None
    best_oracle_crps = min(completed, key=lambda row: float(row["oracle_mean_crps"])) if completed else None
    selectable = [
        row
        for row in completed
        if row.get("selection_primary_nfe10_16_mase_gain_min") is not None
        and row.get("selection_secondary_unseen_mase_gain_mean") is not None
        and row.get("selection_crps_guardrail_unseen_gain_mean") is not None
        and bool(row.get("selection_coverage_complete", False))
        and bool(row.get("selection_crps_guardrail_passed", False))
    ]
    best_calibration_selection = max(
        selectable,
        key=lambda row: (
            float(row["selection_primary_nfe10_16_mase_gain_min"]),
            float(row["selection_secondary_unseen_mase_gain_mean"]),
            float(row["selection_crps_guardrail_unseen_gain_mean"]),
        ),
        default=None,
    )
    payload = {
        "artifact": "gipo_tfv1_contv3_verification_matrix",
        "locked_test_used_for_selection": False,
        "input_manifest": manifest,
        "run_count": len(rows),
        "completed_count": len(completed),
        "best_student_crps_run_id": None if best_student_crps is None else best_student_crps["run_id"],
        "best_student_mase_run_id": None if best_student_mase is None else best_student_mase["run_id"],
        "best_oracle_crps_run_id": None if best_oracle_crps is None else best_oracle_crps["run_id"],
        "best_calibration_selection_run_id": None if best_calibration_selection is None else best_calibration_selection["run_id"],
        "selection_protocol": "calibration_unseen_nfe_complete_coverage_crps_guardrail_then_primary_min_mase_gain_at_10_16_then_mean_mase_then_mean_crps",
        "selection_required_nfes": list(REQUIRED_SELECTION_NFES),
        "selection_crps_mean_guardrail_floor": CRPS_MEAN_GUARDRAIL_FLOOR,
        "selection_crps_per_nfe_guardrail_floor": CRPS_PER_NFE_GUARDRAIL_FLOOR,
        "rows": rows,
    }
    matrix_path = summary_dir / "tfv1_contv3_verification_matrix.json"
    matrix_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# GIPO TFv1 Continuous-v3 Verification",
        "",
        "Locked-test rows are not used for selection in this matrix.",
        "",
        f"- Completed calibration unseen-NFE reports: {len(completed)} / {len(rows)}",
        f"- Best student CRPS run: {payload['best_student_crps_run_id']}",
        f"- Best student MASE run: {payload['best_student_mase_run_id']}",
        f"- Best teacher-oracle CRPS run: {payload['best_oracle_crps_run_id']}",
        f"- Calibration selection run: {payload['best_calibration_selection_run_id']}",
        "",
        "## Rows",
    ]
    for row in rows:
        if row.get("status") != "completed":
            lines.append(f"- `{row['run_id']}`: missing reports")
            continue
        lines.append(
            "- `{run}`: student CRPS {scrps:.5f}, MASE {smase:.5f}; oracle CRPS {ocrps:.5f}, MASE {omase:.5f}; mode `{mode}`, features `{features}`".format(
                run=row["run_id"],
                scrps=float(row["student_mean_crps"]),
                smase=float(row["student_mean_mase"]),
                ocrps=float(row["oracle_mean_crps"]),
                omase=float(row["oracle_mean_mase"]),
                mode=row.get("student_target_mode"),
                features=row.get("setting_encoder_mode") or row.get("setting_feature_mode"),
            )
        )
        lines.append(
            "  - selection: coverage `{coverage}`, CRPS guardrail `{guardrail}`, min MASE gain at NFE 10/16 `{primary}`, all-unseen MASE gain `{mase}`, all-unseen CRPS gain `{crps}`".format(
                coverage=row.get("selection_coverage_complete"),
                guardrail=row.get("selection_crps_guardrail_passed"),
                primary=row.get("selection_primary_nfe10_16_mase_gain_min"),
                mase=row.get("selection_secondary_unseen_mase_gain_mean"),
                crps=row.get("selection_crps_guardrail_unseen_gain_mean"),
            )
        )
    (summary_dir / "final_tfv1_contv3_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect GIPO TFv1 continuous-v3 verification summaries.")
    parser.add_argument("--root", default="/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_tfv1_contv3_20260604")
    return parser


def main() -> None:
    payload = collect(build_argparser().parse_args())
    print(json.dumps({"completed_count": payload["completed_count"], "locked_test_used_for_selection": False}, sort_keys=True))


if __name__ == "__main__":
    main()
