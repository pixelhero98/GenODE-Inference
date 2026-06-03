from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


RUN_SPECS: tuple[dict[str, str], ...] = (
    {"source": "reference", "run_id": "temp_fixed_005_b128"},
    {"source": "gap", "run_id": "margin_fixed005_b128"},
    {"source": "gap", "run_id": "nferich_fixed005_b128"},
    {"source": "gap", "run_id": "nferich_margin_fixed005_b128"},
    {"source": "gap", "run_id": "mase06_nferich_margin_fixed005_b128"},
    {"source": "gap", "run_id": "mase075_nferich_margin_fixed005_b128"},
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _source_root(spec: Mapping[str, str], args: argparse.Namespace) -> Path:
    if spec["source"] == "reference":
        return Path(args.reference_root)
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


def collect(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    reference_root = Path(args.reference_root)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest = _maybe_json(summary_dir / "gap_inputs_manifest.json")

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
        target = _target_summary(training)
        last_loss = _last_loss(training)
        student_crps = student.get("mean_crps")
        student_mase = student.get("mean_mase")
        oracle_crps = oracle.get("mean_crps")
        oracle_mase = oracle.get("mean_mase")
        if run_id == "temp_fixed_005_b128":
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
                "teacher_utility_weights": training.get("teacher_utility_weights", {}),
                "teacher_candidate_ess_p50": target.get("teacher_candidate_ess_p50"),
                "teacher_candidate_max_weight_p95": target.get("teacher_candidate_max_weight_p95"),
                "hard_target_fraction": target.get("hard_target_fraction"),
                "student_kl_ce_last": last_loss.get("student_kl_ce_loss"),
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
    payload = {
        "artifact": "gipo_teacher_student_gap_verification_matrix",
        "locked_test_used_for_selection": False,
        "input_manifest": manifest,
        "run_count": len(rows),
        "completed_count": len(completed),
        "best_student_crps_run_id": None if best_student_crps is None else best_student_crps["run_id"],
        "best_student_mase_run_id": None if best_student_mase is None else best_student_mase["run_id"],
        "best_oracle_crps_run_id": None if best_oracle_crps is None else best_oracle_crps["run_id"],
        "rows": rows,
    }
    matrix_path = summary_dir / "gap_verification_matrix.json"
    matrix_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# GIPO Teacher-Student Gap Verification",
        "",
        "Locked-test rows are not used for selection in this matrix.",
        "",
        f"- Completed calibration unseen-NFE reports: {len(completed)} / {len(rows)}",
        f"- Best student CRPS run: {payload['best_student_crps_run_id']}",
        f"- Best student MASE run: {payload['best_student_mase_run_id']}",
        f"- Best teacher-oracle CRPS run: {payload['best_oracle_crps_run_id']}",
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
                features=row.get("setting_feature_mode"),
            )
        )
    (summary_dir / "final_gap_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect GIPO teacher-student gap verification summaries.")
    parser.add_argument("--root", default="/scratch/b35z/pixelhero.b35z/genode/outputs/verification_teacher_student_gap_20260603")
    parser.add_argument("--reference_root", default="/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602")
    return parser


def main() -> None:
    payload = collect(build_argparser().parse_args())
    print(json.dumps({"completed_count": payload["completed_count"], "locked_test_used_for_selection": False}, sort_keys=True))


if __name__ == "__main__":
    main()
