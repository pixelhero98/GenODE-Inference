from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


RUN_SPECS: tuple[dict[str, Any], ...] = (
    {"run_id": "seen_additive_mase06_b128", "conditioning_style": "additive_mlp_v1", "weights": {"crps": 0.40, "mase": 0.60}},
    {"run_id": "seen_additive_mase075_b128", "conditioning_style": "additive_mlp_v1", "weights": {"crps": 0.25, "mase": 0.75}},
)
CANARY_RUN_ID = "canary_additive_overfit_b128"
OLD_OFFICIAL_SEEN_LOCKED = {"run_id": "temp_fixed_005_b128", "crps": 3.38033, "mase": 2.52072}
VECTOR_BASELINE_RUN_ID = "vec_teacher_mase06_b128"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _gain(value: Any, reference: Any) -> float | None:
    value_f = _safe_float(value)
    reference_f = _safe_float(reference)
    if value_f is None or reference_f is None or abs(reference_f) <= 1e-12:
        return None
    return float((reference_f - value_f) / reference_f)


def _comparison(summary: Mapping[str, Any], *, summary_path: Path) -> dict[str, Any]:
    value = str(summary.get("comparison_summary_path", "") or "").strip()
    if not value:
        return {}
    path = Path(value)
    return _maybe_json(path if path.is_absolute() else summary_path.parent / path)


def _missing_count(summary: Mapping[str, Any], comparison: Mapping[str, Any]) -> int:
    keys = (
        "missing_expected_cells",
        "missing_baseline_cells",
        "missing_ser_ptg_cells",
        "missing_student_cells",
    )
    return sum(len(summary.get(key, []) or comparison.get(key, []) or []) for key in keys)


def _last_loss(training: Mapping[str, Any]) -> dict[str, Any]:
    losses = list(training.get("losses") or [])
    return dict(losses[-1]) if losses else {}


def _training_row(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(spec["run_id"])
    path = root / "policy_runs" / run_id / "gipo_training_summary.json"
    if not path.exists():
        return {"run_id": run_id, "status": "missing_training", "training_summary": str(path)}
    summary = _read_json(path)
    teacher_cfg = dict(summary.get("teacher_model_config") or {})
    student_cfg = dict(summary.get("student_model_config") or {})
    student_training = dict(summary.get("student_training") or {})
    weights = dict(summary.get("teacher_utility_weights") or {})
    diagnostics = dict(summary.get("nfe_sequence_diagnostics") or {})
    fit_diag = dict(diagnostics.get("fit_rows") or {})
    issues: list[str] = []
    expected_style = str(spec.get("conditioning_style", ""))
    if expected_style and teacher_cfg.get("conditioning_style") != expected_style:
        issues.append("teacher_conditioning_style")
    if expected_style and student_cfg.get("conditioning_style") != expected_style:
        issues.append("student_conditioning_style")
    if teacher_cfg.get("attention_heads") != 4 or student_cfg.get("attention_heads") != 4:
        issues.append("attention_heads")
    if summary.get("setting_encoder_mode") != "continuous_v3":
        issues.append("setting_encoder_mode")
    observed_nfes = list(dict(summary.get("setting_encoder_config") or {}).get("observed_target_nfes") or [])
    if observed_nfes != [4, 8, 12]:
        issues.append(f"observed_target_nfes:{observed_nfes}")
    expected_weights = dict(spec.get("weights") or {})
    for key, expected in expected_weights.items():
        if abs(float(weights.get(key, -1.0)) - float(expected)) > 1e-12:
            issues.append(f"teacher_utility_weight_{key}")
    if float(student_training.get("student_nfe_smoothness_weight", -1.0)) != 0.0:
        issues.append("student_nfe_smoothness_weight")
    if bool(student_training.get("pseudo_distillation_used", False)):
        issues.append("pseudo_distillation_used")
    if summary.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection")
    warnings: list[str] = []
    if int(fit_diag.get("nfe_sequence_pair_count", 0)) == 0:
        warnings.append("zero_fit_nfe_sequence_pairs")
    last = _last_loss(student_training)
    return {
        "run_id": run_id,
        "status": "completed" if not issues else "invalid",
        "issues": issues,
        "warnings": warnings,
        "conditioning_style": student_cfg.get("conditioning_style"),
        "teacher_utility_weights": weights,
        "student_weight_decay": student_training.get("student_weight_decay"),
        "student_entropy_last": last.get("student_entropy"),
        "student_kl_ce_loss_last": last.get("student_kl_ce_loss"),
        "student_total_loss_last": last.get("student_total_loss"),
        "student_nfe_sequence_pair_count": student_training.get("student_nfe_sequence_pair_count"),
        "fit_nfe_sequence_pair_count": fit_diag.get("nfe_sequence_pair_count"),
        "fit_physical_multi_nfe_group_count": fit_diag.get("physical_multi_nfe_group_count"),
        "training_summary": str(path),
    }


def _panel_row(root: Path, run_id: str) -> dict[str, Any]:
    student_path = root / "seen_locked_reports" / "student" / run_id / "locked_test_gipo_policy_summary.json"
    oracle_path = root / "seen_locked_reports" / "oracle" / run_id / "locked_test_gipo_teacher_oracle_policy_summary.json"
    if not student_path.exists() or not oracle_path.exists():
        return {
            "run_id": run_id,
            "panel_status": "missing_report",
            "student_summary": str(student_path),
            "oracle_summary": str(oracle_path),
        }
    student = _read_json(student_path)
    oracle = _read_json(oracle_path)
    student_comparison = _comparison(student, summary_path=student_path)
    oracle_comparison = _comparison(oracle, summary_path=oracle_path)
    issues: list[str] = []
    if _missing_count(student, student_comparison):
        issues.append("student_missing_cells")
    if _missing_count(oracle, oracle_comparison):
        issues.append("oracle_missing_cells")
    if student.get("locked_test_used_for_selection") is not False or oracle.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection")
    return {
        "run_id": run_id,
        "panel_status": "completed" if not issues else "invalid",
        "panel_issues": issues,
        "student_crps": student.get("mean_crps"),
        "student_mase": student.get("mean_mase"),
        "oracle_crps": oracle.get("mean_crps"),
        "oracle_mase": oracle.get("mean_mase"),
        "student_gain_vs_old_official_crps": _gain(student.get("mean_crps"), OLD_OFFICIAL_SEEN_LOCKED["crps"]),
        "student_gain_vs_old_official_mase": _gain(student.get("mean_mase"), OLD_OFFICIAL_SEEN_LOCKED["mase"]),
        "oracle_gain_vs_student_crps": _gain(oracle.get("mean_crps"), student.get("mean_crps")),
        "oracle_gain_vs_student_mase": _gain(oracle.get("mean_mase"), student.get("mean_mase")),
        "student_summary": str(student_path),
        "oracle_summary": str(oracle_path),
        "student_comparison_summary": str(student_comparison.get("summary_path", "")),
    }


def _vector_seen_row(vector_root: Path) -> dict[str, Any]:
    student_path = vector_root / "seen_locked_reports" / "student" / VECTOR_BASELINE_RUN_ID / "locked_test_gipo_policy_summary.json"
    oracle_path = vector_root / "seen_locked_reports" / "oracle" / VECTOR_BASELINE_RUN_ID / "locked_test_gipo_teacher_oracle_policy_summary.json"
    if not student_path.exists():
        return {"run_id": VECTOR_BASELINE_RUN_ID, "panel_status": "missing_report", "student_summary": str(student_path)}
    student = _read_json(student_path)
    oracle = _maybe_json(oracle_path)
    return {
        "run_id": VECTOR_BASELINE_RUN_ID,
        "panel_status": "completed",
        "student_crps": student.get("mean_crps"),
        "student_mase": student.get("mean_mase"),
        "oracle_crps": oracle.get("mean_crps"),
        "oracle_mase": oracle.get("mean_mase"),
        "student_gain_vs_old_official_crps": _gain(student.get("mean_crps"), OLD_OFFICIAL_SEEN_LOCKED["crps"]),
        "student_gain_vs_old_official_mase": _gain(student.get("mean_mase"), OLD_OFFICIAL_SEEN_LOCKED["mase"]),
        "student_summary": str(student_path),
        "oracle_summary": str(oracle_path),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# GIPO Seen-NFE Ablation Summary",
        "",
        f"- Root: `{payload.get('root')}`",
        f"- Old official: CRPS {OLD_OFFICIAL_SEEN_LOCKED['crps']}, MASE {OLD_OFFICIAL_SEEN_LOCKED['mase']}",
        "",
        "## Canary",
    ]
    canary = dict(payload.get("canary") or {})
    lines.append(
        f"- `{CANARY_RUN_ID}` status {canary.get('status')}; CE {canary.get('student_kl_ce_loss_last')}; "
        f"entropy {canary.get('student_entropy_last')}; warnings {canary.get('warnings')}"
    )
    lines.extend(["", "## Seen Locked"])
    for row in payload.get("seen_locked_rows", []):
        lines.append(
            f"- `{row.get('run_id')}` {row.get('panel_status')}: CRPS {row.get('student_crps')}, "
            f"MASE {row.get('student_mase')}; gains vs old CRPS {row.get('student_gain_vs_old_official_crps')}, "
            f"MASE {row.get('student_gain_vs_old_official_mase')}; oracle gap MASE {row.get('oracle_gain_vs_student_mase')}"
        )
    vector = dict(payload.get("vector_baseline") or {})
    lines.append(
        f"- `{VECTOR_BASELINE_RUN_ID}` reference: CRPS {vector.get('student_crps')}, MASE {vector.get('student_mase')}; "
        f"gains vs old CRPS {vector.get('student_gain_vs_old_official_crps')}, MASE {vector.get('student_gain_vs_old_official_mase')}"
    )
    return "\n".join(lines) + "\n"


def collect(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    canary = _training_row(
        root,
        {"run_id": CANARY_RUN_ID, "conditioning_style": "additive_mlp_v1", "weights": {"crps": 0.40, "mase": 0.60}},
    )
    training_rows = [_training_row(root, spec) for spec in RUN_SPECS]
    panel_rows = [_panel_row(root, str(spec["run_id"])) for spec in RUN_SPECS]
    payload = {
        "artifact": "gipo_seen_ablation_summary",
        "root": str(root),
        "locked_test_used_for_selection": False,
        "old_official_seen_locked": OLD_OFFICIAL_SEEN_LOCKED,
        "canary": canary,
        "training_rows": training_rows,
        "seen_locked_rows": panel_rows,
        "vector_baseline": _vector_seen_row(Path(args.vector_root)),
    }
    _write_json(summary_dir / "seen_ablation_summary.json", payload)
    _write_csv(summary_dir / "seen_ablation_training_rows.csv", training_rows + [canary])
    _write_csv(summary_dir / "seen_locked_rows.csv", panel_rows)
    (summary_dir / "final_seen_ablation_report.md").write_text(_markdown(payload), encoding="utf-8")
    return {"run_count": len(training_rows), "locked_test_used_for_selection": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect seen-NFE GIPO ablation summaries.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--vector-root", required=True)
    print(json.dumps(collect(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
