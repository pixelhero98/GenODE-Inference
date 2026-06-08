from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


RUN_ID = "seen_balanced_additive_b128"
EXPECTED_OBSERVED_NFES = (4, 8, 12)
OLD_OFFICIAL_SEEN_LOCKED = {"run_id": "temp_fixed_005_b128", "crps": 3.38033, "mase": 2.52072}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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
    return _read_json(path if path.is_absolute() else summary_path.parent / path)


def _missing_count(summary: Mapping[str, Any], comparison: Mapping[str, Any]) -> int:
    keys = ("missing_expected_cells", "missing_baseline_cells", "missing_ser_ptg_cells", "missing_student_cells")
    return sum(len(summary.get(key, []) or comparison.get(key, []) or []) for key in keys)


def _observed_panel_nfes(summary: Mapping[str, Any], comparison: Mapping[str, Any]) -> list[int]:
    raw = summary.get("target_nfe_values") or comparison.get("target_nfe_values") or []
    if raw:
        return sorted({int(value) for value in raw})
    rankings = list(comparison.get("cell_rankings") or [])
    return sorted({int(row["target_nfe"]) for row in rankings if "target_nfe" in row})


def _last_loss(training: Mapping[str, Any]) -> dict[str, Any]:
    losses = list(training.get("losses") or [])
    return dict(losses[-1]) if losses else {}


def _training_row(root: Path) -> dict[str, Any]:
    path = root / "policy_runs" / RUN_ID / "gipo_training_summary.json"
    if not path.exists():
        return {"run_id": RUN_ID, "status": "missing_training", "issues": ["missing_training"], "training_summary": str(path)}
    summary = _read_json(path)
    input_manifest = _read_json(root / "summary" / "input_manifest.json")
    teacher_cfg = dict(summary.get("teacher_model_config") or {})
    student_cfg = dict(summary.get("student_model_config") or {})
    student_training = dict(summary.get("student_training") or {})
    weights = dict(summary.get("teacher_utility_weights") or {})
    diagnostics = dict(summary.get("nfe_sequence_diagnostics") or {})
    fit_diag = dict(diagnostics.get("fit_rows") or {})
    raw_diag = dict(diagnostics.get("raw_rows") or {})
    sampled_diag = dict(diagnostics.get("sampled_rows") or {})
    issues: list[str] = []
    if int(input_manifest.get("multi_nfe_group_count", 0)) <= 0:
        issues.append("input_manifest_multi_nfe_group_count")
    if teacher_cfg.get("conditioning_style") != "additive_mlp_v1":
        issues.append("teacher_conditioning_style")
    if student_cfg.get("conditioning_style") != "additive_mlp_v1":
        issues.append("student_conditioning_style")
    if teacher_cfg.get("series_conditioning") != "none_context_only":
        issues.append("teacher_series_conditioning")
    if student_cfg.get("series_conditioning") != "none_context_only":
        issues.append("student_series_conditioning")
    if teacher_cfg.get("attention_heads") != 4 or student_cfg.get("attention_heads") != 4:
        issues.append("attention_heads")
    if list(dict(summary.get("setting_encoder_config") or {}).get("observed_target_nfes") or []) != list(EXPECTED_OBSERVED_NFES):
        issues.append("observed_target_nfes")
    if abs(float(weights.get("crps", -1.0)) - 0.5) > 1e-12 or abs(float(weights.get("mase", -1.0)) - 0.5) > 1e-12:
        issues.append("teacher_utility_weights")
    if float(student_training.get("student_nfe_smoothness_weight", -1.0)) != 0.0:
        issues.append("student_nfe_smoothness_weight")
    if bool(student_training.get("pseudo_distillation_used", False)):
        issues.append("pseudo_distillation_used")
    if float(student_training.get("pseudo_target_weight", -1.0)) != 0.0:
        issues.append("pseudo_target_weight")
    if int(student_training.get("student_nfe_sequence_pair_count", 0)) <= 0:
        issues.append("student_nfe_sequence_pair_count")
    if int(fit_diag.get("nfe_sequence_pair_count", 0)) <= 0:
        issues.append("fit_nfe_sequence_pair_count")
    if int(sampled_diag.get("nfe_sequence_pair_count", 0)) <= 0:
        issues.append("sampled_nfe_sequence_pair_count")
    if int(fit_diag.get("physical_multi_nfe_group_count", 0)) <= 0:
        issues.append("fit_physical_multi_nfe_group_count")
    if int(raw_diag.get("physical_multi_nfe_group_count", 0)) <= 0:
        issues.append("raw_physical_multi_nfe_group_count")
    if int(sampled_diag.get("physical_multi_nfe_group_count", 0)) <= 0:
        issues.append("sampled_physical_multi_nfe_group_count")
    if summary.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection")
    last = _last_loss(student_training)
    return {
        "run_id": RUN_ID,
        "status": "completed" if not issues else "invalid",
        "issues": issues,
        "conditioning_style": student_cfg.get("conditioning_style"),
        "series_conditioning": student_cfg.get("series_conditioning"),
        "teacher_utility_weights": weights,
        "student_entropy_last": last.get("student_entropy"),
        "student_kl_ce_loss_last": last.get("student_kl_ce_loss"),
        "student_nfe_sequence_pair_count": student_training.get("student_nfe_sequence_pair_count"),
        "fit_nfe_sequence_pair_count": fit_diag.get("nfe_sequence_pair_count"),
        "sampled_nfe_sequence_pair_count": sampled_diag.get("nfe_sequence_pair_count"),
        "fit_physical_multi_nfe_group_count": fit_diag.get("physical_multi_nfe_group_count"),
        "raw_physical_multi_nfe_group_count": raw_diag.get("physical_multi_nfe_group_count"),
        "sampled_physical_multi_nfe_group_count": sampled_diag.get("physical_multi_nfe_group_count"),
        "input_multi_nfe_group_count": input_manifest.get("multi_nfe_group_count"),
        "training_summary": str(path),
    }


def _panel_row(root: Path) -> dict[str, Any]:
    student_path = root / "seen_locked_reports" / "student" / RUN_ID / "locked_test_gipo_policy_summary.json"
    oracle_path = root / "seen_locked_reports" / "oracle" / RUN_ID / "locked_test_gipo_teacher_oracle_policy_summary.json"
    if not student_path.exists() or not oracle_path.exists():
        return {
            "run_id": RUN_ID,
            "panel_status": "missing_report",
            "panel_issues": ["missing_report"],
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
    if _observed_panel_nfes(student, student_comparison) != list(EXPECTED_OBSERVED_NFES):
        issues.append("student_seen_locked_nfes")
    if _observed_panel_nfes(oracle, oracle_comparison) != list(EXPECTED_OBSERVED_NFES):
        issues.append("oracle_seen_locked_nfes")
    if student.get("locked_test_used_for_selection") is not False or oracle.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection")
    return {
        "run_id": RUN_ID,
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
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    training = dict(payload.get("training") or {})
    panel = dict(payload.get("seen_locked") or {})
    return "\n".join(
        [
            "# GIPO Balanced Seen Multi-NFE Summary",
            "",
            f"- Root: `{payload.get('root')}`",
            f"- Input multi-NFE groups: `{payload.get('input_manifest', {}).get('multi_nfe_group_count')}`",
            f"- Training status: `{training.get('status')}` issues `{training.get('issues')}`",
            f"- Student NFE pairs: `{training.get('student_nfe_sequence_pair_count')}`",
            f"- Student CE: `{training.get('student_kl_ce_loss_last')}`, entropy `{training.get('student_entropy_last')}`",
            f"- Seen locked status: `{panel.get('panel_status')}` issues `{panel.get('panel_issues')}`",
            f"- Seen locked student CRPS/MASE: `{panel.get('student_crps')}` / `{panel.get('student_mase')}`",
            f"- Gains vs old official CRPS/MASE: `{panel.get('student_gain_vs_old_official_crps')}` / `{panel.get('student_gain_vs_old_official_mase')}`",
            f"- Teacher-oracle CRPS/MASE: `{panel.get('oracle_crps')}` / `{panel.get('oracle_mase')}`",
            "",
        ]
    )


def collect(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "artifact": "gipo_balanced_seen_multinfe_summary",
        "root": str(root),
        "mode": str(args.mode),
        "expected_observed_nfes": list(EXPECTED_OBSERVED_NFES),
        "locked_test_used_for_selection": False,
        "old_official_seen_locked": OLD_OFFICIAL_SEEN_LOCKED,
        "input_manifest": _read_json(summary_dir / "input_manifest.json"),
        "training": _training_row(root),
    }
    if str(args.mode) == "final":
        payload["seen_locked"] = _panel_row(root)
    _write_json(summary_dir / "balanced_seen_multinfe_summary.json", payload)
    _write_csv(summary_dir / "balanced_seen_training_rows.csv", [payload["training"]])
    if str(args.mode) == "final":
        _write_csv(summary_dir / "balanced_seen_locked_rows.csv", [payload["seen_locked"]])
    (summary_dir / "final_balanced_seen_multinfe_report.md").write_text(_markdown(payload), encoding="utf-8")
    issues = list(dict(payload.get("training") or {}).get("issues") or [])
    if str(args.mode) == "final":
        issues.extend(list(dict(payload.get("seen_locked") or {}).get("panel_issues") or []))
    if issues:
        raise SystemExit(f"{RUN_ID} balanced seen validation failed: {issues}")
    return {"run_id": RUN_ID, "mode": str(args.mode), "locked_test_used_for_selection": False}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=("training", "final"), default="final")
    return parser


if __name__ == "__main__":
    print(json.dumps(collect(build_argparser().parse_args()), indent=2, sort_keys=True))
