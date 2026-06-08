from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from validate_additive_locked_artifacts import PANELS, validate_artifacts


RUN_ID = "additive_locked_b128"
REQUIRED_CONDITIONING = "additive_mlp_v1"
REQUIRED_SERIES_CONDITIONING = "none_context_only"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _comparison(summary: Mapping[str, Any], *, summary_path: Path) -> dict[str, Any]:
    raw_path = str(summary.get("comparison_summary_path", "") or "").strip()
    if not raw_path:
        return {}
    path = Path(raw_path)
    if not path.is_absolute():
        path = summary_path.parent / path
    return _read_json(path)


def _missing_count(summary: Mapping[str, Any], comparison: Mapping[str, Any]) -> int:
    keys = (
        "missing_expected_cells",
        "missing_baseline_cells",
        "missing_ser_ptg_cells",
        "missing_student_cells",
    )
    return sum(len(summary.get(key, []) or comparison.get(key, []) or []) for key in keys)


def _observed_nfes(summary: Mapping[str, Any], comparison: Mapping[str, Any]) -> list[int]:
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
        return {
            "run_id": RUN_ID,
            "status": "missing",
            "issues": ["missing_training_summary"],
            "training_summary_path": str(path),
        }
    summary = _read_json(path)
    teacher_cfg = dict(summary.get("teacher_model_config") or {})
    student_cfg = dict(summary.get("student_model_config") or {})
    density_meta = dict(summary.get("density_representation") or {})
    student_training = dict(summary.get("student_training") or {})
    pseudo = dict(summary.get("pseudo_distillation") or {})
    setting_cfg = dict(summary.get("setting_encoder_config") or {})
    issues: list[str] = []
    if teacher_cfg.get("conditioning_style") != REQUIRED_CONDITIONING:
        issues.append("teacher_conditioning_style")
    if student_cfg.get("conditioning_style") != REQUIRED_CONDITIONING:
        issues.append("student_conditioning_style")
    if teacher_cfg.get("series_conditioning") != REQUIRED_SERIES_CONDITIONING:
        issues.append("teacher_series_conditioning")
    if student_cfg.get("series_conditioning") != REQUIRED_SERIES_CONDITIONING:
        issues.append("student_series_conditioning")
    if summary.get("gipo_conditioning_style") != REQUIRED_CONDITIONING:
        issues.append("summary_conditioning_style")
    if summary.get("series_conditioning") != REQUIRED_SERIES_CONDITIONING:
        issues.append("summary_series_conditioning")
    if int(density_meta.get("reference_bin_count", -1)) != 128:
        issues.append("density_bin_count")
    if list(setting_cfg.get("observed_target_nfes") or []) != list(PANELS["seen"]):
        issues.append("training_observed_target_nfes")
    if float(summary.get("student_weight_decay", -1.0)) != 0.0001:
        issues.append("student_weight_decay")
    if float(student_training.get("student_nfe_smoothness_weight", -1.0)) != 0.0:
        issues.append("student_nfe_smoothness_weight")
    if bool(student_training.get("pseudo_distillation_used", False)):
        issues.append("student_pseudo_distillation_used")
    if float(student_training.get("pseudo_target_weight", -1.0)) != 0.0:
        issues.append("student_pseudo_target_weight")
    if bool(pseudo.get("pseudo_distillation_requested", False)):
        issues.append("pseudo_distillation_requested")
    if float(pseudo.get("pseudo_target_weight", -1.0)) != 0.0:
        issues.append("pseudo_metadata_target_weight")
    if summary.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection")
    diagnostics = dict(summary.get("nfe_sequence_diagnostics") or {})
    fit_diag = dict(diagnostics.get("fit_rows") or {})
    if int(fit_diag.get("nfe_sequence_pair_count", 0)) <= 0:
        issues.append("fit_nfe_sequence_pair_count")
    if int(fit_diag.get("physical_multi_nfe_group_count", 0)) <= 0:
        issues.append("fit_physical_multi_nfe_group_count")
    last = _last_loss(student_training)
    return {
        "run_id": RUN_ID,
        "status": "completed" if not issues else "invalid",
        "issues": issues,
        "conditioning_style": student_cfg.get("conditioning_style"),
        "series_conditioning": student_cfg.get("series_conditioning"),
        "density_bin_count": density_meta.get("reference_bin_count"),
        "student_entropy_last": _safe_float(last.get("student_entropy")),
        "student_kl_ce_loss_last": _safe_float(last.get("student_kl_ce_loss")),
        "student_nfe_sequence_pair_count": student_training.get("student_nfe_sequence_pair_count"),
        "fit_nfe_sequence_pair_count": fit_diag.get("nfe_sequence_pair_count"),
        "fit_physical_multi_nfe_group_count": fit_diag.get("physical_multi_nfe_group_count"),
        "locked_test_used_for_selection": summary.get("locked_test_used_for_selection"),
        "pseudo_target_weight": pseudo.get("pseudo_target_weight"),
        "pseudo_distillation_requested": pseudo.get("pseudo_distillation_requested"),
        "training_summary_path": str(path),
    }


def _report_row(root: Path, panel: str, mode: str) -> dict[str, Any]:
    expected_nfes = list(PANELS[panel])
    if mode == "student":
        report_path = root / "locked_reports" / panel / "student" / RUN_ID / "locked_test_gipo_policy_summary.json"
    elif mode == "teacher_oracle":
        report_path = root / "locked_reports" / panel / "oracle" / RUN_ID / "locked_test_gipo_teacher_oracle_policy_summary.json"
    else:
        raise ValueError(mode)
    if not report_path.exists():
        return {
            "panel": panel,
            "mode": mode,
            "status": "missing",
            "issues": ["missing_report_summary"],
            "report_summary_path": str(report_path),
        }
    summary = _read_json(report_path)
    comparison = _comparison(summary, summary_path=report_path)
    issues: list[str] = []
    missing_count = _missing_count(summary, comparison)
    if missing_count:
        issues.append("missing_cells")
    if _observed_nfes(summary, comparison) != expected_nfes:
        issues.append("wrong_nfe_panel")
    if summary.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection")
    return {
        "panel": panel,
        "mode": mode,
        "status": "completed" if not issues else "invalid",
        "issues": issues,
        "target_nfe_values": _observed_nfes(summary, comparison),
        "expected_target_nfe_values": expected_nfes,
        "missing_cell_count": int(missing_count),
        "mean_crps": _safe_float(summary.get("mean_crps")),
        "mean_mase": _safe_float(summary.get("mean_mase")),
        "locked_test_used_for_selection": summary.get("locked_test_used_for_selection"),
        "report_summary_path": str(report_path),
        "comparison_summary_path": str(summary.get("comparison_summary_path", "") or ""),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    training = dict(payload.get("training") or {})
    report_rows = list(payload.get("reports") or [])
    lines = [
        "# GIPO Additive Locked Summary",
        "",
        f"- Root: `{payload.get('root')}`",
        f"- Run: `{RUN_ID}`",
        f"- Training status: `{training.get('status')}` issues `{training.get('issues')}`",
        f"- Student CE: `{training.get('student_kl_ce_loss_last')}`, entropy `{training.get('student_entropy_last')}`",
        f"- Locked-test used for selection: `{payload.get('locked_test_used_for_selection')}`",
        "",
        "| panel | mode | status | NFEs | CRPS | MASE | issues |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in report_rows:
        lines.append(
            "| {panel} | {mode} | {status} | {nfes} | {crps} | {mase} | {issues} |".format(
                panel=row.get("panel"),
                mode=row.get("mode"),
                status=row.get("status"),
                nfes=",".join(str(value) for value in row.get("target_nfe_values", [])),
                crps=row.get("mean_crps"),
                mase=row.get("mean_mase"),
                issues=row.get("issues"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    artifact_validation = validate_artifacts(root, PANELS)
    training = _training_row(root)
    reports = [
        _report_row(root, panel, mode)
        for panel in PANELS
        for mode in ("student", "teacher_oracle")
    ]
    all_issues: list[str] = []
    if not artifact_validation.get("validation_passed", False):
        all_issues.append("artifact_validation")
    all_issues.extend(f"training:{issue}" for issue in training.get("issues", []))
    for row in reports:
        all_issues.extend(f"{row.get('panel')}:{row.get('mode')}:{issue}" for issue in row.get("issues", []))
    payload: dict[str, Any] = {
        "artifact": "gipo_additive_locked_final_summary",
        "root": str(root),
        "run_id": RUN_ID,
        "validation_passed": not all_issues,
        "issues": all_issues,
        "expected_panels": {panel: list(nfes) for panel, nfes in PANELS.items()},
        "locked_test_used_for_selection": False,
        "artifact_validation": artifact_validation,
        "training": training,
        "reports": reports,
    }
    _write_json(summary_dir / "additive_locked_final_summary.json", payload)
    _write_csv(summary_dir / "additive_locked_training_summary.csv", [training])
    _write_csv(summary_dir / "additive_locked_report_summary.csv", reports)
    (summary_dir / "additive_locked_final_report.md").write_text(_markdown(payload), encoding="utf-8")
    if all_issues:
        raise SystemExit(f"GIPO additive locked final validation failed: {all_issues}")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    payload = collect(build_argparser().parse_args(argv))
    print(json.dumps({"validation_passed": payload["validation_passed"], "run_id": RUN_ID}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
