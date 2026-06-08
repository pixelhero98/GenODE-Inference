from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_ID = "adaln_locked_b128_multiaxis_select_ab"
ADDITIVE_RUN_ID = "additive_locked_b128_multiaxis_select"
REQUIRED_CONDITIONING = "adaln_zero_v1"
ADDITIVE_CONDITIONING = "additive_mlp_v1"
REQUIRED_SELECTION_MODE = "composite_regret_v1"
REQUIRED_DENSITY_BIN_COUNT = 128
EXPECTED_NFES = {"seen": [4, 8, 12], "unseen": [6, 10, 14, 16]}
PANELS = ("seen", "unseen")
MODES = ("student", "teacher_oracle")


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
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _comparison(summary: Mapping[str, Any], *, summary_path: Path) -> dict[str, Any]:
    raw_path = str(summary.get("comparison_summary_path", "") or "").strip()
    if not raw_path:
        return {}
    path = Path(raw_path)
    if not path.is_absolute():
        path = summary_path.parent / path
    return _read_json(path)


def _target_nfes(summary: Mapping[str, Any], comparison: Mapping[str, Any]) -> list[int]:
    raw = summary.get("target_nfe_values") or comparison.get("target_nfe_values") or []
    if raw:
        return sorted({int(value) for value in raw})
    rankings = list(comparison.get("cell_rankings") or [])
    return sorted({int(row["target_nfe"]) for row in rankings if "target_nfe" in row})


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


def _training_row(root: Path) -> dict[str, Any]:
    summary_path = root / "policy_runs" / RUN_ID / "gipo_training_summary.json"
    final_retrain_path = root / "policy_runs" / RUN_ID / "final_retrain_metadata.json"
    summary = _read_json(summary_path)
    final_retrain_payload = _read_json(final_retrain_path)
    if not summary:
        return {
            "run_id": RUN_ID,
            "status": "missing",
            "issues": ["missing_training_summary"],
            "training_summary_path": str(summary_path),
        }
    teacher_cfg = dict(summary.get("teacher_model_config") or {})
    student_cfg = dict(summary.get("student_model_config") or {})
    density_meta = dict(summary.get("density_representation") or {})
    student_training = dict(summary.get("student_training") or {})
    pseudo = dict(summary.get("pseudo_distillation") or {})
    final_retrain = dict(summary.get("final_teacher_retrain") or summary.get("teacher_final_retrain") or {})
    final_retrain_meta = dict(final_retrain_payload.get("final_retrain") or {})
    issues: list[str] = []
    if summary.get("gipo_conditioning_style") != REQUIRED_CONDITIONING:
        issues.append("summary_conditioning_style")
    if teacher_cfg.get("conditioning_style") != REQUIRED_CONDITIONING:
        issues.append("teacher_conditioning_style")
    if student_cfg.get("conditioning_style") != REQUIRED_CONDITIONING:
        issues.append("student_conditioning_style")
    if summary.get("teacher_checkpoint_selection_mode") != REQUIRED_SELECTION_MODE:
        issues.append("selection_mode")
    if summary.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection")
    if _safe_int(density_meta.get("reference_bin_count")) != REQUIRED_DENSITY_BIN_COUNT:
        issues.append("density_bin_count")
    if _safe_int(summary.get("sampled_context_count")) != 256:
        issues.append("context_sample_count")
    if bool(student_training.get("pseudo_distillation_used", False)):
        issues.append("student_pseudo_distillation_used")
    if _safe_float(student_training.get("student_nfe_smoothness_weight")) != 0.0:
        issues.append("student_nfe_smoothness_weight")
    if _safe_float(student_training.get("pseudo_target_weight")) != 0.0:
        issues.append("student_pseudo_target_weight")
    if bool(pseudo.get("pseudo_distillation_requested", False)):
        issues.append("pseudo_distillation_requested")
    if _safe_float(pseudo.get("pseudo_target_weight")) != 0.0:
        issues.append("pseudo_metadata_target_weight")
    if not bool(final_retrain.get("enabled")) or int(final_retrain.get("selected_step", 0)) <= 0:
        issues.append("final_teacher_retrain")
    if not final_retrain_payload:
        issues.append("missing_final_retrain_metadata")
    elif final_retrain_payload.get("selection_mode") != REQUIRED_SELECTION_MODE:
        issues.append("final_retrain_selection_mode")
    if final_retrain_meta and final_retrain_meta.get("completed") is not True:
        issues.append("final_retrain_not_completed")
    last = _last_loss(student_training)
    return {
        "run_id": RUN_ID,
        "status": "completed" if not issues else "invalid",
        "issues": issues,
        "conditioning_style": student_cfg.get("conditioning_style"),
        "teacher_conditioning_style": teacher_cfg.get("conditioning_style"),
        "density_bin_count": density_meta.get("reference_bin_count"),
        "selection_mode": summary.get("teacher_checkpoint_selection_mode"),
        "sampled_context_count": summary.get("sampled_context_count"),
        "selected_step": final_retrain.get("selected_step"),
        "student_entropy_last": _safe_float(last.get("student_entropy")),
        "student_kl_ce_loss_last": _safe_float(last.get("student_kl_ce_loss")),
        "student_nfe_smoothness_weight": student_training.get("student_nfe_smoothness_weight"),
        "pseudo_target_weight": student_training.get("pseudo_target_weight"),
        "locked_test_used_for_selection": summary.get("locked_test_used_for_selection"),
        "training_summary_path": str(summary_path),
        "final_retrain_metadata_path": str(final_retrain_path),
    }


def _report_summary_path(root: Path, panel: str, mode: str) -> Path:
    if mode == "student":
        return root / "locked_reports" / panel / "student" / RUN_ID / "locked_test_gipo_policy_summary.json"
    return root / "locked_reports" / panel / "oracle" / RUN_ID / "locked_test_gipo_teacher_oracle_policy_summary.json"


def _report_row(root: Path, *, panel: str, mode: str) -> dict[str, Any]:
    path = _report_summary_path(root, panel, mode)
    summary = _read_json(path)
    if not summary:
        return {
            "panel": panel,
            "mode": mode,
            "status": "missing",
            "issues": ["missing_report_summary"],
            "report_summary_path": str(path),
        }
    comparison = _comparison(summary, summary_path=path)
    nfes = _target_nfes(summary, comparison)
    missing = _missing_count(summary, comparison)
    issues: list[str] = []
    if summary.get("conditioning_style") != REQUIRED_CONDITIONING:
        issues.append("conditioning_style")
    if summary.get("teacher_checkpoint_selection_mode") != REQUIRED_SELECTION_MODE:
        issues.append("selection_mode")
    if summary.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection")
    if nfes != EXPECTED_NFES[panel]:
        issues.append("wrong_nfe_panel")
    if missing:
        issues.append("missing_cells")
    if _safe_int(dict(summary.get("density_representation") or {}).get("reference_bin_count")) != REQUIRED_DENSITY_BIN_COUNT:
        issues.append("density_bin_count")
    return {
        "panel": panel,
        "mode": mode,
        "status": "completed" if not issues else "invalid",
        "issues": issues,
        "target_nfe_values": nfes,
        "expected_target_nfe_values": EXPECTED_NFES[panel],
        "missing_cell_count": missing,
        "conditioning_style": summary.get("conditioning_style"),
        "selection_mode": summary.get("teacher_checkpoint_selection_mode"),
        "density_bin_count": dict(summary.get("density_representation") or {}).get("reference_bin_count"),
        "mean_crps": _safe_float(summary.get("mean_crps")),
        "mean_mase": _safe_float(summary.get("mean_mase")),
        "locked_test_used_for_selection": summary.get("locked_test_used_for_selection"),
        "report_summary_path": str(path),
        "comparison_summary_path": summary.get("comparison_summary_path", ""),
    }


def _additive_report_rows(additive_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = additive_root / "summary" / "additive_locked_multiaxis_report_summary.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        out[(str(row.get("panel")), str(row.get("mode")))] = row
    return out


def _comparison_rows(adaln_rows: Sequence[Mapping[str, Any]], additive_rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in adaln_rows:
        key = (str(row["panel"]), str(row["mode"]))
        additive = additive_rows.get(key, {})
        adaln_crps = _safe_float(row.get("mean_crps"))
        adaln_mase = _safe_float(row.get("mean_mase"))
        additive_crps = _safe_float(additive.get("mean_crps"))
        additive_mase = _safe_float(additive.get("mean_mase"))
        rows.append(
            {
                "panel": key[0],
                "mode": key[1],
                "adaln_crps": adaln_crps,
                "additive_crps": additive_crps,
                "delta_crps_adaln_minus_additive": None if adaln_crps is None or additive_crps is None else adaln_crps - additive_crps,
                "adaln_mase": adaln_mase,
                "additive_mase": additive_mase,
                "delta_mase_adaln_minus_additive": None if adaln_mase is None or additive_mase is None else adaln_mase - additive_mase,
                "adaln_status": row.get("status"),
                "adaln_issues": row.get("issues"),
                "additive_found": bool(additive),
            }
        )
    return rows


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# GIPO AdaLN Sidecar A/B Summary",
        "",
        f"- AdaLN root: `{payload['root']}`",
        f"- Additive root: `{payload['additive_root']}`",
        f"- Run: `{payload['run_id']}`",
        f"- Validation passed: `{payload['validation_passed']}` issues `{payload['issues']}`",
        f"- Training: `{payload['training']['status']}` issues `{payload['training']['issues']}`",
        "",
        "| panel | mode | AdaLN CRPS | Additive CRPS | delta | AdaLN MASE | Additive MASE | delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["comparison_rows"]:
        lines.append(
            "| {panel} | {mode} | {adaln_crps} | {additive_crps} | {delta_crps_adaln_minus_additive} | "
            "{adaln_mase} | {additive_mase} | {delta_mase_adaln_minus_additive} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def collect(root: Path, additive_root: Path) -> dict[str, Any]:
    training = _training_row(root)
    report_rows = [_report_row(root, panel=panel, mode=mode) for panel in PANELS for mode in MODES]
    additive_rows = _additive_report_rows(additive_root)
    comparison_rows = _comparison_rows(report_rows, additive_rows)
    issues: list[str] = []
    issues.extend(f"training:{issue}" for issue in training.get("issues", []))
    for row in report_rows:
        issues.extend(f"{row.get('panel')}:{row.get('mode')}:{issue}" for issue in row.get("issues", []))
    if len(additive_rows) != 4:
        issues.append("missing_additive_comparison_rows")
    for row in comparison_rows:
        if not row.get("additive_found"):
            issues.append(f"{row['panel']}:{row['mode']}:missing_additive_comparison")
    payload = {
        "artifact": "gipo_adaln_locked_multiaxis_ab_summary",
        "root": str(root),
        "additive_root": str(additive_root),
        "run_id": RUN_ID,
        "additive_run_id": ADDITIVE_RUN_ID,
        "conditioning_style": REQUIRED_CONDITIONING,
        "additive_conditioning_style": ADDITIVE_CONDITIONING,
        "validation_passed": not issues,
        "issues": issues,
        "training": training,
        "reports": report_rows,
        "comparison_rows": comparison_rows,
    }
    summary_dir = root / "summary"
    _write_json(summary_dir / "adaln_locked_multiaxis_ab_final_summary.json", payload)
    _write_csv(summary_dir / "adaln_locked_multiaxis_ab_report_summary.csv", report_rows)
    _write_csv(summary_dir / "adaln_locked_multiaxis_ab_comparison.csv", comparison_rows)
    _write_csv(summary_dir / "adaln_locked_multiaxis_ab_training_summary.csv", [training])
    (summary_dir / "adaln_locked_multiaxis_ab_final_report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--additive_root", required=True)
    args = parser.parse_args()
    payload = collect(Path(args.root), Path(args.additive_root))
    print(json.dumps({"run_id": RUN_ID, "validation_passed": payload["validation_passed"]}, sort_keys=True))
    if not payload["validation_passed"]:
        raise SystemExit("GIPO AdaLN sidecar A/B collection failed validation.")


if __name__ == "__main__":
    main()
