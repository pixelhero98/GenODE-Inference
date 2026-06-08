from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_IDS = {
    "additive64": "additive_locked_b64_multiaxis_select_ab",
    "adaln64": "adaln_locked_b64_multiaxis_select_ab",
}
COMPARATOR_RUN_IDS = {
    "additive128": "additive_locked_b128_multiaxis_select",
    "adaln128": "adaln_locked_b128_multiaxis_select_ab",
}
REQUIRED_CONDITIONING = {
    "additive64": "additive_mlp_v1",
    "adaln64": "adaln_zero_v1",
}
REQUIRED_SELECTION_MODE = "composite_regret_v1"
REQUIRED_DENSITY_BIN_COUNT = 64
COMPARATOR_DENSITY_BIN_COUNT = 128
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
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _close_float(value: Any, expected: float) -> bool:
    parsed = _safe_float(value)
    return parsed is not None and abs(float(parsed) - float(expected)) <= 1e-12


def _last_loss(training: Mapping[str, Any]) -> dict[str, Any]:
    losses = list(training.get("losses") or [])
    return dict(losses[-1]) if losses else {}


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


def _training_row(root: Path, run_key: str) -> dict[str, Any]:
    run_id = RUN_IDS[run_key]
    required_conditioning = REQUIRED_CONDITIONING[run_key]
    summary_path = root / "policy_runs" / run_id / "gipo_training_summary.json"
    final_retrain_path = root / "policy_runs" / run_id / "final_retrain_metadata.json"
    summary = _read_json(summary_path)
    final_retrain_payload = _read_json(final_retrain_path)
    if not summary:
        return {
            "run_key": run_key,
            "run_id": run_id,
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
    selection_meta = dict(final_retrain_payload.get("selection") or {})
    script_contract = dict(final_retrain_payload.get("script_contract") or {})
    locked_report_nfes = dict(final_retrain_meta.get("locked_report_target_nfe_values") or {})
    source_phases = set(final_retrain_meta.get("source_split_phases_allowed") or [])

    issues: list[str] = []
    if summary.get("gipo_conditioning_style") != required_conditioning:
        issues.append("summary_conditioning_style")
    if teacher_cfg.get("conditioning_style") != required_conditioning:
        issues.append("teacher_conditioning_style")
    if student_cfg.get("conditioning_style") != required_conditioning:
        issues.append("student_conditioning_style")
    if run_key == "adaln64" and summary.get("noncanonical_conditioning_allowed") is not True:
        issues.append("noncanonical_conditioning_not_allowed")
    if run_key == "additive64" and summary.get("noncanonical_conditioning_allowed") is True:
        issues.append("additive_noncanonical_opt_in")
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
    if not _close_float(student_training.get("student_nfe_smoothness_weight"), 0.0):
        issues.append("student_nfe_smoothness_weight")
    if not _close_float(student_training.get("pseudo_target_weight"), 0.0):
        issues.append("student_pseudo_target_weight")
    if bool(pseudo.get("pseudo_distillation_requested", False)):
        issues.append("pseudo_distillation_requested")
    if not _close_float(pseudo.get("pseudo_target_weight"), 0.0):
        issues.append("pseudo_metadata_target_weight")
    if not bool(final_retrain.get("enabled")) or int(final_retrain.get("selected_step", 0)) <= 0:
        issues.append("final_teacher_retrain")
    if not final_retrain_payload:
        issues.append("missing_final_retrain_metadata")
    else:
        if final_retrain_payload.get("run_id") != run_id:
            issues.append("final_retrain_run_id")
        if final_retrain_payload.get("selection_mode") != REQUIRED_SELECTION_MODE:
            issues.append("final_retrain_selection_mode")
        if final_retrain_payload.get("selection_policy") != REQUIRED_SELECTION_MODE:
            issues.append("final_retrain_selection_policy")
        if selection_meta.get("selected_run_id") != run_id:
            issues.append("final_retrain_selected_run_id")
        if selection_meta.get("locked_test_used_for_selection") is not False:
            issues.append("final_retrain_selection_locked_test_used")
    if final_retrain_meta and final_retrain_meta.get("completed") is not True:
        issues.append("final_retrain_not_completed")
    if final_retrain_meta:
        if source_phases != {"train_tuning"}:
            issues.append("final_retrain_source_split_phases")
        if final_retrain_meta.get("locked_test_used_for_selection") is not False:
            issues.append("final_retrain_locked_test_used_for_selection")
        if list(final_retrain_meta.get("training_target_nfe_values") or []) != EXPECTED_NFES["seen"]:
            issues.append("final_retrain_training_nfes")
        if {panel: list(nfes) for panel, nfes in EXPECTED_NFES.items()} != {
            panel: list(locked_report_nfes.get(panel, [])) for panel in EXPECTED_NFES
        }:
            issues.append("final_retrain_locked_report_nfes")
    if script_contract:
        if script_contract.get("conditioning_style") != required_conditioning:
            issues.append("script_contract_conditioning_style")
        if _safe_int(script_contract.get("density_bin_count")) != REQUIRED_DENSITY_BIN_COUNT:
            issues.append("script_contract_density_bin_count")
        if not _close_float(script_contract.get("student_nfe_smoothness_weight"), 0.0):
            issues.append("script_contract_student_nfe_smoothness_weight")
        if not _close_float(script_contract.get("student_pseudo_target_weight"), 0.0):
            issues.append("script_contract_student_pseudo_target_weight")
        if script_contract.get("locked_test_used_for_selection") is not False:
            issues.append("script_contract_locked_test_used_for_selection")

    last = _last_loss(student_training)
    return {
        "run_key": run_key,
        "run_id": run_id,
        "status": "completed" if not issues else "invalid",
        "issues": issues,
        "conditioning_style": student_cfg.get("conditioning_style"),
        "teacher_conditioning_style": teacher_cfg.get("conditioning_style"),
        "noncanonical_conditioning_allowed": summary.get("noncanonical_conditioning_allowed"),
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


def _report_summary_path(root: Path, run_key: str, panel: str, mode: str) -> Path:
    run_id = RUN_IDS[run_key]
    if mode == "student":
        return root / "locked_reports" / panel / "student" / run_id / "locked_test_gipo_policy_summary.json"
    return root / "locked_reports" / panel / "oracle" / run_id / "locked_test_gipo_teacher_oracle_policy_summary.json"


def _report_row(root: Path, *, run_key: str, panel: str, mode: str) -> dict[str, Any]:
    path = _report_summary_path(root, run_key, panel, mode)
    run_id = RUN_IDS[run_key]
    required_conditioning = REQUIRED_CONDITIONING[run_key]
    summary = _read_json(path)
    if not summary:
        return {
            "run_key": run_key,
            "run_id": run_id,
            "panel": panel,
            "mode": mode,
            "status": "missing",
            "issues": ["missing_report_summary"],
            "report_summary_path": str(path),
        }
    comparison = _comparison(summary, summary_path=path)
    nfes = _target_nfes(summary, comparison)
    missing = _missing_count(summary, comparison)
    density_meta = dict(summary.get("density_representation") or {})
    issues: list[str] = []
    if summary.get("conditioning_style") != required_conditioning:
        issues.append("conditioning_style")
    if summary.get("selection_mode") != "reporting":
        issues.append("report_selection_mode")
    if summary.get("teacher_checkpoint_selection_mode") != REQUIRED_SELECTION_MODE:
        issues.append("selection_mode")
    if summary.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection")
    if nfes != EXPECTED_NFES[panel]:
        issues.append("wrong_nfe_panel")
    if missing:
        issues.append("missing_cells")
    if _safe_int(density_meta.get("reference_bin_count")) != REQUIRED_DENSITY_BIN_COUNT:
        issues.append("density_bin_count")
    mean_crps = _safe_float(summary.get("mean_crps"))
    mean_mase = _safe_float(summary.get("mean_mase"))
    if mean_crps is None:
        issues.append("mean_crps")
    if mean_mase is None:
        issues.append("mean_mase")
    return {
        "run_key": run_key,
        "run_id": run_id,
        "panel": panel,
        "mode": mode,
        "status": "completed" if not issues else "invalid",
        "issues": issues,
        "target_nfe_values": nfes,
        "expected_target_nfe_values": EXPECTED_NFES[panel],
        "missing_cell_count": missing,
        "conditioning_style": summary.get("conditioning_style"),
        "selection_mode": summary.get("teacher_checkpoint_selection_mode"),
        "density_bin_count": density_meta.get("reference_bin_count"),
        "mean_crps": mean_crps,
        "mean_mase": mean_mase,
        "locked_test_used_for_selection": summary.get("locked_test_used_for_selection"),
        "report_summary_path": str(path),
        "comparison_summary_path": summary.get("comparison_summary_path", ""),
    }


def _load_report_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _index_rows(rows: Sequence[Mapping[str, Any]], *, run_key: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        copied = dict(row)
        copied.setdefault("run_key", run_key)
        out[(str(copied.get("panel")), str(copied.get("mode")))] = copied
    return out


def _comparator_rows(additive128_root: Path, adaln128_root: Path) -> tuple[dict[str, dict[tuple[str, str], dict[str, Any]]], list[str]]:
    sources = {
        "additive128": additive128_root / "summary" / "additive_locked_multiaxis_report_summary.csv",
        "adaln128": adaln128_root / "summary" / "adaln_locked_multiaxis_ab_report_summary.csv",
    }
    indexed: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    issues: list[str] = []
    for run_key, path in sources.items():
        rows = _load_report_csv(path)
        if len(rows) != 4:
            issues.append(f"{run_key}:missing_comparator_rows")
        for row in rows:
            if _safe_int(row.get("density_bin_count")) != COMPARATOR_DENSITY_BIN_COUNT:
                issues.append(f"{run_key}:{row.get('panel')}:{row.get('mode')}:density_bin_count")
            if str(row.get("status", "completed")) not in {"", "completed"}:
                issues.append(f"{run_key}:{row.get('panel')}:{row.get('mode')}:status")
            if _safe_int(row.get("missing_cell_count"), default=0) != 0:
                issues.append(f"{run_key}:{row.get('panel')}:{row.get('mode')}:missing_cells")
            if _safe_float(row.get("mean_crps")) is None:
                issues.append(f"{run_key}:{row.get('panel')}:{row.get('mode')}:mean_crps")
            if _safe_float(row.get("mean_mase")) is None:
                issues.append(f"{run_key}:{row.get('panel')}:{row.get('mode')}:mean_mase")
        indexed[run_key] = _index_rows(rows, run_key=run_key)
    return indexed, issues


def _report_index(report_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[tuple[str, str], Mapping[str, Any]]]:
    out = {"additive64": {}, "adaln64": {}}
    for row in report_rows:
        run_key = str(row.get("run_key"))
        if run_key in out:
            out[run_key][(str(row.get("panel")), str(row.get("mode")))] = row
    return out


def _metric_comparison(
    *,
    comparison: str,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate_label: str,
    baseline_label: str,
) -> dict[str, Any]:
    candidate_crps = _safe_float(candidate.get("mean_crps"))
    baseline_crps = _safe_float(baseline.get("mean_crps"))
    candidate_mase = _safe_float(candidate.get("mean_mase"))
    baseline_mase = _safe_float(baseline.get("mean_mase"))
    return {
        "comparison": comparison,
        "panel": candidate.get("panel") or baseline.get("panel"),
        "mode": candidate.get("mode") or baseline.get("mode"),
        "candidate": candidate_label,
        "baseline": baseline_label,
        "candidate_crps": candidate_crps,
        "baseline_crps": baseline_crps,
        "delta_crps_candidate_minus_baseline": None
        if candidate_crps is None or baseline_crps is None
        else candidate_crps - baseline_crps,
        "candidate_mase": candidate_mase,
        "baseline_mase": baseline_mase,
        "delta_mase_candidate_minus_baseline": None
        if candidate_mase is None or baseline_mase is None
        else candidate_mase - baseline_mase,
        "candidate_found": bool(candidate),
        "baseline_found": bool(baseline),
    }


def _comparison_rows(
    report_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    current = _report_index(report_rows)
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    specs = (
        ("additive64_vs_additive128", "additive64", "additive128"),
        ("adaln64_vs_adaln128", "adaln64", "adaln128"),
        ("adaln64_vs_additive64", "adaln64", "additive64"),
        ("adaln64_vs_additive128_context", "adaln64", "additive128"),
    )
    for panel in PANELS:
        for mode in MODES:
            key = (panel, mode)
            for comparison, candidate_key, baseline_key in specs:
                candidate = (
                    current.get(candidate_key, {}).get(key, {})
                    if candidate_key in current
                    else comparator_rows.get(candidate_key, {}).get(key, {})
                )
                baseline = (
                    current.get(baseline_key, {}).get(key, {})
                    if baseline_key in current
                    else comparator_rows.get(baseline_key, {}).get(key, {})
                )
                row = _metric_comparison(
                    comparison=comparison,
                    candidate=candidate,
                    baseline=baseline,
                    candidate_label=candidate_key,
                    baseline_label=baseline_key,
                )
                rows.append(row)
                if not row["candidate_found"]:
                    issues.append(f"{comparison}:{panel}:{mode}:missing_candidate")
                if not row["baseline_found"]:
                    issues.append(f"{comparison}:{panel}:{mode}:missing_baseline")
    return rows, issues


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# GIPO 64-Bin Multiaxis A/B Summary",
        "",
        f"- Root: `{payload['root']}`",
        f"- Additive 128 root: `{payload['additive128_root']}`",
        f"- AdaLN 128 root: `{payload['adaln128_root']}`",
        f"- Validation passed: `{payload['validation_passed']}` issues `{payload['issues']}`",
        "",
        "| comparison | panel | mode | candidate CRPS | baseline CRPS | delta | candidate MASE | baseline MASE | delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["comparison_rows"]:
        lines.append(
            "| {comparison} | {panel} | {mode} | {candidate_crps} | {baseline_crps} | "
            "{delta_crps_candidate_minus_baseline} | {candidate_mase} | {baseline_mase} | "
            "{delta_mase_candidate_minus_baseline} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def collect(root: Path, additive128_root: Path, adaln128_root: Path) -> dict[str, Any]:
    training_rows = [_training_row(root, run_key) for run_key in ("additive64", "adaln64")]
    report_rows = [
        _report_row(root, run_key=run_key, panel=panel, mode=mode)
        for run_key in ("additive64", "adaln64")
        for panel in PANELS
        for mode in MODES
    ]
    comparator_rows, comparator_issues = _comparator_rows(additive128_root, adaln128_root)
    comparison_rows, comparison_issues = _comparison_rows(report_rows, comparator_rows)

    issues: list[str] = []
    for row in training_rows:
        issues.extend(f"{row.get('run_key')}:training:{issue}" for issue in row.get("issues", []))
    for row in report_rows:
        issues.extend(
            f"{row.get('run_key')}:{row.get('panel')}:{row.get('mode')}:{issue}"
            for issue in row.get("issues", [])
        )
    issues.extend(f"comparator:{issue}" for issue in comparator_issues)
    issues.extend(f"comparison:{issue}" for issue in comparison_issues)

    payload = {
        "artifact": "gipo_locked_multiaxis_b64_ab_summary",
        "root": str(root),
        "additive128_root": str(additive128_root),
        "adaln128_root": str(adaln128_root),
        "run_ids": RUN_IDS,
        "comparator_run_ids": COMPARATOR_RUN_IDS,
        "density_bin_count": REQUIRED_DENSITY_BIN_COUNT,
        "comparator_density_bin_count": COMPARATOR_DENSITY_BIN_COUNT,
        "selection_mode": REQUIRED_SELECTION_MODE,
        "validation_passed": not issues,
        "issues": issues,
        "training": training_rows,
        "reports": report_rows,
        "comparison_rows": comparison_rows,
    }
    summary_dir = root / "summary"
    _write_json(summary_dir / "gipo_locked_multiaxis_b64_ab_final_summary.json", payload)
    _write_csv(summary_dir / "gipo_locked_multiaxis_b64_ab_training_summary.csv", training_rows)
    _write_csv(summary_dir / "gipo_locked_multiaxis_b64_ab_report_summary.csv", report_rows)
    _write_csv(summary_dir / "gipo_locked_multiaxis_b64_ab_comparison.csv", comparison_rows)
    (summary_dir / "gipo_locked_multiaxis_b64_ab_final_report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--additive128_root", required=True)
    parser.add_argument("--adaln128_root", required=True)
    args = parser.parse_args()
    payload = collect(Path(args.root), Path(args.additive128_root), Path(args.adaln128_root))
    print(json.dumps({"validation_passed": payload["validation_passed"], "run_ids": RUN_IDS}, sort_keys=True))
    if not payload["validation_passed"]:
        raise SystemExit("GIPO 64-bin sidecar A/B collection failed validation.")


if __name__ == "__main__":
    main()
