from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_IDS = {
    "additive64_optim": "additive_locked_b64_multiaxis_opt_guarded",
    "adaln64_optim": "adaln_locked_b64_multiaxis_opt_guarded",
}
BASELINE_B64_RUN_KEYS = {
    "additive64_optim": "additive64",
    "adaln64_optim": "adaln64",
}
BASELINE_128_RUN_KEYS = {
    "additive64_optim": "additive128",
    "adaln64_optim": "adaln128",
}
REQUIRED_CONDITIONING = {
    "additive64_optim": "additive_mlp_v1",
    "adaln64_optim": "adaln_zero_v1",
}
REQUIRED_SELECTION_MODE = "composite_regret_guarded_v1"
REQUIRED_STUDENT_SELECTOR_MODE = "validation_ce_v1"
REQUIRED_DENSITY_BIN_COUNT = 64
COMPARATOR_DENSITY_BIN_COUNT = 128
REQUIRED_STUDENT_SELECTION_HOLDOUT_FRACTION = 0.10
EXPECTED_NFES = {"seen": [4, 8, 12], "unseen": [6, 10, 14, 16]}
PANELS = ("seen", "unseen")
MODES = ("student", "teacher_oracle")
REQUIRED_SCRIPT_CONTRACT = {
    "teacher_checkpoint_every": 50,
    "teacher_log_every": 50,
    "teacher_loss_log_every": 50,
    "student_budget": 1000,
    "student_log_every": 50,
    "student_checkpoint_every": 50,
}


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
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _close_float(value: Any, expected: float) -> bool:
    parsed = _safe_float(value)
    return parsed is not None and abs(float(parsed) - float(expected)) <= 1e-12


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
        return [int(value) for value in raw]
    rankings = list(comparison.get("cell_rankings") or [])
    return sorted({_safe_int(row.get("target_nfe")) for row in rankings if _safe_int(row.get("target_nfe")) >= 0})


def _missing_count(summary: Mapping[str, Any], comparison: Mapping[str, Any]) -> int:
    keys = (
        "missing_expected_cells",
        "missing_context_cells",
        "missing_support_cells",
        "missing_baseline_cells",
        "missing_comparator_cells",
    )
    return sum(len(summary.get(key, []) or comparison.get(key, []) or []) for key in keys)


def _last_loss(training: Mapping[str, Any]) -> dict[str, Any]:
    losses = list(training.get("losses") or [])
    return dict(losses[-1]) if losses else {}


def _training_path(root: Path, run_key: str) -> Path:
    return root / "policy_runs" / RUN_IDS[run_key] / "gipo_training_summary.json"


def _final_retrain_path(root: Path, run_key: str) -> Path:
    return root / "policy_runs" / RUN_IDS[run_key] / "final_retrain_metadata.json"


def _student_selector_from_summary(summary: Mapping[str, Any]) -> Any:
    student_training = dict(summary.get("student_training") or {})
    student_cfg = dict(summary.get("student_model_config") or {})
    for key in ("student_selector_mode", "student_checkpoint_selection_mode", "selection_mode"):
        if key in student_training:
            return student_training.get(key)
    return student_cfg.get("student_selector_mode") or student_cfg.get("student_checkpoint_selection_mode")


def _training_row(root: Path, run_key: str) -> dict[str, Any]:
    run_id = RUN_IDS[run_key]
    required_conditioning = REQUIRED_CONDITIONING[run_key]
    path = _training_path(root, run_key)
    final_path = _final_retrain_path(root, run_key)
    summary = _read_json(path)
    final_payload = _read_json(final_path)
    if not summary:
        return {
            "run_key": run_key,
            "run_id": run_id,
            "status": "missing",
            "issues": ["missing_training_summary"],
            "training_summary_path": str(path),
            "final_retrain_metadata_path": str(final_path),
        }
    teacher_cfg = dict(summary.get("teacher_model_config") or {})
    student_cfg = dict(summary.get("student_model_config") or {})
    density_meta = dict(summary.get("density_representation") or {})
    teacher_training = dict(summary.get("teacher_training") or {})
    student_training = dict(summary.get("student_training") or {})
    pseudo = dict(summary.get("pseudo_distillation") or {})
    selection = dict(summary.get("teacher_checkpoint_selection") or {})
    student_selection = dict(summary.get("student_checkpoint_selection") or student_training.get("student_checkpoint_selection") or {})
    student_final_retrain = dict(summary.get("student_final_retrain") or student_training.get("student_final_retrain") or {})
    final_meta = dict(final_payload.get("final_retrain") or {})
    selection_meta = dict(final_payload.get("selection") or {})
    script_contract = dict(final_payload.get("script_contract") or {})
    issues: list[str] = []

    if summary.get("gipo_conditioning_style") != required_conditioning:
        issues.append("summary_conditioning_style")
    if teacher_cfg.get("conditioning_style") != required_conditioning:
        issues.append("teacher_conditioning_style")
    if student_cfg.get("conditioning_style") != required_conditioning:
        issues.append("student_conditioning_style")
    if summary.get("teacher_checkpoint_selection_mode") != REQUIRED_SELECTION_MODE:
        issues.append("selection_mode")
    if selection.get("selection_protocol") != REQUIRED_SELECTION_MODE:
        issues.append("teacher_selection_protocol")
    if selection.get("selection_mode") != REQUIRED_SELECTION_MODE:
        issues.append("teacher_selection_mode")
    if selection.get("guardrail_protocol") != "nfe_proxy_within_5pct_best_v1":
        issues.append("teacher_selection_guardrail_protocol")
    if summary.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection")
    if _safe_int(density_meta.get("reference_bin_count")) != REQUIRED_DENSITY_BIN_COUNT:
        issues.append("density_bin_count")
    if _safe_int(summary.get("teacher_checkpoint_every")) != REQUIRED_SCRIPT_CONTRACT["teacher_checkpoint_every"]:
        issues.append("teacher_checkpoint_every")
    if _safe_int(summary.get("teacher_loss_log_every")) != REQUIRED_SCRIPT_CONTRACT["teacher_loss_log_every"]:
        issues.append("teacher_loss_log_every")
    if _safe_int(summary.get("student_log_every")) != REQUIRED_SCRIPT_CONTRACT["student_log_every"]:
        issues.append("student_log_every")
    if _safe_int(summary.get("student_checkpoint_every")) != REQUIRED_SCRIPT_CONTRACT["student_checkpoint_every"]:
        issues.append("student_checkpoint_every")
    if summary.get("student_checkpoint_selection_mode") != REQUIRED_STUDENT_SELECTOR_MODE:
        issues.append("student_checkpoint_selection_mode")
    if student_selection.get("selection_protocol") != REQUIRED_STUDENT_SELECTOR_MODE:
        issues.append("student_checkpoint_selection_protocol")
    if student_selection.get("selection_metric") != "validation_ce_loss":
        issues.append("student_checkpoint_selection_metric")
    if _safe_int(student_selection.get("selected_step")) <= 0:
        issues.append("student_checkpoint_selected_step")
    if student_selection.get("locked_test_used_for_selection") is not False:
        issues.append("student_checkpoint_locked_test")
    if student_training.get("student_validation_used_for_selection") is not True:
        issues.append("student_validation_used_for_selection")
    if student_training.get("locked_test_used_for_selection") is not False:
        issues.append("student_locked_test_used_for_selection")
    if not bool(student_final_retrain.get("enabled")) or not bool(student_final_retrain.get("performed")):
        issues.append("student_final_retrain")
    if student_final_retrain.get("selection_protocol") != REQUIRED_STUDENT_SELECTOR_MODE:
        issues.append("student_final_retrain_selection_protocol")
    if _safe_int(student_final_retrain.get("selected_step")) <= 0:
        issues.append("student_final_retrain_selected_step")
    if student_final_retrain.get("locked_test_used_for_selection") is not False:
        issues.append("student_final_retrain_locked_test")
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
    student_selector = _student_selector_from_summary(summary)
    if student_selector not in (None, "", REQUIRED_STUDENT_SELECTOR_MODE):
        issues.append("student_selector_mode")

    if not final_payload:
        issues.append("missing_final_retrain_metadata")
    else:
        if final_payload.get("run_id") != run_id or final_payload.get("canonical_run_id") != run_id:
            issues.append("final_retrain_run_id")
        if final_payload.get("selection_mode") != REQUIRED_SELECTION_MODE:
            issues.append("final_retrain_selection_mode")
        if final_payload.get("selection_policy") != REQUIRED_SELECTION_MODE:
            issues.append("final_retrain_selection_policy")
        if selection_meta.get("selected_run_id") != run_id or list(selection_meta.get("candidate_run_ids") or []) != [run_id]:
            issues.append("final_retrain_selection_run_ids")
        if selection_meta.get("locked_test_used_for_selection") is not False:
            issues.append("final_retrain_selection_locked_test")
        if not bool(final_meta.get("enabled")) or not bool(final_meta.get("completed")):
            issues.append("final_retrain_not_completed")
        if list(final_meta.get("training_target_nfe_values") or []) != EXPECTED_NFES["seen"]:
            issues.append("final_retrain_training_nfes")
        if dict(final_meta.get("locked_report_target_nfe_values") or {}) != EXPECTED_NFES:
            issues.append("final_retrain_locked_report_nfes")
        if set(final_meta.get("source_split_phases_allowed") or []) != {"train_tuning"}:
            issues.append("final_retrain_source_split_phases")
        if final_meta.get("locked_test_used_for_selection") is not False:
            issues.append("final_retrain_locked_test_used")
        if script_contract.get("conditioning_style") != required_conditioning:
            issues.append("script_contract_conditioning_style")
        if _safe_int(script_contract.get("density_bin_count")) != REQUIRED_DENSITY_BIN_COUNT:
            issues.append("script_contract_density_bin_count")
        if script_contract.get("student_selector_mode") != REQUIRED_STUDENT_SELECTOR_MODE:
            issues.append("script_contract_student_selector_mode")
        if not _close_float(script_contract.get("student_selection_holdout_fraction"), REQUIRED_STUDENT_SELECTION_HOLDOUT_FRACTION):
            issues.append("script_contract_student_selection_holdout_fraction")
        if not _close_float(script_contract.get("student_nfe_smoothness_weight"), 0.0):
            issues.append("script_contract_student_nfe_smoothness_weight")
        if not _close_float(script_contract.get("student_pseudo_target_weight"), 0.0):
            issues.append("script_contract_student_pseudo_target_weight")
        if script_contract.get("locked_test_used_for_selection") is not False:
            issues.append("script_contract_locked_test_used")
        for key, expected in REQUIRED_SCRIPT_CONTRACT.items():
            if _safe_int(script_contract.get(key)) != int(expected):
                issues.append(f"script_contract_{key}")

    last_loss = _last_loss(teacher_training)
    return {
        "run_key": run_key,
        "run_id": run_id,
        "status": "completed" if not issues else "invalid",
        "issues": issues,
        "conditioning_style": summary.get("gipo_conditioning_style"),
        "density_bin_count": density_meta.get("reference_bin_count"),
        "selection_mode": summary.get("teacher_checkpoint_selection_mode"),
        "selected_step": selection.get("selected_step"),
        "selected_composite_score": selection.get("selected_composite_checkpoint_score"),
        "selected_mean_validation_soft_regret": selection.get("selected_mean_validation_soft_regret"),
        "selected_mean_validation_top1_regret": selection.get("selected_mean_validation_top1_regret"),
        "student_selector_mode": student_selector,
        "student_budget": student_training.get("student_budget", student_training.get("student_steps")),
        "student_nfe_smoothness_weight": student_training.get("student_nfe_smoothness_weight"),
        "pseudo_target_weight": student_training.get("pseudo_target_weight"),
        "teacher_last_total_loss": last_loss.get("teacher_total_loss"),
        "teacher_last_huber_loss": last_loss.get("teacher_huber_loss"),
        "teacher_last_rank_loss": last_loss.get("teacher_rank_loss"),
        "locked_test_used_for_selection": summary.get("locked_test_used_for_selection"),
        "training_summary_path": str(path),
        "final_retrain_metadata_path": str(final_path),
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
    if missing != 0:
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


def _existing_b64_rows(b64_root: Path) -> tuple[dict[str, dict[tuple[str, str], dict[str, Any]]], list[str]]:
    rows = _load_report_csv(b64_root / "summary" / "gipo_locked_multiaxis_b64_ab_report_summary.csv")
    indexed = {"additive64": {}, "adaln64": {}}
    issues: list[str] = []
    if len(rows) != 8:
        issues.append("existing_b64:missing_comparator_rows")
    for run_key in indexed:
        subset = [row for row in rows if str(row.get("run_key")) == run_key]
        if len(subset) != 4:
            issues.append(f"{run_key}:missing_comparator_rows")
        for row in subset:
            if _safe_int(row.get("density_bin_count")) != REQUIRED_DENSITY_BIN_COUNT:
                issues.append(f"{run_key}:{row.get('panel')}:{row.get('mode')}:density_bin_count")
            if str(row.get("status", "completed")) not in {"", "completed"}:
                issues.append(f"{run_key}:{row.get('panel')}:{row.get('mode')}:status")
            if _safe_int(row.get("missing_cell_count"), default=0) != 0:
                issues.append(f"{run_key}:{row.get('panel')}:{row.get('mode')}:missing_cells")
        indexed[run_key] = _index_rows(subset, run_key=run_key)
    return indexed, issues


def _existing_b128_rows(
    additive128_root: Path, adaln128_root: Path
) -> tuple[dict[str, dict[tuple[str, str], dict[str, Any]]], list[str]]:
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
        indexed[run_key] = _index_rows(rows, run_key=run_key)
    return indexed, issues


def _report_index(report_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[tuple[str, str], Mapping[str, Any]]]:
    out = {run_key: {} for run_key in RUN_IDS}
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
    existing_b64: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
    existing_b128: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    current = _report_index(report_rows)
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for run_key in RUN_IDS:
        b64_key = BASELINE_B64_RUN_KEYS[run_key]
        b128_key = BASELINE_128_RUN_KEYS[run_key]
        specs = [
            (f"{run_key}_vs_existing_{b64_key}", existing_b64.get(b64_key, {}), b64_key),
            (f"{run_key}_vs_existing_{b128_key}", existing_b128.get(b128_key, {}), b128_key),
        ]
        for panel in PANELS:
            for mode in MODES:
                key = (panel, mode)
                candidate = current.get(run_key, {}).get(key, {})
                for comparison, baseline_index, baseline_label in specs:
                    baseline = baseline_index.get(key, {})
                    row = _metric_comparison(
                        comparison=comparison,
                        candidate=candidate,
                        baseline=baseline,
                        candidate_label=run_key,
                        baseline_label=baseline_label,
                    )
                    rows.append(row)
                    if not row["candidate_found"]:
                        issues.append(f"{comparison}:{panel}:{mode}:missing_candidate")
                    if not row["baseline_found"]:
                        issues.append(f"{comparison}:{panel}:{mode}:missing_baseline")
    for panel in PANELS:
        for mode in MODES:
            candidate = current.get("adaln64_optim", {}).get((panel, mode), {})
            baseline = current.get("additive64_optim", {}).get((panel, mode), {})
            row = _metric_comparison(
                comparison="adaln64_optim_vs_additive64_optim",
                candidate=candidate,
                baseline=baseline,
                candidate_label="adaln64_optim",
                baseline_label="additive64_optim",
            )
            rows.append(row)
    return rows, issues


def _optimization_acceptance(comparison_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required_comparisons = {
        "additive64_optim_vs_existing_additive64",
        "adaln64_optim_vs_existing_adaln64",
    }
    student_rows = [
        row
        for row in comparison_rows
        if str(row.get("comparison")) in required_comparisons and str(row.get("mode")) == "student"
    ]
    issues: list[str] = []
    balanced_deltas: list[float] = []
    panel_regressions: list[dict[str, Any]] = []
    for row in student_rows:
        delta_crps = _safe_float(row.get("delta_crps_candidate_minus_baseline"))
        delta_mase = _safe_float(row.get("delta_mase_candidate_minus_baseline"))
        if delta_crps is None or delta_mase is None:
            issues.append(f"{row.get('comparison')}:{row.get('panel')}:missing_student_delta")
            continue
        balanced = 0.5 * float(delta_crps) + 0.5 * float(delta_mase)
        balanced_deltas.append(balanced)
        if float(delta_crps) > 0.005 or float(delta_mase) > 0.005:
            panel_regressions.append(
                {
                    "comparison": row.get("comparison"),
                    "panel": row.get("panel"),
                    "delta_crps": float(delta_crps),
                    "delta_mase": float(delta_mase),
                }
            )
    observed = {(str(row.get("comparison")), str(row.get("panel"))) for row in student_rows}
    for comparison in sorted(required_comparisons):
        for panel in PANELS:
            if (comparison, panel) not in observed:
                issues.append(f"{comparison}:{panel}:missing_student_comparison")
    average_balanced_delta = None if not balanced_deltas else sum(balanced_deltas) / float(len(balanced_deltas))
    if average_balanced_delta is None or average_balanced_delta >= 0.0:
        issues.append("average_student_balanced_delta_not_negative")
    if panel_regressions:
        issues.append("student_panel_regression_over_0p005")
    return {
        "criteria": {
            "average_student_locked_balanced_crps_mase_delta_vs_matching_b64_baseline": "negative",
            "max_allowed_seen_unseen_student_panel_regression_per_metric": 0.005,
        },
        "average_student_balanced_delta": average_balanced_delta,
        "student_balanced_deltas": balanced_deltas,
        "student_panel_regressions": panel_regressions,
        "passed": not issues,
        "issues": issues,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# GIPO Locked Multiaxis b64 Guarded Optimization Summary",
        "",
        f"- Root: `{payload['root']}`",
        f"- Existing b64 root: `{payload['b64_root']}`",
        f"- Additive 128 root: `{payload['additive128_root']}`",
        f"- AdaLN 128 root: `{payload['adaln128_root']}`",
        f"- Validation passed: `{payload['validation_passed']}` issues `{payload['issues']}`",
        f"- Optimization candidate passed: `{payload['optimization_acceptance']['passed']}` "
        f"average balanced delta `{payload['optimization_acceptance']['average_student_balanced_delta']}`",
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


def collect(root: Path, b64_root: Path, additive128_root: Path, adaln128_root: Path) -> dict[str, Any]:
    training_rows = [_training_row(root, run_key) for run_key in RUN_IDS]
    report_rows = [
        _report_row(root, run_key=run_key, panel=panel, mode=mode)
        for run_key in RUN_IDS
        for panel in PANELS
        for mode in MODES
    ]
    existing_b64, b64_issues = _existing_b64_rows(b64_root)
    existing_b128, b128_issues = _existing_b128_rows(additive128_root, adaln128_root)
    comparison_rows, comparison_issues = _comparison_rows(report_rows, existing_b64, existing_b128)
    optimization_acceptance = _optimization_acceptance(comparison_rows)

    issues: list[str] = []
    for row in training_rows:
        issues.extend(f"{row.get('run_key')}:training:{issue}" for issue in row.get("issues", []))
    for row in report_rows:
        issues.extend(
            f"{row.get('run_key')}:{row.get('panel')}:{row.get('mode')}:{issue}"
            for issue in row.get("issues", [])
        )
    issues.extend(f"existing_b64:{issue}" for issue in b64_issues)
    issues.extend(f"existing_b128:{issue}" for issue in b128_issues)
    issues.extend(f"comparison:{issue}" for issue in comparison_issues)

    payload = {
        "artifact": "gipo_locked_multiaxis_b64_optim_summary",
        "root": str(root),
        "b64_root": str(b64_root),
        "additive128_root": str(additive128_root),
        "adaln128_root": str(adaln128_root),
        "run_ids": RUN_IDS,
        "density_bin_count": REQUIRED_DENSITY_BIN_COUNT,
        "selection_mode": REQUIRED_SELECTION_MODE,
        "student_selector_mode": REQUIRED_STUDENT_SELECTOR_MODE,
        "script_contract": REQUIRED_SCRIPT_CONTRACT,
        "protocol_validation_passed": not issues,
        "validation_passed": not issues,
        "issues": issues,
        "optimization_acceptance": optimization_acceptance,
        "improved_candidate": bool(optimization_acceptance["passed"]),
        "training": training_rows,
        "reports": report_rows,
        "comparison_rows": comparison_rows,
    }
    summary_dir = root / "summary"
    _write_json(summary_dir / "gipo_locked_multiaxis_b64_optim_final_summary.json", payload)
    _write_csv(summary_dir / "gipo_locked_multiaxis_b64_optim_training_summary.csv", training_rows)
    _write_csv(summary_dir / "gipo_locked_multiaxis_b64_optim_report_summary.csv", report_rows)
    _write_csv(summary_dir / "gipo_locked_multiaxis_b64_optim_comparison.csv", comparison_rows)
    (summary_dir / "gipo_locked_multiaxis_b64_optim_final_report.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--b64_root", required=True)
    parser.add_argument("--additive128_root", required=True)
    parser.add_argument("--adaln128_root", required=True)
    args = parser.parse_args()
    payload = collect(
        Path(args.root),
        Path(args.b64_root),
        Path(args.additive128_root),
        Path(args.adaln128_root),
    )
    print(json.dumps({"validation_passed": payload["validation_passed"], "run_ids": RUN_IDS}, sort_keys=True))
    if not payload["validation_passed"]:
        raise SystemExit("GIPO 64-bin guarded optimization collection failed validation.")


if __name__ == "__main__":
    main()
