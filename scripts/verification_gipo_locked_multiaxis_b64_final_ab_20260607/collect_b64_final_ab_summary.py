from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_IDS = {
    "additive64_final": "additive_locked_b64_normregret_final",
    "adaln64_final": "adaln_locked_b64_normregret_final",
}
CANONICAL_LABEL = "additive64_final"
SIDECAR_LABELS = ("adaln64_final",)
EXPECTED_CONDITIONING = {
    "additive64_final": "additive_mlp_v1",
    "adaln64_final": "adaln_zero_v1",
}
EXPECTED_NONCANONICAL_ALLOWED = {
    "additive64_final": False,
    "adaln64_final": True,
}
PANELS = {
    "seen": [4, 8, 12],
    "unseen": [6, 10, 14, 16],
}
REQUIRED_SELECTION_MODE = "weighted_normalized_regret_v1"
REQUIRED_STUDENT_SELECTION_MODE = "validation_ce_v1"
REQUIRED_STUDENT_TARGET_PROTOCOL = "teacher_weighted_soft_mixture_v1"
REQUIRED_DENSITY_BIN_COUNT = 64


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


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
    if value is None:
        return None
    return int(value)


def _pseudo_weight(training: Mapping[str, Any]) -> float:
    return float(_nested(training, "pseudo_distillation", "pseudo_target_weight", default=0.0) or 0.0)


def _script_contract(summary_root: Path, run_id: str) -> dict[str, Any]:
    return _load_json(summary_root / "policy_runs" / run_id / "final_retrain_metadata.json").get("script_contract", {})


def _final_retrain_metadata(summary_root: Path, run_id: str) -> dict[str, Any]:
    return _load_json(summary_root / "policy_runs" / run_id / "final_retrain_metadata.json")


def _training_summary(root: Path, run_id: str) -> dict[str, Any]:
    return _load_json(root / "policy_runs" / run_id / "gipo_training_summary.json")


def _report_summary(root: Path, panel: str, run_id: str) -> dict[str, Any]:
    return _load_json(root / "locked_reports" / panel / "student" / run_id / "locked_test_gipo_policy_summary.json")


def _validate_training(
    *,
    root: Path,
    label: str,
    run_id: str,
    training: Mapping[str, Any],
    issues: list[str],
) -> None:
    expected_conditioning = EXPECTED_CONDITIONING[label]
    _issue(issues, str(training.get("status")) == "completed", f"{label}: training status is not completed")
    _issue(issues, str(training.get("gipo_conditioning_style")) == expected_conditioning, f"{label}: wrong gipo_conditioning_style")
    _issue(
        issues,
        str(_nested(training, "teacher_model_config", "conditioning_style")) == expected_conditioning,
        f"{label}: wrong teacher conditioning_style",
    )
    _issue(
        issues,
        str(_nested(training, "student_model_config", "conditioning_style")) == expected_conditioning,
        f"{label}: wrong student conditioning_style",
    )
    _issue(
        issues,
        bool(training.get("noncanonical_conditioning_allowed")) == EXPECTED_NONCANONICAL_ALLOWED[label],
        f"{label}: wrong noncanonical_conditioning_allowed flag",
    )
    _issue(issues, _density_bin_count(training) == REQUIRED_DENSITY_BIN_COUNT, f"{label}: wrong density bin count")
    _issue(issues, bool(training.get("locked_test_used_for_selection", False)) is False, f"{label}: locked test used for selection")
    _issue(issues, _pseudo_weight(training) == 0.0, f"{label}: pseudo target weight is nonzero")
    _issue(issues, str(training.get("student_target_protocol")) == REQUIRED_STUDENT_TARGET_PROTOCOL, f"{label}: wrong student target protocol")

    selection = dict(training.get("teacher_checkpoint_selection", {}) or {})
    _issue(issues, str(training.get("teacher_checkpoint_selection_mode")) == REQUIRED_SELECTION_MODE, f"{label}: wrong teacher selection mode")
    _issue(issues, str(selection.get("selection_protocol")) == REQUIRED_SELECTION_MODE, f"{label}: wrong teacher selection protocol")
    _issue(issues, int(selection.get("selected_step") or 0) > 0, f"{label}: missing teacher selected step")
    _issue(issues, selection.get("selected_weighted_normalized_regret_v1_score") is not None, f"{label}: missing normalized-regret score")
    _issue(issues, bool(selection.get("uses_unseen_nfe_selection_diagnostics", False)), f"{label}: missing unseen-NFE selector diagnostics")
    _issue(issues, bool(selection.get("uses_validation_labels", False)) is False, f"{label}: teacher selection uses validation labels")
    _issue(issues, bool(selection.get("locked_test_used_for_selection", False)) is False, f"{label}: teacher selection used locked test")
    expected_splits = {"context_disjoint", "density_family_holdout", "unseen_nfe_holdout"}
    _issue(
        issues,
        expected_splits <= set((selection.get("selected_normalized_regret_values") or {}).keys()),
        f"{label}: normalized-regret splits do not match J_CD/J_CDN",
    )
    weights = dict(selection.get("selected_component_weights", {}) or {})
    _issue(
        issues,
        abs(float(weights.get("context_disjoint", 0.0)) - 0.25) <= 1e-6
        and abs(float(weights.get("density_family_holdout", 0.0)) - 0.25) <= 1e-6
        and abs(float(weights.get("unseen_nfe_holdout", 0.0)) - 0.50) <= 1e-6,
        f"{label}: normalized-regret weights do not match J_CDN",
    )

    teacher_retrain = dict(training.get("teacher_final_retrain", training.get("final_teacher_retrain", {})) or {})
    _issue(issues, bool(teacher_retrain.get("enabled", False)), f"{label}: missing final teacher retrain")
    _issue(issues, bool(teacher_retrain.get("unseen_selection_diagnostics_used", False)), f"{label}: unseen diagnostics not marked in final teacher retrain")
    _issue(issues, bool(teacher_retrain.get("locked_test_used_for_selection", False)) is False, f"{label}: teacher retrain used locked test")

    unseen_selection = dict(training.get("unseen_nfe_selection", {}) or {})
    _issue(issues, bool(unseen_selection.get("enabled", False)), f"{label}: unseen_nfe_selection disabled")
    _issue(issues, [int(v) for v in unseen_selection.get("target_nfes", [])] == [6, 10, 14, 16], f"{label}: wrong unseen selection NFEs")
    _issue(
        issues,
        str(unseen_selection.get("raw_csv", "")).endswith("/artifacts/unseen_train_supervision_rows.csv")
        or str(unseen_selection.get("raw_csv", "")).endswith("\\artifacts\\unseen_train_supervision_rows.csv"),
        f"{label}: wrong unseen selection rows CSV",
    )
    _issue(issues, bool(unseen_selection.get("used_for_final_fitting", True)) is False, f"{label}: unseen rows used for final fitting")
    _issue(issues, bool(unseen_selection.get("locked_test_used_for_selection", False)) is False, f"{label}: unseen selection used locked test")

    student_selection = dict(training.get("student_checkpoint_selection", {}) or {})
    student_training = dict(training.get("student_training", {}) or {})
    student_retrain = dict(training.get("student_final_retrain", {}) or {})
    _issue(issues, bool(student_training.get("pseudo_distillation_used", False)) is False, f"{label}: student pseudo distillation used")
    _issue(issues, float(student_training.get("pseudo_target_weight", 0.0) or 0.0) == 0.0, f"{label}: student pseudo target weight is nonzero")
    _issue(issues, str(student_training.get("student_target_protocol")) == REQUIRED_STUDENT_TARGET_PROTOCOL, f"{label}: wrong student training target protocol")
    _issue(issues, str(training.get("student_checkpoint_selection_mode")) == REQUIRED_STUDENT_SELECTION_MODE, f"{label}: wrong student selection mode")
    _issue(issues, str(student_selection.get("selection_protocol")) == REQUIRED_STUDENT_SELECTION_MODE, f"{label}: wrong student selection protocol")
    _issue(issues, str(student_selection.get("selection_metric")) == "validation_ce_loss", f"{label}: wrong student selection metric")
    _issue(issues, int(student_selection.get("selected_step") or 0) > 0, f"{label}: missing student selected step")
    _issue(
        issues,
        bool(training.get("student_validation_used_for_selection", False))
        or bool(student_training.get("student_validation_used_for_selection", False)),
        f"{label}: student validation not used for selection",
    )
    _issue(issues, bool(student_retrain.get("enabled", False)) and bool(student_retrain.get("performed", False)), f"{label}: missing student final retrain")
    _issue(issues, bool(student_retrain.get("locked_test_used_for_selection", False)) is False, f"{label}: student retrain used locked test")

    contract = _script_contract(root, run_id)
    _issue(issues, int(contract.get("density_bin_count", -1)) == REQUIRED_DENSITY_BIN_COUNT, f"{label}: script contract wrong density bin count")
    _issue(issues, str(contract.get("student_selector_mode")) == REQUIRED_STUDENT_SELECTION_MODE, f"{label}: script contract wrong student selector")
    _issue(issues, bool(contract.get("locked_test_used_for_selection", False)) is False, f"{label}: script contract locked test selection")
    final_metadata = _final_retrain_metadata(root, run_id)
    final_retrain = dict(final_metadata.get("final_retrain", {}) or {})
    final_selection = dict(final_metadata.get("selection", {}) or {})
    _issue(issues, bool(final_retrain.get("locked_test_used_for_selection", False)) is False, f"{label}: final metadata retrain used locked test")
    _issue(issues, bool(final_selection.get("locked_test_used_for_selection", False)) is False, f"{label}: final metadata selection used locked test")


def _validate_report(
    *,
    root: Path,
    label: str,
    panel: str,
    report: Mapping[str, Any],
    issues: list[str],
) -> None:
    expected_conditioning = EXPECTED_CONDITIONING[label]
    expected_nfes = PANELS[panel]
    _issue(issues, str(report.get("status", "completed")) == "completed", f"{label}/{panel}: report status is not completed")
    _issue(issues, str(report.get("conditioning_style")) == expected_conditioning, f"{label}/{panel}: wrong report conditioning")
    _issue(issues, _density_bin_count(report) == REQUIRED_DENSITY_BIN_COUNT, f"{label}/{panel}: wrong report density bin count")
    _issue(issues, str(report.get("selection_mode")) == "reporting", f"{label}/{panel}: report selection_mode is not reporting")
    _issue(issues, str(report.get("teacher_checkpoint_selection_mode")) == REQUIRED_SELECTION_MODE, f"{label}/{panel}: wrong report teacher selector")
    _issue(issues, bool(report.get("locked_test_used_for_selection", False)) is False, f"{label}/{panel}: locked test used for selection")
    _issue(issues, int(report.get("missing_cell_count", -1)) == 0, f"{label}/{panel}: missing locked cells")
    _issue(issues, [int(v) for v in report.get("target_nfe_values", [])] == expected_nfes, f"{label}/{panel}: wrong target NFEs")
    _issue(issues, report.get("mean_crps") is not None and report.get("mean_mase") is not None, f"{label}/{panel}: missing metrics")
    oracle_path = root / "locked_reports" / panel / "oracle" / RUN_IDS[label] / "locked_test_gipo_teacher_oracle_policy_summary.json"
    _issue(issues, not oracle_path.exists(), f"{label}/{panel}: teacher-oracle report exists in final student-only stream")


def _comparison_rows(results: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel in PANELS:
        baseline = results["additive64_final"][panel]
        candidate = results["adaln64_final"][panel]
        baseline_balanced = 0.5 * float(baseline["mean_crps"]) + 0.5 * float(baseline["mean_mase"])
        candidate_balanced = 0.5 * float(candidate["mean_crps"]) + 0.5 * float(candidate["mean_mase"])
        rows.append(
            {
                "comparison": "adaln64_final_vs_additive64_final",
                "mode": "student",
                "panel": panel,
                "baseline": "additive64_final",
                "candidate": "adaln64_final",
                "baseline_crps": float(baseline["mean_crps"]),
                "baseline_mase": float(baseline["mean_mase"]),
                "candidate_crps": float(candidate["mean_crps"]),
                "candidate_mase": float(candidate["mean_mase"]),
                "baseline_balanced_crps_mase": baseline_balanced,
                "candidate_balanced_crps_mase": candidate_balanced,
                "delta_crps_candidate_minus_baseline": float(candidate["mean_crps"]) - float(baseline["mean_crps"]),
                "delta_mase_candidate_minus_baseline": float(candidate["mean_mase"]) - float(baseline["mean_mase"]),
                "delta_balanced_candidate_minus_baseline": candidate_balanced - baseline_balanced,
            }
        )
    return rows


def _canonical_protocol() -> dict[str, Any]:
    return {
        "policy": "predeclared_additive_canonical_with_adaln_sidecar_v1",
        "canonical_label": CANONICAL_LABEL,
        "canonical_run_id": RUN_IDS[CANONICAL_LABEL],
        "canonical_conditioning_style": EXPECTED_CONDITIONING[CANONICAL_LABEL],
        "sidecar_labels": list(SIDECAR_LABELS),
        "sidecar_run_ids": {label: RUN_IDS[label] for label in SIDECAR_LABELS},
        "sidecar_conditioning_styles": {label: EXPECTED_CONDITIONING[label] for label in SIDECAR_LABELS},
        "sidecar_results_are_reporting_only": True,
        "locked_test_used_for_conditioning_selection": False,
        "selection_source": "predeclared_before_locked_reporting",
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    rows = payload["comparison_rows"]
    protocol = payload["canonical_protocol"]
    lines = [
        "# GIPO Final 64-Bin Conditioning A/B",
        "",
        f"- Validation passed: `{payload['validation_passed']}`",
        f"- Canonical run: `{protocol['canonical_run_id']}`",
        f"- Canonical conditioning: `{protocol['canonical_conditioning_style']}`",
        f"- Sidecar runs: `{protocol['sidecar_run_ids']}`",
        f"- Locked-test conditioning selection: `{protocol['locked_test_used_for_conditioning_selection']}`",
        "",
        "| Panel | Additive CRPS | Additive MASE | AdaLN CRPS | AdaLN MASE | Delta Balanced |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {panel} | {baseline_crps:.6f} | {baseline_mase:.6f} | {candidate_crps:.6f} | "
            "{candidate_mase:.6f} | {delta_balanced_candidate_minus_baseline:.6f} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def collect(root: Path) -> dict[str, Any]:
    issues: list[str] = []
    results: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in RUN_IDS}
    trainings: dict[str, dict[str, Any]] = {}
    for label, run_id in RUN_IDS.items():
        training = _training_summary(root, run_id)
        trainings[label] = training
        _validate_training(root=root, label=label, run_id=run_id, training=training, issues=issues)
        for panel in PANELS:
            report = _report_summary(root, panel, run_id)
            _validate_report(root=root, label=label, panel=panel, report=report, issues=issues)
            results[label][panel] = dict(report)
    comparison_rows = _comparison_rows(results)
    canonical_protocol = _canonical_protocol()
    validation_passed = not issues
    return {
        "artifact": "gipo_locked_multiaxis_b64_final_ab_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "density_bin_count": REQUIRED_DENSITY_BIN_COUNT,
        "teacher_checkpoint_selection_mode": REQUIRED_SELECTION_MODE,
        "student_checkpoint_selection_mode": REQUIRED_STUDENT_SELECTION_MODE,
        "student_target_protocol": REQUIRED_STUDENT_TARGET_PROTOCOL,
        "run_ids": dict(RUN_IDS),
        "reports": results,
        "training": {
            label: {
                "run_id": RUN_IDS[label],
                "conditioning_style": training.get("gipo_conditioning_style"),
                "student_target_protocol": training.get("student_target_protocol"),
                "selected_teacher_step": _nested(training, "teacher_checkpoint_selection", "selected_step"),
                "selected_student_step": _nested(training, "student_checkpoint_selection", "selected_step"),
                "sampled_context_count": training.get("sampled_context_count"),
                "locked_test_used_for_selection": training.get("locked_test_used_for_selection", False),
            }
            for label, training in trainings.items()
        },
        "comparison_rows": comparison_rows,
        "canonical_protocol": canonical_protocol,
        "validation_passed": validation_passed,
        "issues": issues,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect final 64-bin GIPO additive/AdaLN A/B summary.")
    parser.add_argument("--root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    root = Path(args.root)
    payload = collect(root)
    out_dir = root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "gipo_locked_multiaxis_b64_final_ab_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(out_dir / "gipo_locked_multiaxis_b64_final_ab_summary.md", payload)
    print(json.dumps({"summary": str(summary_path), "validation_passed": payload["validation_passed"]}, sort_keys=True))
    if not payload["validation_passed"]:
        raise SystemExit("GIPO final 64-bin conditioning A/B collection failed validation.")


if __name__ == "__main__":
    main()
