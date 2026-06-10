from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ADDITIVE = "additive_mlp_v1"
ADALN = "adaln_zero_v1"
PANELS = {
    "seen": [4, 8, 12],
    "unseen": [6, 10, 14, 16],
}
REQUIRED_SELECTION_MODE = "weighted_normalized_regret_v1"
REQUIRED_STUDENT_SELECTION_MODE = "validation_ce_v1"
REQUIRED_DENSITY_BIN_COUNT = 64
BASELINE_LABEL = "teacher_additive_student_additive"
RUN_SPECS: dict[str, dict[str, Any]] = {
    "teacher_additive_student_additive": {
        "root_key": "same_style",
        "run_id": "additive_locked_b64_normregret_final",
        "teacher": ADDITIVE,
        "student": ADDITIVE,
        "noncanonical_allowed": False,
    },
    "teacher_additive_student_adaln": {
        "root_key": "mixed",
        "run_id": "teacher_additive_student_adaln_locked_b64_normregret_final",
        "teacher": ADDITIVE,
        "student": ADALN,
        "noncanonical_allowed": True,
    },
    "teacher_adaln_student_additive": {
        "root_key": "mixed",
        "run_id": "teacher_adaln_student_additive_locked_b64_normregret_final",
        "teacher": ADALN,
        "student": ADDITIVE,
        "noncanonical_allowed": True,
    },
    "teacher_adaln_student_adaln": {
        "root_key": "same_style",
        "run_id": "adaln_locked_b64_normregret_final",
        "teacher": ADALN,
        "student": ADALN,
        "noncanonical_allowed": True,
    },
}


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


def _run_root(*, root: Path, same_style_root: Path, spec: Mapping[str, Any]) -> Path:
    root_key = str(spec["root_key"])
    if root_key == "mixed":
        return root
    if root_key == "same_style":
        return same_style_root
    raise ValueError(f"Unknown run root key: {root_key}")


def _conditioning_pair(teacher_style: str, student_style: str) -> str:
    if teacher_style == student_style:
        return teacher_style
    return f"teacher_{teacher_style}__student_{student_style}"


def _density_bin_count(payload: Mapping[str, Any]) -> int | None:
    value = _nested(payload, "density_representation", "reference_bin_count")
    if value is None:
        value = _nested(payload, "teacher_model_config", "density_dim")
    if value is None:
        value = _nested(payload, "student_model_config", "density_dim")
    if value is None:
        return None
    return int(value)


def _close_float(value: Any, expected: float, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(value) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def _observed_role_style(training: Mapping[str, Any], role: str) -> str:
    direct = training.get(f"{role}_conditioning_style")
    if direct:
        return str(direct)
    nested = _nested(training, "conditioning_styles", role)
    if nested:
        return str(nested)
    model_style = _nested(training, f"{role}_model_config", "conditioning_style")
    if model_style:
        return str(model_style)
    shared = str(training.get("gipo_conditioning_style", "") or "")
    return shared if shared in {ADDITIVE, ADALN} else ""


def _training_summary(run_root: Path, run_id: str) -> dict[str, Any]:
    return _load_json(run_root / "policy_runs" / run_id / "gipo_training_summary.json")


def _final_retrain_metadata(run_root: Path, run_id: str) -> dict[str, Any]:
    path = run_root / "policy_runs" / run_id / "final_retrain_metadata.json"
    if not path.exists():
        return {}
    return _load_json(path)


def _script_contract(run_root: Path, run_id: str) -> dict[str, Any]:
    return dict(_final_retrain_metadata(run_root, run_id).get("script_contract", {}) or {})


def _report_summary(run_root: Path, panel: str, run_id: str) -> dict[str, Any]:
    return _load_json(run_root / "locked_reports" / panel / "student" / run_id / "locked_test_gipo_policy_summary.json")


def _validate_training(
    *,
    run_root: Path,
    label: str,
    spec: Mapping[str, Any],
    training: Mapping[str, Any],
    issues: list[str],
) -> None:
    run_id = str(spec["run_id"])
    teacher_style = str(spec["teacher"])
    student_style = str(spec["student"])
    expected_pair = _conditioning_pair(teacher_style, student_style)
    selection = dict(training.get("teacher_checkpoint_selection", {}) or {})
    student_training = dict(training.get("student_training", {}) or {})
    student_selection = dict(training.get("student_checkpoint_selection", student_training.get("student_checkpoint_selection", {})) or {})
    teacher_retrain = dict(training.get("teacher_final_retrain", training.get("final_teacher_retrain", {})) or {})
    student_retrain = dict(training.get("student_final_retrain", {}) or {})
    unseen_selection = dict(training.get("unseen_nfe_selection", {}) or {})
    pseudo = dict(training.get("pseudo_distillation", {}) or {})
    final_metadata = _final_retrain_metadata(run_root, run_id)
    final_selection = dict(final_metadata.get("selection", {}) or {})
    final_retrain_metadata = dict(final_metadata.get("final_retrain", {}) or {})

    _issue(issues, str(training.get("status")) == "completed", f"{label}: training status is not completed")
    _issue(issues, _observed_role_style(training, "teacher") == teacher_style, f"{label}: wrong teacher conditioning style")
    _issue(issues, _observed_role_style(training, "student") == student_style, f"{label}: wrong student conditioning style")
    _issue(
        issues,
        str(training.get("conditioning_pair") or training.get("gipo_conditioning_style")) == expected_pair,
        f"{label}: wrong conditioning pair",
    )
    _issue(
        issues,
        bool(training.get("noncanonical_conditioning_allowed")) == bool(spec["noncanonical_allowed"]),
        f"{label}: wrong noncanonical conditioning opt-in",
    )
    _issue(issues, _density_bin_count(training) == REQUIRED_DENSITY_BIN_COUNT, f"{label}: wrong density bin count")
    _issue(issues, bool(training.get("locked_test_used_for_selection", False)) is False, f"{label}: locked test used for selection")
    _issue(issues, bool(pseudo.get("pseudo_distillation_requested", False)) is False, f"{label}: pseudo distillation requested")
    _issue(issues, bool(student_training.get("pseudo_distillation_used", False)) is False, f"{label}: student pseudo distillation used")
    _issue(issues, float(pseudo.get("pseudo_target_weight", 0.0) or 0.0) == 0.0, f"{label}: pseudo target weight is nonzero")
    _issue(issues, float(student_training.get("pseudo_target_weight", 0.0) or 0.0) == 0.0, f"{label}: student pseudo target weight is nonzero")
    _issue(
        issues,
        _close_float(training.get("student_nfe_smoothness_weight", 0.0), 0.0),
        f"{label}: student smoothness weight is nonzero",
    )
    _issue(
        issues,
        _close_float(student_training.get("student_nfe_smoothness_weight", 0.0), 0.0),
        f"{label}: student training smoothness weight is nonzero",
    )

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

    _issue(issues, bool(teacher_retrain.get("enabled", False)), f"{label}: missing final teacher retrain")
    _issue(issues, bool(teacher_retrain.get("unseen_selection_diagnostics_used", False)), f"{label}: unseen diagnostics not marked in final teacher retrain")
    _issue(issues, bool(teacher_retrain.get("locked_test_used_for_selection", False)) is False, f"{label}: teacher retrain used locked test")

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

    _issue(issues, str(training.get("student_checkpoint_selection_mode")) == REQUIRED_STUDENT_SELECTION_MODE, f"{label}: wrong student selection mode")
    _issue(issues, str(student_selection.get("selection_protocol")) == REQUIRED_STUDENT_SELECTION_MODE, f"{label}: wrong student selection protocol")
    _issue(issues, str(student_selection.get("selection_metric")) == "validation_ce_loss", f"{label}: wrong student selection metric")
    _issue(issues, int(student_selection.get("selected_step") or 0) > 0, f"{label}: missing student selected step")
    _issue(issues, bool(student_selection.get("locked_test_used_for_selection", False)) is False, f"{label}: student selection used locked test")
    _issue(
        issues,
        bool(training.get("student_validation_used_for_selection", False))
        or bool(student_training.get("student_validation_used_for_selection", False)),
        f"{label}: student validation not marked for selection",
    )
    _issue(issues, bool(student_training.get("locked_test_used_for_selection", False)) is False, f"{label}: student training used locked test")
    _issue(issues, bool(student_retrain.get("enabled", False)) and bool(student_retrain.get("performed", False)), f"{label}: missing student final retrain")
    _issue(issues, bool(student_retrain.get("locked_test_used_for_selection", False)) is False, f"{label}: student retrain used locked test")

    if final_metadata:
        _issue(issues, str(final_metadata.get("selection_mode")) == REQUIRED_SELECTION_MODE, f"{label}: final metadata wrong selection mode")
        _issue(issues, str(final_metadata.get("selection_policy")) == REQUIRED_SELECTION_MODE, f"{label}: final metadata wrong selection policy")
        _issue(issues, bool(final_selection.get("locked_test_used_for_selection", False)) is False, f"{label}: final metadata selection used locked test")
        _issue(issues, bool(final_retrain_metadata.get("locked_test_used_for_selection", False)) is False, f"{label}: final metadata retrain used locked test")

    contract = _script_contract(run_root, run_id)
    if contract:
        _issue(issues, int(contract.get("density_bin_count", -1)) == REQUIRED_DENSITY_BIN_COUNT, f"{label}: script contract wrong density bin count")
        _issue(issues, str(contract.get("student_selector_mode")) == REQUIRED_STUDENT_SELECTION_MODE, f"{label}: script contract wrong student selector")
        _issue(issues, bool(contract.get("locked_test_used_for_selection", False)) is False, f"{label}: script contract locked test selection")
        if str(spec["root_key"]) == "mixed" and any(
            key in contract for key in ("teacher_conditioning_style", "student_conditioning_style", "conditioning_pair")
        ):
            _issue(issues, str(contract.get("teacher_conditioning_style")) == teacher_style, f"{label}: script contract wrong teacher style")
            _issue(issues, str(contract.get("student_conditioning_style")) == student_style, f"{label}: script contract wrong student style")
            _issue(issues, str(contract.get("conditioning_pair")) == expected_pair, f"{label}: script contract wrong conditioning pair")


def _validate_report(
    *,
    run_root: Path,
    run_id: str,
    label: str,
    panel: str,
    spec: Mapping[str, Any],
    report: Mapping[str, Any],
    issues: list[str],
) -> None:
    teacher_style = str(spec["teacher"])
    student_style = str(spec["student"])
    expected_pair = _conditioning_pair(teacher_style, student_style)
    expected_nfes = PANELS[panel]
    report_student_style = str(report.get("student_conditioning_style", report.get("conditioning_style", "")))
    report_teacher_style = report.get("teacher_conditioning_style")
    report_pair = report.get("conditioning_pair")
    _issue(issues, str(report.get("status", "completed")) == "completed", f"{label}/{panel}: report status is not completed")
    _issue(issues, str(report.get("conditioning_style")) == student_style, f"{label}/{panel}: wrong report student conditioning")
    _issue(issues, report_student_style == student_style, f"{label}/{panel}: wrong student style")
    if str(spec["root_key"]) == "mixed":
        _issue(issues, str(report_teacher_style or "") == teacher_style, f"{label}/{panel}: wrong or missing teacher style")
        _issue(issues, str(report_pair or "") == expected_pair, f"{label}/{panel}: wrong or missing conditioning pair")
    else:
        if report_teacher_style is not None:
            _issue(issues, str(report_teacher_style) == teacher_style, f"{label}/{panel}: wrong teacher style")
        if report_pair is not None:
            _issue(issues, str(report_pair) == expected_pair, f"{label}/{panel}: wrong conditioning pair")
    _issue(issues, _density_bin_count(report) == REQUIRED_DENSITY_BIN_COUNT, f"{label}/{panel}: wrong report density bin count")
    _issue(issues, str(report.get("selection_mode")) == "reporting", f"{label}/{panel}: report selection_mode is not reporting")
    _issue(issues, str(report.get("teacher_checkpoint_selection_mode")) == REQUIRED_SELECTION_MODE, f"{label}/{panel}: wrong report teacher selector")
    _issue(issues, bool(report.get("locked_test_used_for_selection", False)) is False, f"{label}/{panel}: locked test used for selection")
    _issue(issues, int(report.get("missing_cell_count", -1)) == 0, f"{label}/{panel}: missing locked cells")
    _issue(issues, [int(v) for v in report.get("target_nfe_values", [])] == expected_nfes, f"{label}/{panel}: wrong target NFEs")
    _issue(issues, report.get("mean_crps") is not None and report.get("mean_mase") is not None, f"{label}/{panel}: missing metrics")
    oracle_path = run_root / "locked_reports" / panel / "oracle" / run_id / "locked_test_gipo_teacher_oracle_policy_summary.json"
    _issue(issues, not oracle_path.exists(), f"{label}/{panel}: teacher-oracle report exists")


def _four_way_rows(results: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel in PANELS:
        for label, spec in RUN_SPECS.items():
            report = results[label][panel]
            mean_crps = float(report["mean_crps"])
            mean_mase = float(report["mean_mase"])
            rows.append(
                {
                    "label": label,
                    "run_id": spec["run_id"],
                    "panel": panel,
                    "teacher_conditioning_style": spec["teacher"],
                    "student_conditioning_style": spec["student"],
                    "conditioning_pair": _conditioning_pair(str(spec["teacher"]), str(spec["student"])),
                    "mean_crps": mean_crps,
                    "mean_mase": mean_mase,
                    "balanced_crps_mase": 0.5 * mean_crps + 0.5 * mean_mase,
                }
            )
    return rows


def _comparison_rows(four_way_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows_by_key = {(str(row["panel"]), str(row["label"])): row for row in four_way_rows}
    out: list[dict[str, Any]] = []
    for panel in PANELS:
        baseline = rows_by_key[(panel, BASELINE_LABEL)]
        for label in RUN_SPECS:
            if label == BASELINE_LABEL:
                continue
            candidate = rows_by_key[(panel, label)]
            out.append(
                {
                    "comparison": f"{label}_vs_{BASELINE_LABEL}",
                    "panel": panel,
                    "baseline": BASELINE_LABEL,
                    "candidate": label,
                    "baseline_balanced_crps_mase": float(baseline["balanced_crps_mase"]),
                    "candidate_balanced_crps_mase": float(candidate["balanced_crps_mase"]),
                    "delta_crps_candidate_minus_baseline": float(candidate["mean_crps"]) - float(baseline["mean_crps"]),
                    "delta_mase_candidate_minus_baseline": float(candidate["mean_mase"]) - float(baseline["mean_mase"]),
                    "delta_balanced_candidate_minus_baseline": float(candidate["balanced_crps_mase"]) - float(baseline["balanced_crps_mase"]),
                }
            )
    return out


def _effect_rows(four_way_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows_by_key = {(str(row["panel"]), str(row["label"])): row for row in four_way_rows}
    aa = "teacher_additive_student_additive"
    ad = "teacher_additive_student_adaln"
    da = "teacher_adaln_student_additive"
    dd = "teacher_adaln_student_adaln"
    out: list[dict[str, Any]] = []
    for panel in PANELS:
        def metric(label: str, key: str) -> float:
            return float(rows_by_key[(panel, label)][key])

        def add_effect(name: str, positive: str, negative: str) -> None:
            out.append(
                {
                    "effect": name,
                    "panel": panel,
                    "positive_label": positive,
                    "negative_label": negative,
                    "delta_crps": metric(positive, "mean_crps") - metric(negative, "mean_crps"),
                    "delta_mase": metric(positive, "mean_mase") - metric(negative, "mean_mase"),
                    "delta_balanced_crps_mase": metric(positive, "balanced_crps_mase") - metric(negative, "balanced_crps_mase"),
                }
            )

        add_effect("student_adaln_effect_at_teacher_additive", ad, aa)
        add_effect("teacher_adaln_effect_at_student_additive", da, aa)
        add_effect("student_adaln_effect_at_teacher_adaln", dd, da)
        add_effect("teacher_adaln_effect_at_student_adaln", dd, ad)
        out.append(
            {
                "effect": "teacher_student_adaln_interaction",
                "panel": panel,
                "formula": "DD - AD - DA + AA",
                "delta_crps": metric(dd, "mean_crps") - metric(ad, "mean_crps") - metric(da, "mean_crps") + metric(aa, "mean_crps"),
                "delta_mase": metric(dd, "mean_mase") - metric(ad, "mean_mase") - metric(da, "mean_mase") + metric(aa, "mean_mase"),
                "delta_balanced_crps_mase": (
                    metric(dd, "balanced_crps_mase")
                    - metric(ad, "balanced_crps_mase")
                    - metric(da, "balanced_crps_mase")
                    + metric(aa, "balanced_crps_mase")
                ),
            }
        )
    return out


def _protocol(root: Path, same_style_root: Path) -> dict[str, Any]:
    return {
        "policy": "predeclared_teacher_student_conditioning_four_way_ab_v1",
        "baseline_label": BASELINE_LABEL,
        "mixed_root": str(root),
        "same_style_reference_root": str(same_style_root),
        "student_locked_reports_only": True,
        "locked_test_used_for_conditioning_selection": False,
        "selection_source": "predeclared_before_locked_reporting",
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# GIPO Teacher/Student Conditioning Four-Way A/B",
        "",
        f"- Validation passed: `{payload['validation_passed']}`",
        f"- Baseline label: `{payload['protocol']['baseline_label']}`",
        f"- Mixed root: `{payload['protocol']['mixed_root']}`",
        f"- Same-style reference root: `{payload['protocol']['same_style_reference_root']}`",
        f"- Student locked reports only: `{payload['protocol']['student_locked_reports_only']}`",
        "",
        "| Panel | Label | Teacher | Student | CRPS | MASE | Balanced |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in payload["four_way_rows"]:
        lines.append(
            "| {panel} | {label} | {teacher_conditioning_style} | {student_conditioning_style} | "
            "{mean_crps:.6f} | {mean_mase:.6f} | {balanced_crps_mase:.6f} |".format(**row)
        )
    lines.extend(["", "| Panel | Candidate vs AA | Delta CRPS | Delta MASE | Delta Balanced |", "|---|---|---:|---:|---:|"])
    for row in payload["comparison_rows"]:
        lines.append(
            "| {panel} | {candidate} | {delta_crps_candidate_minus_baseline:.6f} | "
            "{delta_mase_candidate_minus_baseline:.6f} | {delta_balanced_candidate_minus_baseline:.6f} |".format(**row)
        )
    lines.extend(["", "| Panel | Effect | Delta CRPS | Delta MASE | Delta Balanced |", "|---|---|---:|---:|---:|"])
    for row in payload["effect_rows"]:
        lines.append(
            "| {panel} | {effect} | {delta_crps:.6f} | {delta_mase:.6f} | "
            "{delta_balanced_crps_mase:.6f} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def collect(root: Path, same_style_root: Path) -> dict[str, Any]:
    issues: list[str] = []
    results: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in RUN_SPECS}
    trainings: dict[str, dict[str, Any]] = {}
    for label, spec in RUN_SPECS.items():
        run_id = str(spec["run_id"])
        run_root = _run_root(root=root, same_style_root=same_style_root, spec=spec)
        training = _training_summary(run_root, run_id)
        trainings[label] = training
        _validate_training(run_root=run_root, label=label, spec=spec, training=training, issues=issues)
        for panel in PANELS:
            report = _report_summary(run_root, panel, run_id)
            _validate_report(
                run_root=run_root,
                run_id=run_id,
                label=label,
                panel=panel,
                spec=spec,
                report=report,
                issues=issues,
            )
            results[label][panel] = dict(report)
    four_way_rows = _four_way_rows(results)
    effect_rows = _effect_rows(four_way_rows)
    validation_passed = not issues
    return {
        "artifact": "gipo_locked_multiaxis_b64_teacher_student_conditioning_ab_four_way_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "same_style_root": str(same_style_root),
        "density_bin_count": REQUIRED_DENSITY_BIN_COUNT,
        "teacher_checkpoint_selection_mode": REQUIRED_SELECTION_MODE,
        "student_checkpoint_selection_mode": REQUIRED_STUDENT_SELECTION_MODE,
        "run_specs": RUN_SPECS,
        "reports": results,
        "training": {
            label: {
                "run_id": RUN_SPECS[label]["run_id"],
                "teacher_conditioning_style": _observed_role_style(training, "teacher"),
                "student_conditioning_style": _observed_role_style(training, "student"),
                "conditioning_pair": training.get("conditioning_pair") or training.get("gipo_conditioning_style"),
                "selected_teacher_step": _nested(training, "teacher_checkpoint_selection", "selected_step"),
                "selected_student_step": _nested(training, "student_checkpoint_selection", "selected_step"),
                "sampled_context_count": training.get("sampled_context_count"),
                "locked_test_used_for_selection": training.get("locked_test_used_for_selection", False),
            }
            for label, training in trainings.items()
        },
        "four_way_rows": four_way_rows,
        "comparison_rows": _comparison_rows(four_way_rows),
        "effect_rows": effect_rows,
        "protocol": _protocol(root, same_style_root),
        "validation_passed": validation_passed,
        "issues": issues,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect four-way teacher/student conditioning A/B summary.")
    parser.add_argument("--root", "--mixed-root", "--mixed_root", dest="root", required=True)
    parser.add_argument(
        "--same-style-root",
        "--same_style_root",
        "--final-ab-root",
        "--final_ab_root",
        dest="same_style_root",
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    root = Path(args.root)
    same_style_root = Path(args.same_style_root)
    payload = collect(root, same_style_root)
    out_dir = root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "gipo_locked_multiaxis_b64_teacher_student_conditioning_ab_four_way_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(out_dir / "gipo_locked_multiaxis_b64_teacher_student_conditioning_ab_four_way_summary.md", payload)
    print(json.dumps({"summary": str(summary_path), "validation_passed": payload["validation_passed"]}, sort_keys=True))
    if not payload["validation_passed"]:
        raise SystemExit("GIPO teacher/student conditioning collection failed validation.")


if __name__ == "__main__":
    main()
