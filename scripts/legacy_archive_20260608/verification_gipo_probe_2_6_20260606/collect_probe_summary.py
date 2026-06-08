from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


VECTOR_RUN_SPECS: tuple[dict[str, Any], ...] = (
    {"run_id": "vec_teacher_eq005_b128", "teacher_utility_weights": {"crps": 0.50, "mase": 0.50}},
    {"run_id": "vec_teacher_mase06_b128", "teacher_utility_weights": {"crps": 0.40, "mase": 0.60}},
    {"run_id": "vec_teacher_crps075_b128", "teacher_utility_weights": {"crps": 0.75, "mase": 0.25}},
)
PROBE_RUN_SPECS: tuple[dict[str, Any], ...] = (
    {"run_id": "probe_pseudo010_mase06_b128", "pseudo_weight": 0.10, "teacher_utility_weights": {"crps": 0.40, "mase": 0.60}, "student_target_mode": "soft_mixture"},
    {"run_id": "probe_pseudo025_mase06_b128", "pseudo_weight": 0.25, "teacher_utility_weights": {"crps": 0.40, "mase": 0.60}, "student_target_mode": "soft_mixture"},
    {"run_id": "probe_pseudo050_mase06_b128", "pseudo_weight": 0.50, "teacher_utility_weights": {"crps": 0.40, "mase": 0.60}, "student_target_mode": "soft_mixture"},
    {"run_id": "probe_pseudo025_mase075_b128", "pseudo_weight": 0.25, "teacher_utility_weights": {"crps": 0.25, "mase": 0.75}, "student_target_mode": "soft_mixture"},
    {"run_id": "probe_pseudo025_margin_mase06_b128", "pseudo_weight": 0.25, "teacher_utility_weights": {"crps": 0.40, "mase": 0.60}, "student_target_mode": "margin_hard_soft"},
    {"run_id": "probe_pseudo025_margin_mase075_b128", "pseudo_weight": 0.25, "teacher_utility_weights": {"crps": 0.25, "mase": 0.75}, "student_target_mode": "margin_hard_soft"},
)
FOLLOWUP_RUN_SPECS: tuple[dict[str, Any], ...] = (
    {"run_id": "probe_pseudo025_mase06_student1000_b128", "pseudo_weight": 0.25, "teacher_utility_weights": {"crps": 0.40, "mase": 0.60}, "student_target_mode": "soft_mixture", "followup_type": "student_steps_1000"},
    {"run_id": "probe_pseudo025_mase06_smooth005_b128", "pseudo_weight": 0.25, "teacher_utility_weights": {"crps": 0.40, "mase": 0.60}, "student_target_mode": "soft_mixture", "followup_type": "smoothness_0.005"},
)
VECTOR_BASELINE_RUN_ID = "vec_teacher_mase06_b128"
OLD_OFFICIAL = {
    "seen_locked": {"crps": 3.38033, "mase": 2.52072, "source": "temp_fixed_005_b128 retained locked"},
    "zero_shot_unseen_locked": {"crps": 3.36889, "mase": 2.52310, "source": "temp_fixed_005_b128 retained unseen locked"},
}
REQUIRED_MODEL_CONFIG = {
    "teacher_architecture": "density_form_transformer_v1",
    "student_architecture": "density_query_transformer_v1",
    "setting_encoder_mode": "continuous_v3",
    "attention_heads": 4,
    "conditioning_style": "additive_mlp_v1",
    "density_token_attention": "bin_self_attention_rope_v1",
}
PRIMARY_NFES = (10, 16)
REQUIRED_NFES = (6, 10, 14, 16)
SEEN_NFES = (4, 8, 12)
EXPECTED_SOLVER_COUNT = 4
CRPS_MEAN_GUARDRAIL_FLOOR = -0.005
CRPS_PER_NFE_GUARDRAIL_FLOOR = -0.01
FOLLOWUP_ORACLE_STUDENT_MASE_GAP = 0.0025


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: Iterable[Any]) -> float | None:
    vals = [float(value) for value in (_safe_float(item) for item in values) if value is not None]
    return None if not vals else float(sum(vals) / len(vals))


def _gain(value: Any, reference: Any) -> float | None:
    value_f = _safe_float(value)
    ref_f = _safe_float(reference)
    if value_f is None or ref_f is None or abs(ref_f) <= 1e-12:
        return None
    return float((ref_f - value_f) / ref_f)


def _comparison_path(summary: Mapping[str, Any], *, summary_path: Path) -> Path | None:
    value = str(summary.get("comparison_summary_path", "") or "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else summary_path.parent / path


def _comparison(summary: Mapping[str, Any], *, summary_path: Path) -> dict[str, Any]:
    path = _comparison_path(summary, summary_path=summary_path)
    return {} if path is None else _maybe_json(path)


def _missing_lists(summary: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, list[Any]]:
    return {
        "missing_expected_cells": list(summary.get("missing_expected_cells", []) or []),
        "missing_baseline_cells": list(comparison.get("missing_baseline_cells", []) or []),
        "missing_ser_ptg_cells": list(comparison.get("missing_ser_ptg_cells", []) or []),
        "missing_student_cells": list(comparison.get("missing_student_cells", []) or []),
    }


def _strict_locked_selection_issue(summary: Mapping[str, Any], *, label: str) -> str | None:
    return None if summary.get("locked_test_used_for_selection") is False else f"{label}_locked_test_used_for_selection"


def _comparison_coverage_issues(
    comparison: Mapping[str, Any],
    *,
    label: str,
    expected_nfes: tuple[int, ...],
) -> list[str]:
    cells = list(comparison.get("cell_rankings", []) or [])
    issues: list[str] = []
    if not cells:
        return [f"{label}_missing_cell_rankings"]
    observed_nfes: set[int] = set()
    observed_keys: set[tuple[str, int]] = set()
    for cell in cells:
        nfe = _safe_float(cell.get("target_nfe"))
        if nfe is None:
            issues.append(f"{label}_missing_target_nfe")
            continue
        nfe_int = int(nfe)
        observed_nfes.add(nfe_int)
        solver = str(cell.get("solver_key") or cell.get("solver_name") or "")
        observed_keys.add((solver, nfe_int))
        comparisons = list(cell.get("student_comparisons", []) or [])
        if not comparisons:
            issues.append(f"{label}_missing_student_comparison:{solver}:{nfe_int}")
    missing_nfes = sorted(set(expected_nfes) - observed_nfes)
    extra_nfes = sorted(observed_nfes - set(expected_nfes))
    if missing_nfes:
        issues.append(f"{label}_missing_nfes:{missing_nfes}")
    if extra_nfes:
        issues.append(f"{label}_unexpected_nfes:{extra_nfes}")
    expected_cells = len(expected_nfes) * EXPECTED_SOLVER_COUNT
    if len(observed_keys) < expected_cells:
        issues.append(f"{label}_insufficient_cells:{len(observed_keys)}<{expected_cells}")
    return issues


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        for row in rows:
            writer.writerow(row)


def _density_entropy(row: Mapping[str, Any]) -> float | None:
    text = row.get("density_mass_json")
    if not text:
        return None
    try:
        mass = [max(float(value), 1e-12) for value in json.loads(str(text))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    total = sum(mass)
    if total <= 0.0:
        return None
    probs = [value / total for value in mass]
    return float(-sum(value * math.log(value) for value in probs))


def _interval_stats(row: Mapping[str, Any]) -> dict[str, float | None]:
    text = row.get("time_grid_json")
    if not text:
        return {"min_interval": None, "max_interval": None, "tail_fraction_after_098": None}
    try:
        grid = [float(value) for value in json.loads(str(text))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"min_interval": None, "max_interval": None, "tail_fraction_after_098": None}
    intervals = [b - a for a, b in zip(grid, grid[1:])]
    if not intervals:
        return {"min_interval": None, "max_interval": None, "tail_fraction_after_098": None}
    internal = grid[1:-1]
    return {
        "min_interval": float(min(intervals)),
        "max_interval": float(max(intervals)),
        "tail_fraction_after_098": float(sum(1 for value in internal if value > 0.98) / max(len(internal), 1)),
    }


def _mass(row: Mapping[str, Any]) -> list[float] | None:
    try:
        return [float(value) for value in json.loads(str(row.get("density_mass_json", "")))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _js_divergence(left: list[float], right: list[float]) -> float:
    eps = 1e-12
    p = [max(float(value), eps) for value in left]
    q = [max(float(value), eps) for value in right]
    sp = sum(p)
    sq = sum(q)
    p = [value / sp for value in p]
    q = [value / sq for value in q]
    m = [(a + b) * 0.5 for a, b in zip(p, q)]
    return float(
        0.5 * sum(a * math.log(a / b) for a, b in zip(p, m))
        + 0.5 * sum(a * math.log(a / b) for a, b in zip(q, m))
    )


def _roughness_by_solver_nfe(rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, str, str, str], list[tuple[int, list[float]]]] = defaultdict(list)
    for row in rows:
        mass = _mass(row)
        if mass is None:
            continue
        key = (
            str(row.get("dataset")),
            str(row.get("seed")),
            str(row.get("solver_key")),
            str(row.get("context_id")),
        )
        grouped[key].append((int(row["target_nfe"]), mass))
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for key, items in grouped.items():
        solver = key[2]
        ordered = sorted(items, key=lambda item: item[0])
        for (_, left), (right_nfe, right) in zip(ordered, ordered[1:]):
            values[(solver, int(right_nfe))].append(_js_divergence(left, right))
    return {key: float(sum(items) / len(items)) for key, items in values.items() if items}


def _decision_diagnostics(rows: list[dict[str, Any]], *, prefix: str) -> dict[tuple[str, int], dict[str, Any]]:
    roughness = _roughness_by_solver_nfe(rows)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("solver_key")), int(row.get("target_nfe", -1)))].append(row)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, group in grouped.items():
        entropy = [_density_entropy(row) for row in group]
        intervals = [_interval_stats(row) for row in group]
        top_counts = Counter(str(row.get("teacher_top_schedule_key", "")) for row in group if row.get("teacher_top_schedule_key"))
        out[key] = {
            f"{prefix}_density_entropy_mean": _mean(entropy),
            f"{prefix}_min_interval_mean": _mean(item["min_interval"] for item in intervals),
            f"{prefix}_max_interval_mean": _mean(item["max_interval"] for item in intervals),
            f"{prefix}_tail_fraction_after_098_mean": _mean(item["tail_fraction_after_098"] for item in intervals),
            f"{prefix}_adjacent_density_js_from_prev_nfe_mean": roughness.get(key),
            f"{prefix}_teacher_candidate_ess_mean": _mean(row.get("teacher_candidate_ess") for row in group),
            f"{prefix}_teacher_candidate_max_weight_mean": _mean(row.get("teacher_candidate_max_weight") for row in group),
            f"{prefix}_teacher_top_margin_mean": _mean(row.get("teacher_top_margin") for row in group),
            f"{prefix}_teacher_hard_target_fraction": _mean(1.0 if str(row.get("teacher_hard_target")).lower() in {"true", "1"} else 0.0 for row in group),
            f"{prefix}_teacher_top_schedule_mode": top_counts.most_common(1)[0][0] if top_counts else "",
        }
    return out


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, float | None]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("solver_key")), int(row.get("target_nfe", -1)))].append(row)
    return {
        key: {
            "crps": _mean(row.get("crps") for row in group),
            "mase": _mean(row.get("mase") for row in group),
        }
        for key, group in grouped.items()
    }


def _cell_gain_panel(comparison: Mapping[str, Any]) -> dict[str, Any]:
    cells = list(comparison.get("cell_rankings", []) or [])
    gains: dict[str, list[float]] = defaultdict(list)
    wins: Counter[str] = Counter()
    per_nfe: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        target_nfe = str(cell.get("target_nfe"))
        comp = dict((cell.get("student_comparisons") or [{}])[0])
        mapping = {
            "crps_vs_uniform": comp.get("student_relative_crps_gain_vs_uniform"),
            "mase_vs_uniform": comp.get("student_relative_mase_gain_vs_uniform"),
            "crps_vs_ser_ptg": comp.get("student_relative_crps_gain_vs_ser_ptg"),
            "mase_vs_ser_ptg": comp.get("student_relative_mase_gain_vs_ser_ptg"),
            "crps_vs_best_fixed": comp.get("student_relative_crps_gain_vs_best_baseline"),
            "mase_vs_best_fixed": comp.get("student_relative_mase_gain_vs_best_baseline"),
        }
        for key, value in mapping.items():
            numeric = _safe_float(value)
            if numeric is None:
                continue
            gains[key].append(numeric)
            per_nfe[target_nfe][key].append(numeric)
            wins[key] += int(numeric > 0.0)
    per_nfe_means = {
        nfe: {key: _mean(values) for key, values in metrics.items()}
        for nfe, metrics in sorted(per_nfe.items(), key=lambda item: int(item[0]))
    }
    primary = [
        float(per_nfe_means[str(nfe)]["mase_vs_best_fixed"])
        for nfe in PRIMARY_NFES
        if str(nfe) in per_nfe_means and per_nfe_means[str(nfe)].get("mase_vs_best_fixed") is not None
    ]
    per_nfe_crps = [
        float(metrics["crps_vs_best_fixed"])
        for metrics in per_nfe_means.values()
        if metrics.get("crps_vs_best_fixed") is not None
    ]
    mean_crps = _mean(gains["crps_vs_best_fixed"])
    return {
        "cell_count": int(len(cells)),
        "mean_gains": {key: _mean(values) for key, values in sorted(gains.items())},
        "win_counts": dict(wins),
        "per_nfe_mean_gains": per_nfe_means,
        "primary_nfe10_16_mase_gain_min": min(primary) if len(primary) == len(PRIMARY_NFES) else None,
        "all_unseen_mase_gain_mean": _mean(gains["mase_vs_best_fixed"]),
        "all_unseen_crps_gain_mean": mean_crps,
        "crps_guardrail_passed": bool(
            mean_crps is not None
            and float(mean_crps) >= CRPS_MEAN_GUARDRAIL_FLOOR
            and (not per_nfe_crps or min(per_nfe_crps) >= CRPS_PER_NFE_GUARDRAIL_FLOOR)
        ),
    }


def _training_payload(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(spec["run_id"])
    path = root / "policy_runs" / run_id / "gipo_training_summary.json"
    if not path.exists():
        return {"status": "missing", "training_summary": str(path)}
    summary = _read_json(path)
    teacher_cfg = dict(summary.get("teacher_model_config") or {})
    student_cfg = dict(summary.get("student_model_config") or {})
    student_training = dict(summary.get("student_training") or {})
    pseudo_summary = dict(summary.get("pseudo_distillation") or {})
    teacher_training = dict(summary.get("teacher_training") or {})
    checkpoint_selection = dict(teacher_training.get("teacher_checkpoint_selection") or {})
    weights = dict(summary.get("teacher_utility_weights") or {})
    issues: list[str] = []
    if summary.get("teacher_architecture") != REQUIRED_MODEL_CONFIG["teacher_architecture"]:
        issues.append("teacher_architecture")
    if summary.get("student_architecture") != REQUIRED_MODEL_CONFIG["student_architecture"]:
        issues.append("student_architecture")
    if summary.get("setting_encoder_mode") != REQUIRED_MODEL_CONFIG["setting_encoder_mode"]:
        issues.append("setting_encoder_mode")
    for name, cfg in (("teacher", teacher_cfg), ("student", student_cfg)):
        if cfg.get("attention_heads") != REQUIRED_MODEL_CONFIG["attention_heads"]:
            issues.append(f"{name}_attention_heads")
        if cfg.get("conditioning_style") != REQUIRED_MODEL_CONFIG["conditioning_style"]:
            issues.append(f"{name}_conditioning_style")
        if cfg.get("density_token_attention") != REQUIRED_MODEL_CONFIG["density_token_attention"]:
            issues.append(f"{name}_density_token_attention")
    if summary.get("locked_test_used_for_selection") is not False:
        issues.append("locked_test_used_for_selection_missing_or_not_false")
    expected_weights = dict(spec.get("teacher_utility_weights") or {})
    for key, expected in expected_weights.items():
        observed = _safe_float(weights.get(key))
        if observed is None or abs(observed - float(expected)) > 1e-12:
            issues.append(f"teacher_utility_weight_{key}")
    expected_pseudo = spec.get("pseudo_weight")
    if expected_pseudo is not None:
        if abs(float(student_training.get("pseudo_target_weight", -1.0) or -1.0) - float(expected_pseudo)) > 1e-12:
            issues.append("student_pseudo_target_weight")
        if abs(float(pseudo_summary.get("pseudo_target_weight", -1.0) or -1.0) - float(expected_pseudo)) > 1e-12:
            issues.append("pseudo_target_weight")
        if float(expected_pseudo) > 0.0 and not bool(student_training.get("pseudo_distillation_used", False)):
            issues.append("pseudo_distillation_used")
        pseudo_splits = set(pseudo_summary.get("pseudo_split_phases") or [])
        if float(expected_pseudo) > 0.0 and pseudo_splits != {"train_tuning"}:
            issues.append(f"pseudo_split_phases:{sorted(pseudo_splits)}")
    expected_target_mode = spec.get("student_target_mode")
    if expected_target_mode and summary.get("student_target_mode") != expected_target_mode:
        issues.append("student_target_mode")
    return {
        "status": "completed" if not issues else "invalid",
        "issues": issues,
        "teacher_selected_checkpoint_step": checkpoint_selection.get("selected_step"),
        "teacher_selected_mean_diagnostic_total_loss": checkpoint_selection.get("selected_mean_diagnostic_total_loss"),
        "teacher_checkpoint_selection": checkpoint_selection,
        "teacher_utility_weights": weights,
        "student_target_mode": summary.get("student_target_mode"),
        "pseudo_distillation": pseudo_summary,
        "student_pseudo_target_summary": dict(student_training.get("student_pseudo_target_summary") or {}),
        "student_training_last_loss": (student_training.get("losses") or [{}])[-1],
        "training_summary": str(path),
    }


def _panel_paths(root: Path, run_id: str, panel: str) -> tuple[Path, Path, str, str]:
    if panel == "validation_unseen":
        return (
            root / "validation_unseen_reports" / "student" / run_id / "unseen_validation_gipo_policy_summary.json",
            root / "validation_unseen_reports" / "oracle" / run_id / "unseen_validation_gipo_teacher_oracle_policy_summary.json",
            "unseen_validation_gipo",
            "unseen_validation_gipo_teacher_oracle",
        )
    if panel == "locked_unseen":
        return (
            root / "locked_unseen_reports" / "student" / run_id / "locked_test_gipo_policy_summary.json",
            root / "locked_unseen_reports" / "oracle" / run_id / "locked_test_gipo_teacher_oracle_policy_summary.json",
            "locked_test_gipo",
            "locked_test_gipo_teacher_oracle",
        )
    raise ValueError(f"Unknown panel {panel!r}.")


def _vector_panel_paths(root: Path, run_id: str, panel: str) -> tuple[Path, Path, str, str]:
    if panel == "seen_locked":
        return (
            root / "seen_locked_reports" / "student" / run_id / "locked_test_gipo_policy_summary.json",
            root / "seen_locked_reports" / "oracle" / run_id / "locked_test_gipo_teacher_oracle_policy_summary.json",
            "locked_test_gipo",
            "locked_test_gipo_teacher_oracle",
        )
    if panel == "zero_shot_unseen_validation":
        return (
            root / "zero_shot_unseen_validation_reports" / "student" / run_id / "zero_shot_unseen_validation_gipo_policy_summary.json",
            root / "zero_shot_unseen_validation_reports" / "oracle" / run_id / "zero_shot_unseen_validation_gipo_teacher_oracle_policy_summary.json",
            "zero_shot_unseen_validation_gipo",
            "zero_shot_unseen_validation_gipo_teacher_oracle",
        )
    if panel == "zero_shot_unseen_locked":
        return (
            root / "zero_shot_unseen_locked_reports" / "student" / run_id / "locked_test_gipo_policy_summary.json",
            root / "zero_shot_unseen_locked_reports" / "oracle" / run_id / "locked_test_gipo_teacher_oracle_policy_summary.json",
            "locked_test_gipo",
            "locked_test_gipo_teacher_oracle",
        )
    raise ValueError(f"Unknown vector panel {panel!r}.")


def _panel_payload(
    *,
    run_id: str,
    student_path: Path,
    oracle_path: Path,
    student_prefix: str,
    oracle_prefix: str,
    old_reference: Mapping[str, Any] | None = None,
    expected_nfes: tuple[int, ...] = REQUIRED_NFES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not student_path.exists() or not oracle_path.exists():
        return {
            "status": "missing",
            "student_summary": str(student_path),
            "oracle_summary": str(oracle_path),
        }, []
    student = _read_json(student_path)
    oracle = _read_json(oracle_path)
    student_comparison = _comparison(student, summary_path=student_path)
    oracle_comparison = _comparison(oracle, summary_path=oracle_path)
    student_missing = _missing_lists(student, student_comparison)
    oracle_missing = _missing_lists(oracle, oracle_comparison)
    coverage_issues = (
        _comparison_coverage_issues(student_comparison, label="student", expected_nfes=expected_nfes)
        + _comparison_coverage_issues(oracle_comparison, label="oracle", expected_nfes=expected_nfes)
    )
    strict_flag_issues = [
        issue
        for issue in (
            _strict_locked_selection_issue(student, label="student"),
            _strict_locked_selection_issue(oracle, label="oracle"),
        )
        if issue is not None
    ]
    missing_count = (
        sum(len(values) for values in student_missing.values())
        + sum(len(values) for values in oracle_missing.values())
        + len(coverage_issues)
        + len(strict_flag_issues)
    )
    old_ref = dict(old_reference or {})
    student_rows = _read_csv(student_path.parent / f"{student_prefix}_aggregate_rows.csv")
    oracle_rows = _read_csv(oracle_path.parent / f"{oracle_prefix}_aggregate_rows.csv")
    student_decisions = _read_csv(student_path.parent / f"{student_prefix}_decisions.csv")
    oracle_decisions = _read_csv(oracle_path.parent / f"{oracle_prefix}_decisions.csv")
    student_metrics = _aggregate_metrics(student_rows)
    oracle_metrics = _aggregate_metrics(oracle_rows)
    student_diag = _decision_diagnostics(student_decisions, prefix="student")
    oracle_diag = _decision_diagnostics(oracle_decisions, prefix="oracle")
    diagnostic_rows: list[dict[str, Any]] = []
    for key in sorted(set(student_metrics) | set(oracle_metrics) | set(student_diag) | set(oracle_diag)):
        solver, nfe = key
        s = student_metrics.get(key, {})
        o = oracle_metrics.get(key, {})
        row: dict[str, Any] = {
            "run_id": run_id,
            "solver_key": solver,
            "target_nfe": int(nfe),
            "student_crps": s.get("crps"),
            "student_mase": s.get("mase"),
            "oracle_crps": o.get("crps"),
            "oracle_mase": o.get("mase"),
            "oracle_gain_vs_student_crps": _gain(o.get("crps"), s.get("crps")),
            "oracle_gain_vs_student_mase": _gain(o.get("mase"), s.get("mase")),
        }
        row.update(student_diag.get(key, {}))
        row.update(oracle_diag.get(key, {}))
        diagnostic_rows.append(row)
    payload = {
        "status": "completed" if missing_count == 0 else "incomplete_missing_cells",
        "student_mean_crps": student.get("mean_crps"),
        "student_mean_mase": student.get("mean_mase"),
        "oracle_mean_crps": oracle.get("mean_crps"),
        "oracle_mean_mase": oracle.get("mean_mase"),
        "oracle_gain_vs_student_crps": _gain(oracle.get("mean_crps"), student.get("mean_crps")),
        "oracle_gain_vs_student_mase": _gain(oracle.get("mean_mase"), student.get("mean_mase")),
        "student_gain_vs_old_official_crps": _gain(student.get("mean_crps"), old_ref.get("crps")),
        "student_gain_vs_old_official_mase": _gain(student.get("mean_mase"), old_ref.get("mase")),
        "old_official_reference": old_ref,
        "student_comparison": _cell_gain_panel(student_comparison),
        "oracle_comparison": _cell_gain_panel(oracle_comparison),
        "student_missing": student_missing,
        "oracle_missing": oracle_missing,
        "coverage_issues": coverage_issues,
        "selection_flag_issues": strict_flag_issues,
        "locked_test_used_for_selection": student.get("locked_test_used_for_selection") is not False or oracle.get("locked_test_used_for_selection") is not False,
        "student_summary": str(student_path),
        "oracle_summary": str(oracle_path),
    }
    return payload, diagnostic_rows


def _collect_vector(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.vector_root)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for spec in VECTOR_RUN_SPECS:
        run_id = str(spec["run_id"])
        training = _training_payload(root, spec)
        panels: dict[str, Any] = {}
        for panel_name in ("seen_locked", "zero_shot_unseen_validation", "zero_shot_unseen_locked"):
            student_path, oracle_path, student_prefix, oracle_prefix = _vector_panel_paths(root, run_id, panel_name)
            panel, panel_diag = _panel_payload(
                run_id=run_id,
                student_path=student_path,
                oracle_path=oracle_path,
                student_prefix=student_prefix,
                oracle_prefix=oracle_prefix,
                old_reference=OLD_OFFICIAL.get(panel_name),
                expected_nfes=SEEN_NFES if panel_name == "seen_locked" else REQUIRED_NFES,
            )
            panels[panel_name] = panel
            for item in panel_diag:
                item["source"] = "vector"
                item["panel"] = panel_name
            diagnostics.extend(panel_diag)
        panel_issues = [
            name
            for name, panel in panels.items()
            if panel.get("status") != "completed" or bool(panel.get("locked_test_used_for_selection", False))
        ]
        rows.append(
            {
                "run_id": run_id,
                "source": "vector",
                "status": "completed" if training.get("status") == "completed" and not panel_issues else "incomplete_or_invalid",
                "training": training,
                "panel_issues": panel_issues,
                "panels": panels,
            }
        )
    best = max(
        (
            row for row in rows
            if row["status"] == "completed"
            and row["panels"]["zero_shot_unseen_validation"]["student_comparison"]["mean_gains"].get("mase_vs_best_fixed") is not None
        ),
        key=lambda row: float(row["panels"]["zero_shot_unseen_validation"]["student_comparison"]["mean_gains"]["mase_vs_best_fixed"]),
        default=None,
    )
    return {
        "artifact": "gipo_probe_refreshed_vector_matrix",
        "source_commit": str(args.source_commit),
        "locked_test_used_for_selection": False,
        "vector_root": str(root),
        "run_count": int(len(rows)),
        "completed_count": int(sum(1 for row in rows if row["status"] == "completed")),
        "best_zero_shot_validation_mase_run_id": None if best is None else best["run_id"],
        "rows": rows,
        "diagnostics": diagnostics,
    }


def _available_probe_specs(root: Path) -> list[dict[str, Any]]:
    specs = [dict(item) for item in PROBE_RUN_SPECS]
    for spec in FOLLOWUP_RUN_SPECS:
        if (root / "policy_runs" / str(spec["run_id"]) / "gipo_training_summary.json").exists():
            specs.append(dict(spec))
    return specs


def _collect_probe(args: argparse.Namespace, *, include_locked: bool) -> dict[str, Any]:
    root = Path(args.root)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for spec in _available_probe_specs(root):
        run_id = str(spec["run_id"])
        training = _training_payload(root, spec)
        panels: dict[str, Any] = {}
        for panel_name in ("validation_unseen", "locked_unseen"):
            if panel_name == "locked_unseen" and not include_locked:
                continue
            student_path, oracle_path, student_prefix, oracle_prefix = _panel_paths(root, run_id, panel_name)
            panel, panel_diag = _panel_payload(
                run_id=run_id,
                student_path=student_path,
                oracle_path=oracle_path,
                student_prefix=student_prefix,
                oracle_prefix=oracle_prefix,
                old_reference=OLD_OFFICIAL.get("zero_shot_unseen_locked") if panel_name == "locked_unseen" else None,
                expected_nfes=REQUIRED_NFES,
            )
            panels[panel_name] = panel
            for item in panel_diag:
                item["source"] = "probe"
                item["panel"] = panel_name
                item["student_target_mode"] = spec.get("student_target_mode", "")
                item["pseudo_weight"] = spec.get("pseudo_weight")
                item["followup_type"] = spec.get("followup_type", "")
            diagnostics.extend(panel_diag)
        selection_panel_issues = [
            name
            for name, panel in panels.items()
            if name == "validation_unseen"
            and (panel.get("status") != "completed" or bool(panel.get("locked_test_used_for_selection", False)))
        ]
        locked_panel_issues = [
            name
            for name, panel in panels.items()
            if name == "locked_unseen"
            and (panel.get("status") != "completed" or bool(panel.get("locked_test_used_for_selection", False)))
        ]
        rows.append(
            {
                "run_id": run_id,
                "source": "probe",
                "status": "completed" if training.get("status") == "completed" and not selection_panel_issues else "incomplete_or_invalid",
                "spec": dict(spec),
                "training": training,
                "selection_panel_issues": selection_panel_issues,
                "locked_panel_issues": locked_panel_issues,
                "panel_issues": selection_panel_issues + locked_panel_issues,
                "panels": panels,
            }
        )
    return {
        "rows": rows,
        "diagnostics": diagnostics,
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
    panel = dict(dict(row.get("panels") or {}).get("validation_unseen") or {})
    gains = dict(dict(panel.get("student_comparison") or {}).get("mean_gains") or {})
    ranking = dict(panel.get("student_comparison") or {})
    primary = _safe_float(ranking.get("primary_nfe10_16_mase_gain_min"))
    secondary = _safe_float(ranking.get("all_unseen_mase_gain_mean"))
    crps = _safe_float(ranking.get("all_unseen_crps_gain_mean"))
    return (
        float(primary if primary is not None else -999.0),
        float(secondary if secondary is not None else -999.0),
        float(crps if crps is not None else -999.0),
        str(row.get("run_id")),
    )


def _select_candidates(probe_rows: list[dict[str, Any]]) -> list[str]:
    completed = [
        row for row in probe_rows
        if row.get("status") == "completed"
        and "validation_unseen" in dict(row.get("panels") or {})
    ]
    guardrail_passed = [
        row for row in completed
        if bool(dict(dict(row["panels"]["validation_unseen"].get("student_comparison") or {})).get("crps_guardrail_passed", False))
    ]
    pool = guardrail_passed or completed
    ordered = sorted(pool, key=_selection_key, reverse=True)
    return [str(row["run_id"]) for row in ordered[:2]]


def _followup_decision(probe_rows: list[dict[str, Any]]) -> tuple[bool, float | None, str | None]:
    completed = [row for row in probe_rows if row.get("status") == "completed" and "validation_unseen" in dict(row.get("panels") or {})]
    guardrail_passed = [
        row for row in completed
        if bool(dict(dict(row["panels"]["validation_unseen"].get("student_comparison") or {})).get("crps_guardrail_passed", False))
    ]
    best = max(guardrail_passed or completed, key=_selection_key, default=None)
    if best is None:
        return False, None, None
    panel = best["panels"]["validation_unseen"]
    gap = _safe_float(panel.get("oracle_gain_vs_student_mase"))
    return bool(gap is not None and gap >= FOLLOWUP_ORACLE_STUDENT_MASE_GAP), gap, str(best["run_id"])


def _write_selection_files(summary_dir: Path, selected: list[str], *, followups_requested: bool) -> None:
    locked = []
    for run_id in [*selected, VECTOR_BASELINE_RUN_ID]:
        if run_id not in locked:
            locked.append(run_id)
    (summary_dir / "selected_locked_run_ids.txt").write_text("\n".join(locked) + "\n", encoding="utf-8")
    followups = [spec["run_id"] for spec in FOLLOWUP_RUN_SPECS] if followups_requested else []
    (summary_dir / "followup_run_ids.txt").write_text("\n".join(followups) + ("\n" if followups else ""), encoding="utf-8")


def _read_selection_file(summary_dir: Path) -> list[str]:
    path = summary_dir / "selected_locked_run_ids.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _vector_locked_row_from_probe_or_reference(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    root = Path(args.root)
    vector_root = Path(args.vector_root)
    spec = next((item for item in VECTOR_RUN_SPECS if str(item["run_id"]) == run_id), {"run_id": run_id})
    training = _training_payload(vector_root, spec)
    student_path, oracle_path, student_prefix, oracle_prefix = _panel_paths(root, run_id, "locked_unseen")
    if not student_path.exists() or not oracle_path.exists():
        student_path, oracle_path, student_prefix, oracle_prefix = _vector_panel_paths(vector_root, run_id, "zero_shot_unseen_locked")
    panel, panel_diag = _panel_payload(
        run_id=run_id,
        student_path=student_path,
        oracle_path=oracle_path,
        student_prefix=student_prefix,
        oracle_prefix=oracle_prefix,
        old_reference=OLD_OFFICIAL.get("zero_shot_unseen_locked"),
        expected_nfes=REQUIRED_NFES,
    )
    for item in panel_diag:
        item["source"] = "vector_locked_selected"
        item["panel"] = "locked_unseen"
    return {
        "run_id": run_id,
        "source": "vector",
        "status": "completed" if training.get("status") == "completed" and panel.get("status") == "completed" else "incomplete_or_invalid",
        "training": training,
        "panel_issues": [] if panel.get("status") == "completed" else ["locked_unseen"],
        "panels": {"locked_unseen": panel},
        "_diagnostics": panel_diag,
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# GIPO Probe 2-6 Summary",
        "",
        f"- Source commit: `{payload.get('source_commit')}`",
        f"- Probe root: `{payload.get('root')}`",
        f"- Vector baseline: `{VECTOR_BASELINE_RUN_ID}`",
        f"- Selected locked run IDs: {', '.join(payload.get('selected_locked_run_ids') or [])}",
        f"- Follow-up requested: {payload.get('followup_requested')} (best={payload.get('followup_trigger_run_id')}, gap={payload.get('followup_oracle_student_mase_gap')})",
        "",
        "## Validation Runs",
    ]
    for row in payload.get("probe_rows", []):
        panel = dict(dict(row.get("panels") or {}).get("validation_unseen") or {})
        ranking = dict(panel.get("student_comparison") or {})
        lines.append(
            "- `{run}` {status}: student CRPS {crps}, MASE {mase}; oracle CRPS {ocrps}, MASE {omase}; primary NFE10/16 MASE {primary}; mean CRPS {crpsgain}; guardrail {guard}".format(
                run=row.get("run_id"),
                status=row.get("status"),
                crps=panel.get("student_mean_crps"),
                mase=panel.get("student_mean_mase"),
                ocrps=panel.get("oracle_mean_crps"),
                omase=panel.get("oracle_mean_mase"),
                primary=ranking.get("primary_nfe10_16_mase_gain_min"),
                crpsgain=ranking.get("all_unseen_crps_gain_mean"),
                guard=ranking.get("crps_guardrail_passed"),
            )
        )
    if payload.get("locked_rows"):
        lines.extend(["", "## Locked Reporting"])
        for row in payload.get("locked_rows", []):
            panel = dict(dict(row.get("panels") or {}).get("locked_unseen") or {})
            lines.append(
                f"- `{row.get('run_id')}`: student CRPS {panel.get('student_mean_crps')}, MASE {panel.get('student_mean_mase')}; "
                f"old official gains CRPS {panel.get('student_gain_vs_old_official_crps')}, MASE {panel.get('student_gain_vs_old_official_mase')}"
            )
    return "\n".join(lines) + "\n"


def collect(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    vector = _collect_vector(args)
    _write_json(summary_dir / "refreshed_vector_teacher_verification_matrix.json", {k: v for k, v in vector.items() if k != "diagnostics"})
    _write_csv(summary_dir / "refreshed_vector_gap_diagnostics.csv", list(vector.get("diagnostics") or []))
    if args.mode == "refresh_vector":
        (summary_dir / "refreshed_vector_teacher_recommendation.md").write_text(
            _markdown_report(
                {
                    "source_commit": args.source_commit,
                    "root": str(root),
                    "selected_locked_run_ids": [VECTOR_BASELINE_RUN_ID],
                    "followup_requested": False,
                    "probe_rows": [],
                    "locked_rows": [],
                }
            ),
            encoding="utf-8",
        )
        return {"mode": args.mode, "vector_completed_count": vector["completed_count"], "locked_test_used_for_selection": False}

    include_locked = args.mode == "final"
    probe = _collect_probe(args, include_locked=include_locked)
    probe_rows = list(probe["rows"])
    followup_requested, followup_gap, followup_run = _followup_decision(probe_rows)
    selected = _select_candidates(probe_rows)
    if args.mode == "final":
        locked_run_ids = _read_selection_file(summary_dir)
        if not locked_run_ids:
            _write_selection_files(summary_dir, selected, followups_requested=followup_requested)
            locked_run_ids = _read_selection_file(summary_dir)
    else:
        _write_selection_files(summary_dir, selected, followups_requested=followup_requested)
        locked_run_ids = _read_selection_file(summary_dir)
    locked_rows = [row for row in probe_rows if str(row.get("run_id")) in locked_run_ids and "locked_unseen" in dict(row.get("panels") or {})]
    locked_extra_diagnostics: list[dict[str, Any]] = []
    if args.mode == "final" and VECTOR_BASELINE_RUN_ID in locked_run_ids:
        vector_locked = _vector_locked_row_from_probe_or_reference(args, VECTOR_BASELINE_RUN_ID)
        locked_rows.append({key: value for key, value in vector_locked.items() if key != "_diagnostics"})
        locked_extra_diagnostics.extend(vector_locked.get("_diagnostics") or [])
    payload = {
        "artifact": "gipo_probe_2_6_matrix",
        "mode": args.mode,
        "source_commit": str(args.source_commit),
        "root": str(root),
        "input_root": str(args.input_root),
        "vector_root": str(args.vector_root),
        "locked_test_used_for_selection": False,
        "selection_protocol": "validation unseen NFE10/16 MASE primary, mean unseen MASE secondary, CRPS guardrail; locked reporting only after selection",
        "followup_threshold_oracle_student_mase_gap": FOLLOWUP_ORACLE_STUDENT_MASE_GAP,
        "followup_requested": followup_requested,
        "followup_oracle_student_mase_gap": followup_gap,
        "followup_trigger_run_id": followup_run,
        "selected_locked_run_ids": locked_run_ids,
        "vector_baseline_run_id": VECTOR_BASELINE_RUN_ID,
        "vector_rows": vector["rows"],
        "probe_rows": probe_rows,
        "locked_rows": locked_rows,
    }
    _write_json(summary_dir / "probe_2_6_matrix.json", payload)
    _write_csv(summary_dir / "teacher_student_gap_diagnostics.csv", list(vector.get("diagnostics") or []) + list(probe.get("diagnostics") or []) + locked_extra_diagnostics)
    (summary_dir / "final_probe_2_6_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return {
        "mode": args.mode,
        "probe_run_count": len(probe_rows),
        "selected_locked_run_ids": locked_run_ids,
        "followup_requested": followup_requested,
        "locked_test_used_for_selection": False,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect GIPO probe 2-6 validation, gap, and locked summaries.")
    parser.add_argument("--mode", choices=("refresh_vector", "validation", "select_locked", "final"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--vector-root", required=True)
    parser.add_argument("--source-commit", default="")
    return parser


def main() -> None:
    payload = collect(build_argparser().parse_args())
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
