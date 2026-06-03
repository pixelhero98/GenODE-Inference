from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


RUN_MATRIX: List[Dict[str, Any]] = [
    {
        "run_id": "diverse_fixed005_b128",
        "temperature_mode": "fixed",
        "teacher_temperature": 0.05,
        "teacher_target_ess": None,
        "density_bin_count": 128,
    },
    {
        "run_id": "diverse_ess25_b128",
        "temperature_mode": "adaptive_ess",
        "teacher_temperature": 0.05,
        "teacher_target_ess": 2.5,
        "density_bin_count": 128,
    },
    {
        "run_id": "diverse_ess40_b128",
        "temperature_mode": "adaptive_ess",
        "teacher_temperature": 0.05,
        "teacher_target_ess": 4.0,
        "density_bin_count": 128,
    },
]


def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _safe_gain(candidate: Any, baseline: Any) -> float | None:
    cand = _finite(candidate)
    base = _finite(baseline)
    if cand is None or base is None or abs(base) < 1e-12:
        return None
    return float((base - cand) / base)


def _mean(values: Iterable[Any]) -> float | None:
    vals = [_finite(value) for value in values]
    vals = [value for value in vals if value is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _comparison_stats(comparison: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not comparison:
        return {}
    rows: List[Mapping[str, Any]] = []
    for cell in comparison.get("cell_rankings", []):
        comparisons = list(cell.get("student_comparisons", []))
        if comparisons:
            rows.append(dict(comparisons[0]))
    out: Dict[str, Any] = {"locked_cell_count": len(rows)}
    for metric in ("crps", "mase"):
        for baseline in ("uniform", "ser_ptg", "best_baseline"):
            key = f"student_relative_{metric}_gain_vs_{baseline}"
            gains = [_finite(row.get(key)) for row in rows]
            gains = [value for value in gains if value is not None]
            out[f"{metric}_gain_vs_{baseline}_mean"] = _mean(gains)
            out[f"{metric}_wins_vs_{baseline}"] = int(sum(1 for value in gains if value > 0.0))
    return out


def _selected_history(summary: Mapping[str, Any]) -> Dict[str, Any]:
    selection = dict(summary.get("teacher_training", {}).get("teacher_checkpoint_selection", {}))
    selected_step = selection.get("selected_step")
    for entry in selection.get("history", []):
        if entry.get("step") == selected_step:
            return dict(entry)
    return {}


def _extract_run(root: Path, spec: Mapping[str, Any], reference_policy: Mapping[str, Any]) -> Dict[str, Any]:
    run_id = str(spec["run_id"])
    policy_dir = root / "policy_runs" / run_id
    report_dir = root / "locked_test_reports" / run_id
    training = _read_json(policy_dir / "gipo_training_summary.json")
    report = _read_json(report_dir / "locked_test_gipo_policy_summary.json")
    comparison = _read_json(report_dir / "locked_test_gipo_comparison_summary.json")
    row: Dict[str, Any] = dict(spec)
    row["policy_dir"] = str(policy_dir)
    row["locked_report_dir"] = str(report_dir)
    row["status"] = "missing_training" if training is None else "trained"
    if training:
        density = dict(training.get("density_representation", {}))
        student = dict(training.get("student_training", {}))
        target = dict(student.get("student_target_summary", {}))
        teacher_selection = dict(training.get("teacher_training", {}).get("teacher_checkpoint_selection", {}))
        selected = _selected_history(training)
        losses = list(student.get("losses", []))
        first_loss = dict(losses[0]) if losses else {}
        last_loss = dict(losses[-1]) if losses else {}
        row.update(
            {
                "policy_id": training.get("policy_id"),
                "support_schedule_keys": "|".join(str(key) for key in training.get("support_schedule_keys", [])),
                "locked_test_used_for_selection": training.get("locked_test_used_for_selection"),
                "reference_grid_hash": density.get("reference_grid_hash"),
                "teacher_selected_step": teacher_selection.get("selected_step"),
                "teacher_selected_mean_diagnostic_total_loss": teacher_selection.get("selected_mean_diagnostic_total_loss"),
                "selected_mean_pairwise_accuracy": selected.get("mean_pairwise_accuracy"),
                "selected_mean_spearman_rank_correlation": selected.get("mean_spearman_rank_correlation"),
                "student_kl_ce_first": first_loss.get("student_kl_ce_loss"),
                "student_kl_ce_last": last_loss.get("student_kl_ce_loss"),
                "student_entropy_first": first_loss.get("student_entropy"),
                "student_entropy_last": last_loss.get("student_entropy"),
            }
        )
        for key in (
            "teacher_temperature_mode",
            "teacher_temperature",
            "teacher_target_ess",
            "teacher_candidate_entropy_p50",
            "teacher_candidate_ess_p05",
            "teacher_candidate_ess_p50",
            "teacher_candidate_ess_p95",
            "teacher_candidate_max_weight_p50",
            "teacher_candidate_max_weight_p95",
            "teacher_chosen_temperature_p50",
            "mean_candidate_count",
            "context_setting_count",
        ):
            row[key] = target.get(key)
    if report:
        row["status"] = "locked_reported"
        row.update(
            {
                "locked_context_row_count": report.get("context_row_count"),
                "locked_aggregate_row_count": report.get("aggregate_row_count"),
                "locked_mean_crps": report.get("mean_crps"),
                "locked_mean_mase": report.get("mean_mase"),
                "crps_gain_vs_reference_policy": _safe_gain(report.get("mean_crps"), reference_policy.get("mean_crps")),
                "mase_gain_vs_reference_policy": _safe_gain(report.get("mean_mase"), reference_policy.get("mean_mase")),
            }
        )
        row.update(_comparison_stats(comparison))
    return row


def _calibration_key(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
    loss = _finite(row.get("teacher_selected_mean_diagnostic_total_loss"))
    acc = _finite(row.get("selected_mean_pairwise_accuracy"))
    max_weight = _finite(row.get("teacher_candidate_max_weight_p95"))
    return (
        float("inf") if loss is None else loss,
        -(0.0 if acc is None else acc),
        float("inf") if max_weight is None else max_weight,
        str(row.get("run_id")),
    )


def _write_csv(path: Path, rows: List[Mapping[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_num(value: Any) -> str:
    val = _finite(value)
    return "pending" if val is None else f"{val:.6g}"


def _fmt_pct(value: Any) -> str:
    val = _finite(value)
    return "pending" if val is None else f"{100.0 * val:.2f}%"


def _markdown(root: Path, rows: List[Mapping[str, Any]], reference_policy: Mapping[str, Any], calibration_best: Mapping[str, Any] | None) -> str:
    completed = [row for row in rows if row.get("status") == "locked_reported"]
    trained = [row for row in rows if row.get("status") in {"trained", "locked_reported"}]
    lines = [
        "# GenODE Density-Diversity Verification",
        "",
        f"Output root: `{root}`",
        "",
        "Locked-test rows are reporting-only. The selected diversity policy is chosen by calibration diagnostics.",
        "",
        "## Reference",
        "",
        f"- Reference policy: `temp_fixed_005_b128`",
        f"- Reference CRPS: {_fmt_num(reference_policy.get('mean_crps'))}",
        f"- Reference MASE: {_fmt_num(reference_policy.get('mean_mase'))}",
        "",
        "## Calibration Ranking",
        "",
    ]
    if calibration_best:
        lines.append(
            f"- Selected calibration run: `{calibration_best.get('run_id')}` "
            f"(holdout loss {_fmt_num(calibration_best.get('teacher_selected_mean_diagnostic_total_loss'))}, "
            f"pairwise accuracy {_fmt_num(calibration_best.get('selected_mean_pairwise_accuracy'))}, "
            f"ESS p50 {_fmt_num(calibration_best.get('teacher_candidate_ess_p50'))}, "
            f"max-weight p95 {_fmt_num(calibration_best.get('teacher_candidate_max_weight_p95'))})."
        )
    else:
        lines.append("- Selected calibration run: pending.")
    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- Trained policies found: {len(trained)} / {len(rows)}",
            f"- Locked-test reports found: {len(completed)} / {len(rows)}",
        ]
    )
    if completed:
        best_crps = min(completed, key=lambda row: float(row["locked_mean_crps"]))
        best_mase = min(completed, key=lambda row: float(row["locked_mean_mase"]))
        lines.extend(
            [
                "",
                "## Locked-Test Outcome",
                "",
                f"- Best CRPS run: `{best_crps.get('run_id')}` with CRPS {_fmt_num(best_crps.get('locked_mean_crps'))}, "
                f"gain vs reference {_fmt_pct(best_crps.get('crps_gain_vs_reference_policy'))}.",
                f"- Best MASE run: `{best_mase.get('run_id')}` with MASE {_fmt_num(best_mase.get('locked_mean_mase'))}, "
                f"gain vs reference {_fmt_pct(best_mase.get('mase_gain_vs_reference_policy'))}.",
                f"- Calibration-selected CRPS: {_fmt_num(calibration_best.get('locked_mean_crps') if calibration_best else None)}.",
                f"- Calibration-selected MASE: {_fmt_num(calibration_best.get('locked_mean_mase') if calibration_best else None)}.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def collect_density_diversity_summary(root: Path, reference_root: Path) -> Dict[str, Any]:
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    reference_policy = _read_json(
        reference_root / "locked_test_reports" / "temp_fixed_005_b128" / "locked_test_gipo_policy_summary.json"
    ) or {"mean_crps": 3.3803307782309573, "mean_mase": 2.520723841365859}
    manifest = _read_json(root / "calibration_inputs" / "calibration_manifest.json") or {}
    rows = [_extract_run(root, spec, reference_policy) for spec in RUN_MATRIX]
    trained = [row for row in rows if row.get("status") in {"trained", "locked_reported"}]
    calibration_best = min(trained, key=_calibration_key) if trained else None
    payload = {
        "artifact": "density_diversity_verification_matrix",
        "root": str(root),
        "reference_root": str(reference_root),
        "reference_policy_run_id": "temp_fixed_005_b128",
        "reference_policy": reference_policy,
        "locked_test_used_for_selection": False,
        "selection_rule": "teacher_holdout_loss_then_pairwise_accuracy_then_target_max_weight_p95",
        "selected_calibration_run_id": None if calibration_best is None else calibration_best.get("run_id"),
        "calibration_manifest": manifest,
        "runs": rows,
    }
    (summary_dir / "verification_matrix.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    columns = [
        "run_id",
        "status",
        "temperature_mode",
        "teacher_temperature",
        "teacher_target_ess",
        "density_bin_count",
        "reference_grid_hash",
        "teacher_selected_step",
        "teacher_selected_mean_diagnostic_total_loss",
        "selected_mean_pairwise_accuracy",
        "selected_mean_spearman_rank_correlation",
        "teacher_candidate_entropy_p50",
        "teacher_candidate_ess_p05",
        "teacher_candidate_ess_p50",
        "teacher_candidate_ess_p95",
        "teacher_candidate_max_weight_p50",
        "teacher_candidate_max_weight_p95",
        "teacher_chosen_temperature_p50",
        "mean_candidate_count",
        "context_setting_count",
        "student_kl_ce_first",
        "student_kl_ce_last",
        "student_entropy_first",
        "student_entropy_last",
        "locked_mean_crps",
        "locked_mean_mase",
        "crps_gain_vs_uniform_mean",
        "crps_gain_vs_ser_ptg_mean",
        "crps_gain_vs_best_baseline_mean",
        "crps_gain_vs_reference_policy",
        "mase_gain_vs_uniform_mean",
        "mase_gain_vs_ser_ptg_mean",
        "mase_gain_vs_best_baseline_mean",
        "mase_gain_vs_reference_policy",
    ]
    _write_csv(summary_dir / "density_diversity_diagnostics.csv", rows, columns)
    (summary_dir / "final_recommendation.md").write_text(
        _markdown(root, rows, reference_policy, calibration_best),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    root = Path(
        os.environ.get(
            "GENODE_VERIFICATION_ROOT",
            "/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603",
        )
    )
    reference_root = Path(
        os.environ.get(
            "GENODE_REFERENCE_ROOT",
            "/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602",
        )
    )
    payload = collect_density_diversity_summary(root, reference_root)
    print(json.dumps({"root": str(root), "selected_calibration_run_id": payload.get("selected_calibration_run_id")}, sort_keys=True))


if __name__ == "__main__":
    main()
