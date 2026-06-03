from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


RUN_MATRIX: List[Dict[str, Any]] = [
    {"run_id": "temp_fixed_005_b128", "sweep": "temperature", "temperature_mode": "fixed", "teacher_temperature": 0.05, "teacher_target_ess": None, "density_bin_count": 128},
    {"run_id": "temp_fixed_0075_b128", "sweep": "temperature", "temperature_mode": "fixed", "teacher_temperature": 0.075, "teacher_target_ess": None, "density_bin_count": 128},
    {"run_id": "temp_fixed_010_b128", "sweep": "temperature", "temperature_mode": "fixed", "teacher_temperature": 0.10, "teacher_target_ess": None, "density_bin_count": 128},
    {"run_id": "temp_fixed_015_b128", "sweep": "temperature", "temperature_mode": "fixed", "teacher_temperature": 0.15, "teacher_target_ess": None, "density_bin_count": 128},
    {"run_id": "temp_ess20_b128", "sweep": "temperature", "temperature_mode": "adaptive_ess", "teacher_temperature": 0.05, "teacher_target_ess": 2.0, "density_bin_count": 128},
    {"run_id": "temp_ess25_b128", "sweep": "temperature,bin", "temperature_mode": "adaptive_ess", "teacher_temperature": 0.05, "teacher_target_ess": 2.5, "density_bin_count": 128},
    {"run_id": "temp_ess30_b128", "sweep": "temperature", "temperature_mode": "adaptive_ess", "teacher_temperature": 0.05, "teacher_target_ess": 3.0, "density_bin_count": 128},
    {"run_id": "bin_ess25_b64", "sweep": "bin", "temperature_mode": "adaptive_ess", "teacher_temperature": 0.05, "teacher_target_ess": 2.5, "density_bin_count": 64},
    {"run_id": "bin_ess25_b256", "sweep": "bin", "temperature_mode": "adaptive_ess", "teacher_temperature": 0.05, "teacher_target_ess": 2.5, "density_bin_count": 256},
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
    rows = []
    for cell in comparison.get("cell_rankings", []):
        comparisons = list(cell.get("student_comparisons", []))
        if not comparisons:
            continue
        row = dict(comparisons[0])
        rows.append(row)
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


def _extract_run(root: Path, spec: Mapping[str, Any], current_policy: Mapping[str, Any]) -> Dict[str, Any]:
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
            "teacher_min_temperature",
            "teacher_max_temperature",
            "teacher_candidate_entropy_mean",
            "teacher_candidate_entropy_p05",
            "teacher_candidate_entropy_p50",
            "teacher_candidate_entropy_p95",
            "teacher_candidate_ess_mean",
            "teacher_candidate_ess_p05",
            "teacher_candidate_ess_p50",
            "teacher_candidate_ess_p95",
            "teacher_candidate_max_weight_mean",
            "teacher_candidate_max_weight_p05",
            "teacher_candidate_max_weight_p50",
            "teacher_candidate_max_weight_p95",
            "teacher_chosen_temperature_mean",
            "teacher_chosen_temperature_p05",
            "teacher_chosen_temperature_p50",
            "teacher_chosen_temperature_p95",
            "mean_teacher_candidate_entropy",
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
                "crps_gain_vs_current_retained": _safe_gain(report.get("mean_crps"), current_policy.get("mean_crps")),
                "mase_gain_vs_current_retained": _safe_gain(report.get("mean_mase"), current_policy.get("mean_mase")),
            }
        )
        row.update(_comparison_stats(comparison))
    return row


def _write_csv(path: Path, rows: List[Mapping[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_pct(value: Any) -> str:
    val = _finite(value)
    if val is None:
        return "pending"
    return f"{100.0 * val:.2f}%"


def _format_float(value: Any) -> str:
    val = _finite(value)
    if val is None:
        return "pending"
    return f"{val:.6g}"


def _recommendation(root: Path, rows: List[Mapping[str, Any]], current_policy: Mapping[str, Any]) -> str:
    complete = [row for row in rows if row.get("status") == "locked_reported"]
    trained = [row for row in rows if row.get("status") in {"trained", "locked_reported"}]
    lines = [
        "# GenODE Adaptive ESS and Density-Bin Verification",
        "",
        f"Output root: `{root}`",
        "",
        "Locked-test rows are reporting-only. Training summaries are required to keep `locked_test_used_for_selection=false`.",
        "",
    ]
    current_crps = current_policy.get("mean_crps")
    current_mase = current_policy.get("mean_mase")
    lines.extend(
        [
            "## Retained Baseline",
            "",
            f"- Current retained policy CRPS: {_format_float(current_crps)}",
            f"- Current retained policy MASE: {_format_float(current_mase)}",
            "",
        ]
    )
    if not complete:
        lines.extend(
            [
                "## Status",
                "",
                f"- Trained policies found: {len(trained)} / {len(rows)}",
                f"- Locked-test reports found: {len(complete)} / {len(rows)}",
                "- Final answers are pending until the Slurm dependency chain completes.",
                "",
            ]
        )
        return "\n".join(lines)

    best_crps = min(complete, key=lambda row: float(row["locked_mean_crps"]))
    best_mase = min(complete, key=lambda row: float(row["locked_mean_mase"]))
    calibration_ranked = sorted(
        trained,
        key=lambda row: (
            _finite(row.get("teacher_selected_mean_diagnostic_total_loss")) if _finite(row.get("teacher_selected_mean_diagnostic_total_loss")) is not None else float("inf"),
            -(_finite(row.get("selected_mean_pairwise_accuracy")) or 0.0),
            _finite(row.get("teacher_candidate_max_weight_p95")) if _finite(row.get("teacher_candidate_max_weight_p95")) is not None else float("inf"),
        ),
    )
    calibration_best = calibration_ranked[0] if calibration_ranked else None
    lines.extend(
        [
            "## Calibration Ranking",
            "",
            "- Ranking uses teacher context/series holdout loss first, then pairwise accuracy and target-mixture health. It does not use locked-test metrics.",
        ]
    )
    if calibration_best:
        lines.append(
            f"- Best calibration-health run: `{calibration_best['run_id']}` "
            f"(holdout loss {_format_float(calibration_best.get('teacher_selected_mean_diagnostic_total_loss'))}, "
            f"ESS p50 {_format_float(calibration_best.get('teacher_candidate_ess_p50'))}, "
            f"max-weight p95 {_format_float(calibration_best.get('teacher_candidate_max_weight_p95'))})."
        )
    lines.extend(
        [
            "",
            "## Locked-Test Outcome",
            "",
            f"- Best CRPS run: `{best_crps['run_id']}` with CRPS {_format_float(best_crps.get('locked_mean_crps'))} "
            f"and gain vs retained current {_format_pct(best_crps.get('crps_gain_vs_current_retained'))}.",
            f"- Best MASE run: `{best_mase['run_id']}` with MASE {_format_float(best_mase.get('locked_mean_mase'))} "
            f"and gain vs retained current {_format_pct(best_mase.get('mase_gain_vs_current_retained'))}.",
            f"- Best CRPS gains vs uniform/SER/best fixed: {_format_pct(best_crps.get('crps_gain_vs_uniform_mean'))}, "
            f"{_format_pct(best_crps.get('crps_gain_vs_ser_ptg_mean'))}, "
            f"{_format_pct(best_crps.get('crps_gain_vs_best_baseline_mean'))}.",
            f"- Best MASE gains vs uniform/SER/best fixed: {_format_pct(best_mase.get('mase_gain_vs_uniform_mean'))}, "
            f"{_format_pct(best_mase.get('mase_gain_vs_ser_ptg_mean'))}, "
            f"{_format_pct(best_mase.get('mase_gain_vs_best_baseline_mean'))}.",
            "",
            "## Answers",
            "",
        ]
    )
    fixed_005 = next((row for row in complete if row["run_id"] == "temp_fixed_005_b128"), None)
    fixed_winner = min(
        [row for row in complete if str(row.get("temperature_mode")) == "fixed" and int(row.get("density_bin_count", 0)) == 128],
        key=lambda row: float(row["locked_mean_crps"]),
        default=None,
    )
    ess_winner = min(
        [row for row in complete if str(row.get("temperature_mode")) == "adaptive_ess" and int(row.get("density_bin_count", 0)) == 128],
        key=lambda row: float(row["locked_mean_crps"]),
        default=None,
    )
    bin_winner = min(
        [row for row in complete if row["run_id"] in {"bin_ess25_b64", "temp_ess25_b128", "bin_ess25_b256"}],
        key=lambda row: float(row["locked_mean_crps"]),
        default=None,
    )
    if fixed_005 and fixed_winner:
        sharper = "yes" if fixed_winner["run_id"] != fixed_005["run_id"] and float(fixed_winner["locked_mean_crps"]) < float(fixed_005["locked_mean_crps"]) else "no"
        lines.append(f"- Fixed `0.05` too sharp: {sharper}; best fixed-temperature run is `{fixed_winner['run_id']}`.")
    if ess_winner:
        lines.append(f"- Best adaptive ESS target at 128 bins: `{ess_winner['run_id']}`.")
    if bin_winner:
        lines.append(f"- Best bin count under ESS 2.5 by locked-test CRPS: `{bin_winner['density_bin_count']}` bins via `{bin_winner['run_id']}`.")
    beats_current = (best_crps.get("crps_gain_vs_current_retained") or 0.0) > 0.0
    lines.append(f"- New best beats retained `gipo_solar_v1_256` on CRPS: {'yes' if beats_current else 'no'}.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    root = Path(os.environ.get("GENODE_VERIFICATION_ROOT", "/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602"))
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    current_policy = _read_json(
        Path("/scratch/b35z/pixelhero.b35z/genode/outputs/fast_compare/current_density/locked_test_gipo_policy_summary.json")
    ) or {}
    rows = [_extract_run(root, spec, current_policy) for spec in RUN_MATRIX]
    payload = {
        "artifact": "verification_ess_bins_matrix",
        "locked_test_used_for_selection": False,
        "run_count": len(rows),
        "completed_locked_report_count": sum(1 for row in rows if row.get("status") == "locked_reported"),
        "current_retained_policy": current_policy,
        "runs": rows,
    }
    (summary_dir / "verification_matrix.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    common_columns = [
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
        "teacher_candidate_entropy_p50",
        "teacher_candidate_ess_p05",
        "teacher_candidate_ess_p50",
        "teacher_candidate_ess_p95",
        "teacher_candidate_max_weight_p05",
        "teacher_candidate_max_weight_p50",
        "teacher_candidate_max_weight_p95",
        "teacher_chosen_temperature_p50",
        "student_kl_ce_first",
        "student_kl_ce_last",
        "student_entropy_first",
        "student_entropy_last",
        "locked_mean_crps",
        "locked_mean_mase",
        "crps_gain_vs_uniform_mean",
        "crps_gain_vs_ser_ptg_mean",
        "crps_gain_vs_best_baseline_mean",
        "crps_gain_vs_current_retained",
        "mase_gain_vs_uniform_mean",
        "mase_gain_vs_ser_ptg_mean",
        "mase_gain_vs_best_baseline_mean",
        "mase_gain_vs_current_retained",
    ]
    temperature_rows = [row for row in rows if "temperature" in str(row.get("sweep", ""))]
    bin_rows = [row for row in rows if "bin" in str(row.get("sweep", ""))]
    _write_csv(summary_dir / "temperature_diagnostics.csv", temperature_rows, common_columns)
    _write_csv(summary_dir / "bin_diagnostics.csv", bin_rows, common_columns)
    (summary_dir / "final_recommendation.md").write_text(_recommendation(root, rows, current_policy), encoding="utf-8")
    print(json.dumps({"summary_dir": str(summary_dir), "completed_locked_report_count": payload["completed_locked_report_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
