from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Mapping, Sequence

from genode.gipo.policy import context_id_from_row, read_metric_rows_csv
from genode.gipo.evaluate_schedule_summary import build_comparison_summary


STUDENT_KEY = "gipo"
GUARDRAIL_FLOOR = -0.0025
ENTROPY_OVERSMOOTH_MARGIN = 0.075


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / float(len(finite)))


def _source_phase(row: Mapping[str, Any]) -> str:
    return str(row.get("source_split_phase") or row.get("split_phase", row.get("split", "")))


def _evaluation_seed(row: Mapping[str, Any]) -> int:
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except (TypeError, ValueError):
            pass
    return int(row.get("seed", row.get("evaluation_seed", 0)))


def _context_key(row: Mapping[str, Any]) -> tuple[str, str, int, str, int, str]:
    return (
        _source_phase(row),
        str(row.get("dataset", row.get("dataset_key", ""))),
        _evaluation_seed(row),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        context_id_from_row(row),
    )


def _filter_to_student_contexts(
    rows: Sequence[Mapping[str, Any]],
    student_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    wanted = {_context_key(row) for row in student_rows}
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            key = _context_key(row)
        except (KeyError, TypeError, ValueError):
            continue
        if key in wanted:
            out.append(dict(row))
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _selected_teacher_diagnostic(training: Mapping[str, Any]) -> dict[str, Any]:
    selection = dict(dict(training.get("teacher_training", {}) or {}).get("teacher_checkpoint_selection", {}) or {})
    selected_step = int(selection.get("selected_step", -1))
    for item in list(selection.get("history", []) or []):
        if int(item.get("step", -2)) == selected_step:
            return dict(item)
    return {}


def _teacher_gate(training: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    selection = dict(dict(training.get("teacher_training", {}) or {}).get("teacher_checkpoint_selection", {}) or {})
    diagnostic = _selected_teacher_diagnostic(training)
    acc = _optional_float(diagnostic.get("mean_pairwise_accuracy"))
    rho = _optional_float(diagnostic.get("mean_spearman_rank_correlation"))
    loss = _optional_float(selection.get("selected_mean_diagnostic_total_loss"))
    failures: list[str] = []
    if bool(training.get("locked_test_used_for_selection", False)):
        failures.append("locked_test_used_for_selection")
    if loss is None:
        failures.append("teacher_loss_not_finite")
    if acc is None or acc < 0.65:
        failures.append("pairwise_accuracy_below_0.65")
    if rho is None or rho < 0.35:
        failures.append("spearman_below_0.35")
    if selection.get("history") and not all(bool(item.get("selection_constraints_passed", True)) for item in selection.get("history", [])):
        failures.append("selection_constraints_failed")
    return not failures, {
        "teacher_selected_step": selection.get("selected_step"),
        "teacher_selected_mean_diagnostic_total_loss": loss,
        "selected_mean_pairwise_accuracy": acc,
        "selected_mean_spearman_rank_correlation": rho,
        "teacher_gate_failures": "|".join(failures),
    }


def _student_diagnostics(training: Mapping[str, Any]) -> dict[str, Any]:
    student = dict(training.get("student_training", {}) or {})
    losses = list(student.get("losses", []) or [])
    first = dict(losses[0]) if losses else {}
    last = dict(losses[-1]) if losses else {}
    target = dict(student.get("student_target_summary", {}) or {})
    density = dict(training.get("density_representation", {}) or {})
    return {
        "density_bin_count": int(density.get("reference_bin_count", 0) or 0),
        "temperature_mode": target.get("teacher_temperature_mode", "fixed"),
        "teacher_temperature": target.get("teacher_temperature"),
        "teacher_target_ess": target.get("teacher_target_ess"),
        "teacher_candidate_ess_p50": target.get("teacher_candidate_ess_p50"),
        "teacher_candidate_max_weight_p95": target.get("teacher_candidate_max_weight_p95"),
        "student_kl_ce_first": first.get("student_kl_ce_loss"),
        "student_kl_ce_last": last.get("student_kl_ce_loss"),
        "student_entropy_first": first.get("student_entropy"),
        "student_entropy_last": last.get("student_entropy"),
    }


def _comparison_metrics(comparison: Mapping[str, Any]) -> dict[str, Any]:
    crps_best: list[float | None] = []
    mase_best: list[float | None] = []
    crps_ser: list[float | None] = []
    mase_ser: list[float | None] = []
    for cell in list(comparison.get("cell_rankings", []) or []):
        comparisons = [
            dict(item)
            for item in list(cell.get("student_comparisons", []) or [])
            if str(item.get("scheduler_key")) == STUDENT_KEY
        ]
        if not comparisons:
            continue
        item = comparisons[0]
        crps_best.append(_optional_float(item.get("student_relative_crps_gain_vs_best_baseline")))
        mase_best.append(_optional_float(item.get("student_relative_mase_gain_vs_best_baseline")))
        crps_ser.append(_optional_float(item.get("student_relative_crps_gain_vs_ser_ptg")))
        mase_ser.append(_optional_float(item.get("student_relative_mase_gain_vs_ser_ptg")))
    crps_best_f = [value for value in crps_best if value is not None]
    mase_best_f = [value for value in mase_best if value is not None]
    return {
        "cell_count": int(len(crps_best_f)),
        "crps_best_fixed_gain_mean": _mean(crps_best),
        "mase_best_fixed_gain_mean": _mean(mase_best),
        "crps_ser_gain_mean": _mean(crps_ser),
        "mase_ser_gain_mean": _mean(mase_ser),
        "crps_win_rate_vs_best_fixed": None if not crps_best_f else float(sum(value > 0.0 for value in crps_best_f) / float(len(crps_best_f))),
        "mase_win_rate_vs_best_fixed": None if not mase_best_f else float(sum(value > 0.0 for value in mase_best_f) / float(len(mase_best_f))),
    }


def _composite(row: Mapping[str, Any]) -> float | None:
    crps_best = _optional_float(row.get("crps_best_fixed_gain_mean"))
    mase_best = _optional_float(row.get("mase_best_fixed_gain_mean"))
    crps_ser = _optional_float(row.get("crps_ser_gain_mean"))
    mase_ser = _optional_float(row.get("mase_ser_gain_mean"))
    crps_win = _optional_float(row.get("crps_win_rate_vs_best_fixed"))
    mase_win = _optional_float(row.get("mase_win_rate_vs_best_fixed"))
    if None in (crps_best, mase_best, crps_ser, mase_ser, crps_win, mase_win):
        return None
    return float(
        0.35 * crps_best
        + 0.20 * mase_best
        + 0.15 * crps_ser
        + 0.10 * mase_ser
        + 0.10 * (crps_win - 0.5)
        + 0.10 * (mase_win - 0.5)
    )


def _fmt_float(value: Any, spec: str) -> str:
    parsed = _optional_float(value)
    return "" if parsed is None else format(float(parsed), spec)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    inputs_manifest = _read_json(root / "selection_inputs" / "student_selection_inputs_manifest.json")
    rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    for run in list(inputs_manifest["runs"]):
        run_id = str(run["run_id"])
        report_dir = root / "calibration_student_reports" / run_id
        context_gipo = read_metric_rows_csv(report_dir / "context_disjoint" / "context_disjoint_gipo_rows.csv")
        series_gipo = read_metric_rows_csv(report_dir / "series_disjoint" / "series_disjoint_gipo_rows.csv")
        student_rows = [*context_gipo, *series_gipo]
        fixed_rows = _filter_to_student_contexts(read_metric_rows_csv(run["fixed_reference_rows"]), student_rows)
        ser_rows = _filter_to_student_contexts(read_metric_rows_csv(run["ser_reference_rows"]), student_rows)
        seeds = sorted({int(row["seed"]) for row in student_rows})
        solvers = sorted({str(row["solver_key"]) for row in student_rows})
        nfes = sorted({int(row["target_nfe"]) for row in student_rows})
        comparison = build_comparison_summary(
            baseline_rows=fixed_rows,
            comparator_rows=ser_rows,
            student_rows=student_rows,
            dataset=str(args.dataset),
            split_phase="student_calibration",
            seeds=seeds,
            solver_names=solvers,
            target_nfe_values=nfes,
        )
        comparisons[run_id] = comparison
        training = _read_json(Path(run["training_summary"]))
        gate_passed, gate = _teacher_gate(training)
        diag = _student_diagnostics(training)
        metric_row = {
            "run_id": run_id,
            "source": run["source"],
            "teacher_gate_passed": bool(gate_passed),
            "locked_test_used_for_selection": False,
            "calibration_context_row_count": int(len(student_rows)),
            **gate,
            **diag,
            **_comparison_metrics(comparison),
        }
        rows.append(metric_row)

    entropy_by_bin: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        entropy = _optional_float(row.get("student_entropy_last"))
        if entropy is not None:
            entropy_by_bin[int(row.get("density_bin_count") or 0)].append(entropy)
    entropy_thresholds = {
        int(bin_count): float(median(values) + ENTROPY_OVERSMOOTH_MARGIN)
        for bin_count, values in entropy_by_bin.items()
        if values
    }
    for row in rows:
        crps_best = _optional_float(row.get("crps_best_fixed_gain_mean"))
        mase_best = _optional_float(row.get("mase_best_fixed_gain_mean"))
        entropy = _optional_float(row.get("student_entropy_last"))
        entropy_threshold = entropy_thresholds.get(int(row.get("density_bin_count") or 0))
        row["entropy_oversmooth_threshold"] = entropy_threshold
        row["entropy_oversmooth_rejected"] = bool(entropy is not None and entropy_threshold is not None and entropy > entropy_threshold)
        row["crps_guardrail_passed"] = bool(crps_best is not None and crps_best >= GUARDRAIL_FLOOR)
        row["mase_guardrail_passed"] = bool(mase_best is not None and mase_best >= GUARDRAIL_FLOOR)
        row["composite_score"] = _composite(row)
        row["selection_passed"] = bool(
            row["teacher_gate_passed"]
            and row["crps_guardrail_passed"]
            and row["mase_guardrail_passed"]
            and not row["entropy_oversmooth_rejected"]
            and row["composite_score"] is not None
        )
        row["simplicity_rank"] = 0 if str(row.get("temperature_mode")) == "fixed" else 1

    candidates = [row for row in rows if row["selection_passed"]]
    selected = max(
        candidates,
        key=lambda row: (
            float(row["composite_score"]),
            -float(_optional_float(row.get("student_kl_ce_last")) or 0.0),
            -float(_optional_float(row.get("student_entropy_last")) or 0.0),
            -int(row["simplicity_rank"]),
            str(row["run_id"]),
        ),
        default=None,
    )

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_dir / "student_selection_scores.csv", rows)
    for run_id, comparison in comparisons.items():
        out_dir = root / "calibration_student_reports" / run_id
        (out_dir / "student_calibration_comparison_summary.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    payload = {
        "artifact": "genode_student_selection_matrix",
        "locked_test_used_for_selection": False,
        "selection_rule": "teacher_gate_then_student_calibration_composite_crps_mase_ser_bestfixed_winrates",
        "guardrail_floor": GUARDRAIL_FLOOR,
        "entropy_oversmooth_margin": ENTROPY_OVERSMOOTH_MARGIN,
        "composite_weights": {
            "crps_best_fixed_gain_mean": 0.35,
            "mase_best_fixed_gain_mean": 0.20,
            "crps_ser_gain_mean": 0.15,
            "mase_ser_gain_mean": 0.10,
            "crps_win_rate_vs_best_fixed_centered": 0.10,
            "mase_win_rate_vs_best_fixed_centered": 0.10,
        },
        "selected_run_id": None if selected is None else selected["run_id"],
        "rows": rows,
    }
    (summary_dir / "student_selection_matrix.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_lines = [
        "# GenODE Student Selection Verification",
        "",
        f"Output root: `{root}`",
        "",
        "Selection uses calibration-only context and series holdouts. Locked-test rows are not used for selection.",
        "",
        "## Selected Policy",
        "",
    ]
    if selected is None:
        report_lines.append("- No run passed all teacher, metric, and entropy guardrails.")
    else:
        report_lines.append(
            "- Selected run: "
            f"`{selected['run_id']}` with composite score {float(selected['composite_score']):.6f}, "
            f"CRPS best-fixed gain {float(selected['crps_best_fixed_gain_mean']):.4%}, "
            f"MASE best-fixed gain {float(selected['mase_best_fixed_gain_mean']):.4%}."
        )
    report_lines.extend(["", "## Scores", ""])
    report_lines.append("| Run | Pass | Score | CRPS vs best fixed | MASE vs best fixed | CRPS win | MASE win | Entropy reject |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda item: (not bool(item["selection_passed"]), -(float(item["composite_score"]) if item["composite_score"] is not None else -999.0), str(item["run_id"]))):
        report_lines.append(
            f"| `{row['run_id']}` | {bool(row['selection_passed'])} | "
            f"{_fmt_float(row.get('composite_score'), '.6f')} | "
            f"{_fmt_float(row.get('crps_best_fixed_gain_mean'), '.4%')} | "
            f"{_fmt_float(row.get('mase_best_fixed_gain_mean'), '.4%')} | "
            f"{_fmt_float(row.get('crps_win_rate_vs_best_fixed'), '.2f')} | "
            f"{_fmt_float(row.get('mase_win_rate_vs_best_fixed'), '.2f')} | "
            f"{bool(row['entropy_oversmooth_rejected'])} |"
        )
    (summary_dir / "final_student_selection_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect calibration-only student selection reports.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="solar_energy_10m")
    return parser


def main() -> None:
    payload = collect(build_argparser().parse_args())
    print(json.dumps({"selected_run_id": payload.get("selected_run_id"), "locked_test_used_for_selection": False}, sort_keys=True))


if __name__ == "__main__":
    main()
