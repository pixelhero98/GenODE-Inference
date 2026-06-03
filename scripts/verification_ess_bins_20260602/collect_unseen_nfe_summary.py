from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


POLICY_RUN_ID = "temp_fixed_005_b128"
UNSEEN_NFES = (6, 10, 14, 16)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _mean(values: Iterable[Any]) -> float | None:
    vals = [_finite(value) for value in values]
    vals = [value for value in vals if value is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _pct(value: Any) -> str:
    val = _finite(value)
    return "n/a" if val is None else f"{100.0 * val:.2f}%"


def _num(value: Any) -> str:
    val = _finite(value)
    return "n/a" if val is None else f"{val:.6g}"


def _first_student_comparison(cell: Mapping[str, Any]) -> Mapping[str, Any]:
    comparisons = list(cell.get("student_comparisons", []))
    if not comparisons:
        return {}
    return dict(comparisons[0])


def _gain_columns(metric: str) -> List[str]:
    return [
        f"student_relative_{metric}_gain_vs_uniform",
        f"student_relative_{metric}_gain_vs_ser_ptg",
        f"student_relative_{metric}_gain_vs_best_baseline",
    ]


def _cell_rows(comparison: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cell in comparison.get("cell_rankings", []):
        student = _first_student_comparison(cell)
        row = {
            "solver_key": str(cell.get("solver_key")),
            "target_nfe": int(cell.get("target_nfe")),
            "best_baseline_by_crps": cell.get("best_baseline_by_crps"),
            "best_baseline_by_mase": cell.get("best_baseline_by_mase"),
            "ser_ptg_crps_mean": cell.get("ser_ptg_crps_mean"),
            "ser_ptg_mase_mean": cell.get("ser_ptg_mase_mean"),
            "student_crps_mean": student.get("student_crps_mean"),
            "student_mase_mean": student.get("student_mase_mean"),
            "crps_ranking": "|".join(str(value) for value in cell.get("crps_ranking", [])),
            "mase_ranking": "|".join(str(value) for value in cell.get("mase_ranking", [])),
        }
        for key in _gain_columns("crps") + _gain_columns("mase"):
            row[key] = student.get(key)
        rows.append(row)
    return sorted(rows, key=lambda row: (int(row["target_nfe"]), str(row["solver_key"])))


def _aggregate(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"cell_count": len(rows)}
    for metric in ("crps", "mase"):
        out[f"mean_student_{metric}"] = _mean(row.get(f"student_{metric}_mean") for row in rows)
        for baseline in ("uniform", "ser_ptg", "best_baseline"):
            key = f"student_relative_{metric}_gain_vs_{baseline}"
            gains = [_finite(row.get(key)) for row in rows]
            gains = [value for value in gains if value is not None]
            out[f"{metric}_gain_vs_{baseline}_mean"] = _mean(gains)
            out[f"{metric}_wins_vs_{baseline}"] = int(sum(1 for value in gains if value > 0.0))
    return out


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    columns = [
        "target_nfe",
        "solver_key",
        "student_crps_mean",
        "student_mase_mean",
        "student_relative_crps_gain_vs_uniform",
        "student_relative_crps_gain_vs_ser_ptg",
        "student_relative_crps_gain_vs_best_baseline",
        "student_relative_mase_gain_vs_uniform",
        "student_relative_mase_gain_vs_ser_ptg",
        "student_relative_mase_gain_vs_best_baseline",
        "best_baseline_by_crps",
        "best_baseline_by_mase",
        "ser_ptg_crps_mean",
        "ser_ptg_mase_mean",
        "crps_ranking",
        "mase_ranking",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown(root: Path, manifest: Mapping[str, Any], report: Mapping[str, Any], overall: Mapping[str, Any], by_nfe: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "# Unseen-NFE Locked-Test Evaluation",
        "",
        f"Output root: `{root}`",
        "",
        "Policy: `temp_fixed_005_b128` (`gipo_77b4267321065ac07bcc0b0f`), selected before this unseen-NFE evaluation.",
        "Target NFEs: `6,10,14,16`; solvers: `euler,heun,midpoint_rk2,dpmpp2m`; seeds: `0,1,2`.",
        "",
        "## Validation",
        "",
        f"- Locked-test-only inputs: {'yes' if manifest.get('locked_test_only_inputs') is True else 'check manifest'}",
        f"- SER reference split: `{manifest.get('ser_reference_split')}`",
        f"- Combined locked context rows: {_num(manifest.get('combined_context_row_count'))}",
        f"- Locked context count: {_num(manifest.get('locked_context_count'))}",
        "",
        "## Overall",
        "",
        f"- Mean CRPS: {_num(report.get('mean_crps'))}",
        f"- Mean MASE: {_num(report.get('mean_mase'))}",
        f"- CRPS gain vs uniform / SER-PTG / best fixed: {_pct(overall.get('crps_gain_vs_uniform_mean'))} / {_pct(overall.get('crps_gain_vs_ser_ptg_mean'))} / {_pct(overall.get('crps_gain_vs_best_baseline_mean'))}",
        f"- MASE gain vs uniform / SER-PTG / best fixed: {_pct(overall.get('mase_gain_vs_uniform_mean'))} / {_pct(overall.get('mase_gain_vs_ser_ptg_mean'))} / {_pct(overall.get('mase_gain_vs_best_baseline_mean'))}",
        f"- CRPS wins vs uniform / SER-PTG / best fixed: {overall.get('crps_wins_vs_uniform')}/{overall.get('cell_count')}, {overall.get('crps_wins_vs_ser_ptg')}/{overall.get('cell_count')}, {overall.get('crps_wins_vs_best_baseline')}/{overall.get('cell_count')}",
        f"- MASE wins vs uniform / SER-PTG / best fixed: {overall.get('mase_wins_vs_uniform')}/{overall.get('cell_count')}, {overall.get('mase_wins_vs_ser_ptg')}/{overall.get('cell_count')}, {overall.get('mase_wins_vs_best_baseline')}/{overall.get('cell_count')}",
        "",
        "## Per-NFE",
        "",
        "| NFE | CRPS | MASE | CRPS gain vs best fixed | MASE gain vs best fixed |",
        "|---:|---:|---:|---:|---:|",
    ]
    for nfe in UNSEEN_NFES:
        stats = by_nfe.get(str(nfe), {})
        lines.append(
            f"| {nfe} | {_num(stats.get('mean_student_crps'))} | {_num(stats.get('mean_student_mase'))} | "
            f"{_pct(stats.get('crps_gain_vs_best_baseline_mean'))} | {_pct(stats.get('mase_gain_vs_best_baseline_mean'))} |"
        )
    lines.extend(
        [
            "",
            "Detailed cell-level gains are in `summary/unseen_nfe_cell_gains.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    root = Path(
        os.environ.get(
            "GENODE_UNSEEN_ROOT",
            "/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602/unseen_nfe_locked_winner_6_10_14_16",
        )
    )
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(root / "inputs" / "unseen_nfe_manifest.json")
    report = _read_json(root / "report" / POLICY_RUN_ID / "locked_test_gipo_policy_summary.json")
    comparison = _read_json(root / "report" / POLICY_RUN_ID / "locked_test_gipo_comparison_summary.json")
    rows = _cell_rows(comparison)
    nfe_values = {int(row["target_nfe"]) for row in rows}
    if nfe_values != set(UNSEEN_NFES):
        raise ValueError(f"Unexpected NFE values in report: {sorted(nfe_values)}")
    overall = _aggregate(rows)
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_nfe"])].append(row)
    by_nfe = {key: _aggregate(value) for key, value in sorted(grouped.items(), key=lambda item: int(item[0]))}
    payload = {
        "artifact": "unseen_nfe_locked_winner_summary",
        "policy_run_id": POLICY_RUN_ID,
        "locked_test_used_for_selection": False,
        "target_nfe_values": list(UNSEEN_NFES),
        "policy_summary": report,
        "comparison_summary_path": str(root / "report" / POLICY_RUN_ID / "locked_test_gipo_comparison_summary.json"),
        "input_manifest": manifest,
        "overall": overall,
        "by_nfe": by_nfe,
        "cell_rows": rows,
    }
    (summary_dir / "unseen_nfe_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(summary_dir / "unseen_nfe_cell_gains.csv", rows)
    (summary_dir / "final_unseen_nfe_report.md").write_text(_markdown(root, manifest, report, overall, by_nfe), encoding="utf-8")
    print(json.dumps({"summary_dir": str(summary_dir), "cell_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
