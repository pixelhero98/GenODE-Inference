from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

REQUIRED_SELECTION_NFES: tuple[int, ...] = (6, 10, 14, 16)
PRIMARY_SELECTION_NFES: tuple[int, ...] = (10, 16)
CRPS_MEAN_GUARDRAIL_FLOOR = -0.005
CRPS_PER_NFE_GUARDRAIL_FLOOR = -0.01


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _comparison_path(summary: Mapping[str, Any], *, summary_path: str | Path | None = None) -> Path | None:
    value = str(summary.get("comparison_summary_path", "") or "").strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or summary_path is None:
        return path
    return Path(summary_path).parent / path


def nfe_gain_panel(summary: Mapping[str, Any], *, summary_path: str | Path | None = None) -> dict[str, Any]:
    path = _comparison_path(summary, summary_path=summary_path)
    if path is None or not path.exists():
        return {}
    comparison = json.loads(path.read_text(encoding="utf-8"))
    expected_nfes = [int(value) for value in comparison.get("target_nfe_values", REQUIRED_SELECTION_NFES)]
    expected_solver_count = len(list(comparison.get("solver_names", []) or []))
    missing_cells = (
        list(summary.get("missing_expected_cells", []) or [])
        + list(comparison.get("missing_baseline_cells", []) or [])
        + list(comparison.get("missing_ser_ptg_cells", []) or [])
        + list(comparison.get("missing_student_cells", []) or [])
    )
    by_nfe: dict[int, list[dict[str, Any]]] = {}
    for cell in list(comparison.get("cell_rankings", []) or []):
        comparisons = list(cell.get("student_comparisons", []) or [])
        student_comparison = dict(comparisons[0]) if comparisons else {}
        if student_comparison.get("student_relative_mase_gain_vs_best_baseline") is None:
            continue
        merged_cell = dict(cell)
        merged_cell.update(student_comparison)
        by_nfe.setdefault(int(cell["target_nfe"]), []).append(merged_cell)
    per_nfe: dict[str, dict[str, Any]] = {}
    for nfe, cells in sorted(by_nfe.items()):
        crps = [
            float(cell["student_relative_crps_gain_vs_best_baseline"])
            for cell in cells
            if cell.get("student_relative_crps_gain_vs_best_baseline") is not None
        ]
        mase = [
            float(cell["student_relative_mase_gain_vs_best_baseline"])
            for cell in cells
            if cell.get("student_relative_mase_gain_vs_best_baseline") is not None
        ]
        per_nfe[str(nfe)] = {
            "cell_count": int(len(cells)),
            "mean_crps_gain_vs_best_fixed": _mean(crps),
            "mean_mase_gain_vs_best_fixed": _mean(mase),
            "mase_win_count_vs_best_fixed": int(sum(1 for value in mase if value > 0.0)),
        }
    interpolation = [per_nfe[str(nfe)] for nfe in (6, 10) if str(nfe) in per_nfe]
    extrapolation = [per_nfe[str(nfe)] for nfe in (14, 16) if str(nfe) in per_nfe]
    all_cells = list(per_nfe.values())

    def panel_mean(items: list[dict[str, Any]], key: str) -> float | None:
        values = [float(item[key]) for item in items if item.get(key) is not None]
        return _mean(values)

    nfe10 = per_nfe.get("10", {})
    nfe16 = per_nfe.get("16", {})
    primary_values = [
        float(item["mean_mase_gain_vs_best_fixed"])
        for item in (nfe10, nfe16)
        if item.get("mean_mase_gain_vs_best_fixed") is not None
    ]
    complete_nfes = [
        int(nfe)
        for nfe in expected_nfes
        if str(nfe) in per_nfe and (expected_solver_count <= 0 or int(per_nfe[str(nfe)]["cell_count"]) >= expected_solver_count)
    ]
    all_unseen_crps = panel_mean(all_cells, "mean_crps_gain_vs_best_fixed")
    per_nfe_crps_values = [
        float(item["mean_crps_gain_vs_best_fixed"])
        for item in all_cells
        if item.get("mean_crps_gain_vs_best_fixed") is not None
    ]
    crps_guardrail_passed = (
        all_unseen_crps is not None
        and float(all_unseen_crps) >= CRPS_MEAN_GUARDRAIL_FLOOR
        and (not per_nfe_crps_values or min(per_nfe_crps_values) >= CRPS_PER_NFE_GUARDRAIL_FLOOR)
    )
    return {
        "per_nfe": per_nfe,
        "expected_target_nfes": expected_nfes,
        "expected_solver_count": int(expected_solver_count),
        "complete_target_nfes": complete_nfes,
        "missing_comparison_cell_count": int(len(missing_cells)),
        "coverage_complete": bool(not missing_cells and set(complete_nfes) >= set(REQUIRED_SELECTION_NFES)),
        "interpolation_mase_gain_vs_best_fixed_mean": panel_mean(interpolation, "mean_mase_gain_vs_best_fixed"),
        "extrapolation_mase_gain_vs_best_fixed_mean": panel_mean(extrapolation, "mean_mase_gain_vs_best_fixed"),
        "all_unseen_mase_gain_vs_best_fixed_mean": panel_mean(all_cells, "mean_mase_gain_vs_best_fixed"),
        "all_unseen_crps_gain_vs_best_fixed_mean": all_unseen_crps,
        "crps_guardrail_passed": bool(crps_guardrail_passed),
        "crps_mean_guardrail_floor": CRPS_MEAN_GUARDRAIL_FLOOR,
        "crps_per_nfe_guardrail_floor": CRPS_PER_NFE_GUARDRAIL_FLOOR,
        "nfe10_16_mase_gain_vs_best_fixed_min": min(primary_values) if len(primary_values) == len(PRIMARY_SELECTION_NFES) else None,
    }


_nfe_gain_panel = nfe_gain_panel
