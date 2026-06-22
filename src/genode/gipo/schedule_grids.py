from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from genode.data.otflow_paths import resolve_project_path
from genode.gipo.density_representation import average_density_masses, density_mass_to_time_grid, grid_to_density_mass, uniform_reference_grid
from genode.gipo.models import validate_time_grid
from genode.gipo.policy import density_mass_for_row
from genode.solver_protocol import normalize_solver_key, normalize_solver_nfe_fields

CANONICAL_DENSITY_BIN_COUNT = 64
ScheduleGridKey = Tuple[Any, ...]


def checkpoint_step_from_payload(payload: Mapping[str, Any], schedule: Mapping[str, Any], item: Mapping[str, Any]) -> int | None:
    for source in (item, schedule, payload):
        for key in ("checkpoint_step", "train_steps", "otflow_train_steps"):
            value = source.get(key)
            if value in (None, ""):
                continue
            return int(value)
    return None


def load_schedule_summary_grids(paths: Sequence[str]) -> Dict[ScheduleGridKey, Tuple[float, ...]]:
    grids: Dict[ScheduleGridKey, Tuple[float, ...]] = {}
    for path_text in paths:
        path = resolve_project_path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Schedule summary not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        schedules = payload.get("schedules")
        if schedules:
            schedule_items = list(schedules)
        else:
            schedule_items = [
                {
                    "scheduler_key": str(payload.get("scheduler_key", payload.get("schedule_key", ""))),
                    "predictions": payload.get("predictions", []) or [],
                }
            ]
        for schedule in schedule_items:
            schedule_key = str(schedule.get("scheduler_key", schedule.get("schedule_key", ""))).strip()
            for item in list(schedule.get("predictions", []) or []):
                solver = normalize_solver_key(str(item["solver_key"]))
                target_nfe = int(item["target_nfe"])
                nfe = normalize_solver_nfe_fields(
                    solver,
                    target_nfe,
                    macro_steps=item.get("macro_steps"),
                    runtime_nfe=item.get("runtime_nfe"),
                    realized_nfe=item.get("realized_nfe"),
                    source=f"schedule summary {path}",
                )
                base_grid = validate_time_grid(item["time_grid"], macro_steps=nfe.macro_steps)
                checkpoint_step = checkpoint_step_from_payload(payload, schedule, item)
                base_key = (schedule_key, solver, target_nfe)
                grids[base_key] = base_grid
                if checkpoint_step is not None:
                    grids[(*base_key, int(checkpoint_step))] = base_grid
                if schedule_key == "ser_ptg_local_defect_eta005":
                    reversed_grid = validate_time_grid(
                        [1.0 - float(value) for value in reversed(base_grid)],
                        macro_steps=nfe.macro_steps,
                    )
                    reversed_key = ("ser_ptg_local_defect_eta005_reversed", solver, target_nfe)
                    grids[reversed_key] = reversed_grid
                    if checkpoint_step is not None:
                        grids[(*reversed_key, int(checkpoint_step))] = reversed_grid
                    reference = uniform_reference_grid(CANONICAL_DENSITY_BIN_COUNT)
                    base_mass = grid_to_density_mass(base_grid, reference_time_grid=reference, macro_steps=nfe.macro_steps)
                    reversed_mass = grid_to_density_mass(reversed_grid, reference_time_grid=reference, macro_steps=nfe.macro_steps)
                    averaged_mass = average_density_masses(base_mass, reversed_mass)
                    avg_key = ("ser_ptg_local_defect_eta005_avg_reversed", solver, target_nfe)
                    avg_grid = density_mass_to_time_grid(
                        averaged_mass,
                        reference_time_grid=reference,
                        macro_steps=nfe.macro_steps,
                    )
                    grids[avg_key] = avg_grid
                    if checkpoint_step is not None:
                        grids[(*avg_key, int(checkpoint_step))] = avg_grid
    return grids


def schedule_grid_coverage_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None,
    reference_time_grid: Sequence[float],
) -> Dict[str, Any]:
    missing: List[Dict[str, Any]] = []
    for row in rows:
        try:
            density_mass_for_row(row, schedule_grids=schedule_grids, reference_time_grid=reference_time_grid)
        except (KeyError, ValueError) as exc:
            checkpoint_step = row.get("checkpoint_step", "")
            missing.append(
                {
                    "scheduler_key": str(row.get("scheduler_key", "")),
                    "solver_key": str(row.get("solver_key", "")),
                    "target_nfe": int(row.get("target_nfe", -1)),
                    "checkpoint_step": "" if checkpoint_step in (None, "") else int(checkpoint_step),
                    "context_id": str(row.get("context_id", "")),
                    "error": str(exc),
                }
            )
    return {
        "row_count": int(len(rows)),
        "missing_grid_row_count": int(len(missing)),
        "missing_grid_rows": missing[:50],
    }


def validate_schedule_grid_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    schedule_grids: Mapping[ScheduleGridKey, Sequence[float]] | None,
    reference_time_grid: Sequence[float],
    label: str,
) -> Dict[str, Any]:
    report = schedule_grid_coverage_report(
        rows,
        schedule_grids=schedule_grids,
        reference_time_grid=reference_time_grid,
    )
    if report["missing_grid_row_count"]:
        raise ValueError(
            f"{label} rows are missing schedule grids for non-fixed schedules: "
            f"{report['missing_grid_rows'][:8]}"
        )
    return report


__all__ = [
    "ScheduleGridKey",
    "checkpoint_step_from_payload",
    "load_schedule_summary_grids",
    "schedule_grid_coverage_report",
    "validate_schedule_grid_coverage",
]
