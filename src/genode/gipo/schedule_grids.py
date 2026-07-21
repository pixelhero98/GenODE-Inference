from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from genode.data.otflow_paths import resolve_project_path
from genode.gipo.density_representation import DENSITY_BIN_COUNT, average_density_masses, density_mass_to_time_grid, grid_to_density_mass, uniform_reference_grid
from genode.gipo.models import validate_time_grid
from genode.gipo.policy import density_mass_for_row
from genode.gipo.schema import reject_retired_evaluation_keys
from genode.solver_protocol import normalize_solver_key, normalize_solver_nfe_fields

ScheduleGridKey = Tuple[Any, ...]


def _reject_retired_keys(payload: Mapping[str, Any], *, source: str) -> None:
    reject_retired_evaluation_keys(payload, source=source)


def _register_grid(
    grids: Dict[ScheduleGridKey, Tuple[float, ...]],
    key: ScheduleGridKey,
    grid: Sequence[float],
    *,
    source: str,
) -> None:
    value = tuple(float(item) for item in grid)
    if key in grids and grids[key] != value:
        raise ValueError(f"Conflicting schedule grids for {key!r} while loading {source}.")
    grids[key] = value


def _checkpoint_step(payload: Mapping[str, Any], schedule: Mapping[str, Any], item: Mapping[str, Any]) -> int | None:
    values = {
        int(source["checkpoint_step"])
        for source in (payload, schedule, item)
        if source.get("checkpoint_step") not in (None, "")
    }
    if len(values) > 1:
        raise ValueError(f"Conflicting checkpoint_step values in schedule prediction metadata: {sorted(values)}.")
    return next(iter(values)) if values else None


def load_schedule_summary_grids(paths: Sequence[str]) -> Dict[ScheduleGridKey, Tuple[float, ...]]:
    grids: Dict[ScheduleGridKey, Tuple[float, ...]] = {}
    for path_text in paths:
        path = resolve_project_path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Schedule summary not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Schedule summary {path} must contain a mapping payload.")
        _reject_retired_keys(payload, source=f"Schedule summary {path}")
        if not str(payload.get("scenario_key", "")).strip():
            raise ValueError(f"Schedule summary {path} requires scenario_key.")
        schedules = payload.get("schedules")
        if schedules:
            schedule_items = list(schedules)
        else:
            schedule_items = [
                {
                    "scheduler_key": str(payload.get("scheduler_key", "")),
                    "predictions": payload.get("predictions", []) or [],
                }
            ]
        for schedule_index, schedule in enumerate(schedule_items):
            if not isinstance(schedule, Mapping):
                raise ValueError(f"Schedule summary {path} schedule {schedule_index} must be a mapping.")
            _reject_retired_keys(schedule, source=f"Schedule summary {path} schedule {schedule_index}")
            schedule_key = str(schedule.get("scheduler_key", "")).strip()
            if not schedule_key:
                raise ValueError(f"Schedule summary {path} schedule {schedule_index} requires scheduler_key.")
            for item_index, item in enumerate(list(schedule.get("predictions", []) or [])):
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"Schedule summary {path} schedule {schedule_index} prediction {item_index} must be a mapping."
                    )
                _reject_retired_keys(
                    item,
                    source=f"Schedule summary {path} schedule {schedule_index} prediction {item_index}",
                )
                solver = normalize_solver_key(str(item["solver_key"]))
                target_nfe = int(item["target_nfe"])
                nfe = normalize_solver_nfe_fields(
                    solver,
                    target_nfe,
                    macro_steps=item.get("macro_steps"),
                    realized_nfe=item.get("realized_nfe"),
                    source=f"schedule summary {path}",
                )
                base_grid = validate_time_grid(item["time_grid"], macro_steps=nfe.macro_steps)
                checkpoint_step = _checkpoint_step(payload, schedule, item)
                base_key = (schedule_key, solver, target_nfe)
                _register_grid(grids, base_key, base_grid, source=str(path))
                if checkpoint_step is not None:
                    _register_grid(grids, (*base_key, int(checkpoint_step)), base_grid, source=str(path))
                if schedule_key == "ser_ptg_local_defect_eta005":
                    reversed_grid = validate_time_grid(
                        [1.0 - float(value) for value in reversed(base_grid)],
                        macro_steps=nfe.macro_steps,
                    )
                    reversed_key = ("ser_ptg_local_defect_eta005_reversed", solver, target_nfe)
                    _register_grid(grids, reversed_key, reversed_grid, source=str(path))
                    if checkpoint_step is not None:
                        _register_grid(grids, (*reversed_key, int(checkpoint_step)), reversed_grid, source=str(path))
                    reference = uniform_reference_grid(DENSITY_BIN_COUNT)
                    base_mass = grid_to_density_mass(base_grid, reference_time_grid=reference, macro_steps=nfe.macro_steps)
                    reversed_mass = grid_to_density_mass(reversed_grid, reference_time_grid=reference, macro_steps=nfe.macro_steps)
                    averaged_mass = average_density_masses(base_mass, reversed_mass)
                    avg_key = ("ser_ptg_local_defect_eta005_avg_reversed", solver, target_nfe)
                    avg_grid = density_mass_to_time_grid(
                        averaged_mass,
                        reference_time_grid=reference,
                        macro_steps=nfe.macro_steps,
                    )
                    _register_grid(grids, avg_key, avg_grid, source=str(path))
                    if checkpoint_step is not None:
                        _register_grid(grids, (*avg_key, int(checkpoint_step)), avg_grid, source=str(path))
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
    "load_schedule_summary_grids",
    "schedule_grid_coverage_report",
    "validate_schedule_grid_coverage",
]
