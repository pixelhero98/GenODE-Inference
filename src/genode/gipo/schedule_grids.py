from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from genode.checkpoint_validation import validate_strict_integer
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
        validate_strict_integer(
            source["checkpoint_step"],
            label="Schedule summary checkpoint_step",
            minimum=1,
        )
        for source in (payload, schedule, item)
        if source.get("checkpoint_step") not in (None, "")
    }
    if len(values) > 1:
        raise ValueError(f"Conflicting checkpoint_step values in schedule prediction metadata: {sorted(values)}.")
    return next(iter(values)) if values else None


def _expected_checkpoint_ids_by_step(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[int, set[str]]:
    by_step: Dict[int, set[str]] = {}
    for row_index, row in enumerate(rows):
        raw_step = row.get("checkpoint_step")
        raw_id = str(row.get("checkpoint_id", "") or "").strip()
        if raw_step in (None, "") or not raw_id:
            raise ValueError(
                "Schedule-summary-bound rows require checkpoint_step and checkpoint_id; "
                f"row {row_index} is incomplete."
            )
        if isinstance(raw_step, bool):
            raise ValueError(f"Row {row_index} checkpoint_step must be an integer.")
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {row_index} checkpoint_step must be an integer.") from exc
        if str(raw_step).strip() != str(step):
            raise ValueError(f"Row {row_index} checkpoint_step must be an integer.")
        by_step.setdefault(step, set()).add(raw_id)
    return by_step


def _validate_summary_checkpoint_ids(
    payload: Mapping[str, Any],
    *,
    path: Any,
    expected_by_step: Mapping[int, set[str]],
) -> None:
    raw_step = payload.get("checkpoint_step")
    if not isinstance(raw_step, int) or isinstance(raw_step, bool):
        raise ValueError(
            f"Schedule summary {path} requires an integer top-level checkpoint_step."
        )
    raw_ids = payload.get("checkpoint_ids")
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise ValueError(f"Schedule summary {path} requires top-level checkpoint_ids.")
    checkpoint_ids = [str(value).strip() for value in raw_ids]
    if not checkpoint_ids or any(not value for value in checkpoint_ids):
        raise ValueError(f"Schedule summary {path} checkpoint_ids must be non-empty strings.")
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError(f"Schedule summary {path} checkpoint_ids must be unique.")
    expected_ids = expected_by_step.get(raw_step)
    if expected_ids is None:
        raise ValueError(
            f"Schedule summary {path} checkpoint_step={raw_step} has no matching metric rows."
        )
    observed_ids = set(checkpoint_ids)
    if observed_ids != expected_ids:
        raise ValueError(
            f"Schedule summary {path} checkpoint_ids do not match metric rows at "
            f"checkpoint_step={raw_step}: expected={sorted(expected_ids)}, "
            f"observed={sorted(observed_ids)}."
        )


def load_schedule_summary_grids(
    paths: Sequence[str],
    *,
    expected_scenario_key: str | None = None,
    expected_reference_split: str | None = None,
    expected_rows: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[ScheduleGridKey, Tuple[float, ...]]:
    grids: Dict[ScheduleGridKey, Tuple[float, ...]] = {}
    scoped_fallback_candidates: Dict[ScheduleGridKey, set[Tuple[float, ...]]] = {}
    expected_by_step = (
        _expected_checkpoint_ids_by_step(expected_rows)
        if expected_rows is not None and paths
        else None
    )

    def register(
        base_key: ScheduleGridKey,
        grid: Sequence[float],
        *,
        checkpoint_step: int | None,
        source: str,
    ) -> None:
        value = tuple(float(item) for item in grid)
        if checkpoint_step is None:
            _register_grid(grids, base_key, value, source=source)
            return
        _register_grid(
            grids,
            (*base_key, int(checkpoint_step)),
            value,
            source=source,
        )
        scoped_fallback_candidates.setdefault(base_key, set()).add(value)

    for path_text in paths:
        path = resolve_project_path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Schedule summary not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Schedule summary {path} must contain a mapping payload.")
        _reject_retired_keys(payload, source=f"Schedule summary {path}")
        if expected_by_step is not None:
            if (
                payload.get("status") != "ready"
                or payload.get("artifact") != "ser_ptg_schedule_summary"
            ):
                raise ValueError(
                    f"Schedule summary {path} must be a ready "
                    "ser_ptg_schedule_summary artifact."
                )
            _validate_summary_checkpoint_ids(
                payload,
                path=path,
                expected_by_step=expected_by_step,
            )
        scenario_key = str(payload.get("scenario_key", "")).strip()
        if not scenario_key:
            raise ValueError(f"Schedule summary {path} requires scenario_key.")
        if expected_scenario_key is not None and scenario_key != str(
            expected_scenario_key
        ).strip():
            raise ValueError(
                f"Schedule summary {path.name!r} belongs to scenario {scenario_key!r}, "
                f"not {str(expected_scenario_key).strip()!r}."
            )
        if expected_reference_split is not None:
            expected_split = str(expected_reference_split).strip()
            reference_split = str(payload.get("reference_split", "") or "").strip()
            if reference_split != expected_split:
                raise ValueError(
                    f"Schedule summary {path.name!r} must declare "
                    f"reference_split={expected_split!r}; "
                    f"got {reference_split or '<missing>'!r}."
                )
            expected_split_keys = {
                "train_tuning": "train",
                "validation_tuning": "val",
            }
            if expected_split not in expected_split_keys:
                raise ValueError(
                    f"Unsupported expected_reference_split={expected_split!r}."
                )
            reference_split_key = str(
                payload.get("reference_split_key", "") or ""
            ).strip()
            expected_split_key = expected_split_keys[expected_split]
            if reference_split_key != expected_split_key:
                raise ValueError(
                    f"Schedule summary {path.name!r} has incompatible "
                    f"reference_split_key={reference_split_key or '<missing>'!r}; "
                    f"expected {expected_split_key!r}."
                )
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
                target_nfe = validate_strict_integer(
                    item.get("target_nfe"),
                    label="Schedule summary target_nfe",
                    minimum=1,
                )
                macro_steps = (
                    validate_strict_integer(
                        item.get("macro_steps"),
                        label="Schedule summary macro_steps",
                        minimum=1,
                    )
                    if item.get("macro_steps") not in (None, "")
                    else None
                )
                realized_nfe = (
                    validate_strict_integer(
                        item.get("realized_nfe"),
                        label="Schedule summary realized_nfe",
                        minimum=1,
                    )
                    if item.get("realized_nfe") not in (None, "")
                    else None
                )
                nfe = normalize_solver_nfe_fields(
                    solver,
                    target_nfe,
                    macro_steps=macro_steps,
                    realized_nfe=realized_nfe,
                    source=f"schedule summary {path}",
                )
                base_grid = validate_time_grid(item["time_grid"], macro_steps=nfe.macro_steps)
                checkpoint_step = _checkpoint_step(payload, schedule, item)
                base_key = (schedule_key, solver, target_nfe)
                register(
                    base_key,
                    base_grid,
                    checkpoint_step=checkpoint_step,
                    source=str(path),
                )
                if schedule_key == "ser_ptg_local_defect_eta005":
                    reversed_grid = validate_time_grid(
                        [1.0 - float(value) for value in reversed(base_grid)],
                        macro_steps=nfe.macro_steps,
                    )
                    reversed_key = ("ser_ptg_local_defect_eta005_reversed", solver, target_nfe)
                    register(
                        reversed_key,
                        reversed_grid,
                        checkpoint_step=checkpoint_step,
                        source=str(path),
                    )
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
                    register(
                        avg_key,
                        avg_grid,
                        checkpoint_step=checkpoint_step,
                        source=str(path),
                    )
    if expected_by_step is None:
        for base_key, candidates in scoped_fallback_candidates.items():
            if base_key in grids:
                if any(candidate != grids[base_key] for candidate in candidates):
                    raise ValueError(
                        f"Unscoped schedule grid for {base_key!r} conflicts with checkpoint-scoped grids."
                    )
            elif len(candidates) == 1:
                grids[base_key] = next(iter(candidates))
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
