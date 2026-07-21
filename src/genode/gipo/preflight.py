from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from genode.experiment_layout import (
    REFERENCE_SUPERVISION_SCHEDULE_KEYS,
    SCENARIO_FAMILY_CONDITIONAL_GENERATION,
    SCENARIO_FAMILY_FORECAST,
    SCENARIO_FAMILY_MOLECULE,
    scenario_family_for_key,
)
from genode.data.otflow_paths import display_project_path, resolve_project_path
from genode.gipo.objectives import (
    CONDITIONAL_METRIC_SPECS,
    FORECAST_METRIC_SPECS,
    MOLECULE_METRIC_SPECS,
    teacher_objective_utility_keys_for_family,
    teacher_objective_utility_keys_for_scenario,
)
from genode.gipo.checkpoint_scope import checkpoint_scope_from_row as _checkpoint_scope_from_row
from genode.gipo.density_representation import uniform_reference_grid
from genode.gipo.policy import teacher_rank_pair_diagnostics
from genode.gipo.schedule_grids import load_schedule_summary_grids, schedule_grid_coverage_report
from genode.gipo.schedule_hash import json_hash as _json_hash
from genode.solver_protocol import normalize_solver_key


@dataclass(frozen=True)
class _RowRecord:
    input_index: int
    source_path: Path
    source_row_number: int
    row: Dict[str, Any]


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _has_nonempty_value(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, None)
    if value is None:
        return False
    return not (isinstance(value, str) and value.strip() == "")


def _optional_int(value: Any) -> int | None:
    if value is None or str(value) == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stable_context_id(
    *,
    scenario_key: str,
    split_phase: str,
    example_idx: int,
    series_id: str,
    series_idx: int,
    target_t: int,
    history_start: int | None = None,
    history_stop: int | None = None,
    context_schema: str = "forecast_window",
) -> str:
    return _json_hash(
        {
            "context_schema": str(context_schema),
            "scenario_key": str(scenario_key),
            "split_phase": str(split_phase),
            "example_idx": int(example_idx),
            "series_id": str(series_id),
            "series_idx": int(series_idx),
            "target_t": int(target_t),
            "history_start": None if history_start is None else int(history_start),
            "history_stop": None if history_stop is None else int(history_stop),
        },
        prefix="ctx",
    )


def _context_id_from_row(row: Mapping[str, Any]) -> str:
    existing = str(row.get("context_id", "") or "").strip()
    if existing:
        return existing
    scenario_key = str(row.get("scenario_key", "")).strip()
    split_phase = str(row.get("split_phase", row.get("split", ""))).strip()
    example_idx_raw = row.get("example_idx", row.get("example_index", None))
    target_t_raw = row.get("target_t", None)
    has_series_identity = str(row.get("series_id", "")).strip() != "" or str(row.get("series_idx", "")).strip() != ""
    context_schema = str(row.get("context_schema", "") or "").strip()
    if context_schema and scenario_key:
        payload = {
            "context_schema": context_schema,
            "scenario_key": scenario_key,
            "split_phase": split_phase,
            "axis_series": str(row.get("axis_series", row.get("series_id", row.get("series_idx", "")))),
            "axis_time_bin": str(row.get("axis_time_bin", "")),
            "axis_record": str(row.get("axis_record", row.get("record_id", ""))),
            "axis_window": str(row.get("axis_window", "")),
            "axis_stratum": str(row.get("axis_stratum", row.get("stratum", ""))),
            "axis_member": str(row.get("axis_member", row.get("member_key", ""))),
            "axis_formula": str(row.get("axis_formula", row.get("formula", ""))),
            "axis_atom_count": str(row.get("axis_atom_count", row.get("atom_count", ""))),
            "axis_trajectory": str(row.get("axis_trajectory", row.get("trajectory_key", row.get("trajectory_id", "")))),
            "axis_iso_id": str(row.get("axis_iso_id", row.get("iso_id", ""))),
            "axis_flags": str(row.get("axis_flags", "")),
            "example_idx": str(row.get("example_idx", row.get("example_index", ""))),
            "target_t": str(row.get("target_t", "")),
            "history_start": str(row.get("history_start", "")),
            "history_stop": str(row.get("history_stop", "")),
        }
        return _json_hash(payload, prefix="ctx")
    missing = []
    if not scenario_key:
        missing.append("scenario_key")
    if not split_phase:
        missing.append("split_phase")
    if example_idx_raw is None or str(example_idx_raw) == "":
        missing.append("example_idx")
    if not has_series_identity:
        missing.append("series_id_or_series_idx")
    if target_t_raw is None or str(target_t_raw) == "":
        missing.append("target_t")
    if missing:
        raise ValueError(f"Context rows require context_id or complete identity fields; missing {missing}.")
    return _stable_context_id(
        scenario_key=scenario_key,
        split_phase=split_phase,
        example_idx=int(example_idx_raw),
        series_id=str(row.get("series_id", "")),
        series_idx=int(row.get("series_idx", 0) or 0),
        target_t=int(target_t_raw),
        history_start=_optional_int(row.get("history_start")),
        history_stop=_optional_int(row.get("history_stop")),
        context_schema=context_schema or "forecast_window",
    )


def _context_embedding_id_from_row(row: Mapping[str, Any]) -> str:
    existing = str(row.get("context_embedding_id", "") or "").strip()
    if existing:
        return existing
    return _context_id_from_row(row)


def _logical_seed_from_row(row: Mapping[str, Any]) -> int | None:
    explicit = row.get("logical_seed", "")
    if explicit is not None and str(explicit).strip() != "":
        return int(explicit)
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except (TypeError, ValueError):
            pass
    return _optional_int(row.get("seed"))


def _context_pair_key(row: Mapping[str, Any], *, pair_on_seed: bool = True) -> Tuple[str, str, int, str, int | None, str]:
    seed = _logical_seed_from_row(row) if pair_on_seed else None
    return (
        str(row.get("scenario_key", "")),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        _context_id_from_row(row),
        seed,
        _checkpoint_scope_from_row(row),
    )


def _validate_gipo_support_schedule_keys(
    support_schedule_keys: Sequence[str],
    *,
    allowed_schedule_keys: Sequence[str] = REFERENCE_SUPERVISION_SCHEDULE_KEYS,
) -> Tuple[str, ...]:
    keys = tuple(str(key) for key in support_schedule_keys)
    if not keys:
        raise ValueError("support_schedule_keys must not be empty.")
    bo_like = sorted(key for key in keys if "bo" in key.lower() or "candidate" in key.lower())
    if bo_like:
        raise ValueError(f"GIPO supervision must not include BO/candidate schedules: {bo_like}")
    allowed = {str(key) for key in allowed_schedule_keys}
    unsupported = sorted(set(keys) - allowed)
    if unsupported:
        raise ValueError(f"GIPO supervision is fixed/SER only; unsupported schedules: {unsupported}")
    return keys


def _validate_teacher_metric_target_keys(keys: Sequence[str] | str | None) -> Tuple[str, ...]:
    if keys is None:
        return ("u_comp_uniform",)
    if isinstance(keys, str):
        raw = [part.strip() for part in keys.split(",")]
    else:
        raw = [str(part).strip() for part in keys]
    out = tuple(part for part in raw if part)
    if not out:
        raise ValueError("teacher_metric_target_keys must contain at least one utility column.")
    duplicates = sorted({key for key in out if out.count(key) > 1})
    if duplicates:
        raise ValueError(f"teacher_metric_target_keys contains duplicates: {duplicates}")
    return out


def _teacher_target_spec_by_utility() -> Dict[str, Any]:
    return {
        spec.utility_key: spec
        for specs in (FORECAST_METRIC_SPECS, CONDITIONAL_METRIC_SPECS, MOLECULE_METRIC_SPECS)
        for spec in specs
    }


def _truthy_row_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _finite_target_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _target_component_applicable(row: Mapping[str, Any], target_key: str) -> bool:
    spec = _teacher_target_spec_by_utility().get(str(target_key))
    if spec is None or not spec.applicable_key:
        return True
    value = row.get(spec.applicable_key)
    if value in (None, ""):
        return False
    return _truthy_row_value(value)


def _infer_single_benchmark_family(rows: Sequence[Mapping[str, Any]]) -> str:
    families = {str(row.get("benchmark_family", "")).strip() for row in rows if str(row.get("benchmark_family", "")).strip()}
    if not families:
        scenario_keys = {
            str(row.get("scenario_key", "")).strip()
            for row in rows
            if str(row.get("scenario_key", "")).strip()
        }
        if not scenario_keys:
            raise ValueError(
                "Automatic GIPO teacher target selection requires scenario_key or an explicit benchmark_family."
            )
        families = {scenario_family_for_key(scenario_key) for scenario_key in scenario_keys}
    if len(families) != 1:
        raise ValueError(f"GIPO training rows must contain exactly one benchmark_family; found {sorted(families)}.")
    family = next(iter(families))
    if family not in {SCENARIO_FAMILY_FORECAST, SCENARIO_FAMILY_CONDITIONAL_GENERATION, SCENARIO_FAMILY_MOLECULE}:
        raise ValueError(f"Unsupported benchmark_family for GIPO teacher target selection: {family!r}.")
    return family


def _infer_single_scenario_key(rows: Sequence[Mapping[str, Any]]) -> str:
    if any("dataset" in row or "dataset_key" in row for row in rows):
        raise ValueError("GIPO rows must use 'scenario_key'; 'dataset' and 'dataset_key' are not supported.")
    scenario_keys = {
        str(row.get("scenario_key", "")).strip()
        for row in rows
        if str(row.get("scenario_key", "")).strip()
    }
    if len(scenario_keys) > 1:
        raise ValueError(f"GIPO training rows must contain exactly one scenario_key; found {sorted(scenario_keys)}.")
    return next(iter(scenario_keys)) if scenario_keys else ""


def _resolve_teacher_metric_target_keys(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    raw = str(args.teacher_metric_target_keys).strip()
    if not raw or raw.lower() == "auto":
        scenario_key = _infer_single_scenario_key(rows)
        if scenario_key:
            try:
                return teacher_objective_utility_keys_for_scenario(scenario_key)
            except (KeyError, ValueError):
                if not any(str(row.get("benchmark_family", "")).strip() for row in rows):
                    raise
        return teacher_objective_utility_keys_for_family(_infer_single_benchmark_family(rows))
    return _validate_teacher_metric_target_keys(raw)


def _read_rows_csvs(paths_text: str) -> Tuple[List[_RowRecord], List[str], List[Dict[str, Any]]]:
    records: List[_RowRecord] = []
    fieldnames: List[str] = []
    inputs: List[Dict[str, Any]] = []
    next_index = 0
    for path_text in _parse_csv(str(paths_text)):
        path = resolve_project_path(path_text)
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            current_fields = [str(field) for field in (reader.fieldnames or [])]
            for field in current_fields:
                if field not in fieldnames:
                    fieldnames.append(field)
            source_count = 0
            for source_count, raw_row in enumerate(reader, start=1):
                row = dict(raw_row)
                if str(row.get("solver_key", "")).strip():
                    row["solver_key"] = normalize_solver_key(str(row["solver_key"]))
                records.append(
                    _RowRecord(
                        input_index=next_index,
                        source_path=path,
                        source_row_number=source_count,
                        row=row,
                    )
                )
                next_index += 1
            inputs.append(
                {
                    "path": display_project_path(path),
                    "row_count": int(source_count),
                    "fieldnames": current_fields,
                }
            )
    return records, fieldnames, inputs


def _observed_support(rows: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    observed = tuple(sorted({str(row["scheduler_key"]) for row in rows if str(row.get("scheduler_key", "")).strip()}))
    protocol_order = tuple(key for key in REFERENCE_SUPERVISION_SCHEDULE_KEYS if key in observed)
    extras = tuple(key for key in observed if key not in protocol_order)
    return _validate_gipo_support_schedule_keys(protocol_order + extras)


def _location(record: _RowRecord) -> Dict[str, Any]:
    return {
        "input_index": int(record.input_index),
        "path": display_project_path(record.source_path),
        "row_number": int(record.source_row_number),
    }


def _cell_payload(key: Tuple[Any, ...]) -> Dict[str, Any]:
    return {
        "scenario_key": str(key[0]),
        "solver_key": str(key[1]),
        "target_nfe": int(key[2]),
        "context_id": str(key[3]),
        "logical_seed": key[4],
        "checkpoint_scope": str(key[5]) if len(key) > 5 else "",
    }


def _row_cell_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return tuple(_context_pair_key(row, pair_on_seed=True))


def _optional_value(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _context_identity_fingerprint(row: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...] | None:
    context_schema = _optional_value(row, "context_schema")
    scenario_key = _optional_value(row, "scenario_key")
    split_phase = _optional_value(row, "split_phase", "split")
    payload: Dict[str, str] = {
        "context_schema": context_schema or "unspecified_context",
        "scenario_key": scenario_key,
        "split_phase": split_phase,
        "axis_series": _optional_value(row, "axis_series", "series_id", "series_idx"),
        "axis_time_bin": _optional_value(row, "axis_time_bin"),
        "axis_record": _optional_value(row, "axis_record", "record_id"),
        "axis_window": _optional_value(row, "axis_window"),
        "axis_stratum": _optional_value(row, "axis_stratum", "stratum"),
        "axis_member": _optional_value(row, "axis_member", "member_key"),
        "axis_formula": _optional_value(row, "axis_formula", "formula"),
        "axis_atom_count": _optional_value(row, "axis_atom_count", "atom_count"),
        "axis_trajectory": _optional_value(row, "axis_trajectory", "trajectory_key", "trajectory_id"),
        "axis_iso_id": _optional_value(row, "axis_iso_id", "iso_id"),
        "axis_flags": _optional_value(row, "axis_flags"),
        "example_idx": _optional_value(row, "example_idx", "example_index"),
        "target_t": _optional_value(row, "target_t"),
        "history_start": _optional_value(row, "history_start"),
        "history_stop": _optional_value(row, "history_stop"),
    }
    specific_keys = [key for key in payload if key not in {"context_schema", "scenario_key", "split_phase"}]
    if not any(payload[key] for key in specific_keys):
        return None
    return tuple(sorted(payload.items()))


def _fingerprint_payload(fingerprint: Tuple[Tuple[str, str], ...]) -> Dict[str, str]:
    return {key: value for key, value in fingerprint if value}


def _identity_conflict_report(
    records: Sequence[_RowRecord],
) -> Tuple[List[Dict[str, Any]], set[int], Dict[int, str], List[Dict[str, Any]]]:
    row_context_ids: Dict[int, str] = {}
    row_fingerprints: Dict[int, Tuple[Tuple[str, str], ...]] = {}
    row_errors: List[Dict[str, Any]] = []
    checkpoint_scope_conflicts: List[Dict[str, Any]] = []
    dirty_rows: set[int] = set()

    for record in records:
        row = record.row
        try:
            context_id = _context_id_from_row(row)
            row_context_ids[record.input_index] = context_id
        except Exception as exc:  # preflight reports row-level problems instead of aborting.
            dirty_rows.add(record.input_index)
            row_errors.append(
                {
                    "type": "row_context_identity_error",
                    "message": str(exc),
                    "location": _location(record),
                }
            )
            continue

        checkpoint_id = str(row.get("checkpoint_id", "") or "").strip()
        if checkpoint_id:
            if str(context_id).startswith(f"{checkpoint_id}:"):
                dirty_rows.add(record.input_index)
                checkpoint_scope_conflicts.append(
                    {
                        "type": "checkpoint_prefixed_context_id",
                        "checkpoint_id": checkpoint_id,
                        "context_id": context_id,
                        "location": _location(record),
                    }
                )
            embedding_id = _context_embedding_id_from_row(row)
            if not str(embedding_id).startswith(f"{checkpoint_id}:"):
                dirty_rows.add(record.input_index)
                checkpoint_scope_conflicts.append(
                    {
                        "type": "checkpoint_context_embedding_scope",
                        "checkpoint_id": checkpoint_id,
                        "context_embedding_id": str(embedding_id),
                        "context_id": context_id,
                        "location": _location(record),
                    }
                )

        fingerprint = _context_identity_fingerprint(row)
        if fingerprint is not None:
            row_fingerprints[record.input_index] = fingerprint

    by_context: Dict[str, Dict[Tuple[Tuple[str, str], ...], List[_RowRecord]]] = defaultdict(lambda: defaultdict(list))
    by_identity: Dict[Tuple[Tuple[str, str], ...], Dict[str, List[_RowRecord]]] = defaultdict(lambda: defaultdict(list))
    records_by_index = {record.input_index: record for record in records}
    for row_index, fingerprint in row_fingerprints.items():
        context_id = row_context_ids.get(row_index)
        if context_id is None:
            continue
        record = records_by_index[row_index]
        by_context[context_id][fingerprint].append(record)
        by_identity[fingerprint][context_id].append(record)

    conflicts: List[Dict[str, Any]] = []
    for context_id, fingerprints in sorted(by_context.items()):
        if len(fingerprints) <= 1:
            continue
        involved = [record for records_for_fp in fingerprints.values() for record in records_for_fp]
        dirty_rows.update(record.input_index for record in involved)
        conflicts.append(
            {
                "type": "context_id_multiple_identities",
                "context_id": context_id,
                "identity_count": int(len(fingerprints)),
                "identities": [
                    {
                        "identity": _fingerprint_payload(fingerprint),
                        "row_locations": [_location(record) for record in records_for_fp],
                    }
                    for fingerprint, records_for_fp in sorted(fingerprints.items(), key=lambda item: repr(item[0]))
                ],
            }
        )

    for fingerprint, context_ids in sorted(by_identity.items(), key=lambda item: repr(item[0])):
        if len(context_ids) <= 1:
            continue
        involved = [record for records_for_context in context_ids.values() for record in records_for_context]
        dirty_rows.update(record.input_index for record in involved)
        conflicts.append(
            {
                "type": "identity_multiple_context_ids",
                "identity": _fingerprint_payload(fingerprint),
                "context_ids": sorted(context_ids),
                "row_locations": [_location(record) for record in involved],
            }
        )

    conflicts.extend(checkpoint_scope_conflicts)
    conflicts.extend(row_errors)
    return conflicts, dirty_rows, row_context_ids, row_errors


def _support_report(
    records: Sequence[_RowRecord],
    support_keys: Sequence[str],
    dirty_rows: set[int],
) -> Tuple[Dict[str, Any], set[Tuple[Any, ...]], List[_RowRecord]]:
    support_keys = tuple(str(key) for key in support_keys)
    support_set = set(support_keys)
    grouped: Dict[Tuple[Any, ...], Dict[str, List[_RowRecord]]] = defaultdict(lambda: {key: [] for key in support_keys})
    extra_support_cells: List[Dict[str, Any]] = []
    row_errors: List[Dict[str, Any]] = []

    for record in records:
        row = record.row
        schedule_key = str(row.get("scheduler_key", "") or "").strip()
        if not schedule_key:
            row_errors.append({"type": "missing_scheduler_key", "location": _location(record)})
            continue
        try:
            cell_key = _row_cell_key(row)
        except Exception as exc:
            row_errors.append(
                {
                    "type": "support_cell_key_error",
                    "scheduler_key": schedule_key,
                    "message": str(exc),
                    "location": _location(record),
                }
            )
            continue
        if schedule_key not in support_set:
            extra_support_cells.append(
                {
                    "scheduler_key": schedule_key,
                    "cell": _cell_payload(cell_key),
                    "location": _location(record),
                }
            )
            continue
        grouped[cell_key][schedule_key].append(record)

    missing_support_cells: List[Dict[str, Any]] = []
    duplicate_support_cells: List[Dict[str, Any]] = []
    complete_cell_keys: set[Tuple[Any, ...]] = set()
    complete_clean_cell_keys: set[Tuple[Any, ...]] = set()

    for cell_key, counts in sorted(grouped.items(), key=lambda item: repr(item[0])):
        missing = [key for key in support_keys if len(counts.get(key, [])) == 0]
        if missing:
            present = {key: len(counts.get(key, [])) for key in support_keys if len(counts.get(key, [])) > 0}
            missing_support_cells.append(
                {
                    "cell": _cell_payload(cell_key),
                    "missing_schedule_keys": missing,
                    "present_support_counts": present,
                }
            )
        for schedule_key in support_keys:
            records_for_schedule = counts.get(schedule_key, [])
            if len(records_for_schedule) > 1:
                duplicate_support_cells.append(
                    {
                        "cell": _cell_payload(cell_key),
                        "scheduler_key": schedule_key,
                        "count": int(len(records_for_schedule)),
                        "row_locations": [_location(record) for record in records_for_schedule],
                    }
                )
        if all(len(counts.get(key, [])) == 1 for key in support_keys):
            complete_cell_keys.add(cell_key)
            support_records = [counts[key][0] for key in support_keys]
            if not any(record.input_index in dirty_rows for record in support_records):
                complete_clean_cell_keys.add(cell_key)

    complete_rows = [
        record
        for record in records
        if str(record.row.get("scheduler_key", "") or "").strip() in support_set
        and record.input_index not in dirty_rows
        and _safe_cell_key(record.row) in complete_clean_cell_keys
    ]
    report = {
        "support_schedule_keys": list(support_keys),
        "support_schedule_count": int(len(support_keys)),
        "uniform_anchor_present": "uniform" in support_set,
        "observed_schedule_keys": sorted(
            {str(record.row.get("scheduler_key", "") or "").strip() for record in records if str(record.row.get("scheduler_key", "") or "").strip()}
        ),
        "support_cell_count": int(len(grouped)),
        "complete_support_cell_count": int(len(complete_cell_keys)),
        "complete_context_identity_clean_support_cell_count": int(len(complete_clean_cell_keys)),
        "incomplete_support_cell_count": int(len(grouped) - len(complete_cell_keys)),
        "missing_support_cell_count": int(len(missing_support_cells)),
        "duplicate_support_cell_count": int(len(duplicate_support_cells)),
        "extra_support_cell_count": int(len(extra_support_cells)),
        "missing_support_cells": missing_support_cells,
        "duplicate_support_cells": duplicate_support_cells,
        "extra_support_cells": extra_support_cells,
        "row_errors": row_errors,
    }
    if "uniform" not in support_set:
        report["support_semantic_errors"] = ["GIPO supervision support must include the uniform reward anchor schedule."]
    else:
        report["support_semantic_errors"] = []
    return report, complete_clean_cell_keys, complete_rows


def _safe_cell_key(row: Mapping[str, Any]) -> Tuple[Any, ...] | None:
    try:
        return _row_cell_key(row)
    except Exception:
        return None


def _coverage_for_rows(rows: Sequence[Mapping[str, Any]], metric_key: str) -> Dict[str, Any]:
    total = int(len(rows))
    present = int(sum(1 for row in rows if _has_nonempty_value(row, metric_key)))
    return {
        "row_count": total,
        "rows_with_value": present,
        "rows_missing_value": int(total - present),
        "coverage_fraction": 0.0 if total == 0 else float(present / total),
    }


def _metric_target_coverage(
    rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
    complete_rows: Sequence[Mapping[str, Any]],
    target_keys: Sequence[str],
) -> Dict[str, Any]:
    metrics = []
    for metric_key in target_keys:
        metrics.append(
            {
                "metric_key": str(metric_key),
                "all_rows": _coverage_for_rows(rows, str(metric_key)),
                "support_rows": _coverage_for_rows(support_rows, str(metric_key)),
                "complete_context_identity_clean_support_rows": _coverage_for_rows(complete_rows, str(metric_key)),
            }
        )
    missing_all = [key for key in target_keys if all(not _has_nonempty_value(row, str(key)) for row in rows)]
    return {
        "teacher_metric_target_keys": list(target_keys),
        "metric_count": int(len(target_keys)),
        "missing_from_all_rows": missing_all,
        "metrics": metrics,
    }


def _write_complete_rows(path: Path, fieldnames: Sequence[str], records: Sequence[_RowRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.row.get(key, "") for key in fieldnames})


def teacher_metric_target_coverage(
    rows: Sequence[Mapping[str, Any]],
    target_keys: Sequence[str] | str | None,
) -> Dict[str, Dict[str, Any]]:
    keys = _validate_teacher_metric_target_keys(target_keys)
    coverage: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        applicable = 0
        valid = 0
        missing = 0
        nonfinite = 0
        inapplicable = 0
        for row in rows:
            if not _target_component_applicable(row, key):
                inapplicable += 1
                continue
            applicable += 1
            raw_value = row.get(key)
            if raw_value in (None, ""):
                missing += 1
                continue
            if _finite_target_value(raw_value) is None:
                nonfinite += 1
                continue
            valid += 1
        coverage[key] = {
            "row_count": int(len(rows)),
            "applicable_count": int(applicable),
            "valid_count": int(valid),
            "missing_count": int(missing),
            "nonfinite_count": int(nonfinite),
            "inapplicable_count": int(inapplicable),
            "coverage_fraction": float(valid / applicable) if applicable else 0.0,
        }
    return coverage


def _teacher_metric_target_validation_report(
    rows: Sequence[Mapping[str, Any]],
    target_keys: Sequence[str],
    *,
    min_coverage_fraction: float,
    min_valid_rows: int,
) -> Dict[str, Any]:
    coverage = teacher_metric_target_coverage(rows, target_keys)
    failures: List[Dict[str, Any]] = []
    for key, item in coverage.items():
        valid_count = int(item["valid_count"])
        fraction = float(item["coverage_fraction"])
        if valid_count < int(min_valid_rows) or fraction + 1e-12 < float(min_coverage_fraction):
            failures.append(
                {
                    "metric_key": str(key),
                    "valid_count": valid_count,
                    "applicable_count": int(item["applicable_count"]),
                    "missing_count": int(item["missing_count"]),
                    "nonfinite_count": int(item["nonfinite_count"]),
                    "coverage_fraction": fraction,
                }
            )
    return {
        "row_count": int(len(rows)),
        "min_coverage_fraction": float(min_coverage_fraction),
        "min_valid_rows": int(min_valid_rows),
        "coverage": coverage,
        "failure_count": int(len(failures)),
        "failures": failures,
    }


def _rank_pair_preflight_report(
    rows: Sequence[Mapping[str, Any]],
    target_keys: Sequence[str],
) -> Dict[str, Any]:
    diagnostics = teacher_rank_pair_diagnostics(
        rows,
        target_keys=target_keys,
        pair_margin=0.0,
        pair_on_seed=True,
    )
    errors: List[str] = []
    if int(diagnostics.get("row_count", 0) or 0) > 0 and int(diagnostics.get("rankable_pair_count", 0) or 0) <= 0:
        errors.append(
            "No rankable teacher pairs were found across complete support cells; "
            "at least one same-context support group must contain different scalarized teacher utilities."
        )
    return {
        **diagnostics,
        "error_count": int(len(errors)),
        "errors": errors,
    }


def validate_gipo_support_preflight_report(report: Mapping[str, Any], *, label: str = "GIPO rows") -> None:
    support_cells = dict(report.get("support_cells", {}) or {})
    identity = dict(report.get("context_identity", {}) or {})
    context_count = dict(report.get("context_count_preflight", {}) or {})
    schedule_grid = dict(report.get("schedule_grid_preflight", {}) or {})
    teacher_metric_targets = dict(report.get("teacher_metric_targets", {}) or {})
    rank_pair_preflight = dict(report.get("rank_pair_preflight", {}) or {})
    bad_group_count = int(support_cells.get("bad_group_count", 0))
    conflict_count = int(identity.get("conflict_group_count", 0))
    context_errors = list(context_count.get("errors", []) or [])
    missing_grid_count = int(schedule_grid.get("missing_grid_row_count", 0) or 0)
    schedule_grid_error = str(report.get("schedule_grid_preflight_error", "") or "")
    metric_failure_count = int(teacher_metric_targets.get("failure_count", 0) or 0)
    rank_pair_error_count = int(rank_pair_preflight.get("error_count", 0) or 0)
    if (
        bad_group_count
        or conflict_count
        or context_errors
        or missing_grid_count
        or schedule_grid_error
        or metric_failure_count
        or rank_pair_error_count
    ):
        raise ValueError(
            f"{label} failed GIPO row preflight: "
            f"bad_support_groups={bad_group_count}, "
            f"context_identity_conflicts={conflict_count}, "
            f"context_count_errors={context_errors}, "
            f"missing_schedule_grid_rows={missing_grid_count}, "
            f"schedule_grid_error={schedule_grid_error!r}, "
            f"teacher_metric_target_failures={metric_failure_count}, "
            f"rank_pair_errors={rank_pair_error_count}, "
            f"support={support_cells}, first_identity={list(identity.get('conflicts', []) or [])[:1]}."
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight GIPO support rows without training.")
    parser.add_argument("--rows_csv", required=True, help="Comma-separated per-example fixed/SER metric rows CSVs.")
    parser.add_argument("--schedule_summary_json", default="", help="Comma-separated schedule summaries used to validate non-fixed schedule grids.")
    parser.add_argument("--min_context_count", type=int, default=3, help="Minimum complete clean contexts required for holdout-based GIPO training.")
    parser.add_argument(
        "--support_schedule_keys",
        default="",
        help="Comma-separated fixed/SER supervision keys. Defaults to observed row keys, matching genode-train-gipo.",
    )
    parser.add_argument(
        "--teacher_metric_target_keys",
        default="auto",
        help="Comma-separated teacher utility columns, or auto for the same family defaults as genode-train-gipo.",
    )
    parser.add_argument(
        "--teacher_metric_min_coverage_fraction",
        type=float,
        default=1.0,
        help="Minimum finite/applicable coverage required for each teacher metric target.",
    )
    parser.add_argument(
        "--teacher_metric_min_valid_rows",
        type=int,
        default=1,
        help="Minimum finite row count required for each teacher metric target.",
    )
    parser.add_argument(
        "--allow_issues",
        action="store_true",
        help="Print the report and exit zero even when issue_count is nonzero.",
    )
    parser.add_argument("--report_json", default="", help="Optional path to also write the JSON report.")
    parser.add_argument(
        "--complete_rows_csv",
        default="",
        help="Optional CSV containing only complete support cells with clean context identity.",
    )
    return parser


def preflight_gipo_rows(args: argparse.Namespace) -> Dict[str, Any]:
    records, fieldnames, inputs = _read_rows_csvs(str(args.rows_csv))
    rows = [record.row for record in records]
    support_keys = (
        _validate_gipo_support_schedule_keys(_parse_csv(str(args.support_schedule_keys)))
        if str(args.support_schedule_keys).strip()
        else _observed_support(rows)
    )
    support_set = set(support_keys)

    identity_conflicts, dirty_rows, _, identity_row_errors = _identity_conflict_report(records)
    support, complete_cell_keys, complete_records = _support_report(records, support_keys, dirty_rows)
    support_rows = [record.row for record in records if str(record.row.get("scheduler_key", "") or "").strip() in support_set]
    complete_clean_rows = [record.row for record in complete_records]
    complete_clean_contexts = sorted({_context_id_from_row(row) for row in complete_clean_rows})
    observed_context_ids: set[str] = set()
    observed_context_row_errors: List[str] = []
    for record in records:
        try:
            observed_context_ids.add(_context_id_from_row(record.row))
        except Exception as exc:
            observed_context_row_errors.append(
                f"{display_project_path(record.source_path)}:{record.source_row_number}: {exc}"
            )
    min_context_count = int(getattr(args, "min_context_count", 3))
    context_count_errors: List[str] = []
    if min_context_count < 1:
        context_count_errors.append("min_context_count must be positive.")
    elif len(complete_clean_contexts) < min_context_count:
        context_count_errors.append(
            f"Only {len(complete_clean_contexts)} complete clean contexts are available; "
            f"at least {min_context_count} are required for holdout-based GIPO training."
        )
    context_count_preflight = {
        "observed_context_count": int(len(observed_context_ids)),
        "complete_clean_context_count": int(len(complete_clean_contexts)),
        "minimum_required_context_count": int(min_context_count),
        "row_error_count": int(len(observed_context_row_errors)),
        "row_errors": observed_context_row_errors,
        "errors": context_count_errors,
    }
    schedule_grid_error = ""
    schedule_grid_report: Dict[str, Any] = {
        "row_count": int(len(rows)),
        "missing_grid_row_count": 0,
        "missing_grid_rows": [],
    }
    try:
        schedule_grids = load_schedule_summary_grids(_parse_csv(str(args.schedule_summary_json)))
        schedule_grid_report = schedule_grid_coverage_report(
            rows,
            schedule_grids=schedule_grids,
            reference_time_grid=uniform_reference_grid(64),
        )
    except Exception as exc:
        schedule_grid_error = str(exc)

    target_resolution_error = ""
    try:
        target_keys = _resolve_teacher_metric_target_keys(
            argparse.Namespace(teacher_metric_target_keys=str(args.teacher_metric_target_keys)),
            rows,
        )
    except Exception as exc:
        target_keys = ()
        target_resolution_error = str(exc)
    metric_min_coverage = float(getattr(args, "teacher_metric_min_coverage_fraction", 1.0))
    if not math.isfinite(metric_min_coverage) or metric_min_coverage < 0.0 or metric_min_coverage > 1.0:
        raise ValueError("teacher_metric_min_coverage_fraction must be finite and in [0, 1].")
    metric_min_valid_rows = int(getattr(args, "teacher_metric_min_valid_rows", 1))
    if metric_min_valid_rows < 0:
        raise ValueError("teacher_metric_min_valid_rows must be nonnegative.")
    metric_target_report = _metric_target_coverage(rows, support_rows, complete_clean_rows, target_keys)
    rank_pair_preflight: Dict[str, Any] = {
        "row_count": int(len(complete_clean_rows)),
        "rankable_pair_count": 0,
        "error_count": 0,
        "errors": [],
    }
    if target_keys:
        metric_validation = _teacher_metric_target_validation_report(
            support_rows,
            target_keys,
            min_coverage_fraction=metric_min_coverage,
            min_valid_rows=metric_min_valid_rows,
        )
        metric_target_report["validation_scope"] = "support_rows"
        metric_target_report["validation"] = metric_validation
        metric_target_report["failure_count"] = int(metric_validation["failure_count"])
        metric_target_report["failures"] = list(metric_validation["failures"])
        if int(metric_validation["failure_count"]) == 0:
            rank_pair_preflight = _rank_pair_preflight_report(complete_clean_rows, target_keys)
        else:
            rank_pair_preflight = {
                "status": "skipped",
                "skip_reason": "metric_target_coverage_failed",
                "row_count": int(len(complete_clean_rows)),
                "rankable_pair_count": 0,
                "error_count": 0,
                "errors": [],
            }
    else:
        metric_target_report["validation_scope"] = ""
        metric_target_report["validation"] = {}
        metric_target_report["failure_count"] = 0
        metric_target_report["failures"] = []

    complete_rows_path = ""
    if str(args.complete_rows_csv).strip():
        complete_path = resolve_project_path(str(args.complete_rows_csv))
        _write_complete_rows(complete_path, fieldnames, complete_records)
        complete_rows_path = display_project_path(complete_path)
    report_json_path = ""
    if str(args.report_json).strip():
        report_json_path = display_project_path(resolve_project_path(str(args.report_json)))

    report: Dict[str, Any] = {
        "artifact": "gipo_preflight_rows_report",
        "schema_version": "genode_gipo_preflight_rows",
        "rows_csv": inputs,
        "row_count": int(len(records)),
        "input_header": list(fieldnames),
        "support": support,
        "teacher_metric_targets": metric_target_report,
        "teacher_metric_target_resolution_error": target_resolution_error,
        "rank_pair_preflight": rank_pair_preflight,
        "context_count_preflight": context_count_preflight,
        "schedule_grid_preflight": schedule_grid_report,
        "schedule_grid_preflight_error": schedule_grid_error,
        "context_identity_conflict_count": int(len(identity_conflicts)),
        "context_identity_conflicts": identity_conflicts,
        "identity_row_error_count": int(len(identity_row_errors)),
        "report_json": report_json_path,
        "complete_rows_csv": complete_rows_path,
        "complete_rows_row_count": int(len(complete_records)),
        "complete_rows_support_cell_count": int(len(complete_cell_keys)),
    }
    issue_count = (
        int(support["missing_support_cell_count"])
        + int(support["duplicate_support_cell_count"])
        + int(support["extra_support_cell_count"])
        + len(support.get("row_errors", []))
        + len(support.get("support_semantic_errors", []))
        + int(len(identity_conflicts))
        + (1 if target_resolution_error else 0)
        + int(metric_target_report.get("failure_count", 0) or 0)
        + int(rank_pair_preflight.get("error_count", 0) or 0)
        + len(context_count_errors)
        + int(schedule_grid_report.get("missing_grid_row_count", 0))
        + (1 if schedule_grid_error else 0)
    )
    report["issue_count"] = int(issue_count)
    report["status"] = "ok" if issue_count == 0 else "issues_found"
    if str(args.report_json).strip():
        report_path = resolve_project_path(str(args.report_json))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    report = preflight_gipo_rows(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if int(report.get("issue_count", 0) or 0) > 0 and not bool(getattr(args, "allow_issues", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
