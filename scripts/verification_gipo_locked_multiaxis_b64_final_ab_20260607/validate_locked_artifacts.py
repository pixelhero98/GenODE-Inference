from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FIXED_SUPPORT_KEYS = (
    "uniform",
    "late_power_3",
    "flowts_power_sampling",
    "ays",
    "gits",
    "ots",
    "late_power_3_reversed",
    "flowts_power_sampling_reversed",
    "ays_reversed",
    "gits_reversed",
    "ots_reversed",
)
SER_SUPPORT_KEY = "ser_ptg_local_defect_eta005"
SUPPORT_SCHEDULE_KEYS = FIXED_SUPPORT_KEYS + (SER_SUPPORT_KEY,)
EXPECTED_FIXED_SCHEDULE_COUNT = 11
EXPECTED_SUPPORT_SCHEDULE_COUNT = 12
if len(FIXED_SUPPORT_KEYS) != EXPECTED_FIXED_SCHEDULE_COUNT:
    raise RuntimeError("FIXED_SUPPORT_KEYS must contain exactly 11 physical clock schedules.")
if len(SUPPORT_SCHEDULE_KEYS) != EXPECTED_SUPPORT_SCHEDULE_COUNT:
    raise RuntimeError("SUPPORT_SCHEDULE_KEYS must contain exactly 11 fixed clocks plus SER.")
PANELS = {
    "seen": (4, 8, 12),
    "unseen": (6, 10, 14, 16),
}
TRAIN_PHASE = "train_tuning"
LOCKED_PHASE = "locked_test"


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty row file: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_embeddings(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        context_ids = [str(value) for value in payload["context_ids"].tolist()]
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(context_ids):
        raise ValueError(f"Embedding table has inconsistent shape: {path}")
    return {context_id: embeddings[idx] for idx, context_id in enumerate(context_ids)}


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _phase(row: Mapping[str, Any]) -> str:
    return str(row.get("source_split_phase") or row.get("split_phase") or row.get("split") or "").strip()


def _raw_phase(row: Mapping[str, Any]) -> str:
    return str(row.get("split_phase") or row.get("split") or "").strip()


def _logical_seed(row: Mapping[str, Any]) -> int:
    explicit = str(row.get("logical_seed", "") or "").strip()
    if explicit:
        return int(explicit)
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except ValueError:
            pass
    return int(row.get("seed", row.get("evaluation_seed", 0)))


def _evaluation_seed(row: Mapping[str, Any]) -> int:
    explicit = str(row.get("evaluation_seed", "") or "").strip()
    if explicit:
        return int(explicit)
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except ValueError:
            pass
    return int(row.get("seed", 0))


def _context_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("context_id", "") or "").strip()
    if not value:
        raise ValueError("Artifact row is missing context_id.")
    return value


def _series_id(row: Mapping[str, Any]) -> str:
    return str(row.get("series_id", row.get("series_idx", "")) or "").strip()


def _support_group_key(row: Mapping[str, Any], *, seed_kind: str) -> tuple[Any, ...]:
    seed = _evaluation_seed(row) if seed_kind == "evaluation" else _logical_seed(row)
    return (
        str(row.get("dataset", row.get("dataset_key", ""))),
        _phase(row),
        seed,
        str(row["solver_key"]),
        int(row["target_nfe"]),
        _context_id(row),
        _series_id(row),
        str(row.get("target_t", "")),
        str(row.get("history_start", "")),
        str(row.get("history_stop", "")),
    )


def _realized_nfe(row: Mapping[str, Any]) -> int | str:
    value = str(row.get("realized_nfe", "") or "").strip()
    if value:
        return int(value)
    fallback = str(row.get("runtime_nfe", "") or row.get("target_nfe", "") or "").strip()
    return int(fallback) if fallback else ""


def _invariance_group_key(row: Mapping[str, Any], *, seed_kind: str) -> tuple[Any, ...]:
    seed = _evaluation_seed(row) if seed_kind == "evaluation" else _logical_seed(row)
    return (
        str(row.get("dataset", row.get("dataset_key", ""))),
        _phase(row),
        seed,
        str(row["solver_key"]),
        int(row["target_nfe"]),
        _realized_nfe(row),
        _context_id(row),
    )


def _nfe_sets(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted({int(row["target_nfe"]) for row in rows})


def _split_phases(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({_phase(row) for row in rows})


def _raw_split_phases(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({_raw_phase(row) for row in rows})


def _schedule_keys(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(row["scheduler_key"]) for row in rows})


def _schedule_nfes(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        out[str(row["scheduler_key"])].add(int(row["target_nfe"]))
    return {key: sorted(values) for key, values in sorted(out.items())}


def _nfe_count_diagnostics(rows: Sequence[Mapping[str, Any]], expected_nfes: Sequence[int]) -> dict[str, Any]:
    expected = [int(value) for value in expected_nfes]
    counts = Counter(int(row["target_nfe"]) for row in rows)
    per_nfe = {str(nfe): int(counts.get(nfe, 0)) for nfe in expected}
    extra = sorted(int(value) for value in counts if int(value) not in set(expected))
    missing = [int(nfe) for nfe in expected if counts.get(int(nfe), 0) <= 0]
    unique_expected_counts = sorted({int(counts.get(nfe, 0)) for nfe in expected})
    return {
        "row_count": int(len(rows)),
        "expected_target_nfes": expected,
        "counts_by_target_nfe": per_nfe,
        "missing_target_nfes": missing,
        "extra_target_nfes": extra,
        "balanced_expected_nfe_counts": not missing and not extra and len(unique_expected_counts) == 1,
    }


def _row_count_ok(manifest: Mapping[str, Any]) -> bool:
    return bool(manifest.get("actual")) and int(manifest.get("actual", -1)) == int(manifest.get("expected", -2))


_CLOCK_DENSITY_VARYING_COLUMNS = {
    "parent_row_signature",
    "row_signature",
    "scheduler_key",
    "scheduler_variant_key",
    "scheduler_variant_name",
    "schedule_name",
    "schedule_grid_hash",
    "reference_time_alignment",
    "runtime_grid_q25",
    "runtime_grid_q50",
    "runtime_grid_q75",
    "reference_macro_steps",
    "candidate_source",
    "source_scheduler_key",
    "gipo_step_budget",
    "time_grid_json",
    "intervals_json",
    "density_mass_hash",
    "density_mass_json",
    "density_protocol",
    "density_domain",
    "reference_grid_hash",
    "internal_fraction_after_098",
    "internal_count_after_098",
    "internal_count",
    "min_interval",
    "max_interval",
    "perturbation_type",
    "perturbation_params_json",
    "validity_flags_json",
    "crps",
    "mase",
    "mse",
    "score_main",
    "utility",
    "u_crps_best",
    "u_mase_best",
    "u_comp_best",
    "u_comp_uniform",
    "u_crps_uniform",
    "u_mase_uniform",
    "best_fixed_crps",
    "best_fixed_mase",
    "uniform_crps",
    "uniform_mase",
    "relative_crps_gain_vs_uniform",
    "relative_mase_gain_vs_uniform",
    "relative_score_gain_vs_uniform",
    "evaluation_protocol_hash",
    "latency_ms_per_sample",
    "row_status",
}


def _physical_cell_invariance_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed_kind: str,
    expected_schedules: Sequence[str],
) -> dict[str, Any]:
    expected = tuple(str(key) for key in expected_schedules)
    expected_set = set(expected)
    expected_count = int(len(expected))
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_invariance_group_key(row, seed_kind=seed_kind)].append(row)
    bad_groups: list[dict[str, Any]] = []
    bad_count = 0
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        schedule_counts = Counter(str(row["scheduler_key"]) for row in group)
        schedules = sorted(schedule_counts)
        missing = [schedule for schedule in expected if schedule_counts.get(schedule, 0) != 1]
        unexpected = sorted(schedule for schedule in schedule_counts if schedule not in expected_set)
        duplicate_or_bad_counts = {
            schedule: count
            for schedule, count in sorted(schedule_counts.items())
            if count != 1 or schedule not in expected_set
        }
        varying: dict[str, list[str]] = {}
        for column in sorted({column for row in group for column in row}):
            if column in _CLOCK_DENSITY_VARYING_COLUMNS:
                continue
            values = {str(row.get(column, "")).strip() for row in group if str(row.get(column, "")).strip()}
            if len(values) > 1:
                varying[column] = sorted(values)[:5]
        schedule_issue = (
            int(len(group)) != expected_count
            or bool(missing)
            or bool(unexpected)
            or bool(duplicate_or_bad_counts)
        )
        if varying or schedule_issue:
            bad_count += 1
            if len(bad_groups) < 8:
                bad_groups.append(
                    {
                        "group": [str(part) for part in key],
                        "row_count": int(len(group)),
                        "schedule_count": int(len(schedules)),
                        "schedules": schedules,
                        "expected_schedule_count": expected_count,
                        "expected_schedules": sorted(expected),
                        "missing_or_duplicate": missing,
                        "unexpected_schedules": unexpected,
                        "schedule_counts": dict(schedule_counts),
                        "varying_non_clock_columns": varying,
                    }
                )
    return {
        "expected_schedule_count": expected_count,
        "expected_schedules": sorted(expected),
        "group_count": int(len(grouped)),
        "bad_group_count": int(bad_count),
        "bad_groups_sample": bad_groups,
    }


def _embedding_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table = _read_embeddings(path)
    required = {_context_id(row) for row in rows}
    missing = sorted(required - set(table))
    first = next(iter(table.values())) if table else np.zeros((0,), dtype=np.float32)
    return {
        "path": str(path),
        "context_count": int(len(table)),
        "required_context_count": int(len(required)),
        "embedding_dim": int(first.shape[0]),
        "missing_required_context_ids": missing[:20],
        "missing_required_context_count": int(len(missing)),
    }


def _group_diagnostics(rows: Sequence[Mapping[str, Any]], *, seed_kind: str) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for row in rows:
        key = list(_support_group_key(row, seed_kind=seed_kind))
        key[4] = "*"
        grouped[tuple(key)].add(int(row["target_nfe"]))
    counts = Counter(tuple(sorted(values)) for values in grouped.values())
    return {
        "group_count": int(len(grouped)),
        "multi_nfe_group_count": int(sum(1 for values in grouped.values() if len(values) > 1)),
        "nfe_sets_top": [
            {"nfes": list(values), "count": int(count)}
            for values, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
        ],
    }


def _support_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_schedules: Sequence[str],
    seed_kind: str,
) -> dict[str, Any]:
    expected = tuple(str(key) for key in expected_schedules)
    expected_set = set(expected)
    grouped: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    bad_schedules: set[str] = set()
    for row in rows:
        schedule = str(row["scheduler_key"])
        if schedule not in expected_set:
            bad_schedules.add(schedule)
        grouped[_support_group_key(row, seed_kind=seed_kind)][schedule] += 1
    bad_groups: list[dict[str, Any]] = []
    for key, counts in grouped.items():
        missing = [schedule for schedule in expected if counts.get(schedule, 0) != 1]
        extra = {schedule: count for schedule, count in counts.items() if schedule not in expected_set or count != 1}
        if missing or extra:
            bad_groups.append({"group": [str(part) for part in key], "missing_or_duplicate": missing, "counts": dict(counts)})
            if len(bad_groups) >= 8:
                break
    return {
        "expected_schedule_keys": list(expected),
        "observed_schedule_keys": _schedule_keys(rows),
        "group_count": int(len(grouped)),
        "bad_schedule_keys": sorted(bad_schedules),
        "bad_group_count_sampled": int(len(bad_groups)),
        "bad_groups_sample": bad_groups,
    }


def _standard_paths(root: Path, panel: str) -> dict[str, Path]:
    base = root / "standard_inputs" / panel
    return {
        "fixed_train_rows": base / "fixed_train_tuning" / "fixed_train_context_rows.csv",
        "fixed_train_embeddings": base / "fixed_train_tuning" / "fixed_train_context_embeddings.npz",
        "ser_train_rows": base / "ser_train_tuning" / "ser_train_context_rows.csv",
        "fixed_locked_rows": base / "fixed_locked" / "fixed_locked_context_rows.csv",
        "fixed_locked_embeddings": base / "fixed_locked" / "fixed_locked_context_embeddings.npz",
        "ser_locked_rows": base / "ser_locked" / "ser_locked_context_rows.csv",
        "ser_summary": base / "ser_reference" / "ser_ptg_schedule_summary.json",
    }


def build_panel_artifacts(root: Path, panel: str) -> dict[str, Path]:
    paths = _standard_paths(root, panel)
    artifacts = root / "artifacts"
    fixed_train = _read_rows(paths["fixed_train_rows"])
    ser_train = _read_rows(paths["ser_train_rows"])
    fixed_locked = _read_rows(paths["fixed_locked_rows"])
    ser_locked = _read_rows(paths["ser_locked_rows"])
    out = {
        "train_supervision_rows": artifacts / f"{panel}_train_supervision_rows.csv",
        "train_context_embeddings": artifacts / f"{panel}_train_context_embeddings.npz",
        "locked_report_context_rows": artifacts / f"{panel}_locked_report_context_rows.csv",
        "locked_context_embeddings": artifacts / f"{panel}_locked_context_embeddings.npz",
        "locked_teacher_support_rows": artifacts / f"{panel}_locked_teacher_support_rows.csv",
    }
    _write_rows(out["train_supervision_rows"], [*fixed_train, *ser_train])
    _copy_file(paths["fixed_train_embeddings"], out["train_context_embeddings"])
    _write_rows(out["locked_report_context_rows"], fixed_locked)
    _copy_file(paths["fixed_locked_embeddings"], out["locked_context_embeddings"])
    _write_rows(out["locked_teacher_support_rows"], [*fixed_locked, *ser_locked])
    return out


def validate_panel(root: Path, panel: str, expected_nfes: Sequence[int]) -> dict[str, Any]:
    artifacts = root / "artifacts"
    paths = {
        **_standard_paths(root, panel),
        "train_supervision_rows": artifacts / f"{panel}_train_supervision_rows.csv",
        "train_context_embeddings": artifacts / f"{panel}_train_context_embeddings.npz",
        "locked_report_context_rows": artifacts / f"{panel}_locked_report_context_rows.csv",
        "locked_context_embeddings": artifacts / f"{panel}_locked_context_embeddings.npz",
        "locked_teacher_support_rows": artifacts / f"{panel}_locked_teacher_support_rows.csv",
    }
    fixed_train = _read_rows(paths["fixed_train_rows"])
    ser_train = _read_rows(paths["ser_train_rows"])
    fixed_locked = _read_rows(paths["fixed_locked_rows"])
    ser_locked = _read_rows(paths["ser_locked_rows"])
    train_rows = _read_rows(paths["train_supervision_rows"])
    report_rows = _read_rows(paths["locked_report_context_rows"])
    support_rows = _read_rows(paths["locked_teacher_support_rows"])
    issues: list[str] = []
    expected_nfes_list = sorted(int(value) for value in expected_nfes)
    if _nfe_sets(train_rows) != expected_nfes_list:
        issues.append("train_nfe_coverage")
    if _nfe_sets(report_rows) != expected_nfes_list:
        issues.append("locked_report_nfe_coverage")
    if _nfe_sets(support_rows) != expected_nfes_list:
        issues.append("locked_support_nfe_coverage")
    if _split_phases(train_rows) != [TRAIN_PHASE]:
        issues.append("train_source_split_phase")
    if LOCKED_PHASE in _raw_split_phases(train_rows):
        issues.append("locked_rows_in_training_supervision")
    if LOCKED_PHASE in _split_phases(train_rows):
        issues.append("locked_source_rows_in_training_supervision")
    if _split_phases(report_rows) != [LOCKED_PHASE]:
        issues.append("locked_report_source_split_phase")
    if _split_phases(support_rows) != [LOCKED_PHASE]:
        issues.append("locked_support_source_split_phase")
    train_support = _support_diagnostics(train_rows, expected_schedules=SUPPORT_SCHEDULE_KEYS, seed_kind="logical")
    locked_support = _support_diagnostics(support_rows, expected_schedules=SUPPORT_SCHEDULE_KEYS, seed_kind="evaluation")
    fixed_train_invariance = _physical_cell_invariance_diagnostics(
        fixed_train,
        seed_kind="logical",
        expected_schedules=FIXED_SUPPORT_KEYS,
    )
    fixed_locked_invariance = _physical_cell_invariance_diagnostics(
        report_rows,
        seed_kind="evaluation",
        expected_schedules=FIXED_SUPPORT_KEYS,
    )
    if train_support["bad_schedule_keys"] or train_support["bad_group_count_sampled"]:
        issues.append("train_support_group_coverage")
    if locked_support["bad_schedule_keys"] or locked_support["bad_group_count_sampled"]:
        issues.append("locked_support_group_coverage")
    if fixed_train_invariance["bad_group_count"]:
        issues.append("fixed_train_cell_invariance")
    if fixed_locked_invariance["bad_group_count"]:
        issues.append("fixed_locked_cell_invariance")
    fixed_train_schedules = _schedule_keys(fixed_train)
    ser_train_schedules = _schedule_keys(ser_train)
    fixed_locked_schedules = _schedule_keys(fixed_locked)
    ser_locked_schedules = _schedule_keys(ser_locked)
    if fixed_train_schedules != sorted(FIXED_SUPPORT_KEYS):
        issues.append("fixed_train_schedule_coverage")
    if ser_train_schedules != [SER_SUPPORT_KEY]:
        issues.append("ser_train_schedule_coverage")
    if fixed_locked_schedules != sorted(FIXED_SUPPORT_KEYS):
        issues.append("fixed_locked_schedule_coverage")
    if ser_locked_schedules != [SER_SUPPORT_KEY]:
        issues.append("ser_locked_schedule_coverage")
    if _schedule_keys(train_rows) != sorted(SUPPORT_SCHEDULE_KEYS):
        issues.append("train_supervision_schedule_coverage")
    if _schedule_keys(report_rows) != sorted(FIXED_SUPPORT_KEYS):
        issues.append("locked_report_fixed_schedule_coverage")
    if _schedule_keys(support_rows) != sorted(SUPPORT_SCHEDULE_KEYS):
        issues.append("locked_support_schedule_coverage")
    train_embeddings = _embedding_manifest(paths["train_context_embeddings"], train_rows)
    locked_embeddings = _embedding_manifest(paths["locked_context_embeddings"], support_rows)
    if train_embeddings["missing_required_context_count"]:
        issues.append("train_embedding_context_coverage")
    if locked_embeddings["missing_required_context_count"]:
        issues.append("locked_embedding_context_coverage")
    train_multi_nfe = _group_diagnostics(train_rows, seed_kind="logical")
    # Locked evaluation seeds are intentionally NFE-offset; logical seed preserves
    # the shared context identity needed for multi-NFE coverage diagnostics.
    locked_multi_nfe = _group_diagnostics(support_rows, seed_kind="logical")
    if int(train_multi_nfe["multi_nfe_group_count"]) <= 0:
        issues.append("train_multi_nfe_group_coverage")
    if int(locked_multi_nfe["multi_nfe_group_count"]) <= 0:
        issues.append("locked_support_multi_nfe_group_coverage")
    row_count_relationships = {
        "train_supervision_matches_sources": {
            "actual": int(len(train_rows)),
            "expected": int(len(fixed_train) + len(ser_train)),
        },
        "locked_report_matches_fixed_source": {
            "actual": int(len(report_rows)),
            "expected": int(len(fixed_locked)),
        },
        "locked_teacher_support_matches_sources": {
            "actual": int(len(support_rows)),
            "expected": int(len(fixed_locked) + len(ser_locked)),
        },
        "train_support_full_reward_groups": {
            "actual": int(len(train_rows)),
            "expected": int(train_support["group_count"]) * EXPECTED_SUPPORT_SCHEDULE_COUNT,
        },
        "locked_support_full_reward_groups": {
            "actual": int(len(support_rows)),
            "expected": int(locked_support["group_count"]) * EXPECTED_SUPPORT_SCHEDULE_COUNT,
        },
        "fixed_train_physical_clock_groups": {
            "actual": int(len(fixed_train)),
            "expected": int(fixed_train_invariance["group_count"]) * EXPECTED_FIXED_SCHEDULE_COUNT,
        },
        "fixed_locked_physical_clock_groups": {
            "actual": int(len(report_rows)),
            "expected": int(fixed_locked_invariance["group_count"]) * EXPECTED_FIXED_SCHEDULE_COUNT,
        },
    }
    bad_row_count_relationships = [
        key for key, relationship in row_count_relationships.items() if not _row_count_ok(relationship)
    ]
    if bad_row_count_relationships:
        issues.extend(f"row_count:{key}" for key in bad_row_count_relationships)
    nfe_count_diagnostics = {
        "fixed_train": _nfe_count_diagnostics(fixed_train, expected_nfes_list),
        "ser_train": _nfe_count_diagnostics(ser_train, expected_nfes_list),
        "train_supervision": _nfe_count_diagnostics(train_rows, expected_nfes_list),
        "fixed_locked": _nfe_count_diagnostics(fixed_locked, expected_nfes_list),
        "ser_locked": _nfe_count_diagnostics(ser_locked, expected_nfes_list),
        "locked_report_context": _nfe_count_diagnostics(report_rows, expected_nfes_list),
        "locked_teacher_support": _nfe_count_diagnostics(support_rows, expected_nfes_list),
    }
    bad_nfe_counts = [
        name
        for name, diagnostic in nfe_count_diagnostics.items()
        if not bool(diagnostic["balanced_expected_nfe_counts"])
    ]
    if bad_nfe_counts:
        issues.extend(f"nfe_row_counts:{name}" for name in bad_nfe_counts)
    manifest = {
        "artifact": "gipo_additive_locked_multiaxis_panel_artifact_manifest",
        "panel": panel,
        "validation_passed": not issues,
        "issues": issues,
        "expected_target_nfes": expected_nfes_list,
        "support_schedule_keys": list(SUPPORT_SCHEDULE_KEYS),
        "expected_fixed_schedule_count": EXPECTED_FIXED_SCHEDULE_COUNT,
        "expected_support_schedule_count": EXPECTED_SUPPORT_SCHEDULE_COUNT,
        "paths": {key: str(value) for key, value in paths.items()},
        "row_counts": {
            "fixed_train": int(len(fixed_train)),
            "ser_train": int(len(ser_train)),
            "fixed_locked": int(len(fixed_locked)),
            "ser_locked": int(len(ser_locked)),
            "train_supervision": int(len(train_rows)),
            "locked_report_context": int(len(report_rows)),
            "locked_teacher_support": int(len(support_rows)),
        },
        "row_count_relationships": row_count_relationships,
        "nfe_count_diagnostics": nfe_count_diagnostics,
        "schedule_nfes": {
            "train_supervision": _schedule_nfes(train_rows),
            "locked_report_context": _schedule_nfes(report_rows),
            "locked_teacher_support": _schedule_nfes(support_rows),
        },
        "split_phases": {
            "train_supervision": _split_phases(train_rows),
            "locked_report_context": _split_phases(report_rows),
            "locked_teacher_support": _split_phases(support_rows),
        },
        "embedding_counts": {
            "train_context_embeddings": train_embeddings,
            "locked_context_embeddings": locked_embeddings,
        },
        "support_diagnostics": {
            "train_supervision": train_support,
            "locked_teacher_support": locked_support,
        },
        "fixed_cell_invariance": {
            "fixed_train": fixed_train_invariance,
            "fixed_locked": fixed_locked_invariance,
        },
        "teacher_oracle_support_sources": {
            "fixed_locked_rows": str(paths["fixed_locked_rows"]),
            "ser_locked_rows": str(paths["ser_locked_rows"]),
            "teacher_support_rows": str(paths["locked_teacher_support_rows"]),
            "support_schedule_keys": list(SUPPORT_SCHEDULE_KEYS),
            "ser_support_key": SER_SUPPORT_KEY,
        },
        "multi_nfe_group_diagnostics": {
            "train_supervision": train_multi_nfe,
            "locked_teacher_support": locked_multi_nfe,
        },
    }
    manifest_path = root / "summary" / f"{panel}_artifact_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_artifacts(root: Path, panels: Iterable[str]) -> None:
    for panel in panels:
        build_panel_artifacts(root, panel)


def validate_artifacts(root: Path, panels: Iterable[str]) -> dict[str, Any]:
    manifests = {panel: validate_panel(root, panel, PANELS[panel]) for panel in panels}
    issues = {panel: manifest["issues"] for panel, manifest in manifests.items() if manifest["issues"]}
    payload = {
        "artifact": "gipo_additive_locked_multiaxis_artifact_validation",
        "root": str(root),
        "validation_passed": not issues,
        "issues": issues,
        "panels": manifests,
    }
    out_path = root / "summary" / "artifact_validation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--panels", default="seen,unseen")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    root = Path(args.root)
    panels = [part.strip() for part in str(args.panels).split(",") if part.strip()]
    unknown = sorted(set(panels) - set(PANELS))
    if unknown:
        raise SystemExit(f"Unknown panels: {unknown}")
    if args.build:
        build_artifacts(root, panels)
    payload = validate_artifacts(root, panels) if args.validate or not args.build else {"validation_passed": True}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload.get("validation_passed", False):
        raise SystemExit("GIPO additive locked artifact validation failed.")


if __name__ == "__main__":
    main()
