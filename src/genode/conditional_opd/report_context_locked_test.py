from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from genode.conditional_opd.context_conditional import (
    ContextSupportStudentMLP,
    EmbeddingNormalizer,
    context_id_from_row,
    load_context_embedding_table,
    read_metric_rows_csv,
    remapped_series_index,
    series_key_from_row,
    validate_context_support_schedule_keys,
)
from genode.conditional_opd.evaluate_schedule_summary import (
    SELECTED_STUDENT_SCHEDULE_KEY,
    build_comparison_summary,
)
from genode.conditional_opd.models import setting_features
from genode.data.otflow_paths import resolve_project_path

EXPECTED_CONTEXT_POLICY_SCHEDULE_KEY = "context_conditional_expected_policy"
PRE_GUARD_ARGMAX_CONTEXT_POLICY_SCHEDULE_KEY = "context_conditional_pre_guard_argmax"
LOCKED_TEST_PHASE = "locked_test"


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in _parse_csv(text)]


def _read_csvs(paths_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path_text in _parse_csv(paths_text):
        rows.extend(read_metric_rows_csv(resolve_project_path(path_text)))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
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


def _finite_metric(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite, got {row[key]!r}.")
    return value


def _evaluation_seed_from_row(row: Mapping[str, Any]) -> int:
    parent = str(row.get("parent_row_signature", "") or "")
    parts = parent.split("|")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except (TypeError, ValueError):
            pass
    return int(row["seed"])


def _group_key(row: Mapping[str, Any]) -> Tuple[str, str, int, str, int, str]:
    return (
        str(row.get("dataset", row.get("dataset_key", ""))),
        str(row.get("split_phase", row.get("split", ""))),
        _evaluation_seed_from_row(row),
        str(row["solver_key"]),
        int(row["target_nfe"]),
        context_id_from_row(row),
    )


def _validate_locked_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("locked_context_rows contains no rows.")
    bad = sorted({str(row.get("split_phase", row.get("split", ""))) for row in rows if str(row.get("split_phase", row.get("split", ""))) != LOCKED_TEST_PHASE})
    if bad:
        raise ValueError(f"Context locked-test reporter only accepts split_phase={LOCKED_TEST_PHASE!r}; found {bad}.")


def _validate_requested_dimensions(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    seeds: Sequence[int],
    solver_names: Sequence[str],
    target_nfe_values: Sequence[int],
    label: str,
) -> None:
    if not rows:
        raise ValueError(f"{label} contains no rows.")
    _validate_locked_rows(rows)
    expected_cells = {
        (int(seed), str(solver), int(target_nfe))
        for seed in seeds
        for solver in solver_names
        for target_nfe in target_nfe_values
    }
    observed_cells = {
        (_evaluation_seed_from_row(row), str(row["solver_key"]), int(row["target_nfe"]))
        for row in rows
    }
    bad_datasets = sorted({str(row.get("dataset", row.get("dataset_key", ""))) for row in rows if str(row.get("dataset", row.get("dataset_key", ""))) != str(dataset)})
    if bad_datasets:
        raise ValueError(f"{label} contains rows for unexpected datasets: {bad_datasets}")
    extra_cells = sorted(observed_cells - expected_cells)
    missing_cells = sorted(expected_cells - observed_cells)
    if extra_cells:
        raise ValueError(f"{label} contains unexpected seed/solver/NFE cells: {extra_cells[:8]}")
    if missing_cells:
        raise ValueError(f"{label} is missing seed/solver/NFE cells: {missing_cells[:8]}")


def _load_student_checkpoint(path: str | Path) -> Tuple[ContextSupportStudentMLP, Dict[str, int], Tuple[str, ...], EmbeddingNormalizer, str, Dict[str, Any]]:
    payload = torch.load(resolve_project_path(str(path)), map_location="cpu")
    support_keys = validate_context_support_schedule_keys(tuple(str(key) for key in payload["support_schedule_keys"]))
    series_index_map = {str(key): int(value) for key, value in dict(payload["series_index_map"]).items()}
    normalizer_payload = payload.get("embedding_normalizer")
    if not normalizer_payload:
        raise ValueError("context_student checkpoint is missing embedding_normalizer.")
    normalizer = EmbeddingNormalizer(
        mean=np.asarray(normalizer_payload["mean"], dtype=np.float32),
        std=np.asarray(normalizer_payload["std"], dtype=np.float32),
    )
    setting_dim = int(setting_features("euler", 4).numel())
    student = ContextSupportStudentMLP(
        setting_dim=setting_dim,
        context_dim=int(payload["context_dim"]),
        num_series=len(series_index_map),
        support_schedule_keys=support_keys,
    )
    student.load_state_dict(payload["state_dict"])
    student.eval()
    guard = payload.get("calibration_holdout_non_regression_guard")
    if not isinstance(guard, dict) or not guard.get("cell_decision_map"):
        raise ValueError("context_student checkpoint is missing frozen calibration_holdout_non_regression_guard.")
    if bool(guard.get("locked_test_used_for_selection", True)):
        raise ValueError("Frozen calibration guard must confirm locked_test_used_for_selection=false.")
    if bool(guard.get("locked_test_used_for_guard_construction", True)):
        raise ValueError("Frozen calibration guard must confirm locked_test_used_for_guard_construction=false.")
    if tuple(str(key) for key in guard.get("support_schedule_keys", [])) != support_keys:
        raise ValueError("Frozen calibration guard support_schedule_keys do not match checkpoint support.")
    source_phases = {str(value) for value in guard.get("source_split_phases", [])}
    if LOCKED_TEST_PHASE in source_phases:
        raise ValueError("Frozen calibration guard source_split_phases must not include locked_test.")
    holdout_names = {str(value) for value in guard.get("observed_calibration_holdout_names", [])}
    if not holdout_names:
        raise ValueError("Frozen calibration guard must record observed_calibration_holdout_names.")
    if LOCKED_TEST_PHASE in holdout_names:
        raise ValueError("Frozen calibration guard holdout provenance must not include locked_test.")
    return student, series_index_map, support_keys, normalizer, str(payload.get("policy_id", "")), dict(guard)


def _aggregate_seed_rows(rows: Sequence[Mapping[str, Any]], *, schedule_key: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("dataset", row.get("dataset_key", ""))),
            str(row.get("split_phase", "")),
            _evaluation_seed_from_row(row),
            str(row["solver_key"]),
            int(row["target_nfe"]),
            str(schedule_key),
        )
        grouped[key].append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, split_phase, seed, solver, target_nfe, key), group in sorted(grouped.items()):
        item: Dict[str, Any] = {
            "dataset": dataset,
            "split_phase": split_phase,
            "seed": int(seed),
            "solver_key": solver,
            "target_nfe": int(target_nfe),
            "scheduler_key": key,
            "n_contexts": int(len(group)),
            "crps": float(np.mean(np.asarray([_finite_metric(row, "crps") for row in group], dtype=np.float64))),
            "mase": float(np.mean(np.asarray([_finite_metric(row, "mase") for row in group], dtype=np.float64))),
        }
        mse_values = [row.get("mse") for row in group if row.get("mse") not in (None, "")]
        if mse_values:
            item["mse"] = float(np.mean(np.asarray([float(value) for value in mse_values], dtype=np.float64)))
        out.append(item)
    return out


def _mean_by_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) not in (None, "")]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _coverage_missing_groups(
    rows: Sequence[Mapping[str, Any]],
    support_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    expected = set(str(key) for key in support_keys)
    by_group: Dict[Tuple[str, str, int, str, int, str], set[str]] = defaultdict(set)
    for row in rows:
        schedule_key = str(row["scheduler_key"])
        if schedule_key in expected:
            by_group[_group_key(row)].add(schedule_key)
    missing: List[Dict[str, Any]] = []
    for key, observed in sorted(by_group.items()):
        if observed != expected:
            dataset, split_phase, seed, solver, target_nfe, context_id = key
            missing.append(
                {
                    "dataset": dataset,
                    "split_phase": split_phase,
                    "seed": int(seed),
                    "solver_key": solver,
                    "target_nfe": int(target_nfe),
                    "context_id": context_id,
                    "missing_schedule_keys": sorted(expected - observed),
                    "extra_schedule_keys": sorted(observed - expected),
                }
            )
    return missing


def report_context_locked_test(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = resolve_project_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    training_summary = json.loads(resolve_project_path(str(args.training_summary)).read_text(encoding="utf-8"))
    if bool(training_summary.get("locked_test_used_for_selection", True)):
        raise ValueError("Refusing locked-test report because training_summary does not confirm locked_test_used_for_selection=false.")
    student, series_index_map, checkpoint_support_keys, normalizer, checkpoint_policy_id, calibration_guard = _load_student_checkpoint(args.context_student_checkpoint)
    summary_policy_id = str(training_summary.get("policy_id", ""))
    if not summary_policy_id or checkpoint_policy_id != summary_policy_id:
        raise ValueError(f"context_student checkpoint policy_id does not match training summary: {checkpoint_policy_id!r} != {summary_policy_id!r}")
    summary_guard = training_summary.get("calibration_holdout_non_regression_guard")
    if not isinstance(summary_guard, dict) or not summary_guard.get("guard_id"):
        raise ValueError("training_summary is missing calibration_holdout_non_regression_guard.")
    if str(summary_guard.get("guard_id")) != str(calibration_guard.get("guard_id")):
        raise ValueError(
            "context_student checkpoint calibration guard does not match training summary: "
            f"{calibration_guard.get('guard_id')!r} != {summary_guard.get('guard_id')!r}"
        )
    if str(summary_guard.get("guard_table_hash", "")) != str(calibration_guard.get("guard_table_hash", "")):
        raise ValueError("context_student checkpoint calibration guard hash does not match training summary.")
    requested_support = (
        validate_context_support_schedule_keys(_parse_csv(str(args.support_schedule_keys)))
        if str(args.support_schedule_keys).strip()
        else checkpoint_support_keys
    )
    if tuple(requested_support) != tuple(checkpoint_support_keys):
        raise ValueError(f"Requested support keys do not match checkpoint support keys: {requested_support} != {checkpoint_support_keys}")
    locked_rows = _read_csvs(str(args.locked_context_rows))
    seeds = _parse_int_csv(str(args.seeds))
    solver_names = _parse_csv(str(args.solver_names))
    target_nfes = _parse_int_csv(str(args.target_nfe_values))
    _validate_requested_dimensions(
        locked_rows,
        dataset=str(args.dataset),
        seeds=seeds,
        solver_names=solver_names,
        target_nfe_values=target_nfes,
        label="locked_context_rows",
    )
    missing = _coverage_missing_groups(locked_rows, requested_support)
    if missing:
        raise ValueError(f"Locked context rows are missing support schedules for {len(missing)} context groups; first={missing[:3]}")
    raw_embeddings = load_context_embedding_table(resolve_project_path(str(args.locked_context_embeddings_npz)))
    embeddings = normalizer.transform_table(raw_embeddings)

    by_group_schedule: Dict[Tuple[Tuple[str, str, int, str, int, str], str], Dict[str, Any]] = {}
    group_refs: Dict[Tuple[str, str, int, str, int, str], Dict[str, Any]] = {}
    duplicates: List[Dict[str, Any]] = []
    for row in locked_rows:
        schedule_key = str(row["scheduler_key"])
        if schedule_key not in set(requested_support):
            continue
        group = _group_key(row)
        if (group, schedule_key) in by_group_schedule:
            duplicates.append(
                {
                    "group": list(group),
                    "scheduler_key": schedule_key,
                }
            )
        by_group_schedule[(group, schedule_key)] = dict(row)
        group_refs.setdefault(group, dict(row))
    if not by_group_schedule:
        raise ValueError("locked_context_rows contains no requested support schedule rows.")
    if duplicates:
        raise ValueError(f"locked_context_rows contains duplicate context/support rows; first={duplicates[:3]}")

    selected_rows: List[Dict[str, Any]] = []
    pre_guard_argmax_rows: List[Dict[str, Any]] = []
    expected_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    student.eval()
    with torch.no_grad():
        for group, ref in sorted(group_refs.items()):
            dataset, split_phase, seed, solver, target_nfe, context_id = group
            if context_id not in embeddings:
                raise KeyError(f"Missing locked-test context embedding for {context_id}.")
            setting = setting_features(solver, target_nfe)[None, :]
            series = torch.tensor([remapped_series_index(ref, series_index_map)], dtype=torch.long)
            context = torch.tensor(np.asarray(embeddings[context_id], dtype=np.float32)[None, :], dtype=torch.float32)
            probabilities = student.probabilities(setting, series, context)[0].detach().cpu().numpy().astype(np.float64)
            student_argmax_idx = int(np.argmax(probabilities))
            student_argmax_support = str(requested_support[student_argmax_idx])
            pre_guard_source = by_group_schedule[(group, student_argmax_support)]
            pre_guard = dict(pre_guard_source)
            pre_guard.update(
                {
                    "seed": int(seed),
                    "scheduler_key": PRE_GUARD_ARGMAX_CONTEXT_POLICY_SCHEDULE_KEY,
                    "schedule_name": "Context-Conditional OPD Student Pre-Guard Argmax",
                    "pre_guard_support_schedule_key": student_argmax_support,
                    "pre_guard_support_probability": float(probabilities[student_argmax_idx]),
                    "context_id": context_id,
                    "reporting_only": True,
                    "locked_test_used_for_selection": False,
                    "locked_test_used_for_guard_construction": False,
                }
            )
            pre_guard_argmax_rows.append(pre_guard)
            guard_decision = dict(calibration_guard.get("cell_decision_map", {}).get(f"{solver}/{target_nfe}", {}))
            if not guard_decision:
                raise ValueError(f"Frozen calibration guard has no decision for cell {(solver, target_nfe)}.")
            deployed_mode = str(guard_decision.get("deployed_mode", "context_student"))
            fallback_support = str(guard_decision.get("fallback_schedule_key", "") or guard_decision.get("best_static_support_schedule_key", ""))
            if deployed_mode == "static_support":
                if fallback_support not in requested_support:
                    raise ValueError(f"Frozen calibration guard fallback {fallback_support!r} is not in requested support.")
                selected_support = fallback_support
                post_guard_source = "fallback"
                guard_applied = True
                guard_passed = False
                guard_reason = "calibration_context_student_failed_non_regression_margin"
            elif deployed_mode == "context_student":
                selected_support = student_argmax_support
                post_guard_source = "argmax"
                guard_applied = True
                guard_passed = True
                guard_reason = "calibration_context_student_passed_non_regression_margin"
            else:
                raise ValueError(f"Unsupported frozen calibration guard deployed_mode={deployed_mode!r}.")
            selected_idx = requested_support.index(selected_support)
            selected_source = by_group_schedule[(group, selected_support)]
            selected = dict(selected_source)
            selected.update(
                {
                    "seed": int(seed),
                    "scheduler_key": SELECTED_STUDENT_SCHEDULE_KEY,
                    "schedule_name": "Context-Conditional OPD Guarded Policy",
                    "selected_support_schedule_key": selected_support,
                    "deployed_mode": deployed_mode,
                    "pre_guard_support_schedule_key": student_argmax_support,
                    "pre_guard_support_probability": float(probabilities[student_argmax_idx]),
                    "post_guard_support_schedule_key": selected_support,
                    "post_guard_source": post_guard_source,
                    "guard_applied": guard_applied,
                    "guard_passed": guard_passed,
                    "guard_reason": guard_reason,
                    "student_argmax_support_schedule_key": student_argmax_support,
                    "student_argmax_support_probability": float(probabilities[student_argmax_idx]),
                    "fallback_schedule_key": fallback_support if deployed_mode == "static_support" else "",
                    "selected_support_probability": float(probabilities[selected_idx]),
                    "calibration_guard_id": str(calibration_guard.get("guard_id", "")),
                    "guard_table_id": str(calibration_guard.get("guard_id", "")),
                    "guard_table_hash": str(calibration_guard.get("guard_table_hash", "")),
                    "calibration_guard_context_student_score": guard_decision.get("context_student_score"),
                    "calibration_guard_best_static_score": guard_decision.get("best_static_score"),
                    "context_id": context_id,
                    "reporting_only": True,
                    "locked_test_used_for_selection": False,
                    "locked_test_used_for_guard_construction": False,
                }
            )
            selected_rows.append(selected)

            expected = dict(ref)
            expected.update(
                {
                    "seed": int(seed),
                    "scheduler_key": EXPECTED_CONTEXT_POLICY_SCHEDULE_KEY,
                    "schedule_name": "Context-Conditional OPD Student Expected Policy",
                    "context_id": context_id,
                    "crps": float(
                        sum(float(probabilities[idx]) * _finite_metric(by_group_schedule[(group, key)], "crps") for idx, key in enumerate(requested_support))
                    ),
                    "mase": float(
                        sum(float(probabilities[idx]) * _finite_metric(by_group_schedule[(group, key)], "mase") for idx, key in enumerate(requested_support))
                    ),
                    "reporting_only": True,
                    "locked_test_used_for_selection": False,
                    "locked_test_used_for_guard_construction": False,
                }
            )
            if all(by_group_schedule[(group, key)].get("mse") not in (None, "") for key in requested_support):
                expected["mse"] = float(sum(float(probabilities[idx]) * float(by_group_schedule[(group, key)]["mse"]) for idx, key in enumerate(requested_support)))
            expected_rows.append(expected)

            decision = {
                "dataset": dataset,
                "split_phase": split_phase,
                "seed": int(seed),
                "solver_key": solver,
                "target_nfe": int(target_nfe),
                "context_id": context_id,
                "series_key": series_key_from_row(ref),
                "selected_support_schedule_key": selected_support,
                "deployed_mode": deployed_mode,
                "pre_guard_support_schedule_key": student_argmax_support,
                "pre_guard_support_probability": float(probabilities[student_argmax_idx]),
                "post_guard_support_schedule_key": selected_support,
                "post_guard_source": post_guard_source,
                "guard_applied": guard_applied,
                "guard_passed": guard_passed,
                "guard_reason": guard_reason,
                "student_argmax_support_schedule_key": student_argmax_support,
                "student_argmax_support_probability": float(probabilities[student_argmax_idx]),
                "fallback_schedule_key": fallback_support if deployed_mode == "static_support" else "",
                "selected_support_probability": float(probabilities[selected_idx]),
                "calibration_guard_id": str(calibration_guard.get("guard_id", "")),
                "guard_table_id": str(calibration_guard.get("guard_id", "")),
                "guard_table_hash": str(calibration_guard.get("guard_table_hash", "")),
                "reporting_only": True,
                "locked_test_used_for_selection": False,
                "locked_test_used_for_guard_construction": False,
            }
            for idx, key in enumerate(requested_support):
                decision[f"prob_{key}"] = float(probabilities[idx])
                decision[f"crps_{key}"] = _finite_metric(by_group_schedule[(group, key)], "crps")
                decision[f"mase_{key}"] = _finite_metric(by_group_schedule[(group, key)], "mase")
            decision_rows.append(decision)

    selected_seed_rows = _aggregate_seed_rows(selected_rows, schedule_key=SELECTED_STUDENT_SCHEDULE_KEY)
    pre_guard_argmax_seed_rows = _aggregate_seed_rows(
        pre_guard_argmax_rows,
        schedule_key=PRE_GUARD_ARGMAX_CONTEXT_POLICY_SCHEDULE_KEY,
    )
    expected_seed_rows = _aggregate_seed_rows(expected_rows, schedule_key=EXPECTED_CONTEXT_POLICY_SCHEDULE_KEY)
    usage_by_cell: Dict[str, Dict[str, int]] = defaultdict(dict)
    argmax_usage_by_cell: Dict[str, Dict[str, int]] = defaultdict(dict)
    usage_counts: Dict[Tuple[str, int], Counter[str]] = defaultdict(Counter)
    argmax_usage_counts: Dict[Tuple[str, int], Counter[str]] = defaultdict(Counter)
    for row in selected_rows:
        usage_counts[(str(row["solver_key"]), int(row["target_nfe"]))][str(row["selected_support_schedule_key"])] += 1
        argmax_usage_counts[(str(row["solver_key"]), int(row["target_nfe"]))][str(row["student_argmax_support_schedule_key"])] += 1
    for (solver, target_nfe), counts in sorted(usage_counts.items()):
        usage_by_cell[f"{solver}/{target_nfe}"] = {key: int(counts.get(key, 0)) for key in requested_support}
    for (solver, target_nfe), counts in sorted(argmax_usage_counts.items()):
        argmax_usage_by_cell[f"{solver}/{target_nfe}"] = {key: int(counts.get(key, 0)) for key in requested_support}
    guard_fallback_count = sum(1 for row in selected_rows if str(row.get("deployed_mode")) == "static_support")
    guard_context_count = sum(1 for row in selected_rows if str(row.get("deployed_mode")) == "context_student")

    _write_csv(out_dir / "context_locked_test_policy_rows.csv", selected_rows)
    _write_csv(out_dir / "context_locked_test_pre_guard_argmax_rows.csv", pre_guard_argmax_rows)
    _write_csv(out_dir / "context_locked_test_expected_policy_rows.csv", expected_rows)
    _write_csv(out_dir / "context_locked_test_policy_seed_rows.csv", selected_seed_rows)
    _write_csv(out_dir / "context_locked_test_pre_guard_argmax_seed_rows.csv", pre_guard_argmax_seed_rows)
    _write_csv(out_dir / "context_locked_test_expected_policy_seed_rows.csv", expected_seed_rows)
    _write_csv(out_dir / "context_locked_test_policy_decisions.csv", decision_rows)

    baseline_rows = _read_csvs(str(args.baseline_rows)) if str(args.baseline_rows).strip() else []
    comparator_rows = _read_csvs(str(args.comparator_rows)) if str(args.comparator_rows).strip() else []
    comparison_summary = None
    expected_comparison_summary = None
    if baseline_rows:
        _validate_requested_dimensions(
            baseline_rows,
            dataset=str(args.dataset),
            seeds=seeds,
            solver_names=solver_names,
            target_nfe_values=target_nfes,
            label="baseline_rows",
        )
        if comparator_rows:
            _validate_requested_dimensions(
                comparator_rows,
                dataset=str(args.dataset),
                seeds=seeds,
                solver_names=solver_names,
                target_nfe_values=target_nfes,
                label="comparator_rows",
            )
        comparison_summary = build_comparison_summary(
            baseline_rows=baseline_rows,
            comparator_rows=comparator_rows,
            student_rows=selected_seed_rows,
            dataset=str(args.dataset),
            split_phase=LOCKED_TEST_PHASE,
            seeds=seeds,
            solver_names=solver_names,
            target_nfe_values=target_nfes,
        )
        (out_dir / "context_locked_test_comparison_summary.json").write_text(
            json.dumps(comparison_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        expected_comparison_summary = build_comparison_summary(
            baseline_rows=baseline_rows,
            comparator_rows=comparator_rows,
            student_rows=expected_seed_rows,
            dataset=str(args.dataset),
            split_phase=LOCKED_TEST_PHASE,
            seeds=seeds,
            solver_names=solver_names,
            target_nfe_values=target_nfes,
        )
        (out_dir / "context_locked_test_expected_comparison_summary.json").write_text(
            json.dumps(expected_comparison_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    summary = {
        "artifact": "context_locked_test_policy_report",
        "dataset": str(args.dataset),
        "split_phase": LOCKED_TEST_PHASE,
        "reporting_only": True,
        "selection_source": "frozen_calibration_guarded_policy",
        "frozen_guard_table_applied": True,
        "calibration_guard_id": str(calibration_guard.get("guard_id", "")),
        "guard_table_id": str(calibration_guard.get("guard_id", "")),
        "guard_table_hash": str(calibration_guard.get("guard_table_hash", "")),
        "guard_source_row_hash": str(calibration_guard.get("source_row_hash", "")),
        "guard_source_context_ids_hash": str(calibration_guard.get("source_context_ids_hash", "")),
        "guard_fallback_count": int(guard_fallback_count),
        "guard_context_count": int(guard_context_count),
        "calibration_holdout_non_regression_guard": calibration_guard,
        "locked_test_used_for_selection": False,
        "locked_test_used_for_guard_construction": False,
        "policy_id": summary_policy_id,
        "support_schedule_keys": list(requested_support),
        "context_row_count": int(len(selected_rows)),
        "pre_guard_argmax_context_row_count": int(len(pre_guard_argmax_rows)),
        "expected_context_row_count": int(len(expected_rows)),
        "seed_row_count": int(len(selected_seed_rows)),
        "pre_guard_argmax_seed_row_count": int(len(pre_guard_argmax_seed_rows)),
        "expected_seed_row_count": int(len(expected_seed_rows)),
        "context_count": int(len({context_id_from_row(row) for row in selected_rows})),
        "series_count": int(len({series_key_from_row(row) for row in selected_rows})),
        "pre_guard_argmax_crps_mean": _mean_by_metric(pre_guard_argmax_rows, "crps"),
        "pre_guard_argmax_mase_mean": _mean_by_metric(pre_guard_argmax_rows, "mase"),
        "selected_guarded_crps_mean": _mean_by_metric(selected_rows, "crps"),
        "selected_guarded_mase_mean": _mean_by_metric(selected_rows, "mase"),
        "expected_policy_crps_mean": _mean_by_metric(expected_rows, "crps"),
        "expected_policy_mase_mean": _mean_by_metric(expected_rows, "mase"),
        "support_usage_by_solver_nfe": usage_by_cell,
        "student_argmax_support_usage_by_solver_nfe": argmax_usage_by_cell,
        "policy_rows_csv": str(out_dir / "context_locked_test_policy_rows.csv"),
        "pre_guard_argmax_rows_csv": str(out_dir / "context_locked_test_pre_guard_argmax_rows.csv"),
        "expected_policy_rows_csv": str(out_dir / "context_locked_test_expected_policy_rows.csv"),
        "policy_seed_rows_csv": str(out_dir / "context_locked_test_policy_seed_rows.csv"),
        "pre_guard_argmax_seed_rows_csv": str(out_dir / "context_locked_test_pre_guard_argmax_seed_rows.csv"),
        "expected_policy_seed_rows_csv": str(out_dir / "context_locked_test_expected_policy_seed_rows.csv"),
        "policy_decisions_csv": str(out_dir / "context_locked_test_policy_decisions.csv"),
        "comparison_summary_json": str(out_dir / "context_locked_test_comparison_summary.json") if comparison_summary else None,
        "expected_comparison_summary_json": str(out_dir / "context_locked_test_expected_comparison_summary.json") if expected_comparison_summary else None,
    }
    (out_dir / "context_locked_test_policy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report locked-test performance for a frozen context-conditional support student.")
    parser.add_argument("--context_student_checkpoint", required=True)
    parser.add_argument("--training_summary", required=True)
    parser.add_argument("--locked_context_rows", required=True, help="Comma-separated fixed/SER locked-test context-row CSVs.")
    parser.add_argument("--locked_context_embeddings_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--support_schedule_keys", default="")
    parser.add_argument("--baseline_rows", default="")
    parser.add_argument("--comparator_rows", default="")
    parser.add_argument("--dataset", default="solar_energy_10m")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--solver_names", default="euler,heun,midpoint_rk2,dpmpp2m")
    parser.add_argument("--target_nfe_values", default="4,8,12")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    summary = report_context_locked_test(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
