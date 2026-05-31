from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from genode.conditional_opd.models import (
    ScheduleStudentMLP,
    ScheduleTeacherMLP,
    count_parameters,
    grid_to_intervals,
    setting_features,
    solver_macro_steps,
    teacher_features,
    validate_time_grid,
)
from genode.conditional_opd.objectives import attach_reward_columns, build_fixed_reference_table, rewards_by_setting, seed_mean_metric_rows
from genode.conditional_opd.ser_ptg_reference import SER_PTG_SCHEDULE_KEY, grid_geometry
from genode.data.otflow_paths import project_outputs_root, resolve_project_path
from genode.evaluation.otflow_evaluation_support import TRAIN_TUNING_PHASE
from genode.models.otflow_train_val import save_json, seed_all
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS, build_schedule_grid

DEFAULT_SOLVERS: Tuple[str, ...] = ("euler", "heun", "midpoint_rk2", "dpmpp2m")
DEFAULT_TARGET_NFES: Tuple[int, ...] = (4, 8, 12)
STUDENT_SCHEDULE_PREFIX = "conditional_opd_student_steps"
DEFAULT_DIRECT_OPD_BUDGETS: Tuple[int, ...] = (5, 10, 15, 20, 25)
DIAGNOSTIC_DIRECT_OPD_BUDGETS: Tuple[int, ...] = (35, 50)
DEFAULT_TEACHER_FIXED_SCHEDULE_KEYS: Tuple[str, ...] = tuple(BASELINE_SCHEDULE_KEYS)
DEFAULT_LATE_BIASED_DEMO_SCHEDULES: Tuple[str, ...] = ("late_power_3",)
DEFAULT_LATE_BIASED_DEMO_WEIGHT = 1.0
STUDENT_SER_PTG_REGULARIZER_NONE = "none"
STUDENT_SER_PTG_REGULARIZER_JS = "js"
STUDENT_SER_PTG_REGULARIZER_KL = "kl"
STUDENT_SER_PTG_REGULARIZERS: Tuple[str, ...] = (
    STUDENT_SER_PTG_REGULARIZER_NONE,
    STUDENT_SER_PTG_REGULARIZER_JS,
    STUDENT_SER_PTG_REGULARIZER_KL,
)
DEFAULT_STUDENT_SER_PTG_REGULARIZATION_EPS = 1e-8
DEFAULT_TEACHER_RANK_TEMPERATURE = 0.5
DEFAULT_TEACHER_REGRESSION_WEIGHT = 0.25
DEFAULT_TEACHER_PAIRS_PER_CANDIDATE = 32
DEFAULT_TEACHER_PAIR_MARGIN = 0.0
DEFAULT_TEACHER_DIAGNOSTIC_HOLDOUT_FRACTION = 0.20
DEFAULT_NEURAL_POLICY_ROOT = project_outputs_root() / "train20_v43_pooled_one_round_calibration" / "neural_policy"


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in _parse_csv(text)]


def _parse_teacher_fixed_schedule_keys(text: str) -> Tuple[str, ...]:
    keys = tuple(_parse_csv(text))
    if not keys:
        raise ValueError("teacher_fixed_schedule_keys must contain at least one fixed schedule.")
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"teacher_fixed_schedule_keys contains duplicates: {duplicates}")
    unsupported = sorted(set(keys) - set(BASELINE_SCHEDULE_KEYS))
    if unsupported:
        raise ValueError(f"teacher_fixed_schedule_keys must be fixed baselines; unsupported: {unsupported}")
    if "uniform" not in keys:
        raise ValueError("teacher_fixed_schedule_keys must include uniform for paired train-set metrics.")
    return keys


def _parse_reward_reference_schedule_keys(text: str, *, teacher_fixed_schedule_keys: Sequence[str]) -> Tuple[str, ...]:
    keys = tuple(_parse_csv(text)) if str(text).strip() else tuple(str(key) for key in teacher_fixed_schedule_keys)
    if not keys:
        raise ValueError("reward_reference_schedule_keys must contain at least one fixed schedule.")
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"reward_reference_schedule_keys contains duplicates: {duplicates}")
    unsupported = sorted(set(keys) - set(BASELINE_SCHEDULE_KEYS))
    if unsupported:
        raise ValueError(f"reward_reference_schedule_keys must be fixed baselines; unsupported: {unsupported}")
    missing_teacher_keys = sorted(set(str(key) for key in teacher_fixed_schedule_keys) - set(keys))
    if missing_teacher_keys:
        raise ValueError(f"reward_reference_schedule_keys must include teacher fixed schedules: {missing_teacher_keys}")
    return keys


def _load_csv_rows(path: str | Path) -> List[Dict[str, Any]]:
    resolved = resolve_project_path(str(path))
    if not resolved.exists():
        raise FileNotFoundError(f"Rows CSV not found: {resolved}")
    with resolved.open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _clean_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    solvers: Sequence[str],
    target_nfes: Sequence[int],
    allowed_schedules: Sequence[str],
) -> List[Dict[str, Any]]:
    solver_set = {str(solver) for solver in solvers}
    nfe_set = {int(nfe) for nfe in target_nfes}
    schedule_set = {str(key) for key in allowed_schedules}
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("dataset")) != str(dataset):
            continue
        if str(row.get("solver_key")) not in solver_set:
            continue
        try:
            target_nfe = int(row.get("target_nfe", 0))
        except (TypeError, ValueError):
            continue
        if target_nfe not in nfe_set:
            continue
        schedule_key = str(row.get("scheduler_key", ""))
        if schedule_key not in schedule_set:
            continue
        try:
            crps = float(row["crps"])
            mase = float(row["mase"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(crps) and np.isfinite(mase) and crps > 0.0 and mase > 0.0):
            continue
        clean = dict(row)
        clean["target_nfe"] = int(target_nfe)
        clean["crps"] = float(crps)
        clean["mase"] = float(mase)
        try:
            clean["seed"] = int(row.get("seed", -1))
        except (TypeError, ValueError):
            clean["seed"] = -1
        out.append(clean)
    return out


def _clean_split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    solvers: Sequence[str],
    target_nfes: Sequence[int],
    allowed_schedules: Sequence[str],
    seeds: Sequence[int],
    required_split_phase: str,
) -> List[Dict[str, Any]]:
    clean = _clean_metric_rows(
        rows,
        dataset=str(dataset),
        solvers=solvers,
        target_nfes=target_nfes,
        allowed_schedules=allowed_schedules,
    )
    split = str(required_split_phase).strip()
    if split:
        clean = [row for row in clean if str(row.get("split_phase")) == split]
    seed_set = {int(seed) for seed in seeds}
    clean = [row for row in clean if int(row.get("seed", -1)) in seed_set]
    return clean


def _assert_expected_train_tuning_metadata(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_sampler: str,
    expected_fraction: str,
    label: str,
) -> None:
    sampler = str(expected_sampler).strip()
    if sampler:
        observed = sorted({str(row.get("train_tuning_sampler", "")) for row in rows})
        if observed != [sampler]:
            raise ValueError(f"{label} rows use train_tuning_sampler={observed}, expected {sampler!r}.")
    fraction_text = str(expected_fraction).strip()
    if fraction_text:
        expected = float(fraction_text)
        mismatches = []
        for row in rows:
            try:
                observed_fraction = float(row.get("train_tuning_fraction", ""))
            except (TypeError, ValueError):
                mismatches.append(row.get("scheduler_key", ""))
                continue
            if abs(observed_fraction - expected) > 1e-12:
                mismatches.append(row.get("scheduler_key", ""))
        if mismatches:
            raise ValueError(f"{label} rows do not match expected train_tuning_fraction={expected}.")


def _load_schedule_summary_grids(path: str | Path) -> Dict[Tuple[str, str, int], Tuple[float, ...]]:
    if not str(path).strip():
        return {}
    resolved = resolve_project_path(str(path))
    if not resolved.exists():
        raise FileNotFoundError(f"Reference schedule summary not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    schedules = payload.get("schedules")
    if schedules:
        schedule_items = list(schedules)
    else:
        schedule_items = [
            {
                "scheduler_key": str(payload.get("schedule_key", payload.get("scheduler_key", SER_PTG_SCHEDULE_KEY))),
                "predictions": payload.get("predictions", []) or [],
            }
        ]
    grids: Dict[Tuple[str, str, int], Tuple[float, ...]] = {}
    for schedule in schedule_items:
        schedule_key = str(schedule.get("scheduler_key", schedule.get("schedule_key", "")))
        for item in list(schedule.get("predictions", []) or []):
            solver = str(item["solver_key"])
            target_nfe = int(item["target_nfe"])
            macro_steps = solver_macro_steps(solver, target_nfe)
            grid = validate_time_grid(item.get("time_grid", []), macro_steps=macro_steps)
            grids[(schedule_key, solver, target_nfe)] = grid
    return grids


def _load_schedule_summary_grids_many(paths: Sequence[str | Path]) -> Dict[Tuple[str, str, int], Tuple[float, ...]]:
    grids: Dict[Tuple[str, str, int], Tuple[float, ...]] = {}
    for path in paths:
        loaded = _load_schedule_summary_grids(path)
        duplicates = sorted(set(grids).intersection(loaded), key=lambda item: (item[0], item[1], item[2]))
        if duplicates:
            raise ValueError(f"Duplicate schedule grid entries in candidate summaries: {duplicates[:12]}")
        grids.update(loaded)
    return grids


def _schedule_keys_from_summary_paths(paths: Sequence[str | Path]) -> List[str]:
    keys: List[str] = []
    for path in paths:
        resolved = resolve_project_path(str(path))
        if not resolved.exists():
            raise FileNotFoundError(f"Schedule summary not found: {resolved}")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        schedules = payload.get("schedules")
        if schedules:
            schedule_items = list(schedules)
        else:
            schedule_items = [
                {
                    "scheduler_key": str(payload.get("schedule_key", payload.get("scheduler_key", ""))),
                }
            ]
        for schedule in schedule_items:
            key = str(schedule.get("scheduler_key", schedule.get("schedule_key", ""))).strip()
            if key:
                keys.append(key)
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate schedule keys in candidate summaries: {duplicates[:12]}")
    reserved = sorted(set(keys).intersection(BASELINE_SCHEDULE_KEYS) | (set(keys) & {SER_PTG_SCHEDULE_KEY}))
    if reserved:
        raise ValueError(f"Generated candidate summaries must not reuse fixed schedule keys: {reserved}")
    return sorted(keys)


def _assert_complete_seed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    solvers: Sequence[str],
    target_nfes: Sequence[int],
    schedules: Sequence[str],
    seeds: Sequence[int],
    label: str,
) -> None:
    observed = set()
    duplicates = []
    for row in rows:
        key = (int(row.get("seed", -1)), str(row["solver_key"]), int(row["target_nfe"]), str(row["scheduler_key"]))
        if key in observed:
            duplicates.append(key)
        observed.add(key)
    if duplicates:
        raise ValueError(f"{label} contains duplicate seed rows: {sorted(duplicates)[:12]}")
    expected = {
        (int(seed), str(solver), int(target_nfe), str(schedule))
        for seed in seeds
        for solver in solvers
        for target_nfe in target_nfes
        for schedule in schedules
    }
    missing = sorted(expected - observed, key=lambda item: (item[1], item[2], item[3], item[0]))
    if missing:
        raise ValueError(f"{label} rows are incomplete: {missing[:12]}")


def _grid_for_schedule(
    schedule_key: str,
    solver_key: str,
    target_nfe: int,
    *,
    summary_grids: Mapping[Tuple[str, str, int], Sequence[float]],
) -> Tuple[float, ...]:
    macro_steps = solver_macro_steps(str(solver_key), int(target_nfe))
    key = str(schedule_key)
    if key in BASELINE_SCHEDULE_KEYS:
        grid = build_schedule_grid(key, macro_steps)
        if grid is None:
            raise ValueError(f"No baseline schedule grid for {key}.")
        return validate_time_grid(grid, macro_steps=macro_steps)
    lookup_key = (key, str(solver_key), int(target_nfe))
    if lookup_key not in summary_grids:
        raise KeyError(f"Missing generated schedule grid for {lookup_key}.")
    return validate_time_grid(summary_grids[lookup_key], macro_steps=macro_steps)


def _padded_intervals_from_grid(grid: Sequence[float], max_macro_steps: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    intervals = torch.zeros(int(max_macro_steps), dtype=torch.float32, device=device)
    raw = torch.tensor(grid_to_intervals(grid), dtype=torch.float32, device=device)
    intervals[: raw.numel()] = raw
    return intervals


def differentiable_teacher_features(
    setting_feature_batch: torch.Tensor,
    interval_batch: torch.Tensor,
    *,
    max_macro_steps: int,
) -> torch.Tensor:
    if setting_feature_batch.ndim != 2:
        raise ValueError("setting_feature_batch must be 2D.")
    if interval_batch.ndim != 2:
        raise ValueError("interval_batch must be 2D.")
    if interval_batch.shape[-1] != int(max_macro_steps):
        raise ValueError(f"interval_batch width {interval_batch.shape[-1]} does not match max_macro_steps={max_macro_steps}.")
    return torch.cat([setting_feature_batch, interval_batch], dim=-1)


def _normalized_targets_by_setting(
    targets: Sequence[float],
    setting_keys: Sequence[Tuple[str, int]],
) -> Tuple[List[float], Dict[str, Dict[str, float]]]:
    grouped: Dict[Tuple[str, int], List[float]] = {}
    for target, setting in zip(targets, setting_keys):
        grouped.setdefault(setting, []).append(float(target))
    normalizers: Dict[Tuple[str, int], Tuple[float, float]] = {}
    metadata: Dict[str, Dict[str, float]] = {}
    for setting, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        scale = 1.4826 * mad
        if not math.isfinite(scale) or scale <= 1e-8:
            scale = float(np.std(arr))
        if not math.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        normalizers[setting] = (median, scale)
        metadata[f"{setting[0]}:{int(setting[1])}"] = {
            "median": float(median),
            "scale": float(scale),
            "count": float(arr.size),
        }
    normalized = []
    for target, setting in zip(targets, setting_keys):
        median, scale = normalizers[setting]
        normalized.append(float((float(target) - median) / scale))
    return normalized, metadata


def _teacher_pair_tensor(
    targets: torch.Tensor,
    setting_keys: Sequence[Tuple[str, int]],
    *,
    margin: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left: List[int] = []
    right: List[int] = []
    signs: List[float] = []
    target_values = [float(value) for value in targets.detach().cpu().tolist()]
    by_setting: Dict[Tuple[str, int], List[int]] = {}
    for idx, setting in enumerate(setting_keys):
        by_setting.setdefault(setting, []).append(int(idx))
    min_delta = float(margin)
    for indices in by_setting.values():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1 :]:
                diff = target_values[i] - target_values[j]
                if abs(diff) <= min_delta:
                    continue
                if diff > 0:
                    left.append(i)
                    right.append(j)
                    signs.append(1.0)
                else:
                    left.append(i)
                    right.append(j)
                    signs.append(-1.0)
    if not left:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        return empty_long, empty_long, empty_float
    return (
        torch.tensor(left, dtype=torch.long, device=device),
        torch.tensor(right, dtype=torch.long, device=device),
        torch.tensor(signs, dtype=torch.float32, device=device),
    )


def _rankdata_average(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    pos = 0
    while pos < arr.size:
        end = pos + 1
        while end < arr.size and arr[order[end]] == arr[order[pos]]:
            end += 1
        avg_rank = 0.5 * (pos + end - 1)
        ranks[order[pos:end]] = avg_rank
        pos = end
    return ranks


def teacher_ranking_diagnostics(
    teacher: ScheduleTeacherMLP,
    rows: Sequence[Mapping[str, Any]],
    *,
    rewards: Mapping[Tuple[str, int], Mapping[str, float]],
    summary_grids: Mapping[Tuple[str, str, int], Sequence[float]],
    max_macro_steps: int,
    top_k: int = 5,
) -> Dict[str, Any]:
    examples: List[Tuple[Tuple[str, int], str, float, float]] = []
    for row in rows:
        solver = str(row["solver_key"])
        target_nfe = int(row["target_nfe"])
        schedule_key = str(row["scheduler_key"])
        grid = _grid_for_schedule(schedule_key, solver, target_nfe, summary_grids=summary_grids)
        features = teacher_features(solver, target_nfe, grid, max_macro_steps=max_macro_steps)
        with torch.no_grad():
            pred = float(teacher(features[None, :]).detach().cpu().item())
        target = float(rewards[(solver, target_nfe)][schedule_key])
        examples.append(((solver, target_nfe), schedule_key, target, pred))
    by_setting: Dict[Tuple[str, int], List[Tuple[str, float, float]]] = {}
    for setting, schedule_key, target, pred in examples:
        by_setting.setdefault(setting, []).append((schedule_key, float(target), float(pred)))
    per_cell: List[Dict[str, Any]] = []
    for setting, values in sorted(by_setting.items(), key=lambda item: item[0]):
        pair_total = 0
        pair_correct = 0
        positive_negative_total = 0
        positive_negative_correct = 0
        for idx, (_, target_i, pred_i) in enumerate(values):
            for _, target_j, pred_j in values[idx + 1 :]:
                diff = target_i - target_j
                if abs(diff) <= 1e-12:
                    continue
                pair_total += 1
                pred_diff = pred_i - pred_j
                if pred_diff == 0.0:
                    pair_correct += 0.5
                elif (diff > 0 and pred_diff > 0) or (diff < 0 and pred_diff < 0):
                    pair_correct += 1
        positives = [(target, pred) for _, target, pred in values if target > 0.0]
        negatives = [(target, pred) for _, target, pred in values if target <= 0.0]
        for _, pred_pos in positives:
            for _, pred_neg in negatives:
                positive_negative_total += 1
                if pred_pos == pred_neg:
                    positive_negative_correct += 0.5
                elif pred_pos > pred_neg:
                    positive_negative_correct += 1
        targets = [target for _, target, _ in values]
        preds = [pred for _, _, pred in values]
        spearman = None
        if len(values) >= 2 and len(set(targets)) > 1 and len(set(preds)) > 1:
            target_ranks = _rankdata_average(targets)
            pred_ranks = _rankdata_average(preds)
            corr = np.corrcoef(target_ranks, pred_ranks)[0, 1]
            spearman = float(corr) if np.isfinite(corr) else None
        k = min(int(top_k), len(values))
        true_top = {schedule_key for schedule_key, _, _ in sorted(values, key=lambda item: item[1], reverse=True)[:k]}
        pred_top = {schedule_key for schedule_key, _, _ in sorted(values, key=lambda item: item[2], reverse=True)[:k]}
        per_cell.append(
            {
                "solver_key": setting[0],
                "target_nfe": int(setting[1]),
                "candidate_count": int(len(values)),
                "pairwise_accuracy": None if pair_total == 0 else float(pair_correct / pair_total),
                "pair_count": int(pair_total),
                "spearman": spearman,
                "top_k": int(k),
                "top_k_recall": None if k == 0 else float(len(true_top.intersection(pred_top)) / k),
                "best_fixed_crossing_pair_accuracy": None
                if positive_negative_total == 0
                else float(positive_negative_correct / positive_negative_total),
                "best_fixed_crossing_pair_count": int(positive_negative_total),
            }
        )
    pair_acc_values = [float(row["pairwise_accuracy"]) for row in per_cell if row.get("pairwise_accuracy") is not None]
    spearman_values = [float(row["spearman"]) for row in per_cell if row.get("spearman") is not None]
    topk_values = [float(row["top_k_recall"]) for row in per_cell if row.get("top_k_recall") is not None]
    crossing_values = [float(row["best_fixed_crossing_pair_accuracy"]) for row in per_cell if row.get("best_fixed_crossing_pair_accuracy") is not None]
    return {
        "diagnostic_split": TRAIN_TUNING_PHASE,
        "diagnostic_rows": "held_out_train_tuning_when_available",
        "uses_validation_labels": False,
        "target_pairwise_accuracy": 0.65,
        "mean_pairwise_accuracy": None if not pair_acc_values else float(np.mean(np.asarray(pair_acc_values, dtype=np.float64))),
        "cells_meeting_pairwise_accuracy_target": int(sum(value >= 0.65 for value in pair_acc_values)),
        "mean_spearman": None if not spearman_values else float(np.mean(np.asarray(spearman_values, dtype=np.float64))),
        "positive_spearman_cells": int(sum(value > 0.0 for value in spearman_values)),
        "mean_top_k_recall": None if not topk_values else float(np.mean(np.asarray(topk_values, dtype=np.float64))),
        "mean_best_fixed_crossing_pair_accuracy": None
        if not crossing_values
        else float(np.mean(np.asarray(crossing_values, dtype=np.float64))),
        "per_cell": per_cell,
    }


def split_teacher_diagnostic_holdout(
    rows: Sequence[Mapping[str, Any]],
    *,
    fraction: float = DEFAULT_TEACHER_DIAGNOSTIC_HOLDOUT_FRACTION,
    seed: int = 0,
    fixed_schedule_keys: Sequence[str] = BASELINE_SCHEDULE_KEYS,
    rewards: Mapping[Tuple[str, int], Mapping[str, float]] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split BO train-tuning rows into fit and teacher-holdout subsets by cell.

    The diagnostic rows stay inside Train20 and never use validation labels, but
    they are held out from teacher fitting so ranking diagnostics select the
    teacher checkpoint. Fixed reference/anchor rows stay in the fit set; only BO
    candidate rows are eligible for the 20% holdout.
    """
    frac = float(fraction)
    if frac < 0.0 or frac >= 1.0:
        raise ValueError(f"teacher_diagnostic_holdout_fraction must be in [0, 1), got {fraction!r}.")
    fixed_keys = {str(key) for key in fixed_schedule_keys}
    by_setting: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        by_setting.setdefault((str(row["solver_key"]), int(row["target_nfe"])), []).append(dict(row))
    fit_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    for setting, setting_rows in sorted(by_setting.items(), key=lambda item: item[0]):
        ordered = sorted(setting_rows, key=lambda row: str(row["scheduler_key"]))
        fixed_rows = [row for row in ordered if str(row["scheduler_key"]) in fixed_keys]
        bo_rows = [row for row in ordered if str(row["scheduler_key"]) not in fixed_keys]
        fit_rows.extend(fixed_rows)
        if frac <= 0.0 or len(bo_rows) < 2:
            fit_rows.extend(bo_rows)
            continue
        holdout_count = int(math.ceil(len(bo_rows) * frac))
        holdout_count = max(1, min(holdout_count, len(bo_rows) - 1))
        stable_solver_hash = sum((idx + 1) * ord(ch) for idx, ch in enumerate(setting[0]))
        rng = np.random.default_rng(int(seed) + 1009 * (stable_solver_hash % 997) + int(setting[1]))
        reward_lookup = rewards or {}
        sorted_by_reward = sorted(
            bo_rows,
            key=lambda row: float(reward_lookup.get(setting, {}).get(str(row["scheduler_key"]), 0.0)),
        )
        reward_quantile_by_key: Dict[str, int] = {}
        for rank, row in enumerate(sorted_by_reward):
            quantile = min(3, int(4 * rank / max(1, len(sorted_by_reward))))
            reward_quantile_by_key[str(row["scheduler_key"])] = int(quantile)
        strata: Dict[Tuple[str, str, int], List[int]] = {}
        for idx, row in enumerate(bo_rows):
            source = str(row.get("candidate_source", "") or "unknown")
            active_round = str(row.get("active_round", "") or "unknown")
            quantile = reward_quantile_by_key.get(str(row["scheduler_key"]), 0)
            strata.setdefault((source, active_round, quantile), []).append(idx)
        holdout_indices: set[int] = set()
        for stratum_idx, indices_list in enumerate(strata.values()):
            if len(holdout_indices) >= holdout_count:
                break
            if not indices_list:
                continue
            local = np.asarray(indices_list, dtype=np.int64)
            rng.shuffle(local)
            local_count = int(math.floor(len(indices_list) * frac))
            if local_count <= 0 and len(indices_list) > 1:
                local_count = 1
            local_count = min(local_count, len(indices_list) - 1 if len(indices_list) > 1 else 1)
            for idx_value in local[:local_count]:
                if len(holdout_indices) < holdout_count:
                    holdout_indices.add(int(idx_value))
        if len(holdout_indices) < holdout_count:
            remaining = [idx for idx in range(len(bo_rows)) if idx not in holdout_indices]
            rng.shuffle(remaining)
            holdout_indices.update(int(idx) for idx in remaining[: holdout_count - len(holdout_indices)])
        for idx, row in enumerate(bo_rows):
            if idx in holdout_indices:
                diagnostic_rows.append(row)
            else:
                fit_rows.append(row)
    return fit_rows, diagnostic_rows


def _teacher_checkpoint_score(diagnostics: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    def value(name: str) -> float:
        raw = diagnostics.get(name)
        if raw is None:
            return float("-inf")
        val = float(raw)
        return val if math.isfinite(val) else float("-inf")

    return (
        value("mean_pairwise_accuracy"),
        value("mean_top_k_recall"),
        value("mean_spearman"),
        value("mean_best_fixed_crossing_pair_accuracy"),
    )


def train_teacher(
    teacher: ScheduleTeacherMLP,
    rows: Sequence[Mapping[str, Any]],
    *,
    rewards: Mapping[Tuple[str, int], Mapping[str, float]],
    summary_grids: Mapping[Tuple[str, str, int], Sequence[float]],
    max_macro_steps: int,
    steps: int,
    lr: float,
    schedule_weights: Mapping[str, float] | None = None,
    rank_temperature: float = DEFAULT_TEACHER_RANK_TEMPERATURE,
    regression_weight: float = DEFAULT_TEACHER_REGRESSION_WEIGHT,
    pairs_per_candidate: int = DEFAULT_TEACHER_PAIRS_PER_CANDIDATE,
    pair_margin: float = DEFAULT_TEACHER_PAIR_MARGIN,
    diagnostic_rows: Sequence[Mapping[str, Any]] = (),
    diagnostic_top_k: int = 5,
) -> Dict[str, Any]:
    teacher_x: List[torch.Tensor] = []
    teacher_y: List[float] = []
    setting_keys: List[Tuple[str, int]] = []
    teacher_w: List[float] = []
    weights = dict(schedule_weights or {})
    for row in rows:
        solver = str(row["solver_key"])
        target_nfe = int(row["target_nfe"])
        schedule_key = str(row["scheduler_key"])
        grid = _grid_for_schedule(schedule_key, solver, target_nfe, summary_grids=summary_grids)
        teacher_x.append(teacher_features(solver, target_nfe, grid, max_macro_steps=max_macro_steps))
        teacher_y.append(float(rewards[(solver, target_nfe)][schedule_key]))
        setting_keys.append((solver, int(target_nfe)))
        teacher_w.append(float(weights.get(schedule_key, 1.0)))
    if not teacher_x:
        raise ValueError("Teacher training requires at least one candidate row.")
    if float(rank_temperature) <= 0.0:
        raise ValueError("rank_temperature must be positive.")
    if float(regression_weight) < 0.0:
        raise ValueError("regression_weight must be nonnegative.")
    tx = torch.stack(teacher_x, dim=0)
    normalized_teacher_y, _normalizers = _normalized_targets_by_setting(teacher_y, setting_keys)
    ty = torch.tensor(normalized_teacher_y, dtype=torch.float32)
    tw = torch.tensor(teacher_w, dtype=torch.float32)
    if torch.any(~torch.isfinite(tw)) or torch.any(tw <= 0):
        raise ValueError("Teacher schedule weights must be finite and positive.")
    pair_left, pair_right, pair_sign = _teacher_pair_tensor(
        ty,
        setting_keys,
        margin=float(pair_margin),
        device=tx.device,
    )
    max_pairs_per_step = max(1, int(pairs_per_candidate)) * int(tx.shape[0])
    opt = torch.optim.AdamW(teacher.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: List[Dict[str, float]] = []
    checkpoint_history: List[Dict[str, Any]] = []
    best_state: Dict[str, torch.Tensor] | None = None
    best_step = 0
    best_score: Tuple[float, float, float, float] | None = None
    best_diagnostics: Dict[str, Any] | None = None
    for step in range(int(steps)):
        pred = teacher(tx)
        huber = F.smooth_l1_loss(pred, ty, reduction="none")
        regression_loss = torch.sum(tw * huber) / torch.sum(tw)
        if pair_left.numel() > 0:
            if pair_left.numel() > max_pairs_per_step:
                selected = torch.randperm(pair_left.numel(), device=tx.device)[:max_pairs_per_step]
                left = pair_left[selected]
                right = pair_right[selected]
                sign = pair_sign[selected]
            else:
                left = pair_left
                right = pair_right
                sign = pair_sign
            margin_logits = sign * (pred[left] - pred[right]) / float(rank_temperature)
            rank_loss = F.softplus(-margin_logits).mean()
            loss = rank_loss + float(regression_weight) * regression_loss
            pair_count = int(left.numel())
        else:
            rank_loss = pred.new_zeros(())
            loss = regression_loss
            pair_count = 0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or step == int(steps) - 1 or (step + 1) % max(1, int(steps) // 10) == 0:
            log_row = {
                "step": int(step + 1),
                "teacher_total_loss": float(loss.detach().cpu().item()),
                "teacher_rank_loss": float(rank_loss.detach().cpu().item()),
                "teacher_huber_loss": float(regression_loss.detach().cpu().item()),
                "teacher_pair_count": int(pair_count),
            }
            if diagnostic_rows:
                diagnostics = teacher_ranking_diagnostics(
                    teacher,
                    diagnostic_rows,
                    rewards=rewards,
                    summary_grids=summary_grids,
                    max_macro_steps=max_macro_steps,
                    top_k=int(diagnostic_top_k),
                )
                score = _teacher_checkpoint_score(diagnostics)
                checkpoint_row = {
                    "step": int(step + 1),
                    "selection_score": [float(x) for x in score],
                    "mean_pairwise_accuracy": diagnostics.get("mean_pairwise_accuracy"),
                    "mean_top_k_recall": diagnostics.get("mean_top_k_recall"),
                    "mean_spearman": diagnostics.get("mean_spearman"),
                    "mean_best_fixed_crossing_pair_accuracy": diagnostics.get("mean_best_fixed_crossing_pair_accuracy"),
                }
                checkpoint_history.append(checkpoint_row)
                log_row.update({f"holdout_{key}": value for key, value in checkpoint_row.items() if key != "step"})
                if best_score is None or score > best_score:
                    best_score = score
                    best_step = int(step + 1)
                    best_state = copy.deepcopy(teacher.state_dict())
                    best_diagnostics = diagnostics
            losses.append(log_row)
    if diagnostic_rows and best_state is not None:
        teacher.load_state_dict(best_state)
    if best_diagnostics is None:
        best_diagnostics = teacher_ranking_diagnostics(
            teacher,
            rows,
            rewards=rewards,
            summary_grids=summary_grids,
            max_macro_steps=max_macro_steps,
            top_k=int(diagnostic_top_k),
        )
        best_step = int(steps)
        best_score = _teacher_checkpoint_score(best_diagnostics)
    return {
        "losses": losses,
        "checkpoint_selection": {
            "selection_unit": "teacher_checkpoint",
            "selection_split": "teacher_holdout" if diagnostic_rows else "teacher_train_fallback",
            "selection_metric_order": [
                "mean_pairwise_accuracy",
                "mean_top_k_recall",
                "mean_spearman",
                "mean_best_fixed_crossing_pair_accuracy",
            ],
            "selected_step": int(best_step),
            "selected_score": [float(x) for x in (best_score or (float("-inf"),) * 4)],
            "history": checkpoint_history,
            "uses_validation_labels": False,
        },
        "selected_diagnostics": best_diagnostics,
    }


def fit_student_to_reference(
    student: ScheduleStudentMLP,
    reference_pairs: Sequence[Tuple[str, int, Sequence[float]]],
    *,
    max_macro_steps: int,
    steps: int,
    lr: float,
) -> List[Dict[str, float]]:
    targets = []
    for solver, target_nfe, grid in reference_pairs:
        macro_steps = solver_macro_steps(str(solver), int(target_nfe))
        target = torch.tensor(grid_to_intervals(validate_time_grid(grid, macro_steps=macro_steps)), dtype=torch.float32)
        targets.append((setting_features(str(solver), int(target_nfe)), target, int(macro_steps)))
    if not targets:
        raise ValueError("Student initialization requires SER-PTG reference targets.")
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: List[Dict[str, float]] = []
    for step in range(int(steps)):
        loss_terms = []
        for feat, target, macro_steps in targets:
            pred = student.intervals(feat[None, :], macro_steps)[0]
            loss_terms.append(F.mse_loss(pred, target))
        loss = torch.stack(loss_terms).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or step == int(steps) - 1 or (step + 1) % max(1, int(steps) // 5) == 0:
            losses.append({"step": int(step + 1), "ser_ptg_init_mse": float(loss.detach().cpu().item())})
    return losses


def _normalize_intervals_for_divergence(intervals: torch.Tensor, *, eps: float) -> torch.Tensor:
    eps_value = float(eps)
    if eps_value <= 0.0:
        raise ValueError(f"SER-PTG regularization eps must be positive, got {eps!r}.")
    clipped = torch.clamp(intervals, min=eps_value)
    return clipped / clipped.sum(dim=-1, keepdim=True)


def ser_ptg_interval_divergence(
    student_intervals: torch.Tensor,
    reference_intervals: torch.Tensor,
    *,
    mode: str,
    eps: float = DEFAULT_STUDENT_SER_PTG_REGULARIZATION_EPS,
) -> torch.Tensor:
    regularizer = str(mode).strip().lower()
    if regularizer not in STUDENT_SER_PTG_REGULARIZERS:
        raise ValueError(f"Unsupported SER-PTG regularizer {mode!r}; expected one of {STUDENT_SER_PTG_REGULARIZERS}.")
    if regularizer == STUDENT_SER_PTG_REGULARIZER_NONE:
        return student_intervals.new_zeros(())
    reference = reference_intervals.to(device=student_intervals.device, dtype=student_intervals.dtype)
    if reference.ndim == 1:
        reference = reference[None, :]
    if tuple(reference.shape) != tuple(student_intervals.shape):
        raise ValueError(
            f"SER-PTG reference interval shape {tuple(reference.shape)} does not match student interval shape {tuple(student_intervals.shape)}."
        )
    student_dist = _normalize_intervals_for_divergence(student_intervals, eps=float(eps))
    reference_dist = _normalize_intervals_for_divergence(reference, eps=float(eps))
    if regularizer == STUDENT_SER_PTG_REGULARIZER_KL:
        return torch.sum(student_dist * torch.log(student_dist / reference_dist), dim=-1).mean()
    mixture = 0.5 * (student_dist + reference_dist)
    student_kl = torch.sum(student_dist * torch.log(student_dist / mixture), dim=-1)
    reference_kl = torch.sum(reference_dist * torch.log(reference_dist / mixture), dim=-1)
    return (0.5 * (student_kl + reference_kl)).mean()


def optimize_student_with_teacher(
    student: ScheduleStudentMLP,
    teacher: ScheduleTeacherMLP,
    settings: Sequence[Tuple[str, int]],
    *,
    max_macro_steps: int,
    steps: int,
    lr: float,
    ser_ptg_reference_pairs: Sequence[Tuple[str, int, Sequence[float]]] = (),
    ser_ptg_regularizer: str = STUDENT_SER_PTG_REGULARIZER_NONE,
    ser_ptg_regularization_weight: float = 0.0,
    ser_ptg_regularization_eps: float = DEFAULT_STUDENT_SER_PTG_REGULARIZATION_EPS,
) -> List[Dict[str, float]]:
    regularizer = str(ser_ptg_regularizer).strip().lower()
    if regularizer not in STUDENT_SER_PTG_REGULARIZERS:
        raise ValueError(f"Unsupported SER-PTG regularizer {ser_ptg_regularizer!r}; expected one of {STUDENT_SER_PTG_REGULARIZERS}.")
    regularization_weight = float(ser_ptg_regularization_weight)
    if regularization_weight < 0.0:
        raise ValueError(f"SER-PTG regularization weight must be non-negative, got {ser_ptg_regularization_weight!r}.")
    regularization_eps = float(ser_ptg_regularization_eps)
    if regularization_eps <= 0.0:
        raise ValueError(f"SER-PTG regularization eps must be positive, got {ser_ptg_regularization_eps!r}.")
    use_regularizer = regularizer != STUDENT_SER_PTG_REGULARIZER_NONE and regularization_weight > 0.0
    reference_targets: Dict[Tuple[str, int], torch.Tensor] = {}
    if use_regularizer:
        for solver, target_nfe, grid in ser_ptg_reference_pairs:
            macro_steps = solver_macro_steps(str(solver), int(target_nfe))
            target = torch.tensor(grid_to_intervals(validate_time_grid(grid, macro_steps=macro_steps)), dtype=torch.float32)
            reference_targets[(str(solver), int(target_nfe))] = target
        missing = sorted({(str(solver), int(target_nfe)) for solver, target_nfe in settings} - set(reference_targets))
        if missing:
            raise ValueError(f"Missing SER-PTG reference targets for student regularization: {missing[:12]}")
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: List[Dict[str, float]] = []
    for step in range(int(steps)):
        utility_terms = []
        regularization_terms = []
        for solver, target_nfe in settings:
            macro_steps = solver_macro_steps(str(solver), int(target_nfe))
            feat = setting_features(str(solver), int(target_nfe))[None, :]
            intervals = student.intervals(feat, macro_steps)
            padded = torch.zeros((1, int(max_macro_steps)), dtype=intervals.dtype, device=intervals.device)
            padded[:, :macro_steps] = intervals
            teacher_input = differentiable_teacher_features(feat, padded, max_macro_steps=max_macro_steps)
            utility_terms.append(teacher(teacher_input).mean())
            if use_regularizer:
                reference = reference_targets[(str(solver), int(target_nfe))]
                regularization_terms.append(
                    ser_ptg_interval_divergence(
                        intervals,
                        reference,
                        mode=regularizer,
                        eps=regularization_eps,
                    )
                )
        objective = torch.stack(utility_terms).mean()
        regularization = (
            torch.stack(regularization_terms).mean()
            if regularization_terms
            else objective.new_zeros(())
        )
        loss = -objective + regularization_weight * regularization
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or step == int(steps) - 1 or (step + 1) % max(1, int(steps) // 5) == 0:
            row = {
                "step": int(step + 1),
                "teacher_predicted_utility": float(objective.detach().cpu().item()),
            }
            if use_regularizer:
                row.update(
                    {
                        "ser_ptg_regularization": float(regularization.detach().cpu().item()),
                        "ser_ptg_regularizer": regularizer,
                        "ser_ptg_regularization_weight": float(regularization_weight),
                        "student_total_loss": float(loss.detach().cpu().item()),
                    }
                )
            losses.append(row)
    return losses


def _student_predictions(
    student: ScheduleStudentMLP,
    *,
    solvers: Sequence[str],
    target_nfes: Sequence[int],
    max_macro_steps: int,
    teacher: ScheduleTeacherMLP | None = None,
    active_round: int | None = None,
    student_seed: int | None = None,
    opd_steps: int | None = None,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for solver in solvers:
        for target_nfe in target_nfes:
            macro_steps = solver_macro_steps(str(solver), int(target_nfe))
            with torch.no_grad():
                intervals_t = student.intervals(setting_features(str(solver), int(target_nfe))[None, :], macro_steps)[0]
            intervals = np.asarray([float(x) for x in intervals_t.detach().cpu().tolist()], dtype=np.float64)
            intervals = np.maximum(intervals, 1e-7)
            intervals = intervals / float(np.sum(intervals))
            grid = np.concatenate([[0.0], np.cumsum(intervals)])
            grid[0] = 0.0
            grid[-1] = 1.0
            values = validate_time_grid([float(x) for x in grid.tolist()], macro_steps=macro_steps)
            utility = None
            if teacher is not None:
                padded = torch.zeros((1, int(max_macro_steps)), dtype=torch.float32)
                padded[:, :macro_steps] = torch.tensor(np.diff(np.asarray(values, dtype=np.float64)), dtype=torch.float32)[None, :]
                with torch.no_grad():
                    utility = float(
                        teacher(
                            differentiable_teacher_features(
                                setting_features(str(solver), int(target_nfe))[None, :],
                                padded,
                                max_macro_steps=max_macro_steps,
                            )
                        )
                        .detach()
                        .cpu()
                        .item()
                    )
            geom = grid_geometry(values)
            predictions.append(
                {
                    "solver_key": str(solver),
                    "target_nfe": int(target_nfe),
                    "runtime_nfe": int(macro_steps),
                    "macro_steps": int(macro_steps),
                    "realized_nfe": int(target_nfe),
                    "time_grid": list(values),
                    "grid_geometry": geom,
                    "max_macro_steps": int(max_macro_steps),
                    "candidate_source": "direct_student",
                    "active_round": "" if active_round is None else int(active_round),
                    "student_seed": "" if student_seed is None else int(student_seed),
                    "opd_steps": "" if opd_steps is None else int(opd_steps),
                    "perturbation_type": "none",
                    "perturbation_params_json": "{}",
                    "intervals_json": json.dumps([float(x) for x in np.diff(np.asarray(values, dtype=np.float64)).tolist()], separators=(",", ":")),
                    "utility": "" if utility is None else float(utility),
                    "validity_flags_json": json.dumps({"finite": True, "monotone": True, "exact_realized_nfe": True}, separators=(",", ":")),
                }
            )
    return predictions


def _candidate_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    rewards: Mapping[Tuple[str, int], Mapping[str, float]],
    teacher_fixed_schedule_keys: Sequence[str] = DEFAULT_TEACHER_FIXED_SCHEDULE_KEYS,
) -> List[Dict[str, Any]]:
    table: List[Dict[str, Any]] = []
    teacher_anchor_keys = {str(key) for key in teacher_fixed_schedule_keys}
    for row in rows:
        solver = str(row["solver_key"])
        target_nfe = int(row["target_nfe"])
        schedule_key = str(row["scheduler_key"])
        table.append(
            {
                "solver_key": solver,
                "target_nfe": int(target_nfe),
                "scheduler_key": schedule_key,
                "crps_seed_mean": float(row["crps"]),
                "mase_seed_mean": float(row["mase"]),
                "crps_seed_std": float(row.get("crps_std", 0.0)),
                "mase_seed_std": float(row.get("mase_std", 0.0)),
                "n_seeds": int(row.get("n_seeds", 0)),
                "seed_values": list(row.get("seed_values", []) or []),
                "teacher_reward": float(rewards[(solver, target_nfe)][schedule_key]),
                "best_fixed_crps": row.get("best_fixed_crps"),
                "best_fixed_mase": row.get("best_fixed_mase"),
                "uniform_crps": row.get("uniform_crps"),
                "uniform_mase": row.get("uniform_mase"),
                "u_crps_best": row.get("u_crps_best"),
                "u_mase_best": row.get("u_mase_best"),
                "u_comp_best": row.get("u_comp_best"),
                "u_comp_uniform": row.get("u_comp_uniform"),
                "is_fixed_reference": schedule_key in BASELINE_SCHEDULE_KEYS,
                "is_teacher_anchor": schedule_key in teacher_anchor_keys,
                "candidate_source": row.get("candidate_source", ""),
                "active_round": row.get("active_round", ""),
                "opd_step_budget": row.get("opd_step_budget", ""),
            }
        )
    table.sort(key=lambda item: (item["solver_key"], item["target_nfe"], item["scheduler_key"]))
    return table


def teacher_schedule_weights(
    schedule_keys: Sequence[str],
    *,
    late_biased_demo_schedules: Sequence[str] = DEFAULT_LATE_BIASED_DEMO_SCHEDULES,
    late_biased_demo_weight: float = DEFAULT_LATE_BIASED_DEMO_WEIGHT,
) -> Dict[str, float]:
    demoted = {str(key) for key in late_biased_demo_schedules}
    weight = float(late_biased_demo_weight)
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"late_biased_demo_weight must be finite and positive, got {late_biased_demo_weight!r}.")
    return {str(key): (weight if str(key) in demoted else 1.0) for key in sorted({str(x) for x in schedule_keys})}


def train_conditional_opd(args: argparse.Namespace) -> Dict[str, Any]:
    seed_all(int(args.seed))
    solvers = tuple(_parse_csv(args.solver_names))
    target_nfes = tuple(_parse_int_csv(args.target_nfe_values))
    seeds = tuple(_parse_int_csv(args.seeds))
    budgets = tuple(_parse_int_csv(args.student_opd_step_values))
    required_split_phase = str(args.required_split_phase)
    if required_split_phase != TRAIN_TUNING_PHASE:
        raise ValueError(f"Conditional OPD teacher rows must use split_phase={TRAIN_TUNING_PHASE!r}.")
    teacher_fixed_schedule_keys = _parse_teacher_fixed_schedule_keys(str(getattr(args, "teacher_fixed_schedule_keys", ",".join(DEFAULT_TEACHER_FIXED_SCHEDULE_KEYS))))
    reward_reference_schedule_keys = _parse_reward_reference_schedule_keys(
        str(getattr(args, "reward_reference_schedule_keys", "")),
        teacher_fixed_schedule_keys=teacher_fixed_schedule_keys,
    )
    if str(args.reference_rows_csv).strip():
        raise ValueError("SER-PTG metric rows are not used for teacher training; pass only --reference_schedule_summary for student initialization.")
    late_biased_demo_schedules = tuple(_parse_csv(getattr(args, "late_biased_demo_schedules", "")))
    late_biased_demo_weight = float(getattr(args, "late_biased_demo_weight", DEFAULT_LATE_BIASED_DEMO_WEIGHT))
    include_diagnostic_budgets = bool(getattr(args, "include_diagnostic_budgets", False))
    student_ser_ptg_regularizer = str(getattr(args, "student_ser_ptg_regularizer", STUDENT_SER_PTG_REGULARIZER_NONE)).strip().lower()
    student_ser_ptg_regularization_weight = float(getattr(args, "student_ser_ptg_regularization_weight", 0.0))
    student_ser_ptg_regularization_eps = float(
        getattr(args, "student_ser_ptg_regularization_eps", DEFAULT_STUDENT_SER_PTG_REGULARIZATION_EPS)
    )
    uses_ser_ptg_regularizer = (
        student_ser_ptg_regularizer != STUDENT_SER_PTG_REGULARIZER_NONE and student_ser_ptg_regularization_weight > 0.0
    )
    diagnostic_requested = sorted(set(int(x) for x in budgets).intersection(DIAGNOSTIC_DIRECT_OPD_BUDGETS))
    if diagnostic_requested and not include_diagnostic_budgets:
        raise ValueError(
            "Direct OPD budgets 35 and 50 are diagnostic-only; pass --include_diagnostic_budgets "
            f"to run {','.join(str(x) for x in diagnostic_requested)}."
        )
    if include_diagnostic_budgets:
        budgets = tuple(sorted(set(int(x) for x in budgets).union(DIAGNOSTIC_DIRECT_OPD_BUDGETS)))
    candidate_rows_paths = tuple(_parse_csv(getattr(args, "candidate_rows_csv", "")))
    candidate_summary_paths = tuple(_parse_csv(getattr(args, "candidate_schedule_summary", "")))
    explicit_candidate_keys = tuple(_parse_csv(getattr(args, "candidate_schedule_keys", "")))
    max_macro_steps = max(solver_macro_steps(solver, nfe) for solver in solvers for nfe in target_nfes)
    setting_dim = int(setting_features("euler", 4).numel())
    teacher_input_dim = setting_dim + int(max_macro_steps)
    teacher = ScheduleTeacherMLP(teacher_input_dim, hidden_dim=256, hidden_layers=3)
    student = ScheduleStudentMLP(setting_dim, max_macro_steps=max_macro_steps, hidden_dim=128, hidden_layers=2)
    teacher_param_count = count_parameters(teacher)
    student_param_count = count_parameters(student)
    if student_param_count >= teacher_param_count:
        raise RuntimeError("Student MLP must be smaller than teacher MLP.")

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.dry_run):
        summary = {
            "status": "dry_run",
            "dataset": str(args.dataset),
            "teacher_parameters": teacher_param_count,
            "student_parameters": student_param_count,
            "solvers": list(solvers),
            "target_nfes": list(target_nfes),
            "student_opd_step_values": [int(x) for x in budgets],
            "schedule_key_prefix": str(getattr(args, "schedule_key_prefix", STUDENT_SCHEDULE_PREFIX)),
            "active_round": int(getattr(args, "active_round", -1)),
            "teacher_fixed_schedule_keys": list(teacher_fixed_schedule_keys),
            "reward_reference_schedule_keys": list(reward_reference_schedule_keys),
            "final_baseline_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
            "reference_schedule_key": SER_PTG_SCHEDULE_KEY,
            "ser_ptg_metric_rows_used": False,
            "evaluated_candidate_schedule_keys": list(explicit_candidate_keys),
            "candidate_rows_csv": [str(path) for path in candidate_rows_paths],
            "candidate_schedule_summary": [str(path) for path in candidate_summary_paths],
            "teacher_metric_split": required_split_phase,
            "teacher_demo_weighting": "behavior_balanced",
            "late_biased_demo_schedules": list(late_biased_demo_schedules),
            "late_biased_demo_weight": float(late_biased_demo_weight),
            "teacher_objective": "ranking_first_best_fixed_composite_with_huber_calibration",
            "teacher_selection_protocol": "pooled_bo_holdout_teacher_checkpoint",
            "teacher_loss_config": {
                "rank_temperature": float(args.teacher_rank_temperature),
                "regression_weight": float(args.teacher_regression_weight),
                "pairs_per_candidate": int(args.teacher_pairs_per_candidate),
                "pair_margin": float(args.teacher_pair_margin),
                "diagnostic_holdout_fraction": float(args.teacher_diagnostic_holdout_fraction),
                "holdout_source": "bo_candidate_rows_only",
                "target_normalization": "median_mad_by_solver_nfe",
            },
            "uses_soft_regularizers": bool(uses_ser_ptg_regularizer),
            "student_objective": "maximize_teacher_predicted_utility_with_optional_ser_ptg_regularization",
            "student_ser_ptg_regularization": {
                "mode": student_ser_ptg_regularizer,
                "weight": float(student_ser_ptg_regularization_weight),
                "eps": float(student_ser_ptg_regularization_eps),
                "reference_schedule_key": SER_PTG_SCHEDULE_KEY,
            },
            "expected_train_tuning_sampler": str(args.expected_train_tuning_sampler),
            "expected_train_tuning_fraction": str(args.expected_train_tuning_fraction),
        }
        save_json(summary, str(out_dir / "conditional_opd_summary.json"))
        return summary

    reward_reference_rows = _clean_split_rows(
        _load_csv_rows(args.rows_csv),
        dataset=str(args.dataset),
        solvers=solvers,
        target_nfes=target_nfes,
        allowed_schedules=reward_reference_schedule_keys,
        seeds=seeds,
        required_split_phase=required_split_phase,
    )
    if not reward_reference_rows:
        raise ValueError(f"No usable baseline CRPS/MASE rows for dataset={args.dataset} in {args.rows_csv}.")
    _assert_expected_train_tuning_metadata(
        reward_reference_rows,
        expected_sampler=str(args.expected_train_tuning_sampler),
        expected_fraction=str(args.expected_train_tuning_fraction),
        label="reward reference train-tuning",
    )
    reference_rows: List[Dict[str, Any]] = []
    _assert_complete_seed_rows(
        reward_reference_rows,
        solvers=solvers,
        target_nfes=target_nfes,
        schedules=reward_reference_schedule_keys,
        seeds=seeds,
        label="reward reference train-tuning",
    )
    baseline_rows = [row for row in reward_reference_rows if str(row["scheduler_key"]) in set(teacher_fixed_schedule_keys)]
    _assert_complete_seed_rows(
        baseline_rows,
        solvers=solvers,
        target_nfes=target_nfes,
        schedules=teacher_fixed_schedule_keys,
        seeds=seeds,
        label="teacher fixed demo train-tuning",
    )
    summary_grids = _load_schedule_summary_grids(args.reference_schedule_summary)
    candidate_summary_keys = _schedule_keys_from_summary_paths(candidate_summary_paths) if candidate_summary_paths else []
    if explicit_candidate_keys:
        missing_keys = sorted(set(explicit_candidate_keys) - set(candidate_summary_keys))
        if missing_keys:
            raise ValueError(f"Explicit candidate schedule keys are not present in candidate summaries: {missing_keys[:12]}")
        evaluated_candidate_schedule_keys = sorted(explicit_candidate_keys)
    else:
        evaluated_candidate_schedule_keys = list(candidate_summary_keys)
    candidate_rows: List[Dict[str, Any]] = []
    if candidate_summary_paths and not candidate_rows_paths:
        raise ValueError("candidate_schedule_summary must be paired with evaluated candidate_rows_csv.")
    if candidate_rows_paths and not evaluated_candidate_schedule_keys:
        raise ValueError("candidate_rows_csv requires candidate_schedule_summary or candidate_schedule_keys.")
    for path in candidate_rows_paths:
        candidate_rows.extend(
            _clean_split_rows(
                _load_csv_rows(path),
                dataset=str(args.dataset),
                solvers=solvers,
                target_nfes=target_nfes,
                allowed_schedules=evaluated_candidate_schedule_keys,
                seeds=seeds,
                required_split_phase=required_split_phase,
            )
        )
    if candidate_rows_paths and not candidate_rows:
        raise ValueError("No evaluated candidate rows survived filtering; check split_phase, dataset, solver/NFE, and candidate schedule keys.")
    if candidate_rows:
        _assert_expected_train_tuning_metadata(
            candidate_rows,
            expected_sampler=str(args.expected_train_tuning_sampler),
            expected_fraction=str(args.expected_train_tuning_fraction),
            label="evaluated generated candidate train-tuning",
        )
    if candidate_rows and not bool(args.allow_incomplete_candidate_rows):
        _assert_complete_seed_rows(
            candidate_rows,
            solvers=solvers,
            target_nfes=target_nfes,
            schedules=evaluated_candidate_schedule_keys,
            seeds=seeds,
            label="evaluated generated candidate train-tuning",
        )
    summary_grids.update(_load_schedule_summary_grids_many(candidate_summary_paths))
    all_rows = reward_reference_rows + candidate_rows
    aggregate_rows = seed_mean_metric_rows(all_rows)
    annotated_aggregate_rows = attach_reward_columns(aggregate_rows, fixed_schedule_keys=reward_reference_schedule_keys)
    reward_reference_table = build_fixed_reference_table(reward_reference_rows, fixed_schedule_keys=reward_reference_schedule_keys)
    teacher_training_schedule_keys = set(str(key) for key in teacher_fixed_schedule_keys).union(evaluated_candidate_schedule_keys)
    teacher_rows = [row for row in annotated_aggregate_rows if str(row["scheduler_key"]) in teacher_training_schedule_keys]
    teacher_candidate_schedule_keys = sorted({str(row["scheduler_key"]) for row in teacher_rows})
    required = {(str(solver), int(nfe), key) for solver in solvers for nfe in target_nfes for key in teacher_fixed_schedule_keys}
    observed = {(str(row["solver_key"]), int(row["target_nfe"]), str(row["scheduler_key"])) for row in teacher_rows}
    missing_baselines = sorted(required - observed, key=lambda item: (item[0], item[1], item[2]))
    if missing_baselines:
        raise ValueError(f"Teacher fixed train-tuning rows are incomplete: {missing_baselines[:12]}")
    rewards = rewards_by_setting(annotated_aggregate_rows, fixed_schedule_keys=reward_reference_schedule_keys)
    teacher_fit_rows, teacher_diagnostic_rows = split_teacher_diagnostic_holdout(
        teacher_rows,
        fraction=float(args.teacher_diagnostic_holdout_fraction),
        seed=int(args.seed),
        fixed_schedule_keys=reward_reference_schedule_keys,
        rewards=rewards,
    )
    if not teacher_fit_rows:
        raise ValueError("Teacher fitting requires at least one train-tuning row after diagnostic holdout.")
    schedule_weights = teacher_schedule_weights(
        teacher_candidate_schedule_keys,
        late_biased_demo_schedules=late_biased_demo_schedules,
        late_biased_demo_weight=late_biased_demo_weight,
    )
    teacher_training = train_teacher(
        teacher,
        teacher_fit_rows,
        rewards=rewards,
        summary_grids=summary_grids,
        max_macro_steps=max_macro_steps,
        steps=int(args.teacher_steps),
        lr=float(args.lr),
        schedule_weights=schedule_weights,
        rank_temperature=float(args.teacher_rank_temperature),
        regression_weight=float(args.teacher_regression_weight),
        pairs_per_candidate=int(args.teacher_pairs_per_candidate),
        pair_margin=float(args.teacher_pair_margin),
        diagnostic_rows=teacher_diagnostic_rows,
        diagnostic_top_k=int(args.teacher_diagnostic_top_k),
    )
    teacher_losses = list(teacher_training["losses"])
    teacher_checkpoint_selection = dict(teacher_training["checkpoint_selection"])
    teacher_diagnostics = dict(teacher_training["selected_diagnostics"])
    teacher_diagnostics.update(
        {
            "fit_row_count": int(len(teacher_fit_rows)),
            "heldout_row_count": int(len(teacher_diagnostic_rows)),
            "diagnostic_row_count": int(len(teacher_diagnostic_rows if teacher_diagnostic_rows else teacher_fit_rows)),
            "diagnostic_rows": "held_out_train_tuning" if teacher_diagnostic_rows else "fit_rows_fallback_no_holdout_available",
            "holdout_fraction": float(args.teacher_diagnostic_holdout_fraction),
            "teacher_holdout_source": "bo_candidate_rows_only",
        }
    )
    settings = [(str(solver), int(target_nfe)) for solver in solvers for target_nfe in target_nfes]
    ser_ptg_targets = [
        (solver, target_nfe, _grid_for_schedule(SER_PTG_SCHEDULE_KEY, solver, target_nfe, summary_grids=summary_grids))
        for solver, target_nfe in settings
    ]
    init_losses = fit_student_to_reference(
        student,
        ser_ptg_targets,
        max_macro_steps=max_macro_steps,
        steps=int(args.student_init_steps),
        lr=float(args.lr),
    )
    base_student_state = copy.deepcopy(student.state_dict())
    budget_schedules: List[Dict[str, Any]] = []
    budget_losses: Dict[str, List[Dict[str, float]]] = {}
    budget_states: Dict[str, Any] = {}
    for budget in budgets:
        budget_student = ScheduleStudentMLP(setting_dim, max_macro_steps=max_macro_steps, hidden_dim=128, hidden_layers=2)
        budget_student.load_state_dict(base_student_state)
        losses = optimize_student_with_teacher(
            budget_student,
            teacher,
            settings,
            max_macro_steps=max_macro_steps,
            steps=int(budget),
            lr=float(args.lr),
            ser_ptg_reference_pairs=ser_ptg_targets,
            ser_ptg_regularizer=student_ser_ptg_regularizer,
            ser_ptg_regularization_weight=student_ser_ptg_regularization_weight,
            ser_ptg_regularization_eps=student_ser_ptg_regularization_eps,
        )
        predictions = _student_predictions(
            budget_student,
            solvers=solvers,
            target_nfes=target_nfes,
            max_macro_steps=max_macro_steps,
            teacher=teacher,
            active_round=getattr(args, "active_round", None),
            student_seed=int(args.seed),
            opd_steps=int(budget),
        )
        for item in predictions:
            item["opd_step_budget"] = int(budget)
        prefix = str(getattr(args, "schedule_key_prefix", STUDENT_SCHEDULE_PREFIX)).strip() or STUDENT_SCHEDULE_PREFIX
        schedule_key = f"{prefix}{int(budget)}"
        schedule_utility_values = [float(item["utility"]) for item in predictions if item.get("utility") not in (None, "")]
        schedule_record = {
            "scheduler_key": schedule_key,
            "schedule_name": f"Conditional OPD Student {int(budget)} updates",
            "comparison_role": "learned_student_budget_sweep",
            "opd_step_budget": int(budget),
            "initialized_from": SER_PTG_SCHEDULE_KEY,
            "student_objective": "maximize_teacher_predicted_utility_with_optional_ser_ptg_regularization",
            "student_ser_ptg_regularization": {
                "mode": student_ser_ptg_regularizer,
                "weight": float(student_ser_ptg_regularization_weight),
                "eps": float(student_ser_ptg_regularization_eps),
                "reference_schedule_key": SER_PTG_SCHEDULE_KEY,
            },
            "candidate_source": "direct_student",
            "active_round": getattr(args, "active_round", ""),
            "student_seed": int(args.seed),
            "opd_steps": int(budget),
            "perturbation_type": "none",
            "perturbation_params_json": "{}",
            "validity_flags_json": json.dumps({"finite": True, "monotone": True, "exact_realized_nfe": True}, separators=(",", ":")),
            "teacher_predicted_utility_mean": None if not schedule_utility_values else float(np.mean(np.asarray(schedule_utility_values, dtype=np.float64))),
            "predictions": predictions,
        }
        budget_schedules.append(
            schedule_record
        )
        budget_losses[str(int(budget))] = losses
        budget_states[str(int(budget))] = copy.deepcopy(budget_student.state_dict())

    schedule_summary = {
        "status": "ready",
        "artifact": "student_budget_schedule_summary",
        "dataset": str(args.dataset),
        "baseline_schedule": False,
        "reference_schedule_key": SER_PTG_SCHEDULE_KEY,
        "student_objective": "maximize_teacher_predicted_utility_with_optional_ser_ptg_regularization",
        "uses_soft_regularizers": bool(uses_ser_ptg_regularizer),
        "student_ser_ptg_regularization": {
            "mode": student_ser_ptg_regularizer,
            "weight": float(student_ser_ptg_regularization_weight),
            "eps": float(student_ser_ptg_regularization_eps),
            "reference_schedule_key": SER_PTG_SCHEDULE_KEY,
        },
        "student_opd_step_values": [int(x) for x in budgets],
        "schedules": budget_schedules,
    }
    save_json(schedule_summary, str(out_dir / "student_budget_schedule_summary.json"))
    torch.save(
        {
            "teacher_state": teacher.state_dict(),
            "base_student_state": base_student_state,
            "budget_student_states": budget_states,
            "setting_dim": int(setting_dim),
            "teacher_input_dim": int(teacher_input_dim),
            "max_macro_steps": int(max_macro_steps),
        },
        out_dir / "conditional_opd.pt",
    )
    summary = {
        "status": "ready",
        "dataset": str(args.dataset),
        "rows_csv": str(resolve_project_path(args.rows_csv)),
        "reference_rows_csv": str(resolve_project_path(args.reference_rows_csv)) if str(args.reference_rows_csv).strip() else "",
        "reference_schedule_summary": str(resolve_project_path(args.reference_schedule_summary)) if str(args.reference_schedule_summary).strip() else "",
        "teacher_parameters": teacher_param_count,
        "student_parameters": student_param_count,
        "teacher_architecture": {"hidden_layers": 3, "hidden_width": 256},
        "student_architecture": {"hidden_layers": 2, "hidden_width": 128},
        "teacher_fixed_schedule_keys": list(teacher_fixed_schedule_keys),
        "reward_reference_schedule_keys": list(reward_reference_schedule_keys),
        "final_baseline_schedule_keys": list(BASELINE_SCHEDULE_KEYS),
        "reference_schedule_key": SER_PTG_SCHEDULE_KEY,
        "ser_ptg_metric_rows_used": False,
        "candidate_schedule_keys": teacher_candidate_schedule_keys,
        "evaluated_candidate_schedule_keys": evaluated_candidate_schedule_keys,
        "candidate_rows_csv": [str(resolve_project_path(path)) for path in candidate_rows_paths],
        "candidate_schedule_summary": [str(resolve_project_path(path)) for path in candidate_summary_paths],
        "teacher_supervision_sources": {
            "teacher_metric_split": required_split_phase,
            "baseline_seed_rows": int(len(baseline_rows)),
            "reward_reference_seed_rows": int(len(reward_reference_rows)),
            "ser_ptg_metric_seed_rows": int(len(reference_rows)),
            "ser_ptg_metric_rows_used": False,
            "evaluated_candidate_seed_rows": int(len(candidate_rows)),
            "aggregate_rows": int(len(aggregate_rows)),
            "teacher_training_rows": int(len(teacher_rows)),
            "teacher_fit_rows": int(len(teacher_fit_rows)),
            "teacher_diagnostic_holdout_rows": int(len(teacher_diagnostic_rows)),
            "teacher_holdout_source": "bo_candidate_rows_only",
            "evaluated_candidate_schedule_count": int(len(evaluated_candidate_schedule_keys)),
        },
        "reward_reference_train_tuning": [
            dict(value) for _, value in sorted(reward_reference_table.items(), key=lambda item: item[0])
        ],
        "teacher_demo_weighting": {
            "mode": "behavior_balanced",
            "late_biased_demo_schedules": list(late_biased_demo_schedules),
            "late_biased_demo_weight": float(late_biased_demo_weight),
            "schedule_weights": schedule_weights,
        },
        "teacher_objective": "ranking_first_best_fixed_composite_with_huber_calibration",
        "teacher_loss_config": {
            "rank_temperature": float(args.teacher_rank_temperature),
            "regression_weight": float(args.teacher_regression_weight),
            "pairs_per_candidate": int(args.teacher_pairs_per_candidate),
            "pair_margin": float(args.teacher_pair_margin),
            "diagnostic_holdout_fraction": float(args.teacher_diagnostic_holdout_fraction),
            "holdout_source": "bo_candidate_rows_only",
            "holdout_stratification": "solver_nfe_candidate_source_active_round_reward_quantile_when_available",
            "target_normalization": "median_mad_by_solver_nfe",
        },
        "ser_ptg_is_section_15_baseline": SER_PTG_SCHEDULE_KEY in BASELINE_SCHEDULE_KEYS,
        "uses_soft_regularizers": bool(uses_ser_ptg_regularizer),
        "student_ser_ptg_regularization": {
            "mode": student_ser_ptg_regularizer,
            "weight": float(student_ser_ptg_regularization_weight),
            "eps": float(student_ser_ptg_regularization_eps),
            "reference_schedule_key": SER_PTG_SCHEDULE_KEY,
        },
        "metric_reward": "paired_best_fixed_equal_weight_negative_log_crps_mase",
        "seed_aggregation": "mean_by_solver_target_nfe_schedule_before_reward",
        "student_objective": "maximize_teacher_predicted_utility_with_optional_ser_ptg_regularization",
        "student_initialization": SER_PTG_SCHEDULE_KEY,
        "student_opd_step_values": [int(x) for x in budgets],
        "schedule_key_prefix": str(getattr(args, "schedule_key_prefix", STUDENT_SCHEDULE_PREFIX)),
        "active_round": int(getattr(args, "active_round", -1)),
        "candidate_table": _candidate_table(
            annotated_aggregate_rows,
            rewards=rewards,
            teacher_fixed_schedule_keys=teacher_fixed_schedule_keys,
        ),
        "teacher_selection_protocol": "pooled_bo_holdout_teacher_checkpoint",
        "teacher_checkpoint_selection": teacher_checkpoint_selection,
        "teacher_losses": teacher_losses,
        "teacher_diagnostics": teacher_diagnostics,
        "student_initialization_losses": init_losses,
        "student_opd_losses": budget_losses,
        "student_budget_schedule_summary": str(out_dir / "student_budget_schedule_summary.json"),
        "predictions": budget_schedules[-1]["predictions"] if budget_schedules else [],
    }
    save_json(summary["reward_reference_train_tuning"], str(out_dir / "reward_reference_train.json"))
    save_json(teacher_diagnostics, str(out_dir / "teacher_diagnostics.json"))
    save_json(summary, str(out_dir / "conditional_opd_summary.json"))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the genODE one-teacher conditional schedule policy.")
    parser.add_argument("--dataset", default="san_francisco_traffic")
    parser.add_argument("--teacher", default="mlp", choices=("mlp",))
    parser.add_argument("--student", default="mlp_small", choices=("mlp_small",))
    parser.add_argument("--solver_names", default=",".join(DEFAULT_SOLVERS))
    parser.add_argument("--target_nfe_values", default=",".join(str(x) for x in DEFAULT_TARGET_NFES))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--rows_csv", default=str(DEFAULT_NEURAL_POLICY_ROOT / "baseline_train_tuning" / "rows.csv"))
    parser.add_argument("--reference_rows_csv", default="")
    parser.add_argument("--reference_schedule_summary", default="")
    parser.add_argument("--teacher_fixed_schedule_keys", default=",".join(DEFAULT_TEACHER_FIXED_SCHEDULE_KEYS))
    parser.add_argument("--reward_reference_schedule_keys", default="")
    parser.add_argument("--candidate_rows_csv", default="")
    parser.add_argument("--candidate_schedule_summary", default="")
    parser.add_argument("--candidate_schedule_keys", default="")
    parser.add_argument("--allow_incomplete_candidate_rows", action="store_true", default=False)
    parser.add_argument("--required_split_phase", default=TRAIN_TUNING_PHASE)
    parser.add_argument("--expected_train_tuning_sampler", default="")
    parser.add_argument("--expected_train_tuning_fraction", default="")
    parser.add_argument("--out_dir", default=str(DEFAULT_NEURAL_POLICY_ROOT))
    parser.add_argument("--teacher_steps", type=int, default=500)
    parser.add_argument("--teacher_rank_temperature", type=float, default=DEFAULT_TEACHER_RANK_TEMPERATURE)
    parser.add_argument("--teacher_regression_weight", type=float, default=DEFAULT_TEACHER_REGRESSION_WEIGHT)
    parser.add_argument("--teacher_pairs_per_candidate", type=int, default=DEFAULT_TEACHER_PAIRS_PER_CANDIDATE)
    parser.add_argument("--teacher_pair_margin", type=float, default=DEFAULT_TEACHER_PAIR_MARGIN)
    parser.add_argument("--teacher_diagnostic_top_k", type=int, default=5)
    parser.add_argument("--teacher_diagnostic_holdout_fraction", type=float, default=DEFAULT_TEACHER_DIAGNOSTIC_HOLDOUT_FRACTION)
    parser.add_argument("--student_init_steps", type=int, default=500)
    parser.add_argument("--student_opd_step_values", default=",".join(str(x) for x in DEFAULT_DIRECT_OPD_BUDGETS))
    parser.add_argument("--include_diagnostic_budgets", action="store_true", default=False)
    parser.add_argument("--schedule_key_prefix", default=STUDENT_SCHEDULE_PREFIX)
    parser.add_argument("--active_round", type=int, default=-1)
    parser.add_argument("--late_biased_demo_schedules", default=",".join(DEFAULT_LATE_BIASED_DEMO_SCHEDULES))
    parser.add_argument("--late_biased_demo_weight", type=float, default=DEFAULT_LATE_BIASED_DEMO_WEIGHT)
    parser.add_argument("--student_ser_ptg_regularizer", choices=STUDENT_SER_PTG_REGULARIZERS, default=STUDENT_SER_PTG_REGULARIZER_NONE)
    parser.add_argument("--student_ser_ptg_regularization_weight", type=float, default=0.0)
    parser.add_argument("--student_ser_ptg_regularization_eps", type=float, default=DEFAULT_STUDENT_SER_PTG_REGULARIZATION_EPS)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = train_conditional_opd(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
