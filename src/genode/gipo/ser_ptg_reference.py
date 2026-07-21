from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from genode.cli import parse_int_csv
from genode.experiment_layout import (
    TRAIN_TUNING_CONTEXT_SAMPLE_COUNT,
    REFERENCE_SEEN_NFES,
    SCENARIO_FAMILY_CONDITIONAL_GENERATION,
    SCENARIO_FAMILY_FORECAST,
    SCENARIO_FAMILY_MOLECULE,
    scenario_family_for_key,
)
from genode.data.molecule_xyz import load_molecule_group_manifest, molecule_group_root, trainable_molecule_group_members
from genode.solver_protocol import (
    SUPPORTED_SOLVER_KEYS,
    normalize_solver_keys,
    solver_eval_multiplier,
    solver_macro_steps,
    solver_order_p,
    solver_runtime_name,
    target_nfe_for_macro_steps,
    uniform_time_grid,
)
from genode.gipo.density_representation import (
    average_density_masses,
    density_mass_hash,
    density_mass_to_time_grid,
    grid_to_density_mass,
    reference_grid_hash,
    uniform_reference_grid,
)
from genode.gipo.models import validate_time_grid
from genode.gipo.schedule_hash import schedule_grid_hash
from genode.data.otflow_paths import (
    backbone_manifest_path,
    cryptos_data_path,
    lobster_synthetic_profile_path,
    long_term_st_data_path,
    project_outputs_root,
    project_dataset_root,
    resolve_project_path,
)
from genode.evaluation.fm_backbone_registry import BACKBONE_NAME_OTFLOW_MOLECULE, MOLECULE_FAMILY, find_backbone_artifact, load_backbone_manifest
from genode.evaluation.molecule_metrics import load_molecule_checkpoint_splits
from genode.evaluation.otflow_sampling_support import _choose_valid_windows
from genode.evaluation.otflow_evaluation_support import (
    DEFAULT_SHARED_BACKBONE_ROOT,
    DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION,
    DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION,
    TRAIN_TUNING_PHASE,
    TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION,
    TRAIN_TUNING_SAMPLING_MODES,
    VALIDATION_PHASE,
    choose_forecast_example_indices,
    choose_forecast_train_tuning_indices,
    load_conditional_generation_checkpoint_splits,
    load_forecast_checkpoint_splits,
    resolved_eval_horizon,
    resolve_reference_macro_steps,
    train_tuning_sampler_key,
    train_tuning_target_example_count,
)
from genode.models.otflow_train_val import _get_dataset_item_by_t, _parse_batch, save_json, seed_all
from genode.runtime import resolve_torch_device

SER_PTG_SCHEDULE_KEY = "ser_ptg_local_defect_eta005"
SER_PTG_SCHEDULE_NAME = "SER-PTG local defect eta=0.05"
SER_PTG_REVERSED_SCHEDULE_KEY = "ser_ptg_local_defect_eta005_reversed"
SER_PTG_AVG_REVERSED_SCHEDULE_KEY = "ser_ptg_local_defect_eta005_avg_reversed"
SER_PTG_EXAMPLE_SELECTION_PROTOCOL = "ser_ptg_reference_global_context_selection"
SER_PTG_LOCAL_DEFECT_PROXY_PROTOCOL = "otflow_midpoint_local_defect_proxy"


def _positive_int(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return parsed


def _optional_positive_int(value: Any, *, name: str) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}.")
    return parsed if parsed > 0 else None


def _reference_example_cap_and_source(args: argparse.Namespace, *, reference_split: str) -> Tuple[int, str]:
    context_cap = _positive_int(getattr(args, "context_sample_count", TRAIN_TUNING_CONTEXT_SAMPLE_COUNT), name="--context_sample_count")
    if str(reference_split) == TRAIN_TUNING_PHASE:
        explicit = _optional_positive_int(getattr(args, "train_tuning_max_examples", None), name="--train_tuning_max_examples")
        if explicit is not None:
            source = str(getattr(args, "train_tuning_max_examples_source", "") or "train_tuning_max_examples")
            return int(explicit), source
        return int(context_cap), "context_sample_count"
    val_windows = int(getattr(args, "val_windows", 0))
    if val_windows < 0:
        raise ValueError(f"--val_windows must be nonnegative, got {val_windows!r}.")
    if val_windows > 0:
        return int(val_windows), "val_windows"
    return int(context_cap), "context_sample_count"


def _deterministically_cap_index_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    seed: int,
    salt: str,
) -> Tuple[List[List[int]], List[Dict[str, Any]], Dict[str, Any]]:
    selected_cap = _positive_int(cap, name="selected_examples_cap")
    normalized: List[Dict[str, Any]] = []
    flat_candidates: List[Tuple[int, int]] = []
    uncapped_total = 0
    for group_idx, group in enumerate(groups):
        candidate = [int(idx) for idx in group.get("candidate_indices", [])]
        uncapped_count = int(group.get("uncapped_candidate_examples", len(candidate)))
        if uncapped_count < len(candidate):
            raise ValueError(
                "uncapped_candidate_examples cannot be smaller than the selected candidate list "
                f"({uncapped_count} < {len(candidate)})."
            )
        normalized.append({**dict(group), "candidate_indices": candidate, "uncapped_candidate_examples": uncapped_count})
        uncapped_total += int(uncapped_count)
        flat_candidates.extend((int(group_idx), int(pos)) for pos in range(len(candidate)))
    if not flat_candidates:
        raise ValueError("SER-PTG global selection requires at least one candidate example.")
    offsets: List[int] = []
    cursor = 0
    for group in normalized:
        offsets.append(int(cursor))
        cursor += len(group["candidate_indices"])
    selected_target = min(int(selected_cap), int(len(flat_candidates)))
    active_groups = [idx for idx, group in enumerate(normalized) if group["candidate_indices"]]
    if len(flat_candidates) <= selected_cap and uncapped_total <= len(flat_candidates):
        kept_positions = list(range(len(flat_candidates)))
        selection_was_capped = False
    else:
        shape = ",".join(str(len(group["candidate_indices"])) for group in normalized)
        token = f"{SER_PTG_EXAMPLE_SELECTION_PROTOCOL}|global|{salt}|{int(seed)}|{shape}|{len(flat_candidates)}|{selected_cap}"
        local_seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(local_seed)
        kept_positions = []
        if selected_target >= len(active_groups):
            for group_idx in active_groups:
                group_size = len(normalized[group_idx]["candidate_indices"])
                token_one = f"{token}|group|{group_idx}|{group_size}"
                candidate_pos = int(hashlib.sha256(token_one.encode("utf-8")).hexdigest()[:16], 16) % int(group_size)
                kept_positions.append(int(offsets[group_idx]) + int(candidate_pos))
        remaining = int(selected_target) - len(kept_positions)
        if remaining > 0:
            kept = set(kept_positions)
            available_positions = [pos for pos in range(len(flat_candidates)) if pos not in kept]
            chosen = rng.choice(np.arange(len(available_positions)), size=int(remaining), replace=False).tolist()
            kept_positions.extend(int(available_positions[int(pos)]) for pos in chosen)
        kept_positions = sorted(kept_positions)
        selection_was_capped = True
    selected_by_group: List[List[int]] = [[] for _ in normalized]
    for flat_pos in kept_positions:
        group_idx, candidate_pos = flat_candidates[int(flat_pos)]
        selected_by_group[group_idx].append(int(normalized[group_idx]["candidate_indices"][candidate_pos]))
    selected_total = int(sum(len(indices) for indices in selected_by_group))
    records: List[Dict[str, Any]] = []
    for group, selected in zip(normalized, selected_by_group):
        record = dict(group.get("selection_record", {}))
        record.update(
            {
                "example_selection_protocol": SER_PTG_EXAMPLE_SELECTION_PROTOCOL,
                "uncapped_candidate_examples": int(group["uncapped_candidate_examples"]),
                "global_uncapped_candidate_examples": int(uncapped_total),
                "candidate_examples_after_initial_selection": int(len(group["candidate_indices"])),
                "global_candidate_examples_after_initial_selection": int(len(flat_candidates)),
                "selected_examples": int(len(selected)),
                "selected_examples_cap": int(selected_cap),
                "global_selected_examples": int(selected_total),
                "selection_was_capped": bool(selection_was_capped or int(group["uncapped_candidate_examples"]) > len(selected)),
                "global_selection_was_capped": bool(selection_was_capped),
                "eval_examples": 0,
                "trace_count": 0,
            }
        )
        records.append(record)
    return selected_by_group, records, {
        "selected_examples": int(selected_total),
        "selected_examples_cap": int(selected_cap),
        "uncapped_candidate_examples": int(uncapped_total),
        "candidate_examples_after_initial_selection": int(len(flat_candidates)),
        "selection_was_capped": bool(selection_was_capped),
    }


def _sum_int_records_by(records: Sequence[Mapping[str, Any]], key: str, value_key: str) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for record in records:
        group_key = str(record.get(key, ""))
        if not group_key:
            continue
        totals[group_key] = int(totals.get(group_key, 0)) + int(record.get(value_key, 0))
    return totals


def solver_order_for_ptg(solver_key: str) -> float:
    return solver_order_p(str(solver_key))


def grid_geometry(time_grid: Sequence[float]) -> Dict[str, float]:
    values = np.asarray([float(x) for x in time_grid], dtype=np.float64)
    internal = values[1:-1]
    intervals = np.diff(values)
    return {
        "internal_fraction_after_098": float(np.mean(internal > 0.98)) if internal.size else 0.0,
        "internal_count_after_098": int(np.sum(internal > 0.98)),
        "internal_count": int(internal.size),
        "min_interval": float(np.min(intervals)) if intervals.size else 0.0,
        "max_interval": float(np.max(intervals)) if intervals.size else 0.0,
    }


def _prediction_with_density_metadata(prediction: Mapping[str, Any], *, scheduler_key: str, schedule_name: str, time_grid: Sequence[float]) -> Dict[str, Any]:
    macro_steps = int(prediction["macro_steps"])
    grid = validate_time_grid(time_grid, macro_steps=macro_steps)
    density_reference = uniform_reference_grid()
    mass = grid_to_density_mass(grid, reference_time_grid=density_reference, macro_steps=macro_steps)
    copied = dict(prediction)
    copied.update(
        {
            "scheduler_key": str(scheduler_key),
            "schedule_name": str(schedule_name),
            "time_grid": list(grid),
            "schedule_grid_hash": schedule_grid_hash(grid),
            "density_protocol": "density_mass",
            "density_reference_grid_hash": reference_grid_hash(density_reference),
            "density_mass": [float(x) for x in mass],
            "density_mass_hash": density_mass_hash(mass, reference_time_grid=density_reference),
            "grid_geometry": grid_geometry(grid),
        }
    )
    return copied


def _derive_ser_reference_predictions(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    physical: List[Dict[str, Any]] = []
    reversed_rows: List[Dict[str, Any]] = []
    averaged_rows: List[Dict[str, Any]] = []
    density_reference = uniform_reference_grid()
    for prediction in predictions:
        macro_steps = int(prediction["macro_steps"])
        base_grid = validate_time_grid(prediction["time_grid"], macro_steps=macro_steps)
        physical_prediction = _prediction_with_density_metadata(
            prediction,
            scheduler_key=SER_PTG_SCHEDULE_KEY,
            schedule_name=SER_PTG_SCHEDULE_NAME,
            time_grid=base_grid,
        )
        reversed_grid = validate_time_grid([1.0 - float(value) for value in reversed(base_grid)], macro_steps=macro_steps)
        reversed_prediction = _prediction_with_density_metadata(
            {**prediction, "schedule_derivation": "reverse_physical_clock"},
            scheduler_key=SER_PTG_REVERSED_SCHEDULE_KEY,
            schedule_name=f"{SER_PTG_SCHEDULE_NAME} reversed",
            time_grid=reversed_grid,
        )
        base_mass = grid_to_density_mass(base_grid, reference_time_grid=density_reference, macro_steps=macro_steps)
        reversed_mass = grid_to_density_mass(reversed_grid, reference_time_grid=density_reference, macro_steps=macro_steps)
        averaged_mass = average_density_masses(base_mass, reversed_mass)
        averaged_grid = density_mass_to_time_grid(
            averaged_mass,
            reference_time_grid=density_reference,
            macro_steps=macro_steps,
        )
        averaged_prediction = _prediction_with_density_metadata(
            {**prediction, "schedule_derivation": "average_physical_and_reversed_density_mass"},
            scheduler_key=SER_PTG_AVG_REVERSED_SCHEDULE_KEY,
            schedule_name=f"{SER_PTG_SCHEDULE_NAME} density average",
            time_grid=averaged_grid,
        )
        physical.append(physical_prediction)
        reversed_rows.append(reversed_prediction)
        averaged_rows.append(averaged_prediction)
    return {
        SER_PTG_SCHEDULE_KEY: physical,
        SER_PTG_REVERSED_SCHEDULE_KEY: reversed_rows,
        SER_PTG_AVG_REVERSED_SCHEDULE_KEY: averaged_rows,
    }


def _normalize_kappa(hardness: Sequence[float], reference_time_grid: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    h = np.clip(np.asarray([float(x) for x in hardness], dtype=np.float64), 0.0, None)
    grid = np.asarray(validate_time_grid(reference_time_grid, macro_steps=len(h)), dtype=np.float64)
    widths = np.diff(grid)
    eps = 1e-8 * max(float(np.mean(h)), 1e-12)
    weighted = h + eps
    denom = float(np.sum(widths * weighted))
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError("Hardness normalization denominator must be positive and finite.")
    return weighted / denom, widths


def ser_ptg_density_mass_from_trace(
    hardness: Sequence[float],
    reference_time_grid: Sequence[float],
    *,
    solver_order_p: float,
    density_floor_eta: float = 0.05,
) -> Tuple[float, ...]:
    """Return the SER-PTG discrete probability mass over reference-grid bins."""

    kappa, widths = _normalize_kappa(hardness, reference_time_grid)
    p = float(solver_order_p)
    if p <= 0.0:
        raise ValueError(f"solver_order_p must be positive, got {solver_order_p}")
    raw_density = np.power(np.clip(kappa, 1e-12, None), 1.0 / (p + 1.0))
    raw_density = raw_density / max(float(np.sum(widths * raw_density)), 1e-12)
    eta = float(density_floor_eta)
    if eta < 0.0 or eta > 1.0:
        raise ValueError(f"density_floor_eta must lie in [0, 1], got {eta}")
    density = (1.0 - eta) * raw_density + eta
    density = density / max(float(np.sum(widths * density)), 1e-12)
    mass = widths * density
    mass = mass / max(float(np.sum(mass)), 1e-12)
    return tuple(float(x) for x in mass.tolist())


def ser_ptg_grid_from_trace(
    hardness: Sequence[float],
    reference_time_grid: Sequence[float],
    *,
    macro_steps: int,
    solver_order_p: float,
    density_floor_eta: float = 0.05,
) -> Tuple[float, ...]:
    mass = np.asarray(
        ser_ptg_density_mass_from_trace(
            hardness,
            reference_time_grid,
            solver_order_p=float(solver_order_p),
            density_floor_eta=float(density_floor_eta),
        ),
        dtype=np.float64,
    )
    cdf = np.concatenate([[0.0], np.cumsum(mass)])
    cdf[-1] = 1.0
    ref = np.asarray(reference_time_grid, dtype=np.float64)
    targets = np.linspace(0.0, 1.0, int(macro_steps) + 1, dtype=np.float64)
    grid = np.interp(targets, cdf, ref)
    grid[0] = 0.0
    grid[-1] = 1.0
    for idx in range(1, len(grid)):
        if grid[idx] <= grid[idx - 1]:
            grid[idx] = min(1.0, grid[idx - 1] + 1e-8)
    grid[-1] = 1.0
    return validate_time_grid(grid.tolist(), macro_steps=int(macro_steps))


def _local_defect_trace_from_oracle(oracle: np.ndarray, reference_time_grid: Sequence[float], *, solver_order_p: float) -> np.ndarray:
    grid = np.asarray(reference_time_grid, dtype=np.float64)
    widths = np.diff(grid)
    return np.asarray(oracle, dtype=np.float64) / (np.power(widths, float(solver_order_p) + 1.0) + 1e-12)


@torch.no_grad()
def collect_batched_local_defect_trace(
    model: Any,
    ds: Any,
    cfg: Any,
    *,
    solver_name: str,
    reference_macro_steps: int,
    solver_order_p: float,
    seed: int,
    example_indices: Sequence[int],
    calibration_trace_samples: int = 1,
    batch_size: int = 64,
    item_getter: Any | None = None,
) -> Dict[str, Any]:
    device = cfg.train.device
    indices = [int(idx) for idx in example_indices]
    if not indices:
        raise ValueError("SER-PTG trace collection requires at least one validation example.")
    reference_time_grid: Optional[List[float]] = None
    trace_rows: List[np.ndarray] = []
    oracle_rows: List[np.ndarray] = []
    effective_batch_size = max(1, int(batch_size))
    for sample_idx in range(int(calibration_trace_samples)):
        for chunk_start in range(0, len(indices), effective_batch_size):
            chunk = indices[chunk_start : chunk_start + effective_batch_size]
            hist_rows = []
            cond_rows = []
            saw_cond = False
            for example_idx in chunk:
                batch = item_getter(ds, int(example_idx)) if item_getter is not None else ds[int(example_idx)]
                hist_t, _tgt_t, _fut_t, cond_t, _meta = _parse_batch(batch)
                hist_rows.append(hist_t.float())
                if cond_t is not None:
                    saw_cond = True
                    cond_rows.append(cond_t.float())
            hist = torch.stack(hist_rows, dim=0).to(device).float()
            cond = torch.stack(cond_rows, dim=0).to(device).float() if saw_cond else None
            seed_all(int(seed) + 1_000_000 * int(sample_idx) + int(chunk_start))
            target_nfe = target_nfe_for_macro_steps(str(solver_name), int(reference_macro_steps))
            trace_kwargs = {
                "solver_key": str(solver_name),
                "target_nfe": target_nfe,
                "time_grid": uniform_time_grid(str(solver_name), target_nfe),
                "include_local_error": True,
            }
            if cond is not None:
                trace_kwargs["cond"] = cond
            _pred, diagnostics = model.sample_future_with_diagnostics(hist, **trace_kwargs)
            grid = [float(x) for x in diagnostics.trajectory.time_grid.detach().cpu().numpy().astype(np.float64).tolist()]
            if reference_time_grid is None:
                reference_time_grid = grid
            elif not np.allclose(np.asarray(reference_time_grid), np.asarray(grid), atol=1e-8, rtol=1e-8):
                raise ValueError("Reference time grids changed during SER-PTG trace collection.")
            oracle = diagnostics.local_error.detach().cpu().numpy().astype(np.float64)
            for row in oracle:
                oracle_rows.append(row)
                trace_rows.append(_local_defect_trace_from_oracle(row, grid, solver_order_p=float(solver_order_p)))
    if reference_time_grid is None or not trace_rows:
        raise ValueError("No SER-PTG traces were collected.")
    return {
        "reference_time_grid": reference_time_grid,
        "local_defect_trace": [float(x) for x in np.mean(np.stack(trace_rows, axis=0), axis=0).tolist()],
        "oracle_local_error_trace": [float(x) for x in np.mean(np.stack(oracle_rows, axis=0), axis=0).tolist()],
        "eval_examples": int(len(indices)),
        "trace_count": int(len(trace_rows)),
    }


@torch.no_grad()
def collect_molecule_local_defect_trace(
    model: Any,
    ds: Any,
    cfg: Any,
    *,
    solver_name: str,
    reference_macro_steps: int,
    solver_order_p: float,
    seed: int,
    example_indices: Sequence[int],
    calibration_trace_samples: int = 1,
    batch_size: int = 64,
) -> Dict[str, Any]:
    device = cfg.train.device
    indices = [int(idx) for idx in example_indices]
    if not indices:
        raise ValueError("Molecule SER-PTG trace collection requires at least one example.")
    reference_time_grid: Optional[List[float]] = None
    trace_rows: List[np.ndarray] = []
    oracle_rows: List[np.ndarray] = []
    effective_batch_size = max(1, int(batch_size))
    for sample_idx in range(int(calibration_trace_samples)):
        for chunk_start in range(0, len(indices), effective_batch_size):
            chunk = indices[chunk_start : chunk_start + effective_batch_size]
            hist_rows = []
            for example_idx in chunk:
                item = ds.eval_item(int(example_idx))
                history = np.asarray(item.get("history_coords", []), dtype=np.float32)
                context = ds.context_features_from_history_coords(history)
                hist = (context - ds.stats.context_mean[None, :]) / ds.stats.context_std[None, :]
                hist_rows.append(torch.from_numpy(hist.astype(np.float32)))
            hist_t = torch.stack(hist_rows, dim=0).to(device).float()
            seed_all(int(seed) + 1_000_000 * int(sample_idx) + int(chunk_start))
            target_nfe = target_nfe_for_macro_steps(str(solver_name), int(reference_macro_steps))
            _pred, diagnostics = model.sample_future_with_diagnostics(
                hist_t,
                solver_key=str(solver_name),
                target_nfe=target_nfe,
                time_grid=uniform_time_grid(str(solver_name), target_nfe),
                include_local_error=True,
            )
            grid = [float(x) for x in diagnostics.trajectory.time_grid.detach().cpu().numpy().astype(np.float64).tolist()]
            if reference_time_grid is None:
                reference_time_grid = grid
            elif not np.allclose(np.asarray(reference_time_grid), np.asarray(grid), atol=1e-8, rtol=1e-8):
                raise ValueError("Reference time grids changed during molecule SER-PTG trace collection.")
            oracle = diagnostics.local_error.detach().cpu().numpy().astype(np.float64)
            for row in oracle:
                oracle_rows.append(row)
                trace_rows.append(_local_defect_trace_from_oracle(row, grid, solver_order_p=float(solver_order_p)))
    if reference_time_grid is None or not trace_rows:
        raise ValueError("No molecule SER-PTG traces were collected.")
    return {
        "reference_time_grid": reference_time_grid,
        "local_defect_trace": [float(x) for x in np.mean(np.stack(trace_rows, axis=0), axis=0).tolist()],
        "oracle_local_error_trace": [float(x) for x in np.mean(np.stack(oracle_rows, axis=0), axis=0).tolist()],
        "eval_examples": int(len(indices)),
        "trace_count": int(len(trace_rows)),
    }


def _choose_molecule_indices(ds: Any, *, count: int, seed: int) -> List[int]:
    total = int(len(ds))
    if total <= 0:
        raise ValueError("Empty molecule SER reference split.")
    target = min(max(1, int(count)), total)
    indices = np.arange(total, dtype=np.int64)
    if target < total:
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(indices, size=target, replace=False))
    return [int(x) for x in indices.tolist()]


def build_ser_ptg_reference(args: argparse.Namespace) -> Dict[str, Any]:
    scenario_key = str(args.scenario_key)
    family = scenario_family_for_key(scenario_key)
    solvers = list(normalize_solver_keys(str(args.solver_names)))
    target_nfes = parse_int_csv(args.target_nfe_values)
    seeds = parse_int_csv(args.seeds)
    reference_split = str(args.reference_split)
    if reference_split not in {TRAIN_TUNING_PHASE, VALIDATION_PHASE}:
        raise ValueError(f"reference_split must be {TRAIN_TUNING_PHASE!r} or {VALIDATION_PHASE!r}.")
    if bool(args.smoke):
        solvers = solvers[:1]
        target_nfes = target_nfes[:1]
        args.val_windows = min(max(1, int(args.val_windows) if int(args.val_windows) > 0 else 2), 2)
    reference_example_cap, reference_example_cap_source = _reference_example_cap_and_source(args, reference_split=reference_split)
    dataset_root = resolve_project_path(str(args.dataset_root))
    shared_backbone_root = resolve_project_path(str(args.shared_backbone_root))
    device = resolve_torch_device(str(args.device))
    split_key = "train" if reference_split == TRAIN_TUNING_PHASE else "val"
    member_refs: List[Dict[str, Any]] = []
    if family == SCENARIO_FAMILY_FORECAST:
        checkpoint = load_forecast_checkpoint_splits(
            cli_args=args,
            dataset_root=dataset_root,
            shared_backbone_root=shared_backbone_root,
            dataset=scenario_key,
            device=device,
        )
        member_refs.append({"checkpoint": checkpoint, "reference_ds": checkpoint["splits"][split_key], "member_key": "", "stratum": ""})
    elif family == SCENARIO_FAMILY_CONDITIONAL_GENERATION:
        checkpoint = load_conditional_generation_checkpoint_splits(
            cli_args=args,
            shared_backbone_root=shared_backbone_root,
            dataset=scenario_key,
            device=device,
        )
        member_refs.append({"checkpoint": checkpoint, "reference_ds": checkpoint["splits"][split_key], "member_key": "", "stratum": ""})
    elif family == SCENARIO_FAMILY_MOLECULE:
        group_root = resolve_project_path(str(getattr(args, "molecule_group_root", "") or molecule_group_root()))
        manifest = load_molecule_group_manifest(scenario_key, group_root)
        backbone_manifest = load_backbone_manifest(resolve_project_path(str(args.backbone_manifest)))
        for member in trainable_molecule_group_members(manifest):
            artifact = find_backbone_artifact(
                backbone_manifest,
                backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
                benchmark_family=MOLECULE_FAMILY,
                dataset_key=scenario_key,
                train_steps=int(args.checkpoint_step),
                member_key=str(member["member_key"]),
                stratum=str(member["stratum"]),
            )
            loaded = load_molecule_checkpoint_splits(
                checkpoint_path=str(artifact["checkpoint_path"]),
                dataset_key=scenario_key,
                stratum=str(member["stratum"]),
                processed_dir=group_root / scenario_key / str(member["processed_dir"]),
                rollout_steps=1,
                stride_eval=1,
                device=device,
            )
            member_refs.append(
                {
                    "checkpoint": {"model": loaded["model"], "cfg": loaded["cfg"], "splits": loaded["splits"], "checkpoint_id": artifact["checkpoint_id"]},
                    "reference_ds": loaded["splits"][split_key],
                    "member_key": str(member["member_key"]),
                    "stratum": str(member["stratum"]),
                }
            )
        if not member_refs:
            raise ValueError(f"Molecule group {scenario_key!r} has no trainable members for SER-PTG.")
    else:
        raise ValueError(f"Unsupported SER-PTG scenario family: {family!r}")
    train_tuning_reference_examples = int(max(len(ref["checkpoint"]["splits"].get("val", [])) for ref in member_refs))
    predictions: List[Dict[str, Any]] = []
    for solver_idx, solver_key in enumerate(solvers):
        solver_name = solver_runtime_name(solver_key)
        solver_p = solver_order_for_ptg(str(solver_key))
        for target_idx, target_nfe in enumerate(target_nfes):
            macro_steps = solver_macro_steps(str(solver_key), int(target_nfe))
            reference_macro_steps = resolve_reference_macro_steps(
                0,
                int(macro_steps),
                reference_macro_factor=float(args.reference_macro_factor),
            )
            seed_traces: List[np.ndarray] = []
            trace_weights: List[int] = []
            reference_grid: Optional[List[float]] = None
            eval_examples = 0
            trace_count = 0
            work_items: List[Dict[str, Any]] = []
            for seed in seeds:
                for member_idx, ref in enumerate(member_refs):
                    checkpoint = ref["checkpoint"]
                    model = checkpoint["model"]
                    cfg = checkpoint["cfg"]
                    reference_ds = ref["reference_ds"]
                    selection_seed = (
                        int(args.train_tuning_seed)
                        + int(seed)
                        + 100_000 * int(member_idx)
                        + 10_000 * int(solver_idx)
                        + 1_000 * int(target_idx)
                    )
                    if family == SCENARIO_FAMILY_FORECAST:
                        if reference_split == TRAIN_TUNING_PHASE:
                            forecast_uncapped_candidate_examples = train_tuning_target_example_count(
                                len(reference_ds),
                                fraction=float(args.train_tuning_fraction),
                                sampling_mode=str(args.train_tuning_sampling_mode),
                                strata=int(args.train_tuning_strata),
                                reference_examples=int(train_tuning_reference_examples),
                                train_split_fraction=float(args.train_tuning_train_split_fraction),
                                val_split_fraction=float(args.train_tuning_val_split_fraction),
                            )
                            candidate_indices = choose_forecast_train_tuning_indices(
                                reference_ds,
                                fraction=float(args.train_tuning_fraction),
                                seed=selection_seed,
                                strata=int(args.train_tuning_strata),
                                dataset=scenario_key,
                                sampling_mode=str(args.train_tuning_sampling_mode),
                                reference_examples=int(train_tuning_reference_examples),
                                train_split_fraction=float(args.train_tuning_train_split_fraction),
                                val_split_fraction=float(args.train_tuning_val_split_fraction),
                                max_examples=int(reference_example_cap),
                            )
                        else:
                            forecast_uncapped_candidate_examples = int(len(reference_ds))
                            validation_request = int(args.val_windows) if int(args.val_windows) > 0 else 0
                            candidate_indices = choose_forecast_example_indices(
                                reference_ds,
                                n_examples=int(validation_request),
                                seed=selection_seed,
                            )
                        item_getter = None
                        collector = collect_batched_local_defect_trace
                        available_examples = int(len(reference_ds))
                        uncapped_candidate_examples = int(forecast_uncapped_candidate_examples)
                    elif family == SCENARIO_FAMILY_CONDITIONAL_GENERATION:
                        horizon = resolved_eval_horizon(args, scenario_key)
                        available = len(getattr(reference_ds, "start_indices", []))
                        window_count = int(args.val_windows) if int(args.val_windows) > 0 else int(reference_example_cap)
                        candidate_indices = _choose_valid_windows(
                            reference_ds,
                            horizon=int(horizon),
                            n_windows=min(max(1, int(window_count)), max(1, int(available))),
                            seed=selection_seed,
                        )
                        def conditional_item_getter(ds, t0):
                            return _get_dataset_item_by_t(ds, int(t0))

                        item_getter = conditional_item_getter
                        collector = collect_batched_local_defect_trace
                        available_examples = int(available)
                        uncapped_candidate_examples = int(available_examples)
                    else:
                        window_count = int(args.val_windows) if int(args.val_windows) > 0 else int(reference_example_cap)
                        candidate_indices = _choose_molecule_indices(reference_ds, count=int(window_count), seed=selection_seed)
                        item_getter = None
                        collector = collect_molecule_local_defect_trace
                        available_examples = int(len(reference_ds))
                        uncapped_candidate_examples = int(available_examples)
                    work_items.append(
                        {
                            "candidate_indices": [int(idx) for idx in candidate_indices],
                            "uncapped_candidate_examples": int(uncapped_candidate_examples),
                            "model": model,
                            "reference_ds": reference_ds,
                            "cfg": cfg,
                            "collector": collector,
                            "item_getter": item_getter,
                            "collector_seed": int(seed) + 100_000 * int(member_idx) + 10_000 * int(solver_idx) + 1_000 * int(target_idx),
                            "selection_record": {
                                "benchmark_family": str(family),
                                "reference_split": str(reference_split),
                                "seed": int(seed),
                                "member_key": str(ref.get("member_key", "")),
                                "stratum": str(ref.get("stratum", "")),
                                "solver_key": str(solver_key),
                                "target_nfe": int(target_nfe),
                                "reference_available_examples": int(available_examples),
                                "selected_examples_cap_source": str(reference_example_cap_source),
                            },
                        }
                    )
            effective_reference_cap = min(2, int(reference_example_cap)) if bool(args.smoke) else int(reference_example_cap)
            selected_groups, selection_records, _global_selection_meta = _deterministically_cap_index_groups(
                work_items,
                cap=int(effective_reference_cap),
                seed=int(args.train_tuning_seed) + 10_000 * int(solver_idx) + 1_000 * int(target_idx),
                salt=f"{family}|{reference_split}|{scenario_key}|{solver_key}|{target_nfe}",
            )
            if bool(args.smoke):
                for record in selection_records:
                    record["smoke_selected_examples_cap"] = 2
            for group_idx, (work_item, chosen) in enumerate(zip(work_items, selected_groups)):
                if not chosen:
                    continue
                collector = work_item["collector"]
                collected = collector(
                    work_item["model"],
                    work_item["reference_ds"],
                    work_item["cfg"],
                    solver_name=solver_name,
                    reference_macro_steps=int(reference_macro_steps),
                    solver_order_p=float(solver_p),
                    seed=int(work_item["collector_seed"]),
                    example_indices=chosen,
                    calibration_trace_samples=int(args.calibration_trace_samples),
                    batch_size=int(args.calibration_batch_size),
                    **({"item_getter": work_item["item_getter"]} if work_item["item_getter"] is not None else {}),
                )
                grid = [float(x) for x in collected["reference_time_grid"]]
                if reference_grid is None:
                    reference_grid = grid
                elif not np.allclose(np.asarray(reference_grid), np.asarray(grid), atol=1e-8, rtol=1e-8):
                    raise ValueError(f"Reference grids changed for {solver_key}/{target_nfe}.")
                seed_traces.append(np.asarray(collected["local_defect_trace"], dtype=np.float64))
                trace_weights.append(max(1, int(collected["trace_count"])))
                eval_examples += int(collected["eval_examples"])
                trace_count += int(collected["trace_count"])
                selection_records[group_idx]["eval_examples"] = int(collected["eval_examples"])
                selection_records[group_idx]["trace_count"] = int(collected["trace_count"])
            if reference_grid is None:
                raise ValueError(f"No reference grid collected for {solver_key}/{target_nfe}.")
            selected_examples = int(sum(int(record["selected_examples"]) for record in selection_records))
            uncapped_candidate_examples = int(sum(int(record["uncapped_candidate_examples"]) for record in selection_records))
            selected_example_caps = sorted({int(record["selected_examples_cap"]) for record in selection_records})
            selection_was_capped = any(bool(record.get("selection_was_capped", False)) for record in selection_records)
            examples_by_seed = _sum_int_records_by(selection_records, "seed", "selected_examples")
            examples_by_member = _sum_int_records_by(selection_records, "member_key", "selected_examples")
            trace_count_by_seed = _sum_int_records_by(selection_records, "seed", "trace_count")
            trace_count_by_member = _sum_int_records_by(selection_records, "member_key", "trace_count")
            per_seed_counts = list(examples_by_seed.values())
            reference_examples_per_seed = int(max(per_seed_counts)) if per_seed_counts else int(selected_examples)
            mean_trace = np.average(np.stack(seed_traces, axis=0), axis=0, weights=np.asarray(trace_weights, dtype=np.float64))
            time_grid = ser_ptg_grid_from_trace(
                mean_trace.tolist(),
                reference_grid,
                macro_steps=int(macro_steps),
                solver_order_p=float(solver_p),
                density_floor_eta=float(args.density_floor_eta),
            )
            predictions.append(
                {
                    "solver_key": str(solver_key),
                    "target_nfe": int(target_nfe),
                    "checkpoint_step": int(args.checkpoint_step),
                    "macro_steps": int(macro_steps),
                    "realized_nfe": int(macro_steps) * int(solver_eval_multiplier(str(solver_key))),
                    "time_grid": list(time_grid),
                    "schedule_grid_hash": schedule_grid_hash(time_grid),
                    "reference_macro_steps": int(reference_macro_steps),
                    "reference_time_grid": reference_grid,
                    "local_defect_trace": [float(x) for x in mean_trace.tolist()],
                    "density_floor_eta": float(args.density_floor_eta),
                    "solver_order_p": float(solver_p),
                    "reference_split": reference_split,
                    "example_selection_protocol": SER_PTG_EXAMPLE_SELECTION_PROTOCOL,
                    "context_sample_count": int(args.context_sample_count),
                    "selected_examples": int(selected_examples),
                    "selected_examples_cap": int(selected_example_caps[0]) if len(selected_example_caps) == 1 else selected_example_caps,
                    "selected_examples_cap_source": str(reference_example_cap_source),
                    "uncapped_candidate_examples": int(uncapped_candidate_examples),
                    "selection_was_capped": bool(selection_was_capped),
                    "selection_records": selection_records,
                    "reference_examples_total": int(selected_examples),
                    "reference_examples_by_seed": examples_by_seed,
                    "reference_examples_by_member": examples_by_member,
                    "reference_examples_per_seed": int(reference_examples_per_seed),
                    "reference_examples_per_seed_min": int(min(per_seed_counts)) if per_seed_counts else int(selected_examples),
                    "reference_examples_per_seed_mean": float(np.mean(per_seed_counts)) if per_seed_counts else float(selected_examples),
                    "reference_seed_count": int(len(examples_by_seed)),
                    "reference_member_count": int(len(examples_by_member)) if examples_by_member else int(len(member_refs)),
                    "reference_selection_group_count": int(len(selection_records)),
                    "train_tuning_fraction": float(args.train_tuning_fraction) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_seed": int(args.train_tuning_seed) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_strata": int(args.train_tuning_strata) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_sampler": train_tuning_sampler_key(str(args.train_tuning_sampling_mode)) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_sampling_mode": str(args.train_tuning_sampling_mode) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_reference_examples": int(train_tuning_reference_examples) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_target_examples": int(selected_examples) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_max_examples": int(reference_example_cap) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_max_examples_source": str(reference_example_cap_source) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_uncapped_candidate_examples": int(uncapped_candidate_examples) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_train_split_fraction": float(args.train_tuning_train_split_fraction) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_val_split_fraction": float(args.train_tuning_val_split_fraction) if reference_split == TRAIN_TUNING_PHASE else "",
                    "trace_count": int(trace_count),
                    "trace_count_by_seed": trace_count_by_seed,
                    "trace_count_by_member": trace_count_by_member,
                    "grid_geometry": grid_geometry(time_grid),
                }
            )
    out_dir = resolve_project_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    derived_predictions = _derive_ser_reference_predictions(predictions)
    schedule = {
        "scheduler_key": SER_PTG_SCHEDULE_KEY,
        "schedule_name": SER_PTG_SCHEDULE_NAME,
        "comparison_role": "reference_comparator",
        "density_floor_eta": float(args.density_floor_eta),
        "predictions": derived_predictions[SER_PTG_SCHEDULE_KEY],
    }
    schedules = [
        schedule,
        {
            "scheduler_key": SER_PTG_REVERSED_SCHEDULE_KEY,
            "schedule_name": f"{SER_PTG_SCHEDULE_NAME} reversed",
            "comparison_role": "reference_comparator",
            "density_floor_eta": float(args.density_floor_eta),
            "schedule_derivation": "reverse_physical_clock",
            "predictions": derived_predictions[SER_PTG_REVERSED_SCHEDULE_KEY],
        },
        {
            "scheduler_key": SER_PTG_AVG_REVERSED_SCHEDULE_KEY,
            "schedule_name": f"{SER_PTG_SCHEDULE_NAME} density average",
            "comparison_role": "reference_comparator",
            "density_floor_eta": float(args.density_floor_eta),
            "schedule_derivation": "average_physical_and_reversed_density_mass",
            "predictions": derived_predictions[SER_PTG_AVG_REVERSED_SCHEDULE_KEY],
        },
    ]
    selected_examples_by_prediction = [int(prediction.get("selected_examples", 0)) for prediction in predictions]
    trace_counts_by_prediction = [int(prediction.get("trace_count", 0)) for prediction in predictions]
    uncapped_examples_by_prediction = [int(prediction.get("uncapped_candidate_examples", 0)) for prediction in predictions]
    summary = {
        "status": "ready",
        "artifact": "ser_ptg_schedule_summary",
        "scenario_key": scenario_key,
        "example_selection_protocol": SER_PTG_EXAMPLE_SELECTION_PROTOCOL,
        "scheduler_key": SER_PTG_SCHEDULE_KEY,
        "schedule_name": SER_PTG_SCHEDULE_NAME,
        "schedules": schedules,
        "baseline_schedule": False,
        "seeds": [int(seed) for seed in seeds],
        "solver_names": solvers,
        "target_nfe_values": [int(nfe) for nfe in target_nfes],
        "reference_split": reference_split,
        "reference_split_key": split_key,
        "density_floor_eta": float(args.density_floor_eta),
        "reference_macro_factor": float(args.reference_macro_factor),
        "calibration_trace_samples": int(args.calibration_trace_samples),
        "reference_examples": int(sum(len(ref["reference_ds"]) for ref in member_refs)),
        "context_sample_count": int(args.context_sample_count),
        "selected_examples_cap": int(reference_example_cap),
        "selected_examples_cap_source": str(reference_example_cap_source),
        "selected_examples": int(max(selected_examples_by_prediction) if selected_examples_by_prediction else 0),
        "selected_examples_per_prediction_max": int(max(selected_examples_by_prediction) if selected_examples_by_prediction else 0),
        "selected_examples_total_across_predictions": int(sum(selected_examples_by_prediction)),
        "trace_count_total_across_predictions": int(sum(trace_counts_by_prediction)),
        "prediction_count": int(len(predictions)),
        "uncapped_candidate_examples": int(max(uncapped_examples_by_prediction) if uncapped_examples_by_prediction else 0),
        "uncapped_candidate_examples_total_across_predictions": int(sum(uncapped_examples_by_prediction)),
        "selection_was_capped": any(bool(prediction.get("selection_was_capped", False)) for prediction in predictions),
        "local_defect_trace_protocol": SER_PTG_LOCAL_DEFECT_PROXY_PROTOCOL,
        "oracle_local_error_semantics": "local_defect_proxy_not_teacher_oracle",
        "train_tuning": {
            "fraction": float(args.train_tuning_fraction),
            "seed": int(args.train_tuning_seed),
            "strata": int(args.train_tuning_strata),
            "sampling_mode": str(args.train_tuning_sampling_mode),
            "sampler": train_tuning_sampler_key(str(args.train_tuning_sampling_mode)),
            "reference_split_key": "val",
            "reference_examples": int(train_tuning_reference_examples),
            "max_examples": int(reference_example_cap),
            "max_examples_source": str(reference_example_cap_source),
            "train_split_fraction": float(args.train_tuning_train_split_fraction),
            "val_split_fraction": float(args.train_tuning_val_split_fraction),
        } if reference_split == TRAIN_TUNING_PHASE else None,
        "checkpoint_step": int(args.checkpoint_step),
        "checkpoint_ids": sorted(str(ref["checkpoint"].get("checkpoint_id", "")) for ref in member_refs),
        "member_keys": sorted(str(ref.get("member_key", "")) for ref in member_refs if str(ref.get("member_key", ""))),
        "predictions": predictions,
    }
    save_json(summary, str(out_dir / "ser_ptg_schedule_summary.json"))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local-defect SER-PTG reference schedules for genODE.")
    parser.add_argument("--scenario_key", default="traffic_hourly")
    parser.add_argument("--density_floor_eta", type=float, default=0.05)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--solver_names", default=",".join(SUPPORTED_SOLVER_KEYS))
    parser.add_argument("--target_nfe_values", default=",".join(str(x) for x in REFERENCE_SEEN_NFES))
    parser.add_argument("--reference_split", choices=(TRAIN_TUNING_PHASE, VALIDATION_PHASE), default=TRAIN_TUNING_PHASE)
    parser.add_argument("--val_windows", type=int, default=0)
    parser.add_argument("--context_sample_count", type=int, default=TRAIN_TUNING_CONTEXT_SAMPLE_COUNT)
    parser.add_argument("--train_tuning_max_examples", type=int, default=0)
    parser.add_argument("--train_tuning_max_examples_source", default="")
    parser.add_argument("--train_tuning_fraction", type=float, default=0.20)
    parser.add_argument("--train_tuning_seed", type=int, default=0)
    parser.add_argument("--train_tuning_strata", type=int, default=20)
    parser.add_argument("--train_tuning_sampling_mode", choices=TRAIN_TUNING_SAMPLING_MODES, default=TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION)
    parser.add_argument("--train_tuning_train_split_fraction", type=float, default=DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION)
    parser.add_argument("--train_tuning_val_split_fraction", type=float, default=DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION)
    parser.add_argument("--calibration_trace_samples", type=int, default=1)
    parser.add_argument("--calibration_batch_size", type=int, default=64)
    parser.add_argument("--reference_macro_factor", type=float, default=4.0)
    parser.add_argument(
        "--out_dir",
        default=str(project_outputs_root() / "ser_ptg_reference"),
    )
    parser.add_argument("--dataset_root", default=str(project_dataset_root()))
    parser.add_argument("--shared_backbone_root", default=str(DEFAULT_SHARED_BACKBONE_ROOT))
    parser.add_argument("--backbone_manifest", default=str(backbone_manifest_path()))
    parser.add_argument("--cryptos_path", default=str(cryptos_data_path()))
    parser.add_argument("--lobster_synthetic_profile_path", default=str(lobster_synthetic_profile_path()))
    parser.add_argument("--long_term_st_path", default=str(long_term_st_data_path()))
    parser.add_argument("--molecule_group_root", default=str(molecule_group_root()))
    parser.add_argument("--checkpoint_step", type=int, default=20000)
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", default=False)
    return parser


def main() -> None:
    summary = build_ser_ptg_reference(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
