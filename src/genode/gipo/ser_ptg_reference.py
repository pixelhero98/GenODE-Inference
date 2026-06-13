from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from genode.canonical_experiment_layout import (
    CANONICAL_CONTEXT_SAMPLE_COUNT,
    CANONICAL_SEEN_NFES,
    SCENARIO_FAMILY_CONDITIONAL_GENERATION,
    SCENARIO_FAMILY_FORECAST,
    SCENARIO_FAMILY_MOLECULE,
    scenario_family_for_key,
)
from genode.data.molecule_xyz import default_molecule_group_root, load_molecule_group_manifest, trainable_molecule_group_members
from genode.solver_protocol import CANONICAL_SOLVER_KEYS, normalize_solver_keys, solver_order_p
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
    default_backbone_manifest_path,
    default_cryptos_data_path,
    default_lobster_synthetic_profile_path,
    default_long_term_st_data_path,
    project_outputs_root,
    project_paper_dataset_root,
    resolve_project_path,
)
from genode.evaluation.fm_backbone_registry import BACKBONE_NAME_OTFLOW_MOLECULE, MOLECULE_FAMILY, find_backbone_artifact, load_backbone_manifest
from genode.evaluation.molecule_metrics import load_molecule_checkpoint_splits
from genode.evaluation.otflow_sampling_support import _choose_valid_windows
from genode.evaluation.otflow_evaluation_support import (
    DEFAULT_SHARED_BACKBONE_ROOT,
    DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION,
    DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION,
    SOLVER_RUNTIME_NAMES,
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
    solver_eval_multiplier,
    solver_macro_steps,
    train_tuning_sampler_key,
)
from genode.models.otflow_train_val import _get_dataset_item_by_t, _parse_batch, save_json, seed_all
from genode.runtime import resolve_torch_device

SER_PTG_SCHEDULE_KEY = "ser_ptg_local_defect_eta005"
SER_PTG_SCHEDULE_NAME = "SER-PTG local defect eta=0.05"
SER_PTG_REVERSED_SCHEDULE_KEY = "ser_ptg_local_defect_eta005_reversed"
SER_PTG_AVG_REVERSED_SCHEDULE_KEY = "ser_ptg_local_defect_eta005_avg_reversed"
DEFAULT_SOLVERS: Tuple[str, ...] = CANONICAL_SOLVER_KEYS
DEFAULT_TARGET_NFES: Tuple[int, ...] = CANONICAL_SEEN_NFES


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in _parse_csv(text)]


def _hash_grid(grid: Sequence[float]) -> str:
    return schedule_grid_hash(grid)


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
            "schedule_grid_hash": _hash_grid(grid),
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
            trace_kwargs = {
                "steps": int(reference_macro_steps),
                "solver": str(solver_name),
                "oracle_local_error": True,
            }
            if cond is not None:
                trace_kwargs["cond"] = cond
            _pred, trace = model.sample_future_trace(hist, **trace_kwargs)
            grid = [float(x) for x in trace["time_grid"].detach().cpu().numpy().astype(np.float64).tolist()]
            if reference_time_grid is None:
                reference_time_grid = grid
            elif not np.allclose(np.asarray(reference_time_grid), np.asarray(grid), atol=1e-8, rtol=1e-8):
                raise ValueError("Reference time grids changed during SER-PTG trace collection.")
            oracle = trace["oracle_local_error"].detach().cpu().numpy().astype(np.float64)
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
            _pred, trace = model.sample_future_trace(
                hist_t,
                steps=int(reference_macro_steps),
                solver=str(solver_name),
                oracle_local_error=True,
            )
            grid = [float(x) for x in trace["time_grid"].detach().cpu().numpy().astype(np.float64).tolist()]
            if reference_time_grid is None:
                reference_time_grid = grid
            elif not np.allclose(np.asarray(reference_time_grid), np.asarray(grid), atol=1e-8, rtol=1e-8):
                raise ValueError("Reference time grids changed during molecule SER-PTG trace collection.")
            oracle = trace["oracle_local_error"].detach().cpu().numpy().astype(np.float64)
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
    dataset = str(args.dataset)
    family = scenario_family_for_key(dataset)
    solvers = list(normalize_solver_keys(str(args.solver_names)))
    target_nfes = _parse_int_csv(str(args.target_nfe_values))
    seeds = _parse_int_csv(str(args.seeds))
    reference_split = str(args.reference_split)
    if reference_split not in {TRAIN_TUNING_PHASE, VALIDATION_PHASE}:
        raise ValueError(f"reference_split must be {TRAIN_TUNING_PHASE!r} or {VALIDATION_PHASE!r}.")
    if bool(args.smoke):
        solvers = solvers[:1]
        target_nfes = target_nfes[:1]
        args.val_windows = min(max(1, int(args.val_windows) if int(args.val_windows) > 0 else 2), 2)
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
            dataset=dataset,
            device=device,
        )
        member_refs.append({"checkpoint": checkpoint, "reference_ds": checkpoint["splits"][split_key], "member_key": "", "stratum": ""})
    elif family == SCENARIO_FAMILY_CONDITIONAL_GENERATION:
        if not hasattr(args, "steps") or int(getattr(args, "steps", 0) or 0) <= 0:
            args.steps = int(getattr(args, "otflow_train_steps", 0) or 0)
        checkpoint = load_conditional_generation_checkpoint_splits(
            cli_args=args,
            shared_backbone_root=shared_backbone_root,
            dataset=dataset,
            device=device,
        )
        member_refs.append({"checkpoint": checkpoint, "reference_ds": checkpoint["splits"][split_key], "member_key": "", "stratum": ""})
    elif family == SCENARIO_FAMILY_MOLECULE:
        group_root = resolve_project_path(str(getattr(args, "molecule_group_root", "") or default_molecule_group_root()))
        manifest = load_molecule_group_manifest(dataset, group_root)
        backbone_manifest = load_backbone_manifest(resolve_project_path(str(args.backbone_manifest)))
        for member in trainable_molecule_group_members(manifest):
            artifact = find_backbone_artifact(
                backbone_manifest,
                backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
                benchmark_family=MOLECULE_FAMILY,
                dataset_key=dataset,
                train_steps=int(args.otflow_train_steps),
                member_key=str(member["member_key"]),
                stratum=str(member["stratum"]),
            )
            loaded = load_molecule_checkpoint_splits(
                checkpoint_path=str(artifact["checkpoint_path"]),
                dataset_key=dataset,
                stratum=str(member["stratum"]),
                processed_dir=group_root / dataset / str(member["processed_dir"]),
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
            raise ValueError(f"Molecule group {dataset!r} has no trainable members for SER-PTG.")
    else:
        raise ValueError(f"Unsupported SER-PTG scenario family: {family!r}")
    train_tuning_reference_examples = int(max(len(ref["checkpoint"]["splits"].get("val", [])) for ref in member_refs))
    predictions: List[Dict[str, Any]] = []
    for solver_idx, solver_key in enumerate(solvers):
        solver_name = str(SOLVER_RUNTIME_NAMES[str(solver_key)])
        solver_p = solver_order_for_ptg(str(solver_key))
        for target_idx, target_nfe in enumerate(target_nfes):
            macro_steps = solver_macro_steps(str(solver_key), int(target_nfe))
            reference_macro_steps = resolve_reference_macro_steps(
                0,
                int(macro_steps),
                reference_macro_factor=float(args.reference_macro_factor),
            )
            seed_traces: List[np.ndarray] = []
            reference_grid: Optional[List[float]] = None
            eval_examples = 0
            trace_count = 0
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
                            chosen = choose_forecast_train_tuning_indices(
                                reference_ds,
                                fraction=float(args.train_tuning_fraction),
                                seed=selection_seed,
                                strata=int(args.train_tuning_strata),
                                dataset=dataset,
                                sampling_mode=str(args.train_tuning_sampling_mode),
                                reference_examples=int(train_tuning_reference_examples),
                                train_split_fraction=float(args.train_tuning_train_split_fraction),
                                val_split_fraction=float(args.train_tuning_val_split_fraction),
                            )
                        else:
                            chosen = choose_forecast_example_indices(
                                reference_ds,
                                n_examples=int(args.val_windows),
                                seed=selection_seed,
                            )
                        item_getter = None
                        collector = collect_batched_local_defect_trace
                    elif family == SCENARIO_FAMILY_CONDITIONAL_GENERATION:
                        horizon = resolved_eval_horizon(args, dataset)
                        available = len(getattr(reference_ds, "start_indices", []))
                        window_count = int(args.val_windows) if int(args.val_windows) > 0 else int(args.context_sample_count)
                        chosen = _choose_valid_windows(
                            reference_ds,
                            horizon=int(horizon),
                            n_windows=min(max(1, int(window_count)), max(1, int(available))),
                            seed=selection_seed,
                        )
                        item_getter = lambda ds, t0: _get_dataset_item_by_t(ds, int(t0))
                        collector = collect_batched_local_defect_trace
                    else:
                        window_count = int(args.val_windows) if int(args.val_windows) > 0 else int(args.context_sample_count)
                        chosen = _choose_molecule_indices(reference_ds, count=int(window_count), seed=selection_seed)
                        item_getter = None
                        collector = collect_molecule_local_defect_trace
                    if bool(args.smoke):
                        chosen = chosen[: min(2, len(chosen))]
                    collected = collector(
                        model,
                        reference_ds,
                        cfg,
                        solver_name=solver_name,
                        reference_macro_steps=int(reference_macro_steps),
                        solver_order_p=float(solver_p),
                        seed=int(seed) + 100_000 * int(member_idx) + 10_000 * int(solver_idx) + 1_000 * int(target_idx),
                        example_indices=chosen,
                        calibration_trace_samples=int(args.calibration_trace_samples),
                        batch_size=int(args.calibration_batch_size),
                        **({"item_getter": item_getter} if item_getter is not None else {}),
                    )
                    grid = [float(x) for x in collected["reference_time_grid"]]
                    if reference_grid is None:
                        reference_grid = grid
                    elif not np.allclose(np.asarray(reference_grid), np.asarray(grid), atol=1e-8, rtol=1e-8):
                        raise ValueError(f"Reference grids changed for {solver_key}/{target_nfe}.")
                    seed_traces.append(np.asarray(collected["local_defect_trace"], dtype=np.float64))
                    eval_examples += int(collected["eval_examples"])
                    trace_count += int(collected["trace_count"])
            if reference_grid is None:
                raise ValueError(f"No reference grid collected for {solver_key}/{target_nfe}.")
            mean_trace = np.mean(np.stack(seed_traces, axis=0), axis=0)
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
                    "checkpoint_step": int(args.otflow_train_steps),
                    "train_steps": int(args.otflow_train_steps),
                    "runtime_nfe": int(macro_steps),
                    "macro_steps": int(macro_steps),
                    "realized_nfe": int(macro_steps) * int(solver_eval_multiplier(str(solver_key))),
                    "time_grid": list(time_grid),
                    "schedule_grid_hash": _hash_grid(time_grid),
                    "reference_macro_steps": int(reference_macro_steps),
                    "reference_time_grid": reference_grid,
                    "local_defect_trace": [float(x) for x in mean_trace.tolist()],
                    "trace_variant": str(args.trace_variant),
                    "density_floor_eta": float(args.density_floor_eta),
                    "solver_order_p": float(solver_p),
                    "reference_split": reference_split,
                    "reference_examples_per_seed": int(eval_examples),
                    "train_tuning_fraction": float(args.train_tuning_fraction) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_seed": int(args.train_tuning_seed) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_strata": int(args.train_tuning_strata) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_sampler": train_tuning_sampler_key(str(args.train_tuning_sampling_mode)) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_sampling_mode": str(args.train_tuning_sampling_mode) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_reference_examples": int(train_tuning_reference_examples) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_target_examples": int(eval_examples) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_train_split_fraction": float(args.train_tuning_train_split_fraction) if reference_split == TRAIN_TUNING_PHASE else "",
                    "train_tuning_val_split_fraction": float(args.train_tuning_val_split_fraction) if reference_split == TRAIN_TUNING_PHASE else "",
                    "trace_count": int(trace_count),
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
        "trace_variant": str(args.trace_variant),
        "density_floor_eta": float(args.density_floor_eta),
        "predictions": derived_predictions[SER_PTG_SCHEDULE_KEY],
    }
    schedules = [
        schedule,
        {
            "scheduler_key": SER_PTG_REVERSED_SCHEDULE_KEY,
            "schedule_name": f"{SER_PTG_SCHEDULE_NAME} reversed",
            "comparison_role": "reference_comparator",
            "trace_variant": str(args.trace_variant),
            "density_floor_eta": float(args.density_floor_eta),
            "schedule_derivation": "reverse_physical_clock",
            "predictions": derived_predictions[SER_PTG_REVERSED_SCHEDULE_KEY],
        },
        {
            "scheduler_key": SER_PTG_AVG_REVERSED_SCHEDULE_KEY,
            "schedule_name": f"{SER_PTG_SCHEDULE_NAME} density average",
            "comparison_role": "reference_comparator",
            "trace_variant": str(args.trace_variant),
            "density_floor_eta": float(args.density_floor_eta),
            "schedule_derivation": "average_physical_and_reversed_density_mass",
            "predictions": derived_predictions[SER_PTG_AVG_REVERSED_SCHEDULE_KEY],
        },
    ]
    summary = {
        "status": "ready",
        "artifact": "ser_ptg_schedule_summary",
        "dataset": dataset,
        "schedule_key": SER_PTG_SCHEDULE_KEY,
        "schedule_name": SER_PTG_SCHEDULE_NAME,
        "schedules": schedules,
        "baseline_schedule": False,
        "seeds": [int(seed) for seed in seeds],
        "solver_names": solvers,
        "target_nfe_values": [int(nfe) for nfe in target_nfes],
        "trace_variant": str(args.trace_variant),
        "reference_split": reference_split,
        "reference_split_key": split_key,
        "density_floor_eta": float(args.density_floor_eta),
        "reference_macro_factor": float(args.reference_macro_factor),
        "calibration_trace_samples": int(args.calibration_trace_samples),
        "reference_examples": int(sum(len(ref["reference_ds"]) for ref in member_refs)),
        "train_tuning": {
            "fraction": float(args.train_tuning_fraction),
            "seed": int(args.train_tuning_seed),
            "strata": int(args.train_tuning_strata),
            "sampling_mode": str(args.train_tuning_sampling_mode),
            "sampler": train_tuning_sampler_key(str(args.train_tuning_sampling_mode)),
            "reference_split_key": "val",
            "reference_examples": int(train_tuning_reference_examples),
            "train_split_fraction": float(args.train_tuning_train_split_fraction),
            "val_split_fraction": float(args.train_tuning_val_split_fraction),
        } if reference_split == TRAIN_TUNING_PHASE else None,
        "checkpoint_step": int(args.otflow_train_steps),
        "checkpoint_ids": sorted(str(ref["checkpoint"].get("checkpoint_id", "")) for ref in member_refs),
        "member_keys": sorted(str(ref.get("member_key", "")) for ref in member_refs if str(ref.get("member_key", ""))),
        "predictions": predictions,
    }
    save_json(summary, str(out_dir / "ser_ptg_schedule_summary.json"))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local-defect SER-PTG reference schedules for genODE.")
    parser.add_argument("--dataset", default="traffic_hourly")
    parser.add_argument("--trace_variant", default="local_defect", choices=("local_defect",))
    parser.add_argument("--density_floor_eta", type=float, default=0.05)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--solver_names", default=",".join(DEFAULT_SOLVERS))
    parser.add_argument("--target_nfe_values", default=",".join(str(x) for x in DEFAULT_TARGET_NFES))
    parser.add_argument("--reference_split", choices=(TRAIN_TUNING_PHASE, VALIDATION_PHASE), default=TRAIN_TUNING_PHASE)
    parser.add_argument("--val_windows", type=int, default=0)
    parser.add_argument("--context_sample_count", type=int, default=CANONICAL_CONTEXT_SAMPLE_COUNT)
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
    parser.add_argument("--dataset_root", default=str(project_paper_dataset_root()))
    parser.add_argument("--shared_backbone_root", default=str(DEFAULT_SHARED_BACKBONE_ROOT))
    parser.add_argument("--backbone_manifest", default=str(default_backbone_manifest_path()))
    parser.add_argument("--cryptos_path", default=str(default_cryptos_data_path()))
    parser.add_argument("--lobster_synthetic_profile_path", default=str(default_lobster_synthetic_profile_path()))
    parser.add_argument("--long_term_st_path", default=str(default_long_term_st_data_path()))
    parser.add_argument("--molecule_group_root", default=str(default_molecule_group_root()))
    parser.add_argument("--otflow_train_steps", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--fu_net_layers", type=int, default=3)
    parser.add_argument("--fu_net_heads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", default=False)
    return parser


def main() -> None:
    summary = build_ser_ptg_reference(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
