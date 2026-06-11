from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from genode.gipo.models import validate_time_grid
from genode.data.otflow_paths import (
    default_backbone_manifest_path,
    project_outputs_root,
    project_paper_dataset_root,
    resolve_project_path,
)
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
    load_forecast_checkpoint_splits,
    resolve_reference_macro_steps,
    solver_eval_multiplier,
    solver_macro_steps,
    train_tuning_sampler_key,
)
from genode.models.otflow_train_val import save_json, seed_all
from genode.runtime import resolve_torch_device

SER_PTG_SCHEDULE_KEY = "ser_ptg_local_defect_eta005"
SER_PTG_SCHEDULE_NAME = "SER-PTG local defect eta=0.05"
DEFAULT_SOLVERS: Tuple[str, ...] = ("euler", "heun", "midpoint_rk2", "dpmpp2m")
DEFAULT_TARGET_NFES: Tuple[int, ...] = (4, 8, 12)


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in _parse_csv(text)]


def _hash_grid(grid: Sequence[float]) -> str:
    return hashlib.sha256(json.dumps([float(x) for x in grid], separators=(",", ":")).encode("utf-8")).hexdigest()


def solver_order_for_ptg(solver_key: str) -> float:
    if str(solver_key) == "euler":
        return 1.0
    if str(solver_key) in {"heun", "midpoint_rk2", "dpmpp2m"}:
        return 2.0
    raise ValueError(f"Unsupported solver key {solver_key!r}.")


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


def _parse_forecast_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any]:
    if len(batch) == 3:
        hist, tgt, meta = batch
        return hist, tgt, None, meta
    if len(batch) == 4:
        hist, tgt, fut, meta = batch
        return hist, tgt, fut, meta
    raise ValueError(f"Unexpected forecast batch format with {len(batch)} items.")


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
            for example_idx in chunk:
                hist_t, _tgt_t, _fut_t, _meta = _parse_forecast_batch(ds[int(example_idx)])
                hist_rows.append(hist_t.float())
            hist = torch.stack(hist_rows, dim=0).to(device).float()
            seed_all(int(seed) + 1_000_000 * int(sample_idx) + int(chunk_start))
            _pred, trace = model.sample_future_trace(
                hist,
                steps=int(reference_macro_steps),
                solver=str(solver_name),
                oracle_local_error=True,
            )
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


def build_ser_ptg_reference(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = str(args.dataset)
    solvers = _parse_csv(str(args.solver_names))
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
    checkpoint = load_forecast_checkpoint_splits(
        cli_args=args,
        dataset_root=dataset_root,
        shared_backbone_root=shared_backbone_root,
        dataset=dataset,
        device=device,
    )
    model = checkpoint["model"]
    cfg = checkpoint["cfg"]
    split_key = "train" if reference_split == TRAIN_TUNING_PHASE else "val"
    reference_ds = checkpoint["splits"][split_key]
    train_tuning_reference_examples = int(len(checkpoint["splits"].get("val", [])))
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
                if reference_split == TRAIN_TUNING_PHASE:
                    chosen = choose_forecast_train_tuning_indices(
                        reference_ds,
                        fraction=float(args.train_tuning_fraction),
                        seed=int(args.train_tuning_seed) + int(seed) + 10_000 * int(solver_idx) + 1_000 * int(target_idx),
                        strata=int(args.train_tuning_strata),
                        dataset=dataset,
                        sampling_mode=str(args.train_tuning_sampling_mode),
                        reference_examples=int(train_tuning_reference_examples),
                        train_split_fraction=float(args.train_tuning_train_split_fraction),
                        val_split_fraction=float(args.train_tuning_val_split_fraction),
                    )
                    if bool(args.smoke):
                        chosen = chosen[: min(2, len(chosen))]
                else:
                    chosen = choose_forecast_example_indices(
                        reference_ds,
                        n_examples=int(args.val_windows),
                        seed=int(seed) + 10_000 * int(solver_idx) + 1_000 * int(target_idx),
                    )
                collected = collect_batched_local_defect_trace(
                    model,
                    reference_ds,
                    cfg,
                    solver_name=solver_name,
                    reference_macro_steps=int(reference_macro_steps),
                    solver_order_p=float(solver_p),
                    seed=int(seed) + 100_000 * int(solver_idx) + 10_000 * int(target_idx),
                    example_indices=chosen,
                    calibration_trace_samples=int(args.calibration_trace_samples),
                    batch_size=int(args.calibration_batch_size),
                )
                grid = [float(x) for x in collected["reference_time_grid"]]
                if reference_grid is None:
                    reference_grid = grid
                elif not np.allclose(np.asarray(reference_grid), np.asarray(grid), atol=1e-8, rtol=1e-8):
                    raise ValueError(f"Reference grids changed for {solver_key}/{target_nfe}.")
                seed_traces.append(np.asarray(collected["local_defect_trace"], dtype=np.float64))
                eval_examples = int(collected["eval_examples"])
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
    schedule = {
        "scheduler_key": SER_PTG_SCHEDULE_KEY,
        "schedule_name": SER_PTG_SCHEDULE_NAME,
        "comparison_role": "reference_comparator",
        "trace_variant": str(args.trace_variant),
        "density_floor_eta": float(args.density_floor_eta),
        "predictions": predictions,
    }
    summary = {
        "status": "ready",
        "artifact": "ser_ptg_schedule_summary",
        "dataset": dataset,
        "schedule_key": SER_PTG_SCHEDULE_KEY,
        "schedule_name": SER_PTG_SCHEDULE_NAME,
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
        "reference_examples": int(len(reference_ds) if reference_split == TRAIN_TUNING_PHASE else (len(reference_ds) if int(args.val_windows) <= 0 else min(int(args.val_windows), len(reference_ds)))),
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
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "checkpoint_path": str(checkpoint["checkpoint_path"]),
        "schedules": [schedule],
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
    parser.add_argument("--otflow_train_steps", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", default=False)
    return parser


def main() -> None:
    summary = build_ser_ptg_reference(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
