from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from genode.experiment_layout import REFERENCE_SEEN_NFES
from genode.data.otflow_paths import display_project_path, project_root
from genode.gipo.schema import reject_retired_evaluation_keys
from genode.solver_protocol import (
    SOLVER_RUNTIME_NAMES,
    solver_macro_steps,
    target_nfe_for_macro_steps,
)

PROJECT_ROOT = project_root()
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUT_ROOT / "ptg_observed_gain"
DEFAULT_INPUT_JSON = DEFAULT_RESULTS_DIR / "ptg_observed_gain_inputs.json"
DEFAULT_POINTS_CSV = DEFAULT_RESULTS_DIR / "ptg_observed_gain_points.csv"
DEFAULT_DIAGNOSTICS_JSON = DEFAULT_RESULTS_DIR / "ptg_diagnostics.json"
DEFAULT_INTEGRATION_ROWS_CSV = DEFAULT_RESULTS_DIR / "ptg_integration_error_rows.csv"
DEFAULT_INTEGRATION_SEED_STATS_CSV = DEFAULT_RESULTS_DIR / "ptg_integration_error_seed_stats.csv"
DEFAULT_ZIP_PATH = DEFAULT_OUTPUT_ROOT / "20k.zip"
DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_ROOT / "figures"
DEFAULT_PNG = DEFAULT_FIGURE_DIR / "ptg_vs_observed_gain_forecast_20k_times_600dpi.png"
DEFAULT_PDF = DEFAULT_FIGURE_DIR / "ptg_vs_observed_gain_forecast_20k_times_600dpi.pdf"

RELATIVE_STATS_NAME = "20k/seed_stats/forecast_baseline_relative_seed_stats.csv"
NATIVE_TRACE_KEY = "info_growth_hardness_by_step"
ORACLE_LOCAL_ERROR_TRACE_KEY = "oracle_local_error_by_step"
LOCAL_DEFECT_TRACE_KEY = "validation_local_defect_trace"
INFO_GROWTH_TRACE_KEY = "validation_info_growth_trace"
DEFAULT_DENSITY_FLOOR_ETA = 0.05
DEFAULT_MAIN_PTG_KEY = "ptg_info_growth_raw"
PTG_X_LABELS: Dict[str, str] = {
    "ptg_info_growth_raw": "Info-growth PTG",
    "ptg_info_growth_reversed": "Info-growth PTG, reversed time",
    "ptg_local_defect_eta005": "Local-defect PTG",
    "ptg_local_defect_reversed_eta005": "Local-defect PTG, reversed time",
}

DATASET_ORDER: Tuple[str, ...] = (
    "solar_energy_10m",
    "traffic_hourly",
    "weather_daily",
)
DATASET_LABELS: Dict[str, str] = {
    "solar_energy_10m": "Solar",
    "traffic_hourly": "Traffic",
    "weather_daily": "Weather",
}
SOLVER_ORDER: Tuple[str, ...] = ("euler", "heun", "midpoint_rk2", "dpmpp2m")
SOLVER_LABELS: Dict[str, str] = {
    "euler": "Euler",
    "heun": "Heun",
    "midpoint_rk2": "Midpoint RK2",
    "dpmpp2m": "DPM++2M",
}
TARGET_NFES: Tuple[int, ...] = REFERENCE_SEEN_NFES
TRANSFER_SCHEDULES: Tuple[str, ...] = ("ays", "gits", "ots")
INTEGRATION_SCHEDULES: Tuple[str, ...] = ("uniform", *TRANSFER_SCHEDULES)
SCHEDULE_LABELS: Dict[str, str] = {
    "uniform": "uniform",
    "ays": "AYS",
    "gits": "GITS",
    "ots": "OTS",
}
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_REFERENCE_MACRO_FACTOR = 4.0
DEFAULT_DENSE_REFERENCE_MACRO_FACTOR = 16.0
DEFAULT_CALIBRATION_TRACE_SAMPLES = 1
DEFAULT_VALIDATION_WINDOWS = 20
DEFAULT_TEST_WINDOWS = 0


class _IndexSubset:
    def __init__(self, base: Any, indices: Sequence[int]):
        self.base = base
        self.indices = [int(idx) for idx in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Any:
        return self.base[self.indices[int(item)]]


@dataclass(frozen=True)
class PtgResult:
    ser: float
    ptg_percent: float
    kappa_integral: float
    rho_integral: float


def parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in parse_csv(text)]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_1d(values: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def validate_time_grid(grid: Sequence[float], *, name: str) -> np.ndarray:
    arr = _finite_1d(grid, name=name)
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two nodes.")
    if abs(float(arr[0])) > 1e-8:
        raise ValueError(f"{name} must start at 0.0.")
    if abs(float(arr[-1]) - 1.0) > 1e-8:
        raise ValueError(f"{name} must end at 1.0.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return arr


def normalize_hardness_for_ptg(
    hardness: Sequence[float],
    reference_time_grid: Sequence[float],
    *,
    eps_multiplier: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    h = np.clip(_finite_1d(hardness, name="hardness"), 0.0, None)
    grid = validate_time_grid(reference_time_grid, name="reference_time_grid")
    if grid.size != h.size + 1:
        raise ValueError(
            "reference_time_grid must have length len(hardness) + 1 "
            f"({h.size + 1}), got {grid.size}."
        )
    widths = np.diff(grid)
    eps_h = float(eps_multiplier) * max(float(np.mean(h)), 1e-12)
    weighted = h + eps_h
    denom = float(np.sum(widths * weighted))
    if denom <= 0.0:
        raise ValueError("Hardness normalization denominator must be positive.")
    kappa = weighted / denom
    integral = float(np.sum(widths * kappa))
    return kappa, widths, float(eps_h), integral


def schedule_density_on_reference_grid(
    schedule_grid: Sequence[float],
    reference_time_grid: Sequence[float],
    *,
    min_density: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    schedule = validate_time_grid(schedule_grid, name="schedule_grid")
    reference = validate_time_grid(reference_time_grid, name="reference_time_grid")
    ref_widths = np.diff(reference)
    step_widths = np.diff(schedule)
    n_steps = int(step_widths.size)
    local_density = 1.0 / (float(n_steps) * step_widths)
    rho = np.zeros(int(ref_widths.size), dtype=np.float64)
    step_idx = 0
    for ref_idx, (left, right) in enumerate(zip(reference[:-1], reference[1:])):
        left_f = float(left)
        right_f = float(right)
        while step_idx < n_steps - 1 and float(schedule[step_idx + 1]) <= left_f + 1e-14:
            step_idx += 1
        cursor = step_idx
        total = 0.0
        while cursor < n_steps and float(schedule[cursor]) < right_f - 1e-14:
            overlap = max(
                0.0,
                min(right_f, float(schedule[cursor + 1])) - max(left_f, float(schedule[cursor])),
            )
            if overlap > 0.0:
                total += overlap * float(local_density[cursor])
            if float(schedule[cursor + 1]) >= right_f - 1e-14:
                break
            cursor += 1
        rho[ref_idx] = max(float(total) / max(float(ref_widths[ref_idx]), 1e-14), float(min_density))
    integral = float(np.sum(ref_widths * rho))
    return rho, integral


def stabilize_density(
    density: Sequence[float],
    reference_time_grid: Sequence[float],
    *,
    eta: float,
) -> Tuple[np.ndarray, float]:
    rho = _finite_1d(density, name="density")
    grid = validate_time_grid(reference_time_grid, name="reference_time_grid")
    if grid.size != rho.size + 1:
        raise ValueError("reference_time_grid must have length len(density) + 1.")
    eta_value = float(eta)
    if eta_value < 0.0 or eta_value > 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}.")
    stabilized = (1.0 - eta_value) * rho + eta_value
    integral = float(np.sum(np.diff(grid) * stabilized))
    return stabilized, integral


def reverse_schedule_grid(schedule_grid: Sequence[float]) -> List[float]:
    grid = validate_time_grid(schedule_grid, name="schedule_grid")
    reversed_grid = 1.0 - grid[::-1]
    reversed_grid[0] = 0.0
    reversed_grid[-1] = 1.0
    return [float(x) for x in validate_time_grid(reversed_grid, name="reversed_schedule_grid").tolist()]


def local_defect_trace_from_oracle(
    oracle_local_error: Sequence[float],
    reference_time_grid: Sequence[float],
    *,
    solver_order_p: float,
    eps: float = 1e-12,
) -> List[float]:
    oracle = np.clip(_finite_1d(oracle_local_error, name="oracle_local_error"), 0.0, None)
    grid = validate_time_grid(reference_time_grid, name="reference_time_grid")
    if grid.size != oracle.size + 1:
        raise ValueError(
            "reference_time_grid must have length len(oracle_local_error) + 1 "
            f"({oracle.size + 1}), got {grid.size}."
        )
    p = float(solver_order_p)
    if p <= 0.0:
        raise ValueError(f"solver_order_p must be positive, got {solver_order_p}.")
    widths = np.diff(grid)
    denom = np.power(widths, p + 1.0) + float(eps)
    return [float(x) for x in (oracle / denom).tolist()]


def ptg_from_trace(
    hardness: Sequence[float],
    reference_time_grid: Sequence[float],
    schedule_grid: Sequence[float],
    *,
    solver_order_p: float,
    density_floor_eta: float = 0.0,
) -> PtgResult:
    kappa, widths, _eps_h, kappa_integral = normalize_hardness_for_ptg(hardness, reference_time_grid)
    rho, rho_integral = schedule_density_on_reference_grid(schedule_grid, reference_time_grid)
    if float(density_floor_eta) > 0.0:
        rho, rho_integral = stabilize_density(rho, reference_time_grid, eta=float(density_floor_eta))
    p = float(solver_order_p)
    if p <= 0.0:
        raise ValueError(f"solver_order_p must be positive, got {solver_order_p}.")
    ser = float(np.sum(widths * kappa * np.power(rho, -p)))
    return PtgResult(
        ser=ser,
        ptg_percent=float(100.0 * (1.0 - ser)),
        kappa_integral=float(kappa_integral),
        rho_integral=float(rho_integral),
    )


def mean_trace(traces: Sequence[Sequence[float]], *, name: str) -> List[float]:
    if not traces:
        raise ValueError(f"{name} must contain at least one trace.")
    arrays = [_finite_1d(trace, name=f"{name}[{idx}]") for idx, trace in enumerate(traces)]
    shape = arrays[0].shape
    for idx, arr in enumerate(arrays[1:], start=1):
        if arr.shape != shape:
            raise ValueError(f"{name}[{idx}] has shape {arr.shape}, expected {shape}.")
    return [float(x) for x in np.mean(np.stack(arrays, axis=0), axis=0).tolist()]


def _select_indices(length: int, n_windows: int, seed: int) -> List[int]:
    length = int(length)
    n_windows = int(n_windows)
    if length <= 0:
        raise ValueError("Cannot sample indices from an empty dataset.")
    if n_windows <= 0:
        n_windows = length
    rng = np.random.default_rng(int(seed))
    replace = n_windows > length
    return [int(x) for x in rng.choice(np.arange(length, dtype=np.int64), size=n_windows, replace=replace).tolist()]


def _project_relative_path(raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _runner_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    from genode.evaluation.diffusion_flow_time_reparameterization import build_argparser

    datasets = parse_csv(str(getattr(args, "scenario_keys", ",".join(DATASET_ORDER))))
    solvers = parse_csv(str(getattr(args, "solvers", ",".join(SOLVER_ORDER))))
    target_nfes = parse_int_csv(str(getattr(args, "target_nfes", ",".join(str(nfe) for nfe in TARGET_NFES))))
    seeds = parse_int_csv(str(getattr(args, "seeds", ",".join(str(seed) for seed in DEFAULT_SEEDS))))
    argv = [
        "--out_root",
        str(DEFAULT_RESULTS_DIR / "_collect_runner_unused"),
        "--forecast_datasets",
        ",".join(datasets),
        "--conditional_generation_datasets",
        "",
        "--target_nfe_values",
        ",".join(str(nfe) for nfe in target_nfes),
        "--solver_names",
        ",".join(solvers),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--device",
        str(getattr(args, "device", "cuda")),
        "--checkpoint_steps",
        "20000",
        "--num_eval_samples",
        "1",
        "--calibration_trace_samples",
        str(int(getattr(args, "calibration_trace_samples", DEFAULT_CALIBRATION_TRACE_SAMPLES))),
        "--eval_windows_val",
        str(int(getattr(args, "val_windows", DEFAULT_VALIDATION_WINDOWS))),
    ]
    backbone_manifest = str(getattr(args, "backbone_manifest", "backbone_manifest.json"))
    if backbone_manifest.strip():
        argv.extend(["--backbone_manifest", str(_project_relative_path(backbone_manifest))])
    return build_argparser().parse_args(argv)


def solver_runtime_name(solver_key: str) -> str:
    key = str(solver_key)
    if key not in SOLVER_RUNTIME_NAMES:
        raise ValueError(f"Unsupported solver key {solver_key!r}.")
    return str(SOLVER_RUNTIME_NAMES[key])


def build_fixed_schedule_grid(schedule_key: str, macro_steps: int) -> List[float]:
    from genode.schedule_transfer.diffusion_flow_schedules import build_schedule_grid

    grid = build_schedule_grid(str(schedule_key), int(macro_steps))
    if grid is None:
        raise ValueError(f"Unable to build schedule grid for {schedule_key!r}.")
    arr = validate_time_grid(grid, name=f"{schedule_key}_grid")
    expected_len = int(macro_steps) + 1
    if arr.size != expected_len:
        raise ValueError(f"{schedule_key} grid must contain {expected_len} nodes, got {arr.size}.")
    return [float(x) for x in arr.tolist()]


def solver_order_for_ptg(solver_key: str) -> float:
    key = str(solver_key)
    if key == "euler":
        return 1.0
    if key in {"heun", "midpoint_rk2", "dpmpp2m"}:
        return 2.0
    raise ValueError(f"Unsupported solver key {solver_key!r}.")


def collect_payload(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from genode.evaluation.otflow_evaluation_support import (
        collect_forecast_calibration,
        load_forecast_checkpoint_splits,
        resolve_reference_macro_steps,
    )
    from genode.data.otflow_paths import project_dataset_root
    from genode.runtime import resolve_torch_device

    datasets = parse_csv(str(args.datasets))
    solvers = parse_csv(str(args.solvers))
    target_nfes = parse_int_csv(str(args.target_nfes))
    seeds = parse_int_csv(str(args.seeds))
    if bool(args.smoke):
        datasets = datasets[:1]
        solvers = solvers[:1]
        target_nfes = target_nfes[:1]
        seeds = seeds[:1]
        args.val_windows = min(int(args.val_windows), 2)
    unknown_datasets = [dataset for dataset in datasets if dataset not in DATASET_ORDER]
    unknown_solvers = [solver for solver in solvers if solver not in SOLVER_ORDER]
    unknown_nfes = [nfe for nfe in target_nfes if int(nfe) not in TARGET_NFES]
    if unknown_datasets:
        raise ValueError(f"Unsupported datasets: {unknown_datasets}")
    if unknown_solvers:
        raise ValueError(f"Unsupported solvers: {unknown_solvers}")
    if unknown_nfes:
        raise ValueError(f"Unsupported target NFEs: {unknown_nfes}")

    cli_args = _runner_cli_args(args)
    device = resolve_torch_device(str(args.device))
    cells: List[Dict[str, Any]] = []
    dataset_root = project_dataset_root()
    for dataset in datasets:
        dataset_idx = DATASET_ORDER.index(str(dataset))
        checkpoint = load_forecast_checkpoint_splits(
            cli_args=cli_args,
            dataset_root=dataset_root,
            shared_backbone_root=Path(str(cli_args.shared_backbone_root)).resolve(),
            dataset=str(dataset),
            device=device,
        )
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        with torch.no_grad():
            for solver_key in solvers:
                runtime_solver = solver_runtime_name(str(solver_key))
                for target_nfe in target_nfes:
                    macro_steps = int(solver_macro_steps(str(solver_key), int(target_nfe)))
                    reference_macro_steps = int(
                        resolve_reference_macro_steps(
                            0,
                            macro_steps,
                            reference_macro_factor=float(args.reference_macro_factor),
                        )
                    )
                    per_seed: List[Dict[str, Any]] = []
                    info_growth_traces: List[List[float]] = []
                    oracle_traces: List[List[float]] = []
                    local_defect_traces: List[List[float]] = []
                    reference_time_grid: Optional[List[float]] = None
                    solver_p = solver_order_for_ptg(str(solver_key))
                    for seed in seeds:
                        seed = int(seed)
                        val_indices = _select_indices(
                            len(splits["val"]),
                            int(args.val_windows),
                            seed + 10_000 * int(dataset_idx),
                        )
                        calibration_seed = seed + 100_000 * int(dataset_idx)
                        calibration = collect_forecast_calibration(
                            model,
                            _IndexSubset(splits["val"], val_indices),
                            cfg,
                            macro_steps=int(reference_macro_steps),
                            solver_name=runtime_solver,
                            seed=int(calibration_seed),
                            calibration_trace_samples=int(args.calibration_trace_samples),
                        )
                        trace = [float(x) for x in calibration[NATIVE_TRACE_KEY]]
                        oracle_trace = [float(x) for x in calibration[ORACLE_LOCAL_ERROR_TRACE_KEY]]
                        grid = [float(x) for x in calibration["reference_time_grid"]]
                        if reference_time_grid is None:
                            reference_time_grid = grid
                        elif not np.allclose(reference_time_grid, grid, atol=1e-8, rtol=1e-8):
                            raise ValueError(
                                f"Reference grids differ for {dataset}/{solver_key}/NFE={target_nfe}."
                            )
                        local_defect_trace = local_defect_trace_from_oracle(
                            oracle_trace,
                            grid,
                            solver_order_p=float(solver_p),
                        )
                        info_growth_traces.append(trace)
                        oracle_traces.append(oracle_trace)
                        local_defect_traces.append(local_defect_trace)
                        per_seed.append(
                            {
                                "seed": int(seed),
                                "validation_indices": val_indices,
                                "validation_windows": int(len(val_indices)),
                                "validation_trace": trace,
                                "validation_info_growth_trace": trace,
                                "validation_oracle_local_error_trace": oracle_trace,
                                "validation_local_defect_trace": local_defect_trace,
                                "info_growth_scale": float(calibration["info_growth_scale"]),
                                "base_info_growth_scale": float(calibration["base_info_growth_scale"]),
                                "signal_validation_spearman": calibration.get("signal_correlations_vs_oracle", {})
                                .get(NATIVE_TRACE_KEY, {})
                                .get("spearman"),
                            }
                        )
                    if reference_time_grid is None:
                        raise ValueError(f"No reference grid collected for {dataset}/{solver_key}/NFE={target_nfe}.")
                    mean_info_growth = mean_trace(info_growth_traces, name=f"{dataset}_{solver_key}_{target_nfe}_info_growth")
                    mean_oracle = mean_trace(oracle_traces, name=f"{dataset}_{solver_key}_{target_nfe}_oracle")
                    mean_local_defect = mean_trace(
                        local_defect_traces,
                        name=f"{dataset}_{solver_key}_{target_nfe}_local_defect",
                    )
                    validate_time_grid(reference_time_grid, name="reference_time_grid")
                    if len(reference_time_grid) != len(mean_local_defect) + 1:
                        raise ValueError(
                            f"{dataset}/{solver_key}/NFE={target_nfe} reference grid length "
                            f"{len(reference_time_grid)} does not match hardness length {len(mean_local_defect)}."
                        )
                    cells.append(
                        {
                            "scenario_key": str(dataset),
                            "scenario_label": DATASET_LABELS[str(dataset)],
                            "solver_key": str(solver_key),
                            "solver_label": SOLVER_LABELS[str(solver_key)],
                            "target_nfe": int(target_nfe),
                            "macro_steps": int(macro_steps),
                            "reference_macro_steps": int(reference_macro_steps),
                            "reference_time_grid": reference_time_grid,
                            "validation_hardness_trace": mean_local_defect,
                            "validation_info_growth_trace": mean_info_growth,
                            "validation_oracle_local_error_trace": mean_oracle,
                            "validation_local_defect_trace": mean_local_defect,
                            "per_seed": per_seed,
                            "checkpoint_path": display_project_path(str(checkpoint["checkpoint_path"])),
                            "checkpoint_id": str(checkpoint["checkpoint_id"]),
                            "checkpoint_step": int(checkpoint["checkpoint_step"]),
                            "train_budget_label": str(checkpoint["train_budget_label"]),
                            "backbone_name": str(checkpoint["backbone_name"]),
                        }
                    )
        del model
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "artifact": "ptg_observed_gain_inputs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": "project_root",
        "scenario_keys": datasets,
        "solvers": solvers,
        "target_nfes": [int(x) for x in target_nfes],
        "seeds": [int(x) for x in seeds],
        "validation_windows": int(args.val_windows),
        "reference_macro_factor": float(args.reference_macro_factor),
        "calibration_trace_samples": int(args.calibration_trace_samples),
        "signal_trace_key": NATIVE_TRACE_KEY,
        "report_trace_key": LOCAL_DEFECT_TRACE_KEY,
        "oracle_local_error_trace_key": ORACLE_LOCAL_ERROR_TRACE_KEY,
        "density_floor_eta": float(DEFAULT_DENSITY_FLOOR_ETA),
        "main_ptg_key": DEFAULT_MAIN_PTG_KEY,
        "test_trace_used": False,
        "cells": cells,
    }


def _eval_dataset_for_integration(ds: Any, *, test_windows: int, seed: int) -> Tuple[Any, int]:
    n_examples = int(len(ds))
    if n_examples <= 0:
        raise ValueError("Locked-test split is empty; cannot collect integration error.")
    window_cap = int(test_windows)
    if window_cap <= 0 or window_cap >= n_examples:
        return ds, n_examples
    indices = _select_indices(n_examples, window_cap, int(seed))
    return _IndexSubset(ds, indices), int(len(indices))


def _sample_forecast_endpoints_norm(
    model: Any,
    ds: Any,
    cfg: Any,
    *,
    solver_name: str,
    macro_steps: int,
    time_grid: Sequence[float],
    seed: int,
    batch_size: int = 64,
) -> Tuple[List[np.ndarray], float]:
    import torch
    from genode.models.otflow_train_val import seed_all

    device = cfg.train.device
    endpoints: List[np.ndarray] = []
    elapsed = 0.0
    effective_batch_size = max(1, int(batch_size))
    target_nfe = target_nfe_for_macro_steps(str(solver_name), int(macro_steps))
    for batch_start in range(0, int(len(ds)), effective_batch_size):
        batch_end = min(int(len(ds)), int(batch_start) + effective_batch_size)
        hist_rows = []
        for example_idx in range(batch_start, batch_end):
            hist_t, _, _, _ = ds[int(example_idx)]
            hist_rows.append(hist_t)
        hist = torch.stack(hist_rows, dim=0).to(device).float()
        draw_seed = int(seed) + 1000 * int(batch_start)
        seed_all(draw_seed)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        pred_norm = model.sample_future(
            hist,
            solver_key=str(solver_name),
            target_nfe=target_nfe,
            time_grid=time_grid,
        )
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elapsed += float(time.perf_counter() - start)
        pred_arr = pred_norm.detach().cpu().numpy().astype(np.float64)
        for batch_idx in range(int(pred_arr.shape[0])):
            endpoints.append(pred_arr[batch_idx].reshape(-1))
    return endpoints, elapsed


def _mean_endpoint_l2(endpoints: Sequence[np.ndarray], references: Sequence[np.ndarray]) -> float:
    if len(endpoints) != len(references):
        raise ValueError(f"Endpoint/reference count mismatch: {len(endpoints)} vs {len(references)}.")
    if not endpoints:
        raise ValueError("Cannot compute integration error from an empty endpoint list.")
    errors: List[float] = []
    for idx, (endpoint, reference) in enumerate(zip(endpoints, references)):
        endpoint_arr = np.asarray(endpoint, dtype=np.float64).reshape(-1)
        reference_arr = np.asarray(reference, dtype=np.float64).reshape(-1)
        if endpoint_arr.shape != reference_arr.shape:
            raise ValueError(f"Endpoint/reference shape mismatch at example {idx}: {endpoint_arr.shape} vs {reference_arr.shape}.")
        errors.append(float(np.linalg.norm(endpoint_arr - reference_arr, ord=2)))
    return float(np.mean(np.asarray(errors, dtype=np.float64)))


def _dense_uniform_grid(n_steps: int) -> List[float]:
    steps = int(n_steps)
    if steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}.")
    return [float(i) / float(steps) for i in range(steps + 1)]


def collect_integration_error_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    import torch
    from genode.evaluation.otflow_evaluation_support import (
        load_forecast_checkpoint_splits,
        resolve_reference_macro_steps,
    )
    from genode.data.otflow_paths import project_dataset_root
    from genode.runtime import resolve_torch_device

    datasets = parse_csv(str(args.datasets))
    solvers = parse_csv(str(args.solvers))
    target_nfes = parse_int_csv(str(args.target_nfes))
    seeds = parse_int_csv(str(args.seeds))
    schedules = parse_csv(str(args.schedules))
    if bool(args.smoke):
        datasets = datasets[:1]
        solvers = solvers[:1]
        target_nfes = target_nfes[:1]
        seeds = seeds[:1]
        schedules = schedules[:2]
        args.test_windows = 2
    unknown_datasets = [dataset for dataset in datasets if dataset not in DATASET_ORDER]
    unknown_solvers = [solver for solver in solvers if solver not in SOLVER_ORDER]
    unknown_nfes = [nfe for nfe in target_nfes if int(nfe) not in TARGET_NFES]
    unknown_schedules = [schedule for schedule in schedules if schedule not in INTEGRATION_SCHEDULES]
    if unknown_datasets:
        raise ValueError(f"Unsupported datasets: {unknown_datasets}")
    if unknown_solvers:
        raise ValueError(f"Unsupported solvers: {unknown_solvers}")
    if unknown_nfes:
        raise ValueError(f"Unsupported target NFEs: {unknown_nfes}")
    if unknown_schedules:
        raise ValueError(f"Unsupported schedules: {unknown_schedules}")
    if "uniform" not in schedules:
        raise ValueError("Integration-error collection requires the uniform schedule denominator.")

    cli_args = _runner_cli_args(args)
    device = resolve_torch_device(str(args.device))
    dataset_root = project_dataset_root()
    rows: List[Dict[str, Any]] = []
    completed_seed_keys = set()
    rows_csv = Path(args.rows_csv) if getattr(args, "rows_csv", None) is not None else None
    if bool(getattr(args, "resume", False)) and rows_csv is not None and rows_csv.exists():
        with rows_csv.open("r", newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        reject_retired_evaluation_keys(rows, source=f"Integration-error rows from {rows_csv.name}")
        grouped_schedules: Dict[Tuple[str, str, int, int], set] = defaultdict(set)
        for row in rows:
            grouped_schedules[
                (
                    str(row["scenario_key"]),
                    str(row["solver_key"]),
                    int(row["target_nfe"]),
                    int(row["evaluation_seed"]),
                )
            ].add(str(row["scheduler_key"]))
        required = set(schedules)
        completed_seed_keys = {key for key, seen in grouped_schedules.items() if required.issubset(seen)}
    total_seed_cells = int(len(datasets) * len(solvers) * len(target_nfes) * len(seeds))
    completed_seed_cells = len(completed_seed_keys)
    for dataset in datasets:
        checkpoint = load_forecast_checkpoint_splits(
            cli_args=cli_args,
            dataset_root=dataset_root,
            shared_backbone_root=Path(str(cli_args.shared_backbone_root)).resolve(),
            dataset=str(dataset),
            device=device,
        )
        model = checkpoint["model"]
        cfg = checkpoint["cfg"]
        splits = checkpoint["splits"]
        test_ds, eval_examples = _eval_dataset_for_integration(
            splits["test"],
            test_windows=int(args.test_windows),
            seed=10_000 + DATASET_ORDER.index(str(dataset)),
        )
        with torch.no_grad():
            for solver_key in solvers:
                runtime_solver = solver_runtime_name(str(solver_key))
                for target_nfe in target_nfes:
                    macro_steps = int(solver_macro_steps(str(solver_key), int(target_nfe)))
                    dense_reference_steps = int(
                        resolve_reference_macro_steps(
                            0,
                            macro_steps,
                            reference_macro_factor=float(args.dense_reference_macro_factor),
                        )
                    )
                    dense_reference_grid = _dense_uniform_grid(dense_reference_steps)
                    schedule_grids = {
                        schedule_key: build_fixed_schedule_grid(str(schedule_key), int(macro_steps))
                        for schedule_key in schedules
                    }
                    for seed in seeds:
                        seed = int(seed)
                        seed_key = (str(dataset), str(solver_key), int(target_nfe), int(seed))
                        if seed_key in completed_seed_keys:
                            print(
                                json.dumps(
                                    {
                                        "progress": f"{completed_seed_cells}/{total_seed_cells}",
                                        "scenario_key": str(dataset),
                                        "solver_key": str(solver_key),
                                        "target_nfe": int(target_nfe),
                                        "seed": int(seed),
                                        "rows": int(len(rows)),
                                        "status": "skipped_existing",
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            continue
                        reference_endpoints, reference_seconds = _sample_forecast_endpoints_norm(
                            model,
                            test_ds,
                            cfg,
                            solver_name=runtime_solver,
                            macro_steps=int(dense_reference_steps),
                            time_grid=dense_reference_grid,
                            seed=int(seed),
                            batch_size=int(args.integration_batch_size),
                        )
                        schedule_errors: Dict[str, float] = {}
                        schedule_seconds: Dict[str, float] = {}
                        for schedule_key in schedules:
                            endpoints, elapsed_seconds = _sample_forecast_endpoints_norm(
                                model,
                                test_ds,
                                cfg,
                                solver_name=runtime_solver,
                                macro_steps=int(macro_steps),
                                time_grid=schedule_grids[str(schedule_key)],
                                seed=int(seed),
                                batch_size=int(args.integration_batch_size),
                            )
                            schedule_errors[str(schedule_key)] = _mean_endpoint_l2(endpoints, reference_endpoints)
                            schedule_seconds[str(schedule_key)] = float(elapsed_seconds)
                        uniform_error = float(schedule_errors["uniform"])
                        if uniform_error <= 0.0 or not math.isfinite(uniform_error):
                            raise ValueError(
                                f"Uniform integration error must be positive for {dataset}/{solver_key}/"
                                f"NFE={target_nfe}/seed={seed}, got {uniform_error}."
                            )
                        for schedule_key in schedules:
                            integration_error = float(schedule_errors[str(schedule_key)])
                            gain = 100.0 * (1.0 - integration_error / uniform_error)
                            rows.append(
                                {
                                    "scenario_key": str(dataset),
                                    "split_phase": "locked_test",
                                    "checkpoint_id": str(checkpoint["checkpoint_id"]),
                                    "checkpoint_path": display_project_path(str(checkpoint["checkpoint_path"])),
                                    "backbone_name": str(checkpoint["backbone_name"]),
                                    "train_budget_label": str(checkpoint["train_budget_label"]),
                                    "checkpoint_step": int(checkpoint["checkpoint_step"]),
                                    "target_nfe": int(target_nfe),
                                    "macro_steps": int(macro_steps),
                                    "dense_reference_macro_steps": int(dense_reference_steps),
                                    "dense_reference_macro_factor": float(args.dense_reference_macro_factor),
                                    "evaluation_seed": int(seed),
                                    "solver_key": str(solver_key),
                                    "solver_name": str(runtime_solver),
                                    "scheduler_key": str(schedule_key),
                                    "scheduler_name": SCHEDULE_LABELS[str(schedule_key)],
                                    "integration_error": float(integration_error),
                                    "uniform_integration_error": float(uniform_error),
                                    "integration_gain_percent": float(gain),
                                    "eval_examples": int(eval_examples),
                                    "reference_seconds": float(reference_seconds),
                                    "schedule_seconds": float(schedule_seconds[str(schedule_key)]),
                                    "integration_batch_size": int(args.integration_batch_size),
                                    "endpoint_space": "normalized_model_output",
                                    "row_status": "ok",
                                }
                            )
                        completed_seed_cells += 1
                        if rows_csv is not None:
                            write_csv_rows(rows, rows_csv)
                        print(
                            json.dumps(
                                {
                                    "progress": f"{completed_seed_cells}/{total_seed_cells}",
                                    "scenario_key": str(dataset),
                                    "solver_key": str(solver_key),
                                    "target_nfe": int(target_nfe),
                                    "seed": int(seed),
                                    "rows": int(len(rows)),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
        del model
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def aggregate_integration_error_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    reject_retired_evaluation_keys(rows, source="Integration-error rows")
    groups: Dict[Tuple[str, int, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["scenario_key"]), int(row["target_nfe"]), str(row["solver_key"]), str(row["scheduler_key"]))
        groups[key].append(row)
    stats_rows: List[Dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        first = group[0]
        seeds = sorted(int(row["evaluation_seed"]) for row in group)
        errors = np.asarray([float(row["integration_error"]) for row in group], dtype=np.float64)
        uniform_errors = np.asarray([float(row["uniform_integration_error"]) for row in group], dtype=np.float64)
        gains = np.asarray([float(row["integration_gain_percent"]) for row in group], dtype=np.float64)
        stats_rows.append(
            {
                "scenario_key": str(first["scenario_key"]),
                "split_phase": str(first["split_phase"]),
                "checkpoint_id": str(first["checkpoint_id"]),
                "backbone_name": str(first["backbone_name"]),
                "train_budget_label": str(first["train_budget_label"]),
                "checkpoint_step": int(first["checkpoint_step"]),
                "target_nfe": int(first["target_nfe"]),
                "macro_steps": int(first["macro_steps"]),
                "dense_reference_macro_steps": int(first["dense_reference_macro_steps"]),
                "dense_reference_macro_factor": float(first["dense_reference_macro_factor"]),
                "solver_key": str(first["solver_key"]),
                "solver_name": str(first["solver_name"]),
                "scheduler_key": str(first["scheduler_key"]),
                "scheduler_name": str(first["scheduler_name"]),
                "n_seeds": int(len(seeds)),
                "seed_values": ";".join(str(seed) for seed in seeds),
                "integration_error_mean": float(np.mean(errors)),
                "integration_error_std": float(np.std(errors, ddof=0)),
                "uniform_integration_error_mean": float(np.mean(uniform_errors)),
                "uniform_integration_error_std": float(np.std(uniform_errors, ddof=0)),
                "integration_gain_percent_mean": float(np.mean(gains)),
                "integration_gain_percent_std": float(np.std(gains, ddof=0)),
                "eval_examples": int(first["eval_examples"]),
                "endpoint_space": str(first["endpoint_space"]),
            }
        )
    return stats_rows


def write_csv_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_integration_gain_rows(path: Path) -> Dict[Tuple[str, int, str, str], Dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    reject_retired_evaluation_keys(rows, source=f"Integration-gain rows from {Path(path).name}")
    selected: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {}
    for row in rows:
        dataset = str(row["scenario_key"])
        solver_key = str(row["solver_key"])
        schedule_key = str(row["scheduler_key"])
        target_nfe = int(row["target_nfe"])
        if dataset not in DATASET_ORDER or solver_key not in SOLVER_ORDER:
            continue
        if target_nfe not in TARGET_NFES or schedule_key not in TRANSFER_SCHEDULES:
            continue
        key = (dataset, target_nfe, solver_key, schedule_key)
        if key in selected:
            raise ValueError(f"Duplicate integration gain row for {key}.")
        selected[key] = {
            "scenario_key": dataset,
            "checkpoint_id": str(row["checkpoint_id"]),
            "checkpoint_step": int(row["checkpoint_step"]),
            "target_nfe": int(target_nfe),
            "macro_steps": int(row["macro_steps"]),
            "solver_key": solver_key,
            "scheduler_key": schedule_key,
            "n_seeds": int(row["n_seeds"]),
            "seed_values": str(row["seed_values"]),
            "integration_error_mean": float(row["integration_error_mean"]),
            "integration_error_std": float(row["integration_error_std"]),
            "uniform_integration_error_mean": float(row["uniform_integration_error_mean"]),
            "uniform_integration_error_std": float(row["uniform_integration_error_std"]),
            "integration_gain_percent_mean": float(row["integration_gain_percent_mean"]),
            "integration_gain_percent_std": float(row["integration_gain_percent_std"]),
            "observed_integration_gain_percent": float(row["integration_gain_percent_mean"]),
            "eval_examples": int(row["eval_examples"]),
            "endpoint_space": str(row.get("endpoint_space", "normalized_model_output")),
        }
    expected = len(DATASET_ORDER) * len(SOLVER_ORDER) * len(TARGET_NFES) * len(TRANSFER_SCHEDULES)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} selected integration rows, got {len(selected)}.")
    return selected


def _load_zip_csv(zip_path: Path, member_name: str) -> List[Dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as handle:
            return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))


def observed_gain_from_relative_row(row: Mapping[str, str]) -> Dict[str, float]:
    gain_crps = 100.0 * (1.0 - float(row["relative_crps_vs_uniform_mean"]))
    gain_mase = 100.0 * (1.0 - float(row["relative_mase_vs_uniform_mean"]))
    return {
        "observed_gain_rcrps_percent": float(gain_crps),
        "observed_gain_rmase_percent": float(gain_mase),
        "observed_gain_avg_percent": float(0.5 * (gain_crps + gain_mase)),
    }


def load_observed_gain_rows(zip_path: Path) -> Dict[Tuple[str, int, str, str], Dict[str, Any]]:
    rows = _load_zip_csv(Path(zip_path), RELATIVE_STATS_NAME)
    reject_retired_evaluation_keys(rows, source=f"Observed-gain rows from {Path(zip_path).name}")
    selected: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {}
    for row in rows:
        dataset = str(row["scenario_key"])
        solver_key = str(row["solver_key"])
        schedule_key = str(row["scheduler_key"])
        target_nfe = int(row["target_nfe"])
        if dataset not in DATASET_ORDER or solver_key not in SOLVER_ORDER:
            continue
        if target_nfe not in TARGET_NFES or schedule_key not in TRANSFER_SCHEDULES:
            continue
        key = (dataset, target_nfe, solver_key, schedule_key)
        if key in selected:
            raise ValueError(f"Duplicate observed gain row for {key}.")
        gains = observed_gain_from_relative_row(row)
        selected[key] = {
            "scenario_key": dataset,
            "checkpoint_id": str(row["checkpoint_id"]),
            "checkpoint_step": int(row["checkpoint_step"]),
            "target_nfe": int(target_nfe),
            "solver_key": solver_key,
            "scheduler_key": schedule_key,
            "relative_crps_vs_uniform_mean": float(row["relative_crps_vs_uniform_mean"]),
            "relative_mase_vs_uniform_mean": float(row["relative_mase_vs_uniform_mean"]),
            "relative_crps_vs_uniform_std": float(row["relative_crps_vs_uniform_std"]),
            "relative_mase_vs_uniform_std": float(row["relative_mase_vs_uniform_std"]),
            "n_seeds": int(row["n_seeds"]),
            **gains,
        }
    expected = len(DATASET_ORDER) * len(SOLVER_ORDER) * len(TARGET_NFES) * len(TRANSFER_SCHEDULES)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} selected observed rows, got {len(selected)}.")
    return selected


def validate_input_payload(payload: Mapping[str, Any]) -> None:
    reject_retired_evaluation_keys(payload, source="PTG input payload")
    cells = list(payload.get("cells", []))
    expected = len(payload.get("scenario_keys", [])) * len(payload.get("solvers", [])) * len(payload.get("target_nfes", []))
    if len(cells) != expected:
        raise ValueError(f"Expected {expected} input cells, got {len(cells)}.")
    seen = set()
    for cell in cells:
        key = (str(cell["scenario_key"]), int(cell["target_nfe"]), str(cell["solver_key"]))
        if key in seen:
            raise ValueError(f"Duplicate input cell {key}.")
        seen.add(key)
        checkpoint_id = str(cell.get("checkpoint_id", "")).strip()
        if not checkpoint_id:
            raise ValueError(f"{key} is missing checkpoint_id.")
        checkpoint_step = int(cell.get("checkpoint_step", 0) or 0)
        if checkpoint_step <= 0:
            raise ValueError(f"{key} requires a positive checkpoint_step.")
        reference_grid = validate_time_grid(cell["reference_time_grid"], name=f"{key}_reference_time_grid")
        required_trace_keys = (
            INFO_GROWTH_TRACE_KEY,
            "validation_oracle_local_error_trace",
            LOCAL_DEFECT_TRACE_KEY,
        )
        for trace_key in required_trace_keys:
            if trace_key not in cell:
                raise ValueError(f"{key} missing required trace {trace_key!r}; rerun collect.")
            trace = _finite_1d(cell[trace_key], name=f"{key}_{trace_key}")
            if reference_grid.size != trace.size + 1:
                raise ValueError(f"{key} reference grid length does not match {trace_key} length.")


def _ptg_variant(
    trace: Sequence[float],
    reference_grid: Sequence[float],
    schedule_grid: Sequence[float],
    *,
    solver_order_p: float,
    density_floor_eta: float = 0.0,
    reverse_schedule: bool = False,
) -> PtgResult:
    grid = reverse_schedule_grid(schedule_grid) if bool(reverse_schedule) else list(schedule_grid)
    return ptg_from_trace(
        trace,
        reference_grid,
        grid,
        solver_order_p=float(solver_order_p),
        density_floor_eta=float(density_floor_eta),
    )


def build_points(
    payload: Mapping[str, Any],
    integration_rows: Mapping[Tuple[str, int, str, str], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    validate_input_payload(payload)
    points: List[Dict[str, Any]] = []
    for cell in payload["cells"]:
        dataset = str(cell["scenario_key"])
        solver_key = str(cell["solver_key"])
        target_nfe = int(cell["target_nfe"])
        macro_steps = int(cell["macro_steps"])
        reference_grid = [float(x) for x in cell["reference_time_grid"]]
        info_growth = [float(x) for x in cell[INFO_GROWTH_TRACE_KEY]]
        local_defect = [float(x) for x in cell[LOCAL_DEFECT_TRACE_KEY]]
        kappa, _widths, eps_h, kappa_integral = normalize_hardness_for_ptg(local_defect, reference_grid)
        for schedule_key in TRANSFER_SCHEDULES:
            schedule_grid = build_fixed_schedule_grid(schedule_key, macro_steps)
            solver_p = solver_order_for_ptg(solver_key)
            info_growth_raw = _ptg_variant(
                info_growth,
                reference_grid,
                schedule_grid,
                solver_order_p=float(solver_p),
            )
            info_growth_reversed = _ptg_variant(
                info_growth,
                reference_grid,
                schedule_grid,
                solver_order_p=float(solver_p),
                reverse_schedule=True,
            )
            local_defect_eta = _ptg_variant(
                local_defect,
                reference_grid,
                schedule_grid,
                solver_order_p=float(solver_p),
                density_floor_eta=DEFAULT_DENSITY_FLOOR_ETA,
            )
            local_defect_reversed_eta = _ptg_variant(
                local_defect,
                reference_grid,
                schedule_grid,
                solver_order_p=float(solver_p),
                density_floor_eta=DEFAULT_DENSITY_FLOOR_ETA,
                reverse_schedule=True,
            )
            obs_key = (dataset, target_nfe, solver_key, schedule_key)
            if obs_key not in integration_rows:
                raise ValueError(f"Missing integration gain row for {obs_key}.")
            observed = dict(integration_rows[obs_key])
            if str(observed.get("checkpoint_id", "")).strip() != str(cell["checkpoint_id"]):
                raise ValueError(f"Checkpoint ID mismatch for {obs_key}.")
            if int(observed.get("checkpoint_step", 0) or 0) != int(cell["checkpoint_step"]):
                raise ValueError(f"Checkpoint step mismatch for {obs_key}.")
            points.append(
                {
                    "scenario_key": dataset,
                    "scenario_label": DATASET_LABELS[dataset],
                    "solver_key": solver_key,
                    "solver_label": SOLVER_LABELS[solver_key],
                    "target_nfe": int(target_nfe),
                    "macro_steps": int(macro_steps),
                    "scheduler_key": schedule_key,
                    "scheduler_label": SCHEDULE_LABELS[schedule_key],
                    "solver_order_p": float(solver_p),
                    "reference_macro_steps": int(cell["reference_macro_steps"]),
                    "ptg_percent": float(local_defect_eta.ptg_percent),
                    "ser": float(local_defect_eta.ser),
                    "ptg_info_growth_raw": float(info_growth_raw.ptg_percent),
                    "ptg_info_growth_reversed": float(info_growth_reversed.ptg_percent),
                    "ptg_local_defect_eta005": float(local_defect_eta.ptg_percent),
                    "ptg_local_defect_reversed_eta005": float(local_defect_reversed_eta.ptg_percent),
                    "ser_info_growth_raw": float(info_growth_raw.ser),
                    "ser_info_growth_reversed": float(info_growth_reversed.ser),
                    "ser_local_defect_eta005": float(local_defect_eta.ser),
                    "ser_local_defect_reversed_eta005": float(local_defect_reversed_eta.ser),
                    "kappa_integral": float(kappa_integral),
                    "ptg_kappa_integral": float(local_defect_eta.kappa_integral),
                    "rho_integral": float(local_defect_eta.rho_integral),
                    "rho_integral_info_growth_raw": float(info_growth_raw.rho_integral),
                    "rho_integral_info_growth_reversed": float(info_growth_reversed.rho_integral),
                    "rho_integral_local_defect_eta005": float(local_defect_eta.rho_integral),
                    "rho_integral_local_defect_reversed_eta005": float(local_defect_reversed_eta.rho_integral),
                    "density_floor_eta": float(DEFAULT_DENSITY_FLOOR_ETA),
                    "eps_h": float(eps_h),
                    "kappa_mean": float(np.mean(kappa)),
                    "schedule_grid": ";".join(f"{float(x):.10g}" for x in schedule_grid),
                    **observed,
                }
            )
    expected = len(DATASET_ORDER) * len(SOLVER_ORDER) * len(TARGET_NFES) * len(TRANSFER_SCHEDULES)
    if (
        tuple(payload.get("scenario_keys", [])) == DATASET_ORDER
        and tuple(payload.get("solvers", [])) == SOLVER_ORDER
        and tuple(int(x) for x in payload.get("target_nfes", [])) == TARGET_NFES
        and len(points) != expected
    ):
        raise ValueError(f"Expected {expected} PTG points, got {len(points)}.")
    if any(str(point["scheduler_key"]) == "late_power_3" for point in points):
        raise ValueError("late_power_3 must be excluded from PTG points.")
    return points


def write_points_csv(points: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not points:
        raise ValueError("Cannot write an empty points CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(points[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(point)


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> Tuple[float, Optional[float]]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size != y_arr.size or x_arr.size < 2:
        return float("nan"), None
    result = spearmanr(x_arr, y_arr)
    return float(result.statistic), float(result.pvalue)


def _axis_limits(values: np.ndarray, *, pad_fraction: float = 0.08) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low = min(float(np.min(finite)), 0.0)
    high = max(float(np.max(finite)), 0.0)
    span = max(high - low, 1.0)
    return low - pad_fraction * span, high + pad_fraction * span


def summarize_ptg_points(points: Sequence[Mapping[str, Any]], *, main_ptg_key: str = DEFAULT_MAIN_PTG_KEY) -> Dict[str, Any]:
    if not points:
        raise ValueError("Cannot summarize an empty point set.")
    y = [float(point["observed_integration_gain_percent"]) for point in points]
    variants: Dict[str, Dict[str, Any]] = {}
    for key, label in PTG_X_LABELS.items():
        if any(key not in point for point in points):
            continue
        x = [float(point[key]) for point in points]
        rho, p_value = spearman_correlation(x, y)
        variants[key] = {
            "label": label,
            "spearman_rho": float(rho),
            "spearman_p_value": None if p_value is None else float(p_value),
            "ptg_min": float(np.min(np.asarray(x, dtype=np.float64))),
            "ptg_max": float(np.max(np.asarray(x, dtype=np.float64))),
            "observed_integration_gain_min": float(np.min(np.asarray(y, dtype=np.float64))),
            "observed_integration_gain_max": float(np.max(np.asarray(y, dtype=np.float64))),
        }
    main = variants.get(str(main_ptg_key), {})
    main_rho = float(main.get("spearman_rho", float("nan")))
    return {
        "artifact": "ptg_observed_gain_diagnostics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "main_ptg_key": str(main_ptg_key),
        "main_spearman_rho": main_rho,
        "main_spearman_positive": bool(math.isfinite(main_rho) and main_rho > 0.0),
        "n_points": int(len(points)),
        "density_floor_eta": float(DEFAULT_DENSITY_FLOOR_ETA),
        "observed_y_key": "observed_integration_gain_percent",
        "variants": variants,
    }


def build_figure(points: Sequence[Mapping[str, Any]], *, x_key: str = DEFAULT_MAIN_PTG_KEY):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 14.0,
            "axes.labelsize": 15.0,
            "xtick.labelsize": 13.0,
            "ytick.labelsize": 13.0,
            "legend.fontsize": 11.2,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    if not points:
        raise ValueError("Cannot build a plot with no points.")
    if any(str(x_key) not in point for point in points):
        raise ValueError(f"All points must contain x_key={x_key!r}.")
    x = np.asarray([float(point[str(x_key)]) for point in points], dtype=np.float64)
    y = np.asarray([float(point["observed_integration_gain_percent"]) for point in points], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.2, 5.65))
    dataset_markers = {
        "solar_energy_10m": "D",
        "traffic_hourly": "^",
        "weather_daily": "o",
    }
    schedule_colors = {
        "ays": "#0072B2",
        "gits": "#D55E00",
        "ots": "#009E73",
    }
    for dataset in DATASET_ORDER:
        for schedule_key in TRANSFER_SCHEDULES:
            selected = [
                point
                for point in points
                if str(point["scenario_key"]) == dataset and str(point["scheduler_key"]) == schedule_key
            ]
            if not selected:
                continue
            ax.scatter(
                [float(point[str(x_key)]) for point in selected],
                [float(point["observed_integration_gain_percent"]) for point in selected],
                marker=dataset_markers[dataset],
                s=42,
                facecolor=schedule_colors[schedule_key],
                edgecolor="white",
                linewidth=0.55,
                alpha=0.82,
            )
    ax.axvline(0.0, color="#6e6e6e", linewidth=1.0, linestyle="--", zorder=0)
    ax.axhline(0.0, color="#6e6e6e", linewidth=1.0, linestyle="--", zorder=0)
    if np.unique(x).size >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#222222", linewidth=1.65, label="Least-squares trend", zorder=3)
    rho, p_value = spearman_correlation(x, y)
    p_text = "n/a" if p_value is None or not math.isfinite(float(p_value)) else f"{float(p_value):.2g}"
    rho_text = "nan" if not math.isfinite(float(rho)) else f"{float(rho):.2f}"
    ax.text(
        0.03,
        0.97,
        f"Spearman rho = {rho_text}\np = {p_text}\nn = {len(points)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cfcfcf", "linewidth": 0.7},
    )
    ax.set_xlabel("Validation predicted transfer gain, PTG (%)")
    ax.set_ylabel("Observed integration-error gain over uniform (%)")
    ax.set_xlim(*_axis_limits(x))
    ax.set_ylim(*_axis_limits(y))
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    dataset_handles = [
        mlines.Line2D(
            [],
            [],
            color="#444444",
            marker=dataset_markers[dataset],
            linestyle="None",
            markersize=7.2,
            label=DATASET_LABELS[dataset],
        )
        for dataset in DATASET_ORDER
    ]
    schedule_handles = [
        mlines.Line2D(
            [],
            [],
            color=schedule_colors[schedule_key],
            marker="o",
            linestyle="None",
            markersize=7.2,
            label=SCHEDULE_LABELS[schedule_key],
        )
        for schedule_key in TRANSFER_SCHEDULES
    ]
    trend_handle = mlines.Line2D([], [], color="#222222", linewidth=1.65, label="Trend")
    first_legend = fig.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="center left",
        bbox_to_anchor=(0.765, 0.40),
        frameon=True,
        framealpha=0.94,
        borderpad=0.55,
        labelspacing=0.35,
        handletextpad=0.45,
    )
    fig.add_artist(first_legend)
    fig.legend(
        handles=[*schedule_handles, trend_handle],
        title="Schedule",
        loc="upper left",
        bbox_to_anchor=(0.765, 0.955),
        frameon=True,
        framealpha=0.94,
        borderpad=0.55,
        labelspacing=0.35,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.15, right=0.74, bottom=0.16, top=0.98)
    return fig, ax


def plot_points(
    points: Sequence[Mapping[str, Any]],
    *,
    png_path: Path,
    pdf_path: Path,
    dpi: int = 600,
    x_key: str = DEFAULT_MAIN_PTG_KEY,
) -> Dict[str, str]:
    fig, _ax = build_figure(points, x_key=str(x_key))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return {"png": display_project_path(png_path), "pdf": display_project_path(pdf_path)}


def plot_points_with_diagnostics(
    points: Sequence[Mapping[str, Any]],
    *,
    png_path: Path,
    pdf_path: Path,
    diagnostics_json_path: Path,
    dpi: int = 600,
    main_ptg_key: str = DEFAULT_MAIN_PTG_KEY,
) -> Dict[str, Any]:
    diagnostics = summarize_ptg_points(points, main_ptg_key=str(main_ptg_key))
    outputs = plot_points(
        points,
        png_path=Path(png_path),
        pdf_path=Path(pdf_path),
        dpi=int(dpi),
        x_key=str(main_ptg_key),
    )
    diagnostics["report_written"] = True
    diagnostics["figure_mode"] = "report"
    diagnostics["figure_outputs"] = dict(outputs)
    diagnostics["report_outputs"] = dict(outputs)
    diagnostics_json_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(diagnostics, diagnostics_json_path)
    return {
        **outputs,
        "diagnostics_json": display_project_path(diagnostics_json_path),
        "report_written": True,
    }


def synthetic_payload() -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    for dataset_idx, dataset in enumerate(DATASET_ORDER):
        for solver_idx, solver_key in enumerate(SOLVER_ORDER):
            for target_nfe in TARGET_NFES:
                macro_steps = target_nfe if solver_key in {"euler", "dpmpp2m"} else target_nfe // 2
                reference_steps = max(32, int(round(DEFAULT_REFERENCE_MACRO_FACTOR * macro_steps)))
                reference_grid = [float(i) / float(reference_steps) for i in range(reference_steps + 1)]
                trace = [
                    0.8
                    + 0.15 * dataset_idx
                    + 0.03 * solver_idx
                    + 0.2 * math.sin(2.0 * math.pi * (float(i) / float(reference_steps) + 0.04 * dataset_idx))
                    for i in range(reference_steps)
                ]
                oracle_trace = [
                    0.02
                    + 0.003 * dataset_idx
                    + 0.002 * solver_idx
                    + 0.004 * math.cos(2.0 * math.pi * float(i) / float(reference_steps))
                    for i in range(reference_steps)
                ]
                local_defect_trace = local_defect_trace_from_oracle(
                    oracle_trace,
                    reference_grid,
                    solver_order_p=solver_order_for_ptg(solver_key),
                )
                cells.append(
                    {
                        "scenario_key": dataset,
                        "scenario_label": DATASET_LABELS[dataset],
                        "solver_key": solver_key,
                        "solver_label": SOLVER_LABELS[solver_key],
                        "target_nfe": int(target_nfe),
                        "macro_steps": int(macro_steps),
                        "reference_macro_steps": int(reference_steps),
                        "reference_time_grid": reference_grid,
                        "validation_hardness_trace": local_defect_trace,
                        "validation_info_growth_trace": trace,
                        "validation_oracle_local_error_trace": oracle_trace,
                        "validation_local_defect_trace": local_defect_trace,
                        "per_seed": [],
                        "checkpoint_id": "synthetic_checkpoint",
                        "checkpoint_step": 20000,
                    }
                )
    return {
        "artifact": "ptg_observed_gain_inputs",
        "generated_at_utc": "synthetic",
        "scenario_keys": list(DATASET_ORDER),
        "solvers": list(SOLVER_ORDER),
        "target_nfes": list(TARGET_NFES),
        "seeds": list(DEFAULT_SEEDS),
        "validation_windows": DEFAULT_VALIDATION_WINDOWS,
        "reference_macro_factor": DEFAULT_REFERENCE_MACRO_FACTOR,
        "calibration_trace_samples": DEFAULT_CALIBRATION_TRACE_SAMPLES,
        "signal_trace_key": NATIVE_TRACE_KEY,
        "report_trace_key": LOCAL_DEFECT_TRACE_KEY,
        "oracle_local_error_trace_key": ORACLE_LOCAL_ERROR_TRACE_KEY,
        "density_floor_eta": DEFAULT_DENSITY_FLOOR_ETA,
        "main_ptg_key": DEFAULT_MAIN_PTG_KEY,
        "test_trace_used": False,
        "cells": cells,
    }


def synthetic_observed_rows() -> Dict[Tuple[str, int, str, str], Dict[str, Any]]:
    rows: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {}
    for dataset_idx, dataset in enumerate(DATASET_ORDER):
        for solver_idx, solver_key in enumerate(SOLVER_ORDER):
            for nfe_idx, target_nfe in enumerate(TARGET_NFES):
                for schedule_idx, schedule_key in enumerate(TRANSFER_SCHEDULES):
                    gain = -2.0 + 0.7 * dataset_idx + 0.4 * solver_idx + 0.5 * nfe_idx + 0.6 * schedule_idx
                    rows[(dataset, int(target_nfe), solver_key, schedule_key)] = {
                        "scenario_key": dataset,
                        "checkpoint_id": "synthetic_checkpoint",
                        "checkpoint_step": 20000,
                        "target_nfe": int(target_nfe),
                        "macro_steps": int(target_nfe if solver_key in {"euler", "dpmpp2m"} else target_nfe // 2),
                        "solver_key": solver_key,
                        "scheduler_key": schedule_key,
                        "n_seeds": 5,
                        "seed_values": "0;1;2;3;4",
                        "integration_error_mean": 1.0 - gain / 100.0,
                        "integration_error_std": 0.0,
                        "uniform_integration_error_mean": 1.0,
                        "uniform_integration_error_std": 0.0,
                        "integration_gain_percent_mean": gain,
                        "integration_gain_percent_std": 0.0,
                        "observed_integration_gain_percent": gain,
                        "eval_examples": 8,
                        "endpoint_space": "normalized_model_output",
                    }
    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the forecast PTG vs observed gain scatter figure.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect validation hardness traces using the diffusion-flow evaluation harness.")
    collect.add_argument("--out-json", type=Path, default=DEFAULT_INPUT_JSON)
    collect.add_argument("--datasets", type=str, default=",".join(DATASET_ORDER))
    collect.add_argument("--solvers", type=str, default=",".join(SOLVER_ORDER))
    collect.add_argument("--target-nfes", type=str, default=",".join(str(nfe) for nfe in TARGET_NFES))
    collect.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    collect.add_argument("--val-windows", type=int, default=DEFAULT_VALIDATION_WINDOWS)
    collect.add_argument("--reference-macro-factor", type=float, default=DEFAULT_REFERENCE_MACRO_FACTOR)
    collect.add_argument("--calibration-trace-samples", type=int, default=DEFAULT_CALIBRATION_TRACE_SAMPLES)
    collect.add_argument("--backbone-manifest", type=str, default="backbone_manifest.json")
    collect.add_argument("--device", type=str, default="auto")
    collect.add_argument("--smoke", action="store_true", help="Collect the first dataset/solver/NFE/seed with a tiny window cap.")

    integration = subparsers.add_parser(
        "collect-integration-error",
        help="Collect locked-test endpoint integration error against a dense reference rollout.",
    )
    integration.add_argument("--rows-csv", type=Path, default=DEFAULT_INTEGRATION_ROWS_CSV)
    integration.add_argument("--seed-stats-csv", type=Path, default=DEFAULT_INTEGRATION_SEED_STATS_CSV)
    integration.add_argument("--datasets", type=str, default=",".join(DATASET_ORDER))
    integration.add_argument("--solvers", type=str, default=",".join(SOLVER_ORDER))
    integration.add_argument("--target-nfes", type=str, default=",".join(str(nfe) for nfe in TARGET_NFES))
    integration.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    integration.add_argument("--schedules", type=str, default=",".join(INTEGRATION_SCHEDULES))
    integration.add_argument("--test-windows", type=int, default=DEFAULT_TEST_WINDOWS)
    integration.add_argument("--integration-batch-size", type=int, default=64)
    integration.add_argument("--dense-reference-macro-factor", type=float, default=DEFAULT_DENSE_REFERENCE_MACRO_FACTOR)
    integration.add_argument("--backbone-manifest", type=str, default="backbone_manifest.json")
    integration.add_argument("--device", type=str, default="auto")
    integration.add_argument("--resume", action="store_true", help="Skip seed cells already present in --rows-csv.")
    integration.add_argument("--smoke", action="store_true", help="Collect one tiny integration-error cell.")

    plot = subparsers.add_parser("plot", help="Join collected PTG inputs with integration-error gains and render the figure.")
    plot.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    plot.add_argument("--integration-error-csv", type=Path, default=DEFAULT_INTEGRATION_SEED_STATS_CSV)
    plot.add_argument("--points-csv", type=Path, default=DEFAULT_POINTS_CSV)
    plot.add_argument("--diagnostics-json", type=Path, default=DEFAULT_DIAGNOSTICS_JSON)
    plot.add_argument("--png", type=Path, default=DEFAULT_PNG)
    plot.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    plot.add_argument("--dpi", type=int, default=600)

    synth = subparsers.add_parser("plot-synthetic", help="Render a synthetic plot for smoke checks.")
    synth.add_argument("--points-csv", type=Path, default=DEFAULT_RESULTS_DIR / "synthetic_ptg_points.csv")
    synth.add_argument("--diagnostics-json", type=Path, default=DEFAULT_RESULTS_DIR / "synthetic_ptg_diagnostics.json")
    synth.add_argument("--png", type=Path, default=DEFAULT_RESULTS_DIR / "synthetic_ptg_vs_observed_gain.png")
    synth.add_argument("--pdf", type=Path, default=DEFAULT_RESULTS_DIR / "synthetic_ptg_vs_observed_gain.pdf")
    synth.add_argument("--dpi", type=int, default=200)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.command == "collect":
        payload = collect_payload(args)
        validate_input_payload(payload)
        write_json(payload, Path(args.out_json))
        print(display_project_path(Path(args.out_json)))
        return
    if args.command == "collect-integration-error":
        rows = collect_integration_error_rows(args)
        stats_rows = aggregate_integration_error_rows(rows)
        write_csv_rows(rows, Path(args.rows_csv))
        write_csv_rows(stats_rows, Path(args.seed_stats_csv))
        print(json.dumps({"rows_csv": display_project_path(Path(args.rows_csv)), "seed_stats_csv": display_project_path(Path(args.seed_stats_csv)), "rows": len(rows), "seed_stats_rows": len(stats_rows)}, indent=2, sort_keys=True))
        return
    if args.command == "plot":
        payload = load_json(Path(args.input_json))
        integration_rows = load_integration_gain_rows(Path(args.integration_error_csv))
        points = build_points(payload, integration_rows)
        write_points_csv(points, Path(args.points_csv))
        outputs = plot_points_with_diagnostics(
            points,
            png_path=Path(args.png),
            pdf_path=Path(args.pdf),
            diagnostics_json_path=Path(args.diagnostics_json),
            dpi=int(args.dpi),
        )
        outputs["points_csv"] = display_project_path(Path(args.points_csv))
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return
    if args.command == "plot-synthetic":
        points = build_points(synthetic_payload(), synthetic_observed_rows())
        write_points_csv(points, Path(args.points_csv))
        outputs = plot_points_with_diagnostics(
            points,
            png_path=Path(args.png),
            pdf_path=Path(args.pdf),
            diagnostics_json_path=Path(args.diagnostics_json),
            dpi=int(args.dpi),
        )
        outputs["points_csv"] = display_project_path(Path(args.points_csv))
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return
    raise ValueError(f"Unsupported command {args.command!r}.")


if __name__ == "__main__":
    main()
