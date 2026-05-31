from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from genode.conditional_opd.models import grid_to_intervals, validate_time_grid
from genode.conditional_opd.ser_ptg_reference import grid_geometry

DEFAULT_DENSITY_GRID_SIZE = 128
LOWRANK_THETA_DIM = 7
DEFAULT_THETA_BOUND = 2.5
DEFAULT_BUMP_MUS: Tuple[float, float, float] = (0.15, 0.50, 0.85)
DEFAULT_BUMP_SIGMAS: Tuple[float, float, float] = (0.08, 0.10, 0.08)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def theta_hash(theta: Sequence[float]) -> str:
    return _hash_payload([round(float(x), 12) for x in theta])


def validate_theta(theta: Sequence[float]) -> Tuple[float, ...]:
    values = tuple(float(x) for x in theta)
    if len(values) != LOWRANK_THETA_DIM:
        raise ValueError(f"Expected theta dimension {LOWRANK_THETA_DIM}, got {len(values)}.")
    if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
        raise ValueError("Theta contains non-finite values.")
    return values


def density_bin_centers(grid_size: int = DEFAULT_DENSITY_GRID_SIZE) -> np.ndarray:
    size = int(grid_size)
    if size not in (64, 128):
        raise ValueError(f"density_grid_size must be 64 or 128 for v4 consistency, got {grid_size!r}.")
    return (np.arange(size, dtype=np.float64) + 0.5) / float(size)


def _base_density_from_schedule_grid(
    base_grid: Sequence[float] | None,
    *,
    macro_steps: int,
    grid_size: int,
) -> np.ndarray:
    centers = density_bin_centers(grid_size)
    if base_grid is None:
        return np.ones_like(centers, dtype=np.float64)
    checked = validate_time_grid(base_grid, macro_steps=int(macro_steps))
    intervals = np.asarray(grid_to_intervals(checked), dtype=np.float64)
    densities = np.empty_like(centers, dtype=np.float64)
    for idx, t in enumerate(centers):
        bin_idx = min(np.searchsorted(np.asarray(checked[1:], dtype=np.float64), float(t), side="right"), int(macro_steps) - 1)
        densities[idx] = (1.0 / float(macro_steps)) / max(float(intervals[bin_idx]), 1e-12)
    densities = np.maximum(densities, 1e-12)
    return densities / float(np.mean(densities))


def lowrank_log_density_delta(
    theta: Sequence[float],
    *,
    grid_size: int = DEFAULT_DENSITY_GRID_SIZE,
) -> np.ndarray:
    values = np.asarray(validate_theta(theta), dtype=np.float64)
    t = density_bin_centers(grid_size)
    delta = (
        values[0] * np.sin(2.0 * np.pi * t)
        + values[1] * np.cos(2.0 * np.pi * t)
        + values[2] * np.sin(4.0 * np.pi * t)
        + values[3] * np.cos(4.0 * np.pi * t)
    )
    for offset, (mu, sigma) in enumerate(zip(DEFAULT_BUMP_MUS, DEFAULT_BUMP_SIGMAS)):
        delta = delta + values[4 + offset] * np.exp(-0.5 * np.square((t - float(mu)) / float(sigma)))
    return delta


def density_from_theta(
    theta: Sequence[float],
    *,
    macro_steps: int,
    base_grid: Sequence[float] | None = None,
    grid_size: int = DEFAULT_DENSITY_GRID_SIZE,
) -> np.ndarray:
    base = _base_density_from_schedule_grid(base_grid, macro_steps=int(macro_steps), grid_size=int(grid_size))
    log_density = np.log(np.maximum(base, 1e-12)) + lowrank_log_density_delta(theta, grid_size=int(grid_size))
    log_density = log_density - float(np.max(log_density))
    density = np.exp(log_density)
    density = np.maximum(density, 1e-12)
    return density / float(np.sum(density))


def schedule_grid_from_theta(
    theta: Sequence[float],
    *,
    macro_steps: int,
    base_grid: Sequence[float] | None = None,
    grid_size: int = DEFAULT_DENSITY_GRID_SIZE,
) -> Tuple[float, ...]:
    density = density_from_theta(theta, macro_steps=int(macro_steps), base_grid=base_grid, grid_size=int(grid_size))
    edges = np.linspace(0.0, 1.0, int(grid_size) + 1, dtype=np.float64)
    cdf = np.concatenate([[0.0], np.cumsum(density)])
    cdf[-1] = 1.0
    quantiles = np.linspace(0.0, 1.0, int(macro_steps) + 1, dtype=np.float64)
    grid = np.interp(quantiles, cdf, edges)
    grid[0] = 0.0
    grid[-1] = 1.0
    return validate_time_grid([float(x) for x in grid.tolist()], macro_steps=int(macro_steps))


def theta_diagnostics(
    theta: Sequence[float],
    *,
    macro_steps: int,
    base_grid: Sequence[float] | None = None,
    grid_size: int = DEFAULT_DENSITY_GRID_SIZE,
) -> Dict[str, Any]:
    grid = schedule_grid_from_theta(theta, macro_steps=int(macro_steps), base_grid=base_grid, grid_size=int(grid_size))
    density = density_from_theta(theta, macro_steps=int(macro_steps), base_grid=base_grid, grid_size=int(grid_size))
    centers = density_bin_centers(grid_size)
    entropy = -float(np.sum(density * np.log(np.maximum(density, 1e-12))))
    return {
        "theta_hash": theta_hash(theta),
        "density_grid_size": int(grid_size),
        "density_entropy": entropy,
        "mass_early": float(np.sum(density[centers < 0.33])),
        "mass_mid": float(np.sum(density[(centers >= 0.33) & (centers < 0.67)])),
        "mass_late": float(np.sum(density[centers >= 0.67])),
        "center_of_mass": float(np.sum(density * centers)),
        "grid_geometry": grid_geometry(grid),
    }


def zero_theta() -> Tuple[float, ...]:
    return tuple(0.0 for _ in range(LOWRANK_THETA_DIM))


def hand_bump_thetas(scale: float = 1.0) -> List[Tuple[float, ...]]:
    out: List[Tuple[float, ...]] = []
    for idx in range(3):
        theta = [0.0] * LOWRANK_THETA_DIM
        theta[4 + idx] = float(scale)
        out.append(tuple(theta))
    for pairs in ((4, 5), (5, 6), (4, 6)):
        theta = [0.0] * LOWRANK_THETA_DIM
        theta[pairs[0]] = float(scale)
        theta[pairs[1]] = -float(scale) if pairs == (4, 6) else float(scale)
        out.append(tuple(theta))
    return out


def sobol_thetas(
    count: int,
    *,
    seed: int,
    bound: float = DEFAULT_THETA_BOUND,
) -> List[Tuple[float, ...]]:
    if int(count) <= 0:
        return []
    engine = torch.quasirandom.SobolEngine(dimension=LOWRANK_THETA_DIM, scramble=True, seed=int(seed))
    raw = engine.draw(int(count)).cpu().numpy().astype(np.float64)
    values = (2.0 * raw - 1.0) * float(bound)
    return [tuple(float(x) for x in row.tolist()) for row in values]


def theta_metadata(theta: Sequence[float], *, source: str, grid_size: int = DEFAULT_DENSITY_GRID_SIZE) -> Dict[str, Any]:
    values = validate_theta(theta)
    return {
        "theta": [float(x) for x in values],
        "theta_hash": theta_hash(values),
        "theta_source": str(source),
        "density_grid_size": int(grid_size),
        "theta_dim": LOWRANK_THETA_DIM,
    }
