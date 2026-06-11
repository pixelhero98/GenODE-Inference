from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from genode.gipo.models import validate_time_grid

DENSITY_PROTOCOL = "density_mass"
DENSITY_DOMAIN = "normalized_model_time_0_1"
DEFAULT_DENSITY_BIN_COUNT = 64
DEFAULT_DENSITY_EPS = 1e-12
DEFAULT_LOG_DENSITY_EPS = 1e-8


def _json_hash(payload: Mapping[str, Any], *, prefix: str) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def uniform_reference_grid(bin_count: int = DEFAULT_DENSITY_BIN_COUNT) -> Tuple[float, ...]:
    bins = int(bin_count)
    if bins <= 0:
        raise ValueError(f"bin_count must be positive, got {bin_count!r}.")
    grid = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    return tuple(float(x) for x in grid.tolist())


def validate_reference_grid(reference_time_grid: Sequence[float]) -> Tuple[float, ...]:
    values = np.asarray([float(x) for x in reference_time_grid], dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("reference_time_grid must be one-dimensional with at least two edges.")
    if not np.all(np.isfinite(values)):
        raise ValueError("reference_time_grid contains non-finite values.")
    if abs(float(values[0])) > 1e-12 or abs(float(values[-1]) - 1.0) > 1e-12:
        raise ValueError("reference_time_grid must start at 0.0 and end at 1.0.")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("reference_time_grid must be strictly increasing.")
    return tuple(float(x) for x in values.tolist())


def reference_grid_hash(reference_time_grid: Sequence[float]) -> str:
    grid = [round(float(x), 12) for x in validate_reference_grid(reference_time_grid)]
    return _json_hash({"density_protocol": DENSITY_PROTOCOL, "reference_time_grid": grid}, prefix="refgrid")


def sanitize_density_mass(
    density_mass: Sequence[float],
    *,
    eps: float = DEFAULT_DENSITY_EPS,
) -> Tuple[float, ...]:
    mass = np.asarray([float(x) for x in density_mass], dtype=np.float64)
    if mass.ndim != 1 or mass.size <= 0:
        raise ValueError("density_mass must be a non-empty 1D vector.")
    if not np.all(np.isfinite(mass)):
        raise ValueError("density_mass contains non-finite values.")
    if np.any(mass < 0.0):
        raise ValueError("density_mass must be nonnegative.")
    eps_value = float(eps)
    if eps_value < 0.0:
        raise ValueError(f"eps must be nonnegative, got {eps!r}.")
    if eps_value > 0.0:
        mass = np.maximum(mass, eps_value)
    total = float(np.sum(mass))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("density_mass must have positive finite sum.")
    mass = mass / total
    return tuple(float(x) for x in mass.tolist())


def density_mass_hash(density_mass: Sequence[float], *, reference_time_grid: Sequence[float] | None = None) -> str:
    mass = [round(float(x), 12) for x in sanitize_density_mass(density_mass, eps=0.0)]
    grid = uniform_reference_grid(len(mass)) if reference_time_grid is None else validate_reference_grid(reference_time_grid)
    return _json_hash(
        {
            "density_protocol": DENSITY_PROTOCOL,
            "reference_grid_hash": reference_grid_hash(grid),
            "density_mass": mass,
        },
        prefix="density",
    )


def grid_to_density_mass(
    time_grid: Sequence[float],
    *,
    reference_time_grid: Sequence[float] | None = None,
    macro_steps: int | None = None,
    eps: float = 0.0,
) -> Tuple[float, ...]:
    steps = int(macro_steps) if macro_steps is not None else len(tuple(time_grid)) - 1
    grid = np.asarray(validate_time_grid(time_grid, macro_steps=steps), dtype=np.float64)
    reference = np.asarray(
        validate_reference_grid(
            uniform_reference_grid(DEFAULT_DENSITY_BIN_COUNT)
            if reference_time_grid is None
            else reference_time_grid
        ),
        dtype=np.float64,
    )
    step_count = int(grid.size - 1)
    if step_count <= 0:
        raise ValueError("time_grid must contain at least one interval.")
    ref_widths = np.diff(reference)
    mass = np.zeros(int(ref_widths.size), dtype=np.float64)
    for step_idx in range(step_count):
        left = float(grid[step_idx])
        right = float(grid[step_idx + 1])
        width = float(right - left)
        if width <= 0.0:
            raise ValueError("time_grid must be strictly increasing.")
        local_density = 1.0 / (float(step_count) * width)
        start = max(0, int(np.searchsorted(reference, left, side="right") - 1))
        stop = min(int(np.searchsorted(reference, right, side="left")), int(ref_widths.size - 1))
        for ref_idx in range(start, stop + 1):
            overlap = max(0.0, min(right, float(reference[ref_idx + 1])) - max(left, float(reference[ref_idx])))
            if overlap > 0.0:
                mass[ref_idx] += local_density * overlap
    return sanitize_density_mass(mass, eps=float(eps))


def density_mass_to_time_grid(
    density_mass: Sequence[float],
    *,
    macro_steps: int,
    reference_time_grid: Sequence[float] | None = None,
    eps: float = DEFAULT_DENSITY_EPS,
) -> Tuple[float, ...]:
    steps = int(macro_steps)
    if steps <= 0:
        raise ValueError(f"macro_steps must be positive, got {macro_steps!r}.")
    mass = np.asarray(sanitize_density_mass(density_mass, eps=float(eps)), dtype=np.float64)
    reference = np.asarray(
        validate_reference_grid(
            uniform_reference_grid(int(mass.size))
            if reference_time_grid is None
            else reference_time_grid
        ),
        dtype=np.float64,
    )
    if reference.size != mass.size + 1:
        raise ValueError("reference_time_grid must have len(density_mass) + 1 edges.")
    cdf = np.concatenate([[0.0], np.cumsum(mass)])
    cdf[-1] = 1.0
    quantiles = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
    grid = np.interp(quantiles, cdf, reference)
    grid[0] = 0.0
    grid[-1] = 1.0
    for idx in range(1, len(grid)):
        if grid[idx] <= grid[idx - 1]:
            grid[idx] = min(1.0, grid[idx - 1] + 1e-8)
    grid[-1] = 1.0
    return validate_time_grid([float(x) for x in grid.tolist()], macro_steps=steps)


def density_log_features(
    density_mass: Sequence[float],
    *,
    reference_time_grid: Sequence[float] | None = None,
    eps: float = DEFAULT_LOG_DENSITY_EPS,
) -> np.ndarray:
    mass = np.asarray(sanitize_density_mass(density_mass, eps=0.0), dtype=np.float64)
    reference = np.asarray(
        validate_reference_grid(
            uniform_reference_grid(int(mass.size))
            if reference_time_grid is None
            else reference_time_grid
        ),
        dtype=np.float64,
    )
    if reference.size != mass.size + 1:
        raise ValueError("reference_time_grid must have len(density_mass) + 1 edges.")
    widths = np.diff(reference)
    density = mass / widths
    return np.log(np.maximum(density, float(eps))).astype(np.float32)


def density_metadata(reference_time_grid: Sequence[float] | None = None) -> Dict[str, Any]:
    grid = validate_reference_grid(
        uniform_reference_grid(DEFAULT_DENSITY_BIN_COUNT)
        if reference_time_grid is None
        else reference_time_grid
    )
    return {
        "density_protocol": DENSITY_PROTOCOL,
        "density_domain": DENSITY_DOMAIN,
        "reference_grid_kind": "uniform_edges",
        "reference_bin_count": int(len(grid) - 1),
        "reference_time_grid": [float(x) for x in grid],
        "reference_grid_hash": reference_grid_hash(grid),
        "grid_to_density_method": "equal_step_mass_overlap",
        "density_to_grid_method": "piecewise_constant_inverse_cdf",
        "log_density_floor": float(DEFAULT_LOG_DENSITY_EPS),
    }


__all__ = [
    "DEFAULT_DENSITY_BIN_COUNT",
    "DEFAULT_DENSITY_EPS",
    "DEFAULT_LOG_DENSITY_EPS",
    "DENSITY_DOMAIN",
    "DENSITY_PROTOCOL",
    "density_log_features",
    "density_mass_hash",
    "density_mass_to_time_grid",
    "density_metadata",
    "grid_to_density_mass",
    "reference_grid_hash",
    "sanitize_density_mass",
    "uniform_reference_grid",
    "validate_reference_grid",
]
