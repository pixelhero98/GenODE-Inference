from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from genode.canonical_experiment_layout import schedule_family_for_key
from genode.schedule_transfer.reference_clocks import (
    DEFAULT_REFERENCE_CLOCK_KEYS,
    REFERENCE_CLOCK_BASE_KEYS,
    REFERENCE_CLOCK_REVERSED_KEYS,
    build_reference_clock_grid,
    reference_clock_provenance,
    reference_clock_registry,
)

BASELINE_SCHEDULE_KEYS: tuple[str, ...] = REFERENCE_CLOCK_BASE_KEYS
TRANSFER_SCHEDULE_KEYS: tuple[str, ...] = (
    "ays_sd15_native",
    "ays_sd15_log_sigma",
    "gits_cifar10_native",
    "gits_cifar10_log_sigma",
    "ots_vp_linear_native",
    "ots_vp_linear_log_sigma",
)
EXPERIMENTAL_REVERSED_SCHEDULE_KEYS: tuple[str, ...] = REFERENCE_CLOCK_REVERSED_KEYS
EXPERIMENTAL_AVERAGED_FIXED_SCHEDULE_KEYS: tuple[str, ...] = ()
EXPERIMENTAL_FIXED_SCHEDULE_KEYS: tuple[str, ...] = DEFAULT_REFERENCE_CLOCK_KEYS


def load_external_schedule_catalog() -> dict[str, dict[str, Any]]:
    return {
        key: spec.as_dict()
        for key, spec in reference_clock_registry().items()
        if not key.endswith("_reversed") and spec.application_behavior == "transferred_reference"
    }


def build_schedule_grid(schedule_key: str, n_steps: int) -> tuple[float, ...] | None:
    """Build a fixed reference grid, returning ``None`` only for externally supplied dynamic clocks."""
    key = str(schedule_key).strip().lower()
    try:
        return build_reference_clock_grid(key, n_steps)
    except KeyError:
        return None


def schedule_display_name(schedule_key: str) -> str:
    key = str(schedule_key).strip().lower()
    try:
        return str(reference_clock_provenance(key)["display_name"])
    except KeyError:
        return str(schedule_key)


def schedule_time_alignment(schedule_key: str) -> str:
    key = str(schedule_key).strip().lower()
    try:
        provenance = reference_clock_provenance(key)
    except KeyError:
        return f"runtime_{key}"
    coordinate = str(provenance["coordinate"])
    suffix = "_reversed" if key.endswith("_reversed") else ""
    return f"runtime_{str(provenance['family'])}_{coordinate}{suffix}"


def schedule_density_family(schedule_key: str) -> str:
    key = str(schedule_key).strip().lower()
    try:
        provenance = reference_clock_provenance(key)
    except KeyError:
        return schedule_family_for_key(key)
    return f"{provenance['family']}_{provenance['coordinate']}"


def fixed_schedule_shape_statistics(time_grid: Sequence[float]) -> dict[str, float | None]:
    grid = np.asarray(time_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        return {"runtime_grid_q25": None, "runtime_grid_q50": None, "runtime_grid_q75": None}
    positions = np.linspace(0.0, 1.0, int(grid.size), dtype=np.float64)
    q25, q50, q75 = np.interp(np.asarray([0.25, 0.5, 0.75], dtype=np.float64), positions, grid)
    return {"runtime_grid_q25": float(q25), "runtime_grid_q50": float(q50), "runtime_grid_q75": float(q75)}


__all__ = [
    "BASELINE_SCHEDULE_KEYS",
    "EXPERIMENTAL_AVERAGED_FIXED_SCHEDULE_KEYS",
    "EXPERIMENTAL_FIXED_SCHEDULE_KEYS",
    "EXPERIMENTAL_REVERSED_SCHEDULE_KEYS",
    "TRANSFER_SCHEDULE_KEYS",
    "build_schedule_grid",
    "fixed_schedule_shape_statistics",
    "load_external_schedule_catalog",
    "schedule_density_family",
    "schedule_display_name",
    "schedule_time_alignment",
]
