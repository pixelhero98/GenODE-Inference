"""The shared realized density/clock contract for every GICO task."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from genode.gico.density_representation import density_mass_to_time_grid, grid_to_density_mass
from genode.gico.networks import DENSITY_BINS, DENSITY_MIXTURE
from genode.schedule_transfer.reference_clocks import build_reference_clock_grid, reference_clock_keys
from genode.solver_protocol import normalize_solver_nfe_fields, solver_macro_steps

REFERENCE_KEYS = reference_clock_keys(("3",))


def validate_mass(mass) -> np.ndarray:
    values = np.asarray(mass, dtype=np.float64)
    if values.shape != (DENSITY_BINS,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("GICO density requires 64 finite nonnegative masses.")
    if not np.isclose(values.sum(), 1, atol=1e-12, rtol=1e-7):
        raise ValueError("Density must sum to one.")
    return values


def materialize(mass, solver: str, nfe: int) -> tuple[float, ...]:
    values = validate_mass(mass)
    guarded = (1 - DENSITY_MIXTURE) * values + DENSITY_MIXTURE / DENSITY_BINS
    steps = solver_macro_steps(solver, nfe)
    grid = density_mass_to_time_grid(guarded, macro_steps=steps, eps=0)
    normalize_solver_nfe_fields(solver, nfe, macro_steps=steps, realized_nfe=nfe, source="GICO density")
    if not np.all(np.diff(np.asarray(grid, dtype=np.float32)) > 0):
        raise ValueError("Realized clock nodes collapse in solver precision.")
    return grid


def reference_densities(solver: str, nfe: int) -> dict[str, tuple[float, ...]]:
    steps = solver_macro_steps(solver, nfe)
    return {
        key: grid_to_density_mass(build_reference_clock_grid(key, steps), macro_steps=steps, eps=0)
        for key in REFERENCE_KEYS
    }


def verify_measurement_clock(row: dict) -> None:
    mass = validate_mass(row["density_mass"])
    declared = np.asarray(row["time_grid"], dtype=np.float64)
    expected = np.asarray(materialize(mass, row["solver"], row["nfe"]))
    if declared.shape != expected.shape or not np.array_equal(declared, expected):
        raise ValueError("Measured clock differs from its 64-bin realization; recollect terminal evidence.")
    if row["schedule_key"] == "uniform" and not np.allclose(mass, 1 / DENSITY_BINS, atol=1e-12, rtol=0):
        raise ValueError("Uniform anchor must use the uniform density.")
    if row["schedule_key"] in REFERENCE_KEYS:
        steps = solver_macro_steps(row["solver"], row["nfe"])
        reference = grid_to_density_mass(
            build_reference_clock_grid(row["schedule_key"], steps), macro_steps=steps, eps=0
        )
        if not np.allclose(mass, reference, atol=1e-12, rtol=0):
            raise ValueError("Reference name does not match its realized density.")


def density_identity(mass) -> str:
    return hashlib.sha256(validate_mass(mass).astype("<f8").tobytes()).hexdigest()


def clock_generator(seed: int, request_id: str, *, device: str = "cpu") -> torch.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int) or not isinstance(request_id, str) or not request_id:
        raise ValueError("Clock sampling requires an integer seed and nonempty request identity.")
    digest = hashlib.sha256(f"genode-clock-rng-v1\0{seed}\0{request_id}".encode()).digest()
    return torch.Generator(device=device).manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
