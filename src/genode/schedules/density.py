from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor

from genode.artifacts.identity import semantic_sha256
from genode.schedules.progress import PROGRESS_PROTOCOL, uniform_time_grid, validate_time_grid

DENSITY_MASS_PROTOCOL = "progress_density_mass_v1"
DEFAULT_DENSITY_BIN_COUNT = 64
_SUM_TOLERANCE = 1e-6


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a positive integer, got {value!r}.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive, got {parsed}.")
    return parsed


def uniform_reference_time_grid(
    bin_count: int = DEFAULT_DENSITY_BIN_COUNT,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return uniform reference-bin edges on canonical progress."""

    return uniform_time_grid(
        _positive_integer(bin_count, field="bin_count"),
        dtype=dtype,
        device=device,
    )


def validate_reference_time_grid(reference_time_grid: Tensor) -> Tensor:
    if not isinstance(reference_time_grid, Tensor):
        raise TypeError("reference_time_grid must be a torch.Tensor.")
    if reference_time_grid.ndim != 1:
        raise ValueError(
            f"reference_time_grid must have shape [bin_count + 1], got {tuple(reference_time_grid.shape)}."
        )
    return validate_time_grid(reference_time_grid)


def validate_density_mass(
    density_mass: Tensor,
    *,
    reference_time_grid: Tensor | None = None,
) -> Tensor:
    """Validate normalized probability mass over canonical progress bins."""

    if not isinstance(density_mass, Tensor):
        raise TypeError("density_mass must be a torch.Tensor.")
    if density_mass.ndim not in {1, 2}:
        raise ValueError(f"density_mass must have shape [bins] or [batch, bins], got {tuple(density_mass.shape)}.")
    if not density_mass.is_floating_point():
        raise TypeError("density_mass must use a floating-point dtype.")
    if int(density_mass.shape[-1]) <= 0:
        raise ValueError("density_mass must contain at least one bin.")
    if not bool(torch.isfinite(density_mass).all()):
        raise ValueError("density_mass must contain only finite values.")
    if not bool(torch.all(density_mass >= 0)):
        raise ValueError("density_mass must be nonnegative.")
    totals = density_mass.to(dtype=torch.float64).sum(dim=-1)
    if not bool(torch.all(totals > 0)):
        raise ValueError("density_mass must have positive total mass.")
    if not bool(
        torch.allclose(
            totals,
            torch.ones_like(totals),
            rtol=_SUM_TOLERANCE,
            atol=_SUM_TOLERANCE,
        )
    ):
        raise ValueError("density_mass must sum to 1 along its last dimension.")
    if reference_time_grid is not None:
        reference = validate_reference_time_grid(reference_time_grid)
        if int(reference.numel()) != int(density_mass.shape[-1]) + 1:
            raise ValueError("reference_time_grid must contain len(density_mass) + 1 edges.")
    return density_mass


def normalize_density_mass(weights: Tensor) -> Tensor:
    """Explicitly normalize nonnegative finite weights into density mass."""

    if not isinstance(weights, Tensor):
        raise TypeError("weights must be a torch.Tensor.")
    if weights.ndim not in {1, 2}:
        raise ValueError(f"weights must have shape [bins] or [batch, bins], got {tuple(weights.shape)}.")
    if not weights.is_floating_point():
        raise TypeError("weights must use a floating-point dtype.")
    if int(weights.shape[-1]) <= 0:
        raise ValueError("weights must contain at least one bin.")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("weights must contain only finite values.")
    if not bool(torch.all(weights >= 0)):
        raise ValueError("weights must be nonnegative.")
    totals = weights.sum(dim=-1, keepdim=True)
    if not bool(torch.all(totals > 0)):
        raise ValueError("weights must have positive total mass.")
    normalized = weights / totals
    return validate_density_mass(normalized)


def _reference_for_density(
    density_mass: Tensor,
    reference_time_grid: Tensor | None,
) -> Tensor:
    reference = (
        uniform_reference_time_grid(
            int(density_mass.shape[-1]),
            dtype=density_mass.dtype,
            device=density_mass.device,
        )
        if reference_time_grid is None
        else validate_reference_time_grid(reference_time_grid)
    )
    if reference.device != density_mass.device:
        raise ValueError("reference_time_grid and density_mass must be on the same device.")
    if reference.dtype != density_mass.dtype:
        raise TypeError("reference_time_grid and density_mass must use the same dtype.")
    if int(reference.numel()) != int(density_mass.shape[-1]) + 1:
        raise ValueError("reference_time_grid must contain len(density_mass) + 1 edges.")
    return reference


def density_mass_to_time_grid(
    density_mass: Tensor,
    *,
    target_nfe: int,
    reference_time_grid: Tensor | None = None,
) -> Tensor:
    """Convert density mass to equal-mass solver quantiles.

    The returned grid has the same rank as ``density_mass``. Each solver
    interval contains ``1 / target_nfe`` probability mass.
    """

    mass = validate_density_mass(
        density_mass,
        reference_time_grid=reference_time_grid,
    )
    steps = _positive_integer(target_nfe, field="target_nfe")
    reference = _reference_for_density(mass, reference_time_grid)
    squeeze = mass.ndim == 1
    rows = mass.unsqueeze(0) if squeeze else mass
    batch_size = int(rows.shape[0])

    if steps == 1:
        grid = uniform_time_grid(
            1,
            batch_size=batch_size,
            dtype=rows.dtype,
            device=rows.device,
        )
        return grid[0] if squeeze else grid

    cumulative = torch.cumsum(rows, dim=-1)
    quantiles = torch.arange(
        1,
        steps,
        dtype=rows.dtype,
        device=rows.device,
    ) / float(steps)
    quantiles = quantiles.unsqueeze(0).expand(batch_size, -1).contiguous()
    indices = torch.searchsorted(
        cumulative.contiguous(),
        quantiles,
        right=False,
    ).clamp_max(int(rows.shape[-1]) - 1)
    previous_indices = (indices - 1).clamp_min(0)
    cumulative_left = torch.gather(cumulative, 1, previous_indices)
    cumulative_left = torch.where(
        indices == 0,
        torch.zeros_like(cumulative_left),
        cumulative_left,
    )
    selected_mass = torch.gather(rows, 1, indices)
    if not bool(torch.all(selected_mass > 0)):
        raise ValueError("density_mass has zero-mass bins at requested solver quantiles.")
    left_edges = reference[:-1][indices]
    right_edges = reference[1:][indices]
    fraction = (quantiles - cumulative_left) / selected_mass
    interior = left_edges + fraction * (right_edges - left_edges)
    grid = torch.cat(
        (
            torch.zeros(
                (batch_size, 1),
                dtype=rows.dtype,
                device=rows.device,
            ),
            interior,
            torch.ones(
                (batch_size, 1),
                dtype=rows.dtype,
                device=rows.device,
            ),
        ),
        dim=1,
    )
    validate_time_grid(grid, target_nfe=steps, batch_size=batch_size)
    return grid[0] if squeeze else grid


def time_grid_to_density_mass(
    time_grid: Tensor,
    *,
    reference_time_grid: Tensor | None = None,
) -> Tensor:
    """Represent an equal-step schedule as mass on reference progress bins."""

    grid = validate_time_grid(time_grid)
    squeeze = grid.ndim == 1
    rows = grid.unsqueeze(0) if squeeze else grid
    bin_count = (
        DEFAULT_DENSITY_BIN_COUNT
        if reference_time_grid is None
        else int(validate_reference_time_grid(reference_time_grid).numel()) - 1
    )
    reference = (
        uniform_reference_time_grid(
            bin_count,
            dtype=rows.dtype,
            device=rows.device,
        )
        if reference_time_grid is None
        else validate_reference_time_grid(reference_time_grid)
    )
    if reference.device != rows.device:
        raise ValueError("reference_time_grid and time_grid must be on the same device.")
    if reference.dtype != rows.dtype:
        raise TypeError("reference_time_grid and time_grid must use the same dtype.")

    left = rows[:, :-1].unsqueeze(-1)
    right = rows[:, 1:].unsqueeze(-1)
    widths = right - left
    reference_left = reference[:-1].reshape(1, 1, -1)
    reference_right = reference[1:].reshape(1, 1, -1)
    overlap = (torch.minimum(right, reference_right) - torch.maximum(left, reference_left)).clamp_min(0)
    step_count = int(rows.shape[-1]) - 1
    local_density = 1.0 / (float(step_count) * widths)
    mass = torch.sum(local_density * overlap, dim=1)
    mass = normalize_density_mass(mass)
    return mass[0] if squeeze else mass


def _tensor_payload(tensor: Tensor) -> dict[str, object]:
    values = tensor.detach().to(device="cpu", dtype=torch.float64).tolist()
    return {
        "shape": [int(value) for value in tensor.shape],
        "values": values,
    }


def reference_time_grid_hash(reference_time_grid: Tensor) -> str:
    reference = validate_reference_time_grid(reference_time_grid)
    return semantic_sha256(
        {
            "progress_protocol": PROGRESS_PROTOCOL,
            "reference_time_grid": _tensor_payload(reference),
        },
        namespace="schedule-reference-grid",
    )


def density_mass_hash(
    density_mass: Tensor,
    *,
    reference_time_grid: Tensor | None = None,
) -> str:
    mass = validate_density_mass(
        density_mass,
        reference_time_grid=reference_time_grid,
    )
    reference = _reference_for_density(mass, reference_time_grid)
    return semantic_sha256(
        {
            "density_mass_protocol": DENSITY_MASS_PROTOCOL,
            "progress_protocol": PROGRESS_PROTOCOL,
            "reference_time_grid_sha256": reference_time_grid_hash(reference),
            "density_mass": _tensor_payload(mass),
        },
        namespace="schedule-density-mass",
    )


def time_grid_hash(time_grid: Tensor) -> str:
    grid = validate_time_grid(time_grid)
    return semantic_sha256(
        {
            "progress_protocol": PROGRESS_PROTOCOL,
            "time_grid": _tensor_payload(grid),
        },
        namespace="schedule-time-grid",
    )


__all__ = [
    "DEFAULT_DENSITY_BIN_COUNT",
    "DENSITY_MASS_PROTOCOL",
    "density_mass_hash",
    "density_mass_to_time_grid",
    "normalize_density_mass",
    "reference_time_grid_hash",
    "time_grid_hash",
    "time_grid_to_density_mass",
    "uniform_reference_time_grid",
    "validate_density_mass",
    "validate_reference_time_grid",
]
