from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor


PROGRESS_PROTOCOL = "noise_to_data_progress_v1"
NOISE_ENDPOINT = 0.0
DATA_ENDPOINT = 1.0


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a positive integer, got {value!r}.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive, got {parsed}.")
    return parsed


def validate_time_grid(
    time_grid: Tensor,
    *,
    target_nfe: int | None = None,
    batch_size: int | None = None,
) -> Tensor:
    """Validate a canonical noise-to-data progress grid.

    A shared grid has shape ``[steps + 1]``. A per-sample grid has shape
    ``[batch, steps + 1]``. Both forms start at pure noise (0) and end at
    data (1); callers must explicitly transform checkpoints that use another
    convention before entering this API.
    """

    if not isinstance(time_grid, Tensor):
        raise TypeError("time_grid must be a torch.Tensor.")
    if time_grid.ndim not in {1, 2}:
        raise ValueError(f"time_grid must have shape [steps + 1] or [batch, steps + 1], got {tuple(time_grid.shape)}.")
    if not time_grid.is_floating_point():
        raise TypeError("time_grid must use a floating-point dtype.")
    if int(time_grid.shape[-1]) < 2:
        raise ValueError("time_grid must contain at least two progress values.")
    if target_nfe is not None:
        expected = _positive_integer(target_nfe, field="target_nfe") + 1
        if int(time_grid.shape[-1]) != expected:
            raise ValueError(
                f"time_grid has {int(time_grid.shape[-1])} values; target_nfe={target_nfe} requires {expected}."
            )
    if batch_size is not None:
        expected_batch = _positive_integer(batch_size, field="batch_size")
        if time_grid.ndim == 2 and int(time_grid.shape[0]) != expected_batch:
            raise ValueError(f"time_grid batch size is {int(time_grid.shape[0])}; expected {expected_batch}.")
    if not bool(torch.isfinite(time_grid).all()):
        raise ValueError("time_grid must contain only finite values.")
    if not bool(torch.all(time_grid[..., 0] == NOISE_ENDPOINT)):
        raise ValueError(f"time_grid must start at the noise endpoint {NOISE_ENDPOINT}.")
    if not bool(torch.all(time_grid[..., -1] == DATA_ENDPOINT)):
        raise ValueError(f"time_grid must end at the data endpoint {DATA_ENDPOINT}.")
    if not bool(torch.all(torch.diff(time_grid, dim=-1) > 0)):
        raise ValueError("time_grid must be strictly increasing.")
    return time_grid


def uniform_time_grid(
    target_nfe: int,
    *,
    batch_size: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Construct a uniform canonical progress grid."""

    steps = _positive_integer(target_nfe, field="target_nfe")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype.")
    grid = torch.linspace(
        NOISE_ENDPOINT,
        DATA_ENDPOINT,
        steps + 1,
        dtype=dtype,
        device=device,
    )
    if batch_size is None:
        return grid
    rows = _positive_integer(batch_size, field="batch_size")
    return grid.unsqueeze(0).expand(rows, -1).clone()


__all__ = [
    "DATA_ENDPOINT",
    "NOISE_ENDPOINT",
    "PROGRESS_PROTOCOL",
    "uniform_time_grid",
    "validate_time_grid",
]
