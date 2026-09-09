"""Solver-grid validation shared with native adapters."""

import math
from collections.abc import Sequence


def validate_time_grid(grid: Sequence[float], *, macro_steps: int) -> tuple[float, ...]:
    values = tuple(float(x) for x in grid)
    if isinstance(macro_steps, bool) or int(macro_steps) != macro_steps or macro_steps < 1:
        raise ValueError("macro_steps must be a positive integer.")
    if len(values) != int(macro_steps) + 1:
        raise ValueError(f"Grid length {len(values)} does not match macro_steps={macro_steps}.")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Schedule grid contains non-finite values.")
    if abs(values[0]) > 1e-8 or abs(values[-1] - 1.0) > 1e-8:
        raise ValueError("Schedule grid must start at 0.0 and end at 1.0.")
    if not all(b > a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError("Schedule grid must be strictly increasing.")
    return values
