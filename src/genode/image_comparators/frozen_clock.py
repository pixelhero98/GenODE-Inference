"""Compose a density clock with a frozen two-time sampler's learned grids."""

import numpy as np


def warp_frozen_grids(clock, integration, model_times):
    """Interpolate both original grids by normalized node index, preserving endpoints.

    This preserves the original sampler at a uniform clock. It does not refit
    either time grid or the transformed field, and consumes no backbone calls.
    """
    u, first, second = (np.asarray(x, dtype=np.float64) for x in (clock, integration, model_times))
    if (
        u.ndim != 1
        or len(u) < 2
        or first.shape != u.shape
        or second.shape != u.shape
        or not all(np.isfinite(x).all() for x in (u, first, second))
        or u[0] != 0
        or u[-1] != 1
        or np.any(np.diff(u) <= 0)
        or np.any(np.diff(first) <= 0)
        or np.any(np.diff(second) < 0)
    ):
        raise ValueError("Frozen clock warp requires increasing integration and nondecreasing model grids.")
    nodes = np.linspace(0, 1, len(u))
    return np.interp(u, nodes, first), np.interp(u, nodes, second)
