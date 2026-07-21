from __future__ import annotations

from typing import Sequence

import numpy as np


NATIVE_INFO_GROWTH_ROW_KEY = "info_growth_hardness"
NATIVE_INFO_GROWTH_TRACE_KEY = "info_growth_hardness_by_step"


def resolved_info_growth_scale(residual_norm_values: Sequence[float]) -> float:
    residual = np.asarray(residual_norm_values, dtype=np.float64)
    residual = residual[np.isfinite(residual)]
    return 1.0 if residual.size == 0 else max(float(np.mean(np.clip(residual, 0.0, None))), 1e-8)


def compute_info_growth_hardness_numpy(
    residual_norm: Sequence[float] | np.ndarray, disagreement: Sequence[float] | np.ndarray, *, scale: float
) -> np.ndarray:
    if float(scale) <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    return np.asarray(disagreement, dtype=np.float64) * np.log1p(
        np.clip(np.asarray(residual_norm, dtype=np.float64), 0.0, None) / float(scale)
    )


__all__ = [
    "NATIVE_INFO_GROWTH_ROW_KEY",
    "NATIVE_INFO_GROWTH_TRACE_KEY",
    "compute_info_growth_hardness_numpy",
    "resolved_info_growth_scale",
]
