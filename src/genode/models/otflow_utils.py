"""otflow_utils.py

Shared utility functions for OTFlow.

Consolidates helpers used across the OTFlow runtime and paper tooling.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


# -------------------------------------------
# Dict flattening / unflattening
# -------------------------------------------
def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Recursively flatten a nested dict to dotted-path -> float entries."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, kk))
        elif isinstance(v, (int, float, np.floating, np.integer, bool)):
            out[kk] = float(v)
    return out


def unflatten_to_nested(flat_aggs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Reverse of flatten_dict (one level of aggregation stats per key)."""
    root: Dict[str, Any] = {}
    for path, stats in flat_aggs.items():
        cur = root
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = stats
    return root


# -------------------------------------------
# Microstructure series from raw L2 book
# -------------------------------------------
def microstructure_series(
    ask_p: np.ndarray,
    ask_v: np.ndarray,
    bid_p: np.ndarray,
    bid_v: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Compute common microstructure series from raw L2 arrays.

    Returns dict with keys: mid, spread, depth, imb, ret.
    """
    mid = 0.5 * (ask_p[:, 0] + bid_p[:, 0])
    spread = ask_p[:, 0] - bid_p[:, 0]
    depth = ask_v.sum(axis=1) + bid_v.sum(axis=1)
    imb = (bid_v.sum(axis=1) - ask_v.sum(axis=1)) / (depth + eps)
    ret = np.zeros_like(mid)
    if len(mid) > 1:
        ret[1:] = np.diff(mid)
    return {
        "mid": mid.astype(np.float32),
        "spread": spread.astype(np.float32),
        "depth": depth.astype(np.float32),
        "imb": imb.astype(np.float32),
        "ret": ret.astype(np.float32),
    }


def keep_last_snapshot_per_bucket(timestamps: np.ndarray, bucket_ns: int) -> np.ndarray:
    """Return a boolean mask that keeps the last row inside each time bucket."""
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be 1D")
    if len(timestamps) == 0:
        return np.zeros(0, dtype=bool)
    if bucket_ns <= 0:
        return np.ones(len(timestamps), dtype=bool)

    buckets = timestamps.astype(np.int64) // int(bucket_ns)
    keep = np.empty(len(timestamps), dtype=bool)
    keep[-1] = True
    keep[:-1] = buckets[:-1] != buckets[1:]
    return keep


__all__ = [
    "flatten_dict",
    "unflatten_to_nested",
    "microstructure_series",
    "keep_last_snapshot_per_bucket",
]
