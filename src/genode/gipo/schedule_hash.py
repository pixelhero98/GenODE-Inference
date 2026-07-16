from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence


def json_hash(payload: Mapping[str, object], *, prefix: str) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def schedule_grid_hash(time_grid: Sequence[float]) -> str:
    values = [round(float(value), 12) for value in time_grid]
    return json_hash({"time_grid": values}, prefix="grid")


__all__ = ["json_hash", "schedule_grid_hash"]
