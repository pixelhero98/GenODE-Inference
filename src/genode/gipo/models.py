from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

SOLVER_TO_ID: Dict[str, int] = {"euler": 0, "heun": 1, "midpoint_rk2": 2, "dpmpp2m": 3}
TARGET_NFES: Tuple[int, ...] = (4, 8, 12)


def solver_macro_steps(solver_key: str, target_nfe: int) -> int:
    if str(solver_key) in {"heun", "midpoint_rk2"}:
        if int(target_nfe) % 2 != 0:
            raise ValueError(f"{solver_key} requires an even target NFE, got {target_nfe}")
        return int(target_nfe) // 2
    if str(solver_key) in {"euler", "dpmpp2m"}:
        return int(target_nfe)
    raise ValueError(f"Unknown solver_key={solver_key}")


def validate_time_grid(grid: Sequence[float], *, macro_steps: int) -> Tuple[float, ...]:
    values = tuple(float(x) for x in grid)
    if len(values) != int(macro_steps) + 1:
        raise ValueError(f"Grid length {len(values)} does not match macro_steps={macro_steps}.")
    if abs(values[0]) > 1e-8 or abs(values[-1] - 1.0) > 1e-8:
        raise ValueError("Schedule grid must start at 0.0 and end at 1.0.")
    if not all(torch.isfinite(torch.tensor(values)).tolist()):
        raise ValueError("Schedule grid contains non-finite values.")
    if not all(b > a for a, b in zip(values, values[1:])):
        raise ValueError("Schedule grid must be strictly increasing.")
    return values


def setting_features(solver_key: str, target_nfe: int) -> torch.Tensor:
    solver_id = SOLVER_TO_ID[str(solver_key)]
    solver_one_hot = F.one_hot(torch.tensor(solver_id), num_classes=len(SOLVER_TO_ID)).float()
    nfe = torch.tensor([float(target_nfe) / float(max(TARGET_NFES))], dtype=torch.float32)
    order = torch.tensor([1.0 if str(solver_key) == "euler" else 2.0], dtype=torch.float32) / 2.0
    return torch.cat([solver_one_hot, nfe, order], dim=0)
