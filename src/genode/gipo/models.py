from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

SOLVER_TO_ID: Dict[str, int] = {"euler": 0, "heun": 1, "midpoint_rk2": 2, "dpmpp2m": 3}
TARGET_NFES: Tuple[int, ...] = (4, 8, 12)
NFE_RICH_REFERENCE = 16
SETTING_FEATURE_MODE_GIPO_V1 = "gipo_v1"
SETTING_FEATURE_MODE_NFE_RICH_V1 = "nfe_rich_v1"


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


def validate_setting_feature_mode(mode: str) -> str:
    value = str(mode).strip() or SETTING_FEATURE_MODE_GIPO_V1
    allowed = {SETTING_FEATURE_MODE_GIPO_V1, SETTING_FEATURE_MODE_NFE_RICH_V1}
    if value not in allowed:
        raise ValueError(f"setting_feature_mode must be one of {sorted(allowed)}, got {mode!r}.")
    return value


def setting_feature_dim(mode: str = SETTING_FEATURE_MODE_GIPO_V1) -> int:
    return int(setting_features("euler", 4, mode=mode).numel())


def setting_features(solver_key: str, target_nfe: int, *, mode: str = SETTING_FEATURE_MODE_GIPO_V1) -> torch.Tensor:
    feature_mode = validate_setting_feature_mode(mode)
    solver_id = SOLVER_TO_ID[str(solver_key)]
    solver_one_hot = F.one_hot(torch.tensor(solver_id), num_classes=len(SOLVER_TO_ID)).float()
    if feature_mode == SETTING_FEATURE_MODE_GIPO_V1:
        nfe = torch.tensor([float(target_nfe) / float(max(TARGET_NFES))], dtype=torch.float32)
        order = torch.tensor([1.0 if str(solver_key) == "euler" else 2.0], dtype=torch.float32) / 2.0
        return torch.cat([solver_one_hot, nfe, order], dim=0)

    target = float(target_nfe)
    reference = float(NFE_RICH_REFERENCE)
    macro_steps = float(solver_macro_steps(str(solver_key), int(target_nfe)))
    order = torch.tensor([1.0 if str(solver_key) == "euler" else 2.0], dtype=torch.float32) / 2.0
    nfe_features = torch.tensor(
        [
            target / reference,
            math.log1p(target) / math.log1p(reference),
            float(min(TARGET_NFES)) / max(target, 1.0),
            macro_steps / reference,
        ],
        dtype=torch.float32,
    )
    return torch.cat([solver_one_hot, nfe_features, order], dim=0)
