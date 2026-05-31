from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

SOLVER_TO_ID: Dict[str, int] = {"euler": 0, "heun": 1, "midpoint_rk2": 2, "dpmpp2m": 3}
TARGET_NFES: Tuple[int, ...] = (4, 8, 12)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int) -> nn.Sequential:
    layers = []
    dim = int(input_dim)
    for _ in range(int(hidden_layers)):
        layers.extend([nn.Linear(dim, int(hidden_dim)), nn.SiLU()])
        dim = int(hidden_dim)
    layers.append(nn.Linear(dim, int(output_dim)))
    return nn.Sequential(*layers)


class ScheduleTeacherMLP(nn.Module):
    """Single MLP teacher that scores a setting/grid pair."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, hidden_layers: int = 3):
        super().__init__()
        self.net = _mlp(int(input_dim), int(hidden_dim), 1, int(hidden_layers))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class ScheduleStudentMLP(nn.Module):
    """Smaller MLP policy that emits schedule interval logits."""

    def __init__(self, setting_dim: int, max_macro_steps: int = 12, hidden_dim: int = 128, hidden_layers: int = 2):
        super().__init__()
        self.max_macro_steps = int(max_macro_steps)
        self.net = _mlp(int(setting_dim), int(hidden_dim), int(max_macro_steps), int(hidden_layers))

    def interval_logits(self, setting_features: torch.Tensor) -> torch.Tensor:
        return self.net(setting_features)

    def intervals(self, setting_features: torch.Tensor, macro_steps: int) -> torch.Tensor:
        logits = self.interval_logits(setting_features)[..., : int(macro_steps)]
        return F.softmax(logits, dim=-1)

    def time_grid(self, setting_features: torch.Tensor, macro_steps: int) -> torch.Tensor:
        weights = self.intervals(setting_features, int(macro_steps))
        zeros = torch.zeros(*weights.shape[:-1], 1, dtype=weights.dtype, device=weights.device)
        grid = torch.cat([zeros, torch.cumsum(weights, dim=-1)], dim=-1)
        grid[..., -1] = 1.0
        return grid


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


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


def grid_to_intervals(grid: Sequence[float]) -> Tuple[float, ...]:
    values = tuple(float(x) for x in grid)
    return tuple(float(b - a) for a, b in zip(values, values[1:]))


def setting_features(solver_key: str, target_nfe: int) -> torch.Tensor:
    solver_id = SOLVER_TO_ID[str(solver_key)]
    solver_one_hot = F.one_hot(torch.tensor(solver_id), num_classes=len(SOLVER_TO_ID)).float()
    nfe = torch.tensor([float(target_nfe) / float(max(TARGET_NFES))], dtype=torch.float32)
    order = torch.tensor([1.0 if str(solver_key) == "euler" else 2.0], dtype=torch.float32) / 2.0
    return torch.cat([solver_one_hot, nfe, order], dim=0)


def teacher_features(solver_key: str, target_nfe: int, grid: Sequence[float], max_macro_steps: int = 12) -> torch.Tensor:
    macro_steps = solver_macro_steps(solver_key, int(target_nfe))
    checked = validate_time_grid(grid, macro_steps=macro_steps)
    intervals = torch.zeros(int(max_macro_steps), dtype=torch.float32)
    raw_intervals = torch.tensor(grid_to_intervals(checked), dtype=torch.float32)
    intervals[: raw_intervals.numel()] = raw_intervals
    return torch.cat([setting_features(solver_key, int(target_nfe)), intervals], dim=0)


@dataclass(frozen=True)
class SchedulePrediction:
    solver_key: str
    target_nfe: int
    macro_steps: int
    time_grid: Tuple[float, ...]

    @classmethod
    def from_student(cls, student: ScheduleStudentMLP, solver_key: str, target_nfe: int) -> "SchedulePrediction":
        macro_steps = solver_macro_steps(str(solver_key), int(target_nfe))
        with torch.no_grad():
            grid = student.time_grid(setting_features(str(solver_key), int(target_nfe))[None, :], macro_steps)[0]
        values = validate_time_grid([float(x) for x in grid.cpu().tolist()], macro_steps=macro_steps)
        return cls(str(solver_key), int(target_nfe), int(macro_steps), values)
