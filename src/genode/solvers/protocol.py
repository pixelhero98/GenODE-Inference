from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from genode.schedules.progress import validate_time_grid


EULER_SOLVER_KEY = "euler"


def validate_target_nfe(target_nfe: object) -> int:
    if isinstance(target_nfe, bool) or not isinstance(target_nfe, Integral):
        raise TypeError(f"target_nfe must be a positive integer, got {target_nfe!r}.")
    parsed = int(target_nfe)
    if parsed <= 0:
        raise ValueError(f"target_nfe must be positive, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class SolverSpecification:
    solver_key: str
    target_nfe: int

    def __post_init__(self) -> None:
        if self.solver_key != EULER_SOLVER_KEY:
            raise ValueError(f"solver_key must be {EULER_SOLVER_KEY!r}, got {self.solver_key!r}.")
        object.__setattr__(
            self,
            "target_nfe",
            validate_target_nfe(self.target_nfe),
        )

    @property
    def macro_steps(self) -> int:
        return self.target_nfe

    @property
    def expected_field_evaluations(self) -> int:
        return self.target_nfe


@dataclass(frozen=True)
class SolverResult:
    specification: SolverSpecification
    initial_state: Tensor
    final_state: Tensor
    time_grid: Tensor
    field_evaluations: int
    trajectory: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.specification, SolverSpecification):
            raise TypeError("specification must be a SolverSpecification.")
        for field_name, state in (
            ("initial_state", self.initial_state),
            ("final_state", self.final_state),
        ):
            if not isinstance(state, Tensor):
                raise TypeError(f"{field_name} must be a torch.Tensor.")
            if state.ndim < 1 or int(state.shape[0]) <= 0:
                raise ValueError(f"{field_name} must have a non-empty batch dimension.")
            if not state.is_floating_point():
                raise TypeError(f"{field_name} must use a floating-point dtype.")
            if not bool(torch.isfinite(state).all()):
                raise ValueError(f"{field_name} must contain only finite values.")
        if self.initial_state.shape != self.final_state.shape:
            raise ValueError("initial_state and final_state shapes must match.")
        if self.initial_state.device != self.final_state.device:
            raise ValueError("initial_state and final_state devices must match.")
        if self.initial_state.dtype != self.final_state.dtype:
            raise TypeError("initial_state and final_state dtypes must match.")
        grid = validate_time_grid(
            self.time_grid,
            target_nfe=self.specification.target_nfe,
            batch_size=int(self.initial_state.shape[0]),
        )
        if grid.device != self.initial_state.device:
            raise ValueError("time_grid and states must be on the same device.")
        if grid.dtype != self.initial_state.dtype:
            raise TypeError("time_grid and states must use the same dtype.")
        if isinstance(self.field_evaluations, bool) or not isinstance(self.field_evaluations, Integral):
            raise TypeError("field_evaluations must be an integer.")
        realized = int(self.field_evaluations)
        if realized != self.specification.expected_field_evaluations:
            raise ValueError(f"field_evaluations={realized} does not match target_nfe={self.specification.target_nfe}.")
        object.__setattr__(self, "field_evaluations", realized)
        if self.trajectory is not None:
            expected_shape = (
                int(self.initial_state.shape[0]),
                self.specification.macro_steps + 1,
                *self.initial_state.shape[1:],
            )
            if tuple(self.trajectory.shape) != expected_shape:
                raise ValueError(f"trajectory has shape {tuple(self.trajectory.shape)}; expected {expected_shape}.")
            if self.trajectory.device != self.initial_state.device:
                raise ValueError("trajectory and states must be on the same device.")
            if self.trajectory.dtype != self.initial_state.dtype:
                raise TypeError("trajectory and states must use the same dtype.")
            if not bool(torch.isfinite(self.trajectory).all()):
                raise ValueError("trajectory must contain only finite values.")
            if not torch.equal(self.trajectory[:, 0], self.initial_state):
                raise ValueError("trajectory does not start at initial_state.")
            if not torch.equal(self.trajectory[:, -1], self.final_state):
                raise ValueError("trajectory does not end at final_state.")

    @property
    def solver_key(self) -> str:
        return self.specification.solver_key

    @property
    def target_nfe(self) -> int:
        return self.specification.target_nfe

    @property
    def macro_steps(self) -> int:
        return self.specification.macro_steps

    @property
    def realized_nfe(self) -> int:
        return self.field_evaluations


@runtime_checkable
class VelocityField(Protocol):
    """A velocity field over canonical noise-to-data progress."""

    def __call__(self, state: Tensor, progress: Tensor, /) -> Tensor: ...


@runtime_checkable
class Solver(Protocol):
    specification: SolverSpecification

    def integrate(
        self,
        field: VelocityField,
        initial_state: Tensor,
        *,
        time_grid: Tensor,
        return_trajectory: bool = False,
    ) -> SolverResult: ...


__all__ = [
    "EULER_SOLVER_KEY",
    "Solver",
    "SolverResult",
    "SolverSpecification",
    "VelocityField",
    "validate_target_nfe",
]
