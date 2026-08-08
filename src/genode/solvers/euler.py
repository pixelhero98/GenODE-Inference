from __future__ import annotations

import torch
from torch import Tensor

from genode.schedules.progress import validate_time_grid
from genode.solvers.protocol import (
    EULER_SOLVER_KEY,
    SolverResult,
    SolverSpecification,
    VelocityField,
)


def _validate_initial_state(initial_state: Tensor) -> Tensor:
    if not isinstance(initial_state, Tensor):
        raise TypeError("initial_state must be a torch.Tensor.")
    if initial_state.ndim < 1 or int(initial_state.shape[0]) <= 0:
        raise ValueError("initial_state must have a non-empty batch dimension.")
    if not initial_state.is_floating_point():
        raise TypeError("initial_state must use a floating-point dtype.")
    if not bool(torch.isfinite(initial_state).all()):
        raise ValueError("initial_state must contain only finite values.")
    return initial_state


def integrate_euler(
    field: VelocityField,
    initial_state: Tensor,
    *,
    target_nfe: int,
    time_grid: Tensor,
    return_trajectory: bool = False,
) -> SolverResult:
    """Integrate a batch with explicit Euler and exact field-call accounting.

    ``initial_state`` may have any rank at least one; its leading dimension is
    the batch. ``time_grid`` may be shared across the batch or contain one
    grid per sample. The field receives progress as a rank-one ``[batch]``
    tensor, leaving benchmark-specific conditioning to an explicit closure or
    adapter.
    """

    if not callable(field):
        raise TypeError("field must be callable.")
    state = _validate_initial_state(initial_state)
    specification = SolverSpecification(
        solver_key=EULER_SOLVER_KEY,
        target_nfe=target_nfe,
    )
    grid = validate_time_grid(
        time_grid,
        target_nfe=specification.target_nfe,
        batch_size=int(state.shape[0]),
    )
    if grid.device != state.device:
        raise ValueError("time_grid and initial_state must be on the same device.")
    if grid.dtype != state.dtype:
        raise TypeError("time_grid and initial_state must use the same dtype.")

    batch_size = int(state.shape[0])
    initial = state
    states = [state] if return_trajectory else None
    field_evaluations = 0
    for step_index in range(specification.macro_steps):
        if grid.ndim == 1:
            progress = grid[step_index].expand(batch_size)
            step_size = (grid[step_index + 1] - grid[step_index]).expand(batch_size)
        else:
            progress = grid[:, step_index]
            step_size = grid[:, step_index + 1] - progress
        velocity = field(state, progress)
        field_evaluations += 1
        if not isinstance(velocity, Tensor):
            raise TypeError("Velocity field must return a torch.Tensor.")
        if velocity.shape != state.shape:
            raise ValueError(f"Velocity field returned shape {tuple(velocity.shape)}; expected {tuple(state.shape)}.")
        if velocity.device != state.device:
            raise ValueError("Velocity field output and state must be on the same device.")
        if velocity.dtype != state.dtype:
            raise TypeError("Velocity field output and state must use the same dtype.")
        if not bool(torch.isfinite(velocity).all()):
            raise ValueError(f"Velocity field returned non-finite values at step {step_index}.")
        broadcast_shape = (batch_size,) + (1,) * (state.ndim - 1)
        state = state + step_size.reshape(broadcast_shape) * velocity
        if not bool(torch.isfinite(state).all()):
            raise ValueError(f"Euler integration produced non-finite state at step {step_index}.")
        if states is not None:
            states.append(state)

    trajectory = None if states is None else torch.stack(states, dim=1)
    return SolverResult(
        specification=specification,
        initial_state=initial,
        final_state=state,
        time_grid=grid,
        field_evaluations=field_evaluations,
        trajectory=trajectory,
    )


class EulerSolver:
    """Configured Euler solver for one supported NFE budget."""

    def __init__(self, target_nfe: int) -> None:
        self.specification = SolverSpecification(
            solver_key=EULER_SOLVER_KEY,
            target_nfe=target_nfe,
        )

    def integrate(
        self,
        field: VelocityField,
        initial_state: Tensor,
        *,
        time_grid: Tensor,
        return_trajectory: bool = False,
    ) -> SolverResult:
        return integrate_euler(
            field,
            initial_state,
            target_nfe=self.specification.target_nfe,
            time_grid=time_grid,
            return_trajectory=return_trajectory,
        )


__all__ = [
    "EulerSolver",
    "integrate_euler",
]
