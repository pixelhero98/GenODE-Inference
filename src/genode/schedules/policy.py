from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, TypeVar, runtime_checkable

import torch
from torch import Tensor

from genode.artifacts.identity import semantic_sha256
from genode.schedules.density import (
    density_mass_hash,
    density_mass_to_time_grid,
    time_grid_hash,
    time_grid_to_density_mass,
    uniform_reference_time_grid,
    validate_density_mass,
    validate_reference_time_grid,
)
from genode.schedules.progress import validate_time_grid
from genode.schedules.specification import ScheduleSpecification


ContextT = TypeVar("ContextT", contravariant=True)
_EXECUTABLE_BINDING_ATOL = 1e-12


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a positive integer, got {value!r}.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class ScheduleBatch:
    """A batch of density representations and executable progress grids."""

    density_mass: Tensor
    reference_time_grid: Tensor
    time_grid: Tensor
    target_nfe: int
    specification: ScheduleSpecification | None = None

    def __post_init__(self) -> None:
        target = _positive_integer(self.target_nfe, field="target_nfe")
        mass = validate_density_mass(
            self.density_mass,
            reference_time_grid=self.reference_time_grid,
        )
        reference = validate_reference_time_grid(self.reference_time_grid)
        grid = validate_time_grid(self.time_grid, target_nfe=target)
        if mass.ndim != 2:
            raise ValueError("ScheduleBatch.density_mass must have shape [batch, bins].")
        if grid.ndim != 2:
            raise ValueError("ScheduleBatch.time_grid must have shape [batch, target_nfe + 1].")
        if int(mass.shape[0]) != int(grid.shape[0]):
            raise ValueError("ScheduleBatch density_mass and time_grid batch sizes differ.")
        if mass.device != grid.device or mass.device != reference.device:
            raise ValueError("ScheduleBatch tensors must all be on the same device.")
        if mass.dtype != grid.dtype or mass.dtype != reference.dtype:
            raise TypeError("ScheduleBatch tensors must all use the same dtype.")
        executable_grid = density_mass_to_time_grid(
            mass,
            target_nfe=target,
            reference_time_grid=reference,
        )
        if not torch.allclose(
            grid.to(dtype=torch.float64),
            executable_grid.to(dtype=torch.float64),
            rtol=0.0,
            atol=_EXECUTABLE_BINDING_ATOL,
        ):
            raise ValueError(
                "ScheduleBatch time_grid is not the executable quantile grid "
                "of its density_mass and reference_time_grid."
            )
        if self.specification is not None and not isinstance(
            self.specification,
            ScheduleSpecification,
        ):
            raise TypeError("specification must be a ScheduleSpecification or None.")
        object.__setattr__(self, "target_nfe", target)
        object.__setattr__(self, "time_grid", executable_grid)

    @classmethod
    def from_density_mass(
        cls,
        density_mass: Tensor,
        *,
        target_nfe: int,
        reference_time_grid: Tensor | None = None,
        specification: ScheduleSpecification | None = None,
    ) -> ScheduleBatch:
        if not isinstance(density_mass, Tensor):
            raise TypeError("density_mass must be a torch.Tensor.")
        if density_mass.ndim != 2:
            raise ValueError("density_mass must have shape [batch, bins].")
        reference = (
            uniform_reference_time_grid(
                int(density_mass.shape[-1]),
                dtype=density_mass.dtype,
                device=density_mass.device,
            )
            if reference_time_grid is None
            else reference_time_grid
        )
        grid = density_mass_to_time_grid(
            density_mass,
            target_nfe=target_nfe,
            reference_time_grid=reference,
        )
        return cls(
            density_mass=density_mass,
            reference_time_grid=reference,
            time_grid=grid,
            target_nfe=target_nfe,
            specification=specification,
        )

    @classmethod
    def from_time_grid(
        cls,
        time_grid: Tensor,
        *,
        reference_time_grid: Tensor | None = None,
        specification: ScheduleSpecification | None = None,
    ) -> ScheduleBatch:
        if not isinstance(time_grid, Tensor):
            raise TypeError("time_grid must be a torch.Tensor.")
        if time_grid.ndim != 2:
            raise ValueError("time_grid must have shape [batch, target_nfe + 1].")
        target_nfe = int(time_grid.shape[-1]) - 1
        reference = (
            uniform_reference_time_grid(
                dtype=time_grid.dtype,
                device=time_grid.device,
            )
            if reference_time_grid is None
            else reference_time_grid
        )
        mass = time_grid_to_density_mass(
            time_grid,
            reference_time_grid=reference,
        )
        return cls(
            density_mass=mass,
            reference_time_grid=reference,
            time_grid=time_grid,
            target_nfe=target_nfe,
            specification=specification,
        )

    @property
    def batch_size(self) -> int:
        return int(self.density_mass.shape[0])

    @property
    def density_bin_count(self) -> int:
        return int(self.density_mass.shape[-1])

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            {
                "target_nfe": self.target_nfe,
                "specification_sha256": (None if self.specification is None else self.specification.sha256),
                "density_mass_sha256": density_mass_hash(
                    self.density_mass,
                    reference_time_grid=self.reference_time_grid,
                ),
                "time_grid_sha256": time_grid_hash(self.time_grid),
            },
            namespace="schedule-batch",
        )


@runtime_checkable
class SchedulePolicy(Protocol[ContextT]):
    """Predict executable Euler schedules from benchmark-defined context."""

    def predict(
        self,
        context: ContextT,
        *,
        target_nfe: int,
    ) -> ScheduleBatch: ...


__all__ = [
    "ScheduleBatch",
    "SchedulePolicy",
]
