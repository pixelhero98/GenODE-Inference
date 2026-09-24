from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
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
    gico_density_mass: Tensor | None = None

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
        if self.specification is not None and not isinstance(
            self.specification,
            ScheduleSpecification,
        ):
            raise TypeError("specification must be a ScheduleSpecification or None.")
        if self.gico_density_mass is not None:
            from genode.gico.clocks import guarded_density_mass, materialize
            from genode.gico.networks import DENSITY_BINS

            raw = self.gico_density_mass
            if not isinstance(raw, Tensor) or raw.shape != mass.shape or raw.shape[-1] != DENSITY_BINS:
                raise ValueError("GICO raw density must have shape [batch, 64].")
            if raw.device != mass.device or raw.dtype != torch.float64 or mass.dtype != torch.float64:
                raise ValueError("GICO schedule provenance requires float64 tensors on the same device.")
            expected_grid = grid.new_tensor([materialize(row, "euler", target) for row in raw.detach().cpu().numpy()])
            guarded = mass.new_tensor(np.stack([guarded_density_mass(row) for row in raw.detach().cpu().numpy()]))
            if not torch.allclose(mass, guarded, rtol=0, atol=_EXECUTABLE_BINDING_ATOL):
                raise ValueError("GICO density_mass must be its raw density with the uniform mixture applied once.")
            if not torch.equal(grid, expected_grid):
                raise ValueError("GICO time_grid must exactly match the shared density decoder.")
            # Preserve the common decoder's exact nodes for measurement replay.
            executable_grid = grid
        elif not torch.allclose(
            grid.to(dtype=torch.float64),
            executable_grid.to(dtype=torch.float64),
            rtol=0.0,
            atol=_EXECUTABLE_BINDING_ATOL,
        ):
            raise ValueError(
                "ScheduleBatch time_grid is not the executable quantile grid "
                "of its density_mass and reference_time_grid."
            )
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

    def gico_measurement_clock(self, index: int) -> dict:
        """Export unguarded masses and the exact common grid for paired evidence."""
        if self.gico_density_mass is None:
            raise ValueError("This schedule has no GICO raw-density provenance.")
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < self.batch_size:
            raise IndexError("GICO measurement index is outside the schedule batch.")
        return {
            "density_mass": self.gico_density_mass[index].detach().cpu().tolist(),
            "time_grid": self.time_grid[index].detach().cpu().tolist(),
            "solver": "euler",
            "nfe": self.target_nfe,
        }

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
                **(
                    {
                        "gico_density_mass_sha256": density_mass_hash(
                            self.gico_density_mass, reference_time_grid=self.reference_time_grid
                        )
                    }
                    if self.gico_density_mass is not None
                    else {}
                ),
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


@runtime_checkable
class IdentifiedSchedulePolicy(SchedulePolicy[ContextT], Protocol[ContextT]):
    """A schedule policy whose executable state has a content identity."""

    @property
    def policy_sha256(self) -> str: ...


__all__ = [
    "IdentifiedSchedulePolicy",
    "ScheduleBatch",
    "SchedulePolicy",
]
