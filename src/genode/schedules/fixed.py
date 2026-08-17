from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral

import torch
from torch import Tensor

from genode.artifacts.identity import semantic_sha256
from genode.schedule_transfer.reference_clocks import (
    build_reference_clock_grid,
    late_p_value_from_key,
    reference_clock_keys,
    reference_clock_provenance,
)
from genode.schedules.density import (
    density_mass_hash,
    time_grid_hash,
    time_grid_to_density_mass,
    uniform_reference_time_grid,
    validate_density_mass,
    validate_reference_time_grid,
)
from genode.schedules.progress import validate_time_grid
from genode.schedules.specification import ScheduleSpecification

FIXED_SCHEDULE_PROTOCOL = "image_reference_clock_schedule_v2"
FIXED_SCHEDULE_TARGET_NFES = (2, 4, 8)
_EXECUTABLE_BINDING_ATOL = 1e-12


def _fixed_target_nfe(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            "Image reference-clock target_nfe must be one of the integer "
            f"values {FIXED_SCHEDULE_TARGET_NFES}, got {value!r}."
        )
    parsed = int(value)
    if parsed not in FIXED_SCHEDULE_TARGET_NFES:
        raise ValueError(f"Image reference-clock target_nfe must be one of {FIXED_SCHEDULE_TARGET_NFES}, got {parsed}.")
    return parsed


def _normalize_fixed_specification(
    specification: ScheduleSpecification,
) -> ScheduleSpecification:
    if not isinstance(specification, ScheduleSpecification):
        raise TypeError("specification must be a ScheduleSpecification.")
    if specification.schedule_parameters:
        raise ValueError(
            "Reference-clock parameters are encoded in the canonical schedule "
            f"key; {specification.schedule_key!r} may not carry parameters."
        )
    try:
        reference_clock_provenance(specification.schedule_key)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported image reference clock {specification.schedule_key!r}.") from exc
    return specification


def _extra_late_p_values_from_keys(keys: Sequence[str]) -> tuple[Decimal, ...]:
    values: set[Decimal] = set()
    for key in keys:
        base_key = str(key).removesuffix("_reversed")
        if base_key.startswith("late_p_"):
            values.add(late_p_value_from_key(base_key))
    return tuple(sorted(values))


def validate_fixed_schedule_keys(
    schedule_keys: Sequence[str],
) -> tuple[str, ...]:
    """Require one complete canonical clock pool in canonical order."""

    keys = tuple(str(value) for value in schedule_keys)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Image reference-clock keys must be non-empty and unique.")
    for key in keys:
        try:
            reference_clock_provenance(key)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Unsupported image reference clock {key!r}.") from exc
    expected = reference_clock_keys(_extra_late_p_values_from_keys(keys))
    if keys != expected:
        raise ValueError(
            "Image reference-clock keys must contain the complete canonical pool "
            "in canonical order, including every non-uniform reversal; "
            f"expected {expected}, got {keys}."
        )
    return keys


def default_fixed_schedule_specifications(
    *,
    extra_late_p_values: str | Sequence[Decimal | float | int | str] = (),
) -> tuple[ScheduleSpecification, ...]:
    """Return the canonical 23-clock image support plus requested late-p pairs."""

    return tuple(ScheduleSpecification(key) for key in reference_clock_keys(extra_late_p_values))


@dataclass(frozen=True)
class FixedSchedule:
    specification: ScheduleSpecification
    target_nfe: int
    time_grid: Tensor
    reference_time_grid: Tensor
    density_mass: Tensor

    def __post_init__(self) -> None:
        specification = _normalize_fixed_specification(self.specification)
        target_nfe = _fixed_target_nfe(self.target_nfe)
        grid = validate_time_grid(self.time_grid, target_nfe=target_nfe)
        reference = validate_reference_time_grid(self.reference_time_grid)
        mass = validate_density_mass(
            self.density_mass,
            reference_time_grid=reference,
        )
        if grid.ndim != 1 or reference.ndim != 1 or mass.ndim != 1:
            raise ValueError("FixedSchedule requires one time grid, reference grid, and density vector.")
        if not (grid.device == reference.device == mass.device and grid.dtype == reference.dtype == mass.dtype):
            raise ValueError("FixedSchedule tensors must share device and dtype.")
        canonical_mass = time_grid_to_density_mass(
            grid,
            reference_time_grid=reference,
        )
        if not torch.allclose(
            mass.to(dtype=torch.float64),
            canonical_mass.to(dtype=torch.float64),
            rtol=0.0,
            atol=_EXECUTABLE_BINDING_ATOL,
        ):
            raise ValueError("FixedSchedule density_mass does not correspond to its time_grid and reference_time_grid.")
        object.__setattr__(self, "specification", specification)
        object.__setattr__(self, "target_nfe", target_nfe)
        object.__setattr__(self, "density_mass", canonical_mass)

    @property
    def calibration_sha256(self) -> None:
        return None

    @property
    def clock_provenance(self) -> dict[str, object]:
        return reference_clock_provenance(self.specification.schedule_key)

    @property
    def time_grid_sha256(self) -> str:
        return time_grid_hash(self.time_grid)

    @property
    def density_mass_sha256(self) -> str:
        return density_mass_hash(
            self.density_mass,
            reference_time_grid=self.reference_time_grid,
        )

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            {
                "protocol": FIXED_SCHEDULE_PROTOCOL,
                "specification_sha256": self.specification.sha256,
                "reference_clock_provenance": self.clock_provenance,
                "target_nfe": self.target_nfe,
                "time_grid_sha256": self.time_grid_sha256,
                "density_mass_sha256": self.density_mass_sha256,
                "calibration_sha256": None,
            },
            namespace="fixed-schedule",
        )


def build_fixed_schedule(
    specification: ScheduleSpecification,
    target_nfe: int,
    *,
    density_bin_count: int = 64,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> FixedSchedule:
    """Build one canonical reference clock as an executable image schedule."""

    normalized = _normalize_fixed_specification(specification)
    target = _fixed_target_nfe(target_nfe)
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype.")
    grid = torch.tensor(
        build_reference_clock_grid(normalized.schedule_key, target),
        dtype=dtype,
        device=device,
    )
    grid = validate_time_grid(grid, target_nfe=target)
    reference = uniform_reference_time_grid(
        density_bin_count,
        dtype=dtype,
        device=device,
    )
    density_mass = time_grid_to_density_mass(
        grid,
        reference_time_grid=reference,
    )
    return FixedSchedule(
        specification=normalized,
        target_nfe=target,
        time_grid=grid,
        reference_time_grid=reference,
        density_mass=density_mass,
    )


def build_default_fixed_schedules(
    target_nfe: int,
    *,
    extra_late_p_values: str | Sequence[Decimal | float | int | str] = (),
    density_bin_count: int = 64,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[FixedSchedule, ...]:
    """Build the canonical image clock pool in canonical order."""

    return tuple(
        build_fixed_schedule(
            specification,
            target_nfe,
            density_bin_count=density_bin_count,
            dtype=dtype,
            device=device,
        )
        for specification in default_fixed_schedule_specifications(
            extra_late_p_values=extra_late_p_values,
        )
    )


@dataclass(frozen=True)
class FixedScheduleGridGroup:
    time_grid_sha256: str
    schedules: tuple[FixedSchedule, ...]

    @property
    def schedule_keys(self) -> tuple[str, ...]:
        return tuple(schedule.specification.schedule_key for schedule in self.schedules)


def group_fixed_schedules_by_time_grid(
    schedules: Sequence[FixedSchedule],
) -> tuple[FixedScheduleGridGroup, ...]:
    """Group exact duplicate grids without discarding schedule identities."""

    grouped: dict[str, list[FixedSchedule]] = {}
    for schedule in schedules:
        if not isinstance(schedule, FixedSchedule):
            raise TypeError("schedules must contain only FixedSchedule values.")
        grouped.setdefault(schedule.time_grid_sha256, []).append(schedule)
    return tuple(
        FixedScheduleGridGroup(
            time_grid_sha256=grid_hash,
            schedules=tuple(group),
        )
        for grid_hash, group in grouped.items()
    )


__all__ = [
    "FIXED_SCHEDULE_PROTOCOL",
    "FIXED_SCHEDULE_TARGET_NFES",
    "FixedSchedule",
    "FixedScheduleGridGroup",
    "build_default_fixed_schedules",
    "build_fixed_schedule",
    "default_fixed_schedule_specifications",
    "group_fixed_schedules_by_time_grid",
    "validate_fixed_schedule_keys",
]
