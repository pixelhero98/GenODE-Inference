from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any

from genode.artifacts.identity import canonical_json_text, semantic_sha256
from genode.schedules.progress import PROGRESS_PROTOCOL


SCHEDULE_SPECIFICATION_PROTOCOL = "schedule_specification_v2"
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")


def _canonical_identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be canonical lower_snake_case, got {value!r}.")
    return value


def _canonical_parameters(
    parameters: Mapping[str, Real] | None,
) -> Mapping[str, float]:
    if parameters is None:
        return MappingProxyType({})
    if not isinstance(parameters, Mapping):
        raise TypeError("schedule_parameters must be a mapping or None.")
    normalized: dict[str, float] = {}
    for name, raw_value in parameters.items():
        key = _canonical_identifier(name, field_name="schedule parameter name")
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise TypeError(f"Schedule parameter {key!r} must be a finite real number, got {raw_value!r}.")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"Schedule parameter {key!r} must be finite, got {raw_value!r}.")
        normalized[key] = value
    return MappingProxyType(dict(sorted(normalized.items())))


def normalize_schedule_parameters(
    schedule_key: str,
    schedule_parameters: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return canonical parameters for a registered fixed schedule family."""

    key = str(schedule_key).strip().lower()
    if not key:
        raise ValueError("schedule_key must not be empty.")
    if schedule_parameters is None:
        raw_parameters: dict[str, Any] = {}
    elif isinstance(schedule_parameters, Mapping):
        raw_parameters = {str(name): value for name, value in schedule_parameters.items()}
    else:
        raise TypeError("schedule_parameters must be a mapping or None.")

    if raw_parameters:
        raise ValueError(
            "Reference-clock parameters are encoded in the canonical schedule "
            f"key; {key!r} does not accept parameters {sorted(raw_parameters)}."
        )
    return {}


def schedule_parameters_json(
    schedule_key: str,
    schedule_parameters: Mapping[str, Any] | None = None,
) -> str:
    """Serialize normalized fixed-schedule parameters deterministically."""

    return canonical_json_text(normalize_schedule_parameters(schedule_key, schedule_parameters))


def parse_schedule_parameters(
    schedule_key: str,
    value: Any,
) -> dict[str, float]:
    """Parse a mapping or JSON artifact field into canonical parameters."""

    key = str(schedule_key).strip().lower()
    missing = value is None or (isinstance(value, str) and not value.strip())
    if missing:
        return normalize_schedule_parameters(key)
    if isinstance(value, Mapping):
        payload: Any = value
    elif isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Schedule {key!r} has invalid schedule_parameters_json.") from exc
    else:
        raise TypeError("Schedule parameters must be a JSON object, mapping, or empty.")
    if not isinstance(payload, Mapping):
        raise ValueError("schedule_parameters_json must encode a JSON object.")
    return normalize_schedule_parameters(key, payload)


@dataclass(frozen=True)
class ScheduleSpecification:
    """Portable identity for one named schedule family and its parameters."""

    schedule_key: str
    schedule_parameters: Mapping[str, Real] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schedule_key",
            _canonical_identifier(self.schedule_key, field_name="schedule_key"),
        )
        object.__setattr__(
            self,
            "schedule_parameters",
            _canonical_parameters(self.schedule_parameters),
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.schedule_key,
                tuple(self.schedule_parameters.items()),
            )
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "protocol": SCHEDULE_SPECIFICATION_PROTOCOL,
            "progress_protocol": PROGRESS_PROTOCOL,
            "schedule_key": self.schedule_key,
            "schedule_parameters": dict(self.schedule_parameters),
        }

    def parameters_json(self) -> str:
        return canonical_json_text(dict(self.schedule_parameters))

    @property
    def sha256(self) -> str:
        return schedule_hash(self)


def schedule_hash(specification: ScheduleSpecification) -> str:
    """Hash a schedule identity under the canonical progress convention."""

    if not isinstance(specification, ScheduleSpecification):
        raise TypeError("specification must be a ScheduleSpecification.")
    return semantic_sha256(
        specification.as_payload(),
        namespace="schedule-specification",
    )


__all__ = [
    "SCHEDULE_SPECIFICATION_PROTOCOL",
    "ScheduleSpecification",
    "normalize_schedule_parameters",
    "parse_schedule_parameters",
    "schedule_hash",
    "schedule_parameters_json",
]
