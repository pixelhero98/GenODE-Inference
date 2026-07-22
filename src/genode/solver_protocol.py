from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import re
from typing import Sequence, Tuple

from torch import Tensor

SUPPORTED_SOLVER_KEYS: Tuple[str, ...] = ("euler", "dpmpp2m", "heun", "midpoint_rk2")
_INTEGER_TEXT = re.compile(r"[+-]?\d+")


def _strict_positive_int(
    value: object,
    *,
    field: str,
    source: str,
    optional: bool,
) -> int | None:
    if value is None:
        if optional:
            return None
        raise ValueError(f"{source} requires {field}.")
    if isinstance(value, str):
        text = value.strip()
        if not text and optional:
            return None
        if _INTEGER_TEXT.fullmatch(text) is None:
            raise ValueError(f"{source} has non-integer {field}={value!r}.")
        parsed = int(text)
    elif isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{source} has non-integer {field}={value!r}.")
    else:
        parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{source} requires positive {field}; got {parsed}.")
    return parsed


def normalize_solver_key(value: str) -> str:
    key = str(value)
    if key not in SUPPORTED_SOLVER_KEYS:
        raise ValueError(f"Unknown solver_key={value!r}; expected one of {SUPPORTED_SOLVER_KEYS}.")
    return key


def normalize_solver_keys(values: Sequence[str] | str, *, reject_duplicates: bool = True) -> Tuple[str, ...]:
    if isinstance(values, str):
        raw = [part.strip() for part in values.split(",") if part.strip()]
    else:
        raw = [str(part).strip() for part in values if str(part).strip()]
    if not raw:
        raise ValueError("At least one solver key is required.")
    normalized = tuple(normalize_solver_key(part) for part in raw)
    if reject_duplicates:
        duplicates = sorted({key for key in normalized if normalized.count(key) > 1})
        if duplicates:
            raise ValueError(f"Duplicate solver keys: {duplicates}.")
    return normalized


def solver_eval_multiplier(solver_key: str) -> int:
    key = normalize_solver_key(solver_key)
    return 2 if key in {"heun", "midpoint_rk2"} else 1


def solver_experiment_scope(solver_key: str) -> str:
    return "solver_transfer" if normalize_solver_key(solver_key) == "dpmpp2m" else "main"


def solver_macro_steps(solver_key: str, target_nfe: int) -> int:
    key = normalize_solver_key(solver_key)
    target = _strict_positive_int(
        target_nfe,
        field="target_nfe",
        source="solver protocol",
        optional=False,
    )
    if target is None:  # Narrow the optional helper's return type for type checkers.
        raise RuntimeError("target_nfe unexpectedly became unavailable.")
    if key in {"heun", "midpoint_rk2"}:
        if target % 2 != 0:
            raise ValueError(f"{key} requires an even target NFE, got {target}.")
        return target // 2
    return target


def target_nfe_for_macro_steps(solver_key: str, macro_steps: int) -> int:
    key = normalize_solver_key(solver_key)
    steps = _strict_positive_int(
        macro_steps,
        field="macro_steps",
        source="solver protocol",
        optional=False,
    )
    if steps is None:  # Narrow the optional helper's return type for type checkers.
        raise RuntimeError("macro_steps unexpectedly became unavailable.")
    return int(steps * solver_eval_multiplier(key))


def uniform_time_grid(solver_key: str, target_nfe: int) -> Tuple[float, ...]:
    macro_steps = solver_macro_steps(solver_key, target_nfe)
    return tuple(float(index) / float(macro_steps) for index in range(macro_steps + 1))


def solver_effective_order(solver_key: str) -> int:
    return 1 if normalize_solver_key(solver_key) == "euler" else 2


def solver_order_p(solver_key: str) -> float:
    return float(solver_effective_order(solver_key))


def expected_realized_nfe(solver_key: str, target_nfe: int) -> int:
    key = normalize_solver_key(solver_key)
    return int(solver_macro_steps(key, target_nfe) * solver_eval_multiplier(key))


@dataclass(frozen=True)
class SolverNFEFields:
    solver_key: str
    target_nfe: int
    macro_steps: int
    realized_nfe: int


@dataclass(frozen=True)
class FlowTrajectory:
    """A batch-major trajectory produced by a deterministic ODE solve.

    ``states`` includes the initial state and every accepted macro-step state,
    so its second dimension is ``macro_steps + 1``.
    """

    initial_state: Tensor
    time_grid: Tensor
    states: Tensor
    final_state: Tensor
    solver_key: str
    target_nfe: int
    macro_steps: int
    realized_nfe: int


@dataclass(frozen=True)
class FlowDiagnostics:
    """Per-step diagnostics for a fixed-schedule flow trajectory."""

    trajectory: FlowTrajectory
    disagreement: Tensor
    velocity_norm: Tensor
    ema_velocity_norm: Tensor
    residual_norm: Tensor
    local_error: Tensor
    field_evals_by_step: Tensor
    mean_field_evals_per_step: float
    mean_total_field_evals_per_rollout: float


def normalize_solver_nfe_fields(
    solver_key: str,
    target_nfe: int,
    *,
    macro_steps: object = None,
    realized_nfe: object = None,
    source: str = "row",
) -> SolverNFEFields:
    """Return normalized macro-step and realized-NFE fields for one solver target."""

    key = normalize_solver_key(str(solver_key))
    target = _strict_positive_int(
        target_nfe,
        field="target_nfe",
        source=source,
        optional=False,
    )
    if target is None:  # Narrow the optional helper's return type for type checkers.
        raise RuntimeError("target_nfe unexpectedly became unavailable.")
    expected_macro = solver_macro_steps(key, target)
    expected_realized = expected_realized_nfe(key, target)
    parsed_macro = _strict_positive_int(
        macro_steps,
        field="macro_steps",
        source=source,
        optional=True,
    )
    parsed_realized = _strict_positive_int(
        realized_nfe,
        field="realized_nfe",
        source=source,
        optional=True,
    )
    if parsed_macro is not None and parsed_macro != expected_macro:
        raise ValueError(
            f"{source} has macro_steps={parsed_macro} for {key}/{target}; expected {expected_macro}."
        )
    if parsed_realized is not None and parsed_realized != expected_realized:
        raise ValueError(
            f"{source} has realized_nfe={parsed_realized} for {key}/{target}; expected {expected_realized}."
        )
    return SolverNFEFields(
        solver_key=key,
        target_nfe=target,
        macro_steps=int(expected_macro),
        realized_nfe=int(expected_realized),
    )


__all__ = [
    "SUPPORTED_SOLVER_KEYS",
    "FlowTrajectory",
    "FlowDiagnostics",
    "SolverNFEFields",
    "expected_realized_nfe",
    "normalize_solver_key",
    "normalize_solver_keys",
    "normalize_solver_nfe_fields",
    "solver_effective_order",
    "solver_eval_multiplier",
    "solver_experiment_scope",
    "solver_macro_steps",
    "solver_order_p",
    "target_nfe_for_macro_steps",
    "uniform_time_grid",
]
