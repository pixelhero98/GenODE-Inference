from __future__ import annotations

import re
from numbers import Integral
from typing import Any, Mapping, Sequence

import torch

from genode.solver_protocol import (
    SolverNFEFields,
)
from genode.solver_protocol import (
    normalize_solver_nfe_fields as _normalize_snapshot_solver_nfe_fields,
)

_INTEGER_TEXT = re.compile(r"[+-]?\d+")


def validate_strict_integer(
    value: Any,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Validate an integer without truncating floats or accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer, got {value!r}.")
    integer = int(value)
    if minimum is not None and integer < int(minimum):
        raise ValueError(f"{label} must be at least {int(minimum)}, got {integer}.")
    if maximum is not None and integer > int(maximum):
        raise ValueError(f"{label} must be at most {int(maximum)}, got {integer}.")
    return integer


def _strict_positive_solver_integer(
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


def normalize_strict_solver_nfe_fields(
    solver_key: str,
    target_nfe: object,
    *,
    macro_steps: object = None,
    runtime_nfe: object = None,
    realized_nfe: object = None,
    source: str = "row",
) -> SolverNFEFields:
    """Apply strict integer validation before the snapshot solver normalizer."""

    target = _strict_positive_solver_integer(
        target_nfe,
        field="target_nfe",
        source=source,
        optional=False,
    )
    if target is None:  # Narrow the helper's return type for type checkers.
        raise RuntimeError("target_nfe unexpectedly became unavailable.")
    parsed_macro = _strict_positive_solver_integer(
        macro_steps,
        field="macro_steps",
        source=source,
        optional=True,
    )
    parsed_runtime = _strict_positive_solver_integer(
        runtime_nfe,
        field="runtime_nfe",
        source=source,
        optional=True,
    )
    parsed_realized = _strict_positive_solver_integer(
        realized_nfe,
        field="realized_nfe",
        source=source,
        optional=True,
    )
    return _normalize_snapshot_solver_nfe_fields(
        solver_key,
        target,
        macro_steps=parsed_macro,
        runtime_nfe=parsed_runtime,
        realized_nfe=parsed_realized,
        source=source,
    )


def validate_tensor_state_dict(
    state: Mapping[str, Any],
    *,
    label: str,
    target_module: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Validate checkpoint tensors before they are loaded into a module.

    When ``target_module`` is supplied, the checkpoint must exactly match the
    target state names, shapes, and dtypes.  This check is intentionally done
    before :meth:`torch.nn.Module.load_state_dict`, which otherwise copies and
    may silently cast compatible-looking tensors.
    """

    if not isinstance(state, Mapping):
        raise ValueError(f"{label} must be a tensor mapping.")

    for raw_name in state:
        if not isinstance(raw_name, str) or not raw_name or raw_name != raw_name.strip():
            raise ValueError(f"{label} contains an invalid parameter name {raw_name!r}.")

    expected_state = None if target_module is None else target_module.state_dict()
    if expected_state is not None:
        missing = sorted(set(expected_state) - set(state))
        unexpected = sorted(set(state) - set(expected_state))
        if missing or unexpected:
            raise ValueError(
                f"{label} keys do not match the target module; missing={missing}, unexpected={unexpected}."
            )

    validated: dict[str, torch.Tensor] = {}
    for raw_name, value in state.items():
        name = raw_name
        if not torch.is_tensor(value):
            raise ValueError(f"{label} contains a non-tensor value at {name!r}.")
        if not value.is_floating_point():
            raise ValueError(f"{label} tensor {name!r} must use a real floating-point dtype, got {value.dtype}.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} contains non-finite tensor values at {name!r}.")
        if expected_state is not None:
            expected = expected_state[name]
            if tuple(value.shape) != tuple(expected.shape):
                raise ValueError(
                    f"{label} tensor {name!r} has shape {tuple(value.shape)}; expected {tuple(expected.shape)}."
                )
            if value.dtype != expected.dtype:
                raise ValueError(f"{label} tensor {name!r} has dtype {value.dtype}; expected {expected.dtype}.")
        validated[name] = value
    if not validated:
        raise ValueError(f"{label} may not be empty.")
    return validated


def validate_locked_test_exclusion(
    payload: Mapping[str, Any],
    *,
    label: str,
    required_root_keys: Sequence[str] = (),
) -> None:
    """Require every locked-test provenance flag to be the literal ``False``.

    The recursive check prevents a top-level clean flag from masking nested
    selection or distillation metadata that records locked-test use.
    """

    for key in required_root_keys:
        if key not in payload or payload[key] is not False:
            raise ValueError(f"{label} requires {key}=false.")

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                if key.startswith("locked_test_used") and child is not False:
                    raise ValueError(f"{label} requires {child_path}=false, got {child!r}.")
                walk(child, child_path)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload, "")


__all__ = [
    "normalize_strict_solver_nfe_fields",
    "validate_locked_test_exclusion",
    "validate_strict_integer",
    "validate_tensor_state_dict",
]
