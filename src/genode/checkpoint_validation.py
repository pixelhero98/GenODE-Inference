from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import torch


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


__all__ = [
    "validate_strict_integer",
    "validate_tensor_state_dict",
]
