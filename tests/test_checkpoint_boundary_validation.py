from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from genode.checkpoint_validation import (
    validate_tensor_state_dict,
)


def test_tensor_state_validation_matches_target_exactly() -> None:
    module = torch.nn.Linear(3, 2)
    state = OrderedDict((name, tensor.detach().clone()) for name, tensor in module.state_dict().items())

    validated = validate_tensor_state_dict(
        state,
        label="test state",
        target_module=module,
    )

    assert tuple(validated) == tuple(module.state_dict())

    missing = OrderedDict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="keys do not match"):
        validate_tensor_state_dict(missing, label="test state", target_module=module)

    wrong_shape = OrderedDict(state)
    first = next(iter(wrong_shape))
    wrong_shape[first] = wrong_shape[first].reshape(-1)
    with pytest.raises(ValueError, match="has shape"):
        validate_tensor_state_dict(wrong_shape, label="test state", target_module=module)

    wrong_dtype = OrderedDict(state)
    wrong_dtype[first] = wrong_dtype[first].to(torch.float64)
    with pytest.raises(ValueError, match="has dtype"):
        validate_tensor_state_dict(wrong_dtype, label="test state", target_module=module)


@pytest.mark.parametrize("dtype", (torch.int64, torch.complex64))
def test_tensor_state_validation_rejects_non_real_floating_tensors(dtype: torch.dtype) -> None:
    module = torch.nn.Linear(3, 2)
    state = OrderedDict((name, tensor.detach().clone()) for name, tensor in module.state_dict().items())
    first = next(iter(state))
    state[first] = state[first].to(dtype)

    with pytest.raises(ValueError, match="real floating-point dtype"):
        validate_tensor_state_dict(state, label="test state", target_module=module)
