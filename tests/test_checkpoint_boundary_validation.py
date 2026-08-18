from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from genode.checkpoint_validation import (
    validate_tensor_state_dict,
)
from genode.gico.models import (
    build_setting_encoder_config,
    setting_feature_dim,
)
from genode.gico.policy import (
    TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET,
    TEACHER_METRIC_MASK_PROTOCOL,
    TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
    TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
    build_gico_student_model,
    validate_gico_teacher_training_metadata,
)


def _setting_config():
    return build_setting_encoder_config(observed_target_nfes=(2, 4, 6, 8))


def _teacher_training_metadata() -> dict[str, object]:
    return {
        "teacher_target": "metric_vector",
        "teacher_metric_targets": ["u_comp_uniform"],
        "teacher_metric_target_protocol": TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
        "teacher_metric_mask_protocol": TEACHER_METRIC_MASK_PROTOCOL,
        "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
        "teacher_checkpoint_selection": {
            "selection_protocol": TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET,
            "locked_test_used_for_selection": False,
        },
    }


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


def test_gico_boundary_rejects_dtype_coercion() -> None:
    config = _setting_config()
    student = build_gico_student_model(
        setting_dim=setting_feature_dim(config=config),
        density_dim=8,
        context_dim=8,
        model_config={
            "hidden_dim": 8,
            "num_layers": 1,
            "attention_heads": 4,
            "dropout": 0.0,
        },
    )
    student_state = OrderedDict((name, tensor.detach().clone()) for name, tensor in student.state_dict().items())
    student_key = next(iter(student_state))
    student_state[student_key] = student_state[student_key].to(torch.float64)
    with pytest.raises(ValueError, match="has dtype"):
        validate_tensor_state_dict(
            student_state,
            label="GICO student state",
            target_module=student,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("teacher_metric_target_protocol", None),
        ("teacher_metric_target_protocol", "unsupported"),
        ("teacher_metric_mask_protocol", None),
        ("teacher_metric_mask_protocol", "unsupported"),
    ),
)
def test_teacher_training_metadata_requires_explicit_supported_protocols(
    field: str,
    value: object,
) -> None:
    metadata = _teacher_training_metadata()
    if value is None:
        metadata.pop(field)
    else:
        metadata[field] = value

    with pytest.raises(ValueError, match=field):
        validate_gico_teacher_training_metadata(metadata)


@pytest.mark.parametrize("value", (None, {}, {"selection_protocol": "unsupported"}))
def test_teacher_training_metadata_requires_explicit_locked_test_exclusion(value: object) -> None:
    metadata = _teacher_training_metadata()
    if value is None:
        metadata.pop("teacher_checkpoint_selection")
    else:
        metadata["teacher_checkpoint_selection"] = value

    with pytest.raises(ValueError, match="teacher_checkpoint_selection"):
        validate_gico_teacher_training_metadata(metadata)

    metadata = _teacher_training_metadata()
    metadata["teacher_checkpoint_selection"] = {
        "selection_protocol": TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET,
        "locked_test_used_for_selection": True,
    }
    with pytest.raises(ValueError, match="locked_test_used_for_selection=false"):
        validate_gico_teacher_training_metadata(metadata)
