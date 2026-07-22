from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch

from genode.checkpoint_validation import validate_tensor_state_dict
from genode.distillation.model import EndpointFlowMap, FlowMapSampler
from genode.distillation.training import _validate_continuous_array
from genode.gipo.models import (
    build_setting_encoder_config,
    setting_feature_dim,
)
from genode.gipo.policy import (
    TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET,
    TEACHER_METRIC_MASK_PROTOCOL,
    TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
    TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
    build_gipo_student_model,
    validate_gipo_teacher_training_metadata,
)
from genode.models.config import OTFlowConfig
from genode.models.otflow_model import OTFlow
from genode.solver_protocol import normalize_solver_nfe_fields


def _tiny_config() -> OTFlowConfig:
    return OTFlowConfig(
        device=torch.device("cpu"),
        levels=1,
        token_dim=4,
        history_len=2,
        hidden_dim=8,
        dropout=0.0,
        ctx_heads=4,
        ctx_layers=1,
        fu_net_layers=1,
        fu_net_heads=4,
        use_amp=False,
        use_minibatch_ot=False,
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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_nfe", True),
        ("target_nfe", 4.0),
        ("target_nfe", 4.5),
        ("macro_steps", False),
        ("macro_steps", 2.0),
        ("macro_steps", 2.5),
        ("realized_nfe", True),
        ("realized_nfe", 4.0),
        ("realized_nfe", 4.5),
    ),
)
def test_solver_nfe_fields_reject_non_integer_types(field: str, value: object) -> None:
    arguments = {field: value}
    if field != "target_nfe":
        arguments["target_nfe"] = 4

    with pytest.raises(ValueError, match=f"non-integer {field}"):
        normalize_solver_nfe_fields("heun", **arguments)


def test_solver_nfe_fields_accept_plain_integer_text_at_artifact_boundary() -> None:
    result = normalize_solver_nfe_fields(
        "heun",
        "4",  # type: ignore[arg-type]
        macro_steps="2",
        realized_nfe="4",
    )

    assert (result.target_nfe, result.macro_steps, result.realized_nfe) == (4, 2, 4)


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


def test_flow_map_and_gipo_loaders_reject_dtype_coercion() -> None:
    config = _setting_config()
    flow_map = EndpointFlowMap(
        _tiny_config(),
        setting_dim=setting_feature_dim(config=config),
        density_dim=8,
    )
    flow_state = OrderedDict(
        (name, tensor.detach().clone()) for name, tensor in flow_map.state_dict().items()
    )
    flow_key = next(iter(flow_state))
    flow_state[flow_key] = flow_state[flow_key].to(torch.float64)
    with pytest.raises(ValueError, match="has dtype"):
        flow_map.load_state_dict(flow_state)

    student = build_gipo_student_model(
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
    student_state = OrderedDict(
        (name, tensor.detach().clone()) for name, tensor in student.state_dict().items()
    )
    student_key = next(iter(student_state))
    student_state[student_key] = student_state[student_key].to(torch.float64)
    with pytest.raises(ValueError, match="has dtype"):
        student.load_state_dict(student_state)


def test_flow_map_sampler_rejects_nonfinite_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_config()
    config = _setting_config()
    flow_map = EndpointFlowMap(
        cfg,
        setting_dim=setting_feature_dim(config=config),
        density_dim=8,
    )
    sampler = FlowMapSampler(
        OTFlow(cfg).eval(),
        flow_map,
        setting_encoder_config=config,
    )
    monkeypatch.setattr(
        flow_map,
        "forward",
        lambda initial, *args, **kwargs: torch.full_like(initial, float("nan")),
    )

    with pytest.raises(ValueError, match="Flow-map output"):
        sampler.map_state(
            torch.zeros(1, cfg.sample_state_dim),
            hist=torch.zeros(1, cfg.history_len, cfg.context_dim),
            solver_key="euler",
            target_nfe=2,
            density_mass=torch.full((1, 8), 1.0 / 8.0),
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
        validate_gipo_teacher_training_metadata(metadata)


@pytest.mark.parametrize(
    "array",
    (
        np.zeros((2, 3), dtype=np.int64),
        np.zeros((2, 3), dtype=np.complex64),
    ),
)
def test_continuous_demonstration_arrays_require_real_floating_dtype(
    array: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="real floating-point dtype"):
        _validate_continuous_array(array, name="Trajectory states", rank=2)


def test_continuous_demonstration_arrays_require_expected_nonempty_rank() -> None:
    with pytest.raises(ValueError, match="rank 3"):
        _validate_continuous_array(
            np.zeros((2, 3), dtype=np.float32),
            name="Context shard ctx_tokens",
            rank=3,
        )
    with pytest.raises(ValueError, match="non-empty"):
        _validate_continuous_array(
            np.zeros((2, 0, 3), dtype=np.float32),
            name="Context shard ctx_tokens",
            rank=3,
        )
