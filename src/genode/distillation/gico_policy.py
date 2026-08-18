from __future__ import annotations

import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

from genode.checkpoint_validation import (
    normalize_strict_solver_nfe_fields as normalize_solver_nfe_fields,
)
from genode.checkpoint_validation import (
    validate_locked_test_exclusion,
    validate_strict_integer,
    validate_tensor_state_dict,
)
from genode.distillation.validation import setting_encoder_config_from_payload
from genode.gico.density_representation import (
    DENSITY_PROTOCOL,
    validate_reference_grid,
)
from genode.gico.models import (
    SettingEncoderConfig,
    setting_feature_dim,
    setting_features,
)
from genode.gico.policy import (
    ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
    GICO_PROTOCOL,
    EmbeddingNormalizer,
    require_current_gico_checkpoint_payload,
)
from genode.gico.policy import (
    build_gico_student_model as _build_current_gico_student_model,
)
from genode.gico.policy import (
    validate_gico_teacher_training_metadata as _validate_current_teacher_training_metadata,
)
from genode.models.conditioning import (
    FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL,
    ConditioningCache,
    frozen_backbone_policy_context_from_cache,
)

_CURRENT_STUDENT_MODEL_CONFIG_FIELDS = frozenset(
    {
        "architecture",
        "setting_dim",
        "density_dim",
        "context_dim",
        "hidden_dim",
        "num_layers",
        "attention_heads",
        "dropout",
        "conditioning_style",
        "density_token_attention",
        "density_feature_mean",
        "density_feature_std",
    }
)


def validate_gico_teacher_training_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require explicit metric protocols before current-schema validation."""

    teacher_training = dict(metadata or {})
    required_protocols = {
        "teacher_metric_target_protocol": "family_metric_utility_vector",
        "teacher_metric_mask_protocol": "row_valid_component_mask",
    }
    for field, expected in required_protocols.items():
        if teacher_training.get(field) != expected:
            raise ValueError(f"GICO checkpoint {field} must be explicitly set to {expected!r}.")
    selection = teacher_training.get("teacher_checkpoint_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("GICO teacher_checkpoint_selection must be an object.")
    if selection.get("locked_test_used_for_selection") is not False:
        raise ValueError("GICO teacher_checkpoint_selection requires locked_test_used_for_selection=false.")
    return dict(_validate_current_teacher_training_metadata(teacher_training))


def build_gico_student_model(
    *,
    architecture: str = ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
    setting_dim: int,
    density_dim: int,
    context_dim: int,
    model_config: Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    """Build the current context-only GICO student architecture."""

    return _build_current_gico_student_model(
        architecture=architecture,
        setting_dim=setting_dim,
        density_dim=density_dim,
        context_dim=context_dim,
        model_config=dict(model_config or {}),
    )


def _validate_student_model_config(
    model_config: Mapping[str, Any],
    *,
    setting_dim: int,
    density_dim: int,
    context_dim: int,
) -> dict[str, Any]:
    """Validate the complete normalized config written by a GICO student."""

    config = dict(model_config)
    if set(config) != _CURRENT_STUDENT_MODEL_CONFIG_FIELDS:
        missing = sorted(_CURRENT_STUDENT_MODEL_CONFIG_FIELDS - set(config))
        unknown = sorted(set(config) - _CURRENT_STUDENT_MODEL_CONFIG_FIELDS)
        raise ValueError(f"GICO student model configuration must be complete; missing={missing}, unknown={unknown}.")
    expected_dimensions = {
        "setting_dim": int(setting_dim),
        "density_dim": int(density_dim),
        "context_dim": int(context_dim),
    }
    for field, expected in expected_dimensions.items():
        actual = validate_strict_integer(
            config[field],
            label=f"GICO student model_config {field}",
            minimum=1,
        )
        if actual != expected:
            raise ValueError(
                f"GICO student model_config {field}={actual} does not match the checkpoint value {expected}."
            )
        config[field] = actual
    for field in ("hidden_dim", "num_layers", "attention_heads"):
        config[field] = validate_strict_integer(
            config[field],
            label=f"GICO student model_config {field}",
            minimum=1,
        )
    dropout = config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, Real):
        raise ValueError("GICO student model_config dropout must be numeric.")
    dropout = float(dropout)
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("GICO student model_config dropout must be finite and in [0, 1).")
    config["dropout"] = dropout
    if config["architecture"] != ARCHITECTURE_DENSITY_QUERY_TRANSFORMER:
        raise ValueError("GICO student model_config architecture does not match the student architecture.")
    if config["density_feature_mean"] is not None or config["density_feature_std"] is not None:
        raise ValueError("GICO student model_config may not contain teacher density-feature statistics.")
    return config


@dataclass(frozen=True)
class GICOSchedule:
    solver_key: str
    target_nfe: int
    macro_steps: int
    realized_nfe: int
    density_mass: torch.Tensor
    time_grid: torch.Tensor


class GICOSchedulePolicy:
    """Validated GICO student used to produce context-dependent ODE schedules."""

    def __init__(
        self,
        student: torch.nn.Module,
        *,
        embedding_normalizer: EmbeddingNormalizer,
        reference_time_grid: tuple[float, ...],
        setting_encoder_config: SettingEncoderConfig,
        checkpoint_payload: Mapping[str, Any],
    ):
        self.student = student
        self.embedding_normalizer = embedding_normalizer
        self.reference_time_grid = reference_time_grid
        self.setting_encoder_config = setting_encoder_config
        metadata = dict(checkpoint_payload)
        metadata.pop("student_state", None)
        self.checkpoint_payload = metadata
        if "context_embedding_kind" in metadata:
            raise ValueError("GICO checkpoints may not contain the retired context_embedding_kind field.")
        context_protocol = str(metadata.get("context_embedding_protocol", ""))
        if context_protocol != FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL:
            raise ValueError(
                f"GICO checkpoint context_embedding_protocol must be {FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL!r}."
            )
        self.context_embedding_protocol = context_protocol
        try:
            parameter = next(student.parameters())
        except StopIteration as exc:
            raise ValueError("GICO student must contain model parameters.") from exc
        if not parameter.is_floating_point():
            raise ValueError("GICO student parameters must use a floating-point dtype.")
        self._student_device = parameter.device
        self._student_dtype = parameter.dtype
        self._normalizer_mean = torch.as_tensor(
            embedding_normalizer.mean,
            device=self._student_device,
            dtype=self._student_dtype,
        )
        self._normalizer_std = torch.as_tensor(
            embedding_normalizer.std,
            device=self._student_device,
            dtype=self._student_dtype,
        )
        self._reference_edges = torch.as_tensor(
            reference_time_grid,
            device=self._student_device,
            dtype=self._student_dtype,
        )
        self._setting_feature_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._quantile_cache: dict[int, torch.Tensor] = {}
        self.student.eval()

    @property
    def density_dim(self) -> int:
        return int(len(self.reference_time_grid) - 1)

    @property
    def setting_dim(self) -> int:
        return setting_feature_dim(
            self.setting_encoder_config.mode,
            config=self.setting_encoder_config,
        )

    def context_embedding_from_cache(
        self,
        conditioning_cache: ConditioningCache,
    ) -> torch.Tensor:
        """Derive the canonical frozen-backbone policy representation."""

        return frozen_backbone_policy_context_from_cache(
            conditioning_cache,
            protocol=self.context_embedding_protocol,
            expected_width=int(self._normalizer_mean.numel()),
        )

    def _time_grid(self, density_mass: torch.Tensor, *, macro_steps: int) -> torch.Tensor:
        """Invert a batch of piecewise-linear density CDFs without host transfers."""

        mass = density_mass.clamp_min(1e-12)
        mass = mass / mass.sum(dim=-1, keepdim=True)
        cdf = torch.cat(
            (
                torch.zeros(
                    int(mass.shape[0]),
                    1,
                    device=mass.device,
                    dtype=mass.dtype,
                ),
                mass.cumsum(dim=-1),
            ),
            dim=-1,
        )
        cdf[:, -1] = 1.0
        step_count = int(macro_steps)
        if step_count not in self._quantile_cache:
            self._quantile_cache[step_count] = torch.linspace(
                0.0,
                1.0,
                step_count + 1,
                device=mass.device,
                dtype=mass.dtype,
            )
        quantiles = self._quantile_cache[step_count].expand(int(mass.shape[0]), -1)
        bin_index = torch.searchsorted(cdf.contiguous(), quantiles.contiguous(), right=True) - 1
        bin_index = bin_index.clamp(min=0, max=self.density_dim - 1)
        cdf_left = torch.gather(cdf, 1, bin_index)
        cdf_right = torch.gather(cdf, 1, bin_index + 1)
        edges = self._reference_edges.expand(int(mass.shape[0]), -1)
        edge_left = torch.gather(edges, 1, bin_index)
        edge_right = torch.gather(edges, 1, bin_index + 1)
        fraction = (quantiles - cdf_left) / (cdf_right - cdf_left).clamp_min(1e-12)
        grid = edge_left + fraction * (edge_right - edge_left)
        grid[:, 0] = 0.0
        grid[:, -1] = 1.0
        return grid

    @torch.no_grad()
    def predict(
        self,
        context_embedding: torch.Tensor,
        *,
        solver_key: str,
        target_nfe: int,
    ) -> GICOSchedule:
        if context_embedding.ndim != 2:
            raise ValueError(
                f"context_embedding must have shape [batch, context_dim], got {tuple(context_embedding.shape)}."
            )
        if not context_embedding.is_floating_point():
            raise ValueError("context_embedding must use a floating-point dtype.")
        if not torch.isfinite(context_embedding).all():
            raise ValueError("context_embedding contains non-finite values.")
        nfe = normalize_solver_nfe_fields(solver_key, target_nfe, source="GICO schedule prediction")
        output_device = context_embedding.device
        output_dtype = context_embedding.dtype
        student_context = context_embedding.to(
            device=self._student_device,
            dtype=self._student_dtype,
        )
        mean = self._normalizer_mean
        std = self._normalizer_std
        if mean.shape != student_context.shape[1:] or std.shape != student_context.shape[1:]:
            raise ValueError(
                "GICO context normalizer is incompatible with the backbone context summary: "
                f"normalizer={tuple(mean.shape)}, context={tuple(student_context.shape[1:])}."
            )
        normalized_context = (student_context - mean) / std.clamp_min(1e-6)
        if not torch.isfinite(normalized_context).all():
            raise ValueError("GICO context normalization produced non-finite values.")
        setting_key = (nfe.solver_key, nfe.target_nfe)
        if setting_key not in self._setting_feature_cache:
            self._setting_feature_cache[setting_key] = setting_features(
                nfe.solver_key,
                nfe.target_nfe,
                mode=self.setting_encoder_config.mode,
                config=self.setting_encoder_config,
            ).to(device=self._student_device, dtype=self._student_dtype)
        features = self._setting_feature_cache[setting_key]
        feature_batch = features.unsqueeze(0).expand(int(student_context.shape[0]), -1)
        student_density = self.student.density_mass(
            feature_batch,
            normalized_context,
        )
        if student_density.shape != (int(student_context.shape[0]), self.density_dim):
            raise ValueError(f"GICO student returned an invalid density shape: {tuple(student_density.shape)}.")
        if (
            not student_density.is_floating_point()
            or not torch.isfinite(student_density).all()
            or bool((student_density < 0.0).any())
            or not torch.allclose(
                student_density.sum(dim=-1),
                torch.ones(
                    int(student_density.shape[0]),
                    device=student_density.device,
                    dtype=student_density.dtype,
                ),
                atol=1e-6,
                rtol=1e-5,
            )
        ):
            raise ValueError("GICO student density must be finite, nonnegative, and sum to one.")
        student_time_grid = self._time_grid(
            student_density,
            macro_steps=nfe.macro_steps,
        )
        density_mass = student_density.to(device=output_device, dtype=output_dtype)
        time_grid = student_time_grid.to(device=output_device, dtype=output_dtype)
        density_tolerance = max(
            1e-6,
            4.0 * float(torch.finfo(density_mass.dtype).eps),
        )
        if (
            not torch.isfinite(density_mass).all()
            or bool((density_mass < 0.0).any())
            or not torch.allclose(
                density_mass.sum(dim=-1),
                torch.ones(
                    int(density_mass.shape[0]),
                    device=density_mass.device,
                    dtype=density_mass.dtype,
                ),
                atol=density_tolerance,
                rtol=density_tolerance,
            )
        ):
            raise ValueError("GICO density is invalid in the requested output dtype.")
        expected_grid_shape = (int(student_context.shape[0]), nfe.macro_steps + 1)
        if (
            time_grid.shape != expected_grid_shape
            or not torch.isfinite(time_grid).all()
            or not bool((torch.diff(time_grid, dim=-1) > 0.0).all())
            or not bool((time_grid[:, 0] == 0.0).all())
            or not bool((time_grid[:, -1] == 1.0).all())
        ):
            raise ValueError("GICO time grid must be finite, strictly increasing, and span [0, 1].")
        return GICOSchedule(
            solver_key=nfe.solver_key,
            target_nfe=nfe.target_nfe,
            macro_steps=nfe.macro_steps,
            realized_nfe=nfe.realized_nfe,
            density_mass=density_mass,
            time_grid=time_grid,
        )


def load_gico_schedule_policy(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> GICOSchedulePolicy:
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"Could not load GICO checkpoint {checkpoint_path.name!r}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("GICO student checkpoint must contain a mapping payload.")
    payload = require_current_gico_checkpoint_payload(
        payload,
        label="GICO checkpoint",
    )
    if str(payload.get("protocol", "")) != GICO_PROTOCOL:
        raise ValueError(f"Unsupported GICO protocol {payload.get('protocol')!r}; expected {GICO_PROTOCOL!r}.")
    if str(payload.get("student_policy_type", "")) != "continuous_density":
        raise ValueError("GICO checkpoint must contain a continuous-density student policy.")
    validate_locked_test_exclusion(
        payload,
        label="GICO checkpoint",
        required_root_keys=("locked_test_used_for_selection",),
    )
    if "context_embedding_kind" in payload:
        raise ValueError("GICO checkpoints may not contain the retired context_embedding_kind field.")
    if str(payload.get("context_embedding_protocol", "")) != FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL:
        raise ValueError(
            f"GICO checkpoint context_embedding_protocol must be {FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL!r}."
        )
    teacher_training = payload.get("teacher_training")
    if not isinstance(teacher_training, Mapping):
        raise ValueError("GICO checkpoint is missing teacher_training metadata.")
    validate_gico_teacher_training_metadata(teacher_training)
    architecture = str(payload.get("student_architecture", ""))
    if architecture != ARCHITECTURE_DENSITY_QUERY_TRANSFORMER:
        raise ValueError(f"GICO student architecture must be {ARCHITECTURE_DENSITY_QUERY_TRANSFORMER!r}.")
    density_metadata = dict(payload.get("density_representation", {}))
    if str(density_metadata.get("density_protocol", "")) != DENSITY_PROTOCOL:
        raise ValueError("GICO checkpoint is missing density-mass metadata.")
    reference_time_grid = validate_reference_grid(density_metadata.get("reference_time_grid", ()))
    density_dim = validate_strict_integer(
        payload.get("density_dim"),
        label="GICO density_dim",
        minimum=2,
    )
    if density_dim != len(reference_time_grid) - 1:
        raise ValueError("GICO density_dim does not match its reference time grid.")
    encoder_payload = payload.get("setting_encoder_config")
    if not isinstance(encoder_payload, Mapping):
        raise ValueError("GICO checkpoint is missing its setting encoder configuration.")
    encoder_config = setting_encoder_config_from_payload(
        encoder_payload,
        require_complete=True,
    )
    expected_setting_dim = setting_feature_dim(encoder_config.mode, config=encoder_config)
    setting_dim = validate_strict_integer(
        payload.get("setting_dim"),
        label="GICO setting_dim",
        minimum=1,
    )
    if setting_dim != expected_setting_dim:
        raise ValueError("GICO setting_dim does not match its setting encoder configuration.")
    normalizer_payload = payload.get("embedding_normalizer")
    if not isinstance(normalizer_payload, Mapping):
        raise ValueError("GICO checkpoint is missing its context embedding normalizer.")
    embedding_normalizer = EmbeddingNormalizer.from_payload(normalizer_payload)
    if embedding_normalizer.mean.shape != embedding_normalizer.std.shape:
        raise ValueError("GICO context embedding normalizer has inconsistent shapes.")
    if not np.isfinite(embedding_normalizer.mean).all() or not np.isfinite(embedding_normalizer.std).all():
        raise ValueError("GICO context embedding normalizer contains non-finite values.")
    if np.any(embedding_normalizer.std <= 0.0):
        raise ValueError("GICO context embedding normalizer must have positive standard deviations.")
    context_dim = validate_strict_integer(
        payload.get("context_dim"),
        label="GICO context_dim",
        minimum=1,
    )
    if embedding_normalizer.mean.shape != (context_dim,):
        raise ValueError("GICO context_dim does not match its context embedding normalizer.")
    model_config = payload.get("student_model_config")
    state = payload.get("student_state")
    if not isinstance(model_config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("GICO checkpoint is missing student model configuration or state.")
    validated_model_config = _validate_student_model_config(
        model_config,
        setting_dim=setting_dim,
        density_dim=density_dim,
        context_dim=context_dim,
    )
    student = build_gico_student_model(
        architecture=architecture,
        setting_dim=expected_setting_dim,
        density_dim=density_dim,
        context_dim=context_dim,
        model_config=validated_model_config,
    ).to(device)
    if student.model_config() != validated_model_config:
        raise ValueError("GICO student model configuration does not match the loaded model.")
    validated_state = validate_tensor_state_dict(
        state,
        label="GICO student state",
        target_module=student,
    )
    try:
        student.load_state_dict(validated_state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"GICO student state is incompatible: {exc}") from exc
    student.eval()
    return GICOSchedulePolicy(
        student,
        embedding_normalizer=embedding_normalizer,
        reference_time_grid=reference_time_grid,
        setting_encoder_config=encoder_config,
        checkpoint_payload=payload,
    )


__all__ = [
    "GICOSchedule",
    "GICOSchedulePolicy",
    "build_gico_student_model",
    "load_gico_schedule_policy",
    "validate_gico_teacher_training_metadata",
]
