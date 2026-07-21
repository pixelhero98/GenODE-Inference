from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from genode.gipo.density_representation import (
    DENSITY_PROTOCOL,
    validate_reference_grid,
)
from genode.gipo.models import (
    SettingEncoderConfig,
    setting_encoder_config_from_payload,
    setting_feature_dim,
    setting_features,
)
from genode.gipo.policy import (
    ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
    GIPO_PROTOCOL,
    MODEL_PAYLOAD_VERSION,
    EmbeddingNormalizer,
    build_gipo_student_model,
    normalize_gipo_checkpoint_payload,
)
from genode.solver_protocol import normalize_solver_nfe_fields


@dataclass(frozen=True)
class GIPOSchedule:
    solver_key: str
    target_nfe: int
    macro_steps: int
    realized_nfe: int
    density_mass: torch.Tensor
    time_grid: torch.Tensor


class GIPOSchedulePolicy:
    """Validated GIPO student used to produce context-dependent ODE schedules."""

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
        try:
            parameter = next(student.parameters())
        except StopIteration as exc:
            raise ValueError("GIPO student must contain model parameters.") from exc
        if not parameter.is_floating_point():
            raise ValueError("GIPO student parameters must use a floating-point dtype.")
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
        quantiles = torch.linspace(
            0.0,
            1.0,
            int(macro_steps) + 1,
            device=mass.device,
            dtype=mass.dtype,
        ).expand(int(mass.shape[0]), -1)
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
        context_summary: torch.Tensor,
        *,
        solver_key: str,
        target_nfe: int,
    ) -> GIPOSchedule:
        if context_summary.ndim != 2:
            raise ValueError(
                "context_summary must have shape [batch, context_dim], "
                f"got {tuple(context_summary.shape)}."
            )
        if not torch.isfinite(context_summary).all():
            raise ValueError("context_summary contains non-finite values.")
        nfe = normalize_solver_nfe_fields(solver_key, target_nfe, source="GIPO schedule prediction")
        output_device = context_summary.device
        output_dtype = context_summary.dtype
        student_context = context_summary.to(
            device=self._student_device,
            dtype=self._student_dtype,
        )
        mean = self._normalizer_mean
        std = self._normalizer_std
        if mean.shape != student_context.shape[1:] or std.shape != student_context.shape[1:]:
            raise ValueError(
                "GIPO context normalizer is incompatible with the backbone context summary: "
                f"normalizer={tuple(mean.shape)}, context={tuple(student_context.shape[1:])}."
            )
        normalized_context = (student_context - mean) / std.clamp_min(1e-6)
        features = setting_features(
            nfe.solver_key,
            nfe.target_nfe,
            mode=self.setting_encoder_config.mode,
            config=self.setting_encoder_config,
        ).to(device=self._student_device, dtype=self._student_dtype)
        feature_batch = features.unsqueeze(0).expand(int(student_context.shape[0]), -1)
        student_density = self.student.density_mass(feature_batch, normalized_context)
        if student_density.shape != (int(student_context.shape[0]), self.density_dim):
            raise ValueError(
                "GIPO student returned an invalid density shape: "
                f"{tuple(student_density.shape)}."
            )
        student_time_grid = self._time_grid(
            student_density,
            macro_steps=nfe.macro_steps,
        )
        density_mass = student_density.to(device=output_device, dtype=output_dtype)
        time_grid = student_time_grid.to(device=output_device, dtype=output_dtype)
        return GIPOSchedule(
            solver_key=nfe.solver_key,
            target_nfe=nfe.target_nfe,
            macro_steps=nfe.macro_steps,
            realized_nfe=nfe.realized_nfe,
            density_mass=density_mass,
            time_grid=time_grid,
        )


def load_gipo_schedule_policy(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> GIPOSchedulePolicy:
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not load GIPO checkpoint {checkpoint_path.name!r}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("GIPO student checkpoint must contain a mapping payload.")
    payload = normalize_gipo_checkpoint_payload(payload)
    if str(payload.get("protocol", "")) != GIPO_PROTOCOL:
        raise ValueError(
            f"Unsupported GIPO protocol {payload.get('protocol')!r}; expected {GIPO_PROTOCOL!r}."
        )
    if int(payload.get("model_payload_version", 0)) != MODEL_PAYLOAD_VERSION:
        raise ValueError(
            f"Unsupported GIPO model_payload_version {payload.get('model_payload_version')!r}; "
            f"expected {MODEL_PAYLOAD_VERSION}."
        )
    if str(payload.get("student_policy_type", "")) != "continuous_density":
        raise ValueError("GIPO checkpoint must contain a continuous-density student policy.")
    if bool(payload.get("locked_test_used_for_selection", False)):
        raise ValueError("GIPO checkpoint indicates that locked-test data was used for selection.")
    architecture = str(payload.get("student_architecture", ""))
    if architecture != ARCHITECTURE_DENSITY_QUERY_TRANSFORMER:
        raise ValueError(
            f"GIPO student architecture must be {ARCHITECTURE_DENSITY_QUERY_TRANSFORMER!r}."
        )
    density_metadata = dict(payload.get("density_representation", {}))
    if str(density_metadata.get("density_protocol", "")) != DENSITY_PROTOCOL:
        raise ValueError("GIPO checkpoint is missing density-mass metadata.")
    reference_time_grid = validate_reference_grid(density_metadata.get("reference_time_grid", ()))
    density_dim = int(payload.get("density_dim", 0))
    if density_dim != len(reference_time_grid) - 1:
        raise ValueError("GIPO density_dim does not match its reference time grid.")
    encoder_payload = payload.get("setting_encoder_config")
    if not isinstance(encoder_payload, Mapping):
        raise ValueError("GIPO checkpoint is missing its setting encoder configuration.")
    encoder_config = setting_encoder_config_from_payload(encoder_payload)
    expected_setting_dim = setting_feature_dim(encoder_config.mode, config=encoder_config)
    if int(payload.get("setting_dim", 0)) != expected_setting_dim:
        raise ValueError("GIPO setting_dim does not match its setting encoder configuration.")
    normalizer_payload = payload.get("embedding_normalizer")
    if not isinstance(normalizer_payload, Mapping):
        raise ValueError("GIPO checkpoint is missing its context embedding normalizer.")
    embedding_normalizer = EmbeddingNormalizer.from_payload(normalizer_payload)
    if embedding_normalizer.mean.shape != embedding_normalizer.std.shape:
        raise ValueError("GIPO context embedding normalizer has inconsistent shapes.")
    if not np.isfinite(embedding_normalizer.mean).all() or not np.isfinite(embedding_normalizer.std).all():
        raise ValueError("GIPO context embedding normalizer contains non-finite values.")
    if np.any(embedding_normalizer.std <= 0.0):
        raise ValueError("GIPO context embedding normalizer must have positive standard deviations.")
    context_dim = int(payload.get("context_dim", 0))
    if context_dim <= 0 or embedding_normalizer.mean.shape != (context_dim,):
        raise ValueError("GIPO context_dim does not match its context embedding normalizer.")
    model_config = payload.get("student_model_config")
    state = payload.get("student_state")
    if not isinstance(model_config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("GIPO checkpoint is missing student model configuration or state.")
    student = build_gipo_student_model(
        architecture=architecture,
        setting_dim=expected_setting_dim,
        density_dim=density_dim,
        context_dim=context_dim,
        model_config=model_config,
    ).to(device)
    result = student.load_state_dict(dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "GIPO student state is incompatible: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}."
        )
    student.eval()
    return GIPOSchedulePolicy(
        student,
        embedding_normalizer=embedding_normalizer,
        reference_time_grid=reference_time_grid,
        setting_encoder_config=encoder_config,
        checkpoint_payload=payload,
    )


__all__ = ["GIPOSchedule", "GIPOSchedulePolicy", "load_gipo_schedule_policy"]
