from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping

import torch
import torch.nn as nn

from genode.gipo.models import (
    SettingEncoderConfig,
    setting_encoder_config_from_payload,
    setting_feature_dim,
    setting_features,
)
from genode.models.conditioning import ConditioningCache, ConditioningState
from genode.models.config import OTFlowConfig
from genode.models.modules import TransformerFUNet, build_mlp
from genode.solver_protocol import normalize_solver_nfe_fields

if TYPE_CHECKING:
    from genode.distillation.gipo_policy import GIPOSchedulePolicy


DEFAULT_DENSITY_DIM = 64
DEFAULT_PSEUDO_HUBER_DELTA = 0.03


def _require_rank_two(value: torch.Tensor, *, name: str, width: int | None = None) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, features], got {tuple(value.shape)}.")
    if width is not None and int(value.shape[1]) != int(width):
        raise ValueError(f"{name} width must be {int(width)}, got {int(value.shape[1])}.")
    return value


class EndpointFlowMap(nn.Module):
    """Map any teacher-trajectory state directly to its terminal state.

    The residual parameterization enforces ``map(x, 1) == x`` without a
    learned boundary penalty. Context features come from the frozen OTFlow
    conditioning backbone used to generate the demonstrations.
    """

    def __init__(
        self,
        cfg: OTFlowConfig,
        *,
        setting_dim: int,
        density_dim: int = DEFAULT_DENSITY_DIM,
        loss_delta: float = DEFAULT_PSEUDO_HUBER_DELTA,
    ):
        super().__init__()
        if str(cfg.model.fu_net_type).strip().lower() != "transformer":
            raise ValueError("EndpointFlowMap currently requires a transformer OTFlow field.")
        if int(setting_dim) <= 0:
            raise ValueError("setting_dim must be positive.")
        if int(density_dim) <= 1:
            raise ValueError("density_dim must be greater than one.")
        if float(loss_delta) <= 0.0:
            raise ValueError("loss_delta must be positive.")
        self.cfg = cfg
        self.setting_dim = int(setting_dim)
        self.density_dim = int(density_dim)
        self.loss_delta = float(loss_delta)
        hidden_dim = int(cfg.model.hidden_dim)
        dropout = float(cfg.model.dropout)
        self.setting_projection = build_mlp(
            self.setting_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            use_res=bool(cfg.model.use_res_mlp),
        )
        self.density_projection = build_mlp(
            self.density_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            use_res=bool(cfg.model.use_res_mlp),
        )
        conditioning_parts = 5 + (1 if int(cfg.model.cond_dim) > 0 else 0)
        self.map_conditioning_projection = build_mlp(
            conditioning_parts * hidden_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            use_res=bool(cfg.model.use_res_mlp),
        )
        self.map_network = TransformerFUNet(cfg)

    def initialize_from_teacher(self, teacher: nn.Module) -> None:
        teacher_type = str(getattr(teacher, "fu_net_type", "")).strip().lower()
        teacher_network = getattr(teacher, "v_net", None)
        if teacher_type != "transformer" or teacher_network is None:
            raise ValueError("Flow-map initialization requires a transformer OTFlow teacher.")
        self.map_network.load_state_dict(teacher_network.state_dict(), strict=True)

    def model_config(self) -> Dict[str, Any]:
        return {
            "architecture": "endpoint_flow_map",
            "sample_state_dim": int(self.cfg.sample_state_dim),
            "hidden_dim": int(self.cfg.model.hidden_dim),
            "setting_dim": int(self.setting_dim),
            "density_dim": int(self.density_dim),
            "loss_delta": float(self.loss_delta),
            "fu_net_type": "transformer",
        }

    def forward(
        self,
        state: torch.Tensor,
        source_time: torch.Tensor,
        conditioning: ConditioningState,
        setting: torch.Tensor,
        density_mass: torch.Tensor,
        *,
        validate_values: bool = False,
    ) -> torch.Tensor:
        state = _require_rank_two(state, name="state", width=int(self.cfg.sample_state_dim))
        source_time = _require_rank_two(source_time, name="source_time", width=1)
        setting = _require_rank_two(setting, name="setting", width=self.setting_dim)
        density_mass = _require_rank_two(density_mass, name="density_mass", width=self.density_dim)
        batch = int(state.shape[0])
        for name, tensor in (
            ("source_time", source_time),
            ("setting", setting),
            ("density_mass", density_mass),
            ("conditioning.ctx", conditioning.ctx),
            ("conditioning.ctx_summary", conditioning.ctx_summary),
            ("conditioning.t_emb", conditioning.t_emb),
            ("conditioning.ctx_tokens", conditioning.ctx_tokens),
        ):
            if int(tensor.shape[0]) != batch:
                raise ValueError(f"{name} batch size does not match state batch size {batch}.")
        if validate_values:
            mass_sum = density_mass.sum(dim=-1)
            valid = (
                torch.isfinite(state).all()
                & torch.isfinite(source_time).all()
                & torch.isfinite(setting).all()
                & torch.isfinite(density_mass).all()
                & (source_time >= 0.0).all()
                & (source_time <= 1.0).all()
                & (density_mass >= 0.0).all()
                & ((mass_sum - 1.0).abs() <= 1e-5).all()
            )
            if not bool(valid):
                raise ValueError(
                    "Flow-map inputs must be finite, source_time must lie in [0, 1], and "
                    "density_mass must be nonnegative and sum to one per example."
                )

        setting_embedding = self.setting_projection(setting)
        log_density = torch.log(density_mass.clamp_min(1e-8))
        log_density = log_density - log_density.mean(dim=-1, keepdim=True)
        density_embedding = self.density_projection(log_density)
        parts = [
            conditioning.ctx,
            conditioning.ctx_summary,
            conditioning.t_emb,
        ]
        if conditioning.cond_emb is not None:
            parts.append(conditioning.cond_emb)
        elif int(self.cfg.model.cond_dim) > 0:
            parts.append(torch.zeros_like(conditioning.ctx_summary))
        parts.extend((setting_embedding, density_embedding))
        map_conditioning = self.map_conditioning_projection(torch.cat(parts, dim=-1))
        residual = self.map_network(state, conditioning.ctx_tokens, map_conditioning)
        return state + (1.0 - source_time).to(dtype=state.dtype) * residual


def endpoint_consistency_loss(
    prediction: torch.Tensor,
    teacher_endpoint: torch.Tensor,
    *,
    delta: float = DEFAULT_PSEUDO_HUBER_DELTA,
) -> torch.Tensor:
    if prediction.shape != teacher_endpoint.shape:
        raise ValueError(
            f"prediction and teacher_endpoint shapes differ: {tuple(prediction.shape)} != "
            f"{tuple(teacher_endpoint.shape)}."
        )
    scale = float(delta)
    if scale <= 0.0:
        raise ValueError("delta must be positive.")
    error = (prediction - teacher_endpoint) / scale
    return (scale * scale * (torch.sqrt(1.0 + error.square()) - 1.0)).mean()


@dataclass(frozen=True)
class FlowMapSample:
    sample: torch.Tensor
    model_evaluations: int
    solver_key: str
    target_nfe: int
    teacher_realized_nfe: int


class FlowMapSampler:
    """Compose a frozen OTFlow conditioner with a one-evaluation flow map."""

    def __init__(
        self,
        backbone_model: nn.Module,
        flow_map: EndpointFlowMap,
        *,
        setting_encoder_config: Mapping[str, Any] | SettingEncoderConfig,
        gipo_policy: "GIPOSchedulePolicy | None" = None,
    ):
        conditioner = getattr(backbone_model, "backbone", backbone_model)
        if not hasattr(conditioner, "build_conditioning"):
            raise ValueError("backbone_model must expose the OTFlow conditioning backbone.")
        self.conditioner = conditioner
        self.flow_map = flow_map
        self.setting_encoder_config = setting_encoder_config_from_payload(setting_encoder_config)
        self.gipo_policy = gipo_policy
        expected_setting_dim = setting_feature_dim(
            self.setting_encoder_config.mode,
            config=self.setting_encoder_config,
        )
        if expected_setting_dim != int(flow_map.setting_dim):
            raise ValueError(
                "Flow-map setting dimension is incompatible with the supplied setting encoder."
            )
        if self.gipo_policy is not None:
            if int(self.gipo_policy.density_dim) != int(flow_map.density_dim):
                raise ValueError("GIPO and flow-map density dimensions are incompatible.")
            if (
                self.gipo_policy.setting_encoder_config.to_payload()
                != self.setting_encoder_config.to_payload()
            ):
                raise ValueError("GIPO and flow-map setting encoders are incompatible.")
        for module in (self.conditioner, self.flow_map):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.conditioner.eval()
        self.flow_map.eval()
        if self.gipo_policy is not None:
            for parameter in self.gipo_policy.student.parameters():
                parameter.requires_grad_(False)
            self.gipo_policy.student.eval()

    @torch.no_grad()
    def map_state(
        self,
        initial_state: torch.Tensor,
        *,
        solver_key: str,
        target_nfe: int,
        density_mass: torch.Tensor | None = None,
        hist: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        conditioning_cache: ConditioningCache | None = None,
        source_time: float | torch.Tensor = 0.0,
    ) -> FlowMapSample:
        if (hist is None) == (conditioning_cache is None):
            raise ValueError("Provide exactly one of hist or conditioning_cache.")
        if conditioning_cache is not None and cond is not None:
            raise ValueError("cond cannot be combined with a precomputed conditioning_cache.")
        if initial_state.ndim != 2 or int(initial_state.shape[1]) != int(
            self.flow_map.cfg.sample_state_dim
        ):
            raise ValueError(
                "initial_state must have shape [batch, sample_state_dim], got "
                f"{tuple(initial_state.shape)}."
            )
        self.conditioner.eval()
        nfe = normalize_solver_nfe_fields(solver_key, target_nfe, source="flow-map sample")
        batch = int(initial_state.shape[0])
        setting = setting_features(
            nfe.solver_key,
            nfe.target_nfe,
            config=self.setting_encoder_config,
        ).to(device=initial_state.device, dtype=initial_state.dtype)
        setting = setting.unsqueeze(0).expand(batch, -1)
        if isinstance(source_time, torch.Tensor):
            source = source_time.to(device=initial_state.device, dtype=initial_state.dtype)
            if source.ndim == 0:
                source = source.expand(batch).unsqueeze(-1)
            elif source.ndim == 1:
                source = source.unsqueeze(-1)
        else:
            source = initial_state.new_full((batch, 1), float(source_time))
        cache = conditioning_cache
        if cache is None:
            assert hist is not None
            cache = self.conditioner.precompute(hist, cond=cond)
        conditioning = self.conditioner.build_conditioning(
            hist=hist,
            x_ref=initial_state,
            t=source,
            cond=None,
            cache=cache,
        )
        self.flow_map.eval()
        if density_mass is None:
            if self.gipo_policy is None:
                raise ValueError(
                    "density_mass is required when the sampler has no bound GIPO policy."
                )
            schedule = self.gipo_policy.predict(
                cache.ctx_summary,
                solver_key=nfe.solver_key,
                target_nfe=nfe.target_nfe,
            )
            density = schedule.density_mass
        else:
            density = density_mass.to(device=initial_state.device, dtype=initial_state.dtype)
        if density.ndim == 1:
            density = density.unsqueeze(0).expand(batch, -1)
        output = self.flow_map(
            initial_state,
            source,
            conditioning,
            setting,
            density,
            validate_values=True,
        )
        return FlowMapSample(
            sample=output,
            model_evaluations=1,
            solver_key=nfe.solver_key,
            target_nfe=nfe.target_nfe,
            teacher_realized_nfe=nfe.realized_nfe,
        )

    def _initial_state(self, hist: torch.Tensor) -> torch.Tensor:
        if hist.ndim < 1:
            raise ValueError(f"hist must include a batch dimension, got {tuple(hist.shape)}.")
        return torch.randn(
            int(hist.shape[0]),
            int(self.flow_map.cfg.sample_state_dim),
            device=hist.device,
            dtype=hist.dtype,
        )

    @torch.no_grad()
    def sample(
        self,
        hist: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        solver_key: str,
        target_nfe: int,
        density_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw one autoregressive sample with a single flow-map evaluation."""

        if int(self.flow_map.cfg.prediction_horizon) != 1:
            raise RuntimeError("Non-autoregressive flow maps use sample_future(...), not sample(...).")
        return self.map_state(
            self._initial_state(hist),
            hist=hist,
            cond=cond,
            solver_key=solver_key,
            target_nfe=target_nfe,
            density_mass=density_mass,
        ).sample

    @torch.no_grad()
    def sample_future(
        self,
        hist: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        solver_key: str,
        target_nfe: int,
        density_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw and reshape one future block with a single map evaluation."""

        flat = self.map_state(
            self._initial_state(hist),
            hist=hist,
            cond=cond,
            solver_key=solver_key,
            target_nfe=target_nfe,
            density_mass=density_mass,
        ).sample
        return flat.reshape(
            int(flat.shape[0]),
            int(self.flow_map.cfg.prediction_horizon),
            int(self.flow_map.cfg.snapshot_dim),
        )


__all__ = [
    "DEFAULT_DENSITY_DIM",
    "DEFAULT_PSEUDO_HUBER_DELTA",
    "EndpointFlowMap",
    "FlowMapSample",
    "FlowMapSampler",
    "endpoint_consistency_loss",
]
