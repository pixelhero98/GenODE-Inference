from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import torch

from genode.distillation.artifacts import CHECKPOINT_PROTOCOL
from genode.distillation.model import EndpointFlowMap, FlowMapSampler
from genode.gipo.models import (
    SettingEncoderConfig,
    setting_encoder_config_from_payload,
    setting_feature_dim,
)
from genode.models.config import OTFlowConfig
from genode.provenance import file_sha256


FLOW_MAP_ARTIFACT_VERSION = 1
QUALITY_STATUS_NOT_EVALUATED = "not_evaluated"


def config_from_payload(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> OTFlowConfig:
    """Reconstruct an :class:`OTFlowConfig` without accepting unknown fields."""

    cfg = OTFlowConfig()
    target_device = torch.device(device)
    section_names = ("data", "model", "fm", "train", "sample")
    unknown_sections = sorted(set(payload) - set(section_names))
    if unknown_sections:
        raise ValueError(
            f"Flow-map checkpoint config contains unknown sections: {unknown_sections}."
        )
    for section_name in section_names:
        section = getattr(cfg, section_name)
        section_type = type(section)
        section_payload = dict(payload.get(section_name, {}))
        valid_fields = {field.name for field in fields(section_type)}
        unknown = sorted(set(section_payload) - valid_fields)
        if unknown:
            raise ValueError(
                f"Flow-map checkpoint config section {section_name!r} contains unknown fields: {unknown}."
            )
        values = {field.name: getattr(section, field.name) for field in fields(section_type)}
        values.update(section_payload)
        if section_name == "train":
            values["device"] = target_device
        setattr(cfg, section_name, section_type(**values))
    cfg.train.device = target_device
    return cfg


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        str(name): value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_flow_map_checkpoint(
    path: str | Path,
    flow_map: EndpointFlowMap,
    *,
    backbone_checkpoint: str | Path,
    gipo_checkpoint: str | Path,
    setting_encoder_config: Mapping[str, Any] | SettingEncoderConfig,
    training_summary: Mapping[str, Any],
    demonstration_manifest_sha256: str,
    overwrite: bool = False,
) -> Path:
    """Write one portable, final flow-map checkpoint.

    Source artifact paths are deliberately not serialized. Their content hashes
    are sufficient for compatibility checks and do not leak workstation layout.
    """

    target = Path(path).expanduser().resolve()
    backbone_path = Path(backbone_checkpoint).expanduser().resolve()
    gipo_path = Path(gipo_checkpoint).expanduser().resolve()
    if not backbone_path.is_file():
        raise ValueError(f"Backbone checkpoint does not exist: {backbone_path.name!r}.")
    if not gipo_path.is_file():
        raise ValueError(f"GIPO checkpoint does not exist: {gipo_path.name!r}.")
    if target in {backbone_path, gipo_path}:
        raise ValueError("Flow-map output checkpoint must differ from every source checkpoint.")
    if target.exists() and not bool(overwrite):
        raise FileExistsError(
            f"Refusing to overwrite existing flow-map checkpoint {target.name!r}."
        )
    manifest_hash = str(demonstration_manifest_sha256)
    if re.fullmatch(r"[0-9a-f]{64}", manifest_hash) is None:
        raise ValueError("demonstration_manifest_sha256 must be a lowercase SHA-256 digest.")
    summary = dict(training_summary)
    if bool(summary.get("locked_test_used_for_selection", False)):
        raise ValueError("Locked-test data may not be used to select a flow-map checkpoint.")
    encoder_config = setting_encoder_config_from_payload(setting_encoder_config)
    payload = {
        "protocol": CHECKPOINT_PROTOCOL,
        "artifact_version": FLOW_MAP_ARTIFACT_VERSION,
        "model_config": flow_map.model_config(),
        "otflow_config": flow_map.cfg.to_dict(),
        "model_state": _cpu_state_dict(flow_map),
        "setting_encoder_config": encoder_config.to_payload(),
        "backbone_checkpoint_sha256": file_sha256(backbone_path),
        "gipo_checkpoint_sha256": file_sha256(gipo_path),
        "demonstration_manifest_sha256": manifest_hash,
        "training_summary": summary,
        "quality_gate": {
            "status": QUALITY_STATUS_NOT_EVALUATED,
            "performance_claim": False,
        },
    }
    _atomic_torch_save(payload, target)
    return target


def load_flow_map_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    backbone_checkpoint: str | Path | None = None,
    gipo_checkpoint: str | Path | None = None,
) -> tuple[EndpointFlowMap, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not load flow-map checkpoint {checkpoint_path.name!r}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Flow-map checkpoint must contain a mapping payload.")
    if str(payload.get("protocol", "")) != CHECKPOINT_PROTOCOL:
        raise ValueError(
            f"Unsupported flow-map protocol {payload.get('protocol')!r}; expected {CHECKPOINT_PROTOCOL!r}."
        )
    if int(payload.get("artifact_version", 0)) != FLOW_MAP_ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported flow-map artifact_version {payload.get('artifact_version')!r}; "
            f"expected {FLOW_MAP_ARTIFACT_VERSION}."
        )
    for hash_key in (
        "backbone_checkpoint_sha256",
        "gipo_checkpoint_sha256",
        "demonstration_manifest_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(payload.get(hash_key, ""))) is None:
            raise ValueError(f"Flow-map checkpoint has an invalid {hash_key} value.")
    training_summary = payload.get("training_summary")
    if not isinstance(training_summary, Mapping):
        raise ValueError("Flow-map checkpoint is missing training_summary.")
    if bool(training_summary.get("locked_test_used_for_selection", False)):
        raise ValueError("Flow-map checkpoint indicates locked-test use during model selection.")
    quality_gate = payload.get("quality_gate")
    if (
        not isinstance(quality_gate, Mapping)
        or str(quality_gate.get("status", "")) != QUALITY_STATUS_NOT_EVALUATED
        or bool(quality_gate.get("performance_claim", False))
    ):
        raise ValueError("Flow-map checkpoint has invalid quality-gate metadata.")
    for source_path, hash_key, label in (
        (backbone_checkpoint, "backbone_checkpoint_sha256", "backbone"),
        (gipo_checkpoint, "gipo_checkpoint_sha256", "GIPO"),
    ):
        if source_path is None:
            continue
        actual = file_sha256(Path(source_path).expanduser().resolve())
        expected = str(payload.get(hash_key, ""))
        if not expected or actual != expected:
            raise ValueError(f"Flow-map checkpoint is not compatible with the supplied {label} checkpoint.")

    cfg_payload = payload.get("otflow_config")
    model_config = dict(payload.get("model_config", {}))
    if not isinstance(cfg_payload, Mapping):
        raise ValueError("Flow-map checkpoint is missing otflow_config.")
    if str(model_config.get("architecture", "")) != "endpoint_flow_map":
        raise ValueError("Flow-map checkpoint has an unsupported architecture.")
    cfg = config_from_payload(cfg_payload, device=device)
    if int(model_config.get("sample_state_dim", -1)) != int(cfg.sample_state_dim):
        raise ValueError("Flow-map model_config sample_state_dim does not match otflow_config.")
    encoder_config = setting_encoder_config_from_payload(payload.get("setting_encoder_config"))
    expected_setting_dim = setting_feature_dim(encoder_config.mode, config=encoder_config)
    if int(model_config.get("setting_dim", -1)) != expected_setting_dim:
        raise ValueError("Flow-map setting_dim does not match its setting encoder configuration.")
    flow_map = EndpointFlowMap(
        cfg,
        setting_dim=int(model_config["setting_dim"]),
        density_dim=int(model_config["density_dim"]),
        loss_delta=float(model_config["loss_delta"]),
    ).to(device)
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("Flow-map checkpoint is missing model_state.")
    result = flow_map.load_state_dict(dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "Flow-map model state is incompatible: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}."
        )
    flow_map.eval()
    metadata = dict(payload)
    metadata.pop("model_state", None)
    return flow_map, metadata


def load_flow_map_sampler(
    path: str | Path,
    *,
    backbone_checkpoint: str | Path,
    gipo_checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[FlowMapSampler, dict[str, Any]]:
    """Load a sampler bound to the exact checkpoints used for distillation."""

    from genode.evaluation.otflow_evaluation_support import load_checkpoint_model

    backbone_path = Path(backbone_checkpoint).expanduser().resolve()
    gipo_path = Path(gipo_checkpoint).expanduser().resolve()
    flow_map, payload = load_flow_map_checkpoint(
        path,
        device=device,
        backbone_checkpoint=backbone_path,
        gipo_checkpoint=gipo_path,
    )
    from genode.distillation.gipo_policy import load_gipo_schedule_policy

    backbone_model, _ = load_checkpoint_model(backbone_path, torch.device(device))
    gipo_policy = load_gipo_schedule_policy(gipo_path, device=device)
    sampler = FlowMapSampler(
        backbone_model,
        flow_map,
        setting_encoder_config=payload["setting_encoder_config"],
        gipo_policy=gipo_policy,
    )
    return sampler, payload


__all__ = [
    "FLOW_MAP_ARTIFACT_VERSION",
    "QUALITY_STATUS_NOT_EVALUATED",
    "config_from_payload",
    "load_flow_map_checkpoint",
    "load_flow_map_sampler",
    "save_flow_map_checkpoint",
]
