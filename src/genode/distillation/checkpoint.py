from __future__ import annotations

from dataclasses import fields
import math
from numbers import Real
import os
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any, Mapping, get_args, get_origin, get_type_hints

import torch

from genode.checkpoint_validation import (
    validate_locked_test_exclusion,
    validate_strict_integer,
    validate_tensor_state_dict,
)
from genode.distillation.artifacts import CHECKPOINT_PROTOCOL, validate_context_binding
from genode.distillation.model import EndpointFlowMap, FlowMapSampler
from genode.experiment_layout import scenario_family_for_key
from genode.gipo.models import (
    SettingEncoderConfig,
    setting_encoder_config_from_payload,
    setting_feature_dim,
)
from genode.models.config import OTFlowConfig
from genode.path_safety import is_link_or_reparse_point
from genode.provenance import file_sha256


FLOW_MAP_ARTIFACT_VERSION = 1
QUALITY_STATUS_NOT_EVALUATED = "not_evaluated"


def config_from_payload(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> OTFlowConfig:
    """Reconstruct an :class:`OTFlowConfig` without accepting unknown fields."""

    if not isinstance(payload, Mapping):
        raise ValueError("Flow-map checkpoint config must be an object.")
    cfg = OTFlowConfig()
    target_device = torch.device(device)
    section_names = ("data", "model", "fm", "train", "sample")
    missing_sections = sorted(set(section_names) - set(payload))
    unknown_sections = sorted(set(payload) - set(section_names))
    if missing_sections or unknown_sections:
        raise ValueError(
            "Flow-map checkpoint config sections must be complete; "
            f"missing={missing_sections}, unknown={unknown_sections}."
        )
    for section_name in section_names:
        section = getattr(cfg, section_name)
        section_type = type(section)
        raw_section_payload = payload[section_name]
        if not isinstance(raw_section_payload, Mapping):
            raise ValueError(
                f"Flow-map checkpoint config section {section_name!r} must be an object."
            )
        section_payload = dict(raw_section_payload)
        valid_fields = {field.name for field in fields(section_type)}
        missing = sorted(valid_fields - set(section_payload))
        unknown = sorted(set(section_payload) - valid_fields)
        if missing or unknown:
            raise ValueError(
                f"Flow-map checkpoint config section {section_name!r} must be complete; "
                f"missing={missing}, unknown={unknown}."
            )
        values = {field.name: getattr(section, field.name) for field in fields(section_type)}
        type_hints = get_type_hints(section_type)
        for field_name, value in section_payload.items():
            annotation = type_hints.get(field_name)
            if annotation is int:
                value = validate_strict_integer(
                    value,
                    label=f"Flow-map otflow_config {section_name}.{field_name}",
                )
            elif get_origin(annotation) is tuple and get_args(annotation) == (int, Ellipsis):
                if not isinstance(value, (list, tuple)):
                    raise ValueError(
                        f"Flow-map otflow_config {section_name}.{field_name} must be an integer sequence."
                    )
                value = tuple(
                    validate_strict_integer(
                        item,
                        label=(
                            f"Flow-map otflow_config {section_name}.{field_name}[{index}]"
                        ),
                    )
                    for index, item in enumerate(value)
                )
            elif annotation is bool:
                if not isinstance(value, bool):
                    raise ValueError(
                        f"Flow-map otflow_config {section_name}.{field_name} must be a boolean."
                    )
            elif annotation is float:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(
                        f"Flow-map otflow_config {section_name}.{field_name} must be numeric."
                    )
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(
                        f"Flow-map otflow_config {section_name}.{field_name} must be finite."
                    )
            elif annotation is str:
                if not isinstance(value, str):
                    raise ValueError(
                        f"Flow-map otflow_config {section_name}.{field_name} must be a string."
                    )
            elif annotation is torch.device:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Flow-map otflow_config {section_name}.{field_name} must be a device string."
                    )
                try:
                    torch.device(value)
                except (RuntimeError, TypeError) as exc:
                    raise ValueError(
                        f"Flow-map otflow_config {section_name}.{field_name} is invalid."
                    ) from exc
            values[field_name] = value
        if section_name == "train":
            values["device"] = target_device
        setattr(cfg, section_name, section_type(**values))
    cfg.train.device = target_device
    return cfg


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {
        str(name): value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }
    return validate_tensor_state_dict(state, label="Flow-map model state")


def _atomic_torch_save(
    payload: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool,
) -> None:
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
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite existing flow-map checkpoint {path.name!r}."
                ) from exc
            temporary.unlink()
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
    demonstration_metadata: Mapping[str, Any] | None = None,
    expected_backbone_checkpoint_sha256: str | None = None,
    expected_gipo_checkpoint_sha256: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one portable, final flow-map checkpoint.

    Source artifact paths are deliberately not serialized. Their content hashes
    are sufficient for compatibility checks and do not leak workstation layout.
    """

    lexical_target = Path(path).expanduser()
    if is_link_or_reparse_point(lexical_target):
        raise ValueError(
            "Flow-map output checkpoint may not be a symlink, junction, or reparse point."
        )
    target = lexical_target.parent.resolve() / lexical_target.name
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
    source_hashes: dict[str, str] = {}
    for source_path, expected_hash, key, label in (
        (
            backbone_path,
            expected_backbone_checkpoint_sha256,
            "backbone_checkpoint_sha256",
            "backbone",
        ),
        (
            gipo_path,
            expected_gipo_checkpoint_sha256,
            "gipo_checkpoint_sha256",
            "GIPO",
        ),
    ):
        actual_hash = file_sha256(source_path)
        if expected_hash is not None:
            expected = str(expected_hash)
            if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                raise ValueError(
                    f"expected_{key} must be a lowercase SHA-256 digest."
                )
            if actual_hash != expected:
                raise ValueError(
                    f"{label} checkpoint changed after its training identity was captured."
                )
            actual_hash = expected
        source_hashes[key] = actual_hash
    manifest_hash = str(demonstration_manifest_sha256)
    if re.fullmatch(r"[0-9a-f]{64}", manifest_hash) is None:
        raise ValueError("demonstration_manifest_sha256 must be a lowercase SHA-256 digest.")
    summary = dict(training_summary)
    validate_locked_test_exclusion(
        summary,
        label="Flow-map training summary",
        required_root_keys=("locked_test_used_for_selection",),
    )
    encoder_config = setting_encoder_config_from_payload(
        setting_encoder_config,
        require_complete=not isinstance(setting_encoder_config, SettingEncoderConfig),
    )
    demonstration_binding: dict[str, Any] | None = None
    demonstration_scenario = ""
    demonstration_family = ""
    if demonstration_metadata is not None:
        metadata = dict(demonstration_metadata)
        raw_binding = metadata.get("context_binding")
        if not isinstance(raw_binding, Mapping):
            raise ValueError("Demonstration metadata is missing its context binding.")
        demonstration_binding = validate_context_binding(raw_binding)
        expected_context_count = validate_strict_integer(
            summary.get("train_context_count"),
            label="Flow-map training train_context_count",
            minimum=0,
        ) + validate_strict_integer(
            summary.get("validation_context_count"),
            label="Flow-map training validation_context_count",
            minimum=0,
        )
        if demonstration_binding["context_count"] != expected_context_count:
            raise ValueError(
                "Demonstration context binding does not match the training context counts."
            )
        demonstration_scenario = str(metadata.get("scenario_key", "")).strip()
        demonstration_family = str(metadata.get("benchmark_family", "")).strip()
        if not demonstration_scenario or not demonstration_family:
            raise ValueError(
                "Demonstration metadata requires scenario_key and benchmark_family."
            )
        try:
            expected_family = scenario_family_for_key(demonstration_scenario)
        except KeyError:
            expected_family = ""
        if expected_family and demonstration_family != expected_family:
            raise ValueError(
                "Demonstration benchmark_family does not match its scenario_key."
            )
    payload = {
        "protocol": CHECKPOINT_PROTOCOL,
        "artifact_version": FLOW_MAP_ARTIFACT_VERSION,
        "model_config": flow_map.model_config(),
        "otflow_config": flow_map.cfg.to_dict(),
        "model_state": _cpu_state_dict(flow_map),
        "setting_encoder_config": encoder_config.to_payload(),
        **source_hashes,
        "demonstration_manifest_sha256": manifest_hash,
        "training_summary": summary,
        "quality_gate": {
            "status": QUALITY_STATUS_NOT_EVALUATED,
            "performance_claim": False,
        },
    }
    if demonstration_binding is not None:
        payload.update(
            {
                "demonstration_context_binding": demonstration_binding,
                "scenario_key": demonstration_scenario,
                "benchmark_family": demonstration_family,
            }
        )
    _atomic_torch_save(payload, target, overwrite=bool(overwrite))
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
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"Could not load flow-map checkpoint {checkpoint_path.name!r}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Flow-map checkpoint must contain a mapping payload.")
    if str(payload.get("protocol", "")) != CHECKPOINT_PROTOCOL:
        raise ValueError(
            f"Unsupported flow-map protocol {payload.get('protocol')!r}; expected {CHECKPOINT_PROTOCOL!r}."
        )
    if validate_strict_integer(
        payload.get("artifact_version"),
        label="Flow-map artifact_version",
        minimum=1,
    ) != FLOW_MAP_ARTIFACT_VERSION:
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
    validate_locked_test_exclusion(
        training_summary,
        label="Flow-map checkpoint training_summary",
        required_root_keys=("locked_test_used_for_selection",),
    )
    raw_context_binding = payload.get("demonstration_context_binding")
    if raw_context_binding is not None:
        if not isinstance(raw_context_binding, Mapping):
            raise ValueError("Flow-map checkpoint has an invalid demonstration context binding.")
        validated_binding = validate_context_binding(raw_context_binding)
        if not str(payload.get("scenario_key", "")).strip() or not str(
            payload.get("benchmark_family", "")
        ).strip():
            raise ValueError(
                "Flow-map checkpoint context binding requires scenario and benchmark metadata."
            )
        checkpoint_scenario = str(payload["scenario_key"]).strip()
        checkpoint_family = str(payload["benchmark_family"]).strip()
        try:
            expected_family = scenario_family_for_key(checkpoint_scenario)
        except KeyError:
            expected_family = ""
        if expected_family and checkpoint_family != expected_family:
            raise ValueError(
                "Flow-map checkpoint benchmark_family does not match its scenario_key."
            )
        expected_context_count = validate_strict_integer(
            training_summary.get("train_context_count"),
            label="Flow-map training_summary train_context_count",
            minimum=0,
        ) + validate_strict_integer(
            training_summary.get("validation_context_count"),
            label="Flow-map training_summary validation_context_count",
            minimum=0,
        )
        if validated_binding["context_count"] != expected_context_count:
            raise ValueError(
                "Flow-map checkpoint context binding does not match its training context counts."
            )
    quality_gate = payload.get("quality_gate")
    if (
        not isinstance(quality_gate, Mapping)
        or str(quality_gate.get("status", "")) != QUALITY_STATUS_NOT_EVALUATED
        or quality_gate.get("performance_claim") is not False
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
    raw_model_config = payload.get("model_config")
    if not isinstance(cfg_payload, Mapping):
        raise ValueError("Flow-map checkpoint is missing otflow_config.")
    if not isinstance(raw_model_config, Mapping):
        raise ValueError("Flow-map checkpoint is missing model_config.")
    model_config = dict(raw_model_config)
    expected_model_config_fields = {
        "architecture",
        "sample_state_dim",
        "hidden_dim",
        "setting_dim",
        "density_dim",
        "loss_delta",
        "fu_net_type",
    }
    missing_model_fields = sorted(expected_model_config_fields - set(model_config))
    extra_model_fields = sorted(set(model_config) - expected_model_config_fields)
    if missing_model_fields or extra_model_fields:
        raise ValueError(
            "Flow-map model_config fields are invalid; "
            f"missing={missing_model_fields}, extra={extra_model_fields}."
        )
    if str(model_config.get("architecture", "")) != "endpoint_flow_map":
        raise ValueError("Flow-map checkpoint has an unsupported architecture.")
    if model_config.get("fu_net_type") != "transformer":
        raise ValueError("Flow-map checkpoint requires fu_net_type='transformer'.")
    cfg = config_from_payload(cfg_payload, device=device)
    sample_state_dim = validate_strict_integer(
        model_config.get("sample_state_dim"),
        label="Flow-map model_config sample_state_dim",
        minimum=1,
    )
    if sample_state_dim != int(cfg.sample_state_dim):
        raise ValueError("Flow-map model_config sample_state_dim does not match otflow_config.")
    hidden_dim = validate_strict_integer(
        model_config.get("hidden_dim"),
        label="Flow-map model_config hidden_dim",
        minimum=1,
    )
    if hidden_dim != int(cfg.model.hidden_dim):
        raise ValueError("Flow-map model_config hidden_dim does not match otflow_config.")
    encoder_config = setting_encoder_config_from_payload(
        payload.get("setting_encoder_config"), require_complete=True
    )
    expected_setting_dim = setting_feature_dim(encoder_config.mode, config=encoder_config)
    setting_dim = validate_strict_integer(
        model_config.get("setting_dim"),
        label="Flow-map model_config setting_dim",
        minimum=1,
    )
    if setting_dim != expected_setting_dim:
        raise ValueError("Flow-map setting_dim does not match its setting encoder configuration.")
    density_dim = validate_strict_integer(
        model_config.get("density_dim"),
        label="Flow-map model_config density_dim",
        minimum=2,
    )
    raw_loss_delta = model_config["loss_delta"]
    if isinstance(raw_loss_delta, bool) or not isinstance(raw_loss_delta, Real):
        raise ValueError("Flow-map model_config loss_delta must be numeric.")
    loss_delta = float(raw_loss_delta)
    if not math.isfinite(loss_delta) or loss_delta <= 0.0:
        raise ValueError("Flow-map model_config loss_delta must be finite and positive.")
    flow_map = EndpointFlowMap(
        cfg,
        setting_dim=setting_dim,
        density_dim=density_dim,
        loss_delta=loss_delta,
    ).to(device)
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("Flow-map checkpoint is missing model_state.")
    validated_state = validate_tensor_state_dict(
        state,
        label="Flow-map model state",
    )
    try:
        flow_map.load_state_dict(validated_state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Flow-map model state is incompatible: {exc}") from exc
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
