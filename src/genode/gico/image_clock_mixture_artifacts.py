from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from genode.artifact_bundle import (
    discard_temporary_bundle_path,
    preflight_artifact_bundle,
    promote_artifact_bundle,
    temporary_bundle_path,
    validate_artifact_bundle,
)
from genode.artifacts.identity import (
    canonical_json_bytes,
    canonical_json_text,
    semantic_sha256,
)
from genode.backbones.adapter import CanonicalNoiseToDataAdapter
from genode.benchmarks.image.artifact_paths import managed_image_leaf_directory
from genode.benchmarks.image.protocol import IMAGE_TARGET_NFES
from genode.gico.image_clock_mixture import (
    ImageGICOBackboneContextClockMixtureModel,
    ImageGICOClockLibrary,
    ImageGICOClockMixtureModelConfig,
    ImageGICOClockMixturePolicy,
    ImageGICOClockRealization,
    build_image_gico_clock_library_from_time_grids,
    image_clock_mixture_serialized_state_sha256,
)
from genode.gico.image_clock_mixture_training import (
    IMAGE_GICO_CLOCK_MIXTURE_TRAINING_PROTOCOL,
    ImageGICOClockMixtureTrainingConfig,
    ImageGICOClockMixtureTrainingResult,
    _context_table_sha256,
    _determinism_contract,
    _validate_library_target_binding,
    recompute_image_gico_clock_mixture_diagnostics,
)
from genode.gico.image_conditional_artifacts import (
    IMAGE_GICO_CONDITIONAL_BUNDLE_PROTOCOL,
    BoundImageGICOConditionalArtifact,
    LoadedImageGICOConditionalArtifact,
    _verified_manifest_artifact_sha256 as _verified_source_manifest_artifact_sha256,
)
from genode.provenance import file_sha256
from genode.schedules.policy import ScheduleBatch


IMAGE_GICO_CLOCK_MIXTURE_BUNDLE_PROTOCOL = "image_gico_complete_clock_mixture_bundle_v1"
_BOUND_ARTIFACT_CONSTRUCTION_TOKEN = object()
_LOADED_ARTIFACT_CONSTRUCTION_TOKEN = object()
_MANIFEST = "manifest.json"
_MODEL_STATE = "clock-mixture-state.pt"
_CLOCK_LIBRARY = "clock-library.json"
_CLOCK_GRID_FILENAMES = {
    2: "complete-clocks-nfe-2.npy",
    4: "complete-clocks-nfe-4.npy",
    8: "complete-clocks-nfe-8.npy",
}
_SHA256_IDENTITY = re.compile(r"(?:(?:[a-z][a-z0-9_.-]*):)?[0-9a-f]{64}\Z")
_ROLE_NAMES = {
    "manifest",
    "model_state",
    "clock_library",
    "clock_grid_nfe_2",
    "clock_grid_nfe_4",
    "clock_grid_nfe_8",
}


def _identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 identity.")
    return value


def _paths(output_dir: str | Path) -> dict[str, Path]:
    root = managed_image_leaf_directory(
        output_dir,
        label="Complete-clock GICO sidecar directory",
    )
    return {
        "manifest": root / _MANIFEST,
        "model_state": root / _MODEL_STATE,
        "clock_library": root / _CLOCK_LIBRARY,
        **{f"clock_grid_nfe_{target_nfe}": root / filename for target_nfe, filename in _CLOCK_GRID_FILENAMES.items()},
    }


def _canonical_payload(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Non-finite JSON constant {value!r}.")),
    )
    if not isinstance(payload, Mapping) or canonical_json_text(payload) != text:
        raise ValueError(f"{path.name} is not canonical JSON.")
    return payload


def _state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{path.name} does not contain a state dictionary.")
    result: dict[str, torch.Tensor] = {}
    for name, value in payload.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{path.name} contains an unsafe state dictionary.")
        if value.layout != torch.strided or value.is_quantized or value.device.type != "cpu":
            raise ValueError(f"{path.name} contains a nonportable state tensor.")
        tensor = value.detach().to(device="cpu").contiguous()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{path.name} contains non-finite state values.")
        result[name] = tensor.clone()
    return MappingProxyType(result)


def _save_npy(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype="<f8")
    if not np.all(np.isfinite(array)):
        raise ValueError("Complete-clock grids must contain only finite values.")
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _clock_grid(
    path: Path,
    *,
    schedule_count: int,
    target_nfe: int,
) -> torch.Tensor:
    value = np.load(path, allow_pickle=False)
    expected_shape = (schedule_count, target_nfe + 1)
    if value.dtype != np.dtype("<f8") or value.shape != expected_shape:
        raise ValueError(f"{path.name} must have dtype <f8 and shape {expected_shape}.")
    array = np.ascontiguousarray(value, dtype="<f8")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path.name} must contain only finite values.")
    return torch.from_numpy(array.copy()).to(dtype=torch.float64)


def _library_payload(library: ImageGICOClockLibrary) -> dict[str, object]:
    if not isinstance(library, ImageGICOClockLibrary):
        raise TypeError("library must be an ImageGICOClockLibrary.")
    return {
        "artifact": "image_gico_complete_clock_library",
        **library.identity_payload(),
        "clock_library_sha256": library.sha256,
    }


def _training_fields() -> set[str]:
    return {
        "protocol",
        "conditioning",
        "context_binding_sha256",
        "normalized_context_table_sha256",
        "target_sha256",
        "clock_library_sha256",
        "training_config",
        "training_config_sha256",
        "model_config",
        "model_config_sha256",
        "model_state_sha256",
        "coverage",
        "execution_device_type",
        "determinism",
        "training_dtype",
        "diagnostic_dtype",
        "target_reconstruction_max_abs_error",
        "final_cross_entropy",
        "final_kl",
        "final_residual_penalty",
        "final_objective",
        "final_barycenter_l1",
        "max_probability_error",
        "teacher_scoring",
        "interpolated_density_teacher_scoring",
        "result_sha256",
    }


def _verified_training_result_sha256(payload: object) -> str:
    if not isinstance(payload, Mapping) or set(payload) != _training_fields():
        raise ValueError("Clock-mixture training-result fields are incomplete or unexpected.")
    body = dict(payload)
    stored = _identity(
        body.pop("result_sha256", None),
        field="training.result_sha256",
    )
    if body.get("protocol") != IMAGE_GICO_CLOCK_MIXTURE_TRAINING_PROTOCOL:
        raise ValueError("Unsupported clock-mixture training protocol.")
    observed = semantic_sha256(
        body,
        namespace="image-gico-complete-clock-mixture-training-result-v1",
    )
    if stored != observed:
        raise ValueError("Clock-mixture training-result identity is inconsistent.")
    return observed


def _manifest_fields() -> set[str]:
    return {
        "artifact",
        "protocol",
        "artifact_sha256",
        "source_artifact_sha256",
        "source_artifact_protocol",
        "target_sha256",
        "context_binding_sha256",
        "backbone_model_key",
        "backbone_protocol_sha256",
        "backbone_checkpoint_sha256",
        "clock_library_sha256",
        "model_config",
        "model_config_sha256",
        "model_state_sha256",
        "serialized_model_state_sha256",
        "training",
        "portable_execution",
        "files",
    }


def _verified_manifest_artifact_sha256(manifest: Mapping[str, Any]) -> str:
    if set(manifest) != _manifest_fields():
        raise ValueError(f"Clock-mixture bundle manifest fields must be exactly {sorted(_manifest_fields())}.")
    if (
        manifest.get("artifact") != "image_gico_complete_clock_mixture_policy"
        or manifest.get("protocol") != IMAGE_GICO_CLOCK_MIXTURE_BUNDLE_PROTOCOL
    ):
        raise ValueError("Unsupported complete-clock GICO sidecar bundle.")
    body = dict(manifest)
    stored = _identity(
        body.pop("artifact_sha256", None),
        field="artifact_sha256",
    )
    observed = semantic_sha256(
        body,
        namespace="image-gico-complete-clock-mixture-artifact-v1",
    )
    if stored != observed:
        raise ValueError("Clock-mixture artifact identity is inconsistent.")
    return observed


def _source_binding(
    source_artifact: LoadedImageGICOConditionalArtifact | BoundImageGICOConditionalArtifact,
) -> Any:
    if type(source_artifact) is LoadedImageGICOConditionalArtifact:
        return source_artifact.context_binding
    if type(source_artifact) is BoundImageGICOConditionalArtifact:
        return source_artifact.prepared_context.binding
    raise TypeError("source_artifact must be a loaded or bound ImageNet GICO artifact.")


def _validate_source(
    source_artifact: LoadedImageGICOConditionalArtifact | BoundImageGICOConditionalArtifact,
) -> str:
    if type(source_artifact) is BoundImageGICOConditionalArtifact:
        source_artifact.verify_execution_identity()
    elif type(source_artifact) is not LoadedImageGICOConditionalArtifact:
        raise TypeError("source_artifact must be a loaded or bound ImageNet GICO artifact.")
    source_sha256 = _verified_source_manifest_artifact_sha256(source_artifact.manifest)
    binding = _source_binding(source_artifact)
    if (
        source_artifact.artifact_sha256 != source_sha256
        or source_artifact.manifest.get("protocol") != IMAGE_GICO_CONDITIONAL_BUNDLE_PROTOCOL
        or source_artifact.manifest.get("target_sha256") != source_artifact.targets.sha256
        or source_artifact.manifest.get("context_binding_sha256") != binding.binding_sha256
    ):
        raise ValueError("Source GICO artifact identity is inconsistent.")
    return source_sha256


def _validate_source_manifest_binding(
    manifest: Mapping[str, Any],
    source_artifact: LoadedImageGICOConditionalArtifact | BoundImageGICOConditionalArtifact,
) -> None:
    source_sha256 = _validate_source(source_artifact)
    binding = _source_binding(source_artifact)
    expected = (
        source_sha256,
        IMAGE_GICO_CONDITIONAL_BUNDLE_PROTOCOL,
        source_artifact.targets.sha256,
        binding.binding_sha256,
        source_artifact.targets.backbone_model_key,
        source_artifact.targets.backbone_protocol_sha256,
        source_artifact.targets.backbone_checkpoint_sha256,
    )
    observed = (
        manifest.get("source_artifact_sha256"),
        manifest.get("source_artifact_protocol"),
        manifest.get("target_sha256"),
        manifest.get("context_binding_sha256"),
        manifest.get("backbone_model_key"),
        manifest.get("backbone_protocol_sha256"),
        manifest.get("backbone_checkpoint_sha256"),
    )
    if observed != expected:
        raise ValueError("Clock-mixture sidecar does not match its source artifact.")


def _reconstructed_library(
    paths: Mapping[str, Path],
    source_artifact: LoadedImageGICOConditionalArtifact | BoundImageGICOConditionalArtifact,
) -> ImageGICOClockLibrary:
    schedule_count = len(source_artifact.targets.schedule_keys)
    grids = tuple(
        _clock_grid(
            paths[f"clock_grid_nfe_{target_nfe}"],
            schedule_count=schedule_count,
            target_nfe=target_nfe,
        )
        for target_nfe in IMAGE_TARGET_NFES
    )
    return build_image_gico_clock_library_from_time_grids(
        source_artifact.targets,
        grids,
    )


def _validate_state_for_library(
    state: Mapping[str, torch.Tensor],
    *,
    library: ImageGICOClockLibrary,
    config: ImageGICOClockMixtureModelConfig,
    canonical_context_table: np.ndarray | torch.Tensor | None = None,
) -> ImageGICOBackboneContextClockMixtureModel:
    contexts = (
        np.zeros((config.class_count, config.context_dim), dtype=np.float32)
        if canonical_context_table is None
        else canonical_context_table
    )
    model = ImageGICOBackboneContextClockMixtureModel(
        config,
        contexts,
        library,
    )
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Clock-mixture state has unexpected or missing tensor names.")
    for name, expected_tensor in expected.items():
        observed = state[name]
        if (
            observed.layout != torch.strided
            or observed.is_quantized
            or observed.device.type != "cpu"
            or observed.shape != expected_tensor.shape
            or observed.dtype != expected_tensor.dtype
        ):
            raise ValueError(f"Clock-mixture tensor {name!r} has an invalid layout, device, shape, or dtype.")
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Clock-mixture state does not match its declared architecture.") from exc
    model.eval()
    model.requires_grad_(False)
    return model


def _validate_training_binding(
    training: object,
    *,
    manifest: Mapping[str, Any],
    source_artifact: LoadedImageGICOConditionalArtifact | BoundImageGICOConditionalArtifact,
    library: ImageGICOClockLibrary,
    config: ImageGICOClockMixtureModelConfig,
) -> ImageGICOClockMixtureTrainingConfig:
    _verified_training_result_sha256(training)
    if not isinstance(training, Mapping):
        raise TypeError("Clock-mixture training result must be a mapping.")
    training_config_payload = training.get("training_config")
    if not isinstance(training_config_payload, Mapping):
        raise ValueError("Clock-mixture training config must be a mapping.")
    training_config = ImageGICOClockMixtureTrainingConfig.from_payload(
        training_config_payload
    )
    if training.get("training_config_sha256") != training_config.sha256:
        raise ValueError("Clock-mixture training-config identity is inconsistent.")
    coverage = training.get("coverage")
    execution_device_type = training.get("execution_device_type")
    determinism = training.get("determinism")
    cuda_workspace = (
        determinism.get("cuda_cublas_workspace_config")
        if isinstance(determinism, Mapping)
        else None
    )
    if (
        execution_device_type == "cpu"
        and cuda_workspace is not None
    ) or (
        execution_device_type == "cuda"
        and cuda_workspace not in {":16:8", ":4096:8"}
    ):
        raise ValueError("Clock-mixture training device determinism is invalid.")
    expected_coverage = {
        "class_count": config.class_count,
        "target_nfes": list(config.target_nfes),
        "schedule_keys": list(library.schedule_keys),
        "schedule_count": library.schedule_count,
        "group_count_by_nfe": list(library.group_counts),
        "class_nfe_row_count": len(config.target_nfes) * config.class_count,
        "all_target_mixture_weights_consumed": True,
    }
    if (
        training.get("conditioning") != "normalized_frozen_backbone_map_label_plus_target_nfe"
        or training.get("context_binding_sha256") != _source_binding(source_artifact).binding_sha256
        or training.get("target_sha256") != source_artifact.targets.sha256
        or training.get("clock_library_sha256") != library.sha256
        or training.get("model_config") != config.as_payload()
        or training.get("model_config_sha256") != config.sha256
        or training.get("model_state_sha256") != manifest.get("model_state_sha256")
        or coverage != expected_coverage
        or training.get("training_dtype") != "float32"
        or training.get("diagnostic_dtype") != "float64"
        or training.get("teacher_scoring") != "none"
        or training.get("interpolated_density_teacher_scoring") is not False
        or execution_device_type not in {"cpu", "cuda"}
        or determinism
        != _determinism_contract(
            execution_device_type=str(execution_device_type),
            cuda_cublas_workspace_config=(
                None if cuda_workspace is None else str(cuda_workspace)
            ),
        )
    ):
        raise ValueError("Clock-mixture training bindings are inconsistent.")
    _identity(
        training.get("normalized_context_table_sha256"),
        field="training.normalized_context_table_sha256",
    )
    reconstruction_error = _validate_library_target_binding(
        library,
        source_artifact.targets,
    )
    for field_name in (
        "target_reconstruction_max_abs_error",
        "final_cross_entropy",
        "final_kl",
        "final_residual_penalty",
        "final_objective",
        "final_barycenter_l1",
        "max_probability_error",
    ):
        value = training.get(field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"training.{field_name} must be finite and nonnegative.")
    if not math.isclose(
        float(training["target_reconstruction_max_abs_error"]),
        reconstruction_error,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "Clock-mixture target-reconstruction diagnostic is inconsistent."
        )
    if float(training["max_probability_error"]) > 1.0:
        raise ValueError("training.max_probability_error must be at most one.")
    expected_objective = (
        float(training["final_cross_entropy"])
        + training_config.residual_penalty_weight
        * float(training["final_residual_penalty"])
    )
    if not math.isclose(
        float(training["final_objective"]),
        expected_objective,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise ValueError("Clock-mixture final objective is inconsistent.")
    return training_config


def _validate_recomputed_training_diagnostics(
    training: Mapping[str, Any],
    *,
    model: ImageGICOBackboneContextClockMixtureModel,
    source_artifact: BoundImageGICOConditionalArtifact,
) -> None:
    training_config_payload = training.get("training_config")
    if not isinstance(training_config_payload, Mapping):
        raise ValueError("Clock-mixture training config must be a mapping.")
    training_config = ImageGICOClockMixtureTrainingConfig.from_payload(
        training_config_payload
    )
    observed = recompute_image_gico_clock_mixture_diagnostics(
        model,
        source_artifact.targets,
        training_config,
    )
    for field_name, observed_value in observed.items():
        stored_value = float(training[field_name])
        if not math.isclose(
            stored_value,
            observed_value,
            rel_tol=1e-5,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"Clock-mixture training diagnostic {field_name!r} "
                "does not match the frozen model."
            )


def _validate_bundle(
    paths: Mapping[str, Path],
    expected_paths: Mapping[str, Path],
    *,
    source_artifact: LoadedImageGICOConditionalArtifact | BoundImageGICOConditionalArtifact,
) -> None:
    if set(paths) != set(expected_paths) or set(paths) != _ROLE_NAMES:
        raise ValueError("Clock-mixture bundle roles are incomplete or unexpected.")
    manifest = _canonical_payload(paths["manifest"])
    _verified_manifest_artifact_sha256(manifest)
    _validate_source_manifest_binding(manifest, source_artifact)

    expected_roles = set(paths) - {"manifest"}
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != expected_roles:
        raise ValueError("Clock-mixture bundle file bindings are incomplete.")
    for role in expected_roles:
        binding = files[role]
        path = paths[role]
        if not isinstance(binding, Mapping):
            raise ValueError("Clock-mixture file bindings must be mappings.")
        if (
            set(binding) != {"filename", "sha256", "size_bytes"}
            or binding.get("filename") != expected_paths[role].name
            or binding.get("sha256") != file_sha256(path)
            or binding.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"Clock-mixture file {role!r} is inconsistent.")

    library = _reconstructed_library(paths, source_artifact)
    library_payload = _canonical_payload(paths["clock_library"])
    if dict(library_payload) != _library_payload(library):
        raise ValueError("Persisted complete-clock library identity is inconsistent.")
    if manifest.get("clock_library_sha256") != library.sha256:
        raise ValueError("Clock-mixture manifest names the wrong clock library.")

    config = ImageGICOClockMixtureModelConfig.for_library(library)
    if manifest.get("model_config") != config.as_payload() or manifest.get("model_config_sha256") != config.sha256:
        raise ValueError("Clock-mixture model configuration is inconsistent.")
    state = _state_dict(paths["model_state"])
    if (
        image_clock_mixture_serialized_state_sha256(state)
        != manifest.get("serialized_model_state_sha256")
    ):
        raise ValueError("Serialized clock-mixture model-state identity is inconsistent.")
    canonical_context_table = (
        source_artifact.normalized_context_table if type(source_artifact) is BoundImageGICOConditionalArtifact else None
    )
    model = _validate_state_for_library(
        state,
        library=library,
        config=config,
        canonical_context_table=canonical_context_table,
    )
    if canonical_context_table is not None and model.state_sha256 != manifest.get("model_state_sha256"):
        raise ValueError("Clock-mixture model-state identity is inconsistent.")
    training = manifest.get("training")
    _validate_training_binding(
        training,
        manifest=manifest,
        source_artifact=source_artifact,
        library=library,
        config=config,
    )
    if type(source_artifact) is BoundImageGICOConditionalArtifact:
        if not isinstance(training, Mapping):
            raise TypeError("Clock-mixture training result must be a mapping.")
        _validate_recomputed_training_diagnostics(
            training,
            model=model,
            source_artifact=source_artifact,
        )
    expected_portable = {
        "load_phase": "metadata_weights_and_exact_clock_library_only",
        "bind_phase": "source_artifact_regenerates_context_from_verified_backbone",
        "requires_explicit_source_artifact": True,
        "requires_explicit_backbone_binding": True,
        "raw_context_table_is_portable": False,
        "conditional_targets_are_portable": False,
        "internal_rng": False,
    }
    if manifest.get("portable_execution") != expected_portable:
        raise ValueError("Clock-mixture portable-execution contract is inconsistent.")


def _validate_loaded_decoder(
    *,
    source_artifact: LoadedImageGICOConditionalArtifact,
    library: ImageGICOClockLibrary,
    manifest: Mapping[str, Any],
    model_state: Mapping[str, torch.Tensor],
) -> None:
    _verified_manifest_artifact_sha256(manifest)
    _validate_source_manifest_binding(manifest, source_artifact)
    library.validate_targets(source_artifact.targets)
    if manifest.get("clock_library_sha256") != library.sha256:
        raise ValueError("Loaded complete-clock library identity changed.")
    config = ImageGICOClockMixtureModelConfig.for_library(library)
    if manifest.get("model_config") != config.as_payload() or manifest.get("model_config_sha256") != config.sha256:
        raise ValueError("Loaded clock-mixture model configuration changed.")
    if (
        image_clock_mixture_serialized_state_sha256(model_state)
        != manifest.get("serialized_model_state_sha256")
    ):
        raise ValueError("Loaded serialized clock-mixture state changed.")
    _validate_state_for_library(
        model_state,
        library=library,
        config=config,
    )
    _validate_training_binding(
        manifest.get("training"),
        manifest=manifest,
        source_artifact=source_artifact,
        library=library,
        config=config,
    )


def _module_tensors(
    module: torch.nn.Module,
) -> Mapping[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for prefix, rows in (
        ("parameter", module.named_parameters()),
        ("buffer", module.named_buffers()),
    ):
        for name, value in rows:
            values[f"{prefix}:{name}"] = value
    return values


def _snapshot_tensors(
    values: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    return MappingProxyType(
        {name: value.detach().to(device="cpu").contiguous().clone() for name, value in values.items()}
    )


def _library_tensors(
    library: ImageGICOClockLibrary,
) -> Mapping[str, torch.Tensor]:
    return {
        "reference_time_grid": library.reference_time_grid,
        "density_mass": library.density_mass,
        "supervision_reference_time_grid": (library.supervision_reference_time_grid),
        "supervision_density_mass": library.supervision_density_mass,
        **{
            f"time_grids_nfe_{target_nfe}": grid
            for target_nfe, grid in zip(
                library.target_nfes,
                library.time_grids,
                strict=True,
            )
        },
    }


def _verify_tensor_snapshot(
    observed: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if set(observed) != set(expected):
        raise ValueError(f"{label} tensor names were modified.")
    for name, expected_tensor in expected.items():
        value = observed[name].detach()
        if (
            value.device.type != "cpu"
            or value.layout != torch.strided
            or value.is_quantized
            or value.dtype != expected_tensor.dtype
            or value.shape != expected_tensor.shape
            or not torch.equal(value, expected_tensor)
        ):
            raise ValueError(f"{label} tensor {name!r} was modified.")


@dataclass(frozen=True)
class BoundImageGICOClockMixtureArtifact:
    source_artifact: BoundImageGICOConditionalArtifact
    policy: ImageGICOClockMixturePolicy
    library: ImageGICOClockLibrary
    manifest: Mapping[str, Any]
    _construction_token: object = field(repr=False, compare=False)
    _bound_artifact_sha256: str = field(init=False, repr=False, compare=False)
    _expected_policy_sha256: str = field(init=False, repr=False, compare=False)
    _expected_module_tensors: Mapping[str, torch.Tensor] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _expected_library_tensors: Mapping[str, torch.Tensor] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _expected_library_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _BOUND_ARTIFACT_CONSTRUCTION_TOKEN:
            raise TypeError(
                "Bound clock-mixture artifacts must be created by LoadedImageGICOClockMixtureArtifact.bind()."
            )
        if type(self.source_artifact) is not BoundImageGICOConditionalArtifact:
            raise TypeError("A bound clock mixture requires its bound source artifact.")
        if type(self.policy) is not ImageGICOClockMixturePolicy:
            raise TypeError("Bound clock-mixture policy must use the canonical policy class.")
        if type(self.policy.model) is not ImageGICOBackboneContextClockMixtureModel:
            raise TypeError("Bound clock-mixture policy must use the canonical model class.")
        if type(self.library) is not ImageGICOClockLibrary:
            raise TypeError("Bound clock mixture must use the canonical library class.")
        artifact_sha256 = _verified_manifest_artifact_sha256(self.manifest)
        object.__setattr__(self, "_bound_artifact_sha256", artifact_sha256)
        object.__setattr__(self, "_expected_policy_sha256", self.policy.policy_sha256)
        object.__setattr__(
            self,
            "_expected_module_tensors",
            _snapshot_tensors(_module_tensors(self.policy)),
        )
        object.__setattr__(
            self,
            "_expected_library_tensors",
            _snapshot_tensors(_library_tensors(self.library)),
        )
        object.__setattr__(self, "_expected_library_sha256", self.library.sha256)
        self.verify_execution_identity()

    @property
    def artifact_sha256(self) -> str:
        value = self.manifest.get("artifact_sha256")
        return _identity(value, field="artifact_sha256")

    def verify_execution_identity(self) -> None:
        self.source_artifact.verify_execution_identity()
        observed_artifact_sha256 = _verified_manifest_artifact_sha256(self.manifest)
        if (
            observed_artifact_sha256 != self._bound_artifact_sha256
            or self.artifact_sha256 != self._bound_artifact_sha256
        ):
            raise ValueError("Bound clock-mixture artifact identity changed.")
        if (
            type(self.policy) is not ImageGICOClockMixturePolicy
            or type(self.policy.model) is not ImageGICOBackboneContextClockMixtureModel
            or type(self.library) is not ImageGICOClockLibrary
            or self.policy.library is not self.library
            or self.policy.model.library is not self.library
        ):
            raise TypeError("Bound clock-mixture execution classes were modified.")
        if self.policy.training or self.policy.model.training:
            raise ValueError("Bound clock-mixture execution requires eval mode.")
        if any(parameter.requires_grad for parameter in self.policy.parameters()):
            raise ValueError("Bound clock-mixture parameters must remain frozen.")
        if (
            self.library.sha256 != self._expected_library_sha256
            or self.manifest.get("clock_library_sha256") != self._expected_library_sha256
            or self.policy.model.config.clock_library_sha256 != self._expected_library_sha256
            or self.manifest.get("model_config") != self.policy.model.config.as_payload()
            or self.manifest.get("model_config_sha256") != self.policy.model.config.sha256
            or self.manifest.get("model_state_sha256") != self.policy.model.state_sha256
            or self.manifest.get("serialized_model_state_sha256")
            != image_clock_mixture_serialized_state_sha256(
                self.policy.model.state_dict()
            )
            or self.policy.policy_sha256 != self._expected_policy_sha256
        ):
            raise ValueError("Bound clock-mixture content identity changed.")
        if (
            self.manifest.get("source_artifact_sha256") != self.source_artifact.artifact_sha256
            or self.manifest.get("target_sha256") != self.source_artifact.targets.sha256
            or self.manifest.get("context_binding_sha256")
            != self.source_artifact.prepared_context.binding.binding_sha256
            or self.policy.context_binding_sha256 != self.source_artifact.prepared_context.binding.binding_sha256
        ):
            raise ValueError("Bound clock-mixture source binding changed.")
        observed_context = self.policy.model.canonical_context_table.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        expected_context = self.source_artifact.normalized_context_table.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        if not torch.equal(observed_context, expected_context):
            raise ValueError("Bound clock-mixture context table changed.")
        _verify_tensor_snapshot(
            _module_tensors(self.policy),
            self._expected_module_tensors,
            label="Bound clock-mixture model",
        )
        _verify_tensor_snapshot(
            _library_tensors(self.library),
            self._expected_library_tensors,
            label="Bound complete-clock library",
        )

    def contexts_for_class_labels(self, labels: torch.Tensor) -> torch.Tensor:
        self.verify_execution_identity()
        if not isinstance(labels, torch.Tensor):
            raise TypeError("Class labels must be a torch.Tensor.")
        return self.source_artifact.contexts_for_class_labels(labels.to(device="cpu"))

    def predict_for_class_labels(
        self,
        labels: torch.Tensor,
        *,
        target_nfe: int,
    ) -> ScheduleBatch:
        contexts = self.contexts_for_class_labels(labels)
        return self.policy.predict(contexts, target_nfe=target_nfe)

    def realize_for_class_labels(
        self,
        labels: torch.Tensor,
        *,
        target_nfe: int,
        uniforms: torch.Tensor,
        alpha: float,
    ) -> ImageGICOClockRealization:
        contexts = self.contexts_for_class_labels(labels)
        return self.policy.sample_realization(
            contexts,
            target_nfe=target_nfe,
            uniforms=uniforms,
            alpha=alpha,
        )


@dataclass(frozen=True)
class LoadedImageGICOClockMixtureArtifact:
    source_artifact: LoadedImageGICOConditionalArtifact
    library: ImageGICOClockLibrary
    manifest: Mapping[str, Any]
    _model_state: Mapping[str, torch.Tensor] = field(repr=False, compare=False)
    _construction_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _LOADED_ARTIFACT_CONSTRUCTION_TOKEN:
            raise TypeError(
                "Loaded clock-mixture artifacts must be created by load_image_gico_clock_mixture_artifact()."
            )
        _validate_source(self.source_artifact)
        if type(self.library) is not ImageGICOClockLibrary:
            raise TypeError("Loaded clock mixture must use the canonical library class.")
        _validate_loaded_decoder(
            source_artifact=self.source_artifact,
            library=self.library,
            manifest=self.manifest,
            model_state=self._model_state,
        )

    @property
    def artifact_sha256(self) -> str:
        value = self.manifest.get("artifact_sha256")
        return _identity(value, field="artifact_sha256")

    def bind(
        self,
        backbone: CanonicalNoiseToDataAdapter,
    ) -> BoundImageGICOClockMixtureArtifact:
        if not isinstance(backbone, CanonicalNoiseToDataAdapter):
            raise TypeError("backbone must be a verified canonical image adapter.")
        _validate_loaded_decoder(
            source_artifact=self.source_artifact,
            library=self.library,
            manifest=self.manifest,
            model_state=self._model_state,
        )
        source = self.source_artifact.bind(backbone)
        source.verify_execution_identity()
        training = self.manifest.get("training")
        if not isinstance(training, Mapping):
            raise TypeError("Clock-mixture training result must be a mapping.")
        observed_context_sha256 = _context_table_sha256(
            source.normalized_context_table.detach().to(device="cpu", dtype=torch.float32).contiguous()
        )
        if training.get("normalized_context_table_sha256") != observed_context_sha256:
            raise ValueError("Verified backbone context does not match clock-mixture training.")
        config = ImageGICOClockMixtureModelConfig.for_library(self.library)
        model = ImageGICOBackboneContextClockMixtureModel(
            config,
            source.prepared_context.normalized_context_table,
            self.library,
        )
        model.load_state_dict(self._model_state, strict=True)
        model.to(device="cpu")
        model.eval()
        model.requires_grad_(False)
        if model.state_sha256 != self.manifest.get("model_state_sha256"):
            raise ValueError("Loaded clock-mixture model state is inconsistent.")
        _validate_recomputed_training_diagnostics(
            training,
            model=model,
            source_artifact=source,
        )
        policy = ImageGICOClockMixturePolicy(
            model,
            self.library,
            context_binding_sha256=(source.prepared_context.binding.binding_sha256),
        )
        policy.eval()
        policy.requires_grad_(False)
        return BoundImageGICOClockMixtureArtifact(
            source_artifact=source,
            policy=policy,
            library=self.library,
            manifest=self.manifest,
            _construction_token=_BOUND_ARTIFACT_CONSTRUCTION_TOKEN,
        )


def _validate_result_source_binding(
    result: ImageGICOClockMixtureTrainingResult,
    source_artifact: BoundImageGICOConditionalArtifact,
) -> ImageGICOClockLibrary:
    if type(result) is not ImageGICOClockMixtureTrainingResult:
        raise TypeError("result must be an ImageGICOClockMixtureTrainingResult.")
    if type(source_artifact) is not BoundImageGICOConditionalArtifact:
        raise TypeError("source_artifact must be a bound source GICO artifact.")
    source_artifact.verify_execution_identity()
    library = result.model.library
    if type(library) is not ImageGICOClockLibrary:
        raise TypeError("Training result must use the canonical clock library.")
    rebuilt = build_image_gico_clock_library_from_time_grids(
        source_artifact.targets,
        library.time_grids,
    )
    if (
        library.sha256 != rebuilt.sha256
        or library.identity_payload() != rebuilt.identity_payload()
        or any(
            not torch.equal(observed, expected)
            for observed, expected in (
                (library.reference_time_grid, rebuilt.reference_time_grid),
                (library.density_mass, rebuilt.density_mass),
                (
                    library.supervision_reference_time_grid,
                    rebuilt.supervision_reference_time_grid,
                ),
                (
                    library.supervision_density_mass,
                    rebuilt.supervision_density_mass,
                ),
            )
        )
        or any(
            not torch.equal(observed, expected)
            for observed, expected in zip(
                library.time_grids,
                rebuilt.time_grids,
                strict=True,
            )
        )
    ):
        raise ValueError("Training-result clock library is not canonical.")
    source_context = source_artifact.normalized_context_table.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    result_context = result.model.canonical_context_table.detach().to(
        device="cpu",
        dtype=torch.float32,
    )
    expected_config = ImageGICOClockMixtureModelConfig.for_library(library)
    if (
        result.target_sha256 != source_artifact.targets.sha256
        or result.context_binding_sha256 != source_artifact.prepared_context.binding.binding_sha256
        or result.clock_library_sha256 != library.sha256
        or result.model.config != expected_config
        or result.model_state_sha256 != result.model.state_sha256
        or result.normalized_context_table_sha256 != _context_table_sha256(source_context)
        or not torch.equal(result_context, source_context)
    ):
        raise ValueError("Training result does not match its bound source artifact.")
    diagnostics = recompute_image_gico_clock_mixture_diagnostics(
        result.model,
        source_artifact.targets,
        result.config,
    )
    for field_name, observed_value in diagnostics.items():
        if not math.isclose(
            float(getattr(result, field_name)),
            observed_value,
            rel_tol=1e-5,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"Training result diagnostic {field_name!r} does not match "
                "its frozen model."
            )
    return library


def save_image_gico_clock_mixture_artifact(
    output_dir: str | Path,
    result: ImageGICOClockMixtureTrainingResult,
    *,
    source_artifact: BoundImageGICOConditionalArtifact,
    overwrite: bool = False,
) -> Mapping[str, Path]:
    library = _validate_result_source_binding(result, source_artifact)
    paths = _paths(output_dir)
    anchor = paths[sorted(paths)[0]]

    def validator(
        current: Mapping[str, Path],
        expected: Mapping[str, Path],
    ) -> None:
        _validate_bundle(
            current,
            expected,
            source_artifact=source_artifact,
        )

    preflight_artifact_bundle(
        anchor,
        paths,
        overwrite=bool(overwrite),
        validator=validator,
    )
    staged = {role: temporary_bundle_path(path) for role, path in paths.items()}
    staged_hashes: dict[str, str] = {}
    try:
        state = {
            name: value.detach().to(device="cpu").contiguous() for name, value in result.model.state_dict().items()
        }
        serialized_model_state_sha256 = (
            image_clock_mixture_serialized_state_sha256(state)
        )
        with staged["model_state"].open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        staged["clock_library"].write_bytes(canonical_json_bytes(_library_payload(library)))
        for target_nfe, grid in zip(
            library.target_nfes,
            library.time_grids,
            strict=True,
        ):
            _save_npy(
                staged[f"clock_grid_nfe_{target_nfe}"],
                grid.detach().to(device="cpu", dtype=torch.float64).numpy(),
            )
        data_roles = tuple(role for role in paths if role != "manifest")
        for role in data_roles:
            staged_hashes[role] = file_sha256(staged[role])
        training = result.manifest_payload()
        source_binding = source_artifact.prepared_context.binding
        config = result.model.config
        manifest_body = {
            "artifact": "image_gico_complete_clock_mixture_policy",
            "protocol": IMAGE_GICO_CLOCK_MIXTURE_BUNDLE_PROTOCOL,
            "source_artifact_sha256": source_artifact.artifact_sha256,
            "source_artifact_protocol": IMAGE_GICO_CONDITIONAL_BUNDLE_PROTOCOL,
            "target_sha256": source_artifact.targets.sha256,
            "context_binding_sha256": source_binding.binding_sha256,
            "backbone_model_key": source_binding.backbone_model_key,
            "backbone_protocol_sha256": source_binding.backbone_protocol_sha256,
            "backbone_checkpoint_sha256": source_binding.backbone_checkpoint_sha256,
            "clock_library_sha256": library.sha256,
            "model_config": config.as_payload(),
            "model_config_sha256": config.sha256,
            "model_state_sha256": result.model_state_sha256,
            "serialized_model_state_sha256": serialized_model_state_sha256,
            "training": training,
            "portable_execution": {
                "load_phase": "metadata_weights_and_exact_clock_library_only",
                "bind_phase": "source_artifact_regenerates_context_from_verified_backbone",
                "requires_explicit_source_artifact": True,
                "requires_explicit_backbone_binding": True,
                "raw_context_table_is_portable": False,
                "conditional_targets_are_portable": False,
                "internal_rng": False,
            },
            "files": {
                role: {
                    "filename": paths[role].name,
                    "sha256": staged_hashes[role],
                    "size_bytes": staged[role].stat().st_size,
                }
                for role in data_roles
            },
        }
        manifest = {
            **manifest_body,
            "artifact_sha256": semantic_sha256(
                manifest_body,
                namespace="image-gico-complete-clock-mixture-artifact-v1",
            ),
        }
        staged["manifest"].write_bytes(canonical_json_bytes(manifest))
        staged_hashes["manifest"] = file_sha256(staged["manifest"])
        promote_artifact_bundle(
            anchor,
            paths,
            staged,
            overwrite=bool(overwrite),
            validator=validator,
        )
        return paths
    finally:
        for role, path in staged.items():
            discard_temporary_bundle_path(
                path,
                paths[role],
                expected_sha256=staged_hashes.get(role),
            )


def load_image_gico_clock_mixture_artifact(
    output_dir: str | Path,
    *,
    source_artifact: LoadedImageGICOConditionalArtifact,
) -> LoadedImageGICOClockMixtureArtifact:
    _validate_source(source_artifact)
    paths = _paths(output_dir)
    anchor = paths[sorted(paths)[0]]

    def validator(
        current: Mapping[str, Path],
        expected: Mapping[str, Path],
    ) -> None:
        _validate_bundle(
            current,
            expected,
            source_artifact=source_artifact,
        )

    validate_artifact_bundle(
        anchor,
        paths,
        validator=validator,
    )
    manifest = _canonical_payload(paths["manifest"])
    library = _reconstructed_library(paths, source_artifact)
    return LoadedImageGICOClockMixtureArtifact(
        source_artifact=source_artifact,
        library=library,
        manifest=manifest,
        _model_state=_state_dict(paths["model_state"]),
        _construction_token=_LOADED_ARTIFACT_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "IMAGE_GICO_CLOCK_MIXTURE_BUNDLE_PROTOCOL",
    "BoundImageGICOClockMixtureArtifact",
    "LoadedImageGICOClockMixtureArtifact",
    "load_image_gico_clock_mixture_artifact",
    "save_image_gico_clock_mixture_artifact",
]
