from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
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
from genode.gico.image_conditional import (
    ImageGICOBackboneContextDensityModel,
    ImageGICOBackboneContextModelConfig,
    ImageGICOBackboneContextSchedulePolicy,
    ImageGICOConditionalTargets,
    ImageGICOFeatureGroups,
)
from genode.gico.image_conditional_context import (
    ImageGICOBackboneContextBinding,
    ImageGICOContextNormalizer,
    PreparedImageGICOBackboneContext,
    bind_image_gico_backbone_context,
)
from genode.gico.image_conditional_training import (
    IMAGE_GICO_BACKBONE_CONTEXT_TRAINING_PROTOCOL,
    ImageGICOBackboneContextTeacher,
    ImageGICOBackboneContextTrainingResult,
    conditional_module_state_sha256,
)
from genode.provenance import file_sha256


IMAGE_GICO_CONDITIONAL_BUNDLE_PROTOCOL = (
    "image_gico_backbone_context_policy_bundle_v4"
)
_MANIFEST = "manifest.json"
_FEATURE_GROUPS = "reward-feature-groups.json"
_TARGETS = "conditional-targets.json"
_STUDENT_STATE = "student-state.pt"
_TEACHER_STATE = "teacher-state.pt"
_DENSITY_TABLE = "class-density-table.npy"
_CONTEXT_MEAN = "context-normalizer-mean.npy"
_CONTEXT_SCALE = "context-normalizer-scale.npy"
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _training_result_identity(payload: object) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError("Backbone-context training result must be a mapping.")
    body = dict(payload)
    stored = body.pop("result_sha256", None)
    if body.get("protocol") != IMAGE_GICO_BACKBONE_CONTEXT_TRAINING_PROTOCOL:
        raise ValueError("Unsupported backbone-context training-result protocol.")
    observed = semantic_sha256(
        body,
        namespace="image-gico-backbone-context-training-result-v4",
    )
    if stored != observed:
        raise ValueError("Backbone-context training-result hash is inconsistent.")
    return observed


def _paths(output_dir: str | Path) -> dict[str, Path]:
    root = managed_image_leaf_directory(
        output_dir,
        label="Backbone-context GICO bundle directory",
    )
    return {
        "manifest": root / _MANIFEST,
        "feature_groups": root / _FEATURE_GROUPS,
        "targets": root / _TARGETS,
        "student_state": root / _STUDENT_STATE,
        "teacher_state": root / _TEACHER_STATE,
        "density_table": root / _DENSITY_TABLE,
        "context_mean": root / _CONTEXT_MEAN,
        "context_scale": root / _CONTEXT_SCALE,
    }


def _canonical_payload(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON constant {value!r}.")
        ),
    )
    if not isinstance(payload, Mapping) or canonical_json_text(payload) != text:
        raise ValueError(f"{path.name} is not canonical JSON.")
    return payload


def _state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{path.name} does not contain a state dictionary.")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    ):
        raise ValueError(f"{path.name} contains an unsafe state dictionary.")
    return MappingProxyType(
        {
            key: value.detach().to(device="cpu").contiguous()
            for key, value in payload.items()
        }
    )


def _npy_array(path: Path, *, shape: tuple[int, ...]) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.dtype("<f4") or value.shape != shape:
        raise ValueError(f"{path.name} has an invalid dtype or shape.")
    result = np.ascontiguousarray(value, dtype="<f4")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{path.name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _save_npy(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value, dtype="<f4"), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_cpu_density_table(
    student_state: Mapping[str, torch.Tensor],
    *,
    config: ImageGICOBackboneContextModelConfig,
    normalized_context_table: np.ndarray,
) -> tuple[ImageGICOBackboneContextDensityModel, np.ndarray]:
    student = ImageGICOBackboneContextDensityModel(
        config,
        normalized_context_table,
    )
    student.load_state_dict(student_state, strict=True)
    student.eval()
    student.requires_grad_(False)
    with torch.inference_mode():
        table = student.canonical_density_table()
    return student, table.numpy().astype("<f4", copy=False)


def _manifest_fields() -> set[str]:
    return {
        "artifact",
        "protocol",
        "artifact_sha256",
        "training",
        "feature_group_sha256",
        "target_sha256",
        "context_binding",
        "context_binding_sha256",
        "backbone_model_key",
        "backbone_protocol_sha256",
        "backbone_checkpoint_sha256",
        "feature_protocol_sha256",
        "class_nfe_contract",
        "schedule_support",
        "student_state_sha256",
        "teacher_state_sha256",
        "density_table_sha256",
        "portable_execution",
        "files",
    }


def _validate_bundle(
    paths: Mapping[str, Path],
    expected_paths: Mapping[str, Path],
) -> None:
    manifest = _canonical_payload(paths["manifest"])
    if set(manifest) != _manifest_fields():
        raise ValueError(
            "Backbone-context bundle manifest fields must be exactly "
            f"{sorted(_manifest_fields())}."
        )
    if (
        manifest.get("artifact") != "image_gico_backbone_context_policy"
        or manifest.get("protocol") != IMAGE_GICO_CONDITIONAL_BUNDLE_PROTOCOL
    ):
        raise ValueError("Unsupported backbone-context GICO policy bundle.")
    manifest_body = dict(manifest)
    stored_artifact_sha256 = manifest_body.pop("artifact_sha256")
    observed_artifact_sha256 = semantic_sha256(
        manifest_body,
        namespace="image-gico-backbone-context-artifact-v4",
    )
    if stored_artifact_sha256 != observed_artifact_sha256:
        raise ValueError("Backbone-context GICO artifact hash is inconsistent.")

    feature_groups = ImageGICOFeatureGroups.from_payload(
        _canonical_payload(paths["feature_groups"])
    )
    targets = ImageGICOConditionalTargets.from_payload(
        _canonical_payload(paths["targets"])
    )
    context_binding = ImageGICOBackboneContextBinding.from_payload(
        manifest["context_binding"]
    )
    training = manifest["training"]
    _training_result_identity(training)
    if not isinstance(training, Mapping):
        raise TypeError("Bundle training result must be a mapping.")
    if (
        manifest["context_binding_sha256"] != context_binding.binding_sha256
        or training.get("context_binding_sha256") != context_binding.binding_sha256
    ):
        raise ValueError("Bundle context bindings are inconsistent.")
    if (
        manifest["feature_group_sha256"] != feature_groups.sha256
        or manifest["target_sha256"] != targets.sha256
        or targets.feature_group_sha256 != feature_groups.sha256
        or targets.feature_protocol_sha256
        != feature_groups.feature_protocol_sha256
        or training.get("feature_group_sha256") != feature_groups.sha256
        or training.get("target_sha256") != targets.sha256
    ):
        raise ValueError("Bundle reward bindings are inconsistent.")
    expected_external = (
        targets.backbone_model_key,
        targets.backbone_protocol_sha256,
        targets.backbone_checkpoint_sha256,
        targets.feature_protocol_sha256,
    )
    observed_external = (
        manifest["backbone_model_key"],
        manifest["backbone_protocol_sha256"],
        manifest["backbone_checkpoint_sha256"],
        manifest["feature_protocol_sha256"],
    )
    context_external = (
        context_binding.backbone_model_key,
        context_binding.backbone_protocol_sha256,
        context_binding.backbone_checkpoint_sha256,
        targets.feature_protocol_sha256,
    )
    if observed_external != expected_external or context_external != expected_external:
        raise ValueError("Bundle backbone, context, and metric bindings disagree.")

    expected_contract = {
        "conditioning": "normalized_frozen_backbone_map_label_plus_target_nfe",
        "class_count": 1_000,
        "context_dim": 768,
        "target_nfes": [2, 4, 8],
        "density_bin_count": targets.density_bin_count,
        "density_table_shape": [3, 1_000, targets.density_bin_count],
        "feature_groups_are_inference_inputs": False,
        "initial_noise_is_an_inference_input": False,
        "raw_context_table_is_portable": False,
    }
    if manifest["class_nfe_contract"] != expected_contract:
        raise ValueError("Bundle class/context/NFE contract is inconsistent.")
    expected_support = {
        "schedule_keys": list(targets.schedule_keys),
        "schedule_sha256s": list(targets.schedule_sha256s),
        "density_mass_sha256s": [
            list(row) for row in targets.density_mass_sha256s
        ],
        "fixed_support_sha256": targets.fixed_support_sha256,
        "reward_evidence_sha256": targets.reward_evidence_sha256,
    }
    if manifest["schedule_support"] != expected_support:
        raise ValueError("Bundle schedule support is inconsistent.")
    if manifest["portable_execution"] != {
        "load_phase": "metadata_and_weights_only",
        "bind_phase": "regenerate_context_from_verified_backbone",
        "requires_explicit_backbone_binding": True,
        "density_table_verified_at_bind": True,
    }:
        raise ValueError("Bundle portable-execution contract is inconsistent.")

    expected_roles = {
        "feature_groups",
        "targets",
        "student_state",
        "teacher_state",
        "density_table",
        "context_mean",
        "context_scale",
    }
    files = manifest["files"]
    if not isinstance(files, Mapping) or set(files) != expected_roles:
        raise ValueError("Bundle file bindings are incomplete.")
    for role in expected_roles:
        binding = files[role]
        path = paths[role]
        if not isinstance(binding, Mapping):
            raise ValueError("Bundle file binding must be a mapping.")
        if (
            binding.get("filename") != expected_paths[role].name
            or binding.get("sha256") != file_sha256(path)
            or binding.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"Bundle file {role!r} is inconsistent.")

    mean = _npy_array(paths["context_mean"], shape=(768,))
    scale = _npy_array(paths["context_scale"], shape=(768,))
    normalizer = ImageGICOContextNormalizer(mean=mean, scale=scale)
    if (
        normalizer.mean_sha256 != context_binding.normalizer_mean_sha256
        or normalizer.scale_sha256 != context_binding.normalizer_scale_sha256
        or normalizer.normalizer_sha256 != context_binding.normalizer_sha256
    ):
        raise ValueError("Portable context normalizer does not match its binding.")

    config = ImageGICOBackboneContextModelConfig(
        density_bin_count=targets.density_bin_count
    )
    if training.get("model_config") != config.as_payload():
        raise ValueError("Bundle model config is inconsistent.")
    dummy_contexts = np.zeros((1_000, 768), dtype=np.float32)
    student = ImageGICOBackboneContextDensityModel(config, dummy_contexts)
    student.load_state_dict(_state_dict(paths["student_state"]), strict=True)
    observed_student_hash = conditional_module_state_sha256(
        student,
        namespace="image-gico-backbone-context-model-state-v4",
    )
    if (
        manifest["student_state_sha256"] != observed_student_hash
        or training.get("model_state_sha256") != observed_student_hash
    ):
        raise ValueError("Backbone-context student state hash is inconsistent.")
    teacher = ImageGICOBackboneContextTeacher(
        density_bin_count=targets.density_bin_count
    )
    teacher.load_state_dict(_state_dict(paths["teacher_state"]), strict=True)
    observed_teacher_hash = conditional_module_state_sha256(
        teacher,
        namespace="image-gico-backbone-context-teacher-state-v4",
    )
    if (
        manifest["teacher_state_sha256"] != observed_teacher_hash
        or training.get("teacher_state_sha256") != observed_teacher_hash
    ):
        raise ValueError("Backbone-context teacher state hash is inconsistent.")
    table = _npy_array(
        paths["density_table"],
        shape=(3, 1_000, targets.density_bin_count),
    )
    if (
        np.any(table < 0.0)
        or not np.allclose(table.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5)
        or manifest["density_table_sha256"] != file_sha256(paths["density_table"])
    ):
        raise ValueError("Portable density-table binding is inconsistent.")


@dataclass(frozen=True)
class BoundImageGICOConditionalArtifact:
    policy: ImageGICOBackboneContextSchedulePolicy
    prepared_context: PreparedImageGICOBackboneContext
    feature_groups: ImageGICOFeatureGroups
    targets: ImageGICOConditionalTargets
    manifest: Mapping[str, Any]

    @property
    def artifact_sha256(self) -> str:
        value = self.manifest.get("artifact_sha256")
        if not isinstance(value, str) or not value:
            raise ValueError("Backbone-context artifact hash is absent.")
        return value

    @property
    def normalized_context_table(self) -> torch.Tensor:
        return self.policy.model.canonical_context_table

    def contexts_for_class_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if not isinstance(labels, torch.Tensor):
            raise TypeError("Class labels must be a torch.Tensor.")
        if labels.ndim != 1 or labels.numel() <= 0:
            raise ValueError("Class labels must have shape [batch].")
        if labels.dtype not in _INTEGER_DTYPES:
            raise TypeError("Class labels must use an integer dtype.")
        table = self.normalized_context_table
        indices = labels.to(device=table.device, dtype=torch.int64)
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= 1_000)):
            raise ValueError("Class labels must be in [0, 1000).")
        contexts = table[indices]
        return contexts.to(device=labels.device, dtype=torch.float32).contiguous()


@dataclass(frozen=True)
class LoadedImageGICOConditionalArtifact:
    feature_groups: ImageGICOFeatureGroups
    targets: ImageGICOConditionalTargets
    context_binding: ImageGICOBackboneContextBinding
    context_normalizer: ImageGICOContextNormalizer
    manifest: Mapping[str, Any]
    _student_state: Mapping[str, torch.Tensor]
    _density_table: np.ndarray

    @property
    def artifact_sha256(self) -> str:
        value = self.manifest.get("artifact_sha256")
        if not isinstance(value, str) or not value:
            raise ValueError("Backbone-context artifact hash is absent.")
        return value

    def bind(
        self,
        backbone: CanonicalNoiseToDataAdapter,
    ) -> BoundImageGICOConditionalArtifact:
        prepared = bind_image_gico_backbone_context(
            self.context_binding,
            self.context_normalizer,
            backbone,
        )
        config = ImageGICOBackboneContextModelConfig(
            density_bin_count=self.targets.density_bin_count
        )
        model, expected_table = _canonical_cpu_density_table(
            self._student_state,
            config=config,
            normalized_context_table=prepared.normalized_context_table,
        )
        if not np.array_equal(expected_table, self._density_table):
            raise ValueError(
                "Portable density table does not match the verified backbone context."
            )
        return BoundImageGICOConditionalArtifact(
            policy=ImageGICOBackboneContextSchedulePolicy(model, self.targets),
            prepared_context=prepared,
            feature_groups=self.feature_groups,
            targets=self.targets,
            manifest=self.manifest,
        )


def save_image_gico_conditional_artifact(
    output_dir: str | Path,
    result: ImageGICOBackboneContextTrainingResult,
    feature_groups: ImageGICOFeatureGroups,
    targets: ImageGICOConditionalTargets,
    prepared_context: PreparedImageGICOBackboneContext,
    *,
    overwrite: bool = False,
) -> Mapping[str, Path]:
    if not isinstance(result, ImageGICOBackboneContextTrainingResult):
        raise TypeError("result must be an ImageGICOBackboneContextTrainingResult.")
    if not isinstance(prepared_context, PreparedImageGICOBackboneContext):
        raise TypeError("prepared_context must be a PreparedImageGICOBackboneContext.")
    if (
        result.feature_group_sha256 != feature_groups.sha256
        or result.target_sha256 != targets.sha256
        or targets.feature_group_sha256 != feature_groups.sha256
        or targets.feature_protocol_sha256
        != feature_groups.feature_protocol_sha256
        or result.context_binding_sha256 != prepared_context.binding.binding_sha256
    ):
        raise ValueError("Training result does not match rewards, targets, and context.")
    expected_external = (
        targets.backbone_model_key,
        targets.backbone_protocol_sha256,
        targets.backbone_checkpoint_sha256,
    )
    context_external = (
        prepared_context.binding.backbone_model_key,
        prepared_context.binding.backbone_protocol_sha256,
        prepared_context.binding.backbone_checkpoint_sha256,
    )
    if context_external != expected_external:
        raise ValueError("Prepared context does not match target backbone bindings.")

    paths = _paths(output_dir)
    anchor = paths[sorted(paths)[0]]
    preflight_artifact_bundle(
        anchor,
        paths,
        overwrite=bool(overwrite),
        validator=_validate_bundle,
    )
    staged = {role: temporary_bundle_path(path) for role, path in paths.items()}
    staged_hashes: dict[str, str] = {}
    try:
        staged["feature_groups"].write_bytes(
            canonical_json_bytes(feature_groups.as_payload())
        )
        staged["targets"].write_bytes(canonical_json_bytes(targets.as_payload()))
        student_state = {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in result.model.state_dict().items()
        }
        teacher_state = {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in result.teacher.state_dict().items()
        }
        torch.save(student_state, staged["student_state"])
        torch.save(teacher_state, staged["teacher_state"])
        config = ImageGICOBackboneContextModelConfig(
            density_bin_count=targets.density_bin_count
        )
        _, density_table = _canonical_cpu_density_table(
            student_state,
            config=config,
            normalized_context_table=prepared_context.normalized_context_table,
        )
        _save_npy(staged["density_table"], density_table)
        _save_npy(staged["context_mean"], prepared_context.normalizer.mean)
        _save_npy(staged["context_scale"], prepared_context.normalizer.scale)

        data_roles = (
            "feature_groups",
            "targets",
            "student_state",
            "teacher_state",
            "density_table",
            "context_mean",
            "context_scale",
        )
        for role in data_roles:
            staged_hashes[role] = file_sha256(staged[role])
        training = result.manifest_payload()
        binding = prepared_context.binding
        manifest_body = {
            "artifact": "image_gico_backbone_context_policy",
            "protocol": IMAGE_GICO_CONDITIONAL_BUNDLE_PROTOCOL,
            "training": training,
            "feature_group_sha256": feature_groups.sha256,
            "target_sha256": targets.sha256,
            "context_binding": binding.as_payload(),
            "context_binding_sha256": binding.binding_sha256,
            "backbone_model_key": targets.backbone_model_key,
            "backbone_protocol_sha256": targets.backbone_protocol_sha256,
            "backbone_checkpoint_sha256": targets.backbone_checkpoint_sha256,
            "feature_protocol_sha256": targets.feature_protocol_sha256,
            "class_nfe_contract": {
                "conditioning": "normalized_frozen_backbone_map_label_plus_target_nfe",
                "class_count": 1_000,
                "context_dim": 768,
                "target_nfes": [2, 4, 8],
                "density_bin_count": targets.density_bin_count,
                "density_table_shape": [3, 1_000, targets.density_bin_count],
                "feature_groups_are_inference_inputs": False,
                "initial_noise_is_an_inference_input": False,
                "raw_context_table_is_portable": False,
            },
            "schedule_support": {
                "schedule_keys": list(targets.schedule_keys),
                "schedule_sha256s": list(targets.schedule_sha256s),
                "density_mass_sha256s": [
                    list(row) for row in targets.density_mass_sha256s
                ],
                "fixed_support_sha256": targets.fixed_support_sha256,
                "reward_evidence_sha256": targets.reward_evidence_sha256,
            },
            "student_state_sha256": result.model_state_sha256,
            "teacher_state_sha256": result.teacher_state_sha256,
            "density_table_sha256": staged_hashes["density_table"],
            "portable_execution": {
                "load_phase": "metadata_and_weights_only",
                "bind_phase": "regenerate_context_from_verified_backbone",
                "requires_explicit_backbone_binding": True,
                "density_table_verified_at_bind": True,
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
                namespace="image-gico-backbone-context-artifact-v4",
            ),
        }
        staged["manifest"].write_bytes(canonical_json_bytes(manifest))
        staged_hashes["manifest"] = file_sha256(staged["manifest"])
        promote_artifact_bundle(
            anchor,
            paths,
            staged,
            overwrite=bool(overwrite),
            validator=_validate_bundle,
        )
        return paths
    finally:
        for role, path in staged.items():
            discard_temporary_bundle_path(
                path,
                paths[role],
                expected_sha256=staged_hashes.get(role),
            )


def load_image_gico_conditional_artifact(
    output_dir: str | Path,
) -> LoadedImageGICOConditionalArtifact:
    paths = _paths(output_dir)
    validate_artifact_bundle(
        paths[sorted(paths)[0]],
        paths,
        validator=_validate_bundle,
    )
    manifest = _canonical_payload(paths["manifest"])
    feature_groups = ImageGICOFeatureGroups.from_payload(
        _canonical_payload(paths["feature_groups"])
    )
    targets = ImageGICOConditionalTargets.from_payload(
        _canonical_payload(paths["targets"])
    )
    context_binding = ImageGICOBackboneContextBinding.from_payload(
        manifest["context_binding"]
    )
    normalizer = ImageGICOContextNormalizer(
        mean=_npy_array(paths["context_mean"], shape=(768,)),
        scale=_npy_array(paths["context_scale"], shape=(768,)),
    )
    density_table = _npy_array(
        paths["density_table"],
        shape=(3, 1_000, targets.density_bin_count),
    )
    return LoadedImageGICOConditionalArtifact(
        feature_groups=feature_groups,
        targets=targets,
        context_binding=context_binding,
        context_normalizer=normalizer,
        manifest=manifest,
        _student_state=_state_dict(paths["student_state"]),
        _density_table=density_table,
    )


__all__ = [
    "IMAGE_GICO_CONDITIONAL_BUNDLE_PROTOCOL",
    "BoundImageGICOConditionalArtifact",
    "LoadedImageGICOConditionalArtifact",
    "load_image_gico_conditional_artifact",
    "save_image_gico_conditional_artifact",
]
