"""Portable, source-bound artifacts for the causal-AR image GICO student."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import Tensor

from genode.artifacts.identity import semantic_sha256
from genode.gico.image_artifact_io import (
    load_float64_npy,
    staged_image_gico_directory,
    validate_image_gico_directory,
    write_float64_npy,
)
from genode.gico.image_causal_policy import (
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    ImageGICOCausalConfig,
    ImageGICOCausalPolicy,
    ImageGICOCausalTransformer,
)
from genode.gico.image_causal_stick import ImageGICOCausalPathBank
from genode.gico.image_causal_training import (
    CAUSAL_TRAINING_REPORT_PROTOCOL,
    IMAGE_GICO_CAUSAL_STATE_NAMESPACE,
    ImageGICOCausalTrainingConfig,
    ImageGICOCausalTrainingResult,
    image_gico_causal_state_sha256,
)
from genode.gico.image_supervision import (
    ImageGICOSupervision,
    image_gico_supervision_array_sha256,
)
from genode.provenance import file_sha256

IMAGE_GICO_CAUSAL_ARTIFACT_PROTOCOL = "image_gico_causal_ar_artifact_v2"
IMAGE_GICO_CAUSAL_ARTIFACT_NAMESPACE = "image-gico-causal-ar-artifact-v2"
_ARTIFACT_FILES = {
    "manifest.json",
    "causal-state.pt",
    "fixed-density-mass.npy",
    "normalized-contexts.npy",
    "reference-time-grid.npy",
}


def _array_content_sha256(value: np.ndarray, *, field: str) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    return semantic_sha256(
        {
            "field": field,
            "dtype": "float64-le",
            "shape": list(array.shape),
            "content_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        },
        namespace="image-gico-causal-artifact-array-v1",
    )


def _state_snapshot(model: ImageGICOCausalTransformer) -> dict[str, Tensor]:
    return {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in sorted(model.state_dict().items())
    }


def _validate_training_report(
    value: object,
    *,
    supervision_sha256: str,
    model_state_sha256: str,
    path_bank: ImageGICOCausalPathBank,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Causal training report must be a mapping.")
    expected_fields = {
        "protocol",
        "config",
        "model_config",
        "supervision_sha256",
        "model_state_sha256",
        "trainable_parameter_count",
        "initial_batch_nll",
        "final_batch_nll",
        "final_preclip_gradient_norm",
        "sampled_cell_stream_sha256",
        "alias_support_sizes",
        "completed_updates",
        "published_checkpoint",
    }
    if set(value) != expected_fields:
        raise ValueError("Causal training report fields are incomplete or unexpected.")
    config = value.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Causal training configuration must be a mapping.")
    try:
        parsed = ImageGICOCausalTrainingConfig(
            updates=config["updates"],
            batch_size=config["batch_size_cells"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            seed=config["seed"],
            gradient_clip_norm=config["gradient_clip_norm"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Causal training configuration is invalid.") from exc
    stream_identity = value.get("sampled_cell_stream_sha256")
    if not isinstance(stream_identity, str) or len(stream_identity) != 64:
        raise ValueError("Causal training sample stream identity is invalid.")
    try:
        int(stream_identity, 16)
    except ValueError as exc:
        raise ValueError("Causal training sample stream identity is invalid.") from exc
    metrics = (
        value.get("initial_batch_nll"),
        value.get("final_batch_nll"),
        value.get("final_preclip_gradient_norm"),
    )
    expected_support_sizes = [len(paths) for paths in path_bank.unique_token_paths_by_nfe]
    if (
        dict(config) != parsed.as_payload()
        or value.get("protocol") != CAUSAL_TRAINING_REPORT_PROTOCOL
        or value.get("model_config") != ImageGICOCausalConfig().as_payload()
        or value.get("supervision_sha256") != supervision_sha256
        or value.get("model_state_sha256") != model_state_sha256
        or value.get("trainable_parameter_count") != EXPECTED_TRAINABLE_PARAMETER_COUNT
        or value.get("completed_updates") != parsed.updates
        or value.get("alias_support_sizes") != expected_support_sizes
        or value.get("published_checkpoint") != "final_only"
        or any(isinstance(metric, bool) or not isinstance(metric, (int, float)) for metric in metrics)
        or not bool(np.isfinite(np.asarray(metrics, dtype=np.float64)).all())
        or any(float(metric) < 0.0 for metric in metrics)
    ):
        raise ValueError("Causal training report contradicts the artifact contract.")


def save_image_gico_causal_artifact(
    result: ImageGICOCausalTrainingResult,
    supervision: ImageGICOSupervision,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    """Publish a final-only causal artifact without overwriting an existing path."""

    if not isinstance(result, ImageGICOCausalTrainingResult):
        raise TypeError("result must be ImageGICOCausalTrainingResult.")
    if not isinstance(supervision, ImageGICOSupervision):
        raise TypeError("supervision must be ImageGICOSupervision.")
    supervision.verify()
    if result.report.supervision_sha256 != supervision.sha256:
        raise ValueError("Training result and supervision identities disagree.")
    if not np.array_equal(result.path_bank.canonical_density_paths, supervision.fixed_density_mass):
        raise ValueError("Causal path-bank support differs from supervision.")
    if result.report.model_config != result.model.config.as_payload():
        raise ValueError("Causal training report and model configuration disagree.")
    if result.model.trainable_parameter_count != EXPECTED_TRAINABLE_PARAMETER_COUNT:
        raise ValueError("Causal model parameter count changed.")
    state = _state_snapshot(result.model)
    state_identity = image_gico_causal_state_sha256(state)
    if result.report.model_state_sha256 != state_identity:
        raise ValueError("Causal training report does not describe the current final model state.")
    _validate_training_report(
        result.report.as_payload(),
        supervision_sha256=supervision.sha256,
        model_state_sha256=state_identity,
        path_bank=result.path_bank,
    )
    with staged_image_gico_directory(
        output_dir,
        expected_members=_ARTIFACT_FILES,
        label="Image GICO causal artifact directory",
    ) as stage:
        state_path = stage / "causal-state.pt"
        torch.save(state, state_path)
        arrays = {
            "fixed_density_mass": write_float64_npy(
                stage / "fixed-density-mass.npy",
                result.path_bank.canonical_density_paths,
            ),
            "reference_time_grid": write_float64_npy(
                stage / "reference-time-grid.npy",
                result.path_bank.reference_time_grid,
            ),
            "normalized_contexts": write_float64_npy(
                stage / "normalized-contexts.npy",
                supervision.normalized_contexts,
            ),
        }
        body = {
            "protocol": IMAGE_GICO_CAUSAL_ARTIFACT_PROTOCOL,
            "student_kind": "stochastic_causal_ar",
            "supervision_kind": supervision.supervision_kind,
            "supervision_sha256": supervision.sha256,
            "context_binding_sha256": supervision.context_binding_sha256,
            "context_count": supervision.context_count,
            "model_config": result.model.config.as_payload(),
            "training_report": result.report.as_payload(),
            "path_bank": result.path_bank.as_metadata_payload(),
            "support_sha256": _array_content_sha256(
                result.path_bank.canonical_density_paths,
                field="fixed_density_mass",
            ),
            "reference_time_grid_sha256": _array_content_sha256(
                result.path_bank.reference_time_grid,
                field="reference_time_grid",
            ),
            "state_sha256": state_identity,
            "state_file_sha256": file_sha256(state_path),
            "array_files": arrays,
        }
        artifact_identity = semantic_sha256(body, namespace=IMAGE_GICO_CAUSAL_ARTIFACT_NAMESPACE)
        manifest = {
            "artifact": "image_gico_causal_ar_student",
            **body,
            "artifact_sha256": artifact_identity,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return MappingProxyType(manifest)


@dataclass(frozen=True, slots=True)
class LoadedImageGICOCausalArtifact:
    artifact_sha256: str
    supervision_sha256: str
    supervision_kind: str
    state_sha256: str
    model: ImageGICOCausalTransformer
    path_bank: ImageGICOCausalPathBank
    support_sha256: str
    reference_time_grid_sha256: str
    normalized_contexts: np.ndarray
    context_binding_sha256: str
    manifest: Mapping[str, Any]

    @property
    def policy(self) -> ImageGICOCausalPolicy:
        self.verify()
        return ImageGICOCausalPolicy(self.model, self.path_bank)

    def verify(self) -> None:
        self.model.verify_protocol_buffers()
        if image_gico_causal_state_sha256(self.model) != self.state_sha256:
            raise ValueError("Loaded causal model state was mutated.")
        if (
            _array_content_sha256(
                self.path_bank.canonical_density_paths,
                field="fixed_density_mass",
            )
            != self.support_sha256
            or _array_content_sha256(
                self.path_bank.reference_time_grid,
                field="reference_time_grid",
            )
            != self.reference_time_grid_sha256
            or self.path_bank.as_metadata_payload() != self.manifest.get("path_bank")
        ):
            raise ValueError("Loaded causal path bank was mutated.")
        if (
            image_gico_supervision_array_sha256(
                self.normalized_contexts,
                field="normalized_contexts",
            )
            != self.context_binding_sha256
        ):
            raise ValueError("Loaded causal deployment contexts were mutated.")
        body = dict(self.manifest)
        observed = body.pop("artifact_sha256", None)
        body.pop("artifact", None)
        if (
            observed != self.artifact_sha256
            or semantic_sha256(body, namespace=IMAGE_GICO_CAUSAL_ARTIFACT_NAMESPACE) != self.artifact_sha256
        ):
            raise ValueError("Loaded causal artifact manifest was mutated.")


def load_image_gico_causal_artifact(
    output_dir: str | Path,
    supervision: ImageGICOSupervision | None = None,
    *,
    device: torch.device | str = "cpu",
    expected_artifact_sha256: str | None = None,
) -> LoadedImageGICOCausalArtifact:
    """Strictly load state, support, lineage, and model configuration."""

    if supervision is not None and not isinstance(supervision, ImageGICOSupervision):
        raise TypeError("supervision must be ImageGICOSupervision or None.")
    if supervision is not None:
        supervision.verify()
    root = validate_image_gico_directory(
        output_dir,
        expected_members=_ARTIFACT_FILES,
        label="Image GICO causal artifact directory",
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest_fields = {
        "artifact",
        "protocol",
        "student_kind",
        "supervision_kind",
        "supervision_sha256",
        "context_binding_sha256",
        "context_count",
        "model_config",
        "training_report",
        "path_bank",
        "support_sha256",
        "reference_time_grid_sha256",
        "state_sha256",
        "state_file_sha256",
        "array_files",
        "artifact_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        raise ValueError("Causal manifest fields are incomplete or unexpected.")
    if (
        manifest.get("artifact") != "image_gico_causal_ar_student"
        or manifest.get("protocol") != IMAGE_GICO_CAUSAL_ARTIFACT_PROTOCOL
        or manifest.get("student_kind") != "stochastic_causal_ar"
        or manifest.get("supervision_kind") not in {"conditional_kid", "unconditional_mixture"}
        or manifest.get("model_config") != ImageGICOCausalConfig().as_payload()
    ):
        raise ValueError("Causal artifact contract or supervision binding changed.")
    body = dict(manifest)
    artifact_identity = body.pop("artifact_sha256", None)
    body.pop("artifact", None)
    expected_identity = semantic_sha256(body, namespace=IMAGE_GICO_CAUSAL_ARTIFACT_NAMESPACE)
    if artifact_identity != expected_identity:
        raise ValueError("Causal artifact identity is inconsistent.")
    if expected_artifact_sha256 is not None and artifact_identity != expected_artifact_sha256:
        raise ValueError("Causal artifact differs from the expected identity.")
    arrays = manifest.get("array_files")
    if not isinstance(arrays, Mapping) or set(arrays) != {
        "fixed_density_mass",
        "normalized_contexts",
        "reference_time_grid",
    }:
        raise ValueError("Causal support arrays are invalid.")
    support = load_float64_npy(root, arrays["fixed_density_mass"], field="fixed_density_mass")
    reference = load_float64_npy(root, arrays["reference_time_grid"], field="reference_time_grid")
    contexts = load_float64_npy(root, arrays["normalized_contexts"], field="normalized_contexts")
    if (
        contexts.ndim != 2
        or contexts.shape[0] < 1
        or contexts.shape[1] != 768
        or not np.array_equal(contexts, contexts.astype(np.float32).astype(np.float64))
        or manifest.get("context_count") != contexts.shape[0]
        or image_gico_supervision_array_sha256(contexts, field="normalized_contexts")
        != manifest.get("context_binding_sha256")
    ):
        raise ValueError("Causal deployment contexts changed.")
    if manifest["supervision_kind"] == "unconditional_mixture" and (
        contexts.shape[0] != 1 or bool(np.any(contexts != 0.0))
    ):
        raise ValueError("Unconditional causal deployment requires one explicit zero context.")
    if manifest["supervision_kind"] == "conditional_kid" and contexts.shape[0] != 1_000:
        raise ValueError("Conditional causal deployment requires its exact 1,000-row context table.")
    if supervision is not None and (
        manifest.get("supervision_sha256") != supervision.sha256
        or manifest.get("supervision_kind") != supervision.supervision_kind
        or manifest.get("context_binding_sha256") != supervision.context_binding_sha256
        or not np.array_equal(support, supervision.fixed_density_mass)
        or not np.array_equal(contexts, supervision.normalized_contexts)
    ):
        raise ValueError("Causal artifact and supervision differ.")
    support_identity = _array_content_sha256(support, field="fixed_density_mass")
    reference_identity = _array_content_sha256(reference, field="reference_time_grid")
    if support_identity != manifest.get("support_sha256") or reference_identity != manifest.get(
        "reference_time_grid_sha256"
    ):
        raise ValueError("Causal support content identity changed.")
    path_bank = ImageGICOCausalPathBank.build(support, reference)
    if path_bank.as_metadata_payload() != manifest.get("path_bank"):
        raise ValueError("Rebuilt causal path bank differs from the manifest.")
    _validate_training_report(
        manifest.get("training_report"),
        supervision_sha256=str(manifest["supervision_sha256"]),
        model_state_sha256=str(manifest["state_sha256"]),
        path_bank=path_bank,
    )
    state_path = root / "causal-state.pt"
    if file_sha256(state_path) != manifest.get("state_file_sha256"):
        raise ValueError("Causal state file hash changed.")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()
    ):
        raise ValueError("Causal state file is not a tensor mapping.")
    state_identity = image_gico_causal_state_sha256(state)
    if state_identity != manifest.get("state_sha256"):
        raise ValueError("Causal semantic state identity changed.")
    with torch.random.fork_rng(devices=[]):
        model = ImageGICOCausalTransformer(ImageGICOCausalConfig())
    model.load_state_dict(dict(state), strict=True)
    model.to(torch.device(device)).eval().requires_grad_(False)
    loaded = LoadedImageGICOCausalArtifact(
        artifact_sha256=artifact_identity,
        supervision_sha256=str(manifest["supervision_sha256"]),
        supervision_kind=str(manifest["supervision_kind"]),
        state_sha256=state_identity,
        model=model,
        path_bank=path_bank,
        support_sha256=support_identity,
        reference_time_grid_sha256=reference_identity,
        normalized_contexts=contexts,
        context_binding_sha256=str(manifest["context_binding_sha256"]),
        manifest=MappingProxyType(manifest),
    )
    loaded.verify()
    return loaded


__all__ = [
    "IMAGE_GICO_CAUSAL_ARTIFACT_PROTOCOL",
    "IMAGE_GICO_CAUSAL_STATE_NAMESPACE",
    "LoadedImageGICOCausalArtifact",
    "image_gico_causal_state_sha256",
    "load_image_gico_causal_artifact",
    "save_image_gico_causal_artifact",
]
