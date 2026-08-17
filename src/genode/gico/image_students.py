"""Unified deterministic and stochastic image GICO student runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from genode.artifacts.identity import semantic_sha256
from genode.gico.image_causal_artifacts import LoadedImageGICOCausalArtifact
from genode.gico.image_causal_policy import FrozenImageGICOCausalRealization
from genode.gico.image_causal_rng import image_gico_causal_uniforms_sha256
from genode.gico.image_causal_stick import (
    DENSITY_BIN_COUNT,
    inverse_cdf_clock_nodes,
)
from genode.gico.image_conditional import (
    IMAGE_GICO_CONDITIONAL_HIDDEN_DIM,
    ImageGICOBackboneContextDensityModel,
    ImageGICOBackboneContextModelConfig,
    ImageGICOConditionalTargets,
)
from genode.gico.image_conditional_training import (
    IMAGE_GICO_BACKBONE_CONTEXT_TEACHER_PROTOCOL,
    IMAGE_GICO_BACKBONE_CONTEXT_TRAINING_PROTOCOL,
    IMAGE_GICO_DENSITY_SUMMARY_PROTOCOL,
    ImageGICOBackboneContextTrainingConfig,
    conditional_module_state_sha256,
    train_image_gico_backbone_context,
)
from genode.gico.image_supervision import (
    ImageGICOStudentKind,
    ImageGICOSupervision,
    image_gico_supervision_array_sha256,
    normalize_image_gico_student_kind,
)
from genode.path_safety import is_link_or_reparse_point
from genode.provenance import file_sha256
from genode.solvers.euler import integrate_euler
from genode.solvers.protocol import SolverResult

IMAGE_GICO_DETERMINISTIC_ARTIFACT_PROTOCOL = "image_gico_deterministic_barycenter_artifact_v1"
IMAGE_GICO_DETERMINISTIC_ARTIFACT_NAMESPACE = "image-gico-deterministic-barycenter-artifact-v1"
IMAGE_GICO_DETERMINISTIC_STATE_NAMESPACE = "image-gico-deterministic-barycenter-state-v1"


def _state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(IMAGE_GICO_DETERMINISTIC_STATE_NAMESPACE.encode("ascii"))
    digest.update(b"\0")
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise TypeError("Model state must map names to tensors.")
        value = tensor.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Model state tensor {name!r} is non-finite.")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        for dimension in value.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(value.numpy().tobytes(order="C"))
    return f"{IMAGE_GICO_DETERMINISTIC_STATE_NAMESPACE}:{digest.hexdigest()}"


def _materialization_array_sha256(value: np.ndarray, *, field: str) -> str:
    array = np.ascontiguousarray(value)
    return semantic_sha256(
        {
            "field": field,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "content_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        },
        namespace="image-gico-materialized-array-v1",
    )


def _validate_deterministic_training_payload(
    value: object,
    *,
    supervision_kind: str,
    context_binding_sha256: str,
    target_sha256: str | None,
    feature_group_sha256: str | None,
    schedule_keys: tuple[str, ...],
    model: ImageGICOBackboneContextDensityModel | None,
) -> None:
    if supervision_kind == "unconditional_mixture":
        if (
            value
            != {
                "protocol": "image_gico_direct_barycenter_v1",
                "conditioning": "unconditional_single_zero_context",
                "trained_parameters": 0,
            }
            or model is not None
        ):
            raise ValueError("Unconditional deterministic training lineage is invalid.")
        if target_sha256 is not None or feature_group_sha256 is not None:
            raise ValueError("Unconditional deterministic artifacts cannot claim conditional lineage.")
        return
    if not isinstance(value, Mapping) or model is None:
        raise ValueError("Conditional deterministic training lineage is missing.")
    expected_fields = {
        "protocol",
        "conditioning",
        "context_binding_sha256",
        "feature_group_usage",
        "target_sha256",
        "feature_group_sha256",
        "training_config",
        "training_config_sha256",
        "model_config",
        "model_config_sha256",
        "model_state_sha256",
        "teacher_protocol",
        "teacher_density_summary_protocol",
        "teacher_state_sha256",
        "teacher_evidence_row_count",
        "teacher_evidence_sha256",
        "final_kl",
        "final_residual_penalty",
        "final_teacher_score",
        "final_objective",
        "conditional_density_range",
        "teacher_schedule_fold_diagnostics",
        "teacher_oof_rmse",
        "teacher_oof_pairwise_accuracy",
        "result_sha256",
    }
    if set(value) != expected_fields:
        raise ValueError("Conditional deterministic training fields are incomplete or unexpected.")
    body = dict(value)
    result_identity = body.pop("result_sha256")
    if result_identity != semantic_sha256(
        body,
        namespace="image-gico-backbone-context-training-result-v4",
    ):
        raise ValueError("Conditional deterministic training identity is inconsistent.")
    config_payload = value["training_config"]
    if not isinstance(config_payload, Mapping):
        raise ValueError("Conditional deterministic training configuration is invalid.")
    try:
        config = ImageGICOBackboneContextTrainingConfig(
            student_steps=config_payload["student_steps"],
            teacher_steps=config_payload["teacher_steps"],
            teacher_batch_size=config_payload["teacher_batch_size"],
            student_learning_rate=config_payload["student_learning_rate"],
            teacher_learning_rate=config_payload["teacher_learning_rate"],
            weight_decay=config_payload["weight_decay"],
            residual_penalty_weight=config_payload["residual_penalty_weight"],
            seed=config_payload["seed"],
            teacher_score_weight=config_payload["teacher_score_weight"],
            teacher_score_warmup_fraction=config_payload["teacher_score_warmup_fraction"],
            teacher_score_clip=config_payload["teacher_score_clip"],
            teacher_rank_temperature=config_payload["teacher_rank_temperature"],
            teacher_regression_weight=config_payload["teacher_regression_weight"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Conditional deterministic training configuration is invalid.") from exc
    diagnostics = value["teacher_schedule_fold_diagnostics"]
    if not isinstance(diagnostics, list) or len(diagnostics) != 23 or len(schedule_keys) != 23:
        raise ValueError("Conditional deterministic teacher diagnostics are missing.")
    diagnostic_fields = {
        "fold",
        "heldout_schedule_key",
        "heldout_row_count",
        "root_mean_squared_error",
        "mean_absolute_error",
    }
    for fold, diagnostic in enumerate(diagnostics):
        metrics = (
            diagnostic.get("root_mean_squared_error") if isinstance(diagnostic, Mapping) else None,
            diagnostic.get("mean_absolute_error") if isinstance(diagnostic, Mapping) else None,
        )
        if (
            not isinstance(diagnostic, Mapping)
            or set(diagnostic) != diagnostic_fields
            or diagnostic.get("fold") != fold
            or not isinstance(diagnostic.get("heldout_schedule_key"), str)
            or not diagnostic["heldout_schedule_key"]
            or diagnostic["heldout_schedule_key"] != schedule_keys[fold]
            or diagnostic.get("heldout_row_count") != 3_000
            or any(isinstance(metric, bool) or not isinstance(metric, (int, float)) for metric in metrics)
            or not bool(np.isfinite(np.asarray(metrics, dtype=np.float64)).all())
            or any(float(metric) < 0.0 for metric in metrics)
        ):
            raise ValueError("Conditional deterministic teacher diagnostics are invalid.")
    evidence_row_count = 3_000 * len(diagnostics)
    expected_evidence = semantic_sha256(
        {
            "target_sha256": target_sha256,
            "row_count": evidence_row_count,
            "coverage": f"all_1000_classes_x_{len(diagnostics)}_schedules_x_3_nfes",
        },
        namespace="image-gico-backbone-context-teacher-evidence-v4",
    )
    metrics = tuple(
        value[field]
        for field in (
            "final_kl",
            "final_residual_penalty",
            "final_teacher_score",
            "final_objective",
            "conditional_density_range",
            "teacher_oof_rmse",
            "teacher_oof_pairwise_accuracy",
        )
    )
    model_state_identity = conditional_module_state_sha256(
        model,
        namespace="image-gico-backbone-context-model-state-v4",
    )
    if (
        dict(config_payload) != config.as_payload()
        or value["training_config_sha256"] != config.sha256
        or value["protocol"] != IMAGE_GICO_BACKBONE_CONTEXT_TRAINING_PROTOCOL
        or value["conditioning"] != "normalized_frozen_backbone_map_label_plus_target_nfe"
        or value["feature_group_usage"] != "reward_shrinkage_only_not_inference_context"
        or value["context_binding_sha256"] != context_binding_sha256
        or value["target_sha256"] != target_sha256
        or value["feature_group_sha256"] != feature_group_sha256
        or value["model_config"] != model.config.as_payload()
        or value["model_config_sha256"] != model.config.sha256
        or value["model_state_sha256"] != model_state_identity
        or value["teacher_protocol"] != IMAGE_GICO_BACKBONE_CONTEXT_TEACHER_PROTOCOL
        or value["teacher_density_summary_protocol"] != IMAGE_GICO_DENSITY_SUMMARY_PROTOCOL
        or not isinstance(value["teacher_state_sha256"], str)
        or not value["teacher_state_sha256"]
        or value["teacher_evidence_row_count"] != evidence_row_count
        or value["teacher_evidence_sha256"] != expected_evidence
        or any(isinstance(metric, bool) or not isinstance(metric, (int, float)) for metric in metrics)
        or not bool(np.isfinite(np.asarray(metrics, dtype=np.float64)).all())
        or float(value["final_kl"]) < 0.0
        or float(value["final_residual_penalty"]) < 0.0
        or float(value["conditional_density_range"]) <= 0.0
        or float(value["teacher_oof_rmse"]) < 0.0
        or not 0.0 <= float(value["teacher_oof_pairwise_accuracy"]) <= 1.0
    ):
        raise ValueError("Conditional deterministic training lineage contradicts the artifact.")


@dataclass(frozen=True, slots=True)
class ImageGICODeterministicTrainingResult:
    supervision_sha256: str
    model: ImageGICOBackboneContextDensityModel | None
    training_payload: Mapping[str, Any]


def train_image_gico_deterministic_student(
    supervision: ImageGICOSupervision,
    *,
    device: torch.device | str = "cpu",
    config: ImageGICOBackboneContextTrainingConfig | None = None,
) -> ImageGICODeterministicTrainingResult:
    """Train the conditional barycenter model or bind an unconditional target."""

    if not isinstance(supervision, ImageGICOSupervision):
        raise TypeError("supervision must be ImageGICOSupervision.")
    if supervision.supervision_kind == "unconditional_mixture":
        if config is not None:
            raise ValueError("Unconditional barycenters do not train a context model.")
        return ImageGICODeterministicTrainingResult(
            supervision_sha256=supervision.sha256,
            model=None,
            training_payload=MappingProxyType(
                {
                    "protocol": "image_gico_direct_barycenter_v1",
                    "conditioning": "unconditional_single_zero_context",
                    "trained_parameters": 0,
                }
            ),
        )
    payload = supervision.conditional_target_payload
    if payload is None:
        raise ValueError("Conditional supervision omitted its target payload.")
    targets = ImageGICOConditionalTargets.from_payload(payload)
    execution_device = torch.device(device)
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        result = train_image_gico_backbone_context(
            targets,
            fixed_density_mass=np.ascontiguousarray(supervision.fixed_density_mass, dtype=np.float64),
            normalized_context_table=np.ascontiguousarray(supervision.normalized_contexts, dtype=np.float32),
            context_binding_sha256=supervision.context_binding_sha256,
            config=config,
            device=execution_device,
        )
    if result.model.config.hidden_dim != IMAGE_GICO_CONDITIONAL_HIDDEN_DIM:
        raise RuntimeError("Deterministic student hidden dimension changed.")
    return ImageGICODeterministicTrainingResult(
        supervision_sha256=supervision.sha256,
        model=result.model,
        training_payload=MappingProxyType(result.manifest_payload()),
    )


def _write_deployment_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    np.save(path, np.ascontiguousarray(value, dtype="<f8"), allow_pickle=False)
    return {
        "file": path.name,
        "sha256": file_sha256(path),
        "shape": list(value.shape),
        "dtype": "float64-le",
    }


def _load_deployment_array(root: Path, descriptor: object, *, field: str) -> np.ndarray:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "file",
        "sha256",
        "shape",
        "dtype",
    }:
        raise ValueError(f"Invalid {field} descriptor.")
    filename = descriptor["file"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"Invalid {field} filename.")
    path = root / filename
    if is_link_or_reparse_point(path) or not path.is_file() or file_sha256(path) != descriptor["sha256"]:
        raise ValueError(f"{field} deployment array changed.")
    value = np.load(path, allow_pickle=False)
    result = np.ascontiguousarray(value, dtype=np.float64)
    if (
        list(result.shape) != descriptor["shape"]
        or descriptor["dtype"] != "float64-le"
        or not bool(np.isfinite(result).all())
    ):
        raise ValueError(f"{field} deployment metadata changed.")
    result.setflags(write=False)
    return result


def save_image_gico_deterministic_artifact(
    result: ImageGICODeterministicTrainingResult,
    supervision: ImageGICOSupervision,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    if not isinstance(result, ImageGICODeterministicTrainingResult):
        raise TypeError("result must be ImageGICODeterministicTrainingResult.")
    if not isinstance(supervision, ImageGICOSupervision):
        raise TypeError("supervision must be ImageGICOSupervision.")
    if result.supervision_sha256 != supervision.sha256:
        raise ValueError("Training result and supervision differ.")
    if supervision.supervision_kind == "conditional_kid":
        if result.model is None or not np.array_equal(
            result.model.canonical_context_table.detach().cpu().numpy(),
            np.asarray(supervision.normalized_contexts, dtype=np.float32),
        ):
            raise ValueError("Deterministic model context table differs from supervision.")
    elif result.model is not None:
        raise ValueError("Unconditional deterministic artifacts must not contain a model.")
    target_sha256 = supervision.source_identities.get("conditional_target")
    feature_group_sha256 = supervision.source_identities.get("feature_group")
    _validate_deterministic_training_payload(
        result.training_payload,
        supervision_kind=supervision.supervision_kind,
        context_binding_sha256=supervision.context_binding_sha256,
        target_sha256=target_sha256,
        feature_group_sha256=feature_group_sha256,
        schedule_keys=supervision.schedule_keys,
        model=result.model,
    )
    target = Path(output_dir).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        state_identity: str | None = None
        state_file_sha256: str | None = None
        model_config: dict[str, object] | None = None
        if result.model is not None:
            state = {
                name: value.detach().to(device="cpu").contiguous().clone()
                for name, value in sorted(result.model.state_dict().items())
            }
            state_identity = _state_sha256(state)
            state_path = stage / "deterministic-state.pt"
            torch.save(state, state_path)
            state_file_sha256 = file_sha256(state_path)
            model_config = result.model.config.as_payload()
        arrays = {
            "normalized_contexts": _write_deployment_array(
                stage / "normalized-contexts.npy",
                supervision.normalized_contexts,
            )
        }
        direct_barycenter_sha256: str | None = None
        if result.model is None:
            arrays["direct_barycenter"] = _write_deployment_array(
                stage / "direct-barycenter.npy",
                supervision.barycenter_density_mass,
            )
            direct_barycenter_sha256 = image_gico_supervision_array_sha256(
                supervision.barycenter_density_mass,
                field="barycenter_density_mass",
            )
        body = {
            "protocol": IMAGE_GICO_DETERMINISTIC_ARTIFACT_PROTOCOL,
            "student_kind": "deterministic_barycenter",
            "supervision_kind": supervision.supervision_kind,
            "supervision_sha256": supervision.sha256,
            "target_sha256": target_sha256,
            "feature_group_sha256": feature_group_sha256,
            "context_binding_sha256": supervision.context_binding_sha256,
            "target_nfes": list(supervision.target_nfes),
            "schedule_keys": list(supervision.schedule_keys),
            "context_count": supervision.context_count,
            "direct_barycenter_sha256": direct_barycenter_sha256,
            "array_files": arrays,
            "model_config": model_config,
            "training": dict(result.training_payload),
            "state_sha256": state_identity,
            "state_file_sha256": state_file_sha256,
        }
        artifact_identity = semantic_sha256(body, namespace=IMAGE_GICO_DETERMINISTIC_ARTIFACT_NAMESPACE)
        manifest = {
            "artifact": "image_gico_deterministic_barycenter_student",
            **body,
            "artifact_sha256": artifact_identity,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return MappingProxyType(manifest)


@dataclass(frozen=True, slots=True)
class LoadedImageGICODeterministicArtifact:
    artifact_sha256: str
    supervision_sha256: str
    supervision_kind: str
    target_nfes: tuple[int, ...]
    normalized_contexts: np.ndarray
    direct_barycenter_density_mass: np.ndarray | None
    model: ImageGICOBackboneContextDensityModel | None
    state_sha256: str | None
    context_binding_sha256: str
    direct_barycenter_sha256: str | None
    manifest: Mapping[str, Any]

    def verify(self) -> None:
        if (
            image_gico_supervision_array_sha256(
                self.normalized_contexts,
                field="normalized_contexts",
            )
            != self.context_binding_sha256
        ):
            raise ValueError("Loaded deterministic deployment contexts were mutated.")
        if self.model is None:
            if self.state_sha256 is not None or self.direct_barycenter_density_mass is None:
                raise ValueError("Direct barycenter artifact has unexpected state.")
            if (
                image_gico_supervision_array_sha256(
                    self.direct_barycenter_density_mass,
                    field="barycenter_density_mass",
                )
                != self.direct_barycenter_sha256
            ):
                raise ValueError("Loaded direct barycenter was mutated.")
        elif _state_sha256(self.model.state_dict()) != self.state_sha256:
            raise ValueError("Loaded deterministic model state was mutated.")
        elif (
            image_gico_supervision_array_sha256(
                self.model.canonical_context_table.detach().cpu().numpy().astype(np.float64),
                field="normalized_contexts",
            )
            != self.context_binding_sha256
        ):
            raise ValueError("Loaded deterministic context table was mutated.")
        elif self.direct_barycenter_density_mass is not None or self.direct_barycenter_sha256 is not None:
            raise ValueError("Conditional deterministic artifact has a direct barycenter.")
        body = dict(self.manifest)
        observed = body.pop("artifact_sha256", None)
        body.pop("artifact", None)
        if (
            observed != self.artifact_sha256
            or semantic_sha256(body, namespace=IMAGE_GICO_DETERMINISTIC_ARTIFACT_NAMESPACE) != self.artifact_sha256
        ):
            raise ValueError("Loaded deterministic artifact manifest was mutated.")


def load_image_gico_deterministic_artifact(
    output_dir: str | Path,
    supervision: ImageGICOSupervision | None = None,
    *,
    device: torch.device | str = "cpu",
    expected_artifact_sha256: str | None = None,
) -> LoadedImageGICODeterministicArtifact:
    if supervision is not None and not isinstance(supervision, ImageGICOSupervision):
        raise TypeError("supervision must be ImageGICOSupervision or None.")
    root = Path(output_dir).expanduser().resolve(strict=True)
    if is_link_or_reparse_point(root) or not root.is_dir():
        raise ValueError("Deterministic artifact root must be a regular directory.")
    manifest_path = root / "manifest.json"
    if is_link_or_reparse_point(manifest_path) or not manifest_path.is_file():
        raise ValueError("Deterministic manifest must be a regular file.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_fields = {
        "artifact",
        "protocol",
        "student_kind",
        "supervision_kind",
        "supervision_sha256",
        "target_sha256",
        "feature_group_sha256",
        "context_binding_sha256",
        "target_nfes",
        "schedule_keys",
        "context_count",
        "direct_barycenter_sha256",
        "array_files",
        "model_config",
        "training",
        "state_sha256",
        "state_file_sha256",
        "artifact_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        raise ValueError("Deterministic manifest fields are incomplete or unexpected.")
    arrays = manifest.get("array_files")
    if not isinstance(arrays, Mapping) or set(arrays) not in (
        {"normalized_contexts"},
        {"normalized_contexts", "direct_barycenter"},
    ):
        raise ValueError("Deterministic deployment arrays are invalid.")
    expected_members = {"manifest.json"} | {
        str(descriptor.get("file")) for descriptor in arrays.values() if isinstance(descriptor, Mapping)
    }
    if manifest.get("state_sha256") is not None:
        expected_members.add("deterministic-state.pt")
    if {path.name for path in root.iterdir()} != expected_members:
        raise ValueError("Deterministic artifact files are unexpected.")
    if (
        manifest.get("artifact") != "image_gico_deterministic_barycenter_student"
        or manifest.get("protocol") != IMAGE_GICO_DETERMINISTIC_ARTIFACT_PROTOCOL
        or manifest.get("student_kind") != "deterministic_barycenter"
        or manifest.get("supervision_kind") not in {"conditional_kid", "unconditional_mixture"}
        or manifest.get("target_nfes") != [2, 4, 8]
    ):
        raise ValueError("Deterministic artifact binding changed.")
    schedule_keys = manifest["schedule_keys"]
    if (
        not isinstance(schedule_keys, list)
        or not schedule_keys
        or len(schedule_keys) != len(set(schedule_keys))
        or any(not isinstance(key, str) or not key for key in schedule_keys)
    ):
        raise ValueError("Deterministic artifact schedule keys are invalid.")
    body = dict(manifest)
    artifact_identity = body.pop("artifact_sha256", None)
    body.pop("artifact", None)
    if artifact_identity != semantic_sha256(body, namespace=IMAGE_GICO_DETERMINISTIC_ARTIFACT_NAMESPACE):
        raise ValueError("Deterministic artifact identity is inconsistent.")
    if expected_artifact_sha256 is not None and artifact_identity != expected_artifact_sha256:
        raise ValueError("Deterministic artifact differs from the expected identity.")

    contexts = _load_deployment_array(root, arrays["normalized_contexts"], field="normalized_contexts")
    if (
        contexts.ndim != 2
        or contexts.shape[0] < 1
        or contexts.shape[1] != 768
        or not np.array_equal(contexts, contexts.astype(np.float32).astype(np.float64))
        or image_gico_supervision_array_sha256(contexts, field="normalized_contexts")
        != manifest.get("context_binding_sha256")
        or manifest.get("context_count") != contexts.shape[0]
    ):
        raise ValueError("Deterministic deployment contexts changed.")
    supervision_kind = str(manifest["supervision_kind"])
    if supervision_kind == "unconditional_mixture" and (contexts.shape[0] != 1 or bool(np.any(contexts != 0.0))):
        raise ValueError("Unconditional deterministic deployment requires one explicit zero context.")
    if supervision_kind == "conditional_kid" and contexts.shape[0] != 1_000:
        raise ValueError("Conditional deterministic deployment requires its exact 1,000-row context table.")
    direct_barycenter = (
        None
        if "direct_barycenter" not in arrays
        else _load_deployment_array(root, arrays["direct_barycenter"], field="direct_barycenter")
    )
    if supervision is not None and (
        manifest.get("supervision_sha256") != supervision.sha256
        or manifest.get("supervision_kind") != supervision.supervision_kind
        or tuple(schedule_keys) != supervision.schedule_keys
        or manifest.get("context_binding_sha256") != supervision.context_binding_sha256
        or not np.array_equal(contexts, supervision.normalized_contexts)
        or (
            direct_barycenter is not None and not np.array_equal(direct_barycenter, supervision.barycenter_density_mass)
        )
    ):
        raise ValueError("Deterministic artifact and supervision differ.")

    model: ImageGICOBackboneContextDensityModel | None = None
    state_identity = manifest.get("state_sha256")
    if supervision_kind == "conditional_kid":
        config = ImageGICOBackboneContextModelConfig(density_bin_count=DENSITY_BIN_COUNT)
        if manifest.get("model_config") != config.as_payload():
            raise ValueError("Deterministic model configuration changed.")
        state_path = root / "deterministic-state.pt"
        if is_link_or_reparse_point(state_path) or file_sha256(state_path) != manifest.get("state_file_sha256"):
            raise ValueError("Deterministic state file changed.")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()
        ):
            raise ValueError("Deterministic state is not a tensor mapping.")
        if _state_sha256(state) != state_identity:
            raise ValueError("Deterministic state identity changed.")
        if direct_barycenter is not None or manifest.get("direct_barycenter_sha256") is not None:
            raise ValueError("Conditional deterministic artifacts must not carry direct targets.")
        context_tensor = torch.as_tensor(np.ascontiguousarray(contexts, dtype=np.float32))
        with torch.random.fork_rng(devices=[]):
            model = ImageGICOBackboneContextDensityModel(config, context_tensor)
        model.load_state_dict(dict(state), strict=True)
        model.to(torch.device(device)).eval().requires_grad_(False)
    elif (
        state_identity is not None
        or manifest.get("model_config") is not None
        or direct_barycenter is None
        or direct_barycenter.shape != (3, 1, DENSITY_BIN_COUNT)
        or bool(np.any(direct_barycenter < 0.0))
        or not bool(
            np.allclose(
                np.sum(direct_barycenter, axis=-1),
                1.0,
                rtol=0.0,
                atol=1e-10,
            )
        )
        or image_gico_supervision_array_sha256(
            direct_barycenter,
            field="barycenter_density_mass",
        )
        != manifest.get("direct_barycenter_sha256")
    ):
        raise ValueError("Unconditional direct barycenter must not have model state.")
    _validate_deterministic_training_payload(
        manifest["training"],
        supervision_kind=supervision_kind,
        context_binding_sha256=str(manifest["context_binding_sha256"]),
        target_sha256=manifest["target_sha256"],
        feature_group_sha256=manifest["feature_group_sha256"],
        schedule_keys=tuple(schedule_keys),
        model=model,
    )
    loaded = LoadedImageGICODeterministicArtifact(
        artifact_sha256=artifact_identity,
        supervision_sha256=str(manifest["supervision_sha256"]),
        supervision_kind=supervision_kind,
        target_nfes=tuple(manifest["target_nfes"]),
        normalized_contexts=contexts,
        direct_barycenter_density_mass=direct_barycenter,
        model=model,
        state_sha256=state_identity,
        context_binding_sha256=str(manifest["context_binding_sha256"]),
        direct_barycenter_sha256=manifest.get("direct_barycenter_sha256"),
        manifest=MappingProxyType(manifest),
    )
    loaded.verify()
    return loaded


def _context_indices(value: Sequence[int], *, context_count: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise TypeError("context_indices must be a nonempty sequence of integers.")
    indices: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError("context_indices must contain integers.")
        if not 0 <= raw < context_count:
            raise ValueError("A context index is outside the supervision table.")
        indices.append(raw)
    return tuple(indices)


@dataclass(frozen=True, slots=True)
class ImageGICOScheduleMaterialization:
    student_kind: ImageGICOStudentKind
    target_nfe: int
    context_indices: tuple[int, ...]
    density_mass: np.ndarray
    time_grids: np.ndarray
    artifact_sha256: str
    supervision_sha256: str
    tokens: np.ndarray | None = None
    uniforms_sha256: str | None = None
    density_mass_sha256: str = dataclass_field(init=False, repr=False)
    time_grids_sha256: str = dataclass_field(init=False, repr=False)
    tokens_sha256: str | None = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        kind = normalize_image_gico_student_kind(self.student_kind)
        density = np.ascontiguousarray(self.density_mass, dtype=np.float64)
        grids = np.ascontiguousarray(self.time_grids, dtype=np.float64)
        expected_rows = len(self.context_indices)
        if self.target_nfe not in (2, 4, 8):
            raise ValueError("target_nfe must be 2, 4, or 8.")
        if expected_rows == 0 or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.context_indices
        ):
            raise ValueError("context_indices must contain nonnegative integers.")
        if density.shape != (expected_rows, DENSITY_BIN_COUNT):
            raise ValueError("density_mass has the wrong materialization shape.")
        if bool(np.any(density < 0.0)) or not bool(np.allclose(np.sum(density, axis=-1), 1.0, rtol=0.0, atol=1e-10)):
            raise ValueError("Materialized density rows must be probability masses.")
        if grids.shape != (expected_rows, self.target_nfe + 1):
            raise ValueError("time_grids has the wrong materialization shape.")
        if (
            not bool(np.isfinite(grids).all())
            or not bool(np.all(np.diff(grids, axis=-1) > 0.0))
            or not bool(np.all(grids[:, 0] == 0.0))
            or not bool(np.all(grids[:, -1] == 1.0))
        ):
            raise ValueError("Materialized time grids must be finite, complete, and strictly increasing.")
        tokens = self.tokens
        if kind == "stochastic_causal_ar":
            if tokens is None or self.uniforms_sha256 is None:
                raise ValueError("Stochastic materialization requires replay evidence.")
            tokens = np.ascontiguousarray(tokens, dtype=np.int64)
            if tokens.shape != (expected_rows, DENSITY_BIN_COUNT - 1):
                raise ValueError("Stochastic token paths have the wrong shape.")
        elif tokens is not None or self.uniforms_sha256 is not None:
            raise ValueError("Deterministic materialization cannot carry random evidence.")
        density.setflags(write=False)
        grids.setflags(write=False)
        if tokens is not None:
            tokens.setflags(write=False)
        object.__setattr__(self, "student_kind", kind)
        object.__setattr__(self, "density_mass", density)
        object.__setattr__(self, "time_grids", grids)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(
            self,
            "density_mass_sha256",
            _materialization_array_sha256(density, field="density_mass"),
        )
        object.__setattr__(
            self,
            "time_grids_sha256",
            _materialization_array_sha256(grids, field="time_grids"),
        )
        object.__setattr__(
            self,
            "tokens_sha256",
            None if tokens is None else _materialization_array_sha256(tokens, field="tokens"),
        )

    def verify(self) -> None:
        if (
            _materialization_array_sha256(self.density_mass, field="density_mass") != self.density_mass_sha256
            or _materialization_array_sha256(self.time_grids, field="time_grids") != self.time_grids_sha256
            or (None if self.tokens is None else _materialization_array_sha256(self.tokens, field="tokens"))
            != self.tokens_sha256
        ):
            raise ValueError("Frozen image GICO materialization was mutated.")


def materialize_image_gico_schedule(
    student_kind: ImageGICOStudentKind,
    *,
    deterministic_artifact: LoadedImageGICODeterministicArtifact | None,
    causal_artifact: LoadedImageGICOCausalArtifact | None,
    target_nfe: int,
    context_indices: Sequence[int],
    uniforms: Tensor | None = None,
) -> ImageGICOScheduleMaterialization:
    """Freeze one complete deterministic or causal schedule batch."""

    kind = normalize_image_gico_student_kind(student_kind)
    reference = np.linspace(0.0, 1.0, DENSITY_BIN_COUNT + 1, dtype=np.float64)
    if kind == "deterministic_barycenter":
        if deterministic_artifact is None:
            raise ValueError("Deterministic materialization requires its artifact.")
        if uniforms is not None or causal_artifact is not None:
            raise ValueError("Deterministic materialization rejects uniforms and causal artifacts.")
        deterministic_artifact.verify()
        if target_nfe not in deterministic_artifact.target_nfes:
            raise ValueError(f"target_nfe must be one of {deterministic_artifact.target_nfes}.")
        indices = _context_indices(
            context_indices,
            context_count=deterministic_artifact.normalized_contexts.shape[0],
        )
        nfe_index = deterministic_artifact.target_nfes.index(target_nfe)
        if deterministic_artifact.model is None:
            if deterministic_artifact.direct_barycenter_density_mass is None:
                raise ValueError("Direct barycenter artifact omitted its deployment target.")
            density = np.ascontiguousarray(
                deterministic_artifact.direct_barycenter_density_mass[nfe_index, list(indices)],
                dtype=np.float64,
            )
        else:
            model = deterministic_artifact.model
            contexts = torch.as_tensor(
                np.ascontiguousarray(
                    deterministic_artifact.normalized_contexts[list(indices)],
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=next(model.parameters()).device,
            )
            target_nfes = torch.full(
                (len(indices),),
                target_nfe,
                dtype=torch.int64,
                device=contexts.device,
            )
            with torch.inference_mode():
                density = model(contexts, target_nfes).detach().cpu().to(torch.float64).numpy()
            density = np.ascontiguousarray(density, dtype=np.float64)
        grids = inverse_cdf_clock_nodes(density, target_nfe, reference)
        return ImageGICOScheduleMaterialization(
            student_kind=kind,
            target_nfe=target_nfe,
            context_indices=indices,
            density_mass=density,
            time_grids=grids,
            artifact_sha256=deterministic_artifact.artifact_sha256,
            supervision_sha256=deterministic_artifact.supervision_sha256,
        )

    if deterministic_artifact is not None:
        raise ValueError("Stochastic materialization rejects deterministic artifacts.")
    if causal_artifact is None or uniforms is None:
        raise ValueError("Stochastic materialization requires a causal artifact and uniforms.")
    causal_artifact.verify()
    if target_nfe not in causal_artifact.path_bank.target_nfes:
        raise ValueError(f"target_nfe must be one of {causal_artifact.path_bank.target_nfes}.")
    indices = _context_indices(
        context_indices,
        context_count=causal_artifact.normalized_contexts.shape[0],
    )
    contexts = torch.as_tensor(
        np.ascontiguousarray(causal_artifact.normalized_contexts[list(indices)], dtype=np.float32),
        dtype=torch.float32,
        device=causal_artifact.model.device,
    )
    realization: FrozenImageGICOCausalRealization = causal_artifact.policy.sample_realization(
        contexts, target_nfe, uniforms
    )
    return ImageGICOScheduleMaterialization(
        student_kind=kind,
        target_nfe=target_nfe,
        context_indices=indices,
        density_mass=np.asarray(realization.raw_densities, dtype=np.float64),
        time_grids=np.asarray(realization.time_grids, dtype=np.float64),
        artifact_sha256=causal_artifact.artifact_sha256,
        supervision_sha256=causal_artifact.supervision_sha256,
        tokens=np.asarray(realization.tokens, dtype=np.int64),
        uniforms_sha256=image_gico_causal_uniforms_sha256(uniforms),
    )


def execute_image_gico_euler(
    field: Callable[[Tensor, Tensor], Tensor],
    initial_state: Tensor,
    materialization: ImageGICOScheduleMaterialization,
) -> SolverResult:
    """Execute only a previously frozen grid and enforce exact NFE accounting."""

    if not isinstance(materialization, ImageGICOScheduleMaterialization):
        raise TypeError("materialization must be ImageGICOScheduleMaterialization.")
    materialization.verify()
    if not isinstance(initial_state, Tensor) or initial_state.shape[0] != len(materialization.context_indices):
        raise ValueError("initial_state batch size must match materialization.")
    grid = torch.as_tensor(
        np.array(materialization.time_grids, dtype=np.float64, copy=True),
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    result = integrate_euler(
        field,
        initial_state,
        target_nfe=materialization.target_nfe,
        time_grid=grid,
    )
    if result.field_evaluations != materialization.target_nfe:
        raise RuntimeError("Euler execution violated exact NFE accounting.")
    return result


__all__ = [
    "IMAGE_GICO_DETERMINISTIC_ARTIFACT_PROTOCOL",
    "ImageGICODeterministicTrainingResult",
    "ImageGICOScheduleMaterialization",
    "LoadedImageGICODeterministicArtifact",
    "execute_image_gico_euler",
    "load_image_gico_deterministic_artifact",
    "materialize_image_gico_schedule",
    "save_image_gico_deterministic_artifact",
    "train_image_gico_deterministic_student",
]
