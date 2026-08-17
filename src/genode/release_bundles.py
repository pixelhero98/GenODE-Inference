from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from genode.artifacts.identity import canonical_json_bytes as identity_canonical_json_bytes
from genode.artifacts.identity import semantic_sha256
from genode.backbones.registry import get_image_backbone_spec
from genode.deterministic_archive import (
    ArchiveEntry,
    canonical_json_bytes,
    contains_local_filesystem_path,
    sha256_file,
    validate_deterministic_zip,
    write_deterministic_zip,
)
from genode.gico.image_conditional import (
    ImageGICOBackboneContextDensityModel,
    ImageGICOBackboneContextModelConfig,
)
from genode.gico.image_conditional_training import ImageGICOBackboneContextTeacher

FROZEN_GICO_POLICY_SCHEMA = "frozen_gico_policy_v1"
FROZEN_BACKBONE_COLLECTION_SCHEMA = "frozen_backbone_collection_v1"
_SOURCE_POLICY_FILES: Mapping[str, str] = {
    "class-density-table.npy": "density_table",
    "conditional-targets.json": "targets",
    "context-normalizer-mean.npy": "context_mean",
    "context-normalizer-scale.npy": "context_scale",
    "reward-feature-groups.json": "feature_groups",
    "student-state.pt": "student_state",
    "teacher-state.pt": "teacher_state",
}
_ARCHIVED_POLICY_FILES: Mapping[str, str] = {
    filename: role for filename, role in _SOURCE_POLICY_FILES.items() if filename.endswith((".npy", ".pt"))
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_NAMESPACE_PATTERN = re.compile(r"^image-[a-z0-9-]+-backbone-context-(?P<kind>[a-z-]+)-v[1-9][0-9]*$")
_IMAGE_POLICY_ARTIFACT_PATTERN = re.compile(r"^image_(?P<token>[a-z0-9]+)_backbone_context_policy$")
_HISTORICAL_POLICY_MANIFEST_DIGESTS = frozenset(
    {
        "23cc57132810438b4a2c76a4b42c66956202765ca7b14529f888c11d3dc0c1a6",
        "e1c196c1df1aa567febfe8854a03b84b46527d71fd57ac8c393b8580a7210ab3",
    }
)
_POLICY_SOURCE_MANIFEST_FIELDS = frozenset(
    {
        "artifact",
        "artifact_sha256",
        "backbone_checkpoint_sha256",
        "backbone_model_key",
        "backbone_protocol_sha256",
        "class_nfe_contract",
        "context_binding",
        "context_binding_sha256",
        "density_table_sha256",
        "feature_group_sha256",
        "feature_protocol_sha256",
        "files",
        "portable_execution",
        "protocol",
        "schedule_support",
        "student_state_sha256",
        "target_sha256",
        "teacher_state_sha256",
        "training",
    }
)
_CONTEXT_BINDING_FIELDS = frozenset(
    {
        "backbone_checkpoint_sha256",
        "backbone_model_key",
        "backbone_protocol_sha256",
        "binding_sha256",
        "class_count",
        "class_order_sha256",
        "context_dim",
        "dtype",
        "normalization_std_floor",
        "normalized_context_table_sha256",
        "normalizer_mean_sha256",
        "normalizer_scale_sha256",
        "normalizer_sha256",
        "protocol",
        "raw_context_table_sha256",
        "selector",
        "source_config_identity",
        "source_revision",
    }
)
_POLICY_TRAINING_FIELDS = frozenset(
    {
        "conditional_density_range",
        "conditioning",
        "context_binding_sha256",
        "feature_group_sha256",
        "feature_group_usage",
        "final_kl",
        "final_objective",
        "final_residual_penalty",
        "final_teacher_score",
        "model_config",
        "model_config_sha256",
        "model_state_sha256",
        "protocol",
        "result_sha256",
        "target_sha256",
        "teacher_density_summary_protocol",
        "teacher_evidence_row_count",
        "teacher_evidence_sha256",
        "teacher_oof_pairwise_accuracy",
        "teacher_oof_rmse",
        "teacher_protocol",
        "teacher_schedule_fold_diagnostics",
        "teacher_state_sha256",
        "training_config",
        "training_config_sha256",
    }
)
_POLICY_MODEL_CONFIG_FIELDS = frozenset(
    {
        "class_count",
        "conditioning",
        "context_dim",
        "density_bin_count",
        "density_floor",
        "global_base",
        "hidden_dim",
        "nfe_embedding_dim",
        "protocol",
        "residual_centering",
        "residual_initialization",
        "target_nfes",
    }
)
_POLICY_CONTRACT_FIELDS = frozenset(
    {
        "class_count",
        "conditioning",
        "context_dim",
        "density_bin_count",
        "density_table_shape",
        "feature_groups_are_inference_inputs",
        "initial_noise_is_an_inference_input",
        "raw_context_table_is_portable",
        "target_nfes",
    }
)
_POLICY_SCHEDULE_SUPPORT_FIELDS = frozenset(
    {
        "density_mass_sha256s",
        "fixed_support_sha256",
        "reward_evidence_sha256",
        "schedule_keys",
        "schedule_sha256s",
    }
)
_FEATURE_GROUP_FIELDS = frozenset(
    {
        "artifact",
        "class_coordinates",
        "class_count",
        "feature_dim",
        "feature_group_sha256",
        "feature_protocol_sha256",
        "group_assignments",
        "group_centroids",
        "group_count",
        "pca_center",
        "pca_components",
        "pca_scales",
        "protocol",
        "real_feature_panel_sha256",
        "samples_per_class",
        "source_panel_fingerprint",
        "usage",
    }
)
_TARGET_FIELDS = frozenset(
    {
        "artifact",
        "backbone_checkpoint_sha256",
        "backbone_model_key",
        "backbone_protocol_sha256",
        "class_count",
        "class_reliability",
        "conditioning",
        "density_bin_count",
        "density_mass",
        "density_mass_sha256s",
        "feature_group_sha256",
        "feature_group_usage",
        "feature_protocol_sha256",
        "fixed_support_sha256",
        "group_reliability",
        "jackknife_standard_errors",
        "mixture_weights",
        "normalized_rewards",
        "protocol",
        "reward_evidence_sha256",
        "schedule_count",
        "schedule_keys",
        "schedule_sha256s",
        "shrinkage_coefficients",
        "target_nfes",
        "target_sha256",
        "temperature_by_nfe",
    }
)
_PORTABLE_EXECUTION_CONTRACT = {
    "bind_phase": "regenerate_context_from_verified_backbone",
    "density_table_verified_at_bind": True,
    "load_phase": "metadata_and_weights_only",
    "requires_explicit_backbone_binding": True,
}
_CHECKPOINT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}


@dataclass(frozen=True)
class NamedCheckpoint:
    name: str
    path: Path


def _plain_digest(value: object) -> str:
    text = str(value or "").strip()
    return text.rsplit(":", 1)[-1] if text else ""


def _require_digest(value: object, *, label: str) -> str:
    digest = _plain_digest(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return digest


def _semantic_identity_namespace(value: object, *, label: str, kind: str) -> tuple[str, str]:
    identity = str(value or "").strip()
    namespace, separator, digest = identity.rpartition(":")
    match = _SEMANTIC_NAMESPACE_PATTERN.fullmatch(namespace)
    if not separator or match is None or match.group("kind") != kind or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a namespaced lowercase SHA-256 {kind} identity.")
    return identity, namespace


def _validate_semantic_record(
    payload: Mapping[str, Any],
    *,
    identity_field: str,
    label: str,
    kind: str,
) -> str:
    body = dict(payload)
    stored, namespace = _semantic_identity_namespace(
        body.pop(identity_field, None),
        label=label,
        kind=kind,
    )
    observed = semantic_sha256(body, namespace=namespace)
    if observed != stored:
        raise ValueError(f"{label} is inconsistent with its payload.")
    return stored


def _source_policy_identity(source_manifest: Mapping[str, Any], *, manifest_digest: str) -> tuple[str, int]:
    if set(source_manifest) != _POLICY_SOURCE_MANIFEST_FIELDS:
        raise ValueError("Policy source manifest fields are incomplete or unexpected.")
    artifact = str(source_manifest.get("artifact", ""))
    match = _IMAGE_POLICY_ARTIFACT_PATTERN.fullmatch(artifact)
    if match is None:
        raise ValueError("Policy source artifact is not an image backbone-context policy.")
    protocol = str(source_manifest.get("protocol", ""))
    protocol_match = re.fullmatch(re.escape(artifact) + r"_bundle_v([1-9][0-9]*)", protocol)
    if protocol_match is None:
        raise ValueError("Policy source bundle protocol is inconsistent with its artifact.")
    version = int(protocol_match.group(1))
    token = match.group("token")
    if token != "gico" and manifest_digest not in _HISTORICAL_POLICY_MANIFEST_DIGESTS:
        raise ValueError("Only current GICO policies and the frozen verified historical policies may be packaged.")
    return token, version


def _validate_payload_semantic_identity(
    payload: Mapping[str, Any],
    *,
    identity_field: str,
    namespace: str,
    label: str,
) -> str:
    body = dict(payload)
    body.pop("artifact", None)
    stored = str(body.pop(identity_field, ""))
    if stored != semantic_sha256(body, namespace=namespace):
        raise ValueError(f"{label} semantic identity is inconsistent.")
    return stored


def _array_identity(value: np.ndarray, *, namespace: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(namespace.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        identity_canonical_json_bytes(
            {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        )
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return f"{namespace}:{digest.hexdigest()}"


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _assert_no_local_json_paths(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_local_json_paths(key, label=f"{label} key")
            _assert_no_local_json_paths(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_local_json_paths(item, label=f"{label}[{index}]")
    elif isinstance(value, str) and contains_local_filesystem_path(value):
        raise ValueError(f"{label} contains a local filesystem path.")


def _json_mapping(path: Path, *, label: str, require_canonical: bool = False) -> Mapping[str, Any]:
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant {value!r}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    result = _require_mapping(payload, label=label)
    if require_canonical:
        canonical = identity_canonical_json_bytes(result)
        if raw not in {canonical, canonical + b"\n"}:
            raise ValueError(f"{label} must use canonical JSON encoding.")
    _assert_no_local_json_paths(result, label=label)
    return result


def _resolve_source_path(source_root: Path, value: object) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Checkpoint manifest contains an empty source path.")
    raw = Path(text)
    portable = PurePosixPath(text.replace("\\", "/"))
    if not raw.is_absolute() and (
        portable.is_absolute()
        or not portable.parts
        or any(part in {"", ".", ".."} for part in portable.parts)
        or portable.parts[0].endswith(":")
    ):
        raise ValueError(f"Unsafe checkpoint source path: {text!r}")
    candidate = raw if raw.is_absolute() else source_root.joinpath(*portable.parts)
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    resolved = absolute.resolve(strict=True)
    try:
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"Checkpoint source escapes source_root: {text!r}") from exc
    return absolute


def _artifact_archive_prefix(artifact: Mapping[str, Any]) -> PurePosixPath:
    parts = [
        str(artifact.get("benchmark_family", "")).strip(),
        str(artifact.get("dataset_key", "")).strip(),
    ]
    for field in ("member_key", "stratum", "variant"):
        value = str(artifact.get(field, "") or "").strip()
        if value:
            parts.append(value)
    train_steps = int(artifact.get("train_steps", 0))
    if train_steps <= 0:
        raise ValueError("Artifact train_steps must be positive for checkpoint packaging.")
    parts.append(f"step-{train_steps}")
    for part in parts:
        if part in {"", ".", ".."} or "/" in part or "\\" in part:
            raise ValueError(f"Unsafe artifact identity component: {part!r}")
    return PurePosixPath("backbones", *parts)


def package_backbone_manifest_checkpoints(
    *,
    manifest_path: str | Path,
    source_root: str | Path,
    output_path: str | Path,
    expected_count: int | None = None,
    include_support_files: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Checkpoint source_root is not a directory: {root}")
    manifest = _json_mapping(manifest_file, label="backbone manifest")
    raw_artifacts = manifest.get("artifacts", [])
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("Backbone manifest artifacts must be a non-empty list.")
    artifacts = [_require_mapping(row, label="backbone artifact") for row in raw_artifacts]
    non_ready = [row for row in artifacts if str(row.get("status", "")) != "ready"]
    if non_ready:
        raise ValueError("Backbone checkpoint release requires every manifest artifact to be ready.")
    if "artifact_count" in manifest and int(manifest["artifact_count"]) != len(artifacts):
        raise ValueError("Backbone manifest artifact_count does not match its artifacts list.")
    if "ready_count" in manifest and int(manifest["ready_count"]) != len(artifacts):
        raise ValueError("Backbone manifest ready_count does not match its ready artifacts.")
    if expected_count is not None and len(artifacts) != int(expected_count):
        raise ValueError(f"Ready checkpoint count is {len(artifacts)}; expected {int(expected_count)}.")
    if not artifacts:
        raise ValueError("Backbone manifest contains no ready artifacts.")

    entries: list[ArchiveEntry] = []
    catalog: list[dict[str, Any]] = []
    seen_checkpoint_ids: set[str] = set()
    for artifact in sorted(artifacts, key=lambda row: str(row.get("checkpoint_id", ""))):
        checkpoint_id = str(artifact.get("checkpoint_id", "")).strip()
        if not checkpoint_id or checkpoint_id in seen_checkpoint_ids:
            raise ValueError(f"Checkpoint identifiers must be non-empty and unique: {checkpoint_id!r}")
        seen_checkpoint_ids.add(checkpoint_id)
        prefix = _artifact_archive_prefix(artifact)
        checkpoint = _resolve_source_path(root, artifact.get("checkpoint_path"))
        if checkpoint.suffix.lower() not in _CHECKPOINT_SUFFIXES:
            raise ValueError(f"Frozen backbone checkpoint has an unsupported filename: {checkpoint.name}")
        if checkpoint.stat().st_size < 1024:
            raise ValueError(f"Frozen backbone checkpoint is too small to be valid: {checkpoint.name}")
        checkpoint_archive_path = (prefix / checkpoint.name).as_posix()
        entries.append(ArchiveEntry(checkpoint, checkpoint_archive_path, "frozen_backbone_checkpoint"))
        support_paths: list[str] = []
        if include_support_files:
            for field, role in (
                ("metadata_path", "checkpoint_metadata"),
                ("summary_path", "checkpoint_summary"),
            ):
                value = str(artifact.get(field, "") or "").strip()
                if not value:
                    continue
                support = _resolve_source_path(root, value)
                if support.suffix.lower() != ".json":
                    raise ValueError(f"Checkpoint support file must be JSON: {support.name}")
                _json_mapping(support, label=f"checkpoint {field}")
                archive_path = (prefix / support.name).as_posix()
                entries.append(ArchiveEntry(support, archive_path, role))
                support_paths.append(archive_path)
        catalog.append(
            {
                "archive_path": checkpoint_archive_path,
                "backbone_name": str(artifact.get("backbone_name", "")),
                "benchmark_family": str(artifact.get("benchmark_family", "")),
                "checkpoint_id": checkpoint_id,
                "dataset_key": str(artifact.get("dataset_key", "")),
                "effective_train_steps": int(artifact.get("effective_train_steps", 0) or 0),
                "member_key": str(artifact.get("member_key", "") or ""),
                "model_cond_dim": int(artifact.get("model_cond_dim", 0) or 0),
                "stratum": str(artifact.get("stratum", "") or ""),
                "support_paths": sorted(support_paths),
                "train_steps": int(artifact.get("train_steps", 0)),
                "variant": str(artifact.get("variant", "") or ""),
            }
        )
    canonical_catalog = canonical_json_bytes(catalog)
    return write_deterministic_zip(
        entries,
        output_path,
        bundle_kind="frozen_backbone_collection",
        metadata={
            "artifact_count": len(catalog),
            "artifacts": catalog,
            "schema_version": FROZEN_BACKBONE_COLLECTION_SCHEMA,
            "source_catalog_sha256": hashlib.sha256(canonical_catalog).hexdigest(),
        },
        overwrite=overwrite,
    )


def package_named_checkpoints(
    checkpoints: Sequence[NamedCheckpoint],
    output_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    entries: list[ArchiveEntry] = []
    names: set[str] = set()
    for checkpoint in checkpoints:
        name = str(checkpoint.name).strip()
        if not name or name in names or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"Checkpoint names must be safe and unique: {name!r}")
        names.add(name)
        path = Path(checkpoint.path).expanduser()
        if path.suffix.lower() not in _CHECKPOINT_SUFFIXES:
            raise ValueError(f"Frozen backbone checkpoint has an unsupported filename: {path.name}")
        if not path.is_file() or path.stat().st_size < 1024:
            raise ValueError(f"Frozen backbone checkpoint is missing or too small to be valid: {path}")
        entries.append(
            ArchiveEntry(
                path,
                PurePosixPath("backbones", name, path.name).as_posix(),
                "frozen_backbone_checkpoint",
            )
        )
    return write_deterministic_zip(
        entries,
        output_path,
        bundle_kind="frozen_backbone_collection",
        metadata={
            **dict(metadata or {}),
            "checkpoint_names": sorted(names),
            "schema_version": FROZEN_BACKBONE_COLLECTION_SCHEMA,
        },
        overwrite=overwrite,
    )


def _frozen_policy_metadata(
    policy_root: Path,
) -> tuple[dict[str, Any], Mapping[str, Any], str, int]:
    source_manifest_path = policy_root / "manifest.json"
    source_manifest = _json_mapping(
        source_manifest_path,
        label="policy source manifest",
        require_canonical=True,
    )
    source_manifest_digest = sha256_file(source_manifest_path)
    policy_token, bundle_version = _source_policy_identity(
        source_manifest,
        manifest_digest=source_manifest_digest,
    )
    _validate_semantic_record(
        source_manifest,
        identity_field="artifact_sha256",
        label="policy artifact identity",
        kind="artifact",
    )
    training = _require_mapping(source_manifest.get("training"), label="policy training record")
    if set(training) != _POLICY_TRAINING_FIELDS:
        raise ValueError("Policy training-result fields are incomplete or unexpected.")
    _validate_semantic_record(
        training,
        identity_field="result_sha256",
        label="policy training-result identity",
        kind="training-result",
    )
    model_config = _require_mapping(training.get("model_config"), label="policy model config")
    if set(model_config) != _POLICY_MODEL_CONFIG_FIELDS:
        raise ValueError("Policy model-config fields are incomplete or unexpected.")
    binding = _require_mapping(source_manifest.get("context_binding"), label="policy context binding")
    if set(binding) != _CONTEXT_BINDING_FIELDS:
        raise ValueError("Policy context-binding fields are incomplete or unexpected.")
    contract = _require_mapping(source_manifest.get("class_nfe_contract"), label="policy inference contract")
    if set(contract) != _POLICY_CONTRACT_FIELDS:
        raise ValueError("Policy inference-contract fields are incomplete or unexpected.")
    schedule_support = _require_mapping(source_manifest.get("schedule_support"), label="policy schedule support")
    if set(schedule_support) != _POLICY_SCHEDULE_SUPPORT_FIELDS:
        raise ValueError("Policy schedule-support fields are incomplete or unexpected.")
    required_positive_ints = {
        "class_count": model_config.get("class_count"),
        "context_dim": model_config.get("context_dim"),
        "density_bin_count": model_config.get("density_bin_count"),
        "hidden_dim": model_config.get("hidden_dim"),
        "nfe_embedding_dim": model_config.get("nfe_embedding_dim"),
    }
    expected_positive_ints = {
        "class_count": 1_000,
        "context_dim": 768,
        "density_bin_count": 64,
        "hidden_dim": 256,
        "nfe_embedding_dim": 16,
    }
    if required_positive_ints != expected_positive_ints:
        raise ValueError(f"Policy model dimensions must be exactly {expected_positive_ints}.")
    target_nfes = [int(value) for value in model_config.get("target_nfes", [])]
    if target_nfes != [2, 4, 8]:
        raise ValueError("Policy target_nfes must be exactly [2, 4, 8].")
    model_protocol = f"image_{policy_token}_backbone_context_nfe_residual_v3"
    expected_model_values = {
        "conditioning": "normalized_frozen_backbone_map_label_plus_target_nfe",
        "density_floor": 1e-8,
        "global_base": "learned_per_nfe_logits",
        "protocol": model_protocol,
        "residual_centering": "canonical_1000_context_table_per_nfe",
        "residual_initialization": "zero_output_layer",
    }
    if any(model_config.get(field) != value for field, value in expected_model_values.items()):
        raise ValueError("Policy model-config semantics are not the canonical ImageNet backbone-context contract.")
    if int(binding.get("class_count", 0)) != int(model_config["class_count"]):
        raise ValueError("Policy class count differs between its context binding and model config.")
    if int(binding.get("context_dim", 0)) != int(model_config["context_dim"]):
        raise ValueError("Policy context width differs between its context binding and model config.")
    expected_shape = [len(target_nfes), int(model_config["class_count"]), int(model_config["density_bin_count"])]
    expected_contract = {
        "class_count": 1_000,
        "conditioning": "normalized_frozen_backbone_map_label_plus_target_nfe",
        "context_dim": 768,
        "density_bin_count": 64,
        "density_table_shape": expected_shape,
        "feature_groups_are_inference_inputs": False,
        "initial_noise_is_an_inference_input": False,
        "raw_context_table_is_portable": False,
        "target_nfes": [2, 4, 8],
    }
    if dict(contract) != expected_contract:
        raise ValueError("Policy inference contract is not the canonical ImageNet backbone-context contract.")
    if dict(source_manifest.get("portable_execution", {})) != _PORTABLE_EXECUTION_CONTRACT:
        raise ValueError("Policy portable-execution contract is incomplete or unsupported.")
    if training.get("protocol") != f"image_{policy_token}_backbone_context_training_v{bundle_version}":
        raise ValueError("Policy training protocol is inconsistent with its source bundle.")
    if training.get("conditioning") != "normalized_frozen_backbone_map_label_plus_target_nfe":
        raise ValueError("Policy training record does not use the frozen backbone context.")
    if training.get("feature_group_usage") != "reward_shrinkage_only_not_inference_context":
        raise ValueError("Policy feature groups must remain reward-only evidence.")
    for field in ("student_state_sha256", "teacher_state_sha256", "density_table_sha256"):
        _require_digest(source_manifest.get(field), label=f"policy {field}")
    schedule_keys = schedule_support.get("schedule_keys", [])
    if (
        not isinstance(schedule_keys, list)
        or not schedule_keys
        or any(not isinstance(value, str) or not value.strip() for value in schedule_keys)
        or len(schedule_keys) != len(set(schedule_keys))
    ):
        raise ValueError("Policy schedule support must contain unique, non-empty schedule keys.")
    metadata = {
        "backbone": {
            "checkpoint_sha256": _require_digest(
                source_manifest.get("backbone_checkpoint_sha256"),
                label="policy backbone checkpoint digest",
            ),
            "model_key": str(source_manifest.get("backbone_model_key", "")),
            "protocol_sha256": _require_digest(
                source_manifest.get("backbone_protocol_sha256"),
                label="policy backbone protocol digest",
            ),
        },
        "context": {
            "binding_digest": _require_digest(binding.get("binding_sha256"), label="policy context binding digest"),
            "class_count": int(binding["class_count"]),
            "context_dim": int(binding["context_dim"]),
            "normalization_std_floor": float(binding.get("normalization_std_floor", 0.0)),
            "selector": str(binding.get("selector", "")),
            "source_revision": str(binding.get("source_revision", "")),
        },
        "inference_contract": {
            "conditioning": str(contract.get("conditioning", "")),
            "density_table_shape": [int(value) for value in contract.get("density_table_shape", [])],
            "feature_groups_are_inference_inputs": bool(contract.get("feature_groups_are_inference_inputs", False)),
            "initial_noise_is_an_inference_input": bool(contract.get("initial_noise_is_an_inference_input", False)),
            "raw_context_table_is_portable": bool(contract.get("raw_context_table_is_portable", False)),
        },
        "model": {
            **{field: int(value) for field, value in required_positive_ints.items()},
            "density_floor": float(model_config.get("density_floor", 0.0)),
            "target_nfes": target_nfes,
        },
        "policy_schema_version": FROZEN_GICO_POLICY_SCHEMA,
        "source_artifact_digest": _require_digest(
            source_manifest.get("artifact_sha256"),
            label="policy artifact digest",
        ),
        "source_manifest_sha256": source_manifest_digest,
        "training_clock_pool": list(schedule_keys),
    }
    if not metadata["backbone"]["model_key"]:
        raise ValueError("Policy backbone model_key may not be empty.")
    if not metadata["context"]["selector"] or not metadata["context"]["source_revision"]:
        raise ValueError("Policy context selector and source revision may not be empty.")
    return metadata, source_manifest, policy_token, bundle_version


def _load_policy_state(path: Path, *, label: str) -> Mapping[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Frozen policy {label} state is not loadable: {exc}") from exc
    if (
        not isinstance(state, Mapping)
        or not state
        or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items())
    ):
        raise ValueError(f"Frozen policy {label} state must be a non-empty tensor state dictionary.")
    return state


def _policy_state_semantic_identity(path: Path, *, stored: object, label: str, kind: str) -> str:
    expected, namespace = _semantic_identity_namespace(stored, label=f"policy {label} state identity", kind=kind)
    state = _load_policy_state(path, label=label)
    payload: dict[str, Any] = {}
    for name, tensor in sorted(state.items()):
        array = tensor.detach().to(device="cpu").contiguous().numpy()
        payload[name] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "values": array.tolist(),
        }
    observed = semantic_sha256(payload, namespace=namespace)
    if observed != expected:
        raise ValueError(f"Frozen policy {label} state identity is inconsistent with its tensors.")
    return expected


def _validate_policy_arrays(policy_root: Path, metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
    model = _require_mapping(metadata.get("model"), label="packaged policy model metadata")
    class_count = int(model["class_count"])
    context_dim = int(model["context_dim"])
    density_bin_count = int(model["density_bin_count"])
    target_nfes = list(model["target_nfes"])
    expected_shapes = {
        "class-density-table.npy": (len(target_nfes), class_count, density_bin_count),
        "context-normalizer-mean.npy": (context_dim,),
        "context-normalizer-scale.npy": (context_dim,),
    }
    arrays: dict[str, np.ndarray] = {}
    for filename, shape in expected_shapes.items():
        try:
            value = np.load(policy_root / filename, allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Frozen policy array is not loadable: {filename}: {exc}") from exc
        if value.shape != shape or value.dtype != np.dtype("<f4") or not np.all(np.isfinite(value)):
            raise ValueError(f"Frozen policy array has invalid shape, dtype, or values: {filename}")
        arrays[filename] = value
    density = arrays["class-density-table.npy"]
    if np.any(density < 0.0) or not np.allclose(density.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("Frozen policy density table must contain normalized nonnegative densities.")
    scale = arrays["context-normalizer-scale.npy"]
    if np.any(scale <= 0.0):
        raise ValueError("Frozen policy context-normalizer scale must be strictly positive.")
    return arrays


def _validate_policy_context_binding(
    source_manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    policy_token: str,
) -> None:
    binding = _require_mapping(source_manifest.get("context_binding"), label="policy context binding")
    expected_protocol = f"image_{policy_token}_backbone_context_binding_v3"
    expected_values = {
        "class_count": 1_000,
        "context_dim": 768,
        "dtype": "float32",
        "normalization_std_floor": 1e-6,
        "protocol": expected_protocol,
        "selector": "native_model.model.map_label",
    }
    if any(binding.get(field) != value for field, value in expected_values.items()):
        raise ValueError("Policy context binding is not the canonical ImageNet map-label contract.")
    binding_identity = _validate_semantic_record(
        binding,
        identity_field="binding_sha256",
        label="policy context-binding identity",
        kind="binding",
    )
    training = _require_mapping(source_manifest.get("training"), label="policy training record")
    if (
        source_manifest.get("context_binding_sha256") != binding_identity
        or training.get("context_binding_sha256") != binding_identity
    ):
        raise ValueError("Policy context-binding identities disagree.")
    for field in ("backbone_model_key", "backbone_protocol_sha256", "backbone_checkpoint_sha256"):
        if binding.get(field) != source_manifest.get(field):
            raise ValueError(f"Policy context binding disagrees with source {field}.")
    if not str(binding.get("source_revision", "")) or not str(binding.get("source_config_identity", "")):
        raise ValueError("Policy context binding requires source revision and config identities.")
    try:
        backbone = get_image_backbone_spec(str(binding["backbone_model_key"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("Policy context binding names an unknown image backbone.") from exc
    if (
        backbone.dataset_key != "imagenet64"
        or backbone.num_conditioning_classes != 1_000
        or binding.get("source_revision") != backbone.source_revision
        or binding.get("source_config_identity") != backbone.source_config_identity
    ):
        raise ValueError("Policy context binding does not match its registered ImageNet backbone.")

    namespace_prefix = f"image-{policy_token}-backbone-context"
    mean = np.ascontiguousarray(arrays["context-normalizer-mean.npy"], dtype="<f4")
    scale = np.ascontiguousarray(arrays["context-normalizer-scale.npy"], dtype="<f4")
    mean_identity = _array_identity(mean, namespace=f"{namespace_prefix}-mean-v3")
    scale_identity = _array_identity(scale, namespace=f"{namespace_prefix}-scale-v3")
    if (
        binding.get("normalizer_mean_sha256") != mean_identity
        or binding.get("normalizer_scale_sha256") != scale_identity
    ):
        raise ValueError("Policy context-normalizer arrays disagree with their binding.")
    normalizer_body = {
        "context_dim": 768,
        "dtype": "float32",
        "mean_sha256": mean_identity,
        "protocol": f"image_{policy_token}_backbone_context_normalizer_v3",
        "scale_sha256": scale_identity,
        "std_floor": 1e-6,
    }
    normalizer_identity = semantic_sha256(
        normalizer_body,
        namespace=f"{namespace_prefix}-normalizer-v3",
    )
    if binding.get("normalizer_sha256") != normalizer_identity:
        raise ValueError("Policy context-normalizer identity is inconsistent.")
    class_order_identity = semantic_sha256(
        {"class_ids": list(range(1_000))},
        namespace=f"{namespace_prefix}-class-order-v3",
    )
    if binding.get("class_order_sha256") != class_order_identity:
        raise ValueError("Policy context class ordering is inconsistent.")
    for field in ("raw_context_table_sha256", "normalized_context_table_sha256"):
        _require_digest(binding.get(field), label=f"policy {field}")


def _validate_policy_architecture(policy_root: Path, *, density_bin_count: int) -> None:
    student_state = _load_policy_state(policy_root / "student-state.pt", label="student")
    teacher_state = _load_policy_state(policy_root / "teacher-state.pt", label="teacher")
    config = ImageGICOBackboneContextModelConfig(density_bin_count=density_bin_count)
    student = ImageGICOBackboneContextDensityModel(
        config,
        np.zeros((1_000, 768), dtype=np.float32),
    )
    teacher = ImageGICOBackboneContextTeacher(density_bin_count=density_bin_count)
    expected_states = {
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
    }
    observed_states = {
        "student": student_state,
        "teacher": teacher_state,
    }
    for label, expected_state in expected_states.items():
        observed_state = observed_states[label]
        if set(observed_state) != set(expected_state):
            raise ValueError(f"Frozen policy {label} state has unexpected or missing tensor names.")
        for name, expected in expected_state.items():
            observed = observed_state[name]
            if (
                observed.layout != torch.strided
                or observed.is_quantized
                or observed.device.type != "cpu"
                or observed.shape != expected.shape
                or observed.dtype != expected.dtype
            ):
                raise ValueError(
                    f"Frozen policy {label} tensor {name!r} has an invalid layout, device, shape, or dtype."
                )
            if (observed.is_floating_point() or observed.is_complex()) and not bool(torch.isfinite(observed).all()):
                raise ValueError(f"Frozen policy {label} tensor {name!r} contains non-finite values.")
    try:
        student.load_state_dict(student_state, strict=True)
        teacher.load_state_dict(teacher_state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Frozen policy state dictionaries do not match the GICO architectures: {exc}") from exc


def _finite_array(value: object, *, label: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"Policy {label} must be finite with shape {shape}.")
    return array


def _validate_policy_evidence_records(
    policy_root: Path,
    source_manifest: Mapping[str, Any],
    *,
    policy_token: str,
    bundle_version: int,
) -> None:
    feature_groups = _json_mapping(
        policy_root / "reward-feature-groups.json",
        label="policy feature groups",
        require_canonical=True,
    )
    targets = _json_mapping(
        policy_root / "conditional-targets.json",
        label="policy conditional targets",
        require_canonical=True,
    )
    if set(feature_groups) != _FEATURE_GROUP_FIELDS or set(targets) != _TARGET_FIELDS:
        raise ValueError("Policy feature-group or target fields are incomplete or unexpected.")
    feature_artifact = f"image_{policy_token}_reward_feature_groups"
    if (
        feature_groups.get("artifact") != feature_artifact
        or re.fullmatch(re.escape(feature_artifact) + r"_v[1-9][0-9]*", str(feature_groups.get("protocol", ""))) is None
        or feature_groups.get("usage") != "reward_shrinkage_only_not_inference_context"
        or feature_groups.get("class_count") != 1_000
        or feature_groups.get("feature_dim") != 64
        or feature_groups.get("group_count") != 32
        or feature_groups.get("samples_per_class") != 64
    ):
        raise ValueError("Policy feature-group evidence is not the canonical ImageNet contract.")
    feature_identity = _validate_payload_semantic_identity(
        feature_groups,
        identity_field="feature_group_sha256",
        namespace=f"image-{policy_token}-reward-feature-groups",
        label="policy feature-group",
    )
    if feature_groups.get("feature_protocol_sha256") != source_manifest.get("feature_protocol_sha256"):
        raise ValueError("Policy feature groups disagree with the source feature protocol.")
    for field in ("source_panel_fingerprint", "feature_protocol_sha256", "real_feature_panel_sha256"):
        _require_digest(feature_groups.get(field), label=f"policy feature-group {field}")

    coordinates = _finite_array(
        feature_groups.get("class_coordinates"),
        label="class coordinates",
        shape=(1_000, 64),
    )
    assignments = np.asarray(feature_groups.get("group_assignments"))
    if (
        assignments.shape != (1_000,)
        or assignments.dtype.kind not in {"i", "u"}
        or not np.array_equal(np.unique(assignments), np.arange(32))
    ):
        raise ValueError("Policy feature-group assignments must cover 32 groups across 1,000 classes.")
    centroids = _finite_array(
        feature_groups.get("group_centroids"),
        label="group centroids",
        shape=(32, 64),
    )
    expected_centroids = np.stack([coordinates[assignments == group].mean(axis=0) for group in range(32)])
    if not np.allclose(centroids, expected_centroids, rtol=1e-10, atol=1e-10):
        raise ValueError("Policy feature-group centroids disagree with their assignments.")
    pca_center = np.asarray(feature_groups.get("pca_center"), dtype=np.float64)
    if pca_center.ndim != 1 or pca_center.size < 64 or not np.all(np.isfinite(pca_center)):
        raise ValueError("Policy PCA center must contain at least 64 finite values.")
    pca_components = _finite_array(
        feature_groups.get("pca_components"),
        label="PCA components",
        shape=(64, int(pca_center.size)),
    )
    pca_scales = _finite_array(feature_groups.get("pca_scales"), label="PCA scales", shape=(64,))
    if np.any(pca_scales <= 0.0):
        raise ValueError("Policy PCA scales must be strictly positive.")
    if not np.allclose(pca_components @ pca_components.T, np.eye(64), rtol=1e-7, atol=1e-7):
        raise ValueError("Policy PCA components must have orthonormal rows.")

    target_artifact = f"image_{policy_token}_conditional_targets"
    if (
        targets.get("artifact") != target_artifact
        or targets.get("protocol") != f"{target_artifact}_v{bundle_version}"
        or targets.get("conditioning") != "classwise_rewards_independent_of_inference_context"
        or targets.get("feature_group_usage") != "reward_shrinkage_only_not_inference_context"
        or targets.get("class_count") != 1_000
        or targets.get("density_bin_count") != 64
        or targets.get("target_nfes") != [2, 4, 8]
    ):
        raise ValueError("Policy conditional targets are not the canonical ImageNet contract.")
    target_identity = _validate_payload_semantic_identity(
        targets,
        identity_field="target_sha256",
        namespace=f"image-{policy_token}-conditional-targets-v{bundle_version}",
        label="policy conditional-target",
    )
    training = _require_mapping(source_manifest.get("training"), label="policy training record")
    if (
        source_manifest.get("feature_group_sha256") != feature_identity
        or training.get("feature_group_sha256") != feature_identity
        or targets.get("feature_group_sha256") != feature_identity
        or source_manifest.get("target_sha256") != target_identity
        or training.get("target_sha256") != target_identity
    ):
        raise ValueError("Policy feature-group and target identities disagree.")
    for field in (
        "backbone_model_key",
        "backbone_protocol_sha256",
        "backbone_checkpoint_sha256",
        "feature_protocol_sha256",
    ):
        if targets.get(field) != source_manifest.get(field):
            raise ValueError(f"Policy conditional targets disagree with source {field}.")

    schedule_support = _require_mapping(source_manifest.get("schedule_support"), label="policy schedule support")
    schedule_keys = targets.get("schedule_keys")
    schedule_count = targets.get("schedule_count")
    if (
        not isinstance(schedule_keys, list)
        or not schedule_keys
        or len(schedule_keys) != len(set(schedule_keys))
        or isinstance(schedule_count, bool)
        or schedule_count != len(schedule_keys)
        or schedule_support.get("schedule_keys") != schedule_keys
        or schedule_support.get("schedule_sha256s") != targets.get("schedule_sha256s")
        or schedule_support.get("density_mass_sha256s") != targets.get("density_mass_sha256s")
        or schedule_support.get("fixed_support_sha256") != targets.get("fixed_support_sha256")
        or schedule_support.get("reward_evidence_sha256") != targets.get("reward_evidence_sha256")
    ):
        raise ValueError("Policy schedule-support evidence is inconsistent.")
    schedule_hashes = targets.get("schedule_sha256s")
    if not isinstance(schedule_hashes, list) or len(schedule_hashes) != schedule_count:
        raise ValueError("Policy schedule identities have an invalid shape.")
    for value in schedule_hashes:
        _require_digest(value, label="policy schedule digest")
    for field in ("reward_evidence_sha256", "fixed_support_sha256"):
        _require_digest(targets.get(field), label=f"policy {field}")
    density_hashes = targets.get("density_mass_sha256s")
    if (
        not isinstance(density_hashes, list)
        or len(density_hashes) != 3
        or any(not isinstance(row, list) or len(row) != schedule_count for row in density_hashes)
    ):
        raise ValueError("Policy density-mass identities have an invalid shape.")
    for row in density_hashes:
        for value in row:
            _require_digest(value, label="policy density-mass digest")

    schedule_width = int(schedule_count)
    mass = _finite_array(targets.get("density_mass"), label="density targets", shape=(3, 1_000, 64))
    weights = _finite_array(
        targets.get("mixture_weights"),
        label="schedule mixture weights",
        shape=(3, 1_000, schedule_width),
    )
    _finite_array(targets.get("normalized_rewards"), label="normalized rewards", shape=weights.shape)
    errors = _finite_array(
        targets.get("jackknife_standard_errors"),
        label="jackknife standard errors",
        shape=weights.shape,
    )
    class_reliability = _finite_array(
        targets.get("class_reliability"),
        label="class reliability",
        shape=weights.shape,
    )
    group_reliability = _finite_array(
        targets.get("group_reliability"),
        label="group reliability",
        shape=weights.shape,
    )
    temperatures = _finite_array(
        targets.get("temperature_by_nfe"),
        label="temperatures",
        shape=(3,),
    )
    coefficients = _finite_array(
        targets.get("shrinkage_coefficients"),
        label="shrinkage coefficients",
        shape=(*weights.shape, 3),
    )
    if (
        np.any(mass < 0.0)
        or not np.allclose(mass.sum(axis=-1), 1.0, rtol=1e-7, atol=1e-7)
        or np.any(weights < 0.0)
        or not np.allclose(weights.sum(axis=-1), 1.0, rtol=1e-7, atol=1e-7)
        or np.any(errors < 0.0)
        or np.any(class_reliability < 0.0)
        or np.any(class_reliability > 1.0)
        or np.any(group_reliability < 0.0)
        or np.any(group_reliability > 1.0)
        or np.any(temperatures <= 0.0)
        or np.any(coefficients < 0.0)
        or np.any(coefficients > 1.0)
        or not np.allclose(coefficients.sum(axis=-1), 1.0, rtol=1e-8, atol=1e-8)
    ):
        raise ValueError("Policy target densities, mixtures, or shrinkage coefficients are invalid.")


def _validate_policy_file_bindings(policy_root: Path, source_manifest: Mapping[str, Any]) -> None:
    bindings = _require_mapping(source_manifest.get("files"), label="policy file bindings")
    expected_roles = set(_SOURCE_POLICY_FILES.values())
    if set(bindings) != expected_roles:
        raise ValueError(f"Policy file bindings must be exactly {sorted(expected_roles)}.")
    for filename, role in _SOURCE_POLICY_FILES.items():
        binding = _require_mapping(bindings.get(role), label=f"policy {role} file binding")
        source = policy_root / filename
        if not source.is_file() or source.stat().st_size <= 0:
            raise ValueError(f"Frozen policy source file is missing or empty: {filename}")
        if source.suffix.lower() == ".json":
            _json_mapping(source, label=f"policy {role}", require_canonical=True)
        if binding.get("filename") != filename:
            raise ValueError(f"Policy {role} binding names the wrong file.")
        if binding.get("size_bytes") != source.stat().st_size or binding.get("sha256") != sha256_file(source):
            raise ValueError(f"Policy {role} file does not match its manifest binding.")


def _validate_policy_semantic_bindings(policy_root: Path, source_manifest: Mapping[str, Any]) -> None:
    training = _require_mapping(source_manifest.get("training"), label="policy training record")
    student_identity = _policy_state_semantic_identity(
        policy_root / "student-state.pt",
        stored=source_manifest.get("student_state_sha256"),
        label="student",
        kind="model-state",
    )
    teacher_identity = _policy_state_semantic_identity(
        policy_root / "teacher-state.pt",
        stored=source_manifest.get("teacher_state_sha256"),
        label="teacher",
        kind="teacher-state",
    )
    if training.get("model_state_sha256") != student_identity:
        raise ValueError("Policy student state identity differs from its training record.")
    if training.get("teacher_state_sha256") != teacher_identity:
        raise ValueError("Policy teacher state identity differs from its training record.")
    observed_density_digest = sha256_file(policy_root / "class-density-table.npy")
    if (
        _require_digest(
            source_manifest.get("density_table_sha256"),
            label="policy density table digest",
        )
        != observed_density_digest
    ):
        raise ValueError("Policy density-table digest differs from its source file.")


def _validated_frozen_policy_source(
    root: Path,
) -> tuple[list[ArchiveEntry], dict[str, Any], Mapping[str, Any]]:
    metadata, source_manifest, policy_token, bundle_version = _frozen_policy_metadata(root)
    entries: list[ArchiveEntry] = []
    for filename, role in _ARCHIVED_POLICY_FILES.items():
        source = root / filename
        if not source.exists():
            raise ValueError(f"Frozen policy file is missing: {filename}")
        if not source.is_file() or source.stat().st_size <= 0:
            raise ValueError(f"Frozen policy file is missing or empty: {source}")
        entries.append(ArchiveEntry(source, f"policy/{filename}", role))
    _validate_policy_file_bindings(root, source_manifest)
    _validate_policy_semantic_bindings(root, source_manifest)
    arrays = _validate_policy_arrays(root, metadata)
    _validate_policy_context_binding(
        source_manifest,
        arrays,
        policy_token=policy_token,
    )
    _validate_policy_evidence_records(
        root,
        source_manifest,
        policy_token=policy_token,
        bundle_version=bundle_version,
    )
    _validate_policy_architecture(root, density_bin_count=int(metadata["model"]["density_bin_count"]))
    for filename in ("student-state.pt", "teacher-state.pt"):
        if (root / filename).stat().st_size < 1024:
            raise ValueError(f"Frozen policy state is too small to be valid: {filename}")
        _load_policy_state(root / filename, label=filename.removesuffix("-state.pt"))
    return entries, metadata, source_manifest


def package_frozen_gico_policy(
    *,
    policy_dir: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(Path(policy_dir).expanduser())))
    if not root.is_dir():
        raise ValueError(f"Policy bundle source is not a directory: {root}")
    entries, metadata, _source_manifest = _validated_frozen_policy_source(root)
    return write_deterministic_zip(
        entries,
        output_path,
        bundle_kind="frozen_gico_policy",
        metadata=metadata,
        overwrite=overwrite,
    )


def _named_checkpoint(value: str) -> NamedCheckpoint:
    name, separator, path = str(value).partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("checkpoints must use NAME=PATH")
    return NamedCheckpoint(name=name, path=Path(path))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or validate deterministic GenODE release archives.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("backbone-manifest")
    manifest_parser.add_argument("--manifest", required=True)
    manifest_parser.add_argument("--source-root", required=True)
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument("--expected-count", type=int)
    manifest_parser.add_argument("--without-support-files", action="store_true")
    manifest_parser.add_argument("--overwrite", action="store_true")

    named_parser = subparsers.add_parser("named-checkpoints")
    named_parser.add_argument("--checkpoint", action="append", type=_named_checkpoint, required=True)
    named_parser.add_argument("--output", required=True)
    named_parser.add_argument("--overwrite", action="store_true")

    policy_parser = subparsers.add_parser("gico-policy")
    policy_parser.add_argument("--policy-dir", required=True)
    policy_parser.add_argument("--output", required=True)
    policy_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--archive", required=True)

    args = parser.parse_args(argv)
    if args.command == "backbone-manifest":
        result = package_backbone_manifest_checkpoints(
            manifest_path=args.manifest,
            source_root=args.source_root,
            output_path=args.output,
            expected_count=args.expected_count,
            include_support_files=not args.without_support_files,
            overwrite=args.overwrite,
        )
    elif args.command == "named-checkpoints":
        result = package_named_checkpoints(
            args.checkpoint,
            args.output,
            overwrite=args.overwrite,
        )
    elif args.command == "gico-policy":
        result = package_frozen_gico_policy(
            policy_dir=args.policy_dir,
            output_path=args.output,
            overwrite=args.overwrite,
        )
    else:
        result = validate_deterministic_zip(args.archive)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "validate" and result.get("status") != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
