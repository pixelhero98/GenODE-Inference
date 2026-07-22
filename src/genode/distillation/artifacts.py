from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from genode.checkpoint_validation import (
    validate_locked_test_exclusion,
    validate_strict_integer,
)
from genode.path_safety import (
    MANIFEST_PARENT_PATH_BASE,
    portable_relative_path,
    resolve_manifest_path_base,
    resolve_portable_relative_path,
)
from genode.provenance import file_sha256


DEMONSTRATION_PROTOCOL = "flow_map_demonstrations"
CHECKPOINT_PROTOCOL = "endpoint_flow_map"
DEMONSTRATION_MANIFEST_NAME = "flow_map_demonstrations.json"
DEMONSTRATION_ARTIFACT_VERSION = 1
DEMONSTRATION_NON_TEST_SPLITS = frozenset(
    {"train", "training", "train_tuning", "validation", "validation_tuning"}
)
DEMONSTRATION_TRAINING_SPLITS = frozenset({"train", "training", "train_tuning"})
CONTEXT_BINDING_DOMAIN = "genode.flow_map.context_content"
IN_MEMORY_CONTEXT_SOURCE_DOMAIN = "genode.flow_map.context_source.in_memory"
CONTEXT_SOURCE_KINDS = frozenset({"in_memory", "npz"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def context_fingerprint(history: Any, condition: Any | None = None) -> str:
    """Hash the physical values that identify one conditioning context.

    Values are normalized to the model's little-endian float32 input precision
    before hashing so independently represented float32/float64 inputs have the
    same portable identity. Signed zero is normalized as well. Display labels
    such as ``context_id`` are deliberately excluded: renaming a sample must
    not bypass train/evaluation split checks.
    """

    digest = hashlib.sha256()
    digest.update(f"{CONTEXT_BINDING_DOMAIN}\0".encode("utf-8"))
    for name, value in (("history", history), ("condition", condition)):
        digest.update(f"{name}\0".encode("utf-8"))
        if value is None:
            digest.update(b"absent\0")
            continue
        array = np.asarray(value)
        if array.dtype.kind != "f" or not bool(np.isfinite(array).all()):
            raise ValueError(
                f"Context fingerprint {name} must contain finite real floating-point values."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = np.array(
                array,
                dtype=np.dtype("<f4"),
                order="C",
                copy=True,
            )
        if not bool(np.isfinite(normalized).all()):
            raise ValueError(
                f"Context fingerprint {name} cannot be represented at float32 model precision."
            )
        normalized[normalized == 0.0] = np.float32(0.0)
        shape = json.dumps(
            [int(size) for size in normalized.shape],
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(shape)
        digest.update(b"\0")
        digest.update(normalized.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def context_binding(context_fingerprints: Sequence[Any]) -> Dict[str, Any]:
    fingerprints = sorted(str(value).strip() for value in context_fingerprints)
    if not fingerprints or any(
        _SHA256_PATTERN.fullmatch(value) is None for value in fingerprints
    ):
        raise ValueError(
            "Context fingerprints must be non-empty lowercase SHA-256 digests."
        )
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Context fingerprints must be unique before context binding.")
    encoded = json.dumps(fingerprints, sort_keys=True, separators=(",", ":"))
    return {
        "algorithm": "sha256",
        "domain": CONTEXT_BINDING_DOMAIN,
        "context_count": len(fingerprints),
        "context_fingerprints": fingerprints,
        "set_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def in_memory_context_source_sha256(
    context_ids: Sequence[Any],
    context_fingerprints: Sequence[Any],
) -> str:
    """Return a semantic identity for contexts supplied directly in memory."""

    ids = [str(value) for value in context_ids]
    fingerprints = [str(value) for value in context_fingerprints]
    if len(ids) != len(fingerprints) or not ids:
        raise ValueError("In-memory context IDs and fingerprints must be aligned and non-empty.")
    if any(not value or value != value.strip() for value in ids):
        raise ValueError("In-memory context IDs must be non-empty and trimmed.")
    if len(set(ids)) != len(ids):
        raise ValueError("In-memory context IDs must be unique.")
    context_binding(fingerprints)
    payload = {
        "domain": IN_MEMORY_CONTEXT_SOURCE_DOMAIN,
        "contexts": [
            {"context_id": context_id, "context_fingerprint": fingerprint}
            for context_id, fingerprint in zip(ids, fingerprints, strict=True)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_context_binding(payload: Mapping[str, Any]) -> Dict[str, Any]:
    binding = dict(payload)
    required = {
        "algorithm",
        "domain",
        "context_count",
        "context_fingerprints",
        "set_sha256",
    }
    if set(binding) != required:
        missing = sorted(required - set(binding))
        extra = sorted(set(binding) - required)
        raise ValueError(
            "Context binding fields are invalid; "
            f"missing={missing}, extra={extra}."
        )
    if binding.get("algorithm") != "sha256" or binding.get("domain") != CONTEXT_BINDING_DOMAIN:
        raise ValueError("Context binding has an unsupported algorithm or domain.")
    raw_fingerprints = binding.get("context_fingerprints")
    if not isinstance(raw_fingerprints, Sequence) or isinstance(
        raw_fingerprints, (str, bytes)
    ):
        raise ValueError("Context binding context_fingerprints must be a list.")
    fingerprints = [str(value) for value in raw_fingerprints]
    if not fingerprints or any(
        _SHA256_PATTERN.fullmatch(value) is None for value in fingerprints
    ):
        raise ValueError("Context binding contains invalid context fingerprints.")
    if fingerprints != sorted(set(fingerprints)):
        raise ValueError("Context binding fingerprints must be unique and sorted.")
    if validate_strict_integer(
        binding.get("context_count"),
        label="Context binding context_count",
        minimum=1,
    ) != len(fingerprints):
        raise ValueError("Context binding count does not match context_fingerprints.")
    encoded = json.dumps(fingerprints, sort_keys=True, separators=(",", ":"))
    expected_set_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if str(binding.get("set_sha256", "")) != expected_set_hash:
        raise ValueError("Context binding set_sha256 does not match context_fingerprints.")
    return {
        "algorithm": "sha256",
        "domain": CONTEXT_BINDING_DOMAIN,
        "context_count": len(fingerprints),
        "context_fingerprints": fingerprints,
        "set_sha256": expected_set_hash,
    }


def _context_identities_from_records(
    root: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    fingerprints: list[str] = []
    context_ids: list[str] = []
    for record in records:
        resolved = resolve_portable_relative_path(
            root,
            str(record.get("path", "")),
            label="context shard path",
            reject_links=True,
        )
        with np.load(resolved, allow_pickle=False) as payload:
            required = {"context_id", "context_fingerprint"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"Context shards are missing arrays: {missing}.")
            raw_ids = np.asarray(payload["context_id"])
            values = np.asarray(payload["context_fingerprint"])
            if (
                raw_ids.ndim != 1
                or raw_ids.dtype.kind not in {"U", "S"}
                or values.ndim != 1
                or values.dtype.kind not in {"U", "S"}
                or raw_ids.shape != values.shape
            ):
                raise ValueError(
                    "Context shard IDs and fingerprints must be aligned string vectors."
                )
            context_ids.extend(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in raw_ids.tolist()
            )
            fingerprints.extend(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in values.tolist()
            )
    if any(not value.strip() or value != value.strip() for value in context_ids):
        raise ValueError("Context shard context_id values must be non-empty and trimmed.")
    if len(set(context_ids)) != len(context_ids):
        raise ValueError("Context shard context_id values must be globally unique.")
    return context_ids, fingerprints


def _resolve_context_source_metadata(
    metadata: Mapping[str, Any],
    *,
    context_ids: Sequence[str],
    context_fingerprints: Sequence[str],
) -> Dict[str, Any]:
    resolved = dict(metadata)
    kind = str(resolved.get("contexts_source_kind", "in_memory")).strip()
    if kind not in CONTEXT_SOURCE_KINDS:
        raise ValueError(
            "Demonstration metadata contexts_source_kind must be 'in_memory' or 'npz'."
        )
    source_sha256 = str(resolved.get("contexts_source_sha256", "")).strip()
    if kind == "in_memory":
        expected = in_memory_context_source_sha256(context_ids, context_fingerprints)
        if source_sha256 and source_sha256 != expected:
            raise ValueError(
                "Demonstration in-memory context source hash does not match its context shards."
            )
        source_sha256 = expected
    if _SHA256_PATTERN.fullmatch(source_sha256) is None:
        raise ValueError(
            "Demonstration metadata contexts_source_sha256 must be a lowercase SHA-256 digest."
        )
    resolved["contexts_source_kind"] = kind
    resolved["contexts_source_sha256"] = source_sha256
    return resolved


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_value) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"Array {name!r} may not use object dtype.")
    if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
        raise ValueError(f"Array {name!r} contains non-finite values.")
    return array


def write_npz_shard(
    root: str | Path,
    relative_path: str,
    arrays: Mapping[str, Any],
) -> Dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    relative = portable_relative_path(relative_path, label="shard path")
    if relative.suffix != ".npz":
        raise ValueError(f"Demonstration shards must use .npz, got {relative.as_posix()!r}.")
    target = resolve_portable_relative_path(root_path, relative.as_posix(), label="shard path", reject_links=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not arrays:
        raise ValueError("A demonstration shard must contain at least one array.")
    normalized = {str(name): _validated_array(str(name), value) for name, value in arrays.items()}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=".tmp.npz",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **normalized)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": relative.as_posix(),
        "size_bytes": int(target.stat().st_size),
        "sha256": file_sha256(target),
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in sorted(normalized.items())
        },
    }


def write_demonstration_manifest(
    output_dir: str | Path,
    *,
    context_shards: Sequence[Mapping[str, Any]],
    trajectory_shards: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not context_shards:
        raise ValueError("At least one context shard is required.")
    if not trajectory_shards:
        raise ValueError("At least one trajectory shard is required.")
    portable_context_shards = [
        _portable_shard_record(root, record, label="context shard")
        for record in context_shards
    ]
    portable_trajectory_shards = [
        _portable_shard_record(root, record, label="trajectory shard")
        for record in trajectory_shards
    ]
    context_ids, context_fingerprints = _context_identities_from_records(
        root, portable_context_shards
    )
    resolved_metadata = _resolve_context_source_metadata(
        metadata,
        context_ids=context_ids,
        context_fingerprints=context_fingerprints,
    )
    derived_binding = context_binding(context_fingerprints)
    declared_binding = resolved_metadata.get("context_binding")
    if declared_binding is not None and validate_context_binding(declared_binding) != derived_binding:
        raise ValueError("Demonstration context binding does not match context shards.")
    resolved_metadata["context_binding"] = derived_binding
    _validate_demonstration_metadata(resolved_metadata)
    manifest_path = root / DEMONSTRATION_MANIFEST_NAME
    payload = {
        "protocol": DEMONSTRATION_PROTOCOL,
        "path_base": MANIFEST_PARENT_PATH_BASE,
        "metadata": resolved_metadata,
        "context_shards": portable_context_shards,
        "trajectory_shards": portable_trajectory_shards,
    }
    write_json(manifest_path, payload)
    return manifest_path


def _portable_shard_record(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Dict[str, Any]:
    core_fields = {"path", "size_bytes", "sha256", "arrays"}
    portable_metadata_fields = {
        "solver_key",
        "target_nfe",
        "context_start",
        "context_stop",
    }
    allowed_fields = core_fields | portable_metadata_fields
    unknown = sorted(set(record) - allowed_fields - {"resolved_path"})
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {unknown}.")
    portable = {
        field: record.get(field)
        for field in allowed_fields
        if field in record or field in core_fields
    }
    _validate_shard_record(root, portable, label=label)
    return portable


def _validate_shard_record(root: Path, record: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    path_value = str(record.get("path", ""))
    resolved = resolve_portable_relative_path(root, path_value, label=f"{label} path", reject_links=True)
    if not resolved.is_file():
        raise ValueError(f"Missing {label}: {path_value!r}.")
    expected_size = validate_strict_integer(
        record.get("size_bytes"),
        label=f"{label} size_bytes",
        minimum=0,
    )
    if expected_size != int(resolved.stat().st_size):
        raise ValueError(
            f"{label} size mismatch for {path_value!r}: expected {expected_size}, "
            f"found {int(resolved.stat().st_size)}."
        )
    expected_hash = str(record.get("sha256", ""))
    actual_hash = file_sha256(resolved)
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(f"{label} checksum mismatch for {path_value!r}.")
    try:
        with np.load(resolved, allow_pickle=False) as payload:
            actual_arrays: dict[str, dict[str, Any]] = {}
            for name in payload.files:
                array = _validated_array(str(name), payload[name])
                actual_arrays[str(name)] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid {label} {path_value!r}: {exc}") from exc
    raw_expected_arrays = record.get("arrays")
    if not isinstance(raw_expected_arrays, Mapping):
        raise ValueError(f"{label} arrays must be an object.")
    expected_arrays: dict[str, dict[str, Any]] = {}
    for raw_name, raw_schema in raw_expected_arrays.items():
        name = str(raw_name)
        if not name or name != name.strip() or not isinstance(raw_schema, Mapping):
            raise ValueError(f"{label} contains an invalid array schema for {raw_name!r}.")
        if set(raw_schema) != {"shape", "dtype"}:
            raise ValueError(
                f"{label} array {name!r} requires exactly shape and dtype metadata."
            )
        raw_shape = raw_schema.get("shape")
        if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
            raise ValueError(f"{label} array {name!r} shape must be an integer sequence.")
        shape = [
            validate_strict_integer(
                value,
                label=f"{label} array {name!r} shape[{index}]",
                minimum=0,
            )
            for index, value in enumerate(raw_shape)
        ]
        dtype = raw_schema.get("dtype")
        if not isinstance(dtype, str) or not dtype or dtype != dtype.strip():
            raise ValueError(f"{label} array {name!r} dtype must be a non-empty string.")
        expected_arrays[name] = {"shape": shape, "dtype": dtype}
    if expected_arrays != actual_arrays:
        raise ValueError(f"{label} array schema mismatch for {path_value!r}.")
    return {**dict(record), "resolved_path": resolved}


def _validate_demonstration_metadata(metadata: Mapping[str, Any]) -> None:
    if validate_strict_integer(
        metadata.get("artifact_version"),
        label="Demonstration artifact_version",
        minimum=1,
    ) != DEMONSTRATION_ARTIFACT_VERSION:
        raise ValueError(
            "Unsupported demonstration artifact version "
            f"{metadata.get('artifact_version')!r}; expected {DEMONSTRATION_ARTIFACT_VERSION}."
        )
    split_phase = str(metadata.get("split_phase", "")).strip()
    if split_phase not in DEMONSTRATION_NON_TEST_SPLITS:
        raise ValueError(
            "Demonstration metadata split_phase must identify a non-test training or "
            f"validation split, got {split_phase!r}."
        )
    for name in ("scenario_key", "benchmark_family"):
        if not str(metadata.get(name, "")).strip():
            raise ValueError(f"Demonstration metadata {name!r} may not be empty.")
    if not str(metadata.get("context_embedding_kind", "") or "").strip():
        raise ValueError(
            "Demonstration metadata requires an explicit context_embedding_kind."
        )
    integer_metadata = {
        name: validate_strict_integer(
            metadata.get(name),
            label=f"Demonstration metadata {name!r}",
            minimum=1,
        )
        for name in ("context_count", "rollouts_per_context", "sample_state_dim")
    }
    integer_metadata["density_bin_count"] = validate_strict_integer(
        metadata.get("density_bin_count"),
        label="Demonstration metadata 'density_bin_count'",
        minimum=2,
    )
    validate_strict_integer(
        metadata.get("collection_seed"),
        label="Demonstration metadata 'collection_seed'",
        minimum=0,
    )
    settings = metadata.get("settings")
    if not isinstance(settings, Sequence) or isinstance(settings, (str, bytes)) or not settings:
        raise ValueError("Demonstration metadata settings must be a non-empty list.")
    setting_keys: list[tuple[str, int]] = []
    for item in settings:
        if not isinstance(item, Mapping):
            raise ValueError("Every demonstration setting must be an object.")
        solver_key = str(item.get("solver_key", "")).strip()
        target_nfe = validate_strict_integer(
            item.get("target_nfe"),
            label="Demonstration setting target_nfe",
            minimum=1,
        )
        if not solver_key:
            raise ValueError("Every demonstration setting requires solver_key and positive target_nfe.")
        setting_keys.append((solver_key, target_nfe))
    if len(set(setting_keys)) != len(setting_keys):
        raise ValueError("Demonstration metadata settings may not contain duplicates.")
    for name in ("backbone_checkpoint_sha256", "gipo_checkpoint_sha256"):
        value = str(metadata.get(name, ""))
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"Demonstration metadata {name!r} must be a lowercase SHA-256 digest.")
    source_kind = str(metadata.get("contexts_source_kind", "")).strip()
    if source_kind not in CONTEXT_SOURCE_KINDS:
        raise ValueError(
            "Demonstration metadata contexts_source_kind must be 'in_memory' or 'npz'."
        )
    if _SHA256_PATTERN.fullmatch(
        str(metadata.get("contexts_source_sha256", ""))
    ) is None:
        raise ValueError(
            "Demonstration metadata contexts_source_sha256 must be a lowercase SHA-256 digest."
        )
    validate_locked_test_exclusion(
        metadata,
        label="Demonstration metadata",
        required_root_keys=("locked_test_used",),
    )
    binding = metadata.get("context_binding")
    if binding is not None:
        if not isinstance(binding, Mapping):
            raise ValueError("Demonstration metadata context_binding must be an object.")
        validated_binding = validate_context_binding(binding)
        if validated_binding["context_count"] != integer_metadata["context_count"]:
            raise ValueError(
                "Demonstration context binding count does not match metadata context_count."
            )


def load_demonstration_manifest(path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read demonstration manifest {manifest_path.name}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Demonstration manifest must contain a JSON object.")
    if str(payload.get("protocol", "")) != DEMONSTRATION_PROTOCOL:
        raise ValueError(
            f"Unsupported demonstration protocol {payload.get('protocol')!r}; "
            f"expected {DEMONSTRATION_PROTOCOL!r}."
        )
    root = resolve_manifest_path_base(manifest_path, payload.get("path_base"))
    metadata = dict(payload.get("metadata", {}))
    _validate_demonstration_metadata(metadata)
    raw_contexts = payload.get("context_shards", [])
    raw_trajectories = payload.get("trajectory_shards", [])
    if (
        not isinstance(raw_contexts, Sequence)
        or isinstance(raw_contexts, (str, bytes))
        or not all(isinstance(record, Mapping) for record in raw_contexts)
    ):
        raise ValueError("Demonstration context_shards must be a list of objects.")
    if (
        not isinstance(raw_trajectories, Sequence)
        or isinstance(raw_trajectories, (str, bytes))
        or not all(isinstance(record, Mapping) for record in raw_trajectories)
    ):
        raise ValueError("Demonstration trajectory_shards must be a list of objects.")
    raw_records = [*raw_contexts, *raw_trajectories]
    paths = [str(record.get("path", "")) for record in raw_records]
    if len(set(paths)) != len(paths):
        raise ValueError("Demonstration shard paths must be unique.")
    contexts = [
        _validate_shard_record(root, record, label="context shard")
        for record in raw_contexts
    ]
    context_ids, context_fingerprints = _context_identities_from_records(root, contexts)
    metadata = _resolve_context_source_metadata(
        metadata,
        context_ids=context_ids,
        context_fingerprints=context_fingerprints,
    )
    derived_binding = context_binding(context_fingerprints)
    declared_binding = metadata.get("context_binding")
    if declared_binding is not None and validate_context_binding(declared_binding) != derived_binding:
        raise ValueError("Demonstration context binding does not match context shards.")
    metadata["context_binding"] = derived_binding
    _validate_demonstration_metadata(metadata)
    trajectories = [
        _validate_shard_record(root, record, label="trajectory shard")
        for record in raw_trajectories
    ]
    if not contexts or not trajectories:
        raise ValueError("Demonstration manifest requires context and trajectory shards.")
    return {
        **dict(payload),
        "metadata": metadata,
        "manifest_path": manifest_path,
        "root": root,
        "context_shards": contexts,
        "trajectory_shards": trajectories,
    }


__all__ = [
    "CONTEXT_SOURCE_KINDS",
    "CONTEXT_BINDING_DOMAIN",
    "CHECKPOINT_PROTOCOL",
    "DEMONSTRATION_ARTIFACT_VERSION",
    "DEMONSTRATION_MANIFEST_NAME",
    "DEMONSTRATION_PROTOCOL",
    "DEMONSTRATION_TRAINING_SPLITS",
    "DEMONSTRATION_NON_TEST_SPLITS",
    "context_binding",
    "context_fingerprint",
    "in_memory_context_source_sha256",
    "load_demonstration_manifest",
    "validate_context_binding",
    "write_demonstration_manifest",
    "write_json",
    "write_npz_shard",
]
