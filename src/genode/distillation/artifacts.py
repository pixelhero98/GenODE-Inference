from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Mapping, Sequence

import numpy as np

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
DEMONSTRATION_TRAINING_SPLITS = frozenset(
    {"train", "training", "train_tuning", "validation", "validation_tuning"}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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
    _validate_demonstration_metadata(metadata)
    manifest_path = root / DEMONSTRATION_MANIFEST_NAME
    payload = {
        "protocol": DEMONSTRATION_PROTOCOL,
        "path_base": MANIFEST_PARENT_PATH_BASE,
        "metadata": dict(metadata),
        "context_shards": [dict(record) for record in context_shards],
        "trajectory_shards": [dict(record) for record in trajectory_shards],
    }
    write_json(manifest_path, payload)
    return manifest_path


def _validate_shard_record(root: Path, record: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    path_value = str(record.get("path", ""))
    resolved = resolve_portable_relative_path(root, path_value, label=f"{label} path", reject_links=True)
    if not resolved.is_file():
        raise ValueError(f"Missing {label}: {path_value!r}.")
    expected_size = int(record.get("size_bytes", -1))
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
    expected_arrays = dict(record.get("arrays", {}))
    if expected_arrays != actual_arrays:
        raise ValueError(f"{label} array schema mismatch for {path_value!r}.")
    return {**dict(record), "resolved_path": resolved}


def _validate_demonstration_metadata(metadata: Mapping[str, Any]) -> None:
    if int(metadata.get("artifact_version", -1)) != DEMONSTRATION_ARTIFACT_VERSION:
        raise ValueError(
            "Unsupported demonstration artifact version "
            f"{metadata.get('artifact_version')!r}; expected {DEMONSTRATION_ARTIFACT_VERSION}."
        )
    split_phase = str(metadata.get("split_phase", "")).strip()
    if split_phase not in DEMONSTRATION_TRAINING_SPLITS:
        raise ValueError(
            "Demonstration metadata split_phase must identify a non-test training or "
            f"validation split, got {split_phase!r}."
        )
    for name in ("scenario_key", "benchmark_family"):
        if not str(metadata.get(name, "")).strip():
            raise ValueError(f"Demonstration metadata {name!r} may not be empty.")
    for name in ("context_count", "rollouts_per_context", "sample_state_dim"):
        try:
            value = int(metadata.get(name, 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Demonstration metadata {name!r} must be a positive integer.") from exc
        if value <= 0:
            raise ValueError(f"Demonstration metadata {name!r} must be a positive integer.")
    if int(metadata.get("density_bin_count", 0)) <= 1:
        raise ValueError("Demonstration metadata density_bin_count must be greater than one.")
    settings = metadata.get("settings")
    if not isinstance(settings, Sequence) or isinstance(settings, (str, bytes)) or not settings:
        raise ValueError("Demonstration metadata settings must be a non-empty list.")
    setting_keys: list[tuple[str, int]] = []
    for item in settings:
        if not isinstance(item, Mapping):
            raise ValueError("Every demonstration setting must be an object.")
        solver_key = str(item.get("solver_key", "")).strip()
        try:
            target_nfe = int(item.get("target_nfe", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Every demonstration setting requires an integer target_nfe.") from exc
        if not solver_key or target_nfe <= 0:
            raise ValueError("Every demonstration setting requires solver_key and positive target_nfe.")
        setting_keys.append((solver_key, target_nfe))
    if len(set(setting_keys)) != len(setting_keys):
        raise ValueError("Demonstration metadata settings may not contain duplicates.")
    for name in ("backbone_checkpoint_sha256", "gipo_checkpoint_sha256"):
        value = str(metadata.get(name, ""))
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"Demonstration metadata {name!r} must be a lowercase SHA-256 digest.")
    if bool(metadata.get("locked_test_used", True)):
        raise ValueError("Demonstration metadata must explicitly declare locked_test_used=false.")


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
    trajectories = [
        _validate_shard_record(root, record, label="trajectory shard")
        for record in raw_trajectories
    ]
    if not contexts or not trajectories:
        raise ValueError("Demonstration manifest requires context and trajectory shards.")
    return {
        **dict(payload),
        "manifest_path": manifest_path,
        "root": root,
        "context_shards": contexts,
        "trajectory_shards": trajectories,
    }


__all__ = [
    "CHECKPOINT_PROTOCOL",
    "DEMONSTRATION_ARTIFACT_VERSION",
    "DEMONSTRATION_MANIFEST_NAME",
    "DEMONSTRATION_PROTOCOL",
    "DEMONSTRATION_TRAINING_SPLITS",
    "load_demonstration_manifest",
    "write_demonstration_manifest",
    "write_json",
    "write_npz_shard",
]
