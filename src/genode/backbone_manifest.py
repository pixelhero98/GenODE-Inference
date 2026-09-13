"""Validate explicit backbone manifests and their retained checkpoint inputs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from genode.canonical_experiment_layout import (
    CANONICAL_CHECKPOINT_STEPS,
    SCENARIO_FAMILY_MOLECULE,
)
from genode.data.otflow_experiment_plan import FORECAST_FAMILY
from genode.data.otflow_paths import resolve_project_path
from genode.evaluation.fm_backbone_registry import BACKBONE_NAME_OTFLOW, BACKBONE_NAME_OTFLOW_MOLECULE

MOLECULE_FAMILY = SCENARIO_FAMILY_MOLECULE

TRAIN_BUDGET_STEPS = CANONICAL_CHECKPOINT_STEPS

PATH_FIELDS = ("checkpoint_path", "summary_path", "metadata_path")

ROOT_FIELDS = (
    "matrix_root",
    "otflow_reuse_root",
    "imported_backbone_root",
    "molecule_group_root",
    "molecule_backbone_root",
)

_EMBEDDED_WINDOWS_PATH = re.compile("(?<![A-Za-z0-9])(?:[A-Za-z]:[\\\\/]|\\\\\\\\[^\\\\/\\s]+[\\\\/][^\\\\/\\s]+)")

_EMBEDDED_POSIX_PATH = re.compile("(?<![:A-Za-z0-9/])/(?:[^/\\s]+/)+[^/\\s]+")

MIN_CHECKPOINT_SIZE_BYTES = 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_relative_path(manifest_path: Path, value: Any, *, path_base: str) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    raw = Path(value)
    if raw.is_absolute():
        return str(raw)
    base = (manifest_path.parent / str(path_base)).resolve()
    return str((base / value).resolve())


def load_portable_backbone_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    path_base = str(payload.get("path_base", "") or "").strip()
    if path_base:
        for artifact in payload.get("artifacts", []):
            if not isinstance(artifact, dict):
                continue
            for field in PATH_FIELDS:
                if field in artifact:
                    artifact[field] = _resolve_manifest_relative_path(resolved, artifact[field], path_base=path_base)
        for field in ROOT_FIELDS:
            if field in payload:
                payload[field] = _resolve_manifest_relative_path(resolved, payload[field], path_base=path_base)
    return payload


def _contains_local_marker(value: str) -> bool:
    return _EMBEDDED_WINDOWS_PATH.search(value) is not None or _EMBEDDED_POSIX_PATH.search(value) is not None


def _known_text_checkpoint_header(header: bytes) -> bool:
    stripped = header.lstrip().lower()
    return (
        stripped.startswith(b"version https://git-lfs.github.com/spec")
        or stripped.startswith(b"<!doctype html")
        or stripped.startswith(b"<html")
    )


def _checkpoint_label(artifact: Mapping[str, Any]) -> str:
    parts = [
        str(artifact.get("checkpoint_id", "") or "").strip(),
        str(artifact.get("benchmark_family", "") or "").strip(),
        str(artifact.get("dataset_key", "") or "").strip(),
        str(artifact.get("train_steps", "") or "").strip(),
    ]
    return "/".join(part for part in parts if part)


def _validate_artifact_checkpoint_integrity(artifact: Mapping[str, Any], checkpoint_path: Path) -> list[str]:
    label = _checkpoint_label(artifact)
    errors: list[str] = []
    if not checkpoint_path.exists():
        return [f"Provided artifact {label} is missing its checkpoint file."]
    if not checkpoint_path.is_file():
        return [f"Provided artifact {label} checkpoint path is not a regular file."]
    size_bytes = int(checkpoint_path.stat().st_size)
    if size_bytes < MIN_CHECKPOINT_SIZE_BYTES:
        errors.append(f"Provided artifact {label} checkpoint is too small to be valid: found {size_bytes} bytes.")
    with checkpoint_path.open("rb") as fh:
        header = fh.read(256)
    if _known_text_checkpoint_header(header):
        errors.append(f"Provided artifact {label} checkpoint looks like text or a pointer.")
    if errors:
        return errors
    if str(artifact.get("backbone_name", "")) == "otflow" and str(artifact.get("benchmark_family", "")) in {
        FORECAST_FAMILY
    }:
        try:
            from genode.evaluation.otflow_evaluation_support import load_otflow_checkpoint_payload

            load_otflow_checkpoint_payload(checkpoint_path, expected_identity=f"provided backbone artifact {label}")
        except Exception as exc:
            detail = (
                str(exc).replace(str(checkpoint_path), "checkpoint").replace(checkpoint_path.as_posix(), "checkpoint")
            )
            if _contains_local_marker(detail):
                detail = type(exc).__name__
            errors.append(f"Provided artifact {label} checkpoint is not loadable: {detail}")
    return errors


def _resolve_loaded_artifact_path(path_value: str) -> Path:
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw
    return resolve_project_path(str(path_value))


def validate_provided_backbone_manifest(
    manifest_path: str | Path, *, scenario_key: str = "", benchmark_family: str = ""
) -> dict[str, Any]:
    resolved_manifest = resolve_project_path(str(manifest_path))
    errors: list[str] = []
    if not resolved_manifest.exists():
        return {
            "status": "failed",
            "errors": ["Provided backbone manifest is missing."],
            "backbone_manifest_sha256": "",
            "artifact_count": 0,
        }
    backbone_manifest_sha256 = _sha256_file(resolved_manifest)
    manifest = load_portable_backbone_manifest(resolved_manifest)
    candidate_artifacts = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if str(artifact.get("status")) == "ready"
        and (not scenario_key or str(artifact.get("dataset_key")) == str(scenario_key))
        and (not benchmark_family or str(artifact.get("benchmark_family")) == str(benchmark_family))
    ]
    expected_backbone_name = (
        BACKBONE_NAME_OTFLOW_MOLECULE if benchmark_family == MOLECULE_FAMILY else BACKBONE_NAME_OTFLOW
    )
    wrong_backbone_names = sorted(
        {
            str(artifact.get("backbone_name", ""))
            for artifact in candidate_artifacts
            if str(artifact.get("backbone_name", "")) != expected_backbone_name
        }
    )
    if wrong_backbone_names:
        errors.append(
            f"Provided backbone artifacts for scenario={scenario_key!r}, family={benchmark_family!r} use backbone names {wrong_backbone_names}; expected {expected_backbone_name!r}."
        )
    matching_artifacts = [
        artifact for artifact in candidate_artifacts if str(artifact.get("backbone_name", "")) == expected_backbone_name
    ]
    if not matching_artifacts:
        errors.append(
            f"No ready provided backbone artifacts match scenario={scenario_key!r}, family={benchmark_family!r}, backbone={expected_backbone_name!r}."
        )
    expected_steps = {int(step) for step in TRAIN_BUDGET_STEPS}
    observed_steps = {int(artifact.get("train_steps", -1)) for artifact in matching_artifacts}
    if matching_artifacts and (not expected_steps.issubset(observed_steps)):
        errors.append(
            f"Provided backbone artifacts have train steps {sorted(observed_steps)}; expected at least {sorted(expected_steps)}."
        )
    lookup_counts: dict[tuple[Any, ...], int] = {}
    for artifact in matching_artifacts:
        key = (
            str(artifact.get("backbone_name", "")),
            str(artifact.get("benchmark_family", "")),
            str(artifact.get("dataset_key", "")),
            int(artifact.get("train_steps", -1)),
            str(artifact.get("member_key", "")) if benchmark_family == MOLECULE_FAMILY else "",
            str(artifact.get("stratum", "")) if benchmark_family == MOLECULE_FAMILY else "",
        )
        lookup_counts[key] = int(lookup_counts.get(key, 0)) + 1
    duplicate_keys = {key: count for key, count in lookup_counts.items() if count != 1}
    if duplicate_keys:
        first_key, first_count = next(iter(sorted(duplicate_keys.items(), key=lambda item: repr(item[0]))))
        errors.append(f"Provided backbone manifest has duplicate runtime lookup key {first_key} count={first_count}.")
    if matching_artifacts and benchmark_family != MOLECULE_FAMILY:
        for step in sorted(expected_steps):
            count = sum(1 for artifact in matching_artifacts if int(artifact.get("train_steps", -1)) == int(step))
            if count != 1:
                errors.append(
                    f"Provided temporal scenario has {count} ready {expected_backbone_name} artifacts for train_steps={step}; expected 1."
                )
    if benchmark_family == MOLECULE_FAMILY:
        members: dict[tuple[str, str, str], set[int]] = {}
        for artifact in matching_artifacts:
            member = (
                str(artifact.get("member_key", "")),
                str(artifact.get("stratum", "")),
                str(artifact.get("variant", "")),
            )
            members.setdefault(member, set()).add(int(artifact.get("train_steps", -1)))
        if matching_artifacts and len(members) != 6:
            errors.append(f"Provided molecule scenario has {len(members)} members; expected 6.")
        for member, member_steps in sorted(members.items()):
            if member_steps != expected_steps:
                errors.append(
                    f"Provided molecule member {'/'.join(member)} has train steps {sorted(member_steps)}; expected {sorted(expected_steps)}."
                )
    for artifact in matching_artifacts:
        for field in PATH_FIELDS:
            value = str(artifact.get(field, "") or "")
            if not value:
                continue
            resolved = _resolve_loaded_artifact_path(value)
            if not resolved.exists():
                errors.append(f"Provided artifact {artifact.get('checkpoint_id', '')} is missing {field}.")
            elif field == "checkpoint_path":
                errors.extend(_validate_artifact_checkpoint_integrity(artifact, resolved))
    return {
        "status": "complete" if not errors else "failed",
        "errors": errors,
        "backbone_manifest_sha256": backbone_manifest_sha256,
        "artifact_count": int(len(matching_artifacts)),
    }
