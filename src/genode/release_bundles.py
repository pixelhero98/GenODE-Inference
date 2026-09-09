from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from genode.artifacts.identity import canonical_json_bytes as identity_canonical_json_bytes
from genode.deterministic_archive import (
    ArchiveEntry,
    canonical_json_bytes,
    contains_local_filesystem_path,
    validate_deterministic_zip,
    write_deterministic_zip,
)

FROZEN_BACKBONE_COLLECTION_SCHEMA = "frozen_backbone_collection_v1"
_CHECKPOINT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}


@dataclass(frozen=True)
class NamedCheckpoint:
    name: str
    path: Path


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


def package_frozen_gico_policy(
    *,
    policy_dir: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Package a validated common GICO artifact without loading old architectures."""
    from genode.gico.policy import GICO_PROTOCOL, load_policy, load_teacher
    from genode.path_safety import is_link_or_reparse_point

    root = Path(os.path.abspath(os.fspath(Path(policy_dir).expanduser())))
    if is_link_or_reparse_point(root) or not root.is_dir():
        raise ValueError("Policy bundle source must be a regular directory.")
    paths = [root / "policy.pt", root / "manifest.json"]
    if any(is_link_or_reparse_point(path) or not path.is_file() for path in paths):
        raise ValueError("Unified GICO packaging requires regular policy.pt and manifest.json files.")
    manifest = _json_mapping(paths[1], label="GICO manifest")
    if manifest.get("protocol") != GICO_PROTOCOL:
        raise ValueError("Incompatible historical GICO artifact; package a newly fitted common policy.")
    policy_bytes = paths[0].read_bytes()
    if hashlib.sha256(policy_bytes).hexdigest() != manifest.get("policy_sha256"):
        raise ValueError("Policy artifact checksum mismatch.")
    payload = torch.load(io.BytesIO(policy_bytes), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("protocol") != GICO_PROTOCOL:
        raise ValueError("Incompatible historical GICO artifact; package a newly fitted common policy.")
    kinds = payload.get("metadata", {}).get("student_kinds", [])
    if not kinds or set(kinds) - {"deterministic", "stochastic"}:
        raise ValueError("GICO artifact must identify its trained student kinds.")
    for kind in kinds:
        policy = load_policy(root, student_kind=kind)
    teacher, _, _ = load_teacher(root)
    if any(not bool(torch.isfinite(value).all()) for value in teacher.state_dict().values()):
        raise ValueError("Policy teacher contains nonfinite parameters.")
    metadata = {
        "schema_version": GICO_PROTOCOL,
        "artifact_sha256": policy.artifact_sha256,
        "task": policy.metadata["task"],
        "backbone": policy.metadata["backbone"],
        "student_kinds": list(kinds),
        "architecture": payload["architecture"],
    }
    _assert_no_local_json_paths(payload["metadata"], label="policy metadata")
    return write_deterministic_zip(
        [
            ArchiveEntry(path, f"policy/{path.name}", "gico_policy" if path.suffix == ".pt" else "gico_manifest")
            for path in paths
        ],
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
