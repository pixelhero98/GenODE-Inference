from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from genode.checkpoint_validation import validate_strict_integer
from genode.distillation.artifacts import (
    DEMONSTRATION_ARTIFACT_VERSION,
    DEMONSTRATION_MANIFEST_NAME,
    DEMONSTRATION_TRAINING_SPLITS,
    context_fingerprint,
    in_memory_context_source_sha256,
    load_demonstration_manifest,
    write_demonstration_manifest,
    write_npz_shard,
)
from genode.data.otflow_paths import resolve_project_path
from genode.distillation.gipo_policy import GIPOSchedulePolicy, load_gipo_schedule_policy
from genode.evaluation.otflow_evaluation_support import load_checkpoint_model
from genode.gipo.density_representation import DENSITY_BIN_COUNT
from genode.models.conditioning import ConditioningCache
from genode.models.otflow_model import OTFlow
from genode.provenance import file_sha256
from genode.path_safety import (
    first_link_or_reparse_component,
    is_link_or_reparse_point,
)
from genode.runtime import resolve_torch_device
from genode.solver_protocol import FlowTrajectory, SUPPORTED_SOLVER_KEYS, normalize_solver_nfe_fields


DEFAULT_DISTILLATION_NFES = (4, 6, 8, 10, 12, 14, 16, 20)
_PROMOTION_JOURNAL_PROTOCOL = "flow_map_demonstration_promotion"
_PROMOTION_JOURNAL_VERSION = 1
_COLLECTION_LOCK_MARKER = b"1"
_PROMOTION_JOURNAL_FIELDS = {
    "protocol",
    "version",
    "target_name",
    "staging_name",
    "staging_manifest_sha256",
    "previous_kind",
    "backup_name",
    "backup_manifest_sha256",
}


@dataclass(frozen=True, order=True)
class DistillationSetting:
    solver_key: str
    target_nfe: int

    def __post_init__(self) -> None:
        normalized = normalize_solver_nfe_fields(
            self.solver_key,
            self.target_nfe,
            source="distillation setting",
        )
        object.__setattr__(self, "solver_key", normalized.solver_key)
        object.__setattr__(self, "target_nfe", normalized.target_nfe)


@dataclass(frozen=True)
class DistillationContexts:
    context_ids: tuple[str, ...]
    histories: torch.Tensor
    conditions: torch.Tensor | None = None

    def validate(self) -> "DistillationContexts":
        if not self.context_ids:
            raise ValueError("At least one context is required for demonstration collection.")
        normalized_ids = tuple(str(value).strip() for value in self.context_ids)
        if any(str(value) != normalized for value, normalized in zip(self.context_ids, normalized_ids, strict=True)):
            raise ValueError("context_ids must not contain leading or trailing whitespace.")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("context_ids must be unique.")
        if any(not value for value in normalized_ids):
            raise ValueError("context_ids may not be empty.")
        if self.histories.ndim != 3:
            raise ValueError(
                "histories must have shape [contexts, history_steps, features], "
                f"got {tuple(self.histories.shape)}."
            )
        if int(self.histories.shape[0]) != len(self.context_ids):
            raise ValueError("context_ids and histories must have the same first dimension.")
        if not self.histories.is_floating_point() or not torch.isfinite(self.histories).all():
            raise ValueError("histories must contain finite floating-point values.")
        if self.conditions is not None:
            if self.conditions.ndim != 2 or int(self.conditions.shape[0]) != len(self.context_ids):
                raise ValueError("conditions must have shape [contexts, condition_features].")
            if not self.conditions.is_floating_point() or not torch.isfinite(self.conditions).all():
                raise ValueError("conditions must contain finite floating-point values.")
        return self

    def content_fingerprints(self) -> tuple[str, ...]:
        """Return label-independent identities for the physical contexts."""

        self.validate()
        fingerprints = tuple(
            context_fingerprint(
                self.histories[index].detach().cpu().numpy(),
                None
                if self.conditions is None
                else self.conditions[index].detach().cpu().numpy(),
            )
            for index in range(len(self.context_ids))
        )
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError(
                "Physical demonstration contexts must be unique even when context_ids differ."
            )
        return fingerprints


@dataclass(frozen=True)
class _CommonContextRollout:
    context_index: int
    seeds: tuple[int, ...]
    initial_state: torch.Tensor
    conditioning_cache: ConditioningCache
    context_embedding: torch.Tensor


def default_distillation_settings() -> tuple[DistillationSetting, ...]:
    return tuple(
        DistillationSetting(solver_key, target_nfe)
        for solver_key in SUPPORTED_SOLVER_KEYS
        for target_nfe in DEFAULT_DISTILLATION_NFES
    )


def parse_distillation_settings(value: str) -> tuple[DistillationSetting, ...]:
    text = str(value).strip()
    if not text:
        return default_distillation_settings()
    settings: list[DistillationSetting] = []
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid setting {item!r}; expected comma-separated solver_key:target_nfe values."
            )
        settings.append(DistillationSetting(parts[0].strip(), int(parts[1])))
    if len(set(settings)) != len(settings):
        raise ValueError("Distillation settings may not contain duplicates.")
    return tuple(settings)


def _cache_to_cpu(cache: ConditioningCache) -> dict[str, torch.Tensor | None]:
    return {
        "ctx_tokens": cache.ctx_tokens.detach().cpu(),
        "ctx_summary": cache.ctx_summary.detach().cpu(),
        "summary": cache.summary.detach().cpu(),
        "cond_emb": None if cache.cond_emb is None else cache.cond_emb.detach().cpu(),
    }


def _concatenate_caches(caches: Sequence[dict[str, torch.Tensor | None]]) -> dict[str, torch.Tensor | None]:
    if not caches:
        raise ValueError("Cannot concatenate an empty context cache list.")
    output: dict[str, torch.Tensor | None] = {}
    for key in ("ctx_tokens", "ctx_summary", "summary", "cond_emb"):
        values = [cache[key] for cache in caches]
        if all(value is None for value in values):
            output[key] = None
            continue
        if any(value is None for value in values):
            raise ValueError(f"Context cache field {key!r} is inconsistently present.")
        tensors = [value for value in values if value is not None]
        output[key] = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
    return output


def _slice_cache(
    cache: dict[str, torch.Tensor | None],
    index: int,
    *,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ConditioningCache:
    def repeated(name: str) -> torch.Tensor | None:
        value = cache[name]
        if value is None:
            return None
        selected = value[index : index + 1].to(device=device, dtype=dtype)
        return selected.expand(repeats, *selected.shape[1:]).contiguous()

    ctx_tokens = repeated("ctx_tokens")
    ctx_summary = repeated("ctx_summary")
    summary = repeated("summary")
    if ctx_tokens is None or ctx_summary is None or summary is None:
        raise ValueError("Context cache is missing required backbone fields.")
    return ConditioningCache(
        ctx_tokens=ctx_tokens,
        ctx_summary=ctx_summary,
        summary=summary,
        cond_emb=repeated("cond_emb"),
    )


def _repeat_cache(
    cache: ConditioningCache,
    *,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ConditioningCache:
    def repeated(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        selected = value.to(device=device, dtype=dtype)
        return selected.expand(repeats, *selected.shape[1:]).contiguous()

    ctx_tokens = repeated(cache.ctx_tokens)
    ctx_summary = repeated(cache.ctx_summary)
    summary = repeated(cache.summary)
    if ctx_tokens is None or ctx_summary is None or summary is None:
        raise ValueError("Context cache is missing required backbone fields.")
    return ConditioningCache(
        ctx_tokens=ctx_tokens,
        ctx_summary=ctx_summary,
        summary=summary,
        cond_emb=repeated(cache.cond_emb),
    )


def _noise_seed(
    base_seed: int,
    *,
    context_fingerprint: str,
    rollout_index: int,
) -> int:
    # Common random numbers make teacher endpoints directly comparable across
    # solver/NFE settings for the same context and rollout.
    encoded = (
        f"{int(base_seed)}\0{context_fingerprint}\0{int(rollout_index)}"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _initial_states(
    *,
    state_dim: int,
    seeds: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows = []
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        rows.append(torch.randn((state_dim,), generator=generator, dtype=torch.float32))
    return torch.stack(rows, dim=0).to(device=device, dtype=dtype)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _promotion_journal_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.promotion.json")


def _collection_lock_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.collection.lock")


def _fsync_directory(directory: Path) -> None:
    """Flush a directory, with an explicitly best-effort Windows fallback."""

    best_effort = os.name == "nt"
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        if best_effort:
            return
        raise
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(descriptor_stat.st_mode):
            raise ValueError(f"Expected a directory while syncing {directory.name!r}.")
        os.fsync(descriptor)
    except OSError:
        if not best_effort:
            raise
    finally:
        os.close(descriptor)


def _fsync_parent_directory(path: Path) -> None:
    """Flush a changed directory entry; only Windows is best-effort."""

    _fsync_directory(path.parent)


def _fsync_regular_file(path: Path) -> None:
    if is_link_or_reparse_point(path) or not path.is_file():
        raise ValueError(f"Cannot sync non-regular managed file {path.name!r}.")
    # Windows requires a writable descriptor for ``fsync`` even though this
    # routine never changes the file contents.
    flags = os.O_RDWR
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(descriptor_stat.st_mode) or (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ) != (path_stat.st_dev, path_stat.st_ino):
            raise ValueError(f"Managed file {path.name!r} changed while it was opened.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_path(source: Path, destination: Path) -> None:
    os.replace(source, destination)
    _fsync_parent_directory(destination)
    if source.parent != destination.parent:
        _fsync_parent_directory(source)


def _install_path_no_replace(source: Path, destination: Path) -> None:
    try:
        os.rename(source, destination)
    except OSError as exc:
        if os.path.lexists(destination):
            raise FileExistsError(
                f"Refusing to overwrite concurrently created path {destination.name!r}."
            ) from exc
        raise
    _fsync_parent_directory(destination)
    if source.parent != destination.parent:
        _fsync_parent_directory(source)


def _unlink_path(path: Path, *, missing_ok: bool = False) -> None:
    existed = path.exists()
    path.unlink(missing_ok=missing_ok)
    if existed:
        _fsync_parent_directory(path)


def _acquire_platform_lock(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        return
    if os.name == "posix":
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    raise OSError(f"Unsupported platform for demonstration collection locking: {os.name!r}.")


@contextmanager
def _exclusive_collection_lock(target: Path) -> Iterator[None]:
    """Hold a cooperative, target-scoped interprocess collection lock.

    The persistent sidecar stays outside the promoted directory, and the OS
    releases its advisory lock when the descriptor closes or the process exits.
    This serializes collectors using this API; it cannot stop an unrelated
    process from deliberately deleting the sidecar or mutating the artifact.
    """

    if os.name not in {"nt", "posix"}:
        raise RuntimeError(
            f"Demonstration collection locking is unsupported on platform {os.name!r}."
        )
    lock_path = _collection_lock_path(target)
    if lock_path.exists() and (
        is_link_or_reparse_point(lock_path) or not lock_path.is_file()
    ):
        raise ValueError("Demonstration collection lock must be a regular file.")

    flags = os.O_RDWR
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        descriptor = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if is_link_or_reparse_point(lock_path) or not lock_path.is_file():
            raise ValueError("Demonstration collection lock must be a regular file.")
        descriptor = os.open(lock_path, flags)

    try:
        os.set_inheritable(descriptor, False)
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise ValueError("Demonstration collection lock must be a regular file.")
        if is_link_or_reparse_point(lock_path):
            raise ValueError("Demonstration collection lock may not be a link or reparse point.")
        path_stat = lock_path.stat(follow_symlinks=False)
        if (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ) != (path_stat.st_dev, path_stat.st_ino):
            raise ValueError("Demonstration collection lock changed while it was opened.")

        try:
            _acquire_platform_lock(descriptor)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise RuntimeError(
                    f"Another demonstration collector is already using target {target.name!r}."
                ) from exc
            raise RuntimeError(
                f"Could not safely acquire the demonstration lock for target {target.name!r}."
            ) from exc
        os.lseek(descriptor, 0, os.SEEK_SET)
        marker = os.read(descriptor, len(_COLLECTION_LOCK_MARKER) + 1)
        if not marker:
            os.lseek(descriptor, 0, os.SEEK_SET)
            written = os.write(descriptor, _COLLECTION_LOCK_MARKER)
            if written != len(_COLLECTION_LOCK_MARKER):
                raise OSError("Could not initialize demonstration collection lock.")
            os.fsync(descriptor)
            _fsync_parent_directory(lock_path)
        elif marker != _COLLECTION_LOCK_MARKER:
            raise ValueError(
                "Refusing to use an unrecognized demonstration collection lock file."
            )
        yield
    finally:
        # Closing the descriptor releases both POSIX flock and Windows byte-range
        # locks, including while unwinding an exception. The persistent sidecar
        # avoids an unlink/recreate race between collectors.
        os.close(descriptor)


def _validated_artifact_manifest_sha256(root: Path) -> str:
    """Validate a complete, exclusively managed artifact and return its manifest hash."""

    if is_link_or_reparse_point(root) or not root.is_dir():
        raise ValueError("Managed demonstration artifact must be a regular directory.")
    manifest_path = root / DEMONSTRATION_MANIFEST_NAME
    if is_link_or_reparse_point(manifest_path) or not manifest_path.is_file():
        raise ValueError(
            "Safe replacement requires an intact demonstration manifest; refusing partial output."
        )
    manifest = load_demonstration_manifest(manifest_path)
    managed_files = {
        manifest_path.resolve(),
        *(
            Path(record["resolved_path"]).resolve()
            for record in [*manifest["context_shards"], *manifest["trajectory_shards"]]
        ),
    }
    managed_directories: set[Path] = set()
    root_resolved = root.resolve()
    for managed_file in managed_files:
        parent = managed_file.parent
        while parent != root_resolved:
            if root_resolved not in parent.parents:
                raise ValueError("Demonstration manifest references a file outside its artifact.")
            managed_directories.add(parent)
            parent = parent.parent

    discovered_files: set[Path] = set()
    discovered_directories: set[Path] = set()
    for path in root.rglob("*"):
        if is_link_or_reparse_point(path):
            raise ValueError("Refusing to replace a demonstration directory containing links.")
        if path.is_file():
            discovered_files.add(path.resolve())
        elif path.is_dir():
            discovered_directories.add(path.resolve())
        else:
            raise ValueError(
                "Refusing to replace a demonstration directory containing special files."
            )
    if discovered_files != managed_files or discovered_directories != managed_directories:
        raise ValueError(
            "Refusing to replace a demonstration directory containing unmanaged paths."
        )
    return file_sha256(manifest_path)


def _fsync_managed_artifact(root: Path) -> str:
    """Validate and durably flush a complete staged artifact before journaling."""

    initial_hash = _validated_artifact_manifest_sha256(root)
    manifest_path = root / DEMONSTRATION_MANIFEST_NAME
    managed_files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix(),
    )
    ordered_files = [path for path in managed_files if path != manifest_path]
    ordered_files.append(manifest_path)
    for path in ordered_files:
        _fsync_regular_file(path)

    managed_directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(
        managed_directories,
        key=lambda path: (-len(path.parts), path.as_posix()),
    ):
        _fsync_directory(directory)
    _fsync_parent_directory(root)

    final_hash = _validated_artifact_manifest_sha256(root)
    if final_hash != initial_hash:
        raise ValueError("Staged demonstration artifact changed while it was being synced.")
    return final_hash


def _validate_output_root(root: Path, *, overwrite: bool) -> None:
    anchor = Path(root.anchor)
    indirect = first_link_or_reparse_component(root, root=anchor)
    if indirect is not None:
        raise ValueError(
            "Refusing a demonstration output through a symlink, junction, or "
            f"reparse point: {indirect}."
        )
    manifest_path = root / DEMONSTRATION_MANIFEST_NAME
    shards_path = root / "shards"
    if root.exists() and not root.is_dir():
        raise ValueError("Demonstration output must be a directory path.")
    if not manifest_path.exists() and not shards_path.exists():
        if root.exists() and any(root.iterdir()):
            raise ValueError("Refusing to write demonstrations into a non-empty unmanaged directory.")
        return
    if not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite an existing demonstration artifact in {root.name!r}."
        )
    _validated_artifact_manifest_sha256(root)


def _promotion_sibling(target: Path, name: object, *, kind: str) -> Path:
    text = str(name)
    expected_prefix = f".{target.name}.{kind}-"
    if not text or Path(text).name != text or not text.startswith(expected_prefix):
        raise ValueError(f"Promotion journal contains an invalid {kind} directory name.")
    path = target.parent / text
    if path == target or path == _promotion_journal_path(target):
        raise ValueError(f"Promotion journal contains an unsafe {kind} directory name.")
    if path.exists() and is_link_or_reparse_point(path):
        raise ValueError(f"Promotion journal {kind} path may not be a link or reparse point.")
    return path


def _write_promotion_journal(path: Path, record: Mapping[str, object]) -> None:
    if path.exists():
        raise ValueError(
            "Refusing to replace an existing promotion journal without recovering it first."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(record), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_path(temporary, path)
    finally:
        _unlink_path(temporary, missing_ok=True)


def _load_promotion_journal(target: Path) -> dict[str, object] | None:
    path = _promotion_journal_path(target)
    if not path.exists():
        return None
    if is_link_or_reparse_point(path) or not path.is_file():
        raise ValueError("Promotion journal must be a regular file.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read promotion journal safely: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != _PROMOTION_JOURNAL_FIELDS:
        raise ValueError("Promotion journal has an invalid schema.")
    if (
        payload.get("protocol") != _PROMOTION_JOURNAL_PROTOCOL
        or type(payload.get("version")) is not int
        or payload.get("version") != _PROMOTION_JOURNAL_VERSION
        or payload.get("target_name") != target.name
    ):
        raise ValueError("Promotion journal identity does not match the requested artifact.")
    previous_kind = payload.get("previous_kind")
    if previous_kind not in {"absent", "empty", "artifact"}:
        raise ValueError("Promotion journal contains an invalid previous artifact kind.")
    if not _is_sha256(payload.get("staging_manifest_sha256")):
        raise ValueError("Promotion journal contains an invalid staging manifest hash.")
    _promotion_sibling(target, payload.get("staging_name"), kind="staging")
    backup_name = payload.get("backup_name")
    backup_hash = payload.get("backup_manifest_sha256")
    if previous_kind == "absent":
        if backup_name != "" or backup_hash != "":
            raise ValueError("Promotion journal has an invalid absent-target backup.")
    else:
        _promotion_sibling(target, backup_name, kind="backup")
        if previous_kind == "artifact" and not _is_sha256(backup_hash):
            raise ValueError("Promotion journal contains an invalid backup manifest hash.")
        if previous_kind == "empty" and backup_hash != "":
            raise ValueError("Promotion journal has an invalid empty-directory backup hash.")
    return dict(payload)


def _require_empty_directory(path: Path, *, label: str) -> None:
    if is_link_or_reparse_point(path) or not path.is_dir() or any(path.iterdir()):
        raise ValueError(f"Promotion {label} must be an empty regular directory.")


def _cleanup_obsolete_artifact(root: Path, expected_manifest_sha256: str) -> None:
    """Remove an exact obsolete artifact, preserving partial or unknown remnants."""

    if not root.exists():
        return
    try:
        actual_hash = _validated_artifact_manifest_sha256(root)
    except (OSError, ValueError):
        return
    if actual_hash != expected_manifest_sha256:
        return
    try:
        shutil.rmtree(root)
    except OSError:
        return
    _fsync_parent_directory(root)


def _cleanup_obsolete_empty_directory(path: Path) -> None:
    if not path.exists():
        return
    try:
        _require_empty_directory(path, label="backup")
        path.rmdir()
    except (OSError, ValueError):
        return
    _fsync_parent_directory(path)


def _cleanup_obsolete_promotion_paths(
    *,
    staging: Path,
    staged_hash: str,
    backup: Path | None,
    backup_hash: str,
    previous_kind: str,
) -> None:
    _cleanup_obsolete_artifact(staging, staged_hash)
    if backup is None:
        return
    if previous_kind == "artifact":
        _cleanup_obsolete_artifact(backup, backup_hash)
    else:
        _cleanup_obsolete_empty_directory(backup)


def _remove_staging_after_collection_failure(staging: Path, target: Path) -> None:
    _promotion_sibling(target, staging.name, kind="staging")
    if is_link_or_reparse_point(staging) or not staging.is_dir():
        raise ValueError("Refusing to remove an unsafe demonstration staging path.")
    for path in staging.rglob("*"):
        if is_link_or_reparse_point(path):
            raise ValueError("Refusing to remove a demonstration staging directory containing links.")
    shutil.rmtree(staging)
    _fsync_parent_directory(staging)


def _prepare_promotion_journal(
    staging: Path,
    target: Path,
    *,
    overwrite: bool = True,
) -> dict[str, object]:
    staging_path = _promotion_sibling(target, staging.name, kind="staging")
    if staging_path != staging:
        raise ValueError("Staging directory must be an absolute sibling of the target.")
    staging_hash = _fsync_managed_artifact(staging)
    if not target.exists():
        previous_kind = "absent"
        backup_name = ""
        backup_hash = ""
    else:
        if is_link_or_reparse_point(target) or not target.is_dir():
            raise ValueError("Demonstration target must be a regular directory.")
        if not bool(overwrite) and any(target.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite concurrently created demonstration artifact {target.name!r}."
            )
        backup = target.with_name(f".{target.name}.backup-{time.time_ns()}")
        while backup.exists():
            backup = target.with_name(f".{target.name}.backup-{time.time_ns()}")
        backup_name = backup.name
        if any(target.iterdir()):
            previous_kind = "artifact"
            backup_hash = _validated_artifact_manifest_sha256(target)
        else:
            previous_kind = "empty"
            backup_hash = ""
    record: dict[str, object] = {
        "protocol": _PROMOTION_JOURNAL_PROTOCOL,
        "version": _PROMOTION_JOURNAL_VERSION,
        "target_name": target.name,
        "staging_name": staging.name,
        "staging_manifest_sha256": staging_hash,
        "previous_kind": previous_kind,
        "backup_name": backup_name,
        "backup_manifest_sha256": backup_hash,
    }
    _write_promotion_journal(_promotion_journal_path(target), record)
    return record


def _recover_interrupted_promotion(target: Path) -> None:
    """Recover a journaled replacement; collection callers hold the target lock."""

    record = _load_promotion_journal(target)
    if record is None:
        return
    journal = _promotion_journal_path(target)
    staging = _promotion_sibling(target, record["staging_name"], kind="staging")
    staged_hash = str(record["staging_manifest_sha256"])
    previous_kind = str(record["previous_kind"])
    backup = (
        None
        if previous_kind == "absent"
        else _promotion_sibling(target, record["backup_name"], kind="backup")
    )
    backup_hash = str(record["backup_manifest_sha256"])

    target_hash: str | None = None
    target_is_empty = False
    if target.exists():
        if is_link_or_reparse_point(target) or not target.is_dir():
            raise ValueError("Cannot recover promotion because the target is not a regular directory.")
        if any(target.iterdir()):
            target_hash = _validated_artifact_manifest_sha256(target)
        else:
            target_is_empty = True

    if target_hash == staged_hash:
        _cleanup_obsolete_promotion_paths(
            staging=staging,
            staged_hash=staged_hash,
            backup=backup,
            backup_hash=backup_hash,
            previous_kind=previous_kind,
        )
        _unlink_path(journal)
        return

    target_matches_previous = (
        previous_kind == "artifact"
        and target_hash == backup_hash
        or previous_kind == "empty"
        and target_is_empty
    )
    if target_matches_previous:
        _cleanup_obsolete_promotion_paths(
            staging=staging,
            staged_hash=staged_hash,
            backup=backup,
            backup_hash=backup_hash,
            previous_kind=previous_kind,
        )
        _unlink_path(journal)
        return

    if not target.exists() and backup is not None and backup.exists():
        if previous_kind == "artifact":
            actual_backup_hash = _validated_artifact_manifest_sha256(backup)
            if actual_backup_hash != backup_hash:
                raise ValueError("Cannot recover promotion from an unexpected backup artifact.")
        else:
            _require_empty_directory(backup, label="backup")
        _replace_path(backup, target)
        _cleanup_obsolete_promotion_paths(
            staging=staging,
            staged_hash=staged_hash,
            backup=backup,
            backup_hash=backup_hash,
            previous_kind=previous_kind,
        )
        _unlink_path(journal)
        return

    if not target.exists() and previous_kind == "absent" and backup is None:
        _cleanup_obsolete_promotion_paths(
            staging=staging,
            staged_hash=staged_hash,
            backup=None,
            backup_hash="",
            previous_kind=previous_kind,
        )
        _unlink_path(journal)
        return

    raise ValueError(
        "Cannot recover demonstration promotion because journaled paths do not match a safe state."
    )


def _promote_staged_artifact(
    staging: Path,
    target: Path,
    *,
    overwrite: bool,
) -> None:
    """Promote a validated artifact; collection callers hold the target lock."""

    record = _prepare_promotion_journal(staging, target, overwrite=overwrite)
    previous_kind = str(record["previous_kind"])
    backup = (
        None
        if previous_kind == "absent"
        else _promotion_sibling(target, record["backup_name"], kind="backup")
    )
    try:
        if backup is not None:
            _replace_path(target, backup)
        if overwrite:
            _replace_path(staging, target)
        else:
            _install_path_no_replace(staging, target)
        _recover_interrupted_promotion(target)
    except BaseException as exc:
        try:
            _recover_interrupted_promotion(target)
        except BaseException as recovery_error:
            if hasattr(exc, "add_note"):
                exc.add_note(f"Automatic promotion recovery also failed: {recovery_error}")
        raise


@torch.no_grad()
def _collect_flow_map_demonstrations_into(
    backbone_model: OTFlow,
    gipo_policy: GIPOSchedulePolicy,
    contexts: DistillationContexts,
    *,
    physical_fingerprints: Sequence[str],
    settings: Sequence[DistillationSetting],
    output_dir: str | Path,
    split_phase: str,
    scenario_key: str,
    benchmark_family: str,
    backbone_checkpoint_sha256: str,
    gipo_checkpoint_sha256: str,
    contexts_source_kind: str,
    contexts_source_sha256: str,
    rollouts_per_context: int = 4,
    context_batch_size: int = 8,
    shard_contexts: int = 8,
    seed: int = 0,
) -> Path:
    """Collect frozen GIPO-guided teacher trajectories as portable NPZ shards."""

    contexts.validate()
    fingerprints = tuple(str(value) for value in physical_fingerprints)
    if len(fingerprints) != len(contexts.context_ids):
        raise ValueError("Physical context fingerprints must align with contexts.")
    requested_settings = tuple(settings)
    if not requested_settings:
        raise ValueError("At least one solver/NFE setting is required.")
    if len(set(requested_settings)) != len(requested_settings):
        raise ValueError("Distillation settings may not contain duplicates.")
    phase = str(split_phase).strip()
    if phase not in DEMONSTRATION_TRAINING_SPLITS:
        raise ValueError(
            f"split_phase must be a non-test training or validation split, got {split_phase!r}."
        )
    rollout_count = int(rollouts_per_context)
    if rollout_count <= 0:
        raise ValueError("rollouts_per_context must be positive.")
    cache_batch = int(context_batch_size)
    shard_size = int(shard_contexts)
    if cache_batch <= 0 or shard_size <= 0:
        raise ValueError("context_batch_size and shard_contexts must be positive.")
    if gipo_policy.density_dim != DENSITY_BIN_COUNT:
        raise ValueError(
            f"Flow-map distillation requires the {DENSITY_BIN_COUNT}-bin GIPO density representation."
        )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    device = next(backbone_model.parameters()).device
    dtype = next(backbone_model.parameters()).dtype
    backbone_model.eval()
    gipo_policy.student.eval()
    context_shards: list[dict[str, object]] = []
    trajectory_shards: list[dict[str, object]] = []
    state_dim = int(backbone_model.cfg.sample_state_dim)
    for shard_index, shard_start in enumerate(range(0, len(contexts.context_ids), shard_size)):
        shard_stop = min(len(contexts.context_ids), shard_start + shard_size)
        cache_parts: list[dict[str, torch.Tensor | None]] = []
        for batch_start in range(shard_start, shard_stop, cache_batch):
            batch_stop = min(shard_stop, batch_start + cache_batch)
            histories = contexts.histories[batch_start:batch_stop].to(device=device, dtype=dtype)
            conditions = None
            if contexts.conditions is not None:
                conditions = contexts.conditions[batch_start:batch_stop].to(
                    device=device,
                    dtype=dtype,
                )
            cache_parts.append(
                _cache_to_cpu(backbone_model.backbone.precompute(histories, cond=conditions))
            )
        shard_cache = _concatenate_caches(cache_parts)
        ctx_summary = shard_cache["ctx_summary"]
        ctx_tokens = shard_cache["ctx_tokens"]
        summary = shard_cache["summary"]
        if ctx_summary is None or ctx_tokens is None or summary is None:
            raise ValueError("Backbone context cache is missing required fields.")
        context_embedding = gipo_policy.context_embedding_from_cache(shard_cache)

        arrays: dict[str, object] = {
            "context_index": np.arange(shard_start, shard_stop, dtype=np.int64),
            "context_id": np.asarray(contexts.context_ids[shard_start:shard_stop], dtype=np.str_),
            "context_fingerprint": np.asarray(
                fingerprints[shard_start:shard_stop],
                dtype=np.str_,
            ),
            "ctx_tokens": ctx_tokens.numpy(),
            "ctx_summary": ctx_summary.numpy(),
            "summary": summary.numpy(),
        }
        if shard_cache["cond_emb"] is not None:
            arrays["cond_emb"] = shard_cache["cond_emb"].numpy()  # type: ignore[union-attr]
        record = write_npz_shard(
            root,
            f"shards/contexts_{shard_index:05d}.npz",
            arrays,
        )
        context_shards.append(record)

        common_rollouts: list[_CommonContextRollout] = []
        for local_index, context_index in enumerate(range(shard_start, shard_stop)):
            seeds = tuple(
                _noise_seed(
                    seed,
                    context_fingerprint=fingerprints[context_index],
                    rollout_index=rollout_index,
                )
                for rollout_index in range(rollout_count)
            )
            common_rollouts.append(
                _CommonContextRollout(
                    context_index=context_index,
                    seeds=seeds,
                    initial_state=_initial_states(
                        state_dim=state_dim,
                        seeds=seeds,
                        device=torch.device("cpu"),
                        dtype=dtype,
                    ),
                    conditioning_cache=_slice_cache(
                        shard_cache,
                        local_index,
                        repeats=1,
                        device=torch.device("cpu"),
                        dtype=dtype,
                    ),
                    context_embedding=context_embedding[local_index : local_index + 1].to(
                        device=torch.device("cpu"),
                        dtype=dtype,
                    ),
                )
            )

        for setting in requested_settings:
            context_index_rows: list[np.ndarray] = []
            seed_rows: list[np.ndarray] = []
            initial_rows: list[np.ndarray] = []
            grid_rows: list[np.ndarray] = []
            state_rows: list[np.ndarray] = []
            density_rows: list[np.ndarray] = []
            for common in common_rollouts:
                schedule = gipo_policy.predict(
                    common.context_embedding.to(device=device, dtype=dtype),
                    solver_key=setting.solver_key,
                    target_nfe=setting.target_nfe,
                )
                initial_state = common.initial_state.to(device=device, dtype=dtype)
                trajectory = backbone_model.solve(
                    initial_state.clone(),
                    conditioning_cache=_repeat_cache(
                        common.conditioning_cache,
                        repeats=rollout_count,
                        device=device,
                        dtype=dtype,
                    ),
                    solver_key=setting.solver_key,
                    target_nfe=setting.target_nfe,
                    time_grid=schedule.time_grid[0],
                    return_trajectory=True,
                )
                if not isinstance(trajectory, FlowTrajectory):
                    raise RuntimeError("OTFlow.solve did not return a trajectory.")
                context_index_rows.append(
                    np.full((rollout_count,), common.context_index, dtype=np.int64)
                )
                seed_rows.append(np.asarray(common.seeds, dtype=np.int64))
                initial_rows.append(initial_state.detach().cpu().numpy())
                grid_rows.append(
                    schedule.time_grid.expand(rollout_count, -1).detach().cpu().numpy()
                )
                state_rows.append(trajectory.states.detach().cpu().numpy())
                density_rows.append(
                    schedule.density_mass.expand(rollout_count, -1).detach().cpu().numpy()
                )
            arrays = {
                "context_index": np.concatenate(context_index_rows, axis=0),
                "noise_seed": np.concatenate(seed_rows, axis=0),
                "initial_state": np.concatenate(initial_rows, axis=0),
                "time_grid": np.concatenate(grid_rows, axis=0),
                "states": np.concatenate(state_rows, axis=0),
                "density_mass": np.concatenate(density_rows, axis=0),
            }
            record = write_npz_shard(
                root,
                (
                    f"shards/trajectories_{setting.solver_key}_nfe{setting.target_nfe}_"
                    f"{shard_index:05d}.npz"
                ),
                arrays,
            )
            record.update(
                {
                    "solver_key": setting.solver_key,
                    "target_nfe": int(setting.target_nfe),
                    "context_start": int(shard_start),
                    "context_stop": int(shard_stop),
                }
            )
            trajectory_shards.append(record)

    metadata = {
        "artifact_version": DEMONSTRATION_ARTIFACT_VERSION,
        "split_phase": phase,
        "scenario_key": str(scenario_key),
        "benchmark_family": str(benchmark_family),
        "context_count": int(len(contexts.context_ids)),
        "rollouts_per_context": rollout_count,
        "sample_state_dim": state_dim,
        "density_bin_count": int(gipo_policy.density_dim),
        "density_reference_time_grid": list(gipo_policy.reference_time_grid),
        "context_embedding_kind": gipo_policy.context_embedding_kind,
        "setting_encoder_config": gipo_policy.setting_encoder_config.to_payload(),
        "settings": [
            {"solver_key": item.solver_key, "target_nfe": int(item.target_nfe)}
            for item in requested_settings
        ],
        "backbone_checkpoint_sha256": str(backbone_checkpoint_sha256),
        "gipo_checkpoint_sha256": str(gipo_checkpoint_sha256),
        "contexts_source_kind": str(contexts_source_kind),
        "contexts_source_sha256": str(contexts_source_sha256),
        "collection_seed": int(seed),
        "classifier_free_guidance_scale": float(backbone_model.cfg.sample.cfg_scale),
        "locked_test_used": False,
    }
    return write_demonstration_manifest(
        root,
        context_shards=context_shards,
        trajectory_shards=trajectory_shards,
        metadata=metadata,
    )


def collect_flow_map_demonstrations(
    backbone_model: OTFlow,
    gipo_policy: GIPOSchedulePolicy,
    contexts: DistillationContexts,
    *,
    settings: Sequence[DistillationSetting],
    output_dir: str | Path,
    split_phase: str,
    scenario_key: str,
    benchmark_family: str,
    backbone_checkpoint_sha256: str,
    gipo_checkpoint_sha256: str,
    contexts_source_kind: str = "in_memory",
    contexts_source_sha256: str = "",
    rollouts_per_context: int = 4,
    context_batch_size: int = 8,
    shard_contexts: int = 8,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Collect demonstrations in staging and promote them with crash recovery."""

    collection_seed = validate_strict_integer(
        seed,
        label="demonstration collection seed",
        minimum=0,
    )
    physical_fingerprints = contexts.content_fingerprints()
    source_kind = str(contexts_source_kind).strip()
    if source_kind not in {"in_memory", "npz"}:
        raise ValueError("contexts_source_kind must be 'in_memory' or 'npz'.")
    source_sha256 = str(contexts_source_sha256).strip()
    if source_kind == "in_memory":
        expected_source_sha256 = in_memory_context_source_sha256(
            contexts.context_ids,
            physical_fingerprints,
        )
        if source_sha256 and source_sha256 != expected_source_sha256:
            raise ValueError(
                "contexts_source_sha256 does not match the in-memory contexts."
            )
        source_sha256 = expected_source_sha256
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError("contexts_source_sha256 must be a lowercase SHA-256 digest.")
    policy_metadata = getattr(gipo_policy, "checkpoint_payload", None)
    if isinstance(policy_metadata, Mapping):
        policy_scenario = str(policy_metadata.get("scenario_key", "") or "").strip()
        policy_family = str(
            policy_metadata.get("benchmark_family", "") or ""
        ).strip()
        if not policy_scenario or not policy_family:
            raise ValueError(
                "GIPO checkpoint requires scenario provenance for demonstration collection."
            )
        if (policy_scenario, policy_family) != (
            str(scenario_key).strip(),
            str(benchmark_family).strip(),
        ):
            raise ValueError(
                "Requested demonstration scenario does not match the GIPO checkpoint."
            )
    target = Path(os.path.abspath(Path(output_dir).expanduser()))
    anchor = Path(target.anchor)
    indirect = first_link_or_reparse_component(target, root=anchor)
    if indirect is not None:
        raise ValueError(
            "Refusing a demonstration output through a symlink, junction, or "
            f"reparse point: {indirect}."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_collection_lock(target):
        _recover_interrupted_promotion(target)
        _validate_output_root(target, overwrite=bool(overwrite))
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.staging-",
                dir=target.parent,
            )
        ).resolve()
        try:
            _collect_flow_map_demonstrations_into(
                backbone_model,
                gipo_policy,
                contexts,
                physical_fingerprints=physical_fingerprints,
                settings=settings,
                output_dir=staging,
                split_phase=split_phase,
                scenario_key=scenario_key,
                benchmark_family=benchmark_family,
                backbone_checkpoint_sha256=backbone_checkpoint_sha256,
                gipo_checkpoint_sha256=gipo_checkpoint_sha256,
                contexts_source_kind=source_kind,
                contexts_source_sha256=source_sha256,
                rollouts_per_context=rollouts_per_context,
                context_batch_size=context_batch_size,
                shard_contexts=shard_contexts,
                seed=collection_seed,
            )
            _promote_staged_artifact(staging, target, overwrite=bool(overwrite))
        except BaseException:
            if staging.exists() and not _promotion_journal_path(target).exists():
                _remove_staging_after_collection_failure(staging, target)
            raise
    return target / DEMONSTRATION_MANIFEST_NAME


def load_distillation_contexts(path: str | Path) -> DistillationContexts:
    input_path = Path(path).expanduser().resolve()
    try:
        with np.load(input_path, allow_pickle=False) as payload:
            required = {"context_ids", "histories"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"Context NPZ is missing arrays: {missing}.")
            ids_array = np.asarray(payload["context_ids"])
            if ids_array.ndim != 1 or ids_array.dtype.kind not in {"U", "S"}:
                raise ValueError("context_ids must be a one-dimensional Unicode or byte-string array.")
            context_ids = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in ids_array.tolist()
            )
            histories_array = np.asarray(payload["histories"])
            if histories_array.dtype.kind != "f":
                raise ValueError("histories must contain floating-point values.")
            histories = torch.from_numpy(
                np.asarray(histories_array, dtype=np.float32)
            )
            conditions = None
            if "conditions" in payload.files:
                conditions_array = np.asarray(payload["conditions"])
                if conditions_array.dtype.kind != "f":
                    raise ValueError("conditions must contain floating-point values.")
                conditions = torch.from_numpy(
                    np.asarray(conditions_array, dtype=np.float32)
                )
    except OSError as exc:
        raise ValueError(f"Could not read context NPZ {input_path.name!r}: {exc}") from exc
    return DistillationContexts(context_ids, histories, conditions).validate()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect frozen GIPO-guided OTFlow trajectories for endpoint-map distillation."
    )
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--gipo-checkpoint", required=True)
    parser.add_argument("--contexts-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-phase", required=True)
    parser.add_argument("--scenario-key", required=True)
    parser.add_argument("--benchmark-family", required=True)
    parser.add_argument(
        "--settings",
        default="",
        help="Comma-separated solver_key:target_nfe pairs; empty uses the supported 4x8 matrix.",
    )
    parser.add_argument("--rollouts-per-context", type=int, default=4)
    parser.add_argument("--context-batch-size", type=int, default=8)
    parser.add_argument("--shard-contexts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argparser().parse_args(list(argv) if argv is not None else None)
    device = resolve_torch_device(str(args.device))
    backbone_path = resolve_project_path(args.backbone_checkpoint)
    gipo_path = resolve_project_path(args.gipo_checkpoint)
    contexts_path = resolve_project_path(args.contexts_npz)
    output_dir = resolve_project_path(args.output_dir)
    output_manifest = (
        output_dir / DEMONSTRATION_MANIFEST_NAME
    )
    if output_manifest in {backbone_path, gipo_path, contexts_path}:
        raise ValueError("Demonstration output must differ from every input artifact path.")
    backbone_model, _ = load_checkpoint_model(backbone_path, device)
    gipo_policy = load_gipo_schedule_policy(gipo_path, device=device)
    manifest = collect_flow_map_demonstrations(
        backbone_model,
        gipo_policy,
        load_distillation_contexts(contexts_path),
        settings=parse_distillation_settings(args.settings),
        output_dir=output_dir,
        split_phase=args.split_phase,
        scenario_key=args.scenario_key,
        benchmark_family=args.benchmark_family,
        backbone_checkpoint_sha256=file_sha256(backbone_path),
        gipo_checkpoint_sha256=file_sha256(gipo_path),
        contexts_source_kind="npz",
        contexts_source_sha256=file_sha256(contexts_path),
        rollouts_per_context=args.rollouts_per_context,
        context_batch_size=args.context_batch_size,
        shard_contexts=args.shard_contexts,
        seed=args.seed,
        overwrite=bool(args.overwrite),
    )
    print(manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DISTILLATION_NFES",
    "DistillationContexts",
    "DistillationSetting",
    "build_argparser",
    "collect_flow_map_demonstrations",
    "default_distillation_settings",
    "load_distillation_contexts",
    "main",
    "parse_distillation_settings",
]
