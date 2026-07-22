from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Callable, Iterator, Mapping
import uuid

from genode.path_safety import is_link_or_reparse_point
from genode.provenance import file_sha256


BUNDLE_TRANSACTION_PROTOCOL = "genode_artifact_bundle_transaction"
BUNDLE_TRANSACTION_VERSION = 1
_LOCK_MARKER = b"genode-artifact-bundle-lock\n"
_ROLE_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_JOURNAL_FIELDS = {
    "protocol",
    "version",
    "state",
    "bundle_key",
    "overwrite",
    "targets",
}
_TARGET_FIELDS = {
    "role",
    "target_name",
    "staging_name",
    "staging_sha256",
    "previous_kind",
    "previous_sha256",
    "backup_name",
}

BundleValidator = Callable[[Mapping[str, Path], Mapping[str, Path]], None]


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _is_sha256(value: object) -> bool:
    return _SHA256_PATTERN.fullmatch(str(value)) is not None


def _normalize_bundle(
    anchor: str | Path,
    targets: Mapping[str, str | Path],
) -> tuple[Path, dict[str, Path]]:
    def normalized_final_path(value: str | Path, *, label: str) -> Path:
        lexical = Path(value).expanduser()
        if is_link_or_reparse_point(lexical):
            raise ValueError(
                f"Artifact bundle {label} may not be a symlink, junction, or reparse point."
            )
        return lexical.parent.resolve() / lexical.name

    normalized_anchor = normalized_final_path(anchor, label="anchor")
    normalized: dict[str, Path] = {}
    for raw_role, raw_path in targets.items():
        role = str(raw_role)
        if _ROLE_PATTERN.fullmatch(role) is None:
            raise ValueError(f"Artifact bundle role {role!r} is invalid.")
        path = normalized_final_path(raw_path, label=f"target {role!r}")
        if path in normalized.values():
            raise ValueError("Artifact bundle target paths must be pairwise distinct.")
        normalized[role] = path
    if not normalized:
        raise ValueError("Artifact bundle requires at least one target.")
    normalized = dict(sorted(normalized.items()))
    normalized_paths = tuple(normalized.values())
    for index, path in enumerate(normalized_paths):
        for other in normalized_paths[index + 1 :]:
            if path in other.parents or other in path.parents:
                raise ValueError(
                    "Artifact bundle targets may not contain one another."
                )
    deterministic_anchor = normalized[next(iter(normalized))]
    if normalized_anchor != deterministic_anchor:
        raise ValueError(
            "Artifact bundle anchor must be the target with the lexicographically first role."
        )
    for candidate in normalized_paths:
        for owner in normalized_paths:
            lock_path = owner.with_name(f".{owner.name}.bundle.lock")
            journal_path = owner.with_name(
                f".{owner.name}.bundle.transaction.json"
            )
            if candidate in {lock_path, journal_path}:
                raise ValueError(
                    "Artifact bundle target collides with a reserved lock or journal sidecar."
                )
            if candidate.parent != owner.parent:
                continue
            staging_prefix = f".{owner.name}.bundle-stage-"
            backup_prefix = f".{owner.name}.bundle-backup-"
            journal_temporary_prefix = f".{journal_path.name}."
            if (
                candidate.name.startswith(staging_prefix)
                or candidate.name.startswith(backup_prefix)
                or (
                    candidate.name.startswith(journal_temporary_prefix)
                    and candidate.name.endswith(".tmp")
                )
            ):
                raise ValueError(
                    "Artifact bundle target collides with a reserved managed sidecar namespace."
                )
    return normalized_anchor, normalized


def validate_artifact_bundle_layout(
    anchor: str | Path,
    targets: Mapping[str, str | Path],
) -> None:
    """Validate bundle paths without creating directories or sidecars."""

    _normalize_bundle(anchor, targets)


def _bundle_key(targets: Mapping[str, Path]) -> str:
    payload = [
        [role, os.path.normcase(str(path))]
        for role, path in sorted(targets.items())
    ]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def bundle_lock_path(anchor: str | Path) -> Path:
    target = Path(anchor).expanduser().resolve()
    return target.with_name(f".{target.name}.bundle.lock")


def bundle_journal_path(anchor: str | Path) -> Path:
    target = Path(anchor).expanduser().resolve()
    return target.with_name(f".{target.name}.bundle.transaction.json")


def temporary_bundle_path(target: str | Path) -> Path:
    path = Path(target).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.bundle-stage-",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def discard_temporary_bundle_path(
    staged: str | Path,
    target: str | Path,
    *,
    expected_sha256: str | None = None,
) -> bool:
    """Remove one unjournaled managed staging file, preserving anything unknown."""

    target_path = Path(target).expanduser().resolve()
    staged_path = Path(staged).expanduser().resolve()
    if staged_path.parent != target_path.parent:
        return False
    try:
        if _staging_path(target_path, staged_path.name) != staged_path:
            return False
        if not _path_exists(staged_path):
            return True
        _require_regular_file(staged_path, label="Managed bundle staging file")
        if expected_sha256 is not None and not _exact_hash(
            staged_path, expected_sha256
        ):
            return False
        _unlink(staged_path)
    except (OSError, ValueError):
        return False
    return True


def _fsync_directory(directory: Path) -> None:
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
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"Expected a directory while syncing {directory.name!r}.")
        try:
            os.fsync(descriptor)
        except OSError:
            if not best_effort:
                raise
    finally:
        os.close(descriptor)


def _fsync_parent(path: Path) -> None:
    _fsync_directory(path.parent)


def _require_regular_file(path: Path, *, label: str) -> None:
    if is_link_or_reparse_point(path) or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path.name!r}.")


def _fsync_regular_file(path: Path) -> str:
    _require_regular_file(path, label="Managed bundle artifact")
    initial_hash = file_sha256(path)
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
            raise ValueError(f"Managed bundle file {path.name!r} changed while open.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    final_hash = file_sha256(path)
    if final_hash != initial_hash:
        raise ValueError(f"Managed bundle file {path.name!r} changed while syncing.")
    return final_hash


def _replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)
    _fsync_parent(destination)
    if source.parent != destination.parent:
        _fsync_parent(source)


def _unlink(path: Path, *, missing_ok: bool = False) -> None:
    existed = _path_exists(path)
    path.unlink(missing_ok=missing_ok)
    if existed:
        _fsync_parent(path)


def _link_without_overwrite(source: Path, destination: Path) -> None:
    """Install a staged file atomically without replacing a concurrent target."""

    os.link(source, destination, follow_symlinks=False)
    _fsync_parent(destination)
    _unlink(source)


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
    raise OSError(f"Artifact bundle locking is unsupported on platform {os.name!r}.")


@contextmanager
def exclusive_bundle_lock(anchor: str | Path) -> Iterator[None]:
    """Hold a persistent, target-scoped interprocess advisory lock."""

    if os.name not in {"nt", "posix"}:
        raise RuntimeError(
            f"Artifact bundle locking is unsupported on platform {os.name!r}."
        )
    lock_path = bundle_lock_path(anchor)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if _path_exists(lock_path) and (
        is_link_or_reparse_point(lock_path) or not lock_path.is_file()
    ):
        raise ValueError("Artifact bundle lock must be a regular file.")
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
            raise ValueError("Artifact bundle lock must be a regular file.")
        descriptor = os.open(lock_path, flags)
    try:
        os.set_inheritable(descriptor, False)
        descriptor_stat = os.fstat(descriptor)
        path_stat = lock_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(descriptor_stat.st_mode) or (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ) != (path_stat.st_dev, path_stat.st_ino):
            raise ValueError("Artifact bundle lock changed while it was opened.")
        if is_link_or_reparse_point(lock_path):
            raise ValueError("Artifact bundle lock may not be a link or reparse point.")
        try:
            _acquire_platform_lock(descriptor)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise RuntimeError(
                    f"Another process is already writing artifact bundle {Path(anchor).name!r}."
                ) from exc
            raise RuntimeError(
                f"Could not safely acquire the artifact bundle lock for {Path(anchor).name!r}."
            ) from exc
        os.lseek(descriptor, 0, os.SEEK_SET)
        marker = os.read(descriptor, len(_LOCK_MARKER) + 1)
        if not marker:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.write(descriptor, _LOCK_MARKER) != len(_LOCK_MARKER):
                raise OSError("Could not initialize artifact bundle lock.")
            os.fsync(descriptor)
            _fsync_parent(lock_path)
        elif marker != _LOCK_MARKER:
            raise ValueError("Refusing to use an unrecognized artifact bundle lock file.")
        yield
    finally:
        os.close(descriptor)


def _staging_path(target: Path, name: object) -> Path:
    text = str(name)
    prefix = f".{target.name}.bundle-stage-"
    if (
        Path(text).name != text
        or not text.startswith(prefix)
        or not text.endswith(".tmp")
    ):
        raise ValueError("Bundle journal contains an invalid staging sidecar name.")
    path = target.parent / text
    if _path_exists(path) and is_link_or_reparse_point(path):
        raise ValueError("Bundle staging sidecar may not be a link or reparse point.")
    return path


def _backup_path(target: Path, name: object) -> Path:
    text = str(name)
    prefix = f".{target.name}.bundle-backup-"
    suffix = text[len(prefix) :] if text.startswith(prefix) else ""
    if Path(text).name != text or not text.startswith(prefix) or re.fullmatch(
        r"[0-9a-f]{32}", suffix
    ) is None:
        raise ValueError("Bundle journal contains an invalid backup sidecar name.")
    path = target.parent / text
    if _path_exists(path) and is_link_or_reparse_point(path):
        raise ValueError("Bundle backup sidecar may not be a link or reparse point.")
    return path


def _journal_payload(
    *,
    targets: Mapping[str, Path],
    staged: Mapping[str, Path],
    staged_hashes: Mapping[str, str],
    previous_kind: str,
    previous_hashes: Mapping[str, str],
    overwrite: bool,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for role, target in targets.items():
        backup_name = (
            f".{target.name}.bundle-backup-{uuid.uuid4().hex}"
            if previous_kind == "file"
            else ""
        )
        records.append(
            {
                "role": role,
                "target_name": target.name,
                "staging_name": staged[role].name,
                "staging_sha256": staged_hashes[role],
                "previous_kind": previous_kind,
                "previous_sha256": previous_hashes.get(role, ""),
                "backup_name": backup_name,
            }
        )
    return {
        "protocol": BUNDLE_TRANSACTION_PROTOCOL,
        "version": BUNDLE_TRANSACTION_VERSION,
        "state": "prepared",
        "bundle_key": _bundle_key(targets),
        "overwrite": bool(overwrite),
        "targets": records,
    }


def _write_journal(path: Path, payload: Mapping[str, object], *, replace: bool) -> None:
    if not replace and _path_exists(path):
        raise ValueError("Recover the existing artifact bundle journal before promotion.")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            _replace(temporary, path)
        else:
            _link_without_overwrite(temporary, path)
    finally:
        if _path_exists(temporary):
            _unlink(temporary)


def _load_journal(anchor: Path, targets: Mapping[str, Path]) -> dict[str, object] | None:
    path = bundle_journal_path(anchor)
    if not _path_exists(path):
        return None
    _require_regular_file(path, label="Artifact bundle journal")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read artifact bundle journal safely: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != _JOURNAL_FIELDS:
        raise ValueError("Artifact bundle journal has an invalid schema.")
    if (
        payload.get("protocol") != BUNDLE_TRANSACTION_PROTOCOL
        or type(payload.get("version")) is not int
        or payload.get("version") != BUNDLE_TRANSACTION_VERSION
        or payload.get("state") not in {"prepared", "committed", "aborted"}
        or payload.get("bundle_key") != _bundle_key(targets)
        or not isinstance(payload.get("overwrite"), bool)
    ):
        raise ValueError("Artifact bundle journal identity is invalid.")
    raw_records = payload.get("targets")
    if not isinstance(raw_records, list) or len(raw_records) != len(targets):
        raise ValueError("Artifact bundle journal target records are invalid.")
    records: dict[str, dict[str, object]] = {}
    previous_kinds: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping) or set(raw_record) != _TARGET_FIELDS:
            raise ValueError("Artifact bundle journal target record has an invalid schema.")
        role = str(raw_record.get("role"))
        if role not in targets or role in records:
            raise ValueError("Artifact bundle journal contains an unexpected target role.")
        target = targets[role]
        if raw_record.get("target_name") != target.name:
            raise ValueError("Artifact bundle journal target identity does not match.")
        _staging_path(target, raw_record.get("staging_name"))
        if not _is_sha256(raw_record.get("staging_sha256")):
            raise ValueError("Artifact bundle journal has an invalid staging hash.")
        previous_kind = str(raw_record.get("previous_kind"))
        previous_kinds.add(previous_kind)
        previous_hash = raw_record.get("previous_sha256")
        backup_name = raw_record.get("backup_name")
        if previous_kind == "absent":
            if previous_hash != "" or backup_name != "":
                raise ValueError("Artifact bundle journal has invalid absent-target fields.")
        elif previous_kind == "file":
            if not _is_sha256(previous_hash):
                raise ValueError("Artifact bundle journal has an invalid previous hash.")
            _backup_path(target, backup_name)
        else:
            raise ValueError("Artifact bundle journal has an invalid previous target kind.")
        records[role] = dict(raw_record)
    if len(previous_kinds) != 1:
        raise ValueError("Artifact bundle journal describes a partial previous bundle.")
    return {**dict(payload), "targets": records}


def _existing_bundle_state(targets: Mapping[str, Path]) -> str:
    present = []
    for target in targets.values():
        if _path_exists(target):
            _require_regular_file(target, label="Artifact bundle target")
            present.append(target)
    if not present:
        return "absent"
    if len(present) != len(targets):
        raise ValueError("Artifact bundle is partial; refusing to replace or resume it.")
    return "file"


def _validate_complete_bundle(
    paths: Mapping[str, Path],
    targets: Mapping[str, Path],
    validator: BundleValidator | None,
) -> None:
    for path in paths.values():
        _require_regular_file(path, label="Artifact bundle member")
    if validator is not None:
        validator(paths, targets)


def _exact_hash(path: Path, expected: str) -> bool:
    if not _path_exists(path):
        return False
    try:
        _require_regular_file(path, label="Managed bundle sidecar")
        return file_sha256(path) == expected
    except (OSError, ValueError):
        return False


def _cleanup_exact(path: Path, expected_hash: str) -> None:
    if not _exact_hash(path, expected_hash):
        return
    try:
        _unlink(path)
    except OSError:
        return


def _records_for_targets(
    payload: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    records = payload.get("targets")
    if not isinstance(records, Mapping):
        raise ValueError("Normalized artifact bundle journal records are unavailable.")
    return records  # type: ignore[return-value]


def _paths_from_records(
    targets: Mapping[str, Path],
    records: Mapping[str, Mapping[str, object]],
    *,
    kind: str,
) -> dict[str, Path]:
    if kind == "target":
        return dict(targets)
    if kind == "staging":
        return {
            role: _staging_path(targets[role], record["staging_name"])
            for role, record in records.items()
        }
    if kind == "backup":
        return {
            role: _backup_path(targets[role], record["backup_name"])
            for role, record in records.items()
        }
    raise ValueError(f"Unknown artifact bundle path kind {kind!r}.")


def _verify_hashes(
    paths: Mapping[str, Path],
    records: Mapping[str, Mapping[str, object]],
    *,
    hash_field: str,
) -> bool:
    return all(
        _exact_hash(paths[role], str(record[hash_field]))
        for role, record in records.items()
    )


def _finalize_journal(
    journal: Path,
    payload: Mapping[str, object],
    *,
    state: str,
) -> dict[str, object]:
    finalized = {**dict(payload), "state": state}
    records = finalized.get("targets")
    if isinstance(records, Mapping):
        finalized["targets"] = list(records.values())
    _write_journal(journal, finalized, replace=True)
    finalized["targets"] = records
    return finalized


def _cleanup_finalized_transaction(
    *,
    journal: Path,
    targets: Mapping[str, Path],
    payload: Mapping[str, object],
) -> None:
    records = _records_for_targets(payload)
    staging = _paths_from_records(targets, records, kind="staging")
    for role, path in staging.items():
        _cleanup_exact(path, str(records[role]["staging_sha256"]))
    if next(iter(records.values()))["previous_kind"] == "file":
        backups = _paths_from_records(targets, records, kind="backup")
        for role, path in backups.items():
            _cleanup_exact(path, str(records[role]["previous_sha256"]))
    _unlink(journal)


def _restore_previous_bundle(
    *,
    targets: Mapping[str, Path],
    payload: Mapping[str, object],
    validator: BundleValidator | None,
) -> None:
    records = _records_for_targets(payload)
    previous_kind = str(next(iter(records.values()))["previous_kind"])
    backups = (
        _paths_from_records(targets, records, kind="backup")
        if previous_kind == "file"
        else {}
    )
    for role, target in targets.items():
        record = records[role]
        staged_hash = str(record["staging_sha256"])
        previous_hash = str(record["previous_sha256"])
        if _path_exists(target):
            if previous_kind == "file" and _exact_hash(target, previous_hash):
                continue
            if _exact_hash(target, staged_hash):
                _unlink(target)
            else:
                raise ValueError(
                    f"Cannot recover artifact bundle because target {target.name!r} is unknown."
                )
        if previous_kind == "file":
            backup = backups[role]
            if not _exact_hash(backup, previous_hash):
                raise ValueError(
                    f"Cannot recover artifact bundle because backup {backup.name!r} is missing or changed."
                )
            try:
                _link_without_overwrite(backup, target)
            except FileExistsError as exc:
                raise ValueError(
                    f"Cannot recover artifact bundle because target {target.name!r} appeared concurrently."
                ) from exc
    if previous_kind == "file":
        if not _verify_hashes(targets, records, hash_field="previous_sha256"):
            raise ValueError("Recovered artifact bundle does not match its previous hashes.")
        _validate_complete_bundle(targets, targets, validator)
    elif any(_path_exists(target) for target in targets.values()):
        raise ValueError("Recovered absent artifact bundle unexpectedly contains targets.")


def _recover_locked(
    anchor: Path,
    targets: Mapping[str, Path],
    *,
    validator: BundleValidator | None,
    force_abort: bool = False,
) -> None:
    payload = _load_journal(anchor, targets)
    if payload is None:
        return
    journal = bundle_journal_path(anchor)
    records = _records_for_targets(payload)
    state = str(payload["state"])
    staged_complete = _verify_hashes(targets, records, hash_field="staging_sha256")
    if state == "committed":
        if not staged_complete:
            raise ValueError("Committed artifact bundle no longer matches its journaled hashes.")
        _validate_complete_bundle(targets, targets, validator)
        _cleanup_finalized_transaction(journal=journal, targets=targets, payload=payload)
        return
    if state == "aborted":
        previous_kind = str(next(iter(records.values()))["previous_kind"])
        if previous_kind == "file":
            if not _verify_hashes(targets, records, hash_field="previous_sha256"):
                raise ValueError("Aborted artifact bundle no longer matches its previous hashes.")
            _validate_complete_bundle(targets, targets, validator)
        elif any(_path_exists(target) for target in targets.values()):
            raise ValueError("Aborted absent artifact bundle unexpectedly contains targets.")
        _cleanup_finalized_transaction(journal=journal, targets=targets, payload=payload)
        return
    if staged_complete and not force_abort:
        _validate_complete_bundle(targets, targets, validator)
        payload = _finalize_journal(journal, payload, state="committed")
    else:
        _restore_previous_bundle(targets=targets, payload=payload, validator=validator)
        payload = _finalize_journal(journal, payload, state="aborted")
    _cleanup_finalized_transaction(journal=journal, targets=targets, payload=payload)


def recover_artifact_bundle(
    anchor: str | Path,
    targets: Mapping[str, str | Path],
    *,
    validator: BundleValidator | None = None,
) -> None:
    normalized_anchor, normalized_targets = _normalize_bundle(anchor, targets)
    with exclusive_bundle_lock(normalized_anchor):
        _recover_locked(
            normalized_anchor,
            normalized_targets,
            validator=validator,
        )


def preflight_artifact_bundle(
    anchor: str | Path,
    targets: Mapping[str, str | Path],
    *,
    overwrite: bool,
    validator: BundleValidator | None = None,
) -> None:
    normalized_anchor, normalized_targets = _normalize_bundle(anchor, targets)
    for target in normalized_targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_bundle_lock(normalized_anchor):
        _recover_locked(
            normalized_anchor,
            normalized_targets,
            validator=validator,
        )
        state = _existing_bundle_state(normalized_targets)
        if state == "absent":
            return
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing artifact bundle {normalized_anchor.name!r}."
            )
        _validate_complete_bundle(normalized_targets, normalized_targets, validator)


def validate_artifact_bundle(
    anchor: str | Path,
    targets: Mapping[str, str | Path],
    *,
    validator: BundleValidator | None = None,
) -> None:
    """Recover, then validate one complete bundle while holding its target lock."""

    normalized_anchor, normalized_targets = _normalize_bundle(anchor, targets)
    with exclusive_bundle_lock(normalized_anchor):
        _recover_locked(
            normalized_anchor,
            normalized_targets,
            validator=validator,
        )
        if _existing_bundle_state(normalized_targets) != "file":
            raise FileNotFoundError(
                f"Artifact bundle {normalized_anchor.name!r} is not complete."
            )
        _validate_complete_bundle(normalized_targets, normalized_targets, validator)


def promote_artifact_bundle(
    anchor: str | Path,
    targets: Mapping[str, str | Path],
    staged: Mapping[str, str | Path],
    *,
    overwrite: bool,
    validator: BundleValidator | None = None,
    precommit_validator: Callable[[], None] | None = None,
) -> None:
    normalized_anchor, normalized_targets = _normalize_bundle(anchor, targets)
    normalized_staged = {
        str(role): Path(path).expanduser().resolve() for role, path in staged.items()
    }
    if set(normalized_staged) != set(normalized_targets):
        raise ValueError("Staged artifact roles must exactly match bundle target roles.")
    for role, path in normalized_staged.items():
        if path == normalized_targets[role] or path.parent != normalized_targets[role].parent:
            raise ValueError("Staged bundle files must be distinct siblings of their targets.")
        if _staging_path(normalized_targets[role], path.name) != path:
            raise ValueError("Staged bundle path does not use the managed sidecar format.")
    _validate_complete_bundle(normalized_staged, normalized_targets, validator)
    staged_hashes = {
        role: _fsync_regular_file(path) for role, path in normalized_staged.items()
    }
    for parent in {path.parent for path in normalized_staged.values()}:
        _fsync_directory(parent)

    with exclusive_bundle_lock(normalized_anchor):
        _recover_locked(
            normalized_anchor,
            normalized_targets,
            validator=validator,
        )
        previous_kind = _existing_bundle_state(normalized_targets)
        if previous_kind == "file" and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing artifact bundle {normalized_anchor.name!r}."
            )
        if previous_kind == "file":
            _validate_complete_bundle(normalized_targets, normalized_targets, validator)
        previous_hashes = (
            {role: file_sha256(path) for role, path in normalized_targets.items()}
            if previous_kind == "file"
            else {}
        )
        if precommit_validator is not None:
            precommit_validator()
        payload = _journal_payload(
            targets=normalized_targets,
            staged=normalized_staged,
            staged_hashes=staged_hashes,
            previous_kind=previous_kind,
            previous_hashes=previous_hashes,
            overwrite=overwrite,
        )
        journal = bundle_journal_path(normalized_anchor)
        _write_journal(journal, payload, replace=False)
        normalized_payload = _load_journal(normalized_anchor, normalized_targets)
        if normalized_payload is None:
            raise RuntimeError("Artifact bundle journal disappeared before promotion.")
        records = _records_for_targets(normalized_payload)
        try:
            if previous_kind == "file":
                backups = _paths_from_records(
                    normalized_targets, records, kind="backup"
                )
                for role, target in normalized_targets.items():
                    if file_sha256(target) != previous_hashes[role]:
                        raise ValueError(
                            f"Artifact bundle target {target.name!r} changed before promotion."
                        )
                    _link_without_overwrite(target, backups[role])
            elif any(_path_exists(target) for target in normalized_targets.values()):
                raise FileExistsError(
                    "An artifact bundle target appeared concurrently before promotion."
                )
            for role, target in normalized_targets.items():
                try:
                    _link_without_overwrite(normalized_staged[role], target)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Artifact bundle target {target.name!r} appeared concurrently."
                    ) from exc
                if file_sha256(target) != staged_hashes[role]:
                    raise ValueError(
                        f"Promoted artifact bundle target {target.name!r} changed unexpectedly."
                    )
            if precommit_validator is not None:
                precommit_validator()
            _validate_complete_bundle(normalized_targets, normalized_targets, validator)
            normalized_payload = _finalize_journal(
                journal, normalized_payload, state="committed"
            )
        except BaseException as exc:
            try:
                _recover_locked(
                    normalized_anchor,
                    normalized_targets,
                    validator=validator,
                    force_abort=True,
                )
            except BaseException as recovery_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        f"Automatic artifact bundle recovery also failed: {recovery_error}"
                    )
            raise
        _cleanup_finalized_transaction(
            journal=journal,
            targets=normalized_targets,
            payload=normalized_payload,
        )


__all__ = [
    "BundleValidator",
    "bundle_journal_path",
    "bundle_lock_path",
    "discard_temporary_bundle_path",
    "exclusive_bundle_lock",
    "preflight_artifact_bundle",
    "promote_artifact_bundle",
    "recover_artifact_bundle",
    "temporary_bundle_path",
    "validate_artifact_bundle",
    "validate_artifact_bundle_layout",
]
