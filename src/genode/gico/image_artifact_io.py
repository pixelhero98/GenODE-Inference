"""Private filesystem primitives for flat Image GICO artifact directories."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from genode.benchmarks.image.artifact_paths import managed_image_leaf_directory
from genode.path_safety import is_link_or_reparse_point

_MANIFEST_NAME = "manifest.json"
_FLOAT64_DESCRIPTOR_FIELDS = {"file", "sha256", "shape", "dtype"}
_LOWERCASE_SHA256 = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class _PathIdentity:
    device: int
    inode: int
    kind: int


@dataclass(slots=True)
class _PublicationState:
    target: Path
    stage: Path
    expected_members: tuple[str, ...]
    stage_identity: _PathIdentity
    source_identities: dict[str, _PathIdentity] = field(default_factory=dict)
    target_identity: _PathIdentity | None = None
    committed: bool = False


def _identity(value: os.stat_result) -> _PathIdentity:
    return _PathIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        kind=stat.S_IFMT(value.st_mode),
    )


def _lstat_identity(path: Path) -> _PathIdentity:
    return _identity(path.lstat())


def _has_identity(path: Path, expected: _PathIdentity) -> bool:
    try:
        return _lstat_identity(path) == expected
    except OSError:
        return False


def _portable_member_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Artifact member names must be strings.")
    if (
        not value
        or value.strip() != value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or Path(value).name != value
    ):
        raise ValueError(f"Artifact member name is not a portable leaf: {value!r}.")
    return value


def _expected_member_names(values: Collection[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise TypeError("expected_members must be a collection of portable filenames.")
    names = tuple(_portable_member_name(value) for value in values)
    if not names or len(set(names)) != len(names):
        raise ValueError("expected_members must be nonempty and contain no duplicates.")
    if _MANIFEST_NAME not in names:
        raise ValueError(f"expected_members must include {_MANIFEST_NAME!r}.")
    return tuple(sorted(names))


def _validated_raw_leaf(path: str | Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.name in {"", ".", ".."} or raw.name.strip() != raw.name:
        raise ValueError(f"{label} must name one managed leaf.")
    if ".." in raw.parts:
        raise ValueError(f"{label} may not contain '..' path components.")
    if is_link_or_reparse_point(raw.absolute()):
        raise ValueError(f"{label} must not be a symlink, junction, or reparse point.")
    return raw


def _normalized_leaf_directory(path: str | Path, *, label: str, create_parent: bool) -> Path:
    raw = _validated_raw_leaf(path, label=label)
    if create_parent:
        raw.parent.mkdir(parents=True, exist_ok=True)
    return managed_image_leaf_directory(raw, label=label)


def _require_directory_identity(path: Path, expected: _PathIdentity, *, label: str) -> None:
    if (
        not _has_identity(path, expected)
        or expected.kind != stat.S_IFDIR
        or is_link_or_reparse_point(path)
        or not path.is_dir()
    ):
        raise ValueError(f"{label} changed or is not a regular directory.")


def _regular_members(
    root: Path,
    *,
    root_identity: _PathIdentity,
    expected_members: tuple[str, ...],
    require_exact: bool,
) -> dict[str, _PathIdentity]:
    _require_directory_identity(root, root_identity, label="Image GICO artifact directory")
    entries = {path.name: path for path in root.iterdir()}
    expected = set(expected_members)
    if require_exact and set(entries) != expected:
        raise ValueError("Image GICO artifact files are incomplete or unexpected.")
    if not set(entries).issubset(expected):
        raise ValueError("Image GICO artifact directory contains unexpected members.")

    identities: dict[str, _PathIdentity] = {}
    for name, path in entries.items():
        if is_link_or_reparse_point(path) or not path.is_file():
            raise ValueError(f"Image GICO artifact member {name!r} must be a regular file.")
        observed = _lstat_identity(path)
        if observed.kind != stat.S_IFREG:
            raise ValueError(f"Image GICO artifact member {name!r} must be a regular file.")
        identities[name] = observed
    _require_directory_identity(root, root_identity, label="Image GICO artifact directory")
    return identities


def _fsync_directory(path: Path, *, expected: _PathIdentity | None = None) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        observed = _identity(os.fstat(descriptor))
        if observed.kind != stat.S_IFDIR or (expected is not None and observed != expected):
            raise ValueError(f"Directory {path.name!r} changed while it was opened for syncing.")
        try:
            os.fsync(descriptor)
        except OSError:
            if os.name != "nt":
                raise
    finally:
        os.close(descriptor)


def _fsync_regular_file(path: Path, *, expected: _PathIdentity) -> None:
    flags = os.O_RDWR
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        descriptor_identity = _identity(os.fstat(descriptor))
        if (
            descriptor_identity != expected
            or descriptor_identity.kind != stat.S_IFREG
            or not _has_identity(path, expected)
            or is_link_or_reparse_point(path)
        ):
            raise ValueError(f"Artifact member {path.name!r} changed while it was opened for syncing.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reserve_target_directory(target: Path) -> _PathIdentity:
    target.mkdir()
    observed = _lstat_identity(target)
    if observed.kind != stat.S_IFDIR or is_link_or_reparse_point(target):
        raise ValueError("Reserved Image GICO target is not a regular directory.")
    return observed


def _link_regular_file_no_replace(
    source: Path,
    destination: Path,
    *,
    expected: _PathIdentity,
) -> _PathIdentity:
    if (
        expected.kind != stat.S_IFREG
        or not _has_identity(source, expected)
        or is_link_or_reparse_point(source)
        or not source.is_file()
    ):
        raise ValueError(f"Staged artifact member {source.name!r} changed before installation.")
    os.link(source, destination, follow_symlinks=False)
    installed = _lstat_identity(destination)
    if installed != expected or installed.kind != stat.S_IFREG or is_link_or_reparse_point(destination):
        raise ValueError(f"Installed artifact member {destination.name!r} changed unexpectedly.")
    if not _has_identity(source, expected):
        raise ValueError(f"Staged artifact member {source.name!r} changed during installation.")
    return installed


def _unlink_if_identity(path: Path, expected: _PathIdentity) -> bool:
    if not _has_identity(path, expected) or is_link_or_reparse_point(path):
        return False
    try:
        if expected.kind == stat.S_IFREG and path.is_file():
            path.unlink()
            return True
    except OSError:
        return False
    return False


def _remove_owned_directory(
    root: Path,
    *,
    root_identity: _PathIdentity,
    member_identities: Mapping[str, _PathIdentity],
) -> bool:
    if not _has_identity(root, root_identity) or is_link_or_reparse_point(root) or not root.is_dir():
        return False
    for name, expected in member_identities.items():
        _unlink_if_identity(root / name, expected)
    if not _has_identity(root, root_identity) or is_link_or_reparse_point(root):
        return False
    try:
        if any(root.iterdir()):
            return False
        root.rmdir()
        _fsync_directory(root.parent)
    except OSError:
        return False
    return True


def _snapshot_cleanup_members(state: _PublicationState) -> dict[str, _PathIdentity]:
    if state.source_identities:
        return dict(state.source_identities)
    try:
        return _regular_members(
            state.stage,
            root_identity=state.stage_identity,
            expected_members=state.expected_members,
            require_exact=False,
        )
    except (OSError, ValueError):
        return {}


def _commit_publication(state: _PublicationState) -> None:
    sources = _regular_members(
        state.stage,
        root_identity=state.stage_identity,
        expected_members=state.expected_members,
        require_exact=True,
    )
    state.source_identities.update(sources)
    for name in state.expected_members:
        _fsync_regular_file(state.stage / name, expected=sources[name])
    _fsync_directory(state.stage, expected=state.stage_identity)

    state.target_identity = _reserve_target_directory(state.target)
    data_members = tuple(name for name in state.expected_members if name != _MANIFEST_NAME)
    for name in data_members:
        _require_directory_identity(
            state.target,
            state.target_identity,
            label="Reserved Image GICO target",
        )
        _link_regular_file_no_replace(
            state.stage / name,
            state.target / name,
            expected=sources[name],
        )
    _fsync_directory(state.target, expected=state.target_identity)

    _require_directory_identity(
        state.target,
        state.target_identity,
        label="Reserved Image GICO target",
    )
    _link_regular_file_no_replace(
        state.stage / _MANIFEST_NAME,
        state.target / _MANIFEST_NAME,
        expected=sources[_MANIFEST_NAME],
    )
    _fsync_directory(state.target, expected=state.target_identity)
    _fsync_directory(state.target.parent)
    state.committed = True


@contextmanager
def staged_image_gico_directory(
    output_dir: str | Path,
    *,
    expected_members: Collection[str],
    label: str = "Image GICO artifact directory",
) -> Iterator[Path]:
    """Stage and manifest-commit one flat, strict, no-replace artifact directory."""

    names = _expected_member_names(expected_members)
    target = _normalized_leaf_directory(output_dir, label=label, create_parent=True)
    if os.path.lexists(target):
        raise FileExistsError(f"Refusing to overwrite existing {label}: {target.name!r}.")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    stage_identity = _lstat_identity(stage)
    if stage_identity.kind != stat.S_IFDIR or is_link_or_reparse_point(stage):
        raise ValueError("Image GICO staging path is not a regular directory.")
    state = _PublicationState(
        target=target,
        stage=stage,
        expected_members=names,
        stage_identity=stage_identity,
    )
    try:
        yield stage
        _commit_publication(state)
    except BaseException:
        if state.target_identity is not None and not state.committed:
            _remove_owned_directory(
                state.target,
                root_identity=state.target_identity,
                member_identities=state.source_identities,
            )
        raise
    finally:
        stage_members = _snapshot_cleanup_members(state)
        _remove_owned_directory(
            state.stage,
            root_identity=state.stage_identity,
            member_identities=stage_members,
        )


def validate_image_gico_directory(
    output_dir: str | Path,
    *,
    expected_members: Collection[str],
    label: str = "Image GICO artifact directory",
) -> Path:
    """Validate one committed flat artifact directory without following its leaf."""

    names = _expected_member_names(expected_members)
    root = image_gico_directory_root(output_dir, label=label)
    root_identity = _lstat_identity(root)
    _regular_members(
        root,
        root_identity=root_identity,
        expected_members=names,
        require_exact=True,
    )
    return root


def image_gico_directory_root(
    output_dir: str | Path,
    *,
    label: str = "Image GICO artifact directory",
) -> Path:
    """Resolve an existing artifact root while rejecting a linked lexical leaf."""

    root = _normalized_leaf_directory(output_dir, label=label, create_parent=False)
    if not os.path.lexists(root):
        raise FileNotFoundError(root)
    if is_link_or_reparse_point(root) or not root.is_dir():
        raise ValueError(f"{label} must be a regular directory.")
    return root


def _sha256_regular_file(path: Path, *, expected: _PathIdentity | None = None) -> tuple[str, _PathIdentity]:
    if is_link_or_reparse_point(path) or not path.is_file():
        raise ValueError(f"Artifact member {path.name!r} must be a regular file.")
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        observed = _identity(os.fstat(descriptor))
        if observed.kind != stat.S_IFREG or (expected is not None and observed != expected):
            raise ValueError(f"Artifact member {path.name!r} changed while it was opened.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if not _has_identity(path, observed) or is_link_or_reparse_point(path):
            raise ValueError(f"Artifact member {path.name!r} changed while it was hashed.")
        return digest, observed
    finally:
        os.close(descriptor)


def write_float64_npy(path: str | Path, value: np.ndarray) -> dict[str, Any]:
    """Write one finite little-endian float64 NPY file without replacement."""

    target = Path(path)
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.floating):
        raise TypeError("Image GICO artifact arrays must use a floating-point dtype.")
    array = np.ascontiguousarray(raw, dtype="<f8")
    if array.size == 0 or not bool(np.isfinite(array).all()):
        raise ValueError("Image GICO artifact arrays must be nonempty and finite.")
    if os.path.lexists(target):
        raise FileExistsError(target)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(target, flags, 0o600)
    identity = _identity(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        _unlink_if_identity(target, identity)
        raise
    os.close(descriptor)
    try:
        digest, observed = _sha256_regular_file(target, expected=identity)
        if observed != identity:
            raise ValueError("Image GICO artifact array changed after writing.")
    except BaseException:
        _unlink_if_identity(target, identity)
        raise
    return {
        "file": target.name,
        "sha256": digest,
        "shape": list(array.shape),
        "dtype": "float64-le",
    }


def load_float64_npy(root: str | Path, descriptor: object, *, field: str) -> np.ndarray:
    """Strictly load one finite little-endian float64 NPY descriptor."""

    if not isinstance(descriptor, Mapping) or set(descriptor) != _FLOAT64_DESCRIPTOR_FIELDS:
        raise ValueError(f"Invalid {field} array descriptor.")
    filename = _portable_member_name(descriptor["file"])
    digest = descriptor["sha256"]
    shape = descriptor["shape"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _LOWERCASE_SHA256 for character in digest)
    ):
        raise ValueError(f"Invalid {field} array SHA-256 digest.")
    if (
        not isinstance(shape, list)
        or any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0 for dimension in shape)
        or descriptor["dtype"] != "float64-le"
    ):
        raise ValueError(f"Invalid {field} array metadata.")

    root_path = _normalized_leaf_directory(root, label=f"{field} array root", create_parent=False)
    if is_link_or_reparse_point(root_path) or not root_path.is_dir():
        raise ValueError(f"{field} array root must be a regular directory.")
    path = root_path / filename
    observed_digest, identity = _sha256_regular_file(path)
    if observed_digest != digest:
        raise ValueError(f"{field} array hash changed.")

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    file_descriptor = os.open(path, flags)
    try:
        if _identity(os.fstat(file_descriptor)) != identity:
            raise ValueError(f"{field} array changed before loading.")
        with os.fdopen(file_descriptor, "rb", closefd=False) as handle:
            array = np.load(handle, allow_pickle=False)
    finally:
        os.close(file_descriptor)
    if not _has_identity(path, identity) or is_link_or_reparse_point(path):
        raise ValueError(f"{field} array changed while loading.")
    if (
        array.dtype != np.dtype("<f8")
        or list(array.shape) != shape
        or array.size == 0
        or not bool(np.isfinite(array).all())
    ):
        raise ValueError(f"{field} array metadata or values changed.")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "image_gico_directory_root",
    "load_float64_npy",
    "staged_image_gico_directory",
    "validate_image_gico_directory",
    "write_float64_npy",
]
