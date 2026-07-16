from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath


MANIFEST_PARENT_PATH_BASE = "manifest_parent"


def portable_relative_path(value: str | os.PathLike[str], *, label: str = "path") -> PurePosixPath:
    """Validate and return a portable POSIX-relative path.

    Public manifests use forward-slash paths so they have identical meaning on
    POSIX and Windows.  Rejecting Windows syntax explicitly is important when a
    manifest is validated on POSIX before being consumed on Windows.
    """

    raw_text = os.fspath(value)
    text = raw_text.strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty portable relative path.")
    if text != raw_text:
        raise ValueError(f"{label} may not have leading or trailing whitespace: {raw_text!r}.")
    if "\x00" in text or "\\" in text:
        raise ValueError(f"{label} must use portable POSIX separators: {text!r}.")

    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    raw_parts = text.split("/")
    if posix.is_absolute() or windows.is_absolute() or bool(windows.drive) or bool(windows.root):
        raise ValueError(f"{label} must be relative: {text!r}.")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{label} may not contain empty, '.' or '..' components: {text!r}.")
    return posix


def is_link_or_reparse_point(path: str | os.PathLike[str]) -> bool:
    """Return whether *path* is a symlink, junction, or other reparse point."""

    candidate = Path(path)
    if candidate.is_symlink():
        return True
    is_junction = getattr(os.path, "isjunction", None)
    if is_junction is not None and is_junction(candidate):
        return True
    try:
        attributes = int(getattr(candidate.lstat(), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))


def first_link_or_reparse_component(path: Path, *, root: Path) -> Path | None:
    """Find filesystem indirection between *root* and *path*, including root."""

    root_absolute = root.expanduser().absolute()
    path_absolute = path.expanduser().absolute()
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError(f"Path {path_absolute} is outside root {root_absolute}.") from exc

    current = root_absolute
    if is_link_or_reparse_point(current):
        return current
    for part in relative.parts:
        current = current / part
        if is_link_or_reparse_point(current):
            return current
    return None


def resolve_portable_relative_path(
    root: str | os.PathLike[str],
    value: str | os.PathLike[str],
    *,
    label: str = "path",
    reject_links: bool = False,
) -> Path:
    """Resolve a validated portable relative path under *root*.

    Existing symlink/reparse components are optionally rejected.  Resolved
    containment is always checked so an existing indirection cannot escape the
    declared root.
    """

    root_path = Path(root).expanduser().resolve()
    relative = portable_relative_path(value, label=label)
    candidate = root_path.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root_path):
        raise ValueError(f"{label} escapes its declared root: {str(value)!r}.")
    if reject_links:
        indirect = first_link_or_reparse_component(candidate, root=root_path)
        if indirect is not None:
            raise ValueError(f"{label} traverses a symlink, junction, or reparse point: {indirect}.")
    return resolved


def resolve_manifest_path_base(manifest_path: str | os.PathLike[str], value: object) -> Path:
    """Resolve the single portable manifest anchor.

    ``path_base`` is an enum rather than a filesystem path. Keeping the
    manifest beside its declared base removes the need for ``.`` or ``..``
    traversal while preserving relocatable member paths.
    """

    token = str(value or "").strip()
    if token != MANIFEST_PARENT_PATH_BASE:
        raise ValueError(
            f"Manifest path_base must be {MANIFEST_PARENT_PATH_BASE!r}, got {token!r}."
        )
    return Path(manifest_path).expanduser().resolve().parent


__all__ = [
    "first_link_or_reparse_component",
    "is_link_or_reparse_point",
    "portable_relative_path",
    "MANIFEST_PARENT_PATH_BASE",
    "resolve_manifest_path_base",
    "resolve_portable_relative_path",
]
