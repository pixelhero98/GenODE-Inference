from __future__ import annotations

import os
from pathlib import Path

from genode.path_safety import is_link_or_reparse_point


def _checked_raw_path(
    path: str | Path,
    *,
    label: str,
) -> Path:
    raw = Path(path).expanduser()
    if raw.name in {"", ".", ".."} or raw.name.strip() != raw.name:
        raise ValueError(f"{label} must name one managed leaf.")
    if ".." in raw.parts:
        raise ValueError(f"{label} may not contain '..' path components.")

    absolute = raw.absolute()
    if is_link_or_reparse_point(absolute):
        raise ValueError(f"{label} must not be a symlink, junction, or reparse point.")
    return raw


def managed_image_leaf_directory(
    path: str | Path,
    *,
    label: str,
) -> Path:
    """Return one safe image-artifact leaf beneath an existing raw parent."""

    raw = _checked_raw_path(path, label=label)
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} parent must be an existing directory.") from exc
    if not parent.is_dir():
        raise ValueError(f"{label} parent must be an existing directory.")
    target = parent / raw.name
    if os.path.lexists(target) and not target.is_dir():
        raise ValueError(f"{label} exists but is not a directory.")
    return target


def managed_image_json_path(
    path: str | Path,
    *,
    label: str,
) -> Path:
    """Return one safe managed JSON leaf beneath an existing raw parent."""

    raw = _checked_raw_path(path, label=label)
    if raw.suffix.lower() != ".json":
        raise ValueError(f"{label} must name one .json file.")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} parent must be an existing directory.") from exc
    if not parent.is_dir():
        raise ValueError(f"{label} parent must be an existing directory.")
    target = parent / raw.name
    if os.path.lexists(target) and not target.is_file():
        raise ValueError(f"{label} exists but is not a regular file.")
    return target


__all__ = [
    "managed_image_json_path",
    "managed_image_leaf_directory",
]
