from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from genode.data.otflow_paths import display_project_path


def _file_sha256(resolved_path: str) -> str:
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of the file's current bytes.

    Integrity checks intentionally do not memoize by pathname: callers may
    validate artifacts after an atomic replacement in the same process.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Expected a regular file for provenance hashing: {display_project_path(resolved)}")
    return _file_sha256(str(resolved))


def path_fingerprint(
    path: str | Path,
    *,
    manifest_names: Sequence[str] = ("manifest.json", "backbone_manifest.json", "package_manifest.json"),
) -> Dict[str, Any]:
    """Describe an input using sanitized location metadata and stable content hashes.

    Directory inputs must expose at least one named authoritative manifest. This
    avoids both machine-specific directory metadata and expensive, ambiguous
    recursive hashing of unrelated artifacts.
    """

    resolved = Path(path).expanduser().resolve()
    record: Dict[str, Any] = {
        "logical_path": display_project_path(resolved),
        "exists": bool(resolved.exists()),
    }
    if not resolved.exists():
        record["kind"] = "missing"
        return record
    if resolved.is_file():
        record.update(
            {
                "kind": "file",
                "size_bytes": int(resolved.stat().st_size),
                "sha256": file_sha256(resolved),
            }
        )
        return record
    if not resolved.is_dir():
        raise ValueError(f"Unsupported provenance input: {display_project_path(resolved)}")

    manifests = []
    for name in manifest_names:
        candidate = resolved / str(name)
        if candidate.is_file():
            manifests.append(
                {
                    "name": str(name),
                    "size_bytes": int(candidate.stat().st_size),
                    "sha256": file_sha256(candidate),
                }
            )
    if not manifests:
        expected = ", ".join(str(name) for name in manifest_names)
        raise ValueError(
            f"Directory provenance requires an authoritative manifest ({expected}): "
            f"{display_project_path(resolved)}"
        )
    encoded = "\n".join(
        f"{row['name']}\0{row['size_bytes']}\0{row['sha256']}" for row in manifests
    ).encode("utf-8")
    record.update(
        {
            "kind": "directory",
            "manifests": manifests,
            "content_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    )
    return record


def fingerprint_identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip display-only paths from a fingerprint before protocol hashing."""

    return {key: value for key, value in record.items() if key != "logical_path"}


__all__ = [
    "file_sha256",
    "fingerprint_identity",
    "path_fingerprint",
]
