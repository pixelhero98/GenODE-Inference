from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from genode.path_safety import is_link_or_reparse_point
from genode.provenance import file_sha256

from .registry import get_image_backbone_spec


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class CheckpointBinding:
    filename: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.filename or Path(self.filename).name != self.filename:
            raise ValueError("Checkpoint filename must be one portable basename.")
        if "\\" in self.filename or "/" in self.filename:
            raise ValueError("Checkpoint filename must not contain path separators.")
        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("Checkpoint sha256 must be one lowercase SHA-256 digest.")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise ValueError("Checkpoint size_bytes must be a positive integer.")

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_manifest_dict(cls, value: object) -> CheckpointBinding:
        if not isinstance(value, Mapping):
            raise ValueError("Checkpoint binding must be a JSON object.")
        expected = {"filename", "sha256", "size_bytes"}
        if set(value) != expected:
            raise ValueError(f"Checkpoint binding fields must be exactly {sorted(expected)}.")
        filename = value["filename"]
        sha256 = value["sha256"]
        size_bytes = value["size_bytes"]
        if not isinstance(filename, str) or not isinstance(sha256, str):
            raise ValueError("Checkpoint filename and sha256 must be strings.")
        return cls(filename=filename, sha256=sha256, size_bytes=size_bytes)  # type: ignore[arg-type]


def _checked_checkpoint_path(model_key: str, checkpoint_path: str | Path) -> Path:
    spec = get_image_backbone_spec(model_key)
    raw_path = Path(checkpoint_path).expanduser()
    if raw_path.name != spec.checkpoint_filename:
        raise ValueError(
            f"Checkpoint for {model_key!r} must be named {spec.checkpoint_filename!r}, got {raw_path.name!r}."
        )
    if is_link_or_reparse_point(raw_path):
        raise ValueError("Checkpoint must not be a symlink, junction, or reparse point.")
    path = raw_path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("Checkpoint must be a regular file.")
    return path


def _hash_file_without_adoption(path: Path) -> tuple[str, int]:
    before = path.stat()
    if before.st_size <= 0:
        raise ValueError("Checkpoint must not be empty.")
    digest = file_sha256(path)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError("Checkpoint changed while it was being hashed.")
    return digest, int(after.st_size)


def bind_checkpoint(model_key: str, checkpoint_path: str | Path) -> CheckpointBinding:
    path = _checked_checkpoint_path(model_key, checkpoint_path)
    digest, size_bytes = _hash_file_without_adoption(path)
    return CheckpointBinding(filename=path.name, sha256=digest, size_bytes=size_bytes)


def verify_checkpoint_binding(
    model_key: str,
    checkpoint_path: str | Path,
    binding: CheckpointBinding,
) -> Path:
    observed = bind_checkpoint(model_key, checkpoint_path)
    if observed != binding:
        raise ValueError("Checkpoint bytes do not match the portable manifest binding.")
    return _checked_checkpoint_path(model_key, checkpoint_path)


def validate_formal_image_checkpoint_binding(
    model_key: str,
    binding: CheckpointBinding,
) -> None:
    """Require the authors' published filename and byte count.

    RF++ publishes no cryptographic checkpoint digest. The portable binding's
    SHA-256 is therefore a local content identity; the upstream Drive filename
    and byte count are the available acquisition metadata.
    """

    if not isinstance(binding, CheckpointBinding):
        raise TypeError("binding must be a CheckpointBinding.")
    spec = get_image_backbone_spec(model_key)
    if binding.filename != spec.checkpoint_filename or binding.size_bytes != spec.checkpoint_published_size_bytes:
        raise ValueError(
            f"Formal {model_key!r} requires the authors' "
            f"{spec.checkpoint_filename!r} object with published size "
            f"{spec.checkpoint_published_size_bytes} bytes."
        )


__all__ = [
    "CheckpointBinding",
    "bind_checkpoint",
    "validate_formal_image_checkpoint_binding",
    "verify_checkpoint_binding",
]
