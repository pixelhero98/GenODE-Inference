from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import PurePath
import re
from typing import Any


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _absolute_path_string(value: str) -> bool:
    return value.startswith(("/", "\\", "~/", "~\\")) or _WINDOWS_ABSOLUTE_PATH.match(value) is not None


def _json_value(value: Any, *, location: str) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if _absolute_path_string(value):
            raise ValueError(f"{location} contains an absolute path string.")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite float.")
        return value
    if isinstance(value, PurePath):
        path = value.as_posix()
        if value.is_absolute():
            raise ValueError(f"{location} contains an absolute path.")
        return path
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} contains a non-string object key {key!r}.")
            normalized[key] = _json_value(item, location=f"{location}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, location=f"{location}[{index}]") for index, item in enumerate(value)]
    raise TypeError(
        f"{location} contains unsupported value type {type(value).__name__}; "
        "artifact identities accept only explicit JSON values and relative paths."
    )


def canonical_json_text(payload: Any) -> str:
    """Serialize a strict JSON value with one deterministic representation."""

    normalized = _json_value(payload, location="payload")
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(payload: Any) -> bytes:
    return canonical_json_text(payload).encode("utf-8")


def semantic_sha256(payload: Any, *, namespace: str) -> str:
    """Return a namespaced SHA-256 identity for a strict JSON payload."""

    normalized_namespace = str(namespace).strip()
    if not normalized_namespace:
        raise ValueError("namespace must be non-empty.")
    digest = hashlib.sha256()
    digest.update(normalized_namespace.encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(payload))
    return f"{normalized_namespace}:{digest.hexdigest()}"


__all__ = [
    "canonical_json_bytes",
    "canonical_json_text",
    "semantic_sha256",
]
