"""Replayable SHA-256 counter uniforms for causal-AR clock sampling."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Sequence

import torch
from torch import Tensor

from genode.gico.image_causal_stick import STICK_ACTION_COUNT, TARGET_NFES

IMAGE_GICO_CAUSAL_RNG_PROTOCOL = "image_gico_causal_ar_sha256_counter_rng_v1"
IMAGE_GICO_CAUSAL_UNIFORMS_NAMESPACE = "image-gico-causal-ar-uniforms-f64le-v1"
_DOMAIN = b"image-gico-causal-ar-clock-v1\0"
_IDENTITY = re.compile(r"(?:[a-z][a-z0-9_.-]*:)?[0-9a-f]{64}\Z")
_MANTISSA_BITS = 52
_OPEN_DENOMINATOR = 2**_MANTISSA_BITS + 1


def _identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 identity.")
    return value


def _field(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(4, "big") + encoded


def _open_unit_float64(digest: bytes) -> float:
    mantissa = int.from_bytes(digest[:8], "big") >> (64 - _MANTISSA_BITS)
    value = float(mantissa + 1) / float(_OPEN_DENOMINATOR)
    if not 0.0 < value < 1.0 or not math.isfinite(value):
        raise RuntimeError("Counter conversion left the open unit interval.")
    return value


def derive_image_gico_causal_uniforms(
    *,
    artifact_sha256: object,
    request_sha256: object,
    target_nfe: object,
    sample_keys: Sequence[str | int],
) -> Tensor:
    """Return replayable CPU-float64 uniforms with shape ``[batch, 63]``."""

    artifact = _identity(artifact_sha256, field="artifact_sha256")
    request = _identity(request_sha256, field="request_sha256")
    if isinstance(target_nfe, bool) or not isinstance(target_nfe, int):
        raise TypeError("target_nfe must be an integer.")
    if target_nfe not in TARGET_NFES:
        raise ValueError(f"target_nfe must be one of {TARGET_NFES}.")
    if isinstance(sample_keys, (str, bytes)) or not isinstance(sample_keys, Sequence):
        raise TypeError("sample_keys must be a sequence.")
    if not sample_keys:
        raise ValueError("sample_keys must be nonempty.")
    normalized: list[str] = []
    for value in sample_keys:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError("sample_keys must contain strings or integers.")
        text = str(value)
        if not text:
            raise ValueError("sample_keys must not contain empty strings.")
        normalized.append(text)
    if len(set(normalized)) != len(normalized):
        raise ValueError("sample_keys must be unique within a request.")
    prefix = b"".join(
        (
            _DOMAIN,
            _field(IMAGE_GICO_CAUSAL_RNG_PROTOCOL),
            _field(artifact),
            _field(request),
            target_nfe.to_bytes(1, "big"),
        )
    )
    rows = [
        [
            _open_unit_float64(hashlib.sha256(prefix + _field(sample_key) + step.to_bytes(1, "big")).digest())
            for step in range(STICK_ACTION_COUNT)
        ]
        for sample_key in normalized
    ]
    return torch.tensor(rows, dtype=torch.float64, device="cpu")


def image_gico_causal_uniforms_sha256(uniforms: Tensor) -> str:
    if (
        not isinstance(uniforms, Tensor)
        or uniforms.device.type != "cpu"
        or uniforms.dtype != torch.float64
        or uniforms.ndim != 2
        or uniforms.shape[0] <= 0
        or uniforms.shape[1] != STICK_ACTION_COUNT
    ):
        raise ValueError("uniforms must be CPU float64 with shape [batch, 63].")
    digest = hashlib.sha256()
    digest.update(IMAGE_GICO_CAUSAL_UNIFORMS_NAMESPACE.encode("ascii"))
    digest.update(b"\0")
    digest.update(int(uniforms.shape[0]).to_bytes(8, "big"))
    for value in uniforms.reshape(-1).tolist():
        parsed = float(value)
        if not 0.0 < parsed < 1.0 or not math.isfinite(parsed):
            raise ValueError("uniforms must lie in the finite open unit interval.")
        digest.update(struct.pack("<d", parsed))
    return f"{IMAGE_GICO_CAUSAL_UNIFORMS_NAMESPACE}:{digest.hexdigest()}"


__all__ = [
    "IMAGE_GICO_CAUSAL_RNG_PROTOCOL",
    "derive_image_gico_causal_uniforms",
    "image_gico_causal_uniforms_sha256",
]
