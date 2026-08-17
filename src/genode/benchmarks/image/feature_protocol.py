from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from genode.artifacts.identity import canonical_json_bytes, semantic_sha256
from genode.path_safety import is_link_or_reparse_point
from genode.provenance import file_sha256

TORCH_FIDELITY_DISTRIBUTION = "torch-fidelity"
TORCH_FIDELITY_VERSION = "0.4.0"
TORCH_FIDELITY_FEATURE_EXTRACTOR = "inception-v3-compat"
TORCH_FIDELITY_FEATURE_LAYER = "2048"
TORCH_FIDELITY_FEATURE_DIMENSION = 2_048
TORCH_FIDELITY_INCEPTION_WEIGHTS_FILENAME = "weights-inception-2015-12-05-6726825d.pth"
TORCH_FIDELITY_INCEPTION_WEIGHTS_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
TORCH_FIDELITY_INCEPTION_WEIGHTS_SIZE_BYTES = 95_628_359
TORCH_FIDELITY_INCEPTION_WEIGHTS_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"
)

IMAGE_FEATURE_PROTOCOL_SCHEMA = "genode_image_feature_protocol_v1"
IMAGE_FEATURE_BLOCK_PROTOCOL = "genode_image_feature_block_v1"
_FEATURE_MATRIX_HASH_SCHEME = b"genode-image-feature-matrix-f32le-v1\0"
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _raw_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
    return value


def _semantic_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a namespaced semantic SHA-256 identity.")
    namespace, separator, digest = value.rpartition(":")
    if not separator or not namespace or namespace.strip() != namespace or _RAW_SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a namespaced semantic SHA-256 identity.")
    return value


@dataclass(frozen=True, slots=True)
class ImageFeatureProtocol:
    """The one concrete feature space used by image KID and locked metrics."""

    inception_weights_sha256: str = TORCH_FIDELITY_INCEPTION_WEIGHTS_SHA256

    def __post_init__(self) -> None:
        digest = _raw_sha256(
            self.inception_weights_sha256,
            label="inception_weights_sha256",
        )
        if digest != TORCH_FIDELITY_INCEPTION_WEIGHTS_SHA256:
            raise ValueError("Image metrics require the pinned torch-fidelity InceptionV3 weight file.")
        object.__setattr__(self, "inception_weights_sha256", digest)

    def preprocessing_payload(self) -> dict[str, object]:
        return {
            "color_space": "rgb",
            "input_dtype": "uint8",
            "input_layout": "nchw",
            "input_value_range": [0, 255],
            "resize": {
                "implementation": "torch_fidelity_tensorflow1x_bilinear",
                "output_size": [299, 299],
                "align_corners": False,
            },
            "normalization": "(x-128)/128",
        }

    @property
    def preprocessing_sha256(self) -> str:
        return semantic_sha256(
            self.preprocessing_payload(),
            namespace="image-feature-preprocessing",
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": IMAGE_FEATURE_PROTOCOL_SCHEMA,
            "backend": {
                "distribution": TORCH_FIDELITY_DISTRIBUTION,
                "version": TORCH_FIDELITY_VERSION,
            },
            "extractor": {
                "key": TORCH_FIDELITY_FEATURE_EXTRACTOR,
                "feature_layer": TORCH_FIDELITY_FEATURE_LAYER,
                "feature_dimension": TORCH_FIDELITY_FEATURE_DIMENSION,
                "internal_dtype": "float32",
                "weights": {
                    "filename": TORCH_FIDELITY_INCEPTION_WEIGHTS_FILENAME,
                    "sha256": self.inception_weights_sha256,
                    "size_bytes": (TORCH_FIDELITY_INCEPTION_WEIGHTS_SIZE_BYTES),
                    "source_url": TORCH_FIDELITY_INCEPTION_WEIGHTS_URL,
                    "implicit_download": False,
                },
            },
            "preprocessing": self.preprocessing_payload(),
            "preprocessing_sha256": self.preprocessing_sha256,
            "kid": {
                "estimator": "unbiased_mmd2",
                "kernel": "polynomial",
                "degree": 3,
                "gamma": "inverse_feature_dimension",
                "coef0": 1.0,
            },
            "locked_metrics": {
                "inception_score_layer": "logits_unbiased",
                "fid_layer": TORCH_FIDELITY_FEATURE_LAYER,
                "precision_recall_layer": (TORCH_FIDELITY_FEATURE_LAYER),
            },
        }

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            self.identity_payload(),
            namespace="image-feature-protocol",
        )

    def as_payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "protocol_sha256": self.sha256,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ImageFeatureProtocol:
        if not isinstance(payload, Mapping):
            raise TypeError("Image feature protocol must be a mapping.")
        protocol = cls()
        if canonical_json_bytes(dict(payload)) != canonical_json_bytes(protocol.as_payload()):
            raise ValueError("Image feature protocol does not match the pinned torch-fidelity protocol.")
        return protocol


_VERIFIED_WEIGHTS_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedImageFeatureWeights:
    """Runtime-only verified weight path; never serialize this object."""

    path: Path
    protocol: ImageFeatureProtocol

    def __init__(
        self,
        path: Path,
        protocol: ImageFeatureProtocol,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _VERIFIED_WEIGHTS_CONSTRUCTION_TOKEN:
            raise TypeError("VerifiedImageFeatureWeights must be created by verify_image_feature_weights().")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "protocol", protocol)


def verify_image_feature_weights(
    path: str | Path,
    *,
    protocol: ImageFeatureProtocol | None = None,
) -> VerifiedImageFeatureWeights:
    selected_protocol = ImageFeatureProtocol() if protocol is None else protocol
    if not isinstance(selected_protocol, ImageFeatureProtocol):
        raise TypeError("protocol must be an ImageFeatureProtocol.")
    raw = Path(path).expanduser()
    if raw.name != TORCH_FIDELITY_INCEPTION_WEIGHTS_FILENAME:
        raise ValueError("Image feature weights must use the pinned torch-fidelity filename.")
    if is_link_or_reparse_point(raw):
        raise ValueError("Image feature weights must not be a symlink, junction, or reparse point.")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Image feature weights must be a regular file.")
    before = resolved.stat()
    if before.st_size != TORCH_FIDELITY_INCEPTION_WEIGHTS_SIZE_BYTES:
        raise ValueError("Image feature weight-file size does not match the pinned file.")
    digest = file_sha256(resolved)
    after = resolved.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("Image feature weights changed while being verified.")
    if digest != selected_protocol.inception_weights_sha256:
        raise ValueError("Image feature weight-file SHA-256 does not match the protocol.")
    return VerifiedImageFeatureWeights(
        path=resolved,
        protocol=selected_protocol,
        _token=_VERIFIED_WEIGHTS_CONSTRUCTION_TOKEN,
    )


def feature_matrix_sha256(
    features: object,
    *,
    expected_rows: int | None = None,
) -> tuple[str, bytes, int]:
    if isinstance(features, torch.Tensor):
        if features.device.type != "cpu":
            raise ValueError("Feature matrices must be supplied on CPU.")
        features = features.detach().numpy()
    matrix = np.asarray(features)
    if matrix.ndim != 2:
        raise ValueError("Feature matrix must have shape [samples, 2048].")
    if matrix.shape[1] != TORCH_FIDELITY_FEATURE_DIMENSION:
        raise ValueError("Feature matrix must use the torch-fidelity InceptionV3 2048 layer.")
    if matrix.shape[0] < 2:
        raise ValueError("Feature matrix requires at least two rows.")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError("Feature matrix row count does not match its request.")
    if not np.issubdtype(matrix.dtype, np.floating):
        raise ValueError("Feature matrix must use a floating-point dtype.")
    canonical = np.ascontiguousarray(matrix, dtype="<f4")
    if not np.all(np.isfinite(canonical)):
        raise ValueError("Feature matrix contains non-finite values.")
    encoded = canonical.tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(_FEATURE_MATRIX_HASH_SCHEME)
    digest.update(int(canonical.shape[0]).to_bytes(8, "big"))
    digest.update(int(canonical.shape[1]).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest(), encoded, int(canonical.shape[0])


@dataclass(frozen=True, slots=True)
class FeatureBlock:
    """An immutable generated or real feature block bound to one request."""

    role: str
    request_sha256: str
    panel_fingerprint: str
    block_index: int
    feature_protocol: ImageFeatureProtocol
    row_count: int
    feature_matrix_sha256: str
    _feature_bytes: bytes = field(repr=False)
    real_feature_panel_sha256: str | None = None
    block_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.role not in {"generated", "real"}:
            raise ValueError("Feature-block role must be 'generated' or 'real'.")
        request_hash = _semantic_sha256(
            self.request_sha256,
            label="request_sha256",
        )
        panel_hash = _semantic_sha256(
            self.panel_fingerprint,
            label="panel_fingerprint",
        )
        if isinstance(self.block_index, bool) or not isinstance(self.block_index, int) or self.block_index < 0:
            raise ValueError("block_index must be a nonnegative integer.")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count < 2:
            raise ValueError("row_count must be an integer of at least two.")
        if not isinstance(self.feature_protocol, ImageFeatureProtocol):
            raise TypeError("feature_protocol must be an ImageFeatureProtocol.")
        matrix_hash = _raw_sha256(
            self.feature_matrix_sha256,
            label="feature_matrix_sha256",
        )
        encoded = bytes(self._feature_bytes)
        expected_bytes = self.row_count * TORCH_FIDELITY_FEATURE_DIMENSION * 4
        if len(encoded) != expected_bytes:
            raise ValueError("Feature-block byte length does not match its shape.")
        observed_hash, observed_bytes, observed_rows = feature_matrix_sha256(
            np.frombuffer(encoded, dtype="<f4").reshape(
                self.row_count,
                TORCH_FIDELITY_FEATURE_DIMENSION,
            ),
            expected_rows=self.row_count,
        )
        if observed_hash != matrix_hash or observed_bytes != encoded or observed_rows != self.row_count:
            raise ValueError("Feature-block content does not match its matrix hash.")
        if self.role == "real":
            real_panel_hash = _semantic_sha256(
                self.real_feature_panel_sha256,
                label="real_feature_panel_sha256",
            )
        else:
            if self.real_feature_panel_sha256 is not None:
                raise ValueError("Generated feature blocks cannot claim a real-feature panel.")
            real_panel_hash = None

        object.__setattr__(self, "request_sha256", request_hash)
        object.__setattr__(self, "panel_fingerprint", panel_hash)
        object.__setattr__(self, "feature_matrix_sha256", matrix_hash)
        object.__setattr__(self, "_feature_bytes", encoded)
        object.__setattr__(
            self,
            "real_feature_panel_sha256",
            real_panel_hash,
        )
        object.__setattr__(
            self,
            "block_sha256",
            semantic_sha256(
                {
                    "protocol": IMAGE_FEATURE_BLOCK_PROTOCOL,
                    "role": self.role,
                    "request_sha256": request_hash,
                    "panel_fingerprint": panel_hash,
                    "block_index": self.block_index,
                    "row_count": self.row_count,
                    "feature_protocol_sha256": self.feature_protocol.sha256,
                    "feature_matrix_sha256": matrix_hash,
                    "real_feature_panel_sha256": real_panel_hash,
                },
                namespace="image-feature-block",
            ),
        )

    @classmethod
    def from_features(
        cls,
        features: object,
        *,
        role: str,
        request_sha256: str,
        panel_fingerprint: str,
        block_index: int,
        feature_protocol: ImageFeatureProtocol,
        real_feature_panel_sha256: str | None = None,
        expected_rows: int | None = None,
    ) -> FeatureBlock:
        matrix_hash, encoded, row_count = feature_matrix_sha256(
            features,
            expected_rows=expected_rows,
        )
        return cls(
            role=role,
            request_sha256=request_sha256,
            panel_fingerprint=panel_fingerprint,
            block_index=block_index,
            feature_protocol=feature_protocol,
            row_count=row_count,
            feature_matrix_sha256=matrix_hash,
            _feature_bytes=encoded,
            real_feature_panel_sha256=real_feature_panel_sha256,
        )

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.frombuffer(self._feature_bytes, dtype="<f4").reshape(
            self.row_count,
            TORCH_FIDELITY_FEATURE_DIMENSION,
        )
        matrix.setflags(write=False)
        return matrix


__all__ = [
    "FeatureBlock",
    "IMAGE_FEATURE_BLOCK_PROTOCOL",
    "IMAGE_FEATURE_PROTOCOL_SCHEMA",
    "ImageFeatureProtocol",
    "TORCH_FIDELITY_DISTRIBUTION",
    "TORCH_FIDELITY_FEATURE_DIMENSION",
    "TORCH_FIDELITY_FEATURE_EXTRACTOR",
    "TORCH_FIDELITY_FEATURE_LAYER",
    "TORCH_FIDELITY_INCEPTION_WEIGHTS_FILENAME",
    "TORCH_FIDELITY_INCEPTION_WEIGHTS_SHA256",
    "TORCH_FIDELITY_INCEPTION_WEIGHTS_SIZE_BYTES",
    "TORCH_FIDELITY_VERSION",
    "VerifiedImageFeatureWeights",
    "feature_matrix_sha256",
    "verify_image_feature_weights",
]
