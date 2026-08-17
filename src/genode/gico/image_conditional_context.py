from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch

from genode.artifacts.identity import canonical_json_bytes, semantic_sha256
from genode.backbones.adapter import (
    IMAGE_BACKBONE_CONTEXT_DIM,
    IMAGE_BACKBONE_CONTEXT_SELECTOR,
    CanonicalNoiseToDataAdapter,
)
from genode.backbones.registry import get_image_backbone_spec

IMAGE_GICO_BACKBONE_CONTEXT_PROTOCOL = "image_gico_backbone_context_binding_v3"
IMAGE_GICO_BACKBONE_CONTEXT_CLASS_COUNT = 1_000
IMAGE_GICO_BACKBONE_CONTEXT_DTYPE = "float32"
IMAGE_GICO_BACKBONE_CONTEXT_STD_FLOOR = 1.0e-6
_SHA256_IDENTITY = re.compile(r"^[a-z0-9-]+:[0-9a-f]{64}$")


def _array_identity(array: np.ndarray, *, namespace: str) -> str:
    value = np.asarray(array)
    if not value.flags.c_contiguous:
        raise ValueError("Context identity requires a C-contiguous array.")
    digest = hashlib.sha256()
    digest.update(namespace.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        canonical_json_bytes(
            {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            }
        )
    )
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return f"{namespace}:{digest.hexdigest()}"


def _identity(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a namespaced SHA-256 identity.")
    return value


def _plain_sha256(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _canonical_vector(
    value: np.ndarray | torch.Tensor,
    *,
    field_name: str,
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError(f"{field_name} must not require gradients.")
        array = value.detach().to(device="cpu").numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise TypeError(f"{field_name} must be a NumPy array or tensor.")
    if array.shape != (IMAGE_BACKBONE_CONTEXT_DIM,):
        raise ValueError(f"{field_name} must have shape [{IMAGE_BACKBONE_CONTEXT_DIM}].")
    if array.dtype != np.float32:
        raise ValueError(f"{field_name} must use float32 values.")
    result = np.ascontiguousarray(array, dtype="<f4")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must contain only finite values.")
    return result


def _canonical_table(
    value: np.ndarray | torch.Tensor,
    *,
    field_name: str,
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError(f"{field_name} must not require gradients.")
        if value.device.type != "cpu":
            raise ValueError(f"{field_name} must be materialized on CPU.")
        array = value.detach().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise TypeError(f"{field_name} must be a NumPy array or tensor.")
    expected_shape = (
        IMAGE_GICO_BACKBONE_CONTEXT_CLASS_COUNT,
        IMAGE_BACKBONE_CONTEXT_DIM,
    )
    if array.shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {list(expected_shape)}.")
    if array.dtype != np.float32:
        raise ValueError(f"{field_name} must use float32 values.")
    result = np.ascontiguousarray(array, dtype="<f4")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must contain only finite values.")
    return result


@dataclass(frozen=True, slots=True)
class ImageGICOContextNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    mean_sha256: str = field(init=False)
    scale_sha256: str = field(init=False)
    normalizer_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        mean = _canonical_vector(self.mean, field_name="context normalizer mean")
        scale = _canonical_vector(self.scale, field_name="context normalizer scale")
        if np.any(scale < np.float32(IMAGE_GICO_BACKBONE_CONTEXT_STD_FLOOR)):
            raise ValueError("Context normalizer scale violates the standard-deviation floor.")
        mean.setflags(write=False)
        scale.setflags(write=False)
        mean_sha256 = _array_identity(
            mean,
            namespace="image-gico-backbone-context-mean-v3",
        )
        scale_sha256 = _array_identity(
            scale,
            namespace="image-gico-backbone-context-scale-v3",
        )
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "mean_sha256", mean_sha256)
        object.__setattr__(self, "scale_sha256", scale_sha256)
        object.__setattr__(
            self,
            "normalizer_sha256",
            semantic_sha256(
                {
                    "protocol": "image_gico_backbone_context_normalizer_v3",
                    "dtype": IMAGE_GICO_BACKBONE_CONTEXT_DTYPE,
                    "context_dim": IMAGE_BACKBONE_CONTEXT_DIM,
                    "std_floor": IMAGE_GICO_BACKBONE_CONTEXT_STD_FLOOR,
                    "mean_sha256": mean_sha256,
                    "scale_sha256": scale_sha256,
                },
                namespace="image-gico-backbone-context-normalizer-v3",
            ),
        )

    def normalize(self, raw_table: np.ndarray | torch.Tensor) -> np.ndarray:
        table = _canonical_table(raw_table, field_name="raw backbone context table")
        normalized = (table - self.mean[None, :]) / self.scale[None, :]
        result = np.ascontiguousarray(normalized, dtype="<f4")
        if not np.all(np.isfinite(result)):
            raise ValueError("Normalized backbone contexts must be finite.")
        return result


@dataclass(frozen=True, slots=True)
class ImageGICOBackboneContextBinding:
    backbone_model_key: str
    backbone_protocol_sha256: str
    backbone_checkpoint_sha256: str
    source_revision: str
    source_config_identity: str
    raw_context_table_sha256: str
    normalized_context_table_sha256: str
    normalizer_mean_sha256: str
    normalizer_scale_sha256: str
    normalizer_sha256: str
    selector: str = IMAGE_BACKBONE_CONTEXT_SELECTOR
    class_count: int = IMAGE_GICO_BACKBONE_CONTEXT_CLASS_COUNT
    context_dim: int = IMAGE_BACKBONE_CONTEXT_DIM
    dtype: str = IMAGE_GICO_BACKBONE_CONTEXT_DTYPE
    normalization_std_floor: float = IMAGE_GICO_BACKBONE_CONTEXT_STD_FLOOR
    class_order_sha256: str = field(
        default_factory=lambda: semantic_sha256(
            {"class_ids": list(range(IMAGE_GICO_BACKBONE_CONTEXT_CLASS_COUNT))},
            namespace="image-gico-backbone-context-class-order-v3",
        )
    )
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.backbone_model_key, str) or not self.backbone_model_key:
            raise ValueError("backbone_model_key must be a non-empty string.")
        spec = get_image_backbone_spec(self.backbone_model_key)
        if spec.dataset_key != "imagenet64" or spec.num_conditioning_classes != self.class_count:
            raise ValueError("Backbone contexts require a 1,000-class ImageNet-64 model.")
        if self.source_revision != spec.source_revision:
            raise ValueError("Backbone context source revision does not match the registry.")
        if self.source_config_identity != spec.source_config_identity:
            raise ValueError("Backbone context source config does not match the registry.")
        _plain_sha256(
            self.backbone_protocol_sha256,
            field_name="backbone_protocol_sha256",
        )
        _plain_sha256(
            self.backbone_checkpoint_sha256,
            field_name="backbone_checkpoint_sha256",
        )
        for field_name in (
            "raw_context_table_sha256",
            "normalized_context_table_sha256",
            "normalizer_mean_sha256",
            "normalizer_scale_sha256",
            "normalizer_sha256",
            "class_order_sha256",
        ):
            _identity(getattr(self, field_name), field_name=field_name)
        if self.selector != IMAGE_BACKBONE_CONTEXT_SELECTOR:
            raise ValueError("Unsupported backbone context selector.")
        if self.class_count != IMAGE_GICO_BACKBONE_CONTEXT_CLASS_COUNT:
            raise ValueError("Backbone context class_count must be 1000.")
        if self.context_dim != IMAGE_BACKBONE_CONTEXT_DIM:
            raise ValueError("Backbone context dimension must be 768.")
        if self.dtype != IMAGE_GICO_BACKBONE_CONTEXT_DTYPE:
            raise ValueError("Backbone contexts must use float32.")
        if self.normalization_std_floor != IMAGE_GICO_BACKBONE_CONTEXT_STD_FLOOR:
            raise ValueError("Unsupported backbone context normalization floor.")
        object.__setattr__(
            self,
            "binding_sha256",
            semantic_sha256(
                self.identity_payload(),
                namespace="image-gico-backbone-context-binding-v3",
            ),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "protocol": IMAGE_GICO_BACKBONE_CONTEXT_PROTOCOL,
            "selector": self.selector,
            "class_count": self.class_count,
            "context_dim": self.context_dim,
            "dtype": self.dtype,
            "normalization_std_floor": self.normalization_std_floor,
            "class_order_sha256": self.class_order_sha256,
            "raw_context_table_sha256": self.raw_context_table_sha256,
            "normalized_context_table_sha256": self.normalized_context_table_sha256,
            "normalizer_mean_sha256": self.normalizer_mean_sha256,
            "normalizer_scale_sha256": self.normalizer_scale_sha256,
            "normalizer_sha256": self.normalizer_sha256,
            "backbone_model_key": self.backbone_model_key,
            "backbone_protocol_sha256": self.backbone_protocol_sha256,
            "backbone_checkpoint_sha256": self.backbone_checkpoint_sha256,
            "source_revision": self.source_revision,
            "source_config_identity": self.source_config_identity,
        }

    def as_payload(self) -> dict[str, object]:
        return {**self.identity_payload(), "binding_sha256": self.binding_sha256}

    @classmethod
    def from_payload(cls, payload: object) -> ImageGICOBackboneContextBinding:
        if not isinstance(payload, Mapping):
            raise TypeError("Backbone context binding must be a mapping.")
        expected = {
            "protocol",
            "selector",
            "class_count",
            "context_dim",
            "dtype",
            "normalization_std_floor",
            "class_order_sha256",
            "raw_context_table_sha256",
            "normalized_context_table_sha256",
            "normalizer_mean_sha256",
            "normalizer_scale_sha256",
            "normalizer_sha256",
            "backbone_model_key",
            "backbone_protocol_sha256",
            "backbone_checkpoint_sha256",
            "source_revision",
            "source_config_identity",
            "binding_sha256",
        }
        if set(payload) != expected:
            raise ValueError(f"Backbone context binding fields must be exactly {sorted(expected)}.")
        if payload["protocol"] != IMAGE_GICO_BACKBONE_CONTEXT_PROTOCOL:
            raise ValueError("Unsupported backbone context binding protocol.")
        binding = cls(
            backbone_model_key=payload["backbone_model_key"],  # type: ignore[arg-type]
            backbone_protocol_sha256=payload["backbone_protocol_sha256"],  # type: ignore[arg-type]
            backbone_checkpoint_sha256=payload["backbone_checkpoint_sha256"],  # type: ignore[arg-type]
            source_revision=payload["source_revision"],  # type: ignore[arg-type]
            source_config_identity=payload["source_config_identity"],  # type: ignore[arg-type]
            raw_context_table_sha256=payload["raw_context_table_sha256"],  # type: ignore[arg-type]
            normalized_context_table_sha256=payload["normalized_context_table_sha256"],  # type: ignore[arg-type]
            normalizer_mean_sha256=payload["normalizer_mean_sha256"],  # type: ignore[arg-type]
            normalizer_scale_sha256=payload["normalizer_scale_sha256"],  # type: ignore[arg-type]
            normalizer_sha256=payload["normalizer_sha256"],  # type: ignore[arg-type]
            selector=payload["selector"],  # type: ignore[arg-type]
            class_count=payload["class_count"],  # type: ignore[arg-type]
            context_dim=payload["context_dim"],  # type: ignore[arg-type]
            dtype=payload["dtype"],  # type: ignore[arg-type]
            normalization_std_floor=payload["normalization_std_floor"],  # type: ignore[arg-type]
            class_order_sha256=payload["class_order_sha256"],  # type: ignore[arg-type]
        )
        if payload["binding_sha256"] != binding.binding_sha256:
            raise ValueError("Backbone context binding hash is inconsistent.")
        return binding


@dataclass(frozen=True, slots=True)
class PreparedImageGICOBackboneContext:
    binding: ImageGICOBackboneContextBinding
    normalizer: ImageGICOContextNormalizer
    normalized_context_table: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ImageGICOBackboneContextBinding):
            raise TypeError("binding must be an ImageGICOBackboneContextBinding.")
        if not isinstance(self.normalizer, ImageGICOContextNormalizer):
            raise TypeError("normalizer must be an ImageGICOContextNormalizer.")
        table = _canonical_table(
            self.normalized_context_table,
            field_name="normalized backbone context table",
        )
        if (
            self.normalizer.mean_sha256 != self.binding.normalizer_mean_sha256
            or self.normalizer.scale_sha256 != self.binding.normalizer_scale_sha256
            or self.normalizer.normalizer_sha256 != self.binding.normalizer_sha256
            or _array_identity(
                table,
                namespace="image-gico-normalized-backbone-context-table-v3",
            )
            != self.binding.normalized_context_table_sha256
        ):
            raise ValueError("Prepared backbone context does not match its binding.")
        table.setflags(write=False)
        object.__setattr__(self, "normalized_context_table", table)


def prepare_image_gico_backbone_context(
    backbone: CanonicalNoiseToDataAdapter,
) -> PreparedImageGICOBackboneContext:
    if not isinstance(backbone, CanonicalNoiseToDataAdapter):
        raise TypeError("backbone must be a CanonicalNoiseToDataAdapter.")
    if backbone.training or any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("Backbone context extraction requires a frozen eval backbone.")
    raw = _canonical_table(
        backbone.canonical_conditioning_table(),
        field_name="canonical backbone context table",
    )
    mean64 = raw.astype(np.float64).mean(axis=0)
    std64 = raw.astype(np.float64).std(axis=0, ddof=0)
    mean = np.ascontiguousarray(mean64.astype("<f4"))
    scale = np.ascontiguousarray(np.maximum(std64, IMAGE_GICO_BACKBONE_CONTEXT_STD_FLOOR).astype("<f4"))
    normalizer = ImageGICOContextNormalizer(mean=mean, scale=scale)
    normalized = normalizer.normalize(raw)
    spec = get_image_backbone_spec(backbone.manifest.model_key)
    binding = ImageGICOBackboneContextBinding(
        backbone_model_key=backbone.manifest.model_key,
        backbone_protocol_sha256=backbone.manifest.protocol_sha256,
        backbone_checkpoint_sha256=backbone.manifest.checkpoint.sha256,
        source_revision=spec.source_revision,
        source_config_identity=spec.source_config_identity,
        raw_context_table_sha256=_array_identity(
            raw,
            namespace="image-gico-raw-backbone-context-table-v3",
        ),
        normalized_context_table_sha256=_array_identity(
            normalized,
            namespace="image-gico-normalized-backbone-context-table-v3",
        ),
        normalizer_mean_sha256=normalizer.mean_sha256,
        normalizer_scale_sha256=normalizer.scale_sha256,
        normalizer_sha256=normalizer.normalizer_sha256,
    )
    return PreparedImageGICOBackboneContext(
        binding=binding,
        normalizer=normalizer,
        normalized_context_table=normalized,
    )


def bind_image_gico_backbone_context(
    binding: ImageGICOBackboneContextBinding,
    normalizer: ImageGICOContextNormalizer,
    backbone: CanonicalNoiseToDataAdapter,
) -> PreparedImageGICOBackboneContext:
    prepared = prepare_image_gico_backbone_context(backbone)
    if prepared.binding != binding:
        raise ValueError("Loaded backbone does not match the portable context binding.")
    if (
        prepared.normalizer.mean_sha256 != normalizer.mean_sha256
        or prepared.normalizer.scale_sha256 != normalizer.scale_sha256
        or prepared.normalizer.normalizer_sha256 != normalizer.normalizer_sha256
        or not np.array_equal(prepared.normalizer.mean, normalizer.mean)
        or not np.array_equal(prepared.normalizer.scale, normalizer.scale)
    ):
        raise ValueError("Portable context normalizer does not match the loaded backbone.")
    return PreparedImageGICOBackboneContext(
        binding=binding,
        normalizer=normalizer,
        normalized_context_table=prepared.normalized_context_table,
    )


__all__ = [
    "IMAGE_GICO_BACKBONE_CONTEXT_CLASS_COUNT",
    "IMAGE_GICO_BACKBONE_CONTEXT_DTYPE",
    "IMAGE_GICO_BACKBONE_CONTEXT_PROTOCOL",
    "IMAGE_GICO_BACKBONE_CONTEXT_STD_FLOOR",
    "ImageGICOBackboneContextBinding",
    "ImageGICOContextNormalizer",
    "PreparedImageGICOBackboneContext",
    "bind_image_gico_backbone_context",
    "prepare_image_gico_backbone_context",
]
