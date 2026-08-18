from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from genode.artifacts.identity import canonical_json_bytes, canonical_json_text, semantic_sha256

from ._strict_json import loads_strict_json
from .checkpoint import CheckpointBinding, bind_checkpoint
from .registry import get_image_backbone_spec, get_image_dataset_spec

IMAGE_BACKBONE_PROTOCOL_SCHEMA = "genode_image_backbone_protocol_v1"
IMAGE_EVALUATION_NFES = (2, 4, 8)
IMAGE_EVALUATION_SOLVER = "euler"
DEFAULT_NATIVE_TIME_EPSILON = 1e-5
_PROTOCOL_IDENTITY_NAMESPACE = "genode-image-backbone-protocol-v1"
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "backbone",
    "dataset",
    "checkpoint",
    "time_adapter",
    "evaluation",
    "protocol_sha256",
}


def _protocol_sha256(payload: Mapping[str, object]) -> str:
    identity = semantic_sha256(payload, namespace=_PROTOCOL_IDENTITY_NAMESPACE)
    return identity.removeprefix(f"{_PROTOCOL_IDENTITY_NAMESPACE}:")


def _same_canonical_json(left: object, right: object) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


@dataclass(frozen=True, slots=True)
class ImageBackboneManifest:
    model_key: str
    checkpoint: CheckpointBinding
    native_time_epsilon: float = DEFAULT_NATIVE_TIME_EPSILON

    def __post_init__(self) -> None:
        spec = get_image_backbone_spec(self.model_key)
        if self.checkpoint.filename != spec.checkpoint_filename:
            raise ValueError(
                f"Manifest checkpoint filename must be {spec.checkpoint_filename!r} for model {self.model_key!r}."
            )
        epsilon = self.native_time_epsilon
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise ValueError("native_time_epsilon must be a finite real number.")
        normalized = float(epsilon)
        if not math.isfinite(normalized) or not 0.0 < normalized < 0.5:
            raise ValueError("native_time_epsilon must be finite and strictly between 0 and 0.5.")
        object.__setattr__(self, "native_time_epsilon", normalized)

    def identity_payload(self) -> dict[str, object]:
        spec = get_image_backbone_spec(self.model_key)
        dataset = get_image_dataset_spec(spec.dataset_key)
        return {
            "schema_version": IMAGE_BACKBONE_PROTOCOL_SCHEMA,
            "backbone": spec.to_manifest_dict(),
            "dataset": dataset.to_manifest_dict(),
            "checkpoint": self.checkpoint.to_manifest_dict(),
            "time_adapter": {
                "canonical_coordinate": "u_noise_to_data",
                "canonical_state_interval": [0.0, 1.0],
                "field_evaluation_interval": "0 <= u < 1",
                "native_coordinate": "t_data_to_noise",
                "native_time_expression": "clamp(1-u,epsilon,1-epsilon)",
                "native_time_epsilon": self.native_time_epsilon,
                "canonical_velocity_expression": "-native_velocity",
            },
            "evaluation": {
                "solver": IMAGE_EVALUATION_SOLVER,
                "target_nfes": list(IMAGE_EVALUATION_NFES),
                "uses_exact_left_endpoint_evaluations": True,
                "evaluates_final_endpoint": False,
            },
        }

    @property
    def protocol_sha256(self) -> str:
        return _protocol_sha256(self.identity_payload())

    def to_manifest_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        return {**payload, "protocol_sha256": self.protocol_sha256}

    def to_json(self) -> str:
        return canonical_json_text(self.to_manifest_dict())

    @classmethod
    def from_manifest_dict(cls, value: object) -> ImageBackboneManifest:
        if not isinstance(value, Mapping):
            raise ValueError("Image backbone manifest must be a JSON object.")
        if set(value) != _TOP_LEVEL_FIELDS:
            raise ValueError(f"Image backbone manifest fields must be exactly {sorted(_TOP_LEVEL_FIELDS)}.")
        if value["schema_version"] != IMAGE_BACKBONE_PROTOCOL_SCHEMA:
            raise ValueError(f"Unsupported image backbone schema {value['schema_version']!r}.")

        backbone = value["backbone"]
        if not isinstance(backbone, Mapping) or not isinstance(backbone.get("key"), str):
            raise ValueError("Manifest backbone must contain a string key.")
        model_key = backbone["key"]
        spec = get_image_backbone_spec(model_key)
        if not _same_canonical_json(dict(backbone), spec.to_manifest_dict()):
            raise ValueError("Manifest backbone metadata does not match the pinned registry.")

        dataset = value["dataset"]
        if not isinstance(dataset, Mapping) or not _same_canonical_json(
            dict(dataset),
            spec.dataset.to_manifest_dict(),
        ):
            raise ValueError("Manifest dataset metadata does not match the pinned registry.")

        time_adapter = value["time_adapter"]
        if not isinstance(time_adapter, Mapping):
            raise ValueError("Manifest time_adapter must be a JSON object.")
        epsilon = time_adapter.get("native_time_epsilon")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise ValueError("Manifest native_time_epsilon must be a finite real number.")

        checkpoint = CheckpointBinding.from_manifest_dict(value["checkpoint"])
        manifest = cls(
            model_key=model_key,
            checkpoint=checkpoint,
            native_time_epsilon=float(epsilon),
        )
        payload = {key: item for key, item in value.items() if key != "protocol_sha256"}
        expected_hash = value["protocol_sha256"]
        if not isinstance(expected_hash, str) or expected_hash != _protocol_sha256(payload):
            raise ValueError("Manifest protocol_sha256 does not match its serialized payload.")
        if not _same_canonical_json(payload, manifest.identity_payload()):
            raise ValueError("Manifest protocol fields do not match the supported image protocol.")
        return manifest

    @classmethod
    def from_json(cls, text: str) -> ImageBackboneManifest:
        value = loads_strict_json(text, label="Image backbone manifest")
        return cls.from_manifest_dict(value)


def build_image_backbone_manifest(
    model_key: str,
    checkpoint_path: str | Path,
    *,
    native_time_epsilon: float = DEFAULT_NATIVE_TIME_EPSILON,
) -> ImageBackboneManifest:
    return ImageBackboneManifest(
        model_key=model_key,
        checkpoint=bind_checkpoint(model_key, checkpoint_path),
        native_time_epsilon=native_time_epsilon,
    )


__all__ = [
    "DEFAULT_NATIVE_TIME_EPSILON",
    "IMAGE_BACKBONE_PROTOCOL_SCHEMA",
    "IMAGE_EVALUATION_NFES",
    "IMAGE_EVALUATION_SOLVER",
    "ImageBackboneManifest",
    "build_image_backbone_manifest",
]
