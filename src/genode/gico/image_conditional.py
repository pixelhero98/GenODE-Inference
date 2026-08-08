from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from genode.artifacts.identity import semantic_sha256
from genode.backbones.registry import get_image_backbone_spec
from genode.benchmarks.image.protocol import (
    IMAGE_SCHEDULE_KEYS,
    IMAGE_TARGET_NFES,
    normalize_image_nfe,
)
from genode.gico.image_model import DEFAULT_DENSITY_FLOOR
from genode.schedules import (
    ScheduleBatch,
    ScheduleSpecification,
    uniform_reference_time_grid,
    validate_fixed_schedule_keys,
)


IMAGE_GICO_FEATURE_GROUP_PROTOCOL = "image_gico_reward_feature_groups_v2"
IMAGE_GICO_CONDITIONAL_REWARD_PROTOCOL = "image_gico_conditional_reward_v2"
IMAGE_GICO_CONDITIONAL_TARGET_PROTOCOL = "image_gico_conditional_targets_v4"
IMAGE_GICO_BACKBONE_CONTEXT_MODEL_PROTOCOL = (
    "image_gico_backbone_context_nfe_residual_v3"
)
IMAGE_GICO_CONDITIONAL_POLICY_SPECIFICATION = ScheduleSpecification(
    "image_gico_backbone_context_v3"
)
IMAGE_GICO_CLASS_COUNT = 1_000
IMAGE_GICO_BACKBONE_CONTEXT_DIM = 768
IMAGE_GICO_FEATURE_DIM = 64
IMAGE_GICO_FEATURE_GROUP_COUNT = 32
IMAGE_GICO_REWARD_SAMPLES_PER_CLASS = 64
IMAGE_GICO_DEFAULT_SCHEDULE_COUNT = len(IMAGE_SCHEDULE_KEYS)
IMAGE_GICO_NFE_EMBEDDING_DIM = 16
IMAGE_GICO_CONDITIONAL_HIDDEN_DIM = 256
IMAGE_GICO_CONDITIONAL_REWARD_CLIP = 5.0
_RAW_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_FEATURE_PROTOCOL_NAMESPACE = "image-feature-protocol"


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a positive integer.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive.")
    return parsed


def _finite_positive(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite positive real.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field} must be a finite positive real.")
    return parsed


def _identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty identity string.")
    return value


def _raw_sha256_identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256 digest.")
    return value


def _feature_protocol_identity(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a namespaced lowercase SHA-256 identity."
        )
    namespace, separator, digest = value.rpartition(":")
    if (
        separator != ":"
        or namespace != _FEATURE_PROTOCOL_NAMESPACE
        or _RAW_SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError(
            f"{field} must use the {_FEATURE_PROTOCOL_NAMESPACE!r} namespace "
            "and one lowercase SHA-256 digest."
        )
    return value


def _validate_imagenet_target_binding(
    *,
    backbone_model_key: object,
    backbone_protocol_sha256: object,
    backbone_checkpoint_sha256: object,
    feature_protocol_sha256: object,
) -> None:
    model_key = _identity(backbone_model_key, field="backbone_model_key")
    backbone = get_image_backbone_spec(model_key)
    if (
        backbone.dataset_key != "imagenet64"
        or backbone.conditioning != "class_conditional"
        or backbone.num_conditioning_classes != IMAGE_GICO_CLASS_COUNT
    ):
        raise ValueError(
            "Conditional targets require a registered class-conditional "
            "ImageNet-64 backbone."
        )
    _raw_sha256_identity(
        backbone_protocol_sha256,
        field="backbone_protocol_sha256",
    )
    _raw_sha256_identity(
        backbone_checkpoint_sha256,
        field="backbone_checkpoint_sha256",
    )
    _feature_protocol_identity(
        feature_protocol_sha256,
        field="feature_protocol_sha256",
    )


def _finite_array(
    value: object,
    *,
    field: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{field} must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite values.")
    return array


def _sorted_cluster_order(centroids: np.ndarray) -> np.ndarray:
    return np.asarray(
        sorted(
            range(centroids.shape[0]),
            key=lambda index: tuple(float(value) for value in centroids[index]),
        ),
        dtype=np.int64,
    )


def _deterministic_feature_clusters(
    coordinates: np.ndarray,
    *,
    group_count: int,
    max_iterations: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster class coordinates without random initialization or tie ambiguity."""

    class_count = coordinates.shape[0]
    if group_count <= 0 or group_count > class_count:
        raise ValueError("group_count must be in [1, class_count].")
    squared_norm = np.sum(coordinates * coordinates, axis=1)
    center_indices = [int(np.argmax(squared_norm))]
    minimum_distance = np.sum(
        (coordinates - coordinates[center_indices[0]]) ** 2,
        axis=1,
    )
    for _ in range(1, group_count):
        next_index = int(np.argmax(minimum_distance))
        center_indices.append(next_index)
        distance = np.sum(
            (coordinates - coordinates[next_index]) ** 2,
            axis=1,
        )
        minimum_distance = np.minimum(minimum_distance, distance)
    centroids = coordinates[np.asarray(center_indices, dtype=np.int64)].copy()
    previous: np.ndarray | None = None
    for _ in range(max_iterations):
        distances = np.sum(
            (coordinates[:, None, :] - centroids[None, :, :]) ** 2,
            axis=2,
        )
        assignments = np.argmin(distances, axis=1).astype(np.int64)
        if previous is not None and np.array_equal(assignments, previous):
            break
        previous = assignments.copy()
        assigned_distance = distances[
            np.arange(class_count, dtype=np.int64),
            assignments,
        ]
        for group_index in range(group_count):
            members = np.flatnonzero(assignments == group_index)
            if members.size == 0:
                replacement = int(np.argmax(assigned_distance))
                assignments[replacement] = group_index
                assigned_distance[replacement] = -1.0
        centroids = np.stack(
            [
                coordinates[assignments == group_index].mean(axis=0)
                for group_index in range(group_count)
            ]
        )
    else:
        raise RuntimeError("Deterministic feature grouping did not converge.")
    order = _sorted_cluster_order(centroids)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(group_count, dtype=np.int64)
    assignments = inverse[assignments]
    centroids = centroids[order]
    return assignments, centroids


@dataclass(frozen=True)
class ImageGICOFeatureGroups:
    """Reward-only class grouping evidence; never an inference context."""

    class_coordinates: tuple[tuple[float, ...], ...]
    group_assignments: tuple[int, ...]
    group_centroids: tuple[tuple[float, ...], ...]
    pca_center: tuple[float, ...]
    pca_components: tuple[tuple[float, ...], ...]
    pca_scales: tuple[float, ...]
    source_panel_fingerprint: str
    feature_protocol_sha256: str
    real_feature_panel_sha256: str
    samples_per_class: int = IMAGE_GICO_REWARD_SAMPLES_PER_CLASS
    group_count: int = IMAGE_GICO_FEATURE_GROUP_COUNT

    def __post_init__(self) -> None:
        samples = _positive_integer(self.samples_per_class, field="samples_per_class")
        groups = _positive_integer(self.group_count, field="group_count")
        if samples != IMAGE_GICO_REWARD_SAMPLES_PER_CLASS:
            raise ValueError("ImageNet conditional rewards require 64 real samples per class.")
        if groups != IMAGE_GICO_FEATURE_GROUP_COUNT:
            raise ValueError("ImageNet conditional rewards require exactly 32 feature groups.")
        coordinates = _finite_array(
            self.class_coordinates,
            field="class_coordinates",
            shape=(IMAGE_GICO_CLASS_COUNT, IMAGE_GICO_FEATURE_DIM),
        )
        assignments = np.asarray(self.group_assignments)
        if assignments.shape != (IMAGE_GICO_CLASS_COUNT,) or assignments.dtype.kind not in {"i", "u"}:
            raise ValueError("group_assignments must contain 1,000 integer entries.")
        assignments = assignments.astype(np.int64)
        if (
            np.any(assignments < 0)
            or np.any(assignments >= groups)
            or not np.array_equal(np.unique(assignments), np.arange(groups))
        ):
            raise ValueError("group_assignments must cover every group exactly within [0, 31].")
        centroids = _finite_array(
            self.group_centroids,
            field="group_centroids",
            shape=(groups, IMAGE_GICO_FEATURE_DIM),
        )
        expected_centroids = np.stack(
            [coordinates[assignments == group].mean(axis=0) for group in range(groups)]
        )
        if not np.allclose(centroids, expected_centroids, rtol=1e-10, atol=1e-10):
            raise ValueError("group_centroids do not match the assigned class coordinates.")
        center = _finite_array(self.pca_center, field="pca_center")
        if center.ndim != 1 or center.size < IMAGE_GICO_FEATURE_DIM:
            raise ValueError("pca_center must contain at least 64 feature entries.")
        components = _finite_array(
            self.pca_components,
            field="pca_components",
            shape=(IMAGE_GICO_FEATURE_DIM, int(center.size)),
        )
        scales = _finite_array(
            self.pca_scales,
            field="pca_scales",
            shape=(IMAGE_GICO_FEATURE_DIM,),
        )
        if np.any(scales <= 0.0):
            raise ValueError("pca_scales must be strictly positive.")
        if not np.allclose(
            components @ components.T,
            np.eye(IMAGE_GICO_FEATURE_DIM),
            rtol=1e-7,
            atol=1e-7,
        ):
            raise ValueError("pca_components must have orthonormal rows.")
        for field in (
            "source_panel_fingerprint",
            "feature_protocol_sha256",
            "real_feature_panel_sha256",
        ):
            _identity(getattr(self, field), field=field)
        object.__setattr__(self, "samples_per_class", samples)
        object.__setattr__(self, "group_count", groups)
        object.__setattr__(
            self,
            "class_coordinates",
            tuple(tuple(float(value) for value in row) for row in coordinates),
        )
        object.__setattr__(
            self,
            "group_assignments",
            tuple(int(value) for value in assignments),
        )
        object.__setattr__(
            self,
            "group_centroids",
            tuple(tuple(float(value) for value in row) for row in centroids),
        )
        object.__setattr__(self, "pca_center", tuple(float(value) for value in center))
        object.__setattr__(
            self,
            "pca_components",
            tuple(tuple(float(value) for value in row) for row in components),
        )
        object.__setattr__(self, "pca_scales", tuple(float(value) for value in scales))

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            self.identity_payload(),
            namespace="image-gico-reward-feature-groups",
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "protocol": IMAGE_GICO_FEATURE_GROUP_PROTOCOL,
            "usage": "reward_shrinkage_only_not_inference_context",
            "class_count": IMAGE_GICO_CLASS_COUNT,
            "feature_dim": IMAGE_GICO_FEATURE_DIM,
            "group_count": self.group_count,
            "samples_per_class": self.samples_per_class,
            "source_panel_fingerprint": self.source_panel_fingerprint,
            "feature_protocol_sha256": self.feature_protocol_sha256,
            "real_feature_panel_sha256": self.real_feature_panel_sha256,
            "pca_center": list(self.pca_center),
            "pca_components": [list(row) for row in self.pca_components],
            "pca_scales": list(self.pca_scales),
            "class_coordinates": [list(row) for row in self.class_coordinates],
            "group_assignments": list(self.group_assignments),
            "group_centroids": [list(row) for row in self.group_centroids],
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "artifact": "image_gico_reward_feature_groups",
            **self.identity_payload(),
            "feature_group_sha256": self.sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ImageGICOFeatureGroups:
        if not isinstance(payload, Mapping):
            raise TypeError("Feature-group payload must be a mapping.")
        expected = {
            "artifact",
            "protocol",
            "usage",
            "class_count",
            "feature_dim",
            "group_count",
            "samples_per_class",
            "source_panel_fingerprint",
            "feature_protocol_sha256",
            "real_feature_panel_sha256",
            "pca_center",
            "pca_components",
            "pca_scales",
            "class_coordinates",
            "group_assignments",
            "group_centroids",
            "feature_group_sha256",
        }
        if set(payload) != expected:
            raise ValueError(f"Feature-group fields must be exactly {sorted(expected)}.")
        if (
            payload["artifact"] != "image_gico_reward_feature_groups"
            or payload["protocol"] != IMAGE_GICO_FEATURE_GROUP_PROTOCOL
            or payload["usage"] != "reward_shrinkage_only_not_inference_context"
            or payload["class_count"] != IMAGE_GICO_CLASS_COUNT
            or payload["feature_dim"] != IMAGE_GICO_FEATURE_DIM
        ):
            raise ValueError("Unsupported feature-group artifact.")
        artifact = cls(
            class_coordinates=tuple(tuple(row) for row in payload["class_coordinates"]),
            group_assignments=tuple(payload["group_assignments"]),
            group_centroids=tuple(tuple(row) for row in payload["group_centroids"]),
            pca_center=tuple(payload["pca_center"]),
            pca_components=tuple(tuple(row) for row in payload["pca_components"]),
            pca_scales=tuple(payload["pca_scales"]),
            source_panel_fingerprint=payload["source_panel_fingerprint"],
            feature_protocol_sha256=payload["feature_protocol_sha256"],
            real_feature_panel_sha256=payload["real_feature_panel_sha256"],
            samples_per_class=payload["samples_per_class"],
            group_count=payload["group_count"],
        )
        if payload["feature_group_sha256"] != artifact.sha256:
            raise ValueError("Feature-group content hash is inconsistent.")
        return artifact


def build_image_gico_feature_groups(
    class_means: np.ndarray | Tensor,
    *,
    source_panel_fingerprint: str,
    feature_protocol_sha256: str,
    real_feature_panel_sha256: str,
) -> ImageGICOFeatureGroups:
    """Build deterministic PCA-64 reward groups from 1,000 class means."""

    means = _finite_array(class_means, field="class_means")
    if means.ndim != 2 or means.shape[0] != IMAGE_GICO_CLASS_COUNT:
        raise ValueError("class_means must have shape [1000, feature_dim].")
    if means.shape[1] < IMAGE_GICO_FEATURE_DIM:
        raise ValueError("class_means must contain at least 64 feature dimensions.")
    center = means.mean(axis=0)
    centered = means - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:IMAGE_GICO_FEATURE_DIM].copy()
    singular_values = singular_values[:IMAGE_GICO_FEATURE_DIM].copy()
    if np.any(singular_values <= 0.0):
        raise ValueError("Class means do not support a 64-dimensional PCA basis.")
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    scales = singular_values / math.sqrt(float(IMAGE_GICO_CLASS_COUNT - 1))
    coordinates = (centered @ components.T) / scales[None, :]
    assignments, centroids = _deterministic_feature_clusters(
        coordinates,
        group_count=IMAGE_GICO_FEATURE_GROUP_COUNT,
    )
    return ImageGICOFeatureGroups(
        class_coordinates=tuple(tuple(float(value) for value in row) for row in coordinates),
        group_assignments=tuple(int(value) for value in assignments),
        group_centroids=tuple(tuple(float(value) for value in row) for row in centroids),
        pca_center=tuple(float(value) for value in center),
        pca_components=tuple(tuple(float(value) for value in row) for row in components),
        pca_scales=tuple(float(value) for value in scales),
        source_panel_fingerprint=source_panel_fingerprint,
        feature_protocol_sha256=feature_protocol_sha256,
        real_feature_panel_sha256=real_feature_panel_sha256,
    )


@dataclass(frozen=True)
class ImageGICOConditionalTargets:
    density_mass: tuple[tuple[tuple[float, ...], ...], ...]
    mixture_weights: tuple[tuple[tuple[float, ...], ...], ...]
    normalized_rewards: tuple[tuple[tuple[float, ...], ...], ...]
    jackknife_standard_errors: tuple[tuple[tuple[float, ...], ...], ...]
    class_reliability: tuple[tuple[tuple[float, ...], ...], ...]
    group_reliability: tuple[tuple[tuple[float, ...], ...], ...]
    shrinkage_coefficients: tuple[
        tuple[tuple[tuple[float, ...], ...], ...],
        ...,
    ]
    temperature_by_nfe: tuple[float, ...]
    target_nfes: tuple[int, ...]
    schedule_keys: tuple[str, ...]
    schedule_sha256s: tuple[str, ...]
    density_mass_sha256s: tuple[tuple[str, ...], ...]
    feature_group_sha256: str
    reward_evidence_sha256: str
    fixed_support_sha256: str
    backbone_model_key: str
    backbone_protocol_sha256: str
    backbone_checkpoint_sha256: str
    feature_protocol_sha256: str
    density_bin_count: int

    def __post_init__(self) -> None:
        if tuple(self.target_nfes) != tuple(IMAGE_TARGET_NFES):
            raise ValueError(f"target_nfes must be exactly {IMAGE_TARGET_NFES}.")
        bins = _positive_integer(self.density_bin_count, field="density_bin_count")
        keys = validate_fixed_schedule_keys(self.schedule_keys)
        schedule_count = len(keys)
        mass = _finite_array(
            self.density_mass,
            field="density_mass",
            shape=(3, IMAGE_GICO_CLASS_COUNT, bins),
        )
        weights = _finite_array(
            self.mixture_weights,
            field="mixture_weights",
            shape=(3, IMAGE_GICO_CLASS_COUNT, schedule_count),
        )
        _finite_array(
            self.normalized_rewards,
            field="normalized_rewards",
            shape=weights.shape,
        )
        errors = _finite_array(
            self.jackknife_standard_errors,
            field="jackknife_standard_errors",
            shape=weights.shape,
        )
        class_reliability = _finite_array(
            self.class_reliability,
            field="class_reliability",
            shape=weights.shape,
        )
        group_reliability = _finite_array(
            self.group_reliability,
            field="group_reliability",
            shape=weights.shape,
        )
        coefficients = _finite_array(
            self.shrinkage_coefficients,
            field="shrinkage_coefficients",
            shape=(*weights.shape, 3),
        )
        if np.any(mass < 0.0) or not np.allclose(
            mass.sum(axis=-1),
            1.0,
            rtol=1e-7,
            atol=1e-7,
        ):
            raise ValueError("Every conditional density target must be normalized.")
        if np.any(weights < 0.0) or not np.allclose(
            weights.sum(axis=-1),
            1.0,
            rtol=1e-7,
            atol=1e-7,
        ):
            raise ValueError("Every schedule mixture must be normalized.")
        for field, values in (
            ("jackknife_standard_errors", errors),
            ("class_reliability", class_reliability),
            ("group_reliability", group_reliability),
            ("shrinkage_coefficients", coefficients),
        ):
            if np.any(values < 0.0):
                raise ValueError(f"{field} must be nonnegative.")
        if (
            np.any(class_reliability > 1.0)
            or np.any(group_reliability > 1.0)
            or np.any(coefficients > 1.0)
        ):
            raise ValueError("Shrinkage reliabilities and coefficients must be in [0, 1].")
        if not np.allclose(coefficients.sum(axis=-1), 1.0, rtol=1e-8, atol=1e-8):
            raise ValueError("Class/group/global shrinkage coefficients must sum to one.")
        temperatures = tuple(
            _finite_positive(value, field="temperature_by_nfe")
            for value in self.temperature_by_nfe
        )
        if len(temperatures) != 3:
            raise ValueError("temperature_by_nfe must contain exactly three values.")
        schedule_hashes = tuple(
            _identity(value, field="schedule_sha256s")
            for value in self.schedule_sha256s
        )
        if len(schedule_hashes) != schedule_count:
            raise ValueError(
                f"schedule_sha256s must contain {schedule_count} entries."
            )
        density_hashes = tuple(
            tuple(_identity(value, field="density_mass_sha256s") for value in row)
            for row in self.density_mass_sha256s
        )
        if len(density_hashes) != 3 or any(
            len(row) != schedule_count for row in density_hashes
        ):
            raise ValueError(
                "density_mass_sha256s must have shape "
                f"[3, {schedule_count}]."
            )
        for field in (
            "feature_group_sha256",
            "reward_evidence_sha256",
            "fixed_support_sha256",
        ):
            _identity(getattr(self, field), field=field)
        _validate_imagenet_target_binding(
            backbone_model_key=self.backbone_model_key,
            backbone_protocol_sha256=self.backbone_protocol_sha256,
            backbone_checkpoint_sha256=self.backbone_checkpoint_sha256,
            feature_protocol_sha256=self.feature_protocol_sha256,
        )
        object.__setattr__(self, "density_bin_count", bins)
        object.__setattr__(self, "temperature_by_nfe", temperatures)
        object.__setattr__(self, "schedule_keys", keys)
        object.__setattr__(self, "schedule_sha256s", schedule_hashes)
        object.__setattr__(self, "density_mass_sha256s", density_hashes)

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            self.identity_payload(),
            namespace="image-gico-conditional-targets-v4",
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "protocol": IMAGE_GICO_CONDITIONAL_TARGET_PROTOCOL,
            "conditioning": "classwise_rewards_independent_of_inference_context",
            "feature_group_usage": "reward_shrinkage_only_not_inference_context",
            "target_nfes": list(self.target_nfes),
            "class_count": IMAGE_GICO_CLASS_COUNT,
            "schedule_count": len(self.schedule_keys),
            "density_bin_count": self.density_bin_count,
            "schedule_keys": list(self.schedule_keys),
            "schedule_sha256s": list(self.schedule_sha256s),
            "density_mass_sha256s": [list(row) for row in self.density_mass_sha256s],
            "temperature_by_nfe": list(self.temperature_by_nfe),
            "feature_group_sha256": self.feature_group_sha256,
            "reward_evidence_sha256": self.reward_evidence_sha256,
            "fixed_support_sha256": self.fixed_support_sha256,
            "backbone_model_key": self.backbone_model_key,
            "backbone_protocol_sha256": self.backbone_protocol_sha256,
            "backbone_checkpoint_sha256": self.backbone_checkpoint_sha256,
            "feature_protocol_sha256": self.feature_protocol_sha256,
            "density_mass": self.density_mass,
            "mixture_weights": self.mixture_weights,
            "normalized_rewards": self.normalized_rewards,
            "jackknife_standard_errors": self.jackknife_standard_errors,
            "class_reliability": self.class_reliability,
            "group_reliability": self.group_reliability,
            "shrinkage_coefficients": self.shrinkage_coefficients,
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "artifact": "image_gico_conditional_targets",
            **self.identity_payload(),
            "target_sha256": self.sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ImageGICOConditionalTargets:
        if not isinstance(payload, Mapping):
            raise TypeError("Conditional target payload must be a mapping.")
        expected = {
            "artifact",
            "protocol",
            "conditioning",
            "feature_group_usage",
            "target_nfes",
            "class_count",
            "schedule_count",
            "density_bin_count",
            "schedule_keys",
            "schedule_sha256s",
            "density_mass_sha256s",
            "temperature_by_nfe",
            "feature_group_sha256",
            "reward_evidence_sha256",
            "fixed_support_sha256",
            "backbone_model_key",
            "backbone_protocol_sha256",
            "backbone_checkpoint_sha256",
            "feature_protocol_sha256",
            "density_mass",
            "mixture_weights",
            "normalized_rewards",
            "jackknife_standard_errors",
            "class_reliability",
            "group_reliability",
            "shrinkage_coefficients",
            "target_sha256",
        }
        if set(payload) != expected:
            raise ValueError(f"Conditional target fields must be exactly {sorted(expected)}.")
        raw_schedule_keys = payload["schedule_keys"]
        if not isinstance(raw_schedule_keys, (list, tuple)):
            raise ValueError("Conditional target schedule_keys must be a sequence.")
        if (
            payload["artifact"] != "image_gico_conditional_targets"
            or payload["protocol"] != IMAGE_GICO_CONDITIONAL_TARGET_PROTOCOL
            or payload["conditioning"]
            != "classwise_rewards_independent_of_inference_context"
            or payload["feature_group_usage"] != "reward_shrinkage_only_not_inference_context"
            or payload["class_count"] != IMAGE_GICO_CLASS_COUNT
            or isinstance(payload["schedule_count"], bool)
            or not isinstance(payload["schedule_count"], Integral)
            or int(payload["schedule_count"]) != len(raw_schedule_keys)
        ):
            raise ValueError("Unsupported conditional target artifact.")
        target = cls(
            density_mass=tuple(
                tuple(tuple(row) for row in nfe_rows)
                for nfe_rows in payload["density_mass"]
            ),
            mixture_weights=tuple(
                tuple(tuple(row) for row in nfe_rows)
                for nfe_rows in payload["mixture_weights"]
            ),
            normalized_rewards=tuple(
                tuple(tuple(row) for row in nfe_rows)
                for nfe_rows in payload["normalized_rewards"]
            ),
            jackknife_standard_errors=tuple(
                tuple(tuple(row) for row in nfe_rows)
                for nfe_rows in payload["jackknife_standard_errors"]
            ),
            class_reliability=tuple(
                tuple(tuple(row) for row in nfe_rows)
                for nfe_rows in payload["class_reliability"]
            ),
            group_reliability=tuple(
                tuple(tuple(row) for row in nfe_rows)
                for nfe_rows in payload["group_reliability"]
            ),
            shrinkage_coefficients=tuple(
                tuple(
                    tuple(tuple(coefficients) for coefficients in class_rows)
                    for class_rows in nfe_rows
                )
                for nfe_rows in payload["shrinkage_coefficients"]
            ),
            temperature_by_nfe=tuple(payload["temperature_by_nfe"]),
            target_nfes=tuple(payload["target_nfes"]),
            schedule_keys=tuple(raw_schedule_keys),
            schedule_sha256s=tuple(payload["schedule_sha256s"]),
            density_mass_sha256s=tuple(
                tuple(row) for row in payload["density_mass_sha256s"]
            ),
            feature_group_sha256=payload["feature_group_sha256"],
            reward_evidence_sha256=payload["reward_evidence_sha256"],
            fixed_support_sha256=payload["fixed_support_sha256"],
            backbone_model_key=payload["backbone_model_key"],
            backbone_protocol_sha256=payload["backbone_protocol_sha256"],
            backbone_checkpoint_sha256=payload["backbone_checkpoint_sha256"],
            feature_protocol_sha256=payload["feature_protocol_sha256"],
            density_bin_count=payload["density_bin_count"],
        )
        if payload["target_sha256"] != target.sha256:
            raise ValueError("Conditional target content hash is inconsistent.")
        return target

    def density_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        return torch.tensor(self.density_mass, dtype=dtype, device=device)


def _jackknife_standard_error(jackknife_advantage: np.ndarray) -> np.ndarray:
    count = jackknife_advantage.shape[-1]
    center = jackknife_advantage.mean(axis=-1, keepdims=True)
    variance = (
        float(count - 1)
        / float(count)
        * np.sum((jackknife_advantage - center) ** 2, axis=-1)
    )
    return np.sqrt(np.maximum(variance, 0.0))


def _hierarchical_shrinkage(
    advantages: np.ndarray,
    standard_errors: np.ndarray,
    assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shrunk = np.empty_like(advantages)
    class_reliability = np.empty_like(advantages)
    group_reliability = np.empty_like(advantages)
    coefficients = np.empty((*advantages.shape, 3), dtype=np.float64)
    noise = standard_errors**2
    for nfe_index in range(advantages.shape[0]):
        for schedule_index in range(advantages.shape[1]):
            values = advantages[nfe_index, schedule_index]
            variances = noise[nfe_index, schedule_index]
            global_mean = float(values.mean())
            group_means = np.empty(IMAGE_GICO_FEATURE_GROUP_COUNT, dtype=np.float64)
            group_noise = np.empty_like(group_means)
            group_sizes = np.empty_like(group_means)
            for group in range(IMAGE_GICO_FEATURE_GROUP_COUNT):
                members = assignments == group
                size = int(np.sum(members))
                group_sizes[group] = float(size)
                group_means[group] = float(values[members].mean())
                group_noise[group] = float(np.sum(variances[members]) / float(size * size))
            residual = values - group_means[assignments]
            within_signal = max(
                float(np.mean(residual**2) - np.mean(variances)),
                0.0,
            )
            between_observed = float(
                np.sum(group_sizes * (group_means - global_mean) ** 2)
                / float(IMAGE_GICO_CLASS_COUNT)
            )
            between_noise = float(
                np.sum(group_sizes * group_noise)
                / float(IMAGE_GICO_CLASS_COUNT)
            )
            between_signal = max(between_observed - between_noise, 0.0)
            class_denominator = within_signal + variances
            class_weight = np.divide(
                within_signal,
                class_denominator,
                out=np.zeros_like(variances),
                where=class_denominator > 0.0,
            )
            group_denominator = between_signal + group_noise
            group_weight_by_group = np.divide(
                between_signal,
                group_denominator,
                out=np.zeros_like(group_noise),
                where=group_denominator > 0.0,
            )
            group_weight = group_weight_by_group[assignments]
            pooled = (
                group_weight * group_means[assignments]
                + (1.0 - group_weight) * global_mean
            )
            shrunk[nfe_index, schedule_index] = (
                class_weight * values + (1.0 - class_weight) * pooled
            )
            class_reliability[nfe_index, schedule_index] = class_weight
            group_reliability[nfe_index, schedule_index] = group_weight
            coefficients[nfe_index, schedule_index, :, 0] = class_weight
            coefficients[nfe_index, schedule_index, :, 1] = (
                (1.0 - class_weight) * group_weight
            )
            coefficients[nfe_index, schedule_index, :, 2] = (
                (1.0 - class_weight) * (1.0 - group_weight)
            )
    return shrunk, class_reliability, group_reliability, coefficients


def build_image_gico_conditional_targets(
    *,
    class_kid: np.ndarray,
    jackknife_class_kid: np.ndarray,
    reward_scales: np.ndarray,
    fixed_density_mass: np.ndarray,
    schedule_keys: Sequence[str],
    schedule_sha256s: Sequence[str],
    density_mass_sha256s: Sequence[Sequence[str]],
    feature_groups: ImageGICOFeatureGroups,
    reward_evidence_sha256: str,
    fixed_support_sha256: str,
    backbone_model_key: str,
    backbone_protocol_sha256: str,
    backbone_checkpoint_sha256: str,
    feature_protocol_sha256: str,
) -> ImageGICOConditionalTargets:
    keys = validate_fixed_schedule_keys(schedule_keys)
    schedule_count = len(keys)
    kids = _finite_array(
        class_kid,
        field="class_kid",
        shape=(3, schedule_count, IMAGE_GICO_CLASS_COUNT),
    )
    jackknife = _finite_array(
        jackknife_class_kid,
        field="jackknife_class_kid",
        shape=(
            3,
            schedule_count,
            IMAGE_GICO_CLASS_COUNT,
            IMAGE_GICO_REWARD_SAMPLES_PER_CLASS,
        ),
    )
    scales = _finite_array(reward_scales, field="reward_scales", shape=(3,))
    if np.any(scales <= 0.0):
        raise ValueError("reward_scales must be strictly positive.")
    masses = _finite_array(fixed_density_mass, field="fixed_density_mass")
    if masses.ndim != 3 or masses.shape[:2] != (3, schedule_count):
        raise ValueError(
            f"fixed_density_mass must have shape [3, {schedule_count}, bins]."
        )
    if np.any(masses < 0.0) or not np.allclose(
        masses.sum(axis=-1),
        1.0,
        rtol=1e-7,
        atol=1e-7,
    ):
        raise ValueError("Every fixed density mass row must be normalized.")
    if not isinstance(feature_groups, ImageGICOFeatureGroups):
        raise TypeError("feature_groups must be ImageGICOFeatureGroups.")
    _validate_imagenet_target_binding(
        backbone_model_key=backbone_model_key,
        backbone_protocol_sha256=backbone_protocol_sha256,
        backbone_checkpoint_sha256=backbone_checkpoint_sha256,
        feature_protocol_sha256=feature_protocol_sha256,
    )
    if feature_groups.feature_protocol_sha256 != feature_protocol_sha256:
        raise ValueError(
            "Conditional targets and reward feature groups must use the same "
            "feature protocol identity."
        )
    uniform = keys.index("uniform")
    advantages = kids[:, uniform : uniform + 1, :] - kids
    jackknife_advantage = (
        jackknife[:, uniform : uniform + 1, :, :] - jackknife
    )
    standard_errors = _jackknife_standard_error(jackknife_advantage)
    assignments = np.asarray(feature_groups.group_assignments, dtype=np.int64)
    shrunk, class_reliability, group_reliability, coefficients = (
        _hierarchical_shrinkage(advantages, standard_errors, assignments)
    )
    normalized = np.clip(
        shrunk / scales[:, None, None],
        -IMAGE_GICO_CONDITIONAL_REWARD_CLIP,
        IMAGE_GICO_CONDITIONAL_REWARD_CLIP,
    )
    normalized[:, uniform, :] = 0.0
    standard_errors[:, uniform, :] = 0.0
    class_reliability[:, uniform, :] = 0.0
    group_reliability[:, uniform, :] = 0.0
    coefficients[:, uniform, :, :] = (0.0, 0.0, 1.0)
    density_hashes = tuple(tuple(str(value) for value in row) for row in density_mass_sha256s)
    if len(density_hashes) != 3 or any(
        len(row) != schedule_count for row in density_hashes
    ):
        raise ValueError(
            "density_mass_sha256s must have shape "
            f"[3, {schedule_count}]."
        )
    schedule_weights = np.empty(
        (3, IMAGE_GICO_CLASS_COUNT, schedule_count),
        dtype=np.float64,
    )
    for nfe_index in range(3):
        groups_by_hash: dict[str, list[int]] = {}
        for schedule_index, density_hash in enumerate(density_hashes[nfe_index]):
            groups_by_hash.setdefault(density_hash, []).append(schedule_index)
        unique_groups = tuple(groups_by_hash.values())
        for class_index in range(IMAGE_GICO_CLASS_COUNT):
            group_logits = np.asarray(
                [
                    float(np.mean(normalized[nfe_index, group, class_index]))
                    for group in unique_groups
                ],
                dtype=np.float64,
            )
            group_logits -= float(np.max(group_logits))
            group_probabilities = np.exp(group_logits)
            group_probabilities /= float(np.sum(group_probabilities))
            row = np.zeros(schedule_count, dtype=np.float64)
            for probability, group in zip(group_probabilities, unique_groups, strict=True):
                row[group] = probability / float(len(group))
            schedule_weights[nfe_index, class_index] = row
    density = np.einsum("ncs,nsb->ncb", schedule_weights, masses)
    density /= density.sum(axis=-1, keepdims=True)
    return ImageGICOConditionalTargets(
        density_mass=tuple(
            tuple(tuple(float(value) for value in row) for row in nfe_rows)
            for nfe_rows in density
        ),
        mixture_weights=tuple(
            tuple(tuple(float(value) for value in row) for row in nfe_rows)
            for nfe_rows in schedule_weights
        ),
        normalized_rewards=tuple(
            tuple(tuple(float(value) for value in row) for row in nfe_rows)
            for nfe_rows in normalized.transpose(0, 2, 1)
        ),
        jackknife_standard_errors=tuple(
            tuple(tuple(float(value) for value in row) for row in nfe_rows)
            for nfe_rows in standard_errors.transpose(0, 2, 1)
        ),
        class_reliability=tuple(
            tuple(tuple(float(value) for value in row) for row in nfe_rows)
            for nfe_rows in class_reliability.transpose(0, 2, 1)
        ),
        group_reliability=tuple(
            tuple(tuple(float(value) for value in row) for row in nfe_rows)
            for nfe_rows in group_reliability.transpose(0, 2, 1)
        ),
        shrinkage_coefficients=tuple(
            tuple(
                tuple(
                    tuple(float(value) for value in coefficient)
                    for coefficient in class_rows
                )
                for class_rows in nfe_rows
            )
            for nfe_rows in coefficients.transpose(0, 2, 1, 3)
        ),
        temperature_by_nfe=tuple(float(value) for value in scales),
        target_nfes=tuple(IMAGE_TARGET_NFES),
        schedule_keys=keys,
        schedule_sha256s=tuple(str(value) for value in schedule_sha256s),
        density_mass_sha256s=density_hashes,
        feature_group_sha256=feature_groups.sha256,
        reward_evidence_sha256=reward_evidence_sha256,
        fixed_support_sha256=fixed_support_sha256,
        backbone_model_key=backbone_model_key,
        backbone_protocol_sha256=backbone_protocol_sha256,
        backbone_checkpoint_sha256=backbone_checkpoint_sha256,
        feature_protocol_sha256=feature_protocol_sha256,
        density_bin_count=int(masses.shape[-1]),
    )


def validate_image_gico_backbone_context_tensor(
    value: np.ndarray | Tensor,
    *,
    field: str,
    expected_rows: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Validate a frozen, normalized backbone-context tensor."""

    if isinstance(value, np.ndarray):
        if value.dtype.kind != "f":
            raise TypeError(f"{field} must use a floating-point dtype.")
        tensor = torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
    elif isinstance(value, Tensor):
        if not value.is_floating_point():
            raise TypeError(f"{field} must use a floating-point dtype.")
        tensor = value
    else:
        raise TypeError(f"{field} must be a NumPy array or torch.Tensor.")
    if tensor.ndim != 2 or tensor.shape[1] != IMAGE_GICO_BACKBONE_CONTEXT_DIM:
        raise ValueError(
            f"{field} must have shape [batch, {IMAGE_GICO_BACKBONE_CONTEXT_DIM}]."
        )
    if tensor.shape[0] <= 0 or (
        expected_rows is not None and tensor.shape[0] != expected_rows
    ):
        expected = "a nonempty batch" if expected_rows is None else str(expected_rows)
        raise ValueError(f"{field} must contain {expected} rows.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{field} must contain only finite values.")
    target_device = tensor.device if device is None else torch.device(device)
    return tensor.detach().to(device=target_device, dtype=torch.float32).contiguous()


@dataclass(frozen=True)
class ImageGICOBackboneContextModelConfig:
    density_bin_count: int
    context_dim: int = IMAGE_GICO_BACKBONE_CONTEXT_DIM
    class_count: int = IMAGE_GICO_CLASS_COUNT
    nfe_embedding_dim: int = IMAGE_GICO_NFE_EMBEDDING_DIM
    hidden_dim: int = IMAGE_GICO_CONDITIONAL_HIDDEN_DIM
    density_floor: float = DEFAULT_DENSITY_FLOOR
    target_nfes: tuple[int, ...] = tuple(IMAGE_TARGET_NFES)

    def __post_init__(self) -> None:
        for field in (
            "density_bin_count",
            "context_dim",
            "class_count",
            "nfe_embedding_dim",
            "hidden_dim",
        ):
            object.__setattr__(
                self,
                field,
                _positive_integer(getattr(self, field), field=field),
            )
        if self.context_dim != IMAGE_GICO_BACKBONE_CONTEXT_DIM:
            raise ValueError("ImageNet GICO requires 768-dimensional backbone contexts.")
        if self.class_count != IMAGE_GICO_CLASS_COUNT:
            raise ValueError("Conditional ImageNet GICO requires 1,000 classes.")
        if tuple(self.target_nfes) != tuple(IMAGE_TARGET_NFES):
            raise ValueError(f"target_nfes must be exactly {IMAGE_TARGET_NFES}.")
        if isinstance(self.density_floor, bool) or not isinstance(self.density_floor, Real):
            raise TypeError("density_floor must be a finite nonnegative real.")
        floor = float(self.density_floor)
        if (
            not math.isfinite(floor)
            or floor < 0.0
            or floor * self.density_bin_count >= 1.0
        ):
            raise ValueError("density_floor is incompatible with density_bin_count.")
        object.__setattr__(self, "density_floor", floor)

    def as_payload(self) -> dict[str, Any]:
        return {
            "protocol": IMAGE_GICO_BACKBONE_CONTEXT_MODEL_PROTOCOL,
            "conditioning": "normalized_frozen_backbone_map_label_plus_target_nfe",
            "density_bin_count": self.density_bin_count,
            "context_dim": self.context_dim,
            "class_count": self.class_count,
            "nfe_embedding_dim": self.nfe_embedding_dim,
            "hidden_dim": self.hidden_dim,
            "density_floor": self.density_floor,
            "target_nfes": list(self.target_nfes),
            "global_base": "learned_per_nfe_logits",
            "residual_centering": "canonical_1000_context_table_per_nfe",
            "residual_initialization": "zero_output_layer",
        }

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            self.as_payload(),
            namespace="image-gico-backbone-context-model-config-v3",
        )


class ImageGICOBackboneContextDensityModel(nn.Module):
    """Global NFE logits plus a centered frozen-backbone-context residual."""

    def __init__(
        self,
        config: ImageGICOBackboneContextModelConfig,
        canonical_context_table: np.ndarray | Tensor,
    ) -> None:
        super().__init__()
        if not isinstance(config, ImageGICOBackboneContextModelConfig):
            raise TypeError("config must be an ImageGICOBackboneContextModelConfig.")
        self.config = config
        table = validate_image_gico_backbone_context_tensor(
            canonical_context_table,
            field="canonical_context_table",
            expected_rows=config.class_count,
        )
        self.register_buffer("_canonical_context_table", table, persistent=False)
        self.nfe_embedding = nn.Embedding(
            len(config.target_nfes),
            config.nfe_embedding_dim,
        )
        input_dim = config.context_dim + config.nfe_embedding_dim
        self.context_network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.density_bin_count),
        )
        self.global_logits_by_nfe = nn.Parameter(
            torch.zeros(len(config.target_nfes), config.density_bin_count)
        )
        final = self.context_network[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("Image GICO context network must end in a linear layer.")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def canonical_context_table(self) -> Tensor:
        return self._canonical_context_table

    def _validate_contexts(self, contexts: Tensor) -> Tensor:
        if not isinstance(contexts, Tensor):
            raise TypeError("contexts must be a torch.Tensor.")
        if contexts.dtype != torch.float32:
            raise TypeError("contexts must use torch.float32.")
        if contexts.device != self.global_logits_by_nfe.device:
            raise ValueError("contexts and the conditional model must share a device.")
        if contexts.ndim != 2 or contexts.shape[1] != self.config.context_dim:
            raise ValueError(
                f"contexts must have shape [batch, {self.config.context_dim}]."
            )
        if contexts.shape[0] <= 0:
            raise ValueError("contexts must contain at least one row.")
        if not bool(torch.isfinite(contexts).all()):
            raise ValueError("contexts must contain only finite values.")
        return contexts

    def _nfe_indices(self, target_nfes: Tensor) -> Tensor:
        if not isinstance(target_nfes, Tensor):
            raise TypeError("target_nfes must be a torch.Tensor.")
        if target_nfes.ndim != 1 or target_nfes.numel() <= 0:
            raise ValueError("target_nfes must have shape [batch].")
        if target_nfes.dtype == torch.bool or target_nfes.is_floating_point():
            raise TypeError("target_nfes must use an integer dtype.")
        if target_nfes.device != self.global_logits_by_nfe.device:
            raise ValueError("target_nfes and the conditional model must share a device.")
        supported = torch.tensor(
            self.config.target_nfes,
            dtype=target_nfes.dtype,
            device=target_nfes.device,
        )
        matches = target_nfes[:, None] == supported[None, :]
        if not bool(torch.all(matches.sum(dim=1) == 1)):
            raise ValueError(f"target_nfes must contain only {self.config.target_nfes}.")
        return torch.argmax(matches.to(dtype=torch.int64), dim=1)

    def _raw_residual(self, contexts: Tensor, nfe_indices: Tensor) -> Tensor:
        conditioning = torch.cat(
            (contexts, self.nfe_embedding(nfe_indices)),
            dim=-1,
        )
        return self.context_network(conditioning)

    def _residual_centers(self) -> Tensor:
        rows = []
        for nfe_index in range(len(self.config.target_nfes)):
            indices = torch.full(
                (self.config.class_count,),
                nfe_index,
                dtype=torch.int64,
                device=self.global_logits_by_nfe.device,
            )
            rows.append(
                self._raw_residual(self.canonical_context_table, indices).mean(dim=0)
            )
        return torch.stack(rows)

    def centered_residual_table(self) -> Tensor:
        rows = []
        for nfe_index in range(len(self.config.target_nfes)):
            indices = torch.full(
                (self.config.class_count,),
                nfe_index,
                dtype=torch.int64,
                device=self.global_logits_by_nfe.device,
            )
            residual = self._raw_residual(self.canonical_context_table, indices)
            rows.append(residual - residual.mean(dim=0, keepdim=True))
        return torch.stack(rows)

    def canonical_density_table(self) -> Tensor:
        """Return the centered `[NFE, class, bin]` table for canonical contexts."""

        residual = self.centered_residual_table()
        probabilities = torch.softmax(
            self.global_logits_by_nfe[:, None, :] + residual,
            dim=-1,
        )
        floor = self.config.density_floor
        if floor == 0.0:
            return probabilities
        free_mass = 1.0 - floor * float(self.config.density_bin_count)
        return floor + free_mass * probabilities

    def forward(self, contexts: Tensor, target_nfes: Tensor) -> Tensor:
        validated_contexts = self._validate_contexts(contexts)
        indices = self._nfe_indices(target_nfes)
        if validated_contexts.shape[0] != indices.shape[0]:
            raise ValueError("contexts and target_nfes must have the same batch size.")
        residual = (
            self._raw_residual(validated_contexts, indices)
            - self._residual_centers()[indices]
        )
        probabilities = torch.softmax(
            self.global_logits_by_nfe[indices] + residual,
            dim=-1,
        )
        floor = self.config.density_floor
        if floor == 0.0:
            return probabilities
        free_mass = 1.0 - floor * float(self.config.density_bin_count)
        return floor + free_mass * probabilities


class ImageGICOBackboneContextSchedulePolicy(nn.Module):
    """Schedule policy driven only by frozen normalized backbone contexts and NFE."""

    def __init__(
        self,
        model: ImageGICOBackboneContextDensityModel,
        targets: ImageGICOConditionalTargets,
    ) -> None:
        super().__init__()
        if not isinstance(model, ImageGICOBackboneContextDensityModel):
            raise TypeError("model must be an ImageGICOBackboneContextDensityModel.")
        if not isinstance(targets, ImageGICOConditionalTargets):
            raise TypeError("targets must be ImageGICOConditionalTargets.")
        if model.config.density_bin_count != targets.density_bin_count:
            raise ValueError("Model and target density bin counts disagree.")
        self.model = model
        self.targets = targets

    def predict(self, contexts: Tensor, *, target_nfe: int) -> ScheduleBatch:
        nfe = normalize_image_nfe(target_nfe)
        validated_contexts = self.model._validate_contexts(contexts)
        target_nfes = torch.full(
            (validated_contexts.shape[0],),
            nfe,
            dtype=torch.int64,
            device=validated_contexts.device,
        )
        with torch.no_grad():
            density = self.model(validated_contexts, target_nfes)
        reference = uniform_reference_time_grid(
            self.model.config.density_bin_count,
            dtype=density.dtype,
            device=density.device,
        )
        return ScheduleBatch.from_density_mass(
            density,
            target_nfe=nfe,
            reference_time_grid=reference,
            specification=IMAGE_GICO_CONDITIONAL_POLICY_SPECIFICATION,
        )

    def class_density_table(self, target_nfe: int) -> Tensor:
        nfe = normalize_image_nfe(target_nfe)
        nfe_index = self.model.config.target_nfes.index(nfe)
        with torch.no_grad():
            return self.model.canonical_density_table()[nfe_index]


__all__ = [
    "IMAGE_GICO_BACKBONE_CONTEXT_DIM",
    "IMAGE_GICO_BACKBONE_CONTEXT_MODEL_PROTOCOL",
    "IMAGE_GICO_CLASS_COUNT",
    "IMAGE_GICO_CONDITIONAL_POLICY_SPECIFICATION",
    "IMAGE_GICO_CONDITIONAL_REWARD_PROTOCOL",
    "IMAGE_GICO_CONDITIONAL_TARGET_PROTOCOL",
    "IMAGE_GICO_FEATURE_DIM",
    "IMAGE_GICO_FEATURE_GROUP_COUNT",
    "IMAGE_GICO_FEATURE_GROUP_PROTOCOL",
    "IMAGE_GICO_NFE_EMBEDDING_DIM",
    "IMAGE_GICO_REWARD_SAMPLES_PER_CLASS",
    "IMAGE_GICO_DEFAULT_SCHEDULE_COUNT",
    "ImageGICOBackboneContextDensityModel",
    "ImageGICOBackboneContextModelConfig",
    "ImageGICOBackboneContextSchedulePolicy",
    "ImageGICOConditionalTargets",
    "ImageGICOFeatureGroups",
    "build_image_gico_conditional_targets",
    "build_image_gico_feature_groups",
    "validate_image_gico_backbone_context_tensor",
]
