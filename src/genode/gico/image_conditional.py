"""Training-only feature groups and paired conditional KID shrinkage.

This module estimates reward evidence; all teacher and student architectures
live in the common GICO implementation.
"""

from __future__ import annotations

import numpy as np

from genode.artifacts.identity import semantic_sha256


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
        if minimum_distance[next_index] <= 0:
            raise ValueError("Feature groups require at least one distinct coordinate per requested group.")
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
                donor_sizes = np.bincount(assignments, minlength=group_count)[assignments]
                candidates = np.where(donor_sizes > 1, assigned_distance, -1)
                replacement = int(np.argmax(candidates))
                assignments[replacement] = group_index
                assigned_distance[replacement] = -1.0
        centroids = np.stack(
            [coordinates[assignments == group_index].mean(axis=0) for group_index in range(group_count)]
        )
    else:
        raise RuntimeError("Deterministic feature grouping did not converge.")
    order = _sorted_cluster_order(centroids)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(group_count, dtype=np.int64)
    assignments = inverse[assignments]
    centroids = centroids[order]
    return assignments, centroids


def fit_feature_groups(
    class_means: np.ndarray,
    *,
    fit_split: str,
    source_reference_id: str,
    group_count: int = 32,
    component_count: int = 64,
) -> dict:
    """Fit deterministic PCA and k-means using training/calibration real features."""
    if fit_split not in {"train", "calibration"} or not source_reference_id:
        raise ValueError("Feature groups require training/calibration reference provenance.")
    values = np.asarray(class_means, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or min(values.shape) < component_count:
        raise ValueError("Finite class means must support the requested PCA dimension.")
    if not 1 <= group_count <= len(values):
        raise ValueError("Feature-group count must lie between one and class count.")
    center = values.mean(axis=0)
    _, singular_values, vectors = np.linalg.svd(values - center, full_matrices=False)
    components = vectors[:component_count].copy()
    scales = singular_values[:component_count] / np.sqrt(len(values) - 1)
    if np.any(scales <= np.finfo(np.float64).eps):
        raise ValueError("Class means have a degenerate PCA feature basis.")
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    coordinates = ((values - center) @ components.T) / scales
    assignments, centroids = _deterministic_feature_clusters(coordinates, group_count=group_count)
    payload = {
        "protocol": "image_reward_feature_groups_v3",
        "fit_split": fit_split,
        "source_reference_id": source_reference_id,
        "assignments": assignments.tolist(),
        "center": center.tolist(),
        "components": components.tolist(),
        "scales": scales.tolist(),
        "centroids": centroids.tolist(),
    }
    return {**payload, "sha256": semantic_sha256(payload, namespace="image-feature-groups")}


def paired_kid_shrinkage(
    class_kid: np.ndarray,
    jackknife_class_kid: np.ndarray,
    assignments: np.ndarray,
    *,
    uniform_index: int,
    fit_split: str,
) -> dict:
    """Estimate equally weighted class -> group -> global KID improvements.

    KID shape is [settings, schedules, classes]; jackknife adds a last paired
    delete-one block axis. Uniform and candidate must share those exact blocks.
    Coefficients are fitted from this training/calibration evidence only. Raw
    class KID is retained separately and is the metric used for evaluation.
    """
    if fit_split not in {"train", "calibration"}:
        raise ValueError("Conditional reward shrinkage may only fit training/calibration evidence.")
    kids = np.asarray(class_kid, dtype=np.float64)
    jackknife = np.asarray(jackknife_class_kid, dtype=np.float64)
    groups = np.asarray(assignments)
    if kids.ndim != 3 or jackknife.ndim != 4 or jackknife.shape[:-1] != kids.shape or jackknife.shape[-1] < 2:
        raise ValueError("Require class KID [settings,schedules,classes] and at least two paired jackknife blocks.")
    if not np.isfinite(kids).all() or not np.isfinite(jackknife).all():
        raise ValueError("KID and jackknife measurements must be finite.")
    if groups.shape != kids.shape[-1:] or not np.issubdtype(groups.dtype, np.integer) or np.any(groups < 0):
        raise ValueError("Feature groups require a nonnegative integer assignment for each class.")
    if not 0 <= uniform_index < kids.shape[1]:
        raise ValueError("Uniform schedule index is outside the measured schedule axis.")
    groups = groups.astype(np.int64)
    group_ids = np.unique(groups)
    if not np.array_equal(group_ids, np.arange(len(group_ids))):
        raise ValueError("Feature groups must be contiguous and nonempty.")
    advantages = kids[:, uniform_index : uniform_index + 1] - kids
    paired = jackknife[:, uniform_index : uniform_index + 1] - jackknife
    blocks = paired.shape[-1]
    variance = (blocks - 1) / blocks * ((paired - paired.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    shrunk = np.empty_like(advantages)
    coefficients = np.empty((*advantages.shape, 3), dtype=np.float64)
    sizes = np.bincount(groups)
    for setting in range(len(kids)):
        for schedule in range(kids.shape[1]):
            values = advantages[setting, schedule]
            global_mean = values.mean()
            means = np.asarray([values[groups == g].mean() for g in group_ids])
            group_replicates = np.stack([paired[setting, schedule, groups == g].mean(axis=0) for g in group_ids])
            global_replicates = paired[setting, schedule].mean(axis=0)
            class_residuals = paired[setting, schedule] - group_replicates[groups]
            group_residuals = group_replicates - global_replicates
            # Shrink contrasts, not absolute measurements: each estimated
            # target shares uncertainty with the quantity being shrunk.
            # Common fluctuations cancel and must not erase measured contrasts.
            class_noise = (
                (blocks - 1)
                / blocks
                * np.square(class_residuals - class_residuals.mean(axis=-1, keepdims=True)).sum(axis=-1)
            )
            group_noise = (
                (blocks - 1)
                / blocks
                * np.square(group_residuals - group_residuals.mean(axis=-1, keepdims=True)).sum(axis=-1)
            )
            within = max(float(np.mean((values - means[groups]) ** 2) - class_noise.mean()), 0.0)
            between = max(float(np.sum(sizes * ((means - global_mean) ** 2 - group_noise)) / len(values)), 0.0)
            cw = np.divide(within, within + class_noise, out=np.zeros_like(class_noise), where=within + class_noise > 0)
            gw = np.divide(
                between, between + group_noise, out=np.zeros_like(group_noise), where=between + group_noise > 0
            )[groups]
            coefficients[setting, schedule] = np.stack([cw, (1 - cw) * gw, (1 - cw) * (1 - gw)], axis=-1)
            shrunk[setting, schedule] = cw * values + (1 - cw) * (gw * means[groups] + (1 - gw) * global_mean)
    return {
        "raw_class_kid": kids,
        "raw_improvements": advantages,
        "standard_errors": np.sqrt(variance),
        "shrunk_improvements": shrunk,
        "coefficients": coefficients,
        "unshrunk_class_conditional_kid": kids.mean(axis=-1),
    }
