"""Frozen, indexed geometric features and a fair trajectory ensemble energy score."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MoleculeFeatureMap:
    """A geometry map fitted once from the training checkpoint reference.

    Pair distances and signed tetrahedron volumes retain atom indices. Each
    block is divided by the square root of its full trajectory dimension, so
    its squared Euclidean contribution is an average across atoms and time.
    """

    atom_count: int
    anchor_triangle: tuple[int, int, int]
    length_scale: float
    reference_sha256: str
    version: str = "indexed_trajectory_geometry_v1"

    def __post_init__(self) -> None:
        if self.version != "indexed_trajectory_geometry_v1":
            raise ValueError("Unsupported molecular feature map version.")
        if self.atom_count < 3 or len(self.anchor_triangle) != 3:
            raise ValueError("Molecular feature map requires at least three atoms and an anchor triangle.")
        if tuple(sorted(set(self.anchor_triangle))) != self.anchor_triangle or any(
            i < 0 or i >= self.atom_count for i in self.anchor_triangle
        ):
            raise ValueError("Anchor triangle must contain three increasing valid atom indices.")
        if not np.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError("Molecular length scale must be finite and positive.")
        if len(self.reference_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.reference_sha256):
            raise ValueError("Molecular feature map requires a training reference SHA256.")

    @classmethod
    def fit(cls, training_reference_coords: np.ndarray) -> MoleculeFeatureMap:
        coords = np.asarray(training_reference_coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 3 or not np.isfinite(coords).all():
            raise ValueError("Training reference must be finite [atoms >= 3, 3] coordinates.")
        left, right = np.triu_indices(len(coords), k=1)
        length = float(np.sqrt(np.mean(np.sum((coords[left] - coords[right]) ** 2, axis=-1))))
        if not np.isfinite(length) or length <= 0:
            raise ValueError("Degenerate training reference has no positive RMS pair distance.")
        normalized = (coords - coords[0]) / length
        # Lexicographic atom-index ordering resolves all geometric ties.
        triangle = next(
            (
                (i, j, k)
                for i, j, k in combinations(range(len(coords)), 3)
                if np.linalg.norm(np.cross(normalized[j] - normalized[i], normalized[k] - normalized[i])) > 1e-10
            ),
            None,
        )
        if triangle is None:
            raise ValueError("Degenerate training reference has no non-collinear anchor triangle.")
        fingerprint = hashlib.sha256(np.ascontiguousarray(coords, dtype="<f8").tobytes()).hexdigest()
        return cls(len(coords), triangle, length, fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "anchor_triangle": list(self.anchor_triangle)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MoleculeFeatureMap:
        return cls(**{**payload, "anchor_triangle": tuple(payload["anchor_triangle"])})

    def transform(self, trajectories: np.ndarray) -> np.ndarray:
        """Map [..., horizon, atoms, 3] to [..., indexed trajectory features]."""
        coords = np.asarray(trajectories, dtype=np.float64)
        if coords.ndim < 3 or coords.shape[-2:] != (self.atom_count, 3) or coords.shape[-3] < 1:
            raise ValueError(
                f"Expected [..., positive horizon, {self.atom_count}, 3] trajectories, got {coords.shape}."
            )
        if not np.isfinite(coords).all():
            raise ValueError("Molecular trajectories must contain only finite coordinates.")
        left, right = np.triu_indices(self.atom_count, k=1)
        pair = np.linalg.norm((coords[..., left, :] - coords[..., right, :]) / self.length_scale, axis=-1)
        i, j, k = self.anchor_triangle
        relative = (coords - coords[..., i : i + 1, :]) / self.length_scale
        normal = np.cross(relative[..., j, :], relative[..., k, :])
        # Signed tetrahedron volumes, indexed by each non-anchor atom.
        volume_atoms = [a for a in range(self.atom_count) if a not in self.anchor_triangle]
        volumes = np.sum(normal[..., None, :] * relative[..., volume_atoms, :], axis=-1) / 6.0
        leading = coords.shape[:-3]
        pair = pair.reshape(*leading, coords.shape[-3] * len(left))
        blocks = [pair / np.sqrt(pair.shape[-1])]
        if volume_atoms:
            volumes = volumes.reshape(*leading, coords.shape[-3] * len(volume_atoms))
            blocks.append(volumes / np.sqrt(volumes.shape[-1]))
        return np.concatenate(blocks, axis=-1)


def fair_energy_score(ensemble: np.ndarray, observation: np.ndarray) -> float:
    """Fair finite-ensemble energy estimator for one observed feature vector.

    Members must be independent draws. Equality of their values is allowed;
    independence is an RNG/sampling protocol obligation, not a value test.
    """
    members = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(observation, dtype=np.float64)
    if members.ndim != 2 or members.shape[0] < 2 or members.shape[1] < 1 or truth.shape != members.shape[1:]:
        raise ValueError("Fair energy score requires [members >= 2, features] and one [features] observation.")
    if not np.isfinite(members).all() or not np.isfinite(truth).all():
        raise ValueError("Energy score inputs must be finite.")
    count = members.shape[0]
    observation_term = np.linalg.norm(members - truth, axis=-1).mean()
    pair_term = sum(float(np.linalg.norm(members[i + 1 :] - members[i], axis=-1).sum()) for i in range(count - 1))
    score = float(observation_term - pair_term / (count * (count - 1)))
    # Euclidean triangle inequality guarantees nonnegativity; only roundoff can
    # cross zero, for example when two members straddle the observation.
    if score < -32 * np.finfo(np.float64).eps * max(1.0, float(observation_term)):
        raise ArithmeticError("Fair energy score violated the Euclidean triangle inequality.")
    return max(0.0, score)


def molecule_energy_score(
    ensemble_trajectories: np.ndarray,
    observed_future: np.ndarray,
    feature_map: MoleculeFeatureMap,
) -> float:
    ensemble = np.asarray(ensemble_trajectories, dtype=np.float64)
    observation = np.asarray(observed_future, dtype=np.float64)
    if ensemble.ndim != 4 or observation.shape != ensemble.shape[1:]:
        raise ValueError("Require [members, horizon, atoms, 3] ensemble and one matching [horizon, atoms, 3] future.")
    return fair_energy_score(feature_map.transform(ensemble), feature_map.transform(observation))
