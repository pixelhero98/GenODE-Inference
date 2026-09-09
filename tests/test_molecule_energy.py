from __future__ import annotations

import json

import numpy as np
import pytest

from genode.evaluation.molecule_energy import MoleculeFeatureMap, fair_energy_score, molecule_energy_score


@pytest.fixture
def geometry():
    return np.asarray([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0.2, 0.3, 1.5]], dtype=float)


def test_fair_energy_analytic_values_and_spread():
    assert fair_energy_score(np.asarray([[-1], [1]]), np.asarray([0])) == 0
    assert fair_energy_score(np.asarray([[1], [1]]), np.asarray([0])) == 1
    assert fair_energy_score(np.asarray([[0], [2]]), np.asarray([3])) == 1
    # Three pair distances total eight; observation distances total six.
    assert fair_energy_score(np.asarray([[0], [2], [4]]), np.asarray([0])) == pytest.approx(2 - 8 / 6)


def test_geometry_rigid_motion_and_mirror_sensitivity(geometry):
    feature_map = MoleculeFeatureMap.fit(geometry)
    trajectory = np.stack([geometry, geometry * 1.1])
    rotation, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    np.testing.assert_allclose(
        feature_map.transform(trajectory),
        feature_map.transform(trajectory @ rotation + [4, -2, 8]),
        atol=1e-14,
    )
    reflection = trajectory * [-1, 1, 1]
    original = feature_map.transform(trajectory)
    reflected = feature_map.transform(reflection)
    np.testing.assert_allclose(original[:12], reflected[:12])
    np.testing.assert_allclose(original[12:], -reflected[12:])
    assert molecule_energy_score(np.stack([reflection, reflection]), trajectory, feature_map) > 0
    assert molecule_energy_score(np.stack([trajectory, trajectory]), trajectory, feature_map) == 0


def test_geometry_preserves_atom_and_horizon_order(geometry):
    feature_map = MoleculeFeatureMap.fit(geometry)
    trajectory = np.stack([geometry, geometry * 1.2])
    features = feature_map.transform(trajectory)
    assert not np.allclose(features, feature_map.transform(trajectory[::-1]))
    assert not np.allclose(features, feature_map.transform(trajectory[:, [1, 0, 2, 3]]))


def test_reference_is_frozen_and_serializable(geometry):
    feature_map = MoleculeFeatureMap.fit(geometry)
    assert feature_map.anchor_triangle == (0, 1, 2)
    expected_length = np.sqrt(np.mean([np.sum((geometry[i] - geometry[j]) ** 2) for i in range(4) for j in range(i)]))
    assert feature_map.length_scale == pytest.approx(expected_length)
    restored = MoleculeFeatureMap.from_dict(json.loads(json.dumps(feature_map.to_dict())))
    assert restored == feature_map
    assert restored.to_dict() == feature_map.to_dict()
    np.testing.assert_equal(restored.transform(geometry[None]), feature_map.transform(geometry[None]))
    # Scaling coordinates and fitting the reference together leaves features unchanged.
    np.testing.assert_allclose(
        MoleculeFeatureMap.fit(geometry * 7).transform(geometry[None] * 7), restored.transform(geometry[None])
    )
    assert feature_map.to_dict() == restored.to_dict()


def test_lexicographic_triangle_skips_collinear_atoms():
    reference = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert MoleculeFeatureMap.fit(reference).anchor_triangle == (0, 1, 3)
    assert MoleculeFeatureMap.fit(reference.copy()).anchor_triangle == (0, 1, 3)


@pytest.mark.parametrize(
    "reference", [np.zeros((4, 3)), np.eye(2, 3), np.full((4, 3), np.nan), np.arange(12).reshape(4, 3)]
)
def test_degenerate_reference_rejected(reference):
    with pytest.raises(ValueError):
        MoleculeFeatureMap.fit(reference)


def test_invalid_ensembles_and_repeated_truth_rejected(geometry):
    feature_map = MoleculeFeatureMap.fit(geometry)
    truth = geometry[None]
    members = np.stack([truth, truth])
    for invalid in (members[:1], members * np.nan):
        with pytest.raises(ValueError):
            molecule_energy_score(invalid, truth, feature_map)
    for invalid_truth in (members, geometry, np.repeat(truth, 2, axis=0)):
        with pytest.raises(ValueError):
            molecule_energy_score(members, invalid_truth, feature_map)
    for invalid in (np.empty((0, 4, 3)), np.zeros((1, 3, 3)), np.full_like(truth, np.inf)):
        with pytest.raises(ValueError):
            feature_map.transform(invalid)


def test_energy_nonnegative_for_random_ensembles():
    rng = np.random.default_rng(43)
    for count in (2, 3, 16):
        for _ in range(10):
            assert fair_energy_score(rng.normal(size=(count, 15)), rng.normal(size=15)) >= 0
