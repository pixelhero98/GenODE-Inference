from __future__ import annotations

import copy

import numpy as np
import pytest

from genode.artifacts.identity import semantic_sha256
from genode.gico.clocks import materialize, reference_densities
from genode.gico.image_conditional import fit_feature_groups, paired_kid_shrinkage
from genode.gico.image_supervision import prepare_image_rows
from tests.test_image_primary_runtime import _frozen_cifar_backbone, _frozen_imagenet_backbone


def image_manifest(task="cifar10"):
    backbone = (
        _frozen_cifar_backbone(digest="a" * 64)
        if task == "cifar10"
        else _frozen_imagenet_backbone(digest="b" * 64, offset=0)
    )
    result = {"backbone_manifest": backbone.manifest.to_manifest_dict(), "rows": []}
    if task == "imagenet64":
        result["native_context_table"] = backbone.canonical_conditioning_table().tolist()
        groups = {
            "assignments": (np.arange(1000) % 32).tolist(),
            "fit_split": "train",
            "source_reference_id": "train-real",
        }
        result["feature_groups"] = {**groups, "sha256": semantic_sha256(groups, namespace="image-feature-groups")}
    for split in ("train", "validation"):
        for schedule in ("uniform", "late_p_3"):
            mass = np.asarray(reference_densities("euler", 2)[schedule])
            for label in range(1000 if task == "imagenet64" else 1):
                kid = 0.1 if schedule == "uniform" else 0.09 - (label % 7) * 0.001
                row = {
                    "solver": "euler",
                    "nfe": 2,
                    "split": split,
                    "seed": 3,
                    "ensemble_size": 8,
                    "reference_id": f"{split}-real",
                    "measurement_protocol": "kid-paired-block-v1",
                    "schedule_key": schedule,
                    "density_mass": mass.tolist(),
                    "time_grid": list(materialize(mass, "euler", 2)),
                    "metrics": {"kid": kid},
                }
                if task == "imagenet64":
                    row.update(class_id=label, jackknife_kid=[kid - 0.001, kid + 0.001])
                result["rows"].append(row)
    return result


def test_cifar_uses_zero_context_and_preserves_raw_paired_measurements():
    manifest = image_manifest()
    rows, contexts, metadata = prepare_image_rows(manifest)
    assert len(rows) == 4 and set(map(tuple, contexts.values())) == {(0.0,)}
    assert rows[0]["metrics"] == {"kid": 0.1}
    assert all("reward_metrics" not in row for row in rows)
    assert metadata["backbone_binding"]["context_source"] == "zero"
    assert metadata["raw_metric_report"][0]["global_kid"] == 0.1
    assert rows[0]["context_id"] != rows[2]["context_id"]
    assert rows[0]["context_id"] == rows[1]["context_id"]


def test_old_executed_clock_is_rejected():
    manifest = image_manifest()
    manifest["rows"][1]["time_grid"][1] += 0.01
    with pytest.raises(ValueError, match="recollect"):
        prepare_image_rows(manifest)


def test_imagenet_preserves_raw_kid_and_only_fits_training_evidence():
    manifest = image_manifest("imagenet64")
    original = copy.deepcopy(manifest["rows"])
    rows, contexts, metadata = prepare_image_rows(manifest)
    assert len(contexts) == 2000
    assert metadata["backbone_binding"]["context_source"] == "native_class_embedding"
    assert metadata["raw_metric_report"][0]["unshrunk_class_conditional_kid"] == pytest.approx(0.1)
    assert metadata["global_metric_report"] == []
    assert all(row["metrics"] == before["metrics"] for row, before in zip(rows, original, strict=True))
    assert all("reward_metrics" in row for row in rows if row["split"] == "train")
    assert all("reward_metrics" not in row for row in rows if row["split"] == "validation")
    train = [row for row in rows if row["split"] == "train"]
    assert train[0]["reward_metrics"] == train[0]["metrics"]
    assert sum(train[1000]["reward_estimator"]["coefficients"]) == pytest.approx(1)
    manifest["feature_groups"]["fit_split"] = "validation"
    with pytest.raises(ValueError, match="provenance"):
        prepare_image_rows(manifest)


def test_paired_jackknife_cancels_shared_noise_and_shrinkage_preserves_equal_class_weight():
    kids = np.asarray([[[1.0, 1.0, 1.0, 1.0], [0.8, 0.7, 0.4, 0.3]]])
    jackknife = kids[..., None] + np.asarray([-0.1, 0.1])
    result = paired_kid_shrinkage(kids, jackknife, np.asarray([0, 0, 1, 1]), uniform_index=0, fit_split="train")
    np.testing.assert_allclose(result["standard_errors"], 0, atol=1e-15)
    np.testing.assert_allclose(result["shrunk_improvements"], kids[:, :1] - kids)
    np.testing.assert_allclose(result["unshrunk_class_conditional_kid"], kids.mean(axis=-1))
    assert np.allclose(result["coefficients"].sum(axis=-1), 1)
    with pytest.raises(ValueError, match="training/calibration"):
        paired_kid_shrinkage(kids, jackknife, np.asarray([0, 0, 1, 1]), uniform_index=0, fit_split="test")


def test_training_feature_groups_are_reproducible_and_split_bound():
    values = np.random.default_rng(1).normal(size=(20, 8))
    first = fit_feature_groups(
        values, fit_split="train", source_reference_id="real-train", group_count=3, component_count=4
    )
    assert first == fit_feature_groups(
        values, fit_split="train", source_reference_id="real-train", group_count=3, component_count=4
    )
    assert set(first["assignments"]) == {0, 1, 2}
    with pytest.raises(ValueError, match="training/calibration"):
        fit_feature_groups(values, fit_split="validation", source_reference_id="real-val")


def test_renaming_split_context_does_not_hide_panel_leakage():
    manifest = image_manifest()
    for row in manifest["rows"]:
        row["reference_id"] = "same-panel"
    with pytest.raises(ValueError, match="disjoint paired measurement panels"):
        prepare_image_rows(manifest)


@pytest.mark.parametrize("pattern,expected_group_weight", [("common", 1.0), ("within", 1.0), ("between", 0.75)])
def test_group_shrinkage_uses_paired_group_global_contrasts(pattern, expected_group_weight):
    kids = np.array([[[10.0, 10.0, 10.0, 10.0], [9.0, 9.0, 5.0, 5.0]]])
    jackknife = np.repeat(kids[..., None], 2, axis=-1)
    offsets = np.tile([-1.0, 1.0], (4, 1))
    if pattern == "within":
        offsets[[1, 3]] *= -1
    elif pattern == "between":
        offsets[[2, 3]] *= -1
    jackknife[0, 1] -= offsets
    result = paired_kid_shrinkage(kids, jackknife, np.array([0, 0, 1, 1]), uniform_index=0, fit_split="train")
    # Every marginal class variance is 1. Only opposing group shifts create
    # uncertainty in the group-minus-global contrast; other shifts cancel.
    np.testing.assert_allclose(result["standard_errors"][0, 1], 1.0)
    expected_coefficients = np.tile([0.0, expected_group_weight, 1 - expected_group_weight], (4, 1))
    np.testing.assert_allclose(result["coefficients"][0, 1], expected_coefficients)
    independent_group_weight = 3.5 / (3.5 + 0.5)
    assert not np.isclose(expected_group_weight, independent_group_weight)
    expected = expected_group_weight * np.array([1, 1, 5, 5]) + (1 - expected_group_weight) * 3
    np.testing.assert_allclose(result["shrunk_improvements"][0, 1], expected)


def test_common_jackknife_noise_preserves_exact_class_and_group_contrasts():
    kids = np.array([[[10.0] * 4, [9.0, 8.0, 6.0, 5.0]]])
    jackknife = np.repeat(kids[..., None], 2, axis=-1)
    jackknife[0, 1] += [-10.0, 10.0]
    result = paired_kid_shrinkage(kids, jackknife, np.array([0, 0, 1, 1]), uniform_index=0, fit_split="train")
    np.testing.assert_allclose(result["standard_errors"][0, 1], 10.0)
    np.testing.assert_allclose(result["shrunk_improvements"][0, 1], [1.0, 2.0, 4.0, 5.0])
    np.testing.assert_allclose(result["coefficients"][0, 1, :, 0], 1.0)


def test_noisy_class_contrasts_shrink_to_precise_group_means():
    kids = np.array([[[10.0] * 4, [9.0, 7.0, 5.0, 3.0]]])
    jackknife = np.repeat(kids[..., None], 2, axis=-1)
    offsets = np.tile([-1.0, 1.0], (4, 1))
    offsets[[1, 3]] *= -1
    jackknife[0, 1] += offsets
    result = paired_kid_shrinkage(kids, jackknife, np.array([0, 0, 1, 1]), uniform_index=0, fit_split="train")
    np.testing.assert_allclose(result["shrunk_improvements"][0, 1], [2.0, 2.0, 6.0, 6.0])
    np.testing.assert_allclose(result["coefficients"][0, 1], np.tile([0.0, 1.0, 0.0], (4, 1)))


@pytest.mark.parametrize("task", ["cifar10", "imagenet64"])
def test_image_split_panel_reuse_is_rejected_even_when_generation_seed_changes(task):
    manifest = image_manifest(task)
    for row in manifest["rows"]:
        row["reference_id"] = "reused-reference-panel"
        if row["split"] == "validation":
            row["seed"] = 999
    with pytest.raises(ValueError, match="disjoint paired measurement panels"):
        prepare_image_rows(manifest)
