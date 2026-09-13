"""Native image protocol tests use labelled synthetic measurement fixtures."""

from copy import deepcopy

import numpy as np
import pytest

from genode.gico.evidence import prepare_evidence
from genode.gico.image_objective import validate_image_rows
from genode.gico.kid_objective import IMAGE_KID_OBJECTIVE, reference_identity
from genode.gico.rewards import calibrate_rewards, construct_rewards
from tests.test_frozen_bezier_kid import evidence_rows


def native_rows(task="cifar10", classes=(0, 1)):
    rows = []
    for label in classes if task == "imagenet64" else (None,):
        for original in evidence_rows():
            row = deepcopy(original)
            generator = {
                "backbone": "fixture-native",
                "checkpoint_sha256": "a" * 64,
                "protocol_sha256": "b" * 64,
                "context_table_sha256": "c" * 64,
                "context_source": "zero" if label is None else "native_class_embedding",
                "clock_protocol": "density64-euler",
            }
            row.update(task=task, backbone=generator["backbone"], backbone_binding=generator, class_id=label)
            row["context_id"] += f":{label}"
            row["measurement_protocol"] = IMAGE_KID_OBJECTIVE
            row["image_objective"].update(protocol=IMAGE_KID_OBJECTIVE, generator=generator)
            if label is not None:
                row["sample_block"]["class_id"] = label
                row["reference_block"]["class_id"] = label
                row["reference_block"]["indices"] = [i + 1000 * label for i in row["reference_block"]["indices"]]
            row["reference_id"] = reference_identity(row["reference_block"])
            rows.append(row)
    return rows


@pytest.mark.parametrize("task", ["cifar10", "imagenet64"])
def test_native_kid_preserves_negative_estimates_and_paired_differences(task):
    rows = native_rows(task)
    validate_image_rows(rows)
    calibration = calibrate_rewards([r for r in rows if r["split"] == "train"])
    cells = construct_rewards(rows, calibration)
    assert calibration.metric_keys == ("kid",)
    assert any(c["metrics"]["kid"] < 0 for c in cells)
    assert all(c["reward"] == 0 if c["schedule_key"] == "uniform" else c["reward"] > 0 for c in cells)


@pytest.mark.parametrize("mutation", ["class", "incomplete", "mismatch", "overlap", "lpips", "solver"])
def test_native_kid_rejects_mislabelled_or_unpaired_blocks(mutation):
    rows = native_rows("imagenet64")
    row = rows[-1]
    if mutation == "class":
        row["sample_block"]["class_id"] = 7
    elif mutation == "incomplete":
        row["sample_block"]["noise_sha256"].pop()
    elif mutation == "mismatch":
        row["sample_block"]["noise_sha256"][0] = "f" * 64
    elif mutation == "overlap":
        row["sample_block"]["seeds"][1] = 0
    elif mutation == "lpips":
        row["metrics"]["lpips"] = 0.1
    else:
        row["solver"] = "heun"
    with pytest.raises(ValueError):
        validate_image_rows(rows)


def test_imagenet_research_requires_all_classes_before_fitting():
    rows = native_rows("imagenet64")
    contexts = {r["context_id"]: [float(r["class_id"])] for r in rows}
    assert prepare_evidence(rows, contexts, purpose="functional").task == "imagenet64"
    with pytest.raises(ValueError, match="1000-class"):
        prepare_evidence(rows, contexts)


def test_imagenet_calibration_does_not_overweight_a_repeated_class():
    rows = [r for r in native_rows("imagenet64") if r["split"] == "train"]
    for row in rows:
        if row["class_id"] == 1 and row["schedule_key"] != "uniform":
            row["metrics"]["kid"] *= 4
    original = calibrate_rewards(rows)
    extra = []
    for row in deepcopy(rows):
        if row["class_id"] != 1:
            continue
        row["context_id"] += ":repeat"
        row["panel_id"] += ":repeat"
        extra.append(row)
    repeated = calibrate_rewards(rows + extra)
    np.testing.assert_allclose(original.reward_scale, repeated.reward_scale, rtol=1e-12)
