"""Native image protocol tests use labelled synthetic measurement fixtures."""

from copy import deepcopy

import numpy as np
import pytest

from genode.gico.evidence import prepare_evidence
from genode.gico.image_objective import validate_image_rows
from genode.gico.kid_objective import IMAGE_KID_OBJECTIVE, reference_identity
from genode.gico.rewards import TASK_METRICS, calibrate_rewards, construct_rewards, measurement_metrics
from tests.test_frozen_bezier_kid import evidence_rows


def native_rows(task="cifar10", classes=(0, 1)):
    rows = []
    templates = evidence_rows()
    for label in classes if task == "imagenet64" else (None,):
        for original in templates:
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
    with pytest.raises(ValueError, match="completed collection manifest"):
        prepare_evidence(rows, contexts, purpose="research")


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


@pytest.mark.parametrize("task", ["cifar10", "imagenet64"])
@pytest.mark.parametrize("protocol", [None, "unknown"])
def test_image_metric_routing_requires_an_explicit_objective(task, protocol):
    assert TASK_METRICS[task] == ("kid",)
    row = {"task": task, "metrics": {"lpips": 0.1}, "image_objective": {"protocol": protocol}}
    with pytest.raises(ValueError, match="explicit GICO KID or GICO-TF"):
        measurement_metrics(row)
    row["image_objective"]["protocol"] = "paired-lpips-v1"
    assert measurement_metrics(row) == ("lpips",)


def native_manifest(model_key):
    """Synthetic complete panels for the registered native backbone interfaces."""
    from genode.backbones import CheckpointBinding, ImageBackboneManifest
    from genode.backbones.registry import get_image_backbone_spec
    from genode.gico.image_conditional_context import context_binding

    spec = get_image_backbone_spec(model_key)
    backbone = ImageBackboneManifest(model_key, CheckpointBinding(spec.checkpoint_filename, "a" * 64, 1))
    table = np.zeros((1000, 768) if spec.dataset_key == "imagenet64" else (1, 1), dtype=np.float32)
    if spec.dataset_key == "imagenet64":
        table[:, 0] = np.arange(1000)
    binding = {**context_binding(backbone, table), "backbone": model_key, "clock_protocol": "density64-euler"}
    rows = native_rows(spec.dataset_key, classes=range(1000))
    for row in rows:
        row.update(backbone=model_key, backbone_binding=binding)
        row["image_objective"]["generator"] = binding
    return {"backbone_manifest": backbone.to_manifest_dict(), "native_context_table": table.tolist(), "rows": rows}


@pytest.mark.parametrize("task", ["cifar10", "imagenet64"])
@pytest.mark.parametrize("kind", ["GICO-det-policy", "GICO-sto-policy"])
def test_native_kid_both_students_retain_objective_through_fit_and_replay(tmp_path, monkeypatch, task, kind):
    from genode.gico.policy import load_policy
    from genode.gico.training import fit

    rows = native_rows(task)
    contexts = {r["context_id"]: [float(r["class_id"] or 0)] for r in rows}
    monkeypatch.setattr("genode.gico.training.accumulated_step", lambda *a, **kw: 0.0)
    metadata = fit(
        rows,
        contexts,
        tmp_path / "policy",
        purpose="functional",
        student_kind=kind,
        device="cpu",
        teacher_steps=2,
        student_steps=2,
        teacher_checkpoint_every=1,
        student_checkpoint_every=1,
    )
    assert metadata["image_objective"]["protocol"] == IMAGE_KID_OBJECTIVE
    assert all(c["metric_keys"] == ("kid",) for c in metadata["reward_calibrations"].values())
    policy = load_policy(tmp_path / "policy", student_kind=kind)
    first = policy.materialize([0.0], "euler", 4, seed=11, request_id="heldout-member")
    assert first == policy.materialize([0.0], "euler", 4, seed=11, request_id="heldout-member")
    assert len(first) == 5
