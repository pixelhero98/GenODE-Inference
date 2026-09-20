"""Deterministic surrogate selection, independent of terminal generation."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from genode.gico import training
from genode.gico.deterministic_selection import balanced_mean, score_deterministic, select_deterministic
from genode.gico.evidence import prepare_evidence
from genode.gico.networks import DensityTeacher, DeterministicStudent, ModelConfig
from genode.gico.policy import load_policy
from genode.gico.profiles import resolve_profile
from genode.gico.selection import state_fingerprint, teacher_fingerprint
from tests.test_unified_gico_policies import corrupt_payload, untrained_artifact  # noqa: F401
from tests.test_unified_gico_rewards import reference_evidence


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def row(step, kl, score):
    return {"step": step, "validation_distillation": kl, "predicted_utility": score, "coefficient": 0.01}


def test_defaults_and_invalid_allowances():
    config = resolve_profile("imagenet64")
    assert config.teacher_checkpoint_every == 20
    assert config.deterministic_checkpoint_every == 10
    assert config.student_checkpoint_every == 100
    assert config.deterministic_kl_allowance == 0.15
    assert config.teacher_steps == config.student_steps == 2000
    for bad in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            replace(config, deterministic_kl_allowance=bad)
    with pytest.raises(ValueError):
        replace(config, deterministic_checkpoint_every=0)


def test_relative_gate_final_minimum_and_earlier_ties():
    assert select_deterministic([row(1, 100, 1), row(2, 115, 2)], 0.15)["step"] == 2
    records = [row(1300, 1.14, 5), row(1310, 1, 2), row(1320, 1.14, 5), row(1330, 1.16, 100)]
    assert select_deterministic(records, 0.15)["step"] == 1300
    assert select_deterministic(records, 0.10)["step"] == 1310
    records.append(row(1340, 0.9, 1))
    assert select_deterministic(records, 0.15)["step"] == 1310
    records.append(row(1350, 0, -1))
    assert select_deterministic(records, 0.15)["step"] == 1350
    records.insert(0, {**row(1000, 0, 1000), "coefficient": 0})
    assert select_deterministic(records, 0.15)["step"] == 1350
    with pytest.raises(ValueError):
        select_deterministic([row(1, -0.1, 1)], 0.15)


def test_equal_setting_and_class_weighting():
    groups = [
        [{"context_id": c, "class_id": k, "solver": "euler", "nfe": n}]
        for c, k, n in [("a", 0, 2), ("b", 0, 2), ("c", 1, 2), ("d", 0, 4)]
    ]
    assert balanced_mean([0, 0, 12, 2], groups, "imagenet64") == 4
    assert balanced_mean([0, 0, 12, 2], groups, "traffic_hourly") == 3
    with pytest.raises(ValueError):
        balanced_mean([0], groups, "imagenet64")


def test_raw_native_teacher_score_reuses_one_global_student_density():
    evidence = prepare_evidence(*reference_evidence(), purpose="functional")
    conditioning = replace(evidence.conditioning, context_mode="global")
    architecture = ModelConfig(conditioning.width, 2, dropout=0.1)
    model = DeterministicStudent(architecture).eval()
    teacher = DensityTeacher(architecture).eval().requires_grad_(False)
    teacher.output.bias.data.copy_(torch.tensor([10.0, 30.0]))
    teacher.output.weight.data.zero_()
    groups = evidence.groups("validation")
    native = torch.tensor(evidence.conditioning.transform([3, 7], "euler", 4))[None]
    global_c = torch.tensor(conditioning.transform([3, 7], "euler", 4))[None]
    assert not torch.equal(native, global_c)
    with torch.no_grad():
        mass = model(global_c)
    targets = [(global_c, None, None, None, None, native)] * len(groups)
    identity = teacher_fingerprint(teacher, evidence.conditioning, 20, 0.05)
    before = state_fingerprint(teacher)
    rng = torch.random.get_rng_state().clone()
    with (
        patch.object(model, "forward", side_effect=AssertionError("Density already computed")),
        patch.object(teacher, "forward", wraps=teacher.forward) as forward,
    ):
        result = score_deterministic(
            model, teacher, conditioning, targets, [mass] * len(groups), groups, evidence, (0.25, 0.75), 10, identity
        )
    assert forward.call_count == len(groups)
    assert all(call.args[0] is native and call.args[1] is mass for call in forward.call_args_list)
    assert result["predicted_utility"] == pytest.approx(25 * evidence.calibrations["euler"].reward_scale)
    assert "utility" not in result and "measurements_sha256" not in result
    assert state_fingerprint(teacher) == before
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.fixture
def fitted(tmp_path):
    destination = tmp_path / "fit"
    metadata = training.fit(
        *reference_evidence(),
        destination,
        student_kind="GICO-det-policy",
        device="cpu",
        teacher_steps=2,
        teacher_checkpoint_every=1,
        student_steps=5,
        deterministic_checkpoint_every=1,
        student_context_mode="global",
        purpose="functional",
    )
    return destination, metadata


def test_new_artifact_roundtrip_and_checkpoint_history(fitted):
    destination, metadata = fitted
    records = metadata["history"]["students"]["GICO-det-policy"]
    assert [r["step"] for r in records] == [1, 2, 3, 4, 5]
    assert [r["step"] for r in records if "predicted_utility" in r] == [4, 5]
    assert metadata["history"]["student_selection"]["GICO-det-policy"] == select_deterministic(records, 0.15)
    with patch("genode.gico.policy.DensityTeacher", side_effect=AssertionError("Inference is student only")):
        policy = load_policy(destination)
    assert np.isclose(policy.density([1, 2], "euler", 4).sum(), 1)


@pytest.mark.parametrize(
    "field",
    ["teacher_weights", "teacher_shape", "teacher_identity", "missing_checkpoint", "missing_profile", "bad_score"],
)
def test_new_artifact_rejects_tampering(fitted, tmp_path, field):
    destination, _ = fitted

    def change(payload):
        metadata = payload["metadata"]
        if field == "teacher_weights":
            next(iter(payload["teacher"].values())).add_(1)
        elif field == "teacher_shape":
            payload["teacher"]["output.weight"] = payload["teacher"]["output.weight"].flatten()
        elif field == "teacher_identity":
            metadata["history"]["students"]["GICO-det-policy"][-1]["selection_teacher_fingerprint"] = "bad"
        elif field == "missing_checkpoint":
            metadata["history"]["students"]["GICO-det-policy"].pop(0)
        elif field == "missing_profile":
            metadata["fitting_profile"].pop("deterministic_kl_allowance")
        else:
            metadata["history"]["students"]["GICO-det-policy"][-1]["predicted_utility"] = float("nan")

    corrupt_payload(destination, tmp_path / "bad", change)
    with pytest.raises(ValueError):
        load_policy(tmp_path / "bad")


def test_validation_cadence_does_not_change_optimization_or_rng():
    evidence = prepare_evidence(*reference_evidence(), purpose="functional")
    config = resolve_profile(
        evidence.task,
        teacher_steps=2,
        teacher_checkpoint_every=1,
        student_steps=5,
        student_context_mode="global",
        dropout=0.1,
    )
    captures = []

    def forbidden(*a, **kw):
        raise AssertionError("Deterministic selection must not call a generator/scorer")

    for cadence in (1, 5):
        checkpoints = {}

        def capture(kind, step, model, record, checkpoints=checkpoints):
            checkpoints[step] = (
                deepcopy(model.state_dict()),
                record["objective"],
                torch.random.get_rng_state().clone(),
            )

        training.fit_models(
            evidence,
            replace(config, deterministic_checkpoint_every=cadence),
            student_kind="GICO-det-policy",
            device="cpu",
            checkpoint_callback=capture,
        )
        captures.append(checkpoints[5])
    first, second = captures
    assert first[1] == second[1] and torch.equal(first[2], second[2])
    for key in first[0]:
        assert torch.equal(first[0][key], second[0][key])


@pytest.mark.parametrize("kind", ["GICO-det-policy", "GICO-sto-policy"])
def test_incomplete_profile_is_rejected(untrained_artifact, tmp_path, kind):  # noqa: F811
    source, *_ = untrained_artifact

    def old_profile(payload):
        profile = payload["metadata"]["fitting_profile"]
        profile.pop("deterministic_checkpoint_every")
        profile.pop("deterministic_kl_allowance")
        profile["teacher_checkpoint_every"] = 100

    corrupt_payload(source, tmp_path / "legacy", old_profile)
    with pytest.raises(ValueError, match="incomplete"):
        load_policy(tmp_path / "legacy", student_kind=kind)
