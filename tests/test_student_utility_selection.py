"""Selection and schedule contracts without generator evaluation or optimizer updates."""

import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from genode.gico.evidence import prepare_evidence
from genode.gico.networks import DeterministicStudent, ModelConfig, StochasticStudent
from genode.gico.profiles import SCORE_SCHEDULES, SCORE_WEIGHTS
from genode.gico.selection import StudentCandidate, candidate_fingerprint, evaluate_candidate, measured_utility
from genode.gico.training import score_coefficient
from tests.selection_fixtures import evaluator_for
from tests.test_unified_gico_rewards import reference_evidence


@pytest.mark.parametrize("schedule", SCORE_SCHEDULES)
@pytest.mark.parametrize("beta", SCORE_WEIGHTS)
@pytest.mark.parametrize("horizon", [500, 2000])
def test_schedule_boundaries(schedule, beta, horizon):
    def coefficient(step):
        return score_coefficient(step - 1, horizon, beta, schedule)

    assert coefficient(int(horizon * 0.6)) == 0
    assert coefficient(int(horizon * 0.6) + 1) > 0
    assert coefficient(int(horizon * 0.8)) == pytest.approx(beta / 2 if schedule == "linear_60_40" else beta)
    assert coefficient(horizon) == pytest.approx(beta)
    if schedule == "constant_60_40":
        assert coefficient(int(horizon * 0.6) + 1) == beta


def fixture(kind="GICO-det-policy", task="traffic_hourly"):
    rows, contexts = reference_evidence(task=task)
    evidence = prepare_evidence(rows, contexts)
    config = ModelConfig(evidence.conditioning.width, len(evidence.calibrations["euler"].metric_keys), dropout=0.05)
    model = (DeterministicStudent if kind == "GICO-det-policy" else StochasticStudent)(config).eval()
    candidate = StudentCandidate(
        model, evidence.conditioning, kind, 500, 0.01, candidate_fingerprint(model, evidence.conditioning, kind, 500)
    )
    return rows, contexts, evidence, candidate


def test_distinct_generation_seeds_cannot_reuse_the_same_clock_rng_inputs():
    rows, contexts, evidence, candidate = fixture("GICO-sto-policy")
    extra = deepcopy([row for row in rows if row["split"] == "validation"])
    for row in extra:
        row["seed"] += 1
    rows.extend(extra)
    evidence = prepare_evidence(rows, contexts)
    measured = evaluator_for(rows, contexts)(candidate)
    assert measured_utility(measured, evidence, candidate, clock_replicates=4)["utility"] > 0
    students = [r for r in measured if r["schedule_key"] == "student"]
    first = students[0]
    other = next(r for r in students if r["seed"] != first["seed"])
    other["sample_clocks"][0] = deepcopy(first["sample_clocks"][0])
    with pytest.raises(ValueError, match="independent clocks"):
        measured_utility(measured, evidence, candidate, clock_replicates=4)


def test_evaluation_restores_all_rngs_and_does_not_mutate_live_model():
    rows, contexts, evidence, candidate = fixture()
    candidate.model.train()
    original = candidate_fingerprint(candidate.model, candidate.conditioning, candidate.student_kind, 500)
    torch_state, numpy_state, random_state = torch.random.get_rng_state(), np.random.get_state(), random.getstate()

    def evaluator(snapshot):
        torch.randn(9)
        np.random.random(9)
        random.random()
        assert not snapshot.model.training and all(not p.requires_grad for p in snapshot.model.parameters())
        return evaluator_for(rows, contexts)(snapshot)

    result = evaluate_candidate(
        candidate.model, candidate.conditioning, "GICO-det-policy", 500, 0.01, evaluator, evidence, 4
    )
    assert result["utility"] > 0 and candidate.model.training
    assert candidate_fingerprint(candidate.model, candidate.conditioning, candidate.student_kind, 500) == original
    assert torch.equal(torch_state, torch.random.get_rng_state())
    np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
    assert random_state == random.getstate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "test"),
        ("seed", 111),
        ("reference_id", "wrong"),
        ("selection_checkpoint_id", "wrong"),
        ("backbone", "wrong"),
    ],
)
def test_wrong_pair_or_split_rejected(field, value):
    rows, contexts, evidence, candidate = fixture()
    measurements = evaluator_for(rows, contexts)(candidate)
    measurements[1][field] = value
    with pytest.raises(ValueError):
        measured_utility(measurements, evidence, candidate, clock_replicates=4)


def test_utility_uses_frozen_calibration_and_correct_direction():
    rows, contexts, evidence, candidate = fixture(task="cifar10")
    measurements = evaluator_for(rows, contexts, factor=1)(candidate)
    assert measured_utility(measurements, evidence, candidate, clock_replicates=4)["utility"] == 0
    measurements[1]["metrics"]["lpips"] -= 0.1
    scale = evidence.calibrations["euler"].reward_scale
    assert measured_utility(measurements, evidence, candidate, clock_replicates=4)["utility"] == pytest.approx(
        0.1 / scale
    )


def test_duplicate_stochastic_innovations_and_incomplete_ensemble_rejected():
    rows, contexts, evidence, candidate = fixture("GICO-sto-policy")
    measurements = evaluator_for(rows, contexts, replicates=2)(candidate)
    bad = deepcopy(measurements)
    bad[1]["sample_clocks"][1] = deepcopy(bad[1]["sample_clocks"][0])
    with pytest.raises(ValueError, match="independent clocks"):
        measured_utility(bad, evidence, candidate, clock_replicates=2)
    bad = deepcopy(measurements)
    clocks = bad[1].pop("sample_clocks")
    bad[1].update(clocks[0])
    with pytest.raises(ValueError, match="one clock per member"):
        measured_utility(bad, evidence, candidate, clock_replicates=2)
    assert measured_utility(measurements, evidence, candidate, clock_replicates=2)["clock_replicates"] == 2
    # Negative raw repeats must not disappear into a positive average.
    measurements[3]["metrics"]["crps"] = -1
    with pytest.raises(ValueError, match="before repeat averaging"):
        measured_utility(measurements, evidence, candidate, clock_replicates=2)


def test_snapshot_conditioning_mutation_fails():
    rows, contexts, evidence, candidate = fixture()

    def evaluator(snapshot):
        snapshot.conditioning.context.mean[0] += 1
        return evaluator_for(rows, contexts)(snapshot)

    with pytest.raises(ValueError, match="mutated"):
        evaluate_candidate(
            candidate.model, candidate.conditioning, "GICO-det-policy", 500, 0.01, evaluator, evidence, 4
        )


def test_factory_validation_and_dry_run_do_not_import_runtime():
    from genode.gico.train_gico import load_selection_evaluator

    assert load_selection_evaluator({"factory": "unavailable_runtime:factory", "config": {}}, dry_run=True) is None
    with pytest.raises(ValueError, match="requires"):
        load_selection_evaluator(None)
    with pytest.raises(ValueError, match="module:factory"):
        load_selection_evaluator({"factory": "exec()", "config": {}})


def test_history_selects_utility_when_distillation_is_unchanged(monkeypatch):
    from genode.gico import training

    rows, contexts, evidence, _ = fixture()
    config = replace(
        training.TrainingConfig(),
        teacher_steps=2,
        student_steps=5,
        teacher_checkpoint_every=1,
        student_checkpoint_every=1,
    )
    monkeypatch.setattr(training, "accumulated_step", lambda *a, **kw: 0.0)
    seen = []

    def evaluator(candidate):
        seen.append(candidate.step)
        return evaluator_for(rows, contexts, factor=1 - candidate.step * 0.01)(candidate)

    _, _, history = training.fit_models(
        evidence, config, student_kind="GICO-det-policy", device="cpu", selection_evaluator=evaluator
    )
    assert seen == [4, 5]
    assert history["student_selection"]["GICO-det-policy"]["step"] == 5


@pytest.mark.parametrize(
    "task",
    [
        "solar_energy_10m",
        "weather_daily",
        "molecule_3d_set1",
        "molecule_3d_set2",
        "molecule_3d_set3",
        "sana",
        "sd15",
        "imagenet64",
    ],
)
def test_all_task_selection_uses_its_current_calibrated_objective(task):
    from genode.gico.rewards import TASK_METRICS
    from tests.image_fixtures import image_fields

    rows, contexts = reference_evidence()
    for row in rows:
        value = row["metrics"]["crps"]
        row["task"] = task
        row["metrics"] = {key: value ** (1 + i / 10) for i, key in enumerate(TASK_METRICS[task])}
        if task == "imagenet64":
            row.update(image_fields(row))
    evidence = prepare_evidence(rows, contexts, purpose="functional")
    model = DeterministicStudent(ModelConfig(evidence.conditioning.width, len(TASK_METRICS[task]))).eval()
    candidate = StudentCandidate(
        model,
        evidence.conditioning,
        "GICO-det-policy",
        500,
        0.01,
        candidate_fingerprint(model, evidence.conditioning, "GICO-det-policy", 500),
    )
    measurements = evaluator_for(rows, contexts)(candidate)
    expected = evidence.calibrations["euler"].scalar({**measurements[1], "anchor_metrics": measurements[0]["metrics"]})
    assert measured_utility(measurements, evidence, candidate, clock_replicates=4)["utility"] == pytest.approx(expected)


def test_replicate_metrics_are_averaged_before_log_improvement():
    rows, contexts, evidence, candidate = fixture("GICO-sto-policy")
    measured = evaluator_for(rows, contexts, replicates=2)(candidate)
    for i, factor in ((1, 0.25), (3, 0.75)):
        measured[i]["metrics"] = {k: v * factor for k, v in measured[0]["metrics"].items()}
    calibration = evidence.calibrations["euler"]
    expected = calibration.scalar(
        {
            **measured[1],
            "anchor_metrics": measured[0]["metrics"],
            "metrics": {k: v * 0.5 for k, v in measured[0]["metrics"].items()},
        }
    )
    assert measured_utility(measured, evidence, candidate, clock_replicates=2)["utility"] == pytest.approx(expected)
