"""Sparse collection isolation and policy-specific selection regressions."""

import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from genode.gico.clocks import density_identity
from genode.gico.collection import (
    CollectionConfig,
    collect,
    complete_collection,
    digest,
    plan_collection,
    validate_collection,
)
from genode.gico.deterministic_selection import select_deterministic
from genode.gico.evidence import prepare_evidence
from genode.gico.networks import DensityTeacher, ModelConfig, StochasticStudent
from genode.gico.profiles import TrainingConfig
from genode.gico.stochastic_selection import distribution_kl, score_stochastic, select_stochastic


def sparse_evidence():
    inventory = [{"context_id": str(i), "stratum": str(i % 5)} for i in range(50)]
    plan = plan_collection(
        "traffic_hourly",
        "fixture-backbone",
        inventory,
        [("euler", 4)],
        source_revision="a" * 40,
        config=CollectionConfig(context_budget=50),
    )

    def measure(request):
        moment = float(np.asarray(request["density_mass"]) @ np.linspace(1, 3, 64))
        return {
            **{k: v for k, v in request.items() if k != "request_id"},
            "ensemble_size": request["sample_count"],
            "reference_id": "reference:" + request["context_id"],
            "measurement_protocol": "fixture-complete-forecast",
            "metrics": {"crps": 3 + moment, "mase": 2 + moment**2},
        }

    rows, manifest = collect(plan, measure)
    contexts = {r["context_id"]: [float(i), float(i % 7)] for i, r in enumerate(inventory)}
    return rows, contexts, manifest


def test_sparse_research_calibration_excludes_both_holdouts():
    rows, contexts, manifest = sparse_evidence()
    evidence = prepare_evidence(rows, contexts, collection_manifest=manifest)
    altered = deepcopy(rows)
    for row in altered:
        if row["split"] == "validation" or evidence.is_density_holdout(row):
            row["metrics"] = {key: value * 1000 for key, value in row["metrics"].items()}
    changed_contexts = deepcopy(contexts)
    for context in manifest["split_contexts"]["validation"]:
        changed_contexts[context] = [1e9, -1e9]
    alternate = prepare_evidence(altered, changed_contexts, collection_manifest=complete_collection(manifest, altered))
    assert evidence.calibrations == alternate.calibrations
    assert digest(evidence.conditioning.to_payload()) == digest(alternate.conditioning.to_payload())
    assert max(len(group) for group in evidence.groups("train")) == 2


@pytest.mark.parametrize("mutation", ["missing-request", "density-holdout", "support", "split"])
def test_manifest_semantics_cannot_be_bypassed_by_rehashing(mutation):
    _, _, manifest = sparse_evidence()
    manifest = deepcopy(manifest)
    if mutation == "missing-request":
        manifest["requests"].pop()
    elif mutation == "density-holdout":
        manifest["density_holdout"]["euler:4"] = []
    elif mutation == "support":
        manifest["reference_support"]["euler:4"]["uniform"]["density_mass"][0] += 0.1
    else:
        manifest["split_contexts"]["train"].append(manifest["split_contexts"]["validation"].pop())
    planned = {k: v for k, v in manifest.items() if k not in {"plan_sha256", "measurements_sha256", "complete_solves"}}
    planned["state"] = "planned"
    manifest["plan_sha256"] = digest(planned)
    with pytest.raises(ValueError):
        validate_collection(manifest)


def test_sparse_students_use_full_support_without_measurement_calls(monkeypatch):
    from genode.gico import evaluators, task_evaluators, training

    rows, contexts, manifest = sparse_evidence()
    evidence = prepare_evidence(rows, contexts, collection_manifest=manifest)
    pool = evidence.reference_support["euler:4"]
    expected = {v["density_identity"] for v in pool.values()}
    seen = []

    def no_optimizer(model, optimizer, groups, loss_fn, **kwargs):
        if isinstance(model, DensityTeacher):
            for _, masses, _, _ in groups:
                assert not {density_identity(m.tolist()) for m in masses} & set(evidence.density_holdout["euler:4"])
        else:
            for target in groups:
                assert {density_identity(m.tolist()) for m in target[1]} == expected
            seen.append(type(model).__name__)
        return 0.0

    def forbidden(*a, **kw):
        raise AssertionError("Generator/scorer invoked during fitting")

    monkeypatch.setattr(training, "accumulated_step", no_optimizer)
    monkeypatch.setattr(evaluators, "execute_request", forbidden)
    monkeypatch.setattr(task_evaluators, "load_task_evaluator", forbidden)
    monkeypatch.setattr(evaluators, "build_collector", forbidden)
    teacher, students, history = training.fit_models(
        evidence,
        replace(TrainingConfig(), teacher_steps=1, student_steps=1, stochastic_likelihood_samples=2),
        student_kind="both",
        device="cpu",
    )
    assert set(seen) == {"DeterministicStudent", "StochasticStudent"}
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert history["teacher_selection"]["density_regret"] is not None
    assert len(students) == 2


def test_policy_gates_use_final_minimum_and_earlier_ties():
    def record(step, kl, utility, coefficient=1):
        return {"step": step, "validation_distillation": kl, "predicted_utility": utility, "coefficient": coefficient}

    rows = [record(10, 1, 100, 0), record(20, 117, 9), record(30, 100, 3), record(40, 100, 3)]
    assert select_deterministic(rows)["step"] == 30
    assert select_stochastic(rows)["step"] == 20
    rows.append(record(50, 90, 2))
    assert select_stochastic(rows)["step"] == 30


def test_distribution_kl_uses_full_mixture_joint_likelihoods():
    class Gaussian:
        def ratios(self, refs):
            return refs

        def conditional_parameters(self, condition, ratios):
            return torch.zeros_like(ratios), torch.ones_like(ratios) * 0.1

    refs = torch.zeros(2, 63, dtype=torch.double)
    refs[1, 0] = 0.2
    weights = torch.tensor([0.25, 0.75], dtype=torch.double)
    noise = torch.randn(32, 63, generator=torch.Generator().manual_seed(2), dtype=torch.double)
    samples = refs[:, None] + 0.1 * noise
    log_components = torch.distributions.Normal(refs, 0.1).log_prob(samples[:, :, None]).sum(-1)
    log_p = torch.logsumexp(log_components + weights.log(), -1)
    log_q = torch.distributions.Normal(samples.new_tensor(0.0), samples.new_tensor(0.1)).log_prob(samples).sum(-1)
    ratio = log_p - log_q
    expected = ((ratio + torch.expm1(-ratio)).mean(-1) * weights).sum()
    actual = distribution_kl(Gaussian(), torch.zeros(1, 2), refs, weights, noise, 0.1)
    torch.testing.assert_close(actual, expected)
    assert actual > 0
    assert distribution_kl(Gaussian(), torch.zeros(1, 2), refs[:1], weights.new_ones(1), noise, 0.1) == pytest.approx(
        0, abs=1e-12
    )


@pytest.mark.parametrize("tail_score", [-50.0, -30.0])
def test_distribution_kl_handles_underflow_without_dropping_positive_components(tail_score):
    from genode.gico.training import reference_weights

    class Gaussian:
        def __init__(self):
            self.samples = 0

        def ratios(self, refs):
            return refs

        def conditional_parameters(self, condition, ratios):
            self.samples += len(ratios)
            return torch.zeros_like(ratios), torch.ones_like(ratios)

    refs = torch.zeros(2, 63, dtype=torch.double)
    refs[1, 0] = 10
    weights = reference_weights(torch.tensor([0.0, tail_score]), temperature=0.05, reward_scale=1.0)
    noise = torch.randn(32, 63, generator=torch.Generator().manual_seed(2), dtype=torch.double)
    condition = torch.zeros(1, 2)
    model = Gaussian()
    actual = distribution_kl(model, condition, refs, weights, noise, 0.1)
    assert torch.isfinite(actual) and actual >= 0
    if tail_score == -50:
        assert weights[1] == 0
        expected = distribution_kl(Gaussian(), condition, refs[:1], weights[:1], noise, 0.1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert model.samples == 32
        reversed_value = distribution_kl(Gaussian(), condition, refs.flip(0), weights.flip(0), noise, 0.1)
        torch.testing.assert_close(actual, reversed_value, atol=0, rtol=0)
    else:
        assert 0 < weights[1] < 1e-250
        assert model.samples == 64


def test_stochastic_selection_is_replayable_and_rng_isolated():
    from tests.test_unified_gico_rewards import reference_evidence

    rows, contexts = reference_evidence()
    evidence = prepare_evidence(rows, contexts, purpose="functional")
    group = evidence.groups("validation")[0]
    condition = torch.tensor(evidence.conditioning.transform(contexts[group[0]["context_id"]], "euler", 4))[
        None
    ].float()
    config = ModelConfig(evidence.conditioning.width, 2)
    model = StochasticStudent(config).eval()
    teacher = DensityTeacher(config).eval().requires_grad_(False)
    mass = torch.tensor([r["density_mass"] for r in group], dtype=torch.double)
    target = (condition, mass, torch.ones(len(mass)) / len(mass), torch.tensor(0), torch.tensor(1), condition)
    profile = TrainingConfig()
    torch_state, np_state, py_state = torch.random.get_rng_state(), np.random.get_state(), random.getstate()
    first = score_stochastic(
        model, teacher, evidence.conditioning, [target], [group], evidence, [0.5, 0.5], 100, "teacher", profile
    )
    second = score_stochastic(
        model, teacher, evidence.conditioning, [target], [group], evidence, [0.5, 0.5], 100, "teacher", profile
    )
    assert first == second
    assert first["clock_replicates"] == 4 and first["kl_samples_per_reference"] == 32
    assert torch.equal(torch_state, torch.random.get_rng_state())
    np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
    assert py_state == random.getstate()
