"""Short forward/backward and artifact checks; no optimizer or fitting loop."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest
import torch

from genode.gico.clocks import clock_generator, materialize, reference_densities
from genode.gico.evidence import prepare_evidence
from genode.gico.networks import (
    DENSITY_BINS,
    DENSITY_MIXTURE,
    DensityTeacher,
    DeterministicStudent,
    ModelConfig,
    StochasticStudent,
    density_kl,
    guarded_mass,
)
from genode.gico.policy import GICO_PROTOCOL, load_policy, save_artifact
from genode.gico.profiles import AUXILIARY_NORMALIZATION, TEMPERATURE_UNITS, resolve_profile
from genode.gico.training import SCORE_WEIGHTS, TrainingConfig, score_coefficient, teacher_loss, teacher_score
from tests.test_unified_gico_rewards import reference_evidence


@pytest.fixture(autouse=True, scope="module")
def limited_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def models():
    config = ModelConfig(condition_dim=5, metric_count=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        return DensityTeacher(config), DeterministicStudent(config), StochasticStudent(config)


def test_fixed_transformer_architecture_and_normalized_density_outputs():
    teacher, deterministic, stochastic = models()
    conditions = torch.zeros(2, 5)
    for model in (teacher, deterministic, stochastic):
        assert len(model.blocks) == 2
        for block in model.blocks:
            assert block.norm_first and block.self_attn.num_heads == 4
            assert block.linear1.in_features == 128 and block.linear1.out_features == 256
            assert block.dropout.p == 0
    with torch.no_grad():
        mass = deterministic(conditions)
        assert mass.shape == (2, 64) and torch.all(mass > 0)
        torch.testing.assert_close(mass.sum(-1), torch.ones(2, dtype=torch.float64))
        assert teacher(conditions, mass).shape == (2, 2)
    with pytest.raises(ValueError, match="Unified GICO"):
        ModelConfig(5, 2, width=32)


def test_stochastic_conditionals_are_causal_and_use_observed_prefix():
    _, _, student = models()
    with torch.no_grad():
        # The prescribed zero-initialized output is prefix-independent; make
        # this a non-vacuous architectural check without training the model.
        student.output.weight.copy_(torch.linspace(-0.02, 0.02, 256).reshape(2, 128))
    condition = torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.1]])
    first = torch.zeros(1, 63)
    second = first.clone()
    second[:, 17:] = torch.linspace(1, 4, 46)
    mean_a, std_a = student.conditional_parameters(condition, first)
    mean_b, std_b = student.conditional_parameters(condition, second)
    torch.testing.assert_close(mean_a[:, :18], mean_b[:, :18], atol=1e-7, rtol=0)
    torch.testing.assert_close(std_a[:, :18], std_b[:, :18], atol=1e-7, rtol=0)
    assert torch.max(torch.abs(mean_a[:, 18:] - mean_b[:, 18:])) > 1e-5
    assert torch.all((std_b >= 0.05) & (std_b <= 2))


def test_stochastic_nll_is_mean_over_63_coordinates_and_ratio_transform_roundtrips():
    _, _, student = models()
    condition = torch.zeros(2, 5)
    target = torch.stack((torch.zeros(63), torch.ones(63)))
    means, stds = student.conditional_parameters(condition, target)
    pointwise = -torch.distributions.Normal(means, stds).log_prob(target)
    torch.testing.assert_close(student.nll(condition, target), pointwise.sum(-1) / 63)
    mass = torch.tensor(list(reference_densities("euler", 4).values())[:3], dtype=torch.float64)
    torch.testing.assert_close(student.density(student.ratios(mass)), guarded_mass(mass), atol=1e-12, rtol=1e-10)
    assert TrainingConfig().target_smoothing == 0.1
    with pytest.raises(ValueError, match="fixed at 0.1"):
        TrainingConfig(target_smoothing=0)


def test_sampling_is_replayable_and_independent_of_generation_rng():
    _, _, student = models()
    conditions = torch.zeros(1, 5)
    with torch.no_grad():
        before = torch.random.get_rng_state().clone()
        mass_a = student.sample(conditions, generator=clock_generator(9, "image-27"))
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        torch.randn(257)  # Advance the independent generation stream.
        mass_b = student.sample(conditions, generator=clock_generator(9, "image-27"))
        mass_c = student.sample(conditions, generator=clock_generator(9, "image-28"))
    torch.testing.assert_close(mass_a, mass_b, atol=0, rtol=0)
    assert not torch.equal(mass_a, mass_c)
    for reference in reference_densities("euler", 4).values():
        assert not np.allclose(mass_a[0].numpy(), reference, atol=1e-8, rtol=0)
    with pytest.raises(ValueError, match="explicit clock RNG"):
        student.sample(conditions)
    with pytest.raises(ValueError, match="request identity"):
        clock_generator(9, "")


@pytest.mark.parametrize("kind", ["deterministic", "stochastic"])
def test_teacher_score_has_finite_density_gradients_to_both_students_with_teacher_frozen(kind):
    teacher, deterministic, stochastic = models()
    student = deterministic if kind == "deterministic" else stochastic
    teacher.eval().requires_grad_(False)
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    condition = torch.tensor([[0.1, 0.2, -0.3, 0.4, 0.5]])
    mass = (
        student(condition)
        if kind == "deterministic"
        else student.sample(condition, innovations=torch.linspace(-1, 1, 63)[None])
    )
    mass.retain_grad()
    score = teacher_score(teacher, condition, mass)
    (-0.05 * score).backward()
    assert mass.grad is not None and torch.isfinite(mass.grad).all() and mass.grad.abs().sum() > 0
    gradients = [parameter.grad for parameter in student.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key], atol=0, rtol=0)


def test_teacher_score_clipping_and_equal_component_ranking():
    teacher, _, _ = models()
    with torch.no_grad():
        teacher.output.weight.zero_()
        teacher.output.bias.fill_(100)
    condition = torch.zeros(1, 5)
    mass = torch.full((1, 64), 1 / 64, dtype=torch.float64, requires_grad=True)
    assert teacher_score(teacher, condition, mass).item() == 5
    teacher.output.bias.data.fill_(-100)
    assert teacher_score(teacher, condition, mass).item() == -5
    truth = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    ordered = teacher_loss(truth, truth)
    reversed_score = teacher_loss(truth.flip(0), truth)
    assert ordered < reversed_score
    tied = torch.zeros(2, 2)
    assert teacher_loss(tied, tied).item() == 0


@pytest.mark.parametrize("weight", SCORE_WEIGHTS)
def test_score_weight_choices_and_exact_sixty_percent_ramp(weight):
    assert TrainingConfig(teacher_score_weight=weight).teacher_score_weight == weight
    assert score_coefficient(0, 100, weight) == 0
    assert score_coefficient(59, 100, weight) == 0
    assert score_coefficient(79, 100, weight) == pytest.approx(weight / 2)
    assert score_coefficient(99, 100, weight) == pytest.approx(weight)
    assert score_coefficient(150, 100, weight) == pytest.approx(weight)


def test_unsupported_score_weights_and_invalid_densities_fail():
    with pytest.raises(ValueError, match="one of"):
        TrainingConfig(teacher_score_weight=0.02)
    with pytest.raises(ValueError, match="Unsupported"):
        score_coefficient(1, 10, 0.02)
    uniform = torch.full((1, 64), 1 / 64, dtype=torch.float64)
    assert density_kl(uniform, uniform).item() == pytest.approx(0, abs=1e-12)
    spike = torch.zeros(1, DENSITY_BINS, dtype=torch.float64)
    spike[0, 10] = 1
    guarded = guarded_mass(spike)
    assert guarded[0, 0].item() == pytest.approx(DENSITY_MIXTURE / DENSITY_BINS)
    assert guarded.sum().item() == pytest.approx(1)
    with pytest.raises(ValueError, match="sum to one"):
        guarded_mass(uniform * 2)
    with pytest.raises(ValueError):
        materialize(np.full(64, np.nan), "euler", 4)
    with pytest.raises(ValueError):
        materialize(np.full(64, 1 / 64), "heun", 3)


def test_reference_temperature_uses_unclipped_pre_normalization_utilities():
    from genode.gico.training import reference_weights

    scores = torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float64)
    expected = (scores * 0.001 / 0.05).softmax(0)
    actual = reference_weights(scores, temperature=0.05, reward_scale=0.001)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, reference_weights(scores / 10, temperature=0.05, reward_scale=0.01))
    assert bool((actual > 0).all())


@pytest.fixture(scope="module")
def untrained_artifact(tmp_path_factory):
    rows, contexts = reference_evidence()
    evidence = prepare_evidence(rows, contexts)
    config = ModelConfig(evidence.conditioning.width, 2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        teacher = DensityTeacher(config)
        students = {"deterministic": DeterministicStudent(config), "stochastic": StochasticStudent(config)}
    metadata = {
        "task": evidence.task,
        "backbone": evidence.backbone,
        "purpose": "research",
        "solvers": list(evidence.calibrations),
        "reward_calibrations": {k: v.to_payload() for k, v in evidence.calibrations.items()},
        "evidence_sha256": evidence.evidence_sha256,
        "locked_test_used": False,
        "split_contexts": {"train": ["train-a"], "validation": ["validation-a"]},
        "reference_densities": {
            f"{r['solver']}:{r['nfe']}:{r['schedule_key']}": r["density_mass"] for r in evidence.cells
        },
        "reference_grids": {f"{r['solver']}:{r['nfe']}:{r['schedule_key']}": r["time_grid"] for r in evidence.cells},
        "measurement_protocols": ["paired-terminal-v3"],
        "fitting_profile": {"task": evidence.task, **asdict(resolve_profile(evidence.task, dropout=0))},
        "metric_weights": [0.5, 0.5],
        "temperature_units": TEMPERATURE_UNITS,
        "auxiliary_normalization": AUXILIARY_NORMALIZATION,
        "teacher_selection_criterion": "heldout_reference_utility_regret",
        "student_selection_criterion": "post_ramp_validation_distillation",
        "selected_temperature": 0.05,
        "history": {"student_selection": {k: {"step": 500, "coefficient": 0.01} for k in students}},
    }
    root = tmp_path_factory.mktemp("untrained-artifact") / "policy"
    save_artifact(root, teacher, students, evidence.conditioning, deepcopy(metadata))
    return root, teacher, students, evidence


@pytest.mark.parametrize("kind", ["deterministic", "stochastic"])
def test_standalone_artifact_roundtrip_without_loading_teacher(untrained_artifact, kind):
    root, _, students, evidence = untrained_artifact
    condition = torch.tensor(evidence.conditioning.transform([1, 2], "euler", 4)[None])
    students[kind].eval().requires_grad_(False)
    with torch.inference_mode():
        expected = (
            students[kind](condition)
            if kind == "deterministic"
            else students[kind].sample(condition, generator=clock_generator(31, "trajectory-6"))
        )
    with patch(
        "genode.gico.policy.DensityTeacher", side_effect=AssertionError("Inference must not instantiate teacher")
    ):
        loaded = load_policy(root / "policy.pt", student_kind=kind, expected_backbone=evidence.backbone)
        mass = loaded.density([1, 2], "euler", 4, seed=31, request_id="trajectory-6")
        grid = loaded.materialize([1, 2], "euler", 4, seed=31, request_id="trajectory-6")
    expected_mass = expected[0].numpy().astype(np.float64)
    expected_mass /= expected_mass.sum()
    np.testing.assert_array_equal(mass, expected_mass)
    assert grid == materialize(mass, "euler", 4)
    assert not hasattr(loaded, "teacher")
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    with pytest.raises(ValueError, match="backbone identities differ"):
        load_policy(root, expected_backbone="wrong-backbone")


def corrupt_payload(source, destination, change):
    shutil.copytree(source, destination)
    payload = torch.load(destination / "policy.pt", weights_only=True)
    change(payload)
    torch.save(payload, destination / "policy.pt")
    manifest = {
        "protocol": GICO_PROTOCOL,
        "policy_sha256": hashlib.sha256((destination / "policy.pt").read_bytes()).hexdigest(),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest))


def test_artifact_checksum_and_old_versions_are_rejected(untrained_artifact, tmp_path):
    root, _, _, _ = untrained_artifact
    damaged = tmp_path / "damaged"
    shutil.copytree(root, damaged)
    with (damaged / "policy.pt").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_policy(damaged)
    manifest = json.loads((damaged / "manifest.json").read_text())
    manifest["protocol"] = "old-gico"
    (damaged / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported artifact version"):
        load_policy(damaged)


@pytest.mark.parametrize(
    "mutation",
    ["missing-calibration", "changed-reference-grid", "bad-reference-density", "split-overlap", "nonfinite-weights"],
)
def test_semantically_corrupt_artifacts_are_rejected_even_with_valid_checksum(untrained_artifact, tmp_path, mutation):
    root, _, _, _ = untrained_artifact

    def change(payload):
        metadata = payload["metadata"]
        if mutation == "missing-calibration":
            metadata["reward_calibrations"] = {}
        elif mutation == "changed-reference-grid":
            metadata["reference_grids"][next(iter(metadata["reference_grids"]))][1] += 0.01
        elif mutation == "bad-reference-density":
            metadata["reference_densities"][next(iter(metadata["reference_densities"]))][0] = -1
        elif mutation == "split-overlap":
            metadata["split_contexts"]["validation"] = ["train-a"]
        else:
            parameter = next(iter(payload["students"]["deterministic"].values()))
            parameter.view(-1)[0] = float("nan")

    destination = tmp_path / mutation
    corrupt_payload(root, destination, change)
    with pytest.raises(ValueError):
        load_policy(destination)


def test_nontrivial_ratio_normalizers_are_inverted_without_changing_density():
    mean = torch.linspace(-2, 3, 63)
    scale = torch.linspace(0.2, 2.5, 63)
    student = StochasticStudent(ModelConfig(5, 2), mean, scale)
    masses = torch.tensor(list(reference_densities("euler", 8).values())[5:8], dtype=torch.float64)
    log_mass = guarded_mass(masses).log()
    expected = (log_mass[:, :-1] - log_mass[:, -1:] - mean) / scale
    torch.testing.assert_close(student.ratios(masses), expected)
    torch.testing.assert_close(student.density(expected), guarded_mass(masses), atol=1e-12, rtol=1e-10)
    with pytest.raises(ValueError, match="positive"):
        StochasticStudent(ModelConfig(5, 2), mean, torch.zeros(63))


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["metadata"]["fitting_profile"].update(teacher_density_normalization="running"),
        lambda p: p["metadata"]["fitting_profile"].update(context_mode="global"),
        lambda p: p["metadata"]["fitting_profile"].update(width=64),
        lambda p: p["teacher"]["density_scale"].zero_(),
    ],
)
def test_teacher_loader_rejects_invalid_profile_or_normalizer(untrained_artifact, tmp_path, change):
    from genode.gico.policy import load_teacher

    root, _, _, _ = untrained_artifact
    damaged = tmp_path / "teacher-invalid"
    corrupt_payload(root, damaged, change)
    with pytest.raises(ValueError):
        load_teacher(damaged)


def test_global_normalized_width64_artifact_roundtrip(untrained_artifact, tmp_path):
    from dataclasses import replace

    from genode.gico.policy import load_teacher

    root, _, _, evidence = untrained_artifact
    metadata = torch.load(root / "policy.pt", weights_only=True)["metadata"]
    config = ModelConfig(evidence.conditioning.width, 2, width=64)
    teacher = DensityTeacher(config, torch.linspace(-5, -3, 64), torch.linspace(0.1, 1, 64)).eval()
    student = DeterministicStudent(config).eval()
    metadata["fitting_profile"].update(
        width=64, context_mode="global", teacher_density_normalization="training_reference"
    )
    metadata["history"]["student_selection"] = {"deterministic": {"step": 500, "coefficient": 0.01}}
    path = tmp_path / "global"
    save_artifact(
        path, teacher, {"deterministic": student}, replace(evidence.conditioning, context_mode="global"), metadata
    )
    policy = load_policy(path)
    np.testing.assert_array_equal(policy.density([1, 2], "euler", 4), policy.density([9, -5], "euler", 4))
    restored, conditioning, _ = load_teacher(path)
    assert conditioning.context_mode == "global"
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
