"""Small objective and optimizer-step checks for task-specific fitting."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from genode.gico.networks import DensityTeacher, DeterministicStudent, ModelConfig, StochasticStudent
from genode.gico.profiles import resolve_profile
from genode.gico.rewards import MOLECULE_METRICS, RewardCalibration
from genode.gico.training import (
    accumulated_step,
    sample_groups,
    scalarize,
    selection_key,
    student_losses,
    utility_regret,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def test_profiles_preserve_task_settings_and_reject_obsolete_options():
    for task in ("sana", "sd15", "weather_daily", "molecule_3d_set1"):
        p = resolve_profile(task)
        assert p.teacher_steps == p.student_steps == 2000 and p.dropout == 0.01
        assert p.teacher_batch_groups == 64 and p.student_batch_contexts == 512
        assert p.teacher_learning_rate == p.student_learning_rate == 0.001
        assert p.teacher_score_weight == (0.05 if task in ("sana", "sd15") else 0.01)
    for task in ("cifar10", "imagenet64"):
        p = resolve_profile(task)
        assert p.student_steps == 2000 and p.dropout == 0 and p.student_learning_rate == 0.001
    with pytest.raises(ValueError, match="Unknown fitting"):
        resolve_profile("sana", steps=500)
    p = resolve_profile("sana", teacher_learning_rate=0.002, student_steps=600)
    assert resolve_profile("sana", **asdict(p)) == p


def test_distinct_group_sampling_caps_effective_batch():
    rng = np.random.default_rng(3)
    for count, batch in ((100, 64), (700, 512), (13, 64)):
        sampled = sample_groups(rng, count, batch)
        assert len(sampled) == len(set(sampled)) == min(count, batch)


def test_accumulation_matches_equal_group_loss_with_unequal_group_sizes():
    first = torch.nn.Linear(1, 1, bias=False).double()
    first.weight.data.fill_(0.03)
    second = deepcopy(first)
    groups = [torch.arange(1, n + 1, dtype=torch.float64)[:, None] for n in (2, 7, 3, 11, 4)]
    losses = []
    for model, micro in ((first, 1), (second, 5)):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        losses.append(
            accumulated_step(
                model,
                optimizer,
                groups,
                lambda batch, model=model: torch.stack([model(x).square().mean() for x in batch]),
                microbatch_contexts=micro,
            )
        )
    assert losses[0] == pytest.approx(losses[1])
    torch.testing.assert_close(first.weight, second.weight, rtol=1e-12, atol=1e-12)


def test_regret_uses_measured_utility_and_stable_tie_breaking():
    truth = torch.tensor([0.0, 2.0, 1.0], dtype=torch.float64)
    correct = utility_regret(truth, truth, temperature=0.05, reward_scale=1)
    wrong = utility_regret(truth.flip(0), truth, temperature=0.05, reward_scale=1)
    uniform = utility_regret(torch.zeros_like(truth), truth, temperature=0.05, reward_scale=1)
    assert correct < 1e-7 and uniform == pytest.approx(1)
    # Compare equivalent scalar-normalization conventions.
    torch.testing.assert_close(wrong, utility_regret(truth.flip(0) / 10, truth / 10, temperature=0.05, reward_scale=10))
    candidates = [(1.0, 0.1, 100), (1.0, 0.05, 300), (1.0, 0.05, 200)]
    assert min(candidates, key=lambda x: selection_key(*x, 0.05)) == (1.0, 0.05, 200)


def test_molecular_scalarization_is_weighted_log_improvement():
    calibration = RewardCalibration(
        "molecule_3d_set1", "b", "euler", MOLECULE_METRICS, (1e-6,) * 5, (1.0,) * 5, 2.0, ("train",), (4,)
    )
    metrics = dict(zip(MOLECULE_METRICS, (1.0, 2.0, 4.0, 8.0, 16.0), strict=True))
    cell = {
        "task": calibration.task,
        "backbone": "b",
        "solver": "euler",
        "metrics": metrics,
        "anchor_metrics": dict.fromkeys(MOLECULE_METRICS, 2.0),
    }
    vector = np.log(2.0 + 1e-6) - np.log(np.array(list(metrics.values())) + 1e-6)
    assert calibration.scalar(cell) == pytest.approx(vector @ np.array([0.4, 0.15, 0.15, 0.15, 0.15]) / 2)
    assert calibration.scalar({**cell, "metrics": cell["anchor_metrics"]}) == 0
    assert RewardCalibration.from_payload(calibration.to_payload()) == calibration


@pytest.mark.parametrize("kind", ("GICO-det-policy", "GICO-sto-policy"))
@pytest.mark.parametrize("coefficient", (0.01, 0.05, 0.1))
def test_actual_student_objective_score_term_has_gradients_without_teacher_updates(kind, coefficient):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        architecture = ModelConfig(5, 2, dropout=0.05)
        teacher = DensityTeacher(architecture).eval().requires_grad_(False)
        model = (
            DeterministicStudent(architecture) if kind == "GICO-det-policy" else StochasticStudent(architecture)
        ).eval()
    context = torch.zeros(1, 5)
    refs = torch.softmax(torch.stack((torch.linspace(-1, 1, 64), torch.linspace(1, -1, 64))), -1).double()
    with torch.no_grad():
        scores = scalarize(teacher(context.expand(2, -1), refs))
    groups = [(context, refs, torch.tensor([0.4, 0.6], dtype=torch.float64), scores.mean(), torch.tensor(1.0), context)]
    original = deepcopy(teacher.state_dict())
    config = resolve_profile("sana", stochastic_likelihood_samples=2, stochastic_score_samples=1)
    values = []
    for beta in (0, coefficient):
        model.zero_grad(set_to_none=True)
        value = student_losses(
            groups,
            model=model,
            kind=kind,
            teacher=teacher,
            config=config,
            weights=(0.5, 0.5),
            coefficient=beta,
            generator=torch.Generator().manual_seed(9),
            score_rng=torch.Generator().manual_seed(10),
        ).sum()
        value.backward()
        values.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
    difference = values[1] - values[0]
    assert torch.isfinite(difference).all() and difference.abs().max() > 0
    assert all(p.grad is None for p in teacher.parameters())
    for key in original:
        torch.testing.assert_close(original[key], teacher.state_dict()[key], atol=0, rtol=0)


def test_dropout_is_disabled_for_deterministic_and_replayable_stochastic_inference():
    config = ModelConfig(5, 2, dropout=0.05)
    model = DeterministicStudent(config)
    # Unzero output so hidden dropout affects visible predictions.
    torch.nn.init.normal_(model.output.weight, std=0.01)
    context = torch.zeros(1, 5)
    model.train()
    assert not torch.equal(model(context), model(context))
    model.eval()
    torch.testing.assert_close(model(context), model(context), atol=0, rtol=0)
    stochastic = StochasticStudent(config).eval()
    a = stochastic.sample(context, generator=torch.Generator().manual_seed(12))
    b = stochastic.sample(context, generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_global_conditioning_preserves_settings_and_zeroes_only_context():
    from dataclasses import replace

    from genode.gico.conditioning import Conditioning
    from genode.gico.evidence import prepare_evidence
    from tests.test_unified_gico_rewards import reference_evidence

    rows, contexts = reference_evidence()
    evidence = prepare_evidence(rows, contexts)
    original = evidence.conditioning
    global_condition = replace(original, context_mode="global")
    a = global_condition.transform([1, 2], "euler", 4)
    b = global_condition.transform([7, -9], "euler", 4)
    np.testing.assert_array_equal(a, b)
    assert not a[:2].any()
    np.testing.assert_array_equal(a[2:], original.transform([1, 2], "euler", 4)[2:])
    assert not np.array_equal(a, global_condition.transform([1, 2], "euler", 8))
    assert Conditioning.from_payload(global_condition.to_payload()).context_mode == "global"
    with pytest.raises(ValueError, match="width"):
        global_condition.transform([1], "euler", 4)
    model = DeterministicStudent(ModelConfig(len(a), 2, dropout=0.05)).eval()
    torch.testing.assert_close(model(torch.tensor(a[None])), model(torch.tensor(b[None])), atol=0, rtol=0)


@pytest.mark.parametrize(
    "option,value",
    [
        ("teacher_context_mode", "unknown"),
        ("student_context_mode", "unknown"),
        ("width", 64),
        ("teacher_density_normalization", "training_reference"),
    ],
)
def test_ablation_profiles_reject_unknown_semantics(option, value):
    with pytest.raises(ValueError):
        resolve_profile("sana", **{option: value})


def test_rope_and_causal_parameters_match_every_prefix():
    from genode.gico.networks import density_rope

    torch.manual_seed(19)
    x = torch.randn(2, 4, 64, 32)
    for length in (1, 2, 17, 63):
        torch.testing.assert_close(density_rope(x[..., :length, :]), density_rope(x)[..., :length, :], atol=0, rtol=0)
    student = StochasticStudent(ModelConfig(5, 2, dropout=0.05)).eval()
    torch.nn.init.normal_(student.output.weight, std=0.03)
    condition, ratios = torch.randn(2, 5), torch.randn(2, 63)
    full_mean, full_std = student.conditional_parameters(condition, ratios)
    for length in (1, 2, 17, 63):
        shifted = torch.cat((torch.zeros(2, 1), ratios[:, : length - 1]), 1)
        mean, std = student._distribution_head(condition, shifted)
        torch.testing.assert_close(mean, full_mean[:, :length], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(std, full_std[:, :length], atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", ("GICO-det-policy", "GICO-sto-policy"))
def test_global_student_score_chasing_keeps_original_teacher_contexts(kind):
    from unittest.mock import patch

    torch.manual_seed(43)
    config = resolve_profile(
        "sana",
        teacher_context_mode="native",
        student_context_mode="global",
        stochastic_likelihood_samples=2,
        stochastic_score_samples=2,
    )
    architecture = ModelConfig(5, 2)
    teacher = DensityTeacher(architecture).eval().requires_grad_(False)
    model = (
        DeterministicStudent(architecture) if kind == "GICO-det-policy" else StochasticStudent(architecture)
    ).eval()
    teacher_contexts = torch.tensor([[1.0, 2.0, 0.0, 0.0, 1.0], [3.0, -4.0, 0.0, 0.0, 1.0]])
    student_context = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
    refs = torch.softmax(torch.randn(2, 64), -1).double()
    groups = [
        (student_context, refs, torch.tensor([0.4, 0.6]), torch.tensor(0.0), torch.tensor(1.0), c[None])
        for c in teacher_contexts
    ]
    with patch.object(teacher, "forward", wraps=teacher.forward) as observed:
        losses = student_losses(
            groups,
            model=model,
            kind=kind,
            teacher=teacher,
            config=config,
            weights=(0.5, 0.5),
            coefficient=0.05,
            generator=torch.Generator().manual_seed(1),
            score_rng=torch.Generator().manual_seed(2),
        )
    expected = teacher_contexts if kind == "GICO-det-policy" else teacher_contexts.repeat_interleave(2, 0)
    torch.testing.assert_close(observed.call_args.args[0], expected, atol=0, rtol=0)
    assert losses.shape == (2,)
    losses.mean().backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_backbone_specific_parameters_are_not_shared():
    a = resolve_profile(
        "sana", backbone="checkpoint-a", teacher_score_weight=0.05, temperatures=(0.025,), preferred_temperature=0.025
    )
    b = resolve_profile(
        "sana", backbone="checkpoint-b", teacher_score_weight=0.1, temperatures=(0.1,), preferred_temperature=0.1
    )
    assert (
        a.backbone != b.backbone
        and a.temperatures != b.temperatures
        and a.teacher_score_weight != b.teacher_score_weight
    )
