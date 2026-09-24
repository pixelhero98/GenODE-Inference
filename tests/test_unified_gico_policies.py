"""Short forward/backward and artifact checks; no optimizer or fitting loop."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from genode.gico.clocks import clock_generator, materialize, reference_densities
from genode.gico.evidence import content_hash, prepare_evidence
from genode.gico.networks import (
    DENSITY_BINS,
    DENSITY_MIXTURE,
    DeterministicPolicy,
    ModelConfig,
    StochasticPolicy,
    UtilitySurrogate,
    density_kl,
    guarded_mass,
)
from genode.gico.policy import GICO_PROTOCOL, GICO_SCHEMA_VERSION, load_policy, save_artifact
from genode.gico.profiles import AUXILIARY_NORMALIZATION, TEMPERATURE_UNITS, resolve_profile
from genode.gico.selection import utility_surrogate_fingerprint
from genode.gico.training import (
    REFINEMENT_WEIGHT,
    TrainingConfig,
    score_coefficient,
    utility_surrogate_loss,
    utility_surrogate_score,
)
from tests.selection_fixtures import fixture_history
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
        return UtilitySurrogate(config), DeterministicPolicy(config), StochasticPolicy(config)


def test_fixed_transformer_architecture_and_normalized_density_outputs():
    utility_surrogate, deterministic, stochastic = models()
    conditions = torch.zeros(2, 5)
    for model in (utility_surrogate, deterministic, stochastic):
        assert len(model.blocks) == 2
        for block in model.blocks:
            assert isinstance(block.norm1, torch.nn.LayerNorm) and block.heads == 4
            assert block.ff[0].in_features == 128 and block.ff[0].out_features == 256
            assert block.dropout.p == 0
    with torch.no_grad():
        mass = deterministic(conditions)
        assert mass.shape == (2, 64) and torch.all(mass > 0)
        torch.testing.assert_close(mass.sum(-1), torch.ones(2, dtype=torch.float64))
        assert utility_surrogate(conditions, mass).shape == (2, 2)
    with pytest.raises(ValueError, match="Unified GICO"):
        ModelConfig(5, 2, width=32)


def test_stochastic_conditionals_are_causal_and_use_observed_prefix():
    _, _, policy = models()
    with torch.no_grad():
        # The prescribed zero-initialized output is prefix-independent; make
        # this a non-vacuous architectural check without training the model.
        policy.output.weight.copy_(torch.linspace(-0.02, 0.02, 256).reshape(2, 128))
    condition = torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.1]])
    first = torch.zeros(1, 63)
    second = first.clone()
    second[:, 17:] = torch.linspace(1, 4, 46)
    mean_a, std_a = policy.conditional_parameters(condition, first)
    mean_b, std_b = policy.conditional_parameters(condition, second)
    torch.testing.assert_close(mean_a[:, :18], mean_b[:, :18], atol=1e-7, rtol=0)
    torch.testing.assert_close(std_a[:, :18], std_b[:, :18], atol=1e-7, rtol=0)
    assert torch.max(torch.abs(mean_a[:, 18:] - mean_b[:, 18:])) > 1e-5
    assert torch.all((std_b >= 0.05) & (std_b <= 2))


def test_stochastic_nll_is_mean_over_63_coordinates_and_ratio_transform_roundtrips():
    _, _, policy = models()
    condition = torch.zeros(2, 5)
    target = torch.stack((torch.zeros(63), torch.ones(63)))
    means, stds = policy.conditional_parameters(condition, target)
    pointwise = -torch.distributions.Normal(means, stds).log_prob(target)
    torch.testing.assert_close(policy.nll(condition, target), pointwise.sum(-1) / 63)
    mass = torch.tensor(list(reference_densities("euler", 4).values())[:3], dtype=torch.float64)
    torch.testing.assert_close(policy.density(policy.ratios(mass)), guarded_mass(mass), atol=1e-12, rtol=1e-10)
    assert TrainingConfig().target_smoothing == 0.1
    with pytest.raises(ValueError, match="fixed at 0.1"):
        TrainingConfig(target_smoothing=0)


def test_sampling_is_replayable_and_independent_of_generation_rng():
    _, _, policy = models()
    conditions = torch.zeros(1, 5)
    with torch.no_grad():
        before = torch.random.get_rng_state().clone()
        mass_a = policy.sample(conditions, generator=clock_generator(9, "image-27"))
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        torch.randn(257)  # Advance the independent generation stream.
        mass_b = policy.sample(conditions, generator=clock_generator(9, "image-27"))
        mass_c = policy.sample(conditions, generator=clock_generator(9, "image-28"))
    torch.testing.assert_close(mass_a, mass_b, atol=0, rtol=0)
    assert not torch.equal(mass_a, mass_c)
    for reference in reference_densities("euler", 4).values():
        assert not np.allclose(mass_a[0].numpy(), reference, atol=1e-8, rtol=0)
    with pytest.raises(ValueError, match="explicit clock RNG"):
        policy.sample(conditions)
    with pytest.raises(ValueError, match="request identity"):
        clock_generator(9, "")


@pytest.mark.parametrize("kind", ["deterministic", "stochastic"])
def test_utility_surrogate_score_has_finite_density_gradients_to_both_policies_with_utility_surrogate_frozen(kind):
    utility_surrogate, deterministic, stochastic = models()
    policy = deterministic if kind == "deterministic" else stochastic
    utility_surrogate.eval().requires_grad_(False)
    utility_surrogate_before = {key: value.clone() for key, value in utility_surrogate.state_dict().items()}
    condition = torch.tensor([[0.1, 0.2, -0.3, 0.4, 0.5]])
    mass = (
        policy(condition)
        if kind == "deterministic"
        else policy.sample(condition, innovations=torch.linspace(-1, 1, 63)[None])
    )
    mass.retain_grad()
    score = utility_surrogate_score(utility_surrogate, condition, mass)
    (-0.05 * score).backward()
    assert mass.grad is not None and torch.isfinite(mass.grad).all() and mass.grad.abs().sum() > 0
    gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    assert all(parameter.grad is None for parameter in utility_surrogate.parameters())
    if kind == "stochastic":
        assert (policy.output.weight.grad.abs().sum(1) > 0).all()
        assert (policy.output.bias.grad.abs() > 0).all()
    for key, value in utility_surrogate.state_dict().items():
        torch.testing.assert_close(value, utility_surrogate_before[key], atol=0, rtol=0)


def test_utility_surrogate_score_clipping_and_equal_component_ranking():
    utility_surrogate, _, _ = models()
    with torch.no_grad():
        utility_surrogate.output.weight.zero_()
        utility_surrogate.output.bias.fill_(100)
    condition = torch.zeros(1, 5)
    mass = torch.full((1, 64), 1 / 64, dtype=torch.float64, requires_grad=True)
    assert utility_surrogate_score(utility_surrogate, condition, mass).item() == 5
    utility_surrogate.output.bias.data.fill_(-100)
    assert utility_surrogate_score(utility_surrogate, condition, mass).item() == -5
    truth = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    ordered = utility_surrogate_loss(truth, truth)
    reversed_score = utility_surrogate_loss(truth.flip(0), truth)
    assert ordered < reversed_score
    tied = torch.zeros(2, 2)
    assert utility_surrogate_loss(tied, tied).item() == 0


@pytest.mark.parametrize("weight", (REFINEMENT_WEIGHT,))
def test_score_weight_choices_and_exact_sixty_percent_ramp(weight):
    assert TrainingConfig(refinement_weight=weight).refinement_weight == weight
    assert score_coefficient(0, 100, weight) == 0
    assert score_coefficient(59, 100, weight) == 0
    assert score_coefficient(79, 100, weight) == pytest.approx(weight / 2)
    assert score_coefficient(99, 100, weight) == pytest.approx(weight)
    assert score_coefficient(150, 100, weight) == pytest.approx(weight)


def test_unsupported_score_weights_and_invalid_densities_fail():
    with pytest.raises(ValueError, match="fixed"):
        TrainingConfig(refinement_weight=0.02)
    with pytest.raises(ValueError, match="fixed"):
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
    evidence = prepare_evidence(rows, contexts, purpose="functional")
    config = ModelConfig(evidence.conditioning.width, 2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        utility_surrogate = UtilitySurrogate(config)
        policies = {"deterministic": DeterministicPolicy(config), "stochastic": StochasticPolicy(config)}
    metadata = {
        "task": evidence.task,
        "backbone": evidence.backbone,
        "purpose": "functional",
        "collection_manifest": evidence.collection_manifest,
        "collection_sha256": content_hash(evidence.collection_manifest),
        "source_code_sha256": "a" * 64,
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
        "fitting_profile": {
            "task": evidence.task,
            **asdict(resolve_profile(evidence.task, backbone=evidence.backbone, dropout=0)),
        },
        "metric_weights": [0.5, 0.5],
        "temperature_units": TEMPERATURE_UNITS,
        "auxiliary_normalization": AUXILIARY_NORMALIZATION,
        "utility_surrogate_selection_criterion": "heldout_component_mse_then_reference_regret",
        "utility_surrogate_prediction_semantics": "native_context",
        "policy_selection_criterion": "heldout_calibrated_utility_surrogate_utility_with_policy_kl",
        "selected_temperature": 0.05,
        "history": {
            "evidence_fingerprint": evidence.evidence_sha256,
            "calibration_fingerprint": content_hash({k: v.to_payload() for k, v in evidence.calibrations.items()}),
            "collection_fingerprint": content_hash(evidence.collection_manifest),
            "support_fingerprint": content_hash(evidence.reference_support),
            "source_fingerprint": "a" * 64,
            **fixture_history(policies, evidence, rows, contexts, utility_surrogate=utility_surrogate),
            "density_holdout": evidence.density_holdout,
            "utility_surrogate": [
                {
                    "step": n,
                    "temperature": t,
                    "regret": 0.1 + (t != 0.05) + (n != 20),
                    "context_regret": 0.1 + (t != 0.05) + (n != 20),
                    "density_regret": 0.1 + (t != 0.05) + (n != 20),
                    "component_mse": 0.1 + (n != 20),
                    "context_mse": 0.1 + (n != 20),
                    "density_mse": 0.1 + (n != 20),
                    "stage": "ranked_regression",
                }
                for n in range(20, 2001, 20)
                for t in (0.05, 0.1, 0.5)
            ],
            "utility_surrogate_selection": {
                "step": 20,
                "temperature": 0.05,
                "regret": 0.1,
                "context_regret": 0.1,
                "density_regret": 0.1,
                "component_mse": 0.1,
                "context_mse": 0.1,
                "density_mse": 0.1,
                "stage": "ranked_regression",
            },
            "utility_surrogate_selection_fingerprint": utility_surrogate_fingerprint(
                utility_surrogate, evidence.conditioning, 20, 0.05
            ),
        },
    }
    root = tmp_path_factory.mktemp("untrained-artifact") / "policy"
    save_artifact(root, utility_surrogate, policies, evidence.conditioning, deepcopy(metadata))
    return root, utility_surrogate, policies, evidence


def test_utility_surrogate_reuse_requires_identical_evidence_and_utility_surrogate_settings(untrained_artifact):
    from genode.gico.training import reuse_utility_surrogate

    root, original, _, evidence = untrained_artifact
    config = resolve_profile(evidence.task, backbone=evidence.backbone, dropout=0)
    utility_surrogate, metadata = reuse_utility_surrogate(root, evidence, replace(config, policy_context_mode="global"))
    assert all(not p.requires_grad for p in utility_surrogate.parameters()) and not utility_surrogate.training
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, utility_surrogate.state_dict()[key], atol=0, rtol=0)
    assert metadata["reused_artifact_sha256"] == hashlib.sha256((root / "policy.pt").read_bytes()).hexdigest()
    for altered in (replace(evidence, evidence_sha256="changed"), replace(evidence, purpose="research")):
        with pytest.raises(ValueError, match="differs from fitting evidence"):
            reuse_utility_surrogate(root, altered, config)
    for setting in (
        {"utility_surrogate_context_mode": "global"},
        {"seed": 1},
        {"utility_surrogate_learning_rate": 0.002},
    ):
        with pytest.raises(ValueError, match="differs"):
            reuse_utility_surrogate(root, evidence, replace(config, **setting))


def test_standalone_utility_surrogate_artifact_roundtrip_and_tampering(untrained_artifact, tmp_path):
    from genode.gico.policy import load_utility_surrogate

    root, original, _, _ = untrained_artifact
    payload = torch.load(root / "policy.pt", weights_only=True)
    metadata = deepcopy(payload["metadata"])
    metadata["history"]["policies"] = {}
    metadata["history"]["policy_selection"] = {}
    standalone = tmp_path / "surrogate"
    from genode.gico.conditioning import Conditioning

    payload_conditioning = Conditioning.from_payload(payload["utility_surrogate_conditioning"])
    save_artifact(standalone, original, {}, payload_conditioning, metadata)
    assert (standalone / "surrogate.pt").is_file()
    assert not (standalone / "policy.pt").exists()
    loaded, conditioning, selected = load_utility_surrogate(standalone)
    assert conditioning.to_payload() == payload_conditioning.to_payload()
    assert selected["selected_temperature"] == 0.05
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="only a utility surrogate"):
        load_policy(standalone)
    damaged = tmp_path / "surrogate-damaged"
    shutil.copytree(standalone, damaged)
    with (damaged / "surrogate.pt").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_utility_surrogate(damaged)
    wrong_grid = tmp_path / "surrogate-grid"
    shutil.copytree(standalone, wrong_grid)
    changed = torch.load(wrong_grid / "surrogate.pt", weights_only=True)
    key = next(iter(changed["metadata"]["reference_grids"]))
    changed["metadata"]["reference_grids"][key][1] += 0.01
    torch.save(changed, wrong_grid / "surrogate.pt")
    manifest = json.loads((wrong_grid / "manifest.json").read_text())
    manifest["artifact_sha256"] = hashlib.sha256((wrong_grid / "surrogate.pt").read_bytes()).hexdigest()
    (wrong_grid / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reference grid"):
        load_utility_surrogate(wrong_grid)


def test_utility_surrogate_checkpoint_uses_component_mse_before_regret(untrained_artifact, tmp_path):
    root, *_ = untrained_artifact
    still_selected = tmp_path / "lower-regret"

    def lower_regret(payload):
        for row in payload["metadata"]["history"]["utility_surrogate"]:
            if row["step"] == 40:
                row["regret"] = row["context_regret"] = row["density_regret"] = 0.0

    corrupt_payload(root, still_selected, lower_regret)
    assert load_policy(still_selected).metadata["history"]["utility_surrogate_selection"]["step"] == 20

    wrong_selection = tmp_path / "lower-mse"

    def lower_mse(payload):
        for row in payload["metadata"]["history"]["utility_surrogate"]:
            if row["step"] == 40:
                row["component_mse"] = row["context_mse"] = row["density_mse"] = 0.01

    corrupt_payload(root, wrong_selection, lower_mse)
    with pytest.raises(ValueError, match="MSE before temperature regret"):
        load_policy(wrong_selection)


@pytest.mark.parametrize("mode", ["native", "global"])
def test_reuse_bypasses_utility_surrogate_fitting_and_records_source(untrained_artifact, monkeypatch, mode):
    from genode.gico import training

    root, original, _, evidence = untrained_artifact
    before = (root / "policy.pt").read_bytes()
    config = resolve_profile(
        evidence.task,
        backbone=evidence.backbone,
        dropout=0,
        policy_context_mode=mode,
        policy_steps=2,
        policy_checkpoint_every=1,
        stochastic_likelihood_samples=1,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Reusing an artifact must skip utility_surrogate training")

    monkeypatch.setattr(training, "_fit_utility_surrogate", forbidden)
    # No local fitting: exercise target construction/selection with optimizer steps mocked.
    monkeypatch.setattr(training, "accumulated_step", lambda *args, **kwargs: 0.0)
    utility_surrogate, policies, history = training.fit_models(
        evidence,
        config,
        policy_kind="stochastic",
        device="cpu",
        utility_surrogate_artifact=root,
    )
    assert set(policies) == {"stochastic"}
    assert history["utility_surrogate_selection"]["temperature"] == 0.05
    assert history["policy_selection"]["stochastic"]["step"] == 2
    assert history["utility_surrogate_source_artifact_sha256"] == hashlib.sha256(before).hexdigest()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, utility_surrogate.state_dict()[key], atol=0, rtol=0)
    assert before == (root / "policy.pt").read_bytes()


@pytest.mark.parametrize("kind", ["deterministic", "stochastic"])
def test_standalone_artifact_roundtrip_without_loading_utility_surrogate(untrained_artifact, kind):
    root, _, policies, evidence = untrained_artifact
    condition = torch.tensor(evidence.conditioning.transform([1, 2], "euler", 4)[None])
    policies[kind].eval().requires_grad_(False)
    with torch.inference_mode():
        expected = (
            policies[kind](condition)
            if kind == "deterministic"
            else policies[kind].sample(condition, generator=clock_generator(31, "trajectory-6"))
        )
    with patch(
        "genode.gico.policy.UtilitySurrogate",
        side_effect=AssertionError("Inference must not instantiate utility_surrogate"),
    ):
        loaded = load_policy(root / "policy.pt", policy_kind=kind, expected_backbone=evidence.backbone)
        mass = loaded.density([1, 2], "euler", 4, seed=31, request_id="trajectory-6")
        grid = loaded.materialize([1, 2], "euler", 4, seed=31, request_id="trajectory-6")
    expected_mass = expected[0].numpy().astype(np.float64)
    expected_mass /= expected_mass.sum()
    np.testing.assert_array_equal(mass, expected_mass)
    assert grid == materialize(mass, "euler", 4)
    assert not hasattr(loaded, "utility_surrogate")
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
        "schema_version": GICO_SCHEMA_VERSION,
        "artifact_kind": "policy",
        "artifact_sha256": hashlib.sha256((destination / "policy.pt").read_bytes()).hexdigest(),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("kind", ["deterministic", "stochastic"])
def test_all_policy_loading_requires_selected_utility_surrogate_proof(untrained_artifact, tmp_path, kind):
    from genode.gico.policy import load_utility_surrogate

    root, *_ = untrained_artifact
    path = tmp_path / "old-utility_surrogate"
    corrupt_payload(root, path, lambda p: p["metadata"]["history"].pop("utility_surrogate_selection_fingerprint"))
    with pytest.raises(ValueError, match="selected-weight fingerprint"):
        load_policy(path, policy_kind=kind)
    with pytest.raises(ValueError, match="selected-weight fingerprint"):
        load_utility_surrogate(path)


@pytest.mark.parametrize("mutation", ["weights", "history", "clock_scope"])
def test_utility_surrogate_proof_and_clock_scope_reject_tampering(untrained_artifact, tmp_path, mutation):
    from genode.gico.policy import load_utility_surrogate

    root, *_ = untrained_artifact
    path = tmp_path / "tampered-utility_surrogate"

    def change(payload):
        if mutation == "weights":
            key = next(iter(payload["utility_surrogate"]))
            payload["utility_surrogate"][key].add_(0.1)
        elif mutation == "history":
            payload["metadata"]["history"]["utility_surrogate"].append({"step": 1, "temperature": 0.05, "regret": -1.0})
        else:
            payload["metadata"]["clock_scope"] = "per_horizon"

    corrupt_payload(root, path, change)
    with pytest.raises(ValueError):
        load_utility_surrogate(path)


@pytest.mark.parametrize("kind", ["GICO-det-policy", "GICO-sto-policy", "gico-deterministic"])
def test_old_public_selector_aliases_are_rejected(untrained_artifact, kind):
    root, *_ = untrained_artifact
    with pytest.raises(ValueError):
        load_policy(root, policy_kind=kind)


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
            parameter = next(iter(payload["policies"]["deterministic"].values()))
            parameter.view(-1)[0] = float("nan")

    destination = tmp_path / mutation
    corrupt_payload(root, destination, change)
    with pytest.raises(ValueError):
        load_policy(destination)


def test_nontrivial_ratio_normalizers_are_inverted_without_changing_density():
    mean = torch.linspace(-2, 3, 63)
    scale = torch.linspace(0.2, 2.5, 63)
    policy = StochasticPolicy(ModelConfig(5, 2), mean, scale)
    masses = torch.tensor(list(reference_densities("euler", 8).values())[5:8], dtype=torch.float64)
    log_mass = guarded_mass(masses).log()
    expected = (log_mass[:, :-1] - log_mass[:, -1:] - mean) / scale
    torch.testing.assert_close(policy.ratios(masses), expected)
    torch.testing.assert_close(policy.density(expected), guarded_mass(masses), atol=1e-12, rtol=1e-10)
    with pytest.raises(ValueError, match="positive"):
        StochasticPolicy(ModelConfig(5, 2), mean, torch.zeros(63))


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["metadata"]["fitting_profile"].update(utility_surrogate_density_normalization="running"),
        lambda p: p["metadata"]["fitting_profile"].update(utility_surrogate_context_mode="global"),
        lambda p: p["metadata"]["fitting_profile"].update(width=64),
        lambda p: p.update(architecture_protocol="wrong-rope"),
    ],
)
def test_utility_surrogate_loader_rejects_invalid_profile_or_normalizer(untrained_artifact, tmp_path, change):
    from genode.gico.policy import load_utility_surrogate

    root, _, _, _ = untrained_artifact
    damaged = tmp_path / "utility_surrogate-invalid"
    corrupt_payload(root, damaged, change)
    with pytest.raises(ValueError):
        load_utility_surrogate(damaged)


def test_v5_requires_archived_runtime(untrained_artifact, tmp_path):
    from genode.gico.policy import load_utility_surrogate

    root, *_ = untrained_artifact
    old = tmp_path / "v5"
    shutil.copytree(root, old)
    manifest_path = old / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol"] = "genode-gico-v5"
    manifest_path.write_text(json.dumps(manifest))
    for loader in (load_utility_surrogate, load_policy):
        with pytest.raises(ValueError, match="Unsupported artifact version"):
            loader(old)


def test_selected_measurements_bind_actual_policy_parameters(untrained_artifact, tmp_path):
    root, _, _, _ = untrained_artifact
    damaged = tmp_path / "state-mismatch"

    def change(payload):
        next(iter(payload["policies"]["deterministic"].values())).add_(0.01)

    corrupt_payload(root, damaged, change)
    with pytest.raises(ValueError, match="stored policy"):
        load_policy(damaged)


@pytest.mark.parametrize("kind", ("deterministic", "stochastic"))
def test_prompt_utility_surrogate_global_policy_artifact_roundtrip(untrained_artifact, tmp_path, kind):
    from dataclasses import replace

    from genode.gico.policy import load_utility_surrogate

    root, utility_surrogate, policies, evidence = untrained_artifact
    metadata = torch.load(root / "policy.pt", weights_only=True)["metadata"]
    metadata["fitting_profile"].update(policy_context_mode="global")
    metadata["history"].update(
        fixture_history(
            {kind: policies[kind]},
            evidence,
            *reference_evidence(),
            conditioning=replace(evidence.conditioning, context_mode="global"),
            utility_surrogate=utility_surrogate,
        )
    )
    path = tmp_path / "global"
    save_artifact(
        path, utility_surrogate, {kind: policies[kind]}, replace(evidence.conditioning, context_mode="global"), metadata
    )
    policy = load_policy(path, policy_kind=kind)
    np.testing.assert_array_equal(
        policy.density([1, 2], "euler", 4, request_id="same"), policy.density([9, -5], "euler", 4, request_id="same")
    )
    restored, conditioning, _ = load_utility_surrogate(path)
    assert conditioning.context_mode == "native"
    assert not np.array_equal(conditioning.transform([1, 2], "euler", 4), conditioning.transform([9, -5], "euler", 4))
    for key, value in utility_surrogate.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
