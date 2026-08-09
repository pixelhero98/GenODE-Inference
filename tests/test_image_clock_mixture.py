from __future__ import annotations

from functools import lru_cache
import math
import threading

import numpy as np
import pytest
import torch

from genode.benchmarks.image.protocol import IMAGE_SCHEDULE_KEYS, IMAGE_TARGET_NFES
from genode.gico.image_clock_mixture import (
    ImageGICOBackboneContextClockMixtureModel,
    ImageGICOClockMixtureModelConfig,
    ImageGICOClockMixturePolicy,
    _open_unit_interval_float64,
    build_image_gico_clock_library,
    build_image_gico_clock_library_from_time_grids,
    derive_image_gico_clock_uniforms,
    image_gico_clock_sample_key,
    image_gico_clock_uniform,
)
from genode.gico.image_clock_mixture_training import (
    ImageGICOClockMixtureTrainingConfig,
    _aggregate_groups,
    _deterministic_training_scope,
    _grouped_cross_entropy,
    train_image_gico_clock_mixture,
)
from genode.gico.image_conditional import ImageGICOConditionalTargets
from genode.schedules import build_default_fixed_schedules
from genode.schedules.density import density_mass_to_time_grid


def _tuple3(array: np.ndarray) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(
        tuple(tuple(float(value) for value in row) for row in plane)
        for plane in array
    )


def _tuple4(
    array: np.ndarray,
) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]:
    return tuple(
        tuple(
            tuple(tuple(float(value) for value in row) for row in matrix)
            for matrix in plane
        )
        for plane in array
    )


@lru_cache(maxsize=1)
def _targets() -> ImageGICOConditionalTargets:
    schedules = tuple(
        build_default_fixed_schedules(target_nfe, dtype=torch.float64)
        for target_nfe in IMAGE_TARGET_NFES
    )
    supervision = np.stack(
        tuple(
            torch.stack(tuple(schedule.density_mass for schedule in row)).numpy()
            for row in schedules
        )
    )
    class_coordinate = np.linspace(-1.0, 1.0, 1_000, dtype=np.float64)
    schedule_coordinate = np.linspace(-0.75, 0.75, len(IMAGE_SCHEDULE_KEYS), dtype=np.float64)
    logits = np.empty((3, 1_000, len(IMAGE_SCHEDULE_KEYS)), dtype=np.float64)
    for nfe_index in range(3):
        logits[nfe_index] = (
            0.17 * float(nfe_index + 1) * schedule_coordinate[None, :]
            + class_coordinate[:, None]
            * np.cos(np.arange(len(IMAGE_SCHEDULE_KEYS), dtype=np.float64))[None, :]
        )
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)
    density = np.einsum("ncs,nsb->ncb", weights, supervision)
    zeros = np.zeros_like(weights)
    coefficients = np.zeros((*weights.shape, 3), dtype=np.float64)
    coefficients[..., 2] = 1.0
    return ImageGICOConditionalTargets(
        density_mass=_tuple3(density),
        mixture_weights=_tuple3(weights),
        normalized_rewards=_tuple3(logits),
        jackknife_standard_errors=_tuple3(zeros),
        class_reliability=_tuple3(zeros),
        group_reliability=_tuple3(zeros),
        shrinkage_coefficients=_tuple4(coefficients),
        temperature_by_nfe=(0.25, 0.5, 1.0),
        target_nfes=tuple(IMAGE_TARGET_NFES),
        schedule_keys=tuple(IMAGE_SCHEDULE_KEYS),
        schedule_sha256s=tuple(
            schedule.specification.sha256 for schedule in schedules[0]
        ),
        density_mass_sha256s=tuple(
            tuple(schedule.density_mass_sha256 for schedule in row)
            for row in schedules
        ),
        feature_group_sha256="image-gico-feature-groups:" + "1" * 64,
        reward_evidence_sha256="image-gico-reward-evidence:" + "2" * 64,
        fixed_support_sha256="image-gico-fixed-support:" + "3" * 64,
        backbone_model_key="imagenet64_rfpp_config_e",
        backbone_protocol_sha256="4" * 64,
        backbone_checkpoint_sha256="5" * 64,
        feature_protocol_sha256="image-feature-protocol:" + "6" * 64,
        density_bin_count=64,
    )


@lru_cache(maxsize=1)
def _library():
    return build_image_gico_clock_library(_targets())


def _context_table() -> torch.Tensor:
    table = torch.zeros((1_000, 768), dtype=torch.float32)
    table[:, 0] = torch.linspace(-2.0, 2.0, 1_000)
    table[:, 1] = torch.linspace(1.0, -1.0, 1_000)
    return table


def _conditional_policy() -> ImageGICOClockMixturePolicy:
    library = _library()
    model = ImageGICOBackboneContextClockMixtureModel(
        ImageGICOClockMixtureModelConfig.for_library(
            library,
            nfe_embedding_dim=2,
            hidden_dim=4,
        ),
        _context_table(),
        library,
    )
    with torch.no_grad():
        model.nfe_embedding.weight.zero_()
        first = model.context_network[0]
        second = model.context_network[2]
        final = model.context_network[4]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0] = 1.0
        second.weight.zero_()
        second.bias.zero_()
        second.weight[0, 0] = 1.0
        final.weight.zero_()
        final.bias.zero_()
        final.weight[0, 0] = 1.5
        final.weight[1, 0] = -1.5
        base = torch.linspace(-0.3, 0.3, library.schedule_count)
        model.global_logits_by_nfe.copy_(base.repeat(3, 1))
    model.eval()
    return ImageGICOClockMixturePolicy(
        model,
        library,
        context_binding_sha256="image-gico-context-binding:" + "7" * 64,
    )


def test_canonical_clock_library_is_exact_complete_and_target_bound() -> None:
    library = _library()
    assert library.schedule_keys == tuple(IMAGE_SCHEDULE_KEYS)
    assert library.schedule_count == 23
    assert library.group_counts == (23, 23, 23)
    assert library.density_bin_count == 170
    assert library.reference_time_grid.shape == (171,)
    assert torch.allclose(
        library.reference_time_grid,
        1.0 - torch.flip(library.reference_time_grid, dims=(0,)),
        rtol=0.0,
        atol=2e-16,
    )

    for nfe_index, target_nfe in enumerate(IMAGE_TARGET_NFES):
        reconstructed = density_mass_to_time_grid(
            library.density_mass[nfe_index],
            target_nfe=target_nfe,
            reference_time_grid=library.reference_time_grid,
        )
        assert torch.allclose(
            reconstructed,
            library.time_grids[nfe_index],
            rtol=0.0,
            atol=2e-14,
        )
        for node in library.time_grids[nfe_index].reshape(-1):
            assert bool(torch.any(library.reference_time_grid == node))

    for key_index, key in enumerate(library.schedule_keys):
        if not key.endswith("_reversed"):
            continue
        base_index = library.schedule_keys.index(key.removesuffix("_reversed"))
        for nfe_index in range(3):
            assert torch.allclose(
                library.density_mass[nfe_index, key_index],
                torch.flip(library.density_mass[nfe_index, base_index], dims=(0,)),
                rtol=0.0,
                atol=2e-14,
            )


def test_clock_policy_is_conditional_and_alpha_endpoints_are_replayable() -> None:
    policy = _conditional_policy()
    contexts = policy.model.canonical_context_table[torch.tensor([0, 999])]
    target_nfe = 4
    nfe_index = IMAGE_TARGET_NFES.index(target_nfe)
    probabilities = policy.model.schedule_probabilities(
        contexts,
        torch.full((2,), target_nfe, dtype=torch.int64),
    )
    assert not torch.allclose(probabilities[0], probabilities[1])

    low_high = torch.tensor(
        [0.0, torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()],
        dtype=torch.float64,
    )
    before_rng = torch.random.get_rng_state().clone()
    mean_a = policy.sample_realization(
        contexts,
        target_nfe=target_nfe,
        uniforms=low_high,
        alpha=0.0,
    )
    mean_b = policy.sample_realization(
        contexts,
        target_nfe=target_nfe,
        uniforms=torch.tensor([0.25, 0.75], dtype=torch.float64),
        alpha=0.0,
    )
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert mean_a.sha256 == mean_b.sha256
    assert mean_a.uniforms_sha256 is None
    assert mean_a.selected_schedule_indices == (None, None)
    assert torch.allclose(
        mean_a.schedule.density_mass,
        policy.predict(contexts, target_nfe=target_nfe).density_mass,
        rtol=0.0,
        atol=0.0,
    )

    complete = policy.sample_realization(
        contexts,
        target_nfe=target_nfe,
        uniforms=low_high,
        alpha=1.0,
    )
    assert complete.selected_schedule_indices[0] == 0
    assert complete.selected_schedule_indices[1] == policy.library.schedule_count - 1
    for row, selected in enumerate(complete.selected_schedule_indices):
        assert selected is not None
        assert torch.equal(
            complete.schedule.density_mass[row],
            policy.library.density_mass[nfe_index, selected],
        )
        assert torch.allclose(
            complete.schedule.time_grid[row],
            policy.library.time_grids[nfe_index][selected],
            rtol=0.0,
            atol=2e-14,
        )

    partial = policy.sample_realization(
        contexts,
        target_nfe=target_nfe,
        uniforms=low_high,
        alpha=0.25,
    )
    selected_density = torch.stack(
        tuple(
            policy.library.density_mass[nfe_index, index]
            for index in complete.selected_schedule_indices
            if index is not None
        )
    )
    expected = mean_a.schedule.density_mass + 0.25 * (
        selected_density - mean_a.schedule.density_mass
    )
    assert torch.allclose(partial.schedule.density_mass, expected, rtol=1e-13, atol=1e-13)
    full_distance = torch.abs(selected_density - mean_a.schedule.density_mass).sum(dim=-1)
    partial_distance = torch.abs(partial.schedule.density_mass - mean_a.schedule.density_mass).sum(dim=-1)
    assert torch.allclose(partial_distance, 0.25 * full_distance, rtol=1e-12, atol=1e-12)


def test_clock_library_and_execution_buffer_mutation_changes_identity() -> None:
    original = _library()
    restored = build_image_gico_clock_library_from_time_grids(
        _targets(),
        tuple(table.clone() for table in original.time_grids),
    )
    assert restored.sha256 == original.sha256
    model = ImageGICOBackboneContextClockMixtureModel(
        ImageGICOClockMixtureModelConfig.for_library(
            restored,
            nfe_embedding_dim=2,
            hidden_dim=4,
        ),
        _context_table(),
        restored,
    )
    state_before = model.state_sha256
    with torch.no_grad():
        model._group_expand[0, 0, 0] += 0.125
    assert model.state_sha256 != state_before

    library_before = restored.sha256
    row = restored.density_mass[0, 0]
    positive = torch.nonzero(row > 1e-5, as_tuple=False).flatten()
    assert positive.numel() >= 2
    with torch.no_grad():
        row[positive[0]] += 1e-6
        row[positive[1]] -= 1e-6
    assert restored.sha256 != library_before
    policy = ImageGICOClockMixturePolicy(
        model,
        restored,
        context_binding_sha256="image-gico-context-binding:" + "9" * 64,
    )
    with pytest.raises(ValueError, match="library identity changed"):
        policy.predict(model.canonical_context_table[:1], target_nfe=2)


def test_clock_sampling_uses_left_closed_cdf_boundaries() -> None:
    policy = _conditional_policy()
    context = policy.model.canonical_context_table[:1]
    probabilities = policy.model.schedule_probabilities(
        context,
        torch.tensor([2], dtype=torch.int64),
    ).to(dtype=torch.float64)
    probabilities /= probabilities.sum(dim=-1, keepdim=True)
    boundary = torch.cumsum(probabilities, dim=-1)[0, 0]
    realization = policy.sample_realization(
        context,
        target_nfe=2,
        uniforms=boundary.reshape(1).contiguous(),
        alpha=1.0,
    )
    assert realization.selected_schedule_indices == (0,)


def test_clock_mixture_expectation_and_covariance_scale_with_alpha() -> None:
    policy = _conditional_policy()
    context = policy.model.canonical_context_table[123:124]
    target_nfe = 8
    nfe_index = IMAGE_TARGET_NFES.index(target_nfe)
    probabilities = policy.model.schedule_probabilities(
        context,
        torch.tensor([target_nfe], dtype=torch.int64),
    )[0].to(dtype=torch.float64)
    probabilities /= probabilities.sum()
    bank = policy.library.density_mass[nfe_index]
    mean = probabilities @ bank
    deviations = bank - mean
    covariance = torch.einsum("s,sb,sc->bc", probabilities, deviations, deviations)
    alpha = 0.4
    interpolated = mean + alpha * deviations
    observed_mean = probabilities @ interpolated
    observed_covariance = torch.einsum(
        "s,sb,sc->bc",
        probabilities,
        interpolated - observed_mean,
        interpolated - observed_mean,
    )
    assert torch.allclose(observed_mean, mean, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        observed_covariance,
        alpha**2 * covariance,
        rtol=1e-11,
        atol=1e-13,
    )


def test_grouped_loss_is_invariant_to_duplicate_alias_probability() -> None:
    probabilities = torch.tensor(
        [[[0.10, 0.20, 0.30, 0.40], [0.25, 0.25, 0.20, 0.30]]],
        dtype=torch.float64,
    )
    groups = (((0, 1), (2,), (3,)),)
    aggregated = _aggregate_groups(probabilities[0], groups[0])
    expected = torch.tensor(
        [[0.30, 0.30, 0.40], [0.50, 0.20, 0.30]],
        dtype=torch.float64,
    )
    assert torch.allclose(aggregated, expected)
    target = (expected.clone(),)
    original = _grouped_cross_entropy(probabilities, target, groups)
    swapped = probabilities.clone()
    swapped[..., 0], swapped[..., 1] = (
        probabilities[..., 1],
        probabilities[..., 0],
    )
    assert torch.equal(original, _grouped_cross_entropy(swapped, target, groups))


def test_clock_policy_rejects_implicit_or_invalid_randomness() -> None:
    policy = _conditional_policy()
    context = policy.model.canonical_context_table[:1]
    with pytest.raises(TypeError, match="CPU torch.float64"):
        policy.sample_realization(
            context,
            target_nfe=2,
            uniforms=torch.tensor([0.5], dtype=torch.float32),
            alpha=1.0,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        policy.sample_realization(
            context,
            target_nfe=2,
            uniforms=torch.tensor([1.0], dtype=torch.float64),
            alpha=1.0,
        )
    with pytest.raises(TypeError, match="alpha"):
        policy.sample_realization(
            context,
            target_nfe=2,
            uniforms=torch.tensor([0.5], dtype=torch.float64),
            alpha=True,
        )
    with pytest.raises(TypeError, match="integer dtype"):
        policy.model.schedule_probabilities(
            context,
            torch.tensor([2 + 0j], dtype=torch.complex64),
        )


def test_clock_counter_rng_has_pinned_content_bound_vectors() -> None:
    key = image_gico_clock_sample_key(
        sampling_plan_sha256="plan:" + "1" * 64,
        policy_artifact_sha256="artifact:" + "2" * 64,
        clock_library_sha256="library:" + "3" * 64,
        context_binding_sha256="context:" + "4" * 64,
        root_seed=20_260_808,
        latent_seed=2_100_000_123,
        class_label=123,
        target_nfe=4,
    )
    assert key == (
        "image-gico-complete-clock-sample-key-v2:"
        "b2ad5d2e7c903cefd8fac8caab029f948959439c881d434f626b6f05f30a6d58"
    )
    assert image_gico_clock_uniform(key) == 0.2224714600266346
    assert image_gico_clock_uniform(key, draw_index=1) == 0.2769573325576015
    assert _open_unit_interval_float64(0) > 0.0
    assert _open_unit_interval_float64(2**52 - 1) < 1.0
    before = torch.random.get_rng_state().clone()
    values = derive_image_gico_clock_uniforms((key, key), draw_index=1)
    assert torch.equal(before, torch.random.get_rng_state())
    assert values.dtype == torch.float64
    assert values.device.type == "cpu"
    assert values.is_contiguous()
    assert values.tolist() == [0.2769573325576015, 0.2769573325576015]


def test_complete_clock_training_scope_serializes_process_global_state() -> None:
    device = torch.device("cpu")
    torch.manual_seed(320)
    rng_before = torch.random.get_rng_state().clone()
    global_state_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    second_started = threading.Event()
    second_entered = threading.Event()
    release_second = threading.Event()
    thread_errors: list[Exception] = []

    def run_second_scope() -> None:
        try:
            second_started.set()
            with (
                _deterministic_training_scope(device),
                torch.random.fork_rng(devices=[]),
            ):
                if not torch.are_deterministic_algorithms_enabled():
                    raise AssertionError(
                        "Deterministic algorithms were disabled inside the training scope."
                    )
                second_entered.set()
                if not release_second.wait(timeout=5.0):
                    raise TimeoutError(
                        "Timed out waiting to leave the second training scope."
                    )
                torch.random.default_generator.manual_seed(202)
        except Exception as exc:
            thread_errors.append(exc)

    worker = threading.Thread(
        target=run_second_scope,
        name="image-gico-determinism-regression",
        daemon=True,
    )
    scopes_overlapped = False
    try:
        with (
            _deterministic_training_scope(device),
            torch.random.fork_rng(devices=[]),
        ):
            torch.random.default_generator.manual_seed(101)
            worker.start()
            assert second_started.wait(timeout=5.0)
            scopes_overlapped = second_entered.wait(timeout=2.0)
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert torch.get_float32_matmul_precision() == "highest"
    finally:
        release_second.set()
        if worker.ident is not None:
            worker.join(timeout=5.0)

    assert not scopes_overlapped
    assert second_entered.is_set()
    assert not worker.is_alive()
    assert not thread_errors
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert global_state_before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


def test_complete_clock_training_is_rng_isolated_and_reproducible() -> None:
    targets = _targets()
    library = _library()
    contexts = _context_table()
    config = ImageGICOClockMixtureTrainingConfig(steps=1, seed=19)
    torch.manual_seed(321)
    before = torch.random.get_rng_state().clone()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    precision_before = torch.get_float32_matmul_precision()
    first = train_image_gico_clock_mixture(
        targets,
        library=library,
        normalized_context_table=contexts,
        context_binding_sha256="image-gico-context-binding:" + "8" * 64,
        config=config,
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.are_deterministic_algorithms_enabled() is deterministic_before
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only_before
    assert torch.get_float32_matmul_precision() == precision_before
    second = train_image_gico_clock_mixture(
        targets,
        library=library,
        normalized_context_table=contexts,
        context_binding_sha256="image-gico-context-binding:" + "8" * 64,
        config=config,
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert first.model_state_sha256 == second.model_state_sha256
    assert first.sha256 == second.sha256
    assert not first.model.training
    assert first.target_reconstruction_max_abs_error < 1e-12
    assert first.group_count_by_nfe == (23, 23, 23)
    assert first.final_kl >= 0.0
    assert math.isfinite(first.final_barycenter_l1)
    assert first.identity_payload()["determinism"] == {
        "protocol": "image_gico_clock_mixture_determinism_v1",
        "rng_isolation": "torch.random.fork_rng",
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "float32_matmul_precision": "highest",
        "cuda_cublas_workspace_config": None,
        "cuda_tf32_enabled": None,
        "cudnn_benchmark": None,
        "cudnn_deterministic": None,
        "optimizer_foreach": False,
    }
    frozen_state_sha256 = first.model_state_sha256
    with torch.no_grad():
        first.model.global_logits_by_nfe[0, 0] += 0.25
    assert first.model_state_sha256 == frozen_state_sha256
    assert first.model.state_sha256 != frozen_state_sha256
