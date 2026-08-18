from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from genode.gico.image_causal_artifacts import (
    image_gico_causal_state_sha256,
    load_image_gico_causal_artifact,
    save_image_gico_causal_artifact,
)
from genode.gico.image_causal_policy import (
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    ImageGICOCausalTransformer,
)
from genode.gico.image_causal_rng import (
    derive_image_gico_causal_uniforms,
    image_gico_causal_uniforms_sha256,
)
from genode.gico.image_causal_stick import (
    DENSITY_BIN_COUNT,
    MAXIMUM_CLOCK_NODE_DRIFT,
    STICK_ACTION_COUNT,
    ImageGICOCausalPathBank,
    inverse_cdf_clock_nodes,
)
from genode.gico.image_causal_training import (
    ImageGICOCausalTrainingConfig,
    path_log_probs_from_teacher_forced_logits,
    terminal_weighted_path_nll,
    train_image_gico_causal_student,
)
from genode.gico.image_students import (
    ImageGICOScheduleMaterialization,
    execute_image_gico_euler,
    load_image_gico_deterministic_artifact,
    materialize_image_gico_schedule,
    save_image_gico_deterministic_artifact,
    train_image_gico_deterministic_student,
)
from genode.gico.image_supervision import (
    IMAGE_GICO_TARGET_NFES,
    build_image_gico_unconditional_supervision,
)
from genode.gico.image_training_rng import resolve_image_gico_training_device


def _support_and_weights() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.arange(64, dtype=np.float64) + 0.5
    uniform = np.full(64, 1.0 / 64.0, dtype=np.float64)
    sine = 1.0 + 0.15 * np.sin(2.0 * np.pi * coordinates / 64.0)
    cosine = 1.0 + 0.15 * np.cos(2.0 * np.pi * coordinates / 64.0)
    sine /= sine.sum(dtype=np.float64)
    cosine /= cosine.sum(dtype=np.float64)
    support = np.stack([[uniform, sine, sine, cosine]] * 3)
    weights = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.1, 0.1, 0.4],
        ],
        dtype=np.float64,
    )
    return support, weights


def _supervision():
    support, weights = _support_and_weights()
    return build_image_gico_unconditional_supervision(
        target_nfes=IMAGE_GICO_TARGET_NFES,
        schedule_keys=("uniform", "sine-a", "sine-b", "cosine"),
        fixed_density_mass=support,
        mixture_weights=weights,
        source_identities={"mixture_evidence": "b" * 64},
    )


def _path_bank() -> ImageGICOCausalPathBank:
    support, _ = _support_and_weights()
    return ImageGICOCausalPathBank.build(support, np.linspace(0.0, 1.0, 65, dtype=np.float64))


def test_quantizer_uses_portable_invariants_and_preserves_alias_mass() -> None:
    bank = _path_bank()
    supervision = _supervision()
    aggregated = bank.aggregate_teacher_weights(supervision.mixture_weights)

    assert bank.token_paths.shape == (3, 4, STICK_ACTION_COUNT)
    assert np.allclose(bank.decoded_density_paths.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12)
    assert bank.diagnostics.maximum_observed_clock_node_drift < MAXIMUM_CLOCK_NODE_DRIFT
    assert all(paths.shape == (3, STICK_ACTION_COUNT) for paths in bank.unique_token_paths_by_nfe)
    assert all(len(trie.children_by_prefix) > 0 for trie in bank.tries)
    assert np.array_equal(bank.token_paths[:, 1], bank.token_paths[:, 2])
    for nfe_index, row in enumerate(aggregated):
        alias_index = int(bank.schedule_to_alias_index[nfe_index, 1])
        assert alias_index == int(bank.schedule_to_alias_index[nfe_index, 2])
        expected_alias_mass = supervision.mixture_weights[nfe_index, 0, 1:3].sum()
        assert row[0, alias_index] == pytest.approx(expected_alias_mass, abs=1e-15)
        assert row.sum() == pytest.approx(1.0, abs=1e-12)


def test_path_bank_rejects_noncanonical_density_coordinates() -> None:
    support, _ = _support_and_weights()
    squared = np.linspace(0.0, 1.0, 65, dtype=np.float64) ** 2
    with pytest.raises(ValueError, match="protocol-fixed uniform"):
        ImageGICOCausalPathBank.build(support, squared)


def test_prefix_trie_allows_only_causal_children_and_complete_support() -> None:
    bank = _path_bank()
    path = tuple(int(value) for value in bank.unique_token_paths_by_nfe[1][0])
    trie = bank.tries[1]

    for step, token in enumerate(path):
        children = trie.valid_children(path[:step])
        assert token in children
        mask = trie.child_mask(path[:step])
        assert mask.dtype == np.bool_
        assert bool(mask[token])
        assert int(mask.sum()) == len(children)
    assert trie.valid_children(path) == ()
    assert trie.alias_members(path)
    unsupported_root = next(token for token in range(256) if token not in trie.valid_children(()))
    with pytest.raises(KeyError, match="outside the frozen teacher support"):
        trie.valid_children((unsupported_root,))


def test_causal_transformer_protocol_parameter_count() -> None:
    model = ImageGICOCausalTransformer()

    assert model.trainable_parameter_count == EXPECTED_TRAINABLE_PARAMETER_COUNT == 339_184


def test_counter_rng_replays_and_binds_artifact_request_and_sample_keys() -> None:
    arguments = {
        "artifact_sha256": "a" * 64,
        "request_sha256": "b" * 64,
        "target_nfe": 4,
        "sample_keys": ["image-0", "image-1"],
    }
    first = derive_image_gico_causal_uniforms(**arguments)
    replay = derive_image_gico_causal_uniforms(**arguments)
    changed_request = derive_image_gico_causal_uniforms(**{**arguments, "request_sha256": "c" * 64})
    reordered = derive_image_gico_causal_uniforms(**{**arguments, "sample_keys": ["image-1", "image-0"]})

    assert torch.equal(first, replay)
    assert not torch.equal(first, changed_request)
    assert torch.equal(first[0], reordered[1])
    assert torch.equal(first[1], reordered[0])
    assert bool(torch.all((first > 0.0) & (first < 1.0)))
    assert image_gico_causal_uniforms_sha256(first) == image_gico_causal_uniforms_sha256(replay)


def test_training_rng_scope_selects_only_the_execution_device(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_image_gico_training_device(torch.device("cpu")) == (torch.device("cpu"), [])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert resolve_image_gico_training_device(torch.device("cuda:2")) == (
        torch.device("cuda:2"),
        [2],
    )
    assert resolve_image_gico_training_device(torch.device("cuda")) == (
        torch.device("cuda:3"),
        [3],
    )


def test_terminal_weighted_complete_path_objective() -> None:
    logits = torch.zeros((2, STICK_ACTION_COUNT, 256), dtype=torch.float32)
    tokens = torch.zeros((2, STICK_ACTION_COUNT), dtype=torch.int64)
    path_log_probs = path_log_probs_from_teacher_forced_logits(logits, tokens)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    loss = terminal_weighted_path_nll(path_log_probs[None], weights)

    assert path_log_probs.shape == (2,)
    assert loss.item() == pytest.approx(np.log(256.0), rel=1e-6)


def test_one_step_training_artifact_strict_load_materialization_and_exact_nfe(tmp_path: Path) -> None:
    supervision = _supervision()
    training_rng_state = torch.random.get_rng_state().clone()
    result = train_image_gico_causal_student(
        supervision,
        device="cpu",
        config=ImageGICOCausalTrainingConfig(updates=1, batch_size=2, seed=17),
    )
    assert torch.equal(torch.random.get_rng_state(), training_rng_state)
    assert result.report.completed_updates == 1
    assert result.report.trainable_parameter_count == EXPECTED_TRAINABLE_PARAMETER_COUNT
    assert result.report.model_state_sha256 == image_gico_causal_state_sha256(result.model)
    assert np.isfinite(result.report.final_batch_nll)

    state = {name: tensor.detach().clone() for name, tensor in result.model.state_dict().items()}
    with torch.no_grad():
        result.model.global_logits[0, 0, 0].add_(1.0)
    mutated_dir = tmp_path / "mutated-causal-artifact"
    with pytest.raises(ValueError, match="current final model state"):
        save_image_gico_causal_artifact(result, supervision, mutated_dir)
    assert not mutated_dir.exists()
    result.model.load_state_dict(state, strict=True)

    artifact_dir = tmp_path / "causal-artifact"
    manifest = save_image_gico_causal_artifact(result, supervision, artifact_dir)
    rng_state = torch.random.get_rng_state().clone()
    loaded = load_image_gico_causal_artifact(
        artifact_dir,
        expected_artifact_sha256=manifest["artifact_sha256"],
    )
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    uniforms = derive_image_gico_causal_uniforms(
        artifact_sha256=loaded.artifact_sha256,
        request_sha256="d" * 64,
        target_nfe=4,
        sample_keys=["sample-0"],
    )
    materialized = materialize_image_gico_schedule(
        "stochastic_causal_ar",
        deterministic_artifact=None,
        causal_artifact=loaded,
        target_nfe=4,
        context_indices=[0],
        uniforms=uniforms,
    )
    assert materialized.time_grids.shape == (1, 5)
    assert materialized.tokens is not None
    assert loaded.path_bank.tries[1].alias_members(tuple(int(value) for value in materialized.tokens[0]))

    calls = 0

    def field(state: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        assert progress.shape == (1,)
        return torch.ones_like(state)

    execution = execute_image_gico_euler(field, torch.zeros((1, 3), dtype=torch.float64), materialized)
    assert calls == execution.field_evaluations == execution.target_nfe == 4
    assert torch.allclose(execution.final_state, torch.ones_like(execution.final_state))

    def grid_mutating_field(state: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        progress.add_(1e-3)
        return torch.ones_like(state)

    with pytest.raises(RuntimeError, match="changed the frozen materialization grid"):
        execute_image_gico_euler(
            grid_mutating_field,
            torch.zeros((1, 3), dtype=torch.float64),
            materialized,
        )

    materialized.time_grids.setflags(write=True)
    materialized.time_grids[0, 1] += 1e-3
    with pytest.raises(ValueError, match="materialization was mutated"):
        execute_image_gico_euler(field, torch.zeros((1, 3), dtype=torch.float64), materialized)

    mask_mutated = load_image_gico_causal_artifact(artifact_dir)
    mask_mutated.model._causal_mask.zero_()
    with pytest.raises(ValueError, match="attention mask was mutated"):
        mask_mutated.verify()

    loaded.path_bank.canonical_density_paths.setflags(write=True)
    loaded.path_bank.canonical_density_paths[0, 0, 0] += 1e-6
    with pytest.raises(ValueError, match="path bank was mutated"):
        loaded.verify()

    state_mutated = load_image_gico_causal_artifact(artifact_dir)
    with torch.no_grad():
        state_mutated.model.global_logits[0, 0, 0].add_(1.0)
    with pytest.raises(ValueError, match="model state was mutated"):
        state_mutated.verify()

    context_mutated = load_image_gico_causal_artifact(artifact_dir)
    context_mutated.normalized_contexts.setflags(write=True)
    context_mutated.normalized_contexts[0, 0] = 1.0
    with pytest.raises(ValueError, match="deployment contexts were mutated"):
        context_mutated.verify()

    (artifact_dir / "unexpected.txt").write_text("not part of the artifact", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or unexpected"):
        load_image_gico_causal_artifact(artifact_dir, supervision)


def test_materialization_binds_density_grid_and_lineage() -> None:
    density = np.full((1, DENSITY_BIN_COUNT), 1.0 / DENSITY_BIN_COUNT, dtype=np.float64)
    reference = np.linspace(0.0, 1.0, DENSITY_BIN_COUNT + 1, dtype=np.float64)
    grid = inverse_cdf_clock_nodes(density, 4, reference)
    with pytest.raises(ValueError, match="artifact_sha256"):
        ImageGICOScheduleMaterialization(
            student_kind="deterministic_barycenter",
            target_nfe=4,
            context_indices=(0,),
            density_mass=density,
            time_grids=grid,
            artifact_sha256="not-a-hash",
            supervision_sha256="b" * 64,
        )

    mismatched = np.array(grid, copy=True)
    mismatched[0, 1] += 1e-3
    with pytest.raises(ValueError, match="canonical inverse-CDF"):
        ImageGICOScheduleMaterialization(
            student_kind="deterministic_barycenter",
            target_nfe=4,
            context_indices=(0,),
            density_mass=density,
            time_grids=mismatched,
            artifact_sha256="a" * 64,
            supervision_sha256="b" * 64,
        )

    materialization = ImageGICOScheduleMaterialization(
        student_kind="deterministic_barycenter",
        target_nfe=4,
        context_indices=(0,),
        density_mass=density,
        time_grids=grid,
        artifact_sha256="a" * 64,
        supervision_sha256="b" * 64,
    )
    object.__setattr__(materialization, "artifact_sha256", "c" * 64)
    with pytest.raises(ValueError, match="materialization was mutated"):
        materialization.verify()


def test_direct_deterministic_artifact_loads_and_materializes_without_supervision(tmp_path: Path) -> None:
    supervision = _supervision()
    result = train_image_gico_deterministic_student(supervision)
    artifact_dir = tmp_path / "deterministic-artifact"
    manifest = save_image_gico_deterministic_artifact(result, supervision, artifact_dir)

    loaded = load_image_gico_deterministic_artifact(
        artifact_dir,
        expected_artifact_sha256=manifest["artifact_sha256"],
    )
    materialized = materialize_image_gico_schedule(
        "deterministic_barycenter",
        deterministic_artifact=loaded,
        causal_artifact=None,
        target_nfe=8,
        context_indices=[0],
    )

    assert loaded.supervision_sha256 == supervision.sha256
    assert materialized.supervision_sha256 == supervision.sha256
    assert np.array_equal(materialized.density_mass, supervision.barycenter_density_mass[2])
    assert materialized.time_grids.shape == (1, 9)

    context_mutated = load_image_gico_deterministic_artifact(artifact_dir)
    context_mutated.normalized_contexts.setflags(write=True)
    context_mutated.normalized_contexts[0, 0] = 1.0
    with pytest.raises(ValueError, match="deployment contexts were mutated"):
        context_mutated.verify()

    barycenter_mutated = load_image_gico_deterministic_artifact(artifact_dir)
    assert barycenter_mutated.direct_barycenter_density_mass is not None
    barycenter_mutated.direct_barycenter_density_mass.setflags(write=True)
    barycenter_mutated.direct_barycenter_density_mass[0, 0, 0] += 1e-3
    with pytest.raises(ValueError, match="direct barycenter was mutated"):
        barycenter_mutated.verify()
