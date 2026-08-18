from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from genode.benchmarks.image.protocol import IMAGE_SCHEDULE_KEYS
from genode.gico.image_conditional import (
    build_image_gico_conditional_targets,
    build_image_gico_feature_groups,
)
from genode.gico.image_conditional_context import prepare_image_gico_backbone_context
from genode.gico.image_supervision import (
    IMAGE_GICO_CONTEXT_DIM,
    IMAGE_GICO_TARGET_NFES,
    build_image_gico_conditional_supervision,
    build_image_gico_unconditional_supervision,
    load_image_gico_supervision,
    save_image_gico_supervision,
)
from tests.test_image_primary_runtime import _frozen_imagenet_backbone


def _support_and_weights() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.arange(64, dtype=np.float64) + 0.5
    uniform = np.full(64, 1.0 / 64.0, dtype=np.float64)
    sine = 1.0 + 0.15 * np.sin(2.0 * np.pi * coordinates / 64.0)
    cosine = 1.0 + 0.15 * np.cos(2.0 * np.pi * coordinates / 64.0)
    sine /= sine.sum(dtype=np.float64)
    cosine /= cosine.sum(dtype=np.float64)
    # The repeated sine schedule is a deliberate density alias.
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


def _unconditional_supervision():
    support, weights = _support_and_weights()
    return build_image_gico_unconditional_supervision(
        target_nfes=IMAGE_GICO_TARGET_NFES,
        schedule_keys=("uniform", "sine-a", "sine-b", "cosine"),
        fixed_density_mass=support,
        mixture_weights=weights,
        source_identities={"mixture_evidence": "a" * 64},
    )


def test_unconditional_supervision_is_an_explicit_singleton_zero_context() -> None:
    supervision = _unconditional_supervision()

    assert supervision.supervision_kind == "unconditional_mixture"
    assert supervision.normalized_contexts.shape == (1, IMAGE_GICO_CONTEXT_DIM)
    assert np.array_equal(supervision.normalized_contexts, np.zeros((1, IMAGE_GICO_CONTEXT_DIM)))
    assert supervision.normalized_rewards is None
    assert supervision.reward_diagnostics["synthetic_class_labels"] is False


def test_unconditional_supervision_requires_authenticated_source_identity() -> None:
    support, weights = _support_and_weights()
    with pytest.raises(ValueError, match="SHA-256 identity"):
        build_image_gico_unconditional_supervision(
            target_nfes=IMAGE_GICO_TARGET_NFES,
            schedule_keys=("uniform", "sine-a", "sine-b", "cosine"),
            fixed_density_mass=support,
            mixture_weights=weights,
            source_identities={"mixture_evidence": "not-authenticated"},
        )


def test_shared_law_preserves_mass_and_constructs_the_exact_barycenter() -> None:
    supervision = _unconditional_supervision()
    expected = np.einsum(
        "ncs,nsb->ncb",
        supervision.mixture_weights,
        supervision.fixed_density_mass,
        dtype=np.float64,
    )

    assert np.allclose(supervision.fixed_density_mass.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(supervision.mixture_weights.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(supervision.barycenter_density_mass.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(supervision.barycenter_density_mass, expected, rtol=0.0, atol=2e-12)
    # The aliases retain separate source weights in the shared supervision law.
    assert np.array_equal(supervision.fixed_density_mass[:, 1], supervision.fixed_density_mass[:, 2])
    assert np.all(supervision.mixture_weights[:, 0, 1:3].sum(axis=-1) > 0.0)


def test_supervision_round_trip_is_identity_preserving_and_no_replace(tmp_path: Path) -> None:
    supervision = _unconditional_supervision()
    output = tmp_path / "supervision"
    manifest = save_image_gico_supervision(supervision, output)
    loaded = load_image_gico_supervision(output)

    assert manifest["supervision_sha256"] == supervision.sha256
    assert loaded.sha256 == supervision.sha256
    assert np.array_equal(loaded.fixed_density_mass, supervision.fixed_density_mass)
    assert np.array_equal(loaded.mixture_weights, supervision.mixture_weights)
    assert np.array_equal(loaded.barycenter_density_mass, supervision.barycenter_density_mass)
    with pytest.raises(FileExistsError):
        save_image_gico_supervision(supervision, output)


def test_supervision_loader_rejects_array_tampering(tmp_path: Path) -> None:
    output = tmp_path / "supervision"
    save_image_gico_supervision(_unconditional_supervision(), output)
    array_path = output / "mixture-weights.npy"
    contents = bytearray(array_path.read_bytes())
    contents[-1] ^= 1
    array_path.write_bytes(contents)

    with pytest.raises(ValueError, match="hash changed"):
        load_image_gico_supervision(output)


def test_supervision_save_rejects_post_construction_law_mutation(tmp_path: Path) -> None:
    supervision = _unconditional_supervision()
    construction_identity = supervision.sha256
    supervision.mixture_weights.setflags(write=True)
    first = float(supervision.mixture_weights[0, 0, 0])
    last = float(supervision.mixture_weights[0, 0, -1])
    supervision.mixture_weights[0, 0, 0] = last
    supervision.mixture_weights[0, 0, -1] = first

    assert supervision.sha256 == construction_identity
    with pytest.raises(ValueError, match="scientific law was mutated"):
        supervision.verify()
    output = tmp_path / "mutated-supervision"
    with pytest.raises(ValueError, match="scientific law was mutated"):
        save_image_gico_supervision(supervision, output)
    assert not output.exists()


def test_supervision_rejects_consistent_law_rebinding() -> None:
    supervision = _unconditional_supervision()
    supervision.mixture_weights.setflags(write=True)
    first = float(supervision.mixture_weights[0, 0, 0])
    last = float(supervision.mixture_weights[0, 0, -1])
    supervision.mixture_weights[0, 0, 0] = last
    supervision.mixture_weights[0, 0, -1] = first
    supervision.barycenter_density_mass.setflags(write=True)
    supervision.barycenter_density_mass[...] = np.einsum(
        "ncs,nsb->ncb",
        supervision.mixture_weights,
        supervision.fixed_density_mass,
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="construction identity was mutated"):
        supervision.verify()


def test_conditional_supervision_enforces_reward_alias_and_barycenter_law() -> None:
    schedule_count = len(IMAGE_SCHEDULE_KEYS)
    backbone = _frozen_imagenet_backbone(digest="3" * 64, offset=0.125)
    prepared = prepare_image_gico_backbone_context(backbone)
    groups = build_image_gico_feature_groups(
        np.random.default_rng(41).normal(size=(1_000, 64)),
        source_panel_fingerprint="panel:" + "1" * 64,
        feature_protocol_sha256="image-feature-protocol:" + "2" * 64,
        real_feature_panel_sha256="real:" + "3" * 64,
    )
    generator = np.random.default_rng(42)
    class_kid = generator.normal(size=(3, schedule_count, 1_000)).astype(np.float32)
    jackknife = np.repeat(class_kid[..., None], 64, axis=-1)
    jackknife += np.linspace(-0.01, 0.01, 64, dtype=np.float32)[None, None, None, :]
    support = generator.uniform(size=(3, schedule_count, 64)).astype(np.float64)
    support /= support.sum(axis=-1, keepdims=True)
    support[:, 2] = support[:, 1]
    density_hashes = tuple(
        tuple(
            f"density:{(nfe_index * schedule_count + (1 if schedule_index == 2 else schedule_index)):064x}"
            for schedule_index in range(schedule_count)
        )
        for nfe_index in range(3)
    )
    targets = build_image_gico_conditional_targets(
        class_kid=class_kid,
        jackknife_class_kid=jackknife,
        reward_scales=np.asarray((0.25, 0.5, 1.0), dtype=np.float32),
        fixed_density_mass=support,
        schedule_keys=IMAGE_SCHEDULE_KEYS,
        schedule_sha256s=tuple(f"schedule:{index:064x}" for index in range(schedule_count)),
        density_mass_sha256s=density_hashes,
        feature_groups=groups,
        reward_evidence_sha256="evidence:" + "4" * 64,
        fixed_support_sha256="support:" + "5" * 64,
        backbone_model_key=backbone.manifest.model_key,
        backbone_protocol_sha256=backbone.manifest.protocol_sha256,
        backbone_checkpoint_sha256=backbone.manifest.checkpoint.sha256,
        feature_protocol_sha256=groups.feature_protocol_sha256,
    )
    supervision = build_image_gico_conditional_supervision(
        targets=targets,
        fixed_density_mass=support,
        normalized_contexts=prepared.normalized_context_table,
    )

    assert np.array_equal(supervision.mixture_weights, np.asarray(targets.mixture_weights))
    assert np.array_equal(supervision.normalized_rewards, np.asarray(targets.normalized_rewards))
    assert np.allclose(
        supervision.barycenter_density_mass,
        np.einsum("ncs,nsb->ncb", supervision.mixture_weights, support),
        rtol=0.0,
        atol=2e-12,
    )
    assert np.array_equal(supervision.mixture_weights[:, :, 1], supervision.mixture_weights[:, :, 2])

    bad_rewards = np.asarray(targets.normalized_rewards, dtype=np.float64).copy()
    bad_rewards[0, 0, 0] = 6.0
    clipped_target = replace(
        targets,
        normalized_rewards=tuple(tuple(tuple(row) for row in nfe) for nfe in bad_rewards),
    )
    with pytest.raises(ValueError, match="clipped"):
        build_image_gico_conditional_supervision(
            targets=clipped_target,
            fixed_density_mass=support,
            normalized_contexts=prepared.normalized_context_table,
        )

    bad_hashes = [list(row) for row in density_hashes]
    bad_hashes[0][2] = "density:" + "f" * 64
    alias_evasion = replace(targets, density_mass_sha256s=tuple(tuple(row) for row in bad_hashes))
    with pytest.raises(ValueError, match="aliases disagree"):
        build_image_gico_conditional_supervision(
            targets=alias_evasion,
            fixed_density_mass=support,
            normalized_contexts=prepared.normalized_context_table,
        )
