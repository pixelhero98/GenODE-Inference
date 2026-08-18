from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from genode.artifacts.identity import semantic_sha256
from genode.backbones import (
    IMAGE_BACKBONE_REGISTRY,
    CanonicalNoiseToDataAdapter,
    CheckpointBinding,
    ImageBackboneManifest,
    build_image_backbone_manifest,
    load_verified_image_backbone,
)
from genode.backbones import loading as backbone_loading
from genode.benchmarks.image.protocol import IMAGE_SCHEDULE_KEYS
from genode.benchmarks.image.runtime import (
    ImageEulerSampler,
    ImageGenerationRequest,
    policy_schedule_request_hashes,
)
from genode.gico.image_conditional import (
    build_image_gico_conditional_targets,
    build_image_gico_feature_groups,
)
from genode.gico.image_conditional_artifacts import (
    BoundImageGICOConditionalArtifact,
    load_image_gico_conditional_artifact,
    save_image_gico_conditional_artifact,
)
from genode.gico.image_conditional_context import prepare_image_gico_backbone_context
from genode.gico.image_conditional_training import (
    ImageGICOBackboneContextTrainingConfig,
    train_image_gico_backbone_context,
)
from genode.gico.image_students import (
    load_image_gico_deterministic_artifact,
    materialize_image_gico_schedule,
    save_image_gico_deterministic_artifact,
    train_image_gico_deterministic_student,
)
from genode.gico.image_supervision import build_image_gico_conditional_supervision
from genode.schedules.policy import ScheduleBatch


class DhariwalUNet(nn.Module):
    def __init__(self, *, offset: float = 0.0) -> None:
        super().__init__()
        self.map_label = nn.Linear(1_000, 768, bias=False)
        with torch.no_grad():
            values = torch.arange(768_000, dtype=torch.float32).reshape(768, 1_000)
            self.map_label.weight.copy_(values / 768_000.0 + float(offset))


class EDMPrecondVel(nn.Module):
    def __init__(self, *, offset: float = 0.0) -> None:
        super().__init__()
        self.model = DhariwalUNet(offset=offset)

    def forward(
        self,
        state: torch.Tensor,
        native_time: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del native_time, class_labels, kwargs
        return torch.zeros_like(state)


def _frozen_imagenet_backbone(*, digest: str, offset: float) -> CanonicalNoiseToDataAdapter:
    adapter = CanonicalNoiseToDataAdapter(
        EDMPrecondVel(offset=offset),
        ImageBackboneManifest(
            model_key="imagenet64_rfpp_config_e",
            checkpoint=CheckpointBinding(
                filename="imagenet-configE.pth",
                sha256=digest,
                size_bytes=1,
            ),
        ),
    )
    adapter.eval()
    adapter.requires_grad_(False)
    return adapter


def _frozen_cifar_backbone(*, digest: str) -> CanonicalNoiseToDataAdapter:
    adapter = CanonicalNoiseToDataAdapter(
        EDMPrecondVel(),
        ImageBackboneManifest(
            model_key="cifar10_rfpp_config_g",
            checkpoint=CheckpointBinding(
                filename="cifar-configG.pth",
                sha256=digest,
                size_bytes=1,
            ),
        ),
    )
    adapter.eval()
    adapter.requires_grad_(False)
    return adapter


class _IdentifiedUniformPolicy:
    policy_sha256 = semantic_sha256(
        {"policy": "unconditional-uniform"},
        namespace="test-image-policy",
    )

    def predict(
        self,
        context: torch.Tensor,
        *,
        target_nfe: int,
    ) -> ScheduleBatch:
        density = torch.full(
            (int(context.shape[0]), 64),
            1.0 / 64.0,
            dtype=torch.float32,
        )
        return ScheduleBatch.from_density_mass(
            density,
            target_nfe=target_nfe,
        )


def test_all_four_backbones_bind_and_load_with_explicit_conditioning_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert set(IMAGE_BACKBONE_REGISTRY) == {
        "cifar10_rfpp_config_g",
        "imagenet64_rfpp_config_e",
        "cifar10_edm_ve_as_1rf",
        "imagenet64_edm_ve_as_1rf",
    }
    monkeypatch.setattr(
        backbone_loading,
        "verify_user_supplied_rfpp_source_root",
        lambda source_root, spec, timeout: Path(source_root).resolve(),
    )

    for model_key, spec in IMAGE_BACKBONE_REGISTRY.items():
        checkpoint = tmp_path / spec.checkpoint_filename
        checkpoint.write_bytes(f"frozen-{model_key}".encode("ascii"))
        manifest = build_image_backbone_manifest(model_key, checkpoint)
        loaded = load_verified_image_backbone(
            manifest,
            checkpoint_path=checkpoint,
            source_root=tmp_path / "verified-source",
            factory=lambda *, source_root, checkpoint_path, spec: EDMPrecondVel(),
        )
        assert loaded.manifest == manifest
        assert not loaded.training
        assert not any(parameter.requires_grad for parameter in loaded.parameters())

        state = torch.zeros((2, *spec.image_shape), dtype=torch.float32)
        progress = torch.zeros(2, dtype=torch.float32)
        if spec.dataset_key == "cifar10":
            assert spec.conditioning == "unconditional"
            assert spec.num_conditioning_classes == 0
            assert torch.equal(loaded(state, progress), state)
            with pytest.raises(ValueError, match="unconditional"):
                loaded(state, progress, torch.tensor([0, 1]))
            with pytest.raises(ValueError, match="ImageNet-64"):
                loaded.encode_conditioning(torch.tensor([0, 1]))
        else:
            assert spec.conditioning == "class_conditional"
            assert spec.num_conditioning_classes == 1_000
            labels = torch.tensor([0, 999], dtype=torch.int64)
            assert torch.equal(loaded(state, progress, labels), state)
            one_hot = torch.nn.functional.one_hot(labels, num_classes=1_000).float()
            with torch.no_grad():
                native_context = loaded.native_model.model.map_label(one_hot)
            assert torch.equal(loaded.encode_conditioning(labels), native_context)


def test_unconditional_cifar_executes_a_content_identified_policy() -> None:
    backbone = _frozen_cifar_backbone(digest="2" * 64)
    policy = _IdentifiedUniformPolicy()
    context = torch.zeros((2, 1), dtype=torch.float32)
    schedule = policy.predict(context, target_nfe=2)
    output_hash, grid_hash, execution_hash, mass_hash = policy_schedule_request_hashes(schedule)
    request = ImageGenerationRequest(
        source_request_sha256=semantic_sha256(
            {"request": "unconditional-policy"},
            namespace="test-image-request",
        ),
        backbone_manifest=backbone.manifest,
        latent_seeds=(11, 13),
        class_labels=None,
        target_nfe=2,
        schedule_policy_sha256=policy.policy_sha256,
        schedule_output_sha256=output_hash,
        time_grid_sha256=grid_hash,
        execution_time_grid_sha256=execution_hash,
        density_mass_sha256=mass_hash,
    )

    generated = ImageEulerSampler(backbone, device="cpu").sample_policy(
        request,
        policy,
        context=context,
    )

    assert generated.field_evaluations == 2
    assert generated.request.class_labels is None
    assert generated.schedule.schedule_policy_sha256 == policy.policy_sha256


def test_imagenet_teacher_student_artifact_round_trip_and_euler_evaluation(
    tmp_path: Path,
) -> None:
    schedule_count = len(IMAGE_SCHEDULE_KEYS)
    assert schedule_count == 23
    backbone = _frozen_imagenet_backbone(digest="3" * 64, offset=0.125)
    prepared = prepare_image_gico_backbone_context(backbone)
    assert prepared.normalized_context_table.shape == (1_000, 768)

    groups = build_image_gico_feature_groups(
        np.random.default_rng(4).normal(size=(1_000, 64)),
        source_panel_fingerprint="panel:" + "1" * 64,
        feature_protocol_sha256="image-feature-protocol:" + "2" * 64,
        real_feature_panel_sha256="real:" + "3" * 64,
    )
    generator = np.random.default_rng(5)
    class_kid = generator.normal(size=(3, schedule_count, 1_000)).astype(np.float32)
    jackknife = np.repeat(class_kid[..., None], 64, axis=-1)
    jackknife += np.linspace(-0.01, 0.01, 64, dtype=np.float32)[None, None, None, :]
    masses = generator.uniform(size=(3, schedule_count, 64)).astype(np.float64)
    masses /= masses.sum(axis=-1, keepdims=True, dtype=np.float64)
    schedule_sha256s = tuple(
        semantic_sha256({"schedule_key": key}, namespace="test-image-schedule") for key in IMAGE_SCHEDULE_KEYS
    )
    density_mass_sha256s = tuple(
        tuple(
            semantic_sha256(masses[nfe_index, schedule_index].tolist(), namespace="test-image-density")
            for schedule_index in range(schedule_count)
        )
        for nfe_index in range(3)
    )
    targets = build_image_gico_conditional_targets(
        class_kid=class_kid,
        jackknife_class_kid=jackknife,
        reward_scales=np.asarray((0.25, 0.5, 1.0), dtype=np.float32),
        fixed_density_mass=masses,
        schedule_keys=IMAGE_SCHEDULE_KEYS,
        schedule_sha256s=schedule_sha256s,
        density_mass_sha256s=density_mass_sha256s,
        feature_groups=groups,
        reward_evidence_sha256=semantic_sha256("evidence", namespace="test-image-reward-evidence"),
        fixed_support_sha256=semantic_sha256(masses.tolist(), namespace="test-image-fixed-support"),
        backbone_model_key=backbone.manifest.model_key,
        backbone_protocol_sha256=backbone.manifest.protocol_sha256,
        backbone_checkpoint_sha256=backbone.manifest.checkpoint.sha256,
        feature_protocol_sha256=groups.feature_protocol_sha256,
    )
    supervision = build_image_gico_conditional_supervision(
        targets=targets,
        fixed_density_mass=masses,
        normalized_contexts=prepared.normalized_context_table,
    )
    unified_result = train_image_gico_deterministic_student(
        supervision,
        config=ImageGICOBackboneContextTrainingConfig(
            teacher_steps=1,
            student_steps=1,
            teacher_batch_size=8,
        ),
    )
    unified_dir = tmp_path / "unified-deterministic-policy"
    unified_manifest = save_image_gico_deterministic_artifact(
        unified_result,
        supervision,
        unified_dir,
    )
    unified_artifact = load_image_gico_deterministic_artifact(
        unified_dir,
        expected_artifact_sha256=unified_manifest["artifact_sha256"],
    )
    materialized = materialize_image_gico_schedule(
        "deterministic_barycenter",
        deterministic_artifact=unified_artifact,
        causal_artifact=None,
        target_nfe=4,
        context_indices=[0, 999],
    )
    assert materialized.density_mass.shape == (2, 64)
    assert materialized.time_grids.shape == (2, 5)
    assert materialized.artifact_sha256 == unified_manifest["artifact_sha256"]
    assert materialized.supervision_sha256 == supervision.sha256

    result = train_image_gico_backbone_context(
        targets,
        fixed_density_mass=masses,
        normalized_context_table=prepared.normalized_context_table,
        context_binding_sha256=prepared.binding.binding_sha256,
        config=ImageGICOBackboneContextTrainingConfig(
            teacher_steps=1,
            student_steps=2,
            teacher_batch_size=8,
        ),
    )
    policy_dir = tmp_path / "policy"
    paths = save_image_gico_conditional_artifact(
        policy_dir,
        result,
        groups,
        targets,
        prepared,
    )
    assert paths["teacher_state"].is_file()
    assert paths["student_state"].is_file()

    portable = load_image_gico_conditional_artifact(policy_dir)
    bound = portable.bind(backbone)
    with pytest.raises(TypeError, match="must be created by"):
        BoundImageGICOConditionalArtifact(
            policy=bound.policy,
            prepared_context=bound.prepared_context,
            feature_groups=bound.feature_groups,
            targets=bound.targets,
            manifest=bound.manifest,
            _construction_token=object(),
        )
    labels = torch.tensor([0, 999], dtype=torch.int64)
    contexts = bound.contexts_for_class_labels(labels)
    raw_contexts = backbone.encode_conditioning(labels).cpu().numpy()
    expected_contexts = torch.from_numpy(
        np.ascontiguousarray(
            (raw_contexts - prepared.normalizer.mean[None, :]) / prepared.normalizer.scale[None, :],
            dtype=np.float32,
        )
    )
    assert torch.equal(contexts, expected_contexts)

    schedule = bound.policy.predict(contexts, target_nfe=4)
    output_hash, grid_hash, execution_hash, mass_hash = policy_schedule_request_hashes(
        schedule,
        preserve_batch=True,
    )
    request = ImageGenerationRequest(
        source_request_sha256=semantic_sha256(
            {"request": "backbone-context-policy"},
            namespace="test-image-request",
        ),
        backbone_manifest=backbone.manifest,
        latent_seeds=(101, 103),
        class_labels=(0, 999),
        target_nfe=4,
        schedule_policy_sha256=portable.artifact_sha256,
        schedule_output_sha256=output_hash,
        time_grid_sha256=grid_hash,
        execution_time_grid_sha256=execution_hash,
        density_mass_sha256=mass_hash,
    )
    generated = ImageEulerSampler(
        backbone,
        device="cpu",
        execution_batch_size=2,
    ).sample_gico(request, bound)
    assert generated.field_evaluations == 4
    assert generated.schedule.source_kind == "contextual_schedule_policy"
    assert torch.equal(generated.images, generated.noise.values)
    assert generated.request.class_labels == (0, 999)

    original_weight = bound.policy.model.global_logits_by_nfe[0, 0].item()
    with torch.no_grad():
        bound.policy.model.global_logits_by_nfe[0, 0] = original_weight + 0.25
    with pytest.raises(ValueError, match="student state was modified"):
        ImageEulerSampler(backbone, device="cpu").sample_gico(request, bound)
    with torch.no_grad():
        bound.policy.model.global_logits_by_nfe[0, 0] = original_weight

    original_context = bound.policy.model.canonical_context_table[0, 0].item()
    with torch.no_grad():
        bound.policy.model.canonical_context_table[0, 0] = original_context + 0.25
    with pytest.raises(ValueError, match="context table was modified"):
        ImageEulerSampler(backbone, device="cpu").sample_gico(request, bound)
    with torch.no_grad():
        bound.policy.model.canonical_context_table[0, 0] = original_context

    with pytest.raises(ValueError, match="must use sample_gico"):
        ImageEulerSampler(backbone, device="cpu").sample_policy(
            request,
            bound.policy,
            context=bound.contexts_for_class_labels(torch.tensor([999, 0])),
        )

    mismatched_identity_request = ImageGenerationRequest(
        source_request_sha256=request.source_request_sha256,
        backbone_manifest=request.backbone_manifest,
        latent_seeds=request.latent_seeds,
        class_labels=request.class_labels,
        target_nfe=request.target_nfe,
        schedule_policy_sha256="9" * 64,
        schedule_output_sha256=request.schedule_output_sha256,
        time_grid_sha256=request.time_grid_sha256,
        execution_time_grid_sha256=request.execution_time_grid_sha256,
        density_mass_sha256=request.density_mass_sha256,
    )
    with pytest.raises(ValueError, match="artifact identity"):
        ImageEulerSampler(backbone, device="cpu").sample_gico(
            mismatched_identity_request,
            bound,
        )
