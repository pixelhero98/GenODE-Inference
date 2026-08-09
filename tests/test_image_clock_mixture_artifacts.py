from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from genode.artifacts.identity import canonical_json_bytes, semantic_sha256
from genode.backbones.protocol import ImageBackboneManifest
from genode.benchmarks.image.runtime import (
    GeneratedBatchProvenance,
    GeneratedImageBatch,
    ImageGICOClockRealizationBinding,
    ImageEulerSampler,
    ImageGenerationRequest,
    policy_schedule_request_hashes,
)
from genode.gico.image_clock_mixture import (
    build_image_gico_clock_library,
    image_clock_mixture_serialized_state_sha256,
)
from genode.gico.image_clock_mixture_artifacts import (
    BoundImageGICOClockMixtureArtifact,
    load_image_gico_clock_mixture_artifact,
    save_image_gico_clock_mixture_artifact,
)
from genode.gico.image_clock_mixture_training import (
    ImageGICOClockMixtureTrainingConfig,
    train_image_gico_clock_mixture,
)
from genode.gico.image_conditional import build_image_gico_feature_groups
from genode.gico.image_conditional_artifacts import (
    load_image_gico_conditional_artifact,
    save_image_gico_conditional_artifact,
)
from genode.gico.image_conditional_context import (
    prepare_image_gico_backbone_context,
)
from genode.gico.image_conditional_training import (
    ImageGICOBackboneContextTrainingConfig,
    train_image_gico_backbone_context,
)
from genode.provenance import file_sha256
from genode.schedules.policy import ScheduleBatch
from tests.test_image_clock_mixture import _targets
from tests.test_image_primary_runtime import _frozen_imagenet_backbone


def _request(
    *,
    artifact_sha256: str,
    backbone_manifest: ImageBackboneManifest,
    schedule: ScheduleBatch,
    label: str,
) -> ImageGenerationRequest:
    output_sha256, grid_sha256, execution_sha256, density_sha256 = (
        policy_schedule_request_hashes(schedule, preserve_batch=True)
    )
    return ImageGenerationRequest(
        source_request_sha256=semantic_sha256(
            {"request": label},
            namespace="test-image-clock-mixture-request",
        ),
        backbone_manifest=backbone_manifest,
        latent_seeds=(101, 103),
        class_labels=(0, 999),
        target_nfe=4,
        schedule_policy_sha256=artifact_sha256,
        schedule_output_sha256=output_sha256,
        time_grid_sha256=grid_sha256,
        execution_time_grid_sha256=execution_sha256,
        density_mass_sha256=density_sha256,
    )


def _provenance(generated: GeneratedImageBatch) -> GeneratedBatchProvenance:
    return GeneratedBatchProvenance(
        request=generated.request,
        noise_batch_sha256=generated.noise.batch_sha256,
        schedule=generated.schedule,
        field_evaluations=generated.field_evaluations,
        shape=tuple(generated.images.shape),
        content_sha256=generated.content_sha256,
        metric_shape=tuple(generated.metric_images.shape),
        metric_content_sha256=generated.metric_content_sha256,
    )


def test_complete_clock_sidecar_round_trip_is_source_bound_and_runtime_safe(
    tmp_path: Path,
) -> None:
    backbone = _frozen_imagenet_backbone(digest="5" * 64, offset=0.125)
    prepared = prepare_image_gico_backbone_context(backbone)
    feature_protocol_sha256 = "image-feature-protocol:" + "6" * 64
    feature_groups = build_image_gico_feature_groups(
        np.random.default_rng(4).normal(size=(1_000, 64)),
        source_panel_fingerprint="panel:" + "1" * 64,
        feature_protocol_sha256=feature_protocol_sha256,
        real_feature_panel_sha256="real:" + "3" * 64,
    )
    targets = replace(
        _targets(),
        feature_group_sha256=feature_groups.sha256,
        feature_protocol_sha256=feature_protocol_sha256,
        backbone_model_key=backbone.manifest.model_key,
        backbone_protocol_sha256=backbone.manifest.protocol_sha256,
        backbone_checkpoint_sha256=backbone.manifest.checkpoint.sha256,
    )
    library = build_image_gico_clock_library(targets)
    source_result = train_image_gico_backbone_context(
        targets,
        fixed_density_mass=library.supervision_density_mass.numpy(),
        normalized_context_table=prepared.normalized_context_table,
        context_binding_sha256=prepared.binding.binding_sha256,
        config=ImageGICOBackboneContextTrainingConfig(
            teacher_steps=1,
            student_steps=1,
            teacher_batch_size=8,
        ),
    )
    source_dir = tmp_path / "source-policy"
    source_paths = save_image_gico_conditional_artifact(
        source_dir,
        source_result,
        feature_groups,
        targets,
        prepared,
    )
    source_hashes = {
        role: file_sha256(path) for role, path in source_paths.items()
    }
    loaded_source = load_image_gico_conditional_artifact(source_dir)
    bound_source = loaded_source.bind(backbone)

    result = train_image_gico_clock_mixture(
        targets,
        library=library,
        normalized_context_table=prepared.normalized_context_table,
        context_binding_sha256=prepared.binding.binding_sha256,
        config=ImageGICOClockMixtureTrainingConfig(steps=1, seed=19),
    )
    decoder_dir = tmp_path / "clock-mixture"
    decoder_paths = save_image_gico_clock_mixture_artifact(
        decoder_dir,
        result,
        source_artifact=bound_source,
    )
    repeated_decoder_dir = tmp_path / "clock-mixture-repeated"
    repeated_decoder_paths = save_image_gico_clock_mixture_artifact(
        repeated_decoder_dir,
        result,
        source_artifact=bound_source,
    )
    assert {
        role: path.read_bytes() for role, path in decoder_paths.items()
    } == {
        role: path.read_bytes()
        for role, path in repeated_decoder_paths.items()
    }
    assert set(decoder_paths) == {
        "manifest",
        "model_state",
        "clock_library",
        "clock_grid_nfe_2",
        "clock_grid_nfe_4",
        "clock_grid_nfe_8",
    }
    assert not {
        "conditional-targets.json",
        "context-normalizer-mean.npy",
        "context-normalizer-scale.npy",
        "class-density-table.npy",
        "reward-feature-groups.json",
    }.intersection(path.name for path in decoder_dir.iterdir())
    assert source_hashes == {
        role: file_sha256(path) for role, path in source_paths.items()
    }
    manifest = json.loads(decoder_paths["manifest"].read_text(encoding="utf-8"))
    serialized_state = torch.load(
        decoder_paths["model_state"],
        map_location="cpu",
        weights_only=True,
    )
    assert manifest["serialized_model_state_sha256"] == (
        image_clock_mixture_serialized_state_sha256(serialized_state)
    )

    loaded = load_image_gico_clock_mixture_artifact(
        decoder_dir,
        source_artifact=loaded_source,
    )
    bound = loaded.bind(backbone)
    assert bound.library.sha256 == library.sha256
    for observed, expected in zip(
        bound.library.time_grids,
        library.time_grids,
        strict=True,
    ):
        assert torch.equal(observed, expected)
    with pytest.raises(TypeError, match="must be created by"):
        BoundImageGICOClockMixtureArtifact(
            source_artifact=bound.source_artifact,
            policy=bound.policy,
            library=bound.library,
            manifest=bound.manifest,
            _construction_token=object(),
        )

    labels = torch.tensor([0, 999], dtype=torch.int64)
    barycenter = bound.predict_for_class_labels(labels, target_nfe=4)
    barycenter_request = _request(
        artifact_sha256=bound.artifact_sha256,
        backbone_manifest=backbone.manifest,
        schedule=barycenter,
        label="barycenter",
    )
    sampler = ImageEulerSampler(backbone, device="cpu", execution_batch_size=2)
    generated_mean = sampler.sample_gico_clock_barycenter(
        barycenter_request,
        bound,
    )
    assert generated_mean.field_evaluations == 4
    assert generated_mean.schedule.source_kind == "contextual_schedule_policy"

    uniforms = torch.tensor([0.125, 0.875], dtype=torch.float64)
    realization = bound.realize_for_class_labels(
        labels,
        target_nfe=4,
        uniforms=uniforms,
        alpha=1.0,
    )
    realization_request = _request(
        artifact_sha256=bound.artifact_sha256,
        backbone_manifest=backbone.manifest,
        schedule=realization.schedule,
        label="complete-clock-realization",
    )
    generated_clock = sampler.sample_gico_clock_realization(
        realization_request,
        bound,
        uniforms=uniforms,
        alpha=1.0,
    )
    assert generated_clock.field_evaluations == 4
    assert generated_clock.schedule.source_kind == "contextual_schedule_policy"

    same_clock_uniforms_a = torch.zeros(2, dtype=torch.float64)
    same_clock_uniforms_b = torch.nextafter(
        same_clock_uniforms_a,
        torch.ones_like(same_clock_uniforms_a),
    )
    same_clock_realization_a = bound.realize_for_class_labels(
        labels,
        target_nfe=4,
        uniforms=same_clock_uniforms_a,
        alpha=1.0,
    )
    same_clock_realization_b = bound.realize_for_class_labels(
        labels,
        target_nfe=4,
        uniforms=same_clock_uniforms_b,
        alpha=1.0,
    )
    assert same_clock_realization_a.selected_schedule_indices == (0, 0)
    assert (
        same_clock_realization_b.selected_schedule_indices
        == same_clock_realization_a.selected_schedule_indices
    )
    assert same_clock_realization_a.schedule.sha256 == (
        same_clock_realization_b.schedule.sha256
    )
    assert same_clock_realization_a.sha256 != same_clock_realization_b.sha256
    same_clock_request_a = _request(
        artifact_sha256=bound.artifact_sha256,
        backbone_manifest=backbone.manifest,
        schedule=same_clock_realization_a.schedule,
        label="same-clock-different-uniforms",
    )
    same_clock_request_b = _request(
        artifact_sha256=bound.artifact_sha256,
        backbone_manifest=backbone.manifest,
        schedule=same_clock_realization_b.schedule,
        label="same-clock-different-uniforms",
    )
    assert same_clock_request_a.request_sha256 == same_clock_request_b.request_sha256
    generated_same_clock_a = sampler.sample_gico_clock_realization(
        same_clock_request_a,
        bound,
        uniforms=same_clock_uniforms_a,
        alpha=1.0,
    )
    generated_same_clock_b = sampler.sample_gico_clock_realization(
        same_clock_request_b,
        bound,
        uniforms=same_clock_uniforms_b,
        alpha=1.0,
    )
    assert torch.equal(
        generated_same_clock_a.images,
        generated_same_clock_b.images,
    )
    realization_binding_a = generated_same_clock_a.request.clock_realization
    realization_binding_b = generated_same_clock_b.request.clock_realization
    assert realization_binding_a is not None
    assert realization_binding_b is not None
    assert generated_same_clock_a.schedule.clock_realization == realization_binding_a
    assert generated_same_clock_b.schedule.clock_realization == realization_binding_b
    assert realization_binding_a.realization_sha256 == same_clock_realization_a.sha256
    assert realization_binding_a.alpha == same_clock_realization_a.alpha
    assert realization_binding_a.probability_sha256 == (
        same_clock_realization_a.probability_sha256
    )
    assert realization_binding_a.uniforms_sha256 == (
        same_clock_realization_a.uniforms_sha256
    )
    assert realization_binding_a.selected_schedule_indices == (
        same_clock_realization_a.selected_schedule_indices
    )
    assert realization_binding_a.selected_schedule_keys == (
        same_clock_realization_a.selected_schedule_keys
    )
    assert realization_binding_a != realization_binding_b
    assert generated_same_clock_a.request.request_sha256 != (
        generated_same_clock_b.request.request_sha256
    )
    assert generated_same_clock_a.schedule.binding_sha256 != (
        generated_same_clock_b.schedule.binding_sha256
    )
    assert generated_same_clock_a.batch_sha256 != generated_same_clock_b.batch_sha256

    generated_payload = generated_same_clock_a.as_payload()
    generated_round_trip = GeneratedImageBatch.from_payload(
        generated_payload,
        images=generated_same_clock_a.images,
    )
    assert generated_round_trip.batch_sha256 == generated_same_clock_a.batch_sha256
    assert generated_round_trip.request.clock_realization == realization_binding_a
    request_identity = generated_payload["request"]["identity"]
    persisted_binding = ImageGICOClockRealizationBinding.from_payload(
        request_identity["clock_realization"]
    )
    assert persisted_binding == realization_binding_a
    integer_alpha_payload = dict(realization_binding_a.as_payload())
    integer_alpha_payload["alpha"] = 1
    with pytest.raises(TypeError, match="JSON floating-point"):
        ImageGICOClockRealizationBinding.from_payload(integer_alpha_payload)
    assert semantic_sha256(
        persisted_binding.realization_identity_payload(),
        namespace="image-gico-clock-realization-v1",
    ) == same_clock_realization_a.sha256
    provenance_a = _provenance(generated_same_clock_a)
    provenance_b = _provenance(generated_same_clock_b)
    assert provenance_a.provenance_sha256 != provenance_b.provenance_sha256
    provenance_round_trip = GeneratedBatchProvenance.from_payload(
        provenance_a.as_payload()
    )
    assert provenance_round_trip.provenance_sha256 == provenance_a.provenance_sha256
    assert provenance_round_trip.request.clock_realization == realization_binding_a

    replayed_same_clock_a = sampler.sample_gico_clock_realization(
        generated_round_trip.request,
        bound,
        uniforms=same_clock_uniforms_a,
        alpha=1.0,
    )
    assert replayed_same_clock_a.batch_sha256 == generated_same_clock_a.batch_sha256
    with pytest.raises(ValueError, match="different clock realization"):
        sampler.sample_gico_clock_realization(
            generated_round_trip.request,
            bound,
            uniforms=same_clock_uniforms_b,
            alpha=1.0,
        )

    zero_realization = bound.realize_for_class_labels(
        labels,
        target_nfe=4,
        uniforms=uniforms,
        alpha=0.0,
    )
    zero_request = _request(
        artifact_sha256=bound.artifact_sha256,
        backbone_manifest=backbone.manifest,
        schedule=zero_realization.schedule,
        label="zero-alpha-barycenter",
    )
    generated_zero = sampler.sample_gico_clock_realization(
        zero_request,
        bound,
        uniforms=uniforms,
        alpha=0.0,
    )
    assert torch.equal(generated_zero.images, generated_mean.images)

    original = bound.policy.model.global_logits_by_nfe[0, 0].item()
    with torch.no_grad():
        bound.policy.model.global_logits_by_nfe[0, 0] = original + 0.25
    with pytest.raises(ValueError, match="modified|identity changed"):
        bound.verify_execution_identity()
    with torch.no_grad():
        bound.policy.model.global_logits_by_nfe[0, 0] = original
    bound.verify_execution_identity()

    resigned_state_dir = tmp_path / "clock-mixture-resigned-state"
    shutil.copytree(decoder_dir, resigned_state_dir)
    resigned_state_path = resigned_state_dir / "clock-mixture-state.pt"
    resigned_state = torch.load(
        resigned_state_path,
        map_location="cpu",
        weights_only=True,
    )
    resigned_state["global_logits_by_nfe"] = resigned_state[
        "global_logits_by_nfe"
    ].clone()
    resigned_state["global_logits_by_nfe"][0, 0] += 0.125
    with resigned_state_path.open("wb") as handle:
        torch.save(resigned_state, handle)
    resigned_manifest_path = resigned_state_dir / "manifest.json"
    resigned_manifest = json.loads(
        resigned_manifest_path.read_text(encoding="utf-8")
    )
    resigned_manifest["files"]["model_state"] = {
        "filename": resigned_state_path.name,
        "sha256": file_sha256(resigned_state_path),
        "size_bytes": resigned_state_path.stat().st_size,
    }
    resigned_body = dict(resigned_manifest)
    resigned_body.pop("artifact_sha256")
    resigned_manifest["artifact_sha256"] = semantic_sha256(
        resigned_body,
        namespace="image-gico-complete-clock-mixture-artifact-v1",
    )
    resigned_manifest_path.write_bytes(canonical_json_bytes(resigned_manifest))
    with pytest.raises(ValueError, match="Serialized|serialized"):
        load_image_gico_clock_mixture_artifact(
            resigned_state_dir,
            source_artifact=loaded_source,
        )

    resigned_training_dir = tmp_path / "clock-mixture-resigned-training"
    shutil.copytree(decoder_dir, resigned_training_dir)
    resigned_training_manifest_path = resigned_training_dir / "manifest.json"
    resigned_training_manifest = json.loads(
        resigned_training_manifest_path.read_text(encoding="utf-8")
    )
    training = resigned_training_manifest["training"]
    training["training_config"]["steps"] = 0
    training["training_config_sha256"] = semantic_sha256(
        training["training_config"],
        namespace="image-gico-complete-clock-mixture-training-config-v1",
    )
    training_body = dict(training)
    training_body.pop("result_sha256")
    training["result_sha256"] = semantic_sha256(
        training_body,
        namespace="image-gico-complete-clock-mixture-training-result-v1",
    )
    artifact_body = dict(resigned_training_manifest)
    artifact_body.pop("artifact_sha256")
    resigned_training_manifest["artifact_sha256"] = semantic_sha256(
        artifact_body,
        namespace="image-gico-complete-clock-mixture-artifact-v1",
    )
    resigned_training_manifest_path.write_bytes(
        canonical_json_bytes(resigned_training_manifest)
    )
    with pytest.raises((TypeError, ValueError), match="steps|positive"):
        load_image_gico_clock_mixture_artifact(
            resigned_training_dir,
            source_artifact=loaded_source,
        )

    decoder_paths["clock_library"].write_bytes(
        decoder_paths["clock_library"].read_bytes() + b"\n"
    )
    with pytest.raises(ValueError, match="canonical JSON|inconsistent"):
        load_image_gico_clock_mixture_artifact(
            decoder_dir,
            source_artifact=loaded_source,
        )
