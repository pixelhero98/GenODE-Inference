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
from genode.benchmarks.image.runtime import (
    ImageEulerSampler,
    ImageGenerationRequest,
    policy_schedule_request_hashes,
)
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


class _CommonPolicyFixture:
    artifact_sha256 = "a" * 64

    def __init__(self, backbone, student_kind="GICO-sto-policy"):
        from genode.backbones.registry import get_image_backbone_spec
        from genode.gico.image_conditional_context import native_contexts

        _, binding = native_contexts(backbone)
        self.metadata = {
            "task": get_image_backbone_spec(backbone.manifest.model_key).dataset_key,
            "backbone": backbone.manifest.model_key,
            "backbone_binding": binding,
        }
        self.contexts = []
        self.student_kind = student_kind

    def density(self, context, solver, nfe, *, seed=0, request_id=""):
        from genode.gico.clocks import clock_generator

        self.contexts.append(np.asarray(context).copy())
        assert solver == "euler"
        identity = request_id if self.student_kind == "GICO-sto-policy" else str(np.asarray(context).tolist())
        values = torch.softmax(torch.randn(64, generator=clock_generator(seed, identity)), dim=0)
        mass = values.numpy().astype(np.float64)
        return mass / mass.sum()


@pytest.mark.parametrize("student_kind", ["GICO-det-policy", "GICO-sto-policy"])
@pytest.mark.parametrize("clock_seed", [0, 8, 51])
def test_common_image_measurement_exports_raw_mass_and_exact_decoded_grid(student_kind, clock_seed):
    from dataclasses import replace

    from genode.gico.clocks import materialize, verify_measurement_clock

    backbone = _frozen_cifar_backbone(digest="b" * 64)
    sampler = ImageEulerSampler(backbone, device="cpu")
    policy = _CommonPolicyFixture(backbone, student_kind)
    schedule = sampler.gico_schedule(
        policy, target_nfe=8, class_labels=None, sample_keys=("first", "second"), clock_seed=clock_seed
    )
    assert schedule.gico_density_mass is not None
    assert not torch.equal(schedule.gico_density_mass, schedule.density_mass)
    for index in range(2):
        row = {**schedule.gico_measurement_clock(index), "schedule_key": "candidate"}
        verify_measurement_clock(row)
        assert row["time_grid"] == list(materialize(row["density_mass"], "euler", 8))
        np.testing.assert_array_equal(row["time_grid"], schedule.time_grid[index].numpy())
        with pytest.raises(ValueError, match="recollect"):
            verify_measurement_clock({**row, "density_mass": schedule.density_mass[index].tolist()})
    # A caller cannot attach a different raw density to an otherwise valid grid.
    with pytest.raises(ValueError, match="uniform mixture|shared density decoder"):
        replace(schedule, gico_density_mass=schedule.density_mass)
    # Shared deterministic rows also retain raw provenance when collapsed for hashing.
    if student_kind == "GICO-det-policy":
        single = replace(
            schedule,
            density_mass=schedule.density_mass[:1],
            time_grid=schedule.time_grid[:1],
            gico_density_mass=schedule.gico_density_mass[:1],
        )
        assert policy_schedule_request_hashes(schedule)[0] == single.sha256


@pytest.mark.parametrize("nfe", [2, 4, 8])
def test_fixed_image_reference_evidence_matches_common_realization_at_every_nfe(nfe):
    from genode.gico.clocks import reference_densities, verify_measurement_clock
    from genode.schedules.fixed import build_default_fixed_schedules, build_fixed_schedule
    from genode.schedules.specification import ScheduleSpecification

    expected = reference_densities("euler", nfe)
    schedules = build_default_fixed_schedules(nfe)
    assert len(schedules) == len(expected) == 25
    for schedule in schedules:
        row = schedule.gico_measurement_clock()
        assert row["density_mass"] == list(expected[row["schedule_key"]])
        verify_measurement_clock(row)
    with pytest.raises(ValueError, match="64 density bins"):
        build_fixed_schedule(ScheduleSpecification("uniform"), nfe, density_bin_count=32)


def test_common_image_policy_uses_native_context_and_independent_replayable_clock_rng():
    backbone = _frozen_imagenet_backbone(digest="b" * 64, offset=0.0)
    sampler = ImageEulerSampler(backbone, device="cpu")
    policy = _CommonPolicyFixture(backbone)
    schedule = sampler.gico_schedule(policy, target_nfe=2, class_labels=(2, 7), sample_keys=("a", "b"), clock_seed=8)
    replay = sampler.gico_schedule(policy, target_nfe=2, class_labels=(2, 7), sample_keys=("a", "b"), clock_seed=8)
    assert torch.equal(schedule.time_grid, replay.time_grid)
    assert not torch.equal(schedule.time_grid[0], schedule.time_grid[1])
    expected = backbone.encode_conditioning(torch.tensor([2, 7])).numpy()
    np.testing.assert_array_equal(policy.contexts[:2], expected)
    output_hash, grid_hash, execution_hash, mass_hash = policy_schedule_request_hashes(schedule)
    request = ImageGenerationRequest(
        source_request_sha256="c" * 64,
        backbone_manifest=backbone.manifest,
        latent_seeds=(31, 47),
        class_labels=(2, 7),
        target_nfe=2,
        schedule_policy_sha256=policy.artifact_sha256,
        schedule_output_sha256=output_hash,
        time_grid_sha256=grid_hash,
        execution_time_grid_sha256=execution_hash,
        density_mass_sha256=mass_hash,
    )
    generated = sampler.sample_gico(request, policy, sample_keys=("a", "b"), clock_seed=8)
    assert generated.field_evaluations == 2
    assert len(policy.contexts) == 6  # Two precomputations and exactly one clock per executed image.
    with pytest.raises(ValueError, match="output/grid/density"):
        sampler.sample_gico(request, policy, sample_keys=("a", "b"), clock_seed=9)
    policy.metadata["backbone_binding"]["checkpoint_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="backbone"):
        sampler.gico_schedule(policy, target_nfe=2, class_labels=(2, 7), sample_keys=("a", "b"))


def test_common_cifar_policy_receives_explicit_zero_context():
    backbone = _frozen_cifar_backbone(digest="a" * 64)
    sampler = ImageEulerSampler(backbone, device="cpu")
    policy = _CommonPolicyFixture(backbone)
    sampler.gico_schedule(policy, target_nfe=2, class_labels=None, sample_keys=("image",))
    np.testing.assert_array_equal(policy.contexts, [[0.0]])
