from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

import genode.release_bundles as release_bundle_module
from genode.artifacts.identity import canonical_json_bytes as identity_canonical_json_bytes
from genode.artifacts.identity import semantic_sha256
from genode.backbones.registry import get_image_backbone_spec
from genode.deterministic_archive import (
    ARCHIVE_MANIFEST_NAME,
    sha256_file,
    validate_deterministic_zip,
)
from genode.release_bundles import (
    FROZEN_GICO_CLOCK_MIXTURE_POLICY_SCHEMA,
    NamedCheckpoint,
    package_backbone_manifest_checkpoints,
    package_frozen_gico_clock_mixture_policy,
    package_frozen_gico_policy,
    package_named_checkpoints,
)
from genode.gico.image_clock_mixture import build_image_gico_clock_library
from genode.gico.image_clock_mixture_artifacts import (
    load_image_gico_clock_mixture_artifact,
    save_image_gico_clock_mixture_artifact,
)
from genode.gico.image_clock_mixture_training import (
    ImageGICOClockMixtureTrainingConfig,
    train_image_gico_clock_mixture,
)
from genode.gico.image_conditional import (
    ImageGICOBackboneContextDensityModel,
    ImageGICOBackboneContextModelConfig,
    build_image_gico_feature_groups,
)
from genode.gico.image_conditional_artifacts import (
    load_image_gico_conditional_artifact,
    save_image_gico_conditional_artifact,
)
from genode.gico.image_conditional_context import (
    prepare_image_gico_backbone_context,
)
from genode.gico.image_conditional_training import (
    ImageGICOBackboneContextTeacher,
    ImageGICOBackboneContextTrainingConfig,
    ImageGICOBackboneContextTrainingResult,
)
from tests.test_image_clock_mixture import _targets
from tests.test_image_primary_runtime import _frozen_imagenet_backbone


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_canonical_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(identity_canonical_json_bytes(value))


def _state_identity(path: Path, *, namespace: str) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    payload = {}
    for name, tensor in sorted(state.items()):
        array = tensor.detach().to(device="cpu").contiguous().numpy()
        payload[name] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "values": array.tolist(),
        }
    return semantic_sha256(payload, namespace=namespace)


def _array_identity(value: np.ndarray, *, namespace: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(namespace.encode("ascii"))
    digest.update(b"\0")
    digest.update(identity_canonical_json_bytes({"dtype": array.dtype.str, "shape": list(array.shape)}))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return f"{namespace}:{digest.hexdigest()}"


def _write_policy(policy: Path) -> str:
    policy.mkdir()
    torch.manual_seed(0)
    token = "gico"
    bundle_version = 4
    namespace_prefix = f"image-{token}-backbone-context"
    model_config = ImageGICOBackboneContextModelConfig(density_bin_count=64)
    student = ImageGICOBackboneContextDensityModel(
        model_config,
        np.zeros((1_000, 768), dtype=np.float32),
    )
    teacher = ImageGICOBackboneContextTeacher(density_bin_count=64)
    torch.save(student.state_dict(), policy / "student-state.pt")
    torch.save(teacher.state_dict(), policy / "teacher-state.pt")
    density_table = np.full((3, 1_000, 64), 1.0 / 64.0, dtype=np.float32)
    context_mean = np.zeros((768,), dtype=np.float32)
    context_scale = np.ones((768,), dtype=np.float32)
    with (policy / "class-density-table.npy").open("wb") as handle:
        np.save(handle, density_table, allow_pickle=False)
    with (policy / "context-normalizer-mean.npy").open("wb") as handle:
        np.save(handle, context_mean, allow_pickle=False)
    with (policy / "context-normalizer-scale.npy").open("wb") as handle:
        np.save(handle, context_scale, allow_pickle=False)

    feature_protocol_sha256 = "image-feature-protocol:" + "2" * 64
    assignments = np.arange(1_000, dtype=np.int64) % 32
    coordinates = np.zeros((1_000, 64), dtype=np.float64)
    feature_body = {
        "protocol": f"image_{token}_reward_feature_groups_v2",
        "usage": "reward_shrinkage_only_not_inference_context",
        "class_count": 1_000,
        "feature_dim": 64,
        "group_count": 32,
        "samples_per_class": 64,
        "source_panel_fingerprint": "panel:" + "3" * 64,
        "feature_protocol_sha256": feature_protocol_sha256,
        "real_feature_panel_sha256": "panel:" + "4" * 64,
        "pca_center": np.zeros(64).tolist(),
        "pca_components": np.eye(64).tolist(),
        "pca_scales": np.ones(64).tolist(),
        "class_coordinates": coordinates.tolist(),
        "group_assignments": assignments.tolist(),
        "group_centroids": np.zeros((32, 64)).tolist(),
    }
    feature_identity = semantic_sha256(
        feature_body,
        namespace=f"image-{token}-reward-feature-groups",
    )
    feature_payload = {
        "artifact": f"image_{token}_reward_feature_groups",
        **feature_body,
        "feature_group_sha256": feature_identity,
    }
    _write_canonical_json(
        policy / "reward-feature-groups.json",
        feature_payload,
    )

    backbone = get_image_backbone_spec("imagenet64_edm_ve_as_1rf")
    backbone_protocol_sha256 = "5" * 64
    backbone_checkpoint_sha256 = "6" * 64
    schedule_keys = ["uniform"]
    schedule_sha256s = ["schedule:" + "7" * 64]
    density_mass_sha256s = [[f"density-{index}:" + str(index + 1) * 64] for index in range(3)]
    weights = np.ones((3, 1_000, 1), dtype=np.float64)
    zeros = np.zeros_like(weights)
    coefficients = np.zeros((*weights.shape, 3), dtype=np.float64)
    coefficients[..., 2] = 1.0
    target_body = {
        "protocol": f"image_{token}_conditional_targets_v{bundle_version}",
        "conditioning": "classwise_rewards_independent_of_inference_context",
        "feature_group_usage": "reward_shrinkage_only_not_inference_context",
        "target_nfes": [2, 4, 8],
        "class_count": 1_000,
        "schedule_count": 1,
        "density_bin_count": 64,
        "schedule_keys": schedule_keys,
        "schedule_sha256s": schedule_sha256s,
        "density_mass_sha256s": density_mass_sha256s,
        "temperature_by_nfe": [1.0, 1.0, 1.0],
        "feature_group_sha256": feature_identity,
        "reward_evidence_sha256": "reward:" + "8" * 64,
        "fixed_support_sha256": "support:" + "9" * 64,
        "backbone_model_key": backbone.key,
        "backbone_protocol_sha256": backbone_protocol_sha256,
        "backbone_checkpoint_sha256": backbone_checkpoint_sha256,
        "feature_protocol_sha256": feature_protocol_sha256,
        "density_mass": density_table.astype(np.float64).tolist(),
        "mixture_weights": weights.tolist(),
        "normalized_rewards": zeros.tolist(),
        "jackknife_standard_errors": zeros.tolist(),
        "class_reliability": zeros.tolist(),
        "group_reliability": zeros.tolist(),
        "shrinkage_coefficients": coefficients.tolist(),
    }
    target_identity = semantic_sha256(
        target_body,
        namespace=f"image-{token}-conditional-targets-v{bundle_version}",
    )
    target_payload = {
        "artifact": f"image_{token}_conditional_targets",
        **target_body,
        "target_sha256": target_identity,
    }
    _write_canonical_json(
        policy / "conditional-targets.json",
        target_payload,
    )
    filenames = {
        "density_table": "class-density-table.npy",
        "targets": "conditional-targets.json",
        "context_mean": "context-normalizer-mean.npy",
        "context_scale": "context-normalizer-scale.npy",
        "feature_groups": "reward-feature-groups.json",
        "student_state": "student-state.pt",
        "teacher_state": "teacher-state.pt",
    }
    files = {
        role: {
            "filename": filename,
            "sha256": sha256_file(policy / filename),
            "size_bytes": (policy / filename).stat().st_size,
        }
        for role, filename in filenames.items()
    }
    student_identity = _state_identity(
        policy / "student-state.pt",
        namespace=f"{namespace_prefix}-model-state-v{bundle_version}",
    )
    teacher_identity = _state_identity(
        policy / "teacher-state.pt",
        namespace=f"{namespace_prefix}-teacher-state-v{bundle_version}",
    )
    mean_identity = _array_identity(context_mean, namespace=f"{namespace_prefix}-mean-v3")
    scale_identity = _array_identity(context_scale, namespace=f"{namespace_prefix}-scale-v3")
    normalizer_body = {
        "context_dim": 768,
        "dtype": "float32",
        "mean_sha256": mean_identity,
        "protocol": f"image_{token}_backbone_context_normalizer_v3",
        "scale_sha256": scale_identity,
        "std_floor": 1e-6,
    }
    normalizer_identity = semantic_sha256(
        normalizer_body,
        namespace=f"{namespace_prefix}-normalizer-v3",
    )
    binding_body = {
        "protocol": f"image_{token}_backbone_context_binding_v3",
        "selector": "native_model.model.map_label",
        "class_count": 1_000,
        "context_dim": 768,
        "dtype": "float32",
        "normalization_std_floor": 1e-6,
        "class_order_sha256": semantic_sha256(
            {"class_ids": list(range(1_000))},
            namespace=f"{namespace_prefix}-class-order-v3",
        ),
        "raw_context_table_sha256": f"image-{token}-raw-backbone-context-table-v3:" + "a" * 64,
        "normalized_context_table_sha256": f"image-{token}-normalized-backbone-context-table-v3:" + "b" * 64,
        "normalizer_mean_sha256": mean_identity,
        "normalizer_scale_sha256": scale_identity,
        "normalizer_sha256": normalizer_identity,
        "backbone_model_key": backbone.key,
        "backbone_protocol_sha256": backbone_protocol_sha256,
        "backbone_checkpoint_sha256": backbone_checkpoint_sha256,
        "source_revision": backbone.source_revision,
        "source_config_identity": backbone.source_config_identity,
    }
    binding_identity = semantic_sha256(
        binding_body,
        namespace=f"{namespace_prefix}-binding-v3",
    )
    binding = {**binding_body, "binding_sha256": binding_identity}
    training_config = {"protocol": f"image_{token}_backbone_context_training_config_v{bundle_version}"}
    training_body = {
        "conditional_density_range": 0.1,
        "conditioning": "normalized_frozen_backbone_map_label_plus_target_nfe",
        "context_binding_sha256": binding_identity,
        "feature_group_sha256": feature_identity,
        "feature_group_usage": "reward_shrinkage_only_not_inference_context",
        "final_kl": 0.1,
        "final_objective": 0.1,
        "final_residual_penalty": 0.0,
        "final_teacher_score": 0.0,
        "model_config": model_config.as_payload(),
        "model_config_sha256": semantic_sha256(
            model_config.as_payload(),
            namespace=f"{namespace_prefix}-model-config-v3",
        ),
        "model_state_sha256": student_identity,
        "protocol": f"image_{token}_backbone_context_training_v{bundle_version}",
        "target_sha256": target_identity,
        "teacher_density_summary_protocol": f"image_{token}_density_summaries_v1",
        "teacher_evidence_row_count": 1,
        "teacher_evidence_sha256": "evidence:" + "c" * 64,
        "teacher_oof_pairwise_accuracy": 1.0,
        "teacher_oof_rmse": 0.0,
        "teacher_protocol": f"image_{token}_backbone_context_nfe_density_teacher_v{bundle_version}",
        "teacher_schedule_fold_diagnostics": [{"fold": 0}],
        "teacher_state_sha256": teacher_identity,
        "training_config": training_config,
        "training_config_sha256": semantic_sha256(
            training_config,
            namespace=f"{namespace_prefix}-training-config-v{bundle_version}",
        ),
    }
    training = {
        **training_body,
        "result_sha256": semantic_sha256(
            training_body,
            namespace=f"{namespace_prefix}-training-result-v{bundle_version}",
        ),
    }
    manifest_body = {
        "artifact": f"image_{token}_backbone_context_policy",
        "protocol": f"image_{token}_backbone_context_policy_bundle_v{bundle_version}",
        "backbone_checkpoint_sha256": backbone_checkpoint_sha256,
        "backbone_model_key": backbone.key,
        "backbone_protocol_sha256": backbone_protocol_sha256,
        "class_nfe_contract": {
            "class_count": 1_000,
            "conditioning": "normalized_frozen_backbone_map_label_plus_target_nfe",
            "context_dim": 768,
            "density_bin_count": 64,
            "density_table_shape": [3, 1_000, 64],
            "feature_groups_are_inference_inputs": False,
            "initial_noise_is_an_inference_input": False,
            "raw_context_table_is_portable": False,
            "target_nfes": [2, 4, 8],
        },
        "context_binding": binding,
        "context_binding_sha256": binding_identity,
        "density_table_sha256": sha256_file(policy / "class-density-table.npy"),
        "feature_group_sha256": feature_identity,
        "feature_protocol_sha256": feature_protocol_sha256,
        "files": files,
        "portable_execution": {
            "bind_phase": "regenerate_context_from_verified_backbone",
            "density_table_verified_at_bind": True,
            "load_phase": "metadata_and_weights_only",
            "requires_explicit_backbone_binding": True,
        },
        "schedule_support": {
            "density_mass_sha256s": density_mass_sha256s,
            "fixed_support_sha256": target_body["fixed_support_sha256"],
            "reward_evidence_sha256": target_body["reward_evidence_sha256"],
            "schedule_keys": schedule_keys,
            "schedule_sha256s": schedule_sha256s,
        },
        "student_state_sha256": student_identity,
        "target_sha256": target_identity,
        "teacher_state_sha256": teacher_identity,
        "training": training,
    }
    artifact_identity = semantic_sha256(
        manifest_body,
        namespace=f"{namespace_prefix}-artifact-v{bundle_version}",
    )
    _write_canonical_json(
        policy / "manifest.json",
        {
            **manifest_body,
            "artifact_sha256": artifact_identity,
        },
    )
    return artifact_identity.rsplit(":", 1)[-1]


def _resign_policy_manifest(policy: Path, manifest: dict[str, object]) -> None:
    training = dict(manifest["training"])
    training_identity = str(training.pop("result_sha256"))
    training_namespace = training_identity.rsplit(":", 1)[0]
    manifest["training"] = {
        **training,
        "result_sha256": semantic_sha256(training, namespace=training_namespace),
    }
    artifact_identity = str(manifest.pop("artifact_sha256"))
    artifact_namespace = artifact_identity.rsplit(":", 1)[0]
    manifest["artifact_sha256"] = semantic_sha256(manifest, namespace=artifact_namespace)
    _write_canonical_json(policy / "manifest.json", manifest)


def _rewrite_policy_state(policy: Path, *, label: str, state: dict[str, torch.Tensor]) -> None:
    path = policy / f"{label}-state.pt"
    torch.save(state, path)
    manifest = json.loads((policy / "manifest.json").read_text(encoding="utf-8"))
    identity_field = f"{label}_state_sha256"
    training_field = "model_state_sha256" if label == "student" else "teacher_state_sha256"
    namespace = str(manifest[identity_field]).rsplit(":", 1)[0]
    identity = _state_identity(path, namespace=namespace)
    manifest[identity_field] = identity
    manifest["training"][training_field] = identity
    binding = manifest["files"][f"{label}_state"]
    binding["sha256"] = sha256_file(path)
    binding["size_bytes"] = path.stat().st_size
    _resign_policy_manifest(policy, manifest)


@pytest.fixture(scope="module")
def _clock_policy_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("release-clock-policy")
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
    torch.manual_seed(11)
    source_model = ImageGICOBackboneContextDensityModel(
        ImageGICOBackboneContextModelConfig(density_bin_count=64),
        prepared.normalized_context_table,
    )
    source_model.eval()
    source_model.requires_grad_(False)
    source_teacher = ImageGICOBackboneContextTeacher(density_bin_count=64)
    source_teacher.eval()
    source_teacher.requires_grad_(False)
    source_result = ImageGICOBackboneContextTrainingResult(
        model=source_model,
        teacher=source_teacher,
        config=ImageGICOBackboneContextTrainingConfig(
            teacher_steps=1,
            student_steps=1,
            teacher_batch_size=8,
        ),
        context_binding_sha256=prepared.binding.binding_sha256,
        target_sha256=targets.sha256,
        feature_group_sha256=feature_groups.sha256,
        final_kl=0.1,
        final_residual_penalty=0.0,
        final_teacher_score=0.0,
        final_objective=0.1,
        conditional_density_range=0.1,
        teacher_schedule_fold_diagnostics=tuple(
            {"fold": index} for index in range(len(targets.schedule_keys))
        ),
        teacher_oof_rmse=0.0,
        teacher_oof_pairwise_accuracy=1.0,
    )
    policy = root / "policy"
    save_image_gico_conditional_artifact(
        policy,
        source_result,
        feature_groups,
        targets,
        prepared,
    )
    loaded_source = load_image_gico_conditional_artifact(policy)
    bound_source = loaded_source.bind(backbone)
    decoder_result = train_image_gico_clock_mixture(
        targets,
        library=library,
        normalized_context_table=prepared.normalized_context_table,
        context_binding_sha256=prepared.binding.binding_sha256,
        config=ImageGICOClockMixtureTrainingConfig(steps=1, seed=19),
    )
    decoder = root / "decoder"
    save_image_gico_clock_mixture_artifact(
        decoder,
        decoder_result,
        source_artifact=bound_source,
    )
    return policy, decoder


def _copy_clock_policy_template(
    template: tuple[Path, Path],
    destination: Path,
) -> tuple[Path, Path]:
    source_policy, source_decoder = template
    policy = destination / "policy"
    decoder = destination / "decoder"
    shutil.copytree(source_policy, policy)
    shutil.copytree(source_decoder, decoder)
    return policy, decoder


def test_checkpoint_collection_contains_only_checkpoints_and_support_files(tmp_path: Path) -> None:
    model = tmp_path / "outputs" / "matrix" / "model.pt"
    metadata = model.with_name("checkpoint_metadata.json")
    summary = model.with_name("artifact_summary.json")
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights" * 200)
    _write_json(metadata, {"portable": True})
    _write_json(summary, {"status": "ready"})
    manifest = tmp_path / "outputs" / "backbone_manifest.json"
    _write_json(
        manifest,
        {
            "artifacts": [
                {
                    "backbone_name": "otflow",
                    "benchmark_family": "temporal_extrapolation",
                    "checkpoint_id": "example_step_4000",
                    "checkpoint_path": "outputs/matrix/model.pt",
                    "dataset_key": "example",
                    "effective_train_steps": 3900,
                    "metadata_path": "outputs/matrix/checkpoint_metadata.json",
                    "model_cond_dim": 0,
                    "status": "ready",
                    "summary_path": "outputs/matrix/artifact_summary.json",
                    "train_steps": 4000,
                }
            ]
        },
    )
    archive = tmp_path / "checkpoints.zip"
    package_backbone_manifest_checkpoints(
        manifest_path=manifest,
        source_root=tmp_path,
        output_path=archive,
        expected_count=1,
    )

    assert validate_deterministic_zip(archive)["status"] == "complete"
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert all("dataset" not in name for name in names)
        assert any(name.endswith("model.pt") for name in names)
        archive_manifest = bundle.read(ARCHIVE_MANIFEST_NAME).decode("utf-8")
        assert str(tmp_path) not in archive_manifest


def test_release_validation_cli_exits_nonzero_for_invalid_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive = tmp_path / "invalid.zip"
    archive.write_bytes(b"not-a-zip")

    with pytest.raises(SystemExit) as exc_info:
        release_bundle_module.main(["validate", "--archive", str(archive)])

    assert exc_info.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"


def test_checkpoint_collection_is_reproducible_across_manifest_order_and_formatting(tmp_path: Path) -> None:
    artifacts: list[dict[str, object]] = []
    for index, dataset in enumerate(("alpha", "beta")):
        checkpoint = tmp_path / "outputs" / dataset / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(bytes([index + 1]) * 2048)
        artifacts.append(
            {
                "backbone_name": "otflow",
                "benchmark_family": "temporal_extrapolation",
                "checkpoint_id": f"{dataset}_step_4000",
                "checkpoint_path": f"outputs/{dataset}/model.pt",
                "dataset_key": dataset,
                "effective_train_steps": 4000,
                "model_cond_dim": 0,
                "status": "ready",
                "train_steps": 4000,
            }
        )
    first_manifest = tmp_path / "manifest-a.json"
    second_manifest = tmp_path / "manifest-b.json"
    first_manifest.write_text(json.dumps({"artifacts": artifacts}, indent=2), encoding="utf-8")
    second_manifest.write_text(json.dumps({"artifacts": list(reversed(artifacts))}), encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    package_backbone_manifest_checkpoints(
        manifest_path=first_manifest,
        source_root=tmp_path,
        output_path=first,
        expected_count=2,
        include_support_files=False,
    )
    package_backbone_manifest_checkpoints(
        manifest_path=second_manifest,
        source_root=tmp_path,
        output_path=second,
        expected_count=2,
        include_support_files=False,
    )
    assert first.read_bytes() == second.read_bytes()


def test_checkpoint_collection_rejects_source_root_escape(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"weights" * 200)
    manifest = root / "manifest.json"
    _write_json(
        manifest,
        {
            "artifacts": [
                {
                    "benchmark_family": "temporal_extrapolation",
                    "checkpoint_id": "escape",
                    "checkpoint_path": "../outside.pt",
                    "dataset_key": "example",
                    "status": "ready",
                    "train_steps": 4000,
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="local filesystem path|Unsafe checkpoint source path|escapes source_root"):
        package_backbone_manifest_checkpoints(
            manifest_path=manifest,
            source_root=root,
            output_path=tmp_path / "escape.zip",
        )


@pytest.mark.parametrize("filename,content", [("not-a-checkpoint.txt", b"x" * 2048), ("empty.pt", b"")])
def test_named_checkpoint_collection_rejects_invalid_checkpoint_files(
    tmp_path: Path,
    filename: str,
    content: bytes,
) -> None:
    checkpoint = tmp_path / filename
    checkpoint.write_bytes(content)

    with pytest.raises(ValueError, match="unsupported filename|too small"):
        package_named_checkpoints(
            [NamedCheckpoint("invalid", checkpoint)],
            tmp_path / "invalid.zip",
        )


def test_checkpoint_only_mode_excludes_support_files(tmp_path: Path) -> None:
    checkpoint = tmp_path / "outputs" / "model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights" * 200)
    metadata = checkpoint.with_name("metadata.json")
    _write_json(metadata, {"portable": True})
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "artifacts": [
                {
                    "benchmark_family": "temporal_extrapolation",
                    "checkpoint_id": "example",
                    "checkpoint_path": "outputs/model.pt",
                    "dataset_key": "example",
                    "metadata_path": "outputs/metadata.json",
                    "status": "ready",
                    "train_steps": 4000,
                }
            ]
        },
    )
    archive = tmp_path / "checkpoint-only.zip"
    package_backbone_manifest_checkpoints(
        manifest_path=manifest,
        source_root=tmp_path,
        output_path=archive,
        include_support_files=False,
    )
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == [
            ARCHIVE_MANIFEST_NAME,
            "backbones/temporal_extrapolation/example/step-4000/model.pt",
        ]


def test_frozen_gico_policy_requires_and_packages_teacher_and_student(tmp_path: Path) -> None:
    policy = tmp_path / "policy"
    source_artifact_digest = _write_policy(policy)
    archive = tmp_path / "policy.zip"
    package_frozen_gico_policy(policy_dir=policy, output_path=archive)

    assert validate_deterministic_zip(archive)["status"] == "complete"
    with zipfile.ZipFile(archive) as bundle:
        assert "policy/student-state.pt" in bundle.namelist()
        assert "policy/teacher-state.pt" in bundle.namelist()
        assert "policy/manifest.json" not in bundle.namelist()
        assert "policy/reward-feature-groups.json" not in bundle.namelist()
        assert "policy/conditional-targets.json" not in bundle.namelist()
        manifest = json.loads(bundle.read(ARCHIVE_MANIFEST_NAME))
        assert manifest["bundle_kind"] == "frozen_gico_policy"
        assert manifest["metadata"]["context"]["selector"] == "native_model.model.map_label"
        assert manifest["metadata"]["source_artifact_digest"] == source_artifact_digest
        assert len(manifest["metadata"]["source_manifest_sha256"]) == 64
        assert manifest["metadata"]["training_clock_pool"] == ["uniform"]
        assert {record["role"] for record in manifest["files"]} == {
            "context_mean",
            "context_scale",
            "density_table",
            "student_state",
            "teacher_state",
        }
    assert ("gi" + "po").encode("ascii") not in archive.read_bytes().lower()


def test_frozen_gico_policy_build_remains_byte_reproducible_and_requires_both_states(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "policy"
    _write_policy(policy)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    package_frozen_gico_policy(policy_dir=policy, output_path=first)
    package_frozen_gico_policy(policy_dir=policy, output_path=second)
    assert first.read_bytes() == second.read_bytes()

    (policy / "teacher-state.pt").unlink()
    with pytest.raises(ValueError, match="missing"):
        package_frozen_gico_policy(policy_dir=policy, output_path=tmp_path / "missing.zip")


def test_frozen_gico_policy_recomputes_semantic_state_identity(tmp_path: Path) -> None:
    policy = tmp_path / "policy"
    _write_policy(policy)
    manifest = json.loads((policy / "manifest.json").read_text(encoding="utf-8"))
    namespace = str(manifest["student_state_sha256"]).rsplit(":", 1)[0]
    forged = f"{namespace}:{'0' * 64}"
    manifest["student_state_sha256"] = forged
    manifest["training"]["model_state_sha256"] = forged
    _resign_policy_manifest(policy, manifest)

    with pytest.raises(ValueError, match="student state identity is inconsistent"):
        package_frozen_gico_policy(policy_dir=policy, output_path=tmp_path / "forged.zip")


def test_frozen_gico_policy_rejects_state_dtype_casting(tmp_path: Path) -> None:
    policy = tmp_path / "policy"
    _write_policy(policy)
    state = torch.load(policy / "student-state.pt", map_location="cpu", weights_only=True)
    name = next(key for key, tensor in state.items() if tensor.is_floating_point())
    state[name] = state[name].to(dtype=torch.float64)
    _rewrite_policy_state(policy, label="student", state=state)

    with pytest.raises(ValueError, match="invalid layout, device, shape, or dtype"):
        package_frozen_gico_policy(policy_dir=policy, output_path=tmp_path / "wrong-dtype.zip")


def test_frozen_gico_policy_rejects_nonfinite_state_tensor(tmp_path: Path) -> None:
    policy = tmp_path / "policy"
    _write_policy(policy)
    state = torch.load(policy / "teacher-state.pt", map_location="cpu", weights_only=True)
    name = next(key for key, tensor in state.items() if tensor.is_floating_point() and tensor.numel())
    state[name] = state[name].clone()
    state[name].reshape(-1)[0] = float("nan")
    torch.save(state, policy / "teacher-state.pt")

    with pytest.raises(ValueError, match="contains non-finite values"):
        release_bundle_module._validate_policy_architecture(policy, density_bin_count=64)


@pytest.mark.parametrize(
    "local_value",
    [
        "prefix,C:/" + "home/private/checkpoint.pt",
        "prefix,/" + "home/private/checkpoint.pt",
        "run:/" + "projects/private/checkpoint.pt",
        "prefix,/opt/private/checkpoint.pt",
        "run:/work/private/checkpoint.pt",
    ],
)
def test_policy_json_precheck_rejects_embedded_local_paths(local_value: str) -> None:
    with pytest.raises(ValueError, match="local filesystem path"):
        release_bundle_module._assert_no_local_json_paths(local_value, label="policy")


def test_frozen_gico_policy_recomputes_density_table_digest(tmp_path: Path) -> None:
    policy = tmp_path / "policy"
    _write_policy(policy)
    manifest = json.loads((policy / "manifest.json").read_text(encoding="utf-8"))
    manifest["density_table_sha256"] = "0" * 64
    _resign_policy_manifest(policy, manifest)

    with pytest.raises(ValueError, match="density-table digest differs"):
        package_frozen_gico_policy(policy_dir=policy, output_path=tmp_path / "forged.zip")


def test_frozen_gico_clock_mixture_policy_is_reproducible_bound_and_complete(
    tmp_path: Path,
    _clock_policy_template: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, decoder = _copy_clock_policy_template(_clock_policy_template, tmp_path)
    first = tmp_path / "clock-policy-first.zip"
    second = tmp_path / "clock-policy-second.zip"
    validation_cache: list[object] = []
    source_cache: list[object] = []
    decoder_cache: list[object] = []
    original_validation = release_bundle_module._validated_frozen_policy_source
    original_source_loader = release_bundle_module.load_image_gico_conditional_artifact
    original_decoder_loader = release_bundle_module.load_image_gico_clock_mixture_artifact

    def recording_validation(root: Path) -> object:
        result = original_validation(root)
        validation_cache.append(result)
        return result

    def recording_source_loader(root: Path) -> object:
        result = original_source_loader(root)
        source_cache.append(result)
        return result

    def recording_decoder_loader(
        root: Path,
        *,
        source_artifact: object,
    ) -> object:
        result = original_decoder_loader(root, source_artifact=source_artifact)
        decoder_cache.append(result)
        return result

    monkeypatch.setattr(
        release_bundle_module,
        "_validated_frozen_policy_source",
        recording_validation,
    )
    monkeypatch.setattr(
        release_bundle_module,
        "load_image_gico_conditional_artifact",
        recording_source_loader,
    )
    monkeypatch.setattr(
        release_bundle_module,
        "load_image_gico_clock_mixture_artifact",
        recording_decoder_loader,
    )
    package_frozen_gico_clock_mixture_policy(
        policy_dir=policy,
        decoder_dir=decoder,
        output_path=first,
    )
    assert len(validation_cache) == len(source_cache) == len(decoder_cache) == 1
    monkeypatch.setattr(
        release_bundle_module,
        "_validated_frozen_policy_source",
        lambda _root: validation_cache[0],
    )
    monkeypatch.setattr(
        release_bundle_module,
        "load_image_gico_conditional_artifact",
        lambda _root: source_cache[0],
    )
    monkeypatch.setattr(
        release_bundle_module,
        "load_image_gico_clock_mixture_artifact",
        lambda _root, *, source_artifact: decoder_cache[0],
    )
    package_frozen_gico_clock_mixture_policy(
        policy_dir=policy,
        decoder_dir=decoder,
        output_path=second,
    )

    assert first.read_bytes() == second.read_bytes()
    assert validate_deterministic_zip(first)["status"] == "complete"
    source_manifest = json.loads((policy / "manifest.json").read_text(encoding="utf-8"))
    decoder_manifest = json.loads((decoder / "manifest.json").read_text(encoding="utf-8"))
    expected_names = {
        ARCHIVE_MANIFEST_NAME,
        "policy/class-density-table.npy",
        "policy/context-normalizer-mean.npy",
        "policy/context-normalizer-scale.npy",
        "policy/student-state.pt",
        "policy/teacher-state.pt",
        "policy/manifest.json",
        "policy/conditional-targets.json",
        "policy/reward-feature-groups.json",
        "decoder/manifest.json",
        "decoder/clock-mixture-state.pt",
        "decoder/clock-library.json",
        "decoder/complete-clocks-nfe-2.npy",
        "decoder/complete-clocks-nfe-4.npy",
        "decoder/complete-clocks-nfe-8.npy",
    }
    with zipfile.ZipFile(first) as bundle:
        assert set(bundle.namelist()) == expected_names
        manifest = json.loads(bundle.read(ARCHIVE_MANIFEST_NAME))
        assert manifest["bundle_kind"] == "frozen_gico_clock_mixture_policy"
        metadata = manifest["metadata"]
        assert metadata["policy_schema_version"] == FROZEN_GICO_CLOCK_MIXTURE_POLICY_SCHEMA
        assert metadata["source_policy"]["artifact_sha256"] == source_manifest["artifact_sha256"]
        assert metadata["clock_mixture"]["artifact_sha256"] == decoder_manifest["artifact_sha256"]
        assert metadata["clock_mixture"]["clock_library_sha256"] == decoder_manifest[
            "clock_library_sha256"
        ]
        assert metadata["clock_mixture"]["execution_model_state_sha256"] == decoder_manifest[
            "model_state_sha256"
        ]
        assert metadata["clock_mixture"]["serialized_model_state_sha256"] == decoder_manifest[
            "serialized_model_state_sha256"
        ]
        assert metadata["inference_contract"]["internal_rng"] is False
        assert metadata["inference_contract"]["target_nfes"] == [2, 4, 8]
        assert {record["role"] for record in manifest["files"]} == {
            "context_mean",
            "context_scale",
            "density_table",
            "student_state",
            "teacher_state",
            "clock_mixture_manifest",
            "clock_mixture_state",
            "clock_library",
            "clock_grid_nfe_2",
            "clock_grid_nfe_4",
            "clock_grid_nfe_8",
            "source_policy_manifest",
            "conditional_targets",
            "reward_feature_groups",
        }
        extracted = tmp_path / "extracted"
        bundle.extractall(extracted)
    extracted_source = load_image_gico_conditional_artifact(extracted / "policy")
    extracted_decoder = load_image_gico_clock_mixture_artifact(
        extracted / "decoder",
        source_artifact=extracted_source,
    )
    assert extracted_source.artifact_sha256 == source_manifest["artifact_sha256"]
    assert extracted_decoder.artifact_sha256 == decoder_manifest["artifact_sha256"]


@pytest.mark.parametrize("mutation", ["missing_decoder_grid", "tampered_decoder_state"])
def test_frozen_gico_clock_mixture_policy_rejects_missing_or_tampered_inputs(
    tmp_path: Path,
    _clock_policy_template: tuple[Path, Path],
    mutation: str,
) -> None:
    policy, decoder = _copy_clock_policy_template(_clock_policy_template, tmp_path)
    if mutation == "missing_decoder_grid":
        (decoder / "complete-clocks-nfe-8.npy").unlink()
    elif mutation == "tampered_decoder_state":
        state_path = decoder / "clock-mixture-state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        name = next(key for key, value in state.items() if value.is_floating_point() and value.numel())
        state[name] = state[name].clone()
        state[name].reshape(-1)[0] += 0.125
        torch.save(state, state_path)
        manifest_path = decoder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["model_state"] = {
            "filename": state_path.name,
            "sha256": sha256_file(state_path),
            "size_bytes": state_path.stat().st_size,
        }
        manifest_body = dict(manifest)
        manifest_body.pop("artifact_sha256")
        manifest["artifact_sha256"] = semantic_sha256(
            manifest_body,
            namespace="image-gico-complete-clock-mixture-artifact-v1",
        )
        _write_canonical_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="incomplete|inconsistent|does not match|missing|partial"):
        package_frozen_gico_clock_mixture_policy(
            policy_dir=policy,
            decoder_dir=decoder,
            output_path=tmp_path / "invalid.zip",
        )


def test_gico_clock_policy_cli_uses_the_strict_packager(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = tmp_path / "policy"
    decoder = tmp_path / "decoder"
    archive = tmp_path / "cli-clock-policy.zip"
    observed: dict[str, object] = {}

    def fake_packager(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"bundle_kind": "frozen_gico_clock_mixture_policy"}

    monkeypatch.setattr(
        release_bundle_module,
        "package_frozen_gico_clock_mixture_policy",
        fake_packager,
    )
    release_bundle_module.main(
        [
            "gico-clock-policy",
            "--policy-dir",
            str(policy),
            "--decoder-dir",
            str(decoder),
            "--output",
            str(archive),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result == {"bundle_kind": "frozen_gico_clock_mixture_policy"}
    assert observed == {
        "policy_dir": str(policy),
        "decoder_dir": str(decoder),
        "output_path": str(archive),
        "overwrite": False,
    }


def test_clock_mixture_packager_explicitly_requires_current_v4_source_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = tmp_path / "historical-policy"
    decoder = tmp_path / "decoder"
    policy.mkdir()
    decoder.mkdir()
    for filename in release_bundle_module._ARCHIVED_CLOCK_MIXTURE_FILES:
        (decoder / filename).write_bytes(b"placeholder")
    monkeypatch.setattr(
        release_bundle_module,
        "_validated_frozen_policy_source",
        lambda _root: (
            [],
            {},
            {"protocol": "image_gico_backbone_context_policy_bundle_v3"},
        ),
    )

    with pytest.raises(ValueError, match="current v4.*gico-policy"):
        package_frozen_gico_clock_mixture_policy(
            policy_dir=policy,
            decoder_dir=decoder,
            output_path=tmp_path / "unsupported.zip",
        )
