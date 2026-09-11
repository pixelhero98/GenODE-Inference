from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import genode.release_bundles as release_bundle_module
from genode.artifacts.identity import canonical_json_bytes as identity_canonical_json_bytes
from genode.deterministic_archive import (
    ARCHIVE_MANIFEST_NAME,
    validate_deterministic_zip,
)
from genode.release_bundles import (
    NamedCheckpoint,
    package_backbone_manifest_checkpoints,
    package_named_checkpoints,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_canonical_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(identity_canonical_json_bytes(value))


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


def test_release_validation_cli_exits_nonzero_for_invalid_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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


def _write_common_artifact(directory):
    from genode.gico.clocks import materialize, reference_densities
    from genode.gico.evidence import prepare_evidence
    from genode.gico.image_supervision import prepare_image_rows
    from genode.gico.networks import DensityTeacher, DeterministicStudent, ModelConfig, StochasticStudent
    from genode.gico.policy import save_artifact
    from tests.test_image_gico_supervision import image_manifest

    rows, contexts, image_metadata = prepare_image_rows(image_manifest())
    for row in list(rows):
        if row["schedule_key"] != "uniform":
            mass = reference_densities("euler", 2)["late_p_3_reversed"]
            rows.append(
                {
                    **row,
                    "schedule_key": "late_p_3_reversed",
                    "density_mass": list(mass),
                    "time_grid": list(materialize(mass, "euler", 2)),
                    "metrics": {"kid": 0.08},
                }
            )
    evidence = prepare_evidence(rows, contexts, purpose="functional")
    config = ModelConfig(condition_dim=evidence.conditioning.width, metric_count=1)
    metadata = {
        "task": evidence.task,
        "backbone": evidence.backbone,
        "solvers": ["euler"],
        "backbone_binding": image_metadata["backbone_binding"],
        "locked_test_used": False,
        "split_contexts": {
            split: sorted({row["context_id"] for row in rows if row["split"] == split})
            for split in ("train", "validation")
        },
        "reward_calibrations": {
            solver: calibration.to_payload() for solver, calibration in evidence.calibrations.items()
        },
        "reference_densities": {f"euler:2:{row['schedule_key']}": row["density_mass"] for row in rows},
        "reference_grids": {f"euler:2:{row['schedule_key']}": row["time_grid"] for row in rows},
    }
    from dataclasses import asdict

    from genode.gico.profiles import AUXILIARY_NORMALIZATION, TEMPERATURE_UNITS, resolve_profile

    metadata.update(
        fitting_profile={"task": evidence.task, **asdict(resolve_profile(evidence.task, backbone=evidence.backbone))},
        metric_weights=[1.0],
        temperature_units=TEMPERATURE_UNITS,
        auxiliary_normalization=AUXILIARY_NORMALIZATION,
        teacher_selection_criterion="heldout_reference_utility_regret",
        student_selection_criterion="post_ramp_validation_distillation",
        selected_temperature=1.0,
        history={
            "student_selection": {k: {"step": 2000, "coefficient": 0.01} for k in ("deterministic", "stochastic")}
        },
    )
    save_artifact(
        directory,
        DensityTeacher(config),
        {"deterministic": DeterministicStudent(config), "stochastic": StochasticStudent(config)},
        evidence.conditioning,
        metadata,
    )


def test_common_policy_release_validates_both_students_and_teacher_and_is_reproducible(tmp_path):
    from genode.release_bundles import package_frozen_gico_policy

    directory = tmp_path / "policy"
    _write_common_artifact(directory)
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    package_frozen_gico_policy(policy_dir=directory, output_path=first)
    package_frozen_gico_policy(policy_dir=directory, output_path=second)
    assert first.read_bytes() == second.read_bytes()
    assert validate_deterministic_zip(first)["status"] == "complete"
    with zipfile.ZipFile(first) as archive:
        assert set(archive.namelist()) == {ARCHIVE_MANIFEST_NAME, "policy/policy.pt", "policy/manifest.json"}
    (directory / "policy.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        package_frozen_gico_policy(policy_dir=directory, output_path=tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()


def test_historical_policy_artifacts_are_not_relabelled(tmp_path):
    import torch

    from genode.release_bundles import package_frozen_gico_policy

    directory = tmp_path / "old-policy"
    directory.mkdir()
    torch.save({"protocol": "image_gico_old"}, directory / "policy.pt")
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="historical"):
        package_frozen_gico_policy(policy_dir=directory, output_path=tmp_path / "old.zip")
