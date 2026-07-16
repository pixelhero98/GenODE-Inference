from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from genode.backbone_packages import (
    PACKAGED_BACKBONE_MANIFEST,
    REDACTED_LOCAL_PATH,
    _copy_tree_or_file,
    _contains_local_marker,
    _rewrite_json_paths,
    validate_backbone_artifact_checkpoint,
    load_portable_backbone_manifest,
    package_backbone_family,
    package_main,
    validate_backbone_package,
    validate_main,
    validate_provided_backbone_manifest,
)
from genode.data.otflow_experiment_plan import CONDITIONAL_GENERATION_FAMILY, FORECAST_FAMILY
from genode.experiment_layout import PAPER_CHECKPOINT_STEPS, SCENARIO_FAMILY_MOLECULE

MOLECULE_FAMILY = SCENARIO_FAMILY_MOLECULE
TRAIN_BUDGET_STEPS = PAPER_CHECKPOINT_STEPS


def _write(path: Path, content: bytes | str = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)


class BackbonePackageTests(unittest.TestCase):
    def test_package_validation_has_no_local_path_bypass(self) -> None:
        self.assertNotIn("require_clean_paths", inspect.signature(validate_backbone_package).parameters)
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as raised:
                validate_main(["package", "--allow_local_paths"])
        self.assertEqual(raised.exception.code, 2)

    def test_path_scrubbing_rejects_absolute_posix_windows_and_unc_paths(self) -> None:
        private_paths = (
            "/opt/private/model.pt",
            "D:\\private\\model.pt",
            "D:private\\model.pt",
            "\\\\server\\share\\model.pt",
            "../private/model.pt",
        )
        for private_path in private_paths:
            with self.subTest(private_path=private_path):
                self.assertTrue(_contains_local_marker(private_path))
                rewritten = _rewrite_json_paths({"checkpoint_path": private_path})
                self.assertEqual(rewritten["checkpoint_path"], REDACTED_LOCAL_PATH)
                self.assertEqual(_rewrite_json_paths({"description": private_path})["description"], private_path)

    def _source_tree(self, root: Path) -> None:
        scenarios = ("solar_energy_10m", "traffic_hourly", "weather_daily")
        artifacts = []
        for scenario in scenarios:
            for train_steps in TRAIN_BUDGET_STEPS:
                label = f"{int(train_steps) // 1000}k"
                base = f"outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}"
                ckpt_rel = f"{base}/model.pt"
                meta_rel = f"{base}/checkpoint_metadata.json"
                summary_rel = f"{base}/artifact_summary.json"
                _write(root / f"outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}/model.pt")
                _write(
                    root / f"outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}/checkpoint_metadata.json",
                    json.dumps(
                        {
                            "checkpoint_path": (
                                "/" + "scratch" + f"/example/genode/outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}/model.pt"
                            )
                        }
                    ),
                )
                _write(
                    root / f"outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}/artifact_summary.json",
                    json.dumps(
                        {
                            "summary_path": (
                                "/" + "projects" + f"/example/genode/outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}/artifact_summary.json"
                            )
                        }
                    ),
                )
                artifacts.append(
                    {
                        "backbone_name": "otflow",
                        "benchmark_family": FORECAST_FAMILY,
                        "dataset_key": scenario,
                        "train_steps": int(train_steps),
                        "train_budget_label": label,
                        "checkpoint_id": f"{scenario}_{label}",
                        "checkpoint_path": ckpt_rel,
                        "summary_path": summary_rel,
                        "metadata_path": meta_rel,
                        "status": "ready",
                        "seed": 0,
                    }
                )
            _write(root / f"paper_datasets/monash/{scenario}/manifest.json", "{}")
        for scenario in ("cryptos", "lobster_synthetic", "long_term_st"):
            for train_steps in TRAIN_BUDGET_STEPS:
                label = f"{int(train_steps) // 1000}k"
                artifacts.append(
                    {
                        "backbone_name": "otflow",
                        "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                        "dataset_key": scenario,
                        "train_steps": int(train_steps),
                        "train_budget_label": label,
                        "checkpoint_id": f"{scenario}_{label}",
                        "checkpoint_path": f"outputs/backbone_matrix/otflow/temporal_conditional_generation_transformer/{label}/{scenario}/model.pt",
                        "summary_path": f"outputs/backbone_matrix/otflow/temporal_conditional_generation_transformer/{label}/{scenario}/artifact_summary.json",
                        "metadata_path": f"outputs/backbone_matrix/otflow/temporal_conditional_generation_transformer/{label}/{scenario}/checkpoint_metadata.json",
                        "status": "ready",
                        "seed": 0,
                    }
                )
        for scenario in ("molecule_3d_set1", "molecule_3d_set2", "molecule_3d_set3"):
            for member_idx in range(6):
                for train_steps in TRAIN_BUDGET_STEPS:
                    artifacts.append(
                        {
                            "backbone_name": "otflow_molecule_3d",
                            "benchmark_family": MOLECULE_FAMILY,
                            "dataset_key": scenario,
                            "member_key": f"member_{member_idx}",
                            "stratum": f"stratum_{member_idx}",
                            "variant": "ar_h1",
                            "train_steps": int(train_steps),
                            "train_budget_label": f"{int(train_steps) // 1000}k",
                            "checkpoint_id": f"{scenario}_member_{member_idx}_{int(train_steps)}",
                            "checkpoint_path": f"outputs/molecule_3d_backbones/{scenario}/member_{member_idx}/ar_h1/{int(train_steps)}_steps/model.pt",
                            "summary_path": f"outputs/molecule_3d_backbones/{scenario}/member_{member_idx}/ar_h1/{int(train_steps)}_steps/artifact_summary.json",
                            "metadata_path": f"outputs/molecule_3d_backbones/{scenario}/member_{member_idx}/ar_h1/{int(train_steps)}_steps/checkpoint_metadata.json",
                            "status": "ready",
                            "seed": 0,
                        }
                    )
        manifest = {
            "version": "fm_backbone_manifest",
            "path_base": "manifest_parent",
            "artifact_count": len(artifacts),
            "ready_count": len(artifacts),
            "missing_count": 0,
            "artifacts": artifacts,
        }
        _write(root / "backbone_manifest.json", json.dumps(manifest))

    def test_package_family_writes_relative_clean_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_dir = Path(tmpdir) / "packages"
            self._source_tree(source_root)

            with mock.patch("genode.backbone_packages.validate_backbone_artifact_checkpoint", return_value=[]):
                summary = package_backbone_family(
                    family="temporal-extrapolation",
                    source_root=source_root,
                    output_dir=output_dir,
                    overwrite=True,
                    make_zip=False,
                )

            package_root = output_dir / "genode_temporal_extrapolation_backbones_datasets"
            self.assertEqual(summary["package_root"], f"external/{package_root.name}")
            with mock.patch("genode.backbone_packages.validate_backbone_artifact_checkpoint", return_value=[]):
                validation = validate_backbone_package(package_root, expected_family="temporal-extrapolation")
            self.assertEqual(validation["status"], "complete", validation.get("errors"))
            raw_manifest = json.loads((package_root / PACKAGED_BACKBONE_MANIFEST).read_text(encoding="utf-8"))
            artifact = raw_manifest["artifacts"][0]
            self.assertEqual(artifact["checkpoint_path"], "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/solar_energy_10m/model.pt")
            self.assertEqual(raw_manifest["path_base"], "manifest_parent")
            loaded = load_portable_backbone_manifest(package_root / PACKAGED_BACKBONE_MANIFEST)
            self.assertTrue(Path(loaded["artifacts"][0]["checkpoint_path"]).exists())

    def test_package_overwrite_allows_only_intended_in_output_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            output_dir = root / "packages"
            self._source_tree(source_root)

            with mock.patch("genode.backbone_packages.validate_backbone_artifact_checkpoint", return_value=[]):
                package_backbone_family(
                    family="temporal-extrapolation",
                    source_root=source_root,
                    output_dir=output_dir,
                    overwrite=True,
                    make_zip=False,
                )
            package_root = output_dir / "genode_temporal_extrapolation_backbones_datasets"
            stale_file = package_root / "stale.txt"
            stale_file.write_text("remove on overwrite", encoding="utf-8")

            with mock.patch("genode.backbone_packages.validate_backbone_artifact_checkpoint", return_value=[]):
                second = package_backbone_family(
                    family="temporal-extrapolation",
                    source_root=source_root,
                    output_dir=output_dir,
                    overwrite=True,
                    make_zip=False,
                )

            self.assertEqual(second["package_root"], f"external/{package_root.name}")
            self.assertFalse(stale_file.exists())
            self.assertTrue((package_root / PACKAGED_BACKBONE_MANIFEST).exists())

    def test_package_source_copy_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            package_root = root / "package"
            private_file = root / "private.txt"
            private_file.write_text("private", encoding="utf-8")
            linked_file = source_root / "data" / "linked.txt"
            linked_file.parent.mkdir(parents=True)
            try:
                os.symlink(private_file, linked_file)
            except OSError as exc:
                self.skipTest(f"Symlink creation is unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "linked or reparse-point source"):
                _copy_tree_or_file(source_root, package_root, "data/linked.txt")

    def test_package_validator_rejects_linked_file_even_when_bytes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            output_dir = root / "packages"
            self._source_tree(source_root)
            with mock.patch("genode.backbone_packages.validate_backbone_artifact_checkpoint", return_value=[]):
                package_backbone_family(
                    family="temporal-extrapolation",
                    source_root=source_root,
                    output_dir=output_dir,
                    overwrite=True,
                    make_zip=False,
                )
            package_root = output_dir / "genode_temporal_extrapolation_backbones_datasets"
            linked_file = package_root / "paper_datasets/monash/solar_energy_10m/manifest.json"
            external_file = root / "external_manifest.json"
            external_file.write_bytes(linked_file.read_bytes())
            linked_file.unlink()
            try:
                os.symlink(external_file, linked_file)
            except OSError as exc:
                self.skipTest(f"Symlink creation is unavailable: {exc}")

            with mock.patch("genode.backbone_packages.validate_backbone_artifact_checkpoint", return_value=[]):
                validation = validate_backbone_package(package_root)

            self.assertEqual(validation["status"], "failed")
            self.assertTrue(any("symlink, junction, or reparse point" in error for error in validation["errors"]))

    def test_package_validator_rejects_reparse_point_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("genode.backbone_packages.is_link_or_reparse_point", return_value=True):
                validation = validate_backbone_package(tmpdir)

        self.assertEqual(validation["status"], "failed")
        self.assertIn("reparse point", validation["errors"][0])

    def test_package_self_validation_finishes_before_existing_zip_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            output_dir = root / "packages"
            output_dir.mkdir()
            self._source_tree(source_root)
            zip_path = output_dir / "genode_temporal_extrapolation_backbones_datasets.zip"
            zip_path.write_bytes(b"existing-package")

            with mock.patch(
                "genode.backbone_packages.validate_backbone_package",
                return_value={"status": "failed", "errors": ["attestation failed"]},
            ):
                with self.assertRaisesRegex(ValueError, "attestation failed"):
                    package_backbone_family(
                        family="temporal-extrapolation",
                        source_root=source_root,
                        output_dir=output_dir,
                        overwrite=True,
                        make_zip=True,
                    )

            self.assertEqual(zip_path.read_bytes(), b"existing-package")

    def test_portable_manifest_rejects_absolute_path_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "backbone_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "path_base": "D:\\private",
                        "artifacts": [{"checkpoint_path": "outputs/model.pt"}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "path_base must be 'manifest_parent'"):
                load_portable_backbone_manifest(manifest_path)

    def test_package_validator_returns_failed_for_unsafe_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            package_root = Path(tmpdir)
            spec = {
                "schema_version": "genode_backbone_package",
                "family": "temporal-extrapolation",
                "scenarios": ["solar_energy_10m", "traffic_hourly", "weather_daily"],
                "expected_artifact_count": 15,
                "data_roots": [
                    "paper_datasets/monash/solar_energy_10m",
                    "paper_datasets/monash/traffic_hourly",
                    "paper_datasets/monash/weather_daily",
                ],
                "files": [],
                "artifact_count": 0,
            }
            (package_root / "package_manifest.json").write_text(json.dumps(spec), encoding="utf-8")
            manifest_path = package_root / PACKAGED_BACKBONE_MANIFEST
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "path_base": "D:\\private",
                        "ready_count": 0,
                        "artifact_count": 0,
                        "artifacts": [{"checkpoint_path": "../private/model.pt"}],
                    }
                ),
                encoding="utf-8",
            )

            validation = validate_backbone_package(package_root)

        self.assertEqual(validation["status"], "failed")
        self.assertTrue(
            any("Invalid packaged backbone manifest paths" in error for error in validation["errors"]),
            validation["errors"],
        )

    def test_molecule_checkpoint_integrity_rejects_corrupt_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_bytes(b"corrupt checkpoint" * 128)
            artifact = {
                "checkpoint_id": "molecule",
                "backbone_name": "otflow_molecule_3d",
                "benchmark_family": MOLECULE_FAMILY,
                "dataset_key": "molecule_3d_set1",
                "member_key": "member",
                "stratum": "Dynamic_Test",
                "train_steps": 4000,
            }

            errors = validate_backbone_artifact_checkpoint(artifact, checkpoint)

        self.assertTrue(any("checkpoint is not loadable" in error for error in errors), errors)

    def test_package_stage_override_and_no_zip_cli_are_removed(self) -> None:
        self.assertNotIn("stage_dir", inspect.signature(package_backbone_family).parameters)
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as stage_error:
                package_main(
                    ["--family", "temporal-extrapolation", "--output_dir", "out", "--stage_dir", "outside"]
                )
        self.assertEqual(stage_error.exception.code, 2)
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as zip_error:
                package_main(["--family", "temporal-extrapolation", "--output_dir", "out", "--no_zip"])
        self.assertEqual(zip_error.exception.code, 2)

    def test_package_builder_rejects_existing_tiny_checkpoint_during_self_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_dir = Path(tmpdir) / "packages"
            self._source_tree(source_root)
            with self.assertRaisesRegex(ValueError, "(?s)failed self-validation.*checkpoint is too small"):
                package_backbone_family(
                    family="temporal-extrapolation",
                    source_root=source_root,
                    output_dir=output_dir,
                    overwrite=True,
                    make_zip=False,
                )
            package_root = output_dir / "genode_temporal_extrapolation_backbones_datasets"
            self.assertFalse(package_root.exists())

    def test_provided_manifest_validation_rejects_unloadable_ready_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/solar_energy_10m/model.pt"
            _write(checkpoint, "exists")
            _write(checkpoint.with_name("checkpoint_metadata.json"), "{}")
            _write(checkpoint.with_name("artifact_summary.json"), "{}")
            manifest_path = root / "backbone_manifest.json"
            _write(
                manifest_path,
                json.dumps(
                    {
                        "version": "fm_backbone_manifest",
                        "path_base": "manifest_parent",
                        "artifact_count": 1,
                        "ready_count": 1,
                        "artifacts": [
                            {
                                "backbone_name": "otflow",
                                "benchmark_family": FORECAST_FAMILY,
                                "dataset_key": "solar_energy_10m",
                                "train_steps": 4000,
                                "train_budget_label": "4k",
                                "checkpoint_id": "solar_energy_10m_4k",
                                "checkpoint_path": "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/solar_energy_10m/model.pt",
                                "summary_path": "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/solar_energy_10m/artifact_summary.json",
                                "metadata_path": "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/solar_energy_10m/checkpoint_metadata.json",
                                "status": "ready",
                                "seed": 0,
                            }
                        ],
                    }
                ),
            )

            validation = validate_provided_backbone_manifest(
                manifest_path,
                scenario_key="solar_energy_10m",
                benchmark_family=FORECAST_FAMILY,
            )

        self.assertEqual(validation["status"], "failed")
        self.assertTrue(
            any("checkpoint is too small to be valid" in error for error in validation["errors"]),
            validation["errors"],
        )

    def test_provided_manifest_validation_is_scoped_to_requested_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = []
            for train_steps in TRAIN_BUDGET_STEPS:
                label = f"{int(train_steps) // 1000}k"
                base = root / f"outputs/backbone_matrix/otflow/temporal_conditional_generation/{label}/lobster_synthetic/transformer"
                _write(base / "model.pt", b"checkpoint" * 256)
                _write(base / "checkpoint_metadata.json", "{}")
                _write(base / "artifact_summary.json", "{}")
                rel = f"outputs/backbone_matrix/otflow/temporal_conditional_generation/{label}/lobster_synthetic/transformer"
                artifacts.append(
                    {
                        "backbone_name": "otflow",
                        "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                        "dataset_key": "lobster_synthetic",
                        "train_steps": int(train_steps),
                        "train_budget_label": label,
                        "checkpoint_id": f"lobster_synthetic_{label}",
                        "checkpoint_path": f"{rel}/model.pt",
                        "summary_path": f"{rel}/artifact_summary.json",
                        "metadata_path": f"{rel}/checkpoint_metadata.json",
                        "status": "ready",
                        "seed": 0,
                    }
                )
                artifacts.append(
                    {
                        "backbone_name": "otflow",
                        "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                        "dataset_key": "cryptos",
                        "train_steps": int(train_steps),
                        "train_budget_label": label,
                        "checkpoint_id": f"cryptos_{label}",
                        "status": "missing",
                        "seed": 0,
                    }
                )
            manifest_path = root / "backbone_manifest.json"
            _write(
                manifest_path,
                json.dumps(
                    {
                        "version": "fm_backbone_manifest",
                        "path_base": "manifest_parent",
                        "artifact_count": len(artifacts),
                        "ready_count": len(TRAIN_BUDGET_STEPS),
                        "missing_count": len(TRAIN_BUDGET_STEPS),
                        "artifacts": artifacts,
                    }
                ),
            )

            with mock.patch("genode.backbone_packages.validate_backbone_artifact_checkpoint", return_value=[]):
                validation = validate_provided_backbone_manifest(
                    manifest_path,
                    scenario_key="lobster_synthetic",
                    benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                )

        self.assertEqual(validation["status"], "complete", validation.get("errors"))
        self.assertEqual(validation["artifact_count"], len(TRAIN_BUDGET_STEPS))

    def test_provided_manifest_validation_rejects_wrong_backbone_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = [
                {
                    "backbone_name": "otflow_molecule_3d",
                    "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                    "dataset_key": "lobster_synthetic",
                    "train_steps": int(train_steps),
                    "train_budget_label": f"{int(train_steps) // 1000}k",
                    "checkpoint_id": f"lobster_synthetic_{int(train_steps)}",
                    "status": "ready",
                    "seed": 0,
                }
                for train_steps in TRAIN_BUDGET_STEPS
            ]
            manifest_path = root / "backbone_manifest.json"
            _write(
                manifest_path,
                json.dumps(
                    {
                        "version": "fm_backbone_manifest",
                        "path_base": "manifest_parent",
                        "artifact_count": len(artifacts),
                        "ready_count": len(artifacts),
                        "artifacts": artifacts,
                    }
                ),
            )

            validation = validate_provided_backbone_manifest(
                manifest_path,
                scenario_key="lobster_synthetic",
                benchmark_family=CONDITIONAL_GENERATION_FAMILY,
            )

        self.assertEqual(validation["status"], "failed")
        self.assertTrue(any("expected 'otflow'" in error for error in validation["errors"]), validation["errors"])
        self.assertTrue(any("No ready provided backbone artifacts match" in error for error in validation["errors"]))

    def test_provided_manifest_validation_rejects_duplicate_runtime_lookup_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = []
            for train_steps in TRAIN_BUDGET_STEPS:
                duplicate_count = 2 if int(train_steps) == int(TRAIN_BUDGET_STEPS[0]) else 1
                for duplicate_idx in range(duplicate_count):
                    label = f"{int(train_steps) // 1000}k"
                    rel = f"outputs/backbone_matrix/otflow/temporal_conditional_generation/{label}/lobster_synthetic/duplicate_{duplicate_idx}"
                    _write(root / rel / "model.pt", b"checkpoint" * 256)
                    _write(root / rel / "checkpoint_metadata.json", "{}")
                    _write(root / rel / "artifact_summary.json", "{}")
                    artifacts.append(
                        {
                            "backbone_name": "otflow",
                            "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                            "dataset_key": "lobster_synthetic",
                            "train_steps": int(train_steps),
                            "train_budget_label": label,
                            "checkpoint_id": f"lobster_synthetic_{label}_{duplicate_idx}",
                            "checkpoint_path": f"{rel}/model.pt",
                            "summary_path": f"{rel}/artifact_summary.json",
                            "metadata_path": f"{rel}/checkpoint_metadata.json",
                            "status": "ready",
                            "seed": 0,
                        }
                    )
            manifest_path = root / "backbone_manifest.json"
            _write(
                manifest_path,
                json.dumps(
                    {
                        "version": "fm_backbone_manifest",
                        "path_base": "manifest_parent",
                        "artifact_count": len(artifacts),
                        "ready_count": len(artifacts),
                        "artifacts": artifacts,
                    }
                ),
            )

            with mock.patch("genode.backbone_packages.validate_backbone_artifact_checkpoint", return_value=[]):
                validation = validate_provided_backbone_manifest(
                    manifest_path,
                    scenario_key="lobster_synthetic",
                    benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                )

        self.assertEqual(validation["status"], "failed")
        self.assertTrue(any("duplicate runtime lookup key" in error for error in validation["errors"]), validation["errors"])
        self.assertTrue(any("expected 1" in error and "train_steps=4000" in error for error in validation["errors"]))

    def test_load_checkpoint_model_wraps_unreadable_torch_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_bytes(b"not a torch checkpoint\n" * 100)
            import torch

            from genode.evaluation.otflow_evaluation_support import load_checkpoint_model

            with self.assertRaisesRegex(RuntimeError, "Invalid OTFlow checkpoint.*torch.load failed"):
                load_checkpoint_model(checkpoint, torch.device("cpu"))

    def test_pipeline_rejects_backbone_training_with_provided_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_dir = Path(tmpdir) / "packages"
            self._source_tree(source_root)
            with mock.patch("genode.backbone_packages.validate_backbone_artifact_checkpoint", return_value=[]):
                package_backbone_family(
                    family="temporal-extrapolation",
                    source_root=source_root,
                    output_dir=output_dir,
                    overwrite=True,
                    make_zip=False,
                )
            package_root = output_dir / "genode_temporal_extrapolation_backbones_datasets"
            try:
                from genode.pipeline.full_pipeline import build_argparser, run_full_pipeline
            except Exception as exc:  # pragma: no cover - exercised only in minimal dependency environments.
                self.skipTest(f"full pipeline dependencies are unavailable: {exc}")
            with mock.patch("genode.backbone_packages.validate_backbone_artifact_checkpoint", return_value=[]):
                args = build_argparser().parse_args(
                    [
                        "--scenario_key",
                        "solar_energy_10m",
                        "--run_root",
                        str(Path(tmpdir) / "run"),
                        "--backbone_package_root",
                        str(package_root),
                        "--dry_run",
                    ]
                )
                with self.assertRaisesRegex(ValueError, "cannot include backbone_training"):
                    run_full_pipeline(args)


if __name__ == "__main__":
    unittest.main()
