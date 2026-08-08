from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from genode.backbone_packages import (
    PACKAGE_MANIFEST_NAME,
    PACKAGE_SCHEMA_VERSION,
    PACKAGED_BACKBONE_MANIFEST,
    backbone_package_protocol_payload,
    load_portable_backbone_manifest,
    package_backbone_family,
    validate_backbone_package,
    validate_provided_backbone_manifest,
)
from genode.data.otflow_experiment_plan import CONDITIONAL_GENERATION_FAMILY
from genode.canonical_experiment_layout import CANONICAL_CHECKPOINT_STEPS, SCENARIO_FAMILY_MOLECULE
from genode.data.otflow_experiment_plan import FORECAST_FAMILY
from genode.deterministic_archive import ARCHIVE_SCHEMA_VERSION, validate_deterministic_zip

MOLECULE_FAMILY = SCENARIO_FAMILY_MOLECULE
TRAIN_BUDGET_STEPS = CANONICAL_CHECKPOINT_STEPS


def _write(path: Path, content: bytes | str = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)


class BackbonePackageTests(unittest.TestCase):
    def _source_tree(self, root: Path) -> None:
        scenarios = ("solar_energy_10m", "traffic_hourly", "weather_daily")
        artifacts = []
        for scenario in scenarios:
            for train_steps in TRAIN_BUDGET_STEPS:
                label = f"{int(train_steps) // 1000}k"
                base = f"genode/outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}"
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
                        "checkpoint_path": f"genode/outputs/backbone_matrix/otflow/temporal_conditional_generation_transformer/{label}/{scenario}/model.pt",
                        "summary_path": f"genode/outputs/backbone_matrix/otflow/temporal_conditional_generation_transformer/{label}/{scenario}/artifact_summary.json",
                        "metadata_path": f"genode/outputs/backbone_matrix/otflow/temporal_conditional_generation_transformer/{label}/{scenario}/checkpoint_metadata.json",
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
            "artifact_count": len(artifacts),
            "ready_count": len(artifacts),
            "missing_count": 0,
            "artifacts": artifacts,
        }
        _write(root / "outputs/backbone_matrix/backbone_manifest.json", json.dumps(manifest))

    def test_package_family_writes_relative_clean_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_dir = Path(tmpdir) / "packages"
            self._source_tree(source_root)

            summary = package_backbone_family(
                family="temporal-extrapolation",
                source_root=source_root,
                output_dir=output_dir,
                overwrite=True,
                make_zip=False,
            )

            package_root = Path(summary["package_root"])
            with mock.patch("genode.backbone_packages._validate_artifact_checkpoint_integrity", return_value=[]):
                validation = validate_backbone_package(package_root, expected_family="temporal-extrapolation")
                relocated_root = Path(tmpdir) / "relocated" / "package"
                shutil.copytree(package_root, relocated_root)
                relocated_validation = validate_backbone_package(
                    relocated_root,
                    expected_family="temporal-extrapolation",
                )
            self.assertEqual(validation["status"], "complete", validation.get("errors"))
            self.assertEqual(validation, relocated_validation)
            self.assertNotIn("package_root", validation)
            self.assertEqual(
                validation["package_manifest_sha256"],
                hashlib.sha256((package_root / PACKAGE_MANIFEST_NAME).read_bytes()).hexdigest(),
            )
            self.assertNotIn(str(Path(tmpdir).resolve()), json.dumps(validation, sort_keys=True))
            raw_manifest = json.loads((package_root / PACKAGED_BACKBONE_MANIFEST).read_text(encoding="utf-8"))
            artifact = raw_manifest["artifacts"][0]
            self.assertEqual(artifact["checkpoint_path"], "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/solar_energy_10m/model.pt")
            self.assertEqual(raw_manifest["path_base"], "../..")
            loaded = load_portable_backbone_manifest(package_root / PACKAGED_BACKBONE_MANIFEST)
            self.assertTrue(Path(loaded["artifacts"][0]["checkpoint_path"]).exists())

    def test_package_protocol_identity_is_content_bound_and_relocation_invariant(self) -> None:
        manifest_bytes = json.dumps(
            {
                "schema_version": PACKAGE_SCHEMA_VERSION,
                "family": "temporal-extrapolation",
                "source_commit": "1" * 40,
                "artifact_count": 12,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmpdir:
            roots = (Path(tmpdir) / "first" / "package", Path(tmpdir) / "second" / "package")
            payloads = []
            for root in roots:
                _write(root / PACKAGE_MANIFEST_NAME, manifest_bytes)
                payloads.append(
                    backbone_package_protocol_payload(
                        argparse.Namespace(
                            backbone_package_root=str(root),
                            use_provided_backbones=False,
                        )
                    )
                )

        self.assertEqual(payloads[0], payloads[1])
        self.assertNotIn("backbone_package_root", payloads[0])
        self.assertEqual(
            payloads[0]["backbone_package_manifest_sha256"],
            hashlib.sha256(manifest_bytes).hexdigest(),
        )
        self.assertNotIn(str(Path(tmpdir).resolve()), json.dumps(payloads[0], sort_keys=True))

    def test_package_validation_redacts_unsafe_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            package_root = Path(tmpdir) / "package"
            local_path = Path(tmpdir).resolve() / "private" / "checkpoint.pt"
            _write(
                package_root / PACKAGE_MANIFEST_NAME,
                json.dumps(
                    {
                        "schema_version": PACKAGE_SCHEMA_VERSION,
                        "family": "temporal-extrapolation",
                        "scenarios": [],
                        "expected_artifact_count": 0,
                        "data_roots": [],
                        "artifact_count": 0,
                        "files": [{"path": str(local_path)}],
                    }
                ),
            )

            validation = validate_backbone_package(package_root)

        self.assertEqual(validation["status"], "failed")
        self.assertTrue(any("unsafe file path" in error for error in validation["errors"]))
        self.assertNotIn(str(local_path), json.dumps(validation, sort_keys=True))

    def test_missing_provided_manifest_error_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir).resolve() / "private" / "backbone_manifest.json"
            validation = validate_provided_backbone_manifest(missing_path)

        self.assertEqual(validation["status"], "failed")
        self.assertEqual(validation["backbone_manifest_sha256"], "")
        self.assertEqual(validation["artifact_count"], 0)
        self.assertNotIn(str(missing_path), json.dumps(validation, sort_keys=True))

    def test_package_family_zip_is_reproducible_and_preserves_archive_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            self._source_tree(source_root)
            first = package_backbone_family(
                family="temporal-extrapolation",
                source_root=source_root,
                output_dir=Path(tmpdir) / "first",
                make_zip=True,
            )
            second = package_backbone_family(
                family="temporal-extrapolation",
                source_root=source_root,
                output_dir=Path(tmpdir) / "second",
                make_zip=True,
            )
            first_zip = Path(first["zip_path"])
            second_zip = Path(second["zip_path"])
            self.assertEqual(first_zip.read_bytes(), second_zip.read_bytes())
            self.assertEqual(validate_deterministic_zip(first_zip)["status"], "complete")
            archive_sidecar = first_zip.with_suffix(first_zip.suffix + ".manifest.json")
            archive_metadata = json.loads(archive_sidecar.read_text(encoding="utf-8"))
            self.assertEqual(archive_metadata["schema_version"], ARCHIVE_SCHEMA_VERSION)
            package_sidecar = Path(first["zip_manifest_path"])
            self.assertEqual(package_sidecar.suffixes[-2:], [".package", ".json"])
            package_metadata = json.loads(package_sidecar.read_text(encoding="utf-8"))
            self.assertEqual(package_metadata["zip_sha256"], archive_metadata["sha256"])

    def test_package_validation_rejects_existing_tiny_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_dir = Path(tmpdir) / "packages"
            self._source_tree(source_root)
            package_root = Path(
                package_backbone_family(
                    family="temporal-extrapolation",
                    source_root=source_root,
                    output_dir=output_dir,
                    overwrite=True,
                    make_zip=False,
                )["package_root"]
            )

            validation = validate_backbone_package(package_root, expected_family="temporal-extrapolation")

        self.assertEqual(validation["status"], "failed")
        self.assertTrue(
            any("checkpoint is too small to be valid" in error for error in validation["errors"]),
            validation["errors"],
        )

    def test_provided_manifest_validation_rejects_unloadable_ready_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/solar_energy_10m/model.pt"
            _write(checkpoint, "exists")
            _write(checkpoint.with_name("checkpoint_metadata.json"), "{}")
            _write(checkpoint.with_name("artifact_summary.json"), "{}")
            manifest_path = root / "outputs/backbone_matrix/backbone_manifest.json"
            _write(
                manifest_path,
                json.dumps(
                    {
                        "version": "fm_backbone_manifest",
                        "path_base": "../..",
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
            manifest_path = root / "outputs/backbone_matrix/backbone_manifest.json"
            _write(
                manifest_path,
                json.dumps(
                    {
                        "version": "fm_backbone_manifest",
                        "path_base": "../..",
                        "artifact_count": len(artifacts),
                        "ready_count": len(TRAIN_BUDGET_STEPS),
                        "missing_count": len(TRAIN_BUDGET_STEPS),
                        "artifacts": artifacts,
                    }
                ),
            )

            with mock.patch("genode.backbone_packages._validate_artifact_checkpoint_integrity", return_value=[]):
                validation = validate_provided_backbone_manifest(
                    manifest_path,
                    scenario_key="lobster_synthetic",
                    benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                )

        self.assertEqual(validation["status"], "complete", validation.get("errors"))
        self.assertEqual(validation["artifact_count"], len(TRAIN_BUDGET_STEPS))
        self.assertNotIn("manifest_path", validation)
        self.assertRegex(validation["backbone_manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn(str(Path(tmpdir).resolve()), json.dumps(validation, sort_keys=True))

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
            manifest_path = root / "outputs/backbone_matrix/backbone_manifest.json"
            _write(
                manifest_path,
                json.dumps(
                    {
                        "version": "fm_backbone_manifest",
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
            manifest_path = root / "outputs/backbone_matrix/backbone_manifest.json"
            _write(
                manifest_path,
                json.dumps(
                    {
                        "version": "fm_backbone_manifest",
                        "path_base": "../..",
                        "artifact_count": len(artifacts),
                        "ready_count": len(artifacts),
                        "artifacts": artifacts,
                    }
                ),
            )

            with mock.patch("genode.backbone_packages._validate_artifact_checkpoint_integrity", return_value=[]):
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
            package_root = Path(
                package_backbone_family(
                    family="temporal-extrapolation",
                    source_root=source_root,
                    output_dir=output_dir,
                    overwrite=True,
                    make_zip=False,
                )["package_root"]
            )
            try:
                from genode.pipeline.full_pipeline import build_argparser, run_full_pipeline
            except Exception as exc:  # pragma: no cover - exercised only in minimal dependency environments.
                self.skipTest(f"full pipeline dependencies are unavailable: {exc}")
            with mock.patch("genode.backbone_packages._validate_artifact_checkpoint_integrity", return_value=[]):
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
