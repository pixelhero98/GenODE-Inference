from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from genode.backbone_packages import (
    PACKAGED_BACKBONE_MANIFEST,
    load_portable_backbone_manifest,
    package_backbone_family,
    validate_backbone_package,
    validate_provided_backbone_manifest,
)
from genode.data.otflow_experiment_plan import CONDITIONAL_GENERATION_FAMILY
from genode.canonical_experiment_layout import CANONICAL_CHECKPOINT_STEPS, SCENARIO_FAMILY_MOLECULE
from genode.data.otflow_experiment_plan import FORECAST_FAMILY

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
            self.assertEqual(validation["status"], "complete", validation.get("errors"))
            raw_manifest = json.loads((package_root / PACKAGED_BACKBONE_MANIFEST).read_text(encoding="utf-8"))
            artifact = raw_manifest["artifacts"][0]
            self.assertEqual(artifact["checkpoint_path"], "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/solar_energy_10m/model.pt")
            self.assertEqual(raw_manifest["path_base"], "../..")
            loaded = load_portable_backbone_manifest(package_root / PACKAGED_BACKBONE_MANIFEST)
            self.assertTrue(Path(loaded["artifacts"][0]["checkpoint_path"]).exists())

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
