from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from genode.backbone_manifest import validate_provided_backbone_manifest
from genode.canonical_experiment_layout import CANONICAL_CHECKPOINT_STEPS
from genode.data.otflow_experiment_plan import FORECAST_FAMILY

TRAIN_BUDGET_STEPS = CANONICAL_CHECKPOINT_STEPS


def _write(path: Path, content: bytes | str = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)


class BackboneManifestTests(unittest.TestCase):
    def test_missing_provided_manifest_error_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir).resolve() / "private" / "backbone_manifest.json"
            validation = validate_provided_backbone_manifest(missing_path)
        self.assertEqual(validation["status"], "failed")
        self.assertEqual(validation["backbone_manifest_sha256"], "")
        self.assertEqual(validation["artifact_count"], 0)
        self.assertNotIn(str(missing_path), json.dumps(validation, sort_keys=True))

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
                manifest_path, scenario_key="solar_energy_10m", benchmark_family=FORECAST_FAMILY
            )
        self.assertEqual(validation["status"], "failed")
        self.assertTrue(
            any("checkpoint is too small to be valid" in error for error in validation["errors"]), validation["errors"]
        )

    def test_provided_manifest_validation_is_scoped_to_requested_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = []
            for train_steps in TRAIN_BUDGET_STEPS:
                label = f"{int(train_steps) // 1000}k"
                base = root / f"outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/traffic_hourly"
                _write(base / "model.pt", b"checkpoint" * 256)
                _write(base / "checkpoint_metadata.json", "{}")
                _write(base / "artifact_summary.json", "{}")
                rel = f"outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/traffic_hourly"
                artifacts.append(
                    {
                        "backbone_name": "otflow",
                        "benchmark_family": FORECAST_FAMILY,
                        "dataset_key": "traffic_hourly",
                        "train_steps": int(train_steps),
                        "train_budget_label": label,
                        "checkpoint_id": f"traffic_hourly_{label}",
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
                        "benchmark_family": FORECAST_FAMILY,
                        "dataset_key": "weather_daily",
                        "train_steps": int(train_steps),
                        "train_budget_label": label,
                        "checkpoint_id": f"weather_daily_{label}",
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
            with mock.patch("genode.backbone_manifest._validate_artifact_checkpoint_integrity", return_value=[]):
                validation = validate_provided_backbone_manifest(
                    manifest_path, scenario_key="traffic_hourly", benchmark_family=FORECAST_FAMILY
                )
        self.assertEqual(validation["status"], "complete", validation.get("errors"))
        self.assertEqual(validation["artifact_count"], len(TRAIN_BUDGET_STEPS))
        self.assertNotIn("manifest_path", validation)
        self.assertRegex(validation["backbone_manifest_sha256"], "^[0-9a-f]{64}$")
        self.assertNotIn(str(Path(tmpdir).resolve()), json.dumps(validation, sort_keys=True))

    def test_provided_manifest_validation_rejects_wrong_backbone_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = [
                {
                    "backbone_name": "otflow_molecule_3d",
                    "benchmark_family": FORECAST_FAMILY,
                    "dataset_key": "traffic_hourly",
                    "train_steps": int(train_steps),
                    "train_budget_label": f"{int(train_steps) // 1000}k",
                    "checkpoint_id": f"traffic_hourly_{int(train_steps)}",
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
                manifest_path, scenario_key="traffic_hourly", benchmark_family=FORECAST_FAMILY
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
                    rel = f"outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/traffic_hourly/duplicate_{duplicate_idx}"
                    _write(root / rel / "model.pt", b"checkpoint" * 256)
                    _write(root / rel / "checkpoint_metadata.json", "{}")
                    _write(root / rel / "artifact_summary.json", "{}")
                    artifacts.append(
                        {
                            "backbone_name": "otflow",
                            "benchmark_family": FORECAST_FAMILY,
                            "dataset_key": "traffic_hourly",
                            "train_steps": int(train_steps),
                            "train_budget_label": label,
                            "checkpoint_id": f"traffic_hourly_{label}_{duplicate_idx}",
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
            with mock.patch("genode.backbone_manifest._validate_artifact_checkpoint_integrity", return_value=[]):
                validation = validate_provided_backbone_manifest(
                    manifest_path, scenario_key="traffic_hourly", benchmark_family=FORECAST_FAMILY
                )
        self.assertEqual(validation["status"], "failed")
        self.assertTrue(
            any("duplicate runtime lookup key" in error for error in validation["errors"]), validation["errors"]
        )
        self.assertTrue(any("expected 1" in error and "train_steps=4000" in error for error in validation["errors"]))

    def test_load_checkpoint_model_wraps_unreadable_torch_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pt"
            checkpoint.write_bytes(b"not a torch checkpoint\n" * 100)
            import torch

            from genode.evaluation.otflow_evaluation_support import load_checkpoint_model

            with self.assertRaisesRegex(RuntimeError, "Invalid OTFlow checkpoint.*torch.load failed"):
                load_checkpoint_model(checkpoint, torch.device("cpu"))
