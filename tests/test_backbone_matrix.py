from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from genode.data.otflow_experiment_plan import experiment_plan_by_key
from genode.evaluation.fm_backbone_registry import (
    ACTIVE_CONDITIONAL_GENERATION_BACKBONE_BUDGETS,
    ACTIVE_FORECAST_BACKBONE_BUDGETS,
    BACKBONE_NAME_OTFLOW,
    CONDITIONAL_GENERATION_FAMILY,
    FORECAST_FAMILY,
    build_backbone_readiness_audit,
    find_backbone_artifact,
    materialize_backbone_manifest,
)
from genode.schedule_transfer.otflow_paper_tables import augment_rows_with_relative_metrics


def _checkpoint_metadata(benchmark_family: str, dataset_key: str, train_steps: int) -> dict:
    spec = experiment_plan_by_key()[str(dataset_key)]
    family_token = "forecast" if benchmark_family == FORECAST_FAMILY else "conditional_generation_transformer"
    metadata = {
        "checkpoint_id": f"{dataset_key}_otflow_{family_token}_{train_steps // 1000}k_seed0",
        "dataset_key": str(dataset_key),
        "benchmark_family": str(benchmark_family),
        "train_steps": int(train_steps),
        "train_budget_label": f"{train_steps // 1000}k",
        "history_len": int(spec.history_len),
        "future_block_len": int(spec.future_block_len),
        "cond_dim": 0,
        "split_stats": {
            "cond_dim": 0,
            "history_len": int(spec.history_len),
        },
    }
    if benchmark_family == CONDITIONAL_GENERATION_FAMILY:
        metadata["field_network_type"] = "transformer"
    return metadata


def _fake_checkpoint_signature(checkpoint_path: Path):
    metadata = json.loads(Path(checkpoint_path).with_name("checkpoint_metadata.json").read_text(encoding="utf-8"))
    spec = experiment_plan_by_key()[str(metadata["dataset_key"])]
    return (
        {
            "model_cond_dim": int(metadata.get("cond_dim", 0) or 0),
            "history_len": int(spec.history_len),
            "future_block_len": int(spec.future_block_len),
            "prediction_horizon": int(spec.future_block_len),
            "train_steps": int(metadata["train_steps"]),
            "field_network_type": str(metadata.get("field_network_type", "transformer")),
        },
        None,
    )


class BackboneMatrixTests(unittest.TestCase):
    def test_manifest_enumerates_all_40_active_target_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = materialize_backbone_manifest(
                matrix_root=Path(tmpdir) / "matrix",
                otflow_reuse_root=Path(tmpdir) / "reuse",
                imported_backbone_root=Path(tmpdir) / "imported",
                write_path=Path(tmpdir) / "backbone_manifest.json",
            )
        self.assertEqual(payload["artifact_count"], 40)
        self.assertEqual(payload["ready_count"], 0)
        self.assertEqual(payload["missing_count"], 40)
        self.assertTrue(all(artifact["backbone_name"] == BACKBONE_NAME_OTFLOW for artifact in payload["artifacts"]))
        self.assertFalse(any(artifact["dataset_key"] == "long_term_headered_ECG_records" for artifact in payload["artifacts"]))
        self.assertTrue(any(artifact["dataset_key"] == "sleep_edf" for artifact in payload["artifacts"]))

    def test_manifest_reuses_existing_otflow_20k_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reuse_root = Path(tmpdir) / "reuse"
            artifact_dir = reuse_root / "forecast" / "electricity"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "model.pt").write_bytes(b"ckpt")
            (artifact_dir / "checkpoint_metadata.json").write_text(
                json.dumps(_checkpoint_metadata(FORECAST_FAMILY, "electricity", 20000)),
                encoding="utf-8",
            )
            with patch(
                "genode.evaluation.fm_backbone_registry._checkpoint_signature",
                side_effect=_fake_checkpoint_signature,
            ):
                payload = materialize_backbone_manifest(
                    matrix_root=Path(tmpdir) / "matrix",
                    otflow_reuse_root=reuse_root,
                    imported_backbone_root=Path(tmpdir) / "imported",
                    write_path=Path(tmpdir) / "backbone_manifest.json",
                )
        artifact = find_backbone_artifact(
            payload,
            backbone_name=BACKBONE_NAME_OTFLOW,
            benchmark_family=FORECAST_FAMILY,
            dataset_key="electricity",
            train_steps=20000,
            status="ready",
        )
        self.assertEqual(artifact["source_kind"], "reused_shared_20k")
        self.assertEqual(artifact["train_budget_label"], "20k")
        self.assertIn("20k", artifact["checkpoint_id"])

    def test_manifest_rejects_unloadable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reuse_root = Path(tmpdir) / "reuse"
            artifact_dir = reuse_root / "forecast" / "electricity"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "model.pt").write_bytes(b"not a torch checkpoint")
            (artifact_dir / "checkpoint_metadata.json").write_text(
                json.dumps(_checkpoint_metadata(FORECAST_FAMILY, "electricity", 20000)),
                encoding="utf-8",
            )
            payload = materialize_backbone_manifest(
                matrix_root=Path(tmpdir) / "matrix",
                otflow_reuse_root=reuse_root,
                imported_backbone_root=Path(tmpdir) / "imported",
                write_path=Path(tmpdir) / "backbone_manifest.json",
            )

        artifact = next(
            row
            for row in payload["artifacts"]
            if row["benchmark_family"] == FORECAST_FAMILY
            and row["dataset_key"] == "electricity"
            and int(row["train_steps"]) == 20000
        )
        self.assertEqual(artifact["status"], "invalid")
        self.assertIn("Unable to load OTFlow checkpoint", artifact["compatibility_error"])

    def test_readiness_audit_normalizes_imported_backbones_and_reports_strict_40_grid_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            imported_root = Path(tmpdir) / "imported"
            missing = {
                (FORECAST_FAMILY, "electricity", 4000),
                (FORECAST_FAMILY, "solar_energy_10m", 4000),
                (FORECAST_FAMILY, "wind_farms_wo_missing", 4000),
                (CONDITIONAL_GENERATION_FAMILY, "cryptos", 8000),
                (CONDITIONAL_GENERATION_FAMILY, "cryptos", 12000),
                (CONDITIONAL_GENERATION_FAMILY, "cryptos", 16000),
                (CONDITIONAL_GENERATION_FAMILY, "es_mbp_10", 4000),
                (CONDITIONAL_GENERATION_FAMILY, "es_mbp_10", 8000),
                (CONDITIONAL_GENERATION_FAMILY, "es_mbp_10", 12000),
                (CONDITIONAL_GENERATION_FAMILY, "es_mbp_10", 16000),
                (CONDITIONAL_GENERATION_FAMILY, "sleep_edf", 8000),
                (CONDITIONAL_GENERATION_FAMILY, "sleep_edf", 12000),
                (CONDITIONAL_GENERATION_FAMILY, "sleep_edf", 16000),
                (CONDITIONAL_GENERATION_FAMILY, "sleep_edf", 20000),
            }

            def _write_imported_artifact(benchmark_family: str, dataset_key: str, train_steps: int) -> None:
                artifact_dir = imported_root / benchmark_family / dataset_key / f"{train_steps // 1000}k"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                (artifact_dir / "model.pt").write_bytes(f"{dataset_key}-{train_steps}".encode("utf-8"))
                metadata = _checkpoint_metadata(benchmark_family, dataset_key, train_steps)
                (artifact_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                (artifact_dir / "artifact_summary.json").write_text(json.dumps(metadata), encoding="utf-8")

            for dataset_key, steps in ACTIVE_FORECAST_BACKBONE_BUDGETS.items():
                for train_steps in steps:
                    if (FORECAST_FAMILY, dataset_key, train_steps) in missing:
                        continue
                    _write_imported_artifact(FORECAST_FAMILY, dataset_key, train_steps)
            for dataset_key, steps in ACTIVE_CONDITIONAL_GENERATION_BACKBONE_BUDGETS.items():
                for train_steps in steps:
                    if (CONDITIONAL_GENERATION_FAMILY, dataset_key, train_steps) in missing:
                        continue
                    _write_imported_artifact(CONDITIONAL_GENERATION_FAMILY, dataset_key, train_steps)

            with patch(
                "genode.evaluation.fm_backbone_registry._checkpoint_signature",
                side_effect=_fake_checkpoint_signature,
            ):
                readiness = build_backbone_readiness_audit(
                    matrix_root=matrix_root,
                    otflow_reuse_root=Path(tmpdir) / "reuse",
                    imported_backbone_root=imported_root,
                    dataset_root=Path(tmpdir) / "datasets",
                    sleep_edf_path=Path(tmpdir) / "sleep_edf.npz",
                    write_path=Path(tmpdir) / "backbone_manifest.json",
                )

            self.assertEqual(readiness["manifest"]["artifact_count"], 40)
            self.assertEqual(readiness["manifest"]["ready_count"], 26)
            self.assertEqual(readiness["manifest"]["missing_count"], 14)
            self.assertEqual(readiness["normalization"]["normalized_count"], 26)
            missing_rows = [
                artifact
                for artifact in readiness["manifest"]["artifacts"]
                if artifact["status"] != "ready"
            ]
            missing_keys = {
                (row["benchmark_family"], row["dataset_key"], int(row["train_steps"]))
                for row in missing_rows
            }
            self.assertEqual(missing_keys, missing)

    def test_imported_artifact_overrides_existing_matrix_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            imported_root = Path(tmpdir) / "imported"
            existing_root = matrix_root / "otflow" / "forecast" / "8k" / "electricity"
            existing_root.mkdir(parents=True, exist_ok=True)
            (existing_root / "model.pt").write_bytes(b"old")
            (existing_root / "checkpoint_metadata.json").write_text(
                json.dumps({**_checkpoint_metadata(FORECAST_FAMILY, "electricity", 8000), "checkpoint_id": "old"}),
                encoding="utf-8",
            )

            imported_artifact = imported_root / FORECAST_FAMILY / "electricity" / "8k"
            imported_artifact.mkdir(parents=True, exist_ok=True)
            (imported_artifact / "model.pt").write_bytes(b"new")
            (imported_artifact / "checkpoint_metadata.json").write_text(
                json.dumps({**_checkpoint_metadata(FORECAST_FAMILY, "electricity", 8000), "checkpoint_id": "new"}),
                encoding="utf-8",
            )
            (imported_artifact / "artifact_summary.json").write_text("{}", encoding="utf-8")

            with patch(
                "genode.evaluation.fm_backbone_registry._checkpoint_signature",
                side_effect=_fake_checkpoint_signature,
            ):
                readiness = build_backbone_readiness_audit(
                    matrix_root=matrix_root,
                    otflow_reuse_root=Path(tmpdir) / "reuse",
                    imported_backbone_root=imported_root,
                    dataset_root=Path(tmpdir) / "datasets",
                    sleep_edf_path=Path(tmpdir) / "sleep_edf.npz",
                    write_path=Path(tmpdir) / "backbone_manifest.json",
                )

            artifact = find_backbone_artifact(
                readiness["manifest"],
                backbone_name=BACKBONE_NAME_OTFLOW,
                benchmark_family=FORECAST_FAMILY,
                dataset_key="electricity",
                train_steps=8000,
                status="ready",
            )
            self.assertEqual(Path(artifact["checkpoint_path"]).read_bytes(), b"new")
            self.assertEqual(artifact["source_kind"], "matrix_output")

    def test_relative_metrics_respect_train_steps(self) -> None:
        rows = [
            {
                "benchmark_family": FORECAST_FAMILY,
                "split_phase": "locked_test",
                "dataset": "electricity",
                "backbone_name": "otflow",
                "checkpoint_id": "shared",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "target_nfe": 10,
                "solver_key": "euler",
                "schedule_name": "uniform",
                "crps": 5.0,
                "experiment_scope": "main",
            },
            {
                "benchmark_family": FORECAST_FAMILY,
                "split_phase": "locked_test",
                "dataset": "electricity",
                "backbone_name": "otflow",
                "checkpoint_id": "shared",
                "train_steps": 4000,
                "train_budget_label": "4k",
                "target_nfe": 10,
                "solver_key": "euler",
                "schedule_name": "flowts_power_sampling",
                "crps": 3.0,
                "experiment_scope": "main",
            },
            {
                "benchmark_family": FORECAST_FAMILY,
                "split_phase": "locked_test",
                "dataset": "electricity",
                "backbone_name": "otflow",
                "checkpoint_id": "shared",
                "train_steps": 4000,
                "train_budget_label": "4k",
                "target_nfe": 10,
                "solver_key": "euler",
                "schedule_name": "uniform",
                "crps": 4.0,
                "experiment_scope": "main",
            },
        ]
        enriched = augment_rows_with_relative_metrics(rows)
        by_schedule = {
            (row["train_steps"], row["schedule_name"]): row for row in enriched
        }
        self.assertIsNone(by_schedule[(4000, "flowts_power_sampling")]["relative_score_gain_vs_uniform"])
        self.assertAlmostEqual(
            by_schedule[(4000, "flowts_power_sampling")]["relative_crps_gain_vs_uniform"],
            0.25,
            places=8,
        )

    def test_conditional_generation_relative_metrics_preserve_seed_paired_gain(self) -> None:
        rows = [
            {
                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                "split_phase": "locked_test",
                "dataset": "cryptos",
                "backbone_name": "otflow",
                "checkpoint_id": "shared",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "seed": 0,
                "score_main": 10.0,
                "experiment_scope": "main",
            },
            {
                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                "split_phase": "locked_test",
                "dataset": "cryptos",
                "backbone_name": "otflow",
                "checkpoint_id": "shared",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "ays",
                "seed": 0,
                "score_main": 8.0,
                "relative_score_gain_vs_uniform": -0.125,
                "experiment_scope": "main",
            },
        ]

        enriched = augment_rows_with_relative_metrics(rows)
        by_schedule = {row["scheduler_key"]: row for row in enriched}
        self.assertEqual(by_schedule["ays"]["relative_score_gain_vs_uniform"], -0.125)


if __name__ == "__main__":
    unittest.main()
