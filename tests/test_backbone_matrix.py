from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from genode.data.otflow_experiment_plan import (
    conditional_generation_dataset_keys,
    forecast_dataset_keys,
    experiment_plan_by_key,
)
from genode.evaluation.fm_backbone_registry import (
    ACTIVE_CONDITIONAL_GENERATION_BACKBONE_BUDGETS,
    ACTIVE_FORECAST_BACKBONE_BUDGETS,
    BACKBONE_NAME_OTFLOW,
    BACKBONE_NAME_OTFLOW_MOLECULE,
    CONDITIONAL_GENERATION_FAMILY,
    FORECAST_FAMILY,
    MOLECULE_FAMILY,
    build_backbone_readiness_audit,
    find_backbone_artifact,
    load_backbone_manifest,
    materialize_backbone_manifest,
)
from genode.data.otflow_paths import display_project_path
from genode.schedule_transfer.otflow_reference_tables import augment_rows_with_relative_metrics


FORECAST_KEYS = ("solar_energy_10m", "traffic_hourly", "weather_daily")
CONDITIONAL_KEYS = ("cryptos", "lobster_synthetic", "long_term_st")


def _checkpoint_metadata(benchmark_family: str, dataset_key: str, train_steps: int) -> dict:
    spec = experiment_plan_by_key()[str(dataset_key)]
    family_token = "temporal_extrapolation" if benchmark_family == FORECAST_FAMILY else "temporal_conditional_generation_transformer"
    metadata = {
        "checkpoint_id": f"{dataset_key}_otflow_{family_token}_{train_steps // 1000}k_seed0",
        "dataset_key": str(dataset_key),
        "benchmark_family": str(benchmark_family),
        "train_steps": int(train_steps),
        "train_budget_label": f"{train_steps // 1000}k",
        "checkpoint_budget_steps": int(train_steps),
        "effective_train_steps": int(train_steps),
        "checkpoint_export_protocol": "exact_budget_step_state",
        "history_len": int(spec.history_len),
        "future_block_len": int(spec.future_block_len),
        "rollout_mode": "non_ar",
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
            "rollout_mode": "non_ar",
            "train_steps": int(metadata["train_steps"]),
            "field_network_type": str(metadata.get("field_network_type", "transformer")),
        },
        None,
    )


class BackboneMatrixTests(unittest.TestCase):
    def test_reference_temporal_matrix_is_exactly_six_datasets(self) -> None:
        self.assertEqual(forecast_dataset_keys(), FORECAST_KEYS)
        self.assertEqual(conditional_generation_dataset_keys(), CONDITIONAL_KEYS)
        self.assertEqual(tuple(ACTIVE_FORECAST_BACKBONE_BUDGETS), FORECAST_KEYS)
        self.assertEqual(tuple(ACTIVE_CONDITIONAL_GENERATION_BACKBONE_BUDGETS), CONDITIONAL_KEYS)

    def test_manifest_enumerates_all_30_active_target_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = materialize_backbone_manifest(
                matrix_root=Path(tmpdir) / "matrix",
                otflow_reuse_root=Path(tmpdir) / "reuse",
                imported_backbone_root=Path(tmpdir) / "imported",
                molecule_group_root=Path(tmpdir) / "empty_molecule_groups",
                write_path=Path(tmpdir) / "backbone_manifest.json",
            )
        self.assertEqual(payload["artifact_count"], 30)
        self.assertEqual(payload["ready_count"], 0)
        self.assertEqual(payload["missing_count"], 30)
        self.assertTrue(all(artifact["backbone_name"] == BACKBONE_NAME_OTFLOW for artifact in payload["artifacts"]))
        active = {(row["benchmark_family"], row["dataset_key"]) for row in payload["artifacts"]}
        self.assertEqual(active, {(FORECAST_FAMILY, key) for key in FORECAST_KEYS} | {(CONDITIONAL_GENERATION_FAMILY, key) for key in CONDITIONAL_KEYS})

    def test_manifest_omitted_roots_follow_explicit_write_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repository_root = root / "repo"
            manifest_path = repository_root / "portable" / "backbone_manifest.json"
            repository_root.mkdir()

            payload = materialize_backbone_manifest(
                budget_steps=(4000,),
                write_path=manifest_path,
            )

            self.assertEqual(payload["matrix_root"], "matrix")
            self.assertEqual(payload["otflow_reuse_root"], "shared_backbones")
            self.assertEqual(payload["imported_backbone_root"], "imported_backbones")
            self.assertEqual(payload["molecule_group_root"], "molecule_3d")
            self.assertEqual(payload["molecule_backbone_root"], "molecule_3d_backbones")
            loaded = load_backbone_manifest(manifest_path)
            self.assertEqual(Path(loaded["matrix_root"]), manifest_path.parent / "matrix")
            self.assertEqual(Path(loaded["molecule_group_root"]), manifest_path.parent / "molecule_3d")

    def test_manifest_loader_resolves_paths_from_explicit_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "backbone_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": "fm_backbone_manifest",
                "path_base": "manifest_parent",
                "artifact_count": 1,
                "ready_count": 1,
                "artifacts": [
                    {
                        "checkpoint_path": "outputs/backbone_matrix/example/model.pt",
                        "summary_path": "outputs/backbone_matrix/example/artifact_summary.json",
                        "metadata_path": "outputs/backbone_matrix/example/checkpoint_metadata.json",
                    }
                ],
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_backbone_manifest(manifest_path)

        artifact = loaded["artifacts"][0]
        self.assertEqual(Path(artifact["checkpoint_path"]), root / "outputs" / "backbone_matrix" / "example" / "model.pt")
        self.assertEqual(Path(artifact["summary_path"]), root / "outputs" / "backbone_matrix" / "example" / "artifact_summary.json")

    def test_manifest_loader_rejects_absolute_path_base(self) -> None:
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
                load_backbone_manifest(manifest_path)

    def test_manifest_loader_rejects_missing_path_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "genode"
            manifest_path = root / "outputs" / "backbone_matrix" / "backbone_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": "fm_backbone_manifest",
                "matrix_root": "outputs/backbone_matrix",
                "artifact_count": 1,
                "ready_count": 1,
                "artifacts": [
                    {
                        "checkpoint_path": "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/traffic_hourly/model.pt",
                        "summary_path": "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/traffic_hourly/artifact_summary.json",
                        "metadata_path": "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/traffic_hourly/checkpoint_metadata.json",
                    }
                ],
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires path_base='manifest_parent'"):
                load_backbone_manifest(manifest_path)

    def test_display_path_uses_logical_outputs_root_for_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            scratch_outputs = Path(tmpdir) / "scratch" / "genode" / "outputs"
            root.mkdir()
            scratch_outputs.mkdir(parents=True)
            outputs_link = root / "outputs"
            try:
                outputs_link.symlink_to(scratch_outputs, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            target = scratch_outputs / "backbone_matrix" / "example" / "model.pt"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"ckpt")

            with patch("genode.data.otflow_paths.project_root", return_value=root):
                display = display_project_path(target)

        self.assertEqual(display, "outputs/backbone_matrix/example/model.pt")

    def test_display_path_redacts_external_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            external = Path(tmpdir) / "private-user" / "artifact.pt"
            root.mkdir()
            external.parent.mkdir()
            external.write_bytes(b"artifact")
            with patch("genode.data.otflow_paths.project_root", return_value=root):
                display = display_project_path(external)

        self.assertEqual(display, "external/artifact.pt")

    def test_manifest_rejects_roots_outside_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            repo.mkdir()
            with self.assertRaisesRegex(ValueError, "must be contained"):
                materialize_backbone_manifest(
                    matrix_root=root / "private" / "matrix",
                    otflow_reuse_root=root / "private" / "reuse",
                    imported_backbone_root=root / "private" / "imported",
                    molecule_group_root=root / "private" / "groups",
                    molecule_backbone_root=root / "private" / "molecule_backbones",
                    write_path=repo / "backbone_manifest.json",
                )

    def test_manifest_extends_temporal_grid_with_trainable_molecule_strata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            group_root = Path(tmpdir) / "groups"
            manifest_dir = group_root / "molecule_3d_set1"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            (manifest_dir / "group_manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": "molecule_3d_group_manifest",
                        "dataset_key": "molecule_3d_set1",
                        "benchmark_family": MOLECULE_FAMILY,
                        "source_zip_names": ["trajectory.zip", "triangulene_3.zip"],
                        "strata": [
                            {
                                "member_key": "trajectory_Dynamic_A",
                                "stratum": "Dynamic_A",
                                "source_zip_name": "trajectory.zip",
                                "trajectory_count": 7,
                                "atom_count": 6,
                                "formula": "C4H2",
                                "processed_dir": "Dynamic_A",
                                "trainable": True,
                                "mixed_shape": False,
                            },
                            {
                                "member_key": "triangulene_3_Dynamic_A",
                                "stratum": "Dynamic_A",
                                "source_zip_name": "triangulene_3.zip",
                                "trajectory_count": 5,
                                "atom_count": 8,
                                "formula": "C6H2",
                                "processed_dir": "triangulene_3_Dynamic_A",
                                "trainable": True,
                                "mixed_shape": False,
                            },
                            {
                                "member_key": "trajectory_Direct_A",
                                "stratum": "Direct_A",
                                "source_zip_name": "trajectory.zip",
                                "trajectory_count": 99,
                                "atom_count": 8,
                                "formula": "C6H2",
                                "processed_dir": "Direct_A",
                                "trainable": True,
                                "mixed_shape": False,
                            },
                            {
                                "member_key": "trajectory_Dynamic_Mixed",
                                "stratum": "Dynamic_Mixed",
                                "source_zip_name": "trajectory.zip",
                                "trajectory_count": 99,
                                "atom_count": 10,
                                "formula": "C8H2",
                                "processed_dir": "Dynamic_Mixed",
                                "trainable": True,
                                "mixed_shape": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = materialize_backbone_manifest(
                matrix_root=Path(tmpdir) / "matrix",
                otflow_reuse_root=Path(tmpdir) / "reuse",
                imported_backbone_root=Path(tmpdir) / "imported",
                molecule_group_root=group_root,
                molecule_backbone_root=Path(tmpdir) / "molecule_backbones",
                write_path=Path(tmpdir) / "backbone_manifest.json",
            )

        self.assertEqual(payload["temporal_artifact_count"], 30)
        self.assertEqual(payload["molecule_stratum_count"], 2)
        self.assertEqual(payload["molecule_artifact_count"], 10)
        self.assertEqual(payload["artifact_count"], 40)
        artifact = find_backbone_artifact(
            payload,
            backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
            benchmark_family=MOLECULE_FAMILY,
            dataset_key="molecule_3d_set1",
            member_key="trajectory_Dynamic_A",
            stratum="Dynamic_A",
            train_steps=4000,
            status="missing",
        )
        self.assertEqual(artifact["atom_count"], 6)
        self.assertEqual(artifact["formula"], "C4H2")
        self.assertEqual(artifact["trajectory_count"], 7)
        self.assertIn("trajectory_Dynamic_A", artifact["checkpoint_path"])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            find_backbone_artifact(
                payload,
                backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
                benchmark_family=MOLECULE_FAMILY,
                dataset_key="molecule_3d_set1",
                stratum="Dynamic_A",
                train_steps=4000,
                status="missing",
            )

    def test_manifest_marks_corrupt_molecule_checkpoint_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_root = root / "groups"
            dataset_key = "molecule_3d_set1"
            member_key = "trajectory_Dynamic_Test"
            stratum = "Dynamic_Test"
            group_manifest_dir = group_root / dataset_key
            group_manifest_dir.mkdir(parents=True)
            group_manifest_dir.joinpath("group_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_key": dataset_key,
                        "strata": [
                            {
                                "member_key": member_key,
                                "stratum": stratum,
                                "source_zip_name": "trajectory.zip",
                                "processed_dir": f"{dataset_key}/{member_key}",
                                "trajectory_count": 2,
                                "atom_count": 1,
                                "formula": "H",
                                "trainable": True,
                                "mixed_shape": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            molecule_root = root / "molecule_backbones"
            artifact_root = molecule_root / dataset_key / member_key / stratum / "ar_h1" / "4000_steps"
            artifact_root.mkdir(parents=True)
            artifact_root.joinpath("model.pt").write_bytes(b"corrupt molecule checkpoint" * 128)
            metadata = {
                "dataset_key": dataset_key,
                "member_key": member_key,
                "stratum": stratum,
                "benchmark_family": MOLECULE_FAMILY,
                "backbone_name": BACKBONE_NAME_OTFLOW_MOLECULE,
                "variant": "ar_h1",
                "train_steps": 4000,
                "history_len": 16,
                "future_block_len": 1,
                "rollout_mode": "autoregressive",
                "source_zip_name": "trajectory.zip",
                "formula": "H",
                "split_stats": {"atom_count": 1, "formula": "H"},
                "cfg": {"model": {"rollout_mode": "autoregressive"}},
            }
            artifact_root.joinpath("checkpoint_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            artifact_root.joinpath("artifact_summary.json").write_text("{}", encoding="utf-8")

            payload = materialize_backbone_manifest(
                matrix_root=root / "matrix",
                otflow_reuse_root=root / "reuse",
                imported_backbone_root=root / "imported",
                molecule_group_root=group_root,
                molecule_backbone_root=molecule_root,
                budget_steps=(4000,),
                write_path=root / "backbone_manifest.json",
            )

        artifact = find_backbone_artifact(
            payload,
            backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
            benchmark_family=MOLECULE_FAMILY,
            dataset_key=dataset_key,
            member_key=member_key,
            stratum=stratum,
            train_steps=4000,
            status="invalid",
        )
        self.assertIn("Unable to load molecule checkpoint", artifact["compatibility_error"])

    def test_manifest_reuses_existing_otflow_20k_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reuse_root = Path(tmpdir) / "reuse"
            artifact_dir = reuse_root / FORECAST_FAMILY / "traffic_hourly"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "model.pt").write_bytes(b"ckpt")
            (artifact_dir / "checkpoint_metadata.json").write_text(
                json.dumps(_checkpoint_metadata(FORECAST_FAMILY, "traffic_hourly", 20000)),
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
                    molecule_group_root=Path(tmpdir) / "empty_molecule_groups",
                    write_path=Path(tmpdir) / "backbone_manifest.json",
                )
        artifact = find_backbone_artifact(
            payload,
            backbone_name=BACKBONE_NAME_OTFLOW,
            benchmark_family=FORECAST_FAMILY,
            dataset_key="traffic_hourly",
            train_steps=20000,
            status="ready",
        )
        self.assertEqual(artifact["source_kind"], "reused_shared_20k")
        self.assertEqual(artifact["train_budget_label"], "20k")
        self.assertEqual(artifact["checkpoint_budget_steps"], 20000)
        self.assertEqual(artifact["effective_train_steps"], 20000)
        self.assertEqual(artifact["checkpoint_export_protocol"], "exact_budget_step_state")

    def test_manifest_rejects_matching_length_autoregressive_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            imported_root = Path(tmpdir) / "imported"
            artifact_dir = imported_root / CONDITIONAL_GENERATION_FAMILY / "cryptos" / "20k"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "model.pt").write_bytes(b"ckpt")
            (artifact_dir / "checkpoint_metadata.json").write_text(
                json.dumps(_checkpoint_metadata(CONDITIONAL_GENERATION_FAMILY, "cryptos", 20000)),
                encoding="utf-8",
            )

            def _ar_signature(checkpoint_path: Path):
                signature, _ = _fake_checkpoint_signature(checkpoint_path)
                signature["rollout_mode"] = "autoregressive"
                return signature, None

            with patch(
                "genode.evaluation.fm_backbone_registry._checkpoint_signature",
                side_effect=_ar_signature,
            ):
                readiness = build_backbone_readiness_audit(
                    matrix_root=Path(tmpdir) / "matrix",
                    otflow_reuse_root=Path(tmpdir) / "reuse",
                    imported_backbone_root=imported_root,
                    molecule_group_root=Path(tmpdir) / "empty_molecule_groups",
                    dataset_root=Path(tmpdir) / "datasets",
                    budget_steps=(20000,),
                    write_path=Path(tmpdir) / "backbone_manifest.json",
                )
                payload = readiness["manifest"]

        rows = [
            row
            for row in payload["artifacts"]
            if row["benchmark_family"] == CONDITIONAL_GENERATION_FAMILY and row["dataset_key"] == "cryptos"
        ]
        self.assertEqual(rows[0]["status"], "invalid")
        self.assertIn("checkpoint rollout_mode='autoregressive'", rows[0]["compatibility_error"])

    def test_readiness_audit_normalizes_imported_backbones_and_reports_strict_30_grid_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            imported_root = Path(tmpdir) / "imported"
            missing = {
                (FORECAST_FAMILY, "traffic_hourly", 4000),
                (FORECAST_FAMILY, "weather_daily", 4000),
                (CONDITIONAL_GENERATION_FAMILY, "lobster_synthetic", 4000),
                (CONDITIONAL_GENERATION_FAMILY, "long_term_st", 8000),
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
                    if (FORECAST_FAMILY, dataset_key, train_steps) not in missing:
                        _write_imported_artifact(FORECAST_FAMILY, dataset_key, train_steps)
            for dataset_key, steps in ACTIVE_CONDITIONAL_GENERATION_BACKBONE_BUDGETS.items():
                for train_steps in steps:
                    if (CONDITIONAL_GENERATION_FAMILY, dataset_key, train_steps) not in missing:
                        _write_imported_artifact(CONDITIONAL_GENERATION_FAMILY, dataset_key, train_steps)

            with patch(
                "genode.evaluation.fm_backbone_registry._checkpoint_signature",
                side_effect=_fake_checkpoint_signature,
            ):
                readiness = build_backbone_readiness_audit(
                    matrix_root=matrix_root,
                    otflow_reuse_root=Path(tmpdir) / "reuse",
                    imported_backbone_root=imported_root,
                    molecule_group_root=Path(tmpdir) / "empty_molecule_groups",
                    dataset_root=Path(tmpdir) / "datasets",
                    lobster_profile_path=Path(tmpdir) / "lobster_profile.json",
                    write_path=Path(tmpdir) / "backbone_manifest.json",
                )

            self.assertEqual(readiness["manifest"]["artifact_count"], 30)
            self.assertEqual(readiness["manifest"]["ready_count"], 26)
            self.assertEqual(readiness["manifest"]["missing_count"], 4)
            self.assertEqual(readiness["normalization"]["normalized_count"], 26)
            artifact = find_backbone_artifact(
                readiness["manifest"],
                backbone_name=BACKBONE_NAME_OTFLOW,
                benchmark_family=FORECAST_FAMILY,
                dataset_key="solar_energy_10m",
                train_steps=8000,
                status="ready",
            )
            self.assertEqual(artifact["source_kind"], "matrix_output")
            self.assertEqual(artifact["checkpoint_budget_steps"], 8000)
            self.assertEqual(artifact["effective_train_steps"], 8000)
            self.assertEqual(artifact["checkpoint_export_protocol"], "exact_budget_step_state")
            missing_keys = {
                (row["benchmark_family"], row["dataset_key"], int(row["train_steps"]))
                for row in readiness["manifest"]["artifacts"]
                if row["status"] != "ready"
            }
            self.assertEqual(missing_keys, missing)

    def test_relative_metrics_respect_train_steps(self) -> None:
        rows = [
            {
                "benchmark_family": FORECAST_FAMILY,
                "split_phase": "locked_test",
                "scenario_key": "traffic_hourly",
                "backbone_name": "otflow",
                "checkpoint_id": "shared",
                "checkpoint_step": 20000,
                "train_budget_label": "20k",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "forecast_crps": 5.0,
                "experiment_scope": "main",
            },
            {
                "benchmark_family": FORECAST_FAMILY,
                "split_phase": "locked_test",
                "scenario_key": "traffic_hourly",
                "backbone_name": "otflow",
                "checkpoint_id": "shared",
                "checkpoint_step": 4000,
                "train_budget_label": "4k",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "flowts_power_sampling",
                "forecast_crps": 3.0,
                "experiment_scope": "main",
            },
            {
                "benchmark_family": FORECAST_FAMILY,
                "split_phase": "locked_test",
                "scenario_key": "traffic_hourly",
                "backbone_name": "otflow",
                "checkpoint_id": "shared",
                "checkpoint_step": 4000,
                "train_budget_label": "4k",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "forecast_crps": 4.0,
                "experiment_scope": "main",
            },
        ]
        enriched = augment_rows_with_relative_metrics(rows)
        by_schedule = {(row["checkpoint_step"], row["scheduler_key"]): row for row in enriched}
        self.assertIsNone(by_schedule[(4000, "flowts_power_sampling")]["relative_score_gain_vs_uniform"])
        self.assertAlmostEqual(by_schedule[(4000, "flowts_power_sampling")]["forecast_relative_crps_gain_vs_uniform"], 0.25)

    def test_conditional_generation_relative_metrics_preserve_seed_paired_gain(self) -> None:
        rows = [
            {
                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                "split_phase": "locked_test",
                "scenario_key": "cryptos",
                "backbone_name": "otflow",
                "checkpoint_id": "shared",
                "checkpoint_step": 20000,
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
                "scenario_key": "cryptos",
                "backbone_name": "otflow",
                "checkpoint_id": "shared",
                "checkpoint_step": 20000,
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
