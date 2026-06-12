import argparse
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from genode.evaluation import diffusion_flow_time_reparameterization as runner
from genode.evaluation.otflow_evaluation_support import CONDITIONAL_GENERATION_FAMILY, FORECAST_FAMILY
from genode.gipo import evaluate_schedule_summary
from genode.gipo.evaluate_schedule_summary import build_comparison_summary
from genode.gipo.objectives import (
    CONDITIONAL_METRIC_SPECS,
    FORECAST_METRIC_SPECS,
    MOLECULE_METRIC_SPECS,
    uniform_anchored_objective_columns,
)
from genode.gipo.policy import (
    GIPO_PROTOCOL,
    _scalarize_teacher_metric_values,
    _teacher_metric_targets,
    _teacher_metric_weights,
    context_embedding_id_from_row,
    read_metric_rows_csv,
    stable_context_id,
)
from genode.gipo import report_locked_test
from genode.pipeline import full_pipeline
from genode.gipo.train_gipo import (
    _merge_embedding_tables_guarded,
    _resolve_teacher_metric_target_keys,
    _validate_context_embedding_checkpoint_scope,
    _validate_support_group_counts,
    _validate_unique_schedule_rows,
)


class GenericContextGipoTests(unittest.TestCase):
    def test_context_schema_fields_are_family_neutral(self) -> None:
        fields = set(runner.CONTEXT_ROW_FIELDS)
        for field in (
            "context_schema",
            "axis_dataset",
            "axis_record",
            "axis_window",
            "axis_member",
            "axis_stratum",
            "axis_formula",
            "axis_atom_count",
            "axis_trajectory",
            "axis_iso_id",
            "u_score_uniform",
            "gipo_reward_protocol",
            "reward_anchor_schedule_key",
            "reward_utility_transform",
            "reward_granularity",
        ):
            self.assertIn(field, fields)
        parser = runner.build_argparser()
        args = parser.parse_args([])
        self.assertEqual(args.context_row_csv_name, "context_rows.csv")
        self.assertEqual(args.context_embeddings_npz_name, "context_embeddings.npz")
        self.assertFalse(args.write_context_rows)

    def test_conditional_context_rows_use_checkpoint_scoped_ids_and_component_utility(self) -> None:
        rows = runner._conditional_context_records(
            benchmark_family=CONDITIONAL_GENERATION_FAMILY,
            dataset="lobster_synthetic",
            split_phase="train_tuning",
            seed=7,
            evaluation_seed=17,
            solver_key="euler",
            target_nfe=4,
            runtime_nfe=4,
            scheduler_key="late_power_3",
            details={"time_grid": [1.0, 0.5, 0.0]},
            checkpoint={"checkpoint_id": "lobster_synthetic_4000_steps"},
            checkpoint_step=4000,
            nfe_role="seen",
            parent_row_signature="parent",
            protocol_hash="proto",
            cfg=SimpleNamespace(history_len=256),
            eval_horizon=200,
            chosen_t0s=[512, 768],
            score_main=0.75,
            uniform_score_main=1.0,
            per_window_metrics_by_t0={
                512: {
                    "score_main": 0.75,
                    "temporal_cw1": 0.5,
                    "temporal_uw1": 0.25,
                    "temporal_tstr_f1": 0.8,
                    "temporal_tstr_f1_applicable": True,
                },
                768: {
                    "score_main": 0.7,
                    "temporal_cw1": 0.4,
                    "temporal_uw1": 0.2,
                    "temporal_tstr_f1": 0.8,
                    "temporal_tstr_f1_applicable": True,
                },
            },
            uniform_per_window_metrics_by_t0={
                512: {
                    "score_main": 1.0,
                    "temporal_cw1": 1.0,
                    "temporal_uw1": 0.5,
                    "temporal_tstr_f1": 0.4,
                    "temporal_tstr_f1_applicable": True,
                },
                768: {
                    "score_main": 0.9,
                    "temporal_cw1": 0.8,
                    "temporal_uw1": 0.4,
                    "temporal_tstr_f1": 0.4,
                    "temporal_tstr_f1_applicable": True,
                },
            },
            metric_row={
                "scheduler_key": "late_power_3",
                "temporal_cw1": 0.5,
                "temporal_uw1": 0.25,
                "temporal_tstr_f1": 0.8,
                "temporal_tstr_f1_applicable": True,
            },
            uniform_metric_row={
                "scheduler_key": "uniform",
                "temporal_cw1": 1.0,
                "temporal_uw1": 0.5,
                "temporal_tstr_f1": 0.4,
                "temporal_tstr_f1_applicable": True,
            },
            evaluation_protocol_hash="protocol-hash",
            chosen_t0s_hash="t0-hash",
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["context_id"].startswith("lobster_synthetic_4000_steps:") for row in rows))
        self.assertTrue(all(row["context_id"] == row["context_embedding_id"] for row in rows))
        self.assertTrue(all(row["context_schema"] == "conditional_generation_window" for row in rows))
        self.assertAlmostEqual(float(rows[0]["u_score_uniform"]), math.log(1.0 / 0.75))
        self.assertAlmostEqual(float(rows[1]["u_score_uniform"]), math.log(0.9 / 0.7))
        self.assertTrue(all(math.isclose(float(row["u_temporal_tstr_f1_uniform"]), math.log(0.8 / 0.4)) for row in rows))
        self.assertTrue(all(float(row["u_comp_uniform"]) > 0.0 for row in rows))
        expected_raw_id = stable_context_id(
            dataset="lobster_synthetic",
            split_phase="train_tuning",
            example_idx=0,
            series_id="lobster_synthetic",
            series_idx=0,
            target_t=512,
            history_start=256,
            history_stop=512,
            context_schema="conditional_generation_window",
        )
        self.assertEqual(rows[0]["context_id"], f"lobster_synthetic_4000_steps:{expected_raw_id}")
        self.assertEqual(rows[0]["evaluation_protocol_hash"], "protocol-hash")
        self.assertEqual(rows[0]["chosen_examples_hash"], "t0-hash")
        self.assertEqual(rows[0]["gipo_reward_protocol"], GIPO_PROTOCOL)
        self.assertEqual(rows[0]["reward_anchor_schedule_key"], "uniform")
        self.assertEqual(rows[0]["reward_utility_transform"], "directional_log_uniform_anchor")
        self.assertEqual({row["axis_window"] for row in rows}, {"512", "768"})

    def test_conditional_context_rows_do_not_promote_score_gain_to_composite(self) -> None:
        with self.assertRaises(ValueError):
            runner._conditional_context_records(
                benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                dataset="lobster_synthetic",
                split_phase="train_tuning",
                seed=7,
                evaluation_seed=17,
                solver_key="euler",
                target_nfe=4,
                runtime_nfe=4,
                scheduler_key="late_power_3",
                details={"time_grid": [0.0, 0.5, 1.0]},
                checkpoint={"checkpoint_id": "lobster_synthetic_4000_steps"},
                checkpoint_step=4000,
                nfe_role="seen",
                parent_row_signature="parent",
                protocol_hash="proto",
                cfg=SimpleNamespace(history_len=256),
                eval_horizon=200,
                chosen_t0s=[512],
                score_main=0.75,
                uniform_score_main=1.0,
            )

    def test_directional_objective_columns_mix_lower_and_higher_metrics(self) -> None:
        row = {"scheduler_key": "late_power_3", "temporal_cw1": 0.5, "temporal_uw1": 1.0, "temporal_tstr_f1": 0.8, "temporal_tstr_f1_applicable": True}
        uniform = {"scheduler_key": "uniform", "temporal_cw1": 1.0, "temporal_uw1": 2.0, "temporal_tstr_f1": 0.4, "temporal_tstr_f1_applicable": True}
        cols = uniform_anchored_objective_columns(row, uniform, CONDITIONAL_METRIC_SPECS)
        self.assertAlmostEqual(float(cols["u_temporal_cw1_uniform"]), math.log(1.0 / 0.5))
        self.assertAlmostEqual(float(cols["u_temporal_uw1_uniform"]), math.log(2.0 / 1.0))
        self.assertAlmostEqual(float(cols["u_temporal_tstr_f1_uniform"]), math.log(0.8 / 0.4))
        uniform_cols = uniform_anchored_objective_columns(uniform, uniform, CONDITIONAL_METRIC_SPECS)
        self.assertEqual(float(uniform_cols["u_comp_uniform"]), 0.0)

    def test_molecule_context_records_attach_directional_rewards(self) -> None:
        uniform = {
            "context_id": "ctx",
            "context_embedding_id": "ckpt:ctx",
            "example_idx": 0,
            "target_t": 12,
            "history_start": 0,
            "history_stop": 12,
            "target_stop": 28,
            "axis_member": "member_a",
            "axis_stratum": "Dynamic_Test",
            "axis_formula": "CH",
            "axis_atom_count": 2,
            "axis_trajectory": "traj",
            "axis_iso_id": "1",
            "axis_window": "12",
            "axis_flags": "{}",
            "molecule_kabsch_rmsd_3d": 1.0,
            "molecule_ensemble_velocity_norm_w1": 1.0,
            "molecule_ensemble_acceleration_norm_w1": 1.0,
            "molecule_rollout_velocity_norm_w1": 1.0,
            "molecule_rollout_acceleration_norm_w1": 1.0,
        }
        candidate = dict(uniform)
        candidate.update(
            {
                "molecule_kabsch_rmsd_3d": 0.5,
                "molecule_ensemble_velocity_norm_w1": 0.5,
                "molecule_ensemble_acceleration_norm_w1": 0.5,
                "molecule_rollout_velocity_norm_w1": 0.5,
                "molecule_rollout_acceleration_norm_w1": 0.5,
            }
        )
        rows = runner._molecule_context_records(
            dataset="molecule_3d_set1",
            split_phase="train_tuning",
            seed=0,
            evaluation_seed=10,
            solver_key="euler",
            target_nfe=4,
            runtime_nfe=4,
            scheduler_key="late_power_3",
            details={"schedule_grid_hash": "grid"},
            checkpoint={"checkpoint_id": "ckpt"},
            checkpoint_step=4000,
            nfe_role="seen",
            parent_row_signature="parent",
            protocol_hash="proto",
            per_context_metrics=[candidate],
            uniform_by_context_id={"ckpt:ctx": dict(uniform, scheduler_key="uniform")},
            rollout_steps=16,
        )
        self.assertEqual(rows[0]["context_schema"], "molecule_3d_window")
        self.assertEqual(rows[0]["context_id"], "ckpt:ctx")
        self.assertEqual(rows[0]["axis_member"], "member_a")
        self.assertAlmostEqual(float(rows[0]["u_molecule_kabsch_rmsd_3d_uniform"]), math.log(1.0 / 0.5))
        self.assertAlmostEqual(float(rows[0]["u_comp_uniform"]), math.log(1.0 / 0.5))
        uniform_cols = uniform_anchored_objective_columns(dict(uniform, scheduler_key="uniform"), dict(uniform, scheduler_key="uniform"), MOLECULE_METRIC_SPECS)
        self.assertEqual(float(uniform_cols["u_comp_uniform"]), 0.0)

    def test_context_embedding_id_is_canonical_lookup_key(self) -> None:
        row = {"context_id": "logical", "context_embedding_id": "ckpt:logical"}
        self.assertEqual(context_embedding_id_from_row(row), "ckpt:logical")
        self.assertEqual(context_embedding_id_from_row({"context_id": "legacy"}), "legacy")

    def test_embedding_merge_rejects_colliding_vectors(self) -> None:
        base = {"ckpt:ctx": [0.0, 1.0]}
        _merge_embedding_tables_guarded(base, {"ckpt:ctx": [0.0, 1.0], "new": [2.0, 3.0]}, label="test")
        self.assertIn("new", base)
        with self.assertRaises(ValueError):
            _merge_embedding_tables_guarded(base, {"ckpt:ctx": [1.0, 0.0]}, label="test")

    def test_support_group_counts_require_uniform_anchor(self) -> None:
        row = {
            "dataset": "lobster_synthetic",
            "split_phase": "train_tuning",
            "seed": 0,
            "solver_key": "euler",
            "target_nfe": 4,
            "context_id": "ctx",
            "series_id": "series",
            "scheduler_key": "late_power_3",
        }
        with self.assertRaises(ValueError):
            _validate_support_group_counts([row], ["late_power_3"])

    def test_duplicate_exact_schedule_rows_are_rejected(self) -> None:
        row = {
            "dataset": "lobster_synthetic",
            "split_phase": "train_tuning",
            "seed": 0,
            "solver_key": "euler",
            "target_nfe": 4,
            "context_id": "ctx",
            "series_id": "series",
            "scheduler_key": "uniform",
        }
        with self.assertRaisesRegex(ValueError, "duplicate schedule rows"):
            _validate_unique_schedule_rows([row, dict(row)], label="test")

        other_checkpoint = dict(row, checkpoint_id="ckpt_b")
        base_checkpoint = dict(row, checkpoint_id="ckpt_a")
        _validate_unique_schedule_rows([base_checkpoint, other_checkpoint], label="test")

    def test_checkpoint_scoped_embedding_id_is_required_when_checkpoint_id_present(self) -> None:
        good = {"checkpoint_id": "ckpt_a", "context_id": "ctx", "context_embedding_id": "ckpt_a:ctx"}
        _validate_context_embedding_checkpoint_scope([good], label="test")
        bad = {"checkpoint_id": "ckpt_b", "context_id": "ctx", "context_embedding_id": "ckpt_a:ctx"}
        with self.assertRaisesRegex(ValueError, "checkpoint-scoped"):
            _validate_context_embedding_checkpoint_scope([bad], label="test")

    def test_duplicate_context_row_signatures_are_rejected_on_load(self) -> None:
        row = {field: "" for field in runner.CONTEXT_ROW_FIELDS}
        row.update({"row_signature": "sig", "context_id": "ctx", "context_embedding_id": "ckpt:ctx"})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "context_rows.csv"
            runner._write_context_row_csv(path, [row, dict(row)])
            with self.assertRaisesRegex(ValueError, "Duplicate context row signature"):
                runner._load_context_rows(path)
            with self.assertRaisesRegex(ValueError, "Duplicate context row signature"):
                evaluate_schedule_summary._load_context_rows(path)

    def test_masked_teacher_targets_and_scalarization_match_composite(self) -> None:
        row = {
            "context_id": "ctx",
            "scheduler_key": "late_power_3",
            "u_temporal_cw1_uniform": 0.2,
            "u_temporal_uw1_uniform": 0.6,
            "u_temporal_tstr_f1_uniform": "",
            "temporal_tstr_f1_applicable": "false",
            "reward_metric_weights_json": '{"u_temporal_cw1_uniform":0.5,"u_temporal_uw1_uniform":0.5}',
            "u_comp_uniform": 0.4,
        }
        target_keys = tuple(spec.utility_key for spec in CONDITIONAL_METRIC_SPECS[:3])
        values, mask = _teacher_metric_targets([row], target_keys=target_keys, device="cpu")
        self.assertEqual(mask.tolist(), [[1.0, 1.0, 0.0]])
        weights = _teacher_metric_weights([row], target_keys=target_keys, batch=1, device=torch.device("cpu"), dtype=torch.float32, target_mask=mask)
        scalar = _scalarize_teacher_metric_values(values, weights, target_keys=target_keys, target_mask=mask)
        self.assertAlmostEqual(float(scalar.item()), float(row["u_comp_uniform"]))

    def test_masked_teacher_scalarization_renormalizes_available_components(self) -> None:
        target_keys = ("u_temporal_cw1_uniform", "u_temporal_uw1_uniform")
        values = torch.tensor([[2.0, 10.0]], dtype=torch.float32)
        mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        raw_weights = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
        scalar = _scalarize_teacher_metric_values(values, raw_weights, target_keys=target_keys, target_mask=mask)
        self.assertAlmostEqual(float(scalar.item()), 2.0)

    def test_row_reward_weights_override_explicit_teacher_weights(self) -> None:
        row = {
            "context_id": "ctx",
            "scheduler_key": "late_power_3",
            "u_temporal_cw1_uniform": 1.0,
            "u_temporal_uw1_uniform": 3.0,
            "reward_metric_weights_json": '{"u_temporal_cw1_uniform":0.25,"u_temporal_uw1_uniform":0.75}',
        }
        target_keys = ("u_temporal_cw1_uniform", "u_temporal_uw1_uniform")
        values, mask = _teacher_metric_targets([row], target_keys=target_keys, device="cpu")
        weights = _teacher_metric_weights(
            [row],
            target_keys=target_keys,
            batch=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
            teacher_utility_weights={"u_temporal_cw1_uniform": 1.0, "u_temporal_uw1_uniform": 0.0},
            target_mask=mask,
        )
        scalar = _scalarize_teacher_metric_values(values, weights, target_keys=target_keys, target_mask=mask)
        self.assertAlmostEqual(float(scalar.item()), 2.5)

    def test_schedule_summary_context_embedding_collisions_are_rejected(self) -> None:
        embeddings = {"ckpt:ctx": [0.0, 1.0]}
        evaluate_schedule_summary._merge_context_embeddings_checked(embeddings, {"ckpt:ctx": [0.0, 1.0]})
        with self.assertRaisesRegex(ValueError, "Context embedding collision"):
            evaluate_schedule_summary._merge_context_embeddings_checked(embeddings, {"ckpt:ctx": [1.0, 0.0]})

    def test_explicit_teacher_target_keys_are_preserved(self) -> None:
        args = argparse.Namespace(teacher_metric_target_keys="u_crps_uniform,u_mase_uniform")
        rows = [
            {
                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                "scheduler_key": "uniform",
                "target_nfe": "4",
                "u_score_uniform": "0.0",
                "u_comp_uniform": "0.0",
            }
        ]
        self.assertEqual(_resolve_teacher_metric_target_keys(args, rows), ("u_crps_uniform", "u_mase_uniform"))

    def test_auto_teacher_target_keys_are_family_vectors_after_csv_roundtrip(self) -> None:
        args = argparse.Namespace(teacher_metric_target_keys="auto")
        row = {field: "" for field in runner.CONTEXT_ROW_FIELDS}
        row.update(
            {
                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                "dataset": "lobster_synthetic",
                "split_phase": "train_tuning",
                "scheduler_key": "uniform",
                "target_nfe": "4",
                "context_id": "ck:ctx",
                "context_embedding_id": "ck:ctx",
                "u_score_uniform": "0.0",
                "u_comp_uniform": "0.0",
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "context_rows.csv"
            runner._write_context_row_csv(path, [row])
            loaded = read_metric_rows_csv(path)
        self.assertEqual(_resolve_teacher_metric_target_keys(args, loaded), tuple(spec.utility_key for spec in CONDITIONAL_METRIC_SPECS))

    def test_auto_teacher_target_keys_cover_all_families(self) -> None:
        args = argparse.Namespace(teacher_metric_target_keys="auto")
        self.assertEqual(
            _resolve_teacher_metric_target_keys(args, [{"benchmark_family": FORECAST_FAMILY, "dataset": "solar_energy_10m"}]),
            tuple(spec.utility_key for spec in FORECAST_METRIC_SPECS),
        )
        self.assertEqual(
            _resolve_teacher_metric_target_keys(args, [{"benchmark_family": CONDITIONAL_GENERATION_FAMILY, "dataset": "lobster_synthetic"}]),
            tuple(spec.utility_key for spec in CONDITIONAL_METRIC_SPECS),
        )
        self.assertEqual(
            _resolve_teacher_metric_target_keys(args, [{"benchmark_family": "molecule_3d_coordinate_generation", "dataset": "molecule_3d_set1"}]),
            tuple(spec.utility_key for spec in MOLECULE_METRIC_SPECS),
        )

    def test_reporter_family_detection_and_conditional_aggregation(self) -> None:
        rows = [
            {
                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                "dataset": "lobster_synthetic",
                "seed": 1,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "gipo",
                "score_main": "0.8",
                "temporal_uw1": "0.2",
                "temporal_cw1": "0.3",
                "temporal_tstr_f1_applicable": "true",
            },
            {
                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                "dataset": "lobster_synthetic",
                "seed": 1,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "gipo",
                "score_main": "0.6",
                "temporal_uw1": "0.4",
                "temporal_cw1": "0.5",
                "temporal_tstr_f1_applicable": "true",
            },
        ]
        self.assertEqual(report_locked_test._benchmark_family_from_rows(rows), CONDITIONAL_GENERATION_FAMILY)
        aggregated = report_locked_test._aggregate_seed_rows(rows, split_phase="locked_test")
        self.assertEqual(len(aggregated), 1)
        self.assertAlmostEqual(float(aggregated[0]["score_main"]), 0.7)
        self.assertAlmostEqual(float(aggregated[0]["temporal_uw1"]), 0.3)
        with self.assertRaises(ValueError):
            report_locked_test._benchmark_family_from_rows([rows[0], {"benchmark_family": FORECAST_FAMILY}])

    def test_reporter_aggregates_molecule_metrics(self) -> None:
        rows = [
            {
                "benchmark_family": "molecule_3d_coordinate_generation",
                "dataset": "molecule_3d_set1",
                "seed": 1,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "gipo",
                "molecule_kabsch_rmsd_3d": "1.0",
                "molecule_rollout_velocity_norm_w1": "2.0",
            },
            {
                "benchmark_family": "molecule_3d_coordinate_generation",
                "dataset": "molecule_3d_set1",
                "seed": 1,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "gipo",
                "molecule_kabsch_rmsd_3d": "3.0",
                "molecule_rollout_velocity_norm_w1": "4.0",
            },
        ]
        aggregated = report_locked_test._aggregate_seed_rows(rows, split_phase="locked_test")
        self.assertEqual(len(aggregated), 1)
        self.assertAlmostEqual(float(aggregated[0]["molecule_kabsch_rmsd_3d"]), 2.0)
        self.assertAlmostEqual(float(aggregated[0]["molecule_rollout_velocity_norm_w1"]), 3.0)

    def test_family_aware_comparison_summary_handles_molecule_metrics(self) -> None:
        baseline = [
            {
                "benchmark_family": "molecule_3d_coordinate_generation",
                "dataset": "molecule_3d_set1",
                "split_phase": "locked_test",
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "uniform",
                "molecule_kabsch_rmsd_3d": 2.0,
                "molecule_rollout_velocity_norm_w1": 2.0,
                "u_comp_uniform": 0.0,
            }
        ]
        student = [
            {
                "benchmark_family": "molecule_3d_coordinate_generation",
                "dataset": "molecule_3d_set1",
                "split_phase": "locked_test",
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "gipo",
                "molecule_kabsch_rmsd_3d": 1.0,
                "molecule_rollout_velocity_norm_w1": 1.0,
                "u_comp_uniform": 0.5,
            }
        ]
        summary = build_comparison_summary(
            baseline_rows=baseline,
            student_rows=student,
            dataset="molecule_3d_set1",
            benchmark_family="molecule_3d_coordinate_generation",
            split_phase="locked_test",
            seeds=[0],
            solver_names=["euler"],
            target_nfe_values=[4],
        )
        ranking = summary["cell_rankings"][0]
        self.assertEqual(summary["benchmark_family"], "molecule_3d_coordinate_generation")
        self.assertEqual(ranking["metric_rankings"]["molecule_kabsch_rmsd_3d"], ["gipo", "uniform"])
        self.assertGreater(ranking["student_comparisons"][0]["student_molecule_kabsch_rmsd_3d_gain_vs_uniform"], 0.0)

    def test_full_pipeline_dry_run_accepts_molecule_scenario_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "molecule_3d_set1",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--molecule_group_root",
                    str(Path(tmpdir) / "missing_group_root"),
                    "--stages",
                    "backbone_training,schedule_rows_seen,locked_test_reports",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
        commands = [cmd for stage in summary["stages"] for cmd in stage["commands"]]
        self.assertTrue(any("--molecule_datasets" in command for command in commands))
        self.assertTrue(any("--allow_execute" in command for command in commands))
        self.assertTrue(any("genode.gipo.report_locked_test" in " ".join(command) for command in commands))
        self.assertFalse(any("report_locked_test" in command for command in commands if command[:1] == ["internal"]))
        self.assertEqual(summary["status"], "dry_run")

    def test_full_pipeline_zero_shot_stage_does_not_use_unseen_selection_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--stages",
                    "gipo_student_seen_only_zero_shot,gipo_student_seen_plus_unseen_pseudo",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
        by_stage = {stage["stage"]: [" ".join(command) for command in stage["commands"]] for stage in summary["stages"]}
        zero_shot = " ".join(by_stage["gipo_student_seen_only_zero_shot"])
        pseudo = " ".join(by_stage["gipo_student_seen_plus_unseen_pseudo"])
        self.assertNotIn("--teacher_unseen_selection_rows_csv", zero_shot)
        self.assertNotIn("--teacher_unseen_selection_context_embeddings_npz", zero_shot)
        self.assertIn("--teacher_metric_target_keys u_temporal_cw1_uniform,u_temporal_uw1_uniform,u_temporal_tstr_f1_uniform", zero_shot)
        self.assertIn("--student_pseudo_rows_csv", pseudo)
        self.assertIn("--teacher_metric_target_keys u_temporal_cw1_uniform,u_temporal_uw1_uniform,u_temporal_tstr_f1_uniform", pseudo)
        self.assertIn("--teacher_utility_weights", pseudo)

    def _dry_run_gipo_commands_for_scenario(self, scenario_key: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    scenario_key,
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--stages",
                    "gipo_student_seen_only_zero_shot,gipo_student_seen_plus_unseen_pseudo",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
        return " ".join(" ".join(command) for stage in summary["stages"] for command in stage["commands"])

    def test_full_pipeline_forecast_uses_forecast_teacher_target_vector(self) -> None:
        commands = self._dry_run_gipo_commands_for_scenario("solar_energy_10m")
        self.assertIn("--teacher_metric_target_keys u_crps_uniform,u_mase_uniform", commands)
        self.assertNotIn("--teacher_metric_target_keys u_comp_uniform", commands)

    def test_full_pipeline_conditional_uses_conditional_teacher_target_vector(self) -> None:
        commands = self._dry_run_gipo_commands_for_scenario("lobster_synthetic")
        expected = ",".join(spec.utility_key for spec in CONDITIONAL_METRIC_SPECS)
        self.assertIn(f"--teacher_metric_target_keys {expected}", commands)
        self.assertIn("u_temporal_cw1_uniform=0.4", commands)
        self.assertNotIn("--teacher_metric_target_keys u_comp_uniform", commands)

    def test_full_pipeline_molecule_uses_molecule_teacher_target_vector(self) -> None:
        commands = self._dry_run_gipo_commands_for_scenario("molecule_3d_set1")
        expected = ",".join(spec.utility_key for spec in MOLECULE_METRIC_SPECS)
        self.assertIn(f"--teacher_metric_target_keys {expected}", commands)
        self.assertIn("u_molecule_kabsch_rmsd_3d_uniform=0.4", commands)
        self.assertNotIn("--teacher_metric_target_keys u_comp_uniform", commands)


if __name__ == "__main__":
    unittest.main()
