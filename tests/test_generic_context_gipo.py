import argparse
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from genode.evaluation import diffusion_flow_time_reparameterization as runner
from genode.evaluation.otflow_evaluation_support import CONDITIONAL_GENERATION_FAMILY, FORECAST_FAMILY
from genode.gipo import evaluate_schedule_summary
from genode.gipo.evaluate_schedule_summary import build_comparison_summary
from genode.gipo.objectives import (
    CONDITIONAL_METRIC_SPECS,
    CONDITIONAL_PRIMARY_ECG_METRIC_SPECS,
    CONDITIONAL_PRIMARY_LOB_METRIC_SPECS,
    FORECAST_METRIC_SPECS,
    MOLECULE_METRIC_SPECS,
    teacher_metric_profile_for_scenario,
    uniform_anchored_objective_columns,
)
from genode.gipo.policy import (
    GIPO_PROTOCOL,
    _scalarize_teacher_metric_values,
    _teacher_metric_targets,
    _teacher_metric_weights,
    checkpoint_scope_from_row,
    context_embedding_id_from_row,
    context_pair_key,
    read_metric_rows_csv,
    save_context_embedding_table,
    split_rows_by_context_holdout,
    stable_context_id,
)
from genode.gipo import report_locked_test
from genode.gipo.ablation_plan import (
    GIPO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX,
    gipo_ablation_arms,
)
from genode.pipeline import full_pipeline
from genode.gipo.train_gipo import (
    _context_sampling_summary,
    _merge_embedding_tables_guarded,
    _resolve_teacher_metric_target_keys,
    _rows_for_context_keys,
    _sample_context_keys_by_checkpoint,
    _split_membership_summary,
    _validate_context_embedding_checkpoint_scope,
    _validate_support_group_counts,
    _validate_unique_schedule_rows,
    train_gipo,
)
from genode.gipo.preflight import _context_identity_fingerprint


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
            "effective_train_steps",
            "checkpoint_export_protocol",
            "u_l1",
            "c_l1",
            "spread_specific_error",
            "imbalance_specific_error",
            "ret_vol_acf_error",
            "impact_response_error",
        ):
            self.assertIn(field, fields)

    def test_context_artifact_arg_defaults(self) -> None:
        parser = runner.build_argparser()
        args = parser.parse_args([])
        self.assertEqual(args.context_row_csv_name, "context_rows.csv")
        self.assertEqual(args.context_embeddings_npz_name, "context_embeddings.npz")
        self.assertFalse(args.write_context_rows)

    def test_locked_report_filters_shared_context_rows_to_baseline_and_ser_sets(self) -> None:
        rows = [
            {"scheduler_key": "uniform"},
            {"scheduler_key": "late_power_3"},
            {"scheduler_key": "ser_ptg_local_defect_eta005"},
            {"scheduler_key": "gipo"},
        ]
        baseline_rows = report_locked_test._filter_rows_to_schedule_keys(rows, ["uniform", "late_power_3"])
        ser_rows = report_locked_test._filter_rows_to_schedule_keys(rows, ["ser_ptg_local_defect_eta005"])

        self.assertEqual([row["scheduler_key"] for row in baseline_rows], ["uniform", "late_power_3"])
        self.assertEqual([row["scheduler_key"] for row in ser_rows], ["ser_ptg_local_defect_eta005"])

    def test_forecast_pseudo_rows_are_reward_materialized_before_student_distillation(self) -> None:
        def write_rows(path: Path, target_nfes: tuple[int, ...]) -> None:
            schedules = {
                "uniform": (2.0, 2.0),
                "late_power_3": (1.0, 1.0),
                "flowts_power_sampling": (1.5, 1.5),
                "ays": (1.4, 1.4),
            }
            fields = [
                "benchmark_family",
                "dataset",
                "split_phase",
                "seed",
                "solver_key",
                "target_nfe",
                "scheduler_key",
                "context_id",
                "series_id",
                "target_t",
                "forecast_crps",
                "forecast_mase",
            ]
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                for ctx_idx in range(3):
                    for nfe in target_nfes:
                        for scheduler_key, (crps, mase) in schedules.items():
                            writer.writerow(
                                {
                                    "benchmark_family": FORECAST_FAMILY,
                                    "dataset": "solar_energy_10m",
                                    "split_phase": "train_tuning",
                                    "seed": 0,
                                    "solver_key": "euler",
                                    "target_nfe": nfe,
                                    "scheduler_key": scheduler_key,
                                    "context_id": f"ctx_{ctx_idx}",
                                    "series_id": f"series_{ctx_idx}",
                                    "target_t": 100 + ctx_idx,
                                    "forecast_crps": crps,
                                    "forecast_mase": mase,
                                }
                            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seen_rows = root / "seen.csv"
            pseudo_rows = root / "pseudo.csv"
            embeddings = root / "context_embeddings.npz"
            write_rows(seen_rows, (4, 8, 12, 16))
            write_rows(pseudo_rows, (6, 10, 14, 20))
            save_context_embedding_table(
                embeddings,
                {"ctx_0": [0.0, 1.0], "ctx_1": [1.0, 0.0], "ctx_2": [0.5, 0.5]},
            )
            args = SimpleNamespace(
                rows_csv=str(seen_rows),
                context_embeddings_npz=str(embeddings),
                schedule_summary_json="",
                out_dir=str(root / "out"),
                support_schedule_keys="uniform,late_power_3,flowts_power_sampling,ays",
                context_sample_count=3,
                context_holdout_fraction=0.5,
                teacher_steps=1,
                teacher_checkpoint_every=1,
                teacher_loss_log_every=0,
                teacher_density_holdout_schedule_keys="flowts_power_sampling,ays",
                teacher_unseen_selection_rows_csv="",
                teacher_unseen_selection_context_embeddings_npz="",
                teacher_unseen_selection_schedule_summary_json="",
                teacher_unseen_selection_target_nfe_values="6,10,14,20",
                student_pseudo_rows_csv=str(pseudo_rows),
                student_pseudo_context_embeddings_npz=str(embeddings),
                student_pseudo_schedule_summary_json="",
                student_pseudo_target_weight=0.25,
                student_steps=1,
                student_log_every=0,
                student_checkpoint_every=1,
                student_selection_holdout_fraction=0.5,
                teacher_lr=1e-3,
                student_lr=1e-3,
                transformer_hidden_dim=16,
                transformer_layers=1,
                transformer_heads=4,
                transformer_dropout=0.0,
                teacher_temperature=0.05,
                teacher_utility_crps_weight=0.5,
                teacher_utility_mase_weight=0.5,
                teacher_metric_target_keys="",
                teacher_utility_weights="",
                student_weight_decay=0.0,
                seed=0,
                device="cpu",
                dry_run=False,
            )

            summary = train_gipo(args)

        self.assertTrue(summary["student_pseudo_distillation"]["enabled"])
        self.assertEqual(summary["student_objective_settings"]["student_teacher_score_weight"], 0.05)
        self.assertEqual(summary["student_objective_settings"]["student_teacher_score_warmup_fraction"], 0.6)
        self.assertEqual(summary["student_objective_settings"]["student_teacher_score_schedule_steps"], 1)
        self.assertEqual(summary["student_objective_settings"]["student_teacher_score_clip"], 5.0)
        self.assertEqual(
            summary["student_objective_settings"]["student_teacher_score_protocol"],
            "late_ramped_per_cell_teacher_utility_z_score",
        )
        self.assertEqual(summary["student_objective_settings"]["student_target_mixture_mode"], "full")
        self.assertFalse(summary["student_objective_settings"]["student_teacher_score_include_pseudo"])
        self.assertFalse(summary["student_objective_settings"]["student_regularizers"]["smooth"])
        self.assertFalse(summary["student_objective_settings"]["student_regularizers"]["guard"])
        pseudo_summary = summary["student_training"]["student_pseudo_target_summary"]
        self.assertTrue(pseudo_summary["pseudo_distillation_used"])
        self.assertEqual(pseudo_summary["pseudo_target_nfes"], [6, 10, 14, 20])
        self.assertEqual(pseudo_summary["student_target_mixture_mode"], "full")

    def test_conditional_context_rows_use_physical_ids_and_checkpoint_scoped_embeddings(self) -> None:
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
        self.assertTrue(all(not row["context_id"].startswith("lobster_synthetic_4000_steps:") for row in rows))
        self.assertTrue(all(row["context_embedding_id"].startswith("lobster_synthetic_4000_steps:") for row in rows))
        self.assertTrue(all(row["context_id"] != row["context_embedding_id"] for row in rows))
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
        self.assertEqual(rows[0]["context_id"], expected_raw_id)
        self.assertEqual(rows[0]["context_embedding_id"], f"lobster_synthetic_4000_steps:{expected_raw_id}")
        self.assertEqual(rows[0]["evaluation_protocol_hash"], "protocol-hash")
        self.assertEqual(rows[0]["chosen_examples_hash"], "t0-hash")
        self.assertEqual(rows[0]["gipo_reward_protocol"], GIPO_PROTOCOL)
        self.assertEqual(rows[0]["reward_anchor_schedule_key"], "uniform")
        self.assertEqual(rows[0]["reward_utility_transform"], "directional_log_uniform_anchor")
        reward_weights = json.loads(str(rows[0]["reward_metric_weights_json"]))
        self.assertEqual(
            set(reward_weights),
            {"u_temporal_uw1_uniform", "u_temporal_cw1_uniform", "u_temporal_tstr_f1_uniform"},
        )
        self.assertEqual({row["axis_window"] for row in rows}, {"512", "768"})

    def test_conditional_context_rows_use_aggregate_primary_metrics_not_score_gain(self) -> None:
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
            metric_row={
                "temporal_uw1": 0.5,
                "temporal_cw1": 0.25,
                "temporal_tstr_f1": 0.8,
                "temporal_tstr_f1_applicable": True,
            },
            uniform_metric_row={
                "temporal_uw1": 1.0,
                "temporal_cw1": 0.5,
                "temporal_tstr_f1": 0.4,
                "temporal_tstr_f1_applicable": True,
            },
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["u_score_uniform"]), math.log(1.0 / 0.75))
        self.assertAlmostEqual(float(rows[0]["u_comp_uniform"]), math.log(2.0))
        self.assertNotAlmostEqual(float(rows[0]["u_comp_uniform"]), float(rows[0]["u_score_uniform"]))

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
            uniform_by_context_id={"ctx": dict(uniform, scheduler_key="uniform")},
            rollout_steps=16,
        )
        self.assertEqual(rows[0]["context_schema"], "molecule_3d_window")
        self.assertEqual(rows[0]["context_id"], "ctx")
        self.assertEqual(rows[0]["context_embedding_id"], "ckpt:ctx")
        self.assertEqual(rows[0]["axis_member"], "member_a")
        self.assertAlmostEqual(float(rows[0]["u_molecule_kabsch_rmsd_3d_uniform"]), math.log(1.0 / 0.5))
        self.assertAlmostEqual(float(rows[0]["u_comp_uniform"]), math.log(1.0 / 0.5))
        uniform_cols = uniform_anchored_objective_columns(dict(uniform, scheduler_key="uniform"), dict(uniform, scheduler_key="uniform"), MOLECULE_METRIC_SPECS)
        self.assertEqual(float(uniform_cols["u_comp_uniform"]), 0.0)

    def test_context_embedding_id_is_canonical_lookup_key(self) -> None:
        row = {"context_id": "logical", "context_embedding_id": "ckpt:logical"}
        self.assertEqual(context_embedding_id_from_row(row), "ckpt:logical")
        self.assertEqual(context_embedding_id_from_row({"context_id": "legacy"}), "legacy")

    def test_preflight_physical_context_fingerprint_ignores_checkpoint_embedding_id(self) -> None:
        base = {
            "context_schema": "forecast_window",
            "dataset": "solar_energy_10m",
            "split_phase": "train_tuning",
            "axis_series": "series_1",
            "axis_time_bin": "144",
            "example_idx": "7",
            "target_t": "144",
            "history_start": "0",
            "history_stop": "144",
        }
        first = {**base, "context_embedding_id": "ckpt_a:ctx"}
        second = {**base, "context_embedding_id": "ckpt_b:ctx"}

        self.assertEqual(_context_identity_fingerprint(first), _context_identity_fingerprint(second))

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

    def test_support_group_counts_are_checkpoint_aware_for_physical_context_ids(self) -> None:
        rows = []
        for checkpoint_id in ("ckpt_a", "ckpt_b"):
            for scheduler_key in ("uniform", "late_power_3"):
                rows.append(
                    {
                        "dataset": "lobster_synthetic",
                        "split_phase": "train_tuning",
                        "seed": 0,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "context_id": "physical_ctx",
                        "context_embedding_id": f"{checkpoint_id}:physical_ctx",
                        "checkpoint_id": checkpoint_id,
                        "series_id": "series",
                        "scheduler_key": scheduler_key,
                    }
                )

        _validate_support_group_counts(rows, ["uniform", "late_power_3"])

        self.assertEqual(len({context_pair_key(row, pair_on_seed=True) for row in rows}), 2)

    def test_context_sampling_is_per_checkpoint_maturity(self) -> None:
        rows = []
        for checkpoint_id in ("ckpt_a", "ckpt_b"):
            for context_idx in range(3):
                for scheduler_key in ("uniform", "late_power_3"):
                    rows.append(
                        {
                            "dataset": "lobster_synthetic",
                            "split_phase": "train_tuning",
                            "seed": 0,
                            "solver_key": "euler",
                            "target_nfe": 4,
                            "context_id": f"ctx_{context_idx}",
                            "context_embedding_id": f"{checkpoint_id}:ctx_{context_idx}",
                            "checkpoint_id": checkpoint_id,
                            "series_id": f"series_{context_idx}",
                            "target_t": 100 + context_idx,
                            "scheduler_key": scheduler_key,
                        }
                    )

        selected = _sample_context_keys_by_checkpoint(rows, sample_count=1, seed=17)
        sampled_rows = _rows_for_context_keys(rows, selected)
        summary = _context_sampling_summary(rows, selected, sample_count=1)

        self.assertEqual(len(selected), 2)
        self.assertEqual({scope for scope, _ in selected}, {"ckpt_a", "ckpt_b"})
        self.assertEqual(len(sampled_rows), 4)
        self.assertEqual(summary["per_checkpoint"]["ckpt_a"]["selected_contexts"], 1)
        self.assertEqual(summary["per_checkpoint"]["ckpt_b"]["selected_contexts"], 1)

    def test_context_holdout_is_physical_across_checkpoint_maturities(self) -> None:
        rows = []
        for checkpoint_id in ("ckpt_a", "ckpt_b"):
            for context_id in ("ctx_a", "ctx_b", "ctx_c", "ctx_d"):
                rows.append(
                    {
                        "dataset": "lobster_synthetic",
                        "split_phase": "train_tuning",
                        "seed": 0,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "context_id": context_id,
                        "context_embedding_id": f"{checkpoint_id}:{context_id}",
                        "checkpoint_id": checkpoint_id,
                        "series_id": context_id,
                        "target_t": 100,
                        "scheduler_key": "uniform",
                    }
                )

        fit_rows, holdout_rows = split_rows_by_context_holdout(rows, holdout_fraction=0.5, seed=3)

        fit_contexts = {row["context_id"] for row in fit_rows}
        holdout_contexts = {row["context_id"] for row in holdout_rows}
        self.assertFalse(fit_contexts & holdout_contexts)
        for context_id in holdout_contexts:
            self.assertEqual({row["checkpoint_id"] for row in holdout_rows if row["context_id"] == context_id}, {"ckpt_a", "ckpt_b"})

    def test_preflight_identity_fingerprint_ignores_checkpoint_scoped_embedding_id(self) -> None:
        row = {
            "dataset": "lobster_synthetic",
            "split_phase": "train_tuning",
            "context_schema": "conditional_generation_window",
            "context_id": "ctx",
            "series_id": "lobster_synthetic",
            "target_t": 512,
            "history_start": 256,
            "history_stop": 512,
        }

        fp_a = _context_identity_fingerprint({**row, "checkpoint_id": "ckpt_a", "context_embedding_id": "ckpt_a:ctx"})
        fp_b = _context_identity_fingerprint({**row, "checkpoint_id": "ckpt_b", "context_embedding_id": "ckpt_b:ctx"})

        self.assertEqual(fp_a, fp_b)

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

    def test_checkpoint_step_only_rows_remain_checkpoint_aware(self) -> None:
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
        early = dict(row, checkpoint_step=4000)
        late = dict(row, checkpoint_step=8000)

        self.assertNotEqual(context_pair_key(early, pair_on_seed=True), context_pair_key(late, pair_on_seed=True))
        _validate_unique_schedule_rows([early, late], label="test")
        self.assertEqual(checkpoint_scope_from_row(dict(row, checkpoint_step="", train_steps=4000)), "checkpoint_step:4000")
        self.assertEqual(checkpoint_scope_from_row(dict(row, checkpoint_step="", train_steps="", otflow_train_steps=8000)), "checkpoint_step:8000")

        train_early = dict(row, train_steps=4000)
        train_late = dict(row, train_steps=8000)
        otflow_late = dict(row, otflow_train_steps=12000)
        self.assertNotEqual(context_pair_key(train_early, pair_on_seed=True), context_pair_key(train_late, pair_on_seed=True))
        self.assertNotEqual(context_pair_key(train_late, pair_on_seed=True), context_pair_key(otflow_late, pair_on_seed=True))
        _validate_unique_schedule_rows([train_early, train_late, otflow_late], label="test")
        split_summary = _split_membership_summary([train_early, train_late, otflow_late])
        self.assertEqual(split_summary["checkpoint_scope_count"], 3)
        self.assertEqual(
            set(split_summary["checkpoint_scopes"]),
            {"checkpoint_step:4000", "checkpoint_step:8000", "checkpoint_step:12000"},
        )

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

    def test_auto_teacher_target_keys_are_scenario_primary_vectors_after_csv_roundtrip(self) -> None:
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
        self.assertEqual(
            _resolve_teacher_metric_target_keys(args, loaded),
            tuple(spec.utility_key for spec in CONDITIONAL_PRIMARY_LOB_METRIC_SPECS),
        )

    def test_auto_teacher_target_keys_cover_all_families(self) -> None:
        args = argparse.Namespace(teacher_metric_target_keys="auto")
        self.assertEqual(
            _resolve_teacher_metric_target_keys(args, [{"benchmark_family": FORECAST_FAMILY, "dataset": "solar_energy_10m"}]),
            tuple(spec.utility_key for spec in FORECAST_METRIC_SPECS),
        )
        self.assertEqual(
            _resolve_teacher_metric_target_keys(args, [{"benchmark_family": CONDITIONAL_GENERATION_FAMILY, "dataset": "lobster_synthetic"}]),
            tuple(spec.utility_key for spec in CONDITIONAL_PRIMARY_LOB_METRIC_SPECS),
        )
        self.assertEqual(
            _resolve_teacher_metric_target_keys(args, [{"benchmark_family": CONDITIONAL_GENERATION_FAMILY, "dataset": "long_term_st"}]),
            tuple(spec.utility_key for spec in CONDITIONAL_PRIMARY_ECG_METRIC_SPECS),
        )
        self.assertEqual(
            _resolve_teacher_metric_target_keys(args, [{"benchmark_family": "molecule_3d_coordinate_generation", "dataset": "molecule_3d_set1"}]),
            tuple(spec.utility_key for spec in MOLECULE_METRIC_SPECS),
        )

    def test_teacher_metric_profiles_canonicalize_conditional_scenarios(self) -> None:
        lob = teacher_metric_profile_for_scenario("cryptos")
        self.assertEqual(
            lob["target_utility_keys"],
            ["u_temporal_uw1_uniform", "u_temporal_cw1_uniform", "u_temporal_tstr_f1_uniform"],
        )
        self.assertTrue(all(math.isclose(float(value), 1.0 / 3.0) for value in lob["target_weights"].values()))
        self.assertIn("spread_specific_error", lob["diagnostic_metric_keys"])

        ecg = teacher_metric_profile_for_scenario("long_term_st")
        self.assertEqual(ecg["target_utility_keys"], ["u_temporal_uw1_uniform", "u_temporal_cw1_uniform"])
        self.assertEqual(ecg["target_weights"], {"u_temporal_uw1_uniform": 0.5, "u_temporal_cw1_uniform": 0.5})

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

    def test_locked_reporter_infers_manual_matrix_selectors_from_context_rows(self) -> None:
        rows = [
            {
                "dataset": "lobster_synthetic",
                "evaluation_seed": "1",
                "solver_key": "rk2",
                "target_nfe": "6",
            },
            {
                "dataset": "lobster_synthetic",
                "evaluation_seed": "0",
                "solver_key": "euler",
                "target_nfe": "4",
            },
        ]

        self.assertEqual(
            report_locked_test._single_value_from_rows(rows, field="dataset", arg_name="dataset"),
            "lobster_synthetic",
        )
        self.assertEqual(
            report_locked_test._infer_int_values_from_rows(rows, field="evaluation_seed", arg_name="seeds"),
            (0, 1),
        )
        self.assertEqual(report_locked_test._infer_solver_values_from_rows(rows), ("euler", "heun"))
        self.assertEqual(
            report_locked_test._infer_int_values_from_rows(rows, field="target_nfe", arg_name="target_nfe_values"),
            (4, 6),
        )
        with self.assertRaisesRegex(ValueError, "not present"):
            report_locked_test._single_value_from_rows(rows, field="dataset", requested="traffic_hourly", arg_name="dataset")

    def test_locked_reporter_representatives_preserve_explicit_evaluation_seed_and_filter_matrix(self) -> None:
        rows = [
            {
                "dataset": "lobster_synthetic",
                "seed": "0",
                "evaluation_seed": "11",
                "solver_key": "euler",
                "target_nfe": "4",
                "scheduler_key": "uniform",
                "context_id": "ctx_a",
                "source_split_phase": "locked_test",
            },
            {
                "dataset": "lobster_synthetic",
                "seed": "0",
                "evaluation_seed": "22",
                "solver_key": "euler",
                "target_nfe": "8",
                "scheduler_key": "uniform",
                "context_id": "ctx_b",
                "source_split_phase": "locked_test",
            },
        ]

        representatives = report_locked_test._representative_context_rows(rows)
        filtered = report_locked_test._filter_representatives_to_matrix(
            representatives,
            dataset="lobster_synthetic",
            seeds=[11],
            solvers=["euler"],
            target_nfes=[4],
        )

        self.assertEqual([row["evaluation_seed"] for row in representatives], [11, 22])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["context_id"], "ctx_a")

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

    def test_conditional_comparison_summary_uses_primary_metrics_and_tstr_high_direction(self) -> None:
        baseline = [
            {
                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                "dataset": "lobster_synthetic",
                "split_phase": "locked_test",
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "uniform",
                "temporal_uw1": 1.0,
                "temporal_cw1": 1.0,
                "temporal_tstr_f1": 0.5,
                "u_comp_uniform": 0.0,
            }
        ]
        student = [
            {
                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                "dataset": "lobster_synthetic",
                "split_phase": "locked_test",
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "gipo",
                "temporal_uw1": 0.5,
                "temporal_cw1": 0.5,
                "temporal_tstr_f1": 0.8,
                "u_comp_uniform": 0.4,
            }
        ]
        summary = build_comparison_summary(
            baseline_rows=baseline,
            student_rows=student,
            dataset="lobster_synthetic",
            benchmark_family=CONDITIONAL_GENERATION_FAMILY,
            split_phase="locked_test",
            seeds=[0],
            solver_names=["euler"],
            target_nfe_values=[4],
        )
        ranking = summary["cell_rankings"][0]
        self.assertEqual(summary["metric_keys"], ["temporal_uw1", "temporal_cw1", "temporal_tstr_f1", "u_comp_uniform"])
        self.assertEqual(ranking["metric_rankings"]["temporal_tstr_f1"], ["gipo", "uniform"])
        self.assertGreater(ranking["student_comparisons"][0]["student_temporal_tstr_f1_gain_vs_uniform"], 0.0)

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

    def test_full_pipeline_requests_exact_budget_temporal_backbones(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "cryptos",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--stages",
                    "backbone_training",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
        commands = [" ".join(cmd) for stage in summary["stages"] for cmd in stage["commands"]]
        backbone_command = next(command for command in commands if "genode.training.train_backbone" in command)
        self.assertIn("--checkpoint_export_mode exact_budget", backbone_command)

    def test_full_pipeline_passes_backbone_manifest_to_molecule_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "molecule_3d_set1",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--molecule_group_root",
                    str(Path(tmpdir) / "groups"),
                    "--molecule_backbone_root",
                    str(Path(tmpdir) / "molecule_backbones"),
                    "--backbone_manifest",
                    str(Path(tmpdir) / "matrix" / "backbone_manifest.json"),
                    "--checkpoint_steps",
                    "4000,8000",
                ]
            )
            member = {"stratum": "Dynamic_A", "processed_dir": "Dynamic_A", "trainable": True}
            with mock.patch.object(full_pipeline, "load_molecule_group_manifest", return_value={"strata": [member]}):
                commands = full_pipeline._backbone_training_commands(args, "molecule_3d_set1", "4000,8000")

        command = " ".join(str(part) for part in commands[0])
        self.assertIn("--backbone_manifest", command)
        self.assertIn(str(Path(tmpdir) / "matrix" / "backbone_manifest.json"), command)

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
        for command in (zero_shot, pseudo):
            self.assertIn("--teacher_metric_target_keys u_temporal_uw1_uniform,u_temporal_cw1_uniform,u_temporal_tstr_f1_uniform", command)
            self.assertIn("--teacher_steps 500", command)
            self.assertIn("--student_steps 500", command)
            self.assertIn("--seen_target_nfe_values 4,8,12,16", command)
            self.assertIn("--pseudo_target_nfe_values 6,10,14,20", command)
            self.assertIn("--student_teacher_score_weight 0.05", command)
            self.assertIn("--student_teacher_score_warmup_fraction 0.6", command)
            self.assertIn("--student_target_mixture_mode full", command)
            self.assertIn("--student_target_elite_fraction 0.3", command)
            self.assertIn("--student_target_elite_k 0", command)
            self.assertIn("--student_target_elite_min_count 2", command)
            self.assertIn("--student_target_elite_blend_all_weight 0.2", command)
        self.assertIn("--student_pseudo_rows_csv", pseudo)
        self.assertNotIn("--student_teacher_score_include_pseudo", pseudo)
        self.assertIn("--teacher_utility_weights", pseudo)
        protocol = full_pipeline._protocol_payload(args)
        self.assertEqual(protocol["gipo_teacher_steps"], 500)
        self.assertEqual(protocol["gipo_student_steps"], 500)
        self.assertEqual(protocol["student_teacher_score_weight"], 0.05)
        self.assertEqual(protocol["student_teacher_score_clip"], 5.0)
        self.assertEqual(protocol["student_teacher_score_protocol"], "late_ramped_per_cell_teacher_utility_z_score")
        self.assertEqual(protocol["student_target_mixture_mode"], "full")

    def test_full_pipeline_gipo_step_budgets_are_protocolized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run_a"),
                    "--stages",
                    "gipo_student_seen_only_zero_shot",
                    "--dry_run",
                ]
            )
            changed_args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run_b"),
                    "--stages",
                    "gipo_student_seen_only_zero_shot",
                    "--gipo_teacher_steps",
                    "501",
                    "--gipo_student_steps",
                    "503",
                    "--dry_run",
                ]
            )

        base_protocol = full_pipeline._protocol_payload(base_args)
        changed_protocol = full_pipeline._protocol_payload(changed_args)
        self.assertEqual(base_protocol["gipo_teacher_steps"], 500)
        self.assertEqual(base_protocol["gipo_student_steps"], 500)
        self.assertEqual(changed_protocol["gipo_teacher_steps"], 501)
        self.assertEqual(changed_protocol["gipo_student_steps"], 503)
        self.assertNotEqual(full_pipeline._json_hash(base_protocol), full_pipeline._json_hash(changed_protocol))

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
        expected = ",".join(spec.utility_key for spec in CONDITIONAL_PRIMARY_LOB_METRIC_SPECS)
        self.assertIn(f"--teacher_metric_target_keys {expected}", commands)
        self.assertIn("u_temporal_uw1_uniform=0.333333", commands)
        self.assertIn("u_temporal_cw1_uniform=0.333333", commands)
        self.assertIn("u_temporal_tstr_f1_uniform=0.333333", commands)
        self.assertNotIn("u_temporal_u_l1_uniform=", commands)
        self.assertNotIn("--teacher_metric_target_keys u_comp_uniform", commands)

    def test_full_pipeline_ecg_uses_two_metric_teacher_target_vector(self) -> None:
        commands = self._dry_run_gipo_commands_for_scenario("long_term_st")
        expected = ",".join(spec.utility_key for spec in CONDITIONAL_PRIMARY_ECG_METRIC_SPECS)
        self.assertIn(f"--teacher_metric_target_keys {expected}", commands)
        self.assertIn("u_temporal_uw1_uniform=0.5", commands)
        self.assertIn("u_temporal_cw1_uniform=0.5", commands)
        self.assertNotIn("u_temporal_tstr_f1_uniform=", commands)
        self.assertNotIn("u_temporal_u_l1_uniform=", commands)

    def test_full_pipeline_molecule_uses_molecule_teacher_target_vector(self) -> None:
        commands = self._dry_run_gipo_commands_for_scenario("molecule_3d_set1")
        expected = ",".join(spec.utility_key for spec in MOLECULE_METRIC_SPECS)
        self.assertIn(f"--teacher_metric_target_keys {expected}", commands)
        self.assertIn("u_molecule_kabsch_rmsd_3d_uniform=0.4", commands)
        self.assertNotIn("--teacher_metric_target_keys u_comp_uniform", commands)

    def test_gipo_ablation_preset_covers_main_and_appendix_arms(self) -> None:
        arms = gipo_ablation_arms(GIPO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX)
        self.assertEqual(len(arms), 16)
        self.assertEqual(len({arm.arm_id for arm in arms}), 16)
        self.assertEqual([arm.paper_group for arm in arms[:6]], ["main"] * 6)
        self.assertNotIn("elite-bend", " ".join(arm.arm_id for arm in arms))
        full_scores_by_mode = {}
        blend_weights_by_mode = {}
        for arm in arms:
            if arm.student_target_mixture_mode == "full":
                full_scores_by_mode.setdefault(arm.student_training_mode, set()).add(float(arm.student_teacher_score_weight))
            if arm.student_target_mixture_mode == "elite_blend":
                blend_weights_by_mode.setdefault(arm.student_training_mode, set()).add(float(arm.student_target_elite_blend_all_weight))
        self.assertEqual(set(full_scores_by_mode), {"seen_only_zero_shot", "seen_plus_unseen_pseudo"})
        for weights in full_scores_by_mode.values():
            self.assertEqual(weights, {0.0, 0.01, 0.05, 0.10})
        for weights in blend_weights_by_mode.values():
            self.assertEqual(weights, {0.10, 0.20, 0.40})

    def test_full_pipeline_ablation_first_default_dry_run_grid_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(run_root),
                    "--ablation_first",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
            manifest = json.loads(
                (run_root / "gipo_ablations" / GIPO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX / "ablation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            [stage["stage"] for stage in summary["stages"]],
            [
                "data_prep",
                "ser_summaries",
                "schedule_rows_seen",
                "schedule_rows_unseen",
                "gipo_ablation_students",
                "gipo_ablation_locked_test_reports",
            ],
        )
        self.assertFalse(any(stage["stage"] == "backbone_training" for stage in summary["stages"]))
        self.assertFalse(any(stage["stage"].startswith("gipo_student_") for stage in summary["stages"]))
        by_stage = {stage["stage"]: stage["commands"] for stage in summary["stages"]}
        self.assertEqual(len(by_stage["gipo_ablation_students"]), 16)
        self.assertEqual(len(by_stage["gipo_ablation_locked_test_reports"]), 160)
        for stage_name in ("schedule_rows_seen", "schedule_rows_unseen"):
            self.assertTrue(by_stage[stage_name])
            for command in by_stage[stage_name]:
                self.assertIn("--forecast_datasets", command)
                self.assertEqual(command[command.index("--forecast_datasets") + 1], "")
                self.assertIn("--conditional_generation_datasets", command)
                self.assertEqual(command[command.index("--conditional_generation_datasets") + 1], "lobster_synthetic")
                self.assertIn("--schedule_summary_json", command)
                self.assertIn("--summary_scheduler_names", command)
        self.assertEqual(manifest["status"], "dry_run")
        self.assertEqual(manifest["arm_count"], 16)
        self.assertEqual(manifest["gipo_teacher_steps"], 500)
        self.assertEqual(manifest["gipo_student_steps"], 500)
        self.assertEqual(manifest["ablation_root"], f"gipo_ablations/{GIPO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX}")
        self.assertTrue(all(not str(value).startswith(("/", "C:")) for arm in manifest["arms"] for value in arm["outputs"].values()))

    def test_full_pipeline_dataset_alias_routes_requested_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--dataset",
                    "solar_energy_10m",
                    "--run_root",
                    str(run_root),
                    "--stages",
                    "schedule_rows_seen",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
            protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))

        commands = [command for stage in summary["stages"] for command in stage["commands"]]
        command = commands[0]
        self.assertEqual(protocol["scenario_key"], "solar_energy_10m")
        self.assertEqual(summary["preflight"]["scenario_key"], "solar_energy_10m")
        self.assertEqual(command[command.index("--forecast_datasets") + 1], "solar_energy_10m")
        self.assertEqual(command[command.index("--conditional_generation_datasets") + 1], "")

    def test_full_pipeline_ablation_first_dry_run_reports_missing_backbone_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--backbone_manifest",
                    str(Path(tmpdir) / "missing" / "backbone_manifest.json"),
                    "--ablation_first",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)

        validation = summary["preflight"]["provided_backbone_validation"]
        self.assertEqual(validation["status"], "skipped_missing_manifest")
        self.assertTrue(any("Backbone manifest is missing" in error for error in validation["errors"]))

    def test_full_pipeline_ser_controls_are_protocolized_and_ser_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--ablation_first",
                    "--ser_calibration_batch_size",
                    "1",
                    "--ser_val_windows",
                    "0",
                    "--ser_train_tuning_max_examples",
                    "17",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
        by_stage = {stage["stage"]: [" ".join(command) for command in stage["commands"]] for stage in summary["stages"]}
        ser_commands = by_stage["ser_summaries"]
        non_ser_commands = [
            command
            for stage, commands in by_stage.items()
            if stage != "ser_summaries"
            for command in commands
        ]
        self.assertTrue(ser_commands)
        self.assertTrue(all("--calibration_batch_size 1" in command for command in ser_commands))
        self.assertTrue(all("--val_windows 0" in command for command in ser_commands))
        self.assertTrue(all("--train_tuning_max_examples 17" in command for command in ser_commands))
        self.assertTrue(all("--train_tuning_max_examples_source train_tuning_max_examples" in command for command in ser_commands))
        self.assertTrue(all("--calibration_batch_size" not in command for command in non_ser_commands))
        self.assertTrue(all("--val_windows" not in command for command in non_ser_commands))
        self.assertTrue(all("--train_tuning_max_examples" not in command for command in non_ser_commands))
        protocol = full_pipeline._protocol_payload(args)
        self.assertEqual(protocol["ser_calibration_batch_size"], 1)
        self.assertEqual(protocol["ser_val_windows"], 0)
        self.assertEqual(protocol["ser_train_tuning_max_examples"], 17)
        self.assertEqual(protocol["ser_train_tuning_effective_max_examples"], 17)
        self.assertEqual(protocol["ser_example_selection_protocol"], "ser_ptg_reference_global_context_capped_v3")
        self.assertEqual(protocol["ser_local_defect_proxy_protocol"], "otflow_midpoint_local_defect_proxy_v1")

    def test_full_pipeline_default_ser_train_tuning_cap_tracks_context_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--ablation_first",
                    "--context_sample_count",
                    "123",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
        by_stage = {stage["stage"]: [" ".join(command) for command in stage["commands"]] for stage in summary["stages"]}
        self.assertTrue(all("--train_tuning_max_examples 123" in command for command in by_stage["ser_summaries"]))
        self.assertTrue(all("--train_tuning_max_examples_source context_sample_count" in command for command in by_stage["ser_summaries"]))
        protocol = full_pipeline._protocol_payload(args)
        self.assertEqual(protocol["ser_train_tuning_max_examples"], 0)
        self.assertEqual(protocol["ser_train_tuning_effective_max_examples"], 123)

    def test_full_pipeline_ablation_first_routes_arm_knobs_and_locked_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--ablation_first",
                    "--gipo_teacher_steps",
                    "37",
                    "--gipo_student_steps",
                    "43",
                    "--seen_nfes",
                    "4,8",
                    "--unseen_nfes",
                    "6,10",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
        by_stage = {stage["stage"]: [" ".join(command) for command in stage["commands"]] for stage in summary["stages"]}
        train_commands = by_stage["gipo_ablation_students"]
        report_commands = by_stage["gipo_ablation_locked_test_reports"]
        for arm in gipo_ablation_arms(GIPO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX):
            command = next(item for item in train_commands if arm.arm_id in item)
            self.assertIn(f"--student_target_mixture_mode {arm.student_target_mixture_mode}", command)
            self.assertIn(f"--student_teacher_score_weight {float(arm.student_teacher_score_weight)}", command)
            self.assertIn("--student_teacher_score_warmup_fraction 0.6", command)
            self.assertIn("--teacher_steps 37", command)
            self.assertIn("--student_steps 43", command)
            self.assertIn("--seen_target_nfe_values 4,8", command)
            self.assertIn("--pseudo_target_nfe_values 6,10", command)
            self.assertIn("--schedule_summary_json", command)
            self.assertNotIn("--student_teacher_score_include_pseudo", command)
            if arm.student_target_mixture_mode == "elite_blend":
                self.assertIn(f"--student_target_elite_blend_all_weight {float(arm.student_target_elite_blend_all_weight)}", command)
            if arm.uses_unseen_pseudo_targets:
                self.assertIn("--student_pseudo_rows_csv", command)
                self.assertIn("--student_pseudo_schedule_summary_json", command)
                self.assertIn("--student_pseudo_target_weight 0.25", command)
            else:
                self.assertNotIn("--student_pseudo_rows_csv", command)
                self.assertNotIn("--student_pseudo_schedule_summary_json", command)
        self.assertTrue(all("gipo_ablations" in command for command in report_commands))
        self.assertTrue(all("locked_test" in command for command in report_commands))
        self.assertTrue(all("--training_summary" in command and "--gipo_student_checkpoint" in command for command in report_commands))

    def test_full_pipeline_default_ablation_first_run_root_is_namespaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_root = Path(tmpdir) / "outputs"
            expected_root = outputs_root / "full_pipeline" / "lobster_synthetic" / "gipo_ablations" / GIPO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--ablation_first",
                    "--dry_run",
                ]
            )
            with mock.patch.object(full_pipeline, "project_outputs_root", return_value=outputs_root):
                summary = full_pipeline.run_full_pipeline(args)
            self.assertTrue((expected_root / "protocol.json").exists())
            self.assertTrue((expected_root / "gipo_ablation_students_manifest.json").exists())
            self.assertTrue((expected_root / "ablation_manifest.json").exists())
            self.assertFalse((outputs_root / "full_pipeline" / "lobster_synthetic" / "protocol.json").exists())
            self.assertIn("gipo_ablations", summary["run_root"])

    def test_full_pipeline_failed_ablation_stage_marks_ablation_manifest_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(run_root),
                    "--ablation_first",
                ]
            )
            failing_stage = full_pipeline.StageCommand(
                full_pipeline.GIPO_ABLATION_STUDENT_STAGE,
                [[full_pipeline.sys.executable, "-c", "import sys; sys.exit(3)"]],
                "gipo_ablation_students_manifest.json",
            )
            with (
                mock.patch.object(full_pipeline, "_validate_inputs_preflight", return_value={"status": "complete"}),
                mock.patch.object(full_pipeline, "_build_stage_commands", return_value=[failing_stage]),
            ):
                with self.assertRaises(RuntimeError):
                    full_pipeline.run_full_pipeline(args)
            manifest = json.loads(
                (run_root / "gipo_ablations" / GIPO_ABLATION_PRESET_PAPER_MAIN_PLUS_APPENDIX / "ablation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failed_stage"], full_pipeline.GIPO_ABLATION_STUDENT_STAGE)
        self.assertEqual(manifest["failed_command_index"], 0)


if __name__ == "__main__":
    unittest.main()
