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
    ABLATION_PRESET_ALL,
    GIPO_POLICY_KEY,
    ablation_student_policies,
    gipo_policy,
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
            "reward_anchor_scheduler_key",
            "reward_utility_transform",
            "reward_granularity",
            "effective_train_steps",
            "checkpoint_export_protocol",
            "locked_test_mode",
            "locked_test_context_limit",
            "locked_test_context_limit_scope",
            "selected_examples_cap_source",
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
        baseline_rows = report_locked_test._filter_rows_to_scheduler_keys(rows, ["uniform", "late_power_3"])
        ser_rows = report_locked_test._filter_rows_to_scheduler_keys(rows, ["ser_ptg_local_defect_eta005"])

        self.assertEqual([row["scheduler_key"] for row in baseline_rows], ["uniform", "late_power_3"])
        self.assertEqual([row["scheduler_key"] for row in ser_rows], ["ser_ptg_local_defect_eta005"])

    def test_forecast_unseen_target_rows_are_reward_materialized_before_student_distillation(self) -> None:
        def write_rows(path: Path, target_nfes: tuple[int, ...]) -> None:
            schedules = {
                "uniform": (2.0, 2.0),
                "late_power_3": (1.0, 1.0),
                "flowts_power_sampling": (1.5, 1.5),
                "ays": (1.4, 1.4),
            }
            fields = [
                "benchmark_family",
                "scenario_key",
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
                                    "scenario_key": "solar_energy_10m",
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
            unseen_target_rows = root / "unseen_target.csv"
            embeddings = root / "context_embeddings.npz"
            write_rows(seen_rows, (4, 8, 12, 16))
            write_rows(unseen_target_rows, (6, 10, 14, 20))
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
                student_unseen_target_rows_csv=str(unseen_target_rows),
                student_unseen_target_context_embeddings_npz=str(embeddings),
                student_unseen_target_schedule_summary_json="",
                student_unseen_target_weight=0.25,
                student_steps=1,
                student_log_every=0,
                student_checkpoint_every=1,
                student_selection_holdout_fraction=0.5,
                teacher_lr=1e-3,
                student_lr=1e-3,
                transformer_hidden_dim=16,
                num_layers=1,
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

        self.assertTrue(summary["student_unseen_target_distillation"]["enabled"])
        self.assertEqual(summary["student_policy_key"], "custom_gipo")
        self.assertEqual(summary["student_objective_settings"]["student_teacher_score_weight"], 0.01)
        self.assertEqual(summary["student_objective_settings"]["student_teacher_score_warmup_fraction"], 0.6)
        self.assertEqual(summary["student_objective_settings"]["student_teacher_score_schedule_steps"], 1)
        self.assertEqual(summary["student_objective_settings"]["student_teacher_score_clip"], 5.0)
        self.assertEqual(
            summary["student_objective_settings"]["student_teacher_score_protocol"],
            "late_ramped_per_cell_teacher_utility_z_score",
        )
        self.assertEqual(summary["student_objective_settings"]["student_target_mixture_mode"], "full")
        self.assertFalse(
            summary["student_objective_settings"]["student_teacher_score_include_unseen_targets"]
        )
        self.assertFalse(summary["student_objective_settings"]["student_regularizers"]["smooth"])
        self.assertFalse(summary["student_objective_settings"]["student_regularizers"]["guard"])
        unseen_target_summary = summary["student_training"]["student_unseen_target_summary"]
        self.assertTrue(unseen_target_summary["unseen_target_distillation_used"])
        self.assertEqual(unseen_target_summary["unseen_target_nfes"], [6, 10, 14, 20])
        self.assertEqual(unseen_target_summary["student_target_mixture_mode"], "full")

    def test_conditional_context_rows_use_physical_ids_and_checkpoint_scoped_embeddings(self) -> None:
        rows = runner._conditional_context_records(
            benchmark_family=CONDITIONAL_GENERATION_FAMILY,
            dataset="lobster_synthetic",
            split_phase="train_tuning",
            seed=7,
            evaluation_seed=17,
            solver_key="euler",
            target_nfe=4,
            macro_steps=4,
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
            scenario_key="lobster_synthetic",
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
        self.assertEqual(rows[0]["reward_anchor_scheduler_key"], "uniform")
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
            macro_steps=4,
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
            macro_steps=4,
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

    def test_context_embedding_id_is_stable_lookup_key(self) -> None:
        row = {"context_id": "logical", "context_embedding_id": "ckpt:logical"}
        self.assertEqual(context_embedding_id_from_row(row), "ckpt:logical")
        self.assertEqual(context_embedding_id_from_row({"context_id": "logical"}), "logical")

    def test_preflight_physical_context_fingerprint_ignores_checkpoint_embedding_id(self) -> None:
        base = {
            "context_schema": "forecast_window",
            "scenario_key": "solar_energy_10m",
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
            "scenario_key": "lobster_synthetic",
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
                        "scenario_key": "lobster_synthetic",
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
                            "scenario_key": "lobster_synthetic",
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
                        "scenario_key": "lobster_synthetic",
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
            "scenario_key": "lobster_synthetic",
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
            "scenario_key": "lobster_synthetic",
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
            "scenario_key": "lobster_synthetic",
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
        self.assertEqual(checkpoint_scope_from_row(dict(row, train_steps=4000)), "")
        split_summary = _split_membership_summary([early, late])
        self.assertEqual(split_summary["checkpoint_scope_count"], 2)
        self.assertEqual(set(split_summary["checkpoint_scopes"]), {"checkpoint_step:4000", "checkpoint_step:8000"})

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
                "scenario_key": "lobster_synthetic",
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
            _resolve_teacher_metric_target_keys(args, [{"benchmark_family": FORECAST_FAMILY, "scenario_key": "solar_energy_10m"}]),
            tuple(spec.utility_key for spec in FORECAST_METRIC_SPECS),
        )
        self.assertEqual(
            _resolve_teacher_metric_target_keys(args, [{"benchmark_family": CONDITIONAL_GENERATION_FAMILY, "scenario_key": "lobster_synthetic"}]),
            tuple(spec.utility_key for spec in CONDITIONAL_PRIMARY_LOB_METRIC_SPECS),
        )
        self.assertEqual(
            _resolve_teacher_metric_target_keys(args, [{"benchmark_family": CONDITIONAL_GENERATION_FAMILY, "scenario_key": "long_term_st"}]),
            tuple(spec.utility_key for spec in CONDITIONAL_PRIMARY_ECG_METRIC_SPECS),
        )
        self.assertEqual(
            _resolve_teacher_metric_target_keys(args, [{"benchmark_family": "molecule_3d_coordinate_generation", "scenario_key": "molecule_3d_set1"}]),
            tuple(spec.utility_key for spec in MOLECULE_METRIC_SPECS),
        )

    def test_teacher_metric_profiles_resolve_conditional_scenarios(self) -> None:
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
                "scenario_key": "lobster_synthetic",
                "seed": 1,
                "solver_key": "euler",
                "target_nfe": 4,
                "checkpoint_step": 4000,
                "scheduler_key": "gipo",
                "method_key": "gipo",
                "gipo_step_budget": 500,
                "mode": "continuous",
                "teacher_final_retrain": "{}",
                "score_main": "0.8",
                "temporal_uw1": "0.2",
                "temporal_cw1": "0.3",
                "temporal_tstr_f1_applicable": "true",
            },
            {
                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                "scenario_key": "lobster_synthetic",
                "seed": 1,
                "solver_key": "euler",
                "target_nfe": 4,
                "checkpoint_step": 4000,
                "scheduler_key": "gipo",
                "method_key": "gipo",
                "gipo_step_budget": 500,
                "mode": "continuous",
                "teacher_final_retrain": "{}",
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
                "scenario_key": "lobster_synthetic",
                "evaluation_seed": "1",
                "solver_key": "heun",
                "target_nfe": "6",
            },
            {
                "scenario_key": "lobster_synthetic",
                "evaluation_seed": "0",
                "solver_key": "euler",
                "target_nfe": "4",
            },
        ]

        self.assertEqual(
            report_locked_test._single_value_from_rows(rows, field="scenario_key", arg_name="scenario_key"),
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
            report_locked_test._single_value_from_rows(rows, field="scenario_key", requested="traffic_hourly", arg_name="scenario_key")

    def test_locked_reporter_representatives_filter_matrix_by_logical_seed(self) -> None:
        rows = [
            {
                "scenario_key": "lobster_synthetic",
                "seed": "0",
                "logical_seed": "0",
                "evaluation_seed": "11",
                "solver_key": "euler",
                "target_nfe": "4",
                "checkpoint_step": "4000",
                "scheduler_key": "uniform",
                "context_id": "ctx_a",
                "source_split_phase": "locked_test",
            },
            {
                "scenario_key": "lobster_synthetic",
                "seed": "0",
                "logical_seed": "0",
                "evaluation_seed": "22",
                "solver_key": "euler",
                "target_nfe": "8",
                "checkpoint_step": "4000",
                "scheduler_key": "uniform",
                "context_id": "ctx_b",
                "source_split_phase": "locked_test",
            },
        ]

        representatives = report_locked_test._representative_context_rows(rows)
        filtered = report_locked_test._filter_representatives_to_matrix(
            representatives,
            scenario_key="lobster_synthetic",
            seeds=[0],
            solvers=["euler"],
            target_nfes=[4],
        )

        self.assertEqual([row["evaluation_seed"] for row in representatives], [11, 22])
        self.assertEqual([row["seed"] for row in representatives], [0, 0])
        self.assertEqual([row["logical_seed"] for row in representatives], [0, 0])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["context_id"], "ctx_a")

    def test_locked_reporter_context_match_uses_logical_seed(self) -> None:
        representative = {
            "scenario_key": "solar_energy_10m",
            "seed": "0",
            "logical_seed": "0",
            "evaluation_seed": "1000",
            "solver_key": "dpmpp2m",
            "target_nfe": "4",
            "checkpoint_step": "4000",
            "context_id": "ctx",
            "source_split_phase": "locked_test",
        }
        uniform = {
            **representative,
            "scheduler_key": "uniform",
        }

        self.assertEqual(
            report_locked_test._context_match_key(representative),
            report_locked_test._context_match_key(uniform),
        )

    def test_locked_reporter_rejects_conflicting_duplicate_context_metadata(self) -> None:
        first = {
            "scenario_key": "solar_energy_10m",
            "benchmark_family": FORECAST_FAMILY,
            "source_split_phase": "locked_test",
            "seed": 0,
            "logical_seed": 0,
            "evaluation_seed": 10,
            "solver_key": "euler",
            "target_nfe": 4,
            "checkpoint_step": 4000,
            "checkpoint_id": "checkpoint",
            "context_id": "context",
            "context_embedding_id": "checkpoint:context",
            "scheduler_key": "uniform",
        }
        conflicting = {**first, "scheduler_key": "gipo", "evaluation_seed": 11}
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate context metadata"):
            report_locked_test._representative_context_rows([first, conflicting])

    def test_locked_reporter_rejects_mixed_checkpoint_steps(self) -> None:
        rows = [
            {"checkpoint_step": 4000},
            {"checkpoint_step": 8000},
        ]

        with self.assertRaisesRegex(ValueError, "mix checkpoint_step values"):
            report_locked_test._validate_checkpoint_step(rows, requested=4000)

    def test_locked_reporter_rejects_rows_from_another_loaded_checkpoint(self) -> None:
        rows = [
            {"checkpoint_step": 4000, "checkpoint_id": "checkpoint-a"},
        ]
        loaded_checkpoint = {
            "checkpoint_step": 4000,
            "checkpoint_id": "checkpoint-b",
        }

        with self.assertRaisesRegex(ValueError, "do not match the loaded backbone artifact"):
            report_locked_test._validate_loaded_checkpoint_identity(rows, loaded_checkpoint)

    def test_strict_locked_report_requires_every_schedule_for_every_context(self) -> None:
        provenance = {
            "locked_test_mode": "full",
            "locked_test_context_limit": "",
            "locked_test_context_limit_scope": "none",
            "selected_examples_cap_source": "locked_test_full",
            "selection_was_capped": False,
            "global_selection_was_capped": False,
        }
        representatives = [
            {
                **provenance,
                "benchmark_family": FORECAST_FAMILY,
                "scenario_key": "solar_energy_10m",
                "source_split_phase": "locked_test",
                "logical_seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "checkpoint_step": 4000,
                "checkpoint_id": "checkpoint",
                "context_id": context_id,
            }
            for context_id in ("context-a", "context-b")
        ]
        baseline_rows = [
            {**row, "scheduler_key": schedule_key}
            for schedule_key in report_locked_test.BASELINE_SCHEDULE_KEYS
            for row in representatives
        ]
        comparator_rows = [
            {**row, "scheduler_key": schedule_key}
            for schedule_key in report_locked_test.SER_REFERENCE_SCHEDULE_KEYS
            for row in representatives
        ]

        report_locked_test._validate_strict_comparison_context_coverage(
            representatives=representatives,
            baseline_rows=baseline_rows,
            comparator_rows=comparator_rows,
        )
        with self.assertRaisesRegex(ValueError, "missing contexts"):
            report_locked_test._validate_strict_comparison_context_coverage(
                representatives=representatives,
                baseline_rows=baseline_rows[:-1],
                comparator_rows=comparator_rows,
            )
        with self.assertRaisesRegex(ValueError, "duplicate contexts"):
            report_locked_test._validate_strict_comparison_context_coverage(
                representatives=representatives,
                baseline_rows=baseline_rows,
                comparator_rows=[*comparator_rows, comparator_rows[0]],
            )

    def test_locked_report_selection_provenance_distinguishes_preview(self) -> None:
        rows = [
            {
                "locked_test_mode": "preview",
                "locked_test_context_limit": "512",
                "locked_test_context_limit_scope": "per_seed",
                "selected_examples_cap_source": "locked_test_preview_contexts",
                "selection_was_capped": "true",
                "global_selection_was_capped": "true",
            }
        ]

        self.assertEqual(
            report_locked_test._locked_test_selection_provenance(rows, split_phase="locked_test"),
            {
                "locked_test_mode": "preview",
                "locked_test_context_limit": 512,
                "locked_test_context_limit_scope": "per_seed",
                "selected_examples_cap_source": "locked_test_preview_contexts",
                "selection_was_capped": True,
                "global_selection_was_capped": True,
            },
        )
        self.assertEqual(
            report_locked_test._report_artifact_name(
                split_phase="locked_test",
                selection_mode=report_locked_test.SELECTION_MODE_REPORTING,
                locked_test_mode="preview",
            ),
            "gipo_locked_test_preview_report",
        )

    def test_full_locked_report_rejects_capped_provenance(self) -> None:
        rows = [
            {
                "locked_test_mode": "full",
                "locked_test_context_limit": "",
                "locked_test_context_limit_scope": "none",
                "selected_examples_cap_source": "locked_test_full",
                "selection_was_capped": "true",
                "global_selection_was_capped": "true",
            }
        ]

        with self.assertRaisesRegex(ValueError, "uncapped selection"):
            report_locked_test._locked_test_selection_provenance(rows, split_phase="locked_test")

    def test_locked_reporter_logical_seed_identity_does_not_fall_back_to_evaluation_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "logical_seed or seed"):
            report_locked_test._context_match_key(
                {
                    "scenario_key": "solar_energy_10m",
                    "evaluation_seed": "1000",
                    "solver_key": "dpmpp2m",
                    "target_nfe": "4",
                    "checkpoint_step": "4000",
                    "context_id": "ctx",
                    "source_split_phase": "locked_test",
                }
            )

    def test_gipo_locked_report_cannot_bypass_comparator_completeness(self) -> None:
        rows = [
            {
                "benchmark_family": FORECAST_FAMILY,
                "scenario_key": "solar_energy_10m",
                "split_phase": "locked_test",
                "source_split_phase": "locked_test",
                "seed": "0",
                "logical_seed": "0",
                "evaluation_seed": str(evaluation_seed),
                "solver_key": "dpmpp2m",
                "target_nfe": str(target_nfe),
                "checkpoint_step": "4000",
                "scheduler_key": "uniform",
                "context_id": f"ctx-{target_nfe}",
                "context_embedding_id": f"ckpt:ctx-{target_nfe}",
                "checkpoint_id": "ckpt",
                "locked_test_mode": "full",
                "locked_test_context_limit": "",
                "locked_test_context_limit_scope": "none",
                "selected_examples_cap_source": "locked_test_full",
                "selection_was_capped": "false",
                "global_selection_was_capped": "false",
                "example_idx": "0",
                "series_id": "solar_energy_10m",
                "target_t": str(target_nfe),
            }
            for target_nfe, evaluation_seed in ((4, 1000), (8, 2000))
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            context_csv = tmp / "context_rows.csv"
            with context_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            training_summary = tmp / "gipo_training_summary.json"
            training_summary.write_text(
                json.dumps(
                    {
                        "protocol": GIPO_PROTOCOL,
                        "student_policy_key": GIPO_POLICY_KEY,
                        "gipo_step_budget": 500,
                        "mode": "continuous",
                        "locked_test_used_for_selection": False,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                gipo_student_checkpoint=str(tmp / "gipo_student.pt"),
                training_summary=str(training_summary),
                checkpoint_step=4000,
                require_teacher_checkpoint_selection_mode="",
                selection_mode=report_locked_test.SELECTION_MODE_REPORTING,
                split_phase="locked_test",
                context_rows=str(context_csv),
                context_embeddings_npz=str(tmp / "context_embeddings.npz"),
                benchmark_family=FORECAST_FAMILY,
                scenario_key="solar_energy_10m",
                seeds="",
                solver_names="",
                target_nfe_values="",
                molecule_group_root="",
                baseline_rows=str(context_csv),
                comparator_rows="",
                allow_incomplete_comparison=True,
            )
            with mock.patch(
                "genode.gipo.report_locked_test._load_student_checkpoint",
                return_value=(
                    mock.Mock(),
                    {},
                    mock.Mock(),
                    {
                        "student_policy_key": GIPO_POLICY_KEY,
                        "gipo_step_budget": 500,
                        "mode": "continuous",
                    },
                ),
            ):
                with self.assertRaisesRegex(ValueError, "requires --comparator_rows") as cm:
                    report_locked_test.report_gipo_locked_test(args)
            self.assertNotIn("missing seed/solver/NFE", str(cm.exception))

    def test_locked_reporter_output_uses_logical_seed_with_offset_uniform_context_match(self) -> None:
        context_rows = [
            {
                "benchmark_family": FORECAST_FAMILY,
                "scenario_key": "solar_energy_10m",
                "split_phase": "locked_test",
                "source_split_phase": "locked_test",
                "seed": "0",
                "logical_seed": "0",
                "evaluation_seed": "1000",
                "solver_key": "dpmpp2m",
                "target_nfe": "4",
                "checkpoint_step": "4000",
                "scheduler_key": "ser_ptg_local_defect_eta005",
                "context_id": "ctx-4",
                "context_embedding_id": "ckpt:ctx-4",
                "checkpoint_id": "ckpt",
                "locked_test_mode": "full",
                "locked_test_context_limit": "",
                "locked_test_context_limit_scope": "none",
                "selected_examples_cap_source": "locked_test_full",
                "selection_was_capped": "false",
                "global_selection_was_capped": "false",
                "example_idx": "0",
                "series_id": "solar_energy_10m",
                "target_t": "4",
                "forecast_crps": "1.50",
                "forecast_mase": "1.25",
            },
            {
                "benchmark_family": FORECAST_FAMILY,
                "scenario_key": "solar_energy_10m",
                "split_phase": "locked_test",
                "source_split_phase": "locked_test",
                "seed": "0",
                "logical_seed": "0",
                "evaluation_seed": "1000",
                "solver_key": "dpmpp2m",
                "target_nfe": "4",
                "checkpoint_step": "4000",
                "scheduler_key": "uniform",
                "context_id": "ctx-4",
                "context_embedding_id": "ckpt:ctx-4",
                "checkpoint_id": "ckpt",
                "locked_test_mode": "full",
                "locked_test_context_limit": "",
                "locked_test_context_limit_scope": "none",
                "selected_examples_cap_source": "locked_test_full",
                "selection_was_capped": "false",
                "global_selection_was_capped": "false",
                "example_idx": "0",
                "series_id": "solar_energy_10m",
                "target_t": "4",
                "forecast_crps": "2.00",
                "forecast_mase": "2.00",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            context_csv = tmp / "context_rows.csv"
            with context_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(context_rows[0]))
                writer.writeheader()
                writer.writerows(context_rows)
            training_summary = tmp / "gipo_training_summary.json"
            training_summary.write_text(
                json.dumps(
                    {
                        "protocol": GIPO_PROTOCOL,
                        "student_policy_key": "custom_gipo",
                        "gipo_step_budget": 500,
                        "mode": "continuous",
                        "locked_test_used_for_selection": False,
                    }
                ),
                encoding="utf-8",
            )
            out_dir = tmp / "report"
            student = mock.Mock()
            normalizer = mock.Mock()
            normalizer.transform_table.side_effect = lambda table: table
            args = SimpleNamespace(
                gipo_student_checkpoint=str(tmp / "gipo_student.pt"),
                training_summary=str(training_summary),
                checkpoint_step=4000,
                require_teacher_checkpoint_selection_mode="",
                selection_mode=report_locked_test.SELECTION_MODE_REPORTING,
                split_phase="locked_test",
                context_rows=str(context_csv),
                context_embeddings_npz=str(tmp / "context_embeddings.npz"),
                benchmark_family=FORECAST_FAMILY,
                scenario_key="solar_energy_10m",
                seeds="0",
                solver_names="dpmpp2m",
                target_nfe_values="4",
                molecule_group_root="",
                baseline_rows=str(context_csv),
                comparator_rows="",
                allow_incomplete_comparison=True,
                dataset_root=str(tmp),
                shared_backbone_root=str(tmp),
                device="cpu",
                num_eval_samples=1,
                forecast_eval_batch_size=1,
                out_dir=str(out_dir),
            )
            with (
                mock.patch(
                    "genode.gipo.report_locked_test._load_student_checkpoint",
                    return_value=(
                        student,
                        normalizer,
                        [0.0, 1.0],
                        {
                            "student_policy_key": "custom_gipo",
                            "gipo_step_budget": 500,
                            "mode": "continuous",
                        },
                    ),
                ),
                mock.patch(
                    "genode.gipo.report_locked_test.load_context_embedding_table",
                    return_value={"ckpt:ctx-4": [0.0, 1.0]},
                ),
                mock.patch(
                    "genode.gipo.report_locked_test.load_forecast_checkpoint_splits",
                    return_value={
                        "model": mock.Mock(),
                        "cfg": SimpleNamespace(),
                        "splits": {"train": object(), "val": object(), "test": object()},
                        "checkpoint_id": "ckpt",
                        "checkpoint_step": 4000,
                    },
                ),
                mock.patch(
                    "genode.gipo.report_locked_test.predict_gipo_density_many",
                    return_value=[
                        {
                            "time_grid": [0.0, 1.0],
                            "density_mass": [1.0],
                            "density_mass_hash": "density",
                            "schedule_grid_hash": "grid",
                            "density_protocol": "protocol",
                            "reference_grid_hash": "reference",
                            "mode": "continuous",
                            "macro_steps": 4,
                        }
                    ],
                ),
                mock.patch(
                    "genode.gipo.report_locked_test.evaluate_forecast_schedule",
                    return_value={"forecast_crps": 1.0, "forecast_mase": 1.0, "forecast_mse": 1.0},
                ) as evaluate_mock,
            ):
                summary = report_locked_test.report_gipo_locked_test(args)

            self.assertEqual(summary["artifact"], "gipo_locked_test_report")
            self.assertEqual(summary["method_key"], "custom_gipo")
            self.assertEqual(summary["gipo_step_budget"], 500)
            self.assertEqual(summary["context_row_count"], 1)
            self.assertEqual(summary["aggregate_row_count"], 1)
            rows = list(csv.DictReader((out_dir / "locked_test_gipo_rows.csv").open(newline="", encoding="utf-8")))
            aggregate_rows = list(csv.DictReader((out_dir / "locked_test_gipo_aggregate_rows.csv").open(newline="", encoding="utf-8")))
            comparison = json.loads((out_dir / "locked_test_gipo_comparison_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["seed"], "0")
            self.assertEqual(rows[0]["logical_seed"], "0")
            self.assertEqual(rows[0]["evaluation_seed"], "1000")
            self.assertEqual(evaluate_mock.call_args.kwargs["seed"], 1000)
            self.assertEqual(aggregate_rows[0]["seed"], "0")
            self.assertEqual(comparison["seeds"], [0])
            self.assertEqual(comparison["method_key"], "custom_gipo")
            self.assertEqual(comparison["missing_student_cells"], [])

    def test_reporter_aggregates_molecule_metrics(self) -> None:
        rows = [
            {
                "benchmark_family": "molecule_3d_coordinate_generation",
                "scenario_key": "molecule_3d_set1",
                "seed": 1,
                "solver_key": "euler",
                "target_nfe": 4,
                "checkpoint_step": 4000,
                "scheduler_key": "gipo",
                "method_key": "gipo",
                "gipo_step_budget": 500,
                "mode": "continuous",
                "teacher_final_retrain": "{}",
                "molecule_kabsch_rmsd_3d": "1.0",
                "molecule_rollout_velocity_norm_w1": "2.0",
            },
            {
                "benchmark_family": "molecule_3d_coordinate_generation",
                "scenario_key": "molecule_3d_set1",
                "seed": 1,
                "solver_key": "euler",
                "target_nfe": 4,
                "checkpoint_step": 4000,
                "scheduler_key": "gipo",
                "method_key": "gipo",
                "gipo_step_budget": 500,
                "mode": "continuous",
                "teacher_final_retrain": "{}",
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
                "scenario_key": "molecule_3d_set1",
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
                "scenario_key": "molecule_3d_set1",
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
            scenario_key="molecule_3d_set1",
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
                "scenario_key": "lobster_synthetic",
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
                "scenario_key": "lobster_synthetic",
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
            scenario_key="lobster_synthetic",
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
                    "backbone_training,schedule_rows_seen,report_gipo_locked_test",
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

    def test_full_pipeline_resume_skips_completed_prefix_and_runs_next_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            run_root.mkdir(parents=True)
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(run_root),
                    "--resume",
                ]
            )
            protocol_hash = full_pipeline._json_hash(full_pipeline._protocol_payload(args))
            first_command = [full_pipeline.sys.executable, "-c", "print('first')"]
            second_command = [full_pipeline.sys.executable, "-c", "print('second')"]
            first_stage = full_pipeline.StageCommand("input_preflight", [first_command], "input_preflight_manifest.json")
            second_stage = full_pipeline.StageCommand("ser_summaries", [second_command], "ser_summaries_manifest.json")
            (run_root / "status.json").write_text(
                json.dumps({"protocol_hash": protocol_hash, "status": "failed"}),
                encoding="utf-8",
            )
            (run_root / first_stage.manifest_name).write_text(
                json.dumps(
                    {
                        "stage": first_stage.stage,
                        "status": "complete",
                        "protocol_hash": protocol_hash,
                        "commands": [full_pipeline._display_command(first_command)],
                        "command_hashes": [full_pipeline._command_hash(first_command)],
                        "command_results": [{"command_index": 0, "returncode": 0, "log_path": "logs/input_preflight_0.log"}],
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(full_pipeline, "_validate_inputs_preflight", return_value={"status": "complete"}),
                mock.patch.object(full_pipeline, "_build_stage_commands", return_value=[first_stage, second_stage]),
                mock.patch.object(full_pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run_mock,
            ):
                summary = full_pipeline.run_full_pipeline(args)
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(run_mock.call_args.args[0], second_command)
        self.assertEqual(summary["skipped_stages"], ["input_preflight"])
        self.assertEqual(summary["executed_stages"], ["ser_summaries"])

    def test_full_pipeline_resume_reruns_backbone_stage_when_required_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            run_root.mkdir(parents=True)
            manifest_path = Path(tmpdir) / "backbone_manifest.json"
            artifact_root = Path(tmpdir) / "artifacts"
            artifact_root.mkdir()
            checkpoint_path = artifact_root / "model.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            (artifact_root / "artifact_summary.json").write_text("{}", encoding="utf-8")
            (artifact_root / "checkpoint_metadata.json").write_text("{}", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": "fm_backbone_manifest",
                        "path_base": "manifest_parent",
                        "artifacts": [
                            {
                                "backbone_name": "otflow",
                                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                                "dataset_key": "lobster_synthetic",
                                "train_steps": 4000,
                                "status": "ready",
                                "checkpoint_id": "lobster_synthetic_4000",
                                "checkpoint_path": "artifacts/model.pt",
                                "summary_path": "artifacts/artifact_summary.json",
                                "metadata_path": "artifacts/checkpoint_metadata.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(run_root),
                    "--stages",
                    "backbone_training",
                    "--checkpoint_steps",
                    "4000",
                    "--resume",
                ]
            )
            protocol_hash = full_pipeline._json_hash(full_pipeline._protocol_payload(args))
            backbone_command = [
                full_pipeline.sys.executable,
                "-m",
                "genode.training.train_backbone",
                "--scenario_key",
                "lobster_synthetic",
                "--checkpoint_steps",
                "4000",
            ]
            backbone_stage = full_pipeline.StageCommand(
                "backbone_training",
                [backbone_command],
                "backbone_training_manifest.json",
            )
            (run_root / "status.json").write_text(
                json.dumps({"protocol_hash": protocol_hash, "status": "failed"}),
                encoding="utf-8",
            )
            (run_root / backbone_stage.manifest_name).write_text(
                json.dumps(
                    {
                        "stage": backbone_stage.stage,
                        "status": "complete",
                        "protocol_hash": protocol_hash,
                        "commands": [full_pipeline._display_command(backbone_command)],
                        "command_hashes": [full_pipeline._command_hash(backbone_command)],
                        "command_results": [
                            {"command_index": 0, "returncode": 0, "log_path": "logs/backbone_training_0.log"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(full_pipeline, "backbone_manifest_path", return_value=manifest_path),
                mock.patch.object(full_pipeline, "validate_backbone_artifact_checkpoint", return_value=[]),
            ):
                is_complete, reason = full_pipeline._stage_manifest_complete(
                    run_root,
                    backbone_stage,
                    protocol_hash=protocol_hash,
                )
            self.assertTrue(is_complete, reason)
            checkpoint_path.unlink()

            with (
                mock.patch.object(full_pipeline, "_validate_inputs_preflight", return_value={"status": "complete"}),
                mock.patch.object(full_pipeline, "_build_stage_commands", return_value=[backbone_stage]),
                mock.patch.object(full_pipeline, "backbone_manifest_path", return_value=manifest_path),
                mock.patch.object(full_pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run_mock,
            ):
                summary = full_pipeline.run_full_pipeline(args)

        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(run_mock.call_args.args[0], backbone_command)
        self.assertEqual(summary["skipped_stages"], [])
        self.assertEqual(summary["executed_stages"], ["backbone_training"])

    def test_backbone_resume_validation_hides_external_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "missing_backbone_manifest.json"
            is_complete, reason = full_pipeline._backbone_manifest_artifacts_complete(
                manifest_path,
                backbone_name="otflow",
                benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                scenario_key="lobster_synthetic",
                checkpoint_steps=(4000,),
            )
        self.assertFalse(is_complete)
        self.assertIn(full_pipeline._display_path(str(manifest_path)), reason)
        self.assertNotIn(str(manifest_path.resolve()), reason)

    def test_full_pipeline_resume_reruns_partial_schedule_stage_despite_combined_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            partial_out = run_root / "schedule_rows" / "seen" / "train_tuning" / "4000_steps"
            partial_out.mkdir(parents=True)
            (partial_out / "combined_summary.json").write_text(json.dumps({"main_table_summary": {"row_count": 1}}), encoding="utf-8")
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(run_root),
                    "--resume",
                ]
            )
            protocol_hash = full_pipeline._json_hash(full_pipeline._protocol_payload(args))
            complete_command = [full_pipeline.sys.executable, "-c", "print('complete')"]
            schedule_command = [full_pipeline.sys.executable, "-c", "print('schedule')"]
            complete_stage = full_pipeline.StageCommand("input_preflight", [complete_command], "input_preflight_manifest.json")
            schedule_stage = full_pipeline.StageCommand("schedule_rows_seen", [schedule_command], "schedule_rows_seen_manifest.json")
            (run_root / "status.json").write_text(json.dumps({"protocol_hash": protocol_hash, "status": "failed"}), encoding="utf-8")
            (run_root / complete_stage.manifest_name).write_text(
                json.dumps(
                    {
                        "stage": complete_stage.stage,
                        "status": "complete",
                        "protocol_hash": protocol_hash,
                        "commands": [full_pipeline._display_command(complete_command)],
                        "command_hashes": [full_pipeline._command_hash(complete_command)],
                        "command_results": [{"command_index": 0, "returncode": 0, "log_path": "logs/input_preflight_0.log"}],
                    }
                ),
                encoding="utf-8",
            )
            def schedule_status(command):
                if list(command) == schedule_command:
                    return {"complete": False, "reason": "missing rows"}
                return None

            with (
                mock.patch.object(full_pipeline, "_validate_inputs_preflight", return_value={"status": "complete"}),
                mock.patch.object(full_pipeline, "_build_stage_commands", return_value=[complete_stage, schedule_stage]),
                mock.patch.object(full_pipeline, "_schedule_row_command_status", side_effect=schedule_status),
                mock.patch.object(full_pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run_mock,
            ):
                summary = full_pipeline.run_full_pipeline(args)
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(run_mock.call_args.args[0], schedule_command)
        self.assertEqual(summary["skipped_stages"], ["input_preflight"])
        self.assertEqual(summary["executed_stages"], ["schedule_rows_seen"])

    def test_full_pipeline_does_not_skip_existing_gipo_outputs_without_resume_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            train_out = run_root / "policies" / "gipo" / "training"
            report_out = run_root / "policies" / "gipo" / "locked_test_reports" / "seen" / "4000_steps"
            train_out.mkdir(parents=True)
            report_out.mkdir(parents=True)
            (train_out / "gipo_training_summary.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            (train_out / "gipo_student.pt").write_bytes(b"checkpoint")
            for name in ("locked_test_gipo_policy_summary.json", "locked_test_gipo_comparison_summary.json"):
                (report_out / name).write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            for name in ("locked_test_gipo_rows.csv", "locked_test_gipo_aggregate_rows.csv", "locked_test_gipo_decisions.csv"):
                (report_out / name).write_text("scenario_key\nlobster_synthetic\n", encoding="utf-8")

            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(run_root),
                ]
            )
            train_command = [
                full_pipeline.sys.executable,
                "-m",
                "genode.gipo.train_gipo",
                "--out_dir",
                str(train_out),
            ]
            report_command = [
                full_pipeline.sys.executable,
                "-m",
                "genode.gipo.report_locked_test",
                "--out_dir",
                str(report_out),
            ]
            stages = [
                full_pipeline.StageCommand("train_gipo", [train_command], "gipo_training_manifest.json"),
                full_pipeline.StageCommand(
                    "report_gipo_locked_test",
                    [report_command],
                    "gipo_locked_test_manifest.json",
                ),
            ]
            with (
                mock.patch.object(full_pipeline, "_validate_inputs_preflight", return_value={"status": "complete"}),
                mock.patch.object(full_pipeline, "_build_stage_commands", return_value=stages),
                mock.patch.object(full_pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run_mock,
            ):
                summary = full_pipeline.run_full_pipeline(args)

            train_manifest = json.loads((run_root / "gipo_training_manifest.json").read_text(encoding="utf-8"))
            report_manifest = json.loads((run_root / "gipo_locked_test_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(summary["status"], "complete")
        self.assertNotIn("skipped", train_manifest["command_results"][0])
        self.assertNotIn("skipped", report_manifest["command_results"][0])

    def test_full_pipeline_overwrite_reruns_existing_gipo_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            train_out = run_root / "policies" / "gipo" / "training"
            train_out.mkdir(parents=True)
            (train_out / "gipo_training_summary.json").write_text(json.dumps({"status": "old"}), encoding="utf-8")
            (train_out / "gipo_student.pt").write_bytes(b"old")

            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(run_root),
                    "--overwrite",
                ]
            )
            train_command = [
                full_pipeline.sys.executable,
                "-m",
                "genode.gipo.train_gipo",
                "--out_dir",
                str(train_out),
            ]
            stage = full_pipeline.StageCommand("train_gipo", [train_command], "gipo_training_manifest.json")
            with (
                mock.patch.object(full_pipeline, "_validate_inputs_preflight", return_value={"status": "complete"}),
                mock.patch.object(full_pipeline, "_build_stage_commands", return_value=[stage]),
                mock.patch.object(full_pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run_mock,
            ):
                summary = full_pipeline.run_full_pipeline(args)
            manifest = json.loads((run_root / "gipo_training_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(summary["status"], "complete")
        self.assertNotIn("skipped", manifest["command_results"][0])

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
        self.assertNotIn("--checkpoint_export_mode", backbone_command)

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
            member = {
                "member_key": "Dynamic_A",
                "stratum": "Dynamic_A",
                "source_zip_name": "molecule.zip",
                "processed_dir": "Dynamic_A",
                "trainable": True,
            }
            with mock.patch.object(full_pipeline, "load_molecule_group_manifest", return_value={"strata": [member]}):
                commands = full_pipeline._backbone_training_commands(args, "molecule_3d_set1", "4000,8000")

        command = " ".join(str(part) for part in commands[0])
        self.assertIn("--backbone_manifest", command)
        self.assertIn(str(Path(tmpdir) / "matrix" / "backbone_manifest.json"), command)

    def test_full_pipeline_gipo_stage_does_not_use_unseen_selection_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--stages",
                    "train_gipo,train_unseen_target_student",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
        by_stage = {stage["stage"]: [" ".join(command) for command in stage["commands"]] for stage in summary["stages"]}
        gipo_command = " ".join(by_stage["train_gipo"])
        unseen_target_command = " ".join(by_stage["train_unseen_target_student"])
        policy = gipo_policy()
        self.assertNotIn("--teacher_unseen_selection_rows_csv", gipo_command)
        self.assertNotIn("--teacher_unseen_selection_context_embeddings_npz", gipo_command)
        self.assertIn("policies/gipo", gipo_command.replace("\\", "/"))
        self.assertIn(
            f"--student_teacher_score_weight {float(policy.student_teacher_score_weight)}",
            gipo_command,
        )
        self.assertIn(
            f"--student_target_mixture_mode {policy.student_target_mixture_mode}",
            gipo_command,
        )
        self.assertNotIn("--student_unseen_target_rows_csv", gipo_command)
        for command in (gipo_command, unseen_target_command):
            self.assertIn("--teacher_metric_target_keys u_temporal_uw1_uniform,u_temporal_cw1_uniform,u_temporal_tstr_f1_uniform", command)
            self.assertIn("--teacher_steps 500", command)
            self.assertIn("--student_steps 500", command)
            self.assertIn("--seen_target_nfe_values 4,8,12,16", command)
            self.assertIn("--unseen_target_nfe_values 6,10,14,20", command)
            self.assertIn("--student_teacher_score_weight 0.01", command)
            self.assertIn("--student_teacher_score_warmup_fraction 0.6", command)
            self.assertIn("--student_target_mixture_mode full", command)
            self.assertIn("--student_target_elite_fraction 0.3", command)
            self.assertIn("--student_target_elite_k 0", command)
            self.assertIn("--student_target_elite_min_count 2", command)
            self.assertIn("--student_target_elite_blend_all_weight 0.2", command)
        self.assertIn("--student_unseen_target_rows_csv", unseen_target_command)
        self.assertNotIn(
            "--student_teacher_score_include_unseen_targets",
            unseen_target_command,
        )
        self.assertIn("--teacher_utility_weights", unseen_target_command)
        protocol = full_pipeline._protocol_payload(args)
        self.assertEqual(protocol["gipo_teacher_steps"], 500)
        self.assertEqual(protocol["gipo_student_steps"], 500)
        self.assertEqual(protocol["gipo_policy_key"], GIPO_POLICY_KEY)
        self.assertEqual(protocol["student_teacher_score_weight"], 0.01)
        self.assertEqual(protocol["student_teacher_score_clip"], 5.0)
        self.assertEqual(protocol["student_teacher_score_protocol"], "late_ramped_per_cell_teacher_utility_z_score")
        self.assertEqual(protocol["student_target_mixture_mode"], "full")

    def test_full_pipeline_runs_only_gipo_policy_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(run_root),
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
            protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))

        stage_names = [stage["stage"] for stage in summary["stages"]]
        self.assertIn("train_gipo", stage_names)
        self.assertIn("report_gipo_locked_test", stage_names)
        self.assertNotIn("train_unseen_target_student", stage_names)
        self.assertNotIn("train_ablation_students", stage_names)

        by_stage = {stage["stage"]: stage["commands"] for stage in summary["stages"]}
        train_command = " ".join(by_stage["train_gipo"][0])
        self.assertIn("policies/gipo", train_command.replace("\\", "/"))
        self.assertIn("--student_teacher_score_weight 0.01", train_command)
        self.assertIn("--student_target_mixture_mode full", train_command)
        self.assertNotIn("--student_unseen_target_rows_csv", train_command)

        report_commands = by_stage["report_gipo_locked_test"]
        self.assertEqual(len(report_commands), 10)
        report_text = [" ".join(command) for command in report_commands]
        self.assertTrue(all("policies/gipo" in command.replace("\\", "/") for command in report_text))
        self.assertTrue(all("gipo_ablations" not in command for command in report_text))
        self.assertEqual(
            {command[command.index("--target_nfe_values") + 1] for command in report_commands},
            {"4,8,12,16", "6,10,14,20"},
        )
        self.assertEqual(
            {
                command[command.index("--out_dir") + 1]
                .replace("\\", "/")
                .split("/locked_test_reports/")[1]
                .split("/")[0]
                for command in report_commands
            },
            {"seen", "unseen"},
        )
        self.assertEqual(protocol["gipo_policy_key"], GIPO_POLICY_KEY)
        self.assertEqual(
            protocol["gipo_policy"]["student_objective_settings"]["student_teacher_score_weight"],
            0.01,
        )
        self.assertFalse(protocol["include_ablations"])

    def test_full_pipeline_rejects_objective_overrides_without_unseen_target_stage(self) -> None:
        cases = (
            (),
            ("--include_ablations",),
        )
        for extra_args in cases:
            with self.subTest(extra_args=extra_args), tempfile.TemporaryDirectory() as tmpdir:
                args = full_pipeline.build_argparser().parse_args(
                    [
                        "--scenario_key",
                        "lobster_synthetic",
                        "--run_root",
                        str(Path(tmpdir) / "run"),
                        "--student_teacher_score_weight",
                        "0.07",
                        "--dry_run",
                        *extra_args,
                    ]
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "only apply to the explicit train_unseen_target_student stage",
                ):
                    full_pipeline.run_full_pipeline(args)

    def test_full_pipeline_gipo_step_budgets_are_protocolized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run_a"),
                    "--stages",
                    "train_gipo",
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
                    "train_gipo",
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

    def test_full_pipeline_unseen_target_stage_objective_knobs_are_protocolized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run_a"),
                    "--stages",
                    "train_gipo,train_unseen_target_student",
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
                    "train_gipo,train_unseen_target_student",
                    "--student_teacher_score_weight",
                    "0.07",
                    "--dry_run",
                ]
            )

        base_protocol = full_pipeline._protocol_payload(base_args)
        changed_protocol = full_pipeline._protocol_payload(changed_args)
        self.assertEqual(base_protocol["student_teacher_score_weight"], 0.01)
        self.assertEqual(changed_protocol["student_teacher_score_weight"], 0.01)
        self.assertEqual(
            base_protocol["unseen_target_student_objective_settings"]["student_teacher_score_weight"],
            0.01,
        )
        self.assertEqual(
            changed_protocol["unseen_target_student_objective_settings"]["student_teacher_score_weight"],
            0.07,
        )
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
                    "train_gipo,train_unseen_target_student",
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

    def test_ablation_preset_excludes_first_class_gipo_policy(self) -> None:
        policies = ablation_student_policies(ABLATION_PRESET_ALL)
        self.assertEqual(len(policies), 15)
        self.assertEqual(len({policy.policy_key for policy in policies}), 15)
        policy = gipo_policy()
        self.assertEqual(policy.policy_key, GIPO_POLICY_KEY)
        self.assertEqual(policy.student_training_mode, "seen_only_zero_shot")
        self.assertEqual(policy.student_target_mixture_mode, "full")
        self.assertEqual(float(policy.student_teacher_score_weight), 0.01)
        self.assertNotIn(GIPO_POLICY_KEY, {item.policy_key for item in policies})
        self.assertEqual([policy.comparison_group for policy in policies[:6]], ["main"] * 6)
        self.assertNotIn("elite-bend", " ".join(policy.policy_key for policy in policies))
        full_scores_by_mode = {}
        blend_weights_by_mode = {}
        for policy in policies:
            if policy.student_target_mixture_mode == "full":
                full_scores_by_mode.setdefault(policy.student_training_mode, set()).add(float(policy.student_teacher_score_weight))
            if policy.student_target_mixture_mode == "elite_blend":
                blend_weights_by_mode.setdefault(policy.student_training_mode, set()).add(float(policy.student_target_elite_blend_all_weight))
        self.assertEqual(set(full_scores_by_mode), {"seen_only_zero_shot", "seen_plus_unseen_target"})
        self.assertEqual(full_scores_by_mode["seen_only_zero_shot"], {0.0, 0.05, 0.10})
        self.assertEqual(full_scores_by_mode["seen_plus_unseen_target"], {0.0, 0.01, 0.05, 0.10})
        for weights in blend_weights_by_mode.values():
            self.assertEqual(weights, {0.10, 0.20, 0.40})

    def test_full_pipeline_include_ablations_adds_grid_after_gipo_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(run_root),
                    "--include_ablations",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
            manifest = json.loads(
                (run_root / "gipo_ablations" / ABLATION_PRESET_ALL / "ablation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            [stage["stage"] for stage in summary["stages"]],
            [
                "input_preflight",
                "backbone_training",
                "ser_summaries",
                "schedule_rows_seen",
                "schedule_rows_unseen",
                "train_gipo",
                "report_gipo_locked_test",
                "train_ablation_students",
                "report_ablation_locked_test",
            ],
        )
        self.assertTrue(any(stage["stage"] == "backbone_training" for stage in summary["stages"]))
        by_stage = {stage["stage"]: stage["commands"] for stage in summary["stages"]}
        self.assertEqual(len(by_stage["train_ablation_students"]), 15)
        self.assertEqual(len(by_stage["report_ablation_locked_test"]), 150)
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
        self.assertEqual(manifest["student_policy_count"], 15)
        self.assertEqual(manifest["gipo_teacher_steps"], 500)
        self.assertEqual(manifest["gipo_student_steps"], 500)
        self.assertEqual(manifest["ablation_root"], f"gipo_ablations/{ABLATION_PRESET_ALL}")
        self.assertTrue(
            all(
                not str(value).startswith(("/", "C:"))
                for policy in manifest["student_policies"]
                for value in policy["outputs"].values()
            )
        )

    def test_full_pipeline_requires_one_scenario_key_without_aliases(self) -> None:
        parser = full_pipeline.build_argparser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--dataset", "solar_energy_10m"])

    def test_full_pipeline_ser_controls_are_protocolized_and_ser_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
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
        self.assertEqual(protocol["ser_example_selection_protocol"], "ser_ptg_reference_global_context_selection")
        self.assertEqual(protocol["ser_local_defect_proxy_protocol"], "otflow_midpoint_local_defect_proxy")
        self.assertEqual(protocol["schedule_context_selection_protocol"], "schedule_evaluation_phase_context_selection")

    def test_full_pipeline_default_ser_train_tuning_cap_tracks_context_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
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
        self.assertEqual(protocol["context_sample_count"], 123)

    def test_full_pipeline_locked_test_is_full_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--context_sample_count",
                    "7",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
        by_stage = {stage["stage"]: stage["commands"] for stage in summary["stages"]}
        schedule_commands = by_stage["schedule_rows_seen"] + by_stage["schedule_rows_unseen"]
        train_commands = [command for command in schedule_commands if command[command.index("--split_phase") + 1] == "train_tuning"]
        locked_commands = [command for command in schedule_commands if command[command.index("--split_phase") + 1] == "locked_test"]
        self.assertTrue(train_commands)
        self.assertTrue(locked_commands)
        for command in train_commands:
            self.assertIn("--context_sample_count", command)
            self.assertEqual(command[command.index("--context_sample_count") + 1], "7")
        for command in locked_commands:
            self.assertNotIn("--context_sample_count", command)
            self.assertNotIn("--locked_test_preview", command)
            self.assertNotIn("--locked_test_preview_contexts", command)
            self.assertNotIn("--eval_windows_val", command)
        self.assertFalse(any(command[command.index("--split_phase") + 1] == "validation_tuning" for command in schedule_commands))
        protocol = full_pipeline._protocol_payload(args)
        self.assertEqual(protocol["schedule_row_split_phases"], ["train_tuning", "locked_test"])
        self.assertEqual(protocol["locked_test_mode"], "full")
        self.assertIsNone(protocol["locked_test_context_limit"])
        self.assertEqual(protocol["locked_test_context_limit_scope"], "none")
        gipo_commands = [" ".join(command) for command in by_stage["train_gipo"]]
        self.assertTrue(all("--context_sample_count 7" in command for command in gipo_commands))
        report_commands = [" ".join(command) for command in by_stage["report_gipo_locked_test"]]
        self.assertTrue(all("--context_sample_count" not in command for command in report_commands))

    def test_full_pipeline_locked_test_preview_maps_per_seed_context_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--context_sample_count",
                    "7",
                    "--locked_test_preview",
                    "--locked_test_preview_contexts",
                    "19",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
        by_stage = {stage["stage"]: stage["commands"] for stage in summary["stages"]}
        schedule_commands = by_stage["schedule_rows_seen"] + by_stage["schedule_rows_unseen"]
        locked_commands = [command for command in schedule_commands if command[command.index("--split_phase") + 1] == "locked_test"]
        validation_commands = [command for command in schedule_commands if command[command.index("--split_phase") + 1] == "validation_tuning"]
        self.assertEqual(validation_commands, [])
        self.assertFalse(any("--eval_windows_val" in command for command in schedule_commands))
        self.assertTrue(locked_commands)
        self.assertTrue(all("--seeds" not in command for command in locked_commands))
        self.assertTrue(all("--locked_test_preview" in command for command in locked_commands))
        self.assertTrue(all(command[command.index("--locked_test_preview_contexts") + 1] == "19" for command in locked_commands))
        self.assertTrue(all("--context_sample_count" not in command for command in locked_commands))
        protocol = full_pipeline._protocol_payload(args)
        self.assertEqual(protocol["schedule_row_split_phases"], ["train_tuning", "locked_test"])
        self.assertEqual(protocol["locked_test_mode"], "preview")
        self.assertEqual(protocol["locked_test_context_limit"], 19)
        self.assertEqual(protocol["locked_test_context_limit_scope"], "per_seed")

    def test_full_pipeline_preview_uses_512_and_rejects_inactive_limit(self) -> None:
        preview_args = full_pipeline.build_argparser().parse_args(
            ["--scenario_key", "lobster_synthetic", "--locked_test_preview", "--dry_run"]
        )
        self.assertEqual(full_pipeline._protocol_payload(preview_args)["locked_test_context_limit"], 512)
        invalid_args = full_pipeline.build_argparser().parse_args(
            ["--scenario_key", "lobster_synthetic", "--locked_test_preview_contexts", "19", "--dry_run"]
        )
        with self.assertRaisesRegex(ValueError, "requires --locked_test_preview"):
            full_pipeline._validate_inputs_preflight(invalid_args)

    def test_full_pipeline_no_ser_fixed_support_skips_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--stages",
                    "input_preflight,schedule_rows_seen,schedule_rows_unseen,train_ablation_students,report_ablation_locked_test",
                    "--schedule_keys",
                    "uniform,late_power_3",
                    "--checkpoint_steps",
                    "4000",
                    "--dry_run",
                ]
            )
            summary = full_pipeline.run_full_pipeline(args)
            protocol = full_pipeline._protocol_payload(args)
        by_stage = {stage["stage"]: stage["commands"] for stage in summary["stages"]}
        self.assertTrue(protocol["include_ablations"])
        self.assertNotIn("ser_summaries", by_stage)
        schedule_commands = by_stage["schedule_rows_seen"] + by_stage["schedule_rows_unseen"]
        split_phases = [command[command.index("--split_phase") + 1] for command in schedule_commands]
        self.assertEqual(set(split_phases), {"train_tuning", "locked_test"})
        self.assertEqual(len(by_stage["schedule_rows_seen"]), 2)
        self.assertEqual(len(by_stage["schedule_rows_unseen"]), 2)
        self.assertFalse(any("ser_ptg" in " ".join(command) for command in schedule_commands))
        self.assertFalse(any("--eval_windows_val" in command for command in schedule_commands))
        self.assertFalse(any(phase == "validation_tuning" for phase in split_phases))
        for command in schedule_commands:
            if command[command.index("--split_phase") + 1] == "train_tuning":
                self.assertEqual(command[command.index("--context_sample_count") + 1], "188")
            else:
                self.assertNotIn("--context_sample_count", command)
        gipo_commands = [" ".join(command) for command in by_stage["train_ablation_students"]]
        self.assertTrue(
            all(
                "--rows_csv" in command
                and "/train_tuning/" in command.replace("\\", "/")
                and "context_rows.csv" in command
                for command in gipo_commands
            )
        )
        self.assertTrue(all("--support_schedule_keys uniform,late_power_3" in command for command in gipo_commands))
        report_commands = [" ".join(command).replace("\\", "/") for command in by_stage["report_ablation_locked_test"]]
        self.assertTrue(all("/locked_test/" in command and "context_rows.csv" in command for command in report_commands))

    def test_full_pipeline_include_ablations_routes_policy_knobs_and_locked_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--include_ablations",
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
        train_commands = by_stage["train_ablation_students"]
        report_commands = by_stage["report_ablation_locked_test"]
        for policy in ablation_student_policies(ABLATION_PRESET_ALL):
            command = next(item for item in train_commands if policy.policy_key in item)
            self.assertIn(f"--student_policy_key {policy.policy_key}", command)
            self.assertIn(f"--student_target_mixture_mode {policy.student_target_mixture_mode}", command)
            self.assertIn(f"--student_teacher_score_weight {float(policy.student_teacher_score_weight)}", command)
            self.assertIn("--student_teacher_score_warmup_fraction 0.6", command)
            self.assertIn("--teacher_steps 37", command)
            self.assertIn("--student_steps 43", command)
            self.assertIn("--seen_target_nfe_values 4,8", command)
            self.assertIn("--unseen_target_nfe_values 6,10", command)
            self.assertIn("--schedule_summary_json", command)
            self.assertNotIn("--student_teacher_score_include_unseen_targets", command)
            if policy.student_target_mixture_mode == "elite_blend":
                self.assertIn(f"--student_target_elite_blend_all_weight {float(policy.student_target_elite_blend_all_weight)}", command)
            if policy.uses_unseen_targets:
                self.assertIn("--student_unseen_target_rows_csv", command)
                self.assertIn("--student_unseen_target_schedule_summary_json", command)
                self.assertIn("--student_unseen_target_weight 0.25", command)
            else:
                self.assertNotIn("--student_unseen_target_rows_csv", command)
                self.assertNotIn("--student_unseen_target_schedule_summary_json", command)
        self.assertTrue(all("gipo_ablations" in command for command in report_commands))
        self.assertTrue(all("locked_test" in command for command in report_commands))
        self.assertTrue(all("--training_summary" in command and "--gipo_student_checkpoint" in command for command in report_commands))

    def test_full_pipeline_ablation_outputs_are_namespaced_below_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_root = Path(tmpdir) / "outputs"
            run_root = outputs_root / "full_pipeline" / "lobster_synthetic"
            expected_ablation_root = run_root / "gipo_ablations" / ABLATION_PRESET_ALL
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--include_ablations",
                    "--dry_run",
                ]
            )
            with mock.patch.object(full_pipeline, "project_outputs_root", return_value=outputs_root):
                summary = full_pipeline.run_full_pipeline(args)
            self.assertTrue((run_root / "protocol.json").exists())
            self.assertTrue((run_root / "ablation_students_manifest.json").exists())
            self.assertTrue((expected_ablation_root / "ablation_manifest.json").exists())
            self.assertNotIn("gipo_ablations", summary["run_root"])

    def test_full_pipeline_failed_ablation_stage_marks_ablation_manifest_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            args = full_pipeline.build_argparser().parse_args(
                [
                    "--scenario_key",
                    "lobster_synthetic",
                    "--run_root",
                    str(run_root),
                    "--include_ablations",
                ]
            )
            failing_stage = full_pipeline.StageCommand(
                full_pipeline.ABLATION_STUDENT_STAGE,
                [[full_pipeline.sys.executable, "-c", "import sys; sys.exit(3)"]],
                "ablation_students_manifest.json",
            )
            with (
                mock.patch.object(full_pipeline, "_validate_inputs_preflight", return_value={"status": "complete"}),
                mock.patch.object(full_pipeline, "_build_stage_commands", return_value=[failing_stage]),
            ):
                with self.assertRaises(RuntimeError):
                    full_pipeline.run_full_pipeline(args)
            manifest = json.loads(
                (run_root / "gipo_ablations" / ABLATION_PRESET_ALL / "ablation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failed_stage"], full_pipeline.ABLATION_STUDENT_STAGE)
        self.assertEqual(manifest["failed_command_index"], 0)


if __name__ == "__main__":
    unittest.main()
