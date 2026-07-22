from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import torch

import genode.evaluation.diffusion_flow_time_reparameterization as runner
from genode.experiment_layout import REFERENCE_SEEN_NFES
from genode.schedule_transfer.diffusion_flow_schedules import (
    BASELINE_SCHEDULE_KEYS,
    EXPERIMENTAL_FIXED_SCHEDULE_KEYS,
    EXPERIMENTAL_REVERSED_SCHEDULE_KEYS,
    TRANSFER_SCHEDULE_KEYS,
    build_schedule_grid,
)
from genode.evaluation.fm_backbone_registry import materialize_backbone_manifest
from genode.schedule_transfer.otflow_reference_registry import (
    MAIN_NFE_VALUES,
    METHOD_KEY,
    reference_registry_snapshot,
    reference_schedule_specs,
    reference_solver_specs,
)
from genode.schedule_transfer.otflow_signal_traces import NATIVE_INFO_GROWTH_TRACE_KEY

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DiffusionFlowReferencePrepTests(unittest.TestCase):
    def test_registry_exposes_diffusion_flow_method_not_tvd(self) -> None:
        snapshot = reference_registry_snapshot()
        self.assertEqual(METHOD_KEY, "diffusion_flow_time_reparameterization")
        self.assertEqual(snapshot["method_key"], "diffusion_flow_time_reparameterization")
        self.assertNotIn("tvd", {spec.key for spec in reference_schedule_specs()})
        self.assertIn("flowts_power_sampling", {spec.key for spec in reference_schedule_specs()})
        self.assertNotIn("atss", {spec.key for spec in reference_schedule_specs()})

    def test_schedule_sets_are_exact(self) -> None:
        self.assertEqual(BASELINE_SCHEDULE_KEYS, ("uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots"))
        self.assertEqual(TRANSFER_SCHEDULE_KEYS, ("ays", "gits", "ots"))
        self.assertNotIn("uniform_reversed", EXPERIMENTAL_REVERSED_SCHEDULE_KEYS)
        self.assertEqual(EXPERIMENTAL_FIXED_SCHEDULE_KEYS[: len(BASELINE_SCHEDULE_KEYS)], BASELINE_SCHEDULE_KEYS)

    def test_registry_exposes_active_baseline_matrix(self) -> None:
        snapshot = reference_registry_snapshot()
        self.assertEqual(MAIN_NFE_VALUES, (4, 8, 12, 16))
        self.assertEqual(snapshot["main_nfe_values"], [4, 8, 12, 16])
        self.assertEqual(REFERENCE_SEEN_NFES, (4, 8, 12, 16))
        self.assertEqual(snapshot["baseline_scheduler_keys"], ["uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots"])

        solver_names = {spec.display_name for spec in reference_solver_specs()}
        self.assertIn("Euler", solver_names)
        self.assertIn("Heun / RK2", solver_names)
        self.assertIn("Midpoint RK2", solver_names)
        self.assertIn("DPM++2M", solver_names)

    def test_readme_describes_gipo_and_distillation_theory(self) -> None:
        text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        lower = text.lower()

        for expected in (
            "normalized time density",
            "inverse cdf",
            "metric-aware teacher",
            "teacher-weighted continuous density",
            "context-disjoint",
            "locked-test",
            "endpoint flow map",
            "terminal identity",
            "strict superiority",
            "transformer otflow",
            "genode-train-gipo",
            "genode-report-gipo-locked-test",
        ):
            self.assertIn(expected, lower)
        for retired in (
            "categorical support",
            "top-1/top-2 categorical",
            "static residual",
            "oracle_context",
            "best_static",
            "support recall",
            "soft penalties",
            "teacher ensemble",
        ):
            self.assertNotIn(retired, lower)

    def test_active_schedule_grids_have_endpoints(self) -> None:
        for key in BASELINE_SCHEDULE_KEYS:
            grid = build_schedule_grid(key, 4)
            self.assertIsNotNone(grid, key)
            self.assertEqual(len(grid), 5)
            self.assertAlmostEqual(grid[0], 0.0)
            self.assertAlmostEqual(grid[-1], 1.0)
            self.assertTrue(all(right > left for left, right in zip(grid, grid[1:])), key)

    def test_active_schedule_grids_reject_non_positive_steps(self) -> None:
        for key in BASELINE_SCHEDULE_KEYS:
            for n_steps in (0, -1):
                with self.assertRaisesRegex(ValueError, "n_steps must be positive"):
                    build_schedule_grid(key, n_steps)

    def test_scheduler_cases_evaluate_uniform_first(self) -> None:
        args = runner.build_argparser().parse_args(["--baseline_scheduler_names", "ays,uniform"])
        cases = runner._scheduler_cases_for_datasets(args, ["traffic_hourly"])
        self.assertEqual([case["scheduler_key"] for case in cases["traffic_hourly"]], ["uniform", "ays"])

    def test_scheduler_cases_accept_explicit_experimental_reversed_keys(self) -> None:
        args = runner.build_argparser().parse_args(["--baseline_scheduler_names", "ays_reversed,uniform"])
        cases = runner._scheduler_cases_for_datasets(args, ["traffic_hourly"])
        self.assertEqual([case["scheduler_key"] for case in cases["traffic_hourly"]], ["uniform", "ays_reversed"])
        default_args = runner.build_argparser().parse_args([])
        self.assertNotIn("ays_reversed", runner._parse_schedule_names(default_args.baseline_scheduler_names))

    def test_scheduler_cases_reject_duplicate_fixed_schedule_names(self) -> None:
        args = runner.build_argparser().parse_args(["--baseline_scheduler_names", "uniform,ays,uniform"])
        with self.assertRaisesRegex(ValueError, "Duplicate fixed diffusion-flow schedules"):
            runner._scheduler_cases_for_datasets(args, ["traffic_hourly"])

    def test_scheduler_cases_reject_duplicate_requested_summary_schedule_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schedules": [
                            {
                                "scheduler_key": "gipo",
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = runner.build_argparser().parse_args(
                [
                    "--baseline_scheduler_names",
                    "uniform",
                    "--schedule_summary_json",
                    str(summary_path),
                    "--summary_scheduler_names",
                    "gipo,gipo",
                ]
            )
            with self.assertRaisesRegex(ValueError, "Duplicate summary diffusion-flow schedules"):
                runner._scheduler_cases_for_datasets(args, ["traffic_hourly"])

    def test_scheduler_cases_reject_duplicate_summary_case_identities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            prediction = {
                "solver_key": "euler",
                "target_nfe": 4,
                "macro_steps": 4,
                "time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            }
            summary_path.write_text(
                json.dumps({"schedules": [{"scheduler_key": "gipo", "predictions": [prediction, prediction]}]}),
                encoding="utf-8",
            )
            args = runner.build_argparser().parse_args(
                [
                    "--baseline_scheduler_names",
                    "uniform",
                    "--schedule_summary_json",
                    str(summary_path),
                    "--summary_scheduler_names",
                    "gipo",
                ]
            )
            with self.assertRaisesRegex(ValueError, "Duplicate scheduler cases"):
                runner._scheduler_cases_for_datasets(args, ["traffic_hourly"])

    def test_scheduler_cases_reject_summary_schedule_that_duplicates_fixed_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schedules": [
                            {
                                "scheduler_key": "uniform",
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = runner.build_argparser().parse_args(
                [
                    "--baseline_scheduler_names",
                    "uniform",
                    "--schedule_summary_json",
                    str(summary_path),
                    "--summary_scheduler_names",
                    "uniform",
                ]
            )
            with self.assertRaisesRegex(ValueError, "Summary-derived schedules duplicate fixed schedules"):
                runner._scheduler_cases_for_datasets(args, ["traffic_hourly"])

    def test_schedule_summary_cases_normalize_rk2_nfe_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "schedules": [
                            {
                                "scheduler_key": "gipo",
                                "predictions": [
                                    {
                                        "solver_key": "heun",
                                        "target_nfe": 4,
                                        "macro_steps": 2,
                                        "time_grid": [0.0, 0.5, 1.0],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cases = runner._load_schedule_summary_cases(str(path))

        self.assertEqual(cases[0]["macro_steps"], 2)
        self.assertEqual(cases[0]["macro_steps"], 2)
        self.assertEqual(cases[0]["realized_nfe"], 4)

    def test_summary_scheduler_cases_are_family_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "schedules": [
                            {
                                "scheduler_key": "gipo",
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = runner.build_argparser().parse_args(
                [
                    "--baseline_scheduler_names",
                    "uniform",
                    "--schedule_summary_json",
                    str(path),
                    "--summary_scheduler_names",
                    "gipo",
                ]
            )
            forecast_cases = runner._scheduler_cases_for_datasets(args, ["traffic_hourly"], include_summary_cases=True)
            conditional_cases = runner._scheduler_cases_for_datasets(args, ["lobster_synthetic"], include_summary_cases=True)

        self.assertEqual([case["scheduler_key"] for case in forecast_cases["traffic_hourly"]], ["uniform", "gipo"])
        self.assertEqual([case["scheduler_key"] for case in conditional_cases["lobster_synthetic"]], ["uniform", "gipo"])

    def test_aggregate_relative_gain_uses_fraction_units(self) -> None:
        rows = [
            {
                "benchmark_family": "temporal_extrapolation",
                "split_phase": "locked_test",
                "seed": 0,
                "scenario_key": "traffic_hourly",
                "checkpoint_id": "ck",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_step": 20000,
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "ays",
                "experiment_scope": "main",
                "row_status": "complete",
                "forecast_crps": 3.0,
            },
            {
                "benchmark_family": "temporal_extrapolation",
                "split_phase": "locked_test",
                "seed": 0,
                "scenario_key": "traffic_hourly",
                "checkpoint_id": "ck",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
                "checkpoint_step": 20000,
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "experiment_scope": "main",
                "row_status": "complete",
                "forecast_crps": 4.0,
            },
        ]

        summary = runner._aggregate_main_table(rows)["seed_summaries"]
        by_schedule = {row["scheduler_key"]: row for row in summary}

        self.assertAlmostEqual(runner._safe_relative_gain(3.0, 4.0), 0.25)
        self.assertAlmostEqual(by_schedule["ays"]["forecast_relative_crps_gain_vs_uniform"], 0.25)
        self.assertAlmostEqual(by_schedule["uniform"]["forecast_relative_crps_gain_vs_uniform"], 0.0)

    def test_native_hardness_trace_is_info_growth(self) -> None:
        self.assertEqual(NATIVE_INFO_GROWTH_TRACE_KEY, "info_growth_hardness_by_step")

    def test_runner_dry_run_writes_combined_summary(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                ]
            )
            payload = runner.run_diffusion_flow_time_reparameterization(args)
            summary = json.loads((Path(tmpdir) / "combined_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["runner_mode"], "diffusion_flow_time_reparameterization")
        self.assertEqual(summary["method_key"], "diffusion_flow_time_reparameterization")
        self.assertEqual(summary["conditional_generation_datasets"], [])
        retired_dataset_key = "lo" + "b_datasets"
        self.assertNotIn(retired_dataset_key, summary)
        self.assertIn("flowts_power_sampling", summary["baseline_scheduler_keys"])
        self.assertEqual(summary["transfer_schedule_keys"], ["ays", "gits", "ots"])

    def test_locked_test_dry_run_combined_summary_round_trips_full_and_preview_provenance(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        expected_by_preview = {
            False: {
                "locked_test_mode": "full",
                "locked_test_context_limit": None,
                "locked_test_context_limit_scope": "none",
                "selected_examples_cap_source": "locked_test_full",
            },
            True: {
                "locked_test_mode": "preview",
                "locked_test_context_limit": 512,
                "locked_test_context_limit_scope": "per_seed",
                "selected_examples_cap_source": "locked_test_preview_contexts",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for preview, expected in expected_by_preview.items():
                with self.subTest(preview=preview):
                    out_root = root / ("preview" if preview else "full")
                    argv = [
                        "--out_root",
                        str(out_root),
                        "--forecast_datasets",
                        "",
                        "--conditional_generation_datasets",
                        "",
                        "--split_phase",
                        "locked_test",
                        "--backbone_manifest",
                        str(manifest),
                    ]
                    if preview:
                        argv.append("--locked_test_preview")
                    returned = runner.run_diffusion_flow_time_reparameterization(
                        runner.build_argparser().parse_args(argv)
                    )
                    persisted = json.loads((out_root / "combined_summary.json").read_text(encoding="utf-8"))
                    for key, value in expected.items():
                        self.assertEqual(returned[key], value)
                        self.assertEqual(persisted[key], value)
                    self.assertFalse(returned["selection_was_capped"])
                    self.assertFalse(returned["global_selection_was_capped"])
                    self.assertFalse(persisted["selection_was_capped"])
                    self.assertFalse(persisted["global_selection_was_capped"])

    def test_runner_rejects_wrong_schedule_summary_scope_before_evaluation_phase(self) -> None:
        mismatches = (
            ({"scenario_key": "weather", "checkpoint_step": 20000}, "scenarios are not selected"),
            ({"scenario_key": "traffic_hourly", "checkpoint_step": 4000}, "checkpoint_step values are not selected"),
        )
        for index, (metadata, message) in enumerate(mismatches):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                summary_path = root / f"summary_{index}.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            **metadata,
                            "schedules": [
                                {
                                    "scheduler_key": "gipo",
                                    "predictions": [
                                        {
                                            "solver_key": "euler",
                                            "target_nfe": 4,
                                            "macro_steps": 4,
                                            "time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
                                        }
                                    ],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                args = runner.build_argparser().parse_args(
                    [
                        "--out_root",
                        str(root / "out"),
                        "--forecast_datasets",
                        "traffic_hourly",
                        "--conditional_generation_datasets",
                        "",
                        "--baseline_scheduler_names",
                        "uniform",
                        "--schedule_summary_json",
                        str(summary_path),
                        "--summary_scheduler_names",
                        "gipo",
                        "--target_nfe_values",
                        "4",
                        "--solver_names",
                        "euler",
                        "--checkpoint_steps",
                        "20000",
                        "--split_phase",
                        "validation_tuning",
                        "--allow_execute",
                    ]
                )
                with mock.patch.object(runner, "validate_execution_preflight"), mock.patch.object(
                    runner,
                    "_run_forecast_phase",
                ) as evaluation_phase:
                    with self.assertRaisesRegex(ValueError, message):
                        runner.run_diffusion_flow_time_reparameterization(args)
                evaluation_phase.assert_not_called()

    def test_conditional_generation_build_row_preserves_full_metrics(self) -> None:
        row = runner._build_row(
            benchmark_family="temporal_conditional_generation",
            split_phase="locked_test",
            seed=0,
            dataset="cryptos",
            checkpoint={
                "checkpoint_id": "ck",
                "checkpoint_path": "outputs/example/model.pt",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
            },
            checkpoint_step=20000,
            nfe_role="seen",
            target_nfe=10,
            macro_steps=10,
            solver_key="euler",
            scheduler_key="uniform",
            context_embedding_kind="ctx_summary",
            details={"reference_macro_steps": 10, "schedule_grid_hash": "grid"},
            metrics={
                "score_main": 0.4,
                "temporal_tstr_f1": 0.5,
                "temporal_tstr_f1_applicable": True,
                "disc_auc": 0.6,
                "disc_auc_gap": 0.1,
                "temporal_uw1": 0.2,
                "temporal_cw1": 0.3,
                "u_l1": 0.7,
                "c_l1": 0.8,
                "spread_specific_error": 0.9,
                "imbalance_specific_error": 1.1,
                "ret_vol_acf_error": 1.2,
                "impact_response_error": 0.25,
                "eval_horizon": 3000,
                "evaluation_protocol_hash": "protocol",
                "chosen_t0s_hash": "windows",
                "chosen_examples_hash": "examples",
            },
            row_signature="sig",
            protocol_hash="hash",
        )

        for key in (
            "disc_auc",
            "disc_auc_gap",
            "temporal_tstr_f1",
            "temporal_tstr_f1_applicable",
            "temporal_uw1",
            "temporal_cw1",
            "u_l1",
            "c_l1",
            "spread_specific_error",
            "imbalance_specific_error",
            "ret_vol_acf_error",
            "impact_response_error",
        ):
            self.assertIn(key, row)
        self.assertEqual(row["eval_horizon"], 3000)
        self.assertEqual(row["schedule_grid_hash"], "grid")
        self.assertEqual(row["chosen_examples_hash"], "examples")
        self.assertEqual(row["context_embedding_kind"], "ctx_summary")
        self.assertIn("context_embedding_kind", runner.ROW_RECORD_FIELDS)

    def test_row_recorder_resume_rejects_protocol_mismatch_without_writing(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--target_nfe_values",
                    "4",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            recorder["fh"].close()
            run_config_path = Path(tmpdir) / "run_config.json"
            original_run_config = run_config_path.read_text(encoding="utf-8")
            row_path = Path(tmpdir) / "rows.jsonl"
            original_rows = '{"protocol_hash":"old","row_status":"complete"}\n'
            row_path.write_text(original_rows, encoding="utf-8")

            args_changed = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--target_nfe_values",
                    "8",
                ]
            )
            with self.assertRaisesRegex(ValueError, "different protocol"):
                runner._init_row_recorder(Path(tmpdir), args_changed)
            self.assertEqual(run_config_path.read_text(encoding="utf-8"), original_run_config)
            self.assertEqual(row_path.read_text(encoding="utf-8"), original_rows)

    def test_row_recorder_rejects_colliding_context_and_parent_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--write_context_rows",
                    "--row_csv_name",
                    "shared.csv",
                    "--context_row_csv_name",
                    "shared.csv",
                ]
            )

            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                runner._init_row_recorder(Path(tmpdir), args)

            self.assertFalse((Path(tmpdir) / "run_config.json").exists())

    def test_runner_rejects_schedule_summary_input_output_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_root = Path(tmpdir)
            collision_path = out_root / "combined_summary.json"
            original = b"schedule summary input must remain unchanged\n"
            collision_path.write_bytes(original)
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    str(out_root),
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--schedule_summary_json",
                    str(collision_path),
                ]
            )

            with self.assertRaisesRegex(ValueError, "input/output_collisions"):
                runner.run_diffusion_flow_time_reparameterization(args)

            self.assertEqual(collision_path.read_bytes(), original)
            self.assertFalse((out_root / "run_config.json").exists())

    def test_row_recorder_no_resume_explicitly_replaces_protocol_outputs(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--target_nfe_values",
                    "4",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            original_protocol_hash = recorder["protocol_hash"]
            recorder["fh"].close()
            row_path = Path(tmpdir) / "rows.jsonl"
            row_path.write_text(
                '{"protocol_hash":"old","row_status":"complete"}\n',
                encoding="utf-8",
            )

            replacement_args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--target_nfe_values",
                    "8",
                    "--no_resume",
                ]
            )
            replacement = runner._init_row_recorder(Path(tmpdir), replacement_args)
            replacement["fh"].close()

            self.assertNotEqual(replacement["protocol_hash"], original_protocol_hash)
            self.assertEqual(row_path.read_text(encoding="utf-8"), "")

    def test_row_recorder_resume_rejects_journal_without_run_config(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                ]
            )
            row_path = Path(tmpdir) / "rows.jsonl"
            original = '{"protocol_hash":"unknown","row_status":"complete"}\n'
            row_path.write_text(original, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "without run_config.json"):
                runner._init_row_recorder(Path(tmpdir), args)

            self.assertFalse((Path(tmpdir) / "run_config.json").exists())
            self.assertEqual(row_path.read_text(encoding="utf-8"), original)

    def test_row_recorder_resume_rejects_mixed_protocol_journal(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            protocol_hash = recorder["protocol_hash"]
            recorder["fh"].close()
            row_path = Path(tmpdir) / "rows.jsonl"
            journal = (
                json.dumps({"protocol_hash": protocol_hash, "row_status": "complete"})
                + "\n"
                + json.dumps({"protocol_hash": "other", "row_status": "complete"})
                + "\n"
            )
            row_path.write_text(journal, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "different protocol"):
                runner._init_row_recorder(Path(tmpdir), args)
            self.assertEqual(row_path.read_text(encoding="utf-8"), journal)

    def test_row_recorder_resume_discards_only_unterminated_final_fragment(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            protocol_hash = recorder["protocol_hash"]
            recorder["fh"].close()
            row = {"protocol_hash": protocol_hash, "row_status": "complete"}
            row_path = Path(tmpdir) / "rows.jsonl"
            row_path.write_text(
                json.dumps(row, sort_keys=True) + "\n" + '{"protocol_hash":',
                encoding="utf-8",
            )

            resumed = runner._init_row_recorder(Path(tmpdir), args)
            resumed["fh"].close()

            self.assertEqual(
                row_path.read_text(encoding="utf-8"),
                json.dumps(row, sort_keys=True) + "\n",
            )

    def test_row_loader_rejects_terminated_or_mid_file_corruption(self) -> None:
        valid = json.dumps({"protocol_hash": "protocol", "row_status": "complete"})
        malformed = '{"protocol_hash":'
        for journal in (
            valid + "\n" + malformed + "\n",
            valid + "\n" + malformed + "\n" + valid + "\n",
        ):
            with self.subTest(journal=journal), tempfile.TemporaryDirectory() as tmpdir:
                row_path = Path(tmpdir) / "rows.jsonl"
                row_path.write_text(journal, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "line 2"):
                    runner._load_rows(row_path, protocol_hash="protocol")

    def test_row_jsonl_compaction_preserves_target_on_serialization_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            row_path = Path(tmpdir) / "rows.jsonl"
            original = '{"existing":true}\n'
            row_path.write_text(original, encoding="utf-8")

            with self.assertRaises(TypeError):
                runner._write_row_jsonl(row_path, [{"not_json": {1}}])

            self.assertEqual(row_path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(Path(tmpdir).glob(".rows.jsonl.*.tmp")), [])

    def test_row_recorder_same_protocol_resume_preserves_rows_jsonl(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--target_nfe_values",
                    "4",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            protocol_hash = recorder["protocol_hash"]
            first = {
                "protocol_hash": protocol_hash,
                "benchmark_family": "temporal_extrapolation",
                "split_phase": "locked_test",
                "seed": 0,
                "scenario_key": "traffic_hourly",
                "target_nfe": 4,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "row_signature": "sig-a",
                "row_status": "complete",
            }
            runner._append_row_record(recorder, first)
            recorder["fh"].close()

            resumed = runner._init_row_recorder(Path(tmpdir), args)
            self.assertIn(runner._row_key(first), resumed["rows_by_key"])
            second = dict(first, scheduler_key="late_power_3", row_signature="sig-b")
            runner._append_row_record(resumed, second)
            resumed["fh"].close()

            lines = [line for line in (Path(tmpdir) / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["row_signature"], "sig-a")
            self.assertEqual(json.loads(lines[1])["row_signature"], "sig-b")

    def test_row_recorder_resume_compacts_duplicate_rows_jsonl(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--target_nfe_values",
                    "4",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            protocol_hash = recorder["protocol_hash"]
            row = {
                "protocol_hash": protocol_hash,
                "benchmark_family": "temporal_extrapolation",
                "split_phase": "locked_test",
                "seed": 0,
                "scenario_key": "traffic_hourly",
                "target_nfe": 4,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "row_signature": "sig-a",
                "row_status": "complete",
            }
            runner._append_row_record(recorder, row)
            recorder["fh"].write(json.dumps(row, sort_keys=True) + "\n")
            recorder["fh"].close()

            resumed = runner._init_row_recorder(Path(tmpdir), args)
            resumed["fh"].close()

            lines = [line for line in (Path(tmpdir) / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["row_signature"], "sig-a")

    def test_append_row_record_rejects_key_collision(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--target_nfe_values",
                    "4",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            protocol_hash = recorder["protocol_hash"]
            row = {
                "protocol_hash": protocol_hash,
                "benchmark_family": "temporal_extrapolation",
                "split_phase": "locked_test",
                "seed": 0,
                "scenario_key": "traffic_hourly",
                "target_nfe": 4,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "row_signature": "sig-a",
                "row_status": "complete",
                "score_main": 1.0,
            }
            runner._append_row_record(recorder, row)
            runner._append_row_record(recorder, row)
            collision = dict(row, score_main=2.0)
            with self.assertRaisesRegex(ValueError, "Schedule row key collision"):
                runner._append_row_record(recorder, collision)
            recorder["fh"].close()

            lines = [line for line in (Path(tmpdir) / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["score_main"], 1.0)

    def test_row_recorder_resume_does_not_skip_parent_with_incomplete_context_artifacts(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "traffic_hourly",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0",
                    "--checkpoint_steps",
                    "4000",
                    "--split_phase",
                    "locked_test",
                    "--write_context_rows",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            protocol_hash = recorder["protocol_hash"]
            parent = {
                "protocol_hash": protocol_hash,
                "benchmark_family": "temporal_extrapolation",
                "split_phase": "locked_test",
                "seed": 0,
                "scenario_key": "traffic_hourly",
                "checkpoint_step": 4000,
                "target_nfe": 4,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "row_signature": "parent-a",
                "row_status": "complete",
                "selected_examples": 2,
            }
            runner._append_row_record(recorder, parent)
            runner._write_context_row_csv(
                Path(tmpdir) / "context_rows.csv",
                [
                    {
                        "protocol_hash": protocol_hash,
                        "parent_row_signature": "parent-a",
                        "row_signature": "ctx-a",
                        "context_id": "ctx-a",
                        "context_embedding_id": "emb-a",
                        "scenario_key": "traffic_hourly",
                        "split_phase": "locked_test",
                        "seed": 0,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "scheduler_key": "uniform",
                        "checkpoint_id": "ck",
                    }
                ],
            )
            runner.save_context_embedding_table(Path(tmpdir) / "context_embeddings.npz", {"emb-a": [1.0, 0.0]})
            recorder["fh"].close()

            resumed = runner._init_row_recorder(Path(tmpdir), args)
            self.assertNotIn(runner._row_key(parent), resumed["rows_by_key"])
            runner._append_row_record(resumed, parent)
            resumed["fh"].close()
            lines = [line for line in (Path(tmpdir) / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(lines), 1)

    def test_schedule_row_output_status_rejects_duplicate_rows_jsonl_keys(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "traffic_hourly",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0",
                    "--checkpoint_steps",
                    "4000",
                    "--split_phase",
                    "locked_test",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            protocol_hash = recorder["protocol_hash"]
            row = {
                "protocol_hash": protocol_hash,
                "benchmark_family": "temporal_extrapolation",
                "split_phase": "locked_test",
                "seed": 0,
                "scenario_key": "traffic_hourly",
                "checkpoint_step": 4000,
                "target_nfe": 4,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "row_signature": "traffic_hourly|locked_test|0|4|euler|uniform|traffic_hourly_otflow_temporal_extrapolation_4k_seed0",
                "row_status": "complete",
            }
            runner._append_row_record(recorder, row)
            recorder["fh"].write(json.dumps(dict(row, score_main=2.0), sort_keys=True) + "\n")
            recorder["fh"].close()

            status = runner.schedule_row_output_status(Path(tmpdir), args)
            self.assertFalse(status["complete"])
            self.assertIn("duplicate rows.jsonl row keys", status["reason"])
            self.assertEqual(status["duplicate_key_count"], 1)
            self.assertEqual(status["duplicate_extra_count"], 1)

    def test_schedule_row_output_status_rejects_stale_combined_summary_with_missing_rows(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "traffic_hourly",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest),
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0",
                    "--checkpoint_steps",
                    "4000",
                    "--split_phase",
                    "locked_test",
                    "--write_context_rows",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            recorder["fh"].close()
            (Path(tmpdir) / "combined_summary.json").write_text(json.dumps({"main_table_summary": {"row_count": 1}}), encoding="utf-8")
            status = runner.schedule_row_output_status(Path(tmpdir), args)
        self.assertFalse(status["complete"])
        self.assertIn("missing complete rows", status["reason"])

    def test_protocol_hash_tracks_data_path_identity(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            data_a = Path(tmpdir) / "cryptos_a.npz"
            data_b = Path(tmpdir) / "cryptos_b.npz"
            data_a.write_bytes(b"a")
            data_b.write_bytes(b"bb")
            args_a = runner.build_argparser().parse_args(
                [
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "cryptos",
                    "--backbone_manifest",
                    str(manifest),
                    "--cryptos_path",
                    str(data_a),
                ]
            )
            args_b = runner.build_argparser().parse_args(
                [
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "cryptos",
                    "--backbone_manifest",
                    str(manifest),
                    "--cryptos_path",
                    str(data_b),
                ]
            )
            self.assertNotEqual(runner._protocol_config_fingerprint(args_a), runner._protocol_config_fingerprint(args_b))

    def test_protocol_hash_tracks_context_reward_protocol(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        args = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "cryptos",
                "--backbone_manifest",
                str(manifest),
                "--write_context_rows",
            ]
        )
        payload = runner._context_reward_protocol_payload(args)
        self.assertEqual(
            payload["conditional_generation_profiles"]["cryptos"]["target_utility_keys"],
            ["u_temporal_uw1_uniform", "u_temporal_cw1_uniform", "u_temporal_tstr_f1_uniform"],
        )
        self.assertFalse(payload["conditional_diagnostic_metrics_are_teacher_targets"])

        with mock.patch.object(runner, "_context_reward_protocol_payload", return_value={"version": "changed"}):
            changed_hash = runner._protocol_config_fingerprint(args)
        self.assertNotEqual(runner._protocol_config_fingerprint(args), changed_hash)

    def test_locked_test_preview_default_only_affects_active_preview_protocols(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"

        def parse_args(split_phase: str, extra_args: list[str] | None = None):
            cli = [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "lobster_synthetic",
                "--backbone_manifest",
                str(manifest),
                "--split_phase",
                str(split_phase),
            ]
            cli.extend(extra_args or [])
            return runner.build_argparser().parse_args(cli)

        train_args = parse_args("train_tuning")
        validation_args = parse_args("validation_tuning")
        locked_args = parse_args("locked_test")
        locked_preview_args = parse_args("locked_test", ["--locked_test_preview"])
        locked_explicit_args = parse_args(
            "locked_test",
            ["--locked_test_preview", "--locked_test_preview_contexts", "19"],
        )
        with mock.patch.object(runner, "LOCKED_TEST_PREVIEW_CONTEXTS", 2048):
            train_hash_2048 = runner._protocol_config_fingerprint(train_args)
            validation_hash_2048 = runner._protocol_config_fingerprint(validation_args)
            locked_hash_2048 = runner._protocol_config_fingerprint(locked_args)
            locked_preview_hash_2048 = runner._protocol_config_fingerprint(locked_preview_args)
            locked_explicit_hash_2048 = runner._protocol_config_fingerprint(locked_explicit_args)
        with mock.patch.object(runner, "LOCKED_TEST_PREVIEW_CONTEXTS", 512):
            train_hash_512 = runner._protocol_config_fingerprint(train_args)
            validation_hash_512 = runner._protocol_config_fingerprint(validation_args)
            locked_hash_512 = runner._protocol_config_fingerprint(locked_args)
            locked_preview_hash_512 = runner._protocol_config_fingerprint(locked_preview_args)
            locked_explicit_hash_512 = runner._protocol_config_fingerprint(locked_explicit_args)

        self.assertEqual(train_hash_2048, train_hash_512)
        self.assertEqual(validation_hash_2048, validation_hash_512)
        self.assertEqual(locked_hash_2048, locked_hash_512)
        self.assertEqual(locked_explicit_hash_2048, locked_explicit_hash_512)
        self.assertNotEqual(locked_preview_hash_2048, locked_preview_hash_512)
        self.assertEqual(
            runner._locked_test_selection_settings(locked_args, "locked_test"),
            {"mode": "full", "context_limit": None, "context_limit_scope": "none"},
        )
        self.assertEqual(
            runner._locked_test_selection_settings(locked_preview_args, "locked_test"),
            {"mode": "preview", "context_limit": 512, "context_limit_scope": "per_seed"},
        )
        self.assertEqual(
            runner._locked_test_selection_settings(locked_explicit_args, "locked_test"),
            {"mode": "preview", "context_limit": 19, "context_limit_scope": "per_seed"},
        )

    def test_locked_test_preview_context_count_requires_preview_flag(self) -> None:
        args = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "lobster_synthetic",
                "--split_phase",
                "locked_test",
                "--locked_test_preview_contexts",
                "19",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --locked_test_preview"):
            runner._prep_summary(args)

    def test_locked_test_preview_rejects_non_locked_split(self) -> None:
        args = runner.build_argparser().parse_args(
            ["--split_phase", "validation_tuning", "--locked_test_preview"]
        )
        with self.assertRaisesRegex(ValueError, "only valid with --split_phase locked_test"):
            runner._prep_summary(args)

    def test_locked_test_preview_requires_positive_context_count(self) -> None:
        for value in ("0", "-1"):
            with self.subTest(value=value):
                args = runner.build_argparser().parse_args(
                    [
                        "--split_phase",
                        "locked_test",
                        "--locked_test_preview",
                        "--locked_test_preview_contexts",
                        value,
                    ]
                )
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    runner._prep_summary(args)

    def test_locked_test_selector_applies_preview_limit_per_seed_across_groups(self) -> None:
        groups = [
            {
                "candidate_indices": list(range(group_idx * 10, group_idx * 10 + 5)),
                "uncapped_candidate_examples": 5,
                "selection_record": {"seed": seed, "selection_seed": 100 + group_idx},
            }
            for group_idx, seed in enumerate((0, 0, 1, 1))
        ]
        selected, records, summary = runner._select_locked_test_context_groups(
            groups,
            context_limit=6,
            seed=17,
            salt="preview-test",
        )
        repeated_selected, repeated_records, repeated_summary = runner._select_locked_test_context_groups(
            groups,
            context_limit=6,
            seed=17,
            salt="preview-test",
        )

        self.assertEqual([values.tolist() for values in selected], [values.tolist() for values in repeated_selected])
        self.assertEqual(records, repeated_records)
        self.assertEqual(summary, repeated_summary)
        selected_by_seed = {
            seed: sum(int(record["selected_examples"]) for record in records if int(record["seed"]) == seed)
            for seed in (0, 1)
        }
        self.assertEqual(selected_by_seed, {0: 6, 1: 6})
        self.assertTrue(all(record["locked_test_mode"] == "preview" for record in records))
        self.assertTrue(all(record["locked_test_context_limit"] == 6 for record in records))
        self.assertEqual(summary["selected_examples"], 12)
        self.assertTrue(summary["selection_was_capped"])

        full_selected, full_records, full_summary = runner._select_locked_test_context_groups(
            groups,
            context_limit=None,
            seed=17,
            salt="full-test",
        )
        self.assertEqual(sum(len(values) for values in full_selected), 20)
        self.assertTrue(all(record["locked_test_mode"] == "full" for record in full_records))
        self.assertEqual(full_summary["selected_examples"], 20)
        self.assertFalse(full_summary["selection_was_capped"])

    def test_preflight_resolves_relative_shared_backbone_root_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            root = Path(tmpdir)
            rel_root = root.relative_to(PROJECT_ROOT).as_posix()
            ckpt_path = root / "temporal_extrapolation" / "traffic_hourly" / "model.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_bytes(b"checkpoint")
            args = runner.build_argparser().parse_args(
                [
                    "--forecast_datasets",
                    "traffic_hourly",
                    "--conditional_generation_datasets",
                    "",
                    "--shared_backbone_root",
                    rel_root,
                    "--backbone_manifest",
                    "",
                    "--checkpoint_steps",
                    "20000",
                    "--allow_execute",
                ]
            )

            runner.validate_execution_preflight(args)

    def test_preflight_rejects_stale_ready_manifest_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "backbone_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "path_base": "manifest_parent",
                        "artifacts": [
                            {
                                "backbone_name": "otflow",
                                "benchmark_family": "temporal_extrapolation",
                                "dataset_key": "traffic_hourly",
                                "train_steps": 20000,
                                "train_budget_label": "20k",
                                "checkpoint_id": "traffic_hourly_otflow_forecast_20k_seed0",
                                "checkpoint_path": "outputs/missing_preflight_checkpoint/model.pt",
                                "summary_path": "",
                                "status": "ready",
                                "seed": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = runner.build_argparser().parse_args(
                [
                    "--forecast_datasets",
                    "traffic_hourly",
                    "--conditional_generation_datasets",
                    "",
                    "--backbone_manifest",
                    str(manifest_path),
                    "--checkpoint_steps",
                    "20000",
                    "--allow_execute",
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "checkpoint files are missing"):
                runner.validate_execution_preflight(args)

    def test_protocol_hash_tracks_selected_seeds(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        args_a = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "",
                "--backbone_manifest",
                str(manifest),
                "--seeds",
                "0",
            ]
        )
        args_b = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "",
                "--backbone_manifest",
                str(manifest),
                "--seeds",
                "1",
            ]
        )
        self.assertNotEqual(runner._protocol_config_fingerprint(args_a), runner._protocol_config_fingerprint(args_b))


    def test_protocol_hash_tracks_train_tuning_sampling_mode(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        args_legacy = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "",
                "--backbone_manifest",
                str(manifest),
                "--split_phase",
                "train_tuning",
                "--train_tuning_sampling_mode",
                "train_window_fraction",
            ]
        )
        args_valnorm = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "",
                "--backbone_manifest",
                str(manifest),
                "--split_phase",
                "train_tuning",
                "--train_tuning_sampling_mode",
                "validation_normalized",
            ]
        )
        self.assertNotEqual(runner._protocol_config_fingerprint(args_legacy), runner._protocol_config_fingerprint(args_valnorm))

    def test_runner_default_split_phase_is_locked_test(self) -> None:
        args = runner.build_argparser().parse_args([
            "--forecast_datasets",
            "",
            "--conditional_generation_datasets",
            "",
        ])
        self.assertEqual(args.split_phase, "locked_test")
        self.assertFalse(args.locked_test_preview)
        self.assertIsNone(args.locked_test_preview_contexts)
        self.assertFalse(hasattr(args, "eval_windows_test"))

    def test_protocol_hash_tracks_selected_split_phase(self) -> None:
        manifest = PROJECT_ROOT / "backbone_manifest.json"
        args_locked = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "",
                "--backbone_manifest",
                str(manifest),
                "--split_phase",
                "locked_test",
            ]
        )
        args_val = runner.build_argparser().parse_args(
            [
                "--forecast_datasets",
                "",
                "--conditional_generation_datasets",
                "",
                "--backbone_manifest",
                str(manifest),
                "--split_phase",
                "validation_tuning",
            ]
        )
        self.assertNotEqual(runner._protocol_config_fingerprint(args_locked), runner._protocol_config_fingerprint(args_val))


    def test_train_tuning_forecast_only_skips_empty_conditional_generation_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "traffic_hourly",
                    "--conditional_generation_datasets",
                    "",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0",
                    "--split_phase",
                    "train_tuning",
                    "--allow_execute",
                ]
            )
            with mock.patch.object(runner, "validate_execution_preflight"), mock.patch.object(runner, "_run_forecast_phase", return_value=[]), mock.patch.object(
                runner,
                "_run_conditional_generation_phase",
                side_effect=AssertionError("conditional generation should not run for an empty dataset list"),
            ) as conditional_phase:
                payload = runner.run_diffusion_flow_time_reparameterization(args)

        conditional_phase.assert_not_called()
        self.assertEqual(payload["prep"]["split_phase"], "train_tuning")

    def test_train_tuning_allows_non_empty_conditional_generation_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "cryptos",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0",
                    "--split_phase",
                    "train_tuning",
                    "--allow_execute",
                ]
            )
            with mock.patch.object(runner, "validate_execution_preflight"), mock.patch.object(runner, "_run_forecast_phase", return_value=[]), mock.patch.object(
                runner,
                "_run_conditional_generation_phase",
                return_value=[{"benchmark_family": runner.CONDITIONAL_GENERATION_FAMILY, "split_phase": "train_tuning"}],
            ) as conditional_phase:
                payload = runner.run_diffusion_flow_time_reparameterization(args)

        conditional_phase.assert_called_once()
        self.assertEqual(payload["prep"]["split_phase"], "train_tuning")

    def test_conditional_generation_accepts_summary_schedule_args_without_forecast_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schedules": [
                            {
                                "scheduler_key": "gipo",
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "lobster_synthetic",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--schedule_summary_json",
                    str(summary_path),
                    "--summary_scheduler_names",
                    "gipo",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0",
                    "--split_phase",
                    "train_tuning",
                    "--allow_execute",
                ]
            )
            with mock.patch.object(runner, "validate_execution_preflight"), mock.patch.object(runner, "_run_forecast_phase", return_value=[]) as forecast_phase, mock.patch.object(
                runner,
                "_run_conditional_generation_phase",
                return_value=[{"benchmark_family": runner.CONDITIONAL_GENERATION_FAMILY, "split_phase": "train_tuning"}],
            ) as conditional_phase:
                payload = runner.run_diffusion_flow_time_reparameterization(args)

        forecast_phase.assert_not_called()
        conditional_phase.assert_called_once()
        cases = conditional_phase.call_args.kwargs["scheduler_cases_by_dataset"]["lobster_synthetic"]
        self.assertEqual([case["scheduler_key"] for case in cases], ["uniform", "gipo"])
        self.assertEqual(payload["schedule_selection_summary"]["summary_scheduler_keys"], ["gipo"])

    def test_forecast_phase_uses_requested_split_dataset(self) -> None:
        class FakeDataset:
            def __init__(self, name: str) -> None:
                self.name = name

            def __len__(self) -> int:
                return 3

        fake_checkpoint = {
            "model": object(),
            "cfg": object(),
            "splits": {"val": FakeDataset("val"), "test": FakeDataset("test")},
            "checkpoint_path": PROJECT_ROOT / "outputs" / "fake_model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }

        def run_for_phase(split_phase: str) -> str:
            seen = []
            with tempfile.TemporaryDirectory() as tmpdir:
                args = runner.build_argparser().parse_args(
                    [
                        "--out_root",
                        tmpdir,
                        "--forecast_datasets",
                        "traffic_hourly",
                        "--conditional_generation_datasets",
                        "",
                        "--baseline_scheduler_names",
                        "uniform",
                        "--target_nfe_values",
                        "4",
                        "--solver_names",
                        "euler",
                        "--seeds",
                        "0",
                        "--split_phase",
                        split_phase,
                        "--backbone_manifest",
                        str(PROJECT_ROOT / "backbone_manifest.json"),
                        "--checkpoint_steps",
                        "20000",
                    ]
                )
                recorder = runner._init_row_recorder(Path(tmpdir), args)
                try:
                    def fake_eval(model, ds, cfg, **kwargs):
                        seen.append(ds.name)
                        return {
                            "forecast_crps": 1.0,
                            "mse": 1.0,
                            "forecast_mase": 1.0,
                            "latency_ms_per_sample": 0.0,
                            "num_eval_samples": 1,
                            "eval_examples": 3,
                            "eval_horizon": 168,
                            "evaluation_protocol_hash": "protocol",
                            "chosen_examples_hash": "examples",
                            "realized_nfe": 4,
                        }

                    with mock.patch.object(runner, "load_forecast_checkpoint_splits", return_value=fake_checkpoint), mock.patch.object(runner, "evaluate_forecast_schedule", side_effect=fake_eval):
                        runner._run_forecast_phase(
                            args,
                            row_recorder=recorder,
                            split_phase=split_phase,
                            seeds=[0],
                            scheduler_cases_by_dataset={"traffic_hourly": [{"scheduler_key": "uniform"}]},
                        )
                finally:
                    recorder["fh"].close()
            return seen[0]

        self.assertEqual(run_for_phase("validation_tuning"), "val")
        self.assertEqual(run_for_phase("locked_test"), "test")

    def test_forecast_phase_passes_logical_seed_separately_from_sampling_seed(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 10

        fake_checkpoint = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
            "checkpoint_path": PROJECT_ROOT / "model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "traffic_hourly",
                    "--conditional_generation_datasets",
                    "",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4,8",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "5",
                    "--split_phase",
                    "validation_tuning",
                    "--eval_windows_val",
                    "1",
                    "--backbone_manifest",
                    str(PROJECT_ROOT / "backbone_manifest.json"),
                    "--checkpoint_steps",
                    "20000",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            calls = []

            def fake_eval(model, ds, cfg, **kwargs):
                del model, ds, cfg
                calls.append(dict(kwargs))
                return {
                    "forecast_crps": 1.0,
                    "mse": 1.0,
                    "forecast_mase": 1.0,
                    "latency_ms_per_sample": 0.0,
                    "num_eval_samples": 1,
                    "eval_examples": 1,
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": int(kwargs["target_nfe"]),
                }

            try:
                with mock.patch.object(runner, "load_forecast_checkpoint_splits", return_value=fake_checkpoint), mock.patch.object(runner, "evaluate_forecast_schedule", side_effect=fake_eval):
                    runner._run_forecast_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="validation_tuning",
                        seeds=[5],
                        scheduler_cases_by_dataset={"traffic_hourly": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()

        self.assertEqual(
            [(call["target_nfe"], call["seed"], call["logical_seed"]) for call in calls],
            [(4, 5, 5), (8, 1005, 5)],
        )

    def test_forecast_phase_locked_test_defaults_to_full_split_not_context_cap(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 1000

        fake_checkpoint = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
            "checkpoint_path": PROJECT_ROOT / "model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "traffic_hourly",
                    "--conditional_generation_datasets",
                    "",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0",
                    "--split_phase",
                    "locked_test",
                    "--context_sample_count",
                    "13",
                    "--backbone_manifest",
                    str(PROJECT_ROOT / "backbone_manifest.json"),
                    "--checkpoint_steps",
                    "20000",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            captured_lengths = []

            def fake_eval(model, ds, cfg, **kwargs):
                del model, ds, cfg
                captured_lengths.append(len(kwargs["example_indices"]))
                return {
                    "forecast_crps": 1.0,
                    "mse": 1.0,
                    "forecast_mase": 1.0,
                    "latency_ms_per_sample": 0.0,
                    "num_eval_samples": 1,
                    "eval_examples": len(kwargs["example_indices"]),
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": 4,
                }

            try:
                with mock.patch.object(runner, "load_forecast_checkpoint_splits", return_value=fake_checkpoint), mock.patch.object(runner, "evaluate_forecast_schedule", side_effect=fake_eval):
                    rows = runner._run_forecast_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="locked_test",
                        seeds=[0],
                        scheduler_cases_by_dataset={"traffic_hourly": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()

        self.assertEqual(captured_lengths, [1000])
        self.assertEqual(rows[0]["selected_examples"], 1000)
        self.assertEqual(rows[0]["selected_examples_cap"], 1000)
        self.assertEqual(rows[0]["selected_examples_cap_source"], "locked_test_full")

    def test_forecast_phase_context_cap_is_global_across_seeds(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 1000

        fake_checkpoint = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
            "checkpoint_path": PROJECT_ROOT / "model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "traffic_hourly",
                    "--conditional_generation_datasets",
                    "",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0,1",
                    "--split_phase",
                    "validation_tuning",
                    "--eval_windows_val",
                    "5",
                    "--backbone_manifest",
                    str(PROJECT_ROOT / "backbone_manifest.json"),
                    "--checkpoint_steps",
                    "20000",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            captured_lengths = []

            def fake_eval(model, ds, cfg, **kwargs):
                del model, ds, cfg
                captured_lengths.append(len(kwargs["example_indices"]))
                return {
                    "forecast_crps": 1.0,
                    "mse": 1.0,
                    "forecast_mase": 1.0,
                    "latency_ms_per_sample": 0.0,
                    "num_eval_samples": 1,
                    "eval_examples": len(kwargs["example_indices"]),
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": 4,
                }

            try:
                with mock.patch.object(runner, "load_forecast_checkpoint_splits", return_value=fake_checkpoint), mock.patch.object(runner, "evaluate_forecast_schedule", side_effect=fake_eval):
                    rows = runner._run_forecast_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="validation_tuning",
                        seeds=[0, 1],
                        scheduler_cases_by_dataset={"traffic_hourly": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()
            with (Path(tmpdir) / "rows.csv").open(newline="", encoding="utf-8") as fh:
                csv_rows = list(csv.DictReader(fh))

        self.assertEqual(sum(captured_lengths), 5)
        self.assertEqual(sum(int(row["selected_examples"]) for row in rows), 5)
        self.assertTrue(all(int(row["global_selected_examples"]) == 5 for row in rows))
        self.assertTrue(all(int(row["selected_examples_cap"]) == 5 for row in rows))
        self.assertTrue(all(row["selected_examples_cap_source"] == "eval_windows_val" for row in rows))
        self.assertTrue(all(int(row["global_selected_examples"]) == 5 for row in csv_rows))

    def test_forecast_phase_train_tuning_defaults_to_context_cap(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 1000

        fake_checkpoint = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
            "checkpoint_path": PROJECT_ROOT / "model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "traffic_hourly",
                    "--conditional_generation_datasets",
                    "",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0",
                    "--split_phase",
                    "train_tuning",
                    "--eval_train_fraction",
                    "1.0",
                    "--context_sample_count",
                    "17",
                    "--backbone_manifest",
                    str(PROJECT_ROOT / "backbone_manifest.json"),
                    "--checkpoint_steps",
                    "20000",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            captured_lengths = []

            def fake_eval(model, ds, cfg, **kwargs):
                del model, ds, cfg
                captured_lengths.append(len(kwargs["example_indices"]))
                return {
                    "forecast_crps": 1.0,
                    "mse": 1.0,
                    "forecast_mase": 1.0,
                    "latency_ms_per_sample": 0.0,
                    "num_eval_samples": 1,
                    "eval_examples": len(kwargs["example_indices"]),
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": 4,
                }

            try:
                with mock.patch.object(runner, "load_forecast_checkpoint_splits", return_value=fake_checkpoint), mock.patch.object(runner, "evaluate_forecast_schedule", side_effect=fake_eval):
                    rows = runner._run_forecast_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="train_tuning",
                        seeds=[0],
                        scheduler_cases_by_dataset={"traffic_hourly": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()

        self.assertEqual(captured_lengths, [17])
        self.assertEqual(rows[0]["selected_examples"], 17)
        self.assertEqual(rows[0]["train_tuning_target_examples"], 1000)
        self.assertEqual(rows[0]["train_tuning_uncapped_candidate_examples"], 1000)
        self.assertEqual(rows[0]["selected_examples_cap_source"], "context_sample_count")

    def test_forecast_train_tuning_uses_twenty_percent_then_global_cap(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 5000

        fake_checkpoint = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
            "checkpoint_path": PROJECT_ROOT / "model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "traffic_hourly",
                    "--conditional_generation_datasets",
                    "",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0,1",
                    "--split_phase",
                    "train_tuning",
                    "--eval_train_fraction",
                    "0.2",
                    "--context_sample_count",
                    "256",
                    "--backbone_manifest",
                    str(PROJECT_ROOT / "backbone_manifest.json"),
                    "--checkpoint_steps",
                    "20000",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            captured_lengths = []

            def fake_eval(model, ds, cfg, **kwargs):
                del model, ds, cfg
                captured_lengths.append(len(kwargs["example_indices"]))
                return {
                    "forecast_crps": 1.0,
                    "mse": 1.0,
                    "forecast_mase": 1.0,
                    "latency_ms_per_sample": 0.0,
                    "num_eval_samples": 1,
                    "eval_examples": len(kwargs["example_indices"]),
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": 4,
                }

            try:
                with mock.patch.object(runner, "load_forecast_checkpoint_splits", return_value=fake_checkpoint), mock.patch.object(runner, "evaluate_forecast_schedule", side_effect=fake_eval):
                    rows = runner._run_forecast_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="train_tuning",
                        seeds=[0, 1],
                        scheduler_cases_by_dataset={"traffic_hourly": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()

        self.assertEqual(sum(captured_lengths), 256)
        self.assertTrue(all(length >= 1 for length in captured_lengths))
        self.assertEqual(sum(int(row["selected_examples"]) for row in rows), 256)
        self.assertTrue(all(int(row["global_selected_examples"]) == 256 for row in rows))
        self.assertTrue(all(int(row["global_uncapped_candidate_examples"]) == 2000 for row in rows))
        self.assertTrue(all(int(row["train_tuning_target_examples"]) == 1000 for row in rows))
        self.assertTrue(all(int(row["train_tuning_uncapped_candidate_examples"]) == 1000 for row in rows))
        self.assertTrue(all(row["selected_examples_cap_source"] == "context_sample_count" for row in rows))
        self.assertTrue(all(row["train_tuning_sampler"] == "temporal_stratified_hash" for row in rows))

    def test_conditional_train_tuning_uses_twenty_percent_then_global_cap(self) -> None:
        class FakeDataset:
            start_indices = list(range(5000))

            def __len__(self) -> int:
                return len(self.start_indices)

        fake_checkpoint = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
            "checkpoint_path": PROJECT_ROOT / "model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "cryptos",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0,1",
                    "--split_phase",
                    "train_tuning",
                    "--eval_train_fraction",
                    "0.2",
                    "--context_sample_count",
                    "256",
                    "--backbone_manifest",
                    str(PROJECT_ROOT / "backbone_manifest.json"),
                    "--checkpoint_steps",
                    "20000",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            captured_windows = []

            def fake_run_fixed_schedule_variant(**kwargs):
                captured_windows.append(len(kwargs["chosen_t0s"]))
                return {
                    "score_main": 1.0,
                    "temporal_uw1": 0.1,
                    "temporal_cw1": 0.2,
                    "temporal_tstr_f1": None,
                    "temporal_tstr_f1_applicable": False,
                    "evaluation_protocol": {"chosen_t0s_hash": "hash"},
                }

            try:
                with mock.patch.object(
                    runner,
                    "load_conditional_generation_checkpoint_splits",
                    return_value=fake_checkpoint,
                ), mock.patch.object(
                    runner,
                    "_choose_valid_windows",
                    side_effect=AssertionError("train_tuning should use stratified train starts directly"),
                ), mock.patch.object(
                    runner,
                    "run_fixed_schedule_variant",
                    side_effect=fake_run_fixed_schedule_variant,
                ):
                    rows = runner._run_conditional_generation_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="train_tuning",
                        seeds=[0, 1],
                        scheduler_cases_by_dataset={"cryptos": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()

        self.assertEqual(sum(captured_windows), 256)
        self.assertTrue(all(length >= 1 for length in captured_windows))
        self.assertEqual(sum(int(row["selected_examples"]) for row in rows), 256)
        self.assertTrue(all(int(row["global_selected_examples"]) == 256 for row in rows))
        self.assertTrue(all(int(row["train_tuning_target_examples"]) == 1000 for row in rows))
        self.assertTrue(all(int(row["train_tuning_uncapped_candidate_examples"]) == 1000 for row in rows))
        self.assertTrue(all(row["selected_examples_cap_source"] == "context_sample_count" for row in rows))
        self.assertTrue(all(row["train_tuning_sampler"] == "temporal_stratified_hash" for row in rows))

    def test_conditional_generation_locked_test_defaults_to_every_eligible_window_per_seed(self) -> None:
        class FakeDataset:
            start_indices = list(range(5000))

            def __len__(self) -> int:
                return len(self.start_indices)

        fake_checkpoint = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
            "checkpoint_path": PROJECT_ROOT / "model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }

        def run_once(dataset: str, seeds: tuple[int, ...] = (0, 1)):
            with tempfile.TemporaryDirectory() as tmpdir:
                args = runner.build_argparser().parse_args(
                    [
                        "--out_root",
                        tmpdir,
                        "--forecast_datasets",
                        "",
                        "--conditional_generation_datasets",
                        str(dataset),
                        "--baseline_scheduler_names",
                        "uniform",
                        "--target_nfe_values",
                        "4",
                        "--solver_names",
                        "euler",
                        "--seeds",
                        ",".join(str(seed) for seed in seeds),
                        "--split_phase",
                        "locked_test",
                        "--context_sample_count",
                        "3",
                        "--backbone_manifest",
                        str(PROJECT_ROOT / "backbone_manifest.json"),
                        "--checkpoint_steps",
                        "20000",
                    ]
                )
                recorder = runner._init_row_recorder(Path(tmpdir), args)
                captured_by_seed = {}
                choose_calls = []

                def fake_choose_valid_windows(ds, *, horizon: int, n_windows: int, seed: int):
                    del ds, horizon
                    choose_calls.append((int(n_windows), int(seed)))
                    start = int(seed) * 10
                    return list(range(start, start + int(n_windows)))

                def fake_run_fixed_schedule_variant(**kwargs):
                    captured_by_seed[int(kwargs["metrics_seed"])] = [int(t0) for t0 in kwargs["chosen_t0s"]]
                    return {
                        "score_main": 1.0,
                        "temporal_uw1": 0.1,
                        "temporal_cw1": 0.2,
                        "temporal_tstr_f1": None,
                        "temporal_tstr_f1_applicable": False,
                        "evaluation_protocol": {"chosen_t0s_hash": "hash"},
                    }

                try:
                    with mock.patch.object(
                        runner,
                        "load_conditional_generation_checkpoint_splits",
                        return_value=fake_checkpoint,
                    ), mock.patch.object(
                        runner,
                        "resolved_validation_windows",
                        side_effect=AssertionError("locked-test full mode should enumerate every eligible window"),
                    ), mock.patch.object(
                        runner,
                        "_choose_valid_windows",
                        side_effect=fake_choose_valid_windows,
                    ), mock.patch.object(
                        runner,
                        "run_fixed_schedule_variant",
                        side_effect=fake_run_fixed_schedule_variant,
                    ):
                        rows = runner._run_conditional_generation_phase(
                            args,
                            row_recorder=recorder,
                            split_phase="locked_test",
                            seeds=list(seeds),
                            scheduler_cases_by_dataset={str(dataset): [{"scheduler_key": "uniform"}]},
                        )
                finally:
                    recorder["fh"].close()
            return rows, captured_by_seed, choose_calls

        for dataset in ("cryptos", "lobster_synthetic", "long_term_st"):
            rows, captured_by_seed, choose_calls = run_once(dataset)
            _, repeated_captured_by_seed, _ = run_once(dataset)
            _, reversed_captured_by_seed, _ = run_once(dataset, seeds=(1, 0))

            self.assertEqual(captured_by_seed, repeated_captured_by_seed)
            self.assertEqual(captured_by_seed, reversed_captured_by_seed)
            self.assertEqual({seed: len(windows) for seed, windows in captured_by_seed.items()}, {0: 5000, 1: 5000})
            self.assertEqual(choose_calls, [])
            rows_by_seed = {int(row["seed"]): row for row in rows}
            self.assertEqual(set(rows_by_seed), {0, 1})
            self.assertEqual(sum(int(row["selected_examples"]) for row in rows), 10000)
            for row in rows_by_seed.values():
                self.assertEqual(row["eval_windows"], 5000)
                self.assertEqual(row["context_sample_count"], 3)
                self.assertEqual(row["selected_examples"], 5000)
                self.assertEqual(row["selected_examples_cap"], 5000)
                self.assertEqual(row["selected_examples_cap_source"], "locked_test_full")
                self.assertEqual(row["selected_examples_cap_scope"], "none")
                self.assertEqual(row["locked_test_mode"], "full")
                self.assertIsNone(row["locked_test_context_limit"])
                self.assertEqual(row["locked_test_context_limit_scope"], "none")
                self.assertNotIn("locked_test_eval_windows_per_seed", row)
                self.assertEqual(row["uncapped_candidate_examples"], 5000)
                self.assertEqual(row["global_selected_examples"], 10000)
                self.assertEqual(row["global_uncapped_candidate_examples"], 10000)
                self.assertFalse(row["selection_was_capped"])
                self.assertFalse(row["global_selection_was_capped"])

    def test_conditional_generation_locked_test_preview_override_is_per_seed(self) -> None:
        class FakeDataset:
            start_indices = list(range(1000))

            def __len__(self) -> int:
                return len(self.start_indices)

        fake_checkpoint = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
            "checkpoint_path": PROJECT_ROOT / "model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "cryptos",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0,1",
                    "--split_phase",
                    "locked_test",
                    "--locked_test_preview",
                    "--locked_test_preview_contexts",
                    "5",
                    "--backbone_manifest",
                    str(PROJECT_ROOT / "backbone_manifest.json"),
                    "--checkpoint_steps",
                    "20000",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            captured_by_seed = {}

            def fake_run_fixed_schedule_variant(**kwargs):
                captured_by_seed[int(kwargs["metrics_seed"])] = [int(t0) for t0 in kwargs["chosen_t0s"]]
                return {
                    "score_main": 1.0,
                    "temporal_uw1": 0.1,
                    "temporal_cw1": 0.2,
                    "temporal_tstr_f1": None,
                    "temporal_tstr_f1_applicable": False,
                    "evaluation_protocol": {"chosen_t0s_hash": "hash"},
                }

            try:
                with mock.patch.object(
                    runner,
                    "load_conditional_generation_checkpoint_splits",
                    return_value=fake_checkpoint,
                ), mock.patch.object(
                    runner,
                    "_choose_valid_windows",
                    side_effect=AssertionError("locked-test cap should sample from the full valid candidate set"),
                ), mock.patch.object(
                    runner,
                    "run_fixed_schedule_variant",
                    side_effect=fake_run_fixed_schedule_variant,
                ):
                    rows = runner._run_conditional_generation_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="locked_test",
                        seeds=[0, 1],
                        scheduler_cases_by_dataset={"cryptos": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()

        self.assertEqual({seed: len(windows) for seed, windows in captured_by_seed.items()}, {0: 5, 1: 5})
        rows_by_seed = {int(row["seed"]): row for row in rows}
        self.assertEqual(set(rows_by_seed), {0, 1})
        self.assertEqual(sum(int(row["selected_examples"]) for row in rows), 10)
        for row in rows_by_seed.values():
            self.assertEqual(row["eval_windows"], 5)
            self.assertEqual(row["selected_examples"], 5)
            self.assertEqual(row["selected_examples_cap"], 5)
            self.assertEqual(row["selected_examples_cap_source"], "locked_test_preview_contexts")
            self.assertEqual(row["selected_examples_cap_scope"], "per_seed")
            self.assertEqual(row["locked_test_mode"], "preview")
            self.assertEqual(row["locked_test_context_limit"], 5)
            self.assertEqual(row["locked_test_context_limit_scope"], "per_seed")
            self.assertNotIn("locked_test_eval_windows_per_seed", row)
            self.assertEqual(row["uncapped_candidate_examples"], 1000)
            self.assertEqual(row["global_selected_examples"], 10)
            self.assertEqual(row["global_uncapped_candidate_examples"], 2000)
            self.assertTrue(row["selection_was_capped"])
            self.assertTrue(row["global_selection_was_capped"])

    def test_conditional_generation_resume_skips_existing_context_embeddings(self) -> None:
        class FakeDataset:
            start_indices = [0, 1]

            def __len__(self) -> int:
                return len(self.start_indices)

            def __getitem__(self, idx):
                t0 = int(self.start_indices[int(idx)])
                return torch.full((2, 3), float(t0)), torch.zeros(3), {"target_t": t0}

        class FakeCache:
            def __init__(self, ctx_summary: torch.Tensor) -> None:
                self.ctx_summary = ctx_summary

        class FakeBackbone(torch.nn.Module):
            def precompute(self, hist_batch: torch.Tensor) -> FakeCache:
                first_value = hist_batch[:, 0, 0].float()
                return FakeCache(torch.stack([first_value, first_value + 1.0], dim=1))

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = FakeBackbone()

        fake_checkpoint = {
            "model": FakeModel(),
            "cfg": SimpleNamespace(history_len=2, batch_size=2),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
            "checkpoint_path": PROJECT_ROOT / "model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }

        def fake_run_fixed_schedule_variant(**kwargs):
            chosen_t0s = [int(x) for x in kwargs["chosen_t0s"].tolist()]
            return {
                "score_main": 1.0,
                "temporal_uw1": 0.1,
                "temporal_cw1": 0.2,
                "temporal_tstr_f1": None,
                "temporal_tstr_f1_applicable": False,
                "evaluation_protocol": {"chosen_t0s_hash": ",".join(str(x) for x in chosen_t0s)},
                "per_window_metric_rows": [{"target_t": t0, "score_main": 1.0} for t0 in chosen_t0s],
            }

        def fake_conditional_context_records(**kwargs):
            parent = str(kwargs["parent_row_signature"])
            protocol_hash = str(kwargs["protocol_hash"])
            checkpoint_id = str(kwargs["checkpoint"]["checkpoint_id"])
            return [
                {
                    "benchmark_family": str(kwargs["benchmark_family"]),
                    "experiment_layout": runner.EXPERIMENT_LAYOUT_ID,
                    "protocol_hash": protocol_hash,
                    "parent_row_signature": parent,
                    "row_signature": f"{parent}:ctx:{int(t0)}",
                    "scenario_family": str(kwargs["benchmark_family"]),
                    "method_key": str(kwargs["details"].get("method_key") or runner.METHOD_KEY),
                    "nfe_role": str(kwargs["nfe_role"]),
                    "checkpoint_step": int(kwargs["checkpoint_step"]),
                    "context_id": f"ctx-{int(t0)}",
                    "context_embedding_id": f"{checkpoint_id}:ctx-{int(t0)}",
                    "context_schema": "conditional_generation_window",
                    "scenario_key": str(kwargs["dataset"]),
                    "split_phase": str(kwargs["split_phase"]),
                    "seed": int(kwargs["seed"]),
                    "solver_key": str(kwargs["solver_key"]),
                    "target_nfe": int(kwargs["target_nfe"]),
                    "scheduler_key": str(kwargs["scheduler_key"]),
                    "schedule_grid_hash": str(kwargs["details"]["schedule_grid_hash"]),
                    "checkpoint_id": checkpoint_id,
                    "target_t": int(t0),
                }
                for t0 in kwargs["chosen_t0s"]
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "cryptos",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0",
                    "--split_phase",
                    "locked_test",
                    "--write_context_rows",
                    "--backbone_manifest",
                    str(PROJECT_ROOT / "backbone_manifest.json"),
                    "--checkpoint_steps",
                    "20000",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            try:
                with mock.patch.object(
                    runner,
                    "load_conditional_generation_checkpoint_splits",
                    return_value=fake_checkpoint,
                ), mock.patch.object(
                    runner,
                    "run_fixed_schedule_variant",
                    side_effect=fake_run_fixed_schedule_variant,
                ), mock.patch.object(
                    runner,
                    "_conditional_context_records",
                    side_effect=fake_conditional_context_records,
                ):
                    first_rows = runner._run_conditional_generation_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="locked_test",
                        seeds=[0],
                        scheduler_cases_by_dataset={"cryptos": [{"scheduler_key": "uniform"}]},
                    )
                self.assertTrue(
                    runner._row_has_complete_context_artifacts(
                        first_rows[0],
                        context_rows_by_signature=recorder["context_rows_by_signature"],
                        context_embeddings=recorder["context_embeddings"],
                    ),
                    msg={
                        "parent": first_rows[0],
                        "context_rows": list(recorder["context_rows_by_signature"].values()),
                        "embedding_ids": sorted(recorder["context_embeddings"]),
                    },
                )
                persisted_context_rows = runner._load_context_rows(
                    recorder["context_csv_path"]
                )
                persisted_embeddings = runner.load_context_embedding_table(
                    recorder["context_embeddings_path"],
                    expected_context_embedding_kind="ctx_summary",
                    require_manifest=True,
                    expected_context_rows=list(persisted_context_rows.values()),
                )
                identity_fields = (
                    "benchmark_family",
                    "experiment_layout",
                    "scenario_key",
                    "scenario_family",
                    "method_key",
                    "nfe_role",
                    "checkpoint_step",
                    "checkpoint_id",
                    "protocol_hash",
                    "split_phase",
                    "seed",
                    "solver_key",
                    "target_nfe",
                    "scheduler_key",
                    "schedule_grid_hash",
                    "context_embedding_kind",
                )
                identity_mismatches = [
                    (
                        field,
                        str(first_rows[0].get(field, "")),
                        str(context_row.get(field, "")),
                    )
                    for context_row in persisted_context_rows.values()
                    for field in identity_fields
                    if str(first_rows[0].get(field, ""))
                    != str(context_row.get(field, ""))
                ]
                self.assertEqual(identity_mismatches, [])
                self.assertTrue(
                    runner._row_has_complete_context_artifacts(
                        first_rows[0],
                        context_rows_by_signature=persisted_context_rows,
                        context_embeddings=persisted_embeddings,
                    )
                )
            finally:
                recorder["fh"].close()

            resumed = runner._init_row_recorder(Path(tmpdir), args)
            try:
                with mock.patch.object(
                    runner,
                    "load_conditional_generation_checkpoint_splits",
                    return_value=fake_checkpoint,
                ), mock.patch.object(
                    runner,
                    "_extract_conditional_context_embeddings",
                    side_effect=AssertionError("complete resumed context embeddings should be reused"),
                ) as extract_embeddings, mock.patch.object(
                    runner,
                    "run_fixed_schedule_variant",
                    side_effect=AssertionError("complete resumed schedule rows should be reused"),
                ) as run_schedule:
                    resumed_rows = runner._run_conditional_generation_phase(
                        args,
                        row_recorder=resumed,
                        split_phase="locked_test",
                        seeds=[0],
                        scheduler_cases_by_dataset={"cryptos": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                resumed["fh"].close()

        self.assertEqual(len(first_rows), 1)
        self.assertEqual(len(resumed_rows), 1)
        self.assertEqual(resumed_rows[0]["row_signature"], first_rows[0]["row_signature"])
        extract_embeddings.assert_not_called()
        run_schedule.assert_not_called()

    def test_conditional_context_rows_flush_and_extract_embeddings_by_chunk(self) -> None:
        class FakeDataset:
            start_indices = [0, 1, 2, 3, 4]

            def __len__(self) -> int:
                return len(self.start_indices)

            def __getitem__(self, idx):
                t0 = int(self.start_indices[int(idx)])
                return torch.full((2, 3), float(t0)), torch.zeros(3), {"target_t": t0}

        class FakeCache:
            def __init__(self, ctx_summary: torch.Tensor) -> None:
                self.ctx_summary = ctx_summary

        class FakeBackbone(torch.nn.Module):
            def precompute(self, hist_batch: torch.Tensor) -> FakeCache:
                first_value = hist_batch[:, 0, 0].float()
                return FakeCache(torch.stack([first_value, first_value + 1.0], dim=1))

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = FakeBackbone()

        fake_checkpoint = {
            "model": FakeModel(),
            "cfg": SimpleNamespace(history_len=2, batch_size=8),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
            "checkpoint_path": PROJECT_ROOT / "model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }

        def fake_run_fixed_schedule_variant(**kwargs):
            chosen_t0s = [int(x) for x in kwargs["chosen_t0s"].tolist()]
            return {
                "score_main": 1.0,
                "temporal_uw1": 0.1,
                "temporal_cw1": 0.2,
                "temporal_tstr_f1": None,
                "temporal_tstr_f1_applicable": False,
                "evaluation_protocol": {"chosen_t0s_hash": ",".join(str(x) for x in chosen_t0s)},
                "per_window_metric_rows": [{"target_t": t0, "score_main": 1.0} for t0 in chosen_t0s],
            }

        def fake_conditional_context_records(**kwargs):
            parent = str(kwargs["parent_row_signature"])
            protocol_hash = str(kwargs["protocol_hash"])
            checkpoint_id = str(kwargs["checkpoint"]["checkpoint_id"])
            return [
                {
                    "protocol_hash": protocol_hash,
                    "parent_row_signature": parent,
                    "row_signature": f"{parent}:ctx:{int(t0)}",
                    "context_id": f"ctx-{int(t0)}",
                    "context_embedding_id": f"{checkpoint_id}:ctx-{int(t0)}",
                    "context_schema": "conditional_generation_window",
                    "scenario_key": str(kwargs["dataset"]),
                    "split_phase": str(kwargs["split_phase"]),
                    "seed": int(kwargs["seed"]),
                    "solver_key": str(kwargs["solver_key"]),
                    "target_nfe": int(kwargs["target_nfe"]),
                    "scheduler_key": str(kwargs["scheduler_key"]),
                    "checkpoint_id": checkpoint_id,
                    "target_t": int(t0),
                }
                for t0 in kwargs["chosen_t0s"]
            ]

        append_row_counts = []
        append_embedding_counts = []
        extract_t0_batches = []
        real_append_context_records = runner._append_context_records
        real_extract_embeddings = runner._extract_conditional_context_embeddings

        def capture_append_context_records(row_recorder, rows, *, context_embeddings, metadata):
            append_row_counts.append(len(rows))
            append_embedding_counts.append(len(context_embeddings))
            return real_append_context_records(
                row_recorder,
                rows,
                context_embeddings=context_embeddings,
                metadata=metadata,
            )

        def capture_extract_embeddings(**kwargs):
            extract_t0_batches.append([int(t0) for t0 in kwargs["chosen_t0s"]])
            return real_extract_embeddings(**kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "cryptos",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0",
                    "--split_phase",
                    "locked_test",
                    "--write_context_rows",
                    "--backbone_manifest",
                    str(PROJECT_ROOT / "backbone_manifest.json"),
                    "--checkpoint_steps",
                    "20000",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            try:
                with mock.patch.object(
                    runner,
                    "CONTEXT_RECORD_FLUSH_WINDOW_COUNT",
                    2,
                ), mock.patch.object(
                    runner,
                    "load_conditional_generation_checkpoint_splits",
                    return_value=fake_checkpoint,
                ), mock.patch.object(
                    runner,
                    "run_fixed_schedule_variant",
                    side_effect=fake_run_fixed_schedule_variant,
                ), mock.patch.object(
                    runner,
                    "_conditional_context_records",
                    side_effect=fake_conditional_context_records,
                ), mock.patch.object(
                    runner,
                    "_append_context_records",
                    side_effect=capture_append_context_records,
                ), mock.patch.object(
                    runner,
                    "_extract_conditional_context_embeddings",
                    side_effect=capture_extract_embeddings,
                ):
                    rows = runner._run_conditional_generation_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="locked_test",
                        seeds=[0],
                        scheduler_cases_by_dataset={"cryptos": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()

            embeddings = runner.load_context_embedding_table(Path(tmpdir) / "context_embeddings.npz")

        self.assertEqual(len(rows), 1)
        self.assertEqual(append_row_counts, [2, 2, 1])
        self.assertEqual(append_embedding_counts, [2, 2, 1])
        self.assertEqual(extract_t0_batches, [[0, 1], [2, 3], [4]])
        self.assertEqual(sorted(embeddings), ["ck:ctx-0", "ck:ctx-1", "ck:ctx-2", "ck:ctx-3", "ck:ctx-4"])

    def test_conditional_context_embedding_export_batches_full_locked_test_windows(self) -> None:
        class FakeDataset:
            start_indices = [50, 10, 30, 20, 40]

            def __getitem__(self, idx):
                t0 = int(self.start_indices[int(idx)])
                return torch.full((2, 3), float(t0)), torch.zeros(3), {"target_t": t0}

        class FakeCache:
            def __init__(self, ctx_summary: torch.Tensor) -> None:
                self.ctx_summary = ctx_summary

        class FakeBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.batch_shapes = []
                self.grad_enabled = []

            def precompute(self, hist_batch: torch.Tensor) -> FakeCache:
                self.batch_shapes.append(tuple(int(dim) for dim in hist_batch.shape))
                self.grad_enabled.append(bool(torch.is_grad_enabled()))
                if int(hist_batch.shape[0]) > 2:
                    raise AssertionError("context embedding export must not precompute the full locked-test set at once")
                first_value = hist_batch[:, 0, 0].float()
                return FakeCache(torch.stack([first_value, first_value + 0.5], dim=1))

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = FakeBackbone()

        model = FakeModel()
        model.train(True)
        embeddings = runner._extract_conditional_context_embeddings(
            model=model,
            ds=FakeDataset(),
            chosen_t0s=[50, 10, 30, 20, 40],
            device=torch.device("cpu"),
            context_embedding_kind="ctx_summary",
            batch_size=2,
        )

        self.assertEqual(model.backbone.batch_shapes, [(2, 2, 3), (2, 2, 3), (1, 2, 3)])
        self.assertEqual(model.backbone.grad_enabled, [False, False, False])
        self.assertTrue(model.training)
        self.assertTrue(model.backbone.training)
        self.assertEqual(list(embeddings), [50, 10, 30, 20, 40])
        self.assertEqual(embeddings[50], [50.0, 50.5])
        self.assertEqual(embeddings[40], [40.0, 40.5])

    def test_conditional_context_embedding_export_batch_size_tracks_physical_batch(self) -> None:
        self.assertEqual(
            runner._context_embedding_export_batch_size(SimpleNamespace(batch_size=8)),
            8,
        )
        self.assertEqual(
            runner._context_embedding_export_batch_size(SimpleNamespace(train=SimpleNamespace(batch_size=2))),
            2,
        )
        self.assertEqual(
            runner._context_embedding_export_batch_size(SimpleNamespace(batch_size=128)),
            runner.CONTEXT_EMBEDDING_EXPORT_MAX_BATCH_SIZE,
        )
        self.assertEqual(
            runner._context_embedding_export_batch_size(SimpleNamespace(batch_size=0)),
            runner.CONTEXT_EMBEDDING_EXPORT_FALLBACK_BATCH_SIZE,
        )

    def test_conditional_generation_validation_eval_windows_val_cap_is_global_across_seeds(self) -> None:
        class FakeDataset:
            start_indices = list(range(1000))

            def __len__(self) -> int:
                return len(self.start_indices)

        fake_checkpoint = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
            "checkpoint_path": PROJECT_ROOT / "model.pt",
            "checkpoint_id": "ck",
            "backbone_name": "otflow",
            "train_steps": 20000,
            "train_budget_label": "20k",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "cryptos",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0,1",
                    "--split_phase",
                    "validation_tuning",
                    "--eval_windows_val",
                    "5",
                    "--backbone_manifest",
                    str(PROJECT_ROOT / "backbone_manifest.json"),
                    "--checkpoint_steps",
                    "20000",
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            captured_windows = []

            def fake_choose_valid_windows(ds, *, horizon: int, n_windows: int, seed: int):
                del ds, horizon, seed
                return list(range(int(n_windows)))

            def fake_run_fixed_schedule_variant(**kwargs):
                captured_windows.append(len(kwargs["chosen_t0s"]))
                return {
                    "score_main": 1.0,
                    "temporal_uw1": 0.1,
                    "temporal_cw1": 0.2,
                    "temporal_tstr_f1": None,
                    "temporal_tstr_f1_applicable": False,
                    "evaluation_protocol": {"chosen_t0s_hash": "hash"},
                }

            try:
                with mock.patch.object(
                    runner,
                    "load_conditional_generation_checkpoint_splits",
                    return_value=fake_checkpoint,
                ), mock.patch.object(runner, "resolved_validation_windows", return_value=999), mock.patch.object(
                    runner,
                    "_choose_valid_windows",
                    side_effect=fake_choose_valid_windows,
                ), mock.patch.object(
                    runner,
                    "run_fixed_schedule_variant",
                    side_effect=fake_run_fixed_schedule_variant,
                ):
                    rows = runner._run_conditional_generation_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="validation_tuning",
                        seeds=[0, 1],
                        scheduler_cases_by_dataset={"cryptos": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()

        self.assertEqual(sum(captured_windows), 5)
        self.assertEqual(sum(int(row["selected_examples"]) for row in rows), 5)
        self.assertTrue(all(int(row["global_selected_examples"]) == 5 for row in rows))
        self.assertTrue(all(row["selected_examples_cap_source"] == "eval_windows_val" for row in rows))

    def test_molecule_locked_test_defaults_to_full_split_not_context_cap(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 1000

        loaded = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
        }
        captured_lengths = []

        def fake_eval(*args, **kwargs):
            del args
            captured_lengths.append(len(kwargs["example_indices"]))
            return {
                "molecule_kabsch_rmsd_3d": 1.0,
                "molecule_ensemble_velocity_norm_w1": 1.0,
                "molecule_ensemble_acceleration_norm_w1": 1.0,
                "molecule_rollout_velocity_norm_w1": 1.0,
                "molecule_rollout_acceleration_norm_w1": 1.0,
                "molecule_coordinate_w1_mean": 1.0,
                "molecule_pair_distance_w1": 1.0,
                "selection_metric_value": 1.0,
                "eval_windows": len(kwargs["example_indices"]),
                "realized_nfe": 4,
                "per_context_rows": [],
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "backbone_manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tmpdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--molecule_datasets",
                    "molecule_3d_set1",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0",
                    "--split_phase",
                    "locked_test",
                    "--context_sample_count",
                    "5",
                    "--backbone_manifest",
                    str(manifest_path),
                    "--checkpoint_steps",
                    "20000",
                    "--molecule_group_root",
                    tmpdir,
                ]
            )
            recorder = runner._init_row_recorder(Path(tmpdir), args)
            try:
                with mock.patch.object(runner, "load_backbone_manifest", return_value={}), mock.patch.object(
                    runner,
                    "load_molecule_group_manifest",
                    return_value={},
                ), mock.patch.object(
                    runner,
                    "trainable_molecule_group_members",
                    return_value=[{"member_key": "member_a", "stratum": "set1", "processed_dir": "member_a"}],
                ), mock.patch.object(
                    runner,
                    "find_backbone_artifact",
                    return_value={
                        "checkpoint_path": str(Path(tmpdir) / "model.pt"),
                        "checkpoint_id": "molecule_ckpt",
                        "train_steps": 20000,
                        "train_budget_label": "20k",
                    },
                ), mock.patch.object(
                    runner,
                    "load_molecule_checkpoint_splits",
                    return_value=loaded,
                ), mock.patch.object(
                    runner,
                    "evaluate_molecule_rollout_schedule",
                    side_effect=fake_eval,
                ):
                    rows = runner._run_molecule_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="locked_test",
                        seeds=[0],
                        scheduler_cases_by_dataset={"molecule_3d_set1": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()

        self.assertEqual(captured_lengths, [1000])
        self.assertEqual(rows[0]["selected_examples"], 1000)
        self.assertEqual(rows[0]["selected_examples_cap"], 1000)
        self.assertEqual(rows[0]["selected_examples_cap_source"], "locked_test_full")
        self.assertEqual(rows[0]["uncapped_candidate_examples"], 1000)
        self.assertFalse(rows[0]["selection_was_capped"])

    def test_molecule_train_tuning_uses_twenty_percent_then_global_cap(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 5000

        loaded = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
        }
        captured_lengths = []

        def fake_eval(*args, **kwargs):
            del args
            captured_lengths.append(len(kwargs["example_indices"]))
            return {
                "molecule_kabsch_rmsd_3d": 1.0,
                "molecule_ensemble_velocity_norm_w1": 1.0,
                "molecule_ensemble_acceleration_norm_w1": 1.0,
                "molecule_rollout_velocity_norm_w1": 1.0,
                "molecule_rollout_acceleration_norm_w1": 1.0,
                "molecule_coordinate_w1_mean": 1.0,
                "molecule_pair_distance_w1": 1.0,
                "selection_metric_value": 1.0,
                "eval_windows": len(kwargs["example_indices"]),
                "realized_nfe": 4,
                "per_context_rows": [],
            }

        def fake_artifact(*args, **kwargs):
            del args
            member_key = str(kwargs["member_key"])
            return {
                "checkpoint_path": str(Path(tempdir) / f"{member_key}.pt"),
                "checkpoint_id": f"molecule_{member_key}",
                "train_steps": 20000,
                "train_budget_label": "20k",
            }

        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "backbone_manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tempdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--molecule_datasets",
                    "molecule_3d_set1",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0,1",
                    "--split_phase",
                    "train_tuning",
                    "--eval_train_fraction",
                    "0.2",
                    "--context_sample_count",
                    "256",
                    "--backbone_manifest",
                    str(manifest_path),
                    "--checkpoint_steps",
                    "20000",
                    "--molecule_group_root",
                    tempdir,
                ]
            )
            recorder = runner._init_row_recorder(Path(tempdir), args)
            try:
                with mock.patch.object(runner, "load_backbone_manifest", return_value={}), mock.patch.object(
                    runner,
                    "load_molecule_group_manifest",
                    return_value={},
                ), mock.patch.object(
                    runner,
                    "trainable_molecule_group_members",
                    return_value=[
                        {"member_key": "member_a", "stratum": "set1", "processed_dir": "member_a"},
                        {"member_key": "member_b", "stratum": "set1", "processed_dir": "member_b"},
                    ],
                ), mock.patch.object(
                    runner,
                    "find_backbone_artifact",
                    side_effect=fake_artifact,
                ), mock.patch.object(
                    runner,
                    "load_molecule_checkpoint_splits",
                    return_value=loaded,
                ), mock.patch.object(
                    runner,
                    "_choose_molecule_indices",
                    side_effect=AssertionError("train_tuning should use stratified train indices directly"),
                ), mock.patch.object(
                    runner,
                    "evaluate_molecule_rollout_schedule",
                    side_effect=fake_eval,
                ):
                    rows = runner._run_molecule_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="train_tuning",
                        seeds=[0, 1],
                        scheduler_cases_by_dataset={"molecule_3d_set1": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()

        self.assertEqual(sum(captured_lengths), 256)
        self.assertTrue(all(length >= 1 for length in captured_lengths))
        self.assertEqual(sum(int(row["selected_examples"]) for row in rows), 256)
        self.assertTrue(all(int(row["global_selected_examples"]) == 256 for row in rows))
        self.assertTrue(all(int(row["global_uncapped_candidate_examples"]) == 4000 for row in rows))
        self.assertTrue(all(int(row["train_tuning_target_examples"]) == 1000 for row in rows))
        self.assertTrue(all(int(row["train_tuning_uncapped_candidate_examples"]) == 1000 for row in rows))
        self.assertTrue(all(row["selected_examples_cap_source"] == "context_sample_count" for row in rows))
        self.assertTrue(all(row["train_tuning_sampler"] == "temporal_stratified_hash" for row in rows))

    def test_molecule_context_cap_is_global_across_seeds_and_members(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 1000

        loaded = {
            "model": object(),
            "cfg": object(),
            "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
        }
        captured_lengths = []

        def fake_eval(*args, **kwargs):
            del args
            captured_lengths.append(len(kwargs["example_indices"]))
            return {
                "molecule_kabsch_rmsd_3d": 1.0,
                "molecule_ensemble_velocity_norm_w1": 1.0,
                "molecule_ensemble_acceleration_norm_w1": 1.0,
                "molecule_rollout_velocity_norm_w1": 1.0,
                "molecule_rollout_acceleration_norm_w1": 1.0,
                "molecule_coordinate_w1_mean": 1.0,
                "molecule_pair_distance_w1": 1.0,
                "selection_metric_value": 1.0,
                "eval_windows": len(kwargs["example_indices"]),
                "realized_nfe": 4,
                "per_context_rows": [],
            }

        def fake_artifact(*args, **kwargs):
            del args
            member_key = str(kwargs["member_key"])
            return {
                "checkpoint_path": str(Path(tempdir) / f"{member_key}.pt"),
                "checkpoint_id": f"molecule_{member_key}",
                "train_steps": 20000,
                "train_budget_label": "20k",
            }

        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "backbone_manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            args = runner.build_argparser().parse_args(
                [
                    "--out_root",
                    tempdir,
                    "--forecast_datasets",
                    "",
                    "--conditional_generation_datasets",
                    "",
                    "--molecule_datasets",
                    "molecule_3d_set1",
                    "--baseline_scheduler_names",
                    "uniform",
                    "--target_nfe_values",
                    "4",
                    "--solver_names",
                    "euler",
                    "--seeds",
                    "0,1",
                    "--split_phase",
                    "validation_tuning",
                    "--eval_windows_val",
                    "5",
                    "--backbone_manifest",
                    str(manifest_path),
                    "--checkpoint_steps",
                    "20000",
                    "--molecule_group_root",
                    tempdir,
                ]
            )
            recorder = runner._init_row_recorder(Path(tempdir), args)
            try:
                with mock.patch.object(runner, "load_backbone_manifest", return_value={}), mock.patch.object(
                    runner,
                    "load_molecule_group_manifest",
                    return_value={},
                ), mock.patch.object(
                    runner,
                    "trainable_molecule_group_members",
                    return_value=[
                        {"member_key": "member_a", "stratum": "set1", "processed_dir": "member_a"},
                        {"member_key": "member_b", "stratum": "set1", "processed_dir": "member_b"},
                    ],
                ), mock.patch.object(
                    runner,
                    "find_backbone_artifact",
                    side_effect=fake_artifact,
                ), mock.patch.object(
                    runner,
                    "load_molecule_checkpoint_splits",
                    return_value=loaded,
                ), mock.patch.object(
                    runner,
                    "evaluate_molecule_rollout_schedule",
                    side_effect=fake_eval,
                ):
                    rows = runner._run_molecule_phase(
                        args,
                        row_recorder=recorder,
                        split_phase="validation_tuning",
                        seeds=[0, 1],
                        scheduler_cases_by_dataset={"molecule_3d_set1": [{"scheduler_key": "uniform"}]},
                    )
            finally:
                recorder["fh"].close()

        self.assertEqual(sum(captured_lengths), 5)
        self.assertTrue(all(length >= 1 for length in captured_lengths))
        self.assertEqual(sum(int(row["selected_examples"]) for row in rows), 5)
        self.assertTrue(all(int(row["global_selected_examples"]) == 5 for row in rows))
        self.assertTrue(all(int(row["selected_examples_cap"]) == 5 for row in rows))
        self.assertTrue(all(row["selected_examples_cap_source"] == "eval_windows_val" for row in rows))

    def test_site_specific_ops_scripts_are_not_in_source_release(self) -> None:
        self.assertFalse((PROJECT_ROOT / "code" / "ops").exists())
        self.assertFalse(any(PROJECT_ROOT.glob("opsi*")))

    def test_retired_source_trees_are_absent(self) -> None:
        self.assertFalse((PROJECT_ROOT / "code").exists())
        self.assertFalse((PROJECT_ROOT / "src" / "old_code").exists())
        self.assertFalse((PROJECT_ROOT / "old_code").exists())

    def test_legacy_cleanup_targets_are_removed(self) -> None:
        removed = {
            "adaptive_noise_sampler_followup.py",
            "adaptive_deterministic_refinement_followup.py",
            "build_adaptive_solver_matched_nfe_study.py",
            "benchmark_otflow_suite.py",
            "baselines.py",
            "deepmarket_baselines.py",
            "temporal_baselines.py",
            "otflow_baselines.py",
            "fm_backbone_readiness_audit.py",
            "merge_otflow_baseline_main_table.py",
            "otflow_dataset_audit.py",
            "otflow_rollout_length_review.py",
        }
        src_root = PROJECT_ROOT / "src"
        self.assertFalse(any((src_root / name).exists() for name in removed))
        source_text = "\n".join(
            path.read_text(encoding="utf-8") for path in src_root.rglob("*.py") if path.name != Path(__file__).name
        )
        for name in removed:
            if name == "baselines.py":
                continue
            self.assertNotIn(name.removesuffix(".py"), source_text)

    def test_retired_generic_naming_tokens_are_absent(self) -> None:
        retired_patterns = (
            r"RectifiedFlowL[O]B",
            r"L[O]BConfig",
            r"L[O]BDataConfig",
            r"WindowedL[O]BParamsDataset",
            r"L[O]B_FAMILY",
            r"--l[o]b_datasets",
            r"l[o]b_conditional_generation",
            r"['\"]l[o]b['\"]",
            r"[/\\]l[o]b[/\\]",
            r"models\.otflow_backbone",
        )
        source_paths = [
            *Path(PROJECT_ROOT / "src").rglob("*.py"),
            *Path(PROJECT_ROOT / "tests").rglob("*.py"),
            *Path(PROJECT_ROOT / "scripts").rglob("*.py"),
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "pyproject.toml",
        ]
        source_text = "\n".join(
            path.read_text(encoding="utf-8") for path in source_paths if path.exists() and path != Path(__file__)
        )
        for pattern in retired_patterns:
            self.assertIsNone(re.search(pattern, source_text), pattern)

    def test_backbone_manifest_tracks_30_active_artifacts_without_private_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = materialize_backbone_manifest(
                matrix_root=root / "matrix",
                otflow_reuse_root=root / "reuse",
                imported_backbone_root=root / "imported",
                write_path=root / "manifest.json",
            )
        self.assertEqual(int(payload.get("artifact_count", 0)), 30)
        self.assertEqual(int(payload.get("ready_count", -1)), 0)
        self.assertEqual(int(payload.get("missing_count", 0)), 30)


if __name__ == "__main__":
    unittest.main()
