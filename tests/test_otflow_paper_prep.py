from __future__ import annotations

import json
import re
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import genode.evaluation.diffusion_flow_time_reparameterization as runner
from genode.schedule_transfer.diffusion_flow_schedules import (
    EXPERIMENTAL_FIXED_SCHEDULE_KEYS,
    EXPERIMENTAL_REVERSED_SCHEDULE_KEYS,
    build_schedule_grid,
)
from genode.evaluation.fm_backbone_registry import materialize_backbone_manifest
from genode.schedule_transfer.otflow_paper_registry import (
    BASELINE_SCHEDULE_KEYS,
    MAIN_NFE_VALUES,
    METHOD_KEY,
    TRANSFER_SCHEDULE_KEYS,
    paper_registry_snapshot,
    paper_schedule_specs,
    paper_solver_specs,
)
from genode.schedule_transfer.otflow_signal_traces import NATIVE_INFO_GROWTH_TRACE_KEY, NATIVE_SIGNAL_TRACE_KEYS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DiffusionFlowPaperPrepTests(unittest.TestCase):
    def test_registry_exposes_diffusion_flow_method_not_tvd(self) -> None:
        snapshot = paper_registry_snapshot()
        self.assertEqual(METHOD_KEY, "diffusion_flow_time_reparameterization")
        self.assertEqual(snapshot["paper_method"], "diffusion_flow_time_reparameterization")
        self.assertFalse(any(spec.comparison_role == "paper_method" and spec.key == "tvd" for spec in paper_schedule_specs()))
        self.assertIn("flowts_power_sampling", {spec.key for spec in paper_schedule_specs()})
        self.assertNotIn("atss", {spec.key for spec in paper_schedule_specs()})

    def test_schedule_sets_are_exact(self) -> None:
        self.assertEqual(BASELINE_SCHEDULE_KEYS, ("uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots"))
        self.assertEqual(TRANSFER_SCHEDULE_KEYS, ("ays", "gits", "ots"))
        self.assertNotIn("uniform_reversed", EXPERIMENTAL_REVERSED_SCHEDULE_KEYS)
        self.assertEqual(EXPERIMENTAL_FIXED_SCHEDULE_KEYS[: len(BASELINE_SCHEDULE_KEYS)], BASELINE_SCHEDULE_KEYS)

    def test_registry_exposes_active_baseline_matrix(self) -> None:
        snapshot = paper_registry_snapshot()
        self.assertEqual(MAIN_NFE_VALUES, (4, 8, 12, 16))
        self.assertEqual(snapshot["main_nfe_values"], [4, 8, 12, 16])
        self.assertEqual(runner.DEFAULT_TARGET_NFE_VALUES, (4, 8, 12, 16))
        self.assertEqual(snapshot["baseline_schedule_keys"], ["uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots"])

        solver_names = {spec.display_name for spec in paper_solver_specs()}
        self.assertIn("Euler", solver_names)
        self.assertIn("Heun / RK2", solver_names)
        self.assertIn("Midpoint RK2", solver_names)
        self.assertIn("DPM++2M", solver_names)

    def test_gipo_docs_describe_active_density_path(self) -> None:
        text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        lower = text.lower()

        self.assertRegex(lower, r"continuous[- ]density")
        self.assertRegex(lower, r"teacher[- ]weighted")
        for expected in (
            "gipo",
            "gipo_density",
            "density_mass",
            "rank",
            "huber",
            "uniform",
            "late_power_3",
            "flowts_power_sampling",
            "ays",
            "gits",
            "ots",
            "ser",
            "context-disjoint",
            "locked-test",
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
                                        "runtime_nfe": 2,
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

        self.assertEqual(cases[0]["runtime_nfe"], 2)
        self.assertEqual(cases[0]["macro_steps"], 2)
        self.assertEqual(cases[0]["realized_nfe"], 4)

    def test_summary_scheduler_cases_are_forecast_only(self) -> None:
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
            conditional_cases = runner._scheduler_cases_for_datasets(args, ["lobster_synthetic"], include_summary_cases=False)

        self.assertEqual([case["scheduler_key"] for case in forecast_cases["traffic_hourly"]], ["uniform", "gipo"])
        self.assertEqual([case["scheduler_key"] for case in conditional_cases["lobster_synthetic"]], ["uniform"])

    def test_aggregate_relative_gain_uses_fraction_units(self) -> None:
        rows = [
            {
                "benchmark_family": "temporal_extrapolation",
                "split_phase": "locked_test",
                "seed": 0,
                "dataset": "traffic_hourly",
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
                "dataset": "traffic_hourly",
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
        self.assertIn("info_growth_hardness_by_step", NATIVE_SIGNAL_TRACE_KEYS)

    def test_runner_dry_run_writes_combined_summary(self) -> None:
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
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
        self.assertIn("flowts_power_sampling", summary["baseline_schedule_keys"])
        self.assertEqual(summary["transfer_schedule_keys"], ["ays", "gits", "ots"])

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
            runtime_nfe=10,
            solver_key="euler",
            scheduler_key="uniform",
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

    def test_row_recorder_drops_stale_protocol_rows(self) -> None:
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
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
            row_path = Path(tmpdir) / "rows.jsonl"
            row_path.write_text('{"protocol_hash":"old","row_status":"complete"}\n', encoding="utf-8")

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
            recorder_changed = runner._init_row_recorder(Path(tmpdir), args_changed)
            recorder_changed["fh"].close()
            self.assertEqual(row_path.read_text(encoding="utf-8"), "")

    def test_protocol_hash_tracks_data_path_identity(self) -> None:
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
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
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
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
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
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
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
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

    def test_protocol_hash_tracks_selected_split_phase(self) -> None:
        manifest = PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"
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
                        str(PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"),
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
                    str(PROJECT_ROOT / "outputs" / "backbone_matrix" / "backbone_manifest.json"),
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
