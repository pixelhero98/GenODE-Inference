from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import genode.schedule_transfer.diffusion_flow_schedules as schedules
from genode.data.otflow_experiment_plan import CONDITIONAL_GENERATION_FAMILY
from genode.evaluation.diffusion_flow_time_reparameterization import (
    _positive_int_field,
    _row_has_complete_context_artifacts,
)
from genode.gipo.density_representation import average_density_masses, grid_to_density_mass, uniform_reference_grid
from genode.gipo.policy import (
    CONTEXT_EMBEDDING_TABLE_ARTIFACT_VERSION,
    _context_embedding_coverage,
    load_context_embedding_table,
    save_context_embedding_table,
)
from genode.schedule_transfer.diffusion_flow_schedules import (
    BASELINE_SCHEDULE_KEYS,
    EXPERIMENTAL_AVERAGED_FIXED_SCHEDULE_KEYS,
    EXPERIMENTAL_FIXED_SCHEDULE_KEYS,
    EXPERIMENTAL_REVERSED_SCHEDULE_KEYS,
    build_schedule_grid,
    schedule_display_name,
    schedule_time_alignment,
)
from genode.schedule_transfer.otflow_schedule_diagnostics import _collect_rollout_diagnostics
from genode.schedule_transfer.otflow_reference_tables import augment_rows_with_relative_metrics


class DiffusionFlowScheduleTests(unittest.TestCase):
    def test_resume_context_artifacts_require_exact_identity_and_unique_ids(self) -> None:
        parent = {
            "row_status": "complete",
            "row_signature": "parent",
            "selected_examples": 1,
            "benchmark_family": "temporal_extrapolation",
            "experiment_layout": "layout",
            "scenario_key": "traffic_hourly",
            "scenario_family": "forecast",
            "method_key": "fixed",
            "nfe_role": "seen",
            "checkpoint_step": 20000,
            "checkpoint_id": "checkpoint",
            "protocol_hash": "protocol",
            "split_phase": "validation_tuning",
            "seed": 0,
            "solver_key": "euler",
            "target_nfe": 4,
            "scheduler_key": "uniform",
            "schedule_grid_hash": "grid",
            "context_embedding_kind": "ctx_summary",
        }
        context = {
            **parent,
            "parent_row_signature": "parent",
            "row_signature": "context-row",
            "context_id": "context-a",
            "context_embedding_id": "embedding-a",
        }
        self.assertTrue(
            _row_has_complete_context_artifacts(
                parent,
                context_rows_by_signature={"context-row": context},
                context_embeddings={"embedding-a": [0.0]},
            )
        )
        self.assertFalse(
            _row_has_complete_context_artifacts(
                parent,
                context_rows_by_signature={
                    "context-row": {**context, "checkpoint_id": "wrong"}
                },
                context_embeddings={"embedding-a": [0.0]},
            )
        )
        self.assertFalse(
            _row_has_complete_context_artifacts(
                parent,
                context_rows_by_signature={
                    "context-row": context,
                    "extra-row": {
                        **context,
                        "row_signature": "extra-row",
                        "context_id": "context-b",
                        "context_embedding_id": "embedding-b",
                    },
                },
                context_embeddings={"embedding-a": [0.0], "embedding-b": [1.0]},
            )
        )

    def test_conditional_resume_context_artifacts_bind_panel_identity(self) -> None:
        chosen_t0s = [10, 20]
        chosen_hash = hashlib.sha256(
            json.dumps(chosen_t0s, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        protocol_hash = "a" * 64
        parent = {
            "row_status": "complete",
            "row_signature": "parent",
            "selected_examples": 2,
            "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
            "experiment_layout": "layout",
            "scenario_key": "lobster_synthetic",
            "scenario_family": CONDITIONAL_GENERATION_FAMILY,
            "method_key": "fixed",
            "nfe_role": "seen",
            "checkpoint_step": 4000,
            "checkpoint_id": "checkpoint",
            "protocol_hash": "protocol",
            "split_phase": "locked_test",
            "seed": 0,
            "solver_key": "euler",
            "target_nfe": 4,
            "scheduler_key": "uniform",
            "schedule_grid_hash": "grid",
            "context_embedding_kind": "ctx_summary",
            "chosen_t0s_hash": chosen_hash,
            "evaluation_protocol_hash": protocol_hash,
            "temporal_uw1": 0.1,
            "temporal_cw1": 0.2,
        }
        context_rows = {}
        context_embeddings = {}
        for example_idx, target_t in enumerate(chosen_t0s):
            row_signature = f"context-row-{example_idx}"
            context_id = f"context-{example_idx}"
            embedding_id = f"embedding-{example_idx}"
            evaluation_seed = 100 + example_idx
            context_rows[row_signature] = {
                **parent,
                "parent_row_signature": "parent",
                "row_signature": row_signature,
                "context_id": context_id,
                "context_embedding_id": embedding_id,
                "context_schema": "conditional_generation_window",
                "example_idx": example_idx,
                "target_t": target_t,
                "evaluation_seed": evaluation_seed,
                "sample_seed_start": evaluation_seed,
                "sample_seed_values_json": json.dumps(
                    [evaluation_seed], separators=(",", ":")
                ),
                "chosen_examples_hash": chosen_hash,
                "evaluation_protocol_hash": protocol_hash,
            }
            context_embeddings[embedding_id] = [float(example_idx)]

        self.assertTrue(
            _row_has_complete_context_artifacts(
                parent,
                context_rows_by_signature=context_rows,
                context_embeddings=context_embeddings,
            )
        )

        mutations = {
            "evaluation_seed": 999,
            "example_idx": 0,
            "target_t": 10,
            "sample_seed_start": 999,
            "sample_seed_values_json": "[999]",
            "chosen_examples_hash": "b" * 64,
            "evaluation_protocol_hash": "c" * 64,
            "temporal_uw1": 0.11,
            "temporal_cw1": 0.22,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                mutated_rows = copy.deepcopy(context_rows)
                mutated_rows["context-row-1"][field] = value
                self.assertFalse(
                    _row_has_complete_context_artifacts(
                        parent,
                        context_rows_by_signature=mutated_rows,
                        context_embeddings=context_embeddings,
                    )
                )

    def test_conditional_embedding_coverage_digest_binds_panel_identity(self) -> None:
        base_rows = [
            {
                "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                "scenario_key": "lobster_synthetic",
                "split_phase": "locked_test",
                "nfe_role": "seen",
                "checkpoint_id": "checkpoint",
                "context_schema": "conditional_generation_window",
                "context_embedding_kind": "ctx_summary",
                "protocol_hash": "protocol",
                "experiment_layout": "layout",
                "scenario_family": CONDITIONAL_GENERATION_FAMILY,
                "method_key": "fixed",
                "checkpoint_step": 4000,
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "uniform",
                "schedule_grid_hash": "uniform-grid",
                "parent_row_signature": "uniform-parent",
                "row_signature": f"uniform-row-{example_idx}",
                "context_id": f"context-{example_idx}",
                "context_embedding_id": f"embedding-{example_idx}",
                "evaluation_seed": example_idx,
                "example_idx": example_idx,
                "target_t": 10 + example_idx,
                "sample_seed_start": example_idx,
                "sample_seed_values_json": json.dumps(
                    [example_idx], separators=(",", ":")
                ),
                "chosen_examples_hash": "a" * 64,
                "evaluation_protocol_hash": "b" * 64,
                "temporal_uw1": 0.1,
                "temporal_cw1": 0.2,
            }
            for example_idx in range(2)
        ]
        embedding_ids = ["embedding-0", "embedding-1"]
        baseline = _context_embedding_coverage(embedding_ids, base_rows)
        mutations = {
            "evaluation_seed": 999,
            "example_idx": 9,
            "target_t": 99,
            "sample_seed_start": 999,
            "sample_seed_values_json": "[999]",
            "chosen_examples_hash": "c" * 64,
            "evaluation_protocol_hash": "d" * 64,
            "temporal_uw1": 0.11,
            "temporal_cw1": 0.22,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                mutated_rows = copy.deepcopy(base_rows)
                mutated_rows[0][field] = value
                observed = _context_embedding_coverage(embedding_ids, mutated_rows)
                self.assertEqual(
                    observed["identity_sha256"], baseline["identity_sha256"]
                )
                self.assertNotEqual(
                    observed["conditional_panel"]["identity_sha256"],
                    baseline["conditional_panel"]["identity_sha256"],
                )

        second_schedule_rows = [
            {
                **row,
                "scheduler_key": "gipo",
                "schedule_grid_hash": f"gipo-grid-{example_idx}",
                "parent_row_signature": "gipo-parent",
                "row_signature": f"gipo-row-{example_idx}",
                "evaluation_protocol_hash": "c" * 64,
                "temporal_uw1": 0.05,
                "temporal_cw1": 0.15,
            }
            for example_idx, row in enumerate(base_rows)
        ]
        multi_schedule = _context_embedding_coverage(
            embedding_ids, [*base_rows, *second_schedule_rows]
        )
        self.assertEqual(multi_schedule["context_count"], 2)
        self.assertEqual(multi_schedule["conditional_panel"]["row_count"], 4)

        forecast_row = {
            **base_rows[0],
            "benchmark_family": "temporal_extrapolation",
            "scenario_key": "traffic_hourly",
        }
        forecast_baseline = _context_embedding_coverage(
            ["embedding-0"], [forecast_row]
        )
        forecast_with_panel_fields_changed = {
            **forecast_row,
            "evaluation_seed": 999,
            "target_t": 999,
        }
        self.assertEqual(
            _context_embedding_coverage(
                ["embedding-0"], [forecast_with_panel_fields_changed]
            )["identity_sha256"],
            forecast_baseline["identity_sha256"],
        )
        self.assertNotIn("conditional_panel", forecast_baseline)

    def test_legacy_v2_conditional_coverage_preserves_shared_embeddings(self) -> None:
        base_row = {
            "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
            "experiment_layout": "layout",
            "scenario_key": "lobster_synthetic",
            "scenario_family": CONDITIONAL_GENERATION_FAMILY,
            "method_key": "fixed",
            "nfe_role": "seen",
            "checkpoint_step": 4000,
            "checkpoint_id": "checkpoint",
            "protocol_hash": "protocol",
            "split_phase": "locked_test",
            "seed": 0,
            "solver_key": "euler",
            "target_nfe": 4,
            "scheduler_key": "uniform",
            "schedule_grid_hash": "uniform-grid",
            "context_schema": "conditional_generation_window",
            "context_embedding_kind": "ctx_summary",
            "parent_row_signature": "uniform-parent",
            "row_signature": "uniform-row",
            "context_id": "context-0",
            "context_embedding_id": "embedding-0",
            "evaluation_seed": 100,
            "example_idx": 0,
            "target_t": 10,
            "sample_seed_start": 100,
            "sample_seed_values_json": "[100]",
            "chosen_examples_hash": "a" * 64,
            "evaluation_protocol_hash": "b" * 64,
            "temporal_uw1": 0.1,
            "temporal_cw1": 0.2,
        }
        context_rows = [
            base_row,
            {
                **base_row,
                "scheduler_key": "gipo",
                "schedule_grid_hash": "gipo-grid",
                "parent_row_signature": "gipo-parent",
                "row_signature": "gipo-row",
                "evaluation_protocol_hash": "c" * 64,
                "temporal_uw1": 0.05,
                "temporal_cw1": 0.15,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "context_embeddings.npz"
            manifest = save_context_embedding_table(
                path,
                {"embedding-0": [0.0, 1.0]},
                metadata={"context_embedding_kind": "ctx_summary"},
                context_rows=context_rows,
            )
            self.assertEqual(
                manifest["artifact_version"],
                CONTEXT_EMBEDDING_TABLE_ARTIFACT_VERSION,
            )
            manifest["artifact_version"] = 2
            manifest["metadata"]["coverage"].pop("conditional_panel")
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            loaded = load_context_embedding_table(
                path,
                expected_context_embedding_kind="ctx_summary",
                require_manifest=True,
                expected_context_rows=context_rows,
            )

        self.assertEqual(set(loaded), {"embedding-0"})

    def test_expected_context_count_rejects_fractional_values(self) -> None:
        self.assertEqual(_positive_int_field({"selected_examples": "4"}, "selected_examples"), 4)
        for value in (4.5, "4.5", "04", True, 0, -1):
            with self.subTest(value=value):
                self.assertIsNone(
                    _positive_int_field({"selected_examples": value}, "selected_examples")
                )

    def test_transfer_schedule_grids_are_valid_for_runtime_steps(self) -> None:
        for schedule_key in ("ays", "gits", "ots"):
            for runtime_steps in (5, 6, 8, 10, 12, 16):
                with self.subTest(schedule_key=schedule_key, runtime_steps=runtime_steps):
                    grid = build_schedule_grid(schedule_key, runtime_steps)
                    self.assertIsNotNone(grid)
                    assert grid is not None
                    self.assertEqual(len(grid), runtime_steps + 1)
                    self.assertAlmostEqual(grid[0], 0.0)
                    self.assertAlmostEqual(grid[-1], 1.0)
                    self.assertTrue(all(b > a for a, b in zip(grid, grid[1:])))

    def test_schedule_grid_rejects_nonpositive_steps(self) -> None:
        for schedule_key in ("uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots"):
            for runtime_steps in (0, -1):
                with self.subTest(schedule_key=schedule_key, runtime_steps=runtime_steps):
                    with self.assertRaisesRegex(ValueError, "n_steps must be positive"):
                        build_schedule_grid(schedule_key, runtime_steps)

    def test_flowts_power_sampling_grid_is_supported(self) -> None:
        grid = build_schedule_grid("flowts_power_sampling", 4)

        self.assertIsNotNone(grid)
        assert grid is not None
        self.assertEqual(len(grid), 5)
        self.assertAlmostEqual(grid[0], 0.0)
        self.assertAlmostEqual(grid[-1], 1.0)
        self.assertTrue(all(b > a for a, b in zip(grid, grid[1:])))

    def test_experimental_reversed_schedule_grids_are_reversed_counterparts(self) -> None:
        self.assertNotIn("uniform_reversed", EXPERIMENTAL_REVERSED_SCHEDULE_KEYS)
        self.assertEqual(EXPERIMENTAL_FIXED_SCHEDULE_KEYS[: len(BASELINE_SCHEDULE_KEYS)], BASELINE_SCHEDULE_KEYS)
        for schedule_key in EXPERIMENTAL_REVERSED_SCHEDULE_KEYS:
            base_key = schedule_key.removesuffix("_reversed")
            with self.subTest(schedule_key=schedule_key):
                base_grid = build_schedule_grid(base_key, 8)
                reversed_grid = build_schedule_grid(schedule_key, 8)
                self.assertIsNotNone(base_grid)
                self.assertIsNotNone(reversed_grid)
                assert base_grid is not None
                assert reversed_grid is not None
                self.assertEqual(len(reversed_grid), len(base_grid))
                self.assertAlmostEqual(reversed_grid[0], 0.0)
                self.assertAlmostEqual(reversed_grid[-1], 1.0)
                self.assertTrue(all(right > left for left, right in zip(reversed_grid, reversed_grid[1:])))
                expected = tuple(1.0 - value for value in reversed(base_grid))
                for observed, target in zip(reversed_grid, expected):
                    self.assertAlmostEqual(observed, target)
                self.assertIn("reversed", schedule_display_name(schedule_key).lower())
                self.assertIn("reversed", schedule_time_alignment(schedule_key))

    def test_averaged_fixed_schedule_grids_average_density_mass(self) -> None:
        reference = uniform_reference_grid()
        for schedule_key in EXPERIMENTAL_AVERAGED_FIXED_SCHEDULE_KEYS:
            base_key = schedule_key.removesuffix("_avg_reversed")
            reversed_key = f"{base_key}_reversed"
            with self.subTest(schedule_key=schedule_key):
                base_grid = build_schedule_grid(base_key, 8)
                reversed_grid = build_schedule_grid(reversed_key, 8)
                averaged_grid = build_schedule_grid(schedule_key, 8)
                self.assertIsNotNone(base_grid)
                self.assertIsNotNone(reversed_grid)
                self.assertIsNotNone(averaged_grid)
                assert base_grid is not None
                assert reversed_grid is not None
                assert averaged_grid is not None
                expected_mass = average_density_masses(
                    grid_to_density_mass(base_grid, reference_time_grid=reference, macro_steps=8),
                    grid_to_density_mass(reversed_grid, reference_time_grid=reference, macro_steps=8),
                )
                observed_mass = grid_to_density_mass(averaged_grid, reference_time_grid=reference, macro_steps=8)
                self.assertAlmostEqual(sum(observed_mass), 1.0, places=6)
                self.assertEqual(len(observed_mass), len(expected_mass))

    def test_seed_paired_relative_mase_gain_is_preserved_in_summary_rows(self) -> None:
        rows = [
            {
                "benchmark_family": "temporal_extrapolation",
                "dataset": "demo",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "train_budget_label": "20k",
                "forecast_crps_mean": 10.0,
                "forecast_mase_mean": 5.0,
            },
            {
                "benchmark_family": "temporal_extrapolation",
                "dataset": "demo",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "gits",
                "train_budget_label": "20k",
                "forecast_crps_mean": 9.0,
                "forecast_mase_mean": 4.0,
                "forecast_relative_crps_gain_vs_uniform_mean": 0.11,
                "forecast_relative_mase_gain_vs_uniform_mean": 0.25,
            },
        ]
        augmented = augment_rows_with_relative_metrics(rows)
        gits = next(row for row in augmented if row["scheduler_key"] == "gits")
        self.assertAlmostEqual(gits["forecast_relative_crps_gain_vs_uniform"], 0.11)
        self.assertAlmostEqual(gits["forecast_relative_mase_gain_vs_uniform"], 0.25)

    def test_rollout_diagnostics_rejects_empty_chosen_t0s(self) -> None:
        cfg = SimpleNamespace(device=torch.device("cpu"))
        ds = SimpleNamespace(cond=None)

        with self.assertRaisesRegex(ValueError, "chosen_t0s must be a non-empty"):
            _collect_rollout_diagnostics(
                object(),
                ds,
                cfg,
                horizon=2,
                macro_steps=3,
                n_windows=1,
                seed=0,
                solver="euler",
                chosen_t0s=[],
            )

    def test_heun_variant_uses_target_nfe_and_solver_macro_steps(self) -> None:
        evaluation = {
            "cmp": {
                "main": {
                    "temporal_tstr_f1_status": "constant_train_class_fallback",
                    "temporal_tstr_f1_train_class_count": 1,
                    "temporal_tstr_f1_test_class_count": 3,
                }
            },
            "meta": {
                "chosen_t0s": [0],
                "chosen_t0s_hash": "hash",
                "horizon": 2,
                "dataset_kind": "synthetic",
                "generation_seed_base": 7,
                "metrics_seed": 11,
                "main_metrics_only": True,
                "per_window_metric_rows": [],
            }
        }
        diagnostics = {
            "mean_field_evals_per_step": 2.0,
            "mean_total_field_evals_per_rollout": 4.0,
        }
        grid_spec = {
            "grid_name": "uniform",
            "grid_kind": "fixed",
            "selection_group": "test",
            "solver_name": "heun",
            "target_nfe": 4,
            "macro_steps": 2,
            "time_grid": (0.0, 0.5, 1.0),
        }
        with (
            mock.patch.object(schedules, "eval_many_windows", return_value=evaluation) as evaluate,
            mock.patch.object(
                schedules,
                "_collect_rollout_diagnostics",
                return_value=diagnostics,
            ) as collect,
            mock.patch.object(
                schedules,
                "_metric_bundle",
                return_value={
                    "score_main": 0.1,
                    "temporal_uw1": 0.2,
                    "temporal_cw1": 0.3,
                },
            ),
            mock.patch.object(schedules.time, "time", side_effect=(10.0, 11.0)),
        ):
            row = schedules.run_fixed_schedule_variant(
                model=object(),
                ds=object(),
                cfg=SimpleNamespace(),
                eval_horizon=2,
                eval_windows=1,
                grid_spec=grid_spec,
                chosen_t0s=[0],
                generation_seed_base=7,
                metrics_seed=11,
                score_main_only=True,
            )

        self.assertEqual(evaluate.call_args.kwargs["nfe"], 4)
        self.assertEqual(collect.call_args.kwargs["macro_steps"], 2)
        self.assertEqual(row["target_total_field_evals"], 4)
        self.assertEqual(
            row["temporal_tstr_f1_status"],
            "constant_train_class_fallback",
        )
        self.assertEqual(row["temporal_tstr_f1_train_class_count"], 1)
        self.assertEqual(row["temporal_tstr_f1_test_class_count"], 3)

        with self.assertRaisesRegex(ValueError, "macro_steps"):
            schedules.run_fixed_schedule_variant(
                model=object(),
                ds=object(),
                cfg=SimpleNamespace(),
                eval_horizon=2,
                eval_windows=1,
                grid_spec={**grid_spec, "macro_steps": 4},
                chosen_t0s=[0],
                generation_seed_base=7,
                metrics_seed=11,
                score_main_only=True,
            )

    def test_fixed_panel_rejects_nonfinite_primary_metrics(self) -> None:
        evaluation = {
            "cmp": {"main": {}},
            "meta": {
                "chosen_t0s": [0],
                "chosen_t0s_hash": "hash",
                "horizon": 2,
                "dataset_kind": "l2",
                "generation_seed_base": 7,
                "metrics_seed": 11,
                "main_metrics_only": False,
                "per_window_metric_rows": [],
            },
        }
        diagnostics = {
            "mean_field_evals_per_step": 1.0,
            "mean_total_field_evals_per_rollout": 2.0,
        }
        grid_spec = {
            "grid_name": "uniform",
            "grid_kind": "fixed",
            "selection_group": "test",
            "solver_name": "euler",
            "target_nfe": 2,
            "macro_steps": 2,
            "time_grid": (0.0, 0.5, 1.0),
        }
        with (
            mock.patch.object(
                schedules, "eval_many_windows", return_value=evaluation
            ),
            mock.patch.object(
                schedules,
                "_collect_rollout_diagnostics",
                return_value=diagnostics,
            ),
            mock.patch.object(
                schedules,
                "_metric_bundle",
                return_value={
                    "score_main": 0.1,
                    "temporal_uw1": float("nan"),
                    "temporal_cw1": 0.2,
                },
            ),
            mock.patch.object(
                schedules.time,
                "time",
                side_effect=(10.0, 11.0, 20.0, 21.0),
            ),
        ):
            for score_main_only in (False, True):
                with self.subTest(score_main_only=score_main_only):
                    with self.assertRaisesRegex(
                        FloatingPointError, "invalid temporal_uw1"
                    ):
                        schedules.run_fixed_schedule_variant(
                            model=object(),
                            ds=object(),
                            cfg=SimpleNamespace(),
                            eval_horizon=2,
                            eval_windows=1,
                            grid_spec=grid_spec,
                            chosen_t0s=[0],
                            generation_seed_base=7,
                            metrics_seed=11,
                            score_main_only=score_main_only,
                        )

    def test_adaptive_panel_forwards_distinct_grids_and_exact_seeds(self) -> None:
        evaluation = {
            "cmp": {
                "main": {
                    "temporal_tstr_f1_status": "failed",
                    "temporal_tstr_f1_train_class_count": 1,
                    "temporal_tstr_f1_test_class_count": 3,
                }
            },
            "meta": {
                "chosen_t0s": [10, 20],
                "chosen_t0s_hash": "panel-hash",
                "horizon": 5,
                "dataset_kind": "l2",
                "generation_seed_values": [100, 101],
                "metrics_seed": 100,
                "main_metrics_only": False,
                "time_grid_mode": "per_window",
                "per_window_metric_rows": [
                    {"target_t": 10, "score_main": 0.2},
                    {"target_t": 20, "score_main": 0.4},
                ],
            },
        }
        metric_bundle = {
            "score_main": 0.3,
            "temporal_uw1": 0.1,
            "temporal_cw1": 0.2,
            "temporal_tstr_f1": None,
            "temporal_tstr_f1_applicable": True,
            "disc_auc": 0.5,
            "disc_auc_gap": 0.0,
            "u_l1": 0.1,
            "c_l1": 0.2,
            "spread_specific_error": 0.3,
            "imbalance_specific_error": 0.4,
            "ret_vol_acf_error": 0.5,
            "impact_response_error": 0.6,
            "efficiency_ms_per_sample": 1.0,
        }
        grids = ((0.0, 0.25, 1.0), (0.0, 0.75, 1.0))
        with (
            mock.patch.object(schedules, "eval_many_windows", return_value=evaluation) as evaluate,
            mock.patch.object(schedules, "_metric_bundle", return_value=metric_bundle),
            mock.patch.object(schedules.time, "time", side_effect=(10.0, 11.0)),
        ):
            row = schedules.run_context_schedule_panel_variant(
                model=object(),
                ds=object(),
                cfg=SimpleNamespace(),
                eval_horizon=5,
                solver_name="euler",
                target_nfe=2,
                time_grids=grids,
                chosen_t0s=[10, 20],
                generation_seed_values=[100, 101],
                metrics_seed=100,
                score_main_only=False,
            )

        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(evaluate.call_args.kwargs["time_grids"], list(grids))
        self.assertEqual(evaluate.call_args.kwargs["chosen_t0s"], [10, 20])
        self.assertEqual(evaluate.call_args.kwargs["generation_seed_values"], [100, 101])
        self.assertNotIn("generation_seed_base", evaluate.call_args.kwargs)
        self.assertEqual(row["panel_context_count"], 2)
        self.assertEqual(row["evaluation_protocol"]["time_grid_mode"], "per_window")
        self.assertIsNone(row["temporal_tstr_f1"])
        self.assertEqual(row["temporal_tstr_f1_status"], "failed")


if __name__ == "__main__":
    unittest.main()
