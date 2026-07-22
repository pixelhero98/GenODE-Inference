from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import genode.schedule_transfer.diffusion_flow_schedules as schedules
from genode.evaluation.diffusion_flow_time_reparameterization import (
    _positive_int_field,
    _row_has_complete_context_artifacts,
)
from genode.gipo.density_representation import average_density_masses, grid_to_density_mass, uniform_reference_grid
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
            mock.patch.object(schedules, "_metric_bundle", return_value={}),
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


if __name__ == "__main__":
    unittest.main()
