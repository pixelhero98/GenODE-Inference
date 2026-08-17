from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from genode.gico.density_representation import average_density_masses, grid_to_density_mass, uniform_reference_grid
from genode.schedule_transfer import diffusion_flow_schedules as schedule_module
from genode.schedule_transfer import otflow_schedule_diagnostics as diagnostics_module
from genode.schedule_transfer.diffusion_flow_schedules import (
    BASELINE_SCHEDULE_KEYS,
    EXPERIMENTAL_AVERAGED_FIXED_SCHEDULE_KEYS,
    EXPERIMENTAL_FIXED_SCHEDULE_KEYS,
    EXPERIMENTAL_REVERSED_SCHEDULE_KEYS,
    build_schedule_grid,
    schedule_display_name,
    schedule_time_alignment,
)
from genode.schedule_transfer.otflow_paper_tables import augment_rows_with_relative_metrics
from genode.schedule_transfer.otflow_schedule_diagnostics import _collect_rollout_diagnostics


class DiffusionFlowScheduleReviewFixTests(unittest.TestCase):
    def test_transfer_schedule_grids_are_valid_for_runtime_steps(self) -> None:
        for schedule_key in (
            "ays_sd15_native",
            "ays_sd15_log_sigma",
            "gits_cifar10_native",
            "gits_cifar10_log_sigma",
            "ots_vp_linear_native",
            "ots_vp_linear_log_sigma",
        ):
            for runtime_steps in (5, 6, 8, 10, 12, 16):
                with self.subTest(schedule_key=schedule_key, runtime_steps=runtime_steps):
                    grid = build_schedule_grid(schedule_key, runtime_steps)
                    self.assertIsNotNone(grid)
                    assert grid is not None
                    self.assertEqual(len(grid), runtime_steps + 1)
                    self.assertAlmostEqual(grid[0], 0.0)
                    self.assertAlmostEqual(grid[-1], 1.0)
                    self.assertTrue(all(b > a for a, b in zip(grid, grid[1:], strict=False)))

    def test_schedule_grid_rejects_nonpositive_steps(self) -> None:
        for schedule_key in BASELINE_SCHEDULE_KEYS:
            for runtime_steps in (0, -1):
                with (
                    self.subTest(schedule_key=schedule_key, runtime_steps=runtime_steps),
                    self.assertRaisesRegex(ValueError, "n_steps must be positive"),
                ):
                    build_schedule_grid(schedule_key, runtime_steps)

    def test_flowts_power_sampling_grid_is_supported(self) -> None:
        grid = build_schedule_grid("flowts_power_0p03", 4)

        self.assertIsNotNone(grid)
        assert grid is not None
        self.assertEqual(len(grid), 5)
        self.assertAlmostEqual(grid[0], 0.0)
        self.assertAlmostEqual(grid[-1], 1.0)
        self.assertTrue(all(b > a for a, b in zip(grid, grid[1:], strict=False)))

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
                self.assertTrue(
                    all(right > left for left, right in zip(reversed_grid, reversed_grid[1:], strict=False))
                )
                expected = tuple(1.0 - value for value in reversed(base_grid))
                for observed, target in zip(reversed_grid, expected, strict=False):
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

    def test_fixed_schedule_reports_realized_heun_evaluations_without_retired_trigger_metrics(self) -> None:
        result = {
            "meta": {
                "chosen_t0s": [3],
                "chosen_t0s_hash": "paired",
                "horizon": 1,
                "dataset_kind": "demo",
                "generation_seed_base": 7,
                "metrics_seed": 11,
                "main_metrics_only": False,
                "per_window_metric_rows": [],
            }
        }
        diagnostics = {
            "n_rollout_calls": 1,
            "macro_steps": 2,
            "field_evals_by_step": [2.0, 2.0],
            "disagreement_by_step": [0.0, 0.1],
            "mean_field_evals_per_step": 2.0,
            "mean_total_field_evals_per_rollout": 4.0,
        }
        with (
            patch.object(schedule_module, "_apply_sample_overrides", return_value={}),
            patch.object(
                schedule_module,
                "_restore_sample_overrides",
            ),
            patch.object(schedule_module, "eval_many_windows", return_value=result),
            patch.object(
                schedule_module,
                "_collect_rollout_diagnostics",
                return_value=diagnostics,
            ),
            patch.object(schedule_module, "_metric_bundle", return_value={}),
        ):
            row = schedule_module.run_fixed_schedule_variant(
                model=object(),
                ds=object(),
                cfg=object(),
                eval_horizon=1,
                eval_windows=1,
                grid_spec={
                    "grid_name": "uniform",
                    "grid_kind": "fixed",
                    "selection_group": "uniform",
                    "solver_name": "heun",
                    "nfe": 2,
                    "time_grid": (0.0, 0.5, 1.0),
                },
                chosen_t0s=(3,),
                generation_seed_base=7,
                metrics_seed=11,
                score_main_only=False,
            )

        self.assertEqual(row["target_total_field_evals"], 4)
        self.assertEqual(row["mean_total_field_evals_per_rollout"], 4.0)
        self.assertNotIn("trigger_rate", row)
        self.assertFalse({"trigger_rate", "trigger_by_step", "eligible_by_step"} & set(row["diag"]))

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

    def test_rollout_diagnostics_consumes_only_canonical_trace_fields(self) -> None:
        hist = torch.zeros(2, 3)
        trace = {
            "field_evals_by_step": torch.tensor([[2.0, 2.0]]),
            "disagreement": torch.tensor([[0.0, 0.25]]),
            "mean_total_field_evals_per_rollout": 4.0,
        }
        with (
            patch.object(diagnostics_module, "_get_dataset_item_by_t", return_value=object()),
            patch.object(
                diagnostics_module,
                "_parse_batch",
                return_value=(hist, None, None, None, None),
            ),
            patch.object(diagnostics_module, "resolve_context_length", return_value=2),
            patch.object(
                diagnostics_module,
                "crop_history_window",
                side_effect=lambda value, _: value,
            ),
            patch.object(diagnostics_module, "_future_time_context_seq", return_value=None),
            patch.object(
                diagnostics_module,
                "_sample_eval_trace",
                return_value=(torch.zeros(1, 1, 3), trace, 1),
            ),
        ):
            result = _collect_rollout_diagnostics(
                object(),
                SimpleNamespace(cond=None),
                SimpleNamespace(device=torch.device("cpu")),
                horizon=1,
                macro_steps=2,
                n_windows=1,
                seed=0,
                solver="heun",
                chosen_t0s=(0,),
            )

        self.assertEqual(
            set(result),
            {
                "n_rollout_calls",
                "macro_steps",
                "field_evals_by_step",
                "disagreement_by_step",
                "mean_field_evals_per_step",
                "mean_total_field_evals_per_rollout",
            },
        )
        self.assertEqual(result["field_evals_by_step"], [2.0, 2.0])
        self.assertEqual(result["mean_total_field_evals_per_rollout"], 4.0)


if __name__ == "__main__":
    unittest.main()
