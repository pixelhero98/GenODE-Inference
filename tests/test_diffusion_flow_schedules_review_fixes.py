from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

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
from genode.schedule_transfer.otflow_paper_tables import augment_rows_with_relative_metrics


class DiffusionFlowScheduleReviewFixTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
