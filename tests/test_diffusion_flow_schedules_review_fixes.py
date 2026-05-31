from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from genode.schedule_transfer.diffusion_flow_schedules import build_schedule_grid
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

    def test_seed_paired_relative_mase_gain_is_preserved_in_summary_rows(self) -> None:
        rows = [
            {
                "benchmark_family": "forecast_extrapolation",
                "dataset": "demo",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "uniform",
                "train_budget_label": "20k",
                "crps_mean": 10.0,
                "mase_mean": 5.0,
            },
            {
                "benchmark_family": "forecast_extrapolation",
                "dataset": "demo",
                "target_nfe": 10,
                "solver_key": "euler",
                "scheduler_key": "gits",
                "train_budget_label": "20k",
                "crps_mean": 9.0,
                "mase_mean": 4.0,
                "relative_crps_gain_vs_uniform_mean": 0.11,
                "relative_mase_gain_vs_uniform_mean": 0.25,
            },
        ]
        augmented = augment_rows_with_relative_metrics(rows)
        gits = next(row for row in augmented if row["scheduler_key"] == "gits")
        self.assertAlmostEqual(gits["relative_crps_gain_vs_uniform"], 0.11)
        self.assertAlmostEqual(gits["relative_mase_gain_vs_uniform"], 0.25)

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
