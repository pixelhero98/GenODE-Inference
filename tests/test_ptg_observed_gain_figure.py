from __future__ import annotations

import tempfile
import unittest
import importlib.util
from unittest import mock
from pathlib import Path

import numpy as np

import genode.visualization.build_ptg_observed_gain_figure as ptg_fig


HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _lightweight_grid(schedule_key: str, runtime_nfe: int):
    base = np.linspace(0.0, 1.0, int(runtime_nfe) + 1, dtype=np.float64)
    if schedule_key == "ays":
        grid = base**1.10
    elif schedule_key == "gits":
        grid = base**1.25
    elif schedule_key == "ots":
        grid = 1.0 - (1.0 - base) ** 1.20
    else:
        grid = base
    grid[0] = 0.0
    grid[-1] = 1.0
    return [float(x) for x in grid.tolist()]


class PtgObservedGainFigureTests(unittest.TestCase):
    def test_hardness_normalization_integrates_to_one(self) -> None:
        reference_grid = [0.0, 0.2, 0.5, 1.0]
        kappa, widths, _eps_h, integral = ptg_fig.normalize_hardness_for_ptg([1.0, 2.0, 4.0], reference_grid)
        self.assertAlmostEqual(float(np.sum(widths * kappa)), 1.0, places=12)
        self.assertAlmostEqual(integral, 1.0, places=12)

    def test_uniform_schedule_density_and_ptg(self) -> None:
        reference_grid = [0.0, 0.1, 0.4, 0.7, 1.0]
        uniform_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        rho, integral = ptg_fig.schedule_density_on_reference_grid(uniform_grid, reference_grid)
        self.assertTrue(np.allclose(rho, np.ones(4), atol=1e-12, rtol=1e-12))
        self.assertAlmostEqual(integral, 1.0, places=12)
        result = ptg_fig.ptg_from_trace([0.2, 0.6, 1.0, 0.4], reference_grid, uniform_grid, solver_order_p=2.0)
        self.assertAlmostEqual(result.ser, 1.0, places=12)
        self.assertAlmostEqual(result.ptg_percent, 0.0, places=12)

    def test_nonuniform_schedule_density_integrates_to_one(self) -> None:
        reference_grid = [0.0, 0.1, 0.25, 0.5, 0.8, 1.0]
        schedule_grid = [0.0, 0.05, 0.2, 0.55, 1.0]
        rho, integral = ptg_fig.schedule_density_on_reference_grid(schedule_grid, reference_grid)
        widths = np.diff(np.asarray(reference_grid, dtype=np.float64))
        self.assertTrue(np.all(rho > 0.0))
        self.assertAlmostEqual(float(np.sum(widths * rho)), 1.0, places=12)
        self.assertAlmostEqual(integral, 1.0, places=12)

    def test_stabilized_density_integrates_to_one(self) -> None:
        reference_grid = [0.0, 0.1, 0.25, 0.5, 0.8, 1.0]
        schedule_grid = [0.0, 0.05, 0.2, 0.55, 1.0]
        rho, _integral = ptg_fig.schedule_density_on_reference_grid(schedule_grid, reference_grid)
        stabilized, stabilized_integral = ptg_fig.stabilize_density(rho, reference_grid, eta=0.05)
        widths = np.diff(np.asarray(reference_grid, dtype=np.float64))
        self.assertTrue(np.all(stabilized > 0.0))
        self.assertAlmostEqual(float(np.sum(widths * stabilized)), 1.0, places=12)
        self.assertAlmostEqual(stabilized_integral, 1.0, places=12)

    def test_local_defect_trace_uses_solver_order_width_scaling(self) -> None:
        reference_grid = [0.0, 0.25, 0.75, 1.0]
        oracle = [0.25, 0.5, 1.0]
        actual = ptg_fig.local_defect_trace_from_oracle(oracle, reference_grid, solver_order_p=2.0, eps=0.0)
        widths = np.diff(np.asarray(reference_grid, dtype=np.float64))
        expected = np.asarray(oracle, dtype=np.float64) / np.power(widths, 3.0)
        self.assertTrue(np.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_reversed_schedule_grid_is_monotone_with_endpoints(self) -> None:
        reversed_grid = ptg_fig.reverse_schedule_grid([0.0, 0.05, 0.2, 0.55, 1.0])
        self.assertEqual(reversed_grid[0], 0.0)
        self.assertEqual(reversed_grid[-1], 1.0)
        self.assertTrue(all(right > left for left, right in zip(reversed_grid, reversed_grid[1:])))

    def test_observed_gain_calculation(self) -> None:
        rows = [
            {
                "dataset": "electricity",
                "split_phase": "locked_test",
                "checkpoint_id": "ckpt",
                "backbone_name": "otflow",
                "train_budget_label": "20k",
                "train_steps": 20000,
                "target_nfe": 10,
                "runtime_nfe": 10,
                "dense_reference_macro_steps": 160,
                "dense_reference_macro_factor": 16.0,
                "evaluation_seed": seed,
                "solver_key": "euler",
                "solver_name": "euler",
                "schedule_key": "ays",
                "schedule_name": "AYS",
                "integration_error": error,
                "uniform_integration_error": uniform,
                "integration_gain_percent": 100.0 * (1.0 - error / uniform),
                "eval_examples": 3,
                "endpoint_space": "normalized_model_output",
            }
            for seed, error, uniform in [(0, 0.8, 1.0), (1, 0.9, 1.2)]
        ]
        stats = ptg_fig.aggregate_integration_error_rows(rows)
        self.assertEqual(len(stats), 1)
        expected = 0.5 * (20.0 + 25.0)
        self.assertAlmostEqual(stats[0]["integration_gain_percent_mean"], expected)

    def test_20k_filter_has_180_points_and_excludes_late_power(self) -> None:
        zip_path = ptg_fig.DEFAULT_ZIP_PATH
        if not zip_path.exists():
            self.skipTest(f"Missing local 20k.zip at {zip_path}")
        rows = ptg_fig.load_observed_gain_rows(zip_path)
        self.assertEqual(len(rows), 180)
        self.assertFalse(any(key[3] == "late_power_3" for key in rows))
        self.assertEqual({key[3] for key in rows}, set(ptg_fig.TRANSFER_SCHEDULES))

    def test_synthetic_points_are_exact_scope(self) -> None:
        with mock.patch.object(ptg_fig, "build_fixed_schedule_grid", side_effect=_lightweight_grid):
            points = ptg_fig.build_points(ptg_fig.synthetic_payload(), ptg_fig.synthetic_observed_rows())
        self.assertEqual(len(points), 180)
        self.assertEqual({point["schedule_key"] for point in points}, set(ptg_fig.TRANSFER_SCHEDULES))
        self.assertTrue(all("observed_integration_gain_percent" in point for point in points))
        for key in (
            "ptg_info_growth_raw",
            "ptg_info_growth_reversed",
            "ptg_local_defect_eta005",
            "ptg_local_defect_reversed_eta005",
        ):
            self.assertTrue(all(key in point for point in points))

    def test_synthetic_plot_has_one_panel_reference_lines_trend_and_spearman(self) -> None:
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib is not installed in this local Python")
        with mock.patch.object(ptg_fig, "build_fixed_schedule_grid", side_effect=_lightweight_grid):
            points = ptg_fig.build_points(ptg_fig.synthetic_payload(), ptg_fig.synthetic_observed_rows())
        fig, ax = ptg_fig.build_figure(points)
        try:
            self.assertEqual(len(fig.axes), 1)
            self.assertIsNone(fig._suptitle)
            lines = ax.get_lines()
            has_x_zero = any(np.allclose(line.get_xdata(), [0.0, 0.0]) for line in lines)
            has_y_zero = any(np.allclose(line.get_ydata(), [0.0, 0.0]) for line in lines)
            has_trend = any(len(line.get_xdata()) > 2 for line in lines)
            self.assertTrue(has_x_zero)
            self.assertTrue(has_y_zero)
            self.assertTrue(has_trend)
            self.assertTrue(any("Spearman rho" in text.get_text() for text in ax.texts))
            self.assertEqual(ax.get_ylabel(), "Observed integration-error gain over uniform (%)")
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plot_points_writes_png_and_pdf(self) -> None:
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib is not installed in this local Python")
        with mock.patch.object(ptg_fig, "build_fixed_schedule_grid", side_effect=_lightweight_grid):
            points = ptg_fig.build_points(ptg_fig.synthetic_payload(), ptg_fig.synthetic_observed_rows())
        with tempfile.TemporaryDirectory() as tmpdir:
            png = Path(tmpdir) / "ptg.png"
            pdf = Path(tmpdir) / "ptg.pdf"
            outputs = ptg_fig.plot_points(points, png_path=png, pdf_path=pdf, dpi=120)
            self.assertEqual(outputs["png"], str(png))
            self.assertEqual(outputs["pdf"], str(pdf))
            self.assertTrue(png.exists())
            self.assertTrue(pdf.exists())
            self.assertGreater(png.stat().st_size, 0)
            self.assertGreater(pdf.stat().st_size, 0)

    def test_diagnostic_summary_contains_main_ptg_status(self) -> None:
        with mock.patch.object(ptg_fig, "build_fixed_schedule_grid", side_effect=_lightweight_grid):
            points = ptg_fig.build_points(ptg_fig.synthetic_payload(), ptg_fig.synthetic_observed_rows())
        summary = ptg_fig.summarize_ptg_points(points)
        self.assertEqual(summary["main_ptg_key"], "ptg_info_growth_raw")
        self.assertEqual(summary["n_points"], 180)
        self.assertEqual(summary["observed_y_key"], "observed_integration_gain_percent")
        self.assertIn("ptg_info_growth_raw", summary["variants"])

    def test_integration_gain_loader_has_180_points_and_excludes_nontransferred(self) -> None:
        stats_rows = []
        for dataset in ptg_fig.DATASET_ORDER:
            for solver_key in ptg_fig.SOLVER_ORDER:
                for target_nfe in ptg_fig.TARGET_NFES:
                    runtime_nfe = target_nfe if solver_key in {"euler", "dpmpp2m"} else target_nfe // 2
                    for schedule_key in (*ptg_fig.INTEGRATION_SCHEDULES, "late_power_3"):
                        stats_rows.append(
                            {
                                "dataset": dataset,
                                "split_phase": "locked_test",
                                "checkpoint_id": "ckpt",
                                "backbone_name": "otflow",
                                "train_budget_label": "20k",
                                "train_steps": 20000,
                                "target_nfe": int(target_nfe),
                                "runtime_nfe": int(runtime_nfe),
                                "dense_reference_macro_steps": int(16 * runtime_nfe),
                                "dense_reference_macro_factor": 16.0,
                                "solver_key": solver_key,
                                "solver_name": solver_key,
                                "schedule_key": schedule_key,
                                "schedule_name": schedule_key,
                                "n_seeds": 5,
                                "seed_values": "0;1;2;3;4",
                                "integration_error_mean": 1.0,
                                "integration_error_std": 0.0,
                                "uniform_integration_error_mean": 1.0,
                                "uniform_integration_error_std": 0.0,
                                "integration_gain_percent_mean": 0.0,
                                "integration_gain_percent_std": 0.0,
                                "eval_examples": 2,
                                "endpoint_space": "normalized_model_output",
                            }
                        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.csv"
            ptg_fig.write_csv_rows(stats_rows, path)
            rows = ptg_fig.load_integration_gain_rows(path)
        self.assertEqual(len(rows), 180)
        self.assertEqual({key[3] for key in rows}, set(ptg_fig.TRANSFER_SCHEDULES))
        self.assertFalse(any(key[3] == "late_power_3" for key in rows))


if __name__ == "__main__":
    unittest.main()
