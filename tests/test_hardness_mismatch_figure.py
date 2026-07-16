from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

import genode.visualization.build_hardness_mismatch_figure as figure_builder
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS, TRANSFER_SCHEDULE_KEYS

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


class NativeInfoGrowthFigureTests(unittest.TestCase):
    def test_active_schedule_order_excludes_tvd(self) -> None:
        self.assertEqual(figure_builder.SCHEDULE_ORDER, BASELINE_SCHEDULE_KEYS)
        self.assertEqual(TRANSFER_SCHEDULE_KEYS, ("ays", "gits", "ots"))
        self.assertNotIn("tvd", figure_builder.SCHEDULE_ORDER)
        self.assertNotIn("atss", figure_builder.SCHEDULE_ORDER)
        self.assertIn("flowts_power_sampling", figure_builder.SCHEDULE_ORDER)

    def test_trace_normalization_is_positive_and_mean_one(self) -> None:
        trace = figure_builder.normalize_trace([0.0, 2.0, 4.0, float("nan")])
        self.assertTrue(all(value >= 0.0 for value in trace))
        self.assertAlmostEqual(float(np.mean(trace)), 1.0)

    def test_schedule_node_summary_marks_transfer_schedules(self) -> None:
        uniform = figure_builder.schedule_node_summary("uniform", 4)
        ays = figure_builder.schedule_node_summary("ays", 4)
        self.assertFalse(uniform["is_transfer_schedule"])
        self.assertTrue(ays["is_transfer_schedule"])
        self.assertEqual(len(uniform["time_grid"]), 5)

    def test_synthetic_payload_uses_native_info_growth_trace(self) -> None:
        payload = figure_builder.synthetic_payload(macro_steps=4)
        self.assertEqual(payload["native_trace_key"], "info_growth_hardness_by_step")
        self.assertEqual(payload["paper_facing_trace"], "native_info_growth")
        self.assertEqual(len(payload["schedule_nodes"]), len(BASELINE_SCHEDULE_KEYS))
        self.assertNotIn("selected_tvd_deltas", payload)

    def test_synthetic_cli_uses_macro_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "payload.json"
            result = figure_builder.main(
                ["synthetic", "--out-json", str(output_path), "--macro-steps", "4"]
            )

        self.assertEqual(result["payload"]["schedule_nodes"][0]["macro_steps"], 4)

    def test_plot_payload_writes_png_and_pdf(self) -> None:
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib is not installed")
        payload = figure_builder.synthetic_payload(macro_steps=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            png = Path(tmpdir) / "native.png"
            pdf = Path(tmpdir) / "native.pdf"
            outputs = figure_builder.plot_payload(payload, png_path=png, pdf_path=pdf, dpi=120)
            self.assertEqual(outputs["png"], str(png))
            self.assertEqual(outputs["pdf"], str(pdf))
            self.assertGreater(png.stat().st_size, 0)
            self.assertGreater(pdf.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
