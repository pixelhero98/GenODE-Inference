from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from genode.conditional_opd.ser_ptg_reference import SER_PTG_SCHEDULE_KEY
from genode.conditional_opd.train20_v43_pooled_calibration import (
    DEFAULT_GEOMETRY_MAX_INTERVAL,
    POOLED_CALIBRATION_PHASE,
    build_argparser,
    combine_origin_metrics,
    run_train20_v43_pooled_calibration,
    select_guarded_teacher_utility_schedule,
    train_v43_policy,
    write_fixed_schedule_summary,
)
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS


class Train20V43PooledCalibrationTests(unittest.TestCase):
    def test_pooled_metrics_are_example_weighted(self) -> None:
        pooled = combine_origin_metrics(
            {
                "crps": 2.0,
                "mse": 4.0,
                "mase": 6.0,
                "latency_ms_per_sample": 8.0,
                "eval_examples": 1,
                "eval_horizon": 2,
                "num_eval_samples": 5,
                "realized_nfe": 4,
                "chosen_examples_hash": "train-hash",
                "evaluation_protocol_hash": "train-proto",
            },
            {
                "crps": 4.0,
                "mse": 8.0,
                "mase": 12.0,
                "latency_ms_per_sample": 16.0,
                "eval_examples": 3,
                "eval_horizon": 2,
                "num_eval_samples": 5,
                "realized_nfe": 4,
                "chosen_examples_hash": "val-hash",
                "evaluation_protocol_hash": "val-proto",
            },
        )
        self.assertEqual(pooled["calibration_origin_counts"], {"train20": 1, "former_val": 3})
        self.assertAlmostEqual(pooled["crps"], 3.5)
        self.assertAlmostEqual(pooled["mase"], 10.5)

    def test_fixed_summary_contains_exact_reward_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "fixed.json"
            args = build_argparser().parse_args(
                [
                    "--stage",
                    "write-fixed-summary",
                    "--fixed_schedule_summary",
                    str(out),
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                ]
            )
            summary = write_fixed_schedule_summary(args)
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(summary["schedule_count"], 6)
        self.assertEqual([row["scheduler_key"] for row in payload["schedules"]], list(BASELINE_SCHEDULE_KEYS))

    def test_final_selection_uses_teacher_score_after_geometry_guard(self) -> None:
        def schedule(key: str, score: float, max_interval: float) -> dict:
            return {
                "scheduler_key": key,
                "opd_step_budget": int(key.rsplit("steps", 1)[-1]),
                "teacher_predicted_utility_mean": score,
                "predictions": [
                    {
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "time_grid": [0.0, max_interval, 1.0],
                        "grid_geometry": {
                            "internal_fraction_after_098": 0.0,
                            "min_interval": min(max_interval, 1.0 - max_interval),
                            "max_interval": max(max_interval, 1.0 - max_interval),
                        },
                        "utility": score,
                    }
                ],
            }

        selection = select_guarded_teacher_utility_schedule(
            {
                "schedules": [
                    schedule("conditional_opd_student_v43_steps25", 0.9, DEFAULT_GEOMETRY_MAX_INTERVAL + 0.001),
                    schedule("conditional_opd_student_v43_steps20", 0.4, 0.8),
                ]
            }
        )
        self.assertEqual(selection["unguarded_top_schedule_key"], "conditional_opd_student_v43_steps25")
        self.assertEqual(selection["selected_schedule_key"], "conditional_opd_student_v43_steps20")
        self.assertTrue(selection["selected_geometry"]["passes_geometry_guard"])

    def test_run_dry_run_records_clean_v43_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_argparser().parse_args(
                [
                    "--out_dir",
                    tmpdir,
                    "--device",
                    "cpu",
                    "--skip_locked_test",
                    "--no_archive_v42_f_output",
                ]
            )
            summary = run_train20_v43_pooled_calibration(args)
        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["bo_rounds"], 1)
        self.assertEqual(summary["bo_candidate_count"], 32)
        self.assertFalse(summary["uses_validation_selection"])
        self.assertFalse(summary["uses_source_balanced_rewards"])
        self.assertEqual(summary["student_initialization"], SER_PTG_SCHEDULE_KEY)
        self.assertEqual(summary["final_selector"], "pooled_teacher_utility_with_hard_geometry_guard")
        self.assertFalse(summary["strict_v43_protocol"])
        flat = " ".join(" ".join(cmd) for cmd in summary["commands"])
        self.assertIn("evaluate-pooled", flat)
        self.assertIn("--dataset san_francisco_traffic", flat)
        self.assertNotIn("train_conditional_opd_v42_f", flat)

    def test_run_dry_run_forwards_locked_test_inference_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_argparser().parse_args(
                [
                    "--out_dir",
                    tmpdir,
                    "--device",
                    "cpu",
                    "--dataset_root",
                    str(Path(tmpdir) / "datasets"),
                    "--shared_backbone_root",
                    str(Path(tmpdir) / "backbones"),
                    "--backbone_manifest",
                    str(Path(tmpdir) / "manifest.json"),
                    "--forecast_eval_batch_size",
                    "7",
                    "--no_archive_v42_f_output",
                ]
            )
            summary = run_train20_v43_pooled_calibration(args)
        locked = [cmd for cmd in summary["commands"] if "genode.conditional_opd.evaluate_schedule_summary" in cmd][0]
        self.assertIn("--forecast_eval_batch_size", locked)
        self.assertIn("7", locked)
        self.assertIn("--dataset_root", locked)
        self.assertIn(str(Path(tmpdir) / "datasets"), locked)
        self.assertIn("--shared_backbone_root", locked)
        self.assertIn(str(Path(tmpdir) / "backbones"), locked)
        self.assertIn("--backbone_manifest", locked)
        self.assertIn(str(Path(tmpdir) / "manifest.json"), locked)

    def test_strict_protocol_rejects_noncanonical_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_argparser().parse_args(
                [
                    "--out_dir",
                    tmpdir,
                    "--device",
                    "cpu",
                    "--calibration_seeds",
                    "0",
                    "--strict_v43_protocol",
                    "--skip_locked_test",
                    "--no_archive_v42_f_output",
                ]
            )
            with self.assertRaises(ValueError):
                run_train20_v43_pooled_calibration(args)

    def test_train_policy_dry_run_rejects_old_protocol_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_argparser().parse_args(
                [
                    "--stage",
                    "train-policy",
                    "--out_dir",
                    tmpdir,
                    "--candidate_schedule_summary",
                    str(Path(tmpdir) / "bo.json"),
                    "--dry_run",
                ]
            )
            Path(args.candidate_schedule_summary).write_text(
                json.dumps({"dataset": "san_francisco_traffic", "schedules": [{"scheduler_key": "bo0", "predictions": []}]}),
                encoding="utf-8",
            )
            summary = train_v43_policy(args)
        self.assertEqual(summary["split_phase"], POOLED_CALIBRATION_PHASE)
        self.assertEqual(summary["reward_reference_source"], "pooled_recomputed_rows_only")
        self.assertFalse(summary["uses_source_balanced_rewards"])
        self.assertFalse(summary["lowest_internal_loss_selector_used"])
        self.assertEqual(summary["student_initialization"], SER_PTG_SCHEDULE_KEY)


if __name__ == "__main__":
    unittest.main()
