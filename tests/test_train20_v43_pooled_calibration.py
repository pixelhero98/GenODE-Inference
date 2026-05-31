from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from genode.conditional_opd.ser_ptg_reference import SER_PTG_SCHEDULE_KEY
from genode.conditional_opd.train20_v43_pooled_calibration import (
    DEFAULT_GEOMETRY_MAX_INTERVAL,
    POOLED_CALIBRATION_PHASE,
    V43_POOLED_CALIBRATION_POOL,
    V43_POOLED_CALIBRATION_PROTOCOL,
    _clean_pooled_rows,
    build_argparser,
    combine_origin_metrics,
    evaluate_pooled_schedule_summary,
    pooled_calibration_origin_indices,
    run_train20_v43_pooled_calibration,
    select_guarded_teacher_utility_schedule,
    train_v43_policy,
    write_fixed_schedule_summary,
)
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS


class Train20V43PooledCalibrationTests(unittest.TestCase):
    def test_combined_pool_sampling_uses_proportional_validation_subset(self) -> None:
        train_indices, val_indices = pooled_calibration_origin_indices(
            80,
            20,
            fraction=0.20,
            seed=11,
            strata=10,
            dataset="unit",
        )
        self.assertEqual(len(train_indices), 16)
        self.assertEqual(len(val_indices), 4)
        self.assertEqual(len(train_indices) + len(val_indices), 20)
        self.assertTrue(all(0 <= int(idx) < 20 for idx in val_indices))

    def test_combined_pool_sampling_rounds_total_before_origin_allocation(self) -> None:
        train_indices, val_indices = pooled_calibration_origin_indices(
            7,
            4,
            fraction=0.20,
            seed=13,
            strata=4,
            dataset="unit",
        )
        self.assertEqual(len(train_indices) + len(val_indices), 2)
        self.assertEqual(len(val_indices), 1)

    def test_huge_raw_train_split_uses_validation_normalized_reference_universe(self) -> None:
        train_indices, val_indices = pooled_calibration_origin_indices(
            1_000_000,
            100,
            fraction=0.20,
            seed=17,
            strata=20,
            dataset="unit",
        )
        self.assertEqual(len(train_indices), 140)
        self.assertEqual(len(val_indices), 20)
        self.assertEqual(len(train_indices) + len(val_indices), 160)
        self.assertNotEqual(len(train_indices), 200_000)
        self.assertLess(len(train_indices) + len(val_indices), 200_000)
        self.assertTrue(all(0 <= int(idx) < 1_000_000 for idx in train_indices))
        self.assertTrue(all(0 <= int(idx) < 100 for idx in val_indices))

    def test_evaluate_pooled_metadata_uses_normalized_combined_reference(self) -> None:
        class FakeDataset:
            def __init__(self, total: int) -> None:
                self.total = int(total)

            def __len__(self) -> int:
                return self.total

        calls = []

        def fake_eval(_model, ds, _cfg, **kwargs):
            indices = [int(idx) for idx in kwargs["example_indices"]]
            calls.append({"total": len(ds), "indices": indices, "label": str(kwargs.get("progress_label", ""))})
            return {
                "crps": 1.0,
                "mse": 2.0,
                "mase": 3.0,
                "latency_ms_per_sample": 4.0,
                "eval_examples": len(indices),
                "eval_horizon": 1,
                "num_eval_samples": int(kwargs["num_eval_samples"]),
                "realized_nfe": int(kwargs["runtime_nfe"]),
                "chosen_examples_hash": f"{len(ds)}:{len(indices)}",
                "evaluation_protocol_hash": f"proto:{len(ds)}:{len(indices)}",
            }

        prediction = {
            "solver_key": "euler",
            "target_nfe": 4,
            "runtime_nfe": 4,
            "scheduler_key": "unit_schedule",
            "schedule_name": "Unit schedule",
            "time_grid": [0.0, 1.0],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = {
                "model": object(),
                "cfg": object(),
                "splits": {"train": FakeDataset(10_000), "val": FakeDataset(100)},
                "checkpoint_id": "unit-checkpoint",
                "checkpoint_path": str(Path(tmpdir) / "checkpoint.pt"),
                "backbone_name": "otflow",
                "train_steps": 1,
                "train_budget_label": "unit",
            }
            args = build_argparser().parse_args(
                [
                    "--stage",
                    "evaluate-pooled",
                    "--schedule_summary",
                    str(Path(tmpdir) / "schedules.json"),
                    "--out_dir",
                    tmpdir,
                    "--row_csv_name",
                    "rows.csv",
                    "--calibration_seeds",
                    "0",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--eval_train_fraction",
                    "0.20",
                    "--train_tuning_seed",
                    "3",
                    "--train_tuning_train_split_fraction",
                    "0.70",
                    "--train_tuning_val_split_fraction",
                    "0.10",
                    "--num_eval_samples",
                    "2",
                    "--device",
                    "cpu",
                ]
            )
            with mock.patch(
                "genode.conditional_opd.train20_v43_pooled_calibration.load_schedule_predictions",
                return_value={("unit_schedule", "euler", 4): prediction},
            ), mock.patch(
                "genode.conditional_opd.train20_v43_pooled_calibration.load_forecast_checkpoint_splits",
                return_value=checkpoint,
            ), mock.patch(
                "genode.conditional_opd.train20_v43_pooled_calibration.evaluate_forecast_schedule",
                side_effect=fake_eval,
            ):
                summary = evaluate_pooled_schedule_summary(args)

            with (Path(tmpdir) / "rows.csv").open("r", newline="", encoding="utf-8") as fh:
                row = next(csv.DictReader(fh))

        train_call = next(call for call in calls if call["label"].endswith(" train"))
        val_call = next(call for call in calls if call["label"].endswith(" validation"))
        self.assertEqual(len(train_call["indices"]), 140)
        self.assertEqual(len(val_call["indices"]), 20)
        self.assertLess(len(train_call["indices"]), int(10_000 * 0.20))
        self.assertTrue(all(0 <= idx < 10_000 for idx in train_call["indices"]))
        self.assertEqual(summary["normalized_train_reference_examples"], 700)
        self.assertEqual(summary["validation_reference_examples"], 100)
        self.assertEqual(summary["combined_reference_examples"], 800)
        self.assertEqual(int(row["train_tuning_reference_examples"]), 800)
        self.assertEqual(int(row["train_tuning_target_examples"]), 160)
        self.assertEqual(json.loads(row["calibration_origin_counts_json"]), {"train": 140, "validation": 20})

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
        self.assertEqual(pooled["calibration_origin_counts"], {"train": 1, "validation": 3})
        self.assertAlmostEqual(pooled["crps"], 3.5)
        self.assertAlmostEqual(pooled["mase"], 10.5)

    def test_clean_rows_rejects_unstamped_pooled_rows(self) -> None:
        unstamped_row = {
            "split_phase": POOLED_CALIBRATION_PHASE,
            "dataset": "unit",
            "seed": 0,
            "target_nfe": 4,
            "solver_key": "euler",
            "scheduler_key": "uniform",
            "crps": 1.0,
            "mase": 1.0,
            "calibration_pool": "unstamped_full_split_pool",
        }
        with self.assertRaises(ValueError):
            _clean_pooled_rows(
                [unstamped_row],
                dataset="unit",
                solvers=("euler",),
                target_nfes=(4,),
                allowed_schedules=("uniform",),
                seeds=(0,),
                label="unstamped",
            )

        canonical_row = dict(unstamped_row)
        canonical_row["calibration_pool"] = V43_POOLED_CALIBRATION_POOL
        canonical_row["calibration_protocol"] = V43_POOLED_CALIBRATION_PROTOCOL
        self.assertEqual(
            len(
                _clean_pooled_rows(
                    [canonical_row],
                    dataset="unit",
                    solvers=("euler",),
                    target_nfes=(4,),
                    allowed_schedules=("uniform",),
                    seeds=(0,),
                    label="canonical",
                )
            ),
            1,
        )

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
                ]
            )
            summary = run_train20_v43_pooled_calibration(args)
        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["bo_rounds"], 1)
        self.assertEqual(summary["bo_candidate_count"], 32)
        self.assertFalse(summary["uses_validation_selection"])
        self.assertEqual(summary["student_initialization"], SER_PTG_SCHEDULE_KEY)
        self.assertEqual(summary["final_selector"], "pooled_teacher_utility_with_hard_geometry_guard")
        self.assertFalse(summary["strict_v43_protocol"])
        flat = " ".join(" ".join(cmd) for cmd in summary["commands"])
        self.assertIn("evaluate-pooled", flat)
        self.assertIn("--dataset san_francisco_traffic", flat)

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
        self.assertFalse(summary["lowest_internal_loss_selector_used"])
        self.assertEqual(summary["student_initialization"], SER_PTG_SCHEDULE_KEY)


if __name__ == "__main__":
    unittest.main()
