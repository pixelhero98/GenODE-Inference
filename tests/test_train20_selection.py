from __future__ import annotations

import tempfile
import unittest

from genode.conditional_opd.legacy.train20_selection import (
    _validate_baseline_validation_rows,
    build_argparser,
    run_train20_expanded_opd_selection,
    select_guarded_validation_schedule,
)
from genode.evaluation.otflow_evaluation_support import VALIDATION_PHASE
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS


class Train20SelectionTests(unittest.TestCase):
    def test_active_round_dry_run_plans_two_rounds_without_execution_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_argparser().parse_args(
                [
                    "--out_dir",
                    tmpdir,
                    "--active_rounds",
                    "2",
                    "--direct_student_seeds",
                    "0,1",
                    "--student_opd_step_values",
                    "5,10,15,20,25",
                    "--device",
                    "cpu",
                    "--eval_train_fraction",
                    "0.10",
                    "--skip_locked_test",
                ]
            )
            summary = run_train20_expanded_opd_selection(args)
            self.assertEqual(args.ser_test_rows, "")
            self.assertEqual(args.train_tuning_sampling_mode, "validation_normalized")
        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["active_rounds"], 2)
        self.assertEqual(summary["direct_student_seeds"], [0, 1])
        self.assertEqual(summary["student_opd_step_values"], [5, 10, 15, 20, 25])
        self.assertEqual(len(summary["rounds"]), 2)
        flat = [str(part) for command in summary["commands"] for part in command]
        self.assertNotIn("--allow_execute", flat)
        self.assertNotIn("ser_ptg_local_defect_full", " ".join(flat))
        self.assertIn("genode.conditional_opd.candidate_pool", flat)
        self.assertIn("genode.conditional_opd.ser_ptg_reference", flat)
        self.assertIn("--teacher_checkpoint_paths", flat)
        baseline_command = next(command for command in summary["commands"] if "genode.evaluation.diffusion_flow_time_reparameterization" in command)
        scheduler_idx = baseline_command.index("--baseline_scheduler_names")
        self.assertEqual(baseline_command[scheduler_idx + 1], "uniform,late_power_3")
        fraction_idx = baseline_command.index("--eval_train_fraction")
        self.assertEqual(baseline_command[fraction_idx + 1], "0.1")
        self.assertIn("--teacher_fixed_schedule_keys", flat)
        self.assertNotIn("--reference_rows_csv", flat)
        self.assertNotIn("ser_ptg_train_tuning/train_tuning_rows.csv", " ".join(flat))
        self.assertEqual(summary["teacher_fixed_schedule_keys"], ["uniform", "late_power_3"])
        self.assertEqual(summary["ser_ptg_usage"], "student_initialization_only")

    def test_validation_normalized_train20_dry_run_propagates_sampling_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_argparser().parse_args(
                [
                    "--out_dir",
                    tmpdir,
                    "--active_rounds",
                    "1",
                    "--direct_student_seeds",
                    "0",
                    "--device",
                    "cpu",
                    "--eval_train_fraction",
                    "0.20",
                    "--train_tuning_sampling_mode",
                    "validation_normalized",
                    "--train_tuning_train_split_fraction",
                    "0.70",
                    "--train_tuning_val_split_fraction",
                    "0.10",
                    "--skip_locked_test",
                ]
            )
            summary = run_train20_expanded_opd_selection(args)
        flat = [str(part) for command in summary["commands"] for part in command]
        self.assertEqual(summary["teacher_train_fraction"], 0.20)
        self.assertEqual(summary["teacher_train_sampling"], "temporal_stratified_validation_normalized_v2")
        self.assertEqual(summary["train_tuning_sampling_mode"], "validation_normalized")
        self.assertIn("--train_tuning_sampling_mode", flat)
        self.assertIn("validation_normalized", flat)
        self.assertIn("--expected_train_tuning_sampler", flat)
        self.assertIn("temporal_stratified_validation_normalized_v2", flat)
        self.assertIn("--expected_train_tuning_fraction", flat)
        self.assertIn("0.2", flat)

    def test_train20_dry_run_propagates_validation_reference_and_ser_regularizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_argparser().parse_args(
                [
                    "--out_dir",
                    tmpdir,
                    "--active_rounds",
                    "1",
                    "--direct_student_seeds",
                    "0",
                    "--device",
                    "cpu",
                    "--eval_train_fraction",
                    "0.20",
                    "--baseline_validation_rows",
                    "outputs/diffusion_flow_time_reparameterization_full_val/rows.csv",
                    "--student_ser_ptg_regularizer",
                    "js",
                    "--student_ser_ptg_regularization_weight",
                    "0.02",
                    "--skip_locked_test",
                ]
            )
            summary = run_train20_expanded_opd_selection(args)
        flat = [str(part) for command in summary["commands"] for part in command]
        self.assertEqual(summary["baseline_validation_rows"], "outputs/diffusion_flow_time_reparameterization_full_val/rows.csv")
        self.assertEqual(summary["ser_ptg_usage"], "student_initialization_and_regularization")
        self.assertEqual(summary["student_ser_ptg_regularization"]["mode"], "js")
        self.assertEqual(summary["student_ser_ptg_regularization"]["weight"], 0.02)
        self.assertIn("--student_ser_ptg_regularizer", flat)
        self.assertIn("js", flat)
        self.assertIn("--student_ser_ptg_regularization_weight", flat)
        self.assertIn("0.02", flat)

    def test_guarded_validation_selection_uses_fixed_reference_rows(self) -> None:
        candidate_rows = []
        for schedule_key, budget, crps, mase in (
            ("conditional_opd_student_steps20", 20, 1.20, 2.20),
            ("conditional_opd_student_steps25", 25, 0.95, 1.95),
        ):
            for seed in (0, 1, 2):
                candidate_rows.append(
                    {
                        "seed": seed,
                        "dataset": "san_francisco_traffic",
                        "split_phase": VALIDATION_PHASE,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "scheduler_key": schedule_key,
                        "opd_step_budget": budget,
                        "crps": crps,
                        "mase": mase,
                        "chosen_examples_hash": "hash-a",
                        "internal_fraction_after_098": 0.0,
                        "min_interval": 0.1,
                        "max_interval": 0.4,
                    }
                )
        fixed_rows = [
            {
                "seed": seed,
                "dataset": "san_francisco_traffic",
                "split_phase": VALIDATION_PHASE,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "crps": 1.0,
                "mase": 2.0,
                "chosen_examples_hash": "hash-a",
            }
            for seed in (0, 1, 2)
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]
        selection = select_guarded_validation_schedule(candidate_rows, reference_rows=fixed_rows)
        self.assertEqual(selection["utility_reference"], "best_fixed_baseline_crps_mase")
        self.assertEqual(selection["fixed_reference_schedule_keys"], list(BASELINE_SCHEDULE_KEYS))
        self.assertEqual(selection["selected_schedule_key"], "conditional_opd_student_steps25")

    def test_baseline_validation_reference_validation_rejects_empty_and_mismatched_hashes(self) -> None:
        candidate_rows = [
            {
                "seed": 0,
                "dataset": "san_francisco_traffic",
                "split_phase": VALIDATION_PHASE,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "conditional_opd_student_steps25",
                "chosen_examples_hash": "candidate-hash",
            }
        ]
        with self.assertRaisesRegex(ValueError, "produced no rows"):
            _validate_baseline_validation_rows(
                [],
                candidate_rows=candidate_rows,
                dataset="san_francisco_traffic",
                seeds=(0,),
                solver_names=("euler",),
                target_nfe_values=(4,),
            )
        fixed_rows = [
            {
                "seed": 0,
                "dataset": "san_francisco_traffic",
                "split_phase": VALIDATION_PHASE,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "chosen_examples_hash": "reference-hash",
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]
        with self.assertRaisesRegex(ValueError, "chosen_examples_hash"):
            _validate_baseline_validation_rows(
                fixed_rows,
                candidate_rows=candidate_rows,
                dataset="san_francisco_traffic",
                seeds=(0,),
                solver_names=("euler",),
                target_nfe_values=(4,),
            )
        missing_hash_candidate_rows = [dict(candidate_rows[0])]
        missing_hash_candidate_rows[0]["chosen_examples_hash"] = ""
        matching_fixed_rows = [dict(row) for row in fixed_rows]
        for row in matching_fixed_rows:
            row["chosen_examples_hash"] = "candidate-hash"
        with self.assertRaisesRegex(ValueError, "candidate rows are missing chosen_examples_hash"):
            _validate_baseline_validation_rows(
                matching_fixed_rows,
                candidate_rows=missing_hash_candidate_rows,
                dataset="san_francisco_traffic",
                seeds=(0,),
                solver_names=("euler",),
                target_nfe_values=(4,),
            )

    def test_geometry_guard_rejects_missing_candidate_geometry_metrics(self) -> None:
        rows = [
            {
                "seed": seed,
                "dataset": "san_francisco_traffic",
                "split_phase": VALIDATION_PHASE,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "conditional_opd_student_steps25",
                "opd_step_budget": 25,
                "crps": 1.0,
                "mase": 2.0,
            }
            for seed in (0, 1, 2)
        ]
        fixed_rows = [
            {
                "seed": seed,
                "dataset": "san_francisco_traffic",
                "split_phase": VALIDATION_PHASE,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "crps": 1.1,
                "mase": 2.1,
            }
            for seed in (0, 1, 2)
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]
        with self.assertRaisesRegex(ValueError, "No validation candidate passed the geometry guard"):
            select_guarded_validation_schedule(rows, reference_rows=fixed_rows)

    def test_locked_test_does_not_auto_run_ser_ptg_comparator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_argparser().parse_args(
                [
                    "--out_dir",
                    tmpdir,
                    "--active_rounds",
                    "0",
                    "--direct_student_seeds",
                    "0",
                    "--device",
                    "cpu",
                ]
            )
            summary = run_train20_expanded_opd_selection(args)
        flat = [str(part) for command in summary["commands"] for part in command]
        self.assertEqual(args.ser_test_rows, "")
        self.assertNotIn("--comparator_rows", flat)
        self.assertNotIn("ser_ptg_train_tuning_locked_test", " ".join(flat))



if __name__ == "__main__":
    unittest.main()
