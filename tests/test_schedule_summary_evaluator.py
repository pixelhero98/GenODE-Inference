from __future__ import annotations

import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from genode.gipo.evaluate_schedule_summary import (
    SELECTED_STUDENT_SCHEDULE_KEY,
    _protocol_hash,
    build_argparser,
    build_comparison_summary,
    evaluate_schedule_summary,
    load_schedule_predictions,
    select_best_validation_schedule,
)
from genode.gipo.ser_ptg_reference import SER_PTG_SCHEDULE_KEY
from genode.evaluation.otflow_evaluation_support import (
    TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
    choose_forecast_train_tuning_indices,
)
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS


def _uniform_grid(n_steps: int) -> list[float]:
    return [float(idx) / float(n_steps) for idx in range(n_steps + 1)]


class ScheduleSummaryEvaluatorTests(unittest.TestCase):
    def test_train_tuning_hash_sampling_is_deterministic_and_stratified(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 100

        first = choose_forecast_train_tuning_indices(FakeDataset(), fraction=0.20, seed=7, strata=20, dataset="sf")
        second = choose_forecast_train_tuning_indices(FakeDataset(), fraction=0.20, seed=7, strata=20, dataset="sf")
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 20)
        self.assertEqual(len({int(idx) // 5 for idx in first.tolist()}), 20)

    def test_validation_normalized_train_tuning_sampling_uses_holdout_scale(self) -> None:
        class FakeTrainDataset:
            def __len__(self) -> int:
                return 14_399_710

        first = choose_forecast_train_tuning_indices(
            FakeTrainDataset(),
            fraction=0.20,
            seed=7,
            strata=20,
            dataset="traffic_hourly",
            sampling_mode=TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
            reference_examples=862,
            train_split_fraction=0.70,
            val_split_fraction=0.10,
        )
        second = choose_forecast_train_tuning_indices(
            FakeTrainDataset(),
            fraction=0.20,
            seed=7,
            strata=20,
            dataset="traffic_hourly",
            sampling_mode=TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
            reference_examples=862,
            train_split_fraction=0.70,
            val_split_fraction=0.10,
        )
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 1207)
        self.assertEqual(len({int(idx) * 20 // 14_399_710 for idx in first.tolist()}), 20)

    def test_schedule_evaluator_protocol_tracks_train_tuning_sampling_mode(self) -> None:
        base = [
            "--dataset",
            "traffic_hourly",
            "--schedule_summary",
            "dummy.json",
            "--split_phase",
            "train_tuning",
            "--seeds",
            "0",
            "--solver_names",
            "euler",
            "--target_nfe_values",
            "4",
            "--device",
            "cpu",
        ]
        legacy = build_argparser().parse_args([*base, "--train_tuning_sampling_mode", "train_window_fraction"])
        valnorm = build_argparser().parse_args([*base, "--train_tuning_sampling_mode", "validation_normalized"])
        valnorm_alt_fraction = build_argparser().parse_args(
            [*base, "--train_tuning_sampling_mode", "validation_normalized", "--train_tuning_train_split_fraction", "0.60"]
        )
        self.assertNotEqual(_protocol_hash(legacy), _protocol_hash(valnorm))
        self.assertNotEqual(_protocol_hash(valnorm), _protocol_hash(valnorm_alt_fraction))

    def test_load_schedule_predictions_validates_ser_ptg_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ser_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset": "traffic_hourly",
                        "schedules": [
                            {
                                "scheduler_key": SER_PTG_SCHEDULE_KEY,
                                "schedule_name": "SER-PTG local defect eta=0.05",
                                "predictions": [
                                    {
                                        "solver_key": "heun",
                                        "target_nfe": 4,
                                        "macro_steps": 2,
                                        "time_grid": _uniform_grid(2),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            predictions = load_schedule_predictions(
                path,
                dataset="traffic_hourly",
                solver_names=("heun",),
                target_nfe_values=(4,),
            )
        self.assertIn((SER_PTG_SCHEDULE_KEY, "heun", 4), predictions)
        self.assertEqual(predictions[(SER_PTG_SCHEDULE_KEY, "heun", 4)]["realized_nfe"], 4)
        self.assertNotIn(SER_PTG_SCHEDULE_KEY, BASELINE_SCHEDULE_KEYS)

    def test_load_schedule_predictions_rejects_empty_filtered_candidate_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "student_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset": "traffic_hourly",
                        "schedules": [
                            {
                                "scheduler_key": "gipo_candidate_steps20",
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing predictions"):
                load_schedule_predictions(
                    path,
                    dataset="traffic_hourly",
                    solver_names=("heun",),
                    target_nfe_values=(4,),
                    require_complete=True,
                )

    def test_comparison_summary_keeps_ser_ptg_as_comparator(self) -> None:
        baseline_rows = [
            {
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "crps": 2.0,
                "mase": 3.0,
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]
        ser_rows = [{"seed": 0, "solver_key": "euler", "target_nfe": 4, "scheduler_key": SER_PTG_SCHEDULE_KEY, "crps": 1.5, "mase": 2.5}]
        student_rows = [
            {"seed": 0, "solver_key": "euler", "target_nfe": 4, "scheduler_key": SELECTED_STUDENT_SCHEDULE_KEY, "crps": 1.25, "mase": 2.0}
        ]
        summary = build_comparison_summary(
            baseline_rows=baseline_rows,
            comparator_rows=ser_rows,
            student_rows=student_rows,
            dataset="traffic_hourly",
            split_phase="locked_test",
            seeds=(0,),
            solver_names=("euler",),
            target_nfe_values=(4,),
        )
        self.assertFalse(summary["ser_ptg_is_baseline"])
        self.assertEqual(summary["observed_ser_ptg_rows"], 1)
        self.assertEqual(summary["observed_student_rows"], 1)
        ranking = summary["cell_rankings"][0]
        self.assertEqual(ranking["crps_ranking"][0], SELECTED_STUDENT_SCHEDULE_KEY)
        self.assertAlmostEqual(ranking["student_relative_crps_gain_vs_ser_ptg"], 1.0 - 1.25 / 1.5)
        self.assertEqual(ranking["student_comparisons"][0]["scheduler_key"], SELECTED_STUDENT_SCHEDULE_KEY)

    def test_comparison_summary_supports_multiple_student_schedules(self) -> None:
        baseline_rows = [
            {
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "crps": 2.0,
                "mase": 3.0,
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]
        student_rows = [
            {"seed": 0, "solver_key": "euler", "target_nfe": 4, "scheduler_key": "density_a", "crps": 1.5, "mase": 2.5},
            {"seed": 0, "solver_key": "euler", "target_nfe": 4, "scheduler_key": "density_b", "crps": 1.25, "mase": 2.25},
        ]
        summary = build_comparison_summary(
            baseline_rows=baseline_rows,
            comparator_rows=[],
            student_rows=student_rows,
            dataset="traffic_hourly",
            split_phase="validation_tuning",
            seeds=(0,),
            solver_names=("euler",),
            target_nfe_values=(4,),
        )
        self.assertEqual(summary["student_schedule_keys"], ["density_a", "density_b"])
        self.assertEqual(summary["expected_student_rows"], 2)
        self.assertEqual(summary["observed_student_rows"], 2)
        self.assertEqual(summary["missing_student_cells"], [])
        comparisons = summary["cell_rankings"][0]["student_comparisons"]
        self.assertEqual([row["scheduler_key"] for row in comparisons], ["density_a", "density_b"])
        self.assertAlmostEqual(comparisons[1]["student_relative_crps_gain_vs_best_baseline"], 1.0 - 1.25 / 2.0)

    def test_validation_schedule_selection_supports_arbitrary_candidate_keys(self) -> None:
        rows = []
        fixed_rows = []
        for schedule_key in BASELINE_SCHEDULE_KEYS:
            fixed_rows.append(
                {
                    "seed": 0,
                    "solver_key": "euler",
                    "target_nfe": 4,
                    "scheduler_key": schedule_key,
                    "crps": 1.0,
                    "mase": 2.0,
                }
            )
        for schedule_key, budget, crps, mase in (
            ("gipo_candidate_steps20", 20, 1.0, 2.0),
            ("gipo_candidate_steps25", 25, 1.0, 2.0),
            ("gipo_candidate_steps35", 35, 0.9, 1.9),
            ("ser_ptg_residual_tail_s200_eps030", None, 1.2, 2.2),
        ):
            for seed in (0, 1, 2):
                row = {
                    "seed": seed,
                    "solver_key": "euler",
                    "target_nfe": 4,
                    "scheduler_key": schedule_key,
                    "crps": crps,
                    "mase": mase,
                }
                if budget is not None:
                    row["gipo_step_budget"] = budget
                rows.append(row)
        selection = select_best_validation_schedule(rows, reference_rows=fixed_rows)
        self.assertEqual(selection["selection_unit"], "generated_schedule_key")
        self.assertEqual(selection["selected_schedule_key"], "gipo_candidate_steps35")
        self.assertEqual(selection["selected_gipo_step_budget"], 35)
        self.assertEqual(selection["utility_reference"], "best_fixed_baseline_crps_mase")
        self.assertTrue(any(row["scheduler_key"] == "ser_ptg_residual_tail_s200_eps030" for row in selection["schedule_table"]))
        self.assertNotIn("eps_rho", selection["schedule_table"][0])
        self.assertNotIn("kl_weight", selection["schedule_table"][0])

    def test_validation_schedule_selection_tie_breaks_smaller_budget(self) -> None:
        rows = []
        fixed_rows = [
            {
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "crps": 2.0,
                "mase": 3.0,
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]
        for schedule_key, budget in (("gipo_candidate_steps20", 20), ("gipo_candidate_steps25", 25)):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "seed": seed,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "scheduler_key": schedule_key,
                        "gipo_step_budget": budget,
                        "crps": 1.0,
                        "mase": 2.0,
                    }
                )
        selection = select_best_validation_schedule(rows, reference_rows=fixed_rows)
        self.assertEqual(selection["selected_schedule_key"], "gipo_candidate_steps20")
        self.assertEqual(
            selection["tie_break"],
            "mean_validation_utility_then_mean_min_metric_utility_then_smaller_gipo_step_budget_then_scheduler_key",
        )
        self.assertIn("mean_min_metric_utility", selection["schedule_table"][0])

    def test_validation_schedule_selection_tie_breaks_worst_metric_before_budget(self) -> None:
        rows = []
        fixed_rows = [
            {
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "crps": 1.0,
                "mase": 1.0,
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]
        # Same composite utility relative to best fixed: 0.5*(+log 2 - log 2) == 0.
        # The 25-step schedule has a better worst-metric utility and should win
        # before the smaller-budget tie-break is considered.
        for schedule_key, budget, crps, mase in (
            ("gipo_candidate_steps20", 20, 0.5, 2.0),
            ("gipo_candidate_steps25", 25, 0.8, 1.25),
        ):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "seed": seed,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "scheduler_key": schedule_key,
                        "gipo_step_budget": budget,
                        "crps": crps,
                        "mase": mase,
                    }
                )
        selection = select_best_validation_schedule(rows, reference_rows=fixed_rows)
        self.assertEqual(selection["selected_schedule_key"], "gipo_candidate_steps25")
        self.assertGreater(selection["schedule_table"][0]["mean_min_metric_utility"], selection["schedule_table"][1]["mean_min_metric_utility"])

    def test_validation_schedule_selection_requires_fixed_reference_rows(self) -> None:
        rows = []
        for schedule_key, budget, crps, mase in (
            ("gipo_candidate_steps20", 20, 1.0, 2.0),
            ("gipo_candidate_steps25", 25, 0.9, 1.9),
        ):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "seed": seed,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "scheduler_key": schedule_key,
                        "gipo_step_budget": budget,
                        "crps": crps,
                        "mase": mase,
                    }
                )
        with self.assertRaisesRegex(ValueError, "fixed baseline reference rows"):
            select_best_validation_schedule(rows)

    def test_budget_only_validation_cli_is_removed(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_argparser().parse_args(
                [
                    "--dataset",
                    "traffic_hourly",
                    "--schedule_summary",
                    "summary.json",
                    "--split_phase",
                    "validation_tuning",
                    "--select_budget_from_validation",
                ]
            )

    def test_evaluate_schedule_summary_writes_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            schedule_path = root / "student_summary.json"
            schedule_path.write_text(
                json.dumps(
                    {
                        "dataset": "traffic_hourly",
                        "schedules": [
                            {
                                "scheduler_key": SELECTED_STUDENT_SCHEDULE_KEY,
                                "schedule_name": "GIPO Student Selected",
                                "gipo_step_budget": 25,
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            class FakeDataset:
                def __len__(self) -> int:
                    return 3

            fake_checkpoint = {
                "model": object(),
                "cfg": object(),
                "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
                "checkpoint_path": root / "model.pt",
                "checkpoint_id": "ck",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
            }
            args = build_argparser().parse_args(
                [
                    "--dataset",
                    "traffic_hourly",
                    "--schedule_summary",
                    str(schedule_path),
                    "--split_phase",
                    "validation_tuning",
                    "--out_dir",
                    str(root / "out"),
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--num_eval_samples",
                    "1",
                    "--eval_windows_val",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            with mock.patch(
                "genode.gipo.evaluate_schedule_summary.load_forecast_checkpoint_splits",
                return_value=fake_checkpoint,
            ), mock.patch(
                "genode.gipo.evaluate_schedule_summary.evaluate_forecast_schedule",
                return_value={
                    "crps": 1.0,
                    "mse": 1.5,
                    "mase": 2.0,
                    "latency_ms_per_sample": 0.25,
                    "num_eval_samples": 1,
                    "eval_examples": 1,
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": 4,
                },
            ):
                summary = evaluate_schedule_summary(args)
            self.assertEqual(summary["observed_rows"], 1)
            with (root / "out" / "validation_rows.csv").open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scheduler_key"], SELECTED_STUDENT_SCHEDULE_KEY)
            self.assertEqual(int(rows[0]["realized_nfe"]), 4)

    def test_evaluate_schedule_summary_writes_train_tuning_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            schedule_path = root / "student_summary.json"
            schedule_path.write_text(
                json.dumps(
                    {
                        "dataset": "traffic_hourly",
                        "schedules": [
                            {
                                "scheduler_key": SELECTED_STUDENT_SCHEDULE_KEY,
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            class FakeDataset:
                def __len__(self) -> int:
                    return 100

            fake_checkpoint = {
                "model": object(),
                "cfg": object(),
                "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
                "checkpoint_path": root / "model.pt",
                "checkpoint_id": "ck",
                "backbone_name": "otflow",
                "train_steps": 20000,
                "train_budget_label": "20k",
            }
            args = build_argparser().parse_args(
                [
                    "--dataset",
                    "traffic_hourly",
                    "--schedule_summary",
                    str(schedule_path),
                    "--split_phase",
                    "train_tuning",
                    "--out_dir",
                    str(root / "out"),
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--num_eval_samples",
                    "1",
                    "--eval_train_fraction",
                    "0.20",
                    "--train_tuning_strata",
                    "20",
                    "--device",
                    "cpu",
                ]
            )
            with mock.patch(
                "genode.gipo.evaluate_schedule_summary.load_forecast_checkpoint_splits",
                return_value=fake_checkpoint,
            ), mock.patch(
                "genode.gipo.evaluate_schedule_summary.evaluate_forecast_schedule",
                return_value={
                    "crps": 1.0,
                    "mse": 1.5,
                    "mase": 2.0,
                    "latency_ms_per_sample": 0.25,
                    "num_eval_samples": 1,
                    "eval_examples": 20,
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": 4,
                },
            ):
                summary = evaluate_schedule_summary(args)
            self.assertEqual(summary["split_phase"], "train_tuning")
            self.assertEqual(summary["train_tuning"]["fraction"], 0.20)
            with (root / "out" / "train_tuning_rows.csv").open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["split_phase"], "train_tuning")
            self.assertEqual(rows[0]["train_tuning_sampler"], "temporal_stratified_hash")
            self.assertEqual(rows[0]["train_tuning_sampling_mode"], "train_window_fraction")


if __name__ == "__main__":
    unittest.main()
