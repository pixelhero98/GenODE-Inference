from __future__ import annotations

import inspect
import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from genode.conditional_opd.models import (
    ScheduleStudentMLP,
    ScheduleTeacherMLP,
    count_parameters,
    setting_features,
    solver_macro_steps,
    validate_time_grid,
)
from genode.conditional_opd.objectives import (
    attach_reward_columns,
    best_fixed_references_by_setting,
    build_fixed_reference_table,
    crps_mase_reward,
    rewards_by_setting,
    seed_mean_metric_rows,
)
from genode.conditional_opd.ser_ptg_reference import SER_PTG_SCHEDULE_KEY, ser_ptg_grid_from_trace
from genode.conditional_opd.train_conditional_opd import (
    _assert_expected_train_tuning_metadata,
    build_argparser,
    optimize_student_with_teacher,
    ser_ptg_interval_divergence,
    split_teacher_diagnostic_holdout,
    teacher_schedule_weights,
    train_conditional_opd,
)
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS, build_schedule_grid


def _uniform_grid(n_steps: int) -> list[float]:
    return [float(idx) / float(n_steps) for idx in range(n_steps + 1)]


class ConditionalOPDPolicyTests(unittest.TestCase):
    def test_baseline_registry_is_section_15_set(self) -> None:
        self.assertEqual(BASELINE_SCHEDULE_KEYS, ("uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots"))

    def test_solver_macro_steps_match_target_nfes(self) -> None:
        expected = {
            "euler": (4, 8, 12),
            "heun": (2, 4, 6),
            "midpoint_rk2": (2, 4, 6),
            "dpmpp2m": (4, 8, 12),
        }
        for solver, macro_steps in expected.items():
            self.assertEqual(tuple(solver_macro_steps(solver, nfe) for nfe in (4, 8, 12)), macro_steps)

    def test_schedule_grids_are_valid_for_all_baselines_solver_nfes(self) -> None:
        for solver in ("euler", "heun", "midpoint_rk2", "dpmpp2m"):
            for target_nfe in (4, 8, 12):
                macro_steps = solver_macro_steps(solver, target_nfe)
                for schedule_key in BASELINE_SCHEDULE_KEYS:
                    with self.subTest(solver=solver, target_nfe=target_nfe, schedule_key=schedule_key):
                        grid = build_schedule_grid(schedule_key, macro_steps)
                        self.assertIsNotNone(grid)
                        assert grid is not None
                        self.assertEqual(validate_time_grid(grid, macro_steps=macro_steps), tuple(grid))

    def test_teacher_and_student_mlp_contract(self) -> None:
        setting_dim = int(setting_features("euler", 4).numel())
        teacher = ScheduleTeacherMLP(setting_dim + 12, hidden_dim=256, hidden_layers=3)
        student = ScheduleStudentMLP(setting_dim, max_macro_steps=12, hidden_dim=128, hidden_layers=2)
        self.assertLess(count_parameters(student), count_parameters(teacher))
        grid = student.time_grid(setting_features("heun", 8)[None, :], macro_steps=4)
        self.assertEqual(tuple(grid.shape), (1, 5))
        self.assertTrue(torch.all(torch.diff(grid, dim=-1) > 0.0))
        self.assertTrue(torch.allclose(grid[:, 0], torch.zeros(1)))
        self.assertTrue(torch.allclose(grid[:, -1], torch.ones(1)))

    def test_reward_uses_crps_and_mase_without_soft_regularizers(self) -> None:
        reward = crps_mase_reward(2.0, 4.0, crps_center=4.0, mase_center=4.0)
        self.assertGreater(reward, 0.0)
        source = inspect.getsource(crps_mase_reward)
        self.assertIn("crps", source)
        self.assertIn("mase", source)
        self.assertNotIn("uncertainty", source)
        self.assertNotIn("soft_penalty", source)
        self.assertNotIn("spacing", source)
        self.assertNotIn("smoothness", source)

    def test_reward_scaling_uses_best_fixed_baseline_reference(self) -> None:
        rows = [
            {"solver_key": "euler", "target_nfe": 4, "scheduler_key": "uniform", "seed": 0, "crps": 2.0, "mase": 3.0},
            {"solver_key": "euler", "target_nfe": 4, "scheduler_key": "ays", "seed": 0, "crps": 1.5, "mase": 2.5},
            {"solver_key": "euler", "target_nfe": 4, "scheduler_key": "learned_a", "seed": 0, "crps": 1.5, "mase": 2.5},
        ]
        refs = best_fixed_references_by_setting(rows, fixed_schedule_keys=BASELINE_SCHEDULE_KEYS)
        self.assertEqual(refs[("euler", 4)]["crps"], 1.5)
        self.assertEqual(refs[("euler", 4)]["mase"], 2.5)
        rewards = rewards_by_setting(rows, fixed_schedule_keys=BASELINE_SCHEDULE_KEYS)
        self.assertAlmostEqual(rewards[("euler", 4)]["ays"], 0.0)
        self.assertAlmostEqual(rewards[("euler", 4)]["learned_a"], 0.0)

    def test_reward_columns_materialize_best_fixed_and_uniform_diagnostics(self) -> None:
        rows = [
            {"solver_key": "euler", "target_nfe": 4, "scheduler_key": "uniform", "seed": 0, "crps": 2.0, "mase": 4.0},
            {"solver_key": "euler", "target_nfe": 4, "scheduler_key": "ays", "seed": 0, "crps": 1.0, "mase": 2.0},
            {"solver_key": "euler", "target_nfe": 4, "scheduler_key": "learned_a", "seed": 0, "crps": 0.5, "mase": 3.0},
        ]
        table = build_fixed_reference_table(rows, fixed_schedule_keys=BASELINE_SCHEDULE_KEYS)
        self.assertEqual(table[("euler", 4)]["best_fixed_crps"], 1.0)
        self.assertEqual(table[("euler", 4)]["best_fixed_mase"], 2.0)
        self.assertEqual(table[("euler", 4)]["uniform_crps"], 2.0)
        annotated = attach_reward_columns(rows, fixed_schedule_keys=BASELINE_SCHEDULE_KEYS)
        learned = [row for row in annotated if row["scheduler_key"] == "learned_a"][0]
        self.assertGreater(float(learned["u_crps_best"]), 0.0)
        self.assertLess(float(learned["u_mase_best"]), 0.0)
        self.assertAlmostEqual(float(learned["u_comp_best"]), float(learned["teacher_reward"]) if "teacher_reward" in learned else float(learned["u_comp_best"]))
        self.assertGreater(float(learned["u_comp_uniform"]), 0.0)

    def test_seed_mean_aggregation_is_row_order_independent(self) -> None:
        rows = [
            {"solver_key": "euler", "target_nfe": 4, "scheduler_key": "uniform", "seed": 2, "crps": 9.0, "mase": 5.0},
            {"solver_key": "euler", "target_nfe": 4, "scheduler_key": "uniform", "seed": 0, "crps": 3.0, "mase": 1.0},
            {"solver_key": "euler", "target_nfe": 4, "scheduler_key": "uniform", "seed": 1, "crps": 6.0, "mase": 3.0},
        ]
        forward = seed_mean_metric_rows(rows)
        backward = seed_mean_metric_rows(list(reversed(rows)))
        self.assertEqual(len(forward), 1)
        self.assertEqual(forward, backward)
        self.assertAlmostEqual(forward[0]["crps"], 6.0)
        self.assertAlmostEqual(forward[0]["mase"], 3.0)
        self.assertEqual(forward[0]["seed_values"], [0, 1, 2])

    def test_teacher_diagnostic_holdout_keeps_rows_train_only_but_out_of_fit(self) -> None:
        rows = [
            {"solver_key": "euler", "target_nfe": 4, "scheduler_key": f"s{i}", "crps": 1.0, "mase": 1.0}
            for i in range(10)
        ]
        fit_rows, diagnostic_rows = split_teacher_diagnostic_holdout(rows, fraction=0.2, seed=3)
        self.assertEqual(len(fit_rows), 8)
        self.assertEqual(len(diagnostic_rows), 2)
        fit_keys = {row["scheduler_key"] for row in fit_rows}
        diagnostic_keys = {row["scheduler_key"] for row in diagnostic_rows}
        self.assertFalse(fit_keys.intersection(diagnostic_keys))

    def test_teacher_diagnostic_holdout_applies_only_to_bo_rows(self) -> None:
        rows = [
            {"solver_key": "euler", "target_nfe": 4, "scheduler_key": schedule_key, "crps": 1.0, "mase": 1.0}
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]
        rows.extend(
            {
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": f"train20_v43_bo_r0_cand{i:03d}",
                "candidate_source": "bayesian_optimization",
                "active_round": i % 2,
                "crps": 1.0,
                "mase": 1.0,
            }
            for i in range(8)
        )
        rewards = {("euler", 4): {f"train20_v43_bo_r0_cand{i:03d}": float(i) for i in range(8)}}
        fit_rows, diagnostic_rows = split_teacher_diagnostic_holdout(
            rows,
            fraction=0.2,
            seed=3,
            fixed_schedule_keys=BASELINE_SCHEDULE_KEYS,
            rewards=rewards,
        )
        self.assertEqual(len(diagnostic_rows), 2)
        self.assertTrue(all(row["scheduler_key"] not in BASELINE_SCHEDULE_KEYS for row in diagnostic_rows))
        self.assertTrue(set(BASELINE_SCHEDULE_KEYS).issubset({row["scheduler_key"] for row in fit_rows}))

    def test_ser_ptg_grid_valid_and_not_section_15_baseline(self) -> None:
        self.assertNotIn(SER_PTG_SCHEDULE_KEY, BASELINE_SCHEDULE_KEYS)
        grid = ser_ptg_grid_from_trace(
            [1.0, 4.0, 1.0, 2.0],
            [0.0, 0.25, 0.5, 0.75, 1.0],
            macro_steps=4,
            solver_order_p=2.0,
            density_floor_eta=0.05,
        )
        self.assertEqual(validate_time_grid(grid, macro_steps=4), tuple(grid))

    def test_student_opd_updates_call_teacher(self) -> None:
        class CountingTeacher(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                weights = torch.arange(1, 13, dtype=features.dtype, device=features.device)
                return (features[:, -12:] * weights).sum(dim=-1)

        student = ScheduleStudentMLP(int(setting_features("euler", 4).numel()), max_macro_steps=12, hidden_dim=16, hidden_layers=1)
        teacher = CountingTeacher()
        optimize_student_with_teacher(student, teacher, [("euler", 4)], max_macro_steps=12, steps=3, lr=1e-3)
        self.assertEqual(teacher.calls, 3)

    def test_ser_ptg_interval_js_regularizer_is_finite(self) -> None:
        student_intervals = torch.tensor([[0.20, 0.30, 0.25, 0.25]], dtype=torch.float32, requires_grad=True)
        reference_intervals = torch.tensor([0.10, 0.40, 0.30, 0.20], dtype=torch.float32)
        divergence = ser_ptg_interval_divergence(student_intervals, reference_intervals, mode="js", eps=1e-8)
        self.assertGreaterEqual(float(divergence.detach().cpu().item()), 0.0)
        divergence.backward()
        self.assertIsNotNone(student_intervals.grad)
        self.assertTrue(torch.isfinite(student_intervals.grad).all())

    def test_student_opd_ser_ptg_regularizer_weight_zero_preserves_teacher_calls(self) -> None:
        class CountingTeacher(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return features[:, -12:].sum(dim=-1)

        student = ScheduleStudentMLP(int(setting_features("euler", 4).numel()), max_macro_steps=12, hidden_dim=16, hidden_layers=1)
        teacher = CountingTeacher()
        losses = optimize_student_with_teacher(
            student,
            teacher,
            [("euler", 4)],
            max_macro_steps=12,
            steps=3,
            lr=1e-3,
            ser_ptg_reference_pairs=[("euler", 4, _uniform_grid(4))],
            ser_ptg_regularizer="js",
            ser_ptg_regularization_weight=0.0,
        )
        self.assertEqual(teacher.calls, 3)
        self.assertTrue(all("ser_ptg_regularization" not in row for row in losses))

    def test_student_opd_ser_ptg_regularizer_logs_finite_loss(self) -> None:
        class CountingTeacher(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return features[:, -12:].sum(dim=-1)

        student = ScheduleStudentMLP(int(setting_features("euler", 4).numel()), max_macro_steps=12, hidden_dim=16, hidden_layers=1)
        teacher = CountingTeacher()
        losses = optimize_student_with_teacher(
            student,
            teacher,
            [("euler", 4)],
            max_macro_steps=12,
            steps=2,
            lr=1e-3,
            ser_ptg_reference_pairs=[("euler", 4, _uniform_grid(4))],
            ser_ptg_regularizer="js",
            ser_ptg_regularization_weight=0.02,
        )
        self.assertTrue(losses)
        self.assertTrue(all(row["ser_ptg_regularizer"] == "js" for row in losses))
        self.assertTrue(all(math.isfinite(float(row["ser_ptg_regularization"])) for row in losses))
        self.assertTrue(all(math.isfinite(float(row["student_total_loss"])) for row in losses))

    def test_student_opd_ser_ptg_regularizer_requires_all_setting_targets(self) -> None:
        student = ScheduleStudentMLP(int(setting_features("euler", 4).numel()), max_macro_steps=12, hidden_dim=16, hidden_layers=1)
        teacher = torch.nn.Linear(12 + int(setting_features("euler", 4).numel()), 1)
        with self.assertRaisesRegex(ValueError, "Missing SER-PTG reference targets"):
            optimize_student_with_teacher(
                student,
                teacher,
                [("euler", 4), ("heun", 4)],
                max_macro_steps=12,
                steps=1,
                lr=1e-3,
                ser_ptg_reference_pairs=[("euler", 4, _uniform_grid(4))],
                ser_ptg_regularizer="js",
                ser_ptg_regularization_weight=0.02,
            )

    def test_direct_opd_defaults_are_clean_guarded_budgets(self) -> None:
        args = build_argparser().parse_args(["--dry_run"])
        budgets = [int(value) for value in args.student_opd_step_values.split(",")]
        self.assertEqual(budgets, [5, 10, 15, 20, 25])
        self.assertEqual(args.required_split_phase, "train_tuning")
        self.assertNotIn(50, budgets)
        self.assertNotIn(100, budgets)
        self.assertNotIn(200, budgets)
        self.assertNotIn(500, budgets)
        self.assertNotIn(35, budgets)

    def test_direct_opd_diagnostic_flag_adds_35_and_50(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_argparser().parse_args(["--dry_run", "--include_diagnostic_budgets", "--out_dir", tmpdir])
            self.assertTrue(args.include_diagnostic_budgets)
            summary = train_conditional_opd(args)
            self.assertEqual(summary["student_opd_step_values"], [5, 10, 15, 20, 25, 35, 50])

    def test_direct_opd_rejects_diagnostic_budgets_without_flag(self) -> None:
        args = build_argparser().parse_args(["--dry_run", "--student_opd_step_values", "35"])
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            train_conditional_opd(args)

    def test_deprecated_student_steps_alias_is_removed(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_argparser().parse_args(["--student_steps", "10"])

    def test_teacher_rejects_non_train_tuning_rows(self) -> None:
        args = build_argparser().parse_args(["--dry_run", "--required_split_phase", "validation_tuning"])
        with self.assertRaisesRegex(ValueError, "train_tuning"):
            train_conditional_opd(args)

    def test_teacher_fixed_demos_default_to_all_fixed_anchors(self) -> None:
        args = build_argparser().parse_args(["--dry_run"])
        self.assertEqual(args.teacher_fixed_schedule_keys, ",".join(BASELINE_SCHEDULE_KEYS))
        self.assertEqual(args.reward_reference_schedule_keys, "")
        self.assertEqual(args.late_biased_demo_schedules, "late_power_3")
        self.assertEqual(args.late_biased_demo_weight, 1.0)
        weights = teacher_schedule_weights(BASELINE_SCHEDULE_KEYS)
        self.assertEqual(set(weights), set(BASELINE_SCHEDULE_KEYS))
        self.assertTrue(all(value == 1.0 for value in weights.values()))
        self.assertEqual(BASELINE_SCHEDULE_KEYS, ("uniform", "late_power_3", "flowts_power_sampling", "ays", "gits", "ots"))

    def test_reward_reference_keys_match_all_fixed_teacher_anchors_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_argparser().parse_args(
                [
                    "--dry_run",
                    "--out_dir",
                    tmpdir,
                    "--reward_reference_schedule_keys",
                    ",".join(BASELINE_SCHEDULE_KEYS),
                ]
            )
            summary = train_conditional_opd(args)
        self.assertEqual(summary["teacher_fixed_schedule_keys"], list(BASELINE_SCHEDULE_KEYS))
        self.assertEqual(summary["reward_reference_schedule_keys"], list(BASELINE_SCHEDULE_KEYS))

    def test_reward_reference_keys_must_include_teacher_fixed_demos(self) -> None:
        args = build_argparser().parse_args(["--dry_run", "--reward_reference_schedule_keys", "uniform"])
        with self.assertRaisesRegex(ValueError, "must include teacher fixed schedules"):
            train_conditional_opd(args)

    def test_ser_ptg_metric_rows_are_rejected_for_teacher_training(self) -> None:
        args = build_argparser().parse_args(["--dry_run", "--reference_rows_csv", "ser_rows.csv"])
        with self.assertRaisesRegex(ValueError, "SER-PTG metric rows are not used"):
            train_conditional_opd(args)

    def test_teacher_rejects_legacy_sampler_rows_when_valnorm_expected(self) -> None:
        rows = [
            {
                "scheduler_key": "uniform",
                "train_tuning_sampler": "temporal_stratified_hash_v1",
                "train_tuning_fraction": "0.2",
            }
        ]
        with self.assertRaisesRegex(ValueError, "train_tuning_sampler"):
            _assert_expected_train_tuning_metadata(
                rows,
                expected_sampler="temporal_stratified_validation_normalized_v2",
                expected_fraction="0.2",
                label="baseline train-tuning",
            )

    def test_20_update_student_grid_is_valid_and_exact_nfe(self) -> None:
        class CountingTeacher(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return features[:, -12:].sum(dim=-1)

        student = ScheduleStudentMLP(int(setting_features("euler", 4).numel()), max_macro_steps=12, hidden_dim=16, hidden_layers=1)
        teacher = CountingTeacher()
        optimize_student_with_teacher(student, teacher, [("euler", 4)], max_macro_steps=12, steps=20, lr=1e-3)
        self.assertEqual(teacher.calls, 20)
        grid = student.time_grid(setting_features("euler", 4)[None, :], macro_steps=solver_macro_steps("euler", 4))[0]
        values = [float(value) for value in grid.detach().cpu().tolist()]
        self.assertEqual(validate_time_grid(values, macro_steps=4), tuple(values))
        self.assertEqual(solver_macro_steps("euler", 4), 4)

    def test_bo_candidate_rows_expand_teacher_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_path = root / "rows.csv"
            candidate_rows_path = root / "candidate_rows.csv"
            ser_summary_path = root / "ser_summary.json"
            candidate_summary_path = root / "candidate_summary.json"
            fieldnames = [
                "dataset",
                "split_phase",
                "seed",
                "solver_key",
                "target_nfe",
                "scheduler_key",
                "crps",
                "mase",
                "train_tuning_sampler",
                "train_tuning_fraction",
            ]
            baseline_rows = []
            candidate_rows = []
            ser_predictions = []
            candidate_keys = [f"train20_v43_bo_r0_cand{idx:03d}" for idx in range(8)]
            candidate_predictions_by_key = {key: [] for key in candidate_keys}
            for solver in ("euler", "heun", "midpoint_rk2", "dpmpp2m"):
                for target_nfe in (4, 8, 12):
                    macro_steps = solver_macro_steps(solver, target_nfe)
                    grid = _uniform_grid(macro_steps)
                    ser_predictions.append({"solver_key": solver, "target_nfe": target_nfe, "time_grid": grid})
                    for candidate_key in candidate_keys:
                        candidate_predictions_by_key[candidate_key].append({"solver_key": solver, "target_nfe": target_nfe, "time_grid": grid})
                    for seed in (0, 1):
                        for idx, schedule_key in enumerate(BASELINE_SCHEDULE_KEYS):
                            baseline_rows.append(
                                {
                                    "dataset": "san_francisco_traffic",
                                    "split_phase": "train_tuning",
                                    "seed": seed,
                                    "solver_key": solver,
                                    "target_nfe": target_nfe,
                                    "scheduler_key": schedule_key,
                                    "crps": 1.0 + 0.01 * idx,
                                    "mase": 2.0 + 0.01 * idx,
                                    "train_tuning_sampler": "temporal_stratified_validation_normalized_v2",
                                    "train_tuning_fraction": "0.2",
                                }
                            )
                        for candidate_idx, candidate_key in enumerate(candidate_keys):
                            candidate_rows.append(
                                {
                                    "dataset": "san_francisco_traffic",
                                    "split_phase": "train_tuning",
                                    "seed": seed,
                                    "solver_key": solver,
                                    "target_nfe": target_nfe,
                                    "scheduler_key": candidate_key,
                                    "crps": 0.95 + 0.01 * candidate_idx,
                                    "mase": 1.95 + 0.01 * candidate_idx,
                                    "train_tuning_sampler": "temporal_stratified_validation_normalized_v2",
                                    "train_tuning_fraction": "0.2",
                                }
                            )
            for path, rows in ((rows_path, baseline_rows), (candidate_rows_path, candidate_rows)):
                with path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            ser_summary_path.write_text(
                json.dumps(
                    {
                        "dataset": "san_francisco_traffic",
                        "schedules": [{"scheduler_key": SER_PTG_SCHEDULE_KEY, "predictions": ser_predictions}],
                    }
                ),
                encoding="utf-8",
            )
            candidate_summary_path.write_text(
                json.dumps(
                    {
                        "dataset": "san_francisco_traffic",
                        "schedules": [
                            {"scheduler_key": key, "predictions": predictions}
                            for key, predictions in candidate_predictions_by_key.items()
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = build_argparser().parse_args(
                [
                    "--rows_csv",
                    str(rows_path),
                    "--reference_schedule_summary",
                    str(ser_summary_path),
                    "--reward_reference_schedule_keys",
                    ",".join(BASELINE_SCHEDULE_KEYS),
                    "--candidate_rows_csv",
                    str(candidate_rows_path),
                    "--candidate_schedule_summary",
                    str(candidate_summary_path),
                    "--seeds",
                    "0,1",
                    "--teacher_steps",
                    "1",
                    "--student_init_steps",
                    "1",
                    "--student_opd_step_values",
                    "1",
                    "--expected_train_tuning_sampler",
                    "temporal_stratified_validation_normalized_v2",
                    "--expected_train_tuning_fraction",
                    "0.2",
                    "--out_dir",
                    str(root / "out"),
                ]
            )
            summary = train_conditional_opd(args)
        sources = summary["teacher_supervision_sources"]
        self.assertEqual(sources["reward_reference_seed_rows"], 144)
        self.assertEqual(sources["baseline_seed_rows"], 144)
        self.assertEqual(sources["evaluated_candidate_seed_rows"], 192)
        self.assertEqual(sources["teacher_training_rows"], 168)
        self.assertEqual(sources["teacher_fit_rows"], 144)
        self.assertEqual(sources["teacher_diagnostic_holdout_rows"], 24)
        self.assertEqual(sources["teacher_holdout_source"], "bo_candidate_rows_only")
        self.assertEqual(summary["reward_reference_schedule_keys"], list(BASELINE_SCHEDULE_KEYS))
        self.assertIn("train20_v43_bo_r0_cand000", summary["candidate_schedule_keys"])
        self.assertEqual(summary["teacher_objective"], "ranking_first_best_fixed_composite_with_huber_calibration")
        self.assertEqual(summary["teacher_selection_protocol"], "pooled_bo_holdout_teacher_checkpoint")
        self.assertEqual(summary["teacher_checkpoint_selection"]["selection_split"], "teacher_holdout")
        self.assertTrue(summary["teacher_checkpoint_selection"]["history"])
        self.assertIn("teacher_rank_loss", summary["teacher_losses"][-1])
        self.assertGreater(summary["teacher_losses"][-1]["teacher_pair_count"], 0)
        self.assertIn("teacher_diagnostics", summary)
        self.assertFalse(summary["teacher_diagnostics"]["uses_validation_labels"])
        self.assertEqual(summary["teacher_diagnostics"]["diagnostic_rows"], "held_out_train_tuning")
        self.assertEqual(summary["teacher_diagnostics"]["teacher_holdout_source"], "bo_candidate_rows_only")
        self.assertEqual(summary["teacher_diagnostics"]["heldout_row_count"], 24)
        first_candidate = summary["candidate_table"][0]
        self.assertIn("u_crps_best", first_candidate)
        self.assertIn("u_mase_best", first_candidate)
        self.assertIn("u_comp_best", first_candidate)
        self.assertIn("u_comp_uniform", first_candidate)

if __name__ == "__main__":
    unittest.main()
