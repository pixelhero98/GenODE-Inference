from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from genode.conditional_opd.bo_candidate_pool import build_argparser as build_bo_argparser
from genode.conditional_opd.bo_candidate_pool import build_and_write_bo_candidate_pool, build_bo_candidate_pool, require_botorch
from genode.conditional_opd.clock_lowrank import (
    DEFAULT_DENSITY_GRID_SIZE,
    LOWRANK_THETA_DIM,
    schedule_grid_from_theta,
    sobol_thetas,
    zero_theta,
)
from genode.conditional_opd.models import solver_macro_steps, validate_time_grid
from genode.conditional_opd.legacy.train20_v4_bo_selection import build_argparser, run_train20_v4_bo_selection
from genode.conditional_opd.legacy.train20_v42_f_final_retrain import (
    build_argparser as build_v42_f_argparser,
    run_train20_v42_f_final_retrain,
)
from genode.conditional_opd.legacy.train20_v42_no_val_selection import (
    build_argparser as build_no_val_argparser,
    run_train20_v42_no_val_selection,
    select_no_validation_student_schedule,
)
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS


SOLVERS = ("euler", "heun", "midpoint_rk2", "dpmpp2m")
NFES = (4, 8, 12)


def _uniform_grid(n_steps: int) -> list[float]:
    return [float(idx) / float(n_steps) for idx in range(int(n_steps) + 1)]


def _write_ser_summary(path: Path) -> None:
    predictions = []
    for solver in SOLVERS:
        for nfe in NFES:
            macro_steps = solver_macro_steps(solver, nfe)
            predictions.append(
                {
                    "solver_key": solver,
                    "target_nfe": int(nfe),
                    "macro_steps": int(macro_steps),
                    "time_grid": _uniform_grid(macro_steps),
                }
            )
    path.write_text(
        json.dumps(
            {
                "status": "ready",
                "dataset": "san_francisco_traffic",
                "schedules": [
                    {
                        "scheduler_key": "ser_ptg_local_defect_eta005",
                        "schedule_name": "SER-PTG test",
                        "predictions": predictions,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class Train20V4BOTests(unittest.TestCase):
    def test_botorch_import_preflight(self) -> None:
        versions = require_botorch()
        self.assertIn("botorch", versions)
        self.assertIn("gpytorch", versions)

    def test_lowrank_grid_is_deterministic_and_uses_128_bins(self) -> None:
        theta = sobol_thetas(1, seed=13)[0]
        self.assertEqual(len(theta), LOWRANK_THETA_DIM)
        first = schedule_grid_from_theta(theta, macro_steps=8, base_grid=_uniform_grid(8), grid_size=DEFAULT_DENSITY_GRID_SIZE)
        second = schedule_grid_from_theta(theta, macro_steps=8, base_grid=_uniform_grid(8), grid_size=DEFAULT_DENSITY_GRID_SIZE)
        self.assertEqual(first, second)
        self.assertEqual(validate_time_grid(first, macro_steps=8), first)

    def test_bo_candidate_pool_emits_complete_12_cell_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ser_path = Path(tmpdir) / "ser.json"
            _write_ser_summary(ser_path)
            pool = build_bo_candidate_pool(
                reference_schedule_summary=ser_path,
                active_round=0,
                seed=0,
                candidate_count=3,
                density_grid_size=128,
            )
        self.assertEqual(pool["candidate_source"], "bayesian_optimization")
        self.assertEqual(pool["density_grid_size"], 128)
        self.assertEqual(pool["schedule_count"], 3)
        for schedule in pool["schedules"]:
            self.assertEqual(schedule["candidate_source"], "bayesian_optimization")
            self.assertTrue(schedule["scheduler_key"].startswith("train20_v43_bo_r0_cand"))
            self.assertEqual(schedule["comparison_role"], "v43_pooled_bo_candidate")
            self.assertEqual(len(schedule["predictions"]), 12)
            self.assertTrue(all(item["density_grid_size"] == 128 for item in schedule["predictions"]))
            self.assertTrue(all(item["bo_theta_json"] for item in schedule["predictions"]))

    def test_bo_candidate_pool_no_longer_exposes_direct_selection_mode(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_bo_argparser().parse_args(["--mode", "select"])

    def test_bo_candidate_pool_cli_accepts_source_balanced_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ser_path = root / "ser.json"
            out_path = root / "pool.json"
            _write_ser_summary(ser_path)
            args = build_bo_argparser().parse_args(
                [
                    "--reference_schedule_summary",
                    str(ser_path),
                    "--out_path",
                    str(out_path),
                    "--active_round",
                    "0",
                    "--candidate_count",
                    "2",
                    "--observed_schedule_summaries",
                    str(root / "observed_summary.json"),
                    "--observed_train_rows_csv",
                    str(root / "observed_train.csv"),
                    "--observed_validation_rows_csv",
                    str(root / "observed_val.csv"),
                    "--fixed_train_rows_csv",
                    str(root / "fixed_train.csv"),
                    "--fixed_validation_rows_csv",
                    str(root / "fixed_val.csv"),
                ]
            )
            summary = build_and_write_bo_candidate_pool(args)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "ready")
        self.assertEqual(payload["artifact"], "v43_pooled_bo_candidate_pool_schedule_summary")
        self.assertEqual(payload["dataset"], "san_francisco_traffic")
        self.assertEqual(payload["schedule_count"], 2)
        self.assertEqual(payload["density_grid_size"], 128)

    def test_bo_candidate_pool_source_balanced_round1_uses_nonempty_rows(self) -> None:
        def write_rows(path: Path, rows: list[dict]) -> None:
            fields = ["dataset", "split_phase", "solver_key", "target_nfe", "scheduler_key", "seed", "crps", "mase"]
            import csv

            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ser_path = root / "ser.json"
            prior_path = root / "prior_bo.json"
            out_path = root / "round1_pool.json"
            _write_ser_summary(ser_path)
            prior_pool = build_bo_candidate_pool(
                reference_schedule_summary=ser_path,
                active_round=0,
                seed=0,
                candidate_count=1,
                density_grid_size=128,
            )
            prior_path.write_text(json.dumps(prior_pool), encoding="utf-8")
            candidate_key = prior_pool["schedules"][0]["scheduler_key"]

            fixed_train_rows: list[dict] = []
            fixed_val_rows: list[dict] = []
            candidate_train_rows: list[dict] = []
            candidate_val_rows: list[dict] = []
            for solver in SOLVERS:
                for nfe in NFES:
                    for idx, key in enumerate(BASELINE_SCHEDULE_KEYS):
                        base = {
                            "dataset": "san_francisco_traffic",
                            "solver_key": solver,
                            "target_nfe": str(nfe),
                            "scheduler_key": key,
                            "seed": "0",
                            "crps": str(1.0 + 0.01 * idx),
                            "mase": str(1.2 + 0.01 * idx),
                        }
                        fixed_train_rows.append({**base, "split_phase": "train_tuning"})
                        fixed_val_rows.append({**base, "split_phase": "validation_tuning", "crps": str(1.1 + 0.01 * idx)})
                    candidate_train_rows.append(
                        {
                            "dataset": "san_francisco_traffic",
                            "split_phase": "train_tuning",
                            "solver_key": solver,
                            "target_nfe": str(nfe),
                            "scheduler_key": candidate_key,
                            "seed": "0",
                            "crps": "0.95",
                            "mase": "1.1",
                        }
                    )
                    candidate_val_rows.append(
                        {
                            "dataset": "san_francisco_traffic",
                            "split_phase": "validation_tuning",
                            "solver_key": solver,
                            "target_nfe": str(nfe),
                            "scheduler_key": candidate_key,
                            "seed": "0",
                            "crps": "1.05",
                            "mase": "1.15",
                        }
                    )

            fixed_train = root / "fixed_train.csv"
            fixed_val = root / "fixed_val.csv"
            cand_train = root / "candidate_train.csv"
            cand_val = root / "candidate_val.csv"
            write_rows(fixed_train, fixed_train_rows)
            write_rows(fixed_val, fixed_val_rows)
            write_rows(cand_train, candidate_train_rows)
            write_rows(cand_val, candidate_val_rows)

            args = build_bo_argparser().parse_args(
                [
                    "--reference_schedule_summary",
                    str(ser_path),
                    "--out_path",
                    str(out_path),
                    "--active_round",
                    "1",
                    "--candidate_count",
                    "2",
                    "--observed_schedule_summaries",
                    str(prior_path),
                    "--observed_train_rows_csv",
                    str(cand_train),
                    "--observed_validation_rows_csv",
                    str(cand_val),
                    "--fixed_train_rows_csv",
                    str(fixed_train),
                    "--fixed_validation_rows_csv",
                    str(fixed_val),
                ]
            )
            summary = build_and_write_bo_candidate_pool(args)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "ready")
        self.assertEqual(payload["active_round"], 1)
        self.assertEqual(payload["schedule_count"], 2)
        self.assertTrue(all(meta["observation_count"] == 1 for meta in payload["bo_acquisition"].values()))

    def test_v4_dry_run_keeps_bo_validation_and_test_seeds_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_argparser().parse_args(
                [
                    "--out_dir",
                    tmpdir,
                    "--device",
                    "cpu",
                    "--bo_rounds",
                    "1",
                    "--bo_candidates_per_round",
                    "2",
                ]
            )
            summary = run_train20_v4_bo_selection(args)
        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["bo_eval_seeds"], [0, 1])
        self.assertEqual(summary["selection_seeds"], [0, 1, 2])
        self.assertEqual(summary["locked_test_seeds"], [0, 1, 2])
        self.assertEqual(summary["density_grid_size"], 128)
        self.assertEqual(summary["bo_teacher_holdout_fraction"], 0.20)
        self.assertEqual(summary["teacher_selection_protocol"], "v4.2_option_a_bo_heldout_teacher_then_guarded_validation_student")
        self.assertEqual(summary["artifact"], "train20_v4_bo_neural_selection")
        flat = [str(part) for command in summary["commands"] for part in command]
        self.assertIn("genode.conditional_opd.bo_candidate_pool", flat)
        self.assertIn("genode.conditional_opd.train_conditional_opd", flat)
        self.assertNotIn("--temperature_values", flat)
        self.assertNotIn("--logit_noise_values", flat)
        self.assertNotIn("--mode select", " ".join(flat))
        baseline = next(command for command in summary["commands"] if "genode.evaluation.diffusion_flow_time_reparameterization" in command)
        self.assertEqual(baseline[baseline.index("--seeds") + 1], "0,1")
        trainer = next(command for command in summary["commands"] if "genode.conditional_opd.train_conditional_opd" in command)
        self.assertEqual(trainer[trainer.index("--seeds") + 1], "0,1")
        self.assertEqual(trainer[trainer.index("--teacher_fixed_schedule_keys") + 1], "uniform,late_power_3,flowts_power_sampling,ays,gits,ots")
        self.assertEqual(trainer[trainer.index("--reward_reference_schedule_keys") + 1], "uniform,late_power_3,flowts_power_sampling,ays,gits,ots")
        self.assertEqual(trainer[trainer.index("--teacher_diagnostic_holdout_fraction") + 1], "0.2")
        self.assertIn("--candidate_rows_csv", trainer)
        self.assertIn("--candidate_schedule_summary", trainer)
        self.assertIn("round_00/bo_train_tuning_eval/train_tuning_rows.csv", trainer[trainer.index("--candidate_rows_csv") + 1])
        self.assertIn("round_00/bo_candidate_schedule_summary.json", trainer[trainer.index("--candidate_schedule_summary") + 1])
        self.assertEqual(trainer[trainer.index("--student_ser_ptg_regularizer") + 1], "none")
        validation = [command for command in summary["commands"] if "validation_tuning" in command][-1]
        self.assertEqual(validation[validation.index("--seeds") + 1], "0,1,2")
        self.assertEqual(
            validation[validation.index("--schedule_summary") + 1],
            str(Path(tmpdir) / "neural_policy" / "student_budget_schedule_summary.json"),
        )
        locked = [command for command in summary["commands"] if "locked_test" in command][-1]
        self.assertEqual(locked[locked.index("--seeds") + 1], "0,1,2")
        self.assertEqual(
            locked[locked.index("--schedule_summary") + 1],
            str(Path(tmpdir) / "selected_schedule_summary.json"),
        )


    def test_v42_f_dry_run_uses_calibration_not_validation_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_v42_f_argparser().parse_args(
                [
                    "--out_dir",
                    tmpdir,
                    "--device",
                    "cpu",
                    "--bo_rounds",
                    "2",
                    "--bo_candidates_per_round",
                    "2",
                    "--skip_locked_test",
                    "--no_clean_output_root",
                ]
            )
            summary = run_train20_v42_f_final_retrain(args)
        self.assertEqual(summary["status"], "dry_run")
        self.assertFalse(summary["uses_validation_selection"])
        self.assertEqual(summary["bo_eval_seeds"], [0, 1])
        self.assertEqual(summary["calibration_val_seeds"], [0, 1, 2])
        self.assertEqual(summary["student_seeds"], [0, 1, 2])
        self.assertEqual(summary["student_initialization"], "uniform")
        self.assertEqual(summary["final_checkpoint_mode"], "lowest_internal_loss")
        commands = summary["commands"]
        validation_eval_commands = [cmd for cmd in commands if "validation_tuning" in cmd]
        self.assertEqual(len(validation_eval_commands), 2)
        self.assertFalse(any("select_guarded_validation" in " ".join(cmd) for cmd in commands))
        round1_generate = [cmd for cmd in commands if "genode.conditional_opd.bo_candidate_pool" in cmd][-1]
        self.assertIn("--observed_train_rows_csv", round1_generate)
        self.assertIn("--observed_validation_rows_csv", round1_generate)
        trainer = [cmd for cmd in commands if "genode.conditional_opd.legacy.train_conditional_opd_v42_f" in cmd][0]
        self.assertEqual(trainer[trainer.index("--teacher_fixed_schedule_keys") + 1], "none")
        self.assertEqual(trainer[trainer.index("--student_seeds") + 1], "0,1,2")
        self.assertNotIn("--student_ser_ptg_regularizer", trainer)

    def test_v4_rejects_non_exact_bo_eval_seeds(self) -> None:
        args = build_argparser().parse_args(["--bo_eval_seeds", "0,1,2"])
        with self.assertRaisesRegex(ValueError, "exactly BO eval seeds 0,1"):
            run_train20_v4_bo_selection(args)
        args = build_argparser().parse_args(["--bo_eval_seeds", "2,3"])
        with self.assertRaisesRegex(ValueError, "exactly BO eval seeds 0,1"):
            run_train20_v4_bo_selection(args)

    def test_v4_rejects_noncanonical_partial_teacher_fixed_demos(self) -> None:
        args = build_argparser().parse_args(["--teacher_fixed_schedule_keys", "uniform,flowts_power_sampling"])
        with self.assertRaisesRegex(ValueError, "requires teacher fixed demos uniform,late_power_3,flowts_power_sampling,ays,gits,ots"):
            run_train20_v4_bo_selection(args)

    def test_v42_no_val_selection_uses_teacher_score_with_geometry_guard(self) -> None:
        def schedule(key: str, budget: int, score: float, max_interval: float) -> dict:
            return {
                "scheduler_key": key,
                "opd_step_budget": budget,
                "teacher_predicted_utility_mean": score,
                "predictions": [
                    {
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "runtime_nfe": 4,
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

        selection = select_no_validation_student_schedule(
            {
                "dataset": "san_francisco_traffic",
                "schedules": [
                    schedule("conditional_opd_student_steps10", 10, 0.1, 0.5),
                    schedule("conditional_opd_student_steps20", 20, 0.4, 0.8),
                    schedule("conditional_opd_student_steps25", 25, 0.9, 0.99),
                ],
            },
            conditional_opd_summary={
                "teacher_selection_protocol": "v4.2_option_a_bo_heldout_teacher_then_guarded_validation_student",
                "teacher_checkpoint_selection": {"selection_split": "teacher_holdout"},
            },
            max_interval_ceiling=0.97,
        )
        self.assertEqual(selection["selection_protocol"], "v4.2_option_b_no_validation_student_selection")
        self.assertFalse(selection["uses_validation_labels_for_selection"])
        self.assertEqual(selection["unguarded_top_schedule_key"], "conditional_opd_student_steps25")
        self.assertEqual(selection["selected_schedule_key"], "conditional_opd_student_steps20")
        self.assertEqual(selection["selected_opd_step_budget"], 20)
        self.assertTrue(selection["selected_geometry"]["passes_geometry_guard"])

    def test_v42_no_val_dry_run_writes_selection_and_locked_test_command_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy_dir = root / "source" / "neural_policy"
            policy_dir.mkdir(parents=True)
            predictions = []
            for solver in SOLVERS:
                for nfe in NFES:
                    macro_steps = solver_macro_steps(solver, nfe)
                    predictions.append(
                        {
                            "solver_key": solver,
                            "target_nfe": int(nfe),
                            "runtime_nfe": int(macro_steps),
                            "time_grid": _uniform_grid(macro_steps),
                            "grid_geometry": {
                                "internal_fraction_after_098": 0.0,
                                "min_interval": 1.0 / float(macro_steps),
                                "max_interval": 1.0 / float(macro_steps),
                            },
                            "utility": 0.5,
                        }
                    )
            (policy_dir / "student_budget_schedule_summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "dataset": "san_francisco_traffic",
                        "schedules": [
                            {
                                "scheduler_key": "conditional_opd_student_steps20",
                                "schedule_name": "Conditional OPD Student 20 updates",
                                "opd_step_budget": 20,
                                "teacher_predicted_utility_mean": 0.5,
                                "predictions": predictions,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (policy_dir / "conditional_opd_summary.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "teacher_selection_protocol": "v4.2_option_a_bo_heldout_teacher_then_guarded_validation_student",
                        "teacher_checkpoint_selection": {"selection_split": "teacher_holdout"},
                    }
                ),
                encoding="utf-8",
            )
            args = build_no_val_argparser().parse_args(
                [
                    "--source_run_dir",
                    str(root / "source"),
                    "--out_dir",
                    str(root / "no_val"),
                    "--device",
                    "cpu",
                ]
            )
            summary = run_train20_v42_no_val_selection(args)
        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["selection_protocol"], "v4.2_option_b_no_validation_student_selection")
        self.assertFalse(summary["uses_validation_labels_for_selection"])
        self.assertFalse(summary["validation_rows_evaluated"])
        self.assertEqual(summary["selected_schedule_key"], "conditional_opd_student_steps20")
        flat = [str(part) for command in summary["commands"] for part in command]
        self.assertIn("genode.conditional_opd.evaluate_schedule_summary", flat)
        self.assertIn("locked_test", flat)
        self.assertNotIn("validation_tuning", flat)


if __name__ == "__main__":
    unittest.main()
