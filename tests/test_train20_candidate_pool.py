from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from genode.conditional_opd.candidate_pool import build_candidate_pool, select_candidate_schedules
from genode.conditional_opd.evaluate_schedule_summary import load_schedule_predictions
from genode.conditional_opd.models import ScheduleTeacherMLP, setting_features


def _uniform_grid(steps: int) -> list[float]:
    return [float(idx) / float(steps) for idx in range(steps + 1)]


def _skew_grid(steps: int) -> list[float]:
    values = [(float(idx) / float(steps)) ** 2 for idx in range(steps + 1)]
    values[0] = 0.0
    values[-1] = 1.0
    return values


def _schedule_summary(path: Path) -> None:
    predictions = []
    for solver, nfes in {
        "euler": (4, 8, 12),
        "heun": (4, 8, 12),
        "midpoint_rk2": (4, 8, 12),
        "dpmpp2m": (4, 8, 12),
    }.items():
        for target_nfe in nfes:
            macro_steps = target_nfe // 2 if solver in {"heun", "midpoint_rk2"} else target_nfe
            predictions.append(
                {
                    "solver_key": solver,
                    "target_nfe": target_nfe,
                    "macro_steps": macro_steps,
                    "time_grid": _skew_grid(macro_steps),
                    "utility": 0.1,
                }
            )
    path.write_text(
        json.dumps(
            {
                "dataset": "san_francisco_traffic",
                "schedules": [
                    {
                        "scheduler_key": "conditional_opd_r0_seed0_steps5",
                        "opd_step_budget": 5,
                        "teacher_predicted_utility_mean": 0.1,
                        "predictions": predictions,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _zero_teacher_checkpoint(path: Path) -> None:
    setting_dim = int(setting_features("euler", 4).numel())
    max_macro_steps = 12
    teacher = ScheduleTeacherMLP(setting_dim + max_macro_steps, hidden_dim=256, hidden_layers=3)
    for param in teacher.parameters():
        param.data.zero_()
    torch.save(
        {
            "teacher_state": teacher.state_dict(),
            "teacher_input_dim": setting_dim + max_macro_steps,
            "max_macro_steps": max_macro_steps,
        },
        path,
    )


class Train20CandidatePoolTests(unittest.TestCase):
    def test_candidate_pool_emits_complete_valid_schedule_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.json"
            _schedule_summary(source)
            pool = build_candidate_pool(
                source_schedule_summaries=(source,),
                active_round=0,
                seed=11,
                temperature_values=(0.85,),
                logit_noise_values=(0.05,),
                dirichlet_student_alpha_values=(100.0,),
                random_dirichlet_alpha_values=(1.0,),
            )
            self.assertGreaterEqual(pool["schedule_count"], 5)
            out = Path(tmpdir) / "pool.json"
            out.write_text(json.dumps(pool), encoding="utf-8")
            for schedule in pool["schedules"]:
                self.assertIn("full_family_grid_hash", schedule)
                self.assertEqual(len(schedule["predictions"]), 12)
                for item in schedule["predictions"]:
                    self.assertIn("intervals_json", item)
                    self.assertIn("validity_flags_json", item)
                    flags = json.loads(item["validity_flags_json"])
                    self.assertTrue(flags["finite"])
                    self.assertTrue(flags["monotone"])
                    self.assertTrue(flags["exact_realized_nfe"])
            predictions = load_schedule_predictions(out, dataset="san_francisco_traffic")
            self.assertEqual(len(predictions), 12 * pool["schedule_count"])

    def test_candidate_pool_scores_variants_with_teacher_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            teacher_path = root / "conditional_opd.pt"
            _schedule_summary(source)
            _zero_teacher_checkpoint(teacher_path)
            pool = build_candidate_pool(
                source_schedule_summaries=(source,),
                teacher_checkpoint_paths=(teacher_path,),
                active_round=0,
                seed=11,
                temperature_values=(0.85,),
                logit_noise_values=(),
                dirichlet_student_alpha_values=(),
                random_dirichlet_alpha_values=(1.0,),
            )
        self.assertGreaterEqual(pool["schedule_count"], 3)
        for schedule in pool["schedules"]:
            self.assertIn("teacher_score_source", schedule)
            self.assertAlmostEqual(float(schedule["teacher_predicted_utility_mean"]), 0.0)
            for item in schedule["predictions"]:
                self.assertAlmostEqual(float(item["utility"]), 0.0)

    def test_k_center_diverse_selection_improves_interval_coverage(self) -> None:
        schedules = []
        for idx, p in enumerate((0.25, 0.30, 0.35, 0.90)):
            grid = [0.0, p, 1.0]
            schedules.append(
                {
                    "scheduler_key": f"s{idx}",
                    "teacher_predicted_utility_mean": 1.0 - 0.01 * idx,
                    "predictions": [{"solver_key": "heun", "target_nfe": 4, "macro_steps": 2, "time_grid": grid}],
                }
            )
        exploit_only, _ = select_candidate_schedules(schedules, exploit_count=3, diverse_count=0, random_count=0, seed=0)
        mixed, _ = select_candidate_schedules(schedules, exploit_count=1, diverse_count=2, random_count=0, seed=0)

        def spread(selected: list[dict]) -> float:
            points = [float(item["predictions"][0]["time_grid"][1]) for item in selected]
            return float(np.max(points) - np.min(points))

        self.assertGreater(spread(mixed), spread(exploit_only))


if __name__ == "__main__":
    unittest.main()
