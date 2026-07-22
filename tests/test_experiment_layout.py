from __future__ import annotations

import unittest

from genode.experiment_layout import (
    AVERAGED_REVERSED_SCHEDULE_KEYS,
    REFERENCE_CHECKPOINT_STEPS,
    TRAIN_TUNING_CONTEXT_SAMPLE_COUNT,
    REFERENCE_SCENARIO_KEYS,
    REFERENCE_SEEN_NFES,
    REFERENCE_SUPERVISION_SCHEDULE_KEYS,
    REFERENCE_UNSEEN_NFES,
    PHYSICAL_SCHEDULE_KEYS,
    REVERSED_SCHEDULE_KEYS,
    target_nfes_for_role,
)
from genode.evaluation import diffusion_flow_time_reparameterization as runner
from genode.solver_protocol import (
    SUPPORTED_SOLVER_KEYS,
    expected_realized_nfe,
    normalize_solver_key,
    normalize_solver_keys,
    normalize_solver_nfe_fields,
    solver_macro_steps,
)
from genode.gipo.train_gipo import build_argparser as build_gipo_argparser


class ExperimentLayoutTests(unittest.TestCase):
    def test_reference_layout_constants(self) -> None:
        self.assertEqual(REFERENCE_SEEN_NFES, (4, 8, 12, 16))
        self.assertEqual(REFERENCE_UNSEEN_NFES, (6, 10, 14, 20))
        self.assertEqual(REFERENCE_CHECKPOINT_STEPS, (4000, 8000, 12000, 16000, 20000))
        self.assertEqual(TRAIN_TUNING_CONTEXT_SAMPLE_COUNT, 188)
        self.assertEqual(SUPPORTED_SOLVER_KEYS, ("euler", "dpmpp2m", "heun", "midpoint_rk2"))
        self.assertEqual(len(REFERENCE_SCENARIO_KEYS), 9)
        self.assertEqual(len(PHYSICAL_SCHEDULE_KEYS), 7)
        self.assertEqual(len(REVERSED_SCHEDULE_KEYS), 6)
        self.assertEqual(len(AVERAGED_REVERSED_SCHEDULE_KEYS), 6)
        self.assertEqual(len(REFERENCE_SUPERVISION_SCHEDULE_KEYS), 19)

    def test_role_and_runner_defaults(self) -> None:
        seen_args = runner.build_argparser().parse_args([])
        self.assertEqual(runner._target_nfe_values_for_args(seen_args), [4, 8, 12, 16])
        self.assertEqual(runner._checkpoint_steps_for_args(seen_args), [4000, 8000, 12000, 16000, 20000])
        unseen_args = runner.build_argparser().parse_args(["--nfe_role", "unseen"])
        self.assertEqual(runner._target_nfe_values_for_args(unseen_args), [6, 10, 14, 20])
        self.assertEqual(target_nfes_for_role("seen"), REFERENCE_SEEN_NFES)
        self.assertEqual(target_nfes_for_role("unseen"), REFERENCE_UNSEEN_NFES)

    def test_runner_rejects_values_outside_reference_protocol(self) -> None:
        bad_nfe = runner.build_argparser().parse_args(["--target_nfe_values", "5"])
        with self.assertRaisesRegex(ValueError, "outside the reference protocol"):
            runner._target_nfe_values_for_args(bad_nfe)
        bad_step = runner.build_argparser().parse_args(["--checkpoint_steps", "123"])
        with self.assertRaisesRegex(ValueError, "outside the reference protocol"):
            runner._checkpoint_steps_for_args(bad_step)

    def test_gipo_parser_defaults(self) -> None:
        args = build_gipo_argparser().parse_args(
            [
                "--rows_csv",
                "rows.csv",
                "--context_embeddings_npz",
                "ctx.npz",
                "--out_dir",
                "out",
            ]
        )
        self.assertEqual(args.context_sample_count, TRAIN_TUNING_CONTEXT_SAMPLE_COUNT)
        self.assertEqual(args.teacher_unseen_selection_target_nfe_values, "6,10,14,20")
        self.assertAlmostEqual(float(args.student_unseen_target_weight), 0.25)
        self.assertAlmostEqual(float(args.student_teacher_score_weight), 0.01)
        self.assertAlmostEqual(float(args.student_teacher_score_warmup_fraction), 0.6)
        self.assertFalse(args.student_teacher_score_include_unseen_targets)
        self.assertEqual(args.student_target_mixture_mode, "full")
        self.assertAlmostEqual(float(args.student_target_elite_fraction), 0.3)
        self.assertEqual(int(args.student_target_elite_k), 0)
        self.assertEqual(int(args.student_target_elite_min_count), 2)
        self.assertAlmostEqual(float(args.student_target_elite_blend_all_weight), 0.2)

    def test_solver_protocol_accepts_only_supported_keys(self) -> None:
        for solver_key in SUPPORTED_SOLVER_KEYS:
            self.assertEqual(normalize_solver_key(solver_key), solver_key)
        self.assertEqual(normalize_solver_keys(",".join(SUPPORTED_SOLVER_KEYS)), SUPPORTED_SOLVER_KEYS)
        for unsupported_alias in ("dpm++2m", "rk2", "midpoint rk2", "Euler"):
            with self.subTest(unsupported_alias=unsupported_alias):
                with self.assertRaisesRegex(ValueError, "Unknown solver_key"):
                    normalize_solver_key(unsupported_alias)
        with self.assertRaisesRegex(ValueError, "Duplicate solver keys"):
            normalize_solver_keys("heun,heun")
        self.assertEqual(solver_macro_steps("heun", 4), 2)
        self.assertEqual(solver_macro_steps("dpmpp2m", 4), 4)
        self.assertEqual(expected_realized_nfe("heun", 4), 4)
        self.assertEqual(expected_realized_nfe("dpmpp2m", 4), 4)
        nfe = normalize_solver_nfe_fields("heun", 4, macro_steps=2, realized_nfe=4)
        self.assertEqual((nfe.macro_steps, nfe.realized_nfe), (2, 4))
        with self.assertRaisesRegex(ValueError, "macro_steps=4"):
            normalize_solver_nfe_fields("heun", 4, macro_steps=4)
        for target_nfe in (4.5, True):
            with self.subTest(target_nfe=target_nfe), self.assertRaisesRegex(
                ValueError,
                "non-integer target_nfe",
            ):
                solver_macro_steps("euler", target_nfe)


if __name__ == "__main__":
    unittest.main()
