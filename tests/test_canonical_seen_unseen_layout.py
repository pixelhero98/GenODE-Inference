from __future__ import annotations

import unittest

from genode.canonical_experiment_layout import (
    AVERAGED_REVERSED_SCHEDULE_KEYS,
    CANONICAL_CHECKPOINT_STEPS,
    CANONICAL_CONTEXT_SAMPLE_COUNT,
    CANONICAL_SCENARIO_KEYS,
    CANONICAL_SEEN_NFES,
    CANONICAL_SOLVER_KEYS,
    CANONICAL_SUPERVISION_SCHEDULE_KEYS,
    CANONICAL_UNSEEN_NFES,
    PHYSICAL_SCHEDULE_KEYS,
    REVERSED_SCHEDULE_KEYS,
    canonical_nfes_for_role,
)
from genode.evaluation import diffusion_flow_time_reparameterization as runner
from genode.solver_protocol import expected_realized_nfe, normalize_solver_key, normalize_solver_keys, solver_macro_steps
from genode.gipo.train_gipo import build_argparser as build_gipo_argparser


class CanonicalSeenUnseenLayoutTests(unittest.TestCase):
    def test_layout_constants_are_canonical(self) -> None:
        self.assertEqual(CANONICAL_SEEN_NFES, (4, 8, 12, 16))
        self.assertEqual(CANONICAL_UNSEEN_NFES, (6, 10, 14, 20))
        self.assertEqual(CANONICAL_CHECKPOINT_STEPS, (4000, 8000, 12000, 16000, 20000))
        self.assertEqual(CANONICAL_SOLVER_KEYS, ("euler", "dpmpp2m", "heun", "midpoint_rk2"))
        self.assertEqual(len(CANONICAL_SCENARIO_KEYS), 9)
        self.assertEqual(len(PHYSICAL_SCHEDULE_KEYS), 7)
        self.assertEqual(len(REVERSED_SCHEDULE_KEYS), 6)
        self.assertEqual(len(AVERAGED_REVERSED_SCHEDULE_KEYS), 6)
        self.assertEqual(len(CANONICAL_SUPERVISION_SCHEDULE_KEYS), 19)

    def test_role_and_runner_defaults(self) -> None:
        seen_args = runner.build_argparser().parse_args([])
        self.assertEqual(runner._target_nfe_values_for_args(seen_args), [4, 8, 12, 16])
        self.assertEqual(runner._checkpoint_steps_for_args(seen_args), [4000, 8000, 12000, 16000, 20000])
        unseen_args = runner.build_argparser().parse_args(["--nfe_role", "unseen"])
        self.assertEqual(runner._target_nfe_values_for_args(unseen_args), [6, 10, 14, 20])
        self.assertEqual(canonical_nfes_for_role("seen"), CANONICAL_SEEN_NFES)
        self.assertEqual(canonical_nfes_for_role("unseen"), CANONICAL_UNSEEN_NFES)

    def test_runner_rejects_noncanonical_nfe_and_checkpoint(self) -> None:
        bad_nfe = runner.build_argparser().parse_args(["--target_nfe_values", "5"])
        with self.assertRaisesRegex(ValueError, "not canonical"):
            runner._target_nfe_values_for_args(bad_nfe)
        bad_step = runner.build_argparser().parse_args(["--checkpoint_steps", "123"])
        with self.assertRaisesRegex(ValueError, "not canonical"):
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
        self.assertEqual(args.context_sample_count, CANONICAL_CONTEXT_SAMPLE_COUNT)
        self.assertEqual(args.teacher_unseen_selection_target_nfe_values, "6,10,14,20")
        self.assertAlmostEqual(float(args.student_pseudo_target_weight), 0.25)

    def test_solver_protocol_is_canonical_and_aliases_do_not_persist(self) -> None:
        self.assertEqual(normalize_solver_key("dpm++2m"), "dpmpp2m")
        self.assertEqual(normalize_solver_key("rk2"), "heun")
        self.assertEqual(normalize_solver_keys("euler,dpm++2m,rk2,midpoint rk2"), CANONICAL_SOLVER_KEYS)
        with self.assertRaisesRegex(ValueError, "Duplicate solver keys"):
            normalize_solver_keys("heun,rk2")
        self.assertEqual(solver_macro_steps("heun", 4), 2)
        self.assertEqual(solver_macro_steps("dpmpp2m", 4), 4)
        self.assertEqual(expected_realized_nfe("heun", 4), 4)
        self.assertEqual(expected_realized_nfe("dpmpp2m", 4), 4)


if __name__ == "__main__":
    unittest.main()
