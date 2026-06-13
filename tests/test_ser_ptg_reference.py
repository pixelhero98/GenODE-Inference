from __future__ import annotations

import unittest

import torch

from genode.gipo.ser_ptg_reference import build_argparser, collect_batched_local_defect_trace
from genode.evaluation.otflow_evaluation_support import build_conditional_generation_dataset_args_from_cfg
from genode.models.config import OTFlowConfig


class SerPtgReferenceTests(unittest.TestCase):
    def test_argparser_provides_conditional_loader_defaults(self) -> None:
        args = build_argparser().parse_args([])
        self.assertEqual(args.dataset_seed, 0)
        self.assertEqual(args.lr, 2e-4)
        self.assertEqual(args.weight_decay, 1e-4)
        self.assertEqual(args.grad_clip, 1.0)
        self.assertEqual(args.hidden_dim, 160)
        self.assertEqual(args.fu_net_layers, 3)
        self.assertEqual(args.fu_net_heads, 4)

    def test_argparser_defaults_rebuild_conditional_dataset_args(self) -> None:
        cfg = OTFlowConfig(
            device=torch.device("cpu"),
            levels=1,
            token_dim=4,
            history_len=12000,
            hidden_dim=160,
            dropout=0.0,
            ctx_heads=4,
            ctx_layers=1,
            fu_net_layers=3,
            fu_net_heads=4,
            rollout_mode="non_ar",
            future_block_len=3000,
            use_cond_features=True,
            cond_standardize=True,
            cond_dim=5,
            use_amp=False,
        )
        cfg.apply_overrides(steps=20000)
        args = build_argparser().parse_args(["--dataset", "long_term_st", "--device", "cpu"])

        dataset_args = build_conditional_generation_dataset_args_from_cfg(
            args,
            "long_term_st",
            "transformer",
            cfg,
        )

        self.assertEqual(dataset_args.seed, 0)
        self.assertEqual(dataset_args.lr, 2e-4)
        self.assertEqual(dataset_args.fu_net_layers, 3)
        self.assertEqual(dataset_args.future_block_len, 3000)

    def test_collect_batched_local_defect_trace_handles_sample_seed_above_numpy_limit(self) -> None:
        cfg = OTFlowConfig(
            device=torch.device("cpu"),
            levels=1,
            token_dim=1,
            history_len=2,
            hidden_dim=8,
            dropout=0.0,
            ctx_heads=1,
            ctx_layers=1,
            use_amp=False,
        )

        class DummyDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int):
                del idx
                return torch.zeros(2, 1), torch.ones(1), {"target_t": 2}

        class DummyModel:
            def sample_future_trace(self, hist, steps=None, solver=None, oracle_local_error=False):
                del solver, oracle_local_error
                width = int(steps) + 1
                trace = {
                    "time_grid": torch.linspace(0.0, 1.0, width, device=hist.device),
                    "oracle_local_error": torch.ones(hist.shape[0], int(steps), device=hist.device),
                }
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device), trace

        result = collect_batched_local_defect_trace(
            DummyModel(),
            DummyDataset(),
            cfg,
            solver_name="euler",
            reference_macro_steps=1,
            solver_order_p=1.0,
            seed=2**32 - 1,
            example_indices=[0],
            calibration_trace_samples=2,
        )
        self.assertEqual(result["trace_count"], 2)


if __name__ == "__main__":
    unittest.main()
