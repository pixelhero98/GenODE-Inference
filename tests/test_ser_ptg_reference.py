from __future__ import annotations

import unittest

import torch

from genode.gipo.ser_ptg_reference import collect_batched_local_defect_trace
from genode.models.config import OTFlowConfig


class SerPtgReferenceTests(unittest.TestCase):
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
