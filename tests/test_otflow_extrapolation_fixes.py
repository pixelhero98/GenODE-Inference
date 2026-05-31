from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from genode.models.config import OTFlowConfig
from genode.data.otflow_datasets import WindowedParamSequenceDataset
from genode.evaluation.otflow_evaluation_support import (
    choose_forecast_example_indices,
    collect_forecast_calibration,
    evaluate_forecast_schedule,
    parse_forecast_datasets,
)
from genode.data.otflow_forecast_data import ForecastExampleRef, ForecastSeriesRecord, MonashForecastWindowDataset, _regular_time_features
from genode.data.otflow_medical_constants import LONG_TERM_HEADERED_ECG_DATASET_KEY, default_long_term_headered_ecg_manifest_path
from genode.data.otflow_medical_datasets import prepare_long_term_headered_ecg_dataset


class ExtrapolationFixesTest(unittest.TestCase):
    def _forecast_record(self, *, time_feature_mode: str) -> ForecastSeriesRecord:
        raw = np.arange(12, dtype=np.float32)
        return ForecastSeriesRecord(
            dataset_key="dummy",
            series_id="series_0",
            raw_values=raw,
            norm_values=raw[:, None],
            time_features=_regular_time_features(12, 1, time_feature_mode=time_feature_mode),
            mean=0.0,
            std=1.0,
            total_length=12,
            train_prefix_end=8,
            val_start=8,
            test_start=10,
        )

    def test_forecast_time_feature_modes_match_context_width(self) -> None:
        refs = [ForecastExampleRef(series_idx=0, target_t=8)]
        expected_extra = {"none": 0, "gap_only": 1, "gap_elapsed": 2}
        for mode, extra_dim in expected_extra.items():
            ds = MonashForecastWindowDataset(
                dataset_key="dummy",
                split_name="val",
                history_len=4,
                horizon=2,
                series_records=[self._forecast_record(time_feature_mode=mode)],
                example_refs=refs,
                time_feature_mode=mode,
            )
            hist, _, _, _ = ds[0]
            self.assertEqual(hist.shape[-1], 1 + extra_dim)

    def test_time_feature_modes_reject_ambiguous_configuration(self) -> None:
        cfg = OTFlowConfig(use_time_features=True, use_time_gaps=True)
        with self.assertRaisesRegex(ValueError, "exactly one mode"):
            _ = cfg.context_dim

    def test_windowed_dataset_includes_last_valid_target(self) -> None:
        params = np.arange(10, dtype=np.float32)[:, None]
        mids = np.arange(10, dtype=np.float32)
        ds_one_step = WindowedParamSequenceDataset(params, mids, history_len=3, future_horizon=0)
        self.assertEqual(ds_one_step.start_indices[-1], 9)
        ds_two_step = WindowedParamSequenceDataset(params, mids, history_len=3, future_horizon=1)
        self.assertEqual(ds_two_step.start_indices[-1], 8)

    def test_parse_forecast_datasets_rejects_high_level_ecg_until_checkpoint_is_supported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown forecast datasets"):
            parse_forecast_datasets(f"electricity,{LONG_TERM_HEADERED_ECG_DATASET_KEY}")
        with self.assertRaisesRegex(ValueError, "Unknown forecast datasets"):
            parse_forecast_datasets("electricity,not_a_dataset")

    def test_stale_ecg_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = default_long_term_headered_ecg_manifest_path(Path(tmpdir))
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_key": LONG_TERM_HEADERED_ECG_DATASET_KEY,
                        "context_length": 2000,
                        "official_horizon": 1000,
                        "series_specs": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Existing ECG manifest does not match"):
                prepare_long_term_headered_ecg_dataset(Path(tmpdir), history_len=4, horizon=2)

    def test_horizon_one_forecast_schedule_uses_target_without_future_tuple(self) -> None:
        cfg = OTFlowConfig(
            device=torch.device("cpu"),
            levels=1,
            token_dim=1,
            history_len=2,
            hidden_dim=8,
            dropout=0.0,
            ctx_heads=1,
            ctx_layers=1,
            rollout_mode="autoregressive",
            future_block_len=1,
            use_amp=False,
        )

        class DummyDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int):
                del idx
                return torch.zeros(2, 1), torch.ones(1), {"target_t": 2}

            def denormalize_block(self, block, idx: int):
                del idx
                return np.asarray(block, dtype=np.float32)

            def mase_denom(self, idx: int) -> float:
                del idx
                return 1.0

        class DummyModel:
            def __init__(self, cfg):
                self.cfg = cfg

            def sample_future(self, hist, steps=None, solver=None):
                del steps, solver
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device)

        metrics = evaluate_forecast_schedule(
            DummyModel(cfg),
            DummyDataset(),
            cfg,
            solver_name="euler",
            runtime_nfe=1,
            time_grid=(0.0, 1.0),
            num_eval_samples=1,
            seed=0,
        )
        self.assertEqual(metrics["eval_examples"], 1)
        self.assertTrue(np.isfinite(metrics["mse"]))

    def test_forecast_schedule_uses_deterministic_example_subset(self) -> None:
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
            horizon = 1

            def __len__(self) -> int:
                return 5

            def __getitem__(self, idx: int):
                return torch.zeros(2, 1), torch.tensor([float(idx)]), {"target_t": int(idx)}

            def denormalize_block(self, block, idx: int):
                del idx
                return np.asarray(block, dtype=np.float32)

            def mase_denom(self, idx: int) -> float:
                del idx
                return 1.0

        class DummyModel:
            def __init__(self, cfg):
                self.cfg = cfg

            def sample_future(self, hist, steps=None, solver=None):
                del steps, solver
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device)

        chosen = choose_forecast_example_indices(DummyDataset(), n_examples=2, seed=7)
        metrics = evaluate_forecast_schedule(
            DummyModel(cfg),
            DummyDataset(),
            cfg,
            solver_name="euler",
            runtime_nfe=1,
            time_grid=(0.0, 1.0),
            num_eval_samples=1,
            seed=3,
            example_indices=chosen,
        )
        self.assertEqual(metrics["eval_examples"], 2)
        self.assertIn("chosen_examples_hash", metrics)

    def test_forecast_schedule_handles_chunk_seed_above_numpy_limit(self) -> None:
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
            horizon = 1

            def __len__(self) -> int:
                return 4296

            def __getitem__(self, idx: int):
                return torch.zeros(2, 1), torch.tensor([float(idx)]), {"target_t": int(idx)}

            def denormalize_block(self, block, idx: int):
                del idx
                return np.asarray(block, dtype=np.float32)

            def mase_denom(self, idx: int) -> float:
                del idx
                return 1.0

        class DummyModel:
            def __init__(self, cfg):
                self.cfg = cfg

            def sample_future(self, hist, steps=None, solver=None):
                del steps, solver
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device)

        metrics = evaluate_forecast_schedule(
            DummyModel(cfg),
            DummyDataset(),
            cfg,
            solver_name="euler",
            runtime_nfe=1,
            time_grid=(0.0, 1.0),
            num_eval_samples=1,
            seed=0,
            example_indices=np.arange(4296, dtype=np.int64),
            batch_size=4295,
        )
        self.assertEqual(metrics["eval_examples"], 4296)

    def test_collect_forecast_calibration_handles_sample_seed_above_numpy_limit(self) -> None:
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
                    "time_grid": torch.linspace(0.0, 1.0, width),
                    "disagreement": torch.ones(hist.shape[0], width, device=hist.device),
                    "residual_norm": torch.ones(hist.shape[0], width, device=hist.device),
                    "oracle_local_error": torch.ones(hist.shape[0], width, device=hist.device),
                }
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device), trace

        result = collect_forecast_calibration(
            DummyModel(),
            DummyDataset(),
            cfg,
            macro_steps=1,
            solver_name="euler",
            seed=2**32 - 1,
            calibration_trace_samples=2,
        )
        self.assertEqual(result["calibration_trace_samples"], 2)


if __name__ == "__main__":
    unittest.main()
