from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from genode.models.config import OTFlowConfig
from genode.solver_protocol import FlowDiagnostics, FlowTrajectory
from genode.models.otflow_train_val import _get_dataset_item_by_t
from genode.data.otflow_datasets import WindowedParamSequenceDataset
from genode.evaluation.otflow_evaluation_support import (
    choose_forecast_example_indices,
    collect_forecast_calibration,
    evaluate_forecast_schedule,
    parse_forecast_datasets,
)
from genode.data.otflow_forecast_data import ForecastExampleRef, ForecastSeriesRecord, MonashForecastWindowDataset, _regular_time_features


class ExtrapolationTests(unittest.TestCase):
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

    def test_dataset_item_lookup_caches_start_indices(self) -> None:
        params = np.arange(20, dtype=np.float32)[:, None]
        mids = np.arange(20, dtype=np.float32)
        ds = WindowedParamSequenceDataset(params, mids, history_len=3, future_horizon=1)

        first = _get_dataset_item_by_t(ds, int(ds.start_indices[2]))
        lookup = getattr(ds, "_start_index_lookup")
        second = _get_dataset_item_by_t(ds, int(ds.start_indices[-1]))

        self.assertEqual(first[-1]["t"], int(ds.start_indices[2]))
        self.assertEqual(second[-1]["t"], int(ds.start_indices[-1]))
        self.assertIs(getattr(ds, "_start_index_lookup"), lookup)

        ds.start_indices = ds.start_indices[3:]
        refreshed = _get_dataset_item_by_t(ds, int(ds.start_indices[0]))
        self.assertEqual(refreshed[-1]["t"], int(ds.start_indices[0]))
        self.assertIsNot(getattr(ds, "_start_index_lookup"), lookup)

    def test_parse_forecast_datasets_rejects_high_level_ecg_until_checkpoint_is_supported(self) -> None:
        retired_ecg_key = "long_term_headered_" + "ECG_records"
        with self.assertRaisesRegex(ValueError, "Unknown forecast datasets"):
            parse_forecast_datasets(f"traffic_hourly,{retired_ecg_key}")
        with self.assertRaisesRegex(ValueError, "Unknown forecast datasets"):
            parse_forecast_datasets("traffic_hourly,not_a_dataset")

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

            def sample_future(self, hist, *, solver_key, target_nfe, time_grid):
                del solver_key, target_nfe, time_grid
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device)

        metrics = evaluate_forecast_schedule(
            DummyModel(cfg),
            DummyDataset(),
            cfg,
            solver_name="euler",
            macro_steps=1,
            time_grid=(0.0, 1.0),
            num_eval_samples=1,
            seed=0,
        )
        self.assertEqual(metrics["eval_examples"], 1)
        self.assertTrue(np.isfinite(metrics["forecast_mse"]))

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

            def sample_future(self, hist, *, solver_key, target_nfe, time_grid):
                del solver_key, target_nfe, time_grid
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device)

        chosen = choose_forecast_example_indices(DummyDataset(), n_examples=2, seed=7)
        metrics = evaluate_forecast_schedule(
            DummyModel(cfg),
            DummyDataset(),
            cfg,
            solver_name="euler",
            macro_steps=1,
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

            def sample_future(self, hist, *, solver_key, target_nfe, time_grid):
                del solver_key, target_nfe, time_grid
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device)

        metrics = evaluate_forecast_schedule(
            DummyModel(cfg),
            DummyDataset(),
            cfg,
            solver_name="euler",
            macro_steps=1,
            time_grid=(0.0, 1.0),
            num_eval_samples=1,
            seed=0,
            example_indices=np.arange(4296, dtype=np.int64),
            batch_size=4295,
        )
        self.assertEqual(metrics["eval_examples"], 4296)

    def test_forecast_context_rows_keep_logical_and_sampling_seeds(self) -> None:
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
            dataset_key = "traffic_hourly"
            split_name = "validation_tuning"
            horizon = 1

            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int):
                del idx
                return (
                    torch.zeros(2, 1),
                    torch.tensor([1.0]),
                    {
                        "series_id": "series_0",
                        "series_idx": 0,
                        "target_t": 42,
                        "history_start": 40,
                        "history_stop": 42,
                    },
                )

            def denormalize_block(self, block, idx: int):
                del idx
                return np.asarray(block, dtype=np.float32)

            def mase_denom(self, idx: int) -> float:
                del idx
                return 1.0

        class DummyModel:
            def __init__(self, cfg):
                self.cfg = cfg

            def sample_future(self, hist, *, solver_key, target_nfe, time_grid):
                del solver_key, target_nfe, time_grid
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device)

        context_rows = []
        for target_nfe, evaluation_seed in ((4, 1003), (8, 2003), (12, 3003)):
            metrics = evaluate_forecast_schedule(
                DummyModel(cfg),
                DummyDataset(),
                cfg,
                solver_name="euler",
                macro_steps=target_nfe,
                target_nfe=target_nfe,
                time_grid=tuple(np.linspace(0.0, 1.0, target_nfe + 1).tolist()),
                num_eval_samples=2,
                seed=evaluation_seed,
                logical_seed=3,
                scheduler_key="uniform",
                scenario_key="traffic_hourly",
                split_phase="validation_tuning",
                checkpoint_id="ck",
                example_indices=np.asarray([0], dtype=np.int64),
                return_per_example_rows=True,
            )
            row = metrics["per_example_rows"][0]
            self.assertEqual(row["seed"], 3)
            self.assertEqual(row["logical_seed"], 3)
            self.assertEqual(row["evaluation_seed"], evaluation_seed)
            self.assertEqual(row["sample_seed_start"], evaluation_seed)
            self.assertEqual(json.loads(row["sample_seed_values_json"]), [evaluation_seed, evaluation_seed + 1_000_000])
            context_rows.append(row)

        self.assertEqual(len({row["row_signature"] for row in context_rows}), 3)

        repeated = evaluate_forecast_schedule(
            DummyModel(cfg),
            DummyDataset(),
            cfg,
            solver_name="euler",
            macro_steps=4,
            target_nfe=4,
            time_grid=tuple(np.linspace(0.0, 1.0, 5).tolist()),
            num_eval_samples=1,
            seed=9999,
            logical_seed=3,
            scheduler_key="uniform",
            scenario_key="traffic_hourly",
            split_phase="validation_tuning",
            checkpoint_id="ck",
            example_indices=np.asarray([0], dtype=np.int64),
            return_per_example_rows=True,
        )["per_example_rows"][0]
        self.assertEqual(repeated["row_signature"], context_rows[0]["row_signature"])

        from genode.gipo.policy import student_nfe_sequence_pairs

        self.assertEqual(student_nfe_sequence_pairs(context_rows), [(0, 1, 4.0), (1, 2, 4.0)])

    def test_forecast_context_seeds_are_indexed_and_schedule_order_independent(self) -> None:
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
            dataset_key = "traffic_hourly"
            split_name = "locked_test"
            horizon = 1

            def __len__(self) -> int:
                return 2

            def __getitem__(self, idx: int):
                return (
                    torch.zeros(2, 1),
                    torch.tensor([float(idx + 1)]),
                    {
                        "series_id": f"series_{idx}",
                        "series_idx": idx,
                        "target_t": 42 + idx,
                        "history_start": 40 + idx,
                        "history_stop": 42 + idx,
                    },
                )

            def denormalize_block(self, block, idx: int):
                del idx
                return np.asarray(block, dtype=np.float32)

            def mase_denom(self, idx: int) -> float:
                del idx
                return 1.0

        class DummyModel:
            def __init__(self, config):
                self.cfg = config

            def sample_future(self, hist, *, solver_key, target_nfe, time_grid):
                del solver_key, target_nfe, time_grid
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device)

        def seeds_by_schedule(schedule_order):
            observed = {}
            for scheduler_key in schedule_order:
                rows = evaluate_forecast_schedule(
                    DummyModel(cfg),
                    DummyDataset(),
                    cfg,
                    solver_name="euler",
                    macro_steps=1,
                    target_nfe=1,
                    time_grid=(0.0, 1.0),
                    num_eval_samples=1,
                    seed=37,
                    logical_seed=7,
                    scheduler_key=scheduler_key,
                    scenario_key="traffic_hourly",
                    split_phase="locked_test",
                    checkpoint_id="ck",
                    example_indices=np.asarray([0, 1], dtype=np.int64),
                    return_per_example_rows=True,
                )["per_example_rows"]
                observed[scheduler_key] = [int(row["evaluation_seed"]) for row in rows]
            return observed

        forward = seeds_by_schedule(("uniform", "gipo"))
        reversed_order = seeds_by_schedule(("gipo", "uniform"))
        self.assertEqual(forward, {"uniform": [37, 38], "gipo": [37, 38]})
        self.assertEqual(reversed_order, forward)

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
            def sample_future_with_diagnostics(
                self,
                hist,
                *,
                solver_key,
                target_nfe,
                time_grid,
                include_local_error=False,
            ):
                self.last_call = {
                    "solver_key": solver_key,
                    "target_nfe": target_nfe,
                    "time_grid": tuple(float(value) for value in time_grid),
                    "include_local_error": include_local_error,
                }
                grid = torch.as_tensor(time_grid, dtype=hist.dtype, device=hist.device)
                macro_steps = int(grid.numel()) - 1
                initial_state = torch.zeros(hist.shape[0], 1, dtype=hist.dtype, device=hist.device)
                states = initial_state[:, None, :].expand(-1, macro_steps + 1, -1).clone()
                trajectory = FlowTrajectory(
                    initial_state=initial_state,
                    time_grid=grid,
                    states=states,
                    final_state=states[:, -1],
                    solver_key=str(solver_key),
                    target_nfe=int(target_nfe),
                    macro_steps=macro_steps,
                    realized_nfe=int(target_nfe),
                )
                values = torch.ones(hist.shape[0], macro_steps, dtype=hist.dtype, device=hist.device)
                diagnostics = FlowDiagnostics(
                    trajectory=trajectory,
                    disagreement=values,
                    velocity_norm=values,
                    ema_velocity_norm=values,
                    residual_norm=values,
                    local_error=values if include_local_error else torch.zeros_like(values),
                    field_evals_by_step=values,
                    mean_field_evals_per_step=1.0,
                    mean_total_field_evals_per_rollout=float(macro_steps),
                )
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device), diagnostics

        model = DummyModel()
        result = collect_forecast_calibration(
            model,
            DummyDataset(),
            cfg,
            macro_steps=1,
            solver_name="euler",
            seed=2**32 - 1,
            calibration_trace_samples=2,
        )
        self.assertEqual(result["calibration_trace_samples"], 2)
        self.assertEqual(
            model.last_call,
            {
                "solver_key": "euler",
                "target_nfe": 1,
                "time_grid": (0.0, 1.0),
                "include_local_error": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
