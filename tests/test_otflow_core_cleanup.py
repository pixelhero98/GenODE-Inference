from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from genode.data.otflow_datasets import L2FeatureMap, build_dataset_splits_from_arrays
from genode.evaluation.otflow_evaluation_support import (
    _forecast_example_detail_metadata,
    load_checkpoint_model,
    resolved_eval_windows,
)
from genode.models.config import OTFlowConfig
from genode.models.otflow_model import OTFLOW_TRACE_FIELDS, OTFlow
from genode.models.otflow_train_val import _temporary_eval_seed, seed_all


class OTFlowCoreCleanupTest(unittest.TestCase):
    def test_forecast_example_metadata_is_strict_and_row_metadata_wins(self) -> None:
        class Dataset:
            def example_metadata(self, example_idx: int):
                self.requested = example_idx
                return {
                    "dataset": "dataset-value",
                    "split_phase": "dataset-split",
                    "series_id": "dataset-series",
                    "series_idx": 3,
                    "target_t": 21,
                    "history_start": 1,
                }

        dataset = Dataset()
        detail = _forecast_example_detail_metadata(
            dataset,
            7,
            {
                "series_id": "row-series",
                "series_idx": 5,
                "target_t": 34,
                "history_start": 2,
            },
            dataset_key="requested-dataset",
            split_phase="locked_test",
        )
        self.assertEqual(dataset.requested, 7)
        self.assertEqual(
            detail,
            {
                "dataset": "requested-dataset",
                "split_phase": "locked_test",
                "example_idx": 7,
                "series_id": "row-series",
                "series_idx": 5,
                "target_t": 34,
                "history_start": 2,
                "history_stop": "",
                "target_stop": "",
            },
        )

        class RaisingDataset:
            def example_metadata(self, example_idx: int):
                del example_idx
                raise RuntimeError("metadata failure")

        with self.assertRaisesRegex(RuntimeError, "metadata failure"):
            _forecast_example_detail_metadata(
                RaisingDataset(),
                0,
                {},
                dataset_key="dataset",
                split_phase="locked_test",
            )

        class InvalidDataset:
            def example_metadata(self, example_idx: int):
                del example_idx
                return ["not", "a", "mapping"]

        with self.assertRaisesRegex(TypeError, "must return a mapping"):
            _forecast_example_detail_metadata(
                InvalidDataset(),
                0,
                {},
                dataset_key="dataset",
                split_phase="locked_test",
            )

    def test_resolved_eval_windows_rejects_unknown_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "split must be 'val' or 'test'"):
            resolved_eval_windows(SimpleNamespace(), "traffic_hourly", "training")

    def test_l2_feature_map_rejects_invalid_constructor_values(self) -> None:
        for levels in (0, -1, True, 1.5, "3"):
            with self.subTest(levels=levels), self.assertRaisesRegex(ValueError, "levels must be a positive integer"):
                L2FeatureMap(levels=levels)
        for eps in (0.0, -1.0, float("nan"), True, "not-a-number"):
            with self.subTest(eps=eps), self.assertRaisesRegex(ValueError, "eps must be finite and positive"):
                L2FeatureMap(eps=eps)

    def test_l2_feature_map_validates_every_input_shape_rank_and_value(self) -> None:
        feature_map = L2FeatureMap(levels=3)
        ask_p = np.asarray([[101.0, 102.0, 103.0], [101.5, 102.5, 103.5]], dtype=np.float32)
        bid_p = np.asarray([[99.0, 98.0, 97.0], [99.5, 98.5, 97.5]], dtype=np.float32)
        volume = np.ones((2, 3), dtype=np.float32)

        params, mids = feature_map.encode_sequence(ask_p, volume, bid_p, volume)
        self.assertEqual(params.shape, (2, 12))
        self.assertEqual(mids.shape, (2,))

        with self.assertRaisesRegex(ValueError, "ask_v shape"):
            feature_map.encode_sequence(ask_p, volume[:, :2], bid_p, volume[:, :2])
        with self.assertRaisesRegex(ValueError, "bid_p must be a rank-2 array"):
            feature_map.encode_sequence(ask_p, volume, bid_p.reshape(-1), volume)
        with self.assertRaisesRegex(ValueError, "ask_p must contain at least one row"):
            feature_map.encode_sequence(ask_p[:0], volume[:0], bid_p[:0], volume[:0])
        complex_prices = ask_p.astype(np.complex64)
        with self.assertRaisesRegex(ValueError, "ask_p must have a real numeric dtype"):
            feature_map.encode_sequence(complex_prices, volume, bid_p, volume)
        nonfinite = volume.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "bid_v must contain only finite values"):
            feature_map.encode_sequence(ask_p, volume, bid_p, nonfinite)
        with self.assertRaisesRegex(ValueError, "L2 input has 3 levels; expected 2"):
            L2FeatureMap(levels=2).encode_sequence(ask_p, volume, bid_p, volume)

    def test_l2_feature_map_decode_uses_explicit_dimension_and_finiteness_checks(self) -> None:
        feature_map = L2FeatureMap(levels=3)
        with self.assertRaisesRegex(ValueError, "params must be a rank-2 array"):
            feature_map.decode_sequence(np.zeros(12, dtype=np.float32), init_mid=100.0)
        with self.assertRaisesRegex(ValueError, "params must contain at least one row"):
            feature_map.decode_sequence(np.zeros((0, 12), dtype=np.float32), init_mid=100.0)
        with self.assertRaisesRegex(ValueError, "params must have a real numeric dtype"):
            feature_map.decode_sequence(np.zeros((2, 12), dtype=np.complex64), init_mid=100.0)
        with self.assertRaisesRegex(ValueError, r"expected exactly 4 \* levels = 12"):
            feature_map.decode_sequence(np.zeros((2, 8), dtype=np.float32), init_mid=100.0)
        nonfinite = np.zeros((2, 12), dtype=np.float32)
        nonfinite[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "params must contain only finite values"):
            feature_map.decode_sequence(nonfinite, init_mid=100.0)
        with self.assertRaisesRegex(ValueError, "init_mid must be finite"):
            feature_map.decode_sequence(np.zeros((2, 12), dtype=np.float32), init_mid=float("nan"))
        with self.assertRaisesRegex(ValueError, "init_mid must be finite"):
            feature_map.decode_sequence(np.zeros((2, 12), dtype=np.float32), init_mid="not-a-number")
        overflowing = np.zeros((2, 12), dtype=np.float32)
        overflowing[:, 6:] = 1_000.0
        with self.assertRaisesRegex(ValueError, "Decoded L2 values exceed"):
            feature_map.decode_sequence(overflowing, init_mid=100.0)

    def test_split_builder_rejects_state_width_before_constructing_datasets(self) -> None:
        cfg = self._cfg()
        with self.assertRaisesRegex(ValueError, "does not match cfg.snapshot_dim=8"):
            build_dataset_splits_from_arrays(
                np.zeros((64, 7), dtype=np.float32),
                np.zeros(64, dtype=np.float32),
                cfg,
            )
        with self.assertRaisesRegex(ValueError, "finite real numeric values"):
            build_dataset_splits_from_arrays(
                np.zeros((64, 8), dtype=np.complex64),
                np.zeros(64, dtype=np.float32),
                cfg,
            )
        with self.assertRaisesRegex(ValueError, "finite real numeric values"):
            build_dataset_splits_from_arrays(
                np.zeros((64, 8), dtype=np.float32),
                np.zeros(64, dtype=np.complex64),
                cfg,
            )

    def test_seed_all_normalizes_generated_seed_above_numpy_limit(self) -> None:
        seed_all(2**32 + 123)
        np_draw = float(np.random.random())
        torch_draw = torch.rand(1)

        seed_all(123)
        self.assertEqual(np_draw, float(np.random.random()))
        self.assertTrue(torch.equal(torch_draw, torch.rand(1)))

        seed_all(2**32 + 123)
        self.assertEqual(np_draw, float(np.random.random()))
        self.assertTrue(torch.equal(torch_draw, torch.rand(1)))

    def test_temporary_eval_seed_normalizes_large_seed_and_restores_state(self) -> None:
        random.seed(77)
        np.random.seed(77)
        torch.manual_seed(77)
        py_before = random.getstate()
        np_before = np.random.get_state()
        torch_before = torch.random.get_rng_state()

        with _temporary_eval_seed(2**32 + 123):
            py_draw = random.random()
            np_draw = float(np.random.random())
            torch_draw = torch.rand(1)

        py_after = random.getstate()
        np_after = np.random.get_state()
        torch_after = torch.random.get_rng_state()
        self.assertEqual(py_before, py_after)
        self.assertEqual(np_before[0], np_after[0])
        self.assertTrue(np.array_equal(np_before[1], np_after[1]))
        self.assertEqual(np_before[2:], np_after[2:])
        self.assertTrue(torch.equal(torch_before, torch_after))

        with _temporary_eval_seed(123):
            self.assertEqual(py_draw, random.random())
            self.assertEqual(np_draw, float(np.random.random()))
            self.assertTrue(torch.equal(torch_draw, torch.rand(1)))

    def _cfg(self, *, use_minibatch_ot: bool = True, rollout_mode: str = "autoregressive") -> OTFlowConfig:
        future_block_len = 2 if rollout_mode == "non_ar" else 1
        cfg = OTFlowConfig(
            device=torch.device("cpu"),
            levels=2,
            history_len=4,
            hidden_dim=16,
            dropout=0.0,
            ctx_heads=4,
            ctx_layers=1,
            fu_net_layers=1,
            fu_net_heads=4,
            rollout_mode=rollout_mode,
            future_block_len=future_block_len,
            use_minibatch_ot=use_minibatch_ot,
            use_amp=False,
        )
        return cfg

    def test_core_loss_logs_only_velocity_regression_terms(self) -> None:
        torch.manual_seed(0)
        cfg = self._cfg(use_minibatch_ot=True)
        model = OTFlow(cfg)
        hist = torch.randn(3, cfg.history_len, cfg.context_dim)
        x = torch.randn(3, cfg.snapshot_dim)

        loss, logs = model.loss(x, hist)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(logs), {"mean", "ot_cost", "ot_used", "loss"})
        self.assertEqual(logs["ot_used"], 1.0)

    def test_minibatch_ot_can_be_disabled(self) -> None:
        torch.manual_seed(1)
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg)
        hist = torch.randn(3, cfg.history_len, cfg.context_dim)
        x = torch.randn(3, cfg.snapshot_dim)

        _, logs = model.loss(x, hist)

        self.assertEqual(logs["ot_used"], 0.0)
        self.assertEqual(logs["ot_cost"], 0.0)

    def test_non_ar_future_block_loss_runs(self) -> None:
        torch.manual_seed(2)
        cfg = self._cfg(use_minibatch_ot=True, rollout_mode="non_ar")
        model = OTFlow(cfg)
        hist = torch.randn(2, cfg.history_len, cfg.context_dim)
        x = torch.randn(2, cfg.snapshot_dim)
        fut = torch.randn(2, 1, cfg.snapshot_dim)

        loss, logs = model.loss(x, hist, fut=fut)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(logs), {"mean", "ot_cost", "ot_used", "loss"})

    def test_canonical_samplers_report_realized_field_evaluations(self) -> None:
        cfg = self._cfg(use_minibatch_ot=True)
        model = OTFlow(cfg)
        hist = torch.randn(2, cfg.history_len, cfg.context_dim)

        for requested, canonical, evaluations_per_step in (
            ("euler", "euler", 1.0),
            ("dpm++2m", "dpmpp2m", 1.0),
            ("heun", "heun", 2.0),
            ("midpoint_rk2", "midpoint_rk2", 2.0),
        ):
            with self.subTest(solver=requested):
                torch.manual_seed(4)
                sample, trace = model.sample_trace(hist, steps=4, solver=requested)

                self.assertEqual(tuple(sample.shape), (2, cfg.snapshot_dim))
                self.assertEqual(tuple(trace), OTFLOW_TRACE_FIELDS)
                self.assertEqual(trace["solver"], canonical)
                self.assertEqual(trace["steps"], 4)
                expected = torch.full_like(trace["field_evals_by_step"], evaluations_per_step)
                self.assertTrue(torch.equal(trace["field_evals_by_step"], expected))
                self.assertEqual(trace["mean_total_field_evals_per_rollout"], 4 * evaluations_per_step)

    def test_noncanonical_solvers_are_rejected_at_model_boundary(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg)
        hist = torch.randn(1, cfg.history_len, cfg.context_dim)

        for solver in (
            "dopri5",
            "dopri5_adaptive",
            "rk45",
            "rk45_adaptive",
            "euler_adaptive",
            "euler_refine_half",
            "euler_refine_heun",
        ):
            with self.subTest(solver=solver), self.assertRaisesRegex(ValueError, "Unknown solver_key"):
                model.sample_trace(hist, steps=2, solver=solver)

    def test_checkpoint_loader_rejects_removed_otflow_keys(self) -> None:
        torch.manual_seed(3)
        cfg = self._cfg(use_minibatch_ot=True)
        model = OTFlow(cfg)
        cfg_dict = cfg.to_dict()
        cfg_dict["model"]["field_parameterization"] = "instantaneous"
        cfg_dict["fm"].update(
            {
                "lambda_mean": 1.0,
                "lambda_consistency": 0.0,
                "lambda_imbalance": 0.0,
                "lambda_causal_ot": 0.0,
                "lambda_current_match": 0.0,
                "lambda_path_fm": 0.0,
                "lambda_mi": 0.0,
                "lambda_mi_critic": 0.0,
                "meanflow_data_proportion": 0.75,
                "meanflow_norm_p": 1.0,
                "meanflow_norm_eps": 0.01,
            }
        )
        state = dict(model.state_dict())
        state["backbone.conditioner.h_mlp.net.0.weight"] = torch.zeros(16, 16)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model.pt"
            torch.save({"cfg": cfg_dict, "model_state": state}, ckpt_path)
            with self.assertRaisesRegex(TypeError, "unsupported keys"):
                load_checkpoint_model(ckpt_path, torch.device("cpu"))

    def test_checkpoint_loader_rejects_unexpected_state_keys(self) -> None:
        torch.manual_seed(4)
        cfg = self._cfg(use_minibatch_ot=True)
        model = OTFlow(cfg)
        state = dict(model.state_dict())
        state["backbone.conditioner.h_mlp.net.0.weight"] = torch.zeros(16, 16)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model.pt"
            torch.save({"cfg": cfg.to_dict(), "model_state": state}, ckpt_path)
            with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                load_checkpoint_model(ckpt_path, torch.device("cpu"))

    def test_checkpoint_loader_ignores_removed_baseline_config_keys(self) -> None:
        torch.manual_seed(5)
        cfg = self._cfg(use_minibatch_ot=True)
        model = OTFlow(cfg)
        cfg_dict = cfg.to_dict()
        cfg_dict["model"].update(
            {
                "baseline_latent_dim": 32,
                "vae_kl_weight": 0.1,
                "timegan_supervision_weight": 10.0,
                "timegan_moment_weight": 10.0,
                "kovae_pred_weight": 1.0,
                "kovae_ridge": 1e-3,
                "gan_noise_dim": 64,
                "cgan_recon_weight": 5.0,
            }
        )
        cfg_dict["nf"] = {"flow_layers": 6, "flow_scale_clip": 2.0, "share_coupling_backbone": True}

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model.pt"
            torch.save({"cfg": cfg_dict, "model_state": model.state_dict()}, ckpt_path)
            loaded, loaded_cfg = load_checkpoint_model(ckpt_path, torch.device("cpu"))

        self.assertIsInstance(loaded, OTFlow)
        self.assertFalse(hasattr(loaded_cfg.model, "baseline_latent_dim"))


if __name__ == "__main__":
    unittest.main()
