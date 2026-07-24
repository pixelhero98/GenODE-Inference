from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import genode.evaluation.otflow_evaluation_support as eval_support
import genode.models.otflow_train_val as train_val
from genode.models.config import OTFlowConfig
from genode.evaluation.fm_backbone_registry import (
    BACKBONE_NAME_OTFLOW,
    CONDITIONAL_GENERATION_FAMILY,
    FORECAST_FAMILY,
    materialize_backbone_manifest,
)
from genode.data.otflow_datasets import L2FeatureMap, build_dataset_splits_from_arrays
from genode.evaluation.otflow_evaluation_support import load_conditional_generation_checkpoint_splits
from genode.models.otflow_model import OTFlow
from genode.models.otflow_train_val import _parse_batch, train_loop


def _tiny_cfg(*, cond_dim: int = 0) -> OTFlowConfig:
    return OTFlowConfig(
        device=torch.device("cpu"),
        levels=1,
        token_dim=4,
        history_len=4,
        hidden_dim=16,
        dropout=0.0,
        ctx_heads=4,
        ctx_layers=1,
        fu_net_layers=1,
        fu_net_heads=4,
        rollout_mode="non_ar",
        future_block_len=2,
        use_cond_features=True,
        cond_standardize=False,
        cond_dim=int(cond_dim),
        use_amp=False,
    )


class ConditionalGenerationTests(unittest.TestCase):
    def test_panel_w1_is_not_the_mean_of_singleton_estimands(self) -> None:
        def metric_series(ask_p, ask_v, bid_p, bid_v):
            del ask_v, bid_p, bid_v
            values = np.asarray(ask_p, dtype=np.float64).ravel()
            return {key: values for key in train_val.CORE_L2_STATS}

        def metric_row(gen_values, true_values):
            gen = np.asarray(gen_values, dtype=np.float64)
            true = np.asarray(true_values, dtype=np.float64)
            return {
                "seq": {
                    "gen": {
                        "ask_p": gen,
                        "ask_v": gen,
                        "bid_p": gen,
                        "bid_v": gen,
                    },
                    "true": {
                        "ask_p": true,
                        "ask_v": true,
                        "bid_p": true,
                        "bid_v": true,
                    },
                }
            }

        rows = [
            metric_row([2.0, 2.0], [0.0, 0.0]),
            metric_row([0.0, 0.0], [2.0, 2.0]),
        ]
        with mock.patch.object(train_val, "microstructure_series", side_effect=metric_series):
            panel = train_val._aggregate_core_l2_distribution_metrics(rows)
            singletons = [
                train_val._aggregate_core_l2_distribution_metrics([row])
                for row in rows
            ]

        self.assertAlmostEqual(panel["unconditional_w1"], 0.0)
        self.assertAlmostEqual(panel["conditional_w1"], 2.0, places=5)
        for singleton in singletons:
            self.assertAlmostEqual(
                singleton["unconditional_w1"],
                singleton["conditional_w1"],
            )
            self.assertGreater(singleton["unconditional_w1"], 1_000_000.0)

    def test_constant_class_tstr_is_reported_instead_of_missing(self) -> None:
        downstream = {
            "real_x": np.asarray(
                [[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.1, 0.0]],
                dtype=np.float32,
            ),
            "real_moves": np.asarray([-3.0, 0.0, 3.0, 0.1], dtype=np.float32),
            "gen_x": np.ones((4, 2), dtype=np.float32),
            "gen_moves": np.asarray([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        }
        distances = {
            "unconditional_w1": 0.1,
            "conditional_w1": 0.2,
            "u_l1": 0.3,
            "c_l1": 0.4,
            "unconditional_w1_by_stat": {"spread": 0.1},
            "conditional_w1_by_stat": {"spread": 0.2},
            "unconditional_l1_by_stat": {"spread": 0.3},
            "conditional_l1_by_stat": {"spread": 0.4},
            "stat_scales": {"spread": 1.0},
        }
        with (
            mock.patch.object(
                train_val,
                "_collect_downstream_examples",
                return_value=downstream,
            ),
            mock.patch.object(
                train_val,
                "_aggregate_core_l2_distribution_metrics",
                return_value=distances,
            ),
            mock.patch.object(
                train_val,
                "_train_small_discriminator_auc",
                return_value=0.5,
            ),
        ):
            metrics = train_val._evaluate_generation_main_metrics(
                [{}],
                SimpleNamespace(device=torch.device("cpu")),
                horizon=10,
                seed=7,
            )

        self.assertTrue(metrics["temporal_tstr_f1_applicable"])
        self.assertEqual(metrics["temporal_tstr_f1_status"], "constant_train_class_fallback")
        self.assertEqual(metrics["temporal_tstr_f1_train_class_count"], 1)
        self.assertEqual(metrics["temporal_tstr_f1_test_class_count"], 3)
        self.assertIsNotNone(metrics["temporal_tstr_f1"])
        self.assertAlmostEqual(float(metrics["temporal_tstr_f1"]), 2.0 / 15.0)
        self.assertAlmostEqual(
            float(metrics["score_main"]),
            (math.log1p(0.1) + math.log1p(0.2)) / 3.0,
        )

    def test_optional_classifier_failures_do_not_discard_distribution_metrics(self) -> None:
        downstream = {
            "real_x": np.asarray(
                [[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0]],
                dtype=np.float32,
            ),
            "real_moves": np.asarray([-3.0, 0.0, 3.0], dtype=np.float32),
            "gen_x": np.asarray(
                [[-2.0, 0.0], [0.1, 0.0], [2.0, 0.0]],
                dtype=np.float32,
            ),
            "gen_moves": np.asarray([-2.0, 0.1, 2.0], dtype=np.float32),
        }
        distances = {
            "unconditional_w1": 0.1,
            "conditional_w1": 0.2,
            "u_l1": 0.3,
            "c_l1": 0.4,
            "unconditional_w1_by_stat": {"spread": 0.1},
            "conditional_w1_by_stat": {"spread": 0.2},
            "unconditional_l1_by_stat": {"spread": 0.3},
            "conditional_l1_by_stat": {"spread": 0.4},
            "stat_scales": {"spread": 1.0},
        }
        with (
            mock.patch.object(
                train_val,
                "_collect_downstream_examples",
                return_value=downstream,
            ),
            mock.patch.object(
                train_val,
                "_aggregate_core_l2_distribution_metrics",
                return_value=distances,
            ),
            mock.patch.object(
                train_val,
                "_train_small_multiclass_mlp_f1",
                side_effect=RuntimeError("classifier failure"),
            ),
            mock.patch.object(
                train_val,
                "_train_small_discriminator_auc",
                side_effect=RuntimeError("discriminator failure"),
            ),
        ):
            metrics = train_val._evaluate_generation_main_metrics(
                [{}],
                SimpleNamespace(device=torch.device("cpu")),
                horizon=10,
                seed=7,
            )

        self.assertIsNone(metrics["temporal_tstr_f1"])
        self.assertEqual(metrics["temporal_tstr_f1_status"], "failed")
        self.assertAlmostEqual(float(metrics["temporal_uw1"]), 0.1)
        self.assertAlmostEqual(float(metrics["temporal_cw1"]), 0.2)
        self.assertTrue(np.isfinite(float(metrics["score_main"])))

    def test_optional_feature_preparation_failure_does_not_discard_distribution_metrics(
        self,
    ) -> None:
        distances = {
            "unconditional_w1": 0.1,
            "conditional_w1": 0.2,
            "u_l1": 0.3,
            "c_l1": 0.4,
            "unconditional_w1_by_stat": {"spread": 0.1},
            "conditional_w1_by_stat": {"spread": 0.2},
            "unconditional_l1_by_stat": {"spread": 0.3},
            "conditional_l1_by_stat": {"spread": 0.4},
            "stat_scales": {"spread": 1.0},
        }
        with (
            mock.patch.object(
                train_val,
                "_collect_downstream_examples",
                side_effect=RuntimeError("feature preparation failure"),
            ),
            mock.patch.object(
                train_val,
                "_aggregate_core_l2_distribution_metrics",
                return_value=distances,
            ),
        ):
            metrics = train_val._evaluate_generation_main_metrics(
                [{}],
                SimpleNamespace(device=torch.device("cpu")),
                horizon=10,
                seed=7,
            )

        self.assertIsNone(metrics["temporal_tstr_f1"])
        self.assertEqual(metrics["temporal_tstr_f1_status"], "failed")
        self.assertEqual(metrics["temporal_tstr_f1_train_class_count"], 0)
        self.assertEqual(metrics["temporal_tstr_f1_test_class_count"], 0)
        self.assertAlmostEqual(float(metrics["temporal_uw1"]), 0.1)
        self.assertAlmostEqual(float(metrics["temporal_cw1"]), 0.2)
        self.assertTrue(np.isfinite(float(metrics["score_main"])))

    def test_eval_many_windows_binds_each_context_to_its_grid_and_seed(self) -> None:
        calls = []

        def evaluate_one(ds, model, cfg, **kwargs):
            del ds, model, cfg
            calls.append(
                (
                    kwargs["t0"],
                    kwargs["seed"],
                    tuple(kwargs["time_grid"]),
                )
            )
            return {
                "cmp": {},
                "timing": {},
                "gen": {},
                "true": {},
                "horizon": {},
                "meta": {"t": kwargs["t0"]},
            }

        main_metrics = {
            "temporal_tstr_f1": 0.5,
            "temporal_tstr_f1_applicable": True,
            "temporal_tstr_f1_status": "trained",
            "temporal_tstr_f1_train_class_count": 3,
            "temporal_tstr_f1_test_class_count": 3,
            "disc_auc": 0.5,
            "disc_auc_gap": 0.0,
            "temporal_uw1": 0.1,
            "temporal_cw1": 0.2,
            "u_l1": 0.3,
            "c_l1": 0.4,
            "temporal_uw1_by_stat": {"spread": 0.1},
            "temporal_cw1_by_stat": {"spread": 0.2},
            "stat_scales": {"spread": 1.0},
            "score_main": 0.2,
            "label_horizon": 1,
            "n_examples_real": 2,
            "n_examples_gen": 2,
            "threshold_abs_move": 0.1,
        }
        ds = SimpleNamespace(dataset_kind="l2", dataset_metadata={})
        grids = ((0.0, 0.2, 1.0), (0.0, 0.8, 1.0))
        with (
            mock.patch.object(
                train_val,
                "_valid_eval_indices",
                return_value=np.asarray([10, 20], dtype=np.int64),
            ),
            mock.patch.object(
                train_val,
                "eval_one_window",
                side_effect=evaluate_one,
            ),
            mock.patch.object(
                train_val,
                "_evaluate_generation_main_metrics",
                return_value=main_metrics,
            ) as reduce_metrics,
        ):
            result = train_val.eval_many_windows(
                ds,
                object(),
                SimpleNamespace(),
                horizon=5,
                nfe=2,
                chosen_t0s=[10, 20],
                generation_seed_values=[700, 701],
                metrics_seed=700,
                solver_key="euler",
                time_grids=grids,
            )

        self.assertEqual(
            calls,
            [(10, 700, grids[0]), (20, 701, grids[1])],
        )
        self.assertEqual(reduce_metrics.call_count, 1)
        self.assertEqual(result["meta"]["generation_seed_values"], [700, 701])
        self.assertEqual(result["meta"]["time_grid_mode"], "per_window")

    def test_fixed_and_adaptive_panels_share_the_same_metric_estimator(self) -> None:
        calls = []

        def metric_series(ask_p, ask_v, bid_p, bid_v):
            del ask_v, bid_p, bid_v
            values = np.asarray(ask_p, dtype=np.float64).ravel()
            return {key: values for key in train_val.CORE_L2_STATS}

        def evaluate_one(ds, model, cfg, **kwargs):
            del ds, model, cfg
            target_t = int(kwargs["t0"])
            calls.append(
                (
                    target_t,
                    int(kwargs["seed"]),
                    tuple(kwargs["time_grid"]),
                )
            )
            if target_t == 10:
                gen_values = np.asarray([2.0, 2.0], dtype=np.float64)
                true_values = np.asarray([0.0, 0.0], dtype=np.float64)
            else:
                gen_values = np.asarray([0.0, 0.0], dtype=np.float64)
                true_values = np.asarray([2.0, 2.0], dtype=np.float64)
            return {
                "cmp": {},
                "timing": {},
                "gen": {},
                "true": {},
                "horizon": {},
                "meta": {"t": target_t},
                "seq": {
                    "gen": {
                        "ask_p": gen_values,
                        "ask_v": gen_values,
                        "bid_p": gen_values,
                        "bid_v": gen_values,
                    },
                    "true": {
                        "ask_p": true_values,
                        "ask_v": true_values,
                        "bid_p": true_values,
                        "bid_v": true_values,
                    },
                },
            }

        downstream = {
            "real_x": np.zeros((0, 1), dtype=np.float32),
            "real_moves": np.zeros(0, dtype=np.float32),
            "gen_x": np.zeros((0, 1), dtype=np.float32),
            "gen_moves": np.zeros(0, dtype=np.float32),
        }
        ds = SimpleNamespace(dataset_kind="l2", dataset_metadata={})
        cfg = SimpleNamespace(device=torch.device("cpu"))
        grid = (0.0, 0.5, 1.0)
        with (
            mock.patch.object(
                train_val,
                "_valid_eval_indices",
                return_value=np.asarray([10, 20], dtype=np.int64),
            ),
            mock.patch.object(
                train_val,
                "eval_one_window",
                side_effect=evaluate_one,
            ),
            mock.patch.object(
                train_val,
                "microstructure_series",
                side_effect=metric_series,
            ),
            mock.patch.object(
                train_val,
                "_collect_downstream_examples",
                return_value=downstream,
            ),
        ):
            fixed = train_val.eval_many_windows(
                ds,
                object(),
                cfg,
                horizon=5,
                nfe=2,
                chosen_t0s=[10, 20],
                generation_seed_base=100,
                metrics_seed=100,
                solver_key="euler",
                time_grid=grid,
            )
            adaptive = train_val.eval_many_windows(
                ds,
                object(),
                cfg,
                horizon=5,
                nfe=2,
                chosen_t0s=[10, 20],
                generation_seed_values=[100, 101],
                metrics_seed=100,
                solver_key="euler",
                time_grids=[grid, grid],
            )

        self.assertEqual(
            calls,
            [
                (10, 100, grid),
                (20, 101, grid),
                (10, 100, grid),
                (20, 101, grid),
            ],
        )
        self.assertEqual(fixed["meta"]["chosen_t0s"], adaptive["meta"]["chosen_t0s"])
        self.assertEqual(
            fixed["meta"]["generation_seed_values"],
            adaptive["meta"]["generation_seed_values"],
        )
        self.assertEqual(fixed["meta"]["time_grid_mode"], "shared")
        self.assertEqual(adaptive["meta"]["time_grid_mode"], "per_window")
        for metric_key in ("temporal_uw1", "temporal_cw1", "stat_scales"):
            with self.subTest(metric_key=metric_key):
                self.assertEqual(
                    fixed["cmp"]["main"][metric_key],
                    adaptive["cmp"]["main"][metric_key],
                )
        self.assertEqual(fixed["cmp"]["score_main"], adaptive["cmp"]["score_main"])
        for scale in fixed["cmp"]["main"]["stat_scales"].values():
            self.assertAlmostEqual(float(scale["mean"]), 1.000001)

    def test_l2_decode_rejects_nonfinite_and_overflowing_parameters(self) -> None:
        feature_map = L2FeatureMap(levels=1)
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            feature_map.decode_sequence(
                np.asarray([[0.0, np.nan, 0.0, 0.0]], dtype=np.float64),
                init_mid=100.0,
            )
        with self.assertRaisesRegex(FloatingPointError, "overflowed"):
            feature_map.decode_sequence(
                np.asarray([[0.0, 1_000.0, 0.0, 0.0]], dtype=np.float64),
                init_mid=100.0,
            )

    def test_manifest_metadata_path_uses_project_relative_resolution(self) -> None:
        with mock.patch.object(eval_support, "resolve_project_path", side_effect=lambda value: Path("/repo") / str(value)):
            path = eval_support._metadata_path_for_checkpoint(
                {"metadata_path": "outputs/backbone_matrix/example/checkpoint_metadata.json"},
                Path("/other/model.pt"),
            )
        self.assertEqual(path, Path("/repo/outputs/backbone_matrix/example/checkpoint_metadata.json"))

    def test_exact_budget_validation_rejects_manifest_metadata_conflict(self) -> None:
        metadata = {
            "train_steps": 8000,
            "checkpoint_budget_steps": 8000,
            "effective_train_steps": 8000,
            "checkpoint_export_protocol": "exact_budget_step_state",
        }
        with self.assertRaisesRegex(RuntimeError, "manifest and checkpoint metadata disagree"):
            eval_support._validate_exact_budget_export(
                checkpoint_path=Path("model.pt"),
                expected_step=8000,
                metadata=metadata,
                manifest_artifact={**metadata, "effective_train_steps": 7600},
            )

    def test_forecast_manifest_checkpoint_branch_returns_resolved_checkpoint_step(self) -> None:
        cfg = _tiny_cfg(cond_dim=0)
        cfg.apply_overrides(steps=8000)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt_path = root / "model.pt"
            ckpt_path.write_bytes(b"placeholder")
            (root / "checkpoint_metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_key": "traffic_hourly",
                        "benchmark_family": FORECAST_FAMILY,
                        "train_steps": 8000,
                        "checkpoint_budget_steps": 8000,
                        "effective_train_steps": 8000,
                        "checkpoint_export_protocol": "exact_budget_step_state",
                        "history_len": int(cfg.history_len),
                        "future_block_len": int(cfg.prediction_horizon),
                    }
                ),
                encoding="utf-8",
            )
            artifact = {
                "checkpoint_path": str(ckpt_path),
                "checkpoint_id": "otflow_temporal_extrapolation_traffic_hourly_8k",
                "train_steps": 8000,
                "train_budget_label": "8k",
                "backbone_name": BACKBONE_NAME_OTFLOW,
            }

            with (
                mock.patch.object(eval_support, "_resolved_manifest_artifact", return_value=artifact),
                mock.patch.object(eval_support, "_resolve_checkpoint_path", return_value=ckpt_path),
                mock.patch.object(eval_support, "load_checkpoint_model", return_value=(object(), cfg)),
                mock.patch.object(eval_support, "_validate_forecast_checkpoint_task"),
                mock.patch.object(eval_support, "_forecast_time_feature_mode", return_value="none"),
                mock.patch.object(eval_support, "build_monash_forecast_splits", return_value={"stats": {}}),
            ):
                result = eval_support.load_forecast_checkpoint_splits(
                    cli_args=SimpleNamespace(checkpoint_step=8000),
                    dataset_root=root,
                    shared_backbone_root=root,
                    dataset="traffic_hourly",
                    device=torch.device("cpu"),
                )

        self.assertEqual(result["checkpoint_step"], 8000)
        self.assertNotIn("train_steps", result)
        self.assertEqual(result["train_budget_label"], "8k")
        self.assertEqual(result["checkpoint_budget_steps"], 8000)
        self.assertEqual(result["effective_train_steps"], 8000)
        self.assertEqual(result["checkpoint_export_protocol"], "exact_budget_step_state")
        self.assertEqual(result["checkpoint_id"], "otflow_temporal_extrapolation_traffic_hourly_8k")

    def test_dataset_builder_updates_model_cond_dim_without_shadow_field(self) -> None:
        rng = np.random.default_rng(0)
        params = rng.normal(size=(80, 4)).astype(np.float32)
        mids = np.linspace(100.0, 101.0, 80, dtype=np.float32)
        cond = np.eye(5, dtype=np.float32)[np.arange(80) % 5]
        cfg = _tiny_cfg(cond_dim=0)

        splits = build_dataset_splits_from_arrays(
            params,
            mids,
            cfg,
            cond_raw_full=cond,
            train_frac=0.6,
            val_frac=0.2,
        )

        self.assertGreater(len(splits["train"]), 0)
        self.assertEqual(cfg.model.cond_dim, 5)
        self.assertNotIn("cond_dim", vars(cfg))
        model = OTFlow(cfg)
        self.assertIsNotNone(model.backbone.conditioner.cond_mlp)

    def test_dataset_builder_rejects_condition_width_mismatch(self) -> None:
        rng = np.random.default_rng(1)
        params = rng.normal(size=(80, 4)).astype(np.float32)
        mids = np.linspace(100.0, 101.0, 80, dtype=np.float32)
        cond = np.eye(5, dtype=np.float32)[np.arange(80) % 5]
        cfg = _tiny_cfg(cond_dim=4)

        with self.assertRaisesRegex(ValueError, "model.cond_dim=4"):
            build_dataset_splits_from_arrays(params, mids, cfg, cond_raw_full=cond)

    def test_parse_batch_distinguishes_batched_and_unbatched_future_from_condition(self) -> None:
        hist_b = torch.zeros(2, 4, 3)
        tgt_b = torch.zeros(2, 3)
        fut_b = torch.zeros(2, 5, 3)
        cond_b = torch.zeros(2, 5)
        meta = {"t": 4}
        self.assertIs(_parse_batch((hist_b, tgt_b, fut_b, meta))[2], fut_b)
        self.assertIs(_parse_batch((hist_b, tgt_b, cond_b, meta))[3], cond_b)

        hist = torch.zeros(4, 3)
        tgt = torch.zeros(3)
        fut = torch.zeros(5, 3)
        cond = torch.zeros(5)
        self.assertIs(_parse_batch((hist, tgt, fut, meta))[2], fut)
        self.assertIs(_parse_batch((hist, tgt, cond, meta))[3], cond)

    def test_loader_rejects_conditional_metadata_with_unconditional_checkpoint(self) -> None:
        cfg = _tiny_cfg(cond_dim=0)
        model = OTFlow(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / CONDITIONAL_GENERATION_FAMILY / "long_term_st" / "transformer"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"cfg": cfg.to_dict(), "model_state": model.state_dict()}, artifact_dir / "model.pt")
            (artifact_dir / "checkpoint_metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_key": "long_term_st",
                        "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                        "train_steps": 20000,
                        "checkpoint_budget_steps": 20000,
                        "effective_train_steps": 20000,
                        "checkpoint_export_protocol": "exact_budget_step_state",
                        "history_len": 12000,
                        "future_block_len": 3000,
                        "field_network_type": "transformer",
                        "split_stats": {"cond_dim": 5, "history_len": 12000},
                    }
                ),
                encoding="utf-8",
            )
            args = type("Args", (), {"backbone_manifest": "", "checkpoint_step": 20000})()
            with self.assertRaisesRegex(RuntimeError, "model.cond_dim=0"):
                load_conditional_generation_checkpoint_splits(
                    cli_args=args,
                    shared_backbone_root=root,
                    dataset="long_term_st",
                    device=torch.device("cpu"),
                )

    def test_readiness_manifest_marks_conditional_checkpoint_without_conditional_state_invalid(self) -> None:
        cfg = _tiny_cfg(cond_dim=0)
        model = OTFlow(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            artifact_dir = matrix_root / "otflow" / CONDITIONAL_GENERATION_FAMILY / "20k" / "long_term_st" / "transformer"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"cfg": cfg.to_dict(), "model_state": model.state_dict()}, artifact_dir / "model.pt")
            (artifact_dir / "checkpoint_metadata.json").write_text(
                json.dumps(
                    {
                        "checkpoint_id": "long_term_bad",
                        "dataset_key": "long_term_st",
                        "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                        "train_steps": 20000,
                        "history_len": 12000,
                        "future_block_len": 3000,
                        "field_network_type": "transformer",
                        "split_stats": {"cond_dim": 5, "history_len": 12000},
                    }
                ),
                encoding="utf-8",
            )

            payload = materialize_backbone_manifest(
                matrix_root=matrix_root,
                otflow_reuse_root=Path(tmpdir) / "reuse",
                imported_backbone_root=Path(tmpdir) / "imported",
                budget_steps=(20000,),
                write_path=Path(tmpdir) / "manifest.json",
            )

        long_term_rows = [
            row
            for row in payload["artifacts"]
            if row["backbone_name"] == BACKBONE_NAME_OTFLOW
            and row["benchmark_family"] == CONDITIONAL_GENERATION_FAMILY
            and row["dataset_key"] == "long_term_st"
        ]
        self.assertEqual(long_term_rows[0]["status"], "invalid")
        self.assertIn("metadata cond_dim=5", long_term_rows[0]["compatibility_error"])

    def test_legacy_model_names_are_rejected(self) -> None:
        rng = np.random.default_rng(3)
        params = rng.normal(size=(80, 4)).astype(np.float32)
        mids = np.linspace(100.0, 101.0, 80, dtype=np.float32)
        cond = np.eye(5, dtype=np.float32)[np.arange(80) % 5]
        cfg = _tiny_cfg(cond_dim=0)
        splits = build_dataset_splits_from_arrays(
            params,
            mids,
            cfg,
            cond_raw_full=cond,
            train_frac=0.6,
            val_frac=0.2,
        )

        with self.assertRaisesRegex(ValueError, "Only model_name='otflow' is supported"):
            train_loop(splits["train"], cfg, model_name="cgan", steps=1)


if __name__ == "__main__":
    unittest.main()
