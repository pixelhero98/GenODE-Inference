from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import genode.evaluation.otflow_evaluation_support as eval_support
from genode.data.otflow_datasets import build_dataset_splits_from_arrays
from genode.evaluation.fm_backbone_registry import BACKBONE_NAME_OTFLOW, FORECAST_FAMILY
from genode.models.conditioning import FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL, frozen_backbone_policy_context
from genode.models.config import OTFlowConfig
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


class SequenceConditioningTests(unittest.TestCase):
    def test_frozen_policy_context_is_exact_projected_summary_without_condition(self) -> None:
        cfg = _tiny_cfg(cond_dim=0)
        backbone = OTFlow(cfg).backbone.eval()
        hist = torch.randn(2, cfg.history_len, cfg.context_dim, requires_grad=True)
        with torch.no_grad():
            expected = backbone.precompute(hist, cond=None).summary
        actual = frozen_backbone_policy_context(
            backbone, hist, protocol=FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL, expected_width=cfg.model.hidden_dim
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(actual.requires_grad)

    def test_frozen_policy_context_appends_exact_backbone_condition_embedding(self) -> None:
        cfg = _tiny_cfg(cond_dim=5)
        backbone = OTFlow(cfg).backbone.eval()
        hist = torch.randn(2, cfg.history_len, cfg.context_dim, requires_grad=True)
        cond = torch.randn(2, cfg.model.cond_dim, requires_grad=True)
        with torch.no_grad():
            cache = backbone.precompute(hist, cond=cond)
            expected = torch.cat([cache.summary, cache.cond_emb], dim=-1)
        actual = frozen_backbone_policy_context(
            backbone,
            hist,
            cond=cond,
            protocol=FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL,
            expected_width=2 * cfg.model.hidden_dim,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(actual.requires_grad)
        with self.assertRaisesRegex(ValueError, "Unsupported frozen-backbone"):
            frozen_backbone_policy_context(backbone, hist, cond=cond, protocol="legacy_context")
        with self.assertRaisesRegex(ValueError, "width mismatch"):
            frozen_backbone_policy_context(backbone, hist, cond=cond, expected_width=cfg.model.hidden_dim)

    def test_forecast_context_export_uses_canonical_frozen_backbone_api(self) -> None:
        cfg = _tiny_cfg(cond_dim=0)
        model = OTFlow(cfg)

        class TinyForecastDataset:
            horizon = 2
            mase_seasonal_period = 1

            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int):
                del idx
                return (
                    torch.zeros(cfg.history_len, cfg.context_dim),
                    torch.ones(cfg.snapshot_dim),
                    torch.ones(1, cfg.snapshot_dim),
                    {"target_t": cfg.history_len, "series_id": "series-0"},
                )

            def denormalize_block(self, block, idx: int):
                del idx
                return np.asarray(block, dtype=np.float32)

            def mase_denom(self, idx: int) -> float:
                del idx
                return 1.0

        canonical_context = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        with (
            mock.patch.object(
                eval_support, "frozen_backbone_policy_context", return_value=canonical_context
            ) as context_api,
            mock.patch.object(
                model, "sample_future", return_value=torch.zeros(1, cfg.prediction_horizon, cfg.snapshot_dim)
            ),
        ):
            result = eval_support.evaluate_forecast_schedule(
                model,
                TinyForecastDataset(),
                cfg,
                solver_name="euler",
                runtime_nfe=1,
                time_grid=(0.0, 1.0),
                num_eval_samples=1,
                seed=0,
                dataset_key="traffic_hourly",
                split_phase="validation_tuning",
                checkpoint_id="checkpoint",
                example_indices=[0],
                return_context_embeddings=True,
            )
        context_api.assert_called_once()
        self.assertIs(context_api.call_args.args[0], model.backbone)
        self.assertEqual(context_api.call_args.kwargs["protocol"], FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL)
        self.assertEqual(next(iter(result["context_embeddings"].values())), [1.0, 2.0, 3.0])
        self.assertEqual(result["context_embedding_protocol"], FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL)
        self.assertEqual(result["context_embedding_width"], 3)

    def test_manifest_metadata_path_uses_project_relative_resolution(self) -> None:
        with mock.patch.object(
            eval_support, "resolve_project_path", side_effect=lambda value: Path("/repo") / str(value)
        ):
            path = eval_support._metadata_path_for_checkpoint(
                {"metadata_path": "outputs/backbone_matrix/example/checkpoint_metadata.json"}, Path("/other/model.pt")
            )
        self.assertEqual(path, Path("/repo/outputs/backbone_matrix/example/checkpoint_metadata.json"))

    def test_forecast_manifest_checkpoint_branch_returns_manifest_train_steps(self) -> None:
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
                        "effective_train_steps": 7600,
                        "checkpoint_export_protocol": "best_validation_state_within_budget",
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
                    cli_args=SimpleNamespace(otflow_train_steps=8000),
                    dataset_root=root,
                    shared_backbone_root=root,
                    dataset="traffic_hourly",
                    device=torch.device("cpu"),
                )
        self.assertEqual(result["train_steps"], 8000)
        self.assertEqual(result["train_budget_label"], "8k")
        self.assertEqual(result["checkpoint_budget_steps"], 8000)
        self.assertEqual(result["effective_train_steps"], 7600)
        self.assertEqual(result["checkpoint_export_protocol"], "best_validation_state_within_budget")
        self.assertEqual(result["checkpoint_id"], "otflow_temporal_extrapolation_traffic_hourly_8k")

    def test_dataset_builder_updates_model_cond_dim_without_shadow_field(self) -> None:
        rng = np.random.default_rng(0)
        params = rng.normal(size=(80, 4)).astype(np.float32)
        mids = np.linspace(100.0, 101.0, 80, dtype=np.float32)
        cond = np.eye(5, dtype=np.float32)[np.arange(80) % 5]
        cfg = _tiny_cfg(cond_dim=0)
        splits = build_dataset_splits_from_arrays(params, mids, cfg, cond_raw_full=cond, train_frac=0.6, val_frac=0.2)
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

    def test_legacy_model_names_are_rejected(self) -> None:
        rng = np.random.default_rng(3)
        params = rng.normal(size=(80, 4)).astype(np.float32)
        mids = np.linspace(100.0, 101.0, 80, dtype=np.float32)
        cond = np.eye(5, dtype=np.float32)[np.arange(80) % 5]
        cfg = _tiny_cfg(cond_dim=0)
        splits = build_dataset_splits_from_arrays(params, mids, cfg, cond_raw_full=cond, train_frac=0.6, val_frac=0.2)
        with self.assertRaisesRegex(ValueError, "Only model_name='otflow' is supported"):
            train_loop(splits["train"], cfg, model_name="cgan", steps=1)


if __name__ == "__main__":
    unittest.main()
