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
from genode.models.config import OTFlowConfig
from genode.evaluation.fm_backbone_registry import (
    BACKBONE_NAME_OTFLOW,
    CONDITIONAL_GENERATION_FAMILY,
    FORECAST_FAMILY,
    materialize_backbone_manifest,
)
from genode.data.otflow_datasets import build_dataset_splits_from_arrays
from genode.evaluation.otflow_evaluation_support import load_conditional_generation_checkpoint_splits
from genode.data.otflow_medical_datasets import prepare_sleep_edf_dataset
from genode.models.otflow_model import OTFlow
from genode.models.otflow_train_val import _parse_batch, select_eval_window_starts, train_loop


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


class ConditionalGenerationFixesTest(unittest.TestCase):
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
                        "history_len": int(cfg.history_len),
                        "future_block_len": int(cfg.prediction_horizon),
                    }
                ),
                encoding="utf-8",
            )
            artifact = {
                "checkpoint_path": str(ckpt_path),
                "checkpoint_id": "otflow_forecast_traffic_hourly_8k",
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
        self.assertEqual(result["checkpoint_id"], "otflow_forecast_traffic_hourly_8k")

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
            artifact_dir = root / "conditional_generation" / "long_term_st" / "transformer"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"cfg": cfg.to_dict(), "model_state": model.state_dict()}, artifact_dir / "model.pt")
            (artifact_dir / "checkpoint_metadata.json").write_text(
                json.dumps(
                    {
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
            args = type("Args", (), {"backbone_manifest": "", "otflow_train_steps": 20000})()
            with self.assertRaisesRegex(RuntimeError, "model.cond_dim=0"):
                load_conditional_generation_checkpoint_splits(
                    cli_args=args,
                    shared_backbone_root=root,
                    dataset="long_term_st",
                    device=torch.device("cpu"),
                )

    def test_sleep_metadata_is_bound_to_requested_npz_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requested = root / "custom_sleep_edf.npz"
            requested.write_bytes(b"placeholder")
            requested.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "dataset_key": "sleep_edf",
                        "history_len": 12000,
                        "official_horizon": 3000,
                        "prepared_npz_path": str(root / "other_sleep_edf.npz"),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not match requested NPZ path"):
                prepare_sleep_edf_dataset(requested)

    def test_readiness_manifest_marks_conditional_checkpoint_without_conditional_state_invalid(self) -> None:
        cfg = _tiny_cfg(cond_dim=0)
        model = OTFlow(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            artifact_dir = matrix_root / "otflow" / "conditional_generation" / "20k" / "long_term_st" / "transformer"
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

    def test_sleep_window_selection_is_stage_stratified(self) -> None:
        rng = np.random.default_rng(2)
        params = rng.normal(size=(80, 4)).astype(np.float32)
        mids = np.linspace(100.0, 101.0, 80, dtype=np.float32)
        cond = np.eye(5, dtype=np.float32)[np.arange(80) % 5]
        cfg = _tiny_cfg(cond_dim=5)
        splits = build_dataset_splits_from_arrays(
            params,
            mids,
            cfg,
            cond_raw_full=cond,
            train_frac=0.6,
            val_frac=0.2,
            dataset_kind="sleep_edf",
            dataset_metadata={"stage_names": ["W", "N1", "N2", "N3", "REM"]},
        )

        chosen = select_eval_window_starts(splits["test"], horizon=2, n_windows=3, seed=5)
        stages = {int(np.argmax(splits["test"].cond[int(t0)])) for t0 in chosen.tolist()}
        self.assertEqual(stages, {0, 1, 2, 3, 4})

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
