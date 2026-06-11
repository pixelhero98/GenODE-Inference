from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from genode.training import train_backbone as train_backbone_module


def _args(**overrides):
    base = {
        "dataset": "solar_energy_10m",
        "dataset_root": ".",
        "device": "cpu",
        "steps": 2,
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "hidden_dim": 8,
        "ctx_encoder": "hybrid",
        "ctx_local_kernel": 3,
        "ctx_pool_scales": "2",
        "fu_net_type": "mlp",
        "fu_net_layers": 1,
        "fu_net_heads": 1,
        "grad_accum_steps": 1,
        "seed": 0,
        "stride_train": 1,
        "log_every": 1,
        "val_every": 1,
        "val_max_batches": None,
        "checkpoint_steps": "1,2",
        "prepare_data": False,
        "use_amp": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))
        self.step = 0

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return {"weight": torch.tensor([float(self.step)])}


class TrainBackboneTests(unittest.TestCase):
    def test_forecast_cfg_uses_non_sf_experiment_spec(self) -> None:
        cfg = train_backbone_module.build_forecast_cfg(_args(dataset="solar_energy_10m", steps=20_000))
        self.assertEqual(cfg.history_len, 1008)
        self.assertEqual(cfg.future_block_len, 1008)
        self.assertEqual(cfg.steps, 20_000)

    def test_training_rejects_non_forecast_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "forecast datasets only"):
            train_backbone_module._forecast_spec("cryptos")

    def test_train_backbone_exports_validation_best_budget_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            fake_model = _FakeModel()

            def fake_train_loop(ds, cfg, *, model_name, steps, log_every, on_step):
                del ds, cfg, model_name, log_every
                for step in range(1, int(steps) + 1):
                    fake_model.step = step
                    on_step(step, fake_model, float(10 - step), {"loss": float(10 - step)})
                return fake_model

            def fake_eval(ds, model, cfg, **kwargs):
                del ds, cfg, kwargs
                return {"loss": {1: 5.0, 2: 4.0}[int(model.step)], "examples": 3, "batches": 1}

            with patch.object(train_backbone_module, "ensure_forecast_dataset"), patch.object(
                train_backbone_module,
                "build_monash_forecast_splits",
                return_value={
                    "train": object(),
                    "val": object(),
                    "stats": {"history_len": 1008, "n_val_examples": 3},
                },
            ), patch.object(train_backbone_module, "train_loop", side_effect=fake_train_loop), patch.object(
                train_backbone_module,
                "evaluate_average_loss",
                side_effect=fake_eval,
            ), patch.object(
                train_backbone_module,
                "project_backbone_matrix_root",
                return_value=matrix_root,
            ), patch.object(
                train_backbone_module,
                "materialize_backbone_manifest",
                return_value={"ready_count": 2},
            ):
                summary = train_backbone_module.train_backbone(_args(dataset_root=tmpdir))

            self.assertEqual(summary["checkpoint_steps"], [1, 2])
            for steps, selected_step in ((1, 1), (2, 2)):
                artifact_root = matrix_root / "otflow" / "temporal_extrapolation" / f"{steps}_steps" / "solar_energy_10m"
                metadata = json.loads((artifact_root / "checkpoint_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["dataset_key"], "solar_energy_10m")
                self.assertEqual(metadata["train_steps"], steps)
                self.assertEqual(metadata["train_budget_label"], f"{steps}_steps")
                self.assertEqual(metadata["cfg"]["train"]["steps"], steps)
                self.assertEqual(metadata["selection"]["selection_metric"], "validation_loss")
                self.assertEqual(metadata["selection"]["selected_step"], selected_step)


if __name__ == "__main__":
    unittest.main()
