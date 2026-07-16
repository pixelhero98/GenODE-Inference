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
        "scenario_key": "solar_energy_10m",
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
        "val_max_batches": None,
        "checkpoint_steps": "1,2",
        "training_state_out": "",
        "resume_training_state": "",
        "save_training_state_every": 0,
        "no_auto_resume_training_state": True,
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
        cfg = train_backbone_module.build_forecast_cfg(_args(scenario_key="solar_energy_10m", steps=20_000))
        self.assertEqual(cfg.history_len, 1008)
        self.assertEqual(cfg.future_block_len, 1008)
        self.assertEqual(cfg.steps, 20_000)

    def test_conditional_cfg_uses_scenario_default_batch_size(self) -> None:
        cfg = train_backbone_module.build_conditional_cfg(_args(scenario_key="long_term_st", batch_size=0, steps=20_000))
        self.assertEqual(cfg.history_len, 12000)
        self.assertEqual(cfg.future_block_len, 3000)
        self.assertEqual(cfg.batch_size, 2)

    def test_cli_uses_scenario_key_without_dataset_alias(self) -> None:
        parser = train_backbone_module.build_argparser()
        args = parser.parse_args(["--scenario_key", "cryptos"])
        self.assertEqual(args.scenario_key, "cryptos")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--dataset", "cryptos"])

    def test_train_backbone_exports_exact_budget_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            fake_model = _FakeModel()

            def fake_train_loop(ds, cfg, *, model_name, steps, log_every, on_step, **kwargs):
                del ds, cfg, model_name, log_every, kwargs
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

            self.assertNotIn("checkpoint_export_mode", summary)
            self.assertEqual(summary["checkpoint_export_protocol"], "exact_budget_step_state")
            self.assertEqual(summary["scenario_key"], "solar_energy_10m")
            self.assertNotIn("dataset", summary)
            for steps in (1, 2):
                artifact_root = matrix_root / "otflow" / "temporal_extrapolation" / f"{steps}_steps" / "solar_energy_10m"
                metadata = json.loads((artifact_root / "checkpoint_metadata.json").read_text(encoding="utf-8"))
                checkpoint = torch.load(artifact_root / "model.pt", map_location="cpu", weights_only=False)
                self.assertEqual(metadata["effective_train_steps"], steps)
                self.assertEqual(metadata["checkpoint_export_protocol"], "exact_budget_step_state")
                self.assertEqual(metadata["selection"]["selected_step"], steps)
                self.assertEqual(float(checkpoint["model_state"]["weight"].item()), float(steps))

    def test_train_backbone_validates_each_exact_budget_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            fake_model = _FakeModel()

            def fake_train_loop(ds, cfg, *, model_name, steps, log_every, on_step, **kwargs):
                del ds, cfg, model_name, log_every, kwargs
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
            for steps in (1, 2):
                artifact_root = matrix_root / "otflow" / "temporal_extrapolation" / f"{steps}_steps" / "solar_energy_10m"
                metadata = json.loads((artifact_root / "checkpoint_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["dataset_key"], "solar_energy_10m")
                self.assertEqual(metadata["train_steps"], steps)
                self.assertEqual(metadata["checkpoint_budget_steps"], steps)
                self.assertEqual(metadata["effective_train_steps"], steps)
                self.assertEqual(metadata["checkpoint_export_protocol"], "exact_budget_step_state")
                self.assertEqual(metadata["train_budget_label"], f"{steps}_steps")
                self.assertEqual(metadata["cfg"]["train"]["steps"], steps)
                self.assertEqual(metadata["selection"]["selection_metric"], "validation_loss")
                self.assertEqual(metadata["selection"]["selected_step"], steps)
                self.assertNotIn("fallback_error", metadata["selection"])

    def test_train_backbone_resumes_from_restart_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            state_path = Path(tmpdir) / "restart.pt"
            fake_model = _FakeModel()

            def fake_eval(ds, model, cfg, **kwargs):
                del ds, cfg, kwargs
                return {"loss": {1: 5.0, 2: 4.0}[int(model.step)], "examples": 3, "batches": 1}

            def interrupted_train_loop(ds, cfg, *, model_name, steps, log_every, on_step, on_training_state, **kwargs):
                del ds, cfg, model_name, steps, log_every, kwargs
                fake_model.step = 1
                on_step(1, fake_model, 9.0, {"loss": 9.0})
                optimizer = torch.optim.SGD(fake_model.parameters(), lr=0.1)
                scaler = torch.cuda.amp.GradScaler(enabled=False)
                on_training_state(
                    1,
                    fake_model,
                    optimizer,
                    None,
                    scaler,
                    {"loader_state": {"seed": 11, "epoch": 0, "batch_index": 1}},
                )
                raise RuntimeError("simulated timeout")

            resume_seen = {}

            def resumed_train_loop(ds, cfg, *, model_name, steps, log_every, on_step, start_step, initial_model_state, **kwargs):
                del ds, cfg, model_name, log_every
                resume_seen["start_step"] = int(start_step)
                resume_seen["weight"] = float(initial_model_state["weight"].item())
                on_training_state = kwargs["on_training_state"]
                for step in range(int(start_step) + 1, int(steps) + 1):
                    fake_model.step = step
                    on_step(step, fake_model, float(10 - step), {"loss": float(10 - step)})
                    optimizer = torch.optim.SGD(fake_model.parameters(), lr=0.1)
                    scaler = torch.cuda.amp.GradScaler(enabled=False)
                    on_training_state(
                        step,
                        fake_model,
                        optimizer,
                        None,
                        scaler,
                        {"loader_state": {"seed": 11, "epoch": 0, "batch_index": step}},
                    )
                return fake_model

            def run_with_patches(train_loop_side_effect, args):
                with patch.object(train_backbone_module, "ensure_forecast_dataset"), patch.object(
                    train_backbone_module,
                    "build_monash_forecast_splits",
                    return_value={
                        "train": object(),
                        "val": object(),
                        "stats": {"history_len": 1008, "n_val_examples": 3},
                    },
                ), patch.object(train_backbone_module, "evaluate_average_loss", side_effect=fake_eval), patch.object(
                    train_backbone_module,
                    "project_backbone_matrix_root",
                    return_value=matrix_root,
                ), patch.object(
                    train_backbone_module,
                    "materialize_backbone_manifest",
                    return_value={"ready_count": 2},
                ), patch.object(
                    train_backbone_module,
                    "train_loop",
                    side_effect=train_loop_side_effect,
                ):
                    return train_backbone_module.train_backbone(args)

            with self.assertRaisesRegex(RuntimeError, "simulated timeout"):
                run_with_patches(
                    interrupted_train_loop,
                    _args(
                        dataset_root=tmpdir,
                        training_state_out=str(state_path),
                        save_training_state_every=1,
                    ),
                )

            self.assertTrue(state_path.exists())

            summary = run_with_patches(
                resumed_train_loop,
                _args(
                    dataset_root=tmpdir,
                    training_state_out=str(state_path),
                    resume_training_state=str(state_path),
                    save_training_state_every=1,
                ),
            )

            self.assertTrue(summary["resumed_from_training_state"])
            self.assertEqual(summary["resume_start_step"], 1)
            self.assertEqual(resume_seen, {"start_step": 1, "weight": 1.0})
            artifact_root = matrix_root / "otflow" / "temporal_extrapolation" / "2_steps" / "solar_energy_10m"
            metadata = json.loads((artifact_root / "checkpoint_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["checkpoint_export_protocol"], "exact_budget_step_state")
            self.assertEqual(metadata["effective_train_steps"], 2)

    def test_train_backbone_rejects_resume_signature_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            state_path = Path(tmpdir) / "restart.pt"
            fake_model = _FakeModel()

            def fake_eval(ds, model, cfg, **kwargs):
                del ds, cfg, kwargs
                return {"loss": {1: 5.0, 2: 4.0}[int(model.step)], "examples": 3, "batches": 1}

            def interrupted_train_loop(ds, cfg, *, model_name, steps, log_every, on_step, on_training_state, **kwargs):
                del ds, cfg, model_name, steps, log_every, kwargs
                fake_model.step = 1
                on_step(1, fake_model, 9.0, {"loss": 9.0})
                optimizer = torch.optim.SGD(fake_model.parameters(), lr=0.1)
                scaler = torch.cuda.amp.GradScaler(enabled=False)
                on_training_state(
                    1,
                    fake_model,
                    optimizer,
                    None,
                    scaler,
                    {"loader_state": {"seed": 11, "epoch": 0, "batch_index": 1}},
                )
                raise RuntimeError("simulated timeout")

            def run_with_patches(train_loop_side_effect, args):
                with patch.object(train_backbone_module, "ensure_forecast_dataset"), patch.object(
                    train_backbone_module,
                    "build_monash_forecast_splits",
                    return_value={
                        "train": object(),
                        "val": object(),
                        "stats": {"history_len": 1008, "n_val_examples": 3},
                    },
                ), patch.object(train_backbone_module, "evaluate_average_loss", side_effect=fake_eval), patch.object(
                    train_backbone_module,
                    "project_backbone_matrix_root",
                    return_value=matrix_root,
                ), patch.object(
                    train_backbone_module,
                    "materialize_backbone_manifest",
                    return_value={"ready_count": 2},
                ), patch.object(
                    train_backbone_module,
                    "train_loop",
                    side_effect=train_loop_side_effect,
                ):
                    return train_backbone_module.train_backbone(args)

            with self.assertRaisesRegex(RuntimeError, "simulated timeout"):
                run_with_patches(
                    interrupted_train_loop,
                    _args(
                        dataset_root=tmpdir,
                        training_state_out=str(state_path),
                        save_training_state_every=1,
                    ),
                )

            with self.assertRaisesRegex(ValueError, "does not match this run signature"):
                run_with_patches(
                    lambda *args, **kwargs: fake_model,
                    _args(
                        dataset_root=tmpdir,
                        training_state_out=str(state_path),
                        resume_training_state=str(state_path),
                        val_max_batches=2,
                    ),
                )

    def test_train_backbone_propagates_validation_failure_without_publishing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            fake_model = _FakeModel()

            def fake_train_loop(ds, cfg, *, model_name, steps, log_every, on_step, **kwargs):
                del ds, cfg, model_name, steps, log_every, kwargs
                fake_model.step = 1
                on_step(1, fake_model, 9.0, {"loss": 9.0})
                return fake_model

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
                side_effect=RuntimeError("validation failed"),
            ), patch.object(
                train_backbone_module,
                "project_backbone_matrix_root",
                return_value=matrix_root,
            ), patch.object(train_backbone_module, "materialize_backbone_manifest") as materialize:
                with self.assertRaisesRegex(RuntimeError, "validation failed"):
                    train_backbone_module.train_backbone(_args(dataset_root=tmpdir, steps=1, checkpoint_steps="1"))

            artifact_root = matrix_root / "otflow" / "temporal_extrapolation" / "1_steps" / "solar_energy_10m"
            self.assertFalse(artifact_root.exists())
            materialize.assert_not_called()

    def test_checkpoint_export_mode_cli_is_removed(self) -> None:
        with self.assertRaises(SystemExit):
            train_backbone_module.build_argparser().parse_args(
                ["--checkpoint_export_mode", "exact_budget"]
            )
        with self.assertRaises(SystemExit):
            train_backbone_module.build_argparser().parse_args(["--val_every", "200"])

    def test_train_backbone_detects_missing_exported_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_root = Path(tmpdir) / "matrix"
            fake_model = _FakeModel()

            def fake_train_loop(ds, cfg, *, model_name, steps, log_every, on_step, **kwargs):
                del ds, cfg, model_name, log_every, kwargs
                for step in range(1, int(steps) + 1):
                    fake_model.step = step
                    on_step(step, fake_model, float(10 - step), {"loss": float(10 - step)})
                return fake_model

            def fake_eval(ds, model, cfg, **kwargs):
                del ds, cfg, kwargs
                return {"loss": {1: 5.0, 2: 4.0}[int(model.step)], "examples": 3, "batches": 1}

            original_save = train_backbone_module._save_backbone_artifact

            def save_then_remove_second_budget_model(**kwargs):
                metadata = original_save(**kwargs)
                if int(kwargs["budget_steps"]) == 2:
                    artifact_root = matrix_root / "otflow" / "temporal_extrapolation" / "2_steps" / "solar_energy_10m"
                    (artifact_root / "model.pt").unlink()
                return metadata

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
                "_save_backbone_artifact",
                side_effect=save_then_remove_second_budget_model,
            ), patch.object(
                train_backbone_module,
                "materialize_backbone_manifest",
                return_value={"ready_count": 2},
            ):
                with self.assertRaisesRegex(RuntimeError, "valid budget artifacts"):
                    train_backbone_module.train_backbone(_args(dataset_root=tmpdir))


if __name__ == "__main__":
    unittest.main()
