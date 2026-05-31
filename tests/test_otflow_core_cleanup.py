from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from genode.models.config import OTFlowConfig
from genode.evaluation.otflow_evaluation_support import load_checkpoint_model
from genode.models.otflow_model import OTFlow
from genode.models.otflow_train_val import _temporary_eval_seed, seed_all


class OTFlowCoreCleanupTest(unittest.TestCase):
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

    def test_adaptive_rk_trace_time_grid_uses_accepted_steps_only(self) -> None:
        torch.manual_seed(6)
        cfg = self._cfg(use_minibatch_ot=False)
        cfg.sample.adaptive_rtol = 1e-3
        cfg.sample.adaptive_atol = 1e-6
        model = OTFlow(cfg)
        hist = torch.randn(1, cfg.history_len, cfg.context_dim)
        errors = iter((2.0, 0.5, 0.5))

        def fake_step(x, t_cur, dt, field_fn, first_eval=None):
            del t_cur, field_fn, first_eval
            return x + float(dt), x, 1

        def fake_error(*args, **kwargs):
            del args, kwargs
            return next(errors)

        model._rk45_embedded_step = fake_step  # type: ignore[method-assign]
        model._adaptive_error_norm = fake_error  # type: ignore[method-assign]

        _, trace = model.sample_trace(hist, steps=1, solver="rk45_adaptive")

        self.assertEqual(trace["accepted_steps"], 2)
        self.assertEqual(trace["rejected_steps"], 1)
        self.assertEqual(trace["steps"], trace["accepted_steps"])
        self.assertEqual(len(trace["time_grid"]), trace["accepted_steps"] + 1)
        self.assertAlmostEqual(float(trace["time_grid"][0]), 0.0)
        self.assertAlmostEqual(float(trace["time_grid"][-1]), 1.0)
        self.assertTrue(torch.all(torch.diff(trace["time_grid"]) > 0))
        self.assertIn("trial_accepted", trace)
        self.assertEqual(len(trace["trial_accepted"]), 3)
        self.assertGreaterEqual(len(trace["trial_accepted"]), trace["steps"])
        self.assertFalse(bool(trace["trial_accepted"][0]))

    def test_adaptive_trigger_mode_none_disables_noise_fields(self) -> None:
        torch.manual_seed(7)
        cfg = self._cfg(use_minibatch_ot=False)
        cfg.sample.adaptive_trigger_mode = "none"
        cfg.sample.adaptive_gamma_max = 1.0
        cfg.sample.adaptive_disable_noise_frac = 0.0
        model = OTFlow(cfg)
        hist = torch.randn(2, cfg.history_len, cfg.context_dim)

        _, trace = model.sample_trace(hist, steps=3, solver="euler_adaptive")

        self.assertFalse(torch.any(trace["fired"]))
        self.assertTrue(torch.all(trace["trigger_strength"] == 0))
        self.assertTrue(torch.all(trace["gamma"] == 0))
        self.assertTrue(torch.all(trace["noise_norm"] == 0))
        self.assertTrue(torch.all(trace["trigger_eligible"] == 0))


if __name__ == "__main__":
    unittest.main()
