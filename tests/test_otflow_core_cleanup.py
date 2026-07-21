from __future__ import annotations

import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from genode.models.config import OTFlowConfig
from genode.data.otflow_paths import display_project_path
from genode.evaluation.otflow_evaluation_support import load_checkpoint_model
from genode.models.otflow_model import OTFlow
from genode.models.otflow_train_val import _temporary_eval_seed, save_json, seed_all
from genode.solver_protocol import FlowTrajectory


class OTFlowCoreCleanupTest(unittest.TestCase):
    def test_save_json_reports_a_sanitized_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "private-output.json"
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                save_json({"ok": True}, str(output_path))

            self.assertEqual(output_path.read_text(encoding="utf-8"), '{\n  "ok": true\n}')
            self.assertEqual(
                captured.getvalue().strip(),
                f"Saved JSON -> {display_project_path(output_path)}",
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

    def test_solve_supports_exactly_the_registered_solvers(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg)
        hist = torch.zeros(2, cfg.history_len, cfg.context_dim)
        initial_state = torch.zeros(2, cfg.sample_state_dim)

        for solver_key, target_nfe, time_grid in (
            ("euler", 4, torch.linspace(0.0, 1.0, 5)),
            ("dpmpp2m", 4, torch.linspace(0.0, 1.0, 5)),
            ("heun", 4, torch.linspace(0.0, 1.0, 3)),
            ("midpoint_rk2", 4, torch.linspace(0.0, 1.0, 3)),
        ):
            field_evaluations = 0

            def constant_field(x, t, hist_arg, cond=None, conditioning_cache=None):
                nonlocal field_evaluations
                del t, hist_arg, cond, conditioning_cache
                field_evaluations += 1
                return torch.ones_like(x)

            model.v_forward = constant_field  # type: ignore[method-assign]
            with self.subTest(solver_key=solver_key):
                result = model.solve(
                    initial_state,
                    hist,
                    solver_key=solver_key,
                    target_nfe=target_nfe,
                    time_grid=time_grid,
                    return_trajectory=True,
                )
                self.assertIsInstance(result, FlowTrajectory)
                assert isinstance(result, FlowTrajectory)
                self.assertEqual(result.solver_key, solver_key)
                self.assertEqual(result.target_nfe, target_nfe)
                self.assertEqual(result.realized_nfe, target_nfe)
                self.assertEqual(result.macro_steps, len(time_grid) - 1)
                self.assertEqual(
                    tuple(result.states.shape),
                    (2, len(time_grid), cfg.sample_state_dim),
                )
                self.assertTrue(torch.equal(result.initial_state, initial_state))
                self.assertTrue(torch.allclose(result.final_state, initial_state + 1.0))
                self.assertTrue(torch.equal(result.final_state, result.states[:, -1]))
                self.assertEqual(field_evaluations, target_nfe)

    def test_solve_matches_linear_field_updates(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg)
        hist = torch.zeros(1, cfg.history_len, cfg.context_dim)
        initial_state = torch.ones(1, cfg.sample_state_dim)

        def linear_field(x, t, hist_arg, cond=None, conditioning_cache=None):
            del t, hist_arg, cond, conditioning_cache
            return x

        model.v_forward = linear_field  # type: ignore[method-assign]
        expected = {
            "euler": 2.44140625,
            "dpmpp2m": 2.59912109375,
            "heun": 2.640625,
            "midpoint_rk2": 2.640625,
        }
        for solver_key, expected_value in expected.items():
            target_nfe = 4
            macro_steps = 2 if solver_key in {"heun", "midpoint_rk2"} else 4
            result = model.solve(
                initial_state,
                hist,
                solver_key=solver_key,
                target_nfe=target_nfe,
                time_grid=torch.linspace(0.0, 1.0, macro_steps + 1),
            )
            assert isinstance(result, torch.Tensor)
            self.assertTrue(
                torch.allclose(result, torch.full_like(result, expected_value), atol=1e-7),
                msg=solver_key,
            )

    def test_sample_draws_noise_then_delegates_to_solve(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg).eval()
        hist = torch.randn(2, cfg.history_len, cfg.context_dim)
        time_grid = torch.linspace(0.0, 1.0, 5)

        torch.manual_seed(41)
        initial_state = torch.randn(2, cfg.sample_state_dim)
        expected = model.solve(
            initial_state,
            hist,
            solver_key="euler",
            target_nfe=4,
            time_grid=time_grid,
        )
        torch.manual_seed(41)
        actual = model.sample(
            hist,
            solver_key="euler",
            target_nfe=4,
            time_grid=time_grid,
        )

        assert isinstance(expected, torch.Tensor)
        self.assertTrue(torch.equal(actual, expected))

    def test_solve_accepts_precomputed_conditioning_without_history(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg).eval()
        hist = torch.randn(2, cfg.history_len, cfg.context_dim)
        initial_state = torch.randn(2, cfg.sample_state_dim)
        cache = model.backbone.precompute(hist)
        kwargs = {
            "solver_key": "midpoint_rk2",
            "target_nfe": 4,
            "time_grid": (0.0, 0.5, 1.0),
        }

        with_history = model.solve(initial_state, hist, **kwargs)
        with_cache = model.solve(initial_state, conditioning_cache=cache, **kwargs)

        assert isinstance(with_history, torch.Tensor)
        assert isinstance(with_cache, torch.Tensor)
        self.assertTrue(torch.allclose(with_history, with_cache, atol=1e-6, rtol=1e-6))

    def test_solve_applies_guidance_with_cache_only_conditioning(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False)
        cfg.model.cond_dim = 3
        cfg.sample.cfg_scale = 2.0
        model = OTFlow(cfg).eval()
        hist = torch.randn(2, cfg.history_len, cfg.context_dim)
        condition = torch.randn(2, cfg.model.cond_dim)
        cache = model.backbone.precompute(hist, cond=condition)
        initial_state = torch.zeros(2, cfg.sample_state_dim)

        actual = model.solve(
            initial_state,
            conditioning_cache=cache,
            solver_key="euler",
            target_nfe=2,
            time_grid=(0.0, 0.5, 1.0),
        )
        assert isinstance(actual, torch.Tensor)
        self.assertTrue(torch.isfinite(actual).all())

        def cache_sensitive_field(x, t, hist_arg, cond=None, conditioning_cache=None):
            del t, hist_arg, cond
            assert conditioning_cache is not None
            value = 1.0 if conditioning_cache.cond_emb is not None else 0.0
            return torch.full_like(x, value)

        model.v_forward = cache_sensitive_field  # type: ignore[method-assign]
        result = model.solve(
            initial_state,
            conditioning_cache=cache,
            solver_key="euler",
            target_nfe=2,
            time_grid=(0.0, 0.5, 1.0),
        )

        assert isinstance(result, torch.Tensor)
        self.assertTrue(torch.equal(result, torch.full_like(result, 2.0)))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model.solve(
                initial_state,
                hist,
                conditioning_cache=cache,
                solver_key="euler",
                target_nfe=2,
                time_grid=(0.0, 0.5, 1.0),
            )

    def test_solve_rejects_aliases_and_inconsistent_grids(self) -> None:
        cfg = self._cfg(use_minibatch_ot=False)
        model = OTFlow(cfg)
        hist = torch.zeros(1, cfg.history_len, cfg.context_dim)
        initial_state = torch.zeros(1, cfg.sample_state_dim)

        with self.assertRaisesRegex(ValueError, "Unknown solver_key"):
            model.solve(
                initial_state,
                hist,
                solver_key="dpm++2m",
                target_nfe=4,
                time_grid=torch.linspace(0.0, 1.0, 5),
            )
        with self.assertRaisesRegex(ValueError, r"macro_steps \+ 1"):
            model.solve(
                initial_state,
                hist,
                solver_key="heun",
                target_nfe=4,
                time_grid=torch.linspace(0.0, 1.0, 5),
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            model.solve(
                initial_state,
                hist,
                solver_key="euler",
                target_nfe=2,
                time_grid=(0.0, 1.0, 1.0),
            )

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

    def test_checkpoint_loader_migrates_retired_inference_only_sample_fields(self) -> None:
        torch.manual_seed(4)
        cfg = self._cfg(use_minibatch_ot=True)
        model = OTFlow(cfg)
        cfg_dict = cfg.to_dict()
        cfg_dict["sample"].update(
            {
                "steps": 8,
                "solver": "heun",
                "time_grid": (0.0, 0.5, 1.0),
                "adaptive_beta": 0.9,
                "adaptive_tau": 0.15,
                "adaptive_kappa": 12.0,
                "adaptive_gamma_max": 0.05,
                "adaptive_cooldown_steps": 0,
                "adaptive_noise_mode": "orthogonal",
                "adaptive_trigger_mode": "adaptive",
                "adaptive_disable_noise_frac": 0.1,
                "adaptive_rtol": 1e-3,
                "adaptive_atol": 1e-6,
                "adaptive_safety": 0.9,
                "adaptive_min_step": 1e-5,
                "adaptive_max_nfe": 512,
                "refine_beta": 0.9,
                "refine_trigger_mode": "zscore",
                "refine_threshold_z": 1.5,
                "refine_threshold_raw": 0.0,
                "refine_step_mu": (),
                "refine_step_sigma": (),
                "refine_step_threshold": (),
                "refine_selected_steps": (),
                "refine_fixed_last_k": 0,
                "refine_sigma_eps": 1e-6,
                "refine_disallow_final_step": True,
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model.pt"
            torch.save({"cfg": cfg_dict, "model_state": model.state_dict()}, ckpt_path)
            loaded, loaded_cfg = load_checkpoint_model(ckpt_path, torch.device("cpu"))

        self.assertIsInstance(loaded, OTFlow)
        self.assertEqual(loaded_cfg.sample.cfg_scale, cfg.sample.cfg_scale)

    def test_checkpoint_loader_rejects_removed_baseline_config_keys(self) -> None:
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
            with self.assertRaisesRegex(ValueError, "retired baseline keys"):
                load_checkpoint_model(ckpt_path, torch.device("cpu"))

if __name__ == "__main__":
    unittest.main()
