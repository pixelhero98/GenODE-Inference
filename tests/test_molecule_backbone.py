from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

from genode.data import molecule_xyz
from genode.evaluation import molecule_metrics
from genode.models.config import OTFlowConfig
from genode.models.otflow_train_val import generate_continuation
from genode.training import train_molecule_backbone as train_molecule_module


def _symbols(atom_count: int = 6) -> list[str]:
    carbon_count = max(1, int(atom_count) - 2)
    return ["C"] * carbon_count + ["H"] * (int(atom_count) - carbon_count)


def _write_xyz_zip(
    path: Path,
    *,
    dataset_key: str = "demo",
    categories: dict[str, tuple[int, int]],
    frames: int = 10,
) -> None:
    with ZipFile(path, "w") as zf:
        for category, (trajectory_count, atom_count) in categories.items():
            symbols = _symbols(atom_count)
            for trajectory_offset in range(trajectory_count):
                rows = []
                iso_id = 1000 + trajectory_offset
                for frame_idx in range(frames):
                    rows.append(str(atom_count))
                    rows.append("")
                    for atom_idx, symbol in enumerate(symbols):
                        x = 0.05 * atom_idx + 0.003 * frame_idx + 0.0001 * trajectory_offset
                        y = 0.02 * atom_idx - 0.002 * frame_idx
                        z = 0.03 * atom_idx + 0.001 * frame_idx
                        rows.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
                name = f"{category}/{dataset_key}_{category}_Iso{iso_id}.trj.xyz"
                zf.writestr(name, "\n".join(rows) + "\n")


class MoleculeBackboneTests(unittest.TestCase):
    def test_discovery_and_prepare_are_generic_and_path_scrubbed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            processed_root = root / "processed"
            _write_xyz_zip(
                zip_path,
                categories={
                    "Dynamic_Test": (8, 6),
                    "Direct_Test": (3, 8),
                },
                frames=8,
            )

            discovered = molecule_xyz.discover_molecule_xyz_strata(zip_path)
            self.assertEqual(discovered["Dynamic_Test"]["atom_count"], 6)
            self.assertFalse(discovered["Dynamic_Test"]["mixed_shape"])
            self.assertTrue(discovered["Dynamic_Test"]["trainable"])
            self.assertFalse(discovered["Direct_Test"]["trainable"])

            summary = molecule_xyz.prepare_molecule_xyz_all_strata(
                zip_path,
                processed_root,
                dataset_key="demo",
                include_pattern="Dynamic_*",
            )
            metadata = summary["strata"]["Dynamic_Test"]
            data = molecule_xyz.load_molecule_processed(
                processed_root / "Dynamic_Test",
                dataset_key="demo",
                stratum="Dynamic_Test",
            )

            self.assertEqual(metadata["dataset_key"], "demo")
            self.assertEqual(metadata["stratum"], "Dynamic_Test")
            self.assertEqual(metadata["xyz_count"], 8)
            self.assertEqual(metadata["atom_count"], 6)
            self.assertEqual(metadata["formula"], "C4H2")
            self.assertEqual(data.coords.shape, (8 * 8, 6, 3))
            self.assertFalse(Path(str(metadata["source_zip"])).is_absolute())
            self.assertNotIn(str(root), json.dumps(metadata))
            split_ids = metadata["split_trajectory_ids"]
            self.assertTrue(split_ids["train"])
            self.assertTrue(split_ids["val"])
            self.assertTrue(split_ids["test"])
            self.assertFalse(set(split_ids["train"]) & set(split_ids["val"]))
            self.assertFalse(set(split_ids["train"]) & set(split_ids["test"]))

    def test_dataset_shapes_splits_and_stride_independent_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            processed = root / "processed"
            _write_xyz_zip(zip_path, categories={"Dynamic_Test": (8, 6)}, frames=12)
            molecule_xyz.prepare_molecule_xyz_zip(
                zip_path,
                processed,
                dataset_key="demo",
                stratum="Dynamic_Test",
            )

            cfg = molecule_xyz.configure_molecule_otflow(
                OTFlowConfig(),
                history_len=2,
                future_horizon=1,
                rollout_mode="autoregressive",
                atom_count=6,
            )
            stride_one = molecule_xyz.build_molecule_dataset_splits(
                processed_dir=processed,
                cfg=cfg,
                history_len=2,
                future_horizon=1,
                stride_train=1,
                stride_eval=1,
                dataset_key="demo",
                stratum="Dynamic_Test",
            )
            stride_three = molecule_xyz.build_molecule_dataset_splits(
                processed_dir=processed,
                cfg=cfg,
                history_len=2,
                future_horizon=1,
                stride_train=3,
                stride_eval=1,
                dataset_key="demo",
                stratum="Dynamic_Test",
            )

            hist, tgt, meta = stride_one["train"][0]
            self.assertEqual(hist.shape, (2, 6 * molecule_xyz.MOLECULE_CONTEXT_ATOM_FEATURE_DIM))
            self.assertEqual(tgt.shape, (6 * 3,))
            self.assertIn("trajectory_id", meta)
            self.assertLess(len(stride_three["train"]), len(stride_one["train"]))
            np.testing.assert_allclose(stride_one["train"].stats.target_mean, stride_three["train"].stats.target_mean)
            np.testing.assert_allclose(stride_one["train"].stats.target_std, stride_three["train"].stats.target_std)
            split_ids = stride_one["data"].metadata["split_trajectory_ids"]
            self.assertFalse(set(split_ids["train"]) & set(split_ids["val"]))
            self.assertFalse(set(split_ids["train"]) & set(split_ids["test"]))

    def test_non_ar_dataset_shapes_without_segment_crossing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            processed = root / "processed"
            _write_xyz_zip(zip_path, categories={"Dynamic_Test": (8, 6)}, frames=12)
            molecule_xyz.prepare_molecule_xyz_zip(
                zip_path,
                processed,
                dataset_key="demo",
                stratum="Dynamic_Test",
            )
            cfg = molecule_xyz.configure_molecule_otflow(
                OTFlowConfig(),
                history_len=2,
                future_horizon=4,
                rollout_mode="non_ar",
                atom_count=6,
            )
            splits = molecule_xyz.build_molecule_dataset_splits(
                processed_dir=processed,
                cfg=cfg,
                history_len=2,
                future_horizon=4,
                stride_train=1,
                stride_eval=1,
                dataset_key="demo",
                stratum="Dynamic_Test",
            )
            hist, tgt, fut, _ = splits["train"][0]
            self.assertEqual(hist.shape, (2, 6 * molecule_xyz.MOLECULE_CONTEXT_ATOM_FEATURE_DIM))
            self.assertEqual(tgt.shape, (18,))
            self.assertEqual(fut.shape, (3, 18))
            for split_name in ("train", "val", "test"):
                ds = splits[split_name]
                for target_idx in ds.start_indices[:20]:
                    seg_end = int(ds.data.segment_ends[np.searchsorted(ds.data.segment_ends, int(target_idx), side="right")])
                    self.assertLess(int(target_idx) + int(ds.future_horizon), seg_end + 1)

    def test_clean_window_filter_excludes_context_discontinuities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            processed = root / "processed"
            _write_xyz_zip(zip_path, categories={"Dynamic_Test": (8, 6)}, frames=12)
            molecule_xyz.prepare_molecule_xyz_zip(
                zip_path,
                processed,
                dataset_key="demo",
                stratum="Dynamic_Test",
            )

            npz_path = molecule_xyz.molecule_processed_npz_path(processed)
            payload = np.load(npz_path, allow_pickle=False)
            arrays = {key: payload[key] for key in payload.files}
            arrays["discontinuity_step_mask"] = arrays["discontinuity_step_mask"].copy()
            train_trajectory = json.loads(molecule_xyz.molecule_processed_metadata_path(processed).read_text())["split_trajectory_ids"]["train"][0]
            train_start = int(np.where(arrays["trajectory_ids"] == int(train_trajectory))[0][0])
            arrays["discontinuity_step_mask"][train_start + 4] = True
            np.savez_compressed(npz_path, **arrays)

            cfg = molecule_xyz.configure_molecule_otflow(
                OTFlowConfig(),
                history_len=2,
                future_horizon=1,
                rollout_mode="autoregressive",
                atom_count=6,
            )
            splits = molecule_xyz.build_molecule_dataset_splits(
                processed_dir=processed,
                cfg=cfg,
                history_len=2,
                future_horizon=1,
                stride_train=1,
                stride_eval=1,
                dataset_key="demo",
                stratum="Dynamic_Test",
            )

            self.assertNotIn(train_start + 4, splits["train"].start_indices.tolist())
            self.assertNotIn(train_start + 5, splits["train"].start_indices.tolist())
            counts = splits["stats"]["filter_counts"]["train"]
            self.assertGreater(counts["discontinuity_window_excluded"], counts["discontinuity_target_excluded"])

    def test_alignment_round_trip(self) -> None:
        rng = np.random.default_rng(0)
        reference = rng.normal(size=(6, 3)).astype(np.float32)
        window = np.stack([reference + 0.01 * step for step in range(4)], axis=0).astype(np.float32)
        local, center, rotation = molecule_xyz.align_window_to_reference(window, reference, anchor_offset=1)
        reconstructed = molecule_xyz.invert_aligned_coords(local, center, rotation)
        np.testing.assert_allclose(reconstructed, window, atol=1e-5)

    def test_config_context_feature_dim_override_is_backward_compatible(self) -> None:
        cfg = OTFlowConfig()
        self.assertEqual(cfg.context_dim, cfg.snapshot_dim)
        cfg.apply_overrides(context_feature_dim=616)
        self.assertEqual(cfg.context_dim, 616)

    def test_generic_rollout_rejects_augmented_non_temporal_context_without_future_features(self) -> None:
        cfg = molecule_xyz.configure_molecule_otflow(
            OTFlowConfig(),
            history_len=2,
            future_horizon=1,
            rollout_mode="autoregressive",
            atom_count=2,
        )
        model = SimpleNamespace(cfg=cfg)
        hist = torch.zeros(1, 2, 2 * molecule_xyz.MOLECULE_CONTEXT_ATOM_FEATURE_DIM)
        with self.assertRaisesRegex(ValueError, "domain-specific rollout"):
            generate_continuation(model, hist, None, steps=1, nfe=1)

    def test_molecule_training_exports_validation_selected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_model = torch.nn.Linear(1, 1)
            fake_model.step = 0  # type: ignore[attr-defined]

            def fake_train_loop(ds, cfg, *, model_name, steps, log_every, on_step):
                del ds, cfg, model_name, log_every
                for step in range(1, int(steps) + 1):
                    fake_model.step = step  # type: ignore[attr-defined]
                    on_step(step, fake_model, float(10 - step), {"loss": float(10 - step)})
                return fake_model

            def fake_eval(ds, model, cfg, **kwargs):
                del ds, cfg, kwargs
                return {"loss": {1: 5.0, 2: 3.0, 3: 4.0}[int(model.step)], "examples": 2, "batches": 1}

            args = SimpleNamespace(
                dataset_key="demo",
                stratum="Dynamic_Test",
                zip_path=None,
                processed_dir=str(root / "processed"),
                out_dir=str(root / "outputs"),
                variant="ar_h1",
                device="cpu",
                steps=3,
                batch_size=2,
                lr=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                hidden_dim=8,
                ctx_encoder="hybrid",
                ctx_local_kernel=3,
                ctx_pool_scales="2",
                fu_net_type="mlp",
                fu_net_layers=1,
                fu_net_heads=1,
                history_len=2,
                future_horizon=1,
                train_variable_context=False,
                train_context_min=2,
                train_context_max=2,
                stride_train=1,
                stride_eval=1,
                log_every=1,
                val_every=1,
                val_max_batches=None,
                seed=0,
                grad_accum_steps=1,
                solver="euler",
                ema_decay=0.0,
                use_swa=False,
                use_minibatch_ot=False,
                prepare_data=False,
                use_amp=False,
                allow_non_trainable=False,
                budget_steps="1,2,3",
            )

            split_stats = {
                "dataset_key": "demo",
                "stratum": "Dynamic_Test",
                "snapshot_dim": 18,
                "context_feature_dim": 66,
                "stats": {
                    "target_mean": [0.0] * 18,
                    "target_std": [1.0] * 18,
                    "context_mean": [0.0] * 66,
                    "context_std": [1.0] * 66,
                    "reference_coords": [[0.0, 0.0, 0.0]] * 6,
                },
            }
            with patch.object(
                train_molecule_module,
                "ensure_molecule_processed",
                return_value={"dataset_key": "demo", "stratum": "Dynamic_Test", "atom_count": 6, "context_feature_dim": 66, "trainable": True},
            ), patch.object(
                train_molecule_module,
                "build_molecule_dataset_splits",
                return_value={
                    "train": [0],
                    "val": [0],
                    "val_clean": [0],
                    "test": [0],
                    "test_clean": [0],
                    "stats": split_stats,
                },
            ), patch.object(
                train_molecule_module,
                "train_loop",
                side_effect=fake_train_loop,
            ), patch.object(
                train_molecule_module,
                "evaluate_average_loss",
                side_effect=fake_eval,
            ), patch.object(
                train_molecule_module,
                "_evaluate_iso_balanced_loss",
                return_value={"loss": 1.0, "examples": 2, "iso_count": 1},
            ):
                summary = train_molecule_module.train_molecule_backbone(args)

            self.assertEqual(summary["selected_step"], 2)
            self.assertEqual(summary["budget_artifacts"]["1"]["selected_step"], 1)
            self.assertEqual(summary["budget_artifacts"]["2"]["selected_step"], 2)
            self.assertEqual(summary["budget_artifacts"]["3"]["selected_step"], 2)
            metadata_path = root / "outputs" / "demo" / "Dynamic_Test" / "ar_h1" / "3_steps" / "checkpoint_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["selection"]["selection_metric"], "clean_validation_loss")
            self.assertEqual(metadata["selection"]["selected_step"], 2)
            self.assertNotIn(str(root), json.dumps(metadata))

    def test_molecule_evaluation_reuses_checkpoint_stats_and_resolves_default_processed_dir(self) -> None:
        cfg = molecule_xyz.configure_molecule_otflow(
            OTFlowConfig(),
            history_len=2,
            future_horizon=1,
            rollout_mode="autoregressive",
            atom_count=2,
        )
        checkpoint_stats = molecule_xyz.MoleculeStats(
            target_mean=np.zeros(6, dtype=np.float32),
            target_std=np.ones(6, dtype=np.float32),
            context_mean=np.zeros(22, dtype=np.float32),
            context_std=np.ones(22, dtype=np.float32),
            reference_coords=np.zeros((2, 3), dtype=np.float32),
        )

        class FakeDataset:
            data = SimpleNamespace(atom_symbols=["C", "H"], atom_count=2)
            stats = checkpoint_stats

            def __len__(self) -> int:
                return 1

        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            return {"val_clean": FakeDataset()}

        args = SimpleNamespace(
            checkpoint="dummy.pt",
            processed_dir=None,
            dataset_key="",
            stratum="",
            split="val_clean",
            device="cpu",
            max_windows=0,
            sample_count=1,
            nfe=1,
            solver="euler",
            stride_eval=5,
            val_max_batches=None,
            seed=0,
            out_json="",
        )

        with patch.object(
            molecule_metrics.torch,
            "load",
            return_value={
                "molecule_stats": checkpoint_stats.to_dict(),
                "dataset_key": "demo",
                "stratum": "Dynamic_Test",
            },
        ), patch.object(
            molecule_metrics,
            "load_checkpoint_model",
            return_value=(SimpleNamespace(), cfg),
        ), patch.object(
            molecule_metrics,
            "build_molecule_dataset_splits",
            side_effect=fake_build,
        ), patch.object(
            molecule_metrics,
            "evaluate_average_loss",
            return_value={"loss": 1.0, "examples": 1, "batches": 1},
        ):
            summary = molecule_metrics.evaluate_molecule_checkpoint(args)

        self.assertEqual(summary["split"], "val_clean")
        self.assertEqual(summary["dataset_key"], "demo")
        self.assertEqual(summary["stratum"], "Dynamic_Test")
        self.assertEqual(captured["processed_dir"], molecule_xyz.default_molecule_processed_dir("demo", "Dynamic_Test"))
        self.assertEqual(captured["stride_train"], 1)
        self.assertEqual(captured["stride_eval"], 5)
        self.assertIsNotNone(captured["stats"])

    def test_molecule_distributional_metrics_are_finite_and_zero_for_identical_inputs(self) -> None:
        coords = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]],
            ],
            dtype=np.float32,
        )
        previous = coords - 0.05
        metrics = molecule_metrics.molecule_distributional_metrics(coords, coords, coords, previous)

        expected_keys = {
            "coord_cw1_mean",
            "coord_cw1_median",
            "coord_cw1_max",
            "pairdist_w1",
            "velocity_norm_w1",
            "acceleration_norm_w1",
        }
        self.assertEqual(set(metrics), expected_keys)
        for value in metrics.values():
            self.assertTrue(np.isfinite(value))
            self.assertEqual(value, 0.0)

    def test_molecule_distributional_metrics_detect_pair_distance_shift_and_shape_errors(self) -> None:
        true = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
        pred = np.asarray([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float32)
        current = true.copy()
        previous = true.copy()

        metrics = molecule_metrics.molecule_distributional_metrics(pred, true, current, previous)

        self.assertAlmostEqual(metrics["pairdist_w1"], 1.0, places=6)
        self.assertAlmostEqual(metrics["coord_cw1_max"], 1.0, places=6)
        self.assertTrue(np.isnan(molecule_metrics.molecule_distributional_metrics(pred, true, current, None)["acceleration_norm_w1"]))
        with self.assertRaisesRegex(ValueError, "pred/true"):
            molecule_metrics.molecule_distributional_metrics(pred[:, :1], true, current, previous)
        with self.assertRaisesRegex(ValueError, "current coordinate"):
            molecule_metrics.molecule_distributional_metrics(pred, true, current[:, :1], previous)
        with self.assertRaisesRegex(ValueError, "previous coordinate"):
            molecule_metrics.molecule_distributional_metrics(pred, true, current, previous[:, :1])

    def test_molecule_evaluation_reports_distributional_metrics(self) -> None:
        cfg = molecule_xyz.configure_molecule_otflow(
            OTFlowConfig(),
            history_len=2,
            future_horizon=1,
            rollout_mode="autoregressive",
            atom_count=2,
        )
        checkpoint_stats = molecule_xyz.MoleculeStats(
            target_mean=np.zeros(6, dtype=np.float32),
            target_std=np.ones(6, dtype=np.float32),
            context_mean=np.zeros(22, dtype=np.float32),
            context_std=np.ones(22, dtype=np.float32),
            reference_coords=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        )

        def context_from_positions(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            context = np.zeros((2, 22), dtype=np.float32)
            context[0] = np.concatenate([np.concatenate([first[0], np.zeros(8)]), np.concatenate([first[1], np.zeros(8)])])
            context[1] = np.concatenate([np.concatenate([second[0], np.zeros(8)]), np.concatenate([second[1], np.zeros(8)])])
            return context

        class FakeDataset:
            H = 2
            future_horizon = 1
            data = SimpleNamespace(atom_symbols=["C", "H"], atom_count=2)
            stats = checkpoint_stats

            def __len__(self) -> int:
                return 2

            def eval_item(self, idx: int):
                previous = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
                current = previous + float(idx) * 0.1
                future = current + 0.05
                return {
                    "context": context_from_positions(previous, current),
                    "history_coords": np.stack([previous, current], axis=0).astype(np.float32),
                    "future_coords": future[None, :, :].astype(np.float32),
                    "current_coords": current.astype(np.float32),
                    "transition": bool(idx == 1),
                    "transition_window": bool(idx == 1),
                }

            def denormalize_target(self, values):
                return np.asarray(values, dtype=np.float32)

        class FakeModel:
            def sample_future(self, hist_t, *, steps: int, solver: str):
                del hist_t, steps, solver
                return torch.full((1, 1, 6), 0.05, dtype=torch.float32)

        calls = []

        def fake_distributional(pred, true, current, previous):
            calls.append((pred.copy(), true.copy(), current.copy(), previous.copy()))
            return {
                "coord_cw1_mean": 1.0,
                "coord_cw1_median": 2.0,
                "coord_cw1_max": 3.0,
                "pairdist_w1": 4.0,
                "velocity_norm_w1": 5.0,
                "acceleration_norm_w1": 6.0,
            }

        args = SimpleNamespace(
            checkpoint="dummy.pt",
            processed_dir="processed",
            dataset_key="demo",
            stratum="Dynamic_Test",
            split="test_clean",
            device="cpu",
            max_windows=2,
            sample_count=1,
            nfe=1,
            solver="euler",
            stride_eval=1,
            val_max_batches=None,
            seed=0,
            out_json="",
        )

        with patch.object(
            molecule_metrics.torch,
            "load",
            return_value={"molecule_stats": checkpoint_stats.to_dict()},
        ), patch.object(
            molecule_metrics,
            "load_checkpoint_model",
            return_value=(FakeModel(), cfg),
        ), patch.object(
            molecule_metrics,
            "build_molecule_dataset_splits",
            return_value={"test_clean": FakeDataset()},
        ), patch.object(
            molecule_metrics,
            "evaluate_average_loss",
            return_value={"loss": 1.0, "examples": 2, "batches": 1},
        ), patch.object(
            molecule_metrics,
            "molecule_distributional_metrics",
            side_effect=fake_distributional,
        ):
            summary = molecule_metrics.evaluate_molecule_checkpoint(args)

        dist = summary["metrics"]["distributional"]
        expected = {
            "coord_cw1_mean",
            "coord_cw1_median",
            "coord_cw1_max",
            "pairdist_w1",
            "velocity_norm_w1",
            "acceleration_norm_w1",
        }
        self.assertEqual(set(dist["all_first_horizon"]), expected)
        self.assertEqual(set(dist["clean_first_horizon"]), expected)
        self.assertEqual(set(dist["transition_first_horizon"]), expected)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0].shape, (2, 2, 3))
        self.assertEqual(calls[1][0].shape, (1, 2, 3))
        self.assertEqual(calls[2][0].shape, (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
