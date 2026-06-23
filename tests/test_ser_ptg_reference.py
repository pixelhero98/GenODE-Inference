from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from genode.gipo import ser_ptg_reference as ser
from genode.gipo.ser_ptg_reference import build_argparser, build_ser_ptg_reference, collect_batched_local_defect_trace
from genode.evaluation.otflow_evaluation_support import build_conditional_generation_dataset_args_from_cfg
from genode.models.config import OTFlowConfig


class SerPtgReferenceTests(unittest.TestCase):
    def test_argparser_provides_conditional_loader_defaults(self) -> None:
        args = build_argparser().parse_args([])
        self.assertEqual(args.dataset_seed, 0)
        self.assertEqual(args.train_tuning_max_examples, 0)
        self.assertEqual(args.lr, 2e-4)
        self.assertEqual(args.weight_decay, 1e-4)
        self.assertEqual(args.grad_clip, 1.0)
        self.assertEqual(args.hidden_dim, 160)
        self.assertEqual(args.fu_net_layers, 3)
        self.assertEqual(args.fu_net_heads, 4)

    def test_argparser_defaults_rebuild_conditional_dataset_args(self) -> None:
        cfg = OTFlowConfig(
            device=torch.device("cpu"),
            levels=1,
            token_dim=4,
            history_len=12000,
            hidden_dim=160,
            dropout=0.0,
            ctx_heads=4,
            ctx_layers=1,
            fu_net_layers=3,
            fu_net_heads=4,
            rollout_mode="non_ar",
            future_block_len=3000,
            use_cond_features=True,
            cond_standardize=True,
            cond_dim=5,
            use_amp=False,
        )
        cfg.apply_overrides(steps=20000)
        args = build_argparser().parse_args(["--dataset", "long_term_st", "--device", "cpu"])

        dataset_args = build_conditional_generation_dataset_args_from_cfg(
            args,
            "long_term_st",
            "transformer",
            cfg,
        )

        self.assertEqual(dataset_args.seed, 0)
        self.assertEqual(dataset_args.lr, 2e-4)
        self.assertEqual(dataset_args.fu_net_layers, 3)
        self.assertEqual(dataset_args.future_block_len, 3000)

    def test_collect_batched_local_defect_trace_handles_sample_seed_above_numpy_limit(self) -> None:
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
            def sample_future_trace(self, hist, steps=None, solver=None, oracle_local_error=False):
                del solver, oracle_local_error
                width = int(steps) + 1
                trace = {
                    "time_grid": torch.linspace(0.0, 1.0, width, device=hist.device),
                    "oracle_local_error": torch.ones(hist.shape[0], int(steps), device=hist.device),
                }
                return torch.zeros(hist.shape[0], 1, 1, device=hist.device), trace

        result = collect_batched_local_defect_trace(
            DummyModel(),
            DummyDataset(),
            cfg,
            solver_name="euler",
            reference_macro_steps=1,
            solver_order_p=1.0,
            seed=2**32 - 1,
            example_indices=[0],
            calibration_trace_samples=2,
        )
        self.assertEqual(result["trace_count"], 2)

    def test_forecast_train_tuning_is_capped_by_context_sample_count(self) -> None:
        captured = []

        def fake_collector(*args, **kwargs):
            del args
            indices = [int(idx) for idx in kwargs["example_indices"]]
            captured.append(indices)
            steps = int(kwargs["reference_macro_steps"])
            return {
                "reference_time_grid": [float(i) / float(steps) for i in range(steps + 1)],
                "local_defect_trace": [1.0] * steps,
                "eval_examples": len(indices),
                "trace_count": len(indices),
            }

        class FakeDataset:
            def __init__(self, size: int) -> None:
                self.size = int(size)

            def __len__(self) -> int:
                return self.size

        checkpoint = {
            "model": object(),
            "cfg": SimpleNamespace(),
            "splits": {"train": FakeDataset(5000), "val": FakeDataset(50)},
            "checkpoint_id": "forecast_ckpt",
        }
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(ser, "load_forecast_checkpoint_splits", return_value=checkpoint), mock.patch.object(
            ser,
            "collect_batched_local_defect_trace",
            side_effect=fake_collector,
        ):
            args = build_argparser().parse_args(
                [
                    "--dataset",
                    "traffic_hourly",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--context_sample_count",
                    "256",
                    "--out_dir",
                    tmpdir,
                    "--device",
                    "cpu",
                ]
            )
            summary = build_ser_ptg_reference(args)

        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), 256)
        prediction = summary["predictions"][0]
        self.assertEqual(prediction["example_selection_protocol"], ser.SER_PTG_EXAMPLE_SELECTION_PROTOCOL)
        self.assertEqual(prediction["selected_examples"], 256)
        self.assertEqual(prediction["selected_examples_cap"], 256)
        self.assertEqual(prediction["selected_examples_cap_source"], "context_sample_count")
        self.assertTrue(prediction["selection_was_capped"])
        self.assertGreater(prediction["uncapped_candidate_examples"], 256)
        self.assertEqual(summary["selected_examples_cap"], 256)
        self.assertEqual(summary["selected_examples_cap_source"], "context_sample_count")
        self.assertEqual(summary["local_defect_trace_protocol"], "otflow_midpoint_local_defect_proxy_v1")
        self.assertEqual(summary["oracle_local_error_semantics"], "local_defect_proxy_not_teacher_oracle")

    def test_forecast_validation_zero_val_windows_uses_context_cap_not_all_rows(self) -> None:
        captured = []

        def fake_collector(*args, **kwargs):
            del args
            indices = [int(idx) for idx in kwargs["example_indices"]]
            captured.append(indices)
            steps = int(kwargs["reference_macro_steps"])
            return {
                "reference_time_grid": [float(i) / float(steps) for i in range(steps + 1)],
                "local_defect_trace": [1.0] * steps,
                "eval_examples": len(indices),
                "trace_count": len(indices),
            }

        class FakeDataset:
            def __init__(self, size: int) -> None:
                self.size = int(size)

            def __len__(self) -> int:
                return self.size

        checkpoint = {
            "model": object(),
            "cfg": SimpleNamespace(),
            "splits": {"train": FakeDataset(10), "val": FakeDataset(2000)},
            "checkpoint_id": "forecast_ckpt",
        }
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(ser, "load_forecast_checkpoint_splits", return_value=checkpoint), mock.patch.object(
            ser,
            "collect_batched_local_defect_trace",
            side_effect=fake_collector,
        ):
            args = build_argparser().parse_args(
                [
                    "--dataset",
                    "traffic_hourly",
                    "--reference_split",
                    "validation_tuning",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--context_sample_count",
                    "256",
                    "--val_windows",
                    "0",
                    "--out_dir",
                    tmpdir,
                    "--device",
                    "cpu",
                ]
            )
            summary = build_ser_ptg_reference(args)

        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), 256)
        prediction = summary["predictions"][0]
        self.assertEqual(prediction["selected_examples"], 256)
        self.assertEqual(prediction["selected_examples_cap"], 256)
        self.assertEqual(prediction["selected_examples_cap_source"], "context_sample_count")
        self.assertTrue(prediction["selection_was_capped"])
        self.assertEqual(prediction["uncapped_candidate_examples"], 2000)

    def test_conditional_generation_over_selection_is_capped(self) -> None:
        captured = []
        requested_windows = []

        def fake_collector(*args, **kwargs):
            del args
            indices = [int(idx) for idx in kwargs["example_indices"]]
            captured.append(indices)
            steps = int(kwargs["reference_macro_steps"])
            return {
                "reference_time_grid": [float(i) / float(steps) for i in range(steps + 1)],
                "local_defect_trace": [1.0] * steps,
                "eval_examples": len(indices),
                "trace_count": len(indices),
            }

        class FakeDataset:
            start_indices = list(range(1000))

            def __len__(self) -> int:
                return 1000

        def fake_choose_valid_windows(ds, horizon: int, n_windows: int, seed: int):
            del ds, horizon, seed
            requested_windows.append(int(n_windows))
            return list(range(int(n_windows) + 64))

        checkpoint = {
            "model": object(),
            "cfg": SimpleNamespace(),
            "splits": {"train": FakeDataset(), "val": FakeDataset()},
            "checkpoint_id": "conditional_ckpt",
        }
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(ser, "load_conditional_generation_checkpoint_splits", return_value=checkpoint), mock.patch.object(
            ser,
            "resolved_eval_horizon",
            return_value=1,
        ), mock.patch.object(ser, "_choose_valid_windows", side_effect=fake_choose_valid_windows), mock.patch.object(
            ser,
            "collect_batched_local_defect_trace",
            side_effect=fake_collector,
        ):
            args = build_argparser().parse_args(
                [
                    "--dataset",
                    "cryptos",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--context_sample_count",
                    "256",
                    "--out_dir",
                    tmpdir,
                    "--device",
                    "cpu",
                ]
            )
            summary = build_ser_ptg_reference(args)

        self.assertEqual(requested_windows, [256])
        self.assertEqual(len(captured[0]), 256)
        prediction = summary["predictions"][0]
        self.assertEqual(prediction["selected_examples"], 256)
        self.assertEqual(prediction["selected_examples_cap"], 256)
        self.assertEqual(prediction["selected_examples_cap_source"], "context_sample_count")
        self.assertTrue(prediction["selection_was_capped"])
        self.assertEqual(prediction["uncapped_candidate_examples"], 1000)
        self.assertEqual(prediction["selection_records"][0]["candidate_examples_after_initial_selection"], 320)
        self.assertEqual(prediction["selection_records"][0]["reference_available_examples"], 1000)

    def test_molecule_over_selection_is_capped(self) -> None:
        captured = []

        def fake_collector(*args, **kwargs):
            del args
            indices = [int(idx) for idx in kwargs["example_indices"]]
            captured.append(indices)
            steps = int(kwargs["reference_macro_steps"])
            return {
                "reference_time_grid": [float(i) / float(steps) for i in range(steps + 1)],
                "local_defect_trace": [1.0] * steps,
                "eval_examples": len(indices),
                "trace_count": len(indices),
            }

        class FakeDataset:
            def __len__(self) -> int:
                return 1000

        loaded = {
            "model": object(),
            "cfg": SimpleNamespace(),
            "splits": {"train": FakeDataset(), "val": FakeDataset()},
        }
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(ser, "load_molecule_group_manifest", return_value={}), mock.patch.object(
            ser,
            "trainable_molecule_group_members",
            return_value=[{"member_key": "member_a", "stratum": "set1", "processed_dir": "member_a"}],
        ), mock.patch.object(ser, "load_backbone_manifest", return_value={}), mock.patch.object(
            ser,
            "find_backbone_artifact",
            return_value={"checkpoint_path": str(Path(tmpdir) / "model.pt"), "checkpoint_id": "molecule_ckpt"},
        ), mock.patch.object(
            ser,
            "load_molecule_checkpoint_splits",
            return_value=loaded,
        ), mock.patch.object(
            ser,
            "_choose_molecule_indices",
            return_value=list(range(400)),
        ), mock.patch.object(
            ser,
            "collect_molecule_local_defect_trace",
            side_effect=fake_collector,
        ):
            args = build_argparser().parse_args(
                [
                    "--dataset",
                    "molecule_3d_set1",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--context_sample_count",
                    "256",
                    "--out_dir",
                    tmpdir,
                    "--molecule_group_root",
                    tmpdir,
                    "--backbone_manifest",
                    str(Path(tmpdir) / "backbone_manifest.json"),
                    "--device",
                    "cpu",
                ]
            )
            summary = build_ser_ptg_reference(args)

        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), 256)
        prediction = summary["predictions"][0]
        self.assertEqual(prediction["selected_examples"], 256)
        self.assertEqual(prediction["selected_examples_cap"], 256)
        self.assertEqual(prediction["selected_examples_cap_source"], "context_sample_count")
        self.assertTrue(prediction["selection_was_capped"])
        self.assertEqual(prediction["uncapped_candidate_examples"], 1000)
        self.assertEqual(prediction["selection_records"][0]["candidate_examples_after_initial_selection"], 400)
        self.assertEqual(prediction["selection_records"][0]["reference_available_examples"], 1000)

    def test_molecule_context_cap_is_global_across_seeds_and_members(self) -> None:
        captured = []

        def fake_collector(*args, **kwargs):
            del args
            indices = [int(idx) for idx in kwargs["example_indices"]]
            captured.append(indices)
            steps = int(kwargs["reference_macro_steps"])
            samples = int(kwargs["calibration_trace_samples"])
            return {
                "reference_time_grid": [float(i) / float(steps) for i in range(steps + 1)],
                "local_defect_trace": [1.0] * steps,
                "eval_examples": len(indices),
                "trace_count": len(indices) * samples,
            }

        class FakeDataset:
            def __len__(self) -> int:
                return 1000

        loaded = {
            "model": object(),
            "cfg": SimpleNamespace(),
            "splits": {"train": FakeDataset(), "val": FakeDataset()},
        }

        def fake_artifact(*args, **kwargs):
            del args
            member_key = str(kwargs["member_key"])
            return {"checkpoint_path": str(Path(tempdir) / f"{member_key}.pt"), "checkpoint_id": f"molecule_{member_key}"}

        with tempfile.TemporaryDirectory() as tempdir, mock.patch.object(ser, "load_molecule_group_manifest", return_value={}), mock.patch.object(
            ser,
            "trainable_molecule_group_members",
            return_value=[
                {"member_key": "member_a", "stratum": "set1", "processed_dir": "member_a"},
                {"member_key": "member_b", "stratum": "set1", "processed_dir": "member_b"},
            ],
        ), mock.patch.object(ser, "load_backbone_manifest", return_value={}), mock.patch.object(
            ser,
            "find_backbone_artifact",
            side_effect=fake_artifact,
        ), mock.patch.object(
            ser,
            "load_molecule_checkpoint_splits",
            return_value=loaded,
        ), mock.patch.object(
            ser,
            "_choose_molecule_indices",
            return_value=list(range(100)),
        ), mock.patch.object(
            ser,
            "collect_molecule_local_defect_trace",
            side_effect=fake_collector,
        ):
            args = build_argparser().parse_args(
                [
                    "--dataset",
                    "molecule_3d_set1",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0,1",
                    "--context_sample_count",
                    "5",
                    "--calibration_trace_samples",
                    "2",
                    "--out_dir",
                    tempdir,
                    "--molecule_group_root",
                    tempdir,
                    "--backbone_manifest",
                    str(Path(tempdir) / "backbone_manifest.json"),
                    "--device",
                    "cpu",
                ]
            )
            summary = build_ser_ptg_reference(args)

        self.assertEqual(sum(len(indices) for indices in captured), 5)
        self.assertTrue(all(len(indices) >= 1 for indices in captured))
        prediction = summary["predictions"][0]
        self.assertEqual(prediction["selected_examples"], 5)
        self.assertEqual(prediction["selected_examples_cap"], 5)
        self.assertEqual(prediction["trace_count"], 10)
        self.assertEqual(prediction["reference_seed_count"], 2)
        self.assertEqual(prediction["reference_member_count"], 2)
        self.assertEqual(prediction["reference_selection_group_count"], 4)
        self.assertEqual(summary["selected_examples"], 5)
        self.assertEqual(summary["trace_count_total_across_predictions"], 10)


if __name__ == "__main__":
    unittest.main()
