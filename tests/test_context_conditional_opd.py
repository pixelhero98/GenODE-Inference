from __future__ import annotations

import tempfile
import unittest
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from genode.conditional_opd import context_conditional as ctx_ops
from genode.conditional_opd.report_context_locked_test import (
    build_argparser as build_locked_report_argparser,
    report_context_locked_test,
)
from genode.conditional_opd.context_conditional import (
    ContextScheduleTeacherMLP,
    ContextSupportStudentMLP,
    attach_uniform_context_rewards,
    build_calibration_holdout_non_regression_guard,
    build_teacher_guided_support_targets,
    build_series_index_map,
    context_id_from_row,
    context_teacher_diagnostics,
    load_context_embedding_table,
    recommended_context_calibration_count,
    context_calibration_train_val_counts,
    sample_context_ids_stratified,
    save_context_embedding_table,
    split_rows_by_context_holdout,
    split_rows_by_series_holdout,
    train_context_support_student,
    train_context_teacher,
    validate_context_support_schedule_keys,
)
from genode.conditional_opd.train_context_conditional_opd import build_argparser as build_train_argparser
from genode.conditional_opd.train_context_conditional_opd import train_context_conditional_opd
from genode.conditional_opd.models import setting_features
from genode.evaluation.otflow_evaluation_support import evaluate_forecast_schedule
from genode.models.config import OTFlowConfig


def _row(
    *,
    schedule: str,
    seed: int,
    context_idx: int,
    crps: float,
    mase: float,
    split_phase: str = "train_tuning",
    series_id: str | None = None,
    solver_key: str = "euler",
    target_nfe: int = 4,
) -> dict:
    return {
        "dataset": "solar_energy_10m",
        "split_phase": split_phase,
        "seed": seed,
        "solver_key": solver_key,
        "target_nfe": target_nfe,
        "scheduler_key": schedule,
        "example_idx": context_idx,
        "series_id": f"series_{context_idx}" if series_id is None else str(series_id),
        "series_idx": context_idx,
        "target_t": 100 + context_idx,
        "history_start": 76 + context_idx,
        "history_stop": 100 + context_idx,
        "crps": crps,
        "mase": mase,
    }


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _guard_for_cell(*, mode: str, fallback: str = "", guard_id: str = "guard_test", support: list[str] | None = None) -> dict:
    support_keys = ["uniform", "ays"] if support is None else list(support)
    decision = {
        "solver_key": "euler",
        "target_nfe": 4,
        "deployed_mode": mode,
        "best_static_support_schedule_key": fallback or support_keys[0],
        "fallback_schedule_key": fallback if mode == "static_support" else "",
        "context_student_score": 0.0,
        "best_static_score": 0.0,
        "score_margin": 0.0,
        "required_margin": 0.001,
    }
    return {
        "artifact": "calibration_holdout_non_regression_guard",
        "enabled": True,
        "guard_id": guard_id,
        "guard_table_hash": f"{guard_id}_hash",
        "locked_test_used_for_selection": False,
        "locked_test_used_for_guard_construction": False,
        "support_schedule_keys": support_keys,
        "source_split_phases": ["train_tuning"],
        "observed_calibration_holdout_names": ["context_disjoint"],
        "cell_decisions": [decision],
        "cell_decision_map": {"euler/4": decision},
    }


def _stamp_holdout(rows: list[dict], holdout_name: str = "context_disjoint") -> list[dict]:
    return [dict(row, calibration_holdout_name=holdout_name) for row in rows]


class ContextConditionalOPDTests(unittest.TestCase):
    def test_context_id_requires_complete_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete identity"):
            context_id_from_row({"solver_key": "euler", "target_nfe": 4})

    def test_uniform_context_rewards_do_not_mix_contexts_or_seeds(self) -> None:
        rows = [
            _row(schedule="uniform", seed=0, context_idx=0, crps=2.0, mase=4.0),
            _row(schedule="ays", seed=0, context_idx=0, crps=1.0, mase=2.0),
            _row(schedule="uniform", seed=1, context_idx=0, crps=10.0, mase=10.0),
            _row(schedule="ays", seed=1, context_idx=0, crps=5.0, mase=20.0),
            _row(schedule="uniform", seed=0, context_idx=1, crps=8.0, mase=8.0),
            _row(schedule="ays", seed=0, context_idx=1, crps=4.0, mase=4.0),
        ]
        rewarded = attach_uniform_context_rewards(rows, support_schedule_keys=("uniform", "ays"))
        by_key = {(row["seed"], row["example_idx"], row["scheduler_key"]): row for row in rewarded}
        self.assertGreater(float(by_key[(0, 0, "ays")]["u_comp_uniform"]), 0.0)
        self.assertAlmostEqual(float(by_key[(1, 0, "ays")]["u_comp_uniform"]), 0.0)
        self.assertGreater(float(by_key[(0, 1, "ays")]["u_comp_uniform"]), 0.0)

    def test_uniform_context_rewards_reject_duplicate_support_rows(self) -> None:
        rows = [
            _row(schedule="uniform", seed=0, context_idx=0, crps=2.0, mase=2.0),
            _row(schedule="ays", seed=0, context_idx=0, crps=1.0, mase=1.0),
            _row(schedule="ays", seed=0, context_idx=0, crps=1.1, mase=1.1),
        ]
        with self.assertRaisesRegex(ValueError, "exactly one row for every support schedule"):
            attach_uniform_context_rewards(rows, support_schedule_keys=("uniform", "ays"))

    def test_bo_like_support_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "BO/candidate"):
            validate_context_support_schedule_keys(("uniform", "bo_schedule_000"))

    def test_context_teacher_and_support_student_train_on_fixed_support(self) -> None:
        rows = []
        embeddings = {}
        for context_idx in range(3):
            uniform = _row(schedule="uniform", seed=0, context_idx=context_idx, crps=2.0, mase=2.0)
            ays = _row(schedule="ays", seed=0, context_idx=context_idx, crps=1.0 + 0.1 * context_idx, mase=1.5)
            rows.extend([uniform, ays])
            embeddings[context_id_from_row(uniform)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
        rewarded = attach_uniform_context_rewards(rows, support_schedule_keys=("uniform", "ays"))
        series_map = build_series_index_map(rewarded)
        setting_dim = int(setting_features("euler", 4).numel())
        teacher = ContextScheduleTeacherMLP(
            setting_dim=setting_dim,
            max_macro_steps=12,
            context_dim=2,
            num_series=len(series_map),
            hidden_dim=16,
            hidden_layers=1,
        )
        teacher_summary = train_context_teacher(
            teacher,
            rewarded,
            context_embeddings=embeddings,
            series_index_map=series_map,
            max_macro_steps=12,
            steps=2,
            lr=1e-3,
        )
        self.assertGreater(teacher_summary["teacher_pair_count"], 0)
        student = ContextSupportStudentMLP(
            setting_dim=setting_dim,
            context_dim=2,
            num_series=len(series_map),
            support_schedule_keys=("uniform", "ays"),
            hidden_dim=16,
            hidden_layers=1,
        )
        student_summary = train_context_support_student(
            student,
            teacher,
            rewarded,
            context_embeddings=embeddings,
            series_index_map=series_map,
            support_schedule_keys=("uniform", "ays"),
            max_macro_steps=12,
            steps=2,
            lr=1e-3,
        )
        self.assertEqual(student_summary["student_policy_type"], "categorical_support")
        self.assertEqual(student_summary["student_objective"], "teacher_guided_top1_top2_categorical_ce")
        self.assertEqual(student_summary["series_unknown_dropout_mode"], "dynamic_per_step")
        self.assertEqual(student_summary["student_target_summary"]["context_setting_count"], 3)
        self.assertIn("student_ce_loss", student_summary["losses"][-1])

    def test_teacher_guided_student_targets_fallback_and_top2_margin(self) -> None:
        class IntervalTeacher(torch.nn.Module):
            def forward(self, setting, interval, series, context):
                del setting, series, context
                return interval[:, 0]

        uniform = _row(schedule="uniform", seed=0, context_idx=0, crps=1.0, mase=1.0)
        bad_ays = _row(schedule="ays", seed=0, context_idx=0, crps=10.0, mase=10.0)
        rows = attach_uniform_context_rewards([uniform, bad_ays], support_schedule_keys=("uniform", "ays"))
        embeddings = {context_id_from_row(uniform): np.asarray([0.0, 1.0], dtype=np.float32)}
        _, _, _, targets, summary = build_teacher_guided_support_targets(
            IntervalTeacher(),
            rows,
            context_embeddings=embeddings,
            series_index_map=build_series_index_map(rows),
            support_schedule_keys=("uniform", "ays"),
            schedule_grids={
                ("uniform", "euler", 4): (0.0, 0.25, 0.5, 0.75, 1.0),
                ("ays", "euler", 4): (0.0, 0.9, 0.95, 0.98, 1.0),
            },
            support_choice_margin=0.001,
        )
        self.assertEqual(summary["target_source_counts"], {"observed_fallback_top1": 1})
        self.assertTrue(torch.allclose(targets[0], torch.tensor([1.0, 0.0])))

        close_ays = _row(schedule="ays", seed=0, context_idx=1, crps=1.0005, mase=1.0005)
        uniform2 = _row(schedule="uniform", seed=0, context_idx=1, crps=1.0, mase=1.0)
        rows2 = attach_uniform_context_rewards([uniform2, close_ays], support_schedule_keys=("uniform", "ays"))
        embeddings2 = {context_id_from_row(uniform2): np.asarray([1.0, 1.0], dtype=np.float32)}
        _, _, _, targets2, summary2 = build_teacher_guided_support_targets(
            IntervalTeacher(),
            rows2,
            context_embeddings=embeddings2,
            series_index_map=build_series_index_map(rows2),
            support_schedule_keys=("uniform", "ays"),
            schedule_grids={
                ("uniform", "euler", 4): (0.0, 0.5, 0.7, 0.9, 1.0),
                ("ays", "euler", 4): (0.0, 0.5005, 0.7, 0.9, 1.0),
            },
            support_choice_margin=0.001,
        )
        self.assertEqual(summary2["target_source_counts"], {"teacher_accepted_top2": 1})
        self.assertTrue(torch.allclose(targets2[0], torch.tensor([0.4, 0.6])))

    def test_context_embedding_sidecar_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embeddings.npz"
            manifest = save_context_embedding_table(path, {"ctx_a": [1.0, 2.0], "ctx_b": [3.0, 4.0]})
            loaded = load_context_embedding_table(path)
        self.assertEqual(manifest["context_count"], 2)
        self.assertTrue(np.allclose(loaded["ctx_a"], np.asarray([1.0, 2.0], dtype=np.float32)))

    def test_context_sample_recommendation_caps_massive_pool(self) -> None:
        self.assertEqual(recommended_context_calibration_count(10_000), 120)
        self.assertEqual(recommended_context_calibration_count(10_000, normalized_combined_reference=10_000), 144)
        self.assertEqual(recommended_context_calibration_count(100), 100)
        self.assertEqual(context_calibration_train_val_counts(10_000), (96, 24))
        sampled = sample_context_ids_stratified(
            [_row(schedule="uniform", seed=0, context_idx=idx, crps=1.0, mase=1.0) for idx in range(200)],
            sample_count=120,
            seed=0,
        )
        self.assertEqual(len(sampled), 120)

    def test_series_disjoint_holdout_keeps_whole_series_out_of_fit(self) -> None:
        rows = []
        for context_idx in range(6):
            rows.append(
                _row(
                    schedule="uniform",
                    seed=0,
                    context_idx=context_idx,
                    crps=1.0,
                    mase=1.0,
                    series_id=f"series_{context_idx // 2}",
                )
            )
        fit_rows, heldout_rows = split_rows_by_series_holdout(rows, holdout_fraction=0.34, seed=1)
        fit_series = {row["series_key"] for row in fit_rows}
        heldout_series = {row["series_key"] for row in heldout_rows}
        self.assertTrue(heldout_series)
        self.assertEqual(fit_series & heldout_series, set())

    def test_context_teacher_diagnostics_allow_unknown_series(self) -> None:
        rows = []
        embeddings = {}
        for context_idx, series_id in enumerate(("fit_series", "heldout_series")):
            uniform = _row(schedule="uniform", seed=0, context_idx=context_idx, crps=2.0, mase=2.0, series_id=series_id)
            ays = _row(schedule="ays", seed=0, context_idx=context_idx, crps=1.0, mase=1.0, series_id=series_id)
            rows.extend([uniform, ays])
            embeddings[context_id_from_row(uniform)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
        rewarded = attach_uniform_context_rewards(rows, support_schedule_keys=("uniform", "ays"))
        fit_rows = [row for row in rewarded if row["series_id"] == "fit_series"]
        heldout_rows = [row for row in rewarded if row["series_id"] == "heldout_series"]
        series_map = build_series_index_map(fit_rows)
        teacher = ContextScheduleTeacherMLP(
            setting_dim=int(setting_features("euler", 4).numel()),
            max_macro_steps=12,
            context_dim=2,
            num_series=len(series_map),
            hidden_dim=16,
            hidden_layers=1,
        )
        diagnostics = context_teacher_diagnostics(
            teacher,
            heldout_rows,
            context_embeddings=embeddings,
            series_index_map=series_map,
            max_macro_steps=12,
            split_name="series_disjoint",
            fit_series_keys=sorted(series_map),
        )
        self.assertEqual(diagnostics["fit_series_overlap_count"], 0)
        self.assertEqual(diagnostics["series_count"], 1)
        self.assertGreater(diagnostics["pair_count"], 0)
        self.assertIn("support_top1_accuracy", diagnostics)
        self.assertIn("support_top2_recall", diagnostics)
        self.assertIn("spearman_rank_correlation", diagnostics)
        self.assertEqual(diagnostics["support_choice_group_count"], 1)

    def test_teacher_checkpoint_selection_prefers_support_choice_metrics(self) -> None:
        def diag(top1: float, top2: float, pairwise: float, spearman: float, total: float) -> dict:
            return {
                "support_top1_accuracy": top1,
                "support_top2_recall": top2,
                "pairwise_accuracy": pairwise,
                "spearman_rank_correlation": spearman,
                "huber_loss": 0.1,
                "total_loss": total,
            }

        history = [
            {
                "step": 1,
                "diagnostics": {
                    "context_disjoint": diag(0.2, 0.4, 0.9, 0.3, 0.5),
                    "series_disjoint": diag(0.2, 0.4, 0.9, 0.3, 0.5),
                },
            },
            {
                "step": 2,
                "diagnostics": {
                    "context_disjoint": diag(0.5, 0.7, 0.7, 0.2, 0.9),
                    "series_disjoint": diag(0.4, 0.6, 0.7, 0.2, 0.9),
                },
            },
        ]
        selection, state = ctx_ops._selected_context_teacher_checkpoint(
            history,
            {1: {"w": torch.tensor([1.0])}, 2: {"w": torch.tensor([2.0])}},
            required_split_names=("context_disjoint", "series_disjoint"),
        )
        self.assertEqual(selection["selected_step"], 2)
        self.assertEqual(float(state["w"][0]), 2.0)
        self.assertEqual(selection["selection_metric"], "context_series_support_top1_top2")
        with self.assertRaisesRegex(ValueError, "no checkpoint satisfying"):
            ctx_ops._selected_context_teacher_checkpoint(
                history,
                {1: {}, 2: {}},
                required_split_names=("context_disjoint", "series_disjoint"),
                min_pairwise_accuracy=0.95,
            )

    def test_support_choice_top2_recall_with_three_support_schedules(self) -> None:
        key = ("solar_energy_10m", "euler", 4, "ctx_a", 0)
        schedules = ["uniform", "ays", "gits"]
        actual = [3.0, 2.0, 1.0]
        second_ranked = ctx_ops._support_choice_metrics(
            [2.0, 3.0, 1.0],
            actual,
            [key, key, key],
            schedules,
        )
        third_ranked = ctx_ops._support_choice_metrics(
            [1.0, 3.0, 2.0],
            actual,
            [key, key, key],
            schedules,
        )
        self.assertEqual(second_ranked["support_top1_accuracy"], 0.0)
        self.assertEqual(second_ranked["support_top2_recall"], 1.0)
        self.assertEqual(third_ranked["support_top1_accuracy"], 0.0)
        self.assertEqual(third_ranked["support_top2_recall"], 0.0)

    def test_context_trainer_excludes_context_and_series_holdouts_from_fit(self) -> None:
        rows = []
        embeddings = {}
        for context_idx in range(8):
            series_id = f"series_{context_idx // 2}"
            uniform = _row(schedule="uniform", seed=0, context_idx=context_idx, crps=2.0, mase=2.0, series_id=series_id)
            ays = _row(schedule="ays", seed=0, context_idx=context_idx, crps=1.0 + 0.05 * context_idx, mase=1.5, series_id=series_id)
            rows.extend([uniform, ays])
            embeddings[context_id_from_row(uniform)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_path = root / "rows.csv"
            embeddings_path = root / "embeddings.npz"
            _write_rows_csv(rows_path, rows)
            save_context_embedding_table(embeddings_path, embeddings)
            args = build_train_argparser().parse_args(
                [
                    "--rows_csv",
                    str(rows_path),
                    "--context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(root / "policy"),
                    "--support_schedule_keys",
                    "uniform,ays",
                    "--context_sample_count",
                    "1000",
                    "--context_holdout_fraction",
                    "0.25",
                    "--series_holdout_fraction",
                    "0.34",
                    "--teacher_steps",
                    "2",
                    "--teacher_checkpoint_every",
                    "1",
                    "--teacher_selection_min_pairwise_accuracy",
                    "0.0",
                    "--teacher_selection_min_spearman",
                    "-1.0",
                    "--student_steps",
                    "1",
                ]
            )
            summary = train_context_conditional_opd(args)
        self.assertEqual(summary["teacher_context_holdout_context_overlap_count"], 0)
        self.assertEqual(summary["teacher_series_holdout_series_overlap_count"], 0)
        self.assertEqual(summary["teacher_selection_protocol"], "context_and_series_disjoint_train_holdout_teacher_checkpoint")
        self.assertIn("teacher_checkpoint_selection", summary)
        self.assertTrue(summary["teacher_checkpoint_selection"]["history"])
        self.assertEqual(summary["teacher_checkpoint_selection"]["selection_metric"], "context_series_support_top1_top2")
        self.assertIn("teacher_context_holdout_diagnostics", summary)
        self.assertIn("teacher_series_holdout_diagnostics", summary)
        self.assertIn("calibration_holdout_non_regression_guard", summary)
        self.assertTrue(summary["calibration_holdout_non_regression_guard"]["cell_decisions"])
        self.assertIn("policy_id", summary)
        self.assertEqual(summary["series_unknown_dropout"], 0.10)
        self.assertEqual(summary["series_unknown_dropout_mode"], "dynamic_per_step_for_student")
        self.assertFalse(summary["locked_test_used_for_selection"])

    def test_context_trainer_rejects_locked_test_rows(self) -> None:
        row = _row(schedule="uniform", seed=0, context_idx=0, crps=1.0, mase=1.0, split_phase="locked_test")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_path = root / "rows.csv"
            embeddings_path = root / "embeddings.npz"
            _write_rows_csv(rows_path, [row])
            save_context_embedding_table(embeddings_path, {context_id_from_row(row): np.asarray([0.0, 0.0], dtype=np.float32)})
            args = build_train_argparser().parse_args(
                [
                    "--rows_csv",
                    str(rows_path),
                    "--context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(root / "policy"),
                    "--support_schedule_keys",
                    "uniform",
                ]
            )
            with self.assertRaisesRegex(ValueError, "refuses locked_test rows"):
                train_context_conditional_opd(args)

    def test_calibration_guard_selects_static_or_context_by_holdout_reward(self) -> None:
        def biased_student() -> ContextSupportStudentMLP:
            student = ContextSupportStudentMLP(
                setting_dim=int(setting_features("euler", 4).numel()),
                context_dim=2,
                num_series=1,
                support_schedule_keys=("uniform", "ays"),
            )
            for param in student.parameters():
                param.data.zero_()
            last_linear = [module for module in student.net.modules() if isinstance(module, torch.nn.Linear)][-1]
            last_linear.bias.data[:] = torch.tensor([0.0, 1.0])
            return student

        uniform = _row(schedule="uniform", seed=0, context_idx=0, crps=1.0, mase=1.0)
        bad_ays = _row(schedule="ays", seed=0, context_idx=0, crps=2.0, mase=2.0)
        rows = attach_uniform_context_rewards([uniform, bad_ays], support_schedule_keys=("uniform", "ays"))
        guard = build_calibration_holdout_non_regression_guard(
            biased_student(),
            _stamp_holdout(rows),
            context_embeddings={context_id_from_row(uniform): np.asarray([0.0, 0.0], dtype=np.float32)},
            series_index_map=build_series_index_map(rows),
            support_schedule_keys=("uniform", "ays"),
            margin=0.001,
            source_holdout_names=("context_disjoint",),
        )
        decision = guard["cell_decision_map"]["euler/4"]
        self.assertEqual(decision["deployed_mode"], "static_support")
        self.assertEqual(decision["fallback_schedule_key"], "uniform")

        class ContextSwitchStudent(torch.nn.Module):
            support_schedule_keys = ("uniform", "ays")

            def probabilities(self, setting, series, context):
                choose_ays = context[:, 0] > 0.5
                probs = torch.zeros((context.shape[0], 2), dtype=torch.float32, device=context.device)
                probs[:, 0] = torch.where(choose_ays, torch.tensor(0.0, device=context.device), torch.tensor(1.0, device=context.device))
                probs[:, 1] = torch.where(choose_ays, torch.tensor(1.0, device=context.device), torch.tensor(0.0, device=context.device))
                return probs

        uniform2a = _row(schedule="uniform", seed=0, context_idx=1, crps=1.0, mase=1.0)
        bad_ays2a = _row(schedule="ays", seed=0, context_idx=1, crps=2.0, mase=2.0)
        uniform2b = _row(schedule="uniform", seed=0, context_idx=2, crps=1.0, mase=1.0)
        good_ays2b = _row(schedule="ays", seed=0, context_idx=2, crps=0.5, mase=0.5)
        rows2 = attach_uniform_context_rewards(
            [uniform2a, bad_ays2a, uniform2b, good_ays2b],
            support_schedule_keys=("uniform", "ays"),
        )
        guard2 = build_calibration_holdout_non_regression_guard(
            ContextSwitchStudent(),
            _stamp_holdout(rows2),
            context_embeddings={
                context_id_from_row(uniform2a): np.asarray([0.0, 0.0], dtype=np.float32),
                context_id_from_row(uniform2b): np.asarray([1.0, 0.0], dtype=np.float32),
            },
            series_index_map=build_series_index_map(rows2),
            support_schedule_keys=("uniform", "ays"),
            margin=0.001,
            source_holdout_names=("context_disjoint",),
        )
        decision2 = guard2["cell_decision_map"]["euler/4"]
        self.assertEqual(decision2["deployed_mode"], "context_student")
        self.assertGreater(decision2["oracle_context_advantage_vs_best_static"], 0.0)
        self.assertEqual(decision2["oracle_support_usage"]["uniform"], 1)
        self.assertEqual(decision2["oracle_support_usage"]["ays"], 1)
        self.assertIn("context_disjoint", decision2["oracle_advantage_by_holdout"])
        self.assertEqual(guard2["observed_calibration_holdout_names"], ["context_disjoint"])

    def test_forecast_evaluator_emits_context_rows_and_embeddings(self) -> None:
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
            horizon = 1
            dataset_key = "solar_energy_10m"
            split_name = "train"

            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int):
                return torch.zeros(2, 1), torch.tensor([float(idx)]), self.example_metadata(idx)

            def example_metadata(self, idx: int):
                return {
                    "dataset_key": "solar_energy_10m",
                    "split": "train",
                    "series_id": "series_0",
                    "series_idx": 0,
                    "target_t": 2,
                    "history_start": 0,
                    "history_stop": 2,
                    "target_stop": 3,
                }

            def denormalize_block(self, block, idx: int):
                del idx
                return np.asarray(block, dtype=np.float32)

            def mase_denom(self, idx: int) -> float:
                del idx
                return 1.0

        class DummyBackbone:
            def precompute(self, hist):
                return SimpleNamespace(ctx_summary=torch.ones(hist.shape[0], 3), summary=torch.ones(hist.shape[0], 3) * 2.0)

        class DummyModel:
            def __init__(self, cfg):
                self.cfg = cfg
                self.backbone = DummyBackbone()

            def eval(self):
                return self

            def sample_future(self, hist, steps=None, solver=None):
                del steps, solver
                return torch.zeros(hist.shape[0], 1, 1)

        metrics = evaluate_forecast_schedule(
            DummyModel(cfg),
            DummyDataset(),
            cfg,
            solver_name="euler",
            runtime_nfe=4,
            target_nfe=4,
            time_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
            num_eval_samples=1,
            seed=0,
            scheduler_key="uniform",
            dataset_key="solar_energy_10m",
            split_phase="train_tuning",
            return_per_example_rows=True,
            return_context_embeddings=True,
        )
        self.assertEqual(metrics["eval_examples"], 1)
        self.assertEqual(len(metrics["per_example_rows"]), 1)
        row = metrics["per_example_rows"][0]
        self.assertEqual(row["split_phase"], "train_tuning")
        self.assertEqual(row["runtime_nfe"], 4)
        self.assertEqual(row["target_nfe"], 4)
        self.assertEqual(row["realized_nfe"], 4)
        self.assertIn(row["context_id"], metrics["context_embeddings"])
        self.assertEqual(len(metrics["context_embeddings"][row["context_id"]]), 3)

    def test_locked_report_argmax_ignores_locked_metric_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context_row = _row(schedule="uniform", seed=0, context_idx=0, crps=1.0, mase=1.0, split_phase="locked_test", series_id="series_0")
            context_id = context_id_from_row(context_row)
            locked_rows = [
                dict(context_row, scheduler_key="uniform", crps=1.0, mase=1.0),
                dict(context_row, scheduler_key="ays", crps=10.0, mase=10.0),
            ]
            locked_rows_path = root / "locked_rows.csv"
            _write_rows_csv(locked_rows_path, locked_rows)
            embeddings_path = root / "locked_embeddings.npz"
            save_context_embedding_table(embeddings_path, {context_id: np.asarray([0.0, 0.0], dtype=np.float32)})
            guard = _guard_for_cell(mode="static_support", fallback="uniform")
            summary_path = root / "context_conditional_summary.json"
            summary_path.write_text(
                json.dumps({"locked_test_used_for_selection": False, "policy_id": "policy_test", "calibration_holdout_non_regression_guard": guard}),
                encoding="utf-8",
            )
            student = ContextSupportStudentMLP(
                setting_dim=int(setting_features("euler", 4).numel()),
                context_dim=2,
                num_series=1,
                support_schedule_keys=("uniform", "ays"),
            )
            for param in student.parameters():
                param.data.zero_()
            last_linear = [module for module in student.net.modules() if isinstance(module, torch.nn.Linear)][-1]
            last_linear.bias.data[:] = torch.tensor([0.0, 1.0])
            checkpoint_path = root / "context_student.pt"
            torch.save(
                {
                    "state_dict": student.state_dict(),
                    "series_index_map": {"series_0": 0},
                    "context_dim": 2,
                    "support_schedule_keys": ["uniform", "ays"],
                    "embedding_normalizer": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    "policy_id": "policy_test",
                    "calibration_holdout_non_regression_guard": guard,
                },
                checkpoint_path,
            )
            args = build_locked_report_argparser().parse_args(
                [
                    "--context_student_checkpoint",
                    str(checkpoint_path),
                    "--training_summary",
                    str(summary_path),
                    "--locked_context_rows",
                    str(locked_rows_path),
                    "--locked_context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(root / "report"),
                    "--support_schedule_keys",
                    "uniform,ays",
                    "--seeds",
                    "0",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                ]
            )
            summary = report_context_locked_test(args)
        self.assertAlmostEqual(float(summary["pre_guard_argmax_crps_mean"]), 10.0)
        self.assertAlmostEqual(float(summary["selected_guarded_crps_mean"]), 1.0)
        self.assertLess(float(summary["expected_policy_crps_mean"]), 10.0)
        self.assertEqual(summary["selection_source"], "frozen_calibration_guarded_policy")
        self.assertEqual(summary["support_usage_by_solver_nfe"]["euler/4"]["uniform"], 1)
        self.assertEqual(summary["student_argmax_support_usage_by_solver_nfe"]["euler/4"]["ays"], 1)
        self.assertTrue(summary["frozen_guard_table_applied"])
        self.assertEqual(summary["guard_fallback_count"], 1)
        self.assertFalse(summary["locked_test_used_for_guard_construction"])
        self.assertFalse(summary["locked_test_used_for_selection"])

    def test_locked_report_rejects_non_locked_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            row = _row(schedule="uniform", seed=0, context_idx=0, crps=1.0, mase=1.0, split_phase="train_tuning")
            rows_path = root / "rows.csv"
            _write_rows_csv(rows_path, [row])
            embeddings_path = root / "embeddings.npz"
            save_context_embedding_table(embeddings_path, {context_id_from_row(row): np.asarray([0.0, 0.0], dtype=np.float32)})
            guard = _guard_for_cell(mode="context_student", support=["uniform"])
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps({"locked_test_used_for_selection": False, "policy_id": "policy_test", "calibration_holdout_non_regression_guard": guard}),
                encoding="utf-8",
            )
            student = ContextSupportStudentMLP(
                setting_dim=int(setting_features("euler", 4).numel()),
                context_dim=2,
                num_series=1,
                support_schedule_keys=("uniform",),
            )
            checkpoint_path = root / "student.pt"
            torch.save(
                {
                    "state_dict": student.state_dict(),
                    "series_index_map": {"series_0": 0},
                    "context_dim": 2,
                    "support_schedule_keys": ["uniform"],
                    "embedding_normalizer": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    "policy_id": "policy_test",
                    "calibration_holdout_non_regression_guard": guard,
                },
                checkpoint_path,
            )
            args = build_locked_report_argparser().parse_args(
                [
                    "--context_student_checkpoint",
                    str(checkpoint_path),
                    "--training_summary",
                    str(summary_path),
                    "--locked_context_rows",
                    str(rows_path),
                    "--locked_context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(root / "report"),
                    "--support_schedule_keys",
                    "uniform",
                    "--seeds",
                    "0",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                ]
            )
            with self.assertRaisesRegex(ValueError, "split_phase='locked_test'"):
                report_context_locked_test(args)

    def test_locked_report_rejects_guard_hash_mismatch_and_locked_source(self) -> None:
        def write_case(root: Path, *, checkpoint_guard: dict, summary_guard: dict):
            row = _row(schedule="uniform", seed=0, context_idx=0, crps=1.0, mase=1.0, split_phase="locked_test")
            rows_path = root / "rows.csv"
            _write_rows_csv(rows_path, [row])
            embeddings_path = root / "embeddings.npz"
            save_context_embedding_table(embeddings_path, {context_id_from_row(row): np.asarray([0.0, 0.0], dtype=np.float32)})
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps({"locked_test_used_for_selection": False, "policy_id": "policy_test", "calibration_holdout_non_regression_guard": summary_guard}),
                encoding="utf-8",
            )
            student = ContextSupportStudentMLP(
                setting_dim=int(setting_features("euler", 4).numel()),
                context_dim=2,
                num_series=1,
                support_schedule_keys=("uniform",),
            )
            checkpoint_path = root / "student.pt"
            torch.save(
                {
                    "state_dict": student.state_dict(),
                    "series_index_map": {"series_0": 0},
                    "context_dim": 2,
                    "support_schedule_keys": ["uniform"],
                    "embedding_normalizer": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    "policy_id": "policy_test",
                    "calibration_holdout_non_regression_guard": checkpoint_guard,
                },
                checkpoint_path,
            )
            return build_locked_report_argparser().parse_args(
                [
                    "--context_student_checkpoint",
                    str(checkpoint_path),
                    "--training_summary",
                    str(summary_path),
                    "--locked_context_rows",
                    str(rows_path),
                    "--locked_context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(root / "report"),
                    "--support_schedule_keys",
                    "uniform",
                    "--seeds",
                    "0",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            guard = _guard_for_cell(mode="context_student", support=["uniform"])
            mismatched = dict(guard, guard_table_hash="different_hash")
            with self.assertRaisesRegex(ValueError, "guard hash"):
                report_context_locked_test(write_case(root, checkpoint_guard=guard, summary_guard=mismatched))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            guard = dict(_guard_for_cell(mode="context_student", support=["uniform"]), source_split_phases=["locked_test"])
            with self.assertRaisesRegex(ValueError, "source_split_phases"):
                report_context_locked_test(write_case(root, checkpoint_guard=guard, summary_guard=guard))


if __name__ == "__main__":
    unittest.main()
