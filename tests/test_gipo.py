from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from genode.gipo.policy import (
    GIPO_PROTOCOL,
    GIPODensityStudentMLP,
    GIPOScheduleTeacherMLP,
    DensityFeatureNormalizer,
    attach_uniform_gipo_rewards,
    build_series_index_map,
    build_teacher_weighted_density_targets,
    context_id_from_row,
    gipo_teacher_diagnostics,
    density_mass_for_row,
    predict_gipo_density,
    save_context_embedding_table,
    train_gipo_student,
    train_gipo_teacher,
    validate_gipo_support_schedule_keys,
)
from genode.gipo.density_representation import (
    DENSITY_PROTOCOL,
    density_log_features,
    density_mass_hash,
    density_mass_to_time_grid,
    density_metadata,
    grid_to_density_mass,
    reference_grid_hash,
    sanitize_density_mass,
    uniform_reference_grid,
    validate_reference_grid,
)
from genode.gipo.models import setting_features
from genode.gipo.train_gipo import (
    build_argparser as build_train_argparser,
    train_gipo,
)


def _row(
    *,
    schedule: str,
    seed: int = 0,
    context_idx: int = 0,
    crps: float = 1.0,
    mase: float = 1.0,
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


def _rewarded_context_rows() -> tuple[list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    embeddings: dict[str, np.ndarray] = {}
    for context_idx in range(3):
        series_id = f"series_{context_idx // 2}"
        uniform = _row(schedule="uniform", context_idx=context_idx, crps=2.0, mase=2.0, series_id=series_id)
        ays = _row(schedule="ays", context_idx=context_idx, crps=1.0 + 0.1 * context_idx, mase=1.2, series_id=series_id)
        rows.extend([uniform, ays])
        embeddings[context_id_from_row(uniform)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
    rewarded = attach_uniform_gipo_rewards(rows, support_schedule_keys=("uniform", "ays"))
    return rewarded, embeddings


def _density_normalizer(rows: list[dict], reference_time_grid: tuple[float, ...]) -> DensityFeatureNormalizer:
    masses = [
        density_mass_for_row(row, schedule_grids={}, reference_time_grid=reference_time_grid)
        for row in rows
    ]
    return DensityFeatureNormalizer.fit(masses, reference_time_grid=reference_time_grid)


class _LastBinTeacher(torch.nn.Module):
    def forward(self, setting, density_feature, series, context):
        del setting, series, context
        return density_feature[:, -1]


class ContextDensityGIPOContractTests(unittest.TestCase):
    def test_density_representation_round_trips_mass_and_metadata(self) -> None:
        reference = uniform_reference_grid(4)
        mass = grid_to_density_mass((0.0, 0.25, 0.5, 0.75, 1.0), reference_time_grid=reference, macro_steps=4)

        self.assertEqual(validate_reference_grid(reference), reference)
        self.assertTrue(np.allclose(mass, np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float64)))
        self.assertAlmostEqual(sum(mass), 1.0)
        self.assertTrue(np.allclose(density_mass_to_time_grid(mass, macro_steps=4, reference_time_grid=reference), reference))
        self.assertEqual(density_metadata(reference)["density_protocol"], DENSITY_PROTOCOL)
        self.assertEqual(density_metadata(reference)["reference_grid_hash"], reference_grid_hash(reference))
        self.assertTrue(density_mass_hash(mass).startswith("density_"))
        self.assertEqual(density_log_features(mass, reference_time_grid=reference).shape, (4,))

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_reference_grid((0.0, 0.5, 0.5, 1.0))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            sanitize_density_mass((0.5, -0.1, 0.6))

    def test_gipo_outputs_mass_and_predictable_grid(self) -> None:
        torch.manual_seed(0)
        reference = uniform_reference_grid(4)
        student = GIPODensityStudentMLP(
            setting_dim=int(setting_features("euler", 4).numel()),
            density_dim=4,
            context_dim=2,
            num_series=1,
            series_embedding_dim=4,
            hidden_dim=8,
            hidden_layers=1,
        )

        setting = setting_features("euler", 4)[None, :].repeat(2, 1)
        mass = student.density_mass(
            setting,
            torch.tensor([0, 99], dtype=torch.long),
            torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32),
        )

        self.assertEqual(tuple(mass.shape), (2, 4))
        self.assertTrue(torch.all(mass > 0.0))
        self.assertTrue(torch.allclose(mass.sum(dim=-1), torch.ones(2), atol=1e-6))

        prediction = predict_gipo_density(
            student,
            row=_row(schedule="uniform", series_id="series_0"),
            context_embedding=np.asarray([0.0, 0.0], dtype=np.float32),
            series_index_map={"series_0": 0},
            reference_time_grid=reference,
        )
        self.assertEqual(prediction["density_protocol"], DENSITY_PROTOCOL)
        self.assertEqual(prediction["reference_bin_count"], 4)
        self.assertEqual(len(prediction["time_grid"]), 5)
        self.assertAlmostEqual(sum(prediction["density_mass"]), 1.0)

    def test_rank_huber_teacher_trains_on_density_features(self) -> None:
        torch.manual_seed(1)
        rewarded, embeddings = _rewarded_context_rows()
        fit_rows = [row for row in rewarded if int(row["example_idx"]) < 2]
        holdout_rows = [row for row in rewarded if int(row["example_idx"]) == 2]
        reference = uniform_reference_grid(8)
        series_map = build_series_index_map(fit_rows)
        normalizer = _density_normalizer(fit_rows, reference)
        teacher = GIPOScheduleTeacherMLP(
            setting_dim=int(setting_features("euler", 4).numel()),
            density_dim=8,
            context_dim=2,
            num_series=len(series_map),
            series_embedding_dim=4,
            hidden_dim=8,
            hidden_layers=1,
        )

        summary = train_gipo_teacher(
            teacher,
            fit_rows,
            context_embeddings=embeddings,
            series_index_map=series_map,
            schedule_grids={},
            reference_time_grid=reference,
            density_normalizer=normalizer,
            steps=2,
            lr=1e-3,
            diagnostic_splits={"context_disjoint": holdout_rows},
            teacher_checkpoint_every=1,
            allowed_schedule_keys=("uniform", "ays"),
        )

        self.assertEqual(summary["teacher_objective"], "pairwise_rank_plus_huber_regression")
        self.assertEqual(summary["teacher_density_feature"], "train_normalized_log_density")
        self.assertGreater(summary["teacher_pair_count"], 0)
        self.assertIn("teacher_rank_loss", summary["losses"][-1])
        self.assertIn("teacher_huber_loss", summary["losses"][-1])
        self.assertEqual(summary["teacher_checkpoint_selection"]["selection_metric"], "mean_context_series_total_loss")

        diagnostics = gipo_teacher_diagnostics(
            teacher,
            holdout_rows,
            context_embeddings=embeddings,
            series_index_map=series_map,
            schedule_grids={},
            reference_time_grid=reference,
            density_normalizer=normalizer,
            fit_series_keys=sorted(series_map),
        )
        self.assertIn("best_candidate_agreement", diagnostics)
        self.assertNotIn("support_top1_accuracy", diagnostics)

    def test_teacher_weighted_density_targets_drive_density_student_kl(self) -> None:
        torch.manual_seed(2)
        reference = uniform_reference_grid(4)
        ser_schedule_key = "ser_ptg_local_defect_eta005"
        rows = [
            _row(schedule="uniform", context_idx=0, series_id="series_0"),
            _row(schedule=ser_schedule_key, context_idx=0, series_id="series_0"),
        ]
        schedule_grids = {(ser_schedule_key, "euler", 4): (0.0, 0.7, 0.8, 0.9, 1.0)}
        context_id = context_id_from_row(rows[0])
        embeddings = {context_id: np.asarray([0.0, 0.0], dtype=np.float32)}
        series_map = build_series_index_map(rows)
        normalizer = DensityFeatureNormalizer(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
        )
        teacher = _LastBinTeacher()

        _, _, _, target_mass, target_summary = build_teacher_weighted_density_targets(
            teacher,
            rows,
            context_embeddings=embeddings,
            series_index_map=series_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference,
            density_normalizer=normalizer,
            temperature=0.05,
        )

        uniform_mass = np.asarray(density_mass_for_row(rows[0], schedule_grids=schedule_grids, reference_time_grid=reference))
        ser_mass = np.asarray(density_mass_for_row(rows[1], schedule_grids=schedule_grids, reference_time_grid=reference))
        utilities = np.asarray(
            [
                normalizer.transform_one(uniform_mass, reference_time_grid=reference)[-1],
                normalizer.transform_one(ser_mass, reference_time_grid=reference)[-1],
            ],
            dtype=np.float64,
        )
        logits = utilities / 0.05
        logits -= float(np.max(logits))
        expected_weights = np.exp(logits)
        expected_weights /= float(np.sum(expected_weights))
        expected_mixture = expected_weights[0] * uniform_mass + expected_weights[1] * ser_mass
        expected_mixture /= float(np.sum(expected_mixture))
        self.assertEqual(target_summary["target_protocol"], "teacher_weighted_density_mle")
        self.assertEqual(target_summary["density_protocol"], DENSITY_PROTOCOL)
        self.assertEqual(target_summary["teacher_temperature_mode"], "fixed")
        self.assertEqual(tuple(target_mass.shape), (1, 4))
        self.assertAlmostEqual(float(target_mass.sum()), 1.0, places=6)
        self.assertTrue(np.allclose(target_mass.detach().cpu().numpy()[0], expected_mixture.astype(np.float32), atol=1e-7))
        self.assertGreater(float(target_mass[0, -1]), float(uniform_mass[-1]))
        for key in (
            "teacher_candidate_entropy_mean",
            "teacher_candidate_entropy_p05",
            "teacher_candidate_entropy_p50",
            "teacher_candidate_entropy_p95",
            "teacher_candidate_ess_mean",
            "teacher_candidate_ess_p05",
            "teacher_candidate_ess_p50",
            "teacher_candidate_ess_p95",
            "teacher_candidate_max_weight_mean",
            "teacher_candidate_max_weight_p05",
            "teacher_candidate_max_weight_p50",
            "teacher_candidate_max_weight_p95",
            "teacher_chosen_temperature_mean",
            "teacher_chosen_temperature_p05",
            "teacher_chosen_temperature_p50",
            "teacher_chosen_temperature_p95",
        ):
            self.assertIn(key, target_summary)

        student = GIPODensityStudentMLP(
            setting_dim=int(setting_features("euler", 4).numel()),
            density_dim=4,
            context_dim=2,
            num_series=len(series_map),
            series_embedding_dim=4,
            hidden_dim=8,
            hidden_layers=1,
        )
        student_summary = train_gipo_student(
            student,
            teacher,
            rows,
            context_embeddings=embeddings,
            series_index_map=series_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference,
            density_normalizer=normalizer,
            steps=2,
            lr=1e-3,
            teacher_temperature=0.05,
            series_unknown_dropout=0.0,
        )
        self.assertEqual(student_summary["student_policy_type"], "continuous_density")
        self.assertEqual(student_summary["student_objective"], "teacher_weighted_density_mle_kl")
        self.assertIn("student_kl_ce_loss", student_summary["losses"][-1])

    def test_adaptive_teacher_temperature_hits_target_ess(self) -> None:
        reference = uniform_reference_grid(4)
        ser_schedule_key = "ser_ptg_local_defect_eta005"
        rows = [
            _row(schedule="uniform", context_idx=0, series_id="series_0"),
            _row(schedule=ser_schedule_key, context_idx=0, series_id="series_0"),
        ]
        schedule_grids = {(ser_schedule_key, "euler", 4): (0.0, 0.7, 0.8, 0.9, 1.0)}
        context_id = context_id_from_row(rows[0])
        embeddings = {context_id: np.asarray([0.0, 0.0], dtype=np.float32)}
        series_map = build_series_index_map(rows)
        normalizer = DensityFeatureNormalizer(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
        )
        teacher = _LastBinTeacher()

        summaries = []
        for target_ess in (1.2, 1.5, 1.8):
            _, _, _, _, target_summary = build_teacher_weighted_density_targets(
                teacher,
                rows,
                context_embeddings=embeddings,
                series_index_map=series_map,
                schedule_grids=schedule_grids,
                reference_time_grid=reference,
                density_normalizer=normalizer,
                temperature_mode="adaptive_ess",
                target_ess=target_ess,
                min_temperature=0.001,
                max_temperature=10.0,
            )
            self.assertEqual(target_summary["teacher_temperature_mode"], "adaptive_ess")
            self.assertAlmostEqual(target_summary["teacher_candidate_ess_mean"], target_ess, places=3)
            summaries.append(target_summary)

        self.assertLess(summaries[0]["teacher_chosen_temperature_mean"], summaries[1]["teacher_chosen_temperature_mean"])
        self.assertLess(summaries[1]["teacher_chosen_temperature_mean"], summaries[2]["teacher_chosen_temperature_mean"])

    def test_gipo_trainer_refuses_locked_test_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            locked_row = _row(schedule="uniform", split_phase="locked_test")
            rows_path = root / "rows.csv"
            embeddings_path = root / "embeddings.npz"
            _write_rows_csv(rows_path, [locked_row])
            save_context_embedding_table(
                embeddings_path,
                {context_id_from_row(locked_row): np.asarray([0.0, 0.0], dtype=np.float32)},
            )
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
                train_gipo(args)

    def test_gipo_trainer_dry_run_persists_split_membership(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows: list[dict] = []
            embeddings: dict[str, np.ndarray] = {}
            for context_idx in range(8):
                series_id = f"series_{context_idx // 2}"
                for schedule, crps in (("uniform", 2.0), ("ays", 1.0 + 0.01 * context_idx)):
                    row = _row(schedule=schedule, context_idx=context_idx, crps=crps, series_id=series_id)
                    rows.append(row)
                embeddings[context_id_from_row(rows[-1])] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
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
                    "8",
                    "--dry_run",
                ]
            )

            summary = train_gipo(args)

            self.assertEqual(summary["status"], "dry_run")
            membership = summary["split_membership"]
            self.assertEqual(set(membership), {"fit", "context_disjoint", "series_disjoint"})
            self.assertGreater(membership["fit"]["context_count"], 0)
            self.assertGreater(membership["context_disjoint"]["context_count"], 0)
            self.assertGreater(membership["series_disjoint"]["series_count"], 0)
            self.assertEqual(len(membership["fit"]["context_id_hash"]), 64)
            self.assertIn("context_ids", membership["context_disjoint"])

    def test_locked_reporter_rejects_legacy_categorical_protocol(self) -> None:
        from genode.gipo.report_locked_test import (
            build_argparser as build_locked_report_argparser,
            report_gipo_locked_test,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            locked_row = _row(schedule="uniform", split_phase="locked_test", series_id="series_0")
            rows_path = root / "locked_rows.csv"
            embeddings_path = root / "embeddings.npz"
            checkpoint_path = root / "legacy_gipo_student.pt"
            summary_path = root / "summary.json"
            _write_rows_csv(rows_path, [locked_row])
            save_context_embedding_table(
                embeddings_path,
                {context_id_from_row(locked_row): np.asarray([0.0, 0.0], dtype=np.float32)},
            )
            legacy_guard = {
                "guard_id": "legacy_guard",
                "guard_table_hash": "legacy_hash",
                "support_schedule_keys": ["uniform"],
                "locked_test_used_for_selection": False,
                "locked_test_used_for_guard_construction": False,
                "source_split_phases": ["train_tuning"],
                "observed_calibration_holdout_names": ["context_disjoint"],
                "cell_decision_map": {"euler/4": {"deployed_mode": "gipo"}},
            }
            torch.save(
                {
                    "protocol": "legacy_categorical_gipo_v1",
                    "student_policy_type": "categorical_support_fixed_ser",
                    "policy_id": "legacy_policy",
                    "state_dict": {},
                    "series_index_map": {"series_0": 0},
                    "context_dim": 2,
                    "support_schedule_keys": ["uniform"],
                    "embedding_normalizer": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    "calibration_holdout_non_regression_guard": legacy_guard,
                },
                checkpoint_path,
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "protocol": "legacy_categorical_gipo_v1",
                        "student_policy_type": "categorical_support_fixed_ser",
                        "policy_id": "legacy_policy",
                        "locked_test_used_for_selection": False,
                        "calibration_holdout_non_regression_guard": legacy_guard,
                    }
                ),
                encoding="utf-8",
            )
            args = build_locked_report_argparser().parse_args(
                [
                    "--gipo_student_checkpoint",
                    str(checkpoint_path),
                    "--training_summary",
                    str(summary_path),
                    "--context_rows",
                    str(rows_path),
                    "--context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(root / "report"),
                    "--seeds",
                    "0",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                ]
            )

            with self.assertRaisesRegex(ValueError, "protocol|continuous_density|gipo_density_v1|categorical"):
                report_gipo_locked_test(args)

    def test_calibration_reporter_refuses_locked_test_rows(self) -> None:
        from genode.gipo.report_locked_test import (
            build_argparser as build_locked_report_argparser,
            report_gipo_locked_test,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = uniform_reference_grid(4)
            locked_row = _row(schedule="uniform", split_phase="locked_test", series_id="series_0")
            rows_path = root / "locked_rows.csv"
            embeddings_path = root / "embeddings.npz"
            checkpoint_path = root / "gipo_student.pt"
            summary_path = root / "summary.json"
            _write_rows_csv(rows_path, [locked_row])
            save_context_embedding_table(
                embeddings_path,
                {context_id_from_row(locked_row): np.asarray([0.0, 0.0], dtype=np.float32)},
            )
            student = GIPODensityStudentMLP(
                setting_dim=int(setting_features("euler", 4).numel()),
                density_dim=4,
                context_dim=2,
                num_series=1,
            )
            torch.save(
                {
                    "protocol": GIPO_PROTOCOL,
                    "student_policy_type": "continuous_density",
                    "student_objective": "teacher_weighted_density_mle_kl",
                    "student_state": student.state_dict(),
                    "setting_dim": int(setting_features("euler", 4).numel()),
                    "density_dim": 4,
                    "context_dim": 2,
                    "series_index_map": {"series_0": 0},
                    "embedding_normalizer": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    "density_representation": density_metadata(reference),
                    "locked_test_used_for_selection": False,
                },
                checkpoint_path,
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "protocol": GIPO_PROTOCOL,
                        "student_policy_type": "continuous_density",
                        "locked_test_used_for_selection": False,
                    }
                ),
                encoding="utf-8",
            )
            args = build_locked_report_argparser().parse_args(
                [
                    "--gipo_student_checkpoint",
                    str(checkpoint_path),
                    "--training_summary",
                    str(summary_path),
                    "--context_rows",
                    str(rows_path),
                    "--context_embeddings_npz",
                    str(embeddings_path),
                    "--split_phase",
                    "locked_test",
                    "--selection_mode",
                    "calibration",
                    "--out_dir",
                    str(root / "report"),
                    "--seeds",
                    "0",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                ]
            )

            with self.assertRaisesRegex(ValueError, "refuses locked_test rows"):
                report_gipo_locked_test(args)

    def test_student_selection_context_filter_uses_logical_parent_seed(self) -> None:
        from scripts.verification_student_selection_20260603.collect_student_selection_summary import (
            _filter_to_student_contexts,
        )

        student = _row(schedule="gipo", seed=0, solver_key="euler", target_nfe=8)
        student["source_split_phase"] = "validation_tuning"
        student["split_phase"] = "context_disjoint"
        baseline = _row(schedule="uniform", seed=1000, solver_key="euler", target_nfe=8)
        baseline["split_phase"] = "validation_tuning"
        context_id = context_id_from_row(baseline)
        student["context_id"] = context_id
        baseline["context_id"] = context_id
        baseline["parent_row_signature"] = (
            "solar_energy_10m|validation_tuning|0|8|euler|uniform|solar_energy_10m_otflow_forecast_20k_seed0"
        )

        self.assertEqual(_filter_to_student_contexts([baseline], [student]), [baseline])

    def test_density_supervision_rejects_bo_like_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "BO/candidate"):
            validate_gipo_support_schedule_keys(("uniform", "bo_candidate_000"))
        self.assertEqual(
            validate_gipo_support_schedule_keys(
                ("uniform", "ays_reversed", "ser_ptg_local_defect_eta005")
            ),
            ("uniform", "ays_reversed", "ser_ptg_local_defect_eta005"),
        )
        self.assertEqual(GIPO_PROTOCOL, "gipo_density_v1")


if __name__ == "__main__":
    unittest.main()
