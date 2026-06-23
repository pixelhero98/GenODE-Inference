from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from genode.canonical_experiment_layout import CANONICAL_SEEN_NFES
from genode.gipo.density_representation import (
    DENSITY_PROTOCOL,
    density_mass_to_time_grid,
    density_metadata,
    grid_to_density_mass,
    uniform_reference_grid,
)
from genode.gipo.models import SETTING_ENCODER_MODE_CONTINUOUS_V3, build_setting_encoder_config, setting_feature_dim, setting_features
from genode.gipo.policy import (
    ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
    ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
    CONDITIONING_STYLE_ADDITIVE_MLP,
    DENSITY_TOKEN_ATTENTION_ROPE,
    GIPO_PROTOCOL,
    GIPODensityFormTeacherTransformer,
    GIPODensityQueryStudentTransformer,
    MODEL_PAYLOAD_VERSION,
    SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
    TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET,
    TEACHER_METRIC_TARGET_KEYS,
    TEACHER_METRIC_MASK_PROTOCOL,
    TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
    TEACHER_OUTPUT_METRIC_VECTOR,
    TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
    DensityFeatureNormalizer,
    build_gipo_student_model,
    build_gipo_teacher_model,
    build_teacher_weighted_density_targets,
    density_mass_for_row,
    _density_mass_to_normalized_log_features_torch,
    _group_indices_by_pair_key,
    _make_group_minibatch_sampler,
    _make_index_minibatch_sampler,
    _student_teacher_score_objective,
    _student_teacher_score_eta,
    realized_nfe_from_row,
    train_gipo_teacher,
    validate_canonical_conditioning_style,
)
from genode.gipo.train_gipo import build_argparser as build_train_argparser


def _teacher_training_payload() -> dict:
    return {
        "teacher_target": "metric_vector",
        "teacher_metric_targets": list(TEACHER_METRIC_TARGET_KEYS),
        "teacher_metric_target_protocol": TEACHER_METRIC_TARGET_PROTOCOL_VECTOR,
        "teacher_metric_mask_protocol": TEACHER_METRIC_MASK_PROTOCOL,
        "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE,
        "teacher_checkpoint_selection": {
            "selection_protocol": TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET,
            "selected_step": 1,
            "uses_validation_labels": False,
            "locked_test_used_for_selection": False,
        },
    }


class _RowScoreTeacher(torch.nn.Module):
    architecture = ARCHITECTURE_DENSITY_FORM_TRANSFORMER
    teacher_metric_targets = ("u_comp_uniform",)

    def forward(self, setting_feature_batch, density_feature_batch, series_index_batch, context_embedding_batch, *, rows=None):
        values = [float(row["teacher_score"]) for row in rows]
        return torch.tensor(values, dtype=density_feature_batch.dtype, device=density_feature_batch.device).unsqueeze(-1)


class _TwoMetricRowScoreTeacher(torch.nn.Module):
    architecture = ARCHITECTURE_DENSITY_FORM_TRANSFORMER
    teacher_metric_targets = ("u_comp_uniform", "u_alt_uniform")

    def forward(self, setting_feature_batch, density_feature_batch, series_index_batch, context_embedding_batch, *, rows=None):
        values = [[float(row["teacher_score"]), float(row["teacher_score"])] for row in rows]
        return torch.tensor(values, dtype=density_feature_batch.dtype, device=density_feature_batch.device)


class _DensityDependentTeacher(torch.nn.Module):
    architecture = ARCHITECTURE_DENSITY_FORM_TRANSFORMER
    teacher_metric_targets = ("u_comp_uniform",)

    def forward(self, setting_feature_batch, density_feature_batch, series_index_batch, context_embedding_batch, *, rows=None):
        return density_feature_batch.sum(dim=1, keepdim=True)


class GIPOCanonicalTests(unittest.TestCase):
    def test_density_grid_roundtrip_uses_canonical_64_bins(self) -> None:
        reference = uniform_reference_grid(64)
        source_grid = (0.0, 0.25, 0.5, 1.0)

        mass = grid_to_density_mass(source_grid, reference_time_grid=reference)
        reconstructed = density_mass_to_time_grid(mass, reference_time_grid=reference, macro_steps=3)
        metadata = density_metadata(reference)

        self.assertEqual(len(mass), 64)
        self.assertAlmostEqual(float(np.sum(mass)), 1.0, places=6)
        self.assertEqual(len(reconstructed), 4)
        self.assertEqual(metadata["density_protocol"], DENSITY_PROTOCOL)
        self.assertEqual(metadata["reference_bin_count"], 64)

    def test_density_mass_for_row_uses_train_steps_for_scoped_summary_grid(self) -> None:
        reference = uniform_reference_grid(4)
        scoped_grid = [0.0, 0.05, 0.4, 0.8, 1.0]
        unscoped_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        row = {
            "solver_key": "euler",
            "target_nfe": 4,
            "scheduler_key": "ser_ptg_test",
            "train_steps": 4000,
        }

        mass = density_mass_for_row(
            row,
            schedule_grids={
                ("ser_ptg_test", "euler", 4): unscoped_grid,
                ("ser_ptg_test", "euler", 4, 4000): scoped_grid,
            },
            reference_time_grid=reference,
        )

        self.assertEqual(mass, grid_to_density_mass(scoped_grid, reference_time_grid=reference, macro_steps=4))

    def test_realized_nfe_from_row_uses_solver_protocol_not_runtime_macro_steps(self) -> None:
        self.assertEqual(
            realized_nfe_from_row({"solver_key": "heun", "target_nfe": 4, "runtime_nfe": 2}),
            4,
        )

    def test_torch_density_features_match_numpy_and_keep_gradients(self) -> None:
        reference = uniform_reference_grid(4)
        masses = [
            (0.40, 0.30, 0.20, 0.10),
            (0.10, 0.20, 0.30, 0.40),
        ]
        normalizer = DensityFeatureNormalizer.fit(masses, reference_time_grid=reference)
        mass = torch.tensor([[0.40, 0.30, 0.20, 0.10]], dtype=torch.float32, requires_grad=True)

        features = _density_mass_to_normalized_log_features_torch(
            mass,
            reference_time_grid=reference,
            density_normalizer=normalizer,
        )
        expected = normalizer.transform_one(masses[0], reference_time_grid=reference)

        self.assertTrue(torch.allclose(features.detach().cpu(), torch.from_numpy(np.asarray([expected])), atol=1e-6))
        features.sum().backward()
        self.assertIsNotNone(mass.grad)
        self.assertTrue(torch.isfinite(mass.grad).all())

    def test_student_teacher_score_eta_late_ramps(self) -> None:
        self.assertEqual(_student_teacher_score_eta(step=0, steps=10, base_weight=0.05, warmup_fraction=0.6), 0.0)
        self.assertAlmostEqual(_student_teacher_score_eta(step=6, steps=10, base_weight=0.05, warmup_fraction=0.6), 0.0125)
        self.assertAlmostEqual(_student_teacher_score_eta(step=9, steps=10, base_weight=0.05, warmup_fraction=0.6), 0.05)

    def test_teacher_score_objective_backpropagates_through_frozen_teacher_input(self) -> None:
        reference = uniform_reference_grid(4)
        normalizer = DensityFeatureNormalizer(mean=np.zeros(4, dtype=np.float32), std=np.ones(4, dtype=np.float32))
        logits = torch.tensor([[0.4, -0.2, 0.1, -0.3]], dtype=torch.float32, requires_grad=True)
        setting = torch.zeros(1, int(setting_features("euler", 4).numel()), dtype=torch.float32)
        series = torch.zeros(1, dtype=torch.long)
        context = torch.zeros(1, 2, dtype=torch.float32)
        rows = [{"solver_key": "euler", "target_nfe": 4, "u_comp_uniform": 1.0}]
        teacher = _DensityDependentTeacher()
        for param in teacher.parameters():
            param.requires_grad_(False)

        objective, score = _student_teacher_score_objective(
            teacher,
            logits,
            setting,
            series,
            context,
            rows,
            reference_time_grid=reference,
            density_normalizer=normalizer,
            score_mean=torch.zeros(1),
            score_std=torch.ones(1),
            teacher_utility_weights=None,
            clip=5.0,
        )
        (-objective).backward()

        self.assertTrue(torch.isfinite(score.detach()).all())
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(torch.sum(torch.abs(logits.grad)).detach().cpu().item()), 0.0)

    def test_teacher_weighted_targets_support_elite_and_blend_modes(self) -> None:
        reference = uniform_reference_grid(64)
        schedules = ("uniform", "late_power_3", "flowts_power_sampling", "ays")
        scores = {
            "uniform": 0.0,
            "late_power_3": 3.0,
            "flowts_power_sampling": 1.0,
            "ays": 2.0,
        }
        rows = [
            {
                "dataset": "lobster_synthetic",
                "split_phase": "train_tuning",
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule,
                "context_id": "ctx_0",
                "series_id": "series_0",
                "target_t": 0,
                "u_comp_uniform": scores[schedule],
                "teacher_score": scores[schedule],
            }
            for schedule in schedules
        ]
        normalizer = DensityFeatureNormalizer.fit(
            [density_mass_for_row(row, schedule_grids=None, reference_time_grid=reference) for row in rows],
            reference_time_grid=reference,
        )
        kwargs = {
            "context_embeddings": {"ctx_0": [0.0, 0.0]},
            "series_index_map": {"series_0": 0},
            "schedule_grids": None,
            "reference_time_grid": reference,
            "density_normalizer": normalizer,
            "supervision_schedule_keys": schedules,
            "temperature": 1.0,
            "device": "cpu",
        }
        teacher = _RowScoreTeacher()

        _, _, _, full_mass, full_summary = build_teacher_weighted_density_targets(teacher, rows, **kwargs)
        _, _, _, elite_mass, elite_summary = build_teacher_weighted_density_targets(
            teacher,
            rows,
            student_target_mixture_mode="elite",
            student_target_elite_k=1,
            student_target_elite_min_count=1,
            **kwargs,
        )
        _, _, _, blend_mass, blend_summary = build_teacher_weighted_density_targets(
            teacher,
            rows,
            student_target_mixture_mode="elite_blend",
            student_target_elite_k=1,
            student_target_elite_min_count=1,
            student_target_elite_blend_all_weight=0.2,
            **kwargs,
        )
        best_mass = torch.tensor(
            [density_mass_for_row(rows[1], schedule_grids=None, reference_time_grid=reference)],
            dtype=torch.float32,
        )

        self.assertEqual(full_summary["student_target_mixture_mode"], "full")
        self.assertEqual(elite_summary["student_target_mixture_mode"], "elite")
        self.assertEqual(blend_summary["student_target_mixture_mode"], "elite_blend")
        self.assertTrue(torch.allclose(elite_mass.cpu(), best_mass, atol=1e-6))
        self.assertTrue(torch.allclose(blend_mass.cpu(), 0.8 * best_mass + 0.2 * full_mass.cpu(), atol=1e-6))
        self.assertEqual(elite_summary["teacher_candidate_elite_count_mean"], 1.0)
        self.assertLess(elite_summary["teacher_candidate_retained_full_weight_mean"], 1.0)

    def test_teacher_weighted_targets_reject_mixed_cell_scalarization(self) -> None:
        reference = uniform_reference_grid(64)
        rows = [
            {
                "dataset": "lobster_synthetic",
                "split_phase": "train_tuning",
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "uniform",
                "context_id": "ctx_0",
                "series_id": "series_0",
                "target_t": 0,
                "u_comp_uniform": 1.0,
                "u_alt_uniform": 1.0,
                "teacher_score": 1.0,
                "u_comp_uniform_weight": 1.0,
                "u_alt_uniform_weight": 0.0,
            },
            {
                "dataset": "lobster_synthetic",
                "split_phase": "train_tuning",
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "late_power_3",
                "context_id": "ctx_0",
                "series_id": "series_0",
                "target_t": 0,
                "u_comp_uniform": 2.0,
                "u_alt_uniform": 2.0,
                "teacher_score": 2.0,
                "u_comp_uniform_weight": 0.0,
                "u_alt_uniform_weight": 1.0,
            },
        ]
        normalizer = DensityFeatureNormalizer.fit(
            [density_mass_for_row(row, schedule_grids=None, reference_time_grid=reference) for row in rows],
            reference_time_grid=reference,
        )

        with self.assertRaisesRegex(ValueError, "identical teacher metric masks"):
            build_teacher_weighted_density_targets(
                _TwoMetricRowScoreTeacher(),
                rows,
                context_embeddings={"ctx_0": [0.0, 0.0]},
                series_index_map={"series_0": 0},
                schedule_grids=None,
                reference_time_grid=reference,
                density_normalizer=normalizer,
                supervision_schedule_keys=("uniform", "late_power_3"),
                temperature=1.0,
                device="cpu",
            )

    def test_minibatch_samplers_do_not_duplicate_within_epoch_wrap(self) -> None:
        index_sampler = _make_index_minibatch_sampler(5, batch_size=3, seed=0)
        first = index_sampler()
        second = index_sampler()
        third = index_sampler()
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(len(second), len(set(second)))
        self.assertEqual(len(third), len(set(third)))
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 2)
        self.assertEqual(len(third), 3)

        group_sampler = _make_group_minibatch_sampler([[idx] for idx in range(5)], group_batch_size=3, seed=0)
        first_groups = group_sampler()
        second_groups = group_sampler()
        third_groups = group_sampler()
        self.assertEqual(len(first_groups), len(set(first_groups)))
        self.assertEqual(len(second_groups), len(set(second_groups)))
        self.assertEqual(len(third_groups), len(set(third_groups)))
        self.assertEqual(len(first_groups), 3)
        self.assertEqual(len(second_groups), 2)
        self.assertEqual(len(third_groups), 3)

    def test_group_indices_accept_mixed_seed_none_values(self) -> None:
        pair_keys = [
            ("dataset", "euler", 4, "ctx_0", 0),
            ("dataset", "euler", 4, "ctx_0", None),
            ("dataset", "euler", 4, "ctx_0", 0),
        ]
        grouped = _group_indices_by_pair_key(pair_keys)
        self.assertEqual(grouped, [[0, 2], [1]])

    def test_teacher_training_summary_keeps_global_pair_count_under_minibatching(self) -> None:
        reference = uniform_reference_grid(64)
        rows = []
        for idx in range(70):
            for schedule, score in (("uniform", 0.0), ("late_power_3", 1.0)):
                rows.append(
                    {
                        "dataset": "lobster_synthetic",
                        "split_phase": "train_tuning",
                        "seed": 0,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "scheduler_key": schedule,
                        "context_id": f"ctx_{idx}",
                        "series_id": f"series_{idx}",
                        "target_t": 0,
                        "u_comp_uniform": score,
                    }
                )
        normalizer = DensityFeatureNormalizer.fit(
            [density_mass_for_row(row, schedule_grids=None, reference_time_grid=reference) for row in rows],
            reference_time_grid=reference,
        )
        teacher = build_gipo_teacher_model(
            architecture=ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
            setting_dim=int(setting_features("euler", 4).numel()),
            density_dim=64,
            context_dim=2,
            num_series=1,
            model_config={"hidden_dim": 16, "hidden_layers": 1, "attention_heads": 4, "dropout": 0.0},
        )

        summary = train_gipo_teacher(
            teacher,
            rows,
            context_embeddings={f"ctx_{idx}": [0.0, 0.0] for idx in range(70)},
            series_index_map={f"series_{idx}": 0 for idx in range(70)},
            schedule_grids=None,
            reference_time_grid=reference,
            density_normalizer=normalizer,
            steps=1,
            teacher_loss_log_every=1,
            final_retrain_mode=True,
            device="cpu",
        )

        self.assertEqual(summary["teacher_pair_count"], 70)
        self.assertEqual(summary["losses"][0]["teacher_pair_count"], 70)
        self.assertLess(summary["losses"][0]["teacher_batch_pair_count"], summary["teacher_pair_count"])

    def test_additive_models_are_context_only_for_series_identity(self) -> None:
        setting_batch = torch.stack(
            [
                setting_features("euler", 4, mode=SETTING_ENCODER_MODE_CONTINUOUS_V3),
                setting_features("euler", 4, mode=SETTING_ENCODER_MODE_CONTINUOUS_V3),
            ]
        )
        series_index = torch.tensor([0, 1], dtype=torch.long)
        context = torch.tensor([[0.1, 0.2], [0.1, 0.2]], dtype=torch.float32)
        density = torch.zeros(2, 64, dtype=torch.float32)
        rows = [
            {"series_id": "series_a", "target_t": 1},
            {"series_id": "series_b", "target_t": 1},
        ]

        student = GIPODensityQueryStudentTransformer(
            setting_dim=int(setting_batch.shape[1]),
            density_dim=64,
            context_dim=2,
            num_series=2,
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
        )
        teacher = GIPODensityFormTeacherTransformer(
            setting_dim=int(setting_batch.shape[1]),
            density_dim=64,
            context_dim=2,
            num_series=2,
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
        )
        student.eval()
        teacher.eval()

        with torch.no_grad():
            student_logits = student.logits(setting_batch, series_index, context, rows=rows)
            teacher_scores = teacher(setting_batch, density, series_index, context, rows=rows)

        self.assertTrue(torch.allclose(student_logits[0], student_logits[1], atol=1e-6))
        self.assertTrue(torch.allclose(teacher_scores[0], teacher_scores[1], atol=1e-6))
        self.assertEqual(student.model_config()["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP)
        self.assertEqual(teacher.model_config()["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP)
        self.assertEqual(student.model_config()["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
        self.assertEqual(teacher.model_config()["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
        self.assertEqual(student.model_config()["density_token_attention"], DENSITY_TOKEN_ATTENTION_ROPE)
        self.assertEqual(teacher.model_config()["density_token_attention"], DENSITY_TOKEN_ATTENTION_ROPE)

    def test_non_additive_conditioning_style_is_rejected(self) -> None:
        legacy_style = "ada" + "adaptive_layer_norm_zero"
        model_kwargs = {
            "setting_dim": 2,
            "density_dim": 64,
            "context_dim": 2,
            "num_series": 1,
            "hidden_dim": 16,
            "hidden_layers": 1,
            "attention_heads": 4,
            "dropout": 0.0,
            "conditioning_style": legacy_style,
        }

        with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp"):
            validate_canonical_conditioning_style({"conditioning_style": legacy_style})
        with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp"):
            GIPODensityQueryStudentTransformer(**model_kwargs)
        with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp"):
            GIPODensityFormTeacherTransformer(**model_kwargs)
        with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp"):
            build_gipo_student_model(
                architecture=ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
                setting_dim=2,
                density_dim=64,
                context_dim=2,
                num_series=1,
                model_config={"conditioning_style": legacy_style},
            )
        with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp"):
            build_gipo_teacher_model(
                architecture=ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
                setting_dim=2,
                density_dim=64,
                context_dim=2,
                num_series=1,
                model_config={"conditioning_style": legacy_style},
            )

    def test_trainer_argparser_exposes_only_canonical_conditioning(self) -> None:
        parser = build_train_argparser()
        options = {option for action in parser._actions for option in action.option_strings}

        self.assertIn("--teacher_unseen_selection_rows_csv", options)
        self.assertNotIn("--gipo_" + "conditioning_style", options)
        self.assertNotIn("--gipo_teacher_" + "conditioning_style", options)
        self.assertNotIn("--gipo_student_" + "conditioning_style", options)
        self.assertNotIn("--allow_noncanonical_conditioning", options)

    def test_student_checkpoint_loader_accepts_only_canonical_additive_student(self) -> None:
        from genode.gipo.report_locked_test import _load_student_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = build_setting_encoder_config(
                SETTING_ENCODER_MODE_CONTINUOUS_V3,
                observed_target_nfes=CANONICAL_SEEN_NFES,
                nfe_reference=16,
                rope_frequencies=(1.0, 2.0),
            )
            setting_dim = setting_feature_dim(SETTING_ENCODER_MODE_CONTINUOUS_V3, config=config)
            student = build_gipo_student_model(
                architecture=ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
                setting_dim=setting_dim,
                density_dim=64,
                context_dim=2,
                num_series=1,
                model_config={"hidden_dim": 16, "hidden_layers": 1, "attention_heads": 4, "dropout": 0.0},
            )
            checkpoint_path = root / "gipo_student.pt"
            payload = {
                "protocol": GIPO_PROTOCOL,
                "model_payload_version": MODEL_PAYLOAD_VERSION,
                "student_policy_type": "continuous_density",
                "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
                "student_model_config": student.model_config(),
                "student_state": student.state_dict(),
                "setting_dim": setting_dim,
                "setting_feature_mode": SETTING_ENCODER_MODE_CONTINUOUS_V3,
                "setting_encoder_config": config.to_payload(),
                "density_dim": 64,
                "context_dim": 2,
                "series_index_map": {"series_0": 0},
                "embedding_normalizer": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                "density_representation": density_metadata(uniform_reference_grid(64)),
                "teacher_training": _teacher_training_payload(),
                "teacher_utility_weights": {"crps": 0.5, "mase": 0.5},
                "locked_test_used_for_selection": False,
            }
            torch.save(payload, checkpoint_path)

            with mock.patch("genode.gipo.report_locked_test.torch.load", return_value=payload) as mocked_load:
                loaded_student, _, _, _, loaded_payload = _load_student_checkpoint(checkpoint_path)
            self.assertTrue(mocked_load.call_args.kwargs["weights_only"])
            self.assertEqual(loaded_student.setting_dim, setting_dim)
            self.assertEqual(loaded_payload["student_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP)
            loaded_student, _, _, _, loaded_payload = _load_student_checkpoint(checkpoint_path)
            self.assertEqual(loaded_student.setting_dim, setting_dim)
            self.assertEqual(loaded_payload["student_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP)

            legacy_path = root / "legacy_student.pt"
            legacy_payload = dict(payload)
            legacy_payload["student_model_config"] = dict(payload["student_model_config"])
            legacy_payload["student_model_config"]["conditioning_style"] = "ada" + "adaptive_layer_norm_zero"
            torch.save(legacy_payload, legacy_path)
            with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp"):
                _load_student_checkpoint(legacy_path)

            scalar_policy_path = root / "categorical_student.pt"
            scalar_policy_payload = dict(payload)
            scalar_policy_payload["student_policy_type"] = "categorical_support_fixed_ser"
            torch.save(scalar_policy_payload, scalar_policy_path)
            with self.assertRaisesRegex(ValueError, "continuous_density"):
                _load_student_checkpoint(scalar_policy_path)

    def test_locked_report_conditioning_metadata_is_canonical(self) -> None:
        from genode.gipo.report_locked_test import _conditioning_metadata_for_summary

        metadata = _conditioning_metadata_for_summary(
            {
                "conditioning_style": CONDITIONING_STYLE_ADDITIVE_MLP,
                "student_model_config": {"conditioning_style": CONDITIONING_STYLE_ADDITIVE_MLP},
            },
            {
                "conditioning_style": CONDITIONING_STYLE_ADDITIVE_MLP,
                "teacher_model_config": {"conditioning_style": CONDITIONING_STYLE_ADDITIVE_MLP},
            },
        )

        self.assertEqual(metadata, {"conditioning_style": CONDITIONING_STYLE_ADDITIVE_MLP})

    def test_teacher_model_config_contains_metric_vector_metadata(self) -> None:
        setting_dim = int(setting_features("euler", 4).numel())
        teacher = build_gipo_teacher_model(
            architecture=ARCHITECTURE_DENSITY_FORM_TRANSFORMER,
            setting_dim=setting_dim,
            density_dim=64,
            context_dim=2,
            num_series=1,
            model_config={"hidden_dim": 16, "hidden_layers": 1, "attention_heads": 4, "dropout": 0.0},
        )

        config = teacher.model_config()
        self.assertEqual(config["teacher_output"], TEACHER_OUTPUT_METRIC_VECTOR)
        self.assertEqual(config["teacher_metric_targets"], list(TEACHER_METRIC_TARGET_KEYS))
        self.assertEqual(config["teacher_metric_target_protocol"], TEACHER_METRIC_TARGET_PROTOCOL_VECTOR)
        self.assertEqual(config["teacher_metric_mask_protocol"], TEACHER_METRIC_MASK_PROTOCOL)
        self.assertEqual(config["teacher_scalarization"], TEACHER_SCALARIZATION_WEIGHTED_AVERAGE)


if __name__ == "__main__":
    unittest.main()
