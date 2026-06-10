from __future__ import annotations

import csv
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from genode.gipo.policy import (
    ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1,
    ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1,
    CONDITIONING_STYLE_ADALN_ZERO_V1,
    CONDITIONING_STYLE_ADDITIVE_MLP_V1,
    DENSITY_TOKEN_ATTENTION_ROPE_V1,
    GIPO_PROTOCOL,
    GIPODensityFormTeacherTransformer,
    GIPODensityQueryStudentTransformer,
    MODEL_PAYLOAD_VERSION,
    SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
    STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE_V1,
    TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1,
    TEACHER_METRIC_TARGET_KEYS,
    TEACHER_OUTPUT_METRIC_VECTOR_V1,
    TEACHER_SCALARIZATION_WEIGHTED_AVERAGE_V1,
    DensityFeatureNormalizer,
    attach_uniform_gipo_rewards,
    build_gipo_student_model,
    build_gipo_teacher_model,
    build_series_index_map,
    build_teacher_weighted_density_prediction_rows,
    build_teacher_weighted_density_targets,
    context_id_from_row,
    density_family_for_schedule_key,
    gipo_teacher_diagnostics,
    density_mass_for_row,
    nfe_sequence_diagnostic_summary,
    logical_seed_from_row,
    predict_gipo_density,
    predict_gipo_density_many,
    save_context_embedding_table,
    split_rows_by_density_family_holdout,
    teacher_selection_candidate_group_key,
    student_nfe_sequence_pair_indices,
    student_nfe_sequence_pairs,
    train_gipo_student,
    train_gipo_teacher,
    validate_density_family_holdout_schedule_keys,
    validate_gipo_conditioning_style,
    validate_gipo_support_schedule_keys,
    validate_teacher_objective_hyperparameters,
    _apply_density_bin_rope,
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
from genode.gipo.models import (
    SETTING_ENCODER_MODE_CONTINUOUS_V3,
    build_setting_encoder_config,
    setting_encoder_config_from_payload,
    setting_feature_dim,
    setting_features,
)
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


def _teacher_training_payload() -> dict:
    return {
        "teacher_target": "metric_vector",
        "teacher_metric_targets": list(TEACHER_METRIC_TARGET_KEYS),
        "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE_V1,
        "teacher_utility_weights": {"crps": 0.5, "mase": 0.5},
        "teacher_checkpoint_selection": {
            "selection_protocol": TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1,
            "selection_mode": TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1,
            "selection_metric": "weighted_normalized_regret_v1_score",
            "selected_step": 1,
            "uses_validation_labels": False,
            "locked_test_used_for_selection": False,
        },
        "teacher_final_retrain": {
            "enabled": True,
            "selected_step": 1,
            "locked_test_used_for_selection": False,
        },
        "final_teacher_retrain": {
            "enabled": True,
            "selected_step": 1,
            "locked_test_used_for_selection": False,
        },
    }


class _LastBinTeacher(torch.nn.Module):
    architecture = ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1

    def forward(self, setting, density_feature, series, context, **kwargs):
        del kwargs
        del setting, series, context
        last = density_feature[:, -1]
        return torch.stack([last, last], dim=-1)


class _ScheduleMetricTeacher(torch.nn.Module):
    architecture = ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1

    def forward(self, setting, density_feature, series, context, **kwargs):
        del setting, series, context
        rows = kwargs.get("rows")
        if rows is None:
            return torch.stack([density_feature[:, 0], density_feature[:, -1]], dim=-1)
        values = [
            (0.0, 1.0) if str(row["scheduler_key"]) == "uniform" else (1.0, 0.0)
            for row in rows
        ]
        return torch.tensor(values, dtype=density_feature.dtype, device=density_feature.device)


class _ScalarTeacher(torch.nn.Module):
    architecture = ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1

    def forward(self, setting, density_feature, series, context, **kwargs):
        del kwargs
        del setting, series, context
        return density_feature[:, -1]


class ContextDensityGIPOContractTests(unittest.TestCase):
    def test_setting_feature_mode_is_canonical_continuous_v3(self) -> None:
        continuous_config = build_setting_encoder_config(
            SETTING_ENCODER_MODE_CONTINUOUS_V3,
            observed_target_nfes=(4, 8, 12),
            nfe_reference=16,
            rope_frequencies=(1.0, 2.0),
        )
        continuous = setting_features("euler", 10, mode=SETTING_ENCODER_MODE_CONTINUOUS_V3, config=continuous_config)

        self.assertEqual(setting_feature_dim(SETTING_ENCODER_MODE_CONTINUOUS_V3, config=continuous_config), int(continuous.numel()))
        self.assertEqual(continuous_config.to_payload()["series_encoding"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
        rope = continuous[-4:].reshape(2, 2)
        self.assertTrue(torch.allclose(torch.sum(rope * rope, dim=1), torch.ones(2), atol=1e-6))
        self.assertEqual(continuous_config.to_payload()["rope_frequencies"], [1.0, 2.0])
        self.assertEqual(setting_encoder_config_from_payload(continuous_config.to_payload()).mode, SETTING_ENCODER_MODE_CONTINUOUS_V3)
        bad_payload = dict(continuous_config.to_payload())
        bad_payload["solver_metadata_version"] = "future_layout"
        with self.assertRaisesRegex(ValueError, "solver_metadata_version"):
            setting_encoder_config_from_payload(bad_payload)
        with self.assertRaisesRegex(ValueError, "requires an even target NFE"):
            setting_features("heun", 5, mode=SETTING_ENCODER_MODE_CONTINUOUS_V3, config=continuous_config)
        with self.assertRaisesRegex(ValueError, "setting_feature_mode"):
            setting_features("euler", 6, mode="nfe_" + "sequence_v2", config=continuous_config)

    def test_series_identity_is_not_part_of_gipo_conditioning(self) -> None:
        torch.manual_seed(21)
        base = _row(schedule="uniform", series_id="series_a", context_idx=0)
        changed = dict(base)
        changed["series_id"] = "series_b"
        changed["series_idx"] = 99
        sx = torch.stack(
            [
                setting_features(base["solver_key"], base["target_nfe"], mode=SETTING_ENCODER_MODE_CONTINUOUS_V3),
                setting_features(changed["solver_key"], changed["target_nfe"], mode=SETTING_ENCODER_MODE_CONTINUOUS_V3),
            ]
        )
        series_idx = torch.tensor([0, 1], dtype=torch.long)
        context = torch.tensor([[0.1, 0.2], [0.1, 0.2]], dtype=torch.float32)
        density = torch.tensor([[1.0, 0.0, -0.5, -1.0], [1.0, 0.0, -0.5, -1.0]], dtype=torch.float32)
        student = GIPODensityQueryStudentTransformer(setting_dim=int(sx.shape[1]), density_dim=4, context_dim=2, num_series=2, hidden_dim=16, hidden_layers=1, attention_heads=4, dropout=0.0)
        teacher = GIPODensityFormTeacherTransformer(setting_dim=int(sx.shape[1]), density_dim=4, context_dim=2, num_series=2, hidden_dim=16, hidden_layers=1, attention_heads=4, dropout=0.0)
        student.eval()
        teacher.eval()

        with torch.no_grad():
            student_logits = student.logits(sx, series_idx, context, rows=[base, changed])
            teacher_scores = teacher(sx, density, series_idx, context, rows=[base, changed])

        self.assertTrue(torch.allclose(student_logits[0], student_logits[1], atol=1e-6))
        self.assertTrue(torch.allclose(teacher_scores[0], teacher_scores[1], atol=1e-6))
        self.assertEqual(student.model_config()["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
        self.assertEqual(teacher.model_config()["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)

    def test_legacy_conditioning_style_is_rejected(self) -> None:
        legacy_style = CONDITIONING_STYLE_ADALN_ZERO_V1
        model_kwargs = {
            "setting_dim": 2,
            "density_dim": 4,
            "context_dim": 2,
            "num_series": 1,
            "hidden_dim": 16,
            "hidden_layers": 1,
            "attention_heads": 4,
            "dropout": 0.0,
            "conditioning_style": legacy_style,
        }

        with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp_v1"):
            validate_gipo_conditioning_style({"conditioning_style": legacy_style})
        with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp_v1"):
            GIPODensityQueryStudentTransformer(**model_kwargs)
        with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp_v1"):
            GIPODensityFormTeacherTransformer(**model_kwargs)
        with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp_v1"):
            build_gipo_student_model(
                architecture=ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1,
                setting_dim=2,
                density_dim=4,
                context_dim=2,
                num_series=1,
                model_config={"conditioning_style": legacy_style},
            )
        with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp_v1"):
            build_gipo_teacher_model(
                architecture=ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1,
                setting_dim=2,
                density_dim=4,
                context_dim=2,
                num_series=1,
                model_config={"conditioning_style": legacy_style},
            )

        self.assertEqual(
            validate_gipo_conditioning_style(
                {"conditioning_style": legacy_style},
                allow_noncanonical=True,
            ),
            CONDITIONING_STYLE_ADALN_ZERO_V1,
        )
        student = GIPODensityQueryStudentTransformer(
            **model_kwargs,
            allow_noncanonical_conditioning=True,
        )
        teacher = GIPODensityFormTeacherTransformer(
            **model_kwargs,
            allow_noncanonical_conditioning=True,
        )
        self.assertEqual(student.model_config()["conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
        self.assertEqual(teacher.model_config()["conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
        built_student = build_gipo_student_model(
            architecture=ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1,
            setting_dim=2,
            density_dim=4,
            context_dim=2,
            num_series=1,
            model_config={"conditioning_style": legacy_style},
            allow_noncanonical_conditioning=True,
        )
        built_teacher = build_gipo_teacher_model(
            architecture=ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1,
            setting_dim=2,
            density_dim=4,
            context_dim=2,
            num_series=1,
            model_config={"conditioning_style": legacy_style},
            allow_noncanonical_conditioning=True,
        )
        sx = torch.zeros(2, 2)
        series_idx = torch.zeros(2, dtype=torch.long)
        context = torch.ones(2, 2)
        density = torch.zeros(2, 4)
        with torch.no_grad():
            self.assertEqual(built_student.logits(sx, series_idx, context).shape, (2, 4))
            self.assertEqual(built_teacher(sx, density, series_idx, context).shape, (2, len(TEACHER_METRIC_TARGET_KEYS)))

    def test_nfe_sequence_diagnostics_detect_multi_nfe_groups(self) -> None:
        rows: list[dict] = []
        for nfe in (4, 8, 12):
            rows.extend(
                [
                    _row(schedule="uniform", context_idx=0, target_nfe=nfe, series_id="series_0"),
                    _row(schedule="ays", context_idx=0, target_nfe=nfe, series_id="series_0"),
                ]
            )

        summary = nfe_sequence_diagnostic_summary(rows)

        self.assertEqual(summary["observed_target_nfes"], [4, 8, 12])
        self.assertEqual(summary["nfe_sequence_pair_count"], 2)
        self.assertEqual(summary["sequence_multi_nfe_group_count"], 1)
        self.assertEqual(summary["physical_multi_nfe_group_count"], 1)

    def test_nfe_sequence_pairs_and_smoothness_loss_are_ordered_by_budget(self) -> None:
        rows: list[dict] = []
        for nfe in (4, 8, 12):
            rows.extend(
                [
                    _row(schedule="uniform", context_idx=0, target_nfe=nfe, series_id="series_0"),
                    _row(schedule="ays", context_idx=0, target_nfe=nfe, series_id="series_0"),
                ]
            )

        self.assertEqual(student_nfe_sequence_pair_indices(rows), [(0, 1), (1, 2)])
        self.assertEqual(student_nfe_sequence_pairs(rows), [(0, 1, 4.0), (1, 2, 4.0)])
        sparse_rows: list[dict] = []
        for nfe in (4, 12):
            sparse_rows.extend(
                [
                    _row(schedule="uniform", context_idx=0, target_nfe=nfe, series_id="series_0"),
                    _row(schedule="ays", context_idx=0, target_nfe=nfe, series_id="series_0"),
                ]
            )
        self.assertEqual(student_nfe_sequence_pairs(sparse_rows), [(0, 1, 8.0)])

    def test_nfe_sequence_pairs_use_logical_seed_not_sampling_seed(self) -> None:
        rows: list[dict] = []
        for target_nfe, evaluation_seed in ((4, 0), (8, 1000), (12, 2000)):
            uniform = _row(schedule="uniform", seed=evaluation_seed, context_idx=0, target_nfe=target_nfe, series_id="series_0")
            uniform["logical_seed"] = 0
            uniform["evaluation_seed"] = evaluation_seed
            uniform["parent_row_signature"] = f"solar_energy_10m|train_tuning|0|{target_nfe}|euler|uniform|ckpt"
            rows.append(uniform)

        self.assertEqual(logical_seed_from_row(rows[0]), 0)
        self.assertEqual(rows[1]["seed"], 1000)
        self.assertEqual(student_nfe_sequence_pair_indices(rows), [(0, 1), (1, 2)])

        parent_only_rows: list[dict] = []
        for target_nfe, evaluation_seed in ((4, 0), (8, 1000), (12, 2000)):
            row = _row(schedule="uniform", seed=evaluation_seed, context_idx=1, target_nfe=target_nfe, series_id="series_1")
            row["parent_row_signature"] = f"solar_energy_10m|train_tuning|2|{target_nfe}|euler|uniform|ckpt"
            parent_only_rows.append(row)
        self.assertEqual([logical_seed_from_row(row) for row in parent_only_rows], [2, 2, 2])
        self.assertEqual(student_nfe_sequence_pairs(parent_only_rows), [(0, 1, 4.0), (1, 2, 4.0)])

    def test_density_query_student_conditions_without_cross_row_attention(self) -> None:
        torch.manual_seed(7)
        rows = [
            _row(schedule="uniform", context_idx=0, target_nfe=4, series_id="series_0"),
            _row(schedule="uniform", context_idx=0, target_nfe=8, series_id="series_0"),
        ]
        config = build_setting_encoder_config(
            SETTING_ENCODER_MODE_CONTINUOUS_V3,
            observed_target_nfes=(4, 8),
            nfe_reference=8,
            rope_frequencies=(1.0, 2.0),
        )
        sx = torch.stack([setting_features(row["solver_key"], row["target_nfe"], mode=SETTING_ENCODER_MODE_CONTINUOUS_V3, config=config) for row in rows])
        series_idx = torch.tensor([0, 0], dtype=torch.long)
        context = torch.tensor([[0.1, 0.2], [2.0, -1.0]], dtype=torch.float32)
        student = GIPODensityQueryStudentTransformer(
            setting_dim=int(sx.shape[1]),
            density_dim=4,
            context_dim=2,
            num_series=1,
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
            conditioning_style=CONDITIONING_STYLE_ADDITIVE_MLP_V1,
        )
        student.eval()

        with torch.no_grad():
            batch_first = student.logits(sx, series_idx, context, rows=rows)[0]
            single_first = student.logits(sx[:1], series_idx[:1], context[:1], rows=rows[:1])[0]
            changed_context = student.logits(sx[:1], series_idx[:1], context[1:2], rows=rows[:1])[0]

        self.assertTrue(torch.allclose(batch_first, single_first, atol=1e-6))
        self.assertFalse(torch.allclose(single_first, changed_context))
        self.assertEqual(student.architecture, ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1)
        self.assertEqual(student.model_config()["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
        self.assertEqual(student.model_config()["density_token_attention"], DENSITY_TOKEN_ATTENTION_ROPE_V1)

    def test_density_form_teacher_attends_density_bins_under_conditioning(self) -> None:
        torch.manual_seed(8)
        rows = [
            _row(schedule="uniform", context_idx=0, target_nfe=4, series_id="series_0"),
            _row(schedule="ays", context_idx=1, target_nfe=8, series_id="series_1"),
        ]
        teacher = GIPODensityFormTeacherTransformer(
            setting_dim=2,
            density_dim=4,
            context_dim=2,
            num_series=2,
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
            conditioning_style=CONDITIONING_STYLE_ADDITIVE_MLP_V1,
        )
        teacher.eval()
        setting = torch.tensor([[0.1, 0.2], [0.3, -0.1]], dtype=torch.float32)
        density = torch.tensor([[2.0, 0.0, -1.0, -2.0], [-2.0, -1.0, 0.0, 2.0]], dtype=torch.float32)
        series = torch.tensor([0, 1], dtype=torch.long)
        context = torch.tensor([[0.5, -0.5], [1.0, 1.0]], dtype=torch.float32)

        with torch.no_grad():
            score_batch = teacher(setting, density, series, context, rows=rows)
            score_single = teacher(setting[:1], density[:1], series[:1], context[:1], rows=rows[:1])[0]
            score_reversed_density = teacher(setting[:1], torch.flip(density[:1], dims=[1]), series[:1], context[:1], rows=rows[:1])[0]

        self.assertEqual(tuple(score_batch.shape), (2, 2))
        self.assertTrue(torch.allclose(score_batch[0], score_single, atol=1e-6))
        self.assertFalse(torch.allclose(score_single, score_reversed_density))
        self.assertEqual(teacher.architecture, ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1)
        self.assertEqual(teacher.model_config()["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
        self.assertEqual(teacher.model_config()["density_token_attention"], DENSITY_TOKEN_ATTENTION_ROPE_V1)
        self.assertEqual(teacher.model_config()["teacher_output"], TEACHER_OUTPUT_METRIC_VECTOR_V1)
        self.assertEqual(teacher.model_config()["teacher_metric_targets"], list(TEACHER_METRIC_TARGET_KEYS))
        self.assertEqual(teacher.model_config()["teacher_scalarization"], TEACHER_SCALARIZATION_WEIGHTED_AVERAGE_V1)
    def test_additive_conditioning_changes_student_logits_immediately(self) -> None:
        torch.manual_seed(18)
        rows = [
            _row(schedule="uniform", context_idx=0, target_nfe=4, series_id="series_0"),
            _row(schedule="uniform", context_idx=1, target_nfe=4, series_id="series_0"),
        ]
        config = build_setting_encoder_config(
            mode=SETTING_ENCODER_MODE_CONTINUOUS_V3,
            observed_target_nfes=(4, 8, 12),
            nfe_reference=12,
            rope_frequencies=(1.0, 2.0),
        )
        sx = torch.stack([setting_features(row["solver_key"], row["target_nfe"], mode=SETTING_ENCODER_MODE_CONTINUOUS_V3, config=config) for row in rows])
        series_idx = torch.tensor([0, 0], dtype=torch.long)
        context = torch.tensor([[0.1, 0.2], [4.0, -3.0]], dtype=torch.float32)
        student = GIPODensityQueryStudentTransformer(
            setting_dim=int(sx.shape[1]),
            density_dim=4,
            context_dim=2,
            num_series=1,
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
            conditioning_style=CONDITIONING_STYLE_ADDITIVE_MLP_V1,
        )
        student.eval()

        with torch.no_grad():
            batch_first = student.logits(sx, series_idx, context, rows=rows)[0]
            single_first = student.logits(sx[:1], series_idx[:1], context[:1], rows=rows[:1])[0]
            changed_context = student.logits(sx[:1], series_idx[:1], context[1:2], rows=rows[:1])[0]

        self.assertTrue(torch.allclose(batch_first, single_first, atol=1e-6))
        self.assertFalse(torch.allclose(single_first, changed_context))
        self.assertEqual(student.model_config()["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)

    def test_additive_conditioning_changes_teacher_score_immediately(self) -> None:
        torch.manual_seed(19)
        rows = [_row(schedule="uniform", context_idx=0, target_nfe=4, series_id="series_0")]
        teacher = GIPODensityFormTeacherTransformer(
            setting_dim=2,
            density_dim=4,
            context_dim=2,
            num_series=1,
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
            conditioning_style=CONDITIONING_STYLE_ADDITIVE_MLP_V1,
        )
        teacher.eval()
        setting = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        density = torch.tensor([[2.0, 0.0, -1.0, -2.0]], dtype=torch.float32)
        series = torch.tensor([0], dtype=torch.long)
        context = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        changed_context = torch.tensor([[3.0, -2.0]], dtype=torch.float32)

        with torch.no_grad():
            score = teacher(setting, density, series, context, rows=rows)[0]
            changed_score = teacher(setting, density, series, changed_context, rows=rows)[0]
            reversed_score = teacher(setting, torch.flip(density, dims=[1]), series, context, rows=rows)[0]

        self.assertFalse(torch.allclose(score, changed_score))
        self.assertFalse(torch.allclose(score, reversed_score))
        self.assertEqual(teacher.model_config()["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)

    def test_density_bin_rope_rotates_qk_by_bin_position(self) -> None:
        q = torch.zeros(1, 4, 1, 4)
        q[..., 0] = 1.0
        q[..., 2] = 1.0
        rotated = _apply_density_bin_rope(q)

        self.assertEqual(tuple(rotated.shape), tuple(q.shape))
        self.assertFalse(torch.allclose(rotated[:, 0], q[:, 0]))
        self.assertFalse(torch.allclose(rotated[:, -1], q[:, -1]))
        self.assertFalse(torch.allclose(rotated[:, 0], rotated[:, -1]))
        original_pair_norms = q[..., 0::2].square() + q[..., 1::2].square()
        rotated_pair_norms = rotated[..., 0::2].square() + rotated[..., 1::2].square()
        self.assertTrue(torch.allclose(rotated_pair_norms, original_pair_norms, atol=1e-6))

    def test_density_token_rope_changes_open_attention_branch(self) -> None:
        torch.manual_seed(9)
        student = GIPODensityQueryStudentTransformer(
            setting_dim=2,
            density_dim=4,
            context_dim=2,
            num_series=1,
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
            conditioning_style=CONDITIONING_STYLE_ADDITIVE_MLP_V1,
        )
        block = student.blocks[0]
        tokens = torch.randn(1, 4, 16)

        with torch.no_grad():
            with_rope = block(tokens)
            with mock.patch("genode.gipo.policy._apply_density_bin_rope", lambda x: x):
                without_rope = block(tokens)

        self.assertFalse(torch.allclose(with_rope, without_rope))

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
        student = GIPODensityQueryStudentTransformer(
            setting_dim=int(setting_features("euler", 4).numel()),
            density_dim=4,
            context_dim=2,
            num_series=1,
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
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
        self.assertAlmostEqual(sum(prediction["density_mass"]), 1.0, places=6)

    def test_rank_huber_teacher_trains_on_density_features(self) -> None:
        torch.manual_seed(1)
        rewarded, embeddings = _rewarded_context_rows()
        fit_rows = [row for row in rewarded if int(row["example_idx"]) < 2]
        holdout_rows = [row for row in rewarded if int(row["example_idx"]) == 2]
        reference = uniform_reference_grid(8)
        series_map = build_series_index_map(fit_rows)
        normalizer = _density_normalizer(fit_rows, reference)
        teacher = GIPODensityFormTeacherTransformer(
            setting_dim=int(setting_features("euler", 4).numel()),
            density_dim=8,
            context_dim=2,
            num_series=len(series_map),
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
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
            teacher_checkpoint_selection_mode=TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1,
            allowed_schedule_keys=("uniform", "ays"),
        )

        self.assertEqual(summary["teacher_objective"], "pairwise_rank_plus_huber_regression")
        self.assertEqual(summary["teacher_target"], "metric_vector")
        self.assertEqual(summary["teacher_metric_targets"], list(TEACHER_METRIC_TARGET_KEYS))
        self.assertEqual(summary["teacher_scalarization"], TEACHER_SCALARIZATION_WEIGHTED_AVERAGE_V1)
        self.assertEqual(summary["teacher_utility_weights"], {"crps": 0.5, "mase": 0.5})
        self.assertEqual(summary["teacher_density_feature"], "train_normalized_log_density")
        self.assertGreater(summary["teacher_pair_count"], 0)
        self.assertIn("teacher_rank_loss", summary["losses"][-1])
        self.assertIn("teacher_huber_loss", summary["losses"][-1])
        self.assertEqual(summary["teacher_checkpoint_selection"]["selection_metric"], "weighted_normalized_regret_v1_score")

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
        self.assertIn("metric_huber_loss_crps", diagnostics)
        self.assertIn("metric_huber_loss_mase", diagnostics)
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
        self.assertEqual(target_summary["teacher_output"], TEACHER_OUTPUT_METRIC_VECTOR_V1)
        self.assertEqual(target_summary["teacher_metric_targets"], list(TEACHER_METRIC_TARGET_KEYS))
        self.assertEqual(target_summary["teacher_scalarization"], TEACHER_SCALARIZATION_WEIGHTED_AVERAGE_V1)
        self.assertEqual(target_summary["teacher_utility_weights"], {"crps": 0.5, "mase": 0.5})
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

        student = GIPODensityQueryStudentTransformer(
            setting_dim=int(setting_features("euler", 4).numel()),
            density_dim=4,
            context_dim=2,
            num_series=len(series_map),
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
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
            student_weight_decay=0.0,
            final_retrain_mode=True,
        )
        self.assertEqual(student_summary["student_policy_type"], "continuous_density")
        self.assertEqual(student_summary["student_objective"], "teacher_weighted_density_mle_kl")
        self.assertEqual(student_summary["teacher_utility_weights"], {"crps": 0.5, "mase": 0.5})
        self.assertEqual(student_summary["student_weight_decay"], 0.0)
        self.assertIn("student_kl_ce_loss", student_summary["losses"][-1])

    def test_student_validation_checkpoint_selection_metadata(self) -> None:
        torch.manual_seed(5)
        reference = uniform_reference_grid(4)
        ser_schedule_key = "ser_ptg_local_defect_eta005"
        train_rows = [
            _row(schedule="uniform", context_idx=0, series_id="series_0"),
            _row(schedule=ser_schedule_key, context_idx=0, series_id="series_0"),
        ]
        validation_rows = [
            _row(schedule="uniform", context_idx=1, series_id="series_0"),
            _row(schedule=ser_schedule_key, context_idx=1, series_id="series_0"),
        ]
        schedule_grids = {
            (ser_schedule_key, "euler", 4): (0.0, 0.7, 0.8, 0.9, 1.0),
        }
        embeddings = {
            context_id_from_row(train_rows[0]): np.asarray([0.0, 0.0], dtype=np.float32),
            context_id_from_row(validation_rows[0]): np.asarray([1.0, 0.0], dtype=np.float32),
        }
        series_map = build_series_index_map(train_rows)
        normalizer = DensityFeatureNormalizer(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
        )
        teacher = _LastBinTeacher()
        student = GIPODensityQueryStudentTransformer(
            setting_dim=int(setting_features("euler", 4).numel()),
            density_dim=4,
            context_dim=2,
            num_series=len(series_map),
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
        )

        student_summary = train_gipo_student(
            student,
            teacher,
            train_rows,
            context_embeddings=embeddings,
            series_index_map=series_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference,
            density_normalizer=normalizer,
            steps=3,
            lr=1e-3,
            teacher_temperature=0.05,
            student_weight_decay=0.0,
            validation_rows=validation_rows,
            student_log_every=1,
            student_checkpoint_every=1,
            student_checkpoint_selection_mode=STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE_V1,
        )

        selection = student_summary["student_checkpoint_selection"]
        self.assertEqual(selection["selection_protocol"], STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE_V1)
        self.assertIn(selection["selected_step"], {1, 2, 3})
        self.assertEqual(len(selection["history"]), 3)
        self.assertFalse(selection["locked_test_used_for_selection"])
        self.assertEqual(student_summary["student_checkpoint_selection_mode"], STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE_V1)
        self.assertTrue(student_summary["student_validation_used_for_selection"])
        self.assertFalse(student_summary["locked_test_used_for_selection"])
        self.assertEqual(student_summary["student_log_every"], 1)
        self.assertEqual(student_summary["student_checkpoint_every"], 1)
        self.assertIn("student_loss_tail_slope", student_summary)
        self.assertEqual(
            student_summary["student_validation_target_summary"]["student_validation_context_count"],
            1,
        )

    def test_student_validation_checkpoint_selection_rejects_locked_rows(self) -> None:
        reference = uniform_reference_grid(4)
        rows = [
            _row(schedule="uniform", context_idx=0, series_id="series_0"),
            _row(schedule="ays", context_idx=0, series_id="series_0"),
        ]
        locked_validation = [
            _row(schedule="uniform", context_idx=1, series_id="series_0", split_phase="locked_test"),
            _row(schedule="ays", context_idx=1, series_id="series_0", split_phase="locked_test"),
        ]
        embeddings = {
            context_id_from_row(rows[0]): np.asarray([0.0, 0.0], dtype=np.float32),
            context_id_from_row(locked_validation[0]): np.asarray([1.0, 0.0], dtype=np.float32),
        }
        series_map = build_series_index_map(rows)
        normalizer = DensityFeatureNormalizer(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
        )
        student = GIPODensityQueryStudentTransformer(
            setting_dim=int(setting_features("euler", 4).numel()),
            density_dim=4,
            context_dim=2,
            num_series=len(series_map),
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
        )
        with self.assertRaisesRegex(ValueError, "locked_test"):
            train_gipo_student(
                student,
                _LastBinTeacher(),
                rows,
                context_embeddings=embeddings,
                series_index_map=series_map,
                schedule_grids={},
                reference_time_grid=reference,
                density_normalizer=normalizer,
                steps=1,
                validation_rows=locked_validation,
                student_checkpoint_selection_mode=STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE_V1,
            )

    def test_student_training_reports_nfe_sequence_pairs_without_smoothness_loss(self) -> None:
        torch.manual_seed(3)
        reference = uniform_reference_grid(4)
        ser_schedule_key = "ser_ptg_local_defect_eta005"
        rows: list[dict] = []
        schedule_grids: dict[tuple[str, str, int], tuple[float, ...]] = {}
        for target_nfe in (4, 8):
            rows.append(_row(schedule="uniform", context_idx=0, series_id="series_0", target_nfe=target_nfe))
            rows.append(_row(schedule=ser_schedule_key, context_idx=0, series_id="series_0", target_nfe=target_nfe))
            schedule_grids[(ser_schedule_key, "euler", target_nfe)] = tuple(np.linspace(0.0, 1.0, target_nfe + 1) ** 2)
        context_id = context_id_from_row(rows[0])
        embeddings = {context_id: np.asarray([0.0, 0.0], dtype=np.float32)}
        series_map = build_series_index_map(rows)
        normalizer = DensityFeatureNormalizer(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
        )
        teacher = _LastBinTeacher()
        student = GIPODensityQueryStudentTransformer(
            setting_dim=int(setting_features("euler", 4).numel()),
            density_dim=4,
            context_dim=2,
            num_series=len(series_map),
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
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
            final_retrain_mode=True,
        )

        self.assertEqual(student_summary["student_nfe_sequence_pair_count"], 1)
        self.assertNotIn("student_nfe_smoothness_loss", student_summary["losses"][-1])

    def test_student_training_can_apply_teacher_scored_pseudo_targets(self) -> None:
        torch.manual_seed(4)
        reference = uniform_reference_grid(4)
        ser_schedule_key = "ser_ptg_local_defect_eta005"
        rows = [
            _row(schedule="uniform", context_idx=0, series_id="series_0", target_nfe=4),
            _row(schedule=ser_schedule_key, context_idx=0, series_id="series_0", target_nfe=4),
        ]
        pseudo_rows = [
            _row(schedule="uniform", context_idx=0, series_id="series_0", target_nfe=6),
            _row(schedule=ser_schedule_key, context_idx=0, series_id="series_0", target_nfe=6),
        ]
        schedule_grids = {
            (ser_schedule_key, "euler", 4): tuple(np.linspace(0.0, 1.0, 5) ** 2),
            (ser_schedule_key, "euler", 6): tuple(np.linspace(0.0, 1.0, 7) ** 2),
        }
        context_id = context_id_from_row(rows[0])
        embeddings = {context_id: np.asarray([0.0, 0.0], dtype=np.float32)}
        series_map = build_series_index_map(rows)
        normalizer = DensityFeatureNormalizer(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
        )
        teacher = _LastBinTeacher()
        student = GIPODensityQueryStudentTransformer(
            setting_dim=int(setting_features("euler", 4).numel()),
            density_dim=4,
            context_dim=2,
            num_series=len(series_map),
            hidden_dim=16,
            hidden_layers=1,
            attention_heads=4,
            dropout=0.0,
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
            pseudo_rows=pseudo_rows,
            pseudo_context_embeddings=embeddings,
            pseudo_schedule_grids=schedule_grids,
            pseudo_target_weight=0.25,
            final_retrain_mode=True,
        )

        last_loss = student_summary["losses"][-1]
        self.assertTrue(student_summary["pseudo_distillation_used"])
        self.assertEqual(student_summary["pseudo_target_weight"], 0.25)
        self.assertGreater(last_loss["student_pseudo_kl_ce_loss"], 0.0)
        self.assertAlmostEqual(
            last_loss["student_pseudo_weighted_loss"],
            0.25 * last_loss["student_pseudo_kl_ce_loss"],
            places=5,
        )
        self.assertEqual(student_summary["student_pseudo_target_summary"]["pseudo_target_nfes"], [6])

    def test_margin_hard_soft_targets_copy_top_teacher_density(self) -> None:
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
            student_target_mode="margin_hard_soft",
            teacher_hard_margin=0.0,
        )
        ser_mass = np.asarray(density_mass_for_row(rows[1], schedule_grids=schedule_grids, reference_time_grid=reference))
        self.assertEqual(target_summary["student_target_mode"], "margin_hard_soft")
        self.assertEqual(target_summary["hard_target_count"], 1)
        self.assertAlmostEqual(target_summary["hard_target_fraction"], 1.0)
        self.assertTrue(np.allclose(target_mass.detach().cpu().numpy()[0], ser_mass.astype(np.float32), atol=1e-7))

        prediction_rows, oracle_summary = build_teacher_weighted_density_prediction_rows(
            teacher,
            rows,
            context_embeddings=embeddings,
            series_index_map=series_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference,
            density_normalizer=normalizer,
            student_target_mode="margin_hard_soft",
            teacher_hard_margin=0.0,
        )
        self.assertEqual(len(prediction_rows), 1)
        self.assertEqual(prediction_rows[0]["teacher_top_schedule_key"], ser_schedule_key)
        self.assertTrue(prediction_rows[0]["teacher_hard_target"])
        self.assertEqual(oracle_summary["hard_target_count"], 1)
        self.assertEqual(len(prediction_rows[0]["time_grid"]), 5)

    def test_teacher_metric_vector_scalarization_controls_oracle_top_schedule(self) -> None:
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
        teacher = _ScheduleMetricTeacher()

        crps_rows, crps_summary = build_teacher_weighted_density_prediction_rows(
            teacher,
            rows,
            context_embeddings=embeddings,
            series_index_map=series_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference,
            density_normalizer=normalizer,
            student_target_mode="margin_hard_soft",
            teacher_hard_margin=0.0,
            teacher_utility_weights={"crps": 1.0, "mase": 0.0},
        )
        mase_rows, mase_summary = build_teacher_weighted_density_prediction_rows(
            teacher,
            rows,
            context_embeddings=embeddings,
            series_index_map=series_map,
            schedule_grids=schedule_grids,
            reference_time_grid=reference,
            density_normalizer=normalizer,
            student_target_mode="margin_hard_soft",
            teacher_hard_margin=0.0,
            teacher_utility_weights={"crps": 0.0, "mase": 1.0},
        )

        self.assertEqual(crps_rows[0]["teacher_top_schedule_key"], ser_schedule_key)
        self.assertEqual(mase_rows[0]["teacher_top_schedule_key"], "uniform")
        self.assertNotEqual(crps_rows[0]["teacher_top_schedule_key"], mase_rows[0]["teacher_top_schedule_key"])
        self.assertEqual(crps_summary["teacher_utility_weights"], {"crps": 1.0, "mase": 0.0})
        self.assertEqual(mase_summary["teacher_utility_weights"], {"crps": 0.0, "mase": 1.0})
        metric_payload = json.loads(crps_rows[0]["teacher_metric_utilities_json"])
        self.assertIn(ser_schedule_key, metric_payload)
        self.assertEqual(sorted(metric_payload[ser_schedule_key]), sorted(TEACHER_METRIC_TARGET_KEYS))

    def test_scalar_teacher_outputs_are_rejected(self) -> None:
        reference = uniform_reference_grid(4)
        rows = [
            _row(schedule="uniform", context_idx=0, series_id="series_0"),
            _row(schedule="ays", context_idx=0, series_id="series_0"),
        ]
        context_id = context_id_from_row(rows[0])
        embeddings = {context_id: np.asarray([0.0, 0.0], dtype=np.float32)}
        series_map = build_series_index_map(rows)
        normalizer = DensityFeatureNormalizer(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "metric vector"):
            build_teacher_weighted_density_targets(
                _ScalarTeacher(),
                rows,
                context_embeddings=embeddings,
                series_index_map=series_map,
                schedule_grids={},
                reference_time_grid=reference,
                density_normalizer=normalizer,
            )

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

    def test_teacher_objective_hyperparameters_are_validated(self) -> None:
        self.assertEqual(
            validate_teacher_objective_hyperparameters(rank_temperature=0.5, regression_weight=0.25, pair_margin=0.0),
            (0.5, 0.25, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "teacher_rank_temperature"):
            validate_teacher_objective_hyperparameters(rank_temperature=0.0, regression_weight=0.25, pair_margin=0.0)
        with self.assertRaisesRegex(ValueError, "teacher_regression_weight"):
            validate_teacher_objective_hyperparameters(rank_temperature=0.5, regression_weight=-0.1, pair_margin=0.0)
        with self.assertRaisesRegex(ValueError, "teacher_pair_margin"):
            validate_teacher_objective_hyperparameters(rank_temperature=0.5, regression_weight=0.25, pair_margin=-1.0)

    def test_density_family_holdout_runs_after_reward_construction(self) -> None:
        rows = [
            _row(schedule="uniform", context_idx=0, crps=2.0, mase=2.0),
            _row(schedule="ays", context_idx=0, crps=1.0, mase=1.2),
        ]

        with self.assertRaisesRegex(ValueError, "after uniform-anchored reward construction"):
            split_rows_by_density_family_holdout(
                rows,
                holdout_schedule_keys=("ays",),
                support_schedule_keys=("uniform", "ays"),
            )

        rewarded = attach_uniform_gipo_rewards(rows, support_schedule_keys=("uniform", "ays"))
        fit_rows, holdout_rows, metadata = split_rows_by_density_family_holdout(
            rewarded,
            holdout_schedule_keys=("ays",),
            support_schedule_keys=("uniform", "ays"),
        )

        self.assertEqual([row["scheduler_key"] for row in fit_rows], ["uniform"])
        self.assertEqual([row["scheduler_key"] for row in holdout_rows], ["ays"])
        self.assertIn("gipo_reward_protocol", holdout_rows[0])
        self.assertIn("u_comp_uniform", holdout_rows[0])
        self.assertEqual(metadata["density_family_holdout_schedule_keys"], ["ays"])
        self.assertEqual(metadata["density_family_fit_row_count"], 1)
        self.assertEqual(metadata["density_family_holdout_row_count"], 1)

    def test_density_family_holdout_refuses_uniform_anchor(self) -> None:
        with self.assertRaisesRegex(ValueError, "uniform|reward anchor"):
            validate_density_family_holdout_schedule_keys(
                ("uniform",),
                support_schedule_keys=("uniform", "ays"),
            )

    def test_weighted_normalized_regret_uses_j_cd_cdn_and_unseen_diagnostics(self) -> None:
        from genode.gipo.train_gipo import _select_weighted_normalized_regret_step

        def _training(split_regrets: dict[str, dict[int, float]]) -> dict:
            steps = sorted({step for values in split_regrets.values() for step in values})
            return {
                "teacher_checkpoint_selection": {
                    "history": [
                        {
                            "step": step,
                            "selection_constraints_passed": True,
                            "diagnostics": {
                                split_name: {
                                    "validation_soft_regret": float(values[step]),
                                    "validation_top1_regret": float(values[step]),
                                    "best_candidate_agreement": 1.0,
                                    "spearman_rank_correlation": 1.0,
                                    "rank_loss": float(values[step]),
                                    "huber_loss": float(values[step]),
                                    "total_loss": float(values[step]),
                                    "complete_candidate_group_count": 1,
                                }
                                for split_name, values in split_regrets.items()
                                if step in values
                            },
                        }
                        for step in steps
                    ]
                }
            }

        selection = _select_weighted_normalized_regret_step(
            context_density_training=_training(
                {
                    "context_disjoint": {1: 10.0, 2: 0.0, 3: 0.0},
                    "density_family_holdout": {1: 10.0, 2: 0.0, 3: 0.0},
                }
            ),
            unseen_nfe_training=_training({"unseen_nfe_holdout": {1: 0.0, 2: 1000.0, 3: 0.0}}),
            component_weights={"context": 0.25, "density_family": 0.25, "unseen_nfe": 0.50},
        )

        self.assertEqual(selection["selection_protocol"], TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1)
        self.assertEqual(selection["selected_step"], 3)
        self.assertTrue(selection["uses_unseen_nfe_selection_diagnostics"])
        self.assertEqual(
            selection["selected_component_weights"],
            {"context_disjoint": 0.25, "density_family_holdout": 0.25, "unseen_nfe_holdout": 0.5},
        )
        self.assertEqual(
            selection["selected_normalized_regret_values"],
            {"context_disjoint": 0.0, "density_family_holdout": 0.0, "unseen_nfe_holdout": 0.0},
        )
        by_step = {int(entry["step"]): entry for entry in selection["history"] if entry.get("selection_constraints_passed")}
        self.assertAlmostEqual(by_step[1]["weighted_normalized_regret_v1_score"], 0.5)
        self.assertAlmostEqual(by_step[2]["weighted_normalized_regret_v1_score"], 0.5)
        self.assertAlmostEqual(by_step[3]["weighted_normalized_regret_v1_score"], 0.0)

    def test_teacher_oracle_support_rows_require_fixed_and_ser_support(self) -> None:
        from genode.gipo.report_teacher_oracle import _validate_teacher_oracle_support_schedule_keys

        ser_key = "ser_ptg_local_defect_eta005"
        self.assertEqual(
            _validate_teacher_oracle_support_schedule_keys(("uniform", ser_key)),
            ("uniform", ser_key),
        )
        with self.assertRaisesRegex(ValueError, "fixed \\+ SER"):
            _validate_teacher_oracle_support_schedule_keys(("uniform",))
        with self.assertRaisesRegex(ValueError, "fixed \\+ SER"):
            _validate_teacher_oracle_support_schedule_keys((ser_key,))

    def test_final_teacher_retrain_metadata_is_persisted(self) -> None:
        def _fake_teacher_training(*args, **kwargs) -> dict:
            del args
            split_names = tuple(dict(kwargs.get("diagnostic_splits", {}) or {}))
            history = [
                {
                    "step": step,
                    "selection_constraints_passed": True,
                    "diagnostics": {
                        split_name: {
                            "validation_soft_regret": regret,
                            "validation_top1_regret": regret,
                            "best_candidate_agreement": 1.0,
                            "spearman_rank_correlation": 1.0,
                            "rank_loss": regret,
                            "huber_loss": regret,
                            "total_loss": regret,
                            "complete_candidate_group_count": 1,
                        }
                        for split_name in split_names
                    },
                }
                for step, regret in ((1, 1.0), (2, 0.1))
            ]
            mode = str(kwargs.get("teacher_checkpoint_selection_mode", TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1))
            return {
                "teacher_objective": "pairwise_rank_plus_huber_regression",
                "teacher_target": "metric_vector",
                "teacher_metric_targets": list(TEACHER_METRIC_TARGET_KEYS),
                "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE_V1,
                "teacher_utility_weights": {"crps": 0.5, "mase": 0.5},
                "teacher_checkpoint_selection": {
                    "selection_protocol": mode,
                    "selection_mode": mode,
                    "selection_metric": "mean_context_series_total_loss",
                    "selected_step": int(kwargs.get("steps", 1)),
                    "uses_validation_labels": False,
                    "locked_test_used_for_selection": False,
                    "history": history,
                },
            }

        def _fake_student_training(*args, **kwargs) -> dict:
            del args
            final_retrain = bool(kwargs.get("final_retrain_mode", False))
            mode = str(kwargs.get("student_checkpoint_selection_mode", STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE_V1))
            steps = int(kwargs.get("steps", 1))
            selected_step = steps if final_retrain else 2
            return {
                "student_objective": "teacher_weighted_density_mle_kl",
                "student_checkpoint_selection_mode": mode,
                "student_checkpoint_selection": {
                    "selection_protocol": "gipo_student_final_retrain_v1" if final_retrain else mode,
                    "selection_mode": "final_retrain" if final_retrain else mode,
                    "selection_metric": "validation_ce_loss"
                    if not final_retrain
                    else "configured_selected_step",
                    "selected_step": selected_step,
                    "history": [
                        {"step": 1, "validation_ce_loss": 1.0, "train_ce_loss": 1.0},
                        {"step": 2, "validation_ce_loss": 0.5, "train_ce_loss": 0.4},
                    ]
                    if not final_retrain
                    else [],
                    "uses_validation_labels": False,
                    "locked_test_used_for_selection": False,
                },
                "student_final_retrain": {"enabled": False},
                "student_target_summary": {
                    "teacher_temperature_mode": "fixed",
                    "student_target_mode": "soft_mixture",
                },
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows: list[dict] = []
            unseen_rows: list[dict] = []
            embeddings: dict[str, np.ndarray] = {}
            unseen_embeddings: dict[str, np.ndarray] = {}
            for context_idx in range(6):
                series_id = f"series_{context_idx}"
                for target_nfe in (4, 8, 12):
                    for schedule, crps in (("uniform", 2.0), ("ays", 1.0 + 0.01 * context_idx)):
                        row = _row(
                            schedule=schedule,
                            context_idx=context_idx,
                            target_nfe=target_nfe,
                            crps=crps,
                            mase=crps,
                            series_id=series_id,
                        )
                        rows.append(row)
                        embeddings[context_id_from_row(row)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
                for target_nfe in (6, 10, 14, 16):
                    for schedule, crps in (("uniform", 2.1), ("ays", 1.1 + 0.01 * context_idx)):
                        row = _row(
                            schedule=schedule,
                            context_idx=context_idx,
                            target_nfe=target_nfe,
                            crps=crps,
                            mase=crps,
                            series_id=series_id,
                        )
                        unseen_rows.append(row)
                        unseen_embeddings[context_id_from_row(row)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
            rows_path = root / "rows.csv"
            embeddings_path = root / "embeddings.npz"
            unseen_rows_path = root / "unseen_rows.csv"
            unseen_embeddings_path = root / "unseen_embeddings.npz"
            out_dir = root / "policy"
            _write_rows_csv(rows_path, rows)
            save_context_embedding_table(embeddings_path, embeddings)
            _write_rows_csv(unseen_rows_path, unseen_rows)
            save_context_embedding_table(unseen_embeddings_path, unseen_embeddings)
            args = build_train_argparser().parse_args(
                [
                    "--rows_csv",
                    str(rows_path),
                    "--context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(out_dir),
                    "--support_schedule_keys",
                    "uniform,ays",
                    "--teacher_density_holdout_schedule_keys",
                    "ays",
                    "--teacher_unseen_selection_rows_csv",
                    str(unseen_rows_path),
                    "--teacher_unseen_selection_context_embeddings_npz",
                    str(unseen_embeddings_path),
                    "--context_sample_count",
                    "6",
                    "--context_holdout_fraction",
                    "0.25",
                    "--teacher_steps",
                    "2",
                    "--teacher_checkpoint_every",
                    "1",
                    "--student_steps",
                    "3",
                    "--student_selection_holdout_fraction",
                    "0.25",
                    "--student_log_every",
                    "1",
                    "--student_checkpoint_every",
                    "1",
                    "--transformer_hidden_dim",
                    "16",
                    "--transformer_layers",
                    "1",
                    "--transformer_heads",
                    "4",
                    "--transformer_dropout",
                    "0",
                    "--gipo_teacher_conditioning_style",
                    CONDITIONING_STYLE_ADALN_ZERO_V1,
                    "--gipo_student_conditioning_style",
                    CONDITIONING_STYLE_ADDITIVE_MLP_V1,
                    "--allow_noncanonical_conditioning",
                    "--student_weight_decay",
                    "0",
                    "--device",
                    "cpu",
                ]
            )

            with mock.patch("genode.gipo.train_gipo.train_gipo_teacher", side_effect=_fake_teacher_training) as teacher_mock:
                with mock.patch("genode.gipo.train_gipo.train_gipo_student", side_effect=_fake_student_training) as student_mock:
                    summary = train_gipo(args)

            teacher_payload = torch.load(out_dir / "gipo_teacher.pt", map_location="cpu")
            student_payload = torch.load(out_dir / "gipo_student.pt", map_location="cpu")

            self.assertEqual(teacher_mock.call_count, 3)
            self.assertEqual(student_mock.call_count, 2)
            self.assertTrue(summary["teacher_final_retrain"]["enabled"])
            self.assertTrue(summary["final_teacher_retrain"]["enabled"])
            self.assertTrue(summary["student_final_retrain"]["enabled"])
            self.assertEqual(summary["student_final_retrain"]["selected_step"], 2)
            self.assertEqual(
                summary["student_checkpoint_selection"]["selection_protocol"],
                STUDENT_CHECKPOINT_SELECTION_VALIDATION_CE_V1,
            )
            self.assertIn("student_selector_training", summary["student_training"])
            self.assertTrue(summary["student_training"]["student_validation_used_for_selection"])
            self.assertEqual(summary["teacher_final_retrain"]["selected_step"], 2)
            self.assertEqual(summary["teacher_conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
            self.assertEqual(summary["student_conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(
                summary["conditioning_pair"],
                f"teacher_{CONDITIONING_STYLE_ADALN_ZERO_V1}__student_{CONDITIONING_STYLE_ADDITIVE_MLP_V1}",
            )
            self.assertEqual(summary["teacher_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
            self.assertEqual(summary["student_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(
                summary["teacher_checkpoint_selection"]["selection_protocol"],
                TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1,
            )
            for payload in (teacher_payload, student_payload):
                self.assertTrue(payload["teacher_final_retrain"]["enabled"])
                self.assertTrue(payload["final_teacher_retrain"]["enabled"])
                self.assertTrue(payload["student_final_retrain"]["enabled"])
                self.assertEqual(payload["student_final_retrain"]["selected_step"], 2)
                self.assertTrue(payload["teacher_training"]["teacher_final_retrain"]["enabled"])
                self.assertTrue(payload["teacher_training"]["final_teacher_retrain"]["enabled"])
                self.assertEqual(payload["teacher_conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
                self.assertEqual(payload["student_conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
                self.assertEqual(
                    payload["conditioning_pair"],
                    f"teacher_{CONDITIONING_STYLE_ADALN_ZERO_V1}__student_{CONDITIONING_STYLE_ADDITIVE_MLP_V1}",
                )
                self.assertEqual(
                    payload["teacher_checkpoint_selection"]["selection_protocol"],
                    TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1,
                )
                self.assertEqual(
                    payload["teacher_training"]["teacher_checkpoint_selection"]["selection_protocol"],
                    TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1,
                )

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

    def test_gipo_trainer_refuses_locked_source_split_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            row = _row(schedule="uniform", split_phase="context_disjoint")
            row["source_split_phase"] = "locked_test"
            rows_path = root / "rows.csv"
            embeddings_path = root / "embeddings.npz"
            _write_rows_csv(rows_path, [row])
            save_context_embedding_table(
                embeddings_path,
                {context_id_from_row(row): np.asarray([0.0, 0.0], dtype=np.float32)},
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
                    "--dry_run",
                ]
            )

            with self.assertRaisesRegex(ValueError, "refuses locked_test rows"):
                train_gipo(args)

    def test_gipo_trainer_requires_canonical_attention_heads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = build_train_argparser().parse_args(
                [
                    "--rows_csv",
                    str(root / "missing_rows.csv"),
                    "--context_embeddings_npz",
                    str(root / "missing_embeddings.npz"),
                    "--out_dir",
                    str(root / "policy"),
                    "--transformer_heads",
                    "2",
                    "--dry_run",
                ]
            )

            with self.assertRaisesRegex(ValueError, "attention_heads=4"):
                train_gipo(args)

    def test_gipo_trainer_dry_run_persists_split_membership(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows: list[dict] = []
            unseen_rows: list[dict] = []
            embeddings: dict[str, np.ndarray] = {}
            for context_idx in range(8):
                series_id = f"series_{context_idx // 2}"
                for target_nfe in (4, 8, 12):
                    for schedule, crps in (("uniform", 2.0), ("ays", 1.0 + 0.01 * context_idx)):
                        row = _row(schedule=schedule, context_idx=context_idx, target_nfe=target_nfe, crps=crps, series_id=series_id)
                        rows.append(row)
                        embeddings[context_id_from_row(row)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
                for target_nfe in (6, 10, 14, 16):
                    for schedule, crps in (("uniform", 2.1), ("ays", 1.1 + 0.01 * context_idx)):
                        row = _row(schedule=schedule, context_idx=context_idx, target_nfe=target_nfe, crps=crps, series_id=series_id)
                        unseen_rows.append(row)
                        embeddings[context_id_from_row(row)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
            rows_path = root / "rows.csv"
            unseen_rows_path = root / "unseen_rows.csv"
            embeddings_path = root / "embeddings.npz"
            _write_rows_csv(rows_path, rows)
            _write_rows_csv(unseen_rows_path, unseen_rows)
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
                    "24",
                    "--teacher_unseen_selection_rows_csv",
                    str(unseen_rows_path),
                    "--gipo_conditioning_style",
                    CONDITIONING_STYLE_ADDITIVE_MLP_V1,
                    "--student_weight_decay",
                    "0",
                    "--dry_run",
                ]
            )

            summary = train_gipo(args)

            self.assertEqual(summary["status"], "dry_run")
            self.assertEqual(summary["student_target_mode"], "soft_mixture")
            self.assertEqual(summary["teacher_architecture"], ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1)
            self.assertEqual(summary["student_architecture"], ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1)
            self.assertEqual(summary["gipo_conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(summary["teacher_conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(summary["student_conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(summary["conditioning_pair"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(summary["teacher_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(summary["student_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(summary["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
            self.assertEqual(summary["teacher_model_config"]["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
            self.assertEqual(summary["student_model_config"]["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
            self.assertEqual(summary["student_weight_decay"], 0.0)
            self.assertEqual(summary["setting_feature_mode"], SETTING_ENCODER_MODE_CONTINUOUS_V3)
            self.assertEqual(summary["setting_encoder_mode"], SETTING_ENCODER_MODE_CONTINUOUS_V3)
            self.assertIn("fit_rows", summary["nfe_sequence_diagnostics"])
            membership = summary["split_membership"]
            self.assertTrue({"fit", "context_disjoint", "density_family_holdout"}.issubset(set(membership)))
            self.assertIn("density_family_holdout", membership)
            self.assertGreater(membership["fit"]["context_count"], 0)
            self.assertGreater(membership["context_disjoint"]["context_count"], 0)
            self.assertEqual(len(membership["fit"]["context_id_hash"]), 64)
            self.assertIn("context_ids", membership["context_disjoint"])

            adaln_without_opt_in = build_train_argparser().parse_args(
                [
                    "--rows_csv",
                    str(rows_path),
                    "--context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(root / "policy_adaln_rejected"),
                    "--support_schedule_keys",
                    "uniform,ays",
                    "--teacher_unseen_selection_rows_csv",
                    str(unseen_rows_path),
                    "--gipo_conditioning_style",
                    CONDITIONING_STYLE_ADALN_ZERO_V1,
                    "--dry_run",
                ]
            )
            with self.assertRaisesRegex(ValueError, "allow_noncanonical_conditioning"):
                train_gipo(adaln_without_opt_in)

            adaln_args = build_train_argparser().parse_args(
                [
                    "--rows_csv",
                    str(rows_path),
                    "--context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(root / "policy_adaln"),
                    "--support_schedule_keys",
                    "uniform,ays",
                    "--context_sample_count",
                    "24",
                    "--teacher_unseen_selection_rows_csv",
                    str(unseen_rows_path),
                    "--gipo_conditioning_style",
                    CONDITIONING_STYLE_ADALN_ZERO_V1,
                    "--allow_noncanonical_conditioning",
                    "--student_weight_decay",
                    "0",
                    "--dry_run",
                ]
            )
            adaln_summary = train_gipo(adaln_args)
            self.assertEqual(adaln_summary["status"], "dry_run")
            self.assertEqual(adaln_summary["gipo_conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
            self.assertEqual(adaln_summary["teacher_conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
            self.assertEqual(adaln_summary["student_conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
            self.assertEqual(adaln_summary["conditioning_pair"], CONDITIONING_STYLE_ADALN_ZERO_V1)
            self.assertTrue(adaln_summary["noncanonical_conditioning_allowed"])
            self.assertEqual(adaln_summary["teacher_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
            self.assertEqual(adaln_summary["student_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)

            role_adaln_without_opt_in = build_train_argparser().parse_args(
                [
                    "--rows_csv",
                    str(rows_path),
                    "--context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(root / "policy_role_adaln_rejected"),
                    "--support_schedule_keys",
                    "uniform,ays",
                    "--teacher_unseen_selection_rows_csv",
                    str(unseen_rows_path),
                    "--gipo_teacher_conditioning_style",
                    CONDITIONING_STYLE_ADALN_ZERO_V1,
                    "--dry_run",
                ]
            )
            with self.assertRaisesRegex(ValueError, "allow_noncanonical_conditioning"):
                train_gipo(role_adaln_without_opt_in)

            conflicting_args = build_train_argparser().parse_args(
                [
                    "--rows_csv",
                    str(rows_path),
                    "--context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(root / "policy_conflicting_conditioning"),
                    "--support_schedule_keys",
                    "uniform,ays",
                    "--teacher_unseen_selection_rows_csv",
                    str(unseen_rows_path),
                    "--gipo_conditioning_style",
                    CONDITIONING_STYLE_ADDITIVE_MLP_V1,
                    "--gipo_teacher_conditioning_style",
                    CONDITIONING_STYLE_ADALN_ZERO_V1,
                    "--allow_noncanonical_conditioning",
                    "--dry_run",
                ]
            )
            with self.assertRaisesRegex(ValueError, "same-style shortcut|conflicting role-specific"):
                train_gipo(conflicting_args)

            role_specific_args = build_train_argparser().parse_args(
                [
                    "--rows_csv",
                    str(rows_path),
                    "--context_embeddings_npz",
                    str(embeddings_path),
                    "--out_dir",
                    str(root / "policy_role_specific"),
                    "--support_schedule_keys",
                    "uniform,ays",
                    "--context_sample_count",
                    "24",
                    "--teacher_unseen_selection_rows_csv",
                    str(unseen_rows_path),
                    "--gipo_teacher_conditioning_style",
                    CONDITIONING_STYLE_ADALN_ZERO_V1,
                    "--gipo_student_conditioning_style",
                    CONDITIONING_STYLE_ADDITIVE_MLP_V1,
                    "--allow_noncanonical_conditioning",
                    "--dry_run",
                ]
            )
            role_specific_summary = train_gipo(role_specific_args)
            self.assertEqual(role_specific_summary["status"], "dry_run")
            self.assertEqual(role_specific_summary["teacher_conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
            self.assertEqual(role_specific_summary["student_conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(
                role_specific_summary["conditioning_pair"],
                f"teacher_{CONDITIONING_STYLE_ADALN_ZERO_V1}__student_{CONDITIONING_STYLE_ADDITIVE_MLP_V1}",
            )
            self.assertEqual(role_specific_summary["teacher_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
            self.assertEqual(role_specific_summary["student_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)

    def test_gipo_trainer_accepts_custom_utility_metric_columns_without_forecast_rewards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows: list[dict] = []
            unseen_rows: list[dict] = []
            embeddings: dict[str, np.ndarray] = {}
            for context_idx in range(8):
                series_id = f"series_{context_idx // 2}"
                for target_nfe in (4, 8, 12):
                    for schedule, utility in (("uniform", 0.0), ("ays", 0.25 + 0.01 * context_idx)):
                        row = _row(schedule=schedule, context_idx=context_idx, target_nfe=target_nfe, series_id=series_id)
                        row.pop("crps")
                        row.pop("mase")
                        row["u_custom_score"] = float(utility)
                        rows.append(row)
                        embeddings[context_id_from_row(row)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
                for target_nfe in (6, 10, 14, 16):
                    for schedule, utility in (("uniform", 0.0), ("ays", 0.20 + 0.01 * context_idx)):
                        row = _row(schedule=schedule, context_idx=context_idx, target_nfe=target_nfe, series_id=series_id)
                        row.pop("crps")
                        row.pop("mase")
                        row["u_custom_score"] = float(utility)
                        unseen_rows.append(row)
                        embeddings[context_id_from_row(row)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
            rows_path = root / "rows.csv"
            unseen_rows_path = root / "unseen_rows.csv"
            embeddings_path = root / "embeddings.npz"
            _write_rows_csv(rows_path, rows)
            _write_rows_csv(unseen_rows_path, unseen_rows)
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
                    "24",
                    "--teacher_unseen_selection_rows_csv",
                    str(unseen_rows_path),
                    "--teacher_metric_target_keys",
                    "u_custom_score",
                    "--teacher_utility_weights",
                    "u_custom_score=1.0",
                    "--dry_run",
                ]
            )

            summary = train_gipo(args)

            self.assertEqual(summary["status"], "dry_run")
            self.assertEqual(summary["teacher_metric_targets"], ["u_custom_score"])
            self.assertEqual(summary["teacher_utility_weights"], {"u_custom_score": 1.0})
            self.assertEqual(summary["teacher_model_config"]["teacher_metric_targets"], ["u_custom_score"])

    def test_gipo_trainer_dry_run_persists_continuous_encoder_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows: list[dict] = []
            unseen_rows: list[dict] = []
            embeddings: dict[str, np.ndarray] = {}
            for context_idx in range(8):
                for target_nfe in (4, 8, 12):
                    for schedule, crps in (("uniform", 2.0), ("ays", 1.0 + 0.01 * context_idx)):
                        row = _row(schedule=schedule, context_idx=context_idx, target_nfe=target_nfe, crps=crps, series_id=f"series_{context_idx // 2}")
                        rows.append(row)
                        embeddings[context_id_from_row(row)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
                for target_nfe in (6, 10, 14, 16):
                    for schedule, crps in (("uniform", 2.1), ("ays", 1.1 + 0.01 * context_idx)):
                        row = _row(schedule=schedule, context_idx=context_idx, target_nfe=target_nfe, crps=crps, series_id=f"series_{context_idx // 2}")
                        unseen_rows.append(row)
                        embeddings[context_id_from_row(row)] = np.asarray([float(context_idx), 1.0], dtype=np.float32)
            rows_path = root / "rows.csv"
            unseen_rows_path = root / "unseen_rows.csv"
            embeddings_path = root / "embeddings.npz"
            _write_rows_csv(rows_path, rows)
            _write_rows_csv(unseen_rows_path, unseen_rows)
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
                    "24",
                    "--teacher_unseen_selection_rows_csv",
                    str(unseen_rows_path),
                    "--dry_run",
                ]
            )

            summary = train_gipo(args)

            self.assertEqual(summary["status"], "dry_run")
            self.assertEqual(summary["teacher_architecture"], ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1)
            self.assertEqual(summary["student_architecture"], ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1)
            self.assertEqual(summary["setting_encoder_mode"], SETTING_ENCODER_MODE_CONTINUOUS_V3)
            self.assertEqual(summary["setting_encoder_config"]["observed_target_nfes"], [4, 8, 12])
            self.assertIn("rope_frequencies", summary["setting_encoder_config"])
            fit_diag = summary["nfe_sequence_diagnostics"]["fit_rows"]
            self.assertGreater(fit_diag["nfe_sequence_pair_count"], 0)
            self.assertGreater(fit_diag["physical_multi_nfe_group_count"], 0)

    def test_student_checkpoint_roundtrip_uses_persisted_setting_encoder_config(self) -> None:
        from genode.gipo.report_locked_test import _load_student_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = uniform_reference_grid(64)
            config = build_setting_encoder_config(
                SETTING_ENCODER_MODE_CONTINUOUS_V3,
                observed_target_nfes=(4, 8, 12),
                nfe_reference=16,
                rope_frequencies=(1.0, 2.0),
            )
            setting_dim = setting_feature_dim(SETTING_ENCODER_MODE_CONTINUOUS_V3, config=config)
            student = GIPODensityQueryStudentTransformer(
                setting_dim=setting_dim,
                density_dim=64,
                context_dim=2,
                num_series=1,
                hidden_dim=16,
                hidden_layers=1,
                attention_heads=4,
                dropout=0.0,
            )
            checkpoint_path = root / "gipo_student.pt"
            torch.save(
                {
                    "protocol": GIPO_PROTOCOL,
                    "model_payload_version": MODEL_PAYLOAD_VERSION,
                    "student_policy_type": "continuous_density",
                    "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1,
                    "student_model_config": student.model_config(),
                    "student_objective": "teacher_weighted_density_mle_kl",
                    "student_state": student.state_dict(),
                    "setting_dim": setting_dim,
                    "setting_feature_mode": SETTING_ENCODER_MODE_CONTINUOUS_V3,
                    "setting_encoder_config": config.to_payload(),
                    "density_dim": 64,
                    "context_dim": 2,
                    "series_index_map": {"series_0": 0},
                    "embedding_normalizer": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    "density_representation": density_metadata(reference),
                    "teacher_training": _teacher_training_payload(),
                    "teacher_utility_weights": {"crps": 0.5, "mase": 0.5},
                    "locked_test_used_for_selection": False,
                },
                checkpoint_path,
            )

            loaded_student, _, _, _, payload = _load_student_checkpoint(checkpoint_path)

            self.assertEqual(loaded_student.setting_dim, setting_dim)
            self.assertEqual(payload["student_architecture"], ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1)
            self.assertEqual(payload["setting_encoder_mode"], SETTING_ENCODER_MODE_CONTINUOUS_V3)
            self.assertEqual(payload["setting_encoder_config"]["rope_frequencies"], [1.0, 2.0])
            self.assertEqual(payload["student_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(payload["student_model_config"]["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
            self.assertEqual(payload["student_model_config"]["density_token_attention"], DENSITY_TOKEN_ATTENTION_ROPE_V1)

            bad_path = root / "bad_gipo_student.pt"
            bad_payload = torch.load(checkpoint_path, map_location="cpu")
            bad_payload["setting_dim"] = int(setting_dim) + 1
            torch.save(bad_payload, bad_path)
            with self.assertRaisesRegex(ValueError, "setting_dim"):
                _load_student_checkpoint(bad_path)

            noncanonical_bins_path = root / "noncanonical_bins_gipo_student.pt"
            noncanonical_bins_payload = torch.load(checkpoint_path, map_location="cpu")
            noncanonical_bins_payload["density_dim"] = 4
            noncanonical_bins_payload["density_representation"] = density_metadata(uniform_reference_grid(4))
            torch.save(noncanonical_bins_payload, noncanonical_bins_path)
            with self.assertRaisesRegex(ValueError, "64 density bins"):
                _load_student_checkpoint(noncanonical_bins_path)

            additive_path = root / "additive_gipo_student.pt"
            additive_payload = torch.load(checkpoint_path, map_location="cpu")
            additive_student = GIPODensityQueryStudentTransformer(
                setting_dim=setting_dim,
                density_dim=64,
                context_dim=2,
                num_series=1,
                hidden_dim=16,
                hidden_layers=1,
                attention_heads=4,
                dropout=0.0,
                conditioning_style=CONDITIONING_STYLE_ADDITIVE_MLP_V1,
            )
            additive_payload["student_model_config"] = additive_student.model_config()
            additive_payload["student_state"] = additive_student.state_dict()
            torch.save(additive_payload, additive_path)
            _, _, _, _, additive_loaded_payload = _load_student_checkpoint(additive_path)
            self.assertEqual(additive_loaded_payload["student_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)

            legacy_style_path = root / "legacy_gipo_student.pt"
            legacy_style_payload = torch.load(checkpoint_path, map_location="cpu")
            legacy_style_payload["student_model_config"]["conditioning_style"] = "ada" + "ln_zero_v1"
            torch.save(legacy_style_payload, legacy_style_path)
            with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp_v1"):
                _load_student_checkpoint(legacy_style_path)

            adaln_path = root / "adaln_gipo_student.pt"
            adaln_payload = torch.load(checkpoint_path, map_location="cpu")
            adaln_student = GIPODensityQueryStudentTransformer(
                setting_dim=setting_dim,
                density_dim=64,
                context_dim=2,
                num_series=1,
                hidden_dim=16,
                hidden_layers=1,
                attention_heads=4,
                dropout=0.0,
                conditioning_style=CONDITIONING_STYLE_ADALN_ZERO_V1,
                allow_noncanonical_conditioning=True,
            )
            adaln_payload["student_model_config"] = adaln_student.model_config()
            adaln_payload["student_state"] = adaln_student.state_dict()
            torch.save(adaln_payload, adaln_path)
            with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp_v1"):
                _load_student_checkpoint(adaln_path)
            _, _, _, _, adaln_loaded_payload = _load_student_checkpoint(
                adaln_path,
                allow_noncanonical_conditioning=True,
            )
            self.assertEqual(adaln_loaded_payload["student_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)

            unknown_style_path = root / "unknown_style_gipo_student.pt"
            unknown_style_payload = torch.load(checkpoint_path, map_location="cpu")
            unknown_style_payload["student_model_config"]["conditioning_style"] = "dit_" + "additive_v1"
            torch.save(unknown_style_payload, unknown_style_path)
            with self.assertRaisesRegex(ValueError, "conditioning_style"):
                _load_student_checkpoint(unknown_style_path)

            additive_attention_path = root / "additive_attention_gipo_student.pt"
            additive_attention_payload = torch.load(checkpoint_path, map_location="cpu")
            additive_attention_payload["student_model_config"]["density_token_attention"] = "bin_" + "self_attention_v1"
            torch.save(additive_attention_payload, additive_attention_path)
            with self.assertRaisesRegex(ValueError, "bin_self_attention_rope_v1"):
                _load_student_checkpoint(additive_attention_path)

            missing_style_path = root / "missing_style_gipo_student.pt"
            missing_style_payload = torch.load(checkpoint_path, map_location="cpu")
            missing_style_payload["student_model_config"].pop("conditioning_style")
            torch.save(missing_style_payload, missing_style_path)
            with self.assertRaisesRegex(ValueError, "conditioning_style"):
                _load_student_checkpoint(missing_style_path)

            missing_attention_path = root / "missing_attention_gipo_student.pt"
            missing_attention_payload = torch.load(checkpoint_path, map_location="cpu")
            missing_attention_payload["student_model_config"].pop("density_token_attention")
            torch.save(missing_attention_payload, missing_attention_path)
            with self.assertRaisesRegex(ValueError, "density_token_attention"):
                _load_student_checkpoint(missing_attention_path)

            bad_heads_path = root / "bad_heads_gipo_student.pt"
            bad_heads_payload = torch.load(checkpoint_path, map_location="cpu")
            bad_heads_payload["student_model_config"]["attention_heads"] = 2
            torch.save(bad_heads_payload, bad_heads_path)
            with self.assertRaisesRegex(ValueError, "attention_heads=4"):
                _load_student_checkpoint(bad_heads_path)

            legacy_selection_path = root / "legacy_selection_gipo_student.pt"
            legacy_selection_payload = torch.load(checkpoint_path, map_location="cpu")
            legacy_selection_payload["teacher_training"]["teacher_checkpoint_selection"][
                "selection_protocol"
            ] = "diagnostic_" + "loss_v1"
            torch.save(legacy_selection_payload, legacy_selection_path)
            with self.assertRaisesRegex(ValueError, "weighted_normalized_regret_v1"):
                _load_student_checkpoint(legacy_selection_path)

    def test_locked_report_conditioning_metadata_preserves_student_compatibility_field(self) -> None:
        from genode.gipo.report_locked_test import _conditioning_metadata_for_summary

        metadata = _conditioning_metadata_for_summary(
            {
                "student_model_config": {"conditioning_style": CONDITIONING_STYLE_ADALN_ZERO_V1},
                "student_conditioning_style": CONDITIONING_STYLE_ADALN_ZERO_V1,
                "conditioning_pair": "teacher_additive_mlp_v1__student_adaln_zero_v1",
            },
            {
                "teacher_model_config": {"conditioning_style": CONDITIONING_STYLE_ADDITIVE_MLP_V1},
                "teacher_conditioning_style": CONDITIONING_STYLE_ADDITIVE_MLP_V1,
            },
        )

        self.assertEqual(metadata["conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
        self.assertEqual(metadata["student_conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)
        self.assertEqual(metadata["teacher_conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
        self.assertEqual(metadata["conditioning_pair"], "teacher_additive_mlp_v1__student_adaln_zero_v1")

    def test_teacher_student_conditioning_collector_validates_four_local_runs(self) -> None:
        import importlib.util

        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "verification_gipo_locked_multiaxis_b64_teacher_student_conditioning_ab_20260610"
            / "collect_teacher_student_conditioning_ab_summary.py"
        )
        spec = importlib.util.spec_from_file_location("collect_teacher_student_conditioning_ab_summary", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector)

        def write_json(path: Path, payload: dict[str, object]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        def training_payload(teacher_style: str, student_style: str, pair: str, noncanonical: bool) -> dict[str, object]:
            return {
                "status": "completed",
                "teacher_conditioning_style": teacher_style,
                "student_conditioning_style": student_style,
                "conditioning_pair": pair,
                "noncanonical_conditioning_allowed": noncanonical,
                "density_representation": {"reference_bin_count": 64},
                "locked_test_used_for_selection": False,
                "teacher_checkpoint_selection_mode": "weighted_normalized_regret_v1",
                "teacher_checkpoint_selection": {
                    "selection_protocol": "weighted_normalized_regret_v1",
                    "selected_step": 1,
                    "selected_weighted_normalized_regret_v1_score": 0.1,
                    "uses_unseen_nfe_selection_diagnostics": True,
                    "uses_validation_labels": False,
                    "locked_test_used_for_selection": False,
                    "selected_normalized_regret_values": {
                        "context_disjoint": 0.1,
                        "density_family_holdout": 0.2,
                        "unseen_nfe_holdout": 0.3,
                    },
                    "selected_component_weights": {
                        "context_disjoint": 0.25,
                        "density_family_holdout": 0.25,
                        "unseen_nfe_holdout": 0.50,
                    },
                },
                "teacher_final_retrain": {
                    "enabled": True,
                    "unseen_selection_diagnostics_used": True,
                    "locked_test_used_for_selection": False,
                },
                "unseen_nfe_selection": {
                    "enabled": True,
                    "target_nfes": [6, 10, 14, 16],
                    "raw_csv": "root/artifacts/unseen_train_supervision_rows.csv",
                    "used_for_final_fitting": False,
                    "locked_test_used_for_selection": False,
                },
                "student_checkpoint_selection_mode": "validation_ce_v1",
                "student_checkpoint_selection": {
                    "selection_protocol": "validation_ce_v1",
                    "selection_metric": "validation_ce_loss",
                    "selected_step": 1,
                    "locked_test_used_for_selection": False,
                },
                "student_training": {
                    "student_validation_used_for_selection": True,
                    "locked_test_used_for_selection": False,
                    "pseudo_distillation_used": False,
                    "pseudo_target_weight": 0.0,
                },
                "student_final_retrain": {
                    "enabled": True,
                    "performed": True,
                    "locked_test_used_for_selection": False,
                },
                "pseudo_distillation": {
                    "pseudo_distillation_requested": False,
                    "pseudo_target_weight": 0.0,
                },
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mixed_root = root / "mixed"
            same_root = root / "same"
            for label, run_spec in collector.RUN_SPECS.items():
                run_root = mixed_root if run_spec["root_key"] == "mixed" else same_root
                run_id = str(run_spec["run_id"])
                teacher_style = str(run_spec["teacher"])
                student_style = str(run_spec["student"])
                pair = collector._conditioning_pair(teacher_style, student_style)
                write_json(
                    run_root / "policy_runs" / run_id / "gipo_training_summary.json",
                    training_payload(teacher_style, student_style, pair, bool(run_spec["noncanonical_allowed"])),
                )
                write_json(
                    run_root / "policy_runs" / run_id / "final_retrain_metadata.json",
                    {
                        "selection_mode": "weighted_normalized_regret_v1",
                        "selection_policy": "weighted_normalized_regret_v1",
                        "selection": {"locked_test_used_for_selection": False},
                        "final_retrain": {"locked_test_used_for_selection": False},
                        "script_contract": {
                            "density_bin_count": 64,
                            "student_selector_mode": "validation_ce_v1",
                            "locked_test_used_for_selection": False,
                            "teacher_conditioning_style": teacher_style,
                            "student_conditioning_style": student_style,
                            "conditioning_pair": pair,
                        },
                    },
                )
                for panel, nfes in collector.PANELS.items():
                    write_json(
                        run_root
                        / "locked_reports"
                        / panel
                        / "student"
                        / run_id
                        / "locked_test_gipo_policy_summary.json",
                        {
                            "status": "completed",
                            "conditioning_style": student_style,
                            "student_conditioning_style": student_style,
                            "teacher_conditioning_style": teacher_style,
                            "conditioning_pair": pair,
                            "density_representation": {"reference_bin_count": 64},
                            "selection_mode": "reporting",
                            "teacher_checkpoint_selection_mode": "weighted_normalized_regret_v1",
                            "locked_test_used_for_selection": False,
                            "missing_cell_count": 0,
                            "target_nfe_values": nfes,
                            "mean_crps": 1.0 + 0.01 * len(label),
                            "mean_mase": 2.0 + 0.01 * len(panel),
                        },
                    )

            payload = collector.collect(mixed_root, same_root)

        self.assertTrue(payload["validation_passed"], payload["issues"])
        self.assertEqual(len(payload["four_way_rows"]), 8)
        self.assertEqual(len(payload["comparison_rows"]), 6)

    def test_nondefault_transformer_architecture_roundtrips_from_checkpoint_config(self) -> None:
        from genode.gipo.report_locked_test import _load_student_checkpoint
        from genode.gipo.report_teacher_oracle import _load_teacher_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = uniform_reference_grid(64)
            setting_dim = int(setting_features("euler", 4).numel())
            student = build_gipo_student_model(
                architecture=ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1,
                setting_dim=setting_dim,
                density_dim=64,
                context_dim=2,
                num_series=1,
                model_config={"hidden_dim": 24, "hidden_layers": 1, "attention_heads": 4, "dropout": 0.0},
            )
            teacher = build_gipo_teacher_model(
                architecture=ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1,
                setting_dim=setting_dim,
                density_dim=64,
                context_dim=2,
                num_series=1,
                model_config={"hidden_dim": 28, "hidden_layers": 1, "attention_heads": 4, "dropout": 0.0},
            )
            base_payload = {
                "protocol": GIPO_PROTOCOL,
                "model_payload_version": MODEL_PAYLOAD_VERSION,
                "setting_dim": setting_dim,
                "density_dim": 64,
                "context_dim": 2,
                "series_index_map": {"series_0": 0},
                "embedding_normalizer": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                "density_representation": density_metadata(reference),
                "teacher_training": _teacher_training_payload(),
                "teacher_utility_weights": {"crps": 0.5, "mase": 0.5},
                "locked_test_used_for_selection": False,
            }
            student_path = root / "student.pt"
            teacher_path = root / "teacher.pt"
            torch.save(
                {
                    **base_payload,
                    "student_policy_type": "continuous_density",
                    "student_objective": "teacher_weighted_density_mle_kl",
                    "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1,
                    "student_model_config": student.model_config(),
                    "student_state": student.state_dict(),
                },
                student_path,
            )
            torch.save(
                {
                    **base_payload,
                    "teacher_architecture": ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1,
                    "teacher_model_config": teacher.model_config(),
                    "teacher_state": teacher.state_dict(),
                    "density_feature_normalizer": DensityFeatureNormalizer(
                        mean=np.zeros(64, dtype=np.float32),
                        std=np.ones(64, dtype=np.float32),
                    ).to_payload(),
                },
                teacher_path,
            )

            loaded_student, _, _, _, student_payload = _load_student_checkpoint(student_path)
            loaded_teacher, _, _, _, _, teacher_payload = _load_teacher_checkpoint(teacher_path)

            self.assertEqual(loaded_student.hidden_dim, 24)
            self.assertEqual(loaded_student.attention_heads, 4)
            self.assertEqual(student_payload["student_model_config"]["hidden_dim"], 24)
            self.assertEqual(student_payload["student_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(student_payload["student_model_config"]["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
            self.assertEqual(student_payload["student_model_config"]["density_token_attention"], DENSITY_TOKEN_ATTENTION_ROPE_V1)
            self.assertEqual(loaded_teacher.hidden_dim, 28)
            self.assertEqual(loaded_teacher.attention_heads, 4)
            self.assertEqual(teacher_payload["teacher_model_config"]["hidden_dim"], 28)
            self.assertEqual(teacher_payload["teacher_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(teacher_payload["teacher_model_config"]["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
            self.assertEqual(teacher_payload["teacher_model_config"]["density_token_attention"], DENSITY_TOKEN_ATTENTION_ROPE_V1)
            self.assertEqual(teacher_payload["teacher_model_config"]["teacher_output"], TEACHER_OUTPUT_METRIC_VECTOR_V1)
            self.assertEqual(teacher_payload["teacher_model_config"]["teacher_metric_targets"], list(TEACHER_METRIC_TARGET_KEYS))
            self.assertEqual(teacher_payload["teacher_model_config"]["teacher_scalarization"], TEACHER_SCALARIZATION_WEIGHTED_AVERAGE_V1)

            bad_student = torch.load(student_path, map_location="cpu")
            bad_student["student_architecture"] = "mlp" + "_v1"
            bad_student_path = root / "bad_student.pt"
            torch.save(bad_student, bad_student_path)
            with self.assertRaisesRegex(ValueError, "density_query_transformer_v1"):
                _load_student_checkpoint(bad_student_path)

    def test_teacher_checkpoint_roundtrip_validates_setting_encoder_dim(self) -> None:
        from genode.gipo.report_teacher_oracle import _load_teacher_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = uniform_reference_grid(64)
            config = build_setting_encoder_config(
                SETTING_ENCODER_MODE_CONTINUOUS_V3,
                observed_target_nfes=(4, 8, 12),
                nfe_reference=16,
                rope_frequencies=(1.0, 2.0),
            )
            setting_dim = setting_feature_dim(SETTING_ENCODER_MODE_CONTINUOUS_V3, config=config)
            teacher = GIPODensityFormTeacherTransformer(
                setting_dim=setting_dim,
                density_dim=64,
                context_dim=2,
                num_series=1,
                hidden_dim=16,
                hidden_layers=1,
                attention_heads=4,
                dropout=0.0,
            )
            checkpoint_path = root / "gipo_teacher.pt"
            torch.save(
                {
                    "protocol": GIPO_PROTOCOL,
                    "model_payload_version": MODEL_PAYLOAD_VERSION,
                    "teacher_architecture": ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1,
                    "teacher_model_config": teacher.model_config(),
                    "teacher_state": teacher.state_dict(),
                    "setting_dim": setting_dim,
                    "setting_feature_mode": SETTING_ENCODER_MODE_CONTINUOUS_V3,
                    "setting_encoder_config": config.to_payload(),
                    "density_dim": 64,
                    "context_dim": 2,
                    "series_index_map": {"series_0": 0},
                    "embedding_normalizer": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    "density_feature_normalizer": DensityFeatureNormalizer(
                        mean=np.zeros(64, dtype=np.float32),
                        std=np.ones(64, dtype=np.float32),
                    ).to_payload(),
                    "density_representation": density_metadata(reference),
                    "teacher_training": _teacher_training_payload(),
                    "teacher_utility_weights": {"crps": 0.5, "mase": 0.5},
                    "locked_test_used_for_selection": False,
                },
                checkpoint_path,
            )
            loaded_teacher, _, _, _, _, payload = _load_teacher_checkpoint(checkpoint_path)
            self.assertEqual(loaded_teacher.setting_dim, setting_dim)
            self.assertEqual(payload["teacher_architecture"], ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1)
            self.assertEqual(payload["setting_encoder_mode"], SETTING_ENCODER_MODE_CONTINUOUS_V3)
            self.assertEqual(payload["teacher_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
            self.assertEqual(payload["teacher_model_config"]["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
            self.assertEqual(payload["teacher_model_config"]["density_token_attention"], DENSITY_TOKEN_ATTENTION_ROPE_V1)
            self.assertEqual(payload["teacher_model_config"]["teacher_output"], TEACHER_OUTPUT_METRIC_VECTOR_V1)
            self.assertEqual(payload["teacher_model_config"]["teacher_metric_targets"], list(TEACHER_METRIC_TARGET_KEYS))
            self.assertEqual(payload["teacher_model_config"]["teacher_scalarization"], TEACHER_SCALARIZATION_WEIGHTED_AVERAGE_V1)

            bad_path = root / "bad_gipo_teacher.pt"
            bad_payload = torch.load(checkpoint_path, map_location="cpu")
            bad_payload["setting_dim"] = int(setting_dim) + 1
            torch.save(bad_payload, bad_path)
            with self.assertRaisesRegex(ValueError, "setting_dim"):
                _load_teacher_checkpoint(bad_path)

            noncanonical_bins_path = root / "noncanonical_bins_gipo_teacher.pt"
            noncanonical_bins_payload = torch.load(checkpoint_path, map_location="cpu")
            noncanonical_bins_payload["density_dim"] = 4
            noncanonical_bins_payload["density_representation"] = density_metadata(uniform_reference_grid(4))
            torch.save(noncanonical_bins_payload, noncanonical_bins_path)
            with self.assertRaisesRegex(ValueError, "64 density bins"):
                _load_teacher_checkpoint(noncanonical_bins_path)

            additive_path = root / "additive_gipo_teacher.pt"
            additive_payload = torch.load(checkpoint_path, map_location="cpu")
            additive_teacher = GIPODensityFormTeacherTransformer(
                setting_dim=setting_dim,
                density_dim=64,
                context_dim=2,
                num_series=1,
                hidden_dim=16,
                hidden_layers=1,
                attention_heads=4,
                dropout=0.0,
                conditioning_style=CONDITIONING_STYLE_ADDITIVE_MLP_V1,
            )
            additive_payload["teacher_model_config"] = additive_teacher.model_config()
            additive_payload["teacher_state"] = additive_teacher.state_dict()
            torch.save(additive_payload, additive_path)
            _, _, _, _, _, additive_loaded_payload = _load_teacher_checkpoint(additive_path)
            self.assertEqual(additive_loaded_payload["teacher_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)

            legacy_style_path = root / "legacy_gipo_teacher.pt"
            legacy_style_payload = torch.load(checkpoint_path, map_location="cpu")
            legacy_style_payload["teacher_model_config"]["conditioning_style"] = "ada" + "ln_zero_v1"
            torch.save(legacy_style_payload, legacy_style_path)
            with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp_v1"):
                _load_teacher_checkpoint(legacy_style_path)

            adaln_path = root / "adaln_gipo_teacher.pt"
            adaln_payload = torch.load(checkpoint_path, map_location="cpu")
            adaln_teacher = GIPODensityFormTeacherTransformer(
                setting_dim=setting_dim,
                density_dim=64,
                context_dim=2,
                num_series=1,
                hidden_dim=16,
                hidden_layers=1,
                attention_heads=4,
                dropout=0.0,
                conditioning_style=CONDITIONING_STYLE_ADALN_ZERO_V1,
                allow_noncanonical_conditioning=True,
            )
            adaln_payload["teacher_model_config"] = adaln_teacher.model_config()
            adaln_payload["teacher_state"] = adaln_teacher.state_dict()
            torch.save(adaln_payload, adaln_path)
            with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp_v1"):
                _load_teacher_checkpoint(adaln_path)
            _, _, _, _, _, adaln_loaded_payload = _load_teacher_checkpoint(
                adaln_path,
                allow_noncanonical_conditioning=True,
            )
            self.assertEqual(adaln_loaded_payload["teacher_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADALN_ZERO_V1)

            unknown_style_path = root / "unknown_style_gipo_teacher.pt"
            unknown_style_payload = torch.load(checkpoint_path, map_location="cpu")
            unknown_style_payload["teacher_model_config"]["conditioning_style"] = "dit_" + "additive_v1"
            torch.save(unknown_style_payload, unknown_style_path)
            with self.assertRaisesRegex(ValueError, "conditioning_style"):
                _load_teacher_checkpoint(unknown_style_path)

            additive_attention_path = root / "additive_attention_gipo_teacher.pt"
            additive_attention_payload = torch.load(checkpoint_path, map_location="cpu")
            additive_attention_payload["teacher_model_config"]["density_token_attention"] = "bin_" + "self_attention_v1"
            torch.save(additive_attention_payload, additive_attention_path)
            with self.assertRaisesRegex(ValueError, "bin_self_attention_rope_v1"):
                _load_teacher_checkpoint(additive_attention_path)

            missing_style_path = root / "missing_style_gipo_teacher.pt"
            missing_style_payload = torch.load(checkpoint_path, map_location="cpu")
            missing_style_payload["teacher_model_config"].pop("conditioning_style")
            torch.save(missing_style_payload, missing_style_path)
            with self.assertRaisesRegex(ValueError, "conditioning_style"):
                _load_teacher_checkpoint(missing_style_path)

            missing_attention_path = root / "missing_attention_gipo_teacher.pt"
            missing_attention_payload = torch.load(checkpoint_path, map_location="cpu")
            missing_attention_payload["teacher_model_config"].pop("density_token_attention")
            torch.save(missing_attention_payload, missing_attention_path)
            with self.assertRaisesRegex(ValueError, "density_token_attention"):
                _load_teacher_checkpoint(missing_attention_path)

            bad_heads_path = root / "bad_heads_gipo_teacher.pt"
            bad_heads_payload = torch.load(checkpoint_path, map_location="cpu")
            bad_heads_payload["teacher_model_config"]["attention_heads"] = 2
            torch.save(bad_heads_payload, bad_heads_path)
            with self.assertRaisesRegex(ValueError, "attention_heads=4"):
                _load_teacher_checkpoint(bad_heads_path)

            scalar_output_path = root / "scalar_output_gipo_teacher.pt"
            scalar_output_payload = torch.load(checkpoint_path, map_location="cpu")
            scalar_output_payload["teacher_model_config"].pop("teacher_output")
            torch.save(scalar_output_payload, scalar_output_path)
            with self.assertRaisesRegex(ValueError, "teacher_output"):
                _load_teacher_checkpoint(scalar_output_path)

            legacy_selection_path = root / "legacy_selection_gipo_teacher.pt"
            legacy_selection_payload = torch.load(checkpoint_path, map_location="cpu")
            legacy_selection_payload["teacher_training"]["teacher_checkpoint_selection"][
                "selection_protocol"
            ] = "diagnostic_" + "loss_v1"
            torch.save(legacy_selection_payload, legacy_selection_path)
            with self.assertRaisesRegex(ValueError, "weighted_normalized_regret_v1"):
                _load_teacher_checkpoint(legacy_selection_path)

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
                    "protocol": "legacy_categorical_" + "gipo_" + "v1",
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
                        "protocol": "legacy_categorical_" + "gipo_" + "v1",
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
            reference = uniform_reference_grid(64)
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
            student = GIPODensityQueryStudentTransformer(
                setting_dim=int(setting_features("euler", 4).numel()),
                density_dim=64,
                context_dim=2,
                num_series=1,
                hidden_dim=16,
                hidden_layers=1,
                attention_heads=4,
                dropout=0.0,
            )
            torch.save(
                {
                    "protocol": GIPO_PROTOCOL,
                    "model_payload_version": MODEL_PAYLOAD_VERSION,
                    "student_policy_type": "continuous_density",
                    "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1,
                    "student_model_config": student.model_config(),
                    "student_objective": "teacher_weighted_density_mle_kl",
                    "student_state": student.state_dict(),
                    "setting_dim": int(setting_features("euler", 4).numel()),
                    "density_dim": 64,
                    "context_dim": 2,
                    "series_index_map": {"series_0": 0},
                    "embedding_normalizer": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    "density_representation": density_metadata(reference),
                    "teacher_training": _teacher_training_payload(),
                    "teacher_utility_weights": {"crps": 0.5, "mase": 0.5},
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

    def test_calibration_reporter_requires_requested_seed_solver_nfe_cells(self) -> None:
        from genode.gipo.report_locked_test import (
            build_argparser as build_locked_report_argparser,
            report_gipo_locked_test,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = uniform_reference_grid(64)
            row = _row(schedule="uniform", split_phase="validation_tuning", series_id="series_0")
            row["evaluation_seed"] = 0
            rows_path = root / "validation_rows.csv"
            embeddings_path = root / "embeddings.npz"
            checkpoint_path = root / "gipo_student.pt"
            summary_path = root / "summary.json"
            _write_rows_csv(rows_path, [row])
            save_context_embedding_table(
                embeddings_path,
                {context_id_from_row(row): np.asarray([0.0, 0.0], dtype=np.float32)},
            )
            student = GIPODensityQueryStudentTransformer(
                setting_dim=int(setting_features("euler", 4).numel()),
                density_dim=64,
                context_dim=2,
                num_series=1,
                hidden_dim=16,
                hidden_layers=1,
                attention_heads=4,
                dropout=0.0,
            )
            torch.save(
                {
                    "protocol": GIPO_PROTOCOL,
                    "model_payload_version": MODEL_PAYLOAD_VERSION,
                    "student_policy_type": "continuous_density",
                    "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1,
                    "student_model_config": student.model_config(),
                    "student_objective": "teacher_weighted_density_mle_kl",
                    "student_state": student.state_dict(),
                    "setting_dim": int(setting_features("euler", 4).numel()),
                    "density_dim": 64,
                    "context_dim": 2,
                    "series_index_map": {"series_0": 0},
                    "embedding_normalizer": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                    "density_representation": density_metadata(reference),
                    "teacher_training": _teacher_training_payload(),
                    "teacher_utility_weights": {"crps": 0.5, "mase": 0.5},
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
                    "validation_tuning",
                    "--selection_mode",
                    "calibration",
                    "--out_dir",
                    str(root / "report"),
                    "--seeds",
                    "0,1",
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                ]
            )

            with self.assertRaisesRegex(ValueError, "missing seed/solver/NFE cells"):
                report_gipo_locked_test(args)

    def test_student_selection_context_filter_uses_logical_parent_seed(self) -> None:
        from genode.gipo.report_locked_test import _filter_rows_to_contexts

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

        self.assertEqual(_filter_rows_to_contexts([baseline], [student]), [baseline])

    def test_gap_collector_marks_report_missing_cells_incomplete(self) -> None:
        from genode.gipo.verification_summary import nfe_gain_panel

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            comparison_path = root / "comparison.json"
            comparison_path.write_text(
                json.dumps(
                    {
                        "target_nfe_values": [6, 10, 14, 16],
                        "solver_names": ["euler"],
                        "missing_baseline_cells": [],
                        "missing_ser_ptg_cells": [],
                        "missing_student_cells": [],
                        "cell_rankings": [
                            {
                                "target_nfe": nfe,
                                "student_relative_mase_gain_vs_best_baseline": 0.1,
                                "student_relative_crps_gain_vs_best_baseline": 0.1,
                            }
                            for nfe in (6, 10, 14, 16)
                        ],
                    }
                ),
                encoding="utf-8",
            )

            panel = nfe_gain_panel(
                {
                    "comparison_summary_path": str(comparison_path),
                    "missing_expected_cells": [["solar_energy_10m", 1, "euler", 16]],
                }
            )

            self.assertFalse(panel["coverage_complete"])
            self.assertGreater(panel["missing_comparison_cell_count"], 0)

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

    def test_teacher_utility_weights_can_emphasize_mase(self) -> None:
        rows = [
            _row(schedule="uniform", context_idx=0, crps=2.0, mase=2.0),
            _row(schedule="ays", context_idx=0, crps=1.0, mase=2.0),
        ]
        crps_only = attach_uniform_gipo_rewards(
            rows,
            support_schedule_keys=("uniform", "ays"),
            utility_crps_weight=1.0,
            utility_mase_weight=0.0,
        )
        mase_only = attach_uniform_gipo_rewards(
            rows,
            support_schedule_keys=("uniform", "ays"),
            utility_crps_weight=0.0,
            utility_mase_weight=1.0,
        )
        crps_ays = next(row for row in crps_only if row["scheduler_key"] == "ays")
        mase_ays = next(row for row in mase_only if row["scheduler_key"] == "ays")
        self.assertGreater(float(crps_ays["u_comp_uniform"]), 0.0)
        self.assertAlmostEqual(float(mase_ays["u_comp_uniform"]), 0.0)
        self.assertEqual(float(mase_ays["u_comp_mase_weight"]), 1.0)


if __name__ == "__main__":
    unittest.main()
