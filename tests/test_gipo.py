from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from genode.gipo.density_representation import (
    DENSITY_PROTOCOL,
    density_mass_to_time_grid,
    density_metadata,
    grid_to_density_mass,
    uniform_reference_grid,
)
from genode.gipo.models import SETTING_ENCODER_MODE_CONTINUOUS_V3, build_setting_encoder_config, setting_feature_dim, setting_features
from genode.gipo.policy import (
    ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1,
    ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1,
    CONDITIONING_STYLE_ADDITIVE_MLP_V1,
    DENSITY_TOKEN_ATTENTION_ROPE_V1,
    GIPO_PROTOCOL,
    GIPODensityFormTeacherTransformer,
    GIPODensityQueryStudentTransformer,
    MODEL_PAYLOAD_VERSION,
    SERIES_CONDITIONING_NONE_CONTEXT_ONLY,
    TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1,
    TEACHER_METRIC_TARGET_KEYS,
    TEACHER_OUTPUT_METRIC_VECTOR_V1,
    TEACHER_SCALARIZATION_WEIGHTED_AVERAGE_V1,
    build_gipo_student_model,
    build_gipo_teacher_model,
    validate_gipo_conditioning_style,
)
from genode.gipo.train_gipo import build_argparser as build_train_argparser


def _teacher_training_payload() -> dict:
    return {
        "teacher_target": "metric_vector",
        "teacher_metric_targets": list(TEACHER_METRIC_TARGET_KEYS),
        "teacher_scalarization": TEACHER_SCALARIZATION_WEIGHTED_AVERAGE_V1,
        "teacher_checkpoint_selection": {
            "selection_protocol": TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1,
            "selected_step": 1,
            "uses_validation_labels": False,
            "locked_test_used_for_selection": False,
        },
    }


class GIPOCanonicalTests(unittest.TestCase):
    def test_density_grid_roundtrip_uses_canonical_64_bins(self) -> None:
        reference = uniform_reference_grid(64)
        source_grid = (0.0, 0.25, 0.5, 1.0)

        mass = grid_to_density_mass(source_grid, reference_grid=reference)
        reconstructed = density_mass_to_time_grid(mass, reference_grid=reference, macro_steps=3)
        metadata = density_metadata(reference)

        self.assertEqual(len(mass), 64)
        self.assertAlmostEqual(float(np.sum(mass)), 1.0, places=6)
        self.assertEqual(len(reconstructed), 4)
        self.assertEqual(metadata["density_protocol"], DENSITY_PROTOCOL)
        self.assertEqual(metadata["reference_bin_count"], 64)

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
        self.assertEqual(student.model_config()["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
        self.assertEqual(teacher.model_config()["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
        self.assertEqual(student.model_config()["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
        self.assertEqual(teacher.model_config()["series_conditioning"], SERIES_CONDITIONING_NONE_CONTEXT_ONLY)
        self.assertEqual(student.model_config()["density_token_attention"], DENSITY_TOKEN_ATTENTION_ROPE_V1)
        self.assertEqual(teacher.model_config()["density_token_attention"], DENSITY_TOKEN_ATTENTION_ROPE_V1)

    def test_non_additive_conditioning_style_is_rejected(self) -> None:
        legacy_style = "ada" + "ln_zero_v1"
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
                density_dim=64,
                context_dim=2,
                num_series=1,
                model_config={"conditioning_style": legacy_style},
            )
        with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp_v1"):
            build_gipo_teacher_model(
                architecture=ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1,
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
        self.assertNotIn("--gipo_conditioning_style", options)
        self.assertNotIn("--gipo_teacher_conditioning_style", options)
        self.assertNotIn("--gipo_student_conditioning_style", options)
        self.assertNotIn("--allow_noncanonical_conditioning", options)

    def test_student_checkpoint_loader_accepts_only_canonical_additive_student(self) -> None:
        from genode.gipo.report_locked_test import _load_student_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = build_setting_encoder_config(
                SETTING_ENCODER_MODE_CONTINUOUS_V3,
                observed_target_nfes=(4, 8, 12),
                nfe_reference=16,
                rope_frequencies=(1.0, 2.0),
            )
            setting_dim = setting_feature_dim(SETTING_ENCODER_MODE_CONTINUOUS_V3, config=config)
            student = build_gipo_student_model(
                architecture=ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1,
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
                "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER_V1,
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

            loaded_student, _, _, _, loaded_payload = _load_student_checkpoint(checkpoint_path)
            self.assertEqual(loaded_student.setting_dim, setting_dim)
            self.assertEqual(loaded_payload["student_model_config"]["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)

            legacy_path = root / "legacy_student.pt"
            legacy_payload = dict(payload)
            legacy_payload["student_model_config"] = dict(payload["student_model_config"])
            legacy_payload["student_model_config"]["conditioning_style"] = "ada" + "ln_zero_v1"
            torch.save(legacy_payload, legacy_path)
            with self.assertRaisesRegex(ValueError, "conditioning_style|additive_mlp_v1"):
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
                "student_model_config": {"conditioning_style": CONDITIONING_STYLE_ADDITIVE_MLP_V1},
                "student_conditioning_style": CONDITIONING_STYLE_ADDITIVE_MLP_V1,
            },
            {
                "teacher_model_config": {"conditioning_style": CONDITIONING_STYLE_ADDITIVE_MLP_V1},
                "teacher_conditioning_style": CONDITIONING_STYLE_ADDITIVE_MLP_V1,
            },
        )

        self.assertEqual(metadata["conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
        self.assertEqual(metadata["student_conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
        self.assertEqual(metadata["teacher_conditioning_style"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)
        self.assertEqual(metadata["conditioning_pair"], CONDITIONING_STYLE_ADDITIVE_MLP_V1)

    def test_teacher_model_config_contains_metric_vector_metadata(self) -> None:
        setting_dim = int(setting_features("euler", 4).numel())
        teacher = build_gipo_teacher_model(
            architecture=ARCHITECTURE_DENSITY_FORM_TRANSFORMER_V1,
            setting_dim=setting_dim,
            density_dim=64,
            context_dim=2,
            num_series=1,
            model_config={"hidden_dim": 16, "hidden_layers": 1, "attention_heads": 4, "dropout": 0.0},
        )

        config = teacher.model_config()
        self.assertEqual(config["teacher_output"], TEACHER_OUTPUT_METRIC_VECTOR_V1)
        self.assertEqual(config["teacher_metric_targets"], list(TEACHER_METRIC_TARGET_KEYS))
        self.assertEqual(config["teacher_scalarization"], TEACHER_SCALARIZATION_WEIGHTED_AVERAGE_V1)


if __name__ == "__main__":
    unittest.main()
