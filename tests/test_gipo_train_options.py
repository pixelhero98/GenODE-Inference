from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from genode.canonical_experiment_layout import (
    CANONICAL_SEEN_NFES,
    CANONICAL_UNSEEN_NFES,
    SCENARIO_FAMILY_CONDITIONAL_GENERATION,
    SCENARIO_FAMILY_FORECAST,
)
from genode.gipo.objectives import CONDITIONAL_PRIMARY_LOB_METRIC_SPECS, FORECAST_METRIC_SPECS
from genode.gipo.policy import GIPO_PROTOCOL, save_context_embedding_table
from genode.gipo.train_gipo import _resolve_teacher_metric_target_keys, build_argparser, train_gipo


SUPPORT_SCHEDULES = ("uniform", "late_power_3")
ROW_FIELDS = (
    "benchmark_family",
    "dataset",
    "split_phase",
    "seed",
    "solver_key",
    "target_nfe",
    "scheduler_key",
    "context_id",
    "context_embedding_id",
    "checkpoint_id",
    "series_id",
    "target_t",
    "gipo_reward_protocol",
    "u_comp_uniform",
    "u_alt_uniform",
)


def _write_rows(
    path: Path,
    *,
    target_nfes: tuple[int, ...],
    schedules: tuple[str, ...] = SUPPORT_SCHEDULES,
    contexts: tuple[str, ...] = ("ctx_0", "ctx_1", "ctx_2"),
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for ctx_idx, context_id in enumerate(contexts):
            for target_nfe in target_nfes:
                for schedule_idx, scheduler_key in enumerate(schedules):
                    writer.writerow(
                        {
                            "benchmark_family": SCENARIO_FAMILY_FORECAST,
                            "dataset": "private_forecast_dataset",
                            "split_phase": "train_tuning",
                            "seed": 0,
                            "solver_key": "euler",
                            "target_nfe": target_nfe,
                            "scheduler_key": scheduler_key,
                            "context_id": context_id,
                            "context_embedding_id": context_id,
                            "checkpoint_id": "",
                            "series_id": f"series_{ctx_idx}",
                            "target_t": 100 + ctx_idx,
                            "gipo_reward_protocol": GIPO_PROTOCOL,
                            "u_comp_uniform": float(schedule_idx),
                            "u_alt_uniform": "",
                        }
                    )


def _write_embeddings(path: Path, contexts: tuple[str, ...] = ("ctx_0", "ctx_1", "ctx_2")) -> None:
    save_context_embedding_table(
        path,
        {context_id: [float(idx), 1.0 - float(idx)] for idx, context_id in enumerate(contexts)},
    )


def _trainer_args(root: Path, rows_csv: Path, embeddings_npz: Path, *extra: str) -> argparse.Namespace:
    return build_argparser().parse_args(
        [
            "--rows_csv",
            str(rows_csv),
            "--context_embeddings_npz",
            str(embeddings_npz),
            "--out_dir",
            str(root / "out"),
            "--support_schedule_keys",
            ",".join(SUPPORT_SCHEDULES),
            "--context_sample_count",
            "3",
            "--context_holdout_fraction",
            "0.5",
            "--teacher_density_holdout_schedule_keys",
            "",
            "--student_selection_holdout_fraction",
            "0.5",
            "--teacher_metric_target_keys",
            "u_comp_uniform",
            "--transformer_hidden_dim",
            "16",
            "--transformer_layers",
            "1",
            "--transformer_heads",
            "4",
            "--transformer_dropout",
            "0.0",
            "--dry_run",
            *extra,
        ]
    )


class GipoTrainOptionsTests(unittest.TestCase):
    def test_parser_defaults_keep_canonical_seen_and_pseudo_nfes(self) -> None:
        args = build_argparser().parse_args(
            [
                "--rows_csv",
                "rows.csv",
                "--context_embeddings_npz",
                "ctx.npz",
                "--out_dir",
                "out",
            ]
        )

        self.assertEqual(args.seen_target_nfe_values, "4,8,12,16")
        self.assertEqual(args.pseudo_target_nfe_values, "6,10,14,20")
        self.assertEqual(args.teacher_metric_min_coverage_fraction, 1.0)
        self.assertEqual(args.teacher_metric_min_valid_rows, 1)

    def test_invalid_step_settings_fail_before_data_loading(self) -> None:
        args = build_argparser().parse_args(
            [
                "--rows_csv",
                "missing.csv",
                "--context_embeddings_npz",
                "missing.npz",
                "--out_dir",
                "out",
                "--teacher_steps",
                "0",
            ]
        )

        with self.assertRaisesRegex(ValueError, "teacher_steps must be positive"):
            train_gipo(args)

    def test_default_density_holdout_fails_early_for_reduced_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_csv = root / "seen.csv"
            embeddings_npz = root / "ctx.npz"
            _write_rows(rows_csv, target_nfes=CANONICAL_SEEN_NFES)
            _write_embeddings(embeddings_npz)
            args = build_argparser().parse_args(
                [
                    "--rows_csv",
                    str(rows_csv),
                    "--context_embeddings_npz",
                    str(embeddings_npz),
                    "--out_dir",
                    str(root / "out"),
                    "--support_schedule_keys",
                    ",".join(SUPPORT_SCHEDULES),
                    "--context_sample_count",
                    "3",
                    "--teacher_metric_target_keys",
                    "u_comp_uniform",
                    "--dry_run",
                ]
            )

            with self.assertRaisesRegex(ValueError, "Default teacher_density_holdout_schedule_keys"):
                train_gipo(args)

    def test_auto_teacher_targets_fall_back_from_noncanonical_dataset_keys(self) -> None:
        args = argparse.Namespace(teacher_metric_target_keys="auto")
        expected = tuple(spec.utility_key for spec in FORECAST_METRIC_SPECS)

        self.assertEqual(
            _resolve_teacher_metric_target_keys(
                args,
                [{"benchmark_family": SCENARIO_FAMILY_FORECAST, "dataset": "private_forecast_dataset"}],
            ),
            expected,
        )
        self.assertEqual(
            _resolve_teacher_metric_target_keys(
                args,
                [{"dataset": "private_forecast_dataset", "forecast_crps": "1.0", "forecast_mase": "1.0"}],
            ),
            expected,
        )
        self.assertEqual(
            _resolve_teacher_metric_target_keys(
                args,
                [{"benchmark_family": SCENARIO_FAMILY_CONDITIONAL_GENERATION, "dataset": "custom_lobster_like"}],
            ),
            tuple(spec.utility_key for spec in CONDITIONAL_PRIMARY_LOB_METRIC_SPECS),
        )

    def test_seen_target_nfe_values_drive_sampled_seen_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_csv = root / "seen.csv"
            embeddings_npz = root / "ctx.npz"
            _write_rows(rows_csv, target_nfes=(5, 7))
            _write_embeddings(embeddings_npz)

            with self.assertRaisesRegex(ValueError, r"seen calibration NFEs \[4, 8, 12, 16\].*found \[5, 7\]"):
                train_gipo(_trainer_args(root, rows_csv, embeddings_npz))

            summary = train_gipo(
                _trainer_args(root, rows_csv, embeddings_npz, "--seen_target_nfe_values", "5,7")
            )

        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["seen_target_nfe_values"], [5, 7])
        self.assertEqual(summary["canonical_seen_nfes"], list(CANONICAL_SEEN_NFES))

    def test_context_sample_count_is_per_checkpoint_maturity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_csv = root / "seen.csv"
            embeddings_npz = root / "ctx.npz"
            contexts = tuple(f"ctx_{idx}" for idx in range(4))
            with rows_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
                writer.writeheader()
                for checkpoint_id in ("ckpt_a", "ckpt_b"):
                    for ctx_idx, context_id in enumerate(contexts):
                        for target_nfe in (5, 7):
                            for schedule_idx, scheduler_key in enumerate(SUPPORT_SCHEDULES):
                                writer.writerow(
                                    {
                                        "benchmark_family": SCENARIO_FAMILY_FORECAST,
                                        "dataset": "private_forecast_dataset",
                                        "split_phase": "train_tuning",
                                        "seed": 0,
                                        "solver_key": "euler",
                                        "target_nfe": target_nfe,
                                        "scheduler_key": scheduler_key,
                                        "context_id": context_id,
                                        "context_embedding_id": f"{checkpoint_id}:{context_id}",
                                        "checkpoint_id": checkpoint_id,
                                        "series_id": f"series_{ctx_idx}",
                                        "target_t": 100 + ctx_idx,
                                        "gipo_reward_protocol": GIPO_PROTOCOL,
                                        "u_comp_uniform": float(schedule_idx),
                                        "u_alt_uniform": "",
                                    }
                                )
            save_context_embedding_table(
                embeddings_npz,
                {
                    f"{checkpoint_id}:{context_id}": [float(ctx_idx), float(checkpoint_idx)]
                    for checkpoint_idx, checkpoint_id in enumerate(("ckpt_a", "ckpt_b"))
                    for ctx_idx, context_id in enumerate(contexts)
                },
            )

            args = _trainer_args(
                root,
                rows_csv,
                embeddings_npz,
                "--seen_target_nfe_values",
                "5,7",
                "--context_sample_count",
                "3",
                "--context_holdout_fraction",
                "0",
                "--student_selection_holdout_fraction",
                "0.5",
            )
            summary = train_gipo(args)

        self.assertEqual(summary["context_sampling"]["sample_count_per_checkpoint"], 3)
        self.assertEqual(summary["context_sampling"]["per_checkpoint"]["ckpt_a"]["selected_contexts"], 3)
        self.assertEqual(summary["context_sampling"]["per_checkpoint"]["ckpt_b"]["selected_contexts"], 3)
        self.assertEqual(summary["context_sampling"]["checkpoint_count"], 2)

    def test_legacy_checkpoint_prefixed_context_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_csv = root / "seen.csv"
            embeddings_npz = root / "ctx.npz"
            contexts = tuple(f"ctx_{idx}" for idx in range(3))
            embeddings = {}
            with rows_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
                writer.writeheader()
                for checkpoint_idx, checkpoint_id in enumerate(("ckpt_a", "ckpt_b")):
                    for ctx_idx, physical_context_id in enumerate(contexts):
                        legacy_context_id = f"{checkpoint_id}:{physical_context_id}"
                        embeddings[legacy_context_id] = [float(ctx_idx), float(checkpoint_idx)]
                        for target_nfe in (5, 7):
                            for schedule_idx, scheduler_key in enumerate(SUPPORT_SCHEDULES):
                                writer.writerow(
                                    {
                                        "benchmark_family": SCENARIO_FAMILY_FORECAST,
                                        "dataset": "private_forecast_dataset",
                                        "split_phase": "train_tuning",
                                        "seed": 0,
                                        "solver_key": "euler",
                                        "target_nfe": target_nfe,
                                        "scheduler_key": scheduler_key,
                                        "context_id": legacy_context_id,
                                        "context_embedding_id": legacy_context_id,
                                        "checkpoint_id": checkpoint_id,
                                        "series_id": f"series_{ctx_idx}",
                                        "target_t": 100 + ctx_idx,
                                        "gipo_reward_protocol": GIPO_PROTOCOL,
                                        "u_comp_uniform": float(schedule_idx),
                                        "u_alt_uniform": "",
                                    }
                                )
            save_context_embedding_table(embeddings_npz, embeddings)

            with self.assertRaisesRegex(ValueError, "physical context_id values without checkpoint prefixes"):
                train_gipo(_trainer_args(root, rows_csv, embeddings_npz, "--seen_target_nfe_values", "5,7"))

    def test_dry_run_reports_split_normalizer_scopes_and_effective_selection_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_csv = root / "seen.csv"
            embeddings_npz = root / "ctx.npz"
            _write_rows(rows_csv, target_nfes=CANONICAL_SEEN_NFES)
            _write_embeddings(embeddings_npz)

            summary = train_gipo(
                _trainer_args(
                    root,
                    rows_csv,
                    embeddings_npz,
                    "--teacher_density_holdout_schedule_keys",
                    "late_power_3",
                )
            )

        self.assertEqual(summary["teacher_selection_nominal_axis_weights"], {"context": 0.25, "density_family": 0.25, "unseen_nfe": 0.5})
        self.assertEqual(summary["teacher_selection_effective_axis_weights"], {"context": 0.5, "density_family": 0.5, "unseen_nfe": 0.0})
        self.assertEqual(summary["teacher_selection_inactive_axes"], ["unseen_nfe"])
        scopes = summary["normalizer_fit_scopes"]
        self.assertEqual(scopes["protocol"], "selector_final_normalizer_scopes_v1")
        self.assertEqual(scopes["selector"]["embedding"]["row_count"], 4)
        self.assertEqual(scopes["selector"]["embedding"]["context_count"], 1)
        self.assertEqual(scopes["selector"]["embedding"]["schedule_count"], 1)
        self.assertEqual(scopes["final"]["embedding"]["row_count"], 24)
        self.assertEqual(scopes["final"]["embedding"]["context_count"], 3)
        self.assertEqual(scopes["final"]["embedding"]["schedule_count"], 2)
        self.assertIn("membership_hash", scopes["selector"]["embedding"])
        self.assertNotEqual(
            scopes["selector"]["embedding"]["membership_hash"],
            scopes["final"]["embedding"]["membership_hash"],
        )

    def test_teacher_metric_coverage_is_strict_by_default_and_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_csv = root / "seen.csv"
            embeddings_npz = root / "ctx.npz"
            _write_rows(rows_csv, target_nfes=CANONICAL_SEEN_NFES)
            _write_embeddings(embeddings_npz)

            with self.assertRaisesRegex(ValueError, r"Teacher metric target coverage failed.*u_alt_uniform"):
                train_gipo(
                    _trainer_args(
                        root,
                        rows_csv,
                        embeddings_npz,
                        "--teacher_metric_target_keys",
                        "u_comp_uniform,u_alt_uniform",
                    )
                )

            summary = train_gipo(
                _trainer_args(
                    root,
                    rows_csv,
                    embeddings_npz,
                    "--teacher_metric_target_keys",
                    "u_comp_uniform,u_alt_uniform",
                    "--teacher_metric_min_coverage_fraction",
                    "0",
                    "--teacher_metric_min_valid_rows",
                    "0",
                )
            )

        coverage = summary["teacher_metric_target_coverage"]
        self.assertEqual(coverage["thresholds"], {"min_coverage_fraction": 0.0, "min_valid_rows": 0})
        self.assertEqual(coverage["scopes"]["rows_csv"]["metrics"]["u_comp_uniform"]["valid_row_count"], 24)
        self.assertEqual(coverage["scopes"]["rows_csv"]["metrics"]["u_alt_uniform"]["valid_row_count"], 0)

    def test_pseudo_target_nfe_values_filter_rows_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seen_csv = root / "seen.csv"
            pseudo_csv = root / "pseudo.csv"
            embeddings_npz = root / "ctx.npz"
            _write_rows(seen_csv, target_nfes=CANONICAL_SEEN_NFES)
            _write_rows(pseudo_csv, target_nfes=(5, 6, 7))
            _write_embeddings(embeddings_npz)

            summary = train_gipo(
                _trainer_args(
                    root,
                    seen_csv,
                    embeddings_npz,
                    "--student_pseudo_rows_csv",
                    str(pseudo_csv),
                    "--student_pseudo_context_embeddings_npz",
                    str(embeddings_npz),
                    "--pseudo_target_nfe_values",
                    "5,7",
                )
            )

        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["pseudo_target_nfe_values"], [5, 7])
        self.assertEqual(summary["canonical_unseen_nfes"], list(CANONICAL_UNSEEN_NFES))
        self.assertEqual(summary["student_pseudo_distillation"]["target_nfes"], [5, 7])
        self.assertEqual(summary["split_counts"]["student_pseudo"]["row_count"], 12)

    def test_pseudo_target_nfe_values_are_named_in_empty_filter_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seen_csv = root / "seen.csv"
            pseudo_csv = root / "pseudo.csv"
            embeddings_npz = root / "ctx.npz"
            _write_rows(seen_csv, target_nfes=CANONICAL_SEEN_NFES)
            _write_rows(pseudo_csv, target_nfes=(6,))
            _write_embeddings(embeddings_npz)

            with self.assertRaisesRegex(ValueError, r"pseudo_target_nfe_values \[5, 7\]"):
                train_gipo(
                    _trainer_args(
                        root,
                        seen_csv,
                        embeddings_npz,
                        "--student_pseudo_rows_csv",
                        str(pseudo_csv),
                        "--student_pseudo_context_embeddings_npz",
                        str(embeddings_npz),
                        "--pseudo_target_nfe_values",
                        "5,7",
                    )
                )

    def test_pseudo_support_schedules_must_match_seen_support_schedules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seen_csv = root / "seen.csv"
            pseudo_csv = root / "pseudo.csv"
            embeddings_npz = root / "ctx.npz"
            _write_rows(seen_csv, target_nfes=CANONICAL_SEEN_NFES)
            _write_rows(pseudo_csv, target_nfes=CANONICAL_UNSEEN_NFES, schedules=("uniform", "ays"))
            _write_embeddings(embeddings_npz)

            with self.assertRaisesRegex(ValueError, r"missing=\['late_power_3'\], extra=\['ays'\]"):
                train_gipo(
                    _trainer_args(
                        root,
                        seen_csv,
                        embeddings_npz,
                        "--student_pseudo_rows_csv",
                        str(pseudo_csv),
                        "--student_pseudo_context_embeddings_npz",
                        str(embeddings_npz),
                    )
                )


if __name__ == "__main__":
    unittest.main()
