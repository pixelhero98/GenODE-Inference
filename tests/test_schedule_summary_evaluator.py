from __future__ import annotations

import csv
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from genode.gipo.ablation_plan import GIPO_POLICY_KEY
from genode.gipo.evaluate_schedule_summary import (
    _load_existing_rows,
    _load_forecast_rows_csv,
    _output_path,
    _protocol_hash,
    _row_has_complete_context_artifacts,
    _split_example_cap,
    _validate_schedule_checkpoint_identity,
    _validate_distinct_artifact_paths,
    _write_jsonl,
    build_argparser,
    build_comparison_summary,
    evaluate_schedule_summary,
    load_schedule_predictions,
    select_best_validation_schedule,
    write_selected_schedule_summary,
)
from genode.gipo.schedule_grids import load_schedule_summary_grids
from genode.gipo.ser_ptg_reference import (
    SER_PTG_AVG_REVERSED_SCHEDULE_KEY,
    SER_PTG_REVERSED_SCHEDULE_KEY,
    SER_PTG_SCHEDULE_KEY,
)
from genode.evaluation.otflow_evaluation_support import (
    TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
    choose_forecast_train_tuning_indices,
    train_tuning_target_example_count,
)
from genode.models.otflow_train_val import save_json as atomic_save_json
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS


def _uniform_grid(n_steps: int) -> list[float]:
    return [float(idx) / float(n_steps) for idx in range(n_steps + 1)]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _panel_fields(
    seed: int,
    *,
    panel: str = "shared",
    protocol: str = "shared",
) -> dict[str, object]:
    return {
        "chosen_examples_hash": _sha256(f"{panel}-{int(seed)}"),
        "evaluation_protocol_hash": _sha256(f"{protocol}-{int(seed)}"),
        "num_eval_samples": 1,
        "eval_examples": 1,
        "eval_horizon": 1,
    }


class ScheduleSummaryEvaluatorTests(unittest.TestCase):
    def test_resume_rejects_existing_rows_from_another_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            original = json.dumps(
                {"protocol_hash": "old", "row_status": "complete"}, sort_keys=True
            ) + "\n"
            path.write_text(original, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "different protocol"):
                _load_existing_rows(path, protocol_hash="new")
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_resume_discards_only_unterminated_final_jsonl_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            row = {
                "protocol_hash": "protocol",
                "row_status": "complete",
                "split_phase": "validation_tuning",
                "seed": 0,
                "scenario_key": "traffic_hourly",
                "target_nfe": 4,
                "solver_key": "euler",
                "scheduler_key": "uniform",
            }
            path.write_text(
                json.dumps(row, sort_keys=True) + "\n" + '{"protocol_hash":',
                encoding="utf-8",
            )

            loaded = _load_existing_rows(path, protocol_hash="protocol")

            self.assertEqual(list(loaded.values()), [row])

    def test_resume_rejects_terminated_or_mid_file_jsonl_corruption(self) -> None:
        valid = json.dumps(
            {"protocol_hash": "protocol", "row_status": "complete"},
            sort_keys=True,
        )
        malformed = '{"protocol_hash":'
        for journal in (
            valid + "\n" + malformed + "\n",
            valid + "\n" + malformed + "\n" + valid + "\n",
        ):
            with self.subTest(journal=journal), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "rows.jsonl"
                path.write_text(journal, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "line 2"):
                    _load_existing_rows(path, protocol_hash="protocol")

    def test_jsonl_compaction_preserves_target_on_serialization_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            original = '{"existing":true}\n'
            path.write_text(original, encoding="utf-8")

            with self.assertRaises(TypeError):
                _write_jsonl(path, [{"not_json": {1}}])

            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(Path(tmpdir).glob(".rows.jsonl.*.tmp")), [])

    def test_output_paths_reject_traversal_and_wrong_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "may not contain"):
                _output_path(
                    root,
                    "../rows.csv",
                    fallback="rows.csv",
                    suffix=".csv",
                    label="row output",
                )
            with self.assertRaisesRegex(ValueError, "must end"):
                _output_path(
                    root,
                    "rows.jsonl",
                    fallback="rows.csv",
                    suffix=".csv",
                    label="row output",
                )

    def test_output_paths_include_implicit_embedding_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "context_embeddings.npz.manifest.json"
            with self.assertRaisesRegex(ValueError, "implicit sidecars"):
                _validate_distinct_artifact_paths(
                    {
                        "context-embedding manifest": manifest,
                        "summary JSON": manifest,
                    }
                )

    def test_evaluator_rejects_input_output_collisions_before_loading(self) -> None:
        cases = (
            ("schedule_summary", "validation_tuning_schedule_summary.json"),
            ("baseline_rows", "validation_rows.csv"),
            ("comparator_rows", "validation_rows.csv"),
            ("selection_reference_rows", "validation_rows.csv"),
        )
        for argument, output_name in cases:
            with self.subTest(argument=argument), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                out_dir = root / "out"
                out_dir.mkdir()
                collision_path = out_dir / output_name
                original = b"input artifact must remain unchanged\n"
                collision_path.write_bytes(original)
                safe_summary = root / "schedule_summary.json"
                cli = [
                    "--scenario_key",
                    "traffic_hourly",
                    "--schedule_summary",
                    str(collision_path if argument == "schedule_summary" else safe_summary),
                    "--split_phase",
                    "validation_tuning",
                    "--out_dir",
                    str(out_dir),
                ]
                if argument != "schedule_summary":
                    cli.extend([f"--{argument}", str(collision_path)])
                args = build_argparser().parse_args(cli)

                with mock.patch(
                    "genode.gipo.evaluate_schedule_summary.load_schedule_predictions"
                ) as load_mock, self.assertRaisesRegex(
                    ValueError,
                    "input/output_collisions",
                ):
                    evaluate_schedule_summary(args)

                load_mock.assert_not_called()
                self.assertEqual(collision_path.read_bytes(), original)

    def test_imported_comparison_rows_validate_solver_nfe_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=(
                        "benchmark_family",
                        "split_phase",
                        "scenario_key",
                        "seed",
                        "solver_key",
                        "target_nfe",
                        "macro_steps",
                        "realized_nfe",
                        "checkpoint_step",
                        "checkpoint_id",
                        "scheduler_key",
                        "forecast_crps",
                        "forecast_mase",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "benchmark_family": "temporal_extrapolation",
                        "split_phase": "validation_tuning",
                        "scenario_key": "traffic_hourly",
                        "seed": 0,
                        "solver_key": "heun",
                        "target_nfe": 4,
                        "macro_steps": 2,
                        "realized_nfe": 8,
                        "checkpoint_step": 20_000,
                        "checkpoint_id": "checkpoint",
                        "scheduler_key": "uniform",
                        "forecast_crps": 1.0,
                        "forecast_mase": 1.0,
                    }
                )

            with self.assertRaisesRegex(ValueError, "realized_nfe=8"):
                _load_forecast_rows_csv(
                    path,
                    scenario_key="traffic_hourly",
                    split_phase="validation_tuning",
                    seeds=(0,),
                    solver_names=("heun",),
                    target_nfe_values=(4,),
                    checkpoint_step=20_000,
                    checkpoint_id="checkpoint",
                )

    def test_imported_comparison_rows_reject_fractional_gipo_step_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.csv"
            row = {
                "split_phase": "validation_tuning",
                "scenario_key": "traffic_hourly",
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "macro_steps": 4,
                "realized_nfe": 4,
                "checkpoint_step": 20_000,
                "checkpoint_id": "checkpoint",
                "scheduler_key": "gipo_candidate",
                "forecast_crps": 1.0,
                "forecast_mase": 1.0,
                "gipo_step_budget": "2.5",
            }
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            with self.assertRaisesRegex(ValueError, "gipo_step_budget must be an integer"):
                _load_forecast_rows_csv(
                    path,
                    scenario_key="traffic_hourly",
                    split_phase="validation_tuning",
                    seeds=(0,),
                    solver_names=("euler",),
                    target_nfe_values=(4,),
                    checkpoint_step=20_000,
                    checkpoint_id="checkpoint",
                )

    def test_schedule_predictions_reject_fractional_gipo_step_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "schedule.json"
            path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "checkpoint_step": 20_000,
                        "checkpoint_ids": ["checkpoint"],
                        "gipo_step_budget": 2.5,
                        "scheduler_key": GIPO_POLICY_KEY,
                        "predictions": [
                            {
                                "solver_key": "euler",
                                "target_nfe": 4,
                                "macro_steps": 4,
                                "realized_nfe": 4,
                                "time_grid": _uniform_grid(4),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "gipo_step_budget must be an integer",
            ):
                load_schedule_predictions(
                    path,
                    scenario_key="traffic_hourly",
                    solver_names=("euler",),
                    target_nfe_values=(4,),
                    require_complete=True,
                )

    def test_checkpoint_identity_is_explicit_and_strict(self) -> None:
        base = {"checkpoint_step": 4000, "checkpoint_ids": ["checkpoint-a"]}
        _validate_schedule_checkpoint_identity(
            {("gipo", "euler", 4): base},
            checkpoint_step=4000,
            checkpoint_id="checkpoint-a",
        )
        for invalid in (
            {"checkpoint_ids": ["checkpoint-a"]},
            {"checkpoint_step": 4000.5, "checkpoint_ids": ["checkpoint-a"]},
            {"checkpoint_step": 4000},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _validate_schedule_checkpoint_identity(
                    {("gipo", "euler", 4): invalid},
                    checkpoint_step=4000,
                    checkpoint_id="checkpoint-a",
                )

    def test_resume_requires_exact_identity_bound_context_artifacts(self) -> None:
        parent = {
            "row_status": "complete",
            "row_signature": "parent-row",
            "selected_examples": 1,
            "benchmark_family": "temporal_extrapolation",
            "protocol_hash": "protocol-a",
            "scenario_key": "traffic_hourly",
            "split_phase": "validation_tuning",
            "seed": 0,
            "solver_key": "euler",
            "target_nfe": 4,
            "scheduler_key": "uniform",
            "checkpoint_step": 4000,
            "checkpoint_id": "checkpoint-a",
        }
        context = {
            **parent,
            "seed": "0",
            "target_nfe": "4",
            "checkpoint_step": "4000",
            "parent_row_signature": "parent-row",
            "row_signature": "context-row",
            "context_id": "context-a",
            "context_embedding_id": "embedding-a",
            "context_embedding_kind": "ctx_summary",
        }
        self.assertTrue(
            _row_has_complete_context_artifacts(
                parent,
                context_rows_by_signature={"context-row": context},
                context_embeddings={"embedding-a": [0.0, 1.0]},
                context_embedding_kind="ctx_summary",
            )
        )
        self.assertFalse(
            _row_has_complete_context_artifacts(
                parent,
                context_rows_by_signature={
                    "context-row": {**context, "scenario_key": "wrong-scenario"}
                },
                context_embeddings={"embedding-a": [0.0, 1.0]},
                context_embedding_kind="ctx_summary",
            )
        )
        self.assertFalse(
            _row_has_complete_context_artifacts(
                parent,
                context_rows_by_signature={"context-row": context},
                context_embeddings={},
                context_embedding_kind="ctx_summary",
            )
        )

    def test_train_tuning_hash_sampling_is_deterministic_and_stratified(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 100

        first = choose_forecast_train_tuning_indices(FakeDataset(), fraction=0.20, seed=7, strata=20, dataset="sf")
        second = choose_forecast_train_tuning_indices(FakeDataset(), fraction=0.20, seed=7, strata=20, dataset="sf")
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 20)
        self.assertEqual(len({int(idx) // 5 for idx in first.tolist()}), 20)

    def test_train_tuning_target_count_matches_small_split_stratified_sampler(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 10

        chosen = choose_forecast_train_tuning_indices(FakeDataset(), fraction=0.20, seed=7, strata=20, dataset="small")
        target = train_tuning_target_example_count(10, fraction=0.20, strata=20)

        self.assertEqual(len(chosen), 10)
        self.assertEqual(target, len(chosen))

    def test_validation_normalized_train_tuning_sampling_uses_holdout_scale(self) -> None:
        class FakeTrainDataset:
            def __len__(self) -> int:
                return 14_399_710

        first = choose_forecast_train_tuning_indices(
            FakeTrainDataset(),
            fraction=0.20,
            seed=7,
            strata=20,
            dataset="traffic_hourly",
            sampling_mode=TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
            reference_examples=862,
            train_split_fraction=0.70,
            val_split_fraction=0.10,
        )
        second = choose_forecast_train_tuning_indices(
            FakeTrainDataset(),
            fraction=0.20,
            seed=7,
            strata=20,
            dataset="traffic_hourly",
            sampling_mode=TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
            reference_examples=862,
            train_split_fraction=0.70,
            val_split_fraction=0.10,
        )
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 1207)
        self.assertEqual(len({int(idx) * 20 // 14_399_710 for idx in first.tolist()}), 20)

    def test_train_tuning_sampling_can_be_capped_before_large_candidate_materialization(self) -> None:
        class FakeTrainDataset:
            def __len__(self) -> int:
                return 1_000_000

        chosen = choose_forecast_train_tuning_indices(
            FakeTrainDataset(),
            fraction=1.0,
            seed=7,
            strata=20,
            dataset="traffic_hourly",
            max_examples=256,
        )

        self.assertEqual(len(chosen), 256)
        self.assertEqual(chosen.tolist(), sorted(chosen.tolist()))
        self.assertEqual(len(set(chosen.tolist())), 256)

    def test_schedule_evaluator_protocol_tracks_train_tuning_sampling_mode(self) -> None:
        base = [
            "--scenario_key",
            "traffic_hourly",
            "--schedule_summary",
            "dummy.json",
            "--split_phase",
            "train_tuning",
            "--seeds",
            "0",
            "--solver_names",
            "euler",
            "--target_nfe_values",
            "4",
            "--device",
            "cpu",
        ]
        legacy = build_argparser().parse_args([*base, "--train_tuning_sampling_mode", "train_window_fraction"])
        valnorm = build_argparser().parse_args([*base, "--train_tuning_sampling_mode", "validation_normalized"])
        valnorm_alt_fraction = build_argparser().parse_args(
            [*base, "--train_tuning_sampling_mode", "validation_normalized", "--train_tuning_train_split_fraction", "0.60"]
        )
        self.assertNotEqual(_protocol_hash(legacy), _protocol_hash(valnorm))
        self.assertNotEqual(_protocol_hash(valnorm), _protocol_hash(valnorm_alt_fraction))

    def test_load_schedule_predictions_validates_ser_ptg_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ser_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "artifact": "ser_ptg_schedule_summary",
                        "scenario_key": "traffic_hourly",
                        "schedules": [
                            {
                                "scheduler_key": SER_PTG_SCHEDULE_KEY,
                                "schedule_name": "SER-PTG local defect eta=0.05",
                                "predictions": [
                                    {
                                        "solver_key": "heun",
                                        "target_nfe": 4,
                                        "macro_steps": 2,
                                        "time_grid": _uniform_grid(2),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            predictions = load_schedule_predictions(
                path,
                scenario_key="traffic_hourly",
                solver_names=("heun",),
                target_nfe_values=(4,),
            )
        self.assertIn((SER_PTG_SCHEDULE_KEY, "heun", 4), predictions)
        self.assertIn((SER_PTG_REVERSED_SCHEDULE_KEY, "heun", 4), predictions)
        self.assertIn((SER_PTG_AVG_REVERSED_SCHEDULE_KEY, "heun", 4), predictions)
        self.assertEqual(predictions[(SER_PTG_SCHEDULE_KEY, "heun", 4)]["realized_nfe"], 4)
        self.assertNotIn(SER_PTG_SCHEDULE_KEY, BASELINE_SCHEDULE_KEYS)

    def test_load_schedule_predictions_rejects_rk2_macro_steps_as_realized_nfe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ser_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "schedules": [
                            {
                                "scheduler_key": SER_PTG_SCHEDULE_KEY,
                                "predictions": [
                                    {
                                        "solver_key": "heun",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(2),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "macro_steps=4"):
                load_schedule_predictions(
                    path,
                    scenario_key="traffic_hourly",
                    solver_names=("heun",),
                    target_nfe_values=(4,),
                )

    def test_load_schedule_predictions_accepts_macro_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ser_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "schedules": [
                            {
                                "scheduler_key": SER_PTG_SCHEDULE_KEY,
                                "predictions": [
                                    {
                                        "solver_key": "heun",
                                        "target_nfe": 4,
                                        "macro_steps": 2,
                                        "time_grid": _uniform_grid(2),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            predictions = load_schedule_predictions(
                path,
                scenario_key="traffic_hourly",
                solver_names=("heun",),
                target_nfe_values=(4,),
            )

        row = predictions[(SER_PTG_SCHEDULE_KEY, "heun", 4)]
        self.assertEqual(row["macro_steps"], 2)
        self.assertEqual(row["realized_nfe"], 4)

    def test_schedule_grid_loader_keeps_claim_grids_checkpoint_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ser_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "artifact": "ser_ptg_schedule_summary",
                        "scenario_key": "traffic_hourly",
                        "checkpoint_step": 4000,
                        "checkpoint_ids": ["forecast-ckpt"],
                        "schedules": [
                            {
                                "scheduler_key": SER_PTG_SCHEDULE_KEY,
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            grids = load_schedule_summary_grids(
                [str(path)],
                expected_rows=[
                    {"checkpoint_step": 4000, "checkpoint_id": "forecast-ckpt"},
                    {"checkpoint_step": 8000, "checkpoint_id": "later-ckpt"},
                ],
            )

        self.assertNotIn((SER_PTG_SCHEDULE_KEY, "euler", 4), grids)
        self.assertIn((SER_PTG_SCHEDULE_KEY, "euler", 4, 4000), grids)
        self.assertNotIn((SER_PTG_SCHEDULE_KEY, "euler", 4, 8000), grids)
        self.assertNotIn((SER_PTG_REVERSED_SCHEDULE_KEY, "euler", 4), grids)
        self.assertNotIn((SER_PTG_AVG_REVERSED_SCHEDULE_KEY, "euler", 4), grids)

    def test_schedule_grid_loader_rejects_checkpoint_identity_mismatch(self) -> None:
        payload = {
            "status": "ready",
            "artifact": "ser_ptg_schedule_summary",
            "scenario_key": "traffic_hourly",
            "checkpoint_step": 4000,
            "checkpoint_ids": ["forecast-ckpt"],
            "schedules": [
                {
                    "scheduler_key": SER_PTG_SCHEDULE_KEY,
                    "predictions": [
                        {
                            "solver_key": "euler",
                            "target_nfe": 4,
                            "macro_steps": 4,
                            "time_grid": _uniform_grid(4),
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ser_summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint_ids do not match"):
                load_schedule_summary_grids(
                    [str(path)],
                    expected_rows=[
                        {"checkpoint_step": 4000, "checkpoint_id": "other-ckpt"}
                    ],
                )

            del payload["checkpoint_ids"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires top-level checkpoint_ids"):
                load_schedule_summary_grids(
                    [str(path)],
                    expected_rows=[
                        {"checkpoint_step": 4000, "checkpoint_id": "forecast-ckpt"}
                    ],
                )

    def test_schedule_grid_loader_rejects_noninteger_nfe_metadata(self) -> None:
        base_prediction = {
            "solver_key": "euler",
            "target_nfe": 4,
            "macro_steps": 4,
            "realized_nfe": 4,
            "time_grid": _uniform_grid(4),
        }
        mutations = (
            ("target_nfe", 4.5),
            ("target_nfe", True),
            ("macro_steps", 4.5),
            ("realized_nfe", True),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            for field, value in mutations:
                with self.subTest(field=field, value=value):
                    prediction = {**base_prediction, field: value}
                    path.write_text(
                        json.dumps(
                            {
                                "scenario_key": "traffic_hourly",
                                "schedules": [
                                    {
                                        "scheduler_key": SER_PTG_SCHEDULE_KEY,
                                        "predictions": [prediction],
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "must be an integer"):
                        load_schedule_summary_grids([str(path)])

    def test_schedule_prediction_loader_rejects_noninteger_target_nfe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            for target_nfe in (4.5, True):
                with self.subTest(target_nfe=target_nfe):
                    path.write_text(
                        json.dumps(
                            {
                                "scenario_key": "traffic_hourly",
                                "schedules": [
                                    {
                                        "scheduler_key": "candidate",
                                        "predictions": [
                                            {
                                                "solver_key": "euler",
                                                "target_nfe": target_nfe,
                                                "macro_steps": 4,
                                                "realized_nfe": 4,
                                                "time_grid": _uniform_grid(4),
                                            }
                                        ],
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(ValueError, "non-integer target_nfe"):
                        load_schedule_predictions(
                            path,
                            scenario_key="traffic_hourly",
                            solver_names=("euler",),
                            target_nfe_values=(4,),
                        )

    def test_schedule_grid_loader_requires_train_reference_split_provenance(self) -> None:
        payload = {
            "status": "ready",
            "artifact": "ser_ptg_schedule_summary",
            "scenario_key": "traffic_hourly",
            "checkpoint_step": 4000,
            "checkpoint_ids": ["forecast-ckpt"],
            "reference_split": "train_tuning",
            "reference_split_key": "train",
            "schedules": [
                {
                    "scheduler_key": SER_PTG_SCHEDULE_KEY,
                    "predictions": [
                        {
                            "solver_key": "euler",
                            "target_nfe": 4,
                            "macro_steps": 4,
                            "time_grid": _uniform_grid(4),
                        }
                    ],
                }
            ],
        }
        expected_rows = [{"checkpoint_step": 4000, "checkpoint_id": "forecast-ckpt"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ser_summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            load_schedule_summary_grids(
                [str(path)],
                expected_scenario_key="traffic_hourly",
                expected_reference_split="train_tuning",
                expected_rows=expected_rows,
            )

            payload["reference_split"] = "validation_tuning"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reference_split='train_tuning'"):
                load_schedule_summary_grids(
                    [str(path)],
                    expected_scenario_key="traffic_hourly",
                    expected_reference_split="train_tuning",
                    expected_rows=expected_rows,
                )

            payload["reference_split"] = "train_tuning"
            del payload["reference_split_key"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reference_split_key='<missing>'"):
                load_schedule_summary_grids(
                    [str(path)],
                    expected_scenario_key="traffic_hourly",
                    expected_reference_split="train_tuning",
                    expected_rows=expected_rows,
                )

            payload["reference_split_key"] = "train"
            payload["status"] = "partial"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a ready"):
                load_schedule_summary_grids(
                    [str(path)],
                    expected_scenario_key="traffic_hourly",
                    expected_reference_split="train_tuning",
                    expected_rows=expected_rows,
                )

    def test_load_schedule_predictions_rejects_empty_filtered_candidate_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "student_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "schedules": [
                            {
                                "scheduler_key": "gipo_candidate_steps20",
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing predictions"):
                load_schedule_predictions(
                    path,
                    scenario_key="traffic_hourly",
                    solver_names=("heun",),
                    target_nfe_values=(4,),
                    require_complete=True,
                )

    def test_schedule_summary_rejects_retired_schema_keys(self) -> None:
        prediction = {
            "solver_key": "euler",
            "target_nfe": 4,
            "macro_steps": 4,
            "time_grid": _uniform_grid(4),
        }
        payloads = (
            {
                "dataset": "traffic_hourly",
                "scheduler_key": "gipo",
                "predictions": [prediction],
            },
            {
                "scenario_key": "traffic_hourly",
                "schedule_key": "gipo",
                "predictions": [prediction],
            },
            {
                "scenario_key": "traffic_hourly",
                "schedules": [
                    {
                        "scheduler_key": "gipo",
                        "gipo_steps": 25,
                        "predictions": [prediction],
                    }
                ],
            },
            {
                "scenario_key": "traffic_hourly",
                "schedules": [
                    {
                        "scheduler_key": "gipo",
                        "student_gipo_steps": 25,
                        "predictions": [prediction],
                    }
                ],
            },
            {
                "scenario_key": "traffic_hourly",
                "schedules": [
                    {
                        "scheduler_key": "gipo",
                        "selected_gipo_step_budget": 25,
                        "predictions": [prediction],
                    }
                ],
            },
            {
                "scenario_key": "traffic_hourly",
                "schedules": [
                    {
                        "scheduler_key": "gipo",
                        "predictions": [{**prediction, "gipo_steps": 25}],
                    }
                ],
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            for index, payload in enumerate(payloads):
                with self.subTest(index=index):
                    path = Path(tmpdir) / f"retired_{index}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "retired evaluation keys"):
                        load_schedule_predictions(
                            path,
                            scenario_key="traffic_hourly",
                            solver_names=("euler",),
                            target_nfe_values=(4,),
                        )

    def test_schedule_grid_loader_rejects_conflicting_checkpoint_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "conflicting_steps.json"
            path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "checkpoint_step": 4000,
                        "schedules": [
                            {
                                "scheduler_key": "gipo",
                                "checkpoint_step": 8000,
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Conflicting checkpoint_step values"):
                load_schedule_summary_grids([str(path)])

    def test_schedule_loader_rejects_conflicting_duplicate_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "conflicting_predictions.json"
            path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "schedules": [
                            {
                                "scheduler_key": "gipo",
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": time_grid,
                                    }
                                ],
                            }
                            for time_grid in (_uniform_grid(4), [0.0, 0.1, 0.4, 0.7, 1.0])
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Conflicting duplicate schedule prediction"):
                load_schedule_predictions(
                    path,
                    scenario_key="traffic_hourly",
                    solver_names=("euler",),
                    target_nfe_values=(4,),
                )

    def test_selected_schedule_summary_round_trip_preserves_shared_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "candidate_schedules.json"
            selected_path = root / "selected_schedule.json"
            teacher_final_retrain = {"enabled": True, "checkpoint_step": 20000}
            source_path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "method_key": "gipo",
                        "mode": "reference_first",
                        "teacher_final_retrain": teacher_final_retrain,
                        "checkpoint_step": 20000,
                        "checkpoint_id": "forecast-ckpt",
                        "checkpoint_ids": ["forecast-ckpt"],
                        "schedules": [
                            {
                                "scheduler_key": "gipo_candidate_steps25",
                                "gipo_step_budget": 25,
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            written = write_selected_schedule_summary(
                source_path,
                {
                    "selected_scheduler_key": "gipo_candidate_steps25",
                    "gipo_step_budget": 25,
                },
                selected_path,
            )
            reloaded = load_schedule_predictions(
                selected_path,
                scenario_key="traffic_hourly",
                solver_names=("euler",),
                target_nfe_values=(4,),
            )

        prediction = reloaded[(GIPO_POLICY_KEY, "euler", 4)]
        expected_metadata = {
            "method_key": "gipo",
            "gipo_step_budget": 25,
            "mode": "reference_first",
            "teacher_final_retrain": teacher_final_retrain,
            "checkpoint_step": 20000,
            "checkpoint_id": "forecast-ckpt",
            "checkpoint_ids": ["forecast-ckpt"],
        }
        for key, expected in expected_metadata.items():
            with self.subTest(key=key):
                self.assertEqual(written[key], expected)
                self.assertEqual(written["schedules"][0][key], expected)
                self.assertEqual(written["predictions"][0][key], expected)
                self.assertEqual(prediction[key], expected)

    def test_evaluation_seed_is_paired_across_schedules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            schedule_path = root / "schedules.json"
            schedule_path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "checkpoint_step": 20000,
                        "checkpoint_ids": ["checkpoint"],
                        "schedules": [
                            {
                                "scheduler_key": scheduler_key,
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                            for scheduler_key in ("uniform", "gipo")
                        ],
                    }
                ),
                encoding="utf-8",
            )

            class FakeDataset:
                def __len__(self) -> int:
                    return 1

            checkpoint = {
                "model": object(),
                "cfg": object(),
                "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
                "checkpoint_path": root / "model.pt",
                "checkpoint_id": "checkpoint",
                "backbone_name": "otflow",
                "checkpoint_step": 20000,
                "train_budget_label": "20k",
            }
            seeds = []

            def fake_eval(*args, **kwargs):
                del args
                seeds.append(int(kwargs["seed"]))
                return {
                    "forecast_crps": 1.0,
                    "forecast_mase": 1.0,
                    "forecast_mse": 1.0,
                    "realized_nfe": 4,
                }

            args = build_argparser().parse_args(
                [
                    "--scenario_key",
                    "traffic_hourly",
                    "--schedule_summary",
                    str(schedule_path),
                    "--split_phase",
                    "validation_tuning",
                    "--out_dir",
                    str(root / "out"),
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "7",
                    "--eval_windows_val",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            with mock.patch(
                "genode.gipo.evaluate_schedule_summary.load_forecast_checkpoint_splits",
                return_value=checkpoint,
            ), mock.patch(
                "genode.gipo.evaluate_schedule_summary.evaluate_forecast_schedule",
                side_effect=fake_eval,
            ):
                evaluate_schedule_summary(args)

            self.assertEqual(seeds, [7, 7])

    def test_comparison_summary_keeps_ser_ptg_as_comparator(self) -> None:
        baseline_rows = [
            {
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "forecast_crps": 2.0,
                "forecast_mase": 3.0,
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]
        ser_rows = [{"seed": 0, "solver_key": "euler", "target_nfe": 4, "scheduler_key": SER_PTG_SCHEDULE_KEY, "forecast_crps": 1.5, "forecast_mase": 2.5}]
        student_rows = [
            {"seed": 0, "solver_key": "euler", "target_nfe": 4, "scheduler_key": GIPO_POLICY_KEY, "forecast_crps": 1.25, "forecast_mase": 2.0}
        ]
        summary = build_comparison_summary(
            baseline_rows=baseline_rows,
            comparator_rows=ser_rows,
            student_rows=student_rows,
            scenario_key="traffic_hourly",
            split_phase="locked_test",
            seeds=(0,),
            solver_names=("euler",),
            target_nfe_values=(4,),
        )
        self.assertFalse(summary["ser_ptg_is_baseline"])
        self.assertEqual(summary["observed_ser_ptg_rows"], 1)
        self.assertEqual(summary["observed_student_rows"], 1)
        ranking = summary["cell_rankings"][0]
        self.assertEqual(ranking["forecast_crps_ranking"][0], GIPO_POLICY_KEY)
        self.assertAlmostEqual(ranking["student_relative_forecast_crps_gain_vs_ser_ptg"], 1.0 - 1.25 / 1.5)
        self.assertEqual(ranking["student_comparisons"][0]["scheduler_key"], GIPO_POLICY_KEY)

    def test_comparison_summary_supports_multiple_student_schedules(self) -> None:
        baseline_rows = [
            {
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "forecast_crps": 2.0,
                "forecast_mase": 3.0,
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]
        student_rows = [
            {"seed": 0, "solver_key": "euler", "target_nfe": 4, "scheduler_key": "density_a", "forecast_crps": 1.5, "forecast_mase": 2.5},
            {"seed": 0, "solver_key": "euler", "target_nfe": 4, "scheduler_key": "density_b", "forecast_crps": 1.25, "forecast_mase": 2.25},
        ]
        summary = build_comparison_summary(
            baseline_rows=baseline_rows,
            comparator_rows=[],
            student_rows=student_rows,
            scenario_key="traffic_hourly",
            split_phase="validation_tuning",
            seeds=(0,),
            solver_names=("euler",),
            target_nfe_values=(4,),
        )
        self.assertEqual(summary["student_scheduler_keys"], ["density_a", "density_b"])
        self.assertEqual(summary["expected_student_rows"], 2)
        self.assertEqual(summary["observed_student_rows"], 2)
        self.assertEqual(summary["missing_student_cells"], [])
        comparisons = summary["cell_rankings"][0]["student_comparisons"]
        self.assertEqual([row["scheduler_key"] for row in comparisons], ["density_a", "density_b"])
        self.assertAlmostEqual(comparisons[1]["student_relative_forecast_crps_gain_vs_best_baseline"], 1.0 - 1.25 / 2.0)

    def test_evaluator_filters_shared_comparison_rows_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            schedule_path = root / "student_summary.json"
            schedule_path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "checkpoint_step": 20000,
                        "checkpoint_ids": ["ck"],
                        "schedules": [
                            {
                                "scheduler_key": GIPO_POLICY_KEY,
                                "schedule_name": "GIPO Student Selected",
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            mixed_rows = [
                {"seed": 0, "solver_key": "euler", "target_nfe": 4, "scheduler_key": BASELINE_SCHEDULE_KEYS[0], "forecast_crps": 2.0, "forecast_mase": 3.0},
                {"seed": 0, "solver_key": "euler", "target_nfe": 4, "scheduler_key": SER_PTG_SCHEDULE_KEY, "forecast_crps": 1.5, "forecast_mase": 2.5},
                {"seed": 0, "solver_key": "euler", "target_nfe": 4, "scheduler_key": GIPO_POLICY_KEY, "forecast_crps": 1.0, "forecast_mase": 2.0},
            ]

            class FakeDataset:
                def __len__(self) -> int:
                    return 1

            fake_checkpoint = {
                "model": object(),
                "cfg": object(),
                "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
                "checkpoint_path": root / "model.pt",
                "checkpoint_id": "ck",
                "backbone_name": "otflow",
                "checkpoint_step": 20000,
                "train_budget_label": "20k",
            }
            args = build_argparser().parse_args(
                [
                    "--scenario_key",
                    "traffic_hourly",
                    "--schedule_summary",
                    str(schedule_path),
                    "--split_phase",
                    "locked_test",
                    "--out_dir",
                    str(root / "out"),
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--num_eval_samples",
                    "1",
                    "--locked_test_preview",
                    "--locked_test_preview_contexts",
                    "1",
                    "--baseline_rows",
                    str(root / "shared_rows.csv"),
                    "--comparator_rows",
                    str(root / "shared_rows.csv"),
                    "--device",
                    "cpu",
                ]
            )
            summary_path = root / "out" / "locked_test_schedule_summary.json"
            write_state = {"comparison_built": False, "summary_writes": 0}

            def build_comparison_once(**kwargs):
                del kwargs
                write_state["comparison_built"] = True
                return {"status": "captured"}

            def save_after_optional_outputs(payload, path):
                if Path(path) == summary_path:
                    self.assertTrue(write_state["comparison_built"])
                    write_state["summary_writes"] += 1
                atomic_save_json(payload, path)

            with mock.patch(
                "genode.gipo.evaluate_schedule_summary.load_forecast_checkpoint_splits",
                return_value=fake_checkpoint,
            ), mock.patch(
                "genode.gipo.evaluate_schedule_summary.evaluate_forecast_schedule",
                return_value={
                    "forecast_crps": 1.0,
                    "forecast_mse": 1.5,
                    "forecast_mase": 2.0,
                    "latency_ms_per_sample": 0.25,
                    "num_eval_samples": 1,
                    "eval_examples": 1,
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": 4,
                },
            ), mock.patch(
                "genode.gipo.evaluate_schedule_summary._load_forecast_rows_csv",
                return_value=mixed_rows,
            ), mock.patch(
                "genode.gipo.evaluate_schedule_summary.build_comparison_summary",
                side_effect=build_comparison_once,
            ) as build_mock:
                with mock.patch(
                    "genode.gipo.evaluate_schedule_summary.save_json",
                    side_effect=save_after_optional_outputs,
                ):
                    evaluate_schedule_summary(args)
                persisted_summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    persisted_summary["comparison_summary"],
                    {"status": "captured"},
                )
                self.assertEqual(write_state["summary_writes"], 1)

        call_kwargs = build_mock.call_args.kwargs
        self.assertEqual([row["scheduler_key"] for row in call_kwargs["baseline_rows"]], [BASELINE_SCHEDULE_KEYS[0]])
        self.assertEqual([row["scheduler_key"] for row in call_kwargs["comparator_rows"]], [SER_PTG_SCHEDULE_KEY])

    def test_mixed_checkpoint_comparison_rows_fail_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            schedule_path = root / "student_summary.json"
            schedule_path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "checkpoint_step": 20000,
                        "checkpoint_ids": ["forecast-ckpt"],
                        "schedules": [
                            {
                                "scheduler_key": GIPO_POLICY_KEY,
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            comparison_path = root / "comparison_rows.csv"
            fieldnames = (
                "benchmark_family",
                "split_phase",
                "scenario_key",
                "seed",
                "solver_key",
                "target_nfe",
                "scheduler_key",
                "checkpoint_step",
                "checkpoint_id",
                "forecast_crps",
                "forecast_mase",
            )
            with comparison_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for checkpoint_id in ("forecast-ckpt", "other-ckpt"):
                    writer.writerow(
                        {
                            "benchmark_family": "temporal_extrapolation",
                            "split_phase": "locked_test",
                            "scenario_key": "traffic_hourly",
                            "seed": 0,
                            "solver_key": "euler",
                            "target_nfe": 4,
                            "scheduler_key": BASELINE_SCHEDULE_KEYS[0],
                            "checkpoint_step": 20000,
                            "checkpoint_id": checkpoint_id,
                            "forecast_crps": 1.0,
                            "forecast_mase": 1.0,
                        }
                    )

            class FakeDataset:
                def __len__(self) -> int:
                    return 1

            checkpoint = {
                "model": object(),
                "cfg": object(),
                "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
                "checkpoint_path": root / "model.pt",
                "checkpoint_id": "forecast-ckpt",
                "backbone_name": "otflow",
                "checkpoint_step": 20000,
                "train_budget_label": "20k",
            }
            args = build_argparser().parse_args(
                [
                    "--scenario_key",
                    "traffic_hourly",
                    "--schedule_summary",
                    str(schedule_path),
                    "--split_phase",
                    "locked_test",
                    "--out_dir",
                    str(root / "out"),
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--baseline_rows",
                    str(comparison_path),
                    "--device",
                    "cpu",
                ]
            )
            with mock.patch(
                "genode.gipo.evaluate_schedule_summary.load_forecast_checkpoint_splits",
                return_value=checkpoint,
            ), mock.patch(
                "genode.gipo.evaluate_schedule_summary.evaluate_forecast_schedule",
            ) as evaluate_mock:
                with self.assertRaisesRegex(ValueError, "do not match the loaded backbone artifact"):
                    evaluate_schedule_summary(args)

            evaluate_mock.assert_not_called()

    def test_validation_schedule_selection_supports_arbitrary_candidate_keys(self) -> None:
        rows = []
        fixed_rows = []
        for schedule_key in BASELINE_SCHEDULE_KEYS:
            for seed in (0, 1, 2):
                fixed_rows.append(
                    {
                        "seed": seed,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "scheduler_key": schedule_key,
                        "forecast_crps": 1.0,
                        "forecast_mase": 2.0,
                        **_panel_fields(seed),
                    }
                )
        for schedule_key, budget, crps, mase in (
            ("gipo_candidate_steps20", 20, 1.0, 2.0),
            ("gipo_candidate_steps25", 25, 1.0, 2.0),
            ("gipo_candidate_steps35", 35, 0.9, 1.9),
            ("ser_ptg_residual_tail_s200_eps030", None, 1.2, 2.2),
        ):
            for seed in (0, 1, 2):
                row = {
                    "seed": seed,
                    "solver_key": "euler",
                    "target_nfe": 4,
                    "scheduler_key": schedule_key,
                    "forecast_crps": crps,
                    "forecast_mase": mase,
                    **_panel_fields(seed),
                }
                if budget is not None:
                    row["gipo_step_budget"] = budget
                rows.append(row)
        selection = select_best_validation_schedule(rows, reference_rows=fixed_rows)
        self.assertEqual(selection["selection_unit"], "generated_schedule_key")
        self.assertEqual(selection["selected_scheduler_key"], "gipo_candidate_steps35")
        self.assertEqual(selection["gipo_step_budget"], 35)
        self.assertEqual(selection["utility_reference"], "best_fixed_baseline_crps_mase")
        self.assertTrue(any(row["scheduler_key"] == "ser_ptg_residual_tail_s200_eps030" for row in selection["schedule_table"]))
        self.assertNotIn("eps_rho", selection["schedule_table"][0])
        self.assertNotIn("kl_weight", selection["schedule_table"][0])

    def test_validation_schedule_selection_rejects_mismatched_physical_panels(self) -> None:
        candidate_rows = [
            {
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "gipo_candidate",
                "forecast_crps": 1.0,
                "forecast_mase": 1.0,
                **_panel_fields(0, panel="candidate"),
            }
        ]
        fixed_rows = [
            {
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "forecast_crps": 2.0,
                "forecast_mase": 2.0,
                **_panel_fields(0, panel="fixed"),
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]

        with self.assertRaisesRegex(ValueError, "chosen-example/evaluation panels"):
            select_best_validation_schedule(candidate_rows, reference_rows=fixed_rows)

    def test_validation_schedule_selection_requires_sha256_panel_digests(self) -> None:
        candidate = {
            "seed": 0,
            "solver_key": "euler",
            "target_nfe": 4,
            "scheduler_key": "gipo_candidate",
            "forecast_crps": 1.0,
            "forecast_mase": 1.0,
            **_panel_fields(0),
        }
        fixed_rows = [
            {
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "forecast_crps": 2.0,
                "forecast_mase": 2.0,
                **_panel_fields(0),
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]
        for field, value in (
            ("chosen_examples_hash", "not-a-digest"),
            ("evaluation_protocol_hash", "A" * 64),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError,
                f"{field} must be a lowercase SHA-256 digest",
            ):
                select_best_validation_schedule(
                    [{**candidate, field: value}],
                    reference_rows=fixed_rows,
                )

    def test_validation_schedule_selection_allows_schedule_specific_protocol_hashes(self) -> None:
        candidate_rows = [
            {
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": "gipo_candidate",
                "forecast_crps": 1.0,
                "forecast_mase": 1.0,
                **_panel_fields(0, protocol="candidate"),
            }
        ]
        fixed_rows = [
            {
                "seed": 0,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "forecast_crps": 2.0,
                "forecast_mase": 2.0,
                **_panel_fields(0, protocol=f"fixed-{schedule_key}"),
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
        ]

        selection = select_best_validation_schedule(
            candidate_rows,
            reference_rows=fixed_rows,
        )

        self.assertEqual(selection["selected_scheduler_key"], "gipo_candidate")

    def test_validation_schedule_selection_tie_breaks_smaller_budget(self) -> None:
        rows = []
        fixed_rows = [
            {
                "seed": seed,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "forecast_crps": 2.0,
                "forecast_mase": 3.0,
                **_panel_fields(seed),
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
            for seed in (0, 1, 2)
        ]
        for schedule_key, budget in (("gipo_candidate_steps20", 20), ("gipo_candidate_steps25", 25)):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "seed": seed,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "scheduler_key": schedule_key,
                        "gipo_step_budget": budget,
                        "forecast_crps": 1.0,
                        "forecast_mase": 2.0,
                        **_panel_fields(seed),
                    }
                )
        selection = select_best_validation_schedule(rows, reference_rows=fixed_rows)
        self.assertEqual(selection["selected_scheduler_key"], "gipo_candidate_steps20")
        self.assertEqual(
            selection["tie_break"],
            "mean_validation_utility_then_mean_min_metric_utility_then_smaller_gipo_step_budget_then_scheduler_key",
        )
        self.assertIn("mean_min_metric_utility", selection["schedule_table"][0])

    def test_validation_schedule_selection_tie_breaks_worst_metric_before_budget(self) -> None:
        rows = []
        fixed_rows = [
            {
                "seed": seed,
                "solver_key": "euler",
                "target_nfe": 4,
                "scheduler_key": schedule_key,
                "forecast_crps": 1.0,
                "forecast_mase": 1.0,
                **_panel_fields(seed),
            }
            for schedule_key in BASELINE_SCHEDULE_KEYS
            for seed in (0, 1, 2)
        ]
        # Same composite utility relative to best fixed: 0.5*(+log 2 - log 2) == 0.
        # The 25-step schedule has a better worst-metric utility and should win
        # before the smaller-budget tie-break is considered.
        for schedule_key, budget, crps, mase in (
            ("gipo_candidate_steps20", 20, 0.5, 2.0),
            ("gipo_candidate_steps25", 25, 0.8, 1.25),
        ):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "seed": seed,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "scheduler_key": schedule_key,
                        "gipo_step_budget": budget,
                        "forecast_crps": crps,
                        "forecast_mase": mase,
                        **_panel_fields(seed),
                    }
                )
        selection = select_best_validation_schedule(rows, reference_rows=fixed_rows)
        self.assertEqual(selection["selected_scheduler_key"], "gipo_candidate_steps25")
        self.assertGreater(selection["schedule_table"][0]["mean_min_metric_utility"], selection["schedule_table"][1]["mean_min_metric_utility"])

    def test_validation_schedule_selection_requires_fixed_reference_rows(self) -> None:
        rows = []
        for schedule_key, budget, crps, mase in (
            ("gipo_candidate_steps20", 20, 1.0, 2.0),
            ("gipo_candidate_steps25", 25, 0.9, 1.9),
        ):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "seed": seed,
                        "solver_key": "euler",
                        "target_nfe": 4,
                        "scheduler_key": schedule_key,
                        "gipo_step_budget": budget,
                        "forecast_crps": crps,
                        "forecast_mase": mase,
                    }
                )
        with self.assertRaisesRegex(ValueError, "fixed baseline reference rows"):
            select_best_validation_schedule(rows)

    def test_budget_only_validation_cli_is_removed(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_argparser().parse_args(
                [
                    "--scenario_key",
                    "traffic_hourly",
                    "--schedule_summary",
                    "summary.json",
                    "--split_phase",
                    "validation_tuning",
                    "--select_budget_from_validation",
                ]
            )

    def test_locked_test_preview_is_explicit_and_defaults_to_512(self) -> None:
        base = [
            "--schedule_summary",
            "summary.json",
            "--split_phase",
            "locked_test",
        ]
        full_args = build_argparser().parse_args(base)
        self.assertEqual(_split_example_cap(full_args, "locked_test"), (None, "locked_test_full"))
        preview_args = build_argparser().parse_args([*base, "--locked_test_preview"])
        self.assertEqual(
            _split_example_cap(preview_args, "locked_test"),
            (512, "locked_test_preview_contexts"),
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_argparser().parse_args([*base, "--eval_windows_test", "1"])

    def test_evaluate_schedule_summary_writes_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            schedule_path = root / "student_summary.json"
            schedule_path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "checkpoint_step": 20000,
                        "checkpoint_ids": ["ck"],
                        "schedules": [
                            {
                                "scheduler_key": GIPO_POLICY_KEY,
                                "schedule_name": "GIPO Student Selected",
                                "gipo_step_budget": 25,
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            class FakeDataset:
                def __len__(self) -> int:
                    return 3

            fake_checkpoint = {
                "model": object(),
                "cfg": object(),
                "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
                "checkpoint_path": root / "model.pt",
                "checkpoint_id": "ck",
                "backbone_name": "otflow",
                "checkpoint_step": 20000,
                "train_budget_label": "20k",
            }
            args = build_argparser().parse_args(
                [
                    "--scenario_key",
                    "traffic_hourly",
                    "--schedule_summary",
                    str(schedule_path),
                    "--split_phase",
                    "validation_tuning",
                    "--out_dir",
                    str(root / "out"),
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--num_eval_samples",
                    "1",
                    "--eval_windows_val",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            with mock.patch(
                "genode.gipo.evaluate_schedule_summary.load_forecast_checkpoint_splits",
                return_value=fake_checkpoint,
            ), mock.patch(
                "genode.gipo.evaluate_schedule_summary.evaluate_forecast_schedule",
                return_value={
                    "forecast_crps": 1.0,
                    "forecast_mse": 1.5,
                    "forecast_mase": 2.0,
                    "latency_ms_per_sample": 0.25,
                    "num_eval_samples": 1,
                    "eval_examples": 1,
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": 4,
                },
            ):
                summary = evaluate_schedule_summary(args)
            self.assertEqual(summary["observed_rows"], 1)
            with (root / "out" / "validation_rows.csv").open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scheduler_key"], GIPO_POLICY_KEY)
            self.assertEqual(int(rows[0]["realized_nfe"]), 4)
            self.assertFalse(Path(summary["row_csv"]).is_absolute())
            self.assertNotIn(str(root), summary["row_csv"])

    def test_evaluate_schedule_summary_validation_defaults_to_context_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            schedule_path = root / "student_summary.json"
            schedule_path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "checkpoint_step": 20000,
                        "checkpoint_ids": ["ck"],
                        "schedules": [
                            {
                                "scheduler_key": GIPO_POLICY_KEY,
                                "gipo_step_budget": 25,
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            class FakeDataset:
                def __len__(self) -> int:
                    return 1000

            fake_checkpoint = {
                "model": object(),
                "cfg": object(),
                "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
                "checkpoint_path": root / "model.pt",
                "checkpoint_id": "ck",
                "backbone_name": "otflow",
                "checkpoint_step": 20000,
                "train_budget_label": "20k",
            }
            captured_lengths = []

            def fake_eval(*args, **kwargs):
                del args
                captured_lengths.append(len(kwargs["example_indices"]))
                return {
                    "forecast_crps": 1.0,
                    "forecast_mse": 1.5,
                    "forecast_mase": 2.0,
                    "latency_ms_per_sample": 0.25,
                    "num_eval_samples": 1,
                    "eval_examples": len(kwargs["example_indices"]),
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": 4,
                }

            args = build_argparser().parse_args(
                [
                    "--scenario_key",
                    "traffic_hourly",
                    "--schedule_summary",
                    str(schedule_path),
                    "--split_phase",
                    "validation_tuning",
                    "--out_dir",
                    str(root / "out"),
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--num_eval_samples",
                    "1",
                    "--context_sample_count",
                    "9",
                    "--device",
                    "cpu",
                ]
            )
            with mock.patch(
                "genode.gipo.evaluate_schedule_summary.load_forecast_checkpoint_splits",
                return_value=fake_checkpoint,
            ), mock.patch(
                "genode.gipo.evaluate_schedule_summary.evaluate_forecast_schedule",
                side_effect=fake_eval,
            ):
                evaluate_schedule_summary(args)

            self.assertEqual(captured_lengths, [9])
            with (root / "out" / "validation_rows.csv").open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["selected_examples"], "9")
            self.assertEqual(rows[0]["selected_examples_cap_source"], "context_sample_count")

    def test_evaluate_schedule_summary_locked_test_defaults_to_full_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            schedule_path = root / "student_summary.json"
            schedule_path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "checkpoint_step": 20000,
                        "checkpoint_ids": ["ck"],
                        "schedules": [
                            {
                                "scheduler_key": GIPO_POLICY_KEY,
                                "gipo_step_budget": 25,
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            class FakeDataset:
                def __len__(self) -> int:
                    return 1000

            fake_checkpoint = {
                "model": object(),
                "cfg": object(),
                "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
                "checkpoint_path": root / "model.pt",
                "checkpoint_id": "ck",
                "backbone_name": "otflow",
                "checkpoint_step": 20000,
                "train_budget_label": "20k",
            }
            captured_lengths = []

            def fake_eval(*args, **kwargs):
                del args
                captured_lengths.append(len(kwargs["example_indices"]))
                context_indices = [int(value) for value in kwargs["example_indices"]]
                return {
                    "forecast_crps": 1.0,
                    "forecast_mse": 1.5,
                    "forecast_mase": 2.0,
                    "latency_ms_per_sample": 0.25,
                    "num_eval_samples": 1,
                    "eval_examples": len(kwargs["example_indices"]),
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": 4,
                    "per_example_rows": [
                        {
                            "row_signature": f"context-row-{index}",
                            "context_id": f"context-{index}",
                            "context_embedding_id": f"ck:context-{index}",
                            "scenario_key": "traffic_hourly",
                            "split_phase": "locked_test",
                            "seed": 0,
                            "logical_seed": 0,
                            "evaluation_seed": 0,
                            "solver_key": "euler",
                            "target_nfe": 4,
                            "realized_nfe": 4,
                            "scheduler_key": GIPO_POLICY_KEY,
                            "example_idx": index,
                            "forecast_crps": 1.0,
                            "forecast_mase": 2.0,
                            "forecast_mse": 1.5,
                        }
                        for index in context_indices
                    ],
                    "context_embeddings": {
                        f"ck:context-{index}": [0.1, 0.2]
                        for index in context_indices
                    },
                }

            args = build_argparser().parse_args(
                [
                    "--scenario_key",
                    "traffic_hourly",
                    "--schedule_summary",
                    str(schedule_path),
                    "--split_phase",
                    "locked_test",
                    "--out_dir",
                    str(root / "out"),
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--num_eval_samples",
                    "1",
                    "--context_sample_count",
                    "9",
                    "--write_context_rows",
                    "--device",
                    "cpu",
                ]
            )
            with mock.patch(
                "genode.gipo.evaluate_schedule_summary.load_forecast_checkpoint_splits",
                return_value=fake_checkpoint,
            ), mock.patch(
                "genode.gipo.evaluate_schedule_summary.evaluate_forecast_schedule",
                side_effect=fake_eval,
            ):
                summary = evaluate_schedule_summary(args)

            self.assertEqual(captured_lengths, [1000])
            with (root / "out" / "test_rows.csv").open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["selected_examples"], "1000")
            self.assertEqual(rows[0]["selected_examples_cap"], "1000")
            self.assertEqual(rows[0]["uncapped_candidate_examples"], "1000")
            self.assertEqual(rows[0]["selected_examples_cap_source"], "locked_test_full")
            self.assertEqual(rows[0]["selection_was_capped"], "False")
            self.assertEqual(rows[0]["global_selection_was_capped"], "False")
            self.assertEqual(rows[0]["locked_test_mode"], "full")
            self.assertEqual(rows[0]["locked_test_context_limit_scope"], "none")
            self.assertEqual(rows[0]["checkpoint_step"], "20000")
            self.assertEqual(summary["locked_test_mode"], "full")
            self.assertIsNone(summary["locked_test_context_limit"])
            self.assertEqual(summary["checkpoint_step"], 20000)
            with (root / "out" / "context_test_rows.csv").open("r", newline="", encoding="utf-8") as fh:
                context_rows = list(csv.DictReader(fh))
            self.assertEqual(context_rows[0]["scenario_key"], "traffic_hourly")
            self.assertEqual(context_rows[0]["checkpoint_step"], "20000")
            self.assertEqual(context_rows[0]["checkpoint_id"], "ck")
            self.assertEqual(context_rows[0]["gipo_step_budget"], "25")
            self.assertEqual(context_rows[0]["locked_test_mode"], "full")
            self.assertEqual(context_rows[0]["selected_examples_cap_source"], "locked_test_full")
            self.assertEqual(context_rows[0]["global_selection_was_capped"], "False")

    def test_evaluate_schedule_summary_writes_train_tuning_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            schedule_path = root / "student_summary.json"
            schedule_path.write_text(
                json.dumps(
                    {
                        "scenario_key": "traffic_hourly",
                        "checkpoint_step": 20000,
                        "checkpoint_ids": ["ck"],
                        "schedules": [
                            {
                                "scheduler_key": GIPO_POLICY_KEY,
                                "predictions": [
                                    {
                                        "solver_key": "euler",
                                        "target_nfe": 4,
                                        "macro_steps": 4,
                                        "time_grid": _uniform_grid(4),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            class FakeDataset:
                def __len__(self) -> int:
                    return 100

            fake_checkpoint = {
                "model": object(),
                "cfg": object(),
                "splits": {"train": FakeDataset(), "val": FakeDataset(), "test": FakeDataset()},
                "checkpoint_path": root / "model.pt",
                "checkpoint_id": "ck",
                "backbone_name": "otflow",
                "checkpoint_step": 20000,
                "train_budget_label": "20k",
            }
            captured_lengths = []

            def fake_eval(*args, **kwargs):
                del args
                captured_lengths.append(len(kwargs["example_indices"]))
                return {
                    "forecast_crps": 1.0,
                    "forecast_mse": 1.5,
                    "forecast_mase": 2.0,
                    "latency_ms_per_sample": 0.25,
                    "num_eval_samples": 1,
                    "eval_examples": len(kwargs["example_indices"]),
                    "eval_horizon": 168,
                    "evaluation_protocol_hash": "protocol",
                    "chosen_examples_hash": "examples",
                    "realized_nfe": 4,
                }

            args = build_argparser().parse_args(
                [
                    "--scenario_key",
                    "traffic_hourly",
                    "--schedule_summary",
                    str(schedule_path),
                    "--split_phase",
                    "train_tuning",
                    "--out_dir",
                    str(root / "out"),
                    "--solver_names",
                    "euler",
                    "--target_nfe_values",
                    "4",
                    "--seeds",
                    "0",
                    "--num_eval_samples",
                    "1",
                    "--eval_train_fraction",
                    "1.0",
                    "--context_sample_count",
                    "7",
                    "--train_tuning_strata",
                    "20",
                    "--device",
                    "cpu",
                ]
            )
            with mock.patch(
                "genode.gipo.evaluate_schedule_summary.load_forecast_checkpoint_splits",
                return_value=fake_checkpoint,
            ), mock.patch(
                "genode.gipo.evaluate_schedule_summary.evaluate_forecast_schedule",
                side_effect=fake_eval,
            ):
                summary = evaluate_schedule_summary(args)
            self.assertEqual(summary["split_phase"], "train_tuning")
            self.assertEqual(summary["train_tuning"]["fraction"], 1.0)
            self.assertEqual(summary["train_tuning"]["max_examples"], 7)
            self.assertEqual(captured_lengths, [7])
            with (root / "out" / "train_tuning_rows.csv").open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["split_phase"], "train_tuning")
            self.assertEqual(rows[0]["train_tuning_sampler"], "temporal_stratified_hash")
            self.assertEqual(rows[0]["train_tuning_sampling_mode"], "train_window_fraction")
            self.assertEqual(rows[0]["train_tuning_target_examples"], "7")
            self.assertEqual(rows[0]["selected_examples_cap_source"], "context_sample_count")


if __name__ == "__main__":
    unittest.main()
