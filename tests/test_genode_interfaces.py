from __future__ import annotations

import importlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tomllib

from genode.runtime import resolve_torch_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tracked_release_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "ls-files", "--cached", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        path
        for relative_path in result.stdout.split("\0")
        if relative_path
        for path in [PROJECT_ROOT / relative_path]
        if path.is_file()
    ]


class GenODEInterfaceTests(unittest.TestCase):
    def test_import_resolves_to_standalone_package(self) -> None:
        module = importlib.import_module("genode")
        module_path = Path(module.__file__).resolve()
        self.assertEqual(module.__name__, "genode")
        self.assertEqual(module_path.name, "__init__.py")
        self.assertEqual(module_path.parent.name, "genode")

    def test_public_entry_points_are_registered(self) -> None:
        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = data["project"]["scripts"]
        expected = {
            "genode-train-backbone",
            "genode-run-schedules",
            "genode-run-full-pipeline",
            "genode-prepare-molecule-xyz",
            "genode-train-molecule-backbone",
            "genode-evaluate-molecule-backbone",
            "genode-train-gipo",
            "genode-preflight-gipo-rows",
            "genode-report-gipo-locked-test",
            "genode-build-ser-ptg-reference",
            "genode-evaluate-schedule-summary",
            "genode-build-hardness-figure",
            "genode-build-ptg-figure",
            "genode-package-backbone-family",
            "genode-validate-backbone-package",
            "genode-collect-flow-map-demonstrations",
            "genode-train-flow-map",
            "genode-evaluate-flow-map",
        }
        self.assertEqual(set(scripts), expected)
        for target in scripts.values():
            module_name, func_name = str(target).split(":", 1)
            self.assertTrue(callable(getattr(importlib.import_module(module_name), func_name)))

    def test_readme_documents_current_interfaces(self) -> None:
        text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# GenODE Inference\n"))
        self.assertIn("--include_flow_map", text)
        self.assertIn("genode-train-gipo", text)
        self.assertIn("genode-collect-flow-map-demonstrations", text)
        self.assertIn("genode-train-flow-map", text)
        self.assertIn("genode-evaluate-flow-map", text)
        self.assertIn('quality_gate.status="not_evaluated"', text)
        self.assertIn("no claim", text.lower())
        self.assertNotIn("paper_gipo", text)
        self.assertNotIn("pseudo_gipo", text)

    def test_shared_csv_parsers_normalize_cli_values(self) -> None:
        from genode.cli import parse_csv, parse_int_csv

        self.assertEqual(parse_csv(" alpha, , beta "), ["alpha", "beta"])
        self.assertEqual(parse_int_csv("4, 8,12"), [4, 8, 12])
        self.assertEqual(parse_int_csv("", default=(6, 10)), [6, 10])

    def test_gipo_trainer_public_contract_matches_required_api(self) -> None:
        from genode.gipo.train_gipo import build_argparser

        parser = build_argparser()
        options = {option for action in parser._actions for option in action.option_strings}

        self.assertIn("--teacher_unseen_selection_rows_csv", options)
        self.assertIn("--student_unseen_target_rows_csv", options)
        self.assertIn("--student_unseen_target_" + "weight", options)
        self.assertIn("--student_teacher_score_" + "weight", options)
        self.assertIn("--student_teacher_score_warmup_" + "fraction", options)
        self.assertIn("--student_teacher_score_include_" + "unseen_targets", options)
        self.assertIn("--student_target_mixture_" + "mode", options)
        self.assertIn("--student_target_elite_" + "fraction", options)
        self.assertIn("--student_target_elite_" + "k", options)
        self.assertIn("--student_target_elite_min_" + "count", options)
        self.assertIn("--student_target_elite_blend_all_" + "weight", options)
        removed_options = {
            "--gipo_" + "conditioning_style",
            "--gipo_teacher_" + "conditioning_style",
            "--gipo_student_" + "conditioning_style",
            "--density_bin_count",
            "--teacher_checkpoint_" + "selection_mode",
            "--student_checkpoint_" + "selection",
            "--teacher_selection_component_" + "weights",
            "--teacher_nfe_" + "proxy_anchor_values",
            "--teacher_fit_checkpoint_" + "selection",
            "--series_holdout_" + "fraction",
            "--teacher_architecture",
            "--student_architecture",
            "--setting_encoder_mode",
            "--setting_feature_mode",
            "--series_unknown_" + "dropout",
            "--student_pseudo_rows_csv",
            "--student_pseudo_target_weight",
            "--student_teacher_score_include_pseudo",
        }
        self.assertFalse(removed_options & options)

    def test_full_pipeline_public_contract_keeps_ablations_and_preview_explicit(self) -> None:
        from genode.distillation.evaluation import (
            build_argparser as build_flow_map_evaluation_parser,
        )
        from genode.pipeline.full_pipeline import build_argparser

        parser = build_argparser()
        options = {option for action in parser._actions for option in action.option_strings}
        help_text = parser.format_help()

        self.assertIn("--include_ablations", options)
        self.assertIn("--include_flow_map", options)
        self.assertIn("--ablation_preset", options)
        self.assertIn("--locked_test_preview", options)
        self.assertIn("--locked_test_preview_contexts", options)
        self.assertIn("--ser_calibration_batch_size", options)
        self.assertIn("--ser_val_windows", options)
        self.assertIn("--ser_train_tuning_max_examples", options)
        self.assertIn("--flow_map_backbone_checkpoint", options)
        self.assertIn("--flow_map_checkpoint", options)
        self.assertIn("--flow_map_contexts_npz", options)
        self.assertIn("--flow_map_quality_rows_csv", options)
        self.assertIn("--flow_map_quality_candidate_catalog", options)
        self.assertIn("--flow_map_quality_contexts_npz", options)
        self.assertIn("--flow_map_quality_sample_panel_npz", options)
        self.assertIn("--flow_map_quality_measurement_protocol", options)
        self.assertIn("GIPO reference workflow", help_text)
        self.assertIn("opt-in ablation grid", help_text)
        evaluation_options = {
            option
            for action in build_flow_map_evaluation_parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--measurement-protocol-json", evaluation_options)

    def test_flow_map_pipeline_is_opt_in_and_code_only_by_default(self) -> None:
        from genode.pipeline import full_pipeline

        parser = full_pipeline.build_argparser()
        default_args = parser.parse_args(["--scenario_key", "solar_energy_10m", "--dry_run"])
        self.assertFalse(
            set(full_pipeline._selected_stage_names(default_args))
            & set(full_pipeline.FLOW_MAP_PIPELINE_STAGES)
        )
        args = parser.parse_args(
            [
                "--scenario_key",
                "solar_energy_10m",
                "--include_flow_map",
                "--flow_map_backbone_checkpoint",
                "backbone.pt",
                "--flow_map_contexts_npz",
                "contexts.npz",
                "--dry_run",
            ]
        )
        self.assertEqual(
            full_pipeline._selected_stage_names(args)[-3:],
            list(full_pipeline.FLOW_MAP_PIPELINE_STAGES),
        )
        protocol = full_pipeline._protocol_payload(args)
        self.assertEqual(protocol["flow_map"]["quality_status"], "not_evaluated")
        self.assertFalse(protocol["flow_map"]["performance_claim"])
        self.assertEqual(protocol["flow_map"]["seed"], 0)

        commands = full_pipeline._build_stage_commands(args, PROJECT_ROOT / "outputs" / "dry_run")
        by_stage = {entry.stage: entry for entry in commands}
        collection = by_stage[full_pipeline.FLOW_MAP_COLLECTION_STAGE].commands[0]
        evaluation = by_stage[full_pipeline.FLOW_MAP_EVALUATION_STAGE].commands[0]
        self.assertIn("genode.distillation.demonstrations", collection)
        self.assertEqual(collection[collection.index("--split-phase") + 1], "train_tuning")
        self.assertIn("--not-evaluated-reason", evaluation)

    def test_flow_map_quality_arguments_require_standalone_evaluation(self) -> None:
        from genode.pipeline import full_pipeline

        parser = full_pipeline.build_argparser()
        unused = parser.parse_args(
            [
                "--scenario_key",
                "solar_energy_10m",
                "--flow_map_quality_rows_csv",
                "rows.csv",
                "--dry_run",
            ]
        )
        with self.assertRaisesRegex(ValueError, "require the evaluate_flow_map stage"):
            full_pipeline._validate_inputs_preflight(unused)

        combined = parser.parse_args(
            [
                "--scenario_key",
                "solar_energy_10m",
                "--include_flow_map",
                "--flow_map_backbone_checkpoint",
                "backbone.pt",
                "--flow_map_contexts_npz",
                "contexts.npz",
                "--flow_map_quality_rows_csv",
                "rows.csv",
                "--flow_map_quality_candidate_catalog",
                "catalog.json",
                "--flow_map_quality_contexts_npz",
                "quality-contexts.npz",
                "--flow_map_quality_sample_panel_npz",
                "quality-panel.npz",
                "--flow_map_quality_measurement_protocol",
                "measurement-protocol.json",
                "--dry_run",
            ]
        )
        with self.assertRaisesRegex(ValueError, "separate evaluation run"):
            full_pipeline._validate_inputs_preflight(combined)

        missing_rows = parser.parse_args(
            [
                "--scenario_key",
                "solar_energy_10m",
                "--stages",
                full_pipeline.FLOW_MAP_EVALUATION_STAGE,
                "--flow_map_backbone_checkpoint",
                "backbone.pt",
                "--flow_map_gipo_checkpoint",
                "gipo.pt",
                "--flow_map_checkpoint",
                "flow-map.pt",
                "--flow_map_quality_candidate_catalog",
                "catalog.json",
                "--dry_run",
            ]
        )
        with self.assertRaisesRegex(
            ValueError,
            "--flow_map_quality_rows_csv is required",
        ):
            full_pipeline._validate_inputs_preflight(missing_rows)

        missing_measurement_protocol = parser.parse_args(
            [
                "--scenario_key",
                "solar_energy_10m",
                "--stages",
                full_pipeline.FLOW_MAP_EVALUATION_STAGE,
                "--flow_map_backbone_checkpoint",
                "backbone.pt",
                "--flow_map_gipo_checkpoint",
                "gipo.pt",
                "--flow_map_checkpoint",
                "flow-map.pt",
                "--flow_map_quality_rows_csv",
                "rows.csv",
                "--flow_map_quality_candidate_catalog",
                "catalog.json",
                "--flow_map_quality_contexts_npz",
                "quality-contexts.npz",
                "--flow_map_quality_sample_panel_npz",
                "quality-panel.npz",
                "--dry_run",
            ]
        )
        with self.assertRaisesRegex(
            ValueError,
            "--flow_map_quality_measurement_protocol is required",
        ):
            full_pipeline._validate_inputs_preflight(
                missing_measurement_protocol
            )

    def test_display_command_sanitizes_every_comma_separated_path(self) -> None:
        from genode.pipeline import full_pipeline

        root = Path.cwd().resolve()
        first = root / "artifacts" / "first.json"
        second = root / "artifacts" / "second.json"
        displayed = full_pipeline._display_command(
            ["--schedule_summary_json", f"{first},{second}"],
            path_base=root,
        )

        self.assertEqual(
            displayed[1],
            "artifacts/first.json,artifacts/second.json",
        )

    def test_flow_map_evaluation_only_requires_and_uses_explicit_checkpoint(self) -> None:
        from genode.pipeline import full_pipeline

        parser = full_pipeline.build_argparser()
        missing = parser.parse_args(
            [
                "--scenario_key",
                "solar_energy_10m",
                "--stages",
                full_pipeline.FLOW_MAP_EVALUATION_STAGE,
                "--flow_map_backbone_checkpoint",
                "backbone.pt",
                "--flow_map_gipo_checkpoint",
                "gipo.pt",
                "--dry_run",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--flow_map_checkpoint"):
            full_pipeline._validate_inputs_preflight(missing)

        args = parser.parse_args(
            [
                "--scenario_key",
                "solar_energy_10m",
                "--stages",
                full_pipeline.FLOW_MAP_EVALUATION_STAGE,
                "--flow_map_backbone_checkpoint",
                "backbone.pt",
                "--flow_map_gipo_checkpoint",
                "gipo.pt",
                "--flow_map_checkpoint",
                "endpoint-map.pt",
                "--dry_run",
            ]
        )
        full_pipeline._validate_inputs_preflight(args)
        command = full_pipeline._build_stage_commands(
            args,
            PROJECT_ROOT / "outputs" / "dry_run",
        )[0].commands[0]
        assert command[command.index("--flow-map-checkpoint") + 1].endswith(
            "endpoint-map.pt"
        )

    def test_flow_map_quality_protocol_binds_inputs_and_hashes_catalog_bytes(self) -> None:
        from genode.distillation.evaluation import (
            _time_grid_sha256,
            candidate_catalog_sha256,
            metric_specs_for_scenario,
            read_candidate_catalog,
        )
        from genode.distillation.measurement_protocol import (
            measurement_protocol_sha256,
            quality_measurement_protocol_payload,
        )
        from genode.pipeline import full_pipeline
        from genode.provenance import file_sha256
        from genode.schedule_transfer.diffusion_flow_schedules import (
            build_schedule_grid,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()
            quality_root = project_root / "quality"
            quality_root.mkdir()
            rows_path = quality_root / "rows.csv"
            rows_path.write_text("split_phase,method\n", encoding="utf-8")
            catalog_path = quality_root / "candidate_catalog.json"
            contexts_path = quality_root / "contexts.npz"
            contexts_path.write_bytes(b"quality-contexts")
            sample_panel_path = quality_root / "sample_panel.npz"
            sample_panel_path.write_bytes(b"quality-sample-panel")
            artifact_root = project_root / "artifacts"
            artifact_root.mkdir()
            backbone_path = artifact_root / "backbone.pt"
            gipo_path = artifact_root / "gipo.pt"
            flow_map_path = artifact_root / "flow_map.pt"
            backbone_path.write_bytes(b"backbone")
            gipo_path.write_bytes(b"gipo")
            flow_map_path.write_bytes(b"flow-map")
            candidates = []
            for suffix, target_nfe in (("low", 4), ("high", 6)):
                candidates.extend(
                    [
                        {
                            "method": "flow_map",
                            "candidate_key": f"flow_map_{suffix}",
                            "solver_key": "euler",
                            "target_nfe": target_nfe,
                            "execution": {
                                "kind": "endpoint_flow_map",
                                "density_source": "bound_gipo_checkpoint",
                            },
                        },
                        {
                            "method": "gipo",
                            "candidate_key": f"gipo_{suffix}",
                            "solver_key": "euler",
                            "target_nfe": target_nfe,
                            "execution": {
                                "kind": "gipo_ode_rollout",
                                "policy_sha256": file_sha256(gipo_path),
                            },
                        },
                    ]
                )
            for suffix, target_nfe, scheduler_key in (
                ("uniform", 4, "uniform"),
                ("late", 6, "late_power_3"),
            ):
                time_grid = build_schedule_grid(scheduler_key, target_nfe)
                assert time_grid is not None
                candidates.append(
                    {
                        "method": "fixed",
                        "candidate_key": f"fixed_{suffix}",
                        "solver_key": "euler",
                        "target_nfe": target_nfe,
                        "execution": {
                            "kind": "fixed_time_grid",
                            "scheduler_key": scheduler_key,
                            "density_source_key": scheduler_key,
                            "time_grid": list(time_grid),
                            "time_grid_sha256": _time_grid_sha256(time_grid),
                        },
                    }
                )
            catalog_path.write_text(json.dumps(candidates), encoding="utf-8")
            catalog_hash = candidate_catalog_sha256(
                read_candidate_catalog(catalog_path)
            )
            metric_payloads = [
                {
                    "name": spec.name,
                    "direction": spec.direction,
                    "weight": float(spec.weight),
                    "applicable_key": spec.applicable_key,
                }
                for spec in metric_specs_for_scenario("solar_energy_10m")
            ]
            measurement_path = quality_root / "measurement_protocol.json"
            measurement_payload = quality_measurement_protocol_payload(
                scenario_key="solar_energy_10m",
                candidate_catalog_sha256=catalog_hash,
                quality_contexts_sha256=file_sha256(contexts_path),
                quality_sample_panel_sha256=file_sha256(sample_panel_path),
                reference_data_sha256="f" * 64,
                artifact_binding={
                    "flow_map_checkpoint_sha256": file_sha256(flow_map_path),
                    "backbone_checkpoint_sha256": file_sha256(backbone_path),
                    "gipo_checkpoint_sha256": file_sha256(gipo_path),
                },
                primary_metrics=metric_payloads,
                runner={
                    "name": "test-runner",
                    "release": "test-release",
                    "implementation_sha256": "d" * 64,
                    "environment_sha256": "e" * 64,
                },
            )
            measurement_path.write_text(
                json.dumps(measurement_payload), encoding="utf-8"
            )
            initial_measurement_hash = measurement_protocol_sha256(
                measurement_payload
            )
            run_root = project_root / "outputs" / "quality_run"
            with mock.patch.dict(
                os.environ,
                {"GENODE_PROJECT_ROOT": str(project_root)},
            ):
                args = full_pipeline.build_argparser().parse_args(
                    [
                        "--scenario_key",
                        "solar_energy_10m",
                        "--run_root",
                        str(run_root),
                        "--stages",
                        full_pipeline.FLOW_MAP_EVALUATION_STAGE,
                        "--flow_map_backbone_checkpoint",
                        "artifacts/backbone.pt",
                        "--flow_map_gipo_checkpoint",
                        "artifacts/gipo.pt",
                        "--flow_map_checkpoint",
                        "artifacts/flow_map.pt",
                        "--flow_map_quality_rows_csv",
                        "quality/rows.csv",
                        "--flow_map_quality_candidate_catalog",
                        "quality/candidate_catalog.json",
                        "--flow_map_quality_contexts_npz",
                        "quality/contexts.npz",
                        "--flow_map_quality_sample_panel_npz",
                        "quality/sample_panel.npz",
                        "--flow_map_quality_measurement_protocol",
                        "quality/measurement_protocol.json",
                        "--dry_run",
                    ]
                )
                protocol_before = full_pipeline._protocol_payload(args)
                command_before = full_pipeline._build_stage_commands(
                    args,
                    run_root,
                )[0].commands[0]
                command_hash_before = full_pipeline._command_hash(command_before)

                catalog_path.write_text(
                    json.dumps(candidates, indent=2),
                    encoding="utf-8",
                )
                protocol_after = full_pipeline._protocol_payload(args)
                command_after = full_pipeline._build_stage_commands(
                    args,
                    run_root,
                )[0].commands[0]
                command_hash_after = full_pipeline._command_hash(command_after)

                measurement_payload["runner"]["release"] = "updated-release"
                measurement_path.write_text(
                    json.dumps(measurement_payload), encoding="utf-8"
                )
                protocol_with_updated_measurement = (
                    full_pipeline._protocol_payload(args)
                )
                command_with_updated_measurement = (
                    full_pipeline._build_stage_commands(args, run_root)[0].commands[0]
                )

            flow_map_protocol = protocol_after["flow_map"]
            self.assertEqual(
                flow_map_protocol["quality_candidate_catalog_sha256"],
                candidate_catalog_sha256(read_candidate_catalog(catalog_path)),
            )
            self.assertEqual(
                flow_map_protocol["quality_rows_sha256"],
                file_sha256(rows_path),
            )
            self.assertEqual(
                flow_map_protocol["quality_contexts_sha256"],
                file_sha256(contexts_path),
            )
            self.assertEqual(
                flow_map_protocol["quality_sample_panel_sha256"],
                file_sha256(sample_panel_path),
            )
            self.assertEqual(
                flow_map_protocol["quality_measurement_protocol_sha256"],
                initial_measurement_hash,
            )
            self.assertEqual(
                protocol_before["flow_map"]["quality_candidate_catalog_sha256"],
                flow_map_protocol["quality_candidate_catalog_sha256"],
            )
            self.assertNotEqual(command_hash_before, command_hash_after)
            self.assertNotEqual(
                flow_map_protocol["quality_measurement_protocol_sha256"],
                protocol_with_updated_measurement["flow_map"][
                    "quality_measurement_protocol_sha256"
                ],
            )
            self.assertNotEqual(
                command_hash_after,
                full_pipeline._command_hash(command_with_updated_measurement),
            )
            self.assertEqual(
                command_after[command_after.index("--quality-protocol-json") + 1],
                str(run_root / "protocol.json"),
            )
            for option in (
                "--rows-csv",
                "--candidate-catalog",
                "--quality-contexts-npz",
                "--quality-sample-panel-npz",
                "--measurement-protocol-json",
                "--flow-map-checkpoint",
                "--backbone-checkpoint",
                "--gipo-checkpoint",
                "--quality-protocol-json",
            ):
                resolved = Path(command_after[command_after.index(option) + 1])
                self.assertTrue(resolved.is_absolute())
                self.assertTrue(resolved.is_relative_to(project_root))

    def test_project_path_resolver_does_not_rewrite_package_prefixes(self) -> None:
        from genode.data import otflow_paths

        with mock.patch.object(otflow_paths, "project_root", return_value=PROJECT_ROOT):
            self.assertEqual(
                otflow_paths.resolve_project_path("genode/outputs/example"),
                (PROJECT_ROOT / "genode" / "outputs" / "example").resolve(),
            )

    def test_gipo_policy_public_surface_excludes_teacher_prediction_helper(self) -> None:
        from genode.gipo import policy

        helper_name = "build_teacher_weighted_density_" + "prediction_rows"
        self.assertFalse(hasattr(policy, helper_name))
        self.assertNotIn(helper_name, getattr(policy, "__all__", ()))

    def test_release_source_has_no_machine_specific_path_markers(self) -> None:
        blocked = (
            "/scratch/",
            "/projects/",
            "/home/",
            "/users/",
            "path.home()",
        )
        offenders: list[str] = []
        for path in _tracked_release_files():
            if path == Path(__file__):
                continue
            if path.is_relative_to(PROJECT_ROOT / "tests"):
                continue
            if path.suffix not in {".py", ".toml", ".md", ".json", ".txt", ".yml", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8").replace("\\", "/").lower()
            for pattern in blocked:
                if pattern in text:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{pattern}")
        self.assertEqual(offenders, [])

    def test_molecule_sources_do_not_embed_local_paths_or_legacy_dataset_constants(self) -> None:
        offenders: list[str] = []
        blocked_patterns = (
            "Path.home()",
            "/" + "home/",
            "/" + "users/",
            "\\\\users\\\\",
            "triangulene_2",
            "trajectory_cleaned",
            "ARTIFACT_EXCLUDED",
        )
        for path in (
            PROJECT_ROOT / "src" / "genode" / "data" / "molecule_xyz.py",
            PROJECT_ROOT / "src" / "genode" / "data" / "prepare_molecule_xyz.py",
            PROJECT_ROOT / "src" / "genode" / "training" / "train_molecule_backbone.py",
            PROJECT_ROOT / "src" / "genode" / "evaluation" / "molecule_metrics.py",
        ):
            text = path.read_text(encoding="utf-8")
            for pattern in blocked_patterns:
                if pattern in text:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{pattern}")
        self.assertEqual(offenders, [])

    def test_auto_device_uses_cuda_when_available(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_torch_device("auto").type, "cuda")

    def test_auto_device_uses_cpu_without_cuda(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_torch_device("auto").type, "cpu")


if __name__ == "__main__":
    unittest.main()
