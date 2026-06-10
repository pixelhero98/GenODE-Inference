from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path
from unittest import mock

import tomllib

from genode.runtime import resolve_torch_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_CLUSTER_PATTERNS = ("/" + "scratch/", "pixel" + "hero", "b" + "35z")


def _active_script_paths() -> list[Path]:
    archive_root = PROJECT_ROOT / "scripts" / "legacy_archive_20260608"
    paths: list[Path] = []
    for path in (PROJECT_ROOT / "scripts").rglob("*"):
        if path.is_file() and archive_root not in path.parents:
            paths.append(path)
    return paths


class GenODEInterfaceTests(unittest.TestCase):
    def test_import_resolves_to_standalone_package(self) -> None:
        module = importlib.import_module("genode")
        module_path = Path(module.__file__).resolve()
        self.assertIn(str(PROJECT_ROOT / "src" / "genode"), str(module_path))
        self.assertNotIn("Diffusion-Flow-Inference", str(module_path))

    def test_public_entry_points_are_registered(self) -> None:
        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = data["project"]["scripts"]
        expected_core = {
            "genode-train-backbone",
            "genode-run-schedules",
            "genode-prepare-molecule-xyz",
            "genode-train-molecule-backbone",
            "genode-eval-molecule-backbone",
            "genode-train-gipo",
            "genode-report-gipo-locked-test",
            "genode-report-gipo-teacher-oracle",
            "genode-build-ser-ptg-reference",
            "genode-evaluate-schedule-summary",
            "genode-build-hardness-figure",
            "genode-build-ptg-figure",
        }
        self.assertTrue(expected_core <= set(scripts))
        for target in scripts.values():
            module_name, func_name = str(target).split(":", 1)
            self.assertTrue(callable(getattr(importlib.import_module(module_name), func_name)))

    def test_docs_describe_gipo_as_additive_context_only(self) -> None:
        texts = [
            (PROJECT_ROOT / "docs" / "gipo.md").read_text(encoding="utf-8"),
            (PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
        ]
        for text in texts:
            lowered = text.lower()
            self.assertIn("additive_mlp_v1", text)
            self.assertIn("context-only", lowered)
            self.assertNotIn("series_" + "hash", lowered)
            self.assertNotIn("hash_" + "fourier", lowered)
            self.assertNotIn("hash-fourier", lowered)

    def test_gipo_trainer_public_contract_is_canonical(self) -> None:
        from genode.gipo.train_gipo import build_argparser

        parser = build_argparser()
        choices_by_dest = {action.dest: set(action.choices or []) for action in parser._actions}
        defaults_by_dest = {action.dest: action.default for action in parser._actions}
        options = {option for action in parser._actions for option in action.option_strings}

        self.assertEqual(defaults_by_dest["gipo_conditioning_style"], "additive_mlp_v1")
        self.assertEqual(choices_by_dest["gipo_conditioning_style"], {"additive_mlp_v1", "adaln_zero_v1"})
        self.assertEqual(choices_by_dest["gipo_teacher_conditioning_style"], {"additive_mlp_v1", "adaln_zero_v1"})
        self.assertEqual(choices_by_dest["gipo_student_conditioning_style"], {"additive_mlp_v1", "adaln_zero_v1"})
        self.assertIn("--allow_noncanonical_conditioning", options)
        self.assertIn("--gipo_teacher_conditioning_style", options)
        self.assertIn("--gipo_student_conditioning_style", options)
        self.assertIn("--teacher_unseen_selection_rows_csv", options)

        removed_options = {
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
            "--student_pseudo_target_" + "weight",
        }
        self.assertFalse(removed_options & options)

    def test_final_b64_ab_is_additive_canonical_with_adaln_sidecar(self) -> None:
        root = PROJECT_ROOT / "scripts" / "verification_gipo_locked_multiaxis_b64_final_ab_20260607"
        files = [
            root / "00_prepare_reused_artifacts.sbatch",
            root / "02_train_additive_locked.sbatch",
            root / "02_train_adaln_locked.sbatch",
            root / "03_report_additive_locked_seen_unseen.sbatch",
            root / "03_report_adaln_locked_seen_unseen.sbatch",
            root / "04_collect_b64_final_ab_summary.sbatch",
            root / "collect_b64_final_ab_summary.py",
            root / "submit_b64_final_ab.sh",
            root / "validate_locked_artifacts.py",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
        additive_train = (root / "02_train_additive_locked.sbatch").read_text(encoding="utf-8")
        adaln_train = (root / "02_train_adaln_locked.sbatch").read_text(encoding="utf-8")
        collect = (root / "collect_b64_final_ab_summary.py").read_text(encoding="utf-8")

        self.assertIn("RUN_ID=additive_locked_b64_normregret_final", additive_train)
        self.assertIn("RUN_ID=adaln_locked_b64_normregret_final", adaln_train)
        self.assertIn("SELECTION_MODE=weighted_normalized_regret_v1", additive_train)
        self.assertIn("DENSITY_BIN_COUNT=64", additive_train)
        self.assertIn("--teacher_unseen_selection_rows_csv", additive_train)
        self.assertNotIn("--allow_noncanonical_conditioning", additive_train)
        self.assertIn("--allow_noncanonical_conditioning", adaln_train)
        self.assertNotIn("--teacher_selection_component_" + "weights", combined)
        self.assertNotIn("--student_pseudo_target_" + "weight", combined)
        self.assertNotIn("python -m genode.gipo.report_teacher_oracle", combined)
        self.assertIn("predeclared_additive_canonical_with_adaln_sidecar_v1", collect)
        self.assertIn('"sidecar_results_are_reporting_only": True', collect)
        self.assertIn('"locked_test_used_for_conditioning_selection": False', collect)
        self.assertNotIn("promotion_" + "decision", collect)
        self.assertNotIn("locked_test_used_for_conditioning_" + "promotion", collect)
        self.assertNotIn("--additive" + "128_root", collect)
        for pattern in PRIVATE_CLUSTER_PATTERNS:
            self.assertNotIn(pattern, combined)

    def test_teacher_student_conditioning_ab_trains_two_mixed_runs_and_collects_four_way(self) -> None:
        root = PROJECT_ROOT / "scripts" / "verification_gipo_locked_multiaxis_b64_teacher_student_conditioning_ab_20260610"
        expected_train_files = {
            "02_train_teacher_additive_student_adaln_locked.sbatch",
            "02_train_teacher_adaln_student_additive_locked.sbatch",
        }
        expected_report_files = {
            "03_report_teacher_additive_student_adaln_locked_seen_unseen.sbatch",
            "03_report_teacher_adaln_student_additive_locked_seen_unseen.sbatch",
        }
        train_files = {path.name for path in root.glob("02_train_*.sbatch")}
        report_files = {path.name for path in root.glob("03_report_*.sbatch")}
        self.assertEqual(train_files, expected_train_files)
        self.assertEqual(report_files, expected_report_files)

        files = [
            root / "00_prepare_reused_artifacts.sbatch",
            *(root / name for name in sorted(expected_train_files)),
            *(root / name for name in sorted(expected_report_files)),
            root / "04_collect_teacher_student_conditioning_ab_summary.sbatch",
            root / "collect_teacher_student_conditioning_ab_summary.py",
            root / "submit_teacher_student_conditioning_ab.sh",
            root / "validate_locked_artifacts.py",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
        train_add_student_adaln = (root / "02_train_teacher_additive_student_adaln_locked.sbatch").read_text(encoding="utf-8")
        train_adaln_student_add = (root / "02_train_teacher_adaln_student_additive_locked.sbatch").read_text(encoding="utf-8")
        report_combined = "\n".join((root / name).read_text(encoding="utf-8") for name in sorted(expected_report_files))
        collect = (root / "collect_teacher_student_conditioning_ab_summary.py").read_text(encoding="utf-8")
        submit = (root / "submit_teacher_student_conditioning_ab.sh").read_text(encoding="utf-8")

        self.assertIn("GENODE_SOURCE_ROOT", combined)
        self.assertIn("RUN_ID=teacher_additive_student_adaln_locked_b64_normregret_final", train_add_student_adaln)
        self.assertIn("TEACHER_CONDITIONING_STYLE=additive_mlp_v1", train_add_student_adaln)
        self.assertIn("STUDENT_CONDITIONING_STYLE=adaln_zero_v1", train_add_student_adaln)
        self.assertIn("RUN_ID=teacher_adaln_student_additive_locked_b64_normregret_final", train_adaln_student_add)
        self.assertIn("TEACHER_CONDITIONING_STYLE=adaln_zero_v1", train_adaln_student_add)
        self.assertIn("STUDENT_CONDITIONING_STYLE=additive_mlp_v1", train_adaln_student_add)
        for train_text in (train_add_student_adaln, train_adaln_student_add):
            self.assertIn("SELECTION_MODE=weighted_normalized_regret_v1", train_text)
            self.assertIn("DENSITY_BIN_COUNT=64", train_text)
            self.assertIn("--teacher_unseen_selection_rows_csv", train_text)
            self.assertIn("--gipo_teacher_conditioning_style", train_text)
            self.assertIn("--gipo_student_conditioning_style", train_text)
            self.assertIn("--allow_noncanonical_conditioning", train_text)
            self.assertNotIn("--gipo_conditioning_style", train_text)
            self.assertIn('"same_style_shortcut_used": False', train_text)

        self.assertIn("python -m genode.gipo.report_locked_test", report_combined)
        self.assertIn('locked_reports/${panel}/student/${RUN_ID}', report_combined)
        self.assertIn("--allow_noncanonical_conditioning", report_combined)
        self.assertIn("teacher_additive_student_adaln_locked_b64_normregret_final", report_combined)
        self.assertIn("teacher_adaln_student_additive_locked_b64_normregret_final", report_combined)

        self.assertIn("--same-style-root", collect)
        self.assertIn("same_style_root", collect)
        self.assertIn("predeclared_teacher_student_conditioning_four_way_ab_v1", collect)
        self.assertIn("teacher_additive_student_additive", collect)
        self.assertIn("teacher_additive_student_adaln", collect)
        self.assertIn("teacher_adaln_student_additive", collect)
        self.assertIn("teacher_adaln_student_adaln", collect)
        self.assertIn("student_locked_reports_only", collect)
        self.assertIn("gipo_locked_multiaxis_b64_teacher_student_conditioning_ab_four_way_summary", collect)

        self.assertEqual(submit.count("02_train_"), 2)
        self.assertEqual(submit.count("03_report_"), 2)
        self.assertIn("afterok:${teacher_additive_student_adaln_report_job}:${teacher_adaln_student_additive_report_job}", submit)
        self.assertNotIn("--student_pseudo_target_" + "weight", combined)
        self.assertNotIn("python -m genode.gipo.report_teacher_oracle", combined)
        self.assertIn("pseudo target weight is nonzero", collect)
        self.assertIn("student_target_protocol", collect)
        self.assertIn("teacher-oracle report exists", collect)
        for pattern in PRIVATE_CLUSTER_PATTERNS:
            self.assertNotIn(pattern, combined)

    def test_checkpoint_maturity_stream_is_additive_only_and_regenerates_artifacts(self) -> None:
        root = PROJECT_ROOT / "scripts" / "verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608"
        files = [
            root / "01_generate_artifacts.sbatch",
            root / "02_train_additive.sbatch",
            root / "03_report_additive_seen_unseen.sbatch",
            root / "04_collect_checkpoint_maturity_summary.sbatch",
            root / "collect_checkpoint_maturity_summary.py",
            root / "submit_checkpoint_maturity.sh",
            root / "validate_locked_artifacts.py",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
        generate = (root / "01_generate_artifacts.sbatch").read_text(encoding="utf-8")
        train = (root / "02_train_additive.sbatch").read_text(encoding="utf-8")
        report = (root / "03_report_additive_seen_unseen.sbatch").read_text(encoding="utf-8")
        collect = (root / "collect_checkpoint_maturity_summary.py").read_text(encoding="utf-8")
        submit = (root / "submit_checkpoint_maturity.sh").read_text(encoding="utf-8")

        self.assertIn("GENODE_OTFLOW_TRAIN_STEPS", generate)
        self.assertIn("--otflow_train_steps \"${GENODE_OTFLOW_TRAIN_STEPS}\"", generate)
        self.assertIn("--build", generate)
        self.assertNotIn("ln -sfn", generate)
        self.assertIn("additive_locked_b64_normregret_ckpt${GENODE_BUDGET_LABEL}", train)
        self.assertIn("CONDITIONING_STYLE=additive_mlp_v1", train)
        self.assertNotIn("--allow_noncanonical_conditioning", train)
        self.assertNotIn("--density_bin_count", train)
        self.assertNotIn("--teacher_checkpoint_" + "selection_mode", train)
        self.assertNotIn("--student_checkpoint_" + "selection", train)
        self.assertNotIn("--teacher_selection_component_" + "weights", train)
        self.assertNotIn("report_teacher_oracle", report)
        self.assertIn("PHYSICAL_SCHEDULES = (", collect)
        self.assertIn("ser_ptg_local_defect_eta005", collect)
        self.assertIn("--candidate", collect)
        self.assertIn("--comparator_root", collect)
        self.assertIn("--comparator_run_id", collect)
        self.assertIn("locked_test_used_for_backbone_maturity_selection", collect)
        self.assertIn("GENODE_MATURITY_SPECS", submit)
        self.assertIn("label,train_steps,root,run_id,report_label_prefix", submit)
        self.assertIn("GENODE_MATURITY_CANDIDATES", submit)
        self.assertIn("GENODE_COMPARATOR_ROOT", submit)
        self.assertNotIn("ROOT16", submit)
        self.assertNotIn("ROOT12", submit)
        self.assertNotIn("GENODE_CKPT16_ROOT", combined)
        self.assertNotIn("GENODE_CKPT12_ROOT", combined)
        for pattern in PRIVATE_CLUSTER_PATTERNS:
            self.assertNotIn(pattern, combined)

    def test_legacy_gipo_script_roots_are_archived(self) -> None:
        archive_root = PROJECT_ROOT / "scripts" / "legacy_archive_20260608"
        manifest = archive_root / "legacy_gipo_script_archive_manifest.json"
        self.assertTrue(manifest.exists())
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        archived = {Path(item["archived_relative_path"]).name for item in payload["archived_script_roots"]}
        expected = {
            "verification_gipo_additive_locked_" + "20260606",
            "verification_gipo_additive_locked_multiaxis_20260606",
            "verification_gipo_adaln_locked_multiaxis_ab_" + "20260607",
            "verification_gipo_locked_multiaxis_b64_ab_" + "20260607",
            "verification_gipo_locked_multiaxis_b64_optim_" + "20260607",
            "verification_gipo_seen_ablation_" + "20260606",
            "verification_gipo_balanced_seen_multinfe_" + "20260606",
            "verification_gipo_probe_2_6_" + "20260606",
        }
        self.assertTrue(expected <= archived)
        active_roots = {path.name for path in (PROJECT_ROOT / "scripts").iterdir() if path.is_dir()}
        self.assertIn("verification_gipo_locked_multiaxis_b64_final_ab_20260607", active_roots)
        self.assertIn("verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608", active_roots)
        self.assertIn("verification_gipo_locked_multiaxis_b64_teacher_student_conditioning_ab_20260610", active_roots)
        self.assertFalse(expected & active_roots)

    def test_retired_flags_do_not_remain_in_active_scripts(self) -> None:
        offenders: list[str] = []
        retired_patterns = (
            "--teacher_selection_component_" + "weights",
            "--teacher_nfe_" + "proxy_anchor_values",
            "--teacher_fit_checkpoint_" + "selection",
            "--series_holdout_" + "fraction",
            "--series_unknown_" + "dropout",
            "--student_pseudo_target_" + "weight",
            "composite_" + "regret_" + "guarded_v1",
            "additive_locked_" + "b128",
            "adaln_locked_" + "b128",
            "b64_multiaxis_opt_" + "guarded",
            "b64_multiaxis_select_" + "ab",
            "locked_test_used_for_conditioning_" + "promotion",
            "--density_bin_count",
            "--teacher_checkpoint_" + "selection_mode",
            "--student_checkpoint_" + "selection",
            *PRIVATE_CLUSTER_PATTERNS,
        )
        for path in _active_script_paths():
            if path.suffix not in {".py", ".sh", ".sbatch"}:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in retired_patterns:
                if pattern in text:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{pattern}")
        self.assertEqual(offenders, [])

    def test_auto_device_uses_cuda_when_available(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_torch_device("auto").type, "cuda")

    def test_auto_device_uses_cpu_without_cuda(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_torch_device("auto").type, "cpu")

    def test_no_upstream_namespace_imports_remain_in_source(self) -> None:
        offenders = []
        for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "tests", PROJECT_ROOT / "scripts"):
            for path in root.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                needle = "diffusion" + "_flow" + "_inference"
                if needle in text:
                    offenders.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(offenders, [])

    def test_molecule_sources_do_not_embed_local_paths_or_legacy_dataset_constants(self) -> None:
        offenders: list[str] = []
        blocked_patterns = (
            "Downloads",
            "PycharmProjects",
            "/home/yzn",
            "Path.home()",
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


if __name__ == "__main__":
    unittest.main()
