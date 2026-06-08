from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path
from unittest import mock

import tomllib

from genode.runtime import resolve_torch_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        self.assertEqual(
            set(scripts),
            {
                "genode-train-backbone",
                "genode-run-schedules",
                "genode-train-gipo",
                "genode-report-gipo-locked-test",
                "genode-report-gipo-teacher-oracle",
                "genode-build-ser-ptg-reference",
                "genode-evaluate-schedule-summary",
                "genode-build-hardness-figure",
                "genode-build-ptg-figure",
            },
        )
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

        self.assertEqual(choices_by_dest["teacher_checkpoint_selection_mode"], {"weighted_normalized_regret_v1"})
        self.assertEqual(choices_by_dest["student_checkpoint_selection"], {"validation_ce_v1"})
        self.assertEqual(defaults_by_dest["density_bin_count"], 64)
        self.assertEqual(defaults_by_dest["gipo_conditioning_style"], "additive_mlp_v1")
        self.assertEqual(choices_by_dest["gipo_conditioning_style"], {"additive_mlp_v1", "adaln_zero_v1"})
        self.assertIn("--allow_noncanonical_conditioning", options)
        self.assertIn("--teacher_unseen_selection_rows_csv", options)
        self.assertIn("--student_checkpoint_selection", options)

        removed_options = {
            "--teacher_selection_component_" + "weights",
            "--teacher_nfe_" + "proxy_anchor_values",
            "--teacher_fit_checkpoint_" + "selection",
            "--series_holdout_" + "fraction",
            "--teacher_architecture",
            "--student_architecture",
            "--setting_encoder_mode",
            "--setting_feature_mode",
            "--series_unknown_" + "dropout",
            "--student_nfe_smoothness_" + "weight",
            "--student_nfe_smoothness_" + "mode",
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
        self.assertNotIn("--student_nfe_smoothness_" + "weight", combined)
        self.assertNotIn("python -m genode.gipo.report_teacher_oracle", combined)
        self.assertIn("predeclared_additive_canonical_with_adaln_sidecar_v1", collect)
        self.assertIn('"sidecar_results_are_reporting_only": True', collect)
        self.assertIn('"locked_test_used_for_conditioning_selection": False', collect)
        self.assertNotIn("promotion_" + "decision", collect)
        self.assertNotIn("locked_test_used_for_conditioning_" + "promotion", collect)
        self.assertNotIn("--additive" + "128_root", collect)

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
        self.assertIn("--density_bin_count \"${DENSITY_BIN_COUNT}\"", train)
        self.assertIn("--teacher_checkpoint_selection_mode \"${SELECTION_MODE}\"", train)
        self.assertIn("--student_checkpoint_selection \"${STUDENT_SELECTOR_MODE}\"", train)
        self.assertNotIn("--teacher_selection_component_" + "weights", train)
        self.assertNotIn("report_teacher_oracle", report)
        self.assertIn("PHYSICAL_SCHEDULES = (", collect)
        self.assertIn("ser_ptg_local_defect_eta005", collect)
        self.assertIn("FINAL20_RUN_ID = \"additive_locked_b64_normregret_final\"", collect)
        self.assertIn("locked_test_used_for_backbone_maturity_selection", collect)
        self.assertIn("GENODE_OTFLOW_TRAIN_STEPS=16000", submit)
        self.assertIn("GENODE_OTFLOW_TRAIN_STEPS=12000", submit)
        self.assertIn("afterok:${report16_job}:${report12_job}", submit)
        self.assertIn("verification_gipo_locked_multiaxis_b64_ckpt16k_20260608", combined)
        self.assertIn("verification_gipo_locked_multiaxis_b64_ckpt12k_20260608", combined)

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
        self.assertFalse(expected & active_roots)

    def test_retired_flags_do_not_remain_in_active_scripts(self) -> None:
        offenders: list[str] = []
        retired_patterns = (
            "--teacher_selection_component_" + "weights",
            "--teacher_nfe_" + "proxy_anchor_values",
            "--teacher_fit_checkpoint_" + "selection",
            "--series_holdout_" + "fraction",
            "--series_unknown_" + "dropout",
            "--student_nfe_smoothness_" + "weight",
            "--student_pseudo_target_" + "weight",
            "composite_" + "regret_" + "guarded_v1",
            "additive_locked_" + "b128",
            "adaln_locked_" + "b128",
            "b64_multiaxis_opt_" + "guarded",
            "b64_multiaxis_select_" + "ab",
            "locked_test_used_for_conditioning_" + "promotion",
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


if __name__ == "__main__":
    unittest.main()
