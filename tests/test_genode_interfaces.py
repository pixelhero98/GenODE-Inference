from __future__ import annotations

import importlib
import unittest
from pathlib import Path
from unittest import mock

import tomllib

from genode.runtime import resolve_torch_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        self.assertEqual(scripts["genode-train-backbone"], "genode.training.train_backbone:main")
        self.assertEqual(scripts["genode-run-schedules"], "genode.evaluation.diffusion_flow_time_reparameterization:main")
        self.assertEqual(scripts["genode-build-ser-ptg-reference"], "genode.gipo.ser_ptg_reference:main")
        self.assertEqual(scripts["genode-train-gipo"], "genode.gipo.train_gipo:main")
        self.assertEqual(scripts["genode-report-gipo-locked-test"], "genode.gipo.report_locked_test:main")
        self.assertEqual(scripts["genode-report-gipo-teacher-oracle"], "genode.gipo.report_teacher_oracle:main")
        self.assertEqual(scripts["genode-evaluate-schedule-summary"], "genode.gipo.evaluate_schedule_summary:main")
        for target in scripts.values():
            module_name, func_name = str(target).split(":", 1)
            self.assertTrue(callable(getattr(importlib.import_module(module_name), func_name)))

    def test_deprecated_context_cli_flags_are_removed(self) -> None:
        from genode.gipo.report_locked_test import build_argparser as build_report_argparser
        from genode.gipo.train_gipo import build_argparser as build_train_argparser

        train_options = {option for action in build_train_argparser()._actions for option in action.option_strings}
        report_options = {option for action in build_report_argparser()._actions for option in action.option_strings}
        self.assertNotIn("--" + "holdout_fraction", train_options)
        self.assertNotIn("--" + "support_schedule_keys", report_options)
        self.assertNotIn("--locked_" + "context_rows", report_options)
        self.assertNotIn("--locked_" + "context_embeddings_npz", report_options)
        self.assertIn("--gipo_student_checkpoint", report_options)
        self.assertIn("--context_rows", report_options)
        self.assertIn("--context_embeddings_npz", report_options)
        self.assertIn("--split_phase", report_options)
        self.assertIn("--selection_mode", report_options)

    def test_gipo_trainer_accepts_only_transformer_continuous_v3_choices(self) -> None:
        from genode.gipo.train_gipo import build_argparser as build_train_argparser

        parser = build_train_argparser()
        choices_by_dest = {action.dest: set(action.choices or []) for action in parser._actions}
        self.assertEqual(choices_by_dest["teacher_architecture"], {"light_transformer_v1"})
        self.assertEqual(choices_by_dest["student_architecture"], {"light_transformer_v1"})
        self.assertEqual(choices_by_dest["setting_encoder_mode"], {"continuous_v3"})
        self.assertEqual(choices_by_dest["setting_feature_mode"], {"continuous_v3"})

    def test_legacy_gipo_namespaces_and_protocols_are_retired(self) -> None:
        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        script_text = "\n".join(sorted(data["project"]["scripts"]))
        self.assertNotIn("context-" + "conditional", script_text)
        self.assertNotIn("conditional-" + "o" + "pd", script_text)
        self.assertNotIn("context-" + "locked-test", script_text)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("genode." + "conditional_" + "o" + "pd")

        offenders = []
        retired_tokens = (
            "genode." + "conditional_" + "o" + "pd",
            "context_" + "density_" + "o" + "pd_v1",
            "genode-train-" + "context-" + "conditional-" + "o" + "pd",
            "genode-report-" + "context-" + "locked-test",
            "bo" + "torch",
            "gpy" + "torch",
            "train" + "20",
            "v" + "4.3",
            "static_" + "residual",
            "static-" + "residual",
        )
        checked_suffixes = {".py", ".sh", ".sbatch"}
        for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "tests", PROJECT_ROOT / "scripts"):
            for path in root.rglob("*"):
                if path.suffix not in checked_suffixes:
                    continue
                text = path.read_text(encoding="utf-8").lower()
                for token in retired_tokens:
                    if token.lower() in text:
                        offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{token}")
        self.assertEqual(offenders, [])

    def test_retired_reporter_flags_do_not_remain_in_scripts(self) -> None:
        offenders = []
        retired_patterns = (
            "--locked_" + "context_rows",
            "--locked_" + "context_embeddings_npz",
            "locked_" + "context_rows=",
            "locked_" + "context_embeddings_npz=",
            "context_" + "student_checkpoint",
            "tests.test_context_" + "conditional_" + "o" + "pd",
            "tests.test_" + "o" + "pd_mlp_components",
        )
        checked_suffixes = {".py", ".sh", ".sbatch"}
        for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "tests", PROJECT_ROOT / "scripts", PROJECT_ROOT / "docs"):
            for path in root.rglob("*"):
                if path.suffix not in checked_suffixes and path.suffix != ".md":
                    continue
                text = path.read_text(encoding="utf-8").lower()
                for pattern in retired_patterns:
                    if pattern.lower() in text:
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
