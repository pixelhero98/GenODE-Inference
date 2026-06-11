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
        expected = {
            "genode-train-backbone",
            "genode-run-schedules",
            "genode-prepare-molecule-xyz",
            "genode-train-molecule-backbone",
            "genode-eval-molecule-backbone",
            "genode-train-gipo",
            "genode-report-gipo-locked-test",
            "genode-build-ser-ptg-reference",
            "genode-evaluate-schedule-summary",
            "genode-build-hardness-figure",
            "genode-build-ptg-figure",
        }
        self.assertEqual(set(scripts), expected)
        for target in scripts.values():
            module_name, func_name = str(target).split(":", 1)
            self.assertTrue(callable(getattr(importlib.import_module(module_name), func_name)))

    def test_single_markdown_file_is_readme(self) -> None:
        markdown_files = sorted(path.relative_to(PROJECT_ROOT).as_posix() for path in PROJECT_ROOT.rglob("*.md") if ".git" not in path.parts)
        self.assertEqual(markdown_files, ["README.md"])
        text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("gipo_density", text)
        self.assertIn("additive_mlp", text)
        self.assertIn("context-only", text.lower())
        self.assertIn("genode-train-gipo", text)
        self.assertIn("genode-report-gipo-locked-test", text)
        self.assertNotIn("allow_noncanonical", text)
        self.assertNotIn("teacher-oracle", text.lower())

    def test_no_tracked_scripts_or_legacy_docs_tree(self) -> None:
        self.assertFalse((PROJECT_ROOT / "scripts").exists())
        self.assertFalse((PROJECT_ROOT / "docs").exists())

    def test_gipo_trainer_public_contract_is_canonical(self) -> None:
        from genode.gipo.train_gipo import build_argparser

        parser = build_argparser()
        options = {option for action in parser._actions for option in action.option_strings}

        self.assertIn("--teacher_unseen_selection_rows_csv", options)
        self.assertIn("--student_pseudo_rows_csv", options)
        self.assertIn("--student_pseudo_target_" + "weight", options)
        removed_options = {
            "--gipo_" + "conditioning_style",
            "--gipo_teacher_" + "conditioning_style",
            "--gipo_student_" + "conditioning_style",
            "--allow_noncanonical_conditioning",
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
        }
        self.assertFalse(removed_options & options)

    def test_gipo_policy_public_surface_excludes_teacher_prediction_helper(self) -> None:
        from genode.gipo import policy

        helper_name = "build_teacher_weighted_density_" + "prediction_rows"
        self.assertFalse(hasattr(policy, helper_name))
        self.assertNotIn(helper_name, getattr(policy, "__all__", ()))

    def test_no_private_paths_or_upstream_namespace_in_tracked_text(self) -> None:
        blocked = (
            "/" + "scratch/",
            "/" + "projects/",
            "/" + "home/",
            "pixel" + "hero",
            "b" + "35z",
            "Py" + "charmProjects",
            "Diffusion-Flow-Inference",
            "diffusion" + "_flow" + "_inference",
        )
        offenders: list[str] = []
        for path in PROJECT_ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path == Path(__file__):
                continue
            if path.suffix not in {".py", ".toml", ".md", ".json", ".txt", ".yml", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in blocked:
                if pattern in text:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{pattern}")
        self.assertEqual(offenders, [])

    def test_molecule_sources_do_not_embed_local_paths_or_legacy_dataset_constants(self) -> None:
        offenders: list[str] = []
        blocked_patterns = (
            "Downloads",
            "Py" + "charmProjects",
            "/" + "home/yzn",
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

    def test_auto_device_uses_cuda_when_available(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_torch_device("auto").type, "cuda")

    def test_auto_device_uses_cpu_without_cuda(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_torch_device("auto").type, "cpu")


if __name__ == "__main__":
    unittest.main()
