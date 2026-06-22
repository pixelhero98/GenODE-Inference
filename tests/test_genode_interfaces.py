from __future__ import annotations

import importlib
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import tomllib

from genode.runtime import resolve_torch_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "data",
    "dist",
    "outputs",
    "paper_datasets",
    "reports",
}


def _is_generated_path(path: Path) -> bool:
    parts = path.relative_to(PROJECT_ROOT).parts
    return bool(GENERATED_ROOTS & set(parts)) or any(part.endswith(".egg-info") for part in parts)


def _source_release_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return [
            path
            for path in PROJECT_ROOT.rglob("*")
            if path.is_file() and not _is_generated_path(path)
        ]
    return [PROJECT_ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]


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
            "genode-run-full-pipeline",
            "genode-prepare-molecule-xyz",
            "genode-train-molecule-backbone",
            "genode-eval-molecule-backbone",
            "genode-train-gipo",
            "genode-preflight-gipo-rows",
            "genode-report-gipo-locked-test",
            "genode-build-ser-ptg-reference",
            "genode-evaluate-schedule-summary",
            "genode-build-hardness-figure",
            "genode-build-ptg-figure",
            "genode-package-backbone-family",
            "genode-validate-backbone-package",
        }
        self.assertEqual(set(scripts), expected)
        for target in scripts.values():
            module_name, func_name = str(target).split(":", 1)
            self.assertTrue(callable(getattr(importlib.import_module(module_name), func_name)))

    def test_single_markdown_file_is_readme(self) -> None:
        markdown_files = sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in _source_release_files()
            if path.suffix == ".md"
        )
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
        self.assertIn("--student_teacher_score_" + "weight", options)
        self.assertIn("--student_teacher_score_warmup_" + "fraction", options)
        self.assertIn("--student_teacher_score_include_" + "pseudo", options)
        self.assertIn("--student_target_mixture_" + "mode", options)
        self.assertIn("--student_target_elite_" + "fraction", options)
        self.assertIn("--student_target_elite_" + "k", options)
        self.assertIn("--student_target_elite_min_" + "count", options)
        self.assertIn("--student_target_elite_blend_all_" + "weight", options)
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

    def test_full_pipeline_public_contract_includes_ablation_first(self) -> None:
        from genode.pipeline.full_pipeline import build_argparser

        parser = build_argparser()
        options = {option for action in parser._actions for option in action.option_strings}

        self.assertIn("--ablation_first", options)
        self.assertIn("--gipo_ablation_preset", options)
        self.assertIn("--ser_calibration_batch_size", options)
        self.assertIn("--ser_val_windows", options)

    def test_project_path_resolver_accepts_legacy_package_prefixes(self) -> None:
        from genode.data import otflow_paths

        with mock.patch.object(otflow_paths, "project_root", return_value=PROJECT_ROOT):
            self.assertEqual(otflow_paths.resolve_project_path("genode/outputs/example"), (PROJECT_ROOT / "outputs" / "example").resolve())
            self.assertEqual(otflow_paths.resolve_project_path("genode/data/example"), (PROJECT_ROOT / "data" / "example").resolve())
            self.assertEqual(
                otflow_paths.resolve_project_path("genode/paper_datasets/example"),
                (PROJECT_ROOT / "paper_datasets" / "example").resolve(),
            )

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
            "Py" + "charmProjects",
            "Diffusion-Flow-Inference",
            "diffusion" + "_flow" + "_inference",
        )
        offenders: list[str] = []
        for path in _source_release_files():
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
            "/" + "home/private_user",
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
