from __future__ import annotations

import importlib
import importlib.util
import subprocess
import tomllib
import unittest
from pathlib import Path
from unittest import mock

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
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return [path for path in PROJECT_ROOT.rglob("*") if path.is_file() and not _is_generated_path(path)]
    return [
        path for line in result.stdout.splitlines() if line.strip() and (path := PROJECT_ROOT / line.strip()).is_file()
    ]


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
            "genode-evaluate-molecule-backbone",
            "genode-train-gico",
            "genode-preflight-gico-rows",
            "genode-report-gico-locked-test",
            "genode-evaluate-schedule-summary",
            "genode-image-gico",
            "genode-latent-clock",
        }
        self.assertEqual(set(scripts), expected)
        for target in scripts.values():
            module_name, func_name = str(target).split(":", 1)
            self.assertTrue(callable(getattr(importlib.import_module(module_name), func_name)))

    def test_publication_metadata_is_gico_only(self) -> None:
        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = data["project"]

        self.assertEqual(project["version"], "0.14.0")
        self.assertEqual(
            project["description"], "GICO inference-clock optimization for frozen flow-matching backbones."
        )

    def test_retired_feature_modules_and_commands_are_absent(self) -> None:
        retired_modules = (
            "genode." + "distillation",
            "genode.gico." + "ser_" + "ptg_reference",
            "genode.visualization",
        )
        for module_name in retired_modules:
            with self.subTest(module_name=module_name):
                spec = importlib.util.find_spec(module_name)
                self.assertTrue(spec is None or spec.loader is None)

        scripts = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
        retired_commands = (
            "genode-build-ser-" + "ptg-reference",
            "genode-build-" + "ptg-figure",
            "genode-collect-" + "flow-map-demonstrations",
            "genode-train-" + "flow-map",
            "genode-evaluate-" + "flow-map",
        )
        for command in retired_commands:
            with self.subTest(command=command):
                self.assertNotIn(command, scripts)

    def test_readme_documents_locked_test_frozen_calibration(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("genode-report-gico-locked-test", readme)
        self.assertIn("frozen calibration to paired test measurements without selection", readme)

    def test_release_markdown_is_current_and_scoped(self) -> None:
        markdown_files = sorted(
            path.relative_to(PROJECT_ROOT).as_posix() for path in _source_release_files() if path.suffix == ".md"
        )
        self.assertEqual(
            markdown_files,
            [
                "CONTRIBUTING.md",
                "README.md",
                "SECURITY.md",
                "THIRD_PARTY_NOTICES.md",
                "docs/evaluators.md",
                "docs/examples.md",
                "docs/frozen-bezier-kid.md",
                "docs/image-comparators.md",
                "docs/image-supervision.md",
                "docs/js-reinforce.md",
                "docs/student-selection.md",
            ],
        )
        text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("25 reference clocks", text)
        self.assertIn("genode-train-gico", text)
        self.assertIn("genode-report-gico-locked-test", text)
        for scenario in (
            "solar_energy_10m",
            "traffic_hourly",
            "weather_daily",
            "cifar10",
            "imagenet64",
            "sana",
            "sd15",
            "molecule_3d_set1",
            "molecule_3d_set2",
            "molecule_3d_set3",
        ):
            self.assertIn(scenario, text)
        self.assertNotIn("## 30-second entry point", text)
        self.assertNotIn("## Guarantees", text)
        self.assertNotIn("allow_noncanonical", text)
        self.assertNotIn("teacher-oracle", text.lower())
        for retired_term in (
            "Consistency " + "distillation",
            "endpoint " + "flow-map",
            "genode-build-ser-" + "ptg-reference",
            "genode-build-" + "ptg-figure",
            "genode-collect-" + "flow-map-demonstrations",
            "genode-train-" + "flow-map",
            "genode-evaluate-" + "flow-map",
        ):
            with self.subTest(retired_term=retired_term):
                self.assertNotIn(retired_term, text)

    def test_publication_ci_covers_supported_matrix_and_image_cli(self) -> None:
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        for required in (
            'python-version: "3.11"',
            'numpy_spec: "numpy>=1.26,<2"',
            'python-version: "3.13"',
            'numpy_spec: "numpy>=2,<3"',
            "python -m ruff format --check .",
            "python -m compileall -q src tests",
            "Inspect wheel and source distribution",
            "Install and test the source distribution",
            "genode-image-gico",
            "prepare",
            "train",
            "genode-train-gico",
            "genode-latent-clock",
            "validate",
            "materialize",
            "retired consistency feature",
            "retired SER/PTG feature",
            "retired_member_fragments",
        ):
            with self.subTest(required=required):
                self.assertIn(required, workflow)

    def test_source_distribution_manifest_includes_publication_metadata(self) -> None:
        manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        for required in (
            "include CONTRIBUTING.md",
            "include SECURITY.md",
            "recursive-include tests *.py",
        ):
            with self.subTest(required=required):
                self.assertIn(required, manifest)

    def test_no_tracked_scripts_or_unregistered_docs(self) -> None:
        self.assertFalse((PROJECT_ROOT / "scripts").exists())
        self.assertEqual(
            {path.name for path in (PROJECT_ROOT / "docs").iterdir()},
            {
                "evaluators.md",
                "examples.md",
                "frozen-bezier-kid.md",
                "image-comparators.md",
                "image-supervision.md",
                "js-reinforce.md",
                "student-selection.md",
            },
        )

    def test_gico_trainer_public_contract_is_canonical(self) -> None:
        from genode.gico.train_gico import build_argparser

        options = {option for action in build_argparser()._actions for option in action.option_strings}
        self.assertEqual(options, {"-h", "--help", "--config", "--student-kind", "--teacher-score-weight", "--dry-run"})

    def test_full_pipeline_public_contract_routes_explicit_gico_configuration(self) -> None:
        from genode.pipeline.full_pipeline import DEFAULT_STAGES, PIPELINE_STAGE_ORDER, build_argparser

        options = {option for action in build_argparser()._actions for option in action.option_strings}
        self.assertTrue({"--gico-config", "--student-kind", "--teacher-score-weight"} <= options)
        self.assertEqual(DEFAULT_STAGES, ("backbone_training", "schedule_rows_seen", "schedule_rows_unseen"))
        self.assertEqual(PIPELINE_STAGE_ORDER, ("data_prep", *DEFAULT_STAGES, "gico_training"))
        self.assertNotIn("--ablation_first", options)

    def test_project_path_resolver_does_not_rewrite_package_prefixes(self) -> None:
        from genode.data import otflow_paths

        with mock.patch.object(otflow_paths, "project_root", return_value=PROJECT_ROOT):
            self.assertEqual(
                otflow_paths.resolve_project_path("genode/outputs/example"),
                (PROJECT_ROOT / "genode" / "outputs" / "example").resolve(),
            )
            self.assertEqual(
                otflow_paths.resolve_project_path("genode/data/example"),
                (PROJECT_ROOT / "genode" / "data" / "example").resolve(),
            )
            self.assertEqual(
                otflow_paths.resolve_project_path("genode/paper_datasets/example"),
                (PROJECT_ROOT / "genode" / "paper_datasets" / "example").resolve(),
            )

    def test_gico_policy_public_surface_excludes_teacher_prediction_helper(self) -> None:
        from genode.gico import policy

        helper_name = "build_teacher_weighted_density_" + "prediction_rows"
        self.assertFalse(hasattr(policy, helper_name))
        self.assertNotIn(helper_name, getattr(policy, "__all__", ()))

    def test_gico_policy_exposes_artifact_and_native_context_interfaces(self) -> None:
        from genode.gico import policy

        for name in (
            "load_policy",
            "save_artifact",
            "load_context_embedding_table",
            "save_context_embedding_table",
            "stable_context_id",
        ):
            self.assertTrue(callable(getattr(policy, name)))

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
