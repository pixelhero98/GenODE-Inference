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
                "genode-train-conditional-opd",
                "genode-run-train20-v43-pooled-calibration",
                "genode-build-ser-ptg-reference",
                "genode-evaluate-schedule-summary",
                "genode-build-hardness-figure",
                "genode-build-ptg-figure",
            },
        )
        self.assertEqual(scripts["genode-train-backbone"], "genode.training.train_backbone:main")
        self.assertEqual(scripts["genode-run-schedules"], "genode.evaluation.diffusion_flow_time_reparameterization:main")
        self.assertEqual(scripts["genode-train-conditional-opd"], "genode.conditional_opd.train_conditional_opd:main")
        self.assertEqual(
            scripts["genode-run-train20-v43-pooled-calibration"],
            "genode.conditional_opd.train20_v43_pooled_calibration:main",
        )
        self.assertNotIn("genode-run-train20-v4-bo-selection", scripts)
        self.assertNotIn("genode-run-train20-v42-no-val-selection", scripts)
        self.assertNotIn("genode-run-train20-v42-f-final-retrain", scripts)
        self.assertNotIn("genode-run-train20-expanded-opd-selection", scripts)
        self.assertNotIn("genode-train-density-opd", scripts)
        self.assertNotIn("genode-train-mlp-flow-opd", scripts)
        self.assertNotIn("genode-run-clean-opd-selection", scripts)
        self.assertNotIn("genode-build-expanded-candidate-summary", scripts)
        self.assertNotIn("genode-evaluate-student-schedules", scripts)
        self.assertNotIn("genode-review-plan-html", scripts)
        self.assertEqual(scripts["genode-build-ser-ptg-reference"], "genode.conditional_opd.ser_ptg_reference:main")
        self.assertEqual(scripts["genode-evaluate-schedule-summary"], "genode.conditional_opd.evaluate_schedule_summary:main")
        for target in scripts.values():
            module_name, func_name = str(target).split(":", 1)
            self.assertTrue(callable(getattr(importlib.import_module(module_name), func_name)))

    def test_auto_device_uses_cuda_when_available(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_torch_device("auto").type, "cuda")

    def test_auto_device_uses_cpu_without_cuda(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_torch_device("auto").type, "cpu")

    def test_v4_orchestration_modules_are_legacy_only(self) -> None:
        importlib.import_module("genode.conditional_opd.legacy.train20_v4_bo_selection")
        importlib.import_module("genode.conditional_opd.legacy.train20_v42_no_val_selection")
        importlib.import_module("genode.conditional_opd.legacy.train20_v42_f_final_retrain")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("genode.conditional_opd.train20_v4_bo_selection")

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
