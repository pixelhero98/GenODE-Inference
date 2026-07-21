from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path
from typing import Any, get_type_hints
from unittest.mock import patch

import numpy as np

from genode.data.otflow_medical_constants import LONG_TERM_ST_DATASET_KEY
from genode.gipo.density_representation import density_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackagingHygieneTests(unittest.TestCase):
    def _prepare_long_term_st_dataset(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required to import the Long-Term ST dataset builder")
        from genode.data.otflow_medical_datasets import prepare_long_term_st_dataset

        return prepare_long_term_st_dataset

    def test_density_metadata_type_hints_resolve(self) -> None:
        hints = get_type_hints(density_metadata)

        self.assertEqual(hints["return"], dict[str, Any])

    def test_medical_dependencies_are_optional(self) -> None:
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = pyproject["project"]["dependencies"]
        medical = pyproject["project"]["optional-dependencies"]["medical"]
        test = pyproject["project"]["optional-dependencies"]["test"]

        self.assertNotIn("wfdb>=4.1", dependencies)
        self.assertFalse(any(str(dep).startswith("pyedflib") for dep in dependencies))
        self.assertEqual(medical, ["wfdb>=4.1"])
        self.assertEqual(test, ["pytest>=8", "ruff>=0.12"])

    def test_project_uses_current_spdx_license_metadata(self) -> None:
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = pyproject["project"]

        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertFalse(any(str(item).startswith("License ::") for item in project["classifiers"]))

    def test_project_root_uses_environment_override(self) -> None:
        from genode.data import otflow_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            configured_root = Path(tmpdir) / "configured_root"
            with patch.dict(os.environ, {"GENODE_PROJECT_ROOT": str(configured_root)}, clear=False):
                self.assertEqual(otflow_paths.project_root(), configured_root.resolve())
                self.assertEqual(
                    otflow_paths.resolve_project_path("outputs/example"),
                    (configured_root / "outputs" / "example").resolve(),
                )

    def test_project_root_defaults_to_working_directory(self) -> None:
        from genode.data import otflow_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            working_root = Path(tmpdir) / "workspace"
            working_root.mkdir()
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(otflow_paths.Path, "cwd", return_value=working_root / "nested" / ".."):
                    self.assertEqual(otflow_paths.project_root(), working_root.resolve())
                    self.assertEqual(
                        otflow_paths.resolve_project_path("data/example"),
                        (working_root / "data" / "example").resolve(),
                    )

    def test_experiment_common_does_not_import_medical_dataset_builder(self) -> None:
        env = dict(os.environ)
        src_path = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        code = (
            "import json, sys\n"
            "import genode.data.experiment_common\n"
            "print(json.dumps('genode.data.otflow_medical_datasets' in sys.modules))\n"
        )

        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )

        self.assertEqual(json.loads(completed.stdout.strip()), False)

    def test_existing_long_term_st_manifest_loading_does_not_require_wfdb(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "series").mkdir()
            for series_name in ("s1", "s2", "s3"):
                np.save(root / "series" / f"{series_name}.npy", np.arange(32, dtype=np.float32))
            manifest = {
                "dataset_key": LONG_TERM_ST_DATASET_KEY,
                "history_len": 20,
                "future_block_len": 5,
                "series_specs": [
                    {
                        "series_id": "s1::ch0::ECG",
                        "record_id": "s1",
                        "group_id": "s1",
                        "channel_index": 0,
                        "channel_name": "ECG",
                        "file_name": "series/s1.npy",
                        "split": "train",
                        "total_length": 32,
                        "source_total_length": 80,
                    },
                    {
                        "series_id": "s2::ch0::ECG",
                        "record_id": "s2",
                        "group_id": "s2",
                        "channel_index": 0,
                        "channel_name": "ECG",
                        "file_name": "series/s2.npy",
                        "split": "val",
                        "total_length": 32,
                        "source_total_length": 80,
                    },
                    {
                        "series_id": "s3::ch0::ECG",
                        "record_id": "s3",
                        "group_id": "s3",
                        "channel_index": 0,
                        "channel_name": "ECG",
                        "file_name": "series/s3.npy",
                        "split": "test",
                        "total_length": 32,
                        "source_total_length": 80,
                    },
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            prepare_long_term_st_dataset = self._prepare_long_term_st_dataset()
            with patch("importlib.util.find_spec", return_value=None):
                loaded = prepare_long_term_st_dataset(root, history_len=20, horizon=5)

            self.assertEqual(loaded["dataset_key"], LONG_TERM_ST_DATASET_KEY)

    def test_raw_long_term_st_preparation_reports_missing_wfdb_extra(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_long_term_st_dataset = self._prepare_long_term_st_dataset()
            with patch("importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(ImportError, r"pip install -e \.\[medical\]"):
                    prepare_long_term_st_dataset(Path(tmpdir) / "prepared", archive_paths=[])

    def test_monash_archive_extraction_rejects_path_traversal(self) -> None:
        from genode.data.otflow_monash_datasets import _extract_zip

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "malicious.zip"
            for member_name in ("../escape.txt", "..\\escape.txt", "C:/escape.txt"):
                with self.subTest(member_name=member_name):
                    with zipfile.ZipFile(archive_path, "w") as archive:
                        archive.writestr(member_name, "blocked")
                    with self.assertRaisesRegex(ValueError, "Archive member"):
                        _extract_zip(archive_path, root / "source")
                    self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
