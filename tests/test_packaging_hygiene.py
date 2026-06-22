from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
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

        self.assertNotIn("wfdb>=4.1", dependencies)
        self.assertFalse(any(str(dep).startswith("pyedflib") for dep in dependencies))
        self.assertEqual(medical, ["wfdb>=4.1"])

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
            np.save(root / "series" / "s1.npy", np.arange(32, dtype=np.float32))
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
                        "file_name": "series/s1.npy",
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
                        "file_name": "series/s1.npy",
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


if __name__ == "__main__":
    unittest.main()
