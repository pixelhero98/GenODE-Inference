from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from genode.data.otflow_medical_constants import LONG_TERM_ST_DATASET_KEY
from genode.data.otflow_medical_datasets import (
    LazyLongTermSTConditionalDataset,
    LongTermSTSeriesSpec,
    _validate_long_term_st_manifest,
    prepare_long_term_st_dataset,
)
from genode.data.experiment_common import OTFLOW_REFERENCE_DATASET_CHOICES, get_otflow_reference_backbone_preset
from genode.data.otflow_experiment_plan import REFERENCE_CONDITIONAL_GENERATION_DATASETS
from genode.evaluation.otflow_evaluation_support import (
    parse_conditional_generation_datasets,
)


def _write_wfdb_record(root: Path, record_id: str, *, length: int = 250) -> None:
    import wfdb

    t = np.linspace(0.0, 1.0, int(length), endpoint=False, dtype=np.float64)
    signal = np.stack(
        [
            np.sin(2.0 * np.pi * 5.0 * t),
            np.cos(2.0 * np.pi * 3.0 * t),
        ],
        axis=1,
    )
    wfdb.wrsamp(
        record_name=str(record_id),
        fs=250,
        units=["mV", "mV"],
        sig_name=["ML2", "MV2"],
        p_signal=signal,
        fmt=["16", "16"],
        write_dir=str(root),
    )


class LongTermSTTests(unittest.TestCase):
    def _write_minimal_manifest(self, root: Path, series_rows: list[dict]) -> Path:
        (root / "series").mkdir(parents=True, exist_ok=True)
        for row in series_rows:
            file_name = str(row["file_name"])
            if file_name.startswith("series/"):
                np.save(root / file_name, np.arange(32, dtype=np.float32))
        manifest = {
            "dataset_key": LONG_TERM_ST_DATASET_KEY,
            "history_len": 20,
            "future_block_len": 5,
            "series_specs": series_rows,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_long_term_st_is_default_conditional_generation(self) -> None:
        self.assertIn(LONG_TERM_ST_DATASET_KEY, REFERENCE_CONDITIONAL_GENERATION_DATASETS)
        self.assertIn(LONG_TERM_ST_DATASET_KEY, OTFLOW_REFERENCE_DATASET_CHOICES)
        self.assertEqual(parse_conditional_generation_datasets(LONG_TERM_ST_DATASET_KEY), [LONG_TERM_ST_DATASET_KEY])
        self.assertEqual(get_otflow_reference_backbone_preset(LONG_TERM_ST_DATASET_KEY)["future_block_len"], 3000)

    def test_lazy_long_term_st_dataset_shapes_and_segment_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "series").mkdir()
            np.save(root / "series" / "s1.npy", np.arange(30, dtype=np.float32))
            np.save(root / "series" / "s2.npy", np.arange(100, 130, dtype=np.float32))
            specs = [
                LongTermSTSeriesSpec("s1::ch0::ML2", "s1", "s1", 0, "ML2", "series/s1.npy", "train", 30, 75),
                LongTermSTSeriesSpec("s2::ch0::ML2", "s2", "s2", 0, "ML2", "series/s2.npy", "train", 30, 75),
            ]
            with LazyLongTermSTConditionalDataset(
                dataset_root=root,
                split_name="train",
                history_len=4,
                horizon=3,
                series_specs=specs,
                mean=10.0,
                std=2.0,
                stride=3,
            ) as ds:
                hist, tgt, fut, meta = ds[0]
                self.assertEqual(tuple(hist.shape), (4, 1))
                self.assertEqual(tuple(tgt.shape), (1,))
                self.assertEqual(tuple(fut.shape), (2, 1))
                self.assertIsNone(ds.cond)
                self.assertEqual(meta["dataset_kind"], LONG_TERM_ST_DATASET_KEY)
                np.testing.assert_array_equal(ds.segment_end_for_t(np.asarray([4, 31])), np.asarray([30, 60]))
                self.assertEqual(ds.params[4:7].shape, (3, 1))

    def test_manifest_validation_rejects_group_leakage_and_unsafe_series_paths(self) -> None:
        base_rows = [
            {
                "series_id": "a",
                "record_id": "s20011",
                "group_id": "same_patient",
                "channel_index": 0,
                "channel_name": "ECG",
                "file_name": "series/a.npy",
                "split": "train",
                "total_length": 32,
                "source_total_length": 80,
            },
            {
                "series_id": "b",
                "record_id": "s20021",
                "group_id": "same_patient",
                "channel_index": 0,
                "channel_name": "ECG",
                "file_name": "series/b.npy",
                "split": "val",
                "total_length": 32,
                "source_total_length": 80,
            },
            {
                "series_id": "c",
                "record_id": "s20031",
                "group_id": "c",
                "channel_index": 0,
                "channel_name": "ECG",
                "file_name": "series/c.npy",
                "split": "test",
                "total_length": 32,
                "source_total_length": 80,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_minimal_manifest(Path(tmpdir), base_rows)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                _validate_long_term_st_manifest(manifest_path, history_len=20, horizon=5)

        safe_rows = [dict(row, group_id=f"group_{idx}") for idx, row in enumerate(base_rows)]
        unsafe_rows = [dict(row) for row in safe_rows]
        unsafe_rows[0]["file_name"] = "../outside.npy"
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_minimal_manifest(Path(tmpdir), unsafe_rows)
            with self.assertRaisesRegex(ValueError, r"may not contain.*'\.\.'"):
                _validate_long_term_st_manifest(manifest_path, history_len=20, horizon=5)

        patient_rows = [dict(row) for row in safe_rows]
        patient_rows[0]["record_id"] = "s20271"
        patient_rows[0]["group_id"] = "s20271"
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_minimal_manifest(Path(tmpdir), patient_rows)
            with self.assertRaisesRegex(ValueError, "same-patient group"):
                _validate_long_term_st_manifest(manifest_path, history_len=20, horizon=5)

    @unittest.skipUnless(importlib.util.find_spec("wfdb") is not None, "wfdb is required")
    def test_zip_preparation_skips_short_records_and_sanitizes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "raw"
            raw.mkdir()
            for record_id in ("s20011", "s20021", "s20031", "s20041"):
                _write_wfdb_record(raw, record_id)
            short_dat = raw / "s20041.dat"
            short_dat.write_bytes(short_dat.read_bytes()[: max(1, short_dat.stat().st_size // 3)])

            archive = root / "long_term_st_test.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                for path in sorted(raw.iterdir()):
                    zf.write(path, arcname=f"long_term_st/{path.name}")

            out_dir = root / "prepared"
            with patch.dict(os.environ, {"OTFLOW_MEDICAL_STAGING_ROOT": str(root / "staging")}):
                manifest = prepare_long_term_st_dataset(
                    out_dir,
                    archive_paths=[archive],
                    force=True,
                    expected_record_count=None,
                    history_len=20,
                    horizon=5,
                )

            self.assertEqual(manifest["dataset_key"], LONG_TERM_ST_DATASET_KEY)
            self.assertEqual(manifest["n_headers"], 4)
            self.assertEqual(manifest["n_records_used"], 3)
            self.assertGreaterEqual(manifest["n_records_skipped"], 1)
            self.assertEqual(manifest["conditioning"], "context_only")
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertEqual(set(manifest["split_counts"]), {"train", "val", "test"})
            encoded = json.dumps(manifest)
            self.assertNotIn(str(root), encoded)
            self.assertNotIn("comments", encoded.lower())
            self.assertFalse(any("s30751_full" in row["file_name"] for row in manifest["series_specs"]))


if __name__ == "__main__":
    unittest.main()
