from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from genode.provenance import clear_provenance_cache, fingerprint_identity, path_fingerprint
from genode.gipo.train_gipo import _artifact_input_summary


class ProvenanceTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_provenance_cache()

    def test_file_identity_ignores_location_and_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir) / "input.json"
            second = Path(second_dir) / "input.json"
            first.write_text('{"value": 1}\n', encoding="utf-8")
            second.write_bytes(first.read_bytes())
            os.utime(second, (1, 1))

            self.assertEqual(
                fingerprint_identity(path_fingerprint(first)),
                fingerprint_identity(path_fingerprint(second)),
            )

    def test_file_identity_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "input.json"
            path.write_text('{"value": 1}\n', encoding="utf-8")
            before = fingerprint_identity(path_fingerprint(path))
            clear_provenance_cache()
            path.write_text('{"value": 2}\n', encoding="utf-8")
            after = fingerprint_identity(path_fingerprint(path))
            self.assertNotEqual(before, after)

    def test_directory_requires_and_hashes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "authoritative manifest"):
                path_fingerprint(root)
            (root / "manifest.json").write_text("{}\n", encoding="utf-8")
            fingerprint = path_fingerprint(root)
            self.assertEqual(fingerprint["kind"], "directory")
            self.assertEqual(fingerprint["manifests"][0]["name"], "manifest.json")

    def test_gipo_input_summary_uses_content_not_location_or_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir) / "rows.csv"
            second = Path(second_dir) / "rows.csv"
            first.write_text("scenario_key\ntraffic_hourly\n", encoding="utf-8")
            second.write_bytes(first.read_bytes())
            os.utime(second, (1, 1))

            first_summary = _artifact_input_summary(str(first))[0]
            second_summary = _artifact_input_summary(str(second))[0]
            self.assertEqual(
                {key: value for key, value in first_summary.items() if key != "logical_path"},
                {key: value for key, value in second_summary.items() if key != "logical_path"},
            )
            self.assertNotIn("mtime_ns", first_summary)
            self.assertNotIn(str(Path(first_dir).resolve()), first_summary["logical_path"])

            clear_provenance_cache()
            second.write_text("scenario_key\nweather_daily\n", encoding="utf-8")
            changed_summary = _artifact_input_summary(str(second))[0]
            self.assertNotEqual(first_summary["sha256"], changed_summary["sha256"])


if __name__ == "__main__":
    unittest.main()
