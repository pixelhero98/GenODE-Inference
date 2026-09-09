from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from genode.provenance import fingerprint_identity, path_fingerprint


class ProvenanceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
