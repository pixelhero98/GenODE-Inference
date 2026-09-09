from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from genode.backbone_packages import _contains_local_marker, _validate_file_record
from genode.benchmarks.image import artifact_paths as image_artifact_paths
from genode.data import molecule_xyz, otflow_datasets, otflow_monash_datasets
from genode.evaluation import diffusion_flow_time_reparameterization as evaluation_runner
from genode.path_safety import portable_relative_path, resolve_portable_relative_path


class PortablePathTests(unittest.TestCase):
    def test_portable_relative_path_rejects_cross_platform_escapes(self) -> None:
        invalid = (
            "",
            "/absolute/path",
            "C:/absolute/path",
            "C:drive-relative",
            "\\\\server\\share\\path",
            "..",
            "../outside",
            "folder/../outside",
            "folder\\..\\outside",
            ".",
            "folder/./file",
            "folder//file",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                portable_relative_path(value)
        self.assertEqual(portable_relative_path("processed/member_a").as_posix(), "processed/member_a")

    def test_resolved_relative_path_rejects_existing_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            outside = Path(tmpdir) / "outside"
            root.mkdir()
            outside.mkdir()
            link = root / "linked"
            try:
                os.symlink(outside, link, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"Directory symlink creation is unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "escapes its declared root"):
                resolve_portable_relative_path(root, "linked/file.npy")

    def test_image_artifact_path_allows_trusted_symlinked_mount_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "storage"
            storage.mkdir()
            mount = root / "mount"
            try:
                mount.symlink_to(storage, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            target = image_artifact_paths.managed_image_leaf_directory(mount / "policy", label="image policy directory")
            self.assertEqual(target, storage.resolve() / "policy")

    def test_image_artifact_path_rejects_leaf_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "storage"
            storage.mkdir()
            target = root / "policy"
            try:
                target.symlink_to(storage, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                image_artifact_paths.managed_image_leaf_directory(target, label="image policy directory")


class MoleculeManifestPathTests(unittest.TestCase):
    def _write_manifest(self, root: Path, *, processed_dir: str, source_zip_name: str) -> None:
        manifest_path = root / "molecule_3d_set1" / "group_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "dataset_key": "molecule_3d_set1",
                    "source_zip_names": [source_zip_name],
                    "strata": [
                        {
                            "member_key": "member_a",
                            "stratum": "Dynamic_A",
                            "processed_dir": processed_dir,
                            "source_zip_name": source_zip_name,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_group_manifest_rejects_windows_and_posix_traversal(self) -> None:
        cases = (
            ("..\\outside", "trajectory.zip"),
            ("C:\\private", "trajectory.zip"),
            ("../outside", "trajectory.zip"),
            ("processed/member", "..\\trajectory.zip"),
            ("processed/member", "C:trajectory.zip"),
            ("processed/member", "folder/trajectory.zip"),
        )
        for processed_dir, source_zip_name in cases:
            with (
                self.subTest(processed_dir=processed_dir, source_zip_name=source_zip_name),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                self._write_manifest(root, processed_dir=processed_dir, source_zip_name=source_zip_name)
                with self.assertRaises(ValueError):
                    molecule_xyz.load_molecule_group_manifest("molecule_3d_set1", root)


class VerifiedDownloadTests(unittest.TestCase):
    def test_sha256_download_is_atomic_and_exact(self) -> None:
        payload = b"verified payload"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "payload.bin"
            with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(payload)):
                otflow_datasets._download_url_to_path(
                    "https://example.invalid/payload.bin",
                    destination,
                    expected_size=len(payload),
                    expected_sha256=digest,
                )
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(list(destination.parent.glob(f".{destination.name}.*.download")), [])

    def test_failed_sha256_download_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "payload.bin"
            destination.write_bytes(b"known-good-existing")
            with (
                mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"too large")),
                self.assertRaisesRegex(ValueError, "exceeded the expected size"),
            ):
                otflow_datasets._download_url_to_path(
                    "https://example.invalid/payload.bin",
                    destination,
                    expected_size=3,
                    expected_sha256=hashlib.sha256(b"abc").hexdigest(),
                )
            self.assertEqual(destination.read_bytes(), b"known-good-existing")
            self.assertEqual(list(destination.parent.glob(f".{destination.name}.*.download")), [])

    def test_md5_download_reuses_valid_file_and_rejects_bad_checksum(self) -> None:
        payload = b"zenodo bytes"
        digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "archive.zip"
            destination.write_bytes(payload)
            with mock.patch("urllib.request.urlopen") as urlopen:
                result = otflow_monash_datasets._download_file(
                    "https://example.invalid/archive.zip", destination, expected_size=len(payload), expected_md5=digest
                )
            self.assertEqual(result, destination)
            urlopen.assert_not_called()
            with (
                mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"wrong bytes")),
                self.assertRaisesRegex(ValueError, "MD5"),
            ):
                otflow_monash_datasets._download_file(
                    "https://example.invalid/archive.zip",
                    destination,
                    expected_size=len(b"wrong bytes"),
                    expected_md5=digest,
                )
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(list(destination.parent.glob(f".{destination.name}.*.download")), [])

    def test_monash_specs_pin_published_size_and_md5(self) -> None:
        observed = {
            spec.key: (int(spec.archive_size_bytes), str(spec.archive_md5))
            for spec in otflow_monash_datasets.MONASH_REFERENCE_DATASETS
        }
        self.assertEqual(
            observed,
            {
                "solar_energy_10m": (4559353, "84c0de18383c911091a3cd274661b029"),
                "traffic_hourly": (22868806, "1cf694f99f95700217845078b467fb24"),
                "weather_daily": (38820451, "57155594af0883ccd5e63a5948976796"),
            },
        )


class SafeZipExtractionTests(unittest.TestCase):
    def test_monash_zip_is_copied_member_by_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("nested/data.tsf", "@data\nseries:1,2,3\n")
            destination = root / "source"
            otflow_monash_datasets._extract_zip(archive_path, destination)
            self.assertEqual((destination / "nested" / "data.tsf").read_text(encoding="utf-8"), "@data\nseries:1,2,3\n")

    def test_monash_zip_allows_trusted_symlinked_mount_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "storage"
            storage.mkdir()
            mount = root / "mount"
            try:
                mount.symlink_to(storage, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            archive_path = root / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("nested/data.tsf", "@data\nseries:1,2,3\n")
            destination = mount / "source"
            otflow_monash_datasets._extract_zip(archive_path, destination)
            self.assertEqual(
                (storage / "source" / "nested" / "data.tsf").read_text(encoding="utf-8"), "@data\nseries:1,2,3\n"
            )

    def test_monash_zip_rejects_destination_that_is_itself_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("nested/data.tsf", "@data\nseries:1,2,3\n")
            storage = root / "storage"
            storage.mkdir()
            destination = root / "source"
            try:
                destination.symlink_to(storage, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "destination may not be a symlink"):
                otflow_monash_datasets._extract_zip(archive_path, destination)
            self.assertEqual(list(storage.iterdir()), [])

    def test_monash_zip_rejects_traversal_and_symbolic_links_before_copying(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for unsafe_name, symbolic_link in (("../outside.tsf", False), ("link.tsf", True)):
                with self.subTest(unsafe_name=unsafe_name, symbolic_link=symbolic_link):
                    archive_path = root / f"unsafe-{int(symbolic_link)}.zip"
                    with zipfile.ZipFile(archive_path, "w") as archive:
                        archive.writestr("safe/data.tsf", "preserve")
                        if symbolic_link:
                            info = zipfile.ZipInfo(unsafe_name)
                            info.create_system = 3
                            info.external_attr = (stat.S_IFLNK | 511) << 16
                            archive.writestr(info, "safe/data.tsf")
                        else:
                            archive.writestr(unsafe_name, "escape")
                    destination = root / f"source-{int(symbolic_link)}"
                    with self.assertRaises(ValueError):
                        otflow_monash_datasets._extract_zip(archive_path, destination)
                    self.assertFalse((root / "outside.tsf").exists())
                    self.assertFalse((destination / "safe" / "data.tsf").exists())


class PackageAttestationTests(unittest.TestCase):
    def test_file_records_require_size_and_sha256_attestations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = root / "payload.bin"
            payload.write_bytes(b"payload")
            errors = _validate_file_record(root, {"path": "payload.bin"})
        self.assertIn("Missing or invalid size_bytes for payload.bin", errors)
        self.assertIn("Missing or invalid SHA256 for payload.bin", errors)

    def test_embedded_local_paths_are_detected_in_free_text(self) -> None:
        local_text = (
            "loaded checkpoint C:\\Users\\person\\model.pt successfully",
            "loaded checkpoint \\\\server\\share\\model.pt successfully",
            "loaded checkpoint /srv/private/model.pt successfully",
        )
        for value in local_text:
            with self.subTest(value=value):
                self.assertTrue(_contains_local_marker(value))
        self.assertFalse(_contains_local_marker("loaded packaged checkpoint successfully"))
        self.assertFalse(_contains_local_marker("source https://github.com/example/project/blob/main/model.py"))


class RunnerOutputPathTests(unittest.TestCase):
    def test_runner_rejects_hostile_output_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            output_root.mkdir()
            hostile_names = (
                "../outside.jsonl",
                "..\\outside.jsonl",
                "/private/rows.jsonl",
                "C:\\private\\rows.jsonl",
                "\\\\server\\share\\rows.jsonl",
                ".",
            )
            for name in hostile_names:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    evaluation_runner._runner_output_path(
                        output_root, name, default="rows.jsonl", label="row JSONL name"
                    )
            self.assertFalse((Path(tmpdir) / "outside.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
