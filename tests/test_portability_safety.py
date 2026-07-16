from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from genode.backbone_packages import _contains_local_marker, _validate_file_record
from genode.data import molecule_xyz, otflow_datasets, otflow_medical_datasets, otflow_monash_datasets
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
            with self.subTest(processed_dir=processed_dir, source_zip_name=source_zip_name):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    self._write_manifest(root, processed_dir=processed_dir, source_zip_name=source_zip_name)
                    with self.assertRaises(ValueError):
                        molecule_xyz.load_molecule_group_manifest("molecule_3d_set1", root)


class VerifiedDownloadTests(unittest.TestCase):
    def test_lobiflow_urls_are_pinned_to_the_declared_revision(self) -> None:
        revision = "2d33cfd6b5e27d2483e2095b22d340813389cd0c"
        self.assertIn(f"/resolve/{revision}/", otflow_datasets.LOBIFLOW_CRYPTOS_NPZ_URL)
        self.assertIn(f"/resolve/{revision}/", otflow_datasets.LOBIFLOW_SYNTHETIC_PROFILE_URL)
        self.assertEqual(otflow_datasets.LOBIFLOW_CRYPTOS_NPZ_SIZE_BYTES, 1_962_160_259)
        self.assertEqual(
            otflow_datasets.LOBIFLOW_CRYPTOS_NPZ_SHA256,
            "124fff5767387373323fcb0ec17cc8b8030fe945d037909786127de6d3942e67",
        )
        self.assertEqual(otflow_datasets.LOBIFLOW_SYNTHETIC_PROFILE_SIZE_BYTES, 7_220)
        self.assertEqual(
            otflow_datasets.LOBIFLOW_SYNTHETIC_PROFILE_SHA256,
            "f92d3ffa3ef3bdbb67d8d45a337328b032727580a89177f967353dccbb40d50f",
        )

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
            with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"too large")):
                with self.assertRaisesRegex(ValueError, "exceeded the expected size"):
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
                    "https://example.invalid/archive.zip",
                    destination,
                    expected_size=len(payload),
                    expected_md5=digest,
                )
            self.assertEqual(result, destination)
            urlopen.assert_not_called()

            with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"wrong bytes")):
                with self.assertRaisesRegex(ValueError, "MD5"):
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
            for spec in otflow_monash_datasets.MONASH_PAPER_DATASETS
        }
        self.assertEqual(
            observed,
            {
                "solar_energy_10m": (4_559_353, "84c0de18383c911091a3cd274661b029"),
                "traffic_hourly": (22_868_806, "1cf694f99f95700217845078b467fb24"),
                "weather_daily": (38_820_451, "57155594af0883ccd5e63a5948976796"),
            },
        )


class MedicalChannelPathTests(unittest.TestCase):
    def test_record_slugs_are_ascii_reserved_safe_and_collision_stable(self) -> None:
        used: set[str] = set()
        first = otflow_medical_datasets._safe_record_name("record", record_index=0, used_slugs=used)
        collision = otflow_medical_datasets._safe_record_name("RECORD", record_index=3, used_slugs=used)
        reserved = otflow_medical_datasets._safe_record_name("CON", record_index=4, used_slugs=used)
        hostile = otflow_medical_datasets._safe_record_name("../patient\\name", record_index=5, used_slugs=used)

        self.assertEqual(first, "record")
        self.assertEqual(collision, "RECORD_3")
        self.assertEqual(reserved, "channel_CON")
        self.assertEqual(hostile, "patient_name")
        for slug in (first, collision, reserved, hostile):
            self.assertRegex(slug, r"^[A-Za-z0-9._-]+$")
            self.assertNotIn("..", slug)

    def test_channel_slugs_are_ascii_reserved_safe_and_collision_stable(self) -> None:
        used: set[str] = set()
        first = otflow_medical_datasets._safe_channel_name("ECG", channel_index=0, used_slugs=used)
        second = otflow_medical_datasets._safe_channel_name("ecg", channel_index=1, used_slugs=used)
        reserved = otflow_medical_datasets._safe_channel_name("CON", channel_index=2, used_slugs=used)
        hostile = otflow_medical_datasets._safe_channel_name("../lead\\name", channel_index=3, used_slugs=used)
        self.assertEqual(first, "ECG")
        self.assertEqual(second, "ecg_1")
        self.assertEqual(reserved, "channel_CON")
        self.assertEqual(hostile, "lead_name")
        for slug in (first, second, reserved, hostile):
            self.assertRegex(slug, r"^[A-Za-z0-9._-]+$")
            self.assertNotIn("..", slug)

    def test_malicious_header_channel_labels_cannot_escape_prepared_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "long_term_st_demo.zip"
            channel_names = ("ECG", "ecg", "CON", "../lead\\outside")
            with zipfile.ZipFile(archive_path, "w") as archive:
                for record_id in ("record_a", "record_b", "record_c"):
                    signal_lines = "\n".join(
                        f"{record_id}_{index}.dat 16 1/mV 16 0 0 0 {name}"
                        for index, name in enumerate(channel_names)
                    )
                    header = f"{record_id} {len(channel_names)} 250 10\n{signal_lines}\n"
                    archive.writestr(f"nested/{record_id}.hea", header)
                    for index in range(len(channel_names)):
                        archive.writestr(f"nested/{record_id}_{index}.dat", b"data")

            class _Record:
                def __init__(self, length: int):
                    self.p_signal = np.arange(length, dtype=np.float32)[:, None]

            def _rdrecord(_path: str, *, sampfrom: int = 0, sampto: int | None = None, channels=None):
                del channels
                stop = 10 if sampto is None else int(sampto)
                return _Record(max(0, stop - int(sampfrom)))

            fake_wfdb = types.SimpleNamespace(rdrecord=_rdrecord)
            prepared_dir = root / "prepared"
            with (
                mock.patch.object(otflow_medical_datasets, "_require_wfdb_for_long_term_st_preparation"),
                mock.patch.dict(sys.modules, {"wfdb": fake_wfdb}),
                mock.patch("scipy.signal.resample_poly", side_effect=lambda values, _up, _down: values),
                mock.patch.dict(os.environ, {"OTFLOW_MEDICAL_STAGING_ROOT": str(root / "staging")}),
            ):
                payload = otflow_medical_datasets.prepare_long_term_st_dataset(
                    prepared_dir,
                    archive_paths=[archive_path],
                    force=True,
                    expected_record_count=3,
                    history_len=2,
                    horizon=1,
                    train_frac=1 / 3,
                    val_frac=1 / 3,
                )

            self.assertEqual(len(payload["series_specs"]), 12)
            for row in payload["series_specs"]:
                file_name = str(row["file_name"])
                resolved = resolve_portable_relative_path(prepared_dir, file_name, reject_links=True)
                self.assertTrue(resolved.is_file())
                self.assertTrue(resolved.is_relative_to(prepared_dir))
            self.assertFalse((root / "outside.npy").exists())

    def test_preparation_rejects_reparse_point_destination_before_reading_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "prepared"
            with mock.patch.object(otflow_medical_datasets, "is_link_or_reparse_point", return_value=True):
                with self.assertRaisesRegex(ValueError, "prepared destination.*reparse point"):
                    otflow_medical_datasets.prepare_long_term_st_dataset(
                        destination,
                        archive_paths=[],
                        force=True,
                    )

    def test_zip_member_copy_rejects_existing_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "source.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("nested/record.hea", "replacement")

            target_root = root / "extracted"
            target_root.mkdir()
            outside = root / "outside.hea"
            outside.write_text("preserve", encoding="utf-8")
            target = target_root / "record.hea"
            try:
                os.symlink(outside, target)
            except OSError as exc:
                self.skipTest(f"File symlink creation is unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "escapes its declared root|symlink"):
                otflow_medical_datasets._copy_zip_member(
                    (archive_path, "nested/record.hea"),
                    target_root=target_root,
                    target_name="record.hea",
                )

            self.assertEqual(outside.read_text(encoding="utf-8"), "preserve")


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
            r"loaded checkpoint C:\Users\person\model.pt successfully",
            r"loaded checkpoint \\server\share\model.pt successfully",
            "loaded checkpoint /srv/private/model.pt successfully",
        )
        for value in local_text:
            with self.subTest(value=value):
                self.assertTrue(_contains_local_marker(value))
        self.assertFalse(_contains_local_marker("loaded packaged checkpoint successfully"))


class RunnerOutputPathTests(unittest.TestCase):
    def test_runner_rejects_hostile_output_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            output_root.mkdir()
            hostile_names = (
                "../outside.jsonl",
                r"..\outside.jsonl",
                "/private/rows.jsonl",
                r"C:\private\rows.jsonl",
                r"\\server\share\rows.jsonl",
                ".",
            )
            for name in hostile_names:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    evaluation_runner._runner_output_path(
                        output_root,
                        name,
                        default="rows.jsonl",
                        label="row JSONL name",
                    )

            self.assertFalse((Path(tmpdir) / "outside.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
