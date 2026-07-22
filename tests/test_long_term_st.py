from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from genode.data import otflow_medical_datasets as medical_module
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


def _write_wfdb_archive(root: Path, record_ids: tuple[str, ...]) -> Path:
    raw = root / "raw"
    raw.mkdir()
    for record_id in record_ids:
        _write_wfdb_record(raw, record_id)
    archive = root / "long_term_st_test.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(raw.iterdir()):
            zf.write(path, arcname=f"long_term_st/{path.name}")
    return archive


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _materialize_interrupted_move(source: Path, destination: Path) -> None:
    """Create a post-rename directory state without relying on Windows rename timing."""

    shutil.copytree(source, destination)
    shutil.rmtree(source)


def _hold_long_term_st_target_lock(
    prepared_dir: str,
    ready,
    release,
) -> None:
    with medical_module._long_term_st_target_lock(Path(prepared_dir)):
        ready.set()
        if not release.wait(timeout=30.0):
            raise RuntimeError("Timed out waiting to release Long-Term ST test lock.")


def _prepare_test_dataset(root: Path, archive: Path, prepared: Path, *, force: bool) -> dict:
    with patch.dict(os.environ, {"OTFLOW_MEDICAL_STAGING_ROOT": str(root / "staging")}):
        return prepare_long_term_st_dataset(
            prepared,
            archive_paths=[archive],
            force=force,
            expected_record_count=None,
            history_len=20,
            horizon=5,
        )


class LongTermSTTests(unittest.TestCase):
    def test_target_lock_rejects_competing_process_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared = root / "prepared"
            context = multiprocessing.get_context("spawn")
            ready = context.Event()
            release = context.Event()
            process = context.Process(
                target=_hold_long_term_st_target_lock,
                args=(str(prepared), ready, release),
            )
            process.start()
            try:
                self.assertTrue(ready.wait(timeout=30.0))
                with (
                    patch.object(
                        medical_module,
                        "_recover_long_term_st_promotion",
                    ) as recover,
                    self.assertRaisesRegex(RuntimeError, "locked by another"),
                ):
                    prepare_long_term_st_dataset(
                        prepared,
                        archive_paths=[],
                        force=True,
                    )
                recover.assert_not_called()
                self.assertFalse(prepared.exists())
            finally:
                release.set()
                process.join(timeout=30.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10.0)

            self.assertEqual(process.exitcode, 0)
            lock_path = medical_module._long_term_st_lock_path(prepared)
            self.assertTrue(lock_path.is_file())
            with medical_module._long_term_st_target_lock(prepared):
                pass

    def test_target_lock_refuses_and_preserves_unknown_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared = Path(tmpdir) / "prepared"
            lock_path = medical_module._long_term_st_lock_path(prepared)
            lock_path.mkdir()
            marker = lock_path / "keep.txt"
            marker.write_text("preserve me", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "lock path must be a regular file"):
                prepare_long_term_st_dataset(
                    prepared,
                    archive_paths=[],
                    force=True,
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")

    def test_target_lock_preserves_existing_regular_lock_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared = Path(tmpdir) / "prepared"
            lock_path = medical_module._long_term_st_lock_path(prepared)
            lock_path.write_bytes(b"")

            with medical_module._long_term_st_target_lock(prepared):
                self.assertTrue(lock_path.is_file())

            self.assertEqual(lock_path.read_bytes(), b"")

    def test_preparation_rejects_indirect_ancestor_before_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            destination = root / "junction" / "nested" / "prepared"
            with (
                patch(
                    "genode.data.otflow_medical_datasets.first_link_or_reparse_component",
                    return_value=root / "junction",
                ),
                patch.object(Path, "mkdir") as mkdir,
                self.assertRaisesRegex(ValueError, "may not traverse"),
            ):
                prepare_long_term_st_dataset(destination)
            mkdir.assert_not_called()

    def test_extraction_rejects_indirect_ancestor_before_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            destination = root / "junction" / "nested" / "extracted"
            with (
                patch(
                    "genode.data.otflow_medical_datasets.first_link_or_reparse_component",
                    return_value=root / "junction",
                ),
                patch.object(Path, "mkdir") as mkdir,
                self.assertRaisesRegex(ValueError, "may not traverse"),
            ):
                medical_module._extract_long_term_st_wfdb_members(
                    source_dir=destination,
                    archive_paths=(),
                    headers={},
                    dat_members={},
                )
            mkdir.assert_not_called()

    def _write_minimal_manifest(
        self,
        root: Path,
        series_rows: list[dict],
        *,
        history_len: int = 20,
        horizon: int = 5,
    ) -> Path:
        (root / "series").mkdir(parents=True, exist_ok=True)
        for row in series_rows:
            file_name = str(row["file_name"])
            if file_name.startswith("series/"):
                np.save(root / file_name, np.arange(32, dtype=np.float32))
        manifest = {
            "dataset_key": LONG_TERM_ST_DATASET_KEY,
            "history_len": int(history_len),
            "future_block_len": int(horizon),
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

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = self._write_minimal_manifest(root, safe_rows)
            np.save(root / "series" / "unreferenced.npy", np.arange(8, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "unmanaged file"):
                _validate_long_term_st_manifest(manifest_path, history_len=20, horizon=5)

    @unittest.skipUnless(importlib.util.find_spec("wfdb") is not None, "wfdb is required")
    def test_zip_preparation_propagates_output_write_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = _write_wfdb_archive(root, ("s20011", "s20021", "s20031"))
            prepared = root / "prepared"

            with (
                patch.dict(os.environ, {"OTFLOW_MEDICAL_STAGING_ROOT": str(root / "staging")}),
                patch(
                    "genode.data.otflow_medical_datasets.np.save",
                    side_effect=OSError("simulated write failure"),
                ),
                self.assertRaisesRegex(OSError, "simulated write failure"),
            ):
                prepare_long_term_st_dataset(
                    prepared,
                    archive_paths=[archive],
                    force=True,
                    expected_record_count=None,
                    history_len=20,
                    horizon=5,
                )

            self.assertFalse(prepared.exists())
            self.assertFalse(list(root.glob(".prepared.staging-*")))
            self.assertFalse(list(root.glob(".prepared.backup-*")))

    @unittest.skipUnless(importlib.util.find_spec("wfdb") is not None, "wfdb is required")
    def test_forced_preparation_failure_preserves_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = _write_wfdb_archive(root, ("s20011", "s20021", "s20031"))
            prepared = root / "prepared"
            _prepare_test_dataset(root, archive, prepared, force=True)
            before = _tree_snapshot(prepared)

            with (
                patch(
                    "genode.data.otflow_medical_datasets.np.save",
                    side_effect=OSError("simulated rebuild failure"),
                ),
                self.assertRaisesRegex(OSError, "simulated rebuild failure"),
            ):
                _prepare_test_dataset(root, archive, prepared, force=True)

            self.assertEqual(_tree_snapshot(prepared), before)
            self.assertFalse(list(root.glob(".prepared.staging-*")))
            self.assertFalse(list(root.glob(".prepared.backup-*")))

    def test_forced_preparation_rebuilds_for_new_task_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared = root / "prepared"
            rows = [
                {
                    "series_id": split,
                    "record_id": split,
                    "group_id": split,
                    "channel_index": 0,
                    "channel_name": "ECG",
                    "file_name": f"series/{split}.npy",
                    "split": split,
                    "total_length": 32,
                    "source_total_length": 80,
                }
                for split in ("train", "val", "test")
            ]
            self._write_minimal_manifest(prepared, rows)

            def rebuild(staging_dir: Path, **kwargs) -> None:
                self._write_minimal_manifest(
                    staging_dir,
                    rows,
                    history_len=int(kwargs["history_len"]),
                    horizon=int(kwargs["horizon"]),
                )

            with patch(
                "genode.data.otflow_medical_datasets._prepare_long_term_st_dataset_into",
                side_effect=rebuild,
            ) as prepare_into:
                manifest = prepare_long_term_st_dataset(
                    prepared,
                    force=True,
                    history_len=24,
                    horizon=6,
                )

            prepare_into.assert_called_once()
            self.assertEqual(manifest["history_len"], 24)
            self.assertEqual(manifest["future_block_len"], 6)

    def test_forced_preparation_refuses_unmanaged_series_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared = root / "prepared"
            rows = [
                {
                    "series_id": split,
                    "record_id": split,
                    "group_id": split,
                    "channel_index": 0,
                    "channel_name": "ECG",
                    "file_name": f"series/{split}.npy",
                    "split": split,
                    "total_length": 32,
                    "source_total_length": 80,
                }
                for split in ("train", "val", "test")
            ]
            self._write_minimal_manifest(prepared, rows)
            unmanaged = prepared / "series" / "unmanaged.npy"
            np.save(unmanaged, np.arange(8, dtype=np.float32))
            before = _tree_snapshot(prepared)

            with (
                patch(
                    "genode.data.otflow_medical_datasets._prepare_long_term_st_dataset_into"
                ) as prepare_into,
                self.assertRaisesRegex(ValueError, "unmanaged file"),
            ):
                prepare_long_term_st_dataset(
                    prepared,
                    force=True,
                    history_len=24,
                    horizon=6,
                )

            prepare_into.assert_not_called()
            self.assertEqual(_tree_snapshot(prepared), before)

    @unittest.skipUnless(importlib.util.find_spec("wfdb") is not None, "wfdb is required")
    def test_promotion_failure_restores_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = _write_wfdb_archive(root, ("s20011", "s20021", "s20031"))
            prepared = root / "prepared"
            _prepare_test_dataset(root, archive, prepared, force=True)
            before = _tree_snapshot(prepared)
            real_replace = medical_module._replace_long_term_st_path

            def fail_stage_promotion(source, destination):
                source_path = Path(source)
                destination_path = Path(destination)
                if source_path.name.startswith(".prepared.staging-") and destination_path == prepared:
                    raise OSError("simulated promotion failure")
                return real_replace(source_path, destination_path)

            with (
                patch.object(
                    medical_module,
                    "_replace_long_term_st_path",
                    side_effect=fail_stage_promotion,
                ),
                self.assertRaisesRegex(OSError, "simulated promotion failure"),
            ):
                _prepare_test_dataset(root, archive, prepared, force=True)

            self.assertEqual(_tree_snapshot(prepared), before)
            self.assertFalse(list(root.glob(".prepared.staging-*")))
            self.assertFalse(list(root.glob(".prepared.backup-*")))

    def test_promotion_restart_restores_backup_after_target_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared = root / "prepared"
            staging = root / ".prepared.staging-interrupted"
            rows = [
                {
                    "series_id": split,
                    "record_id": split,
                    "group_id": split,
                    "channel_index": 0,
                    "channel_name": "ECG",
                    "file_name": f"series/{split}.npy",
                    "split": split,
                    "total_length": 32,
                    "source_total_length": 80,
                }
                for split in ("train", "val", "test")
            ]
            self._write_minimal_manifest(prepared, rows)
            self._write_minimal_manifest(
                staging,
                rows,
                history_len=24,
                horizon=6,
            )

            record = medical_module._prepare_long_term_st_promotion_journal(
                staging,
                prepared,
            )
            backup = prepared.parent / str(record["backup_name"])
            _materialize_interrupted_move(prepared, backup)

            manifest = prepare_long_term_st_dataset(
                prepared,
                history_len=20,
                horizon=5,
            )

            self.assertEqual(manifest["history_len"], 20)
            self.assertEqual(manifest["future_block_len"], 5)
            self.assertFalse(staging.exists())
            self.assertFalse(backup.exists())
            self.assertFalse(
                medical_module._long_term_st_promotion_journal_path(prepared).exists()
            )

    def test_promotion_syncs_complete_staging_tree_before_journal_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared = root / "prepared"
            staging = root / ".prepared.staging-sync"
            rows = [
                {
                    "series_id": split,
                    "record_id": split,
                    "group_id": split,
                    "channel_index": 0,
                    "channel_name": "ECG",
                    "file_name": f"series/{split}.npy",
                    "split": split,
                    "total_length": 32,
                    "source_total_length": 80,
                }
                for split in ("train", "val", "test")
            ]
            self._write_minimal_manifest(prepared, rows)
            self._write_minimal_manifest(staging, rows)
            events: list[tuple[str, str]] = []
            real_file_hash = medical_module._managed_long_term_st_file_sha256
            real_sync_directory = medical_module._sync_long_term_st_directory
            real_write_journal = medical_module._write_long_term_st_promotion_journal

            def tracked_file_hash(path: Path, *, sync: bool) -> str:
                if sync:
                    events.append(("file", path.relative_to(staging).as_posix()))
                return real_file_hash(path, sync=sync)

            def tracked_sync_directory(path: Path) -> None:
                events.append(("directory", str(path)))
                real_sync_directory(path)

            def tracked_write_journal(path: Path, record) -> None:
                events.append(("journal", str(path)))
                real_write_journal(path, record)

            with (
                patch.object(
                    medical_module,
                    "_managed_long_term_st_file_sha256",
                    side_effect=tracked_file_hash,
                ),
                patch.object(
                    medical_module,
                    "_sync_long_term_st_directory",
                    side_effect=tracked_sync_directory,
                ),
                patch.object(
                    medical_module,
                    "_write_long_term_st_promotion_journal",
                    side_effect=tracked_write_journal,
                ),
            ):
                medical_module._prepare_long_term_st_promotion_journal(
                    staging,
                    prepared,
                )

            journal_index = next(
                index for index, event in enumerate(events) if event[0] == "journal"
            )
            synced_files = {
                value for kind, value in events[:journal_index] if kind == "file"
            }
            self.assertEqual(
                synced_files,
                {
                    "manifest.json",
                    "series/train.npy",
                    "series/val.npy",
                    "series/test.npy",
                },
            )
            synced_directories = {
                value for kind, value in events[:journal_index] if kind == "directory"
            }
            self.assertIn(str(staging / "series"), synced_directories)
            self.assertIn(str(staging), synced_directories)
            self.assertIn(str(root), synced_directories)

    def test_promotion_restart_finishes_installed_staging_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared = root / "prepared"
            staging = root / ".prepared.staging-interrupted"
            unrelated_backup = root / ".prepared.backup-unrelated"
            unrelated_backup.mkdir()
            unrelated_file = unrelated_backup / "notes.txt"
            unrelated_file.write_text("preserve me", encoding="utf-8")
            rows = [
                {
                    "series_id": split,
                    "record_id": split,
                    "group_id": split,
                    "channel_index": 0,
                    "channel_name": "ECG",
                    "file_name": f"series/{split}.npy",
                    "split": split,
                    "total_length": 32,
                    "source_total_length": 80,
                }
                for split in ("train", "val", "test")
            ]
            self._write_minimal_manifest(prepared, rows)
            self._write_minimal_manifest(
                staging,
                rows,
                history_len=24,
                horizon=6,
            )

            record = medical_module._prepare_long_term_st_promotion_journal(
                staging,
                prepared,
            )
            backup = prepared.parent / str(record["backup_name"])
            _materialize_interrupted_move(prepared, backup)
            _materialize_interrupted_move(staging, prepared)

            manifest = prepare_long_term_st_dataset(
                prepared,
                history_len=24,
                horizon=6,
            )

            self.assertEqual(manifest["history_len"], 24)
            self.assertEqual(manifest["future_block_len"], 6)
            self.assertFalse(backup.exists())
            self.assertFalse(
                medical_module._long_term_st_promotion_journal_path(prepared).exists()
            )
            self.assertEqual(unrelated_file.read_text(encoding="utf-8"), "preserve me")

    def test_promotion_keeps_journal_until_obsolete_cleanup_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared = root / "prepared"
            staging = root / ".prepared.staging-interrupted"
            rows = [
                {
                    "series_id": split,
                    "record_id": split,
                    "group_id": split,
                    "channel_index": 0,
                    "channel_name": "ECG",
                    "file_name": f"series/{split}.npy",
                    "split": split,
                    "total_length": 32,
                    "source_total_length": 80,
                }
                for split in ("train", "val", "test")
            ]
            self._write_minimal_manifest(prepared, rows)
            self._write_minimal_manifest(
                staging,
                rows,
                history_len=24,
                horizon=6,
            )
            record = medical_module._prepare_long_term_st_promotion_journal(
                staging,
                prepared,
            )
            backup = prepared.parent / str(record["backup_name"])
            _materialize_interrupted_move(prepared, backup)
            _materialize_interrupted_move(staging, prepared)
            journal = medical_module._long_term_st_promotion_journal_path(prepared)
            real_discard = medical_module._discard_obsolete_long_term_st_artifact

            def interrupt_cleanup(path: Path, expected_sha256: str) -> None:
                self.assertTrue(journal.exists())
                raise KeyboardInterrupt("simulated cleanup interruption")

            with (
                patch.object(
                    medical_module,
                    "_discard_obsolete_long_term_st_artifact",
                    side_effect=interrupt_cleanup,
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "cleanup interruption"),
            ):
                medical_module._recover_long_term_st_promotion(prepared)

            self.assertTrue(journal.exists())
            self.assertTrue(backup.exists())
            with patch.object(
                medical_module,
                "_discard_obsolete_long_term_st_artifact",
                side_effect=real_discard,
            ):
                medical_module._recover_long_term_st_promotion(prepared)
            self.assertFalse(journal.exists())
            self.assertFalse(backup.exists())

    def test_promotion_recovery_preserves_unmanaged_backup_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared = root / "prepared"
            staging = root / ".prepared.staging-interrupted"
            rows = [
                {
                    "series_id": split,
                    "record_id": split,
                    "group_id": split,
                    "channel_index": 0,
                    "channel_name": "ECG",
                    "file_name": f"series/{split}.npy",
                    "split": split,
                    "total_length": 32,
                    "source_total_length": 80,
                }
                for split in ("train", "val", "test")
            ]
            self._write_minimal_manifest(prepared, rows)
            self._write_minimal_manifest(
                staging,
                rows,
                history_len=24,
                horizon=6,
            )

            record = medical_module._prepare_long_term_st_promotion_journal(
                staging,
                prepared,
            )
            backup = prepared.parent / str(record["backup_name"])
            _materialize_interrupted_move(prepared, backup)
            _materialize_interrupted_move(staging, prepared)
            unmanaged = backup / "unmanaged.txt"
            unmanaged.write_text("preserve me", encoding="utf-8")

            manifest = prepare_long_term_st_dataset(
                prepared,
                history_len=24,
                horizon=6,
            )

            self.assertEqual(manifest["history_len"], 24)
            self.assertEqual(unmanaged.read_text(encoding="utf-8"), "preserve me")
            self.assertTrue(backup.is_dir())
            self.assertTrue(prepared.is_dir())
            self.assertFalse(
                medical_module._long_term_st_promotion_journal_path(prepared).exists()
            )

    def test_promotion_restart_tolerates_partially_cleaned_obsolete_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared = root / "prepared"
            staging = root / ".prepared.staging-interrupted"
            rows = [
                {
                    "series_id": split,
                    "record_id": split,
                    "group_id": split,
                    "channel_index": 0,
                    "channel_name": "ECG",
                    "file_name": f"series/{split}.npy",
                    "split": split,
                    "total_length": 32,
                    "source_total_length": 80,
                }
                for split in ("train", "val", "test")
            ]
            self._write_minimal_manifest(prepared, rows)
            self._write_minimal_manifest(
                staging,
                rows,
                history_len=24,
                horizon=6,
            )
            record = medical_module._prepare_long_term_st_promotion_journal(
                staging,
                prepared,
            )
            backup = prepared.parent / str(record["backup_name"])
            _materialize_interrupted_move(prepared, backup)
            _materialize_interrupted_move(staging, prepared)
            removed_backup_file = backup / "series" / "train.npy"
            removed_backup_file.unlink()

            manifest = prepare_long_term_st_dataset(
                prepared,
                history_len=24,
                horizon=6,
            )

            self.assertEqual(manifest["history_len"], 24)
            self.assertTrue(backup.is_dir())
            self.assertFalse(removed_backup_file.exists())
            self.assertFalse(
                medical_module._long_term_st_promotion_journal_path(prepared).exists()
            )

    def test_promotion_recovery_never_deletes_backup_for_tampered_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared = root / "prepared"
            staging = root / ".prepared.staging-interrupted"
            rows = [
                {
                    "series_id": split,
                    "record_id": split,
                    "group_id": split,
                    "channel_index": 0,
                    "channel_name": "ECG",
                    "file_name": f"series/{split}.npy",
                    "split": split,
                    "total_length": 32,
                    "source_total_length": 80,
                }
                for split in ("train", "val", "test")
            ]
            self._write_minimal_manifest(prepared, rows)
            self._write_minimal_manifest(
                staging,
                rows,
                history_len=24,
                horizon=6,
            )
            record = medical_module._prepare_long_term_st_promotion_journal(
                staging,
                prepared,
            )
            backup = prepared.parent / str(record["backup_name"])
            _materialize_interrupted_move(prepared, backup)
            _materialize_interrupted_move(staging, prepared)
            backup_before = _tree_snapshot(backup)
            (prepared / "series" / "train.npy").write_bytes(b"truncated")

            with self.assertRaisesRegex(ValueError, "do not match a safe state"):
                prepare_long_term_st_dataset(
                    prepared,
                    history_len=24,
                    horizon=6,
                )

            self.assertEqual(_tree_snapshot(backup), backup_before)
            self.assertTrue(
                medical_module._long_term_st_promotion_journal_path(prepared).is_file()
            )

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
