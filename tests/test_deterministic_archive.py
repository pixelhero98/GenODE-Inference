from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from pathlib import Path

import pytest

import genode.deterministic_archive as deterministic_archive
from genode import artifact_bundle
from genode.artifact_bundle import bundle_journal_path
from genode.deterministic_archive import (
    ARCHIVE_FILE_MODE,
    ARCHIVE_MANIFEST_NAME,
    ARCHIVE_TIMESTAMP,
    ArchiveEntry,
    validate_deterministic_zip,
    write_deterministic_zip,
)


def _archive_bundle_paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output,
        output.with_suffix(output.suffix + ".manifest.json"),
        output.with_suffix(output.suffix + ".sha256"),
    )


def _assert_archive_bundle_complete(output: Path) -> None:
    archive_path, manifest_path, sha256_path = _archive_bundle_paths(output)
    digest = deterministic_archive.sha256_file(archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        manifest_bytes = archive.read(ARCHIVE_MANIFEST_NAME)
        manifest = json.loads(manifest_bytes)
    sidecar = json.loads(manifest_path.read_bytes())
    assert sidecar == {
        "archive": archive_path.name,
        "bundle_kind": manifest["bundle_kind"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "schema_version": deterministic_archive.ARCHIVE_SCHEMA_VERSION,
        "sha256": digest,
        "size_bytes": archive_path.stat().st_size,
    }
    assert sha256_path.read_bytes() == f"{digest}  {archive_path.name}\n".encode("ascii")
    assert validate_deterministic_zip(archive_path)["status"] == "complete"


def test_deterministic_zip_is_byte_identical_across_output_names(tmp_path: Path) -> None:
    first = tmp_path / "student-state.pt"
    second = tmp_path / "teacher-state.pt"
    first.write_bytes(b"student\x00weights\n")
    second.write_bytes(b"teacher\x00weights\n")
    entries = [
        ArchiveEntry(second, "policy/teacher-state.pt", "teacher_state"),
        ArchiveEntry(first, "policy/student-state.pt", "student_state"),
    ]

    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"
    write_deterministic_zip(
        entries,
        archive_a,
        bundle_kind="gico_policy",
        metadata={"policy": "example"},
    )
    write_deterministic_zip(
        list(reversed(entries)),
        archive_b,
        bundle_kind="gico_policy",
        metadata={"policy": "example"},
    )

    assert archive_a.read_bytes() == archive_b.read_bytes()
    assert hashlib.sha256(archive_a.read_bytes()).hexdigest() == (
        "3445f970b0d3b2f7966ffc5a7af02cbfb9467ed0894b66c7f8755719bdf43cf7"
    )
    assert validate_deterministic_zip(archive_a)["status"] == "complete"
    assert validate_deterministic_zip(archive_b)["status"] == "complete"
    with zipfile.ZipFile(archive_a) as archive:
        assert archive.namelist() == [
            ARCHIVE_MANIFEST_NAME,
            "policy/student-state.pt",
            "policy/teacher-state.pt",
        ]
        manifest = json.loads(archive.read(ARCHIVE_MANIFEST_NAME))
        assert [record["path"] for record in manifest["files"]] == [
            "policy/student-state.pt",
            "policy/teacher-state.pt",
        ]
        assert str(tmp_path) not in archive.read(ARCHIVE_MANIFEST_NAME).decode("utf-8")
        for info in archive.infolist():
            assert info.date_time == ARCHIVE_TIMESTAMP
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.create_system == 3
            assert info.external_attr >> 16 == ARCHIVE_FILE_MODE
            assert info.extra == b""
            assert info.comment == b""
        assert archive.getinfo("policy/student-state.pt").extract_version == 45


def test_validator_rejects_bytes_outside_the_canonical_zip_frame(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    archive_path = tmp_path / "archive.zip"
    write_deterministic_zip(
        [ArchiveEntry(source, "payload.bin", "payload")],
        archive_path,
        bundle_kind="test",
    )
    with archive_path.open("ab") as handle:
        handle.write(b"non-canonical-trailer")

    result = validate_deterministic_zip(archive_path)

    assert result["status"] == "failed"
    assert any("trailing bytes" in error for error in result["errors"])


def test_validator_rejects_noncanonical_local_header_metadata(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    archive_path = tmp_path / "archive.zip"
    write_deterministic_zip(
        [ArchiveEntry(source, "payload.bin", "payload")],
        archive_path,
        bundle_kind="test",
    )
    raw = bytearray(archive_path.read_bytes())
    raw[10:12] = bytes((1, 0))
    archive_path.write_bytes(raw)

    result = validate_deterministic_zip(archive_path)

    assert result["status"] == "failed"
    assert any("local header has a non-canonical timestamp" in error for error in result["errors"])


def test_validator_reports_invalid_utf8_central_filename_without_raising(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    archive_path = tmp_path / "archive.zip"
    write_deterministic_zip(
        [ArchiveEntry(source, "payload.bin", "payload")],
        archive_path,
        bundle_kind="test",
    )
    raw = bytearray(archive_path.read_bytes())
    central_offset = raw.find(b"PK\x01\x02")
    assert central_offset >= 0
    flags = struct.unpack_from("<H", raw, central_offset + 8)[0]
    struct.pack_into("<H", raw, central_offset + 8, flags | 0x800)
    raw[central_offset + 46] = 0xFF
    archive_path.write_bytes(raw)

    result = validate_deterministic_zip(archive_path)

    assert result["status"] == "failed"
    assert any("Invalid ZIP archive" in error for error in result["errors"])


def test_canonical_central_zip64_extra_encodes_only_saturated_fields() -> None:
    info = zipfile.ZipInfo("large.bin")
    info.file_size = zipfile.ZIP64_LIMIT + 1
    info.compress_size = zipfile.ZIP64_LIMIT + 1
    info.header_offset = zipfile.ZIP64_LIMIT + 2

    assert deterministic_archive._canonical_central_zip64_extra(info) == struct.pack(
        "<HHQQQ",
        0x0001,
        24,
        info.file_size,
        info.compress_size,
        info.header_offset,
    )


def test_validator_rejects_unnecessary_zip64_directory_and_mismatched_classic_fields(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    archive_path = tmp_path / "archive.zip"
    write_deterministic_zip(
        [ArchiveEntry(source, "payload.bin", "payload")],
        archive_path,
        bundle_kind="test",
    )
    raw = bytearray(archive_path.read_bytes())
    eocd_offset = raw.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    classic = bytearray(raw[eocd_offset : eocd_offset + 22])
    member_count = struct.unpack_from("<H", classic, 10)[0]
    directory_size = struct.unpack_from("<L", classic, 12)[0]
    directory_offset = struct.unpack_from("<L", classic, 16)[0]
    zip64_record = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        member_count,
        member_count,
        directory_size,
        directory_offset,
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, eocd_offset, 1)
    struct.pack_into("<H", classic, 8, 0)
    struct.pack_into("<H", classic, 10, 0)
    struct.pack_into("<L", classic, 16, 0xFFFFFFFF)
    archive_path.write_bytes(raw[:eocd_offset] + zip64_record + locator + classic)

    result = validate_deterministic_zip(archive_path)

    assert result["status"] == "failed"
    assert any("do not mirror" in error for error in result["errors"])
    assert any("unnecessary ZIP64" in error for error in result["errors"])


def test_validator_accepts_required_zip64_directory_with_unsaturated_classic_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CPython deliberately switches to ZIP64 at a conservative 2 GiB limit,
    # while the classic EOCD offset can still represent values up to 4 GiB.
    # Lowering that threshold reproduces the framing without a multi-GB fixture.
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1024)
    source = tmp_path / "payload.bin"
    source.write_bytes(b"x" * 900)
    archive_path = tmp_path / "archive.zip"

    write_deterministic_zip(
        [ArchiveEntry(source, "payload.bin", "payload")],
        archive_path,
        bundle_kind="test",
    )

    raw = archive_path.read_bytes()
    eocd_offset = raw.rfind(b"PK\x05\x06")
    assert raw[eocd_offset - 20 : eocd_offset - 16] == b"PK\x06\x07"
    classic_directory_offset = struct.unpack_from("<L", raw, eocd_offset + 16)[0]
    assert zipfile.ZIP64_LIMIT < classic_directory_offset < 0xFFFFFFFF
    assert validate_deterministic_zip(archive_path)["status"] == "complete"


@pytest.mark.parametrize(
    "archive_path",
    [
        "../escape",
        "/absolute",
        "C:/absolute",
        "C:relative",
        "dir/",
        "dir\\payload.bin",
        "dir//payload.bin",
        "dir/./payload.bin",
        "payload\x00suffix.bin",
        "CON",
        "policy/NUL.pt",
        "payload.",
        "payload ",
        "cafe\u0301/payload.bin",
        f"{'a' * 256}/payload.bin",
        "payload?.bin",
    ],
)
def test_deterministic_zip_rejects_unsafe_member_paths(tmp_path: Path, archive_path: str) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    with pytest.raises(ValueError, match="Unsafe ZIP member path"):
        write_deterministic_zip(
            [ArchiveEntry(source, archive_path, "payload")],
            tmp_path / "out.zip",
            bundle_kind="test",
        )


def test_deterministic_zip_rejects_portable_duplicate_aliases(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    with pytest.raises(ValueError, match="portable comparison"):
        write_deterministic_zip(
            [
                ArchiveEntry(source, "Policy/weights.pt", "first"),
                ArchiveEntry(source, "policy/weights.pt", "second"),
            ],
            tmp_path / "out.zip",
            bundle_kind="test",
        )


@pytest.mark.parametrize(
    "local_value",
    [
        "prefix,C:/" + "home/private/checkpoint.pt",
        "prefix,/" + "home/private/checkpoint.pt",
        "run:/" + "projects/private/checkpoint.pt",
        "prefix,/opt/private/checkpoint.pt",
        "run:/work/private/checkpoint.pt",
        "prefix,/etc/passwd",
        "prefix,//private-server/share/checkpoint.pt",
    ],
)
def test_deterministic_zip_rejects_local_paths_anywhere_in_metadata(
    tmp_path: Path,
    local_value: str,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    with pytest.raises(ValueError, match="local filesystem path"):
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            tmp_path / "out.zip",
            bundle_kind="test",
            metadata={"source": local_value},
        )


def test_deterministic_zip_rejects_symlink_sources(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    link = tmp_path / "payload-link.bin"
    try:
        link.symlink_to(source)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    with pytest.raises(ValueError, match="links or reparse points"):
        write_deterministic_zip(
            [ArchiveEntry(link, "payload.bin", "payload")],
            tmp_path / "out.zip",
            bundle_kind="test",
        )


def test_deterministic_zip_allows_trusted_symlinked_mount_ancestor(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    mount = tmp_path / "mount"
    try:
        mount.symlink_to(storage, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    source = mount / "payload.bin"
    source.write_bytes(b"payload")
    archive_path = mount / "archive.zip"

    write_deterministic_zip(
        [ArchiveEntry(source, "payload.bin", "payload")],
        archive_path,
        bundle_kind="test",
    )

    assert validate_deterministic_zip(archive_path)["status"] == "complete"
    assert (storage / "archive.zip").is_file()


def test_deterministic_zip_rejects_source_swapped_during_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"original")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replacement payload")
    original_resolve = Path.resolve
    swapped = False

    def swap_then_resolve(path: Path, strict: bool = False) -> Path:
        nonlocal swapped
        if path == source and not swapped:
            swapped = True
            source.unlink()
            replacement.replace(source)
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", swap_then_resolve)
    with pytest.raises(RuntimeError, match="changed"):
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            tmp_path / "out.zip",
            bundle_kind="test",
        )
    assert not (tmp_path / "out.zip").exists()


def test_deterministic_zip_detects_source_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"a" * 4096)
    original = deterministic_archive._copy_and_hash

    def mutate_before_archive_copy(source_handle: object, destination: object) -> tuple[int, str]:
        if destination is not None:
            source.write_bytes(b"b" * 4096)
        return original(source_handle, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(deterministic_archive, "_copy_and_hash", mutate_before_archive_copy)
    with pytest.raises(RuntimeError, match="changed"):
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            tmp_path / "out.zip",
            bundle_kind="test",
        )
    assert not (tmp_path / "out.zip").exists()


def test_existing_sidecar_blocks_non_overwriting_build(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    sidecar = tmp_path / "out.zip.manifest.json"
    sidecar.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            tmp_path / "out.zip",
            bundle_kind="test",
        )
    assert sidecar.read_text(encoding="utf-8") == "preserve me"


def test_archive_bundle_staging_failure_leaves_no_partial_publication_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    output = tmp_path / "out.zip"
    real_write_stage_bytes = deterministic_archive._write_stage_bytes

    def fail_manifest_stage(destination: object, data: bytes) -> None:
        if data.startswith(b"{"):
            raise OSError("simulated manifest staging failure")
        real_write_stage_bytes(destination, data)  # type: ignore[arg-type]

    monkeypatch.setattr(
        deterministic_archive,
        "_write_stage_bytes",
        fail_manifest_stage,
    )
    with pytest.raises(OSError, match="manifest staging failure"):
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            output,
            bundle_kind="test",
        )

    assert not any(path.exists() for path in _archive_bundle_paths(output))
    assert not bundle_journal_path(output).exists()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))

    monkeypatch.setattr(
        deterministic_archive,
        "_write_stage_bytes",
        real_write_stage_bytes,
    )
    write_deterministic_zip(
        [ArchiveEntry(source, "payload.bin", "payload")],
        output,
        bundle_kind="test",
    )
    _assert_archive_bundle_complete(output)


@pytest.mark.parametrize(
    "predecessor",
    ("archive_only", "sidecar_only", "invalid_complete"),
)
def test_archive_bundle_overwrite_replaces_partial_or_invalid_predecessor(
    tmp_path: Path,
    predecessor: str,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"replacement payload")
    output = tmp_path / "out.zip"
    archive_path, manifest_path, sha256_path = _archive_bundle_paths(output)
    if predecessor in {"archive_only", "invalid_complete"}:
        archive_path.write_bytes(b"legacy archive")
    if predecessor in {"sidecar_only", "invalid_complete"}:
        manifest_path.write_bytes(b"legacy manifest")
    if predecessor == "invalid_complete":
        sha256_path.write_bytes(b"legacy checksum")

    write_deterministic_zip(
        [ArchiveEntry(source, "payload.bin", "payload")],
        output,
        bundle_kind="test",
        overwrite=True,
    )

    _assert_archive_bundle_complete(output)
    assert archive_path.read_bytes() != b"legacy archive"
    assert manifest_path.read_bytes() != b"legacy manifest"
    assert sha256_path.read_bytes() != b"legacy checksum"
    assert not bundle_journal_path(output).exists()
    assert not list(tmp_path.glob(".*.bundle-backup-*"))


@pytest.mark.parametrize("predecessor", ("archive_only", "sidecar_only"))
def test_archive_bundle_failed_overwrite_restores_partial_predecessor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predecessor: str,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"replacement payload")
    output = tmp_path / "out.zip"
    archive_path, manifest_path, _ = _archive_bundle_paths(output)
    if predecessor == "archive_only":
        archive_path.write_bytes(b"legacy archive")
    else:
        manifest_path.write_bytes(b"legacy manifest")
    previous = {path: path.read_bytes() for path in _archive_bundle_paths(output) if path.exists()}
    real_install = artifact_bundle._link_without_overwrite

    def fail_manifest_promotion(source_path: Path, destination: Path) -> None:
        if destination == manifest_path and ".bundle-stage-" in source_path.name:
            raise OSError("simulated manifest promotion failure")
        real_install(source_path, destination)

    monkeypatch.setattr(
        artifact_bundle,
        "_link_without_overwrite",
        fail_manifest_promotion,
    )
    with pytest.raises(OSError, match="manifest promotion failure"):
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            output,
            bundle_kind="test",
            overwrite=True,
        )

    assert {path: path.read_bytes() for path in _archive_bundle_paths(output) if path.exists()} == previous
    assert not bundle_journal_path(output).exists()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))
    assert not list(tmp_path.glob(".*.bundle-backup-*"))


@pytest.mark.parametrize("preexisting", (False, True))
def test_archive_bundle_promotion_failure_rolls_back_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: bool,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"old payload")
    output = tmp_path / "out.zip"
    if preexisting:
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            output,
            bundle_kind="test",
        )
    previous = {path: path.read_bytes() for path in _archive_bundle_paths(output) if path.exists()}
    source.write_bytes(b"new payload")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    real_install = artifact_bundle._link_without_overwrite

    def fail_manifest_promotion(source_path: Path, destination: Path) -> None:
        if destination == manifest_path and ".bundle-stage-" in source_path.name:
            raise OSError("simulated manifest promotion failure")
        real_install(source_path, destination)

    monkeypatch.setattr(
        artifact_bundle,
        "_link_without_overwrite",
        fail_manifest_promotion,
    )
    with pytest.raises(OSError, match="manifest promotion failure"):
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            output,
            bundle_kind="test",
            overwrite=preexisting,
        )

    assert {path: path.read_bytes() for path in _archive_bundle_paths(output) if path.exists()} == previous
    assert not bundle_journal_path(output).exists()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))
    assert not list(tmp_path.glob(".*.bundle-backup-*"))

    monkeypatch.setattr(artifact_bundle, "_link_without_overwrite", real_install)
    write_deterministic_zip(
        [ArchiveEntry(source, "payload.bin", "payload")],
        output,
        bundle_kind="test",
        overwrite=preexisting,
    )
    _assert_archive_bundle_complete(output)
    if preexisting:
        assert output.read_bytes() != previous[output]


def test_archive_bundle_retry_recovers_prepared_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    output = tmp_path / "out.zip"
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    real_install = artifact_bundle._link_without_overwrite
    real_recover = artifact_bundle._recover_locked

    def interrupt_manifest_promotion(source_path: Path, destination: Path) -> None:
        if destination == manifest_path and ".bundle-stage-" in source_path.name:
            raise OSError("simulated process interruption")
        real_install(source_path, destination)

    def leave_prepared_journal(*args: object, force_abort: bool = False, **kwargs: object) -> None:
        if force_abort:
            raise OSError("simulated unavailable in-process recovery")
        real_recover(*args, force_abort=force_abort, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        artifact_bundle,
        "_link_without_overwrite",
        interrupt_manifest_promotion,
    )
    monkeypatch.setattr(artifact_bundle, "_recover_locked", leave_prepared_journal)
    with pytest.raises(OSError, match="process interruption"):
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            output,
            bundle_kind="test",
        )

    assert output.is_file()
    assert not manifest_path.exists()
    assert bundle_journal_path(output).is_file()

    monkeypatch.setattr(artifact_bundle, "_link_without_overwrite", real_install)
    monkeypatch.setattr(artifact_bundle, "_recover_locked", real_recover)
    write_deterministic_zip(
        [ArchiveEntry(source, "payload.bin", "payload")],
        output,
        bundle_kind="test",
    )

    _assert_archive_bundle_complete(output)
    assert not bundle_journal_path(output).exists()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))
    assert not list(tmp_path.glob(".*.bundle-backup-*"))


def test_archive_bundle_stage_link_swap_cannot_modify_external_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    output = tmp_path / "out.zip"
    external = tmp_path / "external.txt"
    external.write_bytes(b"external owner")
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    probe.unlink()
    real_named_temporary = deterministic_archive.tempfile.NamedTemporaryFile
    swapped_stage: Path | None = None

    class SwapOnClose:
        def __init__(self, handle: object, path: Path) -> None:
            self._handle = handle
            self._path = path
            self._swapped = False

        def __getattr__(self, name: str) -> object:
            return getattr(self._handle, name)

        def close(self) -> None:
            self._handle.close()  # type: ignore[attr-defined]
            if not self._swapped:
                self._path.unlink()
                self._path.symlink_to(external)
                self._swapped = True

    def swap_archive_stage(*args: object, **kwargs: object):
        nonlocal swapped_stage
        handle = real_named_temporary(*args, **kwargs)  # type: ignore[arg-type]
        if kwargs.get("prefix") == ".out.zip.bundle-stage-":
            swapped_stage = Path(handle.name)
            return SwapOnClose(handle, swapped_stage)
        return handle

    monkeypatch.setattr(
        deterministic_archive.tempfile,
        "NamedTemporaryFile",
        swap_archive_stage,
    )
    with pytest.raises(RuntimeError, match="staging path changed while open"):
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            output,
            bundle_kind="test",
        )

    assert external.read_bytes() == b"external owner"
    assert not any(path.exists() for path in _archive_bundle_paths(output))
    assert swapped_stage is not None and swapped_stage.is_symlink()
    swapped_stage.unlink()
    assert not bundle_journal_path(output).exists()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))


def test_staging_cleanup_preserves_managed_name_symlink_referent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out.zip"
    owner = tmp_path / ".out.zip.bundle-stage-owner.tmp"
    owner.write_bytes(b"separately owned staging file")
    linked_stage = tmp_path / ".out.zip.bundle-stage-link.tmp"
    try:
        linked_stage.symlink_to(owner)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    removed = artifact_bundle.discard_temporary_bundle_path(
        linked_stage,
        output,
    )

    assert not removed
    assert linked_stage.is_symlink()
    assert owner.read_bytes() == b"separately owned staging file"
    linked_stage.unlink()
    owner.unlink()


def test_archive_bundle_rejects_post_stage_managed_name_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    output = tmp_path / "out.zip"
    owner = tmp_path / ".out.zip.bundle-stage-owner.tmp"
    real_promote = deterministic_archive.promote_artifact_bundle
    swapped_stage: Path | None = None
    owner_payload = b""

    def swap_before_promotion(
        anchor: Path,
        targets: dict[str, Path],
        staged: dict[str, Path],
        **kwargs: object,
    ) -> None:
        nonlocal swapped_stage, owner_payload
        swapped_stage = Path(staged["archive"])
        owner_payload = swapped_stage.read_bytes()
        owner.write_bytes(owner_payload)
        swapped_stage.unlink()
        try:
            swapped_stage.symlink_to(owner)
        except OSError as exc:
            pytest.skip(f"symlinks are unavailable: {exc}")
        real_promote(anchor, targets, staged, **kwargs)

    monkeypatch.setattr(
        deterministic_archive,
        "promote_artifact_bundle",
        swap_before_promotion,
    )
    with pytest.raises(ValueError, match="staging sidecar may not be a link"):
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            output,
            bundle_kind="test",
        )

    assert not any(path.exists() for path in _archive_bundle_paths(output))
    assert swapped_stage is not None and swapped_stage.is_symlink()
    assert owner.read_bytes() == owner_payload
    swapped_stage.unlink()
    owner.unlink()
    assert not bundle_journal_path(output).exists()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))


@pytest.mark.parametrize("member_suffix", ("", ".manifest.json", ".sha256"))
def test_archive_bundle_rejects_linked_publication_targets(
    tmp_path: Path,
    member_suffix: str,
) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    output = tmp_path / "out.zip"
    external = tmp_path / "external.txt"
    external.write_bytes(b"external owner")
    linked_target = Path(f"{output}{member_suffix}")
    try:
        linked_target.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink, junction, or reparse point"):
        write_deterministic_zip(
            [ArchiveEntry(source, "payload.bin", "payload")],
            output,
            bundle_kind="test",
            overwrite=True,
        )

    assert external.read_bytes() == b"external owner"
    for path in _archive_bundle_paths(output):
        if path != linked_target:
            assert not path.exists()
    assert linked_target.is_symlink()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))


def test_validator_reports_malformed_manifest_without_raising(tmp_path: Path) -> None:
    archive_path = tmp_path / "malformed.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(ARCHIVE_MANIFEST_NAME, b'{"files":[{"path":"payload.bin","size_bytes":"bad"}]}')
        archive.writestr("payload.bin", b"payload")
    result = validate_deterministic_zip(archive_path)
    assert result["status"] == "failed"
    assert result["error_count"] > 0


def test_deterministic_zip_rejects_duplicate_members(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    with pytest.raises(ValueError, match="Duplicate ZIP member path"):
        write_deterministic_zip(
            [
                ArchiveEntry(source, "payload.bin", "first"),
                ArchiveEntry(source, "payload.bin", "second"),
            ],
            tmp_path / "out.zip",
            bundle_kind="test",
        )
