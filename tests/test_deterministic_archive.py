from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import pytest

import genode.deterministic_archive as deterministic_archive
from genode.deterministic_archive import (
    ARCHIVE_FILE_MODE,
    ARCHIVE_MANIFEST_NAME,
    ARCHIVE_TIMESTAMP,
    ArchiveEntry,
    validate_deterministic_zip,
    write_deterministic_zip,
)


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
