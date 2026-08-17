from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
import tempfile
import unicodedata
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

from genode.artifact_bundle import (
    bundle_journal_path,
    discard_temporary_bundle_path,
    preflight_artifact_bundle,
    promote_artifact_bundle,
    validate_artifact_bundle_layout,
)

ARCHIVE_SCHEMA_VERSION = "genode_deterministic_archive_v1"
ARCHIVE_MANIFEST_NAME = "MANIFEST.json"
ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ARCHIVE_FILE_MODE = stat.S_IFREG | 0o644
_COPY_CHUNK_SIZE = 8 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
_EOCD_SIZE = 22
_LOCAL_FILE_HEADER_SIZE = 30
_MAX_ZIP_COMMENT_SIZE = (1 << 16) - 1
_WINDOWS_LOCAL_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
_POSIX_LOCAL_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9/])/(?!/)[^\s,;]+")
_UNC_LOCAL_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9:])//[^/]", re.IGNORECASE)
_TILDE_LOCAL_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9])~[/\\]")
_WINDOWS_RESERVED_COMPONENTS = frozenset(
    {"aux", "con", "nul", "prn"} | {f"com{index}" for index in range(1, 10)} | {f"lpt{index}" for index in range(1, 10)}
)
_WINDOWS_INVALID_COMPONENT = re.compile(r'[<>"|?*]')


@dataclass(frozen=True)
class ArchiveEntry:
    source: Path
    archive_path: str
    role: str


@dataclass(frozen=True)
class _PreparedEntry:
    entry: ArchiveEntry
    identity: tuple[int, int, int, int]
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _SourceSnapshot:
    identity: tuple[int, int, int, int]
    sha256: str
    size_bytes: int


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    return _snapshot_source(_checked_source(path)).sha256


def contains_local_filesystem_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return bool(
        "\x00" in value
        or normalized.startswith(("/", "~/", "//"))
        or "file://" in normalized.lower()
        or _WINDOWS_LOCAL_PATH_PATTERN.search(value)
        or _POSIX_LOCAL_PATH_PATTERN.search(normalized)
        or _UNC_LOCAL_PATH_PATTERN.search(normalized)
        or _TILDE_LOCAL_PATH_PATTERN.search(value)
    )


def _portable_json(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} object keys must be strings.")
            _portable_json(key, label=f"{label} key")
            _portable_json(item, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _portable_json(item, label=f"{label}[{index}]")
        return
    if isinstance(value, str):
        if contains_local_filesystem_path(value):
            raise ValueError(f"{label} contains a local filesystem path.")
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    raise TypeError(f"{label} contains a non-JSON value of type {type(value).__name__}.")


def _safe_archive_path(value: str) -> str:
    normalized = str(value)
    path = PurePosixPath(normalized)
    invalid_component = any(
        unicodedata.normalize("NFC", part) != part
        or part.endswith((".", " "))
        or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_COMPONENTS
        or _WINDOWS_INVALID_COMPONENT.search(part) is not None
        or len(part.encode("utf-8")) > 255
        for part in path.parts
    )
    if (
        not normalized
        or "\\" in normalized
        or any(ord(character) < 32 for character in normalized)
        or normalized.endswith("/")
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(":" in part for part in path.parts)
        or invalid_component
        or len(normalized.encode("utf-8")) > 65_535
        or normalized != path.as_posix()
    ):
        raise ValueError(f"Unsafe ZIP member path: {value!r}")
    return normalized


def _portable_archive_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _is_reparse_stat(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(marker and attributes & marker)


def _absolute_without_links(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_linked_components(path: Path) -> None:
    for candidate in [*reversed(path.parents), path]:
        value = candidate.lstat()
        if stat.S_ISLNK(value.st_mode) or _is_reparse_stat(value):
            raise ValueError(f"Archive sources may not use links or reparse points: {candidate}")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    # Windows reports different ``st_ctime_ns`` values for a path stat and an
    # open-handle stat of the same file. Device, inode, size, and modification
    # time are stable across both views; the streaming SHA-256 comparison below
    # still detects same-size content changes even if timestamps are restored.
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _checked_source(path: str | Path) -> Path:
    requested = _absolute_without_links(Path(path).expanduser())
    requested_before = requested.lstat()
    if stat.S_ISLNK(requested_before.st_mode) or _is_reparse_stat(requested_before):
        raise ValueError(f"Archive sources may not use links or reparse points: {requested}")
    if not stat.S_ISREG(requested_before.st_mode):
        raise ValueError(f"Archive source is not a regular file: {requested}")
    expected_identity = _file_identity(requested_before)
    # Canonicalize after rejecting a linked leaf.  Trusted platform mount
    # aliases may exist above an explicitly selected source or archive, while
    # all subsequent identity and mutation checks operate on one stable,
    # link-free canonical path.
    source = requested.resolve(strict=True)
    requested_after = requested.lstat()
    source_after = source.lstat()
    if stat.S_ISLNK(requested_after.st_mode) or _is_reparse_stat(requested_after):
        raise RuntimeError(f"Archive source changed while it was resolved: {requested}")
    if not stat.S_ISREG(requested_after.st_mode) or not stat.S_ISREG(source_after.st_mode):
        raise ValueError(f"Archive source is not a regular file: {source}")
    if _file_identity(requested_after) != expected_identity or _file_identity(source_after) != expected_identity:
        raise RuntimeError(f"Archive source changed while it was resolved: {requested}")

    _reject_linked_components(source)
    requested_final = requested.lstat()
    source_final = source.lstat()
    if stat.S_ISLNK(requested_final.st_mode) or _is_reparse_stat(requested_final):
        raise RuntimeError(f"Archive source changed while it was checked: {requested}")
    if (
        not stat.S_ISREG(requested_final.st_mode)
        or not stat.S_ISREG(source_final.st_mode)
        or _file_identity(requested_final) != expected_identity
        or _file_identity(source_final) != expected_identity
    ):
        raise RuntimeError(f"Archive source changed while it was checked: {requested}")
    return source


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=ARCHIVE_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = ARCHIVE_FILE_MODE << 16
    info.internal_attr = 0
    return info


def _copy_and_hash(source: BinaryIO, destination: BinaryIO | None) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = source.read(_COPY_CHUNK_SIZE)
        if not chunk:
            break
        if destination is not None:
            destination.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def _canonical_zip_framing_errors(
    path: Path,
    *,
    member_count: int,
    central_directory_start: int,
) -> list[str]:
    """Validate that the central directory occupies one exact, unprefixed ZIP frame."""

    errors: list[str] = []
    file_size = path.stat().st_size
    tail_size = min(file_size, _EOCD_SIZE + _MAX_ZIP_COMMENT_SIZE)
    with path.open("rb") as handle:
        handle.seek(file_size - tail_size)
        tail = handle.read(tail_size)
    relative_eocd = tail.rfind(_EOCD_SIGNATURE)
    if relative_eocd < 0 or len(tail) - relative_eocd < _EOCD_SIZE:
        return ["ZIP archive has no complete end-of-central-directory record."]
    eocd_offset = file_size - tail_size + relative_eocd
    eocd = tail[relative_eocd : relative_eocd + _EOCD_SIZE]
    (
        signature,
        disk_number,
        central_directory_disk,
        disk_entry_count,
        total_entry_count,
        central_directory_size,
        central_directory_offset,
        comment_size,
    ) = struct.unpack("<4s4H2LH", eocd)
    if signature != _EOCD_SIGNATURE:
        return ["ZIP archive has an invalid end-of-central-directory signature."]
    if eocd_offset + _EOCD_SIZE + comment_size != file_size:
        errors.append("ZIP archive contains trailing bytes outside its canonical frame.")
    if comment_size != 0:
        errors.append("ZIP archive has a non-canonical end-of-directory comment.")
    if disk_number != 0 or central_directory_disk != 0:
        errors.append("ZIP archive uses unsupported multi-disk framing.")

    uses_zip64_directory = (
        member_count > zipfile.ZIP_FILECOUNT_LIMIT
        or central_directory_size > zipfile.ZIP64_LIMIT
        or central_directory_offset > zipfile.ZIP64_LIMIT
    )
    # CPython starts using ZIP64 once an offset or size exceeds its
    # conservative 2 GiB ZIP64_LIMIT, but preserves a still-representable
    # 32-bit value in the classic EOCD.  Match those writer conditions rather
    # than relying on saturated fields or locator-like bytes that could occur
    # coincidentally at the tail of a classic central-directory entry.
    if not uses_zip64_directory:
        if disk_entry_count != member_count or total_entry_count != member_count:
            errors.append("ZIP end-of-directory member count is inconsistent.")
        if central_directory_offset + central_directory_size != eocd_offset:
            errors.append("ZIP central directory is not contiguous with its canonical frame.")
        if central_directory_offset != central_directory_start:
            errors.append("ZIP central-directory offset is inconsistent.")
        return errors

    locator_offset = eocd_offset - 20
    if locator_offset < 0:
        errors.append("ZIP64 archive is missing its end-of-directory locator.")
        return errors
    with path.open("rb") as handle:
        handle.seek(locator_offset)
        locator = handle.read(20)
    if len(locator) != 20:
        errors.append("ZIP64 archive has a truncated end-of-directory locator.")
        return errors
    locator_signature, zip64_disk, zip64_eocd_offset, disk_count = struct.unpack("<4sLQL", locator)
    if locator_signature != _ZIP64_LOCATOR_SIGNATURE:
        errors.append("ZIP64 archive has an invalid end-of-directory locator.")
        return errors
    if zip64_disk != 0 or disk_count != 1:
        errors.append("ZIP64 archive uses unsupported multi-disk framing.")
    with path.open("rb") as handle:
        handle.seek(zip64_eocd_offset)
        zip64_eocd = handle.read(56)
    if len(zip64_eocd) != 56:
        errors.append("ZIP64 archive has a truncated end-of-directory record.")
        return errors
    (
        zip64_signature,
        zip64_record_size,
        created_version,
        required_version,
        zip64_disk_number,
        zip64_directory_disk,
        zip64_disk_entries,
        zip64_total_entries,
        zip64_directory_size,
        zip64_directory_offset,
    ) = struct.unpack("<4sQ2H2L4Q", zip64_eocd)
    if zip64_signature != _ZIP64_EOCD_SIGNATURE or zip64_record_size != 44:
        errors.append("ZIP64 archive has a non-canonical end-of-directory record.")
        return errors
    if created_version != 45 or required_version != 45:
        errors.append("ZIP64 archive has non-canonical end-of-directory versions.")
    if zip64_eocd_offset + 12 + zip64_record_size != locator_offset:
        errors.append("ZIP64 end-of-directory records are not contiguous.")
    if zip64_disk_number != 0 or zip64_directory_disk != 0:
        errors.append("ZIP64 archive uses unsupported multi-disk framing.")
    if zip64_disk_entries != member_count or zip64_total_entries != member_count:
        errors.append("ZIP64 end-of-directory member count is inconsistent.")
    if zip64_directory_offset + zip64_directory_size != zip64_eocd_offset:
        errors.append("ZIP64 central directory is not contiguous with its canonical frame.")
    if zip64_directory_offset != central_directory_start:
        errors.append("ZIP64 central-directory offset is inconsistent.")
    expected_classic_disk_entries = 0xFFFF if zip64_disk_entries > zipfile.ZIP_FILECOUNT_LIMIT else zip64_disk_entries
    expected_classic_total_entries = (
        0xFFFF if zip64_total_entries > zipfile.ZIP_FILECOUNT_LIMIT else zip64_total_entries
    )
    expected_classic_directory_size = 0xFFFFFFFF if zip64_directory_size > 0xFFFFFFFF else zip64_directory_size
    expected_classic_directory_offset = 0xFFFFFFFF if zip64_directory_offset > 0xFFFFFFFF else zip64_directory_offset
    if (
        disk_entry_count != expected_classic_disk_entries
        or total_entry_count != expected_classic_total_entries
        or central_directory_size != expected_classic_directory_size
        or central_directory_offset != expected_classic_directory_offset
    ):
        errors.append("ZIP64 classic end-of-directory fields do not mirror the ZIP64 record.")
    if (
        zip64_disk_entries <= zipfile.ZIP_FILECOUNT_LIMIT
        and zip64_total_entries <= zipfile.ZIP_FILECOUNT_LIMIT
        and zip64_directory_size <= zipfile.ZIP64_LIMIT
        and zip64_directory_offset <= zipfile.ZIP64_LIMIT
    ):
        errors.append("ZIP archive uses unnecessary ZIP64 end-of-directory framing.")
    return errors


def _canonical_local_header_errors(
    path: Path,
    infos: Sequence[zipfile.ZipInfo],
    *,
    central_directory_start: int,
) -> list[str]:
    errors: list[str] = []
    ordered = sorted(infos, key=lambda info: info.header_offset)
    if [info.filename for info in ordered] != [info.filename for info in infos]:
        errors.append("ZIP local-member order differs from its canonical central-directory order.")
    expected_offset = 0
    with path.open("rb") as handle:
        for info in ordered:
            if info.header_offset != expected_offset:
                errors.append(f"ZIP local members are not contiguous before {info.filename}.")
            handle.seek(info.header_offset)
            raw_header = handle.read(_LOCAL_FILE_HEADER_SIZE)
            if len(raw_header) != _LOCAL_FILE_HEADER_SIZE:
                errors.append(f"ZIP local header is truncated: {info.filename}")
                continue
            (
                signature,
                required_version,
                flags,
                compression,
                dos_time,
                dos_date,
                crc32,
                compressed_size,
                file_size,
                filename_size,
                extra_size,
            ) = struct.unpack("<4s5H3L2H", raw_header)
            filename = handle.read(filename_size)
            extra = handle.read(extra_size)
            expected_filename = info.filename.encode("ascii" if info.filename.isascii() else "utf-8")
            year, month, day, hour, minute, second = info.date_time
            expected_dos_time = (hour << 11) | (minute << 5) | (second // 2)
            expected_dos_date = ((year - 1980) << 9) | (month << 5) | day
            expected_extra = (
                b""
                if info.filename == ARCHIVE_MANIFEST_NAME
                else struct.pack("<HHQQ", 0x0001, 16, info.file_size, info.compress_size)
            )
            expected_file_size = info.file_size if not expected_extra else 0xFFFFFFFF
            expected_compressed_size = info.compress_size if not expected_extra else 0xFFFFFFFF
            if signature != _LOCAL_FILE_SIGNATURE:
                errors.append(f"ZIP local header has an invalid signature: {info.filename}")
            if required_version != info.extract_version:
                errors.append(f"ZIP local header has a non-canonical version: {info.filename}")
            if flags != info.flag_bits or compression != info.compress_type:
                errors.append(f"ZIP local header flags or compression disagree: {info.filename}")
            if dos_time != expected_dos_time or dos_date != expected_dos_date:
                errors.append(f"ZIP local header has a non-canonical timestamp: {info.filename}")
            if crc32 != info.CRC or compressed_size != expected_compressed_size or file_size != expected_file_size:
                errors.append(f"ZIP local header sizes or CRC disagree: {info.filename}")
            if filename != expected_filename:
                errors.append(f"ZIP local header filename disagrees: {info.filename}")
            if extra != expected_extra:
                errors.append(f"ZIP local header has non-canonical extra metadata: {info.filename}")
            expected_offset = (
                info.header_offset + _LOCAL_FILE_HEADER_SIZE + filename_size + extra_size + info.compress_size
            )
    if expected_offset != central_directory_start:
        errors.append("ZIP local payload region is not contiguous with the central directory.")
    return errors


def _canonical_central_zip64_extra(info: zipfile.ZipInfo) -> bytes:
    """Return the exact ZIP64 central-directory extra emitted by ``zipfile``."""
    values: list[int] = []
    if info.file_size > zipfile.ZIP64_LIMIT:
        values.append(info.file_size)
    if info.compress_size > zipfile.ZIP64_LIMIT:
        values.append(info.compress_size)
    if info.header_offset > zipfile.ZIP64_LIMIT:
        values.append(info.header_offset)
    if not values:
        return b""
    payload = struct.pack(f"<{len(values)}Q", *values)
    return struct.pack("<HH", 0x0001, len(payload)) + payload


def _snapshot_source(source: Path) -> _SourceSnapshot:
    _reject_linked_components(source)
    before = source.lstat()
    expected = _file_identity(before)
    with source.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened_before.st_mode) or _file_identity(opened_before) != expected:
            raise RuntimeError(f"Archive source changed before it could be read: {source}")
        size, digest = _copy_and_hash(handle, None)
        opened_after = os.fstat(handle.fileno())
    _reject_linked_components(source)
    after = source.lstat()
    if _file_identity(opened_after) != expected or _file_identity(after) != expected or size != expected[2]:
        raise RuntimeError(f"Archive source changed while it was being hashed: {source}")
    return _SourceSnapshot(
        identity=expected,
        sha256=digest,
        size_bytes=size,
    )


def _prepare_entries(entries: Iterable[ArchiveEntry]) -> tuple[list[_PreparedEntry], list[dict[str, Any]]]:
    prepared: list[_PreparedEntry] = []
    records: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for entry in entries:
        archive_path = _safe_archive_path(entry.archive_path)
        portable_key = _portable_archive_key(archive_path)
        if portable_key == _portable_archive_key(ARCHIVE_MANIFEST_NAME):
            raise ValueError(f"{ARCHIVE_MANIFEST_NAME} is reserved for the archive manifest.")
        if portable_key in names:
            raise ValueError(
                f"Duplicate ZIP member path under portable comparison: "
                f"{archive_path!r} conflicts with {names[portable_key]!r}"
            )
        names[portable_key] = archive_path
        source = _checked_source(entry.source)
        role = str(entry.role).strip()
        if not role:
            raise ValueError(f"Archive role may not be empty: {archive_path}")
        _portable_json(role, label=f"Archive role for {archive_path}")
        prepared_entry = ArchiveEntry(source=source, archive_path=archive_path, role=role)
        snapshot = _snapshot_source(source)
        prepared.append(
            _PreparedEntry(
                entry=prepared_entry,
                identity=snapshot.identity,
                sha256=snapshot.sha256,
                size_bytes=snapshot.size_bytes,
            )
        )
        records.append(
            {
                "path": archive_path,
                "role": role,
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
            }
        )
    prepared.sort(key=lambda item: item.entry.archive_path)
    records.sort(key=lambda item: str(item["path"]))
    if not prepared:
        raise ValueError("A deterministic archive must contain at least one payload file.")
    return prepared, records


def _archive_bundle_targets(output: Path) -> dict[str, Path]:
    targets = {
        "archive": output,
        "manifest": output.with_suffix(output.suffix + ".manifest.json"),
        "sha256": output.with_suffix(output.suffix + ".sha256"),
    }
    validate_artifact_bundle_layout(output, targets)
    return targets


@contextmanager
def _exclusive_bundle_stage(target: Path) -> Iterator[tuple[Path, BinaryIO]]:
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed by this context manager
        mode="w+b",
        prefix=f".{target.name}.bundle-stage-",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    path = Path(handle.name)
    try:
        yield path, handle
        handle.flush()
        os.fsync(handle.fileno())
        expected_identity = _file_identity(os.fstat(handle.fileno()))
    finally:
        handle.close()
    observed = path.lstat()
    if (
        stat.S_ISLNK(observed.st_mode)
        or _is_reparse_stat(observed)
        or not stat.S_ISREG(observed.st_mode)
        or _file_identity(observed) != expected_identity
    ):
        raise RuntimeError(f"Archive bundle staging path changed while open: {path.name}")


def _write_stage_bytes(destination: BinaryIO, payload: bytes) -> None:
    if destination.write(payload) != len(payload):
        raise OSError("Could not write the complete deterministic archive sidecar.")


def _write_archive_stage(
    destination: BinaryIO,
    *,
    prepared: Sequence[_PreparedEntry],
    records: Sequence[Mapping[str, Any]],
    manifest_bytes: bytes,
) -> None:
    with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
        members: list[tuple[str, bytes | _PreparedEntry]] = [
            (ARCHIVE_MANIFEST_NAME, manifest_bytes),
            *((prepared_entry.entry.archive_path, prepared_entry) for prepared_entry in prepared),
        ]
        expected_by_name = {str(record["path"]): record for record in records}
        for name, value in sorted(members, key=lambda item: item[0]):
            info = _zip_info(name)
            if isinstance(value, bytes):
                archive.writestr(info, value, compress_type=zipfile.ZIP_STORED)
                continue
            expected = expected_by_name[name]
            source_path = value.entry.source
            _reject_linked_components(source_path)
            if _file_identity(source_path.lstat()) != value.identity:
                raise RuntimeError(f"Archive source changed before it was written: {source_path}")
            with source_path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                if _file_identity(os.fstat(source.fileno())) != value.identity:
                    raise RuntimeError(f"Archive source changed before it was written: {source_path}")
                size, digest = _copy_and_hash(source, target)
                opened_after = os.fstat(source.fileno())
            _reject_linked_components(source_path)
            if (
                _file_identity(opened_after) != value.identity
                or _file_identity(source_path.lstat()) != value.identity
                or size != expected["size_bytes"]
                or digest != expected["sha256"]
            ):
                raise RuntimeError(f"Archive source changed while it was being written: {source_path}")


def _validate_archive_bundle_identity(
    paths: Mapping[str, Path],
    targets: Mapping[str, Path],
) -> None:
    validation = validate_deterministic_zip(paths["archive"])
    if validation["status"] != "complete":
        detail = "; ".join(str(error) for error in validation["errors"])
        raise ValueError(f"Deterministic archive bundle contains an invalid ZIP: {detail}")

    with zipfile.ZipFile(paths["archive"], "r") as archive:
        manifest_bytes = archive.read(ARCHIVE_MANIFEST_NAME)
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, Mapping):
        raise ValueError("Deterministic archive manifest must be a JSON object.")

    archive_digest = str(validation["sha256"])
    expected_sidecar = {
        "archive": targets["archive"].name,
        "bundle_kind": manifest["bundle_kind"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "sha256": archive_digest,
        "size_bytes": paths["archive"].stat().st_size,
    }
    try:
        sidecar_bytes = paths["manifest"].read_bytes()
        sidecar = json.loads(
            sidecar_bytes,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant {value!r}")),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Could not read deterministic archive manifest sidecar: {exc}") from exc
    if not isinstance(sidecar, Mapping) or sidecar_bytes != canonical_json_bytes(expected_sidecar):
        raise ValueError("Deterministic archive manifest sidecar does not match the ZIP artifact.")

    expected_sha256 = f"{archive_digest}  {targets['archive'].name}\n".encode("ascii")
    try:
        sha256_bytes = paths["sha256"].read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read deterministic archive SHA-256 sidecar: {exc}") from exc
    if sha256_bytes != expected_sha256:
        raise ValueError("Deterministic archive SHA-256 sidecar does not match the ZIP artifact.")


def write_deterministic_zip(
    entries: Sequence[ArchiveEntry],
    output_path: str | Path,
    *,
    bundle_kind: str,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = _absolute_without_links(Path(output_path).expanduser())
    if output.suffix.lower() != ".zip":
        raise ValueError(f"Deterministic archive output must end in .zip: {output}")
    targets = _archive_bundle_targets(output)
    pending_transaction = os.path.lexists(bundle_journal_path(output))
    if not overwrite and not pending_transaction:
        for target in targets.values():
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
    kind = str(bundle_kind).strip()
    if not kind:
        raise ValueError("bundle_kind may not be empty.")
    _portable_json(kind, label="bundle_kind")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("archive metadata must be a JSON object.")
    portable_metadata = dict(metadata or {})
    _portable_json(portable_metadata, label="archive metadata")
    preflight_artifact_bundle(
        output,
        targets,
        overwrite=overwrite,
        validator=_validate_archive_bundle_identity,
        allow_partial_previous=overwrite,
        validate_previous=False,
    )
    prepared, records = _prepare_entries(entries)
    manifest = {
        "bundle_kind": kind,
        "files": records,
        "metadata": portable_metadata,
        "schema_version": ARCHIVE_SCHEMA_VERSION,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    staged: dict[str, Path] = {}
    staged_hashes: dict[str, str] = {}
    try:
        with _exclusive_bundle_stage(targets["archive"]) as (
            archive_stage,
            archive_handle,
        ):
            staged["archive"] = archive_stage
            _write_archive_stage(
                archive_handle,
                prepared=prepared,
                records=records,
                manifest_bytes=manifest_bytes,
            )
        archive_digest = sha256_file(staged["archive"])
        staged_hashes["archive"] = archive_digest
        sidecar = {
            "archive": output.name,
            "bundle_kind": kind,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "sha256": archive_digest,
            "size_bytes": staged["archive"].stat().st_size,
        }
        sidecar_bytes = canonical_json_bytes(sidecar)
        sha256_bytes = f"{archive_digest}  {output.name}\n".encode("ascii")
        with _exclusive_bundle_stage(targets["manifest"]) as (
            manifest_stage,
            manifest_handle,
        ):
            staged["manifest"] = manifest_stage
            _write_stage_bytes(manifest_handle, sidecar_bytes)
        staged_hashes["manifest"] = hashlib.sha256(sidecar_bytes).hexdigest()
        with _exclusive_bundle_stage(targets["sha256"]) as (
            sha256_stage,
            sha256_handle,
        ):
            staged["sha256"] = sha256_stage
            _write_stage_bytes(sha256_handle, sha256_bytes)
        staged_hashes["sha256"] = hashlib.sha256(sha256_bytes).hexdigest()
        promote_artifact_bundle(
            output,
            targets,
            staged,
            overwrite=overwrite,
            validator=_validate_archive_bundle_identity,
            allow_partial_previous=overwrite,
            validate_previous=False,
        )
    finally:
        cleanup_allowed = not bundle_journal_path(output).exists()
        for role, temporary in staged.items():
            if not cleanup_allowed:
                continue
            discard_temporary_bundle_path(
                temporary,
                targets[role],
                expected_sha256=staged_hashes.get(role),
            )
    return {"archive_path": str(output), "manifest": manifest, "sidecar": sidecar}


def validate_deterministic_zip(path: str | Path) -> dict[str, Any]:
    archive_path = _checked_source(path)
    errors: list[str] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            errors.extend(
                _canonical_zip_framing_errors(
                    archive_path,
                    member_count=len(infos),
                    central_directory_start=archive.start_dir,
                )
            )
            errors.extend(
                _canonical_local_header_errors(
                    archive_path,
                    infos,
                    central_directory_start=archive.start_dir,
                )
            )
            if archive.comment:
                errors.append("ZIP archive has a non-canonical comment.")
            if names != sorted(names):
                errors.append("ZIP members are not lexically sorted.")
            portable_names = [_portable_archive_key(name) for name in names]
            if len(portable_names) != len(set(portable_names)):
                errors.append("ZIP contains duplicate portable member names.")
            for info in infos:
                try:
                    _safe_archive_path(info.filename)
                except ValueError as exc:
                    errors.append(str(exc))
                if info.date_time != ARCHIVE_TIMESTAMP:
                    errors.append(f"ZIP member has a non-canonical timestamp: {info.filename}")
                if info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size:
                    errors.append(f"ZIP member is not stored without compression: {info.filename}")
                if (
                    info.create_system != 3
                    or info.external_attr != ARCHIVE_FILE_MODE << 16
                    or info.internal_attr != 0
                    or info.volume != 0
                    or info.reserved != 0
                ):
                    errors.append(f"ZIP member has non-canonical attributes: {info.filename}")
                if info.extra != _canonical_central_zip64_extra(info) or info.comment:
                    errors.append(f"ZIP member has non-canonical extra metadata: {info.filename}")
                expected_flags = 0 if info.filename.isascii() else 0x800
                if info.flag_bits != expected_flags:
                    errors.append(f"ZIP member has non-canonical flags: {info.filename}")
                expected_version = 20 if info.filename == ARCHIVE_MANIFEST_NAME else 45
                if info.create_version != expected_version or info.extract_version != expected_version:
                    errors.append(f"ZIP member has a non-canonical ZIP version: {info.filename}")

            manifest: Mapping[str, Any] = {}
            if names.count(ARCHIVE_MANIFEST_NAME) != 1:
                errors.append(f"ZIP must contain exactly one {ARCHIVE_MANIFEST_NAME}.")
            else:
                try:
                    raw_manifest = archive.read(ARCHIVE_MANIFEST_NAME)
                    parsed = json.loads(
                        raw_manifest,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"non-finite JSON constant {value!r}")
                        ),
                    )
                except (KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError, zipfile.BadZipFile) as exc:
                    errors.append(f"Invalid archive manifest: {exc}")
                else:
                    if not isinstance(parsed, Mapping):
                        errors.append("Archive manifest must be a JSON object.")
                    else:
                        manifest = parsed
                        try:
                            canonical = canonical_json_bytes(manifest)
                            _portable_json(manifest, label="archive manifest")
                        except (TypeError, ValueError) as exc:
                            errors.append(f"Invalid archive manifest: {exc}")
                        else:
                            if raw_manifest != canonical:
                                errors.append("Archive manifest is not canonical JSON.")
                        if set(manifest) != {"bundle_kind", "files", "metadata", "schema_version"}:
                            errors.append("Archive manifest has unexpected or missing fields.")
                        if manifest.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
                            errors.append("Archive manifest uses an unsupported schema version.")
                        if (
                            not isinstance(manifest.get("bundle_kind"), str)
                            or not str(manifest.get("bundle_kind", "")).strip()
                        ):
                            errors.append("Archive manifest bundle_kind must be a non-empty string.")
                        if not isinstance(manifest.get("metadata"), Mapping):
                            errors.append("Archive manifest metadata field must be an object.")

            records = manifest.get("files", []) if isinstance(manifest, Mapping) else []
            if not isinstance(records, list):
                errors.append("Archive manifest files field must be a list.")
                records = []
            valid_records: list[tuple[str, int, str]] = []
            record_names: list[str] = []
            for record in records:
                if not isinstance(record, Mapping):
                    errors.append("Archive manifest contains a non-mapping file record.")
                    continue
                if set(record) != {"path", "role", "sha256", "size_bytes"}:
                    errors.append("Archive manifest contains a file record with invalid fields.")
                name = record.get("path")
                role = record.get("role")
                digest = record.get("sha256")
                size = record.get("size_bytes")
                if not isinstance(name, str):
                    errors.append("Archive manifest file path must be a string.")
                    continue
                record_names.append(name)
                try:
                    _safe_archive_path(name)
                except ValueError as exc:
                    errors.append(str(exc))
                if _portable_archive_key(name) == _portable_archive_key(ARCHIVE_MANIFEST_NAME):
                    errors.append(f"Archive manifest may not list reserved member {name!r}.")
                if not isinstance(role, str) or not role.strip():
                    errors.append(f"Archive manifest role must be a non-empty string: {name}")
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    errors.append(f"Archive manifest size must be a nonnegative integer: {name}")
                    continue
                if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                    errors.append(f"Archive manifest SHA-256 is invalid: {name}")
                    continue
                valid_records.append((name, size, digest))

            record_keys = [_portable_archive_key(name) for name in record_names]
            if record_names != sorted(record_names) or len(record_keys) != len(set(record_keys)):
                errors.append("Archive manifest file records are not uniquely sorted.")
            expected_names = sorted([ARCHIVE_MANIFEST_NAME, *record_names])
            if names != expected_names:
                errors.append("ZIP members do not exactly match the archive manifest.")
            for name, expected_size, expected_digest in valid_records:
                if names.count(name) != 1:
                    continue
                try:
                    with archive.open(name, "r") as source:
                        size, digest = _copy_and_hash(source, None)
                except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    errors.append(f"Could not read ZIP member {name}: {exc}")
                    continue
                if size != expected_size:
                    errors.append(f"ZIP member size does not match its manifest: {name}")
                if digest != expected_digest:
                    errors.append(f"ZIP member digest does not match its manifest: {name}")
    except (OSError, UnicodeDecodeError, struct.error, NotImplementedError, zipfile.BadZipFile) as exc:
        errors.append(f"Invalid ZIP archive: {exc}")
    return {
        "archive_path": str(archive_path),
        "error_count": len(errors),
        "errors": errors,
        "sha256": sha256_file(archive_path),
        "status": "complete" if not errors else "failed",
    }
