from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import tempfile
import time
import unicodedata
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from genode.models.config import OTFlowConfig
from genode.data.otflow_medical_constants import (
    LONG_TERM_ST_DATASET_KEY,
    LONG_TERM_ST_FREQUENCY_LABEL,
    LONG_TERM_ST_HISTORY_LEN,
    LONG_TERM_ST_HORIZON_LEN,
    LONG_TERM_ST_SAMPLING_RATE_HZ,
    LONG_TERM_ST_SOURCE_SAMPLING_RATE_HZ,
    LONG_TERM_ST_STRIDE,
    long_term_st_manifest_path,
)
from genode.data.otflow_paths import long_term_st_data_path
from genode.path_safety import (
    first_link_or_reparse_component,
    is_link_or_reparse_point,
    portable_relative_path,
    resolve_portable_relative_path,
)
from genode.provenance import file_sha256

LONG_TERM_ST_EXPECTED_RECORDS = 86
LONG_TERM_ST_PATIENT_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("s20271", "s20272", "s20273", "s20274"),
    ("s30731", "s30732"),
    ("s30741", "s30742"),
    ("s30751", "s30752"),
)
_LONG_TERM_ST_PROMOTION_PROTOCOL = "long_term_st_dataset_promotion"
_LONG_TERM_ST_PROMOTION_VERSION = 1
_LONG_TERM_ST_PROMOTION_FIELDS = {
    "protocol",
    "version",
    "target_name",
    "staging_name",
    "staging_artifact_sha256",
    "previous_kind",
    "backup_name",
    "backup_artifact_sha256",
}

def medical_staging_root() -> Path:
    raw = str(os.environ.get("OTFLOW_MEDICAL_STAGING_ROOT", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    raise RuntimeError("Set OTFLOW_MEDICAL_STAGING_ROOT to prepare raw medical datasets.")


def _train_prefix_standardizer(values: np.ndarray, train_prefix_end: int) -> Tuple[float, float]:
    arr = np.asarray(values[: int(train_prefix_end)], dtype=np.float32)
    if arr.size <= 0:
        raise ValueError("Train prefix must be non-empty for normalization.")
    mean = float(arr.mean())
    std = float(arr.std())
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return mean, std


def _time_feature_dim(time_feature_mode: str) -> int:
    mode = str(time_feature_mode)
    if mode == "gap_elapsed":
        return 2
    if mode == "gap_only":
        return 1
    if mode == "none":
        return 0
    raise ValueError(f"Unknown time_feature_mode={time_feature_mode!r}")


def _regular_time_features(start: int, stop: int, *, time_feature_mode: str) -> Optional[np.ndarray]:
    length = max(0, int(stop) - int(start))
    dim = _time_feature_dim(str(time_feature_mode))
    if dim == 0:
        return None
    if length <= 0:
        return np.zeros((0, dim), dtype=np.float32)
    gap = np.zeros((length, 1), dtype=np.float32)
    if dim == 1:
        return gap
    elapsed = np.arange(int(start), int(stop), dtype=np.float32)[:, None]
    return np.concatenate([gap, elapsed], axis=1).astype(np.float32, copy=False)


_WINDOWS_RESERVED_FILE_STEMS = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _ascii_file_slug(name: str, *, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name)).encode("ascii", errors="ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
    slug = re.sub(r"_+", "_", slug)[:80].rstrip("._-") or str(fallback)
    if slug.split(".", maxsplit=1)[0].casefold() in _WINDOWS_RESERVED_FILE_STEMS:
        slug = f"channel_{slug}"
    return slug


def _safe_channel_name(name: str, *, channel_index: int, used_slugs: set[str]) -> str:
    base = _ascii_file_slug(name, fallback=f"channel_{int(channel_index)}")
    candidate = base
    if candidate.casefold() in used_slugs:
        candidate = f"{base}_{int(channel_index)}"
    suffix = 2
    while candidate.casefold() in used_slugs:
        candidate = f"{base}_{int(channel_index)}_{suffix}"
        suffix += 1
    used_slugs.add(candidate.casefold())
    return candidate


def _safe_record_name(record_id: str, *, record_index: int, used_slugs: set[str]) -> str:
    base = _ascii_file_slug(record_id, fallback=f"record_{int(record_index)}")
    candidate = base
    if candidate.casefold() in used_slugs:
        candidate = f"{base}_{int(record_index)}"
    suffix = 2
    while candidate.casefold() in used_slugs:
        candidate = f"{base}_{int(record_index)}_{suffix}"
        suffix += 1
    used_slugs.add(candidate.casefold())
    return candidate


def long_term_st_raw_archive_dir() -> Path:
    return medical_staging_root() / "raw" / "long_term_st"


def long_term_st_source_dir() -> Path:
    return medical_staging_root() / "extracted" / "long_term_st"


def _long_term_st_group_id(record_id: str) -> str:
    record = str(record_id)
    for group in LONG_TERM_ST_PATIENT_GROUPS:
        if record in group:
            return "_".join(group)
    return record


@dataclass(frozen=True)
class LongTermSTHeader:
    record_id: str
    n_sig: int
    sampling_rate_hz: float
    signal_length: int
    channel_names: Tuple[str, ...]
    dat_names: Tuple[str, ...]


@dataclass(frozen=True)
class LongTermSTSeriesSpec:
    series_id: str
    record_id: str
    group_id: str
    channel_index: int
    channel_name: str
    file_name: str
    split: str
    total_length: int
    source_total_length: int


def _coerce_archive_paths(archive_paths: Optional[Union[str, Path, Sequence[str | Path]]]) -> List[Path]:
    if archive_paths is None:
        root = long_term_st_raw_archive_dir()
        candidates = sorted(root.glob("long_term_st*.zip")) if root.exists() else []
    elif isinstance(archive_paths, (str, Path)):
        raw = str(archive_paths)
        parts = [part.strip() for part in raw.split(",") if part.strip()] if "," in raw else [raw]
        candidates = []
        for part in parts:
            path = Path(part).expanduser().resolve()
            if path.is_dir():
                candidates.extend(sorted(path.glob("long_term_st*.zip")))
            else:
                candidates.append(path)
    else:
        candidates = [Path(path).expanduser().resolve() for path in archive_paths]
    resolved = [path.resolve() for path in candidates if path.exists()]
    if not resolved:
        raise FileNotFoundError(
            "No Long-Term ST zip archives found. Place long_term_st*.zip under "
            f"{long_term_st_raw_archive_dir()} or pass archive_paths explicitly."
        )
    return sorted(resolved)


def _require_wfdb_for_long_term_st_preparation() -> None:
    if importlib.util.find_spec("wfdb") is None:
        raise ImportError(
            "wfdb is required to prepare raw Long-Term ST data. "
            "Install the medical extra with: python -m pip install -e .[medical]"
        )


def _parse_long_term_st_header(record_id: str, text: str) -> LongTermSTHeader:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty WFDB header for {record_id}.")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Malformed WFDB header line for {record_id}: {lines[0]!r}")
    n_sig = int(first[1])
    sampling_rate_hz = float(first[2])
    signal_length = int(first[3])
    if len(lines) < 1 + n_sig:
        raise ValueError(f"WFDB header for {record_id} has fewer signal lines than n_sig={n_sig}.")
    dat_names: List[str] = []
    channel_names: List[str] = []
    for channel_index, line in enumerate(lines[1 : 1 + n_sig]):
        parts = line.split()
        if not parts:
            raise ValueError(f"Malformed signal line {channel_index} for {record_id}.")
        dat_names.append(Path(parts[0]).name)
        channel_names.append(str(parts[-1]) if len(parts) > 1 else f"channel_{channel_index}")
    return LongTermSTHeader(
        record_id=str(record_id),
        n_sig=int(n_sig),
        sampling_rate_hz=float(sampling_rate_hz),
        signal_length=int(signal_length),
        channel_names=tuple(channel_names),
        dat_names=tuple(dat_names),
    )


def _scan_long_term_st_archives(archive_paths: Sequence[Path]) -> Tuple[Dict[str, LongTermSTHeader], Dict[str, Tuple[Path, str]], List[Dict[str, Any]]]:
    headers: Dict[str, LongTermSTHeader] = {}
    dat_members: Dict[str, Tuple[Path, str]] = {}
    archive_rows: List[Dict[str, Any]] = []
    for archive_path in archive_paths:
        archive_rows.append(
            {
                "name": str(archive_path.name),
                "size_bytes": int(archive_path.stat().st_size),
                "sha256": file_sha256(archive_path),
            }
        )
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                member_name = str(info.filename)
                base_name = Path(member_name).name
                lower = base_name.lower()
                if lower.endswith(".hea"):
                    record_id = Path(base_name).stem
                    text = zf.read(info).decode("utf-8", errors="replace")
                    headers[record_id] = _parse_long_term_st_header(record_id, text)
                elif lower.endswith(".dat"):
                    dat_members[base_name] = (archive_path, member_name)
    return headers, dat_members, archive_rows


def _copy_zip_member(member: Tuple[Path, str], *, target_root: Path, target_name: str) -> None:
    archive_path, member_name = member
    relative = portable_relative_path(target_name, label="Long-Term ST extraction file")
    if len(relative.parts) != 1:
        raise ValueError(f"Long-Term ST extraction file must be a basename: {target_name!r}.")
    absolute_target_root = target_root.expanduser().absolute()
    indirect = first_link_or_reparse_component(
        absolute_target_root,
        root=Path(absolute_target_root.anchor),
    )
    if indirect is not None:
        raise ValueError(
            "Long-Term ST extraction destination traverses a symlink, junction, or reparse point: "
            f"{indirect}."
        )
    target_root.mkdir(parents=True, exist_ok=True)
    indirect = first_link_or_reparse_component(
        absolute_target_root,
        root=Path(absolute_target_root.anchor),
    )
    if indirect is not None:
        raise ValueError(
            "Long-Term ST extraction destination traverses a symlink, junction, or reparse point: "
            f"{indirect}."
        )
    target = resolve_portable_relative_path(
        target_root,
        relative.as_posix(),
        label="Long-Term ST extraction file",
        reject_links=True,
    )
    with zipfile.ZipFile(archive_path) as zf:
        with zf.open(member_name) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)


def _extract_long_term_st_wfdb_members(
    *,
    source_dir: Path,
    archive_paths: Sequence[Path],
    headers: Mapping[str, LongTermSTHeader],
    dat_members: Mapping[str, Tuple[Path, str]],
) -> List[str]:
    absolute_source_dir = source_dir.expanduser().absolute()
    indirect = first_link_or_reparse_component(
        absolute_source_dir,
        root=Path(absolute_source_dir.anchor),
    )
    if indirect is not None:
        raise ValueError(
            "Long-Term ST extraction destination may not traverse a symlink, junction, "
            f"or reparse point: {indirect}."
        )
    source_dir.mkdir(parents=True, exist_ok=True)
    indirect = first_link_or_reparse_component(
        absolute_source_dir,
        root=Path(absolute_source_dir.anchor),
    )
    if indirect is not None:
        raise ValueError(
            "Long-Term ST extraction destination may not traverse a symlink, junction, "
            f"or reparse point: {indirect}."
        )
    missing_dat_names: List[str] = []

    header_members: Dict[str, Tuple[Path, str]] = {}
    for archive_path in archive_paths:
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                if not info.is_dir() and Path(info.filename).name.lower().endswith(".hea"):
                    header_members[Path(info.filename).name] = (archive_path, str(info.filename))

    for record_id in sorted(headers):
        header_name = f"{record_id}.hea"
        member = header_members.get(header_name)
        if member is None:
            continue
        _copy_zip_member(member, target_root=source_dir, target_name=header_name)

    for dat_name in sorted({name for header in headers.values() for name in header.dat_names}):
        member = dat_members.get(dat_name)
        if member is None:
            missing_dat_names.append(str(dat_name))
            continue
        _copy_zip_member(member, target_root=source_dir, target_name=dat_name)
    return missing_dat_names


def _split_long_term_st_groups(group_ids: Sequence[str], train_frac: float, val_frac: float) -> Dict[str, str]:
    groups = sorted(set(str(group_id) for group_id in group_ids))
    if len(groups) < 3:
        raise ValueError("Long-Term ST requires at least 3 record groups for train/val/test splits.")
    train_count = max(1, int(round(len(groups) * float(train_frac))))
    val_count = max(1, int(round(len(groups) * float(val_frac))))
    if train_count + val_count >= len(groups):
        val_count = max(1, len(groups) - train_count - 1)
    if train_count + val_count >= len(groups):
        train_count = max(1, len(groups) - val_count - 1)
    split_by_group: Dict[str, str] = {}
    for idx, group_id in enumerate(groups):
        if idx < train_count:
            split_by_group[group_id] = "train"
        elif idx < train_count + val_count:
            split_by_group[group_id] = "val"
        else:
            split_by_group[group_id] = "test"
    return split_by_group


def _iter_manifest_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_manifest_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_manifest_strings(nested)


def _looks_like_local_path(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    if text.startswith("~"):
        return True
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if posix.is_absolute() or bool(windows.drive) or bool(windows.root):
        return True
    lowered = text.replace("\\", "/").lower()
    markers = (
        "/" + "home/",
        "/" + "users/",
        "/" + "mnt/",
        "/" + "tmp/",
    )
    return any(marker in lowered for marker in markers)


def _validate_long_term_st_series_file_name(file_name: Any, prepared_dir: Path) -> Path:
    text = str(file_name or "")
    posix = portable_relative_path(text, label="Long-Term ST manifest series file")
    if not posix.parts or posix.parts[0] != "series" or posix.suffix != ".npy":
        raise ValueError(f"Long-Term ST manifest series file must live under series/*.npy: {text!r}.")
    return resolve_portable_relative_path(
        prepared_dir,
        posix.as_posix(),
        label="Long-Term ST manifest series file",
        reject_links=True,
    )


def _validate_long_term_st_manifest_series_specs(
    payload: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    rows = payload.get("series_specs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Long-Term ST manifest must contain non-empty series_specs.")

    prepared_dir = manifest_path.parent.resolve()
    group_split: Dict[str, str] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    seen_series_ids: set[str] = set()
    managed_file_names: set[str] = set()
    seen_file_identities: set[str] = set()
    known_group_by_record = {
        record_id: "_".join(group)
        for group in LONG_TERM_ST_PATIENT_GROUPS
        for record_id in group
    }
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Long-Term ST series_specs[{idx}] must be an object.")
        series_id = str(row.get("series_id", "") or "").strip()
        if not series_id:
            raise ValueError(f"Long-Term ST series_specs[{idx}] must include series_id.")
        if series_id in seen_series_ids:
            raise ValueError(f"Long-Term ST manifest contains duplicate series_id={series_id!r}.")
        seen_series_ids.add(series_id)
        file_name = str(row.get("file_name", "") or "")
        file_identity = file_name.casefold()
        if file_identity in seen_file_identities:
            raise ValueError(f"Long-Term ST manifest contains duplicate file_name={file_name!r}.")
        seen_file_identities.add(file_identity)
        managed_file_names.add(file_name)
        split = str(row.get("split", "")).strip()
        if split not in split_counts:
            raise ValueError(f"Long-Term ST series_specs[{idx}] has invalid split={split!r}.")
        split_counts[split] += 1

        record_id = str(row.get("record_id", "")).strip()
        group_id = str(row.get("group_id", "")).strip()
        if not record_id or not group_id:
            raise ValueError(f"Long-Term ST series_specs[{idx}] must include record_id and group_id.")
        expected_group = known_group_by_record.get(record_id)
        if expected_group is not None and group_id != expected_group:
            raise ValueError(
                f"Long-Term ST manifest record {record_id} must use same-patient group {expected_group!r}, "
                f"got {group_id!r}."
            )

        prior_split = group_split.setdefault(group_id, split)
        if prior_split != split:
            raise ValueError(
                f"Long-Term ST manifest group {group_id!r} appears in multiple splits: "
                f"{prior_split!r} and {split!r}."
            )

        resolved_file = _validate_long_term_st_series_file_name(file_name, prepared_dir)
        if not resolved_file.is_file():
            raise ValueError(f"Long-Term ST series file is missing: {row.get('file_name')!r}.")

    empty_splits = [split for split, count in split_counts.items() if int(count) <= 0]
    if empty_splits:
        raise ValueError(f"Long-Term ST manifest has empty split(s): {', '.join(empty_splits)}.")

    managed_files = {manifest_path.name, *managed_file_names}
    managed_directories: set[str] = set()
    for file_name in managed_file_names:
        parent = PurePosixPath(file_name).parent
        while parent != PurePosixPath("."):
            managed_directories.add(parent.as_posix())
            parent = parent.parent

    def raise_walk_error(error: OSError) -> None:
        raise ValueError(
            f"Could not inspect Long-Term ST prepared output safely: {error}."
        ) from error

    for root_text, directory_names, file_names in os.walk(
        prepared_dir,
        topdown=True,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        root = Path(root_text)
        for name in directory_names:
            candidate = root / name
            if is_link_or_reparse_point(candidate):
                raise ValueError(
                    "Long-Term ST prepared output may not contain symlinks, junctions, "
                    f"or reparse points: {candidate}."
                )
            relative = candidate.relative_to(prepared_dir).as_posix()
            if relative not in managed_directories:
                raise ValueError(
                    f"Long-Term ST prepared output contains an unmanaged directory: {relative!r}."
                )
        for name in file_names:
            candidate = root / name
            if is_link_or_reparse_point(candidate):
                raise ValueError(
                    "Long-Term ST prepared output may not contain symlinks, junctions, "
                    f"or reparse points: {candidate}."
                )
            relative = candidate.relative_to(prepared_dir).as_posix()
            if relative in managed_files:
                continue
            raise ValueError(
                f"Long-Term ST prepared output contains an unmanaged file: {relative!r}."
            )


def _long_term_st_manifest_task_lengths(payload: Mapping[str, Any]) -> Tuple[int, int]:
    values: list[int] = []
    for field in ("history_len", "future_block_len"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
            raise ValueError(
                f"Long-Term ST manifest {field} must be a positive integer, got {value!r}."
            )
        values.append(int(value))
    return values[0], values[1]


def _validate_long_term_st_manifest(
    path: Path,
    *,
    history_len: int | None,
    horizon: int | None,
) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("dataset_key")) != LONG_TERM_ST_DATASET_KEY:
        raise ValueError(f"Unexpected Long-Term ST manifest dataset_key={payload.get('dataset_key')!r}.")
    declared_history_len, declared_horizon = _long_term_st_manifest_task_lengths(payload)
    if (history_len is None) != (horizon is None):
        raise ValueError("Long-Term ST manifest validation requires both task lengths or neither.")
    expected_history_len = declared_history_len if history_len is None else int(history_len)
    expected_horizon = declared_horizon if horizon is None else int(horizon)
    if declared_history_len != expected_history_len or declared_horizon != expected_horizon:
        raise ValueError(
            "Existing Long-Term ST manifest does not match requested task: "
            f"history_len={payload.get('history_len')}, future_block_len={payload.get('future_block_len')}, "
            f"requested history_len={expected_history_len}, horizon={expected_horizon}."
        )
    if any(_looks_like_local_path(value) for value in _iter_manifest_strings(payload)):
        raise ValueError("Existing Long-Term ST manifest contains local filesystem paths; regenerate it.")
    _validate_long_term_st_manifest_series_specs(payload, path)
    return payload


def _validate_long_term_st_output_root(
    prepared_dir: Path,
    *,
    force: bool,
    history_len: int,
    horizon: int,
) -> Dict[str, Any] | None:
    if not prepared_dir.exists():
        return None
    if is_link_or_reparse_point(prepared_dir) or not prepared_dir.is_dir():
        raise ValueError(
            "Long-Term ST prepared destination must be a regular directory, not a "
            "symlink, junction, reparse point, or file: "
            f"{prepared_dir}."
        )
    if not any(prepared_dir.iterdir()):
        return None
    manifest_path = long_term_st_manifest_path(prepared_dir)
    if not manifest_path.is_file() or is_link_or_reparse_point(manifest_path):
        raise ValueError(
            "A non-empty Long-Term ST prepared destination requires an intact regular manifest."
        )
    return _validate_long_term_st_manifest(
        manifest_path,
        history_len=None if bool(force) else int(history_len),
        horizon=None if bool(force) else int(horizon),
    )


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _sync_long_term_st_directory(directory: Path) -> None:
    """Persist directory entries where stdlib directory fsync is supported.

    POSIX directory fsync failures are surfaced because they weaken transaction
    ordering. Windows does not expose portable directory handles through
    ``os.open``; an attempted directory fsync is therefore best effort there.
    """

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            if os.name != "nt":
                raise
    finally:
        os.close(descriptor)


def _replace_long_term_st_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        os.rename(source, destination)
    else:
        os.replace(source, destination)
    _sync_long_term_st_directory(destination.parent)


def _unlink_long_term_st_path(path: Path) -> None:
    path.unlink()
    _sync_long_term_st_directory(path.parent)


def _long_term_st_lock_path(prepared_dir: Path) -> Path:
    return prepared_dir.with_name(f".{prepared_dir.name}.lock")


def _open_long_term_st_lock(path: Path) -> int:
    if is_link_or_reparse_point(path):
        raise ValueError("Long-Term ST lock path must be a regular file.")
    base_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    base_flags |= getattr(os, "O_NOINHERIT", 0)
    base_flags |= getattr(os, "O_NOFOLLOW", 0)
    created = False
    if path.exists():
        if not path.is_file():
            raise ValueError("Long-Term ST lock path must be a regular file.")
        descriptor = os.open(path, base_flags)
    else:
        try:
            descriptor = os.open(
                path,
                base_flags | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            created = True
        except FileExistsError:
            if is_link_or_reparse_point(path) or not path.is_file():
                raise ValueError("Long-Term ST lock path must be a regular file.")
            descriptor = os.open(path, base_flags)

    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise ValueError("Long-Term ST lock path must be a regular file.")
        if is_link_or_reparse_point(path) or not path.is_file():
            raise ValueError("Long-Term ST lock path changed during acquisition.")
        path_stat = path.stat(follow_symlinks=False)
        if (
            int(path_stat.st_dev),
            int(path_stat.st_ino),
        ) != (
            int(descriptor_stat.st_dev),
            int(descriptor_stat.st_ino),
        ):
            raise ValueError("Long-Term ST lock path changed during acquisition.")
        os.set_inheritable(descriptor, False)
        if created:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
            _sync_long_term_st_directory(path.parent)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _long_term_st_target_lock(prepared_dir: Path) -> Iterator[Path]:
    """Hold an interprocess lock for one prepared Long-Term ST target."""

    lock_path = _long_term_st_lock_path(prepared_dir)
    descriptor = _open_long_term_st_lock(lock_path)
    try:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                f"Long-Term ST target {prepared_dir.name!r} is locked by another preparation process."
            ) from exc
        yield lock_path
    finally:
        os.close(descriptor)


def _long_term_st_promotion_journal_path(prepared_dir: Path) -> Path:
    return prepared_dir.with_name(f".{prepared_dir.name}.promotion.json")


def _long_term_st_promotion_sibling(
    prepared_dir: Path,
    name: object,
    *,
    kind: str,
) -> Path:
    text = str(name)
    expected_prefix = f".{prepared_dir.name}.{kind}-"
    if not text or Path(text).name != text or not text.startswith(expected_prefix):
        raise ValueError(
            f"Long-Term ST promotion journal contains an invalid {kind} directory name."
        )
    path = prepared_dir.parent / text
    if path == prepared_dir or path == _long_term_st_promotion_journal_path(prepared_dir):
        raise ValueError(
            f"Long-Term ST promotion journal contains an unsafe {kind} directory name."
        )
    return path


def _managed_long_term_st_file_sha256(path: Path, *, sync: bool) -> str:
    if is_link_or_reparse_point(path) or not path.is_file():
        raise ValueError("Managed Long-Term ST artifact files must be regular files.")
    flags = (os.O_RDWR if sync else os.O_RDONLY) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOINHERIT", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or is_link_or_reparse_point(path)
            or (int(path_stat.st_dev), int(path_stat.st_ino))
            != (int(descriptor_stat.st_dev), int(descriptor_stat.st_ino))
        ):
            raise ValueError("Managed Long-Term ST artifact file changed during hashing.")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if sync:
            os.fsync(descriptor)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _validated_long_term_st_artifact_sha256(
    root: Path,
    *,
    sync: bool = False,
) -> str:
    if is_link_or_reparse_point(root) or not root.is_dir():
        raise ValueError("Managed Long-Term ST artifact must be a regular directory.")
    root = root.resolve(strict=True)
    manifest_path = long_term_st_manifest_path(root)
    if is_link_or_reparse_point(manifest_path) or not manifest_path.is_file():
        raise ValueError(
            "Safe Long-Term ST replacement requires an intact regular manifest."
        )
    payload = _validate_long_term_st_manifest(
        manifest_path,
        history_len=None,
        horizon=None,
    )
    managed_paths = {
        manifest_path.name: manifest_path,
        **{
            str(row["file_name"]): _validate_long_term_st_series_file_name(
                row["file_name"],
                root,
            )
            for row in payload["series_specs"]
        },
    }
    file_records = [
        {
            "path": relative,
            "sha256": _managed_long_term_st_file_sha256(path, sync=sync),
        }
        for relative, path in sorted(managed_paths.items())
    ]
    if sync:
        managed_directories = {root}
        for path in managed_paths.values():
            parent = path.parent
            while parent != root:
                if root not in parent.parents:
                    raise ValueError(
                        "Managed Long-Term ST artifact file escapes its artifact root."
                    )
                managed_directories.add(parent)
                parent = parent.parent
        for directory in sorted(
            managed_directories,
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            _sync_long_term_st_directory(directory)
        _sync_long_term_st_directory(root.parent)
    encoded = json.dumps(
        {
            "protocol": "long_term_st_managed_artifact",
            "files": file_records,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_long_term_st_promotion_journal(
    path: Path,
    record: Mapping[str, object],
) -> None:
    if is_link_or_reparse_point(path) or path.exists():
        raise ValueError(
            "Refusing to replace an existing Long-Term ST promotion journal "
            "without recovering it first."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(record), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_long_term_st_path(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_long_term_st_promotion_journal(
    prepared_dir: Path,
) -> Dict[str, object] | None:
    path = _long_term_st_promotion_journal_path(prepared_dir)
    if is_link_or_reparse_point(path):
        raise ValueError("Long-Term ST promotion journal must be a regular file.")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("Long-Term ST promotion journal must be a regular file.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read Long-Term ST promotion journal safely: {exc}"
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != _LONG_TERM_ST_PROMOTION_FIELDS:
        raise ValueError("Long-Term ST promotion journal has an invalid schema.")
    if (
        payload.get("protocol") != _LONG_TERM_ST_PROMOTION_PROTOCOL
        or type(payload.get("version")) is not int
        or payload.get("version") != _LONG_TERM_ST_PROMOTION_VERSION
        or payload.get("target_name") != prepared_dir.name
    ):
        raise ValueError(
            "Long-Term ST promotion journal identity does not match the requested artifact."
        )
    previous_kind = payload.get("previous_kind")
    if previous_kind not in {"absent", "empty", "artifact"}:
        raise ValueError(
            "Long-Term ST promotion journal contains an invalid previous artifact kind."
        )
    if not _is_sha256(payload.get("staging_artifact_sha256")):
        raise ValueError(
            "Long-Term ST promotion journal contains an invalid staging artifact hash."
        )
    _long_term_st_promotion_sibling(
        prepared_dir,
        payload.get("staging_name"),
        kind="staging",
    )
    backup_name = payload.get("backup_name")
    backup_hash = payload.get("backup_artifact_sha256")
    if previous_kind == "absent":
        if backup_name != "" or backup_hash != "":
            raise ValueError(
                "Long-Term ST promotion journal has an invalid absent-target backup."
            )
    else:
        _long_term_st_promotion_sibling(
            prepared_dir,
            backup_name,
            kind="backup",
        )
        if previous_kind == "artifact" and not _is_sha256(backup_hash):
            raise ValueError(
                "Long-Term ST promotion journal contains an invalid backup artifact hash."
            )
        if previous_kind == "empty" and backup_hash != "":
            raise ValueError(
                "Long-Term ST promotion journal has an invalid empty-directory backup hash."
            )
    return dict(payload)


def _remove_verified_long_term_st_artifact(
    root: Path,
    expected_artifact_sha256: str,
) -> None:
    actual_hash = _validated_long_term_st_artifact_sha256(root)
    if actual_hash != expected_artifact_sha256:
        raise ValueError(
            "Refusing to remove a Long-Term ST artifact with an unexpected hash."
        )
    shutil.rmtree(root)
    _sync_long_term_st_directory(root.parent)


def _discard_obsolete_long_term_st_artifact(
    root: Path,
    expected_artifact_sha256: str,
) -> None:
    """Remove a known obsolete artifact, preserving anything unexpected."""

    if not root.exists() and not is_link_or_reparse_point(root):
        return
    try:
        _remove_verified_long_term_st_artifact(root, expected_artifact_sha256)
    except (OSError, ValueError):
        return


def _require_empty_long_term_st_directory(path: Path, *, label: str) -> None:
    if is_link_or_reparse_point(path) or not path.is_dir() or any(path.iterdir()):
        raise ValueError(
            f"Long-Term ST promotion {label} must be an empty regular directory."
        )


def _discard_obsolete_empty_long_term_st_directory(path: Path) -> None:
    if not path.exists() and not is_link_or_reparse_point(path):
        return
    try:
        _require_empty_long_term_st_directory(path, label="backup")
        path.rmdir()
        _sync_long_term_st_directory(path.parent)
    except (OSError, ValueError):
        return


def _remove_long_term_st_staging_after_failure(
    staging_dir: Path,
    prepared_dir: Path,
) -> None:
    _long_term_st_promotion_sibling(
        prepared_dir,
        staging_dir.name,
        kind="staging",
    )
    if is_link_or_reparse_point(staging_dir) or not staging_dir.is_dir():
        raise ValueError("Refusing to remove an unsafe Long-Term ST staging path.")
    for path in staging_dir.rglob("*"):
        if is_link_or_reparse_point(path):
            raise ValueError(
                "Refusing to remove a Long-Term ST staging directory containing links."
            )
    shutil.rmtree(staging_dir)
    _sync_long_term_st_directory(staging_dir.parent)


def _prepare_long_term_st_promotion_journal(
    staging_dir: Path,
    prepared_dir: Path,
) -> Dict[str, object]:
    staging_path = _long_term_st_promotion_sibling(
        prepared_dir,
        staging_dir.name,
        kind="staging",
    )
    if staging_path != staging_dir:
        raise ValueError(
            "Long-Term ST staging directory must be an absolute sibling of the target."
        )
    staging_hash = _validated_long_term_st_artifact_sha256(
        staging_dir,
        sync=True,
    )
    if not prepared_dir.exists():
        previous_kind = "absent"
        backup_name = ""
        backup_hash = ""
    else:
        if is_link_or_reparse_point(prepared_dir) or not prepared_dir.is_dir():
            raise ValueError("Long-Term ST target must be a regular directory.")
        backup_dir = prepared_dir.with_name(
            f".{prepared_dir.name}.backup-{time.time_ns()}"
        )
        while backup_dir.exists() or is_link_or_reparse_point(backup_dir):
            backup_dir = prepared_dir.with_name(
                f".{prepared_dir.name}.backup-{time.time_ns()}"
            )
        backup_name = backup_dir.name
        if any(prepared_dir.iterdir()):
            previous_kind = "artifact"
            backup_hash = _validated_long_term_st_artifact_sha256(prepared_dir)
        else:
            previous_kind = "empty"
            backup_hash = ""
    record: Dict[str, object] = {
        "protocol": _LONG_TERM_ST_PROMOTION_PROTOCOL,
        "version": _LONG_TERM_ST_PROMOTION_VERSION,
        "target_name": prepared_dir.name,
        "staging_name": staging_dir.name,
        "staging_artifact_sha256": staging_hash,
        "previous_kind": previous_kind,
        "backup_name": backup_name,
        "backup_artifact_sha256": backup_hash,
    }
    _write_long_term_st_promotion_journal(
        _long_term_st_promotion_journal_path(prepared_dir),
        record,
    )
    return record


def _recover_long_term_st_promotion(prepared_dir: Path) -> None:
    record = _load_long_term_st_promotion_journal(prepared_dir)
    if record is None:
        return
    journal = _long_term_st_promotion_journal_path(prepared_dir)
    staging_dir = _long_term_st_promotion_sibling(
        prepared_dir,
        record["staging_name"],
        kind="staging",
    )
    staging_hash = str(record["staging_artifact_sha256"])
    previous_kind = str(record["previous_kind"])
    backup_dir = (
        None
        if previous_kind == "absent"
        else _long_term_st_promotion_sibling(
            prepared_dir,
            record["backup_name"],
            kind="backup",
        )
    )
    backup_hash = str(record["backup_artifact_sha256"])

    target_hash: str | None = None
    target_is_empty = False
    if prepared_dir.exists():
        if is_link_or_reparse_point(prepared_dir) or not prepared_dir.is_dir():
            raise ValueError(
                "Cannot recover Long-Term ST promotion because the target is not a regular directory."
            )
        if any(prepared_dir.iterdir()):
            target_hash = _validated_long_term_st_artifact_sha256(prepared_dir)
        else:
            target_is_empty = True

    if target_hash == staging_hash:
        if staging_dir.exists():
            _discard_obsolete_long_term_st_artifact(staging_dir, staging_hash)
        if backup_dir is not None and backup_dir.exists():
            if previous_kind == "artifact":
                _discard_obsolete_long_term_st_artifact(backup_dir, backup_hash)
            else:
                _discard_obsolete_empty_long_term_st_directory(backup_dir)
        _unlink_long_term_st_path(journal)
        return

    target_matches_previous = (
        previous_kind == "artifact"
        and target_hash == backup_hash
        or previous_kind == "empty"
        and target_is_empty
    )
    if target_matches_previous:
        if backup_dir is not None and backup_dir.exists():
            raise ValueError(
                "Cannot recover Long-Term ST promotion with both target and backup present."
            )
        if staging_dir.exists():
            _discard_obsolete_long_term_st_artifact(staging_dir, staging_hash)
        _unlink_long_term_st_path(journal)
        return

    if not prepared_dir.exists() and backup_dir is not None and backup_dir.exists():
        if previous_kind == "artifact":
            actual_backup_hash = _validated_long_term_st_artifact_sha256(backup_dir)
            if actual_backup_hash != backup_hash:
                raise ValueError(
                    "Cannot recover Long-Term ST promotion from an unexpected backup artifact."
                )
        else:
            _require_empty_long_term_st_directory(backup_dir, label="backup")
        _replace_long_term_st_path(backup_dir, prepared_dir)
        if previous_kind == "artifact":
            restored_hash = _validated_long_term_st_artifact_sha256(prepared_dir)
            if restored_hash != backup_hash:
                raise ValueError(
                    "Restored Long-Term ST artifact does not match the journaled backup."
                )
        else:
            _require_empty_long_term_st_directory(prepared_dir, label="target")
        if staging_dir.exists():
            _discard_obsolete_long_term_st_artifact(staging_dir, staging_hash)
        _unlink_long_term_st_path(journal)
        return

    if (
        not prepared_dir.exists()
        and previous_kind == "absent"
        and backup_dir is None
    ):
        if staging_dir.exists():
            _discard_obsolete_long_term_st_artifact(staging_dir, staging_hash)
        _unlink_long_term_st_path(journal)
        return

    raise ValueError(
        "Cannot recover Long-Term ST promotion because journaled paths do not match a safe state."
    )


def _promote_long_term_st_dataset(
    staging_dir: Path,
    prepared_dir: Path,
) -> None:
    record = _prepare_long_term_st_promotion_journal(staging_dir, prepared_dir)
    previous_kind = str(record["previous_kind"])
    backup_dir = (
        None
        if previous_kind == "absent"
        else _long_term_st_promotion_sibling(
            prepared_dir,
            record["backup_name"],
            kind="backup",
        )
    )
    try:
        if backup_dir is not None:
            _replace_long_term_st_path(prepared_dir, backup_dir)
        _replace_long_term_st_path(staging_dir, prepared_dir)
        _recover_long_term_st_promotion(prepared_dir)
    except BaseException as exc:
        try:
            _recover_long_term_st_promotion(prepared_dir)
        except BaseException as recovery_error:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"Automatic Long-Term ST promotion recovery also failed: {recovery_error}"
                )
        raise


def prepare_long_term_st_dataset(
    out_dir: str | Path | None = None,
    *,
    archive_paths: Optional[Union[str, Path, Sequence[str | Path]]] = None,
    force: bool = False,
    expected_record_count: Optional[int] = LONG_TERM_ST_EXPECTED_RECORDS,
    history_len: int = LONG_TERM_ST_HISTORY_LEN,
    horizon: int = LONG_TERM_ST_HORIZON_LEN,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Dict[str, Any]:
    requested_prepared_dir = Path(
        out_dir or long_term_st_data_path()
    ).expanduser().absolute()
    if is_link_or_reparse_point(requested_prepared_dir):
        raise ValueError(
            "Long-Term ST prepared destination may not be a symlink, junction, "
            f"or reparse point: {requested_prepared_dir}."
        )
    anchor = Path(requested_prepared_dir.anchor)
    indirect = first_link_or_reparse_component(
        requested_prepared_dir,
        root=anchor,
    )
    if indirect is not None:
        raise ValueError(
            "Long-Term ST prepared destination may not traverse a symlink, junction, "
            f"or reparse point: {indirect}."
        )
    requested_prepared_dir.parent.mkdir(parents=True, exist_ok=True)
    indirect = first_link_or_reparse_component(
        requested_prepared_dir,
        root=anchor,
    )
    if indirect is not None:
        raise ValueError(
            "Long-Term ST prepared destination may not traverse a symlink, junction, "
            f"or reparse point: {indirect}."
        )
    prepared_dir = requested_prepared_dir.resolve()
    with _long_term_st_target_lock(prepared_dir):
        return _prepare_long_term_st_dataset_while_locked(
            prepared_dir,
            archive_paths=archive_paths,
            force=bool(force),
            expected_record_count=expected_record_count,
            history_len=int(history_len),
            horizon=int(horizon),
            train_frac=float(train_frac),
            val_frac=float(val_frac),
        )


def _prepare_long_term_st_dataset_while_locked(
    prepared_dir: Path,
    *,
    archive_paths: Optional[Union[str, Path, Sequence[str | Path]]],
    force: bool,
    expected_record_count: Optional[int],
    history_len: int,
    horizon: int,
    train_frac: float,
    val_frac: float,
) -> Dict[str, Any]:
    _recover_long_term_st_promotion(prepared_dir)
    existing = _validate_long_term_st_output_root(
        prepared_dir,
        force=force,
        history_len=history_len,
        horizon=horizon,
    )
    if existing is not None and not force:
        return existing

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{prepared_dir.name}.staging-",
            dir=prepared_dir.parent,
        )
    ).resolve()
    try:
        _prepare_long_term_st_dataset_into(
            staging_dir,
            archive_paths=archive_paths,
            expected_record_count=expected_record_count,
            history_len=history_len,
            horizon=horizon,
            train_frac=train_frac,
            val_frac=val_frac,
        )
        _validate_long_term_st_manifest(
            long_term_st_manifest_path(staging_dir),
            history_len=history_len,
            horizon=horizon,
        )
        _promote_long_term_st_dataset(staging_dir, prepared_dir)
        return _validate_long_term_st_manifest(
            long_term_st_manifest_path(prepared_dir),
            history_len=history_len,
            horizon=horizon,
        )
    finally:
        if (
            staging_dir.exists()
            and not _long_term_st_promotion_journal_path(prepared_dir).exists()
        ):
            _remove_long_term_st_staging_after_failure(staging_dir, prepared_dir)


def _prepare_long_term_st_dataset_into(
    prepared_dir: Path,
    *,
    archive_paths: Optional[Union[str, Path, Sequence[str | Path]]],
    expected_record_count: Optional[int],
    history_len: int,
    horizon: int,
    train_frac: float,
    val_frac: float,
) -> Dict[str, Any]:
    manifest_path = long_term_st_manifest_path(prepared_dir)

    _require_wfdb_for_long_term_st_preparation()
    resolved_archives = _coerce_archive_paths(archive_paths)
    headers, dat_members, archive_rows = _scan_long_term_st_archives(resolved_archives)
    if expected_record_count is not None and len(headers) != int(expected_record_count):
        raise ValueError(
            f"Expected {int(expected_record_count)} Long-Term ST headers, found {len(headers)} in archives."
        )

    source_dir = long_term_st_source_dir()
    missing_dat_names = _extract_long_term_st_wfdb_members(
        source_dir=source_dir,
        archive_paths=resolved_archives,
        headers=headers,
        dat_members=dat_members,
    )

    try:
        import wfdb
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ImportError(
            "wfdb and scipy are required to prepare raw Long-Term ST data. "
            "Install the medical extra with: python -m pip install -e .[medical]"
        ) from exc

    prepared_series_dir = prepared_dir / "series"
    if is_link_or_reparse_point(prepared_series_dir):
        raise ValueError(
            f"Long-Term ST series destination may not be a symlink, junction, or reparse point: {prepared_series_dir}."
        )
    prepared_series_dir.mkdir(parents=True, exist_ok=True)
    missing_dat_set = set(missing_dat_names)
    skipped_records: List[Dict[str, str]] = []
    series_rows: List[Dict[str, Any]] = []
    used_records: set[str] = set()
    min_prepared_length: Optional[int] = None
    max_prepared_length: Optional[int] = None

    used_record_slugs: set[str] = set()
    for record_index, (record_id, header) in enumerate(sorted(headers.items())):
        if any(dat_name in missing_dat_set for dat_name in header.dat_names):
            skipped_records.append({"record_id": str(record_id), "reason": "missing_referenced_dat"})
            continue
        if abs(float(header.sampling_rate_hz) - float(LONG_TERM_ST_SOURCE_SAMPLING_RATE_HZ)) > 1e-6:
            skipped_records.append({"record_id": str(record_id), "reason": "unexpected_sampling_rate"})
            continue
        record_path = source_dir / str(record_id)
        try:
            tail_start = max(0, int(header.signal_length) - 1000)
            tail = wfdb.rdrecord(str(record_path), sampfrom=int(tail_start), sampto=int(header.signal_length), channels=[0])
        except (OSError, ValueError, EOFError) as exc:
            skipped_records.append({"record_id": str(record_id), "reason": f"unreadable_declared_tail:{type(exc).__name__}"})
            continue
        tail_values = np.asarray(tail.p_signal)
        if tail_values.shape[0] != int(header.signal_length) - int(tail_start):
            skipped_records.append({"record_id": str(record_id), "reason": "tail_read_length_mismatch"})
            continue

        record_had_series = False
        group_id = _long_term_st_group_id(str(record_id))
        record_slug = _safe_record_name(
            str(record_id),
            record_index=int(record_index),
            used_slugs=used_record_slugs,
        )
        used_channel_slugs: set[str] = set()
        for channel_index, channel_name in enumerate(header.channel_names):
            try:
                record = wfdb.rdrecord(str(record_path), channels=[int(channel_index)])
            except (OSError, ValueError, EOFError) as exc:
                skipped_records.append(
                    {
                        "record_id": str(record_id),
                        "reason": f"unreadable_channel_{int(channel_index)}:{type(exc).__name__}",
                    }
                )
                continue
            values = np.asarray(record.p_signal, dtype=np.float32)
            if values.ndim == 2:
                values = values[:, 0]
            values = values.astype(np.float32, copy=False).reshape(-1)
            if values.shape[0] != int(header.signal_length):
                skipped_records.append(
                    {"record_id": str(record_id), "reason": f"channel_{int(channel_index)}:full_read_length_mismatch"}
                )
                continue
            if not np.all(np.isfinite(values)):
                skipped_records.append(
                    {"record_id": str(record_id), "reason": f"channel_{int(channel_index)}:nonfinite_signal_values"}
                )
                continue
            downsampled = resample_poly(values, 2, 5).astype(np.float32)
            if downsampled.shape[0] < int(history_len) + int(horizon):
                skipped_records.append(
                    {"record_id": str(record_id), "reason": f"channel_{int(channel_index)}:prepared_series_too_short"}
                )
                continue
            safe_channel = _safe_channel_name(
                str(channel_name),
                channel_index=int(channel_index),
                used_slugs=used_channel_slugs,
            )
            file_name = f"series/{record_slug}__ch{int(channel_index)}_{safe_channel}.npy"
            output_path = resolve_portable_relative_path(
                prepared_dir,
                file_name,
                label="Long-Term ST prepared series file",
                reject_links=True,
            )
            np.save(str(output_path), downsampled.astype(np.float32, copy=False))
            total_length = int(downsampled.shape[0])
            min_prepared_length = total_length if min_prepared_length is None else min(min_prepared_length, total_length)
            max_prepared_length = total_length if max_prepared_length is None else max(max_prepared_length, total_length)
            series_rows.append(
                {
                    "series_id": f"{record_id}::ch{int(channel_index)}::{safe_channel}",
                    "record_id": str(record_id),
                    "group_id": str(group_id),
                    "channel_index": int(channel_index),
                    "channel_name": str(channel_name),
                    "file_name": str(file_name).replace("\\", "/"),
                    "split": "",
                    "total_length": int(total_length),
                    "source_total_length": int(header.signal_length),
                }
            )
            record_had_series = True
        if record_had_series:
            used_records.add(str(record_id))

    if not series_rows:
        raise ValueError("No usable Long-Term ST channel series were prepared.")

    split_by_group = _split_long_term_st_groups(
        [row["group_id"] for row in series_rows],
        train_frac=float(train_frac),
        val_frac=float(val_frac),
    )
    for row in series_rows:
        row["split"] = split_by_group[str(row["group_id"])]

    sum_x = 0.0
    sum_x2 = 0.0
    count = 0
    for row in series_rows:
        if row["split"] != "train":
            continue
        arr = np.load(str(prepared_dir / str(row["file_name"])), mmap_mode="r")
        memory_map = getattr(arr, "_mmap", None)
        try:
            arr64 = np.asarray(arr, dtype=np.float64)
            sum_x += float(np.sum(arr64))
            sum_x2 += float(np.sum(arr64 * arr64))
            count += int(arr64.size)
        finally:
            if memory_map is not None:
                memory_map.close()
            del arr
    if count <= 0:
        raise ValueError("Long-Term ST train split is empty after strict validation.")
    mean = float(sum_x / float(count))
    variance = max(0.0, float(sum_x2 / float(count)) - mean * mean)
    std = float(np.sqrt(variance))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0

    split_counts = {
        split: int(sum(1 for row in series_rows if row["split"] == split))
        for split in ("train", "val", "test")
    }
    record_split_counts = {
        split: int(len({row["record_id"] for row in series_rows if row["split"] == split}))
        for split in ("train", "val", "test")
    }
    payload = {
        "dataset_key": LONG_TERM_ST_DATASET_KEY,
        "display_name": "Long-Term ST (100Hz context-only ECG)",
        "source_sampling_rate_hz": float(LONG_TERM_ST_SOURCE_SAMPLING_RATE_HZ),
        "sampling_rate_hz": float(LONG_TERM_ST_SAMPLING_RATE_HZ),
        "frequency": LONG_TERM_ST_FREQUENCY_LABEL,
        "history_len": int(history_len),
        "future_block_len": int(horizon),
        "context_seconds": float(int(history_len) / float(LONG_TERM_ST_SAMPLING_RATE_HZ)),
        "horizon_seconds": float(int(horizon) / float(LONG_TERM_ST_SAMPLING_RATE_HZ)),
        "target_dim": 1,
        "conditioning": "context_only",
        "normalization_mode": "global_train_split_zscore",
        "global_mean": float(mean),
        "global_std": float(std),
        "archive_files": archive_rows,
        "n_headers": int(len(headers)),
        "n_records_used": int(len(used_records)),
        "n_records_skipped": int(len({row["record_id"] for row in skipped_records})),
        "n_series_used": int(len(series_rows)),
        "split_counts": split_counts,
        "record_split_counts": record_split_counts,
        "min_series_length": int(min_prepared_length or 0),
        "max_series_length": int(max_prepared_length or 0),
        "strict_validation": {
            "expected_record_count": None if expected_record_count is None else int(expected_record_count),
            "skip_unreadable_declared_tail": True,
            "ignore_unreferenced_dat_files": True,
            "ignore_atr_annotations": True,
            "omit_header_notes": True,
        },
        "skipped_records": skipped_records,
        "series_specs": series_rows,
    }
    prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


class _LongTermSTParamsView:
    def __init__(self, dataset: "LazyLongTermSTConditionalDataset"):
        self._dataset = dataset

    def __len__(self) -> int:
        return int(self._dataset.total_length)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if int(step) != 1:
                values = self[slice(start, stop, 1)]
                return values[:: int(step)]
            return self._dataset._read_global_slice(int(start), int(stop), normalized=True)
        idx = int(key)
        if idx < 0:
            idx += len(self)
        return self._dataset._read_global_slice(idx, idx + 1, normalized=True)[0]


class LazyLongTermSTConditionalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        dataset_root: str | Path,
        split_name: str,
        history_len: int,
        horizon: int,
        series_specs: Sequence[LongTermSTSeriesSpec],
        mean: float,
        std: float,
        stride: int,
        sampler_num_samples: Optional[int] = None,
        dataset_metadata: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.dataset_key = LONG_TERM_ST_DATASET_KEY
        self.dataset_kind = LONG_TERM_ST_DATASET_KEY
        self.split_name = str(split_name)
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.history_len = int(history_len)
        self.H = int(history_len)
        self.horizon = int(horizon)
        self.future_horizon = max(0, int(horizon) - 1)
        self.series_specs = list(series_specs)
        self.params_mean = np.asarray([float(mean)], dtype=np.float32)
        self.params_std = np.asarray([float(std)], dtype=np.float32)
        self.cond = None
        self.cond_mean = None
        self.cond_std = None
        self.time_feature_source = "none"
        self.time_gap_scale = None
        self.dataset_metadata = dict(dataset_metadata or {})
        self.stride = int(max(1, stride))
        self._arrays: Dict[int, np.ndarray] = {}
        self._segment_starts = np.cumsum(
            np.asarray([0] + [int(spec.total_length) for spec in self.series_specs[:-1]], dtype=np.int64),
            dtype=np.int64,
        )
        self.segment_ends = np.cumsum(
            np.asarray([int(spec.total_length) for spec in self.series_specs], dtype=np.int64),
            dtype=np.int64,
        )
        self.total_length = int(self.segment_ends[-1]) if len(self.segment_ends) else 0
        self.start_indices = self._build_start_indices()
        self.params = _LongTermSTParamsView(self)
        self.sampler_replacement = bool(self.split_name == "train")
        if self.split_name == "train" and len(self.start_indices) > 0:
            bounded_sample_count = min(int(len(self.start_indices)), 16_384)
            self.sampler_num_samples = int(sampler_num_samples or bounded_sample_count)
        else:
            self.sampler_num_samples = None

    def _build_start_indices(self) -> np.ndarray:
        starts: List[int] = []
        for series_idx, spec in enumerate(self.series_specs):
            first = int(self.history_len)
            last_exclusive = int(spec.total_length) - int(self.horizon) + 1
            if last_exclusive <= first:
                continue
            base = int(self._segment_starts[int(series_idx)])
            starts.extend((base + int(t)) for t in range(first, last_exclusive, int(self.stride)))
        return np.asarray(starts, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.start_indices))

    def _array_for_series(self, series_idx: int) -> np.ndarray:
        idx = int(series_idx)
        if idx not in self._arrays:
            path = self.dataset_root / str(self.series_specs[idx].file_name)
            self._arrays[idx] = np.load(str(path), mmap_mode="r")
        return self._arrays[idx]

    def close(self) -> None:
        for array in self._arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()

    def __enter__(self) -> "LazyLongTermSTConditionalDataset":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _series_index_for_global_t(self, t: Union[int, np.ndarray]) -> np.ndarray:
        arr = np.asarray(t, dtype=np.int64)
        return np.searchsorted(self.segment_ends, arr, side="right").astype(np.int64)

    def segment_end_for_t(self, t: Union[int, np.ndarray]) -> np.ndarray:
        idx = self._series_index_for_global_t(t)
        return self.segment_ends[idx]

    def _resolve_global_slice(self, start: int, stop: int) -> Tuple[int, int, int]:
        if int(start) < 0 or int(stop) < int(start) or int(stop) > int(self.total_length):
            raise IndexError(f"Invalid Long-Term ST slice [{int(start)}, {int(stop)}).")
        series_idx = int(self._series_index_for_global_t(int(start)))
        segment_start = int(self._segment_starts[series_idx])
        segment_end = int(self.segment_ends[series_idx])
        if int(stop) > segment_end:
            raise IndexError("Long-Term ST slices may not cross series boundaries.")
        return series_idx, int(start) - segment_start, int(stop) - segment_start

    def _read_global_slice(self, start: int, stop: int, *, normalized: bool) -> np.ndarray:
        if int(stop) == int(start):
            return np.zeros((0, 1), dtype=np.float32)
        series_idx, local_start, local_stop = self._resolve_global_slice(int(start), int(stop))
        raw = np.asarray(self._array_for_series(series_idx)[int(local_start) : int(local_stop)], dtype=np.float32)
        values = raw.reshape(-1, 1)
        if normalized:
            values = ((values - self.params_mean[None, :]) / self.params_std[None, :]).astype(np.float32)
        return values.astype(np.float32, copy=False)

    def example_metadata(self, idx: int) -> Dict[str, Any]:
        target_t = int(self.start_indices[int(idx)])
        series_idx, local_t, _ = self._resolve_global_slice(target_t, target_t + 1)
        spec = self.series_specs[int(series_idx)]
        return {
            "dataset_key": LONG_TERM_ST_DATASET_KEY,
            "dataset_kind": LONG_TERM_ST_DATASET_KEY,
            "split": self.split_name,
            "series_id": str(spec.series_id),
            "series_idx": int(series_idx),
            "record_id": str(spec.record_id),
            "group_id": str(spec.group_id),
            "channel_index": int(spec.channel_index),
            "channel_name": str(spec.channel_name),
            "target_t": int(target_t),
            "local_target_t": int(local_t),
            "history_start": int(target_t - self.history_len),
            "history_stop": int(target_t),
            "target_stop": int(target_t + self.horizon),
        }

    def future_time_features(self, t0: int, horizon: int) -> Optional[torch.Tensor]:
        del t0, horizon
        return None

    def denormalize_block(self, block: np.ndarray, idx: int = 0) -> np.ndarray:
        del idx
        arr = np.asarray(block, dtype=np.float32)
        return (arr * self.params_std[None, :] + self.params_mean[None, :]).astype(np.float32)

    def __getitem__(self, idx: int):
        target_t = int(self.start_indices[int(idx)])
        window = self._read_global_slice(
            int(target_t) - int(self.history_len),
            int(target_t) + int(self.horizon),
            normalized=True,
        )
        expected = int(self.history_len) + int(self.horizon)
        if window.shape[0] != expected:
            raise ValueError(f"Unexpected Long-Term ST window length: got {window.shape[0]}, expected {expected}.")
        hist = window[: int(self.history_len)]
        block = window[int(self.history_len) :]
        tgt = block[0]
        fut = block[1:] if self.future_horizon > 0 else None
        meta = self.example_metadata(int(idx))
        if fut is None:
            return torch.from_numpy(hist), torch.from_numpy(tgt), meta
        return torch.from_numpy(hist), torch.from_numpy(tgt), torch.from_numpy(fut), meta


def build_dataset_splits_from_long_term_st(
    path: str,
    cfg: OTFlowConfig,
    *,
    stride_train: int = LONG_TERM_ST_STRIDE,
    stride_eval: int = LONG_TERM_ST_STRIDE,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
) -> Dict[str, object]:
    del test_frac
    if int(cfg.history_len) != int(LONG_TERM_ST_HISTORY_LEN):
        raise ValueError(
            f"Long-Term ST uses the locked 120-second context at 100Hz: history_len must be "
            f"{int(LONG_TERM_ST_HISTORY_LEN)}, got {int(cfg.history_len)}."
        )
    if int(cfg.prediction_horizon) != int(LONG_TERM_ST_HORIZON_LEN):
        raise ValueError(
            f"Long-Term ST uses the locked 30-second continuation at 100Hz: prediction_horizon must be "
            f"{int(LONG_TERM_ST_HORIZON_LEN)}, got {int(cfg.prediction_horizon)}."
        )
    if int(cfg.snapshot_dim) != 1:
        raise ValueError(
            f"Long-Term ST is a univariate ECG task; levels*token_dim must equal 1, got {int(cfg.snapshot_dim)}."
        )
    cfg.apply_overrides(use_cond_features=False, cond_standardize=False, cond_dim=0)
    prepared_dir = Path(path or long_term_st_data_path()).expanduser().resolve()
    manifest_path = long_term_st_manifest_path(prepared_dir)
    if not manifest_path.exists():
        prepare_long_term_st_dataset(
            prepared_dir,
            history_len=int(LONG_TERM_ST_HISTORY_LEN),
            horizon=int(LONG_TERM_ST_HORIZON_LEN),
            train_frac=float(train_frac),
            val_frac=float(val_frac),
        )
    manifest = _validate_long_term_st_manifest(
        manifest_path,
        history_len=int(LONG_TERM_ST_HISTORY_LEN),
        horizon=int(LONG_TERM_ST_HORIZON_LEN),
    )
    series_specs = [LongTermSTSeriesSpec(**row) for row in manifest["series_specs"]]
    if not series_specs:
        raise ValueError("No usable Long-Term ST series are listed in the prepared manifest.")
    metadata = {
        "sampling_rate_hz": float(manifest["sampling_rate_hz"]),
        "channel_names": ["ECG"],
        "source_sampling_rate_hz": float(manifest["source_sampling_rate_hz"]),
        "conditioning": "context_only",
    }
    splits: Dict[str, object] = {}
    for split_name, stride in (("train", stride_train), ("val", stride_eval), ("test", stride_eval)):
        split_specs = [spec for spec in series_specs if spec.split == split_name]
        splits[split_name] = LazyLongTermSTConditionalDataset(
            dataset_root=prepared_dir,
            split_name=split_name,
            history_len=int(LONG_TERM_ST_HISTORY_LEN),
            horizon=int(LONG_TERM_ST_HORIZON_LEN),
            series_specs=split_specs,
            mean=float(manifest["global_mean"]),
            std=float(manifest["global_std"]),
            stride=int(stride),
            dataset_metadata=metadata,
        )
    splits["stats"] = {
        "dataset_key": LONG_TERM_ST_DATASET_KEY,
        "dataset_kind": LONG_TERM_ST_DATASET_KEY,
        "frequency": LONG_TERM_ST_FREQUENCY_LABEL,
        "official_horizon": int(LONG_TERM_ST_HORIZON_LEN),
        "experiment_horizon": int(LONG_TERM_ST_HORIZON_LEN),
        "history_len": int(LONG_TERM_ST_HISTORY_LEN),
        "cond_dim": 0,
        "target_dim": 1,
        "sampling_rate_hz": float(LONG_TERM_ST_SAMPLING_RATE_HZ),
        "normalization_mode": "global_train_split_zscore",
        "n_train_examples": int(len(splits["train"])),
        "n_val_examples": int(len(splits["val"])),
        "n_test_examples": int(len(splits["test"])),
        "n_series_used": int(manifest["n_series_used"]),
        "n_records_used": int(manifest["n_records_used"]),
        "n_records_skipped": int(manifest["n_records_skipped"]),
        "dataset_metadata": metadata,
    }
    return splits


__all__ = [
    "LazyLongTermSTConditionalDataset",
    "build_dataset_splits_from_long_term_st",
    "long_term_st_raw_archive_dir",
    "long_term_st_source_dir",
    "prepare_long_term_st_dataset",
    "medical_staging_root",
]
