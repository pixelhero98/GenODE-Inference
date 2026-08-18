from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import torch

from genode.data.otflow_medical_constants import (
    LONG_TERM_ST_DATASET_KEY,
    LONG_TERM_ST_DEFAULT_STRIDE,
    LONG_TERM_ST_FREQUENCY_LABEL,
    LONG_TERM_ST_HISTORY_LEN,
    LONG_TERM_ST_HORIZON_LEN,
    LONG_TERM_ST_SAMPLING_RATE_HZ,
    LONG_TERM_ST_SOURCE_SAMPLING_RATE_HZ,
    default_long_term_st_data_path,
    default_long_term_st_manifest_path,
)
from genode.models.config import OTFlowConfig
from genode.path_safety import (
    is_link_or_reparse_point,
    portable_relative_path,
    resolve_portable_relative_path,
)

LONG_TERM_ST_EXPECTED_RECORDS = 86
LONG_TERM_ST_PATIENT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("s20271", "s20272", "s20273", "s20274"),
    ("s30731", "s30732"),
    ("s30741", "s30742"),
    ("s30751", "s30752"),
)


def medical_staging_root() -> Path:
    raw = str(os.environ.get("OTFLOW_MEDICAL_STAGING_ROOT", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    raise RuntimeError("Set OTFLOW_MEDICAL_STAGING_ROOT to prepare raw medical datasets.")


def _train_prefix_standardizer(values: np.ndarray, train_prefix_end: int) -> tuple[float, float]:
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


def _regular_time_features(start: int, stop: int, *, time_feature_mode: str) -> np.ndarray | None:
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


def _unique_file_slug(base: str, *, index: int, used_slugs: set[str]) -> str:
    candidate = base
    if candidate.casefold() in used_slugs:
        candidate = f"{base}_{int(index)}"
    suffix = 2
    while candidate.casefold() in used_slugs:
        candidate = f"{base}_{int(index)}_{suffix}"
        suffix += 1
    used_slugs.add(candidate.casefold())
    return candidate


def _safe_channel_name(name: str, *, channel_index: int, used_slugs: set[str]) -> str:
    base = _ascii_file_slug(name, fallback=f"channel_{int(channel_index)}")
    return _unique_file_slug(base, index=int(channel_index), used_slugs=used_slugs)


def _safe_record_name(record_id: str, *, record_index: int, used_slugs: set[str]) -> str:
    base = _ascii_file_slug(record_id, fallback=f"record_{int(record_index)}")
    return _unique_file_slug(base, index=int(record_index), used_slugs=used_slugs)


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
    channel_names: tuple[str, ...]
    dat_names: tuple[str, ...]


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


def _coerce_archive_paths(archive_paths: str | Path | Sequence[str | Path] | None) -> list[Path]:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    dat_names: list[str] = []
    channel_names: list[str] = []
    for channel_index, line in enumerate(lines[1 : 1 + n_sig]):
        parts = line.split()
        if not parts:
            raise ValueError(f"Malformed signal line {channel_index} for {record_id}.")
        dat_path = portable_relative_path(parts[0], label=f"WFDB data file for {record_id}")
        if len(dat_path.parts) != 1 or dat_path.suffix.casefold() != ".dat":
            raise ValueError(f"WFDB data file for {record_id} must be a .dat basename: {parts[0]!r}.")
        dat_names.append(dat_path.name)
        channel_names.append(str(parts[-1]) if len(parts) > 1 else f"channel_{channel_index}")
    return LongTermSTHeader(
        record_id=str(record_id),
        n_sig=int(n_sig),
        sampling_rate_hz=float(sampling_rate_hz),
        signal_length=int(signal_length),
        channel_names=tuple(channel_names),
        dat_names=tuple(dat_names),
    )


def _scan_long_term_st_archives(
    archive_paths: Sequence[Path],
) -> tuple[dict[str, LongTermSTHeader], dict[str, tuple[Path, str]], list[dict[str, Any]]]:
    headers: dict[str, LongTermSTHeader] = {}
    dat_members: dict[str, tuple[Path, str]] = {}
    archive_rows: list[dict[str, Any]] = []
    for archive_path in archive_paths:
        archive_rows.append(
            {
                "name": str(archive_path.name),
                "size_bytes": int(archive_path.stat().st_size),
                "sha256": _sha256_file(archive_path),
            }
        )
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                unix_mode = int(info.external_attr) >> 16
                if stat.S_ISLNK(unix_mode):
                    raise ValueError(f"Long-Term ST archive contains a symbolic link: {info.filename!r}.")
                file_type = stat.S_IFMT(unix_mode)
                if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise ValueError(f"Long-Term ST archive contains a special file: {info.filename!r}.")
                if info.is_dir():
                    continue
                member_path = portable_relative_path(str(info.filename), label="Long-Term ST archive member")
                member_name = member_path.as_posix()
                base_name = member_path.name
                lower = base_name.lower()
                if lower.endswith(".hea"):
                    record_id = Path(base_name).stem
                    text = zf.read(info).decode("utf-8", errors="replace")
                    headers[record_id] = _parse_long_term_st_header(record_id, text)
                elif lower.endswith(".dat"):
                    dat_members[base_name] = (archive_path, member_name)
    return headers, dat_members, archive_rows


def _copy_zip_member(member: tuple[Path, str], *, target_root: Path, target_name: str) -> None:
    archive_path, member_name = member
    relative = portable_relative_path(target_name, label="Long-Term ST extraction file")
    if len(relative.parts) != 1:
        raise ValueError(f"Long-Term ST extraction file must be a basename: {target_name!r}.")
    absolute_target_root = target_root.expanduser().absolute()
    if is_link_or_reparse_point(absolute_target_root):
        raise ValueError(
            "Long-Term ST extraction destination may not be a symlink, junction, or reparse point: "
            f"{absolute_target_root}."
        )
    if absolute_target_root.exists() and not absolute_target_root.is_dir():
        raise ValueError(f"Long-Term ST extraction destination must be a directory: {absolute_target_root}.")
    absolute_target_root.mkdir(parents=True, exist_ok=True)
    if is_link_or_reparse_point(absolute_target_root):
        raise ValueError(
            "Long-Term ST extraction destination may not be a symlink, junction, or reparse point: "
            f"{absolute_target_root}."
        )
    canonical_target_root = absolute_target_root.resolve(strict=True)
    target = resolve_portable_relative_path(
        canonical_target_root,
        relative.as_posix(),
        label="Long-Term ST extraction file",
        reject_links=True,
    )
    if target.exists() and not target.is_file():
        raise ValueError(f"Long-Term ST extraction target must be a regular file: {target}.")
    temporary: Path | None = None
    with zipfile.ZipFile(archive_path) as zf:
        info = zf.getinfo(member_name)
        unix_mode = int(info.external_attr) >> 16
        if stat.S_ISLNK(unix_mode):
            raise ValueError(f"Long-Term ST archive contains a symbolic link: {member_name!r}.")
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=canonical_target_root,
                prefix=f".{target.name}.",
                suffix=".extract",
                delete=False,
            ) as dst:
                temporary = Path(dst.name)
                with zf.open(info) as src:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            if temporary.stat().st_size != int(info.file_size):
                raise zipfile.BadZipFile(f"Long-Term ST archive member size mismatch: {member_name!r}.")
            temporary.replace(target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _extract_long_term_st_wfdb_members(
    *,
    source_dir: Path,
    archive_paths: Sequence[Path],
    headers: Mapping[str, LongTermSTHeader],
    dat_members: Mapping[str, tuple[Path, str]],
) -> list[str]:
    absolute_source_dir = source_dir.expanduser().absolute()
    if is_link_or_reparse_point(absolute_source_dir):
        raise ValueError(
            "Long-Term ST extraction destination may not be a symlink, junction, or reparse point: "
            f"{absolute_source_dir}."
        )
    if absolute_source_dir.exists() and not absolute_source_dir.is_dir():
        raise ValueError(f"Long-Term ST extraction destination must be a directory: {absolute_source_dir}.")
    absolute_source_dir.mkdir(parents=True, exist_ok=True)
    if is_link_or_reparse_point(absolute_source_dir):
        raise ValueError(
            "Long-Term ST extraction destination may not be a symlink, junction, or reparse point: "
            f"{absolute_source_dir}."
        )
    absolute_source_dir = absolute_source_dir.resolve(strict=True)
    missing_dat_names: list[str] = []

    header_members: dict[str, tuple[Path, str]] = {}
    for archive_path in archive_paths:
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                member_path = portable_relative_path(str(info.filename), label="Long-Term ST archive member")
                if member_path.name.lower().endswith(".hea"):
                    header_members[member_path.name] = (archive_path, member_path.as_posix())

    for record_id in sorted(headers):
        header_name = f"{record_id}.hea"
        member = header_members.get(header_name)
        if member is None:
            continue
        _copy_zip_member(member, target_root=absolute_source_dir, target_name=header_name)

    for dat_name in sorted({name for header in headers.values() for name in header.dat_names}):
        member = dat_members.get(dat_name)
        if member is None:
            missing_dat_names.append(str(dat_name))
            continue
        _copy_zip_member(member, target_root=absolute_source_dir, target_name=dat_name)
    return missing_dat_names


def _split_long_term_st_groups(group_ids: Sequence[str], train_frac: float, val_frac: float) -> dict[str, str]:
    groups = sorted({str(group_id) for group_id in group_ids})
    if len(groups) < 3:
        raise ValueError("Long-Term ST requires at least 3 record groups for train/val/test splits.")
    train_count = max(1, int(round(len(groups) * float(train_frac))))
    val_count = max(1, int(round(len(groups) * float(val_frac))))
    if train_count + val_count >= len(groups):
        val_count = max(1, len(groups) - train_count - 1)
    if train_count + val_count >= len(groups):
        train_count = max(1, len(groups) - val_count - 1)
    split_by_group: dict[str, str] = {}
    for idx, group_id in enumerate(groups):
        if idx < train_count:
            split_by_group[group_id] = "train"
        elif idx < train_count + val_count:
            split_by_group[group_id] = "val"
        else:
            split_by_group[group_id] = "test"
    return split_by_group


def _is_relative_to_path(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


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
        "py" + "charmprojects",
    )
    return any(marker in lowered for marker in markers)


def _validate_long_term_st_series_file_name(file_name: Any, prepared_dir: Path) -> Path:
    text = str(file_name or "").strip()
    if not text:
        raise ValueError("Long-Term ST manifest contains an empty series file name.")
    if "\\" in text or "\x00" in text:
        raise ValueError(f"Long-Term ST manifest contains an unsafe series file name: {text!r}.")

    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if posix.is_absolute() or bool(windows.drive) or bool(windows.root):
        raise ValueError(f"Long-Term ST manifest series file must be relative: {text!r}.")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError(f"Long-Term ST manifest series file may not traverse directories: {text!r}.")
    if not posix.parts or posix.parts[0] != "series" or posix.suffix != ".npy":
        raise ValueError(f"Long-Term ST manifest series file must live under series/*.npy: {text!r}.")

    resolved = (prepared_dir / Path(*posix.parts)).resolve()
    if not _is_relative_to_path(resolved, prepared_dir):
        raise ValueError(f"Long-Term ST manifest series file escapes the prepared directory: {text!r}.")
    return resolved


def _validate_long_term_st_manifest_series_specs(payload: Mapping[str, Any], manifest_path: Path) -> None:
    rows = payload.get("series_specs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Long-Term ST manifest must contain non-empty series_specs.")

    prepared_dir = manifest_path.parent.resolve()
    group_split: dict[str, str] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    known_group_by_record = {record_id: "_".join(group) for group in LONG_TERM_ST_PATIENT_GROUPS for record_id in group}
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Long-Term ST series_specs[{idx}] must be an object.")
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
                f"Long-Term ST manifest group {group_id!r} appears in multiple splits: {prior_split!r} and {split!r}."
            )

        resolved_file = _validate_long_term_st_series_file_name(row.get("file_name"), prepared_dir)
        if not resolved_file.exists():
            raise ValueError(f"Long-Term ST series file is missing: {row.get('file_name')!r}.")

    empty_splits = [split for split, count in split_counts.items() if int(count) <= 0]
    if empty_splits:
        raise ValueError(f"Long-Term ST manifest has empty split(s): {', '.join(empty_splits)}.")


def _validate_long_term_st_manifest(path: Path, *, history_len: int, horizon: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("dataset_key")) != LONG_TERM_ST_DATASET_KEY:
        raise ValueError(f"Unexpected Long-Term ST manifest dataset_key={payload.get('dataset_key')!r}.")
    if int(payload.get("history_len", -1)) != int(history_len) or int(payload.get("future_block_len", -1)) != int(
        horizon
    ):
        raise ValueError(
            "Existing Long-Term ST manifest does not match requested task: "
            f"history_len={payload.get('history_len')}, future_block_len={payload.get('future_block_len')}, "
            f"requested history_len={int(history_len)}, horizon={int(horizon)}."
        )
    if any(_looks_like_local_path(value) for value in _iter_manifest_strings(payload)):
        raise ValueError("Existing Long-Term ST manifest contains local filesystem paths; regenerate it.")
    _validate_long_term_st_manifest_series_specs(payload, path)
    return payload


def prepare_long_term_st_dataset(
    out_dir: str | Path | None = None,
    *,
    archive_paths: str | Path | Sequence[str | Path] | None = None,
    force: bool = False,
    expected_record_count: int | None = LONG_TERM_ST_EXPECTED_RECORDS,
    history_len: int = LONG_TERM_ST_HISTORY_LEN,
    horizon: int = LONG_TERM_ST_HORIZON_LEN,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> dict[str, Any]:
    prepared_candidate = Path(out_dir or default_long_term_st_data_path()).expanduser().absolute()
    if is_link_or_reparse_point(prepared_candidate):
        raise ValueError(
            f"Long-Term ST prepared destination may not be a symlink, junction, or reparse point: {prepared_candidate}."
        )
    if prepared_candidate.exists() and not prepared_candidate.is_dir():
        raise ValueError(f"Long-Term ST prepared destination must be a directory: {prepared_candidate}.")
    # Canonicalize the selected destination after rejecting a direct link.
    # Trusted mount aliases above it are valid on HPC systems; all generated
    # members are still constrained beneath this canonical directory.
    prepared_dir = prepared_candidate.resolve(strict=False)
    manifest_path = default_long_term_st_manifest_path(prepared_dir)
    if is_link_or_reparse_point(manifest_path):
        raise ValueError(f"Long-Term ST prepared manifest may not be a symlink or reparse point: {manifest_path}.")
    if manifest_path.exists() and not bool(force):
        return _validate_long_term_st_manifest(manifest_path, history_len=int(history_len), horizon=int(horizon))

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

    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_series_dir = resolve_portable_relative_path(
        prepared_dir,
        "series",
        label="Long-Term ST prepared series directory",
        reject_links=True,
    )
    prepared_series_dir.mkdir(parents=True, exist_ok=True)
    missing_dat_set = set(missing_dat_names)
    skipped_records: list[dict[str, str]] = []
    series_rows: list[dict[str, Any]] = []
    used_records: set[str] = set()
    used_record_slugs: set[str] = set()
    min_prepared_length: int | None = None
    max_prepared_length: int | None = None

    for record_index, (record_id, header) in enumerate(sorted(headers.items())):
        safe_record = _safe_record_name(
            str(record_id),
            record_index=int(record_index),
            used_slugs=used_record_slugs,
        )
        if any(dat_name in missing_dat_set for dat_name in header.dat_names):
            skipped_records.append({"record_id": str(record_id), "reason": "missing_referenced_dat"})
            continue
        if abs(float(header.sampling_rate_hz) - float(LONG_TERM_ST_SOURCE_SAMPLING_RATE_HZ)) > 1e-6:
            skipped_records.append({"record_id": str(record_id), "reason": "unexpected_sampling_rate"})
            continue
        record_path = source_dir / str(record_id)
        try:
            tail_start = max(0, int(header.signal_length) - 1000)
            tail = wfdb.rdrecord(
                str(record_path), sampfrom=int(tail_start), sampto=int(header.signal_length), channels=[0]
            )
            tail_values = np.asarray(tail.p_signal)
            if tail_values.shape[0] != int(header.signal_length) - int(tail_start):
                raise ValueError("tail_read_length_mismatch")
        except Exception as exc:
            skipped_records.append(
                {"record_id": str(record_id), "reason": f"unreadable_declared_tail:{type(exc).__name__}"}
            )
            continue

        record_had_series = False
        group_id = _long_term_st_group_id(str(record_id))
        used_channel_slugs: set[str] = set()
        for channel_index, channel_name in enumerate(header.channel_names):
            safe_channel = _safe_channel_name(
                str(channel_name),
                channel_index=int(channel_index),
                used_slugs=used_channel_slugs,
            )
            file_name = f"series/{safe_record}__ch{int(channel_index)}_{safe_channel}.npy"
            series_target = resolve_portable_relative_path(
                prepared_dir,
                file_name,
                label="Long-Term ST prepared series file",
                reject_links=True,
            )
            try:
                record = wfdb.rdrecord(str(record_path), channels=[int(channel_index)])
                values = np.asarray(record.p_signal, dtype=np.float32)
                if values.ndim == 2:
                    values = values[:, 0]
                values = values.astype(np.float32, copy=False).reshape(-1)
                if values.shape[0] != int(header.signal_length):
                    raise ValueError("full_read_length_mismatch")
                if not np.all(np.isfinite(values)):
                    raise ValueError("nonfinite_signal_values")
                downsampled = resample_poly(values, 2, 5).astype(np.float32)
                if downsampled.shape[0] < int(history_len) + int(horizon):
                    raise ValueError("prepared_series_too_short")
                np.save(str(series_target), downsampled.astype(np.float32, copy=False))
                total_length = int(downsampled.shape[0])
                min_prepared_length = (
                    total_length if min_prepared_length is None else min(min_prepared_length, total_length)
                )
                max_prepared_length = (
                    total_length if max_prepared_length is None else max(max_prepared_length, total_length)
                )
                series_rows.append(
                    {
                        "series_id": f"{safe_record}::ch{int(channel_index)}::{safe_channel}",
                        "record_id": str(record_id),
                        "group_id": str(group_id),
                        "channel_index": int(channel_index),
                        "channel_name": str(channel_name),
                        "file_name": file_name,
                        "split": "",
                        "total_length": int(total_length),
                        "source_total_length": int(header.signal_length),
                    }
                )
                record_had_series = True
            except Exception as exc:
                skipped_records.append(
                    {
                        "record_id": str(record_id),
                        "reason": f"channel_{int(channel_index)}:{type(exc).__name__}",
                    }
                )
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
        arr64 = np.asarray(arr, dtype=np.float64)
        sum_x += float(np.sum(arr64))
        sum_x2 += float(np.sum(arr64 * arr64))
        count += int(arr64.size)
    if count <= 0:
        raise ValueError("Long-Term ST train split is empty after strict validation.")
    mean = float(sum_x / float(count))
    variance = max(0.0, float(sum_x2 / float(count)) - mean * mean)
    std = float(np.sqrt(variance))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0

    split_counts = {
        split: int(sum(1 for row in series_rows if row["split"] == split)) for split in ("train", "val", "test")
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
    def __init__(self, dataset: LazyLongTermSTConditionalDataset):
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
        sampler_num_samples: int | None = None,
        dataset_metadata: Mapping[str, Any] | None = None,
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
        self._arrays: dict[int, np.ndarray] = {}
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
            default_samples = min(int(len(self.start_indices)), 16_384)
            self.sampler_num_samples = int(sampler_num_samples or default_samples)
        else:
            self.sampler_num_samples = None

    def _build_start_indices(self) -> np.ndarray:
        starts: list[int] = []
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

    def __enter__(self) -> LazyLongTermSTConditionalDataset:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _series_index_for_global_t(self, t: int | np.ndarray) -> np.ndarray:
        arr = np.asarray(t, dtype=np.int64)
        return np.searchsorted(self.segment_ends, arr, side="right").astype(np.int64)

    def segment_end_for_t(self, t: int | np.ndarray) -> np.ndarray:
        idx = self._series_index_for_global_t(t)
        return self.segment_ends[idx]

    def _resolve_global_slice(self, start: int, stop: int) -> tuple[int, int, int]:
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

    def example_metadata(self, idx: int) -> dict[str, Any]:
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

    def future_time_features(self, t0: int, horizon: int) -> torch.Tensor | None:
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
    stride_train: int = LONG_TERM_ST_DEFAULT_STRIDE,
    stride_eval: int = LONG_TERM_ST_DEFAULT_STRIDE,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float | None = None,
) -> dict[str, object]:
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
    prepared_dir = Path(path or default_long_term_st_data_path()).expanduser().resolve()
    manifest_path = default_long_term_st_manifest_path(prepared_dir)
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
    splits: dict[str, object] = {}
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
    "LONG_TERM_ST_DATASET_KEY",
    "LONG_TERM_ST_DEFAULT_STRIDE",
    "LONG_TERM_ST_FREQUENCY_LABEL",
    "LONG_TERM_ST_HISTORY_LEN",
    "LONG_TERM_ST_HORIZON_LEN",
    "LONG_TERM_ST_SAMPLING_RATE_HZ",
    "LazyLongTermSTConditionalDataset",
    "build_dataset_splits_from_long_term_st",
    "default_long_term_st_data_path",
    "default_long_term_st_manifest_path",
    "long_term_st_raw_archive_dir",
    "long_term_st_source_dir",
    "prepare_long_term_st_dataset",
    "medical_staging_root",
]
