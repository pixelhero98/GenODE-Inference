from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import hashlib
import stat
import tempfile
import urllib.request
import zipfile

from genode.data.otflow_experiment_plan import FORECAST_FAMILY
from genode.path_safety import is_link_or_reparse_point, resolve_portable_relative_path


MONASH_ARCHIVE_URL = "https://forecastingdata.org/"


@dataclass(frozen=True)
class MonashDatasetSpec:
    key: str
    display_name: str
    data_subdir: str
    zenodo_record_id: int
    archive_name: str
    download_url: str
    archive_size_bytes: int
    archive_md5: str
    source_frequency_label: str
    official_horizon: int
    horizon_source: str
    source_url: str = MONASH_ARCHIVE_URL
    benchmark_family: str = FORECAST_FAMILY


@dataclass(frozen=True)
class MonashDatasetManifest:
    dataset_key: str
    official_horizon: int
    context_length: int
    frequency: str
    n_series: int
    min_series_length: int
    max_series_length: int
    target_dim: int = 1


@dataclass(frozen=True)
class TsfHeader:
    frequency: str
    horizon: int
    missing: bool
    equal_length: bool
    attribute_names: Tuple[str, ...]
    data_start_line: int


MONASH_PAPER_DATASETS: Tuple[MonashDatasetSpec, ...] = (
    MonashDatasetSpec(
        key="solar_energy_10m",
        display_name="Solar Energy (Monash, 10m)",
        data_subdir="monash/solar_energy_10m",
        zenodo_record_id=4656144,
        archive_name="solar_10_minutes_dataset.zip",
        download_url="https://zenodo.org/records/4656144/files/solar_10_minutes_dataset.zip?download=1",
        archive_size_bytes=4_559_353,
        archive_md5="84c0de18383c911091a3cd274661b029",
        source_frequency_label="10_minutes",
        official_horizon=1008,
        horizon_source="TSForecasting experiments/fixed_horizon.R solar_10_minutes benchmark horizon.",
    ),
    MonashDatasetSpec(
        key="traffic_hourly",
        display_name="Traffic Hourly (Monash)",
        data_subdir="monash/traffic_hourly",
        zenodo_record_id=4656132,
        archive_name="traffic_hourly_dataset.zip",
        download_url="https://zenodo.org/records/4656132/files/traffic_hourly_dataset.zip?download=1",
        archive_size_bytes=22_868_806,
        archive_md5="1cf694f99f95700217845078b467fb24",
        source_frequency_label="hourly",
        official_horizon=168,
        horizon_source="TSForecasting experiments/fixed_horizon.R traffic_hourly benchmark horizon.",
    ),
    MonashDatasetSpec(
        key="weather_daily",
        display_name="Weather Daily (Monash)",
        data_subdir="monash/weather_daily",
        zenodo_record_id=4654822,
        archive_name="weather_dataset.zip",
        download_url="https://zenodo.org/records/4654822/files/weather_dataset.zip?download=1",
        archive_size_bytes=38_820_451,
        archive_md5="57155594af0883ccd5e63a5948976796",
        source_frequency_label="daily",
        official_horizon=30,
        horizon_source="TSForecasting experiments/fixed_horizon.R weather benchmark horizon.",
    ),
)


def monash_paper_dataset_keys() -> Tuple[str, ...]:
    return tuple(spec.key for spec in MONASH_PAPER_DATASETS)


def get_monash_dataset_spec(dataset_key: str) -> MonashDatasetSpec:
    key = str(dataset_key).strip().lower()
    for spec in MONASH_PAPER_DATASETS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown Monash paper dataset: {dataset_key}")


def monash_manifest_path(dataset_root: str | Path, dataset_key: str) -> Path:
    spec = get_monash_dataset_spec(dataset_key)
    return Path(dataset_root).resolve() / spec.data_subdir / "manifest.json"


def monash_dataset_dir(dataset_root: str | Path, dataset_key: str) -> Path:
    spec = get_monash_dataset_spec(dataset_key)
    return Path(dataset_root).resolve() / spec.data_subdir


def monash_raw_dir(dataset_root: str | Path, dataset_key: str) -> Path:
    return monash_dataset_dir(dataset_root, dataset_key) / "raw"


def monash_source_dir(dataset_root: str | Path, dataset_key: str) -> Path:
    return monash_dataset_dir(dataset_root, dataset_key) / "source"


def monash_archive_path(dataset_root: str | Path, dataset_key: str) -> Path:
    spec = get_monash_dataset_spec(dataset_key)
    return monash_raw_dir(dataset_root, dataset_key) / spec.archive_name


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: str | Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_archive(path: Path, *, expected_size: int, expected_md5: str) -> bool:
    return (
        path.is_file()
        and int(path.stat().st_size) == int(expected_size)
        and _md5_file(path) == str(expected_md5).lower()
    )


def _download_file(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_md5: str,
) -> Path:
    if _verified_archive(destination, expected_size=expected_size, expected_md5=expected_md5):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".download",
            delete=False,
        ) as out_fh:
            temporary = Path(out_fh.name)
            digest = hashlib.md5(usedforsecurity=False)
            total = 0
            with urllib.request.urlopen(url, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > int(expected_size):
                        raise ValueError(
                            f"Download from {url} exceeded the expected size of {int(expected_size)} bytes."
                        )
                    digest.update(chunk)
                    out_fh.write(chunk)
        if total != int(expected_size):
            raise ValueError(f"Download from {url} has size {total}; expected {int(expected_size)} bytes.")
        observed_md5 = digest.hexdigest()
        if observed_md5 != str(expected_md5).lower():
            raise ValueError(
                f"Download from {url} has MD5 {observed_md5}; expected {str(expected_md5).lower()}."
            )
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _extract_zip(archive_path: Path, destination_dir: Path) -> Path:
    if is_link_or_reparse_point(destination_dir):
        raise ValueError(
            f"Archive destination may not be a symlink, junction, or reparse point: {destination_dir}."
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_root = destination_dir.resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Archive contains an unsupported symbolic link: {member.filename!r}")
            member_path = str(member.filename).rstrip("/")
            if not member_path:
                raise ValueError("Archive contains an empty member path.")
            resolve_portable_relative_path(
                destination_root,
                member_path,
                label="Archive member",
                reject_links=True,
            )
        archive.extractall(destination_root)
    return destination_dir


def find_tsf_file(source_dir: str | Path) -> Path:
    candidates = sorted(Path(source_dir).rglob("*.tsf"))
    if not candidates:
        raise FileNotFoundError(f"No .tsf file found under {source_dir}")
    return candidates[0]


def parse_tsf_header(tsf_path: str | Path) -> TsfHeader:
    frequency = ""
    horizon = 0
    missing = False
    equal_length = False
    attribute_names: List[str] = []
    data_start_line = -1
    with Path(tsf_path).open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lower = line.lower()
            if lower.startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 3:
                    attribute_names.append(str(parts[1]))
                continue
            if lower.startswith("@frequency"):
                frequency = str(line.split(maxsplit=1)[1]).strip()
                continue
            if lower.startswith("@horizon"):
                horizon = int(line.split(maxsplit=1)[1])
                continue
            if lower.startswith("@missing"):
                missing = str(line.split(maxsplit=1)[1]).strip().lower() == "true"
                continue
            if lower.startswith("@equallength"):
                equal_length = str(line.split(maxsplit=1)[1]).strip().lower() == "true"
                continue
            if lower.startswith("@data"):
                data_start_line = int(line_number) + 1
                break
    if data_start_line < 0:
        raise ValueError(f"Malformed TSF file without @data section: {tsf_path}")
    return TsfHeader(
        frequency=str(frequency),
        horizon=int(horizon),
        missing=bool(missing),
        equal_length=bool(equal_length),
        attribute_names=tuple(attribute_names),
        data_start_line=int(data_start_line),
    )


def iter_tsf_series(tsf_path: str | Path) -> Iterator[Tuple[int, Dict[str, str], List[Optional[float]]]]:
    header = parse_tsf_header(tsf_path)
    attribute_count = int(len(header.attribute_names))
    with Path(tsf_path).open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            if int(line_number) < int(header.data_start_line):
                continue
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) < attribute_count + 1:
                raise ValueError(f"Malformed TSF data line {line_number} in {tsf_path}")
            attr_values = parts[:attribute_count]
            series_text = ":".join(parts[attribute_count:])
            series_values: List[Optional[float]] = []
            for token in series_text.split(","):
                item = token.strip()
                if not item or item == "?":
                    series_values.append(None)
                else:
                    series_values.append(float(item))
            metadata = {
                str(name): str(value)
                for name, value in zip(header.attribute_names, attr_values)
            }
            yield int(line_number), metadata, series_values


def _context_length(official_horizon: int, min_series_length: int) -> int:
    horizon = int(official_horizon)
    max_allowed = max(1, int(min_series_length) - 2 * int(horizon))
    return int(max(4, min(max_allowed, max(4 * horizon, 64))))


def download_monash_dataset(dataset_root: str | Path, dataset_key: str) -> Dict[str, Any]:
    spec = get_monash_dataset_spec(dataset_key)
    source_dir = monash_source_dir(dataset_root, dataset_key)
    archive_path = monash_archive_path(dataset_root, dataset_key)
    _download_file(
        spec.download_url,
        archive_path,
        expected_size=int(spec.archive_size_bytes),
        expected_md5=str(spec.archive_md5),
    )
    if not any(source_dir.rglob("*.tsf")):
        _extract_zip(archive_path, source_dir)
    tsf_path = find_tsf_file(source_dir)
    header = parse_tsf_header(tsf_path)
    official_horizon = int(spec.official_horizon) if int(spec.official_horizon) > 0 else int(header.horizon)
    if official_horizon <= 0:
        raise ValueError(f"Missing official horizon for Monash dataset: {dataset_key}")

    n_series = 0
    min_series_length = 0
    max_series_length = 0
    for _, _, series_values in iter_tsf_series(tsf_path):
        length = len(series_values)
        if n_series == 0:
            min_series_length = length
            max_series_length = length
        else:
            min_series_length = min(min_series_length, length)
            max_series_length = max(max_series_length, length)
        n_series += 1
    if n_series <= 0:
        raise ValueError(f"No series found in TSF file: {tsf_path}")

    manifest_payload = {
        "dataset_key": spec.key,
        "display_name": spec.display_name,
        "official_horizon": int(official_horizon),
        "context_length": int(_context_length(int(official_horizon), int(min_series_length))),
        "frequency": str(header.frequency or spec.source_frequency_label),
        "n_series": int(n_series),
        "min_series_length": int(min_series_length),
        "max_series_length": int(max_series_length),
        "target_dim": 1,
        "source_url": str(spec.source_url),
        "record_id": int(spec.zenodo_record_id),
        "record_url": f"https://zenodo.org/record/{spec.zenodo_record_id}",
        "download_url": str(spec.download_url),
        "archive_name": str(spec.archive_name),
        "archive_size_bytes": int(archive_path.stat().st_size),
        "archive_md5": _md5_file(archive_path),
        "archive_sha256": _sha256_file(archive_path),
        "source_tsf_name": str(tsf_path.name),
        "header_horizon": int(header.horizon),
        "horizon_source": str(spec.horizon_source),
        "equal_length_header": bool(header.equal_length),
        "missing_header": bool(header.missing),
        "context_length_policy": "4xhorizon_clipped_to_min_length",
    }
    monash_manifest_path(dataset_root, dataset_key).write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )
    return manifest_payload


def download_monash_paper_datasets(dataset_root: str | Path, dataset_keys: Optional[Tuple[str, ...]] = None) -> List[Dict[str, Any]]:
    keys = monash_paper_dataset_keys() if dataset_keys is None else tuple(str(key) for key in dataset_keys)
    return [download_monash_dataset(dataset_root, key) for key in keys]


def load_monash_manifest(path: str | Path) -> MonashDatasetManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MonashDatasetManifest(
        dataset_key=str(payload["dataset_key"]),
        official_horizon=int(payload["official_horizon"]),
        context_length=int(payload["context_length"]),
        frequency=str(payload["frequency"]),
        n_series=int(payload["n_series"]),
        min_series_length=int(payload["min_series_length"]),
        max_series_length=int(payload["max_series_length"]),
        target_dim=int(payload.get("target_dim", 1)),
    )


__all__ = [
    "MONASH_ARCHIVE_URL",
    "MONASH_PAPER_DATASETS",
    "MonashDatasetManifest",
    "MonashDatasetSpec",
    "TsfHeader",
    "monash_archive_path",
    "monash_dataset_dir",
    "monash_manifest_path",
    "monash_raw_dir",
    "monash_source_dir",
    "download_monash_dataset",
    "download_monash_paper_datasets",
    "find_tsf_file",
    "get_monash_dataset_spec",
    "iter_tsf_series",
    "load_monash_manifest",
    "monash_paper_dataset_keys",
    "parse_tsf_header",
]
