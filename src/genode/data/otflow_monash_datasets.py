from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import hashlib
import shutil
import urllib.request
import zipfile


MONASH_ARCHIVE_URL = "https://forecastingdata.org/"


@dataclass(frozen=True)
class MonashDatasetSpec:
    key: str
    display_name: str
    data_subdir: str
    zenodo_record_id: int
    archive_name: str
    download_url: str
    source_frequency_label: str
    official_horizon: int
    horizon_source: str
    source_url: str = MONASH_ARCHIVE_URL
    benchmark_family: str = "forecast_extrapolation"


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
        key="wind_farms_wo_missing",
        display_name="Wind Farms (Monash, W/O Missing)",
        data_subdir="monash/wind_farms_wo_missing",
        zenodo_record_id=4654858,
        archive_name="wind_farms_minutely_dataset_without_missing_values.zip",
        download_url="https://zenodo.org/records/4654858/files/wind_farms_minutely_dataset_without_missing_values.zip?download=1",
        source_frequency_label="minutely",
        official_horizon=60,
        horizon_source="Monash-University/monash_tsf dataset card prediction length for wind_farms_minutely (1T).",
    ),
    MonashDatasetSpec(
        key="san_francisco_traffic",
        display_name="San Francisco Traffic (Monash)",
        data_subdir="monash/san_francisco_traffic",
        zenodo_record_id=4656132,
        archive_name="traffic_hourly_dataset.zip",
        download_url="https://zenodo.org/records/4656132/files/traffic_hourly_dataset.zip?download=1",
        source_frequency_label="hourly",
        official_horizon=168,
        horizon_source="TSForecasting experiments/fixed_horizon.R traffic_hourly benchmark horizon.",
    ),
    MonashDatasetSpec(
        key="london_smart_meters_wo_missing",
        display_name="London Smart Meters (Monash, W/O Missing)",
        data_subdir="monash/london_smart_meters_wo_missing",
        zenodo_record_id=4656091,
        archive_name="london_smart_meters_dataset_without_missing_values.zip",
        download_url="https://zenodo.org/records/4656091/files/london_smart_meters_dataset_without_missing_values.zip?download=1",
        source_frequency_label="half_hourly",
        official_horizon=60,
        horizon_source="Monash-University/monash_tsf dataset card prediction length for london_smart_meters (30T).",
    ),
    MonashDatasetSpec(
        key="electricity",
        display_name="Electricity (Monash)",
        data_subdir="monash/electricity",
        zenodo_record_id=4656140,
        archive_name="electricity_hourly_dataset.zip",
        download_url="https://zenodo.org/records/4656140/files/electricity_hourly_dataset.zip?download=1",
        source_frequency_label="hourly",
        official_horizon=168,
        horizon_source="TSForecasting experiments/fixed_horizon.R electricity_hourly benchmark horizon.",
    ),
    MonashDatasetSpec(
        key="solar_energy_10m",
        display_name="Solar Energy (Monash, 10m)",
        data_subdir="monash/solar_energy_10m",
        zenodo_record_id=4656144,
        archive_name="solar_10_minutes_dataset.zip",
        download_url="https://zenodo.org/records/4656144/files/solar_10_minutes_dataset.zip?download=1",
        source_frequency_label="10_minutes",
        official_horizon=1008,
        horizon_source="TSForecasting experiments/fixed_horizon.R solar_10_minutes benchmark horizon.",
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


def default_manifest_path(dataset_root: str | Path, dataset_key: str) -> Path:
    spec = get_monash_dataset_spec(dataset_key)
    return Path(dataset_root).resolve() / spec.data_subdir / "manifest.json"


def default_dataset_dir(dataset_root: str | Path, dataset_key: str) -> Path:
    spec = get_monash_dataset_spec(dataset_key)
    return Path(dataset_root).resolve() / spec.data_subdir


def default_raw_dir(dataset_root: str | Path, dataset_key: str) -> Path:
    return default_dataset_dir(dataset_root, dataset_key) / "raw"


def default_source_dir(dataset_root: str | Path, dataset_key: str) -> Path:
    return default_dataset_dir(dataset_root, dataset_key) / "source"


def default_audit_path(dataset_root: str | Path, dataset_key: str) -> Path:
    return default_dataset_dir(dataset_root, dataset_key) / "audit.json"


def default_archive_path(dataset_root: str | Path, dataset_key: str) -> Path:
    spec = get_monash_dataset_spec(dataset_key)
    return default_raw_dir(dataset_root, dataset_key) / spec.archive_name


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as out_fh:
        shutil.copyfileobj(response, out_fh)
    return destination


def _extract_zip(archive_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(destination_dir)
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


def _default_context_length(official_horizon: int, min_series_length: int) -> int:
    horizon = int(official_horizon)
    max_allowed = max(1, int(min_series_length) - 2 * int(horizon))
    return int(max(4, min(max_allowed, max(4 * horizon, 64))))


def download_monash_dataset(dataset_root: str | Path, dataset_key: str) -> Dict[str, Any]:
    spec = get_monash_dataset_spec(dataset_key)
    dataset_dir = default_dataset_dir(dataset_root, dataset_key)
    raw_dir = default_raw_dir(dataset_root, dataset_key)
    source_dir = default_source_dir(dataset_root, dataset_key)
    archive_path = default_archive_path(dataset_root, dataset_key)
    if not archive_path.exists():
        _download_file(spec.download_url, archive_path)
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
        "context_length": int(_default_context_length(int(official_horizon), int(min_series_length))),
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
        "archive_sha256": _sha256_file(archive_path),
        "raw_archive_path": str(archive_path),
        "source_tsf_path": str(tsf_path),
        "header_horizon": int(header.horizon),
        "horizon_source": str(spec.horizon_source),
        "equal_length_header": bool(header.equal_length),
        "missing_header": bool(header.missing),
        "context_length_policy": "4xhorizon_clipped_to_min_length",
    }
    default_manifest_path(dataset_root, dataset_key).write_text(
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


def build_single_tail_holdout_plan(
    *,
    series_length: int,
    official_horizon: int,
    context_length: int,
) -> Dict[str, int]:
    total = int(series_length)
    horizon = int(official_horizon)
    context = int(context_length)
    min_required = context + 2 * horizon
    if total < min_required:
        raise ValueError(
            f"Series length {total} is too short for context={context} and two horizon blocks of size {horizon}."
        )
    test_start = total - horizon
    val_start = test_start - horizon
    return {
        "context_length": context,
        "official_horizon": horizon,
        "validation_start": val_start,
        "validation_stop": test_start,
        "test_start": test_start,
        "test_stop": total,
    }


def dataset_prep_stub(dataset_root: str | Path, dataset_key: str) -> Dict[str, Any]:
    spec = get_monash_dataset_spec(dataset_key)
    manifest_path = default_manifest_path(dataset_root, dataset_key)
    status = "ready" if manifest_path.exists() else "missing_manifest"
    manifest_payload = None
    holdout_plan = None
    if manifest_path.exists():
        manifest = load_monash_manifest(manifest_path)
        manifest_payload = asdict(manifest)
        holdout_plan = build_single_tail_holdout_plan(
            series_length=int(manifest.min_series_length),
            official_horizon=int(manifest.official_horizon),
            context_length=int(manifest.context_length),
        )
    return {
        "dataset_key": spec.key,
        "display_name": spec.display_name,
        "source_url": spec.source_url,
        "data_subdir": spec.data_subdir,
        "manifest_path": str(manifest_path),
        "status": status,
        "manifest": manifest_payload,
        "single_tail_holdout": holdout_plan,
    }


def all_dataset_prep_stubs(dataset_root: str | Path) -> List[Dict[str, Any]]:
    return [dataset_prep_stub(dataset_root, spec.key) for spec in MONASH_PAPER_DATASETS]


__all__ = [
    "MONASH_ARCHIVE_URL",
    "MONASH_PAPER_DATASETS",
    "MonashDatasetManifest",
    "MonashDatasetSpec",
    "TsfHeader",
    "all_dataset_prep_stubs",
    "build_single_tail_holdout_plan",
    "default_archive_path",
    "default_audit_path",
    "default_dataset_dir",
    "dataset_prep_stub",
    "default_manifest_path",
    "default_raw_dir",
    "default_source_dir",
    "download_monash_dataset",
    "download_monash_paper_datasets",
    "find_tsf_file",
    "get_monash_dataset_spec",
    "iter_tsf_series",
    "load_monash_manifest",
    "monash_paper_dataset_keys",
    "parse_tsf_header",
]
