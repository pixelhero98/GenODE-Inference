from __future__ import annotations

import json
import os
import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from genode.models.config import OTFlowConfig
from genode.data.otflow_datasets import build_dataset_splits_from_arrays
from genode.data.otflow_medical_constants import (
    DEFAULT_LONG_TERM_ECG_MANIFEST_NAME,
    DEFAULT_SLEEP_EDF_METADATA_NAME,
    DEFAULT_SLEEP_EDF_NPZ_NAME,
    LONG_TERM_HEADERED_ECG_DATASET_KEY,
    LONG_TERM_ST_DATASET_KEY,
    SLEEP_EDF_DATASET_KEY,
    default_long_term_headered_ecg_dataset_dir,
    default_long_term_headered_ecg_manifest_path,
    default_long_term_st_data_path,
    default_long_term_st_manifest_path,
    default_sleep_edf_data_path,
    default_sleep_edf_metadata_path,
    sleep_edf_metadata_path_for_npz,
)

DEFAULT_MEDICAL_STAGING_ROOT: Path | None = None

LONG_TERM_ECG_FREQUENCY_LABEL = "250_hz"
LONG_TERM_ECG_SAMPLING_RATE_HZ = 250.0
LONG_TERM_ECG_MASE_SEASONAL_PERIOD = 250
LONG_TERM_ST_SOURCE_SAMPLING_RATE_HZ = 250.0
LONG_TERM_ST_SAMPLING_RATE_HZ = 100.0
LONG_TERM_ST_FREQUENCY_LABEL = "100_hz"
LONG_TERM_ST_CONTEXT_SECONDS = 120
LONG_TERM_ST_HORIZON_SECONDS = 30
LONG_TERM_ST_HISTORY_LEN = int(LONG_TERM_ST_CONTEXT_SECONDS * LONG_TERM_ST_SAMPLING_RATE_HZ)
LONG_TERM_ST_HORIZON_LEN = int(LONG_TERM_ST_HORIZON_SECONDS * LONG_TERM_ST_SAMPLING_RATE_HZ)
LONG_TERM_ST_DEFAULT_STRIDE = LONG_TERM_ST_HORIZON_LEN
LONG_TERM_ST_EXPECTED_RECORDS = 86
LONG_TERM_ST_PATIENT_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("s20271", "s20272", "s20273", "s20274"),
    ("s30731", "s30732"),
    ("s30741", "s30742"),
    ("s30751", "s30752"),
)
SLEEP_EDF_SAMPLING_RATE_HZ = 100.0
SLEEP_EDF_EPOCH_SECONDS = 30
SLEEP_EDF_EPOCH_SAMPLES = int(SLEEP_EDF_SAMPLING_RATE_HZ * SLEEP_EDF_EPOCH_SECONDS)
SLEEP_EDF_HISTORY_EPOCHS = 4
SLEEP_EDF_HISTORY_LEN = SLEEP_EDF_HISTORY_EPOCHS * SLEEP_EDF_EPOCH_SAMPLES
SLEEP_EDF_HORIZON_LEN = SLEEP_EDF_EPOCH_SAMPLES
SLEEP_EDF_COMMON_CHANNELS: Tuple[str, ...] = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal")
SLEEP_EDF_STAGE_NAMES: Tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")
SLEEP_EDF_STAGE_MAP: Mapping[str, Optional[str]] = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Movement time": None,
    "Sleep stage ?": None,
}


def medical_staging_root() -> Path:
    raw = str(os.environ.get("OTFLOW_MEDICAL_STAGING_ROOT", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    raise RuntimeError("Set OTFLOW_MEDICAL_STAGING_ROOT to prepare raw medical datasets.")


def long_term_headered_ecg_source_dir() -> Path:
    return medical_staging_root() / "extracted" / "long_term_st"


def sleep_edf_source_dir() -> Path:
    return medical_staging_root() / "extracted" / "sleep_edf"

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


def _safe_channel_name(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("/", "_")


@dataclass(frozen=True)
class ECGSeriesSpec:
    series_id: str
    record_id: str
    channel_index: int
    channel_name: str
    sampling_rate_hz: float
    total_length: int
    train_prefix_end: int
    val_start: int
    test_start: int
    mean: float
    std: float


class LazyECGForecastWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        dataset_key: str,
        split_name: str,
        history_len: int,
        horizon: int,
        series_specs: Sequence[ECGSeriesSpec],
        time_feature_mode: str = "gap_elapsed",
        frequency_label: str = LONG_TERM_ECG_FREQUENCY_LABEL,
        mase_seasonal_period: int = LONG_TERM_ECG_MASE_SEASONAL_PERIOD,
        train_stride: int = 1,
    ):
        super().__init__()
        self.dataset_key = str(dataset_key)
        self.split_name = str(split_name)
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.future_horizon = max(0, int(horizon) - 1)
        self.series_specs = list(series_specs)
        self.time_feature_mode = str(time_feature_mode)
        self.frequency_label = str(frequency_label)
        self.mase_seasonal_period = int(max(1, mase_seasonal_period))
        self.params_mean = None
        self.params_std = None
        self.cond_mean = None
        self.cond_std = None
        self.cond = None
        self.time_feature_source = "synthetic_regular_frequency" if self.time_feature_mode != "none" else "none"
        self.time_gap_scale = 1.0 if self.time_feature_mode != "none" else None
        self.normalization_mode = "per_series_train_prefix_zscore"
        self.train_stride = int(max(1, train_stride))
        self._mase_cache: Dict[int, float] = {}
        self._train_counts = self._build_train_counts() if self.split_name == "train" else np.zeros(0, dtype=np.int64)
        self._train_cumulative = (
            np.cumsum(self._train_counts, dtype=np.int64)
            if self._train_counts.size > 0
            else np.zeros(0, dtype=np.int64)
        )
        self.sampler_replacement = bool(self.split_name == "train")
        self.sampler_num_samples = (
            int(
                min(
                    int(self.__len__()),
                    max(8192, 1024 * max(1, len(self.series_specs))),
                )
            )
            if self.split_name == "train" and len(self.series_specs) > 0 and self.__len__() > 0
            else None
        )

    def _build_train_counts(self) -> np.ndarray:
        counts: List[int] = []
        for spec in self.series_specs:
            max_target_t = int(spec.train_prefix_end) - int(self.horizon)
            if max_target_t < int(self.history_len):
                counts.append(0)
                continue
            count = 1 + (int(max_target_t) - int(self.history_len)) // int(self.train_stride)
            counts.append(max(0, int(count)))
        return np.asarray(counts, dtype=np.int64)

    def __len__(self) -> int:
        if self.split_name == "train":
            return int(self._train_cumulative[-1]) if self._train_cumulative.size > 0 else 0
        return int(len(self.series_specs))

    def _resolve_ref(self, idx: int) -> Tuple[int, int]:
        if self.split_name != "train":
            target_t = (
                int(self.series_specs[int(idx)].val_start)
                if self.split_name == "val"
                else int(self.series_specs[int(idx)].test_start)
            )
            return int(idx), int(target_t)

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")
        series_idx = int(np.searchsorted(self._train_cumulative, int(idx), side="right"))
        prev = int(self._train_cumulative[series_idx - 1]) if series_idx > 0 else 0
        offset = int(idx) - int(prev)
        target_t = int(self.history_len) + int(offset) * int(self.train_stride)
        return int(series_idx), int(target_t)

    def _read_channel_slice(self, series_idx: int, start: int, stop: int) -> np.ndarray:
        spec = self.series_specs[int(series_idx)]
        try:
            import wfdb
        except ImportError as exc:
            raise ImportError("wfdb is required for long_term_headered_ECG_records support.") from exc

        record = wfdb.rdrecord(
            str(long_term_headered_ecg_source_dir() / str(spec.record_id)),
            sampfrom=int(start),
            sampto=int(stop),
            channels=[int(spec.channel_index)],
        )
        values = np.asarray(record.p_signal, dtype=np.float32)
        if values.ndim == 2:
            return values[:, 0].astype(np.float32, copy=False)
        return values.astype(np.float32, copy=False).reshape(-1)

    def target_block_raw(self, idx: int) -> np.ndarray:
        series_idx, target_t = self._resolve_ref(int(idx))
        spec = self.series_specs[int(series_idx)]
        raw = self._read_channel_slice(series_idx, target_t, target_t + int(self.horizon))
        if raw.shape[0] != int(self.horizon):
            raise ValueError(
                f"Unexpected raw block length for {spec.series_id}: got {raw.shape[0]}, expected {self.horizon}."
            )
        return raw.astype(np.float32, copy=False)

    def history_block_raw(self, idx: int) -> np.ndarray:
        series_idx, target_t = self._resolve_ref(int(idx))
        raw = self._read_channel_slice(series_idx, int(target_t) - int(self.history_len), int(target_t))
        if raw.shape[0] != int(self.history_len):
            spec = self.series_specs[int(series_idx)]
            raise ValueError(
                f"Unexpected history length for {spec.series_id}: got {raw.shape[0]}, expected {self.history_len}."
            )
        return raw.astype(np.float32, copy=False)

    def train_series_raw(self, series_idx: int) -> np.ndarray:
        spec = self.series_specs[int(series_idx)]
        raw = self._read_channel_slice(int(series_idx), 0, int(spec.train_prefix_end))
        return raw.astype(np.float32, copy=False)

    def target_block_norm(self, idx: int) -> np.ndarray:
        series_idx, _ = self._resolve_ref(int(idx))
        spec = self.series_specs[int(series_idx)]
        raw = self.target_block_raw(int(idx))
        norm = ((raw - float(spec.mean)) / float(spec.std)).astype(np.float32)[:, None]
        return norm

    def denormalize_block(self, block: np.ndarray, idx: int) -> np.ndarray:
        series_idx, _ = self._resolve_ref(int(idx))
        spec = self.series_specs[int(series_idx)]
        return (np.asarray(block, dtype=np.float32) * float(spec.std) + float(spec.mean)).astype(np.float32)

    def mase_denom(self, idx: int) -> float:
        series_idx, _ = self._resolve_ref(int(idx))
        if int(series_idx) in self._mase_cache:
            return float(self._mase_cache[int(series_idx)])
        spec = self.series_specs[int(series_idx)]
        prefix = self._read_channel_slice(int(series_idx), 0, int(spec.train_prefix_end)).astype(np.float64)
        if prefix.size <= 1:
            scale = 1.0
        else:
            seasonal_period = int(max(1, self.mase_seasonal_period))
            if prefix.size > seasonal_period:
                diffs = np.abs(prefix[seasonal_period:] - prefix[:-seasonal_period])
            else:
                diffs = np.abs(np.diff(prefix))
            scale = float(np.mean(diffs)) if diffs.size > 0 else 1.0
            if not np.isfinite(scale) or scale < 1e-12:
                scale = 1.0
        self._mase_cache[int(series_idx)] = float(scale)
        return float(scale)

    def example_metadata(self, idx: int) -> Dict[str, Any]:
        series_idx, target_t = self._resolve_ref(int(idx))
        spec = self.series_specs[int(series_idx)]
        return {
            "dataset_key": self.dataset_key,
            "split": self.split_name,
            "series_id": str(spec.series_id),
            "series_idx": int(series_idx),
            "record_id": str(spec.record_id),
            "channel_index": int(spec.channel_index),
            "channel_name": str(spec.channel_name),
            "sampling_rate_hz": float(spec.sampling_rate_hz),
            "target_t": int(target_t),
            "history_start": int(target_t - self.history_len),
            "history_stop": int(target_t),
            "target_stop": int(target_t + self.horizon),
            "series_mean": float(spec.mean),
            "series_std": float(spec.std),
            "train_prefix_end": int(spec.train_prefix_end),
            "val_start": int(spec.val_start),
            "test_start": int(spec.test_start),
        }

    def future_time_features(self, idx: int) -> Optional[torch.Tensor]:
        if self.time_feature_mode == "none":
            return None
        _, target_t = self._resolve_ref(int(idx))
        features = _regular_time_features(
            int(target_t),
            int(target_t) + int(self.horizon),
            time_feature_mode=str(self.time_feature_mode),
        )
        if features is None:
            return None
        if features.shape[0] > 0 and features.shape[1] >= 2:
            features[:, 1] = features[:, 1] - float(features[0, 1])
        return torch.from_numpy(features)

    def __getitem__(self, idx: int):
        series_idx, target_t = self._resolve_ref(int(idx))
        spec = self.series_specs[int(series_idx)]
        raw_window = self._read_channel_slice(
            int(series_idx),
            int(target_t) - int(self.history_len),
            int(target_t) + int(self.horizon),
        )
        expected = int(self.history_len) + int(self.horizon)
        if raw_window.shape[0] != expected:
            raise ValueError(
                f"Unexpected window length for {spec.series_id}: got {raw_window.shape[0]}, expected {expected}."
            )
        norm_window = ((raw_window - float(spec.mean)) / float(spec.std)).astype(np.float32)[:, None]
        hist = norm_window[: int(self.history_len)]
        if self.time_feature_mode != "none":
            hist_time = _regular_time_features(
                int(target_t) - int(self.history_len),
                int(target_t),
                time_feature_mode=str(self.time_feature_mode),
            )
            if hist_time is None:
                raise ValueError("time_feature_mode produced no history features.")
            if hist_time.shape[0] > 0 and hist_time.shape[1] >= 2:
                hist_time[:, 1] = hist_time[:, 1] - float(hist_time[0, 1])
            hist = np.concatenate([hist, hist_time], axis=1).astype(np.float32, copy=False)
        block = norm_window[int(self.history_len) :]
        tgt = block[0]
        fut = block[1:] if self.future_horizon > 0 else None
        meta = self.example_metadata(int(idx))

        hist_t = torch.from_numpy(hist)
        tgt_t = torch.from_numpy(tgt)
        if fut is None:
            return hist_t, tgt_t, meta
        return hist_t, tgt_t, torch.from_numpy(fut), meta


def _load_long_term_headered_ecg_manifest(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_long_term_headered_records(source_dir: Path) -> Iterable[Path]:
    for hea_path in sorted(source_dir.glob("*.hea")):
        if hea_path.with_suffix(".dat").exists():
            yield hea_path.with_suffix("")


def _long_term_headered_ecg_record_header(wfdb, record_path: Path) -> Tuple[int, float, List[str]]:
    header = wfdb.rdheader(str(record_path))
    total_length = int(getattr(header, "sig_len", 0))
    fs = float(getattr(header, "fs", LONG_TERM_ECG_SAMPLING_RATE_HZ))
    sig_names = list(getattr(header, "sig_name", []))
    n_sig = int(getattr(header, "n_sig", len(sig_names)))
    if not sig_names:
        sig_names = [f"channel_{idx}" for idx in range(int(max(0, n_sig)))]
    return int(total_length), float(fs), [str(name) for name in sig_names]


def _read_long_term_headered_ecg_prefix(
    wfdb,
    record_path: Path,
    *,
    channel_index: int,
    stop: int,
) -> np.ndarray:
    record = wfdb.rdrecord(
        str(record_path),
        sampfrom=0,
        sampto=int(stop),
        channels=[int(channel_index)],
    )
    values = np.asarray(record.p_signal, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, 0]
    return values.astype(np.float32, copy=False).reshape(-1)


def prepare_long_term_headered_ecg_dataset(
    dataset_root: str | Path,
    *,
    history_len: int,
    horizon: int,
    force: bool = False,
) -> Dict[str, Any]:
    manifest_path = default_long_term_headered_ecg_manifest_path(dataset_root)
    if manifest_path.exists() and not bool(force):
        manifest = _load_long_term_headered_ecg_manifest(manifest_path)
        manifest_history = int(manifest.get("context_length", -1))
        manifest_horizon = int(manifest.get("official_horizon", -1))
        if manifest_history != int(history_len) or manifest_horizon != int(horizon):
            raise ValueError(
                "Existing ECG manifest does not match requested task: "
                f"context_length={manifest_history}, official_horizon={manifest_horizon}, "
                f"requested history_len={int(history_len)}, horizon={int(horizon)}."
            )
        if "source_dir" in manifest or any("source_record_path" in row for row in manifest.get("series_specs", [])):
            raise ValueError("Existing ECG manifest contains local source paths; regenerate it with force=True.")
        return manifest

    source_dir = long_term_headered_ecg_source_dir()
    if not source_dir.exists():
        raise FileNotFoundError(
            "Missing long_term_st source directory. Set OTFLOW_MEDICAL_STAGING_ROOT to the audited staging area."
        )
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("wfdb is required for long_term_headered_ECG_records support.") from exc

    dataset_dir = default_long_term_headered_ecg_dataset_dir(dataset_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    heas = sorted(source_dir.glob("*.hea"))
    missing_dat_records = [
        str(path.with_suffix("").name)
        for path in heas
        if not path.with_suffix(".dat").exists()
    ]
    min_total = int(history_len) + 3 * int(horizon)

    series_specs: List[Dict[str, Any]] = []
    n_records_used = 0
    skipped_short = 0
    skipped_errors: List[Dict[str, str]] = []
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    for record_path in _iter_long_term_headered_records(source_dir):
        record_id = str(record_path.name)
        try:
            total_length, fs, sig_names = _long_term_headered_ecg_record_header(wfdb, record_path)
        except Exception as exc:  # pragma: no cover - defensive around third-party readers.
            skipped_errors.append({"record_id": record_id, "error": str(exc)})
            continue

        if total_length < min_total:
            skipped_short += int(len(sig_names))
            continue

        val_start = int(total_length - int(horizon) - int(horizon))
        test_start = int(total_length - int(horizon))
        train_prefix_end = int(val_start)
        min_length = total_length if min_length is None else min(min_length, total_length)
        max_length = total_length if max_length is None else max(max_length, total_length)
        n_records_used += 1

        for channel_index, channel_name in enumerate(sig_names):
            try:
                raw_prefix = _read_long_term_headered_ecg_prefix(
                    wfdb,
                    record_path,
                    channel_index=int(channel_index),
                    stop=int(train_prefix_end),
                )
            except Exception as exc:  # pragma: no cover - defensive around third-party readers.
                skipped_errors.append(
                    {
                        "record_id": record_id,
                        "error": f"{channel_name}: {exc}",
                    }
                )
                continue
            mean, std = _train_prefix_standardizer(raw_prefix, train_prefix_end=train_prefix_end)
            series_specs.append(
                {
                    "series_id": f"{record_id}::{_safe_channel_name(channel_name)}",
                    "record_id": record_id,
                    "channel_index": int(channel_index),
                    "channel_name": str(channel_name),
                    "sampling_rate_hz": float(fs),
                    "total_length": int(total_length),
                    "train_prefix_end": int(train_prefix_end),
                    "val_start": int(val_start),
                    "test_start": int(test_start),
                    "mean": float(mean),
                    "std": float(std),
                }
            )

    payload = {
        "dataset_key": LONG_TERM_HEADERED_ECG_DATASET_KEY,
        "display_name": "Long-Term Headered ECG Records",
        "official_horizon": int(horizon),
        "context_length": int(history_len),
        "frequency": LONG_TERM_ECG_FREQUENCY_LABEL,
        "target_dim": 1,
        "sampling_rate_hz": float(LONG_TERM_ECG_SAMPLING_RATE_HZ),
        "n_records_total": int(len(heas)),
        "n_records_used": int(n_records_used),
        "n_records_missing_dat": int(len(missing_dat_records)),
        "n_series_total": int(len(series_specs)),
        "n_series_used": int(len(series_specs)),
        "n_series_skipped_short": int(skipped_short),
        "n_series_failed_read": int(len(skipped_errors)),
        "min_series_length": int(min_length) if min_length is not None else 0,
        "max_series_length": int(max_length) if max_length is not None else 0,
        "missing_dat_records": missing_dat_records,
        "skipped_errors": skipped_errors,
        "series_specs": series_specs,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def long_term_headered_ecg_prep_stub(
    dataset_root: str | Path,
    *,
    history_len: int,
    horizon: int,
) -> Dict[str, Any]:
    manifest_path = default_long_term_headered_ecg_manifest_path(dataset_root)
    status = "ready" if manifest_path.exists() else "missing_manifest"
    return {
        "dataset_key": LONG_TERM_HEADERED_ECG_DATASET_KEY,
        "display_name": "Long-Term Headered ECG Records",
        "manifest_name": str(manifest_path.name),
        "status": status,
        "single_tail_holdout": {
            "context_length": int(history_len),
            "official_horizon": int(horizon),
        },
    }


def build_long_term_headered_ecg_forecast_splits(
    *,
    dataset_root: str | Path,
    cfg: OTFlowConfig,
    history_len: int,
    horizon: int,
    stride_train: int = 1,
    time_feature_mode: str = "gap_elapsed",
) -> Dict[str, Any]:
    del cfg
    time_feature_mode = str(time_feature_mode)
    manifest = prepare_long_term_headered_ecg_dataset(
        dataset_root,
        history_len=int(history_len),
        horizon=int(horizon),
    )
    series_specs = [ECGSeriesSpec(**row) for row in manifest["series_specs"]]
    if not series_specs:
        raise ValueError("No usable ECG channel series were prepared for long_term_headered_ECG_records.")

    ds_train = LazyECGForecastWindowDataset(
        dataset_key=LONG_TERM_HEADERED_ECG_DATASET_KEY,
        split_name="train",
        history_len=int(history_len),
        horizon=int(horizon),
        series_specs=series_specs,
        time_feature_mode=str(time_feature_mode),
        train_stride=int(max(1, stride_train)),
    )
    ds_val = LazyECGForecastWindowDataset(
        dataset_key=LONG_TERM_HEADERED_ECG_DATASET_KEY,
        split_name="val",
        history_len=int(history_len),
        horizon=int(horizon),
        series_specs=series_specs,
        time_feature_mode=str(time_feature_mode),
    )
    ds_test = LazyECGForecastWindowDataset(
        dataset_key=LONG_TERM_HEADERED_ECG_DATASET_KEY,
        split_name="test",
        history_len=int(history_len),
        horizon=int(horizon),
        series_specs=series_specs,
        time_feature_mode=str(time_feature_mode),
    )
    return {
        "train": ds_train,
        "val": ds_val,
        "test": ds_test,
        "stats": {
            "dataset_key": LONG_TERM_HEADERED_ECG_DATASET_KEY,
            "frequency": LONG_TERM_ECG_FREQUENCY_LABEL,
            "official_horizon": int(horizon),
            "experiment_horizon": int(horizon),
            "history_len": int(history_len),
            "normalization_mode": "per_series_train_prefix_zscore",
            "mase_seasonal_period": int(LONG_TERM_ECG_MASE_SEASONAL_PERIOD),
            "time_features_enabled": bool(time_feature_mode != "none"),
            "time_feature_mode": str(time_feature_mode),
            "time_feature_dim": _time_feature_dim(str(time_feature_mode)),
            "time_feature_source": "synthetic_regular_frequency" if time_feature_mode != "none" else "none",
            "n_train_examples": int(len(ds_train)),
            "n_val_examples": int(len(ds_val)),
            "n_test_examples": int(len(ds_test)),
            "n_series_total": int(manifest["n_series_total"]),
            "n_series_used": int(manifest["n_series_used"]),
            "n_records_total": int(manifest["n_records_total"]),
            "n_records_used": int(manifest["n_records_used"]),
        },
    }


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
                "sha256": _sha256_file(archive_path),
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


def _copy_zip_member(member: Tuple[Path, str], target: Path) -> None:
    archive_path, member_name = member
    target.parent.mkdir(parents=True, exist_ok=True)
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
    source_dir.mkdir(parents=True, exist_ok=True)
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
        _copy_zip_member(member, source_dir / header_name)

    for dat_name in sorted({name for header in headers.values() for name in header.dat_names}):
        member = dat_members.get(dat_name)
        if member is None:
            missing_dat_names.append(str(dat_name))
            continue
        _copy_zip_member(member, source_dir / dat_name)
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
    group_split: Dict[str, str] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    known_group_by_record = {
        record_id: "_".join(group)
        for group in LONG_TERM_ST_PATIENT_GROUPS
        for record_id in group
    }
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
                f"Long-Term ST manifest group {group_id!r} appears in multiple splits: "
                f"{prior_split!r} and {split!r}."
            )

        resolved_file = _validate_long_term_st_series_file_name(row.get("file_name"), prepared_dir)
        if not resolved_file.exists():
            raise ValueError(f"Long-Term ST series file is missing: {row.get('file_name')!r}.")

    empty_splits = [split for split, count in split_counts.items() if int(count) <= 0]
    if empty_splits:
        raise ValueError(f"Long-Term ST manifest has empty split(s): {', '.join(empty_splits)}.")


def _validate_long_term_st_manifest(path: Path, *, history_len: int, horizon: int) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("dataset_key")) != LONG_TERM_ST_DATASET_KEY:
        raise ValueError(f"Unexpected Long-Term ST manifest dataset_key={payload.get('dataset_key')!r}.")
    if int(payload.get("history_len", -1)) != int(history_len) or int(payload.get("future_block_len", -1)) != int(horizon):
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
    archive_paths: Optional[Union[str, Path, Sequence[str | Path]]] = None,
    force: bool = False,
    expected_record_count: Optional[int] = LONG_TERM_ST_EXPECTED_RECORDS,
    history_len: int = LONG_TERM_ST_HISTORY_LEN,
    horizon: int = LONG_TERM_ST_HORIZON_LEN,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Dict[str, Any]:
    prepared_dir = Path(out_dir or default_long_term_st_data_path()).expanduser().resolve()
    manifest_path = default_long_term_st_manifest_path(prepared_dir)
    if manifest_path.exists() and not bool(force):
        return _validate_long_term_st_manifest(manifest_path, history_len=int(history_len), horizon=int(horizon))

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
        raise ImportError("wfdb and scipy are required to prepare long_term_st.") from exc

    prepared_series_dir = prepared_dir / "series"
    prepared_series_dir.mkdir(parents=True, exist_ok=True)
    missing_dat_set = set(missing_dat_names)
    skipped_records: List[Dict[str, str]] = []
    series_rows: List[Dict[str, Any]] = []
    used_records: set[str] = set()
    min_prepared_length: Optional[int] = None
    max_prepared_length: Optional[int] = None

    for record_id, header in sorted(headers.items()):
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
            tail_values = np.asarray(tail.p_signal)
            if tail_values.shape[0] != int(header.signal_length) - int(tail_start):
                raise ValueError("tail_read_length_mismatch")
        except Exception as exc:
            skipped_records.append({"record_id": str(record_id), "reason": f"unreadable_declared_tail:{type(exc).__name__}"})
            continue

        record_had_series = False
        group_id = _long_term_st_group_id(str(record_id))
        for channel_index, channel_name in enumerate(header.channel_names):
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
                safe_channel = _safe_channel_name(str(channel_name)) or f"channel_{channel_index}"
                file_name = f"series/{record_id}__ch{int(channel_index)}_{safe_channel}.npy"
                np.save(str(prepared_dir / file_name), downsampled.astype(np.float32, copy=False))
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
            default_samples = min(int(len(self.start_indices)), 16_384)
            self.sampler_num_samples = int(sampler_num_samples or default_samples)
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
    stride_train: int = LONG_TERM_ST_DEFAULT_STRIDE,
    stride_eval: int = LONG_TERM_ST_DEFAULT_STRIDE,
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


def _sleep_pairing_key(path: Path) -> str:
    stem = str(path.stem).split("-")[0]
    return stem[:7]


def _canonical_sleep_label(raw_label: str) -> Optional[str]:
    return SLEEP_EDF_STAGE_MAP.get(str(raw_label).strip(), None)


def _read_sleep_annotations(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from pyedflib import EdfReader

    reader = EdfReader(str(path))
    try:
        onset, duration, labels = reader.readAnnotations()
    finally:
        reader.close()
    label_list = [str(label) for label in labels]
    return np.asarray(onset, dtype=np.float64), np.asarray(duration, dtype=np.float64), label_list


def _build_sleep_epoch_labels(total_epochs: int, hyp_path: Path) -> np.ndarray:
    epoch_labels = np.full(int(total_epochs), -1, dtype=np.int64)
    onset, duration, labels = _read_sleep_annotations(hyp_path)
    for start_s, duration_s, raw_label in zip(onset.tolist(), duration.tolist(), labels):
        canonical = _canonical_sleep_label(raw_label)
        start_epoch = int(round(float(start_s) / float(SLEEP_EDF_EPOCH_SECONDS)))
        epoch_count = int(round(float(duration_s) / float(SLEEP_EDF_EPOCH_SECONDS)))
        if epoch_count <= 0:
            continue
        stop_epoch = min(int(total_epochs), int(start_epoch) + int(epoch_count))
        if canonical is None:
            continue
        label_idx = int(SLEEP_EDF_STAGE_NAMES.index(str(canonical)))
        epoch_labels[int(start_epoch) : int(stop_epoch)] = int(label_idx)
    return epoch_labels


def prepare_sleep_edf_dataset(
    out_path: str | Path | None = None,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    npz_path = Path(out_path or default_sleep_edf_data_path()).resolve()
    metadata_path = sleep_edf_metadata_path_for_npz(npz_path)
    if npz_path.exists() and metadata_path.exists() and not bool(force):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prepared_npz = Path(str(metadata.get("prepared_npz_path", ""))).expanduser().resolve()
        if prepared_npz != npz_path:
            raise ValueError(
                "Sleep-EDF metadata does not match requested NPZ path: "
                f"prepared_npz_path={prepared_npz}, requested={npz_path}."
            )
        if int(metadata.get("history_len", -1)) != int(SLEEP_EDF_HISTORY_LEN):
            raise ValueError("Sleep-EDF metadata history_len does not match the locked 12000-sample task.")
        if int(metadata.get("official_horizon", -1)) != int(SLEEP_EDF_HORIZON_LEN):
            raise ValueError("Sleep-EDF metadata official_horizon does not match the locked 3000-sample task.")
        return metadata

    source_dir = sleep_edf_source_dir()
    if not source_dir.exists():
        raise FileNotFoundError(
            "Missing Sleep-EDF source directory. Set OTFLOW_MEDICAL_STAGING_ROOT to the audited staging area."
        )
    try:
        from pyedflib import EdfReader
    except ImportError as exc:
        raise ImportError("pyedflib is required for sleep_edf support.") from exc

    psg_paths = sorted(source_dir.glob("*-PSG.edf"))
    hyp_paths = sorted(source_dir.glob("*-Hypnogram.edf"))
    hyp_by_key = {
        _sleep_pairing_key(path): path
        for path in hyp_paths
        if path.exists() and path.stat().st_size > 0
    }

    params_parts: List[np.ndarray] = []
    cond_parts: List[np.ndarray] = []
    mids_parts: List[np.ndarray] = []
    valid_start_parts: List[np.ndarray] = []
    segment_ends: List[int] = []
    matched_pairs: List[Dict[str, Any]] = []
    stage_counts = {name: 0 for name in SLEEP_EDF_STAGE_NAMES}
    running_total = 0

    for psg_path in psg_paths:
        key = _sleep_pairing_key(psg_path)
        hyp_path = hyp_by_key.get(key)
        if hyp_path is None:
            continue

        reader = EdfReader(str(psg_path))
        try:
            labels = [str(label) for label in reader.getSignalLabels()]
            freqs = np.asarray(reader.getSampleFrequencies(), dtype=np.float64)
            channel_indices = []
            for channel_name in SLEEP_EDF_COMMON_CHANNELS:
                if channel_name not in labels:
                    channel_indices = []
                    break
                idx = labels.index(channel_name)
                if abs(float(freqs[idx]) - float(SLEEP_EDF_SAMPLING_RATE_HZ)) > 1e-6:
                    channel_indices = []
                    break
                channel_indices.append(int(idx))
            if len(channel_indices) != len(SLEEP_EDF_COMMON_CHANNELS):
                continue

            channel_arrays = [
                np.asarray(reader.readSignal(int(idx)), dtype=np.float32)
                for idx in channel_indices
            ]
            min_samples = min(int(arr.shape[0]) for arr in channel_arrays)
        finally:
            reader.close()

        usable_samples = int(min_samples // int(SLEEP_EDF_EPOCH_SAMPLES)) * int(SLEEP_EDF_EPOCH_SAMPLES)
        if usable_samples < int(SLEEP_EDF_HISTORY_LEN + SLEEP_EDF_HORIZON_LEN):
            continue

        signal = np.stack([arr[:usable_samples] for arr in channel_arrays], axis=1).astype(np.float32, copy=False)
        total_epochs = int(usable_samples // int(SLEEP_EDF_EPOCH_SAMPLES))
        epoch_labels = _build_sleep_epoch_labels(total_epochs, hyp_path)
        cond = np.zeros((usable_samples, len(SLEEP_EDF_STAGE_NAMES)), dtype=np.float32)
        for epoch_idx, label_idx in enumerate(epoch_labels.tolist()):
            if int(label_idx) < 0:
                continue
            start = int(epoch_idx) * int(SLEEP_EDF_EPOCH_SAMPLES)
            stop = int(start + int(SLEEP_EDF_EPOCH_SAMPLES))
            cond[start:stop, int(label_idx)] = 1.0
            stage_counts[SLEEP_EDF_STAGE_NAMES[int(label_idx)]] += 1

        valid_start_mask = np.zeros(usable_samples, dtype=bool)
        valid_epochs = epoch_labels >= 0
        for epoch_idx in range(int(SLEEP_EDF_HISTORY_EPOCHS), int(total_epochs)):
            left = int(epoch_idx) - int(SLEEP_EDF_HISTORY_EPOCHS)
            if not bool(np.all(valid_epochs[left : int(epoch_idx) + 1])):
                continue
            start = int(epoch_idx) * int(SLEEP_EDF_EPOCH_SAMPLES)
            valid_start_mask[int(start)] = True

        params_parts.append(signal)
        cond_parts.append(cond)
        mids_parts.append(np.zeros(usable_samples, dtype=np.float32))
        valid_start_parts.append(valid_start_mask)
        running_total += int(usable_samples)
        segment_ends.append(int(running_total))
        matched_pairs.append(
            {
                "recording_key": key,
                "psg_file": str(psg_path.name),
                "hypnogram_file": str(hyp_path.name),
                "total_epochs": int(total_epochs),
                "usable_samples": int(usable_samples),
                "valid_target_epochs": int(np.count_nonzero(valid_start_mask)),
            }
        )

    if not params_parts:
        raise ValueError("No usable matched sleep_edf PSG+hypnogram pairs were prepared.")

    params_raw = np.concatenate(params_parts, axis=0).astype(np.float32, copy=False)
    cond_raw = np.concatenate(cond_parts, axis=0).astype(np.float32, copy=False)
    mids = np.concatenate(mids_parts, axis=0).astype(np.float32, copy=False)
    valid_start_mask = np.concatenate(valid_start_parts, axis=0).astype(bool, copy=False)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(npz_path),
        params_raw=params_raw,
        cond_raw=cond_raw,
        mids=mids,
        segment_ends=np.asarray(segment_ends, dtype=np.int64),
        valid_start_mask=valid_start_mask.astype(np.uint8),
    )

    metadata = {
        "dataset_key": SLEEP_EDF_DATASET_KEY,
        "display_name": "Sleep-EDF (3ch, 100Hz)",
        "sampling_rate_hz": float(SLEEP_EDF_SAMPLING_RATE_HZ),
        "epoch_seconds": int(SLEEP_EDF_EPOCH_SECONDS),
        "epoch_samples": int(SLEEP_EDF_EPOCH_SAMPLES),
        "history_len": int(SLEEP_EDF_HISTORY_LEN),
        "official_horizon": int(SLEEP_EDF_HORIZON_LEN),
        "channels": [str(name) for name in SLEEP_EDF_COMMON_CHANNELS],
        "stage_names": [str(name) for name in SLEEP_EDF_STAGE_NAMES],
        "prepared_npz_path": str(npz_path),
        "n_psg_total": int(len(psg_paths)),
        "n_hypnogram_nonzero": int(len(hyp_by_key)),
        "n_recordings_matched": int(len(matched_pairs)),
        "n_segments": int(len(segment_ends)),
        "n_samples_total": int(params_raw.shape[0]),
        "n_valid_target_starts": int(np.count_nonzero(valid_start_mask)),
        "stage_epoch_counts": {key: int(value) for key, value in stage_counts.items()},
        "matched_pairs": matched_pairs,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_dataset_splits_from_sleep_edf(
    path: str,
    cfg: OTFlowConfig,
    *,
    stride_train: int = SLEEP_EDF_EPOCH_SAMPLES,
    stride_eval: int = SLEEP_EDF_EPOCH_SAMPLES,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    if int(cfg.history_len) != int(SLEEP_EDF_HISTORY_LEN):
        raise ValueError(
            f"Sleep-EDF uses the locked 120-second context: history_len must be "
            f"{int(SLEEP_EDF_HISTORY_LEN)}, got {int(cfg.history_len)}."
        )
    if int(cfg.prediction_horizon) != int(SLEEP_EDF_HORIZON_LEN):
        raise ValueError(
            f"Sleep-EDF uses the locked 30-second continuation: prediction_horizon must be "
            f"{int(SLEEP_EDF_HORIZON_LEN)}, got {int(cfg.prediction_horizon)}."
        )
    cfg.apply_overrides(use_cond_features=True, cond_standardize=False)
    resolved_path = Path(path or default_sleep_edf_data_path()).resolve()
    if not resolved_path.exists():
        prepare_sleep_edf_dataset(resolved_path)
    metadata = prepare_sleep_edf_dataset(resolved_path)
    prepared_npz = Path(str(metadata.get("prepared_npz_path", ""))).expanduser().resolve()
    if prepared_npz != resolved_path:
        raise ValueError(
            "Sleep-EDF metadata does not match requested NPZ path: "
            f"prepared_npz_path={prepared_npz}, requested={resolved_path}."
        )
    data = np.load(str(resolved_path), allow_pickle=True)
    params_raw = np.asarray(data["params_raw"], dtype=np.float32)
    cond_raw = np.asarray(data["cond_raw"], dtype=np.float32)
    mids = np.asarray(data["mids"], dtype=np.float32)
    segment_ends = np.asarray(data["segment_ends"], dtype=np.int64)
    valid_start_mask = np.asarray(data["valid_start_mask"], dtype=np.uint8).astype(bool)
    return build_dataset_splits_from_arrays(
        params_raw=params_raw,
        mids=mids,
        cfg=cfg,
        stride_train=int(stride_train),
        stride_eval=int(stride_eval),
        train_frac=float(train_frac),
        val_frac=float(val_frac),
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
        segment_ends=segment_ends,
        cond_raw_full=cond_raw,
        valid_start_mask=valid_start_mask,
        dataset_kind=SLEEP_EDF_DATASET_KEY,
        dataset_metadata={
            "sampling_rate_hz": float(metadata["sampling_rate_hz"]),
            "channel_names": [str(name) for name in metadata["channels"]],
            "stage_names": [str(name) for name in metadata["stage_names"]],
            "epoch_samples": int(metadata["epoch_samples"]),
        },
    )


__all__ = [
    "DEFAULT_MEDICAL_STAGING_ROOT",
    "LONG_TERM_ECG_FREQUENCY_LABEL",
    "LONG_TERM_ECG_MASE_SEASONAL_PERIOD",
    "LONG_TERM_HEADERED_ECG_DATASET_KEY",
    "LONG_TERM_ST_DATASET_KEY",
    "LONG_TERM_ST_DEFAULT_STRIDE",
    "LONG_TERM_ST_FREQUENCY_LABEL",
    "LONG_TERM_ST_HISTORY_LEN",
    "LONG_TERM_ST_HORIZON_LEN",
    "LONG_TERM_ST_SAMPLING_RATE_HZ",
    "LazyLongTermSTConditionalDataset",
    "LazyECGForecastWindowDataset",
    "SLEEP_EDF_COMMON_CHANNELS",
    "SLEEP_EDF_DATASET_KEY",
    "SLEEP_EDF_EPOCH_SAMPLES",
    "SLEEP_EDF_HISTORY_LEN",
    "SLEEP_EDF_HORIZON_LEN",
    "SLEEP_EDF_STAGE_NAMES",
    "build_dataset_splits_from_long_term_st",
    "build_dataset_splits_from_sleep_edf",
    "build_long_term_headered_ecg_forecast_splits",
    "default_long_term_headered_ecg_dataset_dir",
    "default_long_term_headered_ecg_manifest_path",
    "default_long_term_st_data_path",
    "default_long_term_st_manifest_path",
    "default_sleep_edf_data_path",
    "default_sleep_edf_metadata_path",
    "long_term_st_raw_archive_dir",
    "long_term_st_source_dir",
    "prepare_long_term_st_dataset",
    "long_term_headered_ecg_prep_stub",
    "long_term_headered_ecg_source_dir",
    "medical_staging_root",
    "prepare_long_term_headered_ecg_dataset",
    "prepare_sleep_edf_dataset",
    "sleep_edf_source_dir",
]
