from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from genode.models.config import OTFlowConfig
from genode.data.otflow_datasets import build_dataset_splits_from_arrays
from genode.data.otflow_medical_constants import (
    DEFAULT_LONG_TERM_ECG_MANIFEST_NAME,
    DEFAULT_SLEEP_EDF_METADATA_NAME,
    DEFAULT_SLEEP_EDF_NPZ_NAME,
    LONG_TERM_HEADERED_ECG_DATASET_KEY,
    SLEEP_EDF_DATASET_KEY,
    default_long_term_headered_ecg_dataset_dir,
    default_long_term_headered_ecg_manifest_path,
    default_sleep_edf_data_path,
    default_sleep_edf_metadata_path,
    sleep_edf_metadata_path_for_npz,
)

DEFAULT_MEDICAL_STAGING_ROOT: Path | None = None

LONG_TERM_ECG_FREQUENCY_LABEL = "250_hz"
LONG_TERM_ECG_SAMPLING_RATE_HZ = 250.0
LONG_TERM_ECG_MASE_SEASONAL_PERIOD = 250
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
    "LazyECGForecastWindowDataset",
    "SLEEP_EDF_COMMON_CHANNELS",
    "SLEEP_EDF_DATASET_KEY",
    "SLEEP_EDF_EPOCH_SAMPLES",
    "SLEEP_EDF_HISTORY_LEN",
    "SLEEP_EDF_HORIZON_LEN",
    "SLEEP_EDF_STAGE_NAMES",
    "build_dataset_splits_from_sleep_edf",
    "build_long_term_headered_ecg_forecast_splits",
    "default_long_term_headered_ecg_dataset_dir",
    "default_long_term_headered_ecg_manifest_path",
    "default_sleep_edf_data_path",
    "default_sleep_edf_metadata_path",
    "long_term_headered_ecg_prep_stub",
    "long_term_headered_ecg_source_dir",
    "medical_staging_root",
    "prepare_long_term_headered_ecg_dataset",
    "prepare_sleep_edf_dataset",
    "sleep_edf_source_dir",
]
