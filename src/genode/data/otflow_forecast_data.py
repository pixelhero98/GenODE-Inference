from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from genode.models.config import OTFlowConfig
from genode.data.otflow_monash_datasets import (
    default_manifest_path,
    default_source_dir,
    find_tsf_file,
    iter_tsf_series,
    load_monash_manifest,
)


_FREQUENCY_SECONDS: Mapping[str, int] = {
    "yearly": 365 * 24 * 60 * 60,
    "quarterly": 90 * 24 * 60 * 60,
    "monthly": 30 * 24 * 60 * 60,
    "weekly": 7 * 24 * 60 * 60,
    "daily": 24 * 60 * 60,
    "hourly": 60 * 60,
    "half_hourly": 30 * 60,
    "10_minutes": 10 * 60,
    "minutely": 60,
}


def _safe_series_id(metadata: Mapping[str, str], line_number: int) -> str:
    for key in ("series_name", "series_id", "id", "series", "item_id"):
        value = str(metadata.get(key, "")).strip()
        if value:
            return value
    return f"series_{int(line_number)}"


def _fill_missing_values(values: np.ndarray) -> np.ndarray:
    if not np.isnan(values).any():
        return values.astype(np.float32, copy=False)
    idx = np.arange(values.shape[0], dtype=np.float64)
    mask = np.isfinite(values)
    if not np.any(mask):
        raise ValueError("Series contains only missing values.")
    filled = np.interp(idx, idx[mask], values[mask]).astype(np.float32)
    return filled


def _frequency_seconds(label: str) -> int:
    key = str(label).strip().lower()
    return int(_FREQUENCY_SECONDS.get(key, 1))


def infer_mase_seasonal_period(label: str) -> int:
    key = str(label).strip().lower()
    explicit = {
        "10_minutes": 144,
        "half_hourly": 48,
        "hourly": 24,
        "daily": 7,
        "weekly": 52,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1,
        "minutely": 1440,
    }
    if key in explicit:
        return int(max(1, explicit[key]))
    step_seconds = _frequency_seconds(key)
    if step_seconds > 0 and 86_400 % step_seconds == 0:
        return int(max(1, 86_400 // step_seconds))
    return 1


def _time_feature_dim(time_feature_mode: str) -> int:
    mode = str(time_feature_mode)
    if mode == "gap_elapsed":
        return 2
    if mode == "gap_only":
        return 1
    if mode == "none":
        return 0
    raise ValueError(f"Unknown time_feature_mode={time_feature_mode!r}")


def _regular_time_features(length: int, step_seconds: int, *, time_feature_mode: str) -> Optional[np.ndarray]:
    dim = _time_feature_dim(str(time_feature_mode))
    if dim == 0:
        return None
    timestamps = np.arange(int(length), dtype=np.int64) * int(max(1, step_seconds))
    gap_feature = np.zeros((int(length), 1), dtype=np.float32)
    if dim == 1:
        return gap_feature
    elapsed_feature = np.arange(int(length), dtype=np.float32)[:, None]
    if len(timestamps) <= 1:
        return np.concatenate([gap_feature, elapsed_feature], axis=1)
    return np.concatenate([gap_feature, elapsed_feature], axis=1).astype(np.float32)


def _train_prefix_standardizer(values: np.ndarray, train_prefix_end: int) -> Tuple[float, float]:
    train_prefix = np.asarray(values[: int(train_prefix_end)], dtype=np.float32)
    if train_prefix.size <= 0:
        raise ValueError("Train prefix must be non-empty for per-series normalization.")
    mean = float(train_prefix.mean())
    std = float(train_prefix.std())
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return mean, std


@dataclass(frozen=True)
class ForecastSeriesRecord:
    dataset_key: str
    series_id: str
    raw_values: np.ndarray
    norm_values: np.ndarray
    time_features: Optional[np.ndarray]
    mean: float
    std: float
    total_length: int
    train_prefix_end: int
    val_start: int
    test_start: int


@dataclass(frozen=True)
class ForecastExampleRef:
    series_idx: int
    target_t: int


class MonashForecastWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        dataset_key: str,
        split_name: str,
        history_len: int,
        horizon: int,
        series_records: Sequence[ForecastSeriesRecord],
        example_refs: Sequence[ForecastExampleRef],
        time_feature_mode: str = "gap_elapsed",
        frequency_label: str = "",
        mase_seasonal_period: int = 1,
    ):
        super().__init__()
        self.dataset_key = str(dataset_key)
        self.split_name = str(split_name)
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.future_horizon = max(0, int(horizon) - 1)
        self.series_records = list(series_records)
        self.example_refs = list(example_refs)
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

    def __len__(self) -> int:
        return len(self.example_refs)

    def _series_record(self, idx: int) -> ForecastSeriesRecord:
        ref = self.example_refs[int(idx)]
        return self.series_records[int(ref.series_idx)]

    def target_block_norm(self, idx: int) -> np.ndarray:
        ref = self.example_refs[int(idx)]
        series = self.series_records[int(ref.series_idx)]
        start = int(ref.target_t)
        stop = int(start + self.horizon)
        return series.norm_values[start:stop].astype(np.float32, copy=True)

    def target_block_raw(self, idx: int) -> np.ndarray:
        ref = self.example_refs[int(idx)]
        series = self.series_records[int(ref.series_idx)]
        start = int(ref.target_t)
        stop = int(start + self.horizon)
        return series.raw_values[start:stop].astype(np.float32, copy=True)

    def denormalize_block(self, block: np.ndarray, idx: int) -> np.ndarray:
        series = self._series_record(int(idx))
        return (np.asarray(block, dtype=np.float32) * float(series.std) + float(series.mean)).astype(np.float32)

    def mase_denom(self, idx: int) -> float:
        series = self._series_record(int(idx))
        train_prefix = np.asarray(series.raw_values[: int(series.train_prefix_end)], dtype=np.float64).reshape(-1)
        if train_prefix.size <= 1:
            return 1.0
        seasonal_period = int(max(1, self.mase_seasonal_period))
        if train_prefix.size > seasonal_period:
            diffs = np.abs(train_prefix[seasonal_period:] - train_prefix[:-seasonal_period])
        else:
            diffs = np.abs(np.diff(train_prefix))
        if diffs.size <= 0:
            return 1.0
        scale = float(np.mean(diffs))
        if not np.isfinite(scale) or scale < 1e-12:
            return 1.0
        return scale

    def example_metadata(self, idx: int) -> Dict[str, Any]:
        ref = self.example_refs[int(idx)]
        series = self.series_records[int(ref.series_idx)]
        return {
            "dataset_key": self.dataset_key,
            "split": self.split_name,
            "series_id": str(series.series_id),
            "series_idx": int(ref.series_idx),
            "target_t": int(ref.target_t),
            "history_start": int(ref.target_t - self.history_len),
            "history_stop": int(ref.target_t),
            "target_stop": int(ref.target_t + self.horizon),
            "series_mean": float(series.mean),
            "series_std": float(series.std),
            "train_prefix_end": int(series.train_prefix_end),
            "val_start": int(series.val_start),
            "test_start": int(series.test_start),
        }

    def future_time_features(self, idx: int) -> Optional[torch.Tensor]:
        ref = self.example_refs[int(idx)]
        series = self.series_records[int(ref.series_idx)]
        if series.time_features is None:
            return None
        start = int(ref.target_t)
        stop = int(start + self.horizon)
        features = series.time_features[start:stop].astype(np.float32, copy=True)
        if features.shape[0] > 0 and features.shape[1] >= 2:
            features[:, 1] = features[:, 1] - float(features[0, 1])
        return torch.from_numpy(features)

    def __getitem__(self, idx: int):
        ref = self.example_refs[int(idx)]
        series = self.series_records[int(ref.series_idx)]
        target_t = int(ref.target_t)
        hist = series.norm_values[target_t - self.history_len : target_t].astype(np.float32, copy=True)
        if self.time_feature_mode != "none" and series.time_features is not None:
            hist_time = series.time_features[target_t - self.history_len : target_t].astype(np.float32, copy=True)
            if hist_time.shape[0] > 0 and hist_time.shape[1] >= 2:
                hist_time[:, 1] = hist_time[:, 1] - float(hist_time[0, 1])
            hist = np.concatenate([hist, hist_time], axis=1).astype(np.float32, copy=False)
        block = self.target_block_norm(int(idx))
        tgt = block[0]
        fut = block[1:] if self.future_horizon > 0 else None
        meta = self.example_metadata(int(idx))

        hist_t = torch.from_numpy(hist)
        tgt_t = torch.from_numpy(tgt)
        if fut is None:
            return hist_t, tgt_t, meta
        return hist_t, tgt_t, torch.from_numpy(fut), meta


def _build_series_records(
    *,
    dataset_key: str,
    tsf_path: Path,
    history_len: int,
    horizon: int,
    time_feature_mode: str,
    frequency_label: str,
) -> Tuple[List[ForecastSeriesRecord], Dict[str, int]]:
    records: List[ForecastSeriesRecord] = []
    skipped_short = 0
    filled_missing = 0
    step_seconds = _frequency_seconds(frequency_label)
    min_total = int(history_len) + 3 * int(horizon)

    for line_number, metadata, series_values in iter_tsf_series(tsf_path):
        raw = np.asarray(
            [np.nan if value is None else float(value) for value in series_values],
            dtype=np.float32,
        )
        if np.isnan(raw).any():
            filled_missing += 1
            raw = _fill_missing_values(raw)
        total_length = int(raw.shape[0])
        if total_length < min_total:
            skipped_short += 1
            continue
        test_start = total_length - int(horizon)
        val_start = test_start - int(horizon)
        train_prefix_end = int(val_start)
        mean, std = _train_prefix_standardizer(raw, train_prefix_end=train_prefix_end)
        norm = ((raw - float(mean)) / float(std)).astype(np.float32)[:, None]
        time_features = _regular_time_features(
            total_length,
            step_seconds,
            time_feature_mode=str(time_feature_mode),
        )
        records.append(
            ForecastSeriesRecord(
                dataset_key=str(dataset_key),
                series_id=_safe_series_id(metadata, line_number),
                raw_values=raw.astype(np.float32),
                norm_values=norm.astype(np.float32),
                time_features=time_features,
                mean=float(mean),
                std=float(std),
                total_length=int(total_length),
                train_prefix_end=int(train_prefix_end),
                val_start=int(val_start),
                test_start=int(test_start),
            )
        )

    stats = {
        "n_series_total": int(sum(1 for _ in iter_tsf_series(tsf_path))),
        "n_series_used": int(len(records)),
        "n_series_skipped_short": int(skipped_short),
        "n_series_filled_missing": int(filled_missing),
    }
    if not records:
        raise ValueError(
            f"No usable series for dataset={dataset_key} with history_len={history_len} and horizon={horizon}."
        )
    return records, stats


def _train_example_refs(
    series_records: Sequence[ForecastSeriesRecord],
    *,
    history_len: int,
    horizon: int,
    stride: int,
) -> List[ForecastExampleRef]:
    refs: List[ForecastExampleRef] = []
    for series_idx, record in enumerate(series_records):
        max_target_t = int(record.train_prefix_end) - int(horizon)
        for target_t in range(int(history_len), int(max_target_t) + 1, int(max(1, stride))):
            refs.append(ForecastExampleRef(series_idx=int(series_idx), target_t=int(target_t)))
    return refs


def _holdout_example_refs(series_records: Sequence[ForecastSeriesRecord], *, split_name: str) -> List[ForecastExampleRef]:
    refs: List[ForecastExampleRef] = []
    for series_idx, record in enumerate(series_records):
        if str(split_name) == "val":
            target_t = int(record.val_start)
        elif str(split_name) == "test":
            target_t = int(record.test_start)
        else:
            raise ValueError(f"Unknown holdout split: {split_name}")
        refs.append(ForecastExampleRef(series_idx=int(series_idx), target_t=int(target_t)))
    return refs


def build_monash_forecast_splits(
    *,
    dataset_root: str | Path,
    dataset_key: str,
    cfg: OTFlowConfig,
    history_len: int,
    horizon: int,
    stride_train: int = 1,
    time_feature_mode: str = "gap_elapsed",
) -> Dict[str, Any]:
    time_feature_mode = str(time_feature_mode)
    manifest = load_monash_manifest(default_manifest_path(dataset_root, dataset_key))
    tsf_path = find_tsf_file(default_source_dir(dataset_root, dataset_key))
    mase_seasonal_period = infer_mase_seasonal_period(str(manifest.frequency))
    records, record_stats = _build_series_records(
        dataset_key=str(dataset_key),
        tsf_path=tsf_path,
        history_len=int(history_len),
        horizon=int(horizon),
        time_feature_mode=str(time_feature_mode),
        frequency_label=str(manifest.frequency),
    )
    ds_train = MonashForecastWindowDataset(
        dataset_key=str(dataset_key),
        split_name="train",
        history_len=int(history_len),
        horizon=int(horizon),
        series_records=records,
        example_refs=_train_example_refs(records, history_len=int(history_len), horizon=int(horizon), stride=int(stride_train)),
        time_feature_mode=str(time_feature_mode),
        frequency_label=str(manifest.frequency),
        mase_seasonal_period=int(mase_seasonal_period),
    )
    ds_val = MonashForecastWindowDataset(
        dataset_key=str(dataset_key),
        split_name="val",
        history_len=int(history_len),
        horizon=int(horizon),
        series_records=records,
        example_refs=_holdout_example_refs(records, split_name="val"),
        time_feature_mode=str(time_feature_mode),
        frequency_label=str(manifest.frequency),
        mase_seasonal_period=int(mase_seasonal_period),
    )
    ds_test = MonashForecastWindowDataset(
        dataset_key=str(dataset_key),
        split_name="test",
        history_len=int(history_len),
        horizon=int(horizon),
        series_records=records,
        example_refs=_holdout_example_refs(records, split_name="test"),
        time_feature_mode=str(time_feature_mode),
        frequency_label=str(manifest.frequency),
        mase_seasonal_period=int(mase_seasonal_period),
    )
    return {
        "train": ds_train,
        "val": ds_val,
        "test": ds_test,
        "stats": {
            "dataset_key": str(dataset_key),
            "frequency": str(manifest.frequency),
            "official_horizon": int(manifest.official_horizon),
            "experiment_horizon": int(horizon),
            "history_len": int(history_len),
            "normalization_mode": "per_series_train_prefix_zscore",
            "mase_seasonal_period": int(mase_seasonal_period),
            "time_features_enabled": bool(time_feature_mode != "none"),
            "time_feature_mode": str(time_feature_mode),
            "time_feature_dim": _time_feature_dim(str(time_feature_mode)),
            "time_feature_source": "synthetic_regular_frequency" if time_feature_mode != "none" else "none",
            "n_train_examples": int(len(ds_train)),
            "n_val_examples": int(len(ds_val)),
            "n_test_examples": int(len(ds_test)),
            **record_stats,
        },
    }


__all__ = [
    "ForecastExampleRef",
    "ForecastSeriesRecord",
    "MonashForecastWindowDataset",
    "build_monash_forecast_splits",
]
