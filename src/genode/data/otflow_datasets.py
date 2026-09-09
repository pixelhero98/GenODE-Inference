"""Generic sequence windows and chronological splits with training-only normalization."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request

import numpy as np
import torch

from genode.models.config import OTFlowConfig
from genode.path_safety import is_link_or_reparse_point

ArrayLike = np.ndarray | torch.Tensor


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean/std on x [T,D] only."""
    mu = x.mean(axis=0).astype(np.float32)
    sig = (x.std(axis=0) + 1e-06).astype(np.float32)
    return (mu, sig)


def apply_standardizer(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((x - mu[None, :]) / sig[None, :]).astype(np.float32)


def standardize_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sig = fit_standardizer(params)
    return (apply_standardizer(params, mu, sig), mu, sig)


def standardize_cond(cond: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sig = fit_standardizer(cond)
    return (apply_standardizer(cond, mu, sig), mu, sig)


def _future_horizon_from_cfg(cfg: OTFlowConfig) -> int:
    required = 0
    rollout_mode = str(getattr(cfg.model, "rollout_mode", "autoregressive")).strip().lower()
    if rollout_mode == "non_ar":
        required = max(required, max(0, int(getattr(cfg.model, "future_block_len", 1)) - 1))
    return int(max(0, required))


def _time_feature_mode(cfg: OTFlowConfig) -> str:
    use_elapsed = bool(getattr(cfg.model, "use_time_features", False))
    use_gap_only = bool(getattr(cfg.model, "use_time_gaps", False))
    if use_elapsed and use_gap_only:
        raise ValueError("Time features must use exactly one mode: none, gap_only, or gap_elapsed.")
    if use_elapsed:
        return "gap_elapsed"
    if use_gap_only:
        return "gap_only"
    return "none"


def _time_feature_dim(mode: str) -> int:
    mode_key = str(mode)
    if mode_key == "gap_elapsed":
        return 2
    if mode_key == "gap_only":
        return 1
    return 0


def _set_model_cond_dim(cfg: OTFlowConfig, cond_dim: int) -> None:
    resolved = int(cond_dim)
    if resolved <= 0:
        raise ValueError(f"Condition dimension must be positive, got {cond_dim}.")
    current = int(getattr(cfg.model, "cond_dim", 0))
    if current > 0 and current != resolved:
        raise ValueError(f"cfg.model.cond_dim={current} does not match data condition dimension {resolved}.")
    cfg.model.cond_dim = resolved


def _fit_time_gap_scale(timestamps: np.ndarray | None, *, train_end: int, segment_ends: np.ndarray | None) -> float:
    if timestamps is None or int(train_end) <= 1:
        return 1.0
    timestamps = np.asarray(timestamps, dtype=np.int64)
    train_end = min(int(train_end), int(len(timestamps)))
    if train_end <= 1:
        return 1.0
    chunks = []
    if segment_ends is None:
        if train_end > 1:
            chunks.append(np.diff(timestamps[:train_end]))
    else:
        segment_ends = np.asarray(segment_ends, dtype=np.int64)
        seg_starts = _segment_starts_from_ends(segment_ends)
        for seg_start, seg_end in zip(seg_starts, segment_ends, strict=False):
            left = int(seg_start)
            right = min(int(seg_end), int(train_end))
            if right - left > 1:
                chunks.append(np.diff(timestamps[left:right]))
            if int(seg_end) >= int(train_end):
                break
    if not chunks:
        return 1.0
    gaps = np.concatenate(chunks).astype(np.float64)
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if gaps.size == 0:
        return 1.0
    return float(max(np.median(gaps), 1.0))


def _build_time_gap_features(
    timestamps: np.ndarray | None, *, gap_scale: float, segment_ends: np.ndarray | None
) -> np.ndarray | None:
    if timestamps is None:
        return None
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if timestamps.ndim != 1:
        raise ValueError(f"Expected 1D timestamps, got shape={timestamps.shape}.")
    gaps = np.zeros(len(timestamps), dtype=np.float64)
    if len(timestamps) > 1:
        gaps[1:] = np.diff(timestamps).astype(np.float64)
    if segment_ends is not None:
        seg_starts = _segment_starts_from_ends(np.asarray(segment_ends, dtype=np.int64))
        gaps[seg_starts] = 0.0
    else:
        gaps[0] = 0.0
    safe_scale = max(float(gap_scale), 1.0)
    ratio = np.clip(gaps / safe_scale, 0.0001, 10000.0)
    gap_feature = np.log(ratio).astype(np.float32)
    gap_feature[gaps <= 0.0] = 0.0
    return gap_feature[:, None]


def _build_elapsed_time_features(
    timestamps: np.ndarray | None, *, gap_scale: float, segment_ends: np.ndarray | None
) -> np.ndarray | None:
    if timestamps is None:
        return None
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if timestamps.ndim != 1:
        raise ValueError(f"Expected 1D timestamps, got shape={timestamps.shape}.")
    gaps = np.zeros(len(timestamps), dtype=np.float64)
    if len(timestamps) > 1:
        gaps[1:] = np.diff(timestamps).astype(np.float64)
    gaps = np.clip(gaps, 0.0, None)
    safe_scale = max(float(gap_scale), 1.0)
    elapsed = np.zeros(len(timestamps), dtype=np.float64)
    if segment_ends is None:
        if len(timestamps) > 1:
            elapsed[1:] = np.cumsum(gaps[1:] / safe_scale)
    else:
        seg_starts = _segment_starts_from_ends(np.asarray(segment_ends, dtype=np.int64))
        for seg_start, seg_end in zip(seg_starts, np.asarray(segment_ends, dtype=np.int64), strict=False):
            start = int(seg_start)
            stop = int(seg_end)
            if stop - start <= 1:
                continue
            seg_gaps = gaps[start:stop].copy()
            seg_gaps[0] = 0.0
            elapsed[start:stop] = np.cumsum(seg_gaps / safe_scale)
    return elapsed[:, None].astype(np.float32)


def _build_time_features(
    timestamps: np.ndarray | None, *, gap_scale: float, segment_ends: np.ndarray | None, include_elapsed: bool = True
) -> np.ndarray | None:
    gap_feature = _build_time_gap_features(timestamps, gap_scale=float(gap_scale), segment_ends=segment_ends)
    if gap_feature is None:
        return None
    if not bool(include_elapsed):
        return gap_feature.astype(np.float32)
    elapsed_feature = _build_elapsed_time_features(timestamps, gap_scale=float(gap_scale), segment_ends=segment_ends)
    if elapsed_feature is None:
        return None
    return np.concatenate([gap_feature, elapsed_feature], axis=1).astype(np.float32)


class WindowedParamSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        params: np.ndarray,
        mids: np.ndarray,
        history_len: int,
        stride: int = 1,
        params_mean: np.ndarray | None = None,
        params_std: np.ndarray | None = None,
        future_horizon: int = 0,
        cond: np.ndarray | None = None,
        cond_mean: np.ndarray | None = None,
        cond_std: np.ndarray | None = None,
        time_features: np.ndarray | None = None,
        time_gap_features: np.ndarray | None = None,
        elapsed_time_features: np.ndarray | None = None,
        time_gap_scale: float | None = None,
        time_feature_source: str = "none",
        segment_ends: np.ndarray | None = None,
        valid_start_mask: np.ndarray | None = None,
        dataset_kind: str | None = None,
        dataset_metadata: dict[str, object] | None = None,
        global_offset: int = 0,
    ):
        super().__init__()
        self.params = params.astype(np.float32)
        self.mids = mids.astype(np.float32)
        self.H = int(history_len)
        self.stride = int(stride)
        self.future_horizon = int(future_horizon)
        self.params_mean = params_mean
        self.params_std = params_std
        self.cond = cond.astype(np.float32) if cond is not None else None
        self.cond_mean = cond_mean
        self.cond_std = cond_std
        if time_features is None and time_gap_features is not None:
            gap_arr = time_gap_features.astype(np.float32)
            if elapsed_time_features is None:
                time_features = gap_arr
            else:
                elapsed_arr = elapsed_time_features.astype(np.float32)
                time_features = np.concatenate([gap_arr, elapsed_arr], axis=1)
        self.time_features = time_features.astype(np.float32) if time_features is not None else None
        self.time_gap_features = None if self.time_features is None else self.time_features[:, :1]
        self.elapsed_time_features = (
            None if self.time_features is None or self.time_features.shape[1] < 2 else self.time_features[:, 1:2]
        )
        self.time_gap_scale = float(time_gap_scale) if time_gap_scale is not None else None
        self.time_feature_source = str(time_feature_source)
        self.segment_ends = None if segment_ends is None else np.asarray(segment_ends, dtype=np.int64)
        self.valid_start_mask = None if valid_start_mask is None else np.asarray(valid_start_mask, dtype=bool)
        self.dataset_kind = None if dataset_kind is None else str(dataset_kind)
        self.dataset_metadata = {} if dataset_metadata is None else dict(dataset_metadata)
        self.global_offset = int(global_offset)
        self.start_indices = self._build_start_indices()

    def _build_start_indices(self) -> np.ndarray:
        last_exclusive = len(self.params) - max(0, self.future_horizon)
        if self.segment_ends is None:
            starts = np.arange(self.H, last_exclusive, self.stride, dtype=np.int64)
        else:
            starts = []
            seg_starts = np.concatenate(([0], self.segment_ends[:-1]))
            for seg_start, seg_end in zip(seg_starts, self.segment_ends, strict=False):
                local_start = int(seg_start) + self.H
                local_end = int(seg_end) - max(0, self.future_horizon)
                if local_start < local_end:
                    starts.append(np.arange(local_start, local_end, self.stride, dtype=np.int64))
            if not starts:
                return np.empty(0, dtype=np.int64)
            starts = np.concatenate(starts)
        if self.valid_start_mask is not None:
            if len(self.valid_start_mask) != len(self.params):
                raise ValueError("valid_start_mask length mismatch")
            starts = starts[self.valid_start_mask[starts]]
        return starts

    def segment_end_for_t(self, t: int | np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=np.int64)
        if self.segment_ends is None:
            return np.full_like(t_arr, len(self.params), dtype=np.int64)
        idx = np.searchsorted(self.segment_ends, t_arr, side="right")
        return self.segment_ends[idx]

    def __len__(self):
        return len(self.start_indices)

    def has_time_gap_features(self) -> bool:
        return self.time_gap_features is not None

    def has_time_features(self) -> bool:
        return self.time_features is not None

    def _slice_time_features(self, start: int, stop: int) -> np.ndarray | None:
        if self.time_features is None:
            return None
        features = self.time_features[int(start) : int(stop)].astype(np.float32, copy=True)
        if features.shape[0] > 0 and features.shape[1] >= 2:
            features[:, 1] = features[:, 1] - float(features[0, 1])
        return features

    def future_time_features(self, t0: int, horizon: int) -> torch.Tensor | None:
        features = self._slice_time_features(int(t0), int(t0) + int(horizon))
        if features is None:
            return None
        return torch.from_numpy(features)

    def future_time_gap_features(self, t0: int, horizon: int) -> torch.Tensor | None:
        if self.time_gap_features is None:
            return None
        return torch.from_numpy(self.time_gap_features[int(t0) : int(t0) + int(horizon)].astype(np.float32, copy=True))

    def __getitem__(self, idx: int):
        t = int(self.start_indices[idx])
        t_global = self.global_offset + t
        hist = self.params[t - self.H : t]
        hist_time = self._slice_time_features(t - self.H, t)
        if hist_time is not None:
            hist = np.concatenate([hist, hist_time], axis=1).astype(np.float32, copy=False)
        tgt = self.params[t]
        meta = {
            "t": int(t),
            "t_global": int(t_global),
            "mid_prev": float(self.mids[t - 1]),
            "init_mid_for_window": float(self.mids[t - self.H]),
        }
        fut_t = None
        if self.future_horizon > 0:
            fut = self.params[t + 1 : t + 1 + self.future_horizon]
            fut_t = torch.from_numpy(fut)
        hist_t = torch.from_numpy(hist)
        tgt_t = torch.from_numpy(tgt)
        if self.cond is None:
            if fut_t is None:
                return (hist_t, tgt_t, meta)
            return (hist_t, tgt_t, fut_t, meta)
        c = torch.from_numpy(self.cond[t])
        if fut_t is None:
            return (hist_t, tgt_t, c, meta)
        return (hist_t, tgt_t, fut_t, c, meta)


def _resolve_split_bounds(
    T: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float | None = None,
    train_end: int | None = None,
    val_end: int | None = None,
) -> tuple[int, int]:
    """Return (train_end, val_end) as absolute timestep boundaries in [0, T].

    Splits are interpreted over raw timesteps (params rows).
    """
    if test_frac is None:
        test_frac = 1.0 - train_frac - val_frac
    if train_end is None or val_end is None:
        if train_frac <= 0 or val_frac < 0 or test_frac < 0:
            raise ValueError("Invalid split fractions.")
        s = train_frac + val_frac + test_frac
        if abs(s - 1.0) > 1e-06:
            raise ValueError(f"Split fractions must sum to 1.0, got {s:.6f}")
        train_end = int(round(T * train_frac))
        val_end = int(round(T * (train_frac + val_frac)))
    train_end = int(train_end)
    val_end = int(val_end)
    if not 0 < train_end < val_end <= T:
        raise ValueError(f"Invalid split bounds: train_end={train_end}, val_end={val_end}, T={T}")
    return (train_end, val_end)


def _slice_segment_with_history(arr: np.ndarray, start_t: int, end_t: int, history_len: int) -> tuple[np.ndarray, int]:
    """Slice arr so targets in [start_t, end_t) are valid with history.

    Returns
    -------
    arr_seg : np.ndarray
        arr[left:end_t], where left=max(0, start_t-history_len)
    left : int
        Global offset corresponding to local index 0.
    """
    left = max(0, int(start_t) - int(history_len))
    arr_seg = arr[left : int(end_t)]
    return (arr_seg, left)


def _segment_starts_from_ends(segment_ends: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.asarray(segment_ends, dtype=np.int64)[:-1])).astype(np.int64)


def _resolve_segment_split_bounds(
    T: int,
    segment_ends: np.ndarray,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float | None,
    train_end: int | None,
    val_end: int | None,
) -> tuple[int, int]:
    segment_ends = np.asarray(segment_ends, dtype=np.int64)
    if len(segment_ends) < 3:
        raise ValueError("Need at least 3 segments for train/val/test splits.")
    if int(segment_ends[-1]) != int(T):
        raise ValueError("segment_ends must terminate at T.")
    if test_frac is None:
        test_frac = 1.0 - train_frac - val_frac
    if train_end is None or val_end is None:
        s = train_frac + val_frac + test_frac
        if abs(s - 1.0) > 1e-06:
            raise ValueError(f"Split fractions must sum to 1.0, got {s:.6f}")
        n_segments = len(segment_ends)
        train_seg = max(1, int(round(n_segments * train_frac)))
        val_seg = max(train_seg + 1, int(round(n_segments * (train_frac + val_frac))))
        val_seg = min(val_seg, n_segments - 1)
        train_end = int(segment_ends[train_seg - 1])
        val_end = int(segment_ends[val_seg - 1])
    else:
        train_idx = int(np.searchsorted(segment_ends, int(train_end), side="left"))
        val_idx = int(np.searchsorted(segment_ends, int(val_end), side="left"))
        train_idx = min(max(train_idx, 0), len(segment_ends) - 2)
        val_idx = min(max(val_idx, train_idx + 1), len(segment_ends) - 1)
        train_end = int(segment_ends[train_idx])
        val_end = int(segment_ends[val_idx])
    if not 0 < train_end < val_end <= T:
        raise ValueError(f"Invalid segment split bounds: train_end={train_end}, val_end={val_end}, T={T}")
    return (int(train_end), int(val_end))


def _make_windowed_dataset_from_arrays(
    params_full: np.ndarray,
    mids_full: np.ndarray,
    cfg: OTFlowConfig,
    *,
    stride: int,
    start_t: int,
    end_t: int,
    params_mean: np.ndarray | None,
    params_std: np.ndarray | None,
    cond_full: np.ndarray | None,
    cond_mean: np.ndarray | None,
    cond_std: np.ndarray | None,
    time_features_full: np.ndarray | None,
    time_gap_scale: float | None,
    time_feature_source: str = "none",
    segment_ends_full: np.ndarray | None = None,
    valid_start_mask_full: np.ndarray | None = None,
    dataset_kind: str | None = None,
    dataset_metadata: dict[str, object] | None = None,
) -> WindowedParamSequenceDataset:
    """Construct a split dataset [start_t,end_t) with left history buffer and fixed normalization stats."""
    H = int(cfg.history_len)
    local_segment_ends = None
    valid_start_mask_seg = None
    if segment_ends_full is None:
        params_seg_raw, left = _slice_segment_with_history(params_full, start_t, end_t, H)
        mids_seg, left_m = _slice_segment_with_history(mids_full, start_t, end_t, H)
        if left_m != left:
            raise RuntimeError("Unexpected offset mismatch")
        if valid_start_mask_full is not None:
            valid_start_mask_seg, left_v = _slice_segment_with_history(valid_start_mask_full, start_t, end_t, H)
            if left_v != left:
                raise RuntimeError("Valid-start offset mismatch")
    else:
        segment_ends_full = np.asarray(segment_ends_full, dtype=np.int64)
        segment_starts_full = _segment_starts_from_ends(segment_ends_full)
        mask = (segment_ends_full > int(start_t)) & (segment_starts_full < int(end_t))
        if not np.any(mask):
            raise ValueError(f"No segments found inside split [{start_t}, {end_t}).")
        left = int(segment_starts_full[mask][0])
        right = int(segment_ends_full[mask][-1])
        params_seg_raw = params_full[left:right]
        mids_seg = mids_full[left:right]
        local_segment_ends = (segment_ends_full[mask] - left).astype(np.int64)
        if valid_start_mask_full is not None:
            valid_start_mask_seg = np.asarray(valid_start_mask_full[left:right], dtype=bool)
    if params_mean is not None and params_std is not None:
        params_seg = apply_standardizer(params_seg_raw, params_mean, params_std)
    else:
        params_seg = params_seg_raw.astype(np.float32)
    cond_seg = None
    if cond_full is not None:
        if segment_ends_full is None:
            cond_seg_raw, left_c = _slice_segment_with_history(cond_full, start_t, end_t, H)
            if left_c != left:
                raise RuntimeError("Conditioning offset mismatch")
        else:
            cond_seg_raw = cond_full[left:right]
        if cond_mean is not None and cond_std is not None:
            cond_seg = apply_standardizer(cond_seg_raw, cond_mean, cond_std)
        else:
            cond_seg = cond_seg_raw.astype(np.float32)
    time_features_seg = None
    if time_features_full is not None:
        if segment_ends_full is None:
            time_features_seg, left_g = _slice_segment_with_history(time_features_full, start_t, end_t, H)
            if left_g != left:
                raise RuntimeError("Time-feature offset mismatch")
        else:
            time_features_seg = time_features_full[left:right]
        time_features_seg = time_features_seg.astype(np.float32, copy=False)
    ds = WindowedParamSequenceDataset(
        params=params_seg,
        mids=mids_seg,
        history_len=cfg.history_len,
        stride=stride,
        params_mean=params_mean,
        params_std=params_std,
        future_horizon=_future_horizon_from_cfg(cfg),
        cond=cond_seg,
        cond_mean=cond_mean,
        cond_std=cond_std,
        time_features=time_features_seg,
        time_gap_scale=time_gap_scale,
        time_feature_source=time_feature_source,
        segment_ends=local_segment_ends,
        valid_start_mask=valid_start_mask_seg,
        dataset_kind=dataset_kind,
        dataset_metadata=dataset_metadata,
        global_offset=left,
    )
    g = ds.global_offset + ds.start_indices
    mask = (g >= int(start_t)) & (g < int(end_t))
    ds.start_indices = ds.start_indices[mask]
    if len(ds.start_indices) == 0:
        raise ValueError(
            f"Empty split dataset: start_t={start_t}, end_t={end_t}, H={cfg.history_len}, stride={stride}. Increase segment length or reduce history_len."
        )
    return ds


def build_dataset_splits_from_arrays(
    params_raw: np.ndarray,
    mids: np.ndarray,
    cfg: OTFlowConfig,
    *,
    timestamps: np.ndarray | None = None,
    cond_raw_full: np.ndarray | None = None,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float | None = None,
    train_end: int | None = None,
    val_end: int | None = None,
    segment_ends: np.ndarray | None = None,
    valid_start_mask: np.ndarray | None = None,
    dataset_kind: str | None = None,
    dataset_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Chronological train/val/test split with train-only normalization statistics.

    Parameters
    ----------
    params_raw, mids : full timeline arrays [T, D], [T]
    cfg : OTFlowConfig
    stride_train, stride_eval : int
        Often use denser train and sparser eval.
    train_frac/val_frac/test_frac OR train_end/val_end :
        Define split boundaries on raw timesteps.

    Returns
    -------
    dict with keys:
      - 'train', 'val', 'test' : WindowedParamSequenceDataset
      - 'stats' : normalization statistics and split bounds
    """
    params_raw = np.asarray(params_raw)
    if params_raw.ndim != 2:
        raise ValueError(f"params_raw must be rank 2, got shape {params_raw.shape}.")
    if (
        not np.issubdtype(params_raw.dtype, np.number)
        or np.issubdtype(params_raw.dtype, np.complexfloating)
        or (not bool(np.all(np.isfinite(params_raw))))
    ):
        raise ValueError("params_raw must contain finite real numeric values.")
    expected_snapshot_dim = int(cfg.snapshot_dim)
    if params_raw.shape[1] != expected_snapshot_dim:
        raise ValueError(
            f"params_raw width {params_raw.shape[1]} does not match cfg.snapshot_dim={expected_snapshot_dim}."
        )
    mids = np.asarray(mids)
    if mids.ndim != 1:
        raise ValueError(f"mids must be rank 1, got shape {mids.shape}.")
    if (
        not np.issubdtype(mids.dtype, np.number)
        or np.issubdtype(mids.dtype, np.complexfloating)
        or (not bool(np.all(np.isfinite(mids))))
    ):
        raise ValueError("mids must contain finite real numeric values.")
    T = int(len(params_raw))
    if len(mids) != T:
        raise ValueError("params_raw and mids length mismatch")
    if timestamps is not None and len(timestamps) != T:
        raise ValueError("params_raw and timestamps length mismatch")
    if cond_raw_full is not None and len(cond_raw_full) != T:
        raise ValueError("params_raw and cond_raw_full length mismatch")
    if valid_start_mask is not None and len(valid_start_mask) != T:
        raise ValueError("params_raw and valid_start_mask length mismatch")
    if segment_ends is None:
        train_end, val_end = _resolve_split_bounds(
            T, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, train_end=train_end, val_end=val_end
        )
    else:
        train_end, val_end = _resolve_segment_split_bounds(
            T,
            np.asarray(segment_ends, dtype=np.int64),
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            train_end=train_end,
            val_end=val_end,
        )
    if cfg.standardize:
        p_mu, p_sig = fit_standardizer(params_raw[:train_end])
    else:
        p_mu = p_sig = None
    resolved_cond_raw_full = None if cond_raw_full is None else np.asarray(cond_raw_full, dtype=np.float32)
    c_mu = c_sig = None
    if resolved_cond_raw_full is None and cfg.use_cond_features:
        raise ValueError("Conditional features must be supplied explicitly through cond_raw_full.")
    if resolved_cond_raw_full is not None:
        if not bool(cfg.use_cond_features):
            raise ValueError("External conditional features require cfg.data.use_cond_features=True.")
        if cfg.cond_standardize:
            c_mu, c_sig = fit_standardizer(resolved_cond_raw_full[:train_end])
        _set_model_cond_dim(cfg, int(resolved_cond_raw_full.shape[1]))
    time_features_full = None
    time_gap_scale = None
    time_feature_source = "none"
    time_feature_mode = _time_feature_mode(cfg)
    if time_feature_mode != "none":
        time_gap_scale = _fit_time_gap_scale(
            None if timestamps is None else np.asarray(timestamps, dtype=np.int64),
            train_end=int(train_end),
            segment_ends=segment_ends,
        )
        time_features_full = _build_time_features(
            None if timestamps is None else np.asarray(timestamps, dtype=np.int64),
            gap_scale=float(time_gap_scale),
            segment_ends=segment_ends,
            include_elapsed=bool(time_feature_mode == "gap_elapsed"),
        )
        if time_features_full is None:
            time_features_full = np.zeros((T, _time_feature_dim(time_feature_mode)), dtype=np.float32)
            time_feature_source = "missing_timestamps_zero_fill"
        else:
            time_feature_source = "timestamps"
    ds_train = _make_windowed_dataset_from_arrays(
        params_full=params_raw,
        mids_full=mids,
        cfg=cfg,
        stride=stride_train,
        start_t=cfg.history_len,
        end_t=train_end,
        params_mean=p_mu,
        params_std=p_sig,
        cond_full=resolved_cond_raw_full,
        cond_mean=c_mu,
        cond_std=c_sig,
        time_features_full=time_features_full,
        time_gap_scale=time_gap_scale,
        time_feature_source=time_feature_source,
        segment_ends_full=segment_ends,
        valid_start_mask_full=valid_start_mask,
        dataset_kind=dataset_kind,
        dataset_metadata=dataset_metadata,
    )
    ds_val = _make_windowed_dataset_from_arrays(
        params_full=params_raw,
        mids_full=mids,
        cfg=cfg,
        stride=stride_eval,
        start_t=train_end,
        end_t=val_end,
        params_mean=p_mu,
        params_std=p_sig,
        cond_full=resolved_cond_raw_full,
        cond_mean=c_mu,
        cond_std=c_sig,
        time_features_full=time_features_full,
        time_gap_scale=time_gap_scale,
        time_feature_source=time_feature_source,
        segment_ends_full=segment_ends,
        valid_start_mask_full=valid_start_mask,
        dataset_kind=dataset_kind,
        dataset_metadata=dataset_metadata,
    )
    ds_test = _make_windowed_dataset_from_arrays(
        params_full=params_raw,
        mids_full=mids,
        cfg=cfg,
        stride=stride_eval,
        start_t=val_end,
        end_t=T,
        params_mean=p_mu,
        params_std=p_sig,
        cond_full=resolved_cond_raw_full,
        cond_mean=c_mu,
        cond_std=c_sig,
        time_features_full=time_features_full,
        time_gap_scale=time_gap_scale,
        time_feature_source=time_feature_source,
        segment_ends_full=segment_ends,
        valid_start_mask_full=valid_start_mask,
        dataset_kind=dataset_kind,
        dataset_metadata=dataset_metadata,
    )
    stats = {
        "T": int(T),
        "train_end": int(train_end),
        "val_end": int(val_end),
        "test_end": int(T),
        "params_mean": p_mu,
        "params_std": p_sig,
        "cond_mean": c_mu,
        "cond_std": c_sig,
        "cond_dim": int(resolved_cond_raw_full.shape[1]) if resolved_cond_raw_full is not None else 0,
        "history_len": int(cfg.history_len),
        "time_gap_scale": None if time_gap_scale is None else float(time_gap_scale),
        "use_time_gaps": bool(getattr(cfg.model, "use_time_gaps", False)),
        "use_time_features": bool(getattr(cfg.model, "use_time_features", False)),
        "time_feature_mode": str(time_feature_mode),
        "time_feature_dim": 0 if time_features_full is None else int(time_features_full.shape[1]),
        "time_feature_source": str(time_feature_source),
        "n_segments": int(len(segment_ends)) if segment_ends is not None else 1,
        "dataset_kind": None if dataset_kind is None else str(dataset_kind),
        "dataset_metadata": {} if dataset_metadata is None else dict(dataset_metadata),
        "n_valid_target_starts": None
        if valid_start_mask is None
        else int(np.count_nonzero(np.asarray(valid_start_mask, dtype=bool))),
    }
    return {"train": ds_train, "val": ds_val, "test": ds_test, "stats": stats}


def _sha256_path(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(path: str | os.PathLike[str], *, expected_size: int, expected_sha256: str) -> bool:
    resolved = os.fspath(path)
    return (
        not is_link_or_reparse_point(resolved)
        and os.path.isfile(resolved)
        and (os.path.getsize(resolved) == int(expected_size))
        and (_sha256_path(resolved) == str(expected_sha256).lower())
    )


def _download_url_to_path(
    url: str, destination: str | os.PathLike[str], *, expected_size: int, expected_sha256: str
) -> str:
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError(f"expected_size must be a positive integer, got {expected_size!r}.")
    sha256 = str(expected_sha256).strip().lower()
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError(f"expected_sha256 must be a lowercase hexadecimal SHA-256 digest, got {expected_sha256!r}.")
    resolved = os.fspath(destination)
    if is_link_or_reparse_point(resolved):
        raise ValueError(f"Download destination may not be a symlink, junction, or reparse point: {resolved}.")
    if os.path.exists(resolved) and (not os.path.isfile(resolved)):
        raise ValueError(f"Download destination must be a regular file path: {resolved}.")
    destination_dir = os.path.dirname(os.path.abspath(resolved))
    os.makedirs(destination_dir, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination_dir, prefix=f".{os.path.basename(resolved)}.", suffix=".download", delete=False
        ) as out_fh:
            temporary = out_fh.name
            digest = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(str(url), timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > expected_size:
                        raise ValueError(f"Download from {url} exceeded the expected size of {expected_size} bytes.")
                    digest.update(chunk)
                    out_fh.write(chunk)
        if total != expected_size:
            raise ValueError(f"Download from {url} has size {total}; expected {expected_size} bytes.")
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != sha256:
            raise ValueError(f"Download from {url} has SHA-256 {observed_sha256}; expected {sha256}.")
        os.replace(temporary, resolved)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)
    return resolved


__all__ = [
    "WindowedParamSequenceDataset",
    "apply_standardizer",
    "build_dataset_splits_from_arrays",
    "fit_standardizer",
    "standardize_cond",
    "standardize_params",
]
