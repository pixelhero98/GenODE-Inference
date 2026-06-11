"""otflow_datasets.py

Data + representation utilities for Level-2 (L2) limit order books.

Contains:
- L2FeatureMap: valid-by-construction encoding/decoding (raw L2 <-> unconstrained params)
- Standardization helpers
- WindowedParamSequenceDataset (history->target windows; optional future horizon for rollout)
- Builders for prepared NPZ, crypto, and synthetic sequences
- Basic raw-space metrics
- NEW: chronological split builders with train-only normalization (anti-leakage)

Also includes derived microstructure conditioning features (cond) computed from the
parameter sequence: spread, returns, abs returns, microprice deviation, multi-depth
imbalance, Δbest sizes, rolling vol.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Dict, Optional, Tuple, Union
import urllib.request

import numpy as np
import torch

from genode.models.config import OTFlowConfig
from genode.data.otflow_paths import default_lobster_synthetic_profile_path, project_data_root

ArrayLike = Union[np.ndarray, torch.Tensor]
DEFAULT_SYNTHETIC_LENGTH = 2_000_000
DEFAULT_CRYPTOS_NPZ = str(project_data_root() / "cryptos_binance_spot_monthly_1s_l10.npz")
DEFAULT_ES_MBP_10_NPZ = str(project_data_root() / "es_mbp_10.npz")
LOBIFLOW_CRYPTOS_NPZ_URL = (
    "https://huggingface.co/datasets/mpstoryfans/lobiflow/resolve/main/"
    "data/cryptos/cryptos_binance_spot_monthly_1s_l10.npz?download=1"
)
DEFAULT_OPTIVER_NPZ = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data_optiver",
    "optiver_train_8stocks_l2.npz",
)
DEFAULT_LOBSTER_SYNTH_PROFILE = default_lobster_synthetic_profile_path()
LOBSTER_SYNTHETIC_DATASET_KEY = "lobster_synthetic"
LOBIFLOW_SYNTHETIC_PROFILE_URL = (
    "https://huggingface.co/datasets/mpstoryfans/lobiflow/resolve/main/"
    "data/synthetic/lobster_free_sample_profile_10.json"
)


# -----------------------------
# Feature map: valid L2 <-> unconstrained params
# -----------------------------
class L2FeatureMap:
    """Encode/decode between raw L2 snapshots and an unconstrained vector.

    Raw format expected by encode_sequence():
      ask_p, ask_v, bid_p, bid_v each shape [T, L]

    Parameter vector per snapshot (dim=4L):
      [delta_mid, log_spread,
       log_ask_gaps(2..L), log_bid_gaps(2..L),
       log_ask_sizes(1..L), log_bid_sizes(1..L)]
    """

    def __init__(self, levels: int = 10, eps: float = 1e-8):
        self.L = int(levels)
        self.eps = float(eps)

    def encode_sequence(
        self,
        ask_p: np.ndarray,
        ask_v: np.ndarray,
        bid_p: np.ndarray,
        bid_v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T, L = ask_p.shape
        assert L == self.L

        mid = 0.5 * (ask_p[:, 0] + bid_p[:, 0])
        spread = np.maximum(ask_p[:, 0] - bid_p[:, 0], self.eps)

        # delta mid
        delta_mid = np.zeros(T, dtype=np.float32)
        delta_mid[1:] = (mid[1:] - mid[:-1]).astype(np.float32)

        # gaps (positive)
        ask_gaps = np.maximum(np.diff(ask_p, axis=1), self.eps)
        # reverse-then-diff to ensure positive ladder gaps for bid side, then reverse back
        bid_gaps = np.maximum(np.diff(bid_p[:, ::-1], axis=1)[:, ::-1], self.eps)

        # params
        log_spread = np.log(spread + self.eps).astype(np.float32)
        log_ask_gaps = np.log(ask_gaps + self.eps).astype(np.float32)  # [T, L-1]
        log_bid_gaps = np.log(bid_gaps + self.eps).astype(np.float32)  # [T, L-1]
        log_ask_v = np.log(np.maximum(ask_v, self.eps)).astype(np.float32)  # [T, L]
        log_bid_v = np.log(np.maximum(bid_v, self.eps)).astype(np.float32)  # [T, L]

        params = np.concatenate(
            [
                delta_mid[:, None],
                log_spread[:, None],
                log_ask_gaps,
                log_bid_gaps,
                log_ask_v,
                log_bid_v,
            ],
            axis=1,
        ).astype(np.float32)

        return params, mid.astype(np.float32)

    def decode_sequence(self, params: np.ndarray, init_mid: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode params to raw L2 arrays using the mid immediately before the window.

        Notes
        -----
        `delta_mid[t]` is interpreted as `mid[t] - mid[t-1]`. Therefore, `init_mid`
        should be the previous mid (at t-1 for the first decoded row).
        """
        T, D = params.shape
        L = self.L
        assert D == 4 * L, f"Expected D={4*L}, got {D}"

        delta_mid = params[:, 0]
        log_spread = params[:, 1]
        log_ask_gaps = params[:, 2 : 2 + (L - 1)]
        log_bid_gaps = params[:, 2 + (L - 1) : 2 + 2 * (L - 1)]
        log_ask_v = params[:, 2 + 2 * (L - 1) : 2 + 2 * (L - 1) + L]
        log_bid_v = params[:, 2 + 2 * (L - 1) + L :]

        mid = np.zeros(T, dtype=np.float32)
        prev_mid = float(init_mid)
        for t in range(T):
            prev_mid = prev_mid + float(delta_mid[t])
            mid[t] = prev_mid

        spread = np.exp(log_spread)
        ask1 = mid + 0.5 * spread
        bid1 = mid - 0.5 * spread

        ask_p = np.zeros((T, L), dtype=np.float32)
        bid_p = np.zeros((T, L), dtype=np.float32)
        ask_p[:, 0] = ask1
        bid_p[:, 0] = bid1

        ask_gaps = np.exp(log_ask_gaps)
        bid_gaps = np.exp(log_bid_gaps)
        for i in range(1, L):
            ask_p[:, i] = ask_p[:, i - 1] + ask_gaps[:, i - 1]
            bid_p[:, i] = bid_p[:, i - 1] - bid_gaps[:, i - 1]

        ask_v = np.exp(log_ask_v).astype(np.float32)
        bid_v = np.exp(log_bid_v).astype(np.float32)
        return ask_p, ask_v, bid_p, bid_v

# -----------------------------
# Standardization helpers
# -----------------------------
def fit_standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit mean/std on x [T,D] only."""
    mu = x.mean(axis=0).astype(np.float32)
    sig = (x.std(axis=0) + 1e-6).astype(np.float32)
    return mu, sig


def apply_standardizer(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((x - mu[None, :]) / sig[None, :]).astype(np.float32)


def standardize_params(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sig = fit_standardizer(params)
    return apply_standardizer(params, mu, sig), mu, sig


def standardize_cond(cond: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sig = fit_standardizer(cond)
    return apply_standardizer(cond, mu, sig), mu, sig


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


def _use_time_features_enabled(cfg: OTFlowConfig) -> bool:
    return _time_feature_mode(cfg) != "none"


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


def _timestamps_from_loaded_npz(data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    for key in ("local_timestamps", "timestamps", "ts_event", "ts_recv", "ts"):
        if key in data:
            return np.asarray(data[key], dtype=np.int64)
    return None


def _fit_time_gap_scale(
    timestamps: Optional[np.ndarray],
    *,
    train_end: int,
    segment_ends: Optional[np.ndarray],
) -> float:
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
        for seg_start, seg_end in zip(seg_starts, segment_ends):
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
    timestamps: Optional[np.ndarray],
    *,
    gap_scale: float,
    segment_ends: Optional[np.ndarray],
) -> Optional[np.ndarray]:
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
    ratio = np.clip(gaps / safe_scale, 1e-4, 1e4)
    gap_feature = np.log(ratio).astype(np.float32)
    gap_feature[gaps <= 0.0] = 0.0
    return gap_feature[:, None]


def _build_elapsed_time_features(
    timestamps: Optional[np.ndarray],
    *,
    gap_scale: float,
    segment_ends: Optional[np.ndarray],
) -> Optional[np.ndarray]:
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
        for seg_start, seg_end in zip(seg_starts, np.asarray(segment_ends, dtype=np.int64)):
            start = int(seg_start)
            stop = int(seg_end)
            if stop - start <= 1:
                continue
            seg_gaps = gaps[start:stop].copy()
            seg_gaps[0] = 0.0
            elapsed[start:stop] = np.cumsum(seg_gaps / safe_scale)
    return elapsed[:, None].astype(np.float32)


def _build_time_features(
    timestamps: Optional[np.ndarray],
    *,
    gap_scale: float,
    segment_ends: Optional[np.ndarray],
    include_elapsed: bool = True,
) -> Optional[np.ndarray]:
    gap_feature = _build_time_gap_features(
        timestamps,
        gap_scale=float(gap_scale),
        segment_ends=segment_ends,
    )
    if gap_feature is None:
        return None
    if not bool(include_elapsed):
        return gap_feature.astype(np.float32)
    elapsed_feature = _build_elapsed_time_features(
        timestamps,
        gap_scale=float(gap_scale),
        segment_ends=segment_ends,
    )
    if elapsed_feature is None:
        return None
    return np.concatenate([gap_feature, elapsed_feature], axis=1).astype(np.float32)


# -----------------------------
# Derived conditioning features (from params + mids)
# -----------------------------
def build_cond_features(params_raw: np.ndarray, mids: np.ndarray, cfg: OTFlowConfig) -> np.ndarray:
    """Compute per-timestep conditioning features from raw params."""
    L = cfg.levels
    eps = cfg.eps
    T = params_raw.shape[0]

    log_spread = params_raw[:, 1]
    spread = np.exp(log_spread)

    # returns from mids
    ret = np.zeros(T, dtype=np.float32)
    ret[1:] = (mids[1:] - mids[:-1]) / (np.abs(mids[:-1]) + 1.0)
    absret = np.abs(ret)

    # volumes
    off = 2 + 2 * (L - 1)
    log_ask_v = params_raw[:, off : off + L]
    log_bid_v = params_raw[:, off + L : off + 2 * L]
    ask_v = np.exp(log_ask_v)
    bid_v = np.exp(log_bid_v)

    # best prices
    ask1 = mids + 0.5 * spread
    bid1 = mids - 0.5 * spread

    # microprice deviation (normalized by spread)
    micro = (ask1 * bid_v[:, 0] + bid1 * ask_v[:, 0]) / (ask_v[:, 0] + bid_v[:, 0] + eps)
    micro_dev = (micro - mids) / (spread + eps)

    # multi-depth imbalance + depth sums
    feats = [
        log_spread.astype(np.float32)[:, None],
        ret[:, None],
        absret[:, None],
        micro_dev.astype(np.float32)[:, None],
    ]
    for k in cfg.cond_depths:
        kk = int(min(L, max(1, k)))
        b = bid_v[:, :kk].sum(axis=1)
        a = ask_v[:, :kk].sum(axis=1)
        imb = (b - a) / (b + a + eps)
        feats.append(imb.astype(np.float32)[:, None])

    # delta best sizes (relative)
    d_bid1 = np.zeros(T, dtype=np.float32)
    d_ask1 = np.zeros(T, dtype=np.float32)
    d_bid1[1:] = (bid_v[1:, 0] - bid_v[:-1, 0]) / (bid_v[:-1, 0] + eps)
    d_ask1[1:] = (ask_v[1:, 0] - ask_v[:-1, 0]) / (ask_v[:-1, 0] + eps)
    feats.append(d_bid1[:, None])
    feats.append(d_ask1[:, None])

    # rolling volatility of returns
    w = int(max(5, cfg.cond_vol_window))
    vol = np.zeros(T, dtype=np.float32)
    for t in range(T):
        s = max(0, t - w + 1)
        vol[t] = float(np.std(ret[s : t + 1]))
    feats.append(vol[:, None])

    cond = np.concatenate(feats, axis=1).astype(np.float32)
    return cond


# -----------------------------
# Dataset
# -----------------------------
class WindowedParamSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        params: np.ndarray,
        mids: np.ndarray,
        history_len: int,
        stride: int = 1,
        params_mean: Optional[np.ndarray] = None,
        params_std: Optional[np.ndarray] = None,
        future_horizon: int = 0,
        cond: Optional[np.ndarray] = None,
        cond_mean: Optional[np.ndarray] = None,
        cond_std: Optional[np.ndarray] = None,
        time_features: Optional[np.ndarray] = None,
        time_gap_features: Optional[np.ndarray] = None,
        elapsed_time_features: Optional[np.ndarray] = None,
        time_gap_scale: Optional[float] = None,
        time_feature_source: str = "none",
        segment_ends: Optional[np.ndarray] = None,
        valid_start_mask: Optional[np.ndarray] = None,
        dataset_kind: Optional[str] = None,
        dataset_metadata: Optional[Dict[str, object]] = None,
        global_offset: int = 0,  # NEW: maps local t -> original/global t
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
            for seg_start, seg_end in zip(seg_starts, self.segment_ends):
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

    def segment_end_for_t(self, t: Union[int, np.ndarray]) -> np.ndarray:
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

    def _slice_time_features(self, start: int, stop: int) -> Optional[np.ndarray]:
        if self.time_features is None:
            return None
        features = self.time_features[int(start) : int(stop)].astype(np.float32, copy=True)
        if features.shape[0] > 0 and features.shape[1] >= 2:
            features[:, 1] = features[:, 1] - float(features[0, 1])
        return features

    def future_time_features(self, t0: int, horizon: int) -> Optional[torch.Tensor]:
        features = self._slice_time_features(int(t0), int(t0) + int(horizon))
        if features is None:
            return None
        return torch.from_numpy(features)

    def future_time_gap_features(self, t0: int, horizon: int) -> Optional[torch.Tensor]:
        if self.time_gap_features is None:
            return None
        return torch.from_numpy(self.time_gap_features[int(t0) : int(t0) + int(horizon)].astype(np.float32, copy=True))

    def __getitem__(self, idx: int):
        t = int(self.start_indices[idx])  # local index inside this split dataset
        t_global = self.global_offset + t

        hist = self.params[t - self.H : t]
        hist_time = self._slice_time_features(t - self.H, t)
        if hist_time is not None:
            hist = np.concatenate(
                [hist, hist_time],
                axis=1,
            ).astype(np.float32, copy=False)
        tgt = self.params[t]

        meta = {
            "t": int(t),  # local
            "t_global": int(t_global),  # NEW
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
                return hist_t, tgt_t, meta
            return hist_t, tgt_t, fut_t, meta

        c = torch.from_numpy(self.cond[t])
        if fut_t is None:
            return hist_t, tgt_t, c, meta
        return hist_t, tgt_t, fut_t, c, meta


# -----------------------------
# Loaders / builders
# -----------------------------
def load_l2_npz(path: str) -> Dict[str, np.ndarray]:
    """Load a standardized L2 snapshot NPZ prepared by `lob_prepare_dataset.py`.

    Required keys:
      - ask_p, ask_v, bid_p, bid_v : [T,L] float arrays

    Optional keys:
      - mids : [T] float32
      - params_raw : [T,4L] float32
      - ts : [T] timestamps
    """
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    for k in ("ask_p", "ask_v", "bid_p", "bid_v", "mids", "params_raw"):
        if k in out:
            out[k] = out[k].astype(np.float32)
    return out


def _build_windowed_dataset(
    params_raw: np.ndarray,
    mids: np.ndarray,
    cfg: OTFlowConfig,
    stride: int,
    *,
    timestamps: Optional[np.ndarray] = None,
) -> WindowedParamSequenceDataset:
    """Build a windowed LOB dataset from one parameterized array bundle."""
    # params
    if cfg.standardize:
        params, mu, sig = standardize_params(params_raw)
    else:
        params, mu, sig = params_raw.astype(np.float32), None, None

    # cond features
    cond = None
    c_mu = c_sig = None
    if cfg.use_cond_features:
        cond_raw = build_cond_features(params_raw, mids, cfg)
        if cfg.cond_standardize:
            cond, c_mu, c_sig = standardize_cond(cond_raw)
        else:
            cond = cond_raw

        _set_model_cond_dim(cfg, int(cond.shape[1]))

    time_features = None
    time_gap_scale = None
    time_feature_source = "none"
    time_feature_mode = _time_feature_mode(cfg)
    if time_feature_mode != "none":
        time_gap_scale = _fit_time_gap_scale(
            None if timestamps is None else np.asarray(timestamps, dtype=np.int64),
            train_end=len(params_raw),
            segment_ends=None,
        )
        time_features = _build_time_features(
            None if timestamps is None else np.asarray(timestamps, dtype=np.int64),
            gap_scale=float(time_gap_scale),
            segment_ends=None,
            include_elapsed=bool(time_feature_mode == "gap_elapsed"),
        )
        if time_features is None:
            time_features = np.zeros((len(params_raw), _time_feature_dim(time_feature_mode)), dtype=np.float32)
            time_feature_source = "missing_timestamps_zero_fill"
        else:
            time_feature_source = "timestamps"

    return WindowedParamSequenceDataset(
        params=params,
        mids=mids,
        history_len=cfg.history_len,
        stride=stride,
        params_mean=mu,
        params_std=sig,
        future_horizon=_future_horizon_from_cfg(cfg),
        cond=cond,
        cond_mean=c_mu,
        cond_std=c_sig,
        time_features=time_features,
        time_gap_scale=time_gap_scale,
        time_feature_source=time_feature_source,
        global_offset=0,
    )


def default_lobster_synth_profile_path() -> str:
    return DEFAULT_LOBSTER_SYNTH_PROFILE


def default_lobster_synthetic_profile_path_alias() -> str:
    return default_lobster_synth_profile_path()


def validate_lobster_synth_profile(profile: Dict[str, object], *, source: str = "") -> Dict[str, object]:
    profiles = profile.get("profiles", [])
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"LOBSTER synthetic profile at {source or '<memory>'} contains no regimes.")
    required = (
        "rows",
        "tick_size",
        "log_spread_mean",
        "log_spread_std",
        "spread_phi",
        "imb_mean",
        "imb_std",
        "imb_phi",
        "ret_scale_ticks",
        "jump_prob_5ticks",
        "jump_prob_2ticks",
        "seasonality_abs_ret",
        "log_ask_gap_mean",
        "log_ask_gap_std",
        "log_bid_gap_mean",
        "log_bid_gap_std",
        "log_ask_vol_mean",
        "log_ask_vol_std",
        "log_bid_vol_mean",
        "log_bid_vol_std",
    )
    for idx, regime in enumerate(profiles):
        if not isinstance(regime, dict):
            raise ValueError(f"LOBSTER profile regime {idx} must be an object.")
        missing = [key for key in required if key not in regime]
        if missing:
            raise ValueError(f"LOBSTER profile regime {idx} is missing {missing}.")
        if float(regime["tick_size"]) <= 0.0:
            raise ValueError(f"LOBSTER profile regime {idx} has non-positive tick_size.")
    return profile


@lru_cache(maxsize=None)
def load_lobster_synth_profile(path: Optional[str] = None) -> Dict[str, object]:
    resolved = path or default_lobster_synth_profile_path()
    with open(resolved, "r", encoding="utf-8") as f:
        profile = json.load(f)
    return validate_lobster_synth_profile(profile, source=str(resolved))


def download_lobster_synthetic_profile(
    path: str | os.PathLike[str] | None = None,
    *,
    url: str = LOBIFLOW_SYNTHETIC_PROFILE_URL,
    force: bool = False,
) -> Dict[str, object]:
    resolved = os.fspath(path or default_lobster_synth_profile_path())
    if force or not os.path.exists(resolved):
        os.makedirs(os.path.dirname(os.path.abspath(resolved)), exist_ok=True)
        with urllib.request.urlopen(str(url)) as response, open(resolved, "wb") as out_fh:
            out_fh.write(response.read())
        load_lobster_synth_profile.cache_clear()
    return load_lobster_synth_profile(resolved)


def _generate_synthetic_l2(
    levels: int,
    length: int,
    seed: int,
    eps: float = 1e-8,
    *,
    profile_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate LOBSTER-calibrated synthetic L2 data with persistent liquidity regimes."""
    rng = np.random.default_rng(seed)
    L = levels
    T = int(length)
    profile = load_lobster_synth_profile(profile_path)
    regimes = [
        regime for regime in profile["profiles"]
        if len(regime["log_ask_vol_mean"]) >= L and len(regime["log_bid_vol_mean"]) >= L
    ]
    if not regimes:
        raise ValueError(f"No LOBSTER regimes support levels={L}.")

    weights = np.asarray([math.sqrt(float(regime.get("rows", 1.0))) for regime in regimes], dtype=np.float64)
    weights = weights / weights.sum()
    tick_size = float(np.median([float(regime["tick_size"]) for regime in regimes]))
    level_weights = np.exp(-0.35 * np.arange(L, dtype=np.float64))

    ask_p = np.zeros((T, L), dtype=np.float32)
    bid_p = np.zeros((T, L), dtype=np.float32)
    ask_v = np.zeros((T, L), dtype=np.float32)
    bid_v = np.zeros((T, L), dtype=np.float32)
    mid_ticks = np.zeros(T, dtype=np.float64)

    current_regime = None
    spread_log = 0.0
    imbalance = 0.0
    ask_gap_state = np.zeros(max(0, L - 1), dtype=np.float64)
    bid_gap_state = np.zeros(max(0, L - 1), dtype=np.float64)
    ask_log_v_state = np.zeros(L, dtype=np.float64)
    bid_log_v_state = np.zeros(L, dtype=np.float64)
    segment_remaining = 0

    def _sample_segment_length(remaining: int) -> int:
        target = max(128.0, min(4096.0, float(max(1, T)) / 12.0))
        seg_len = int(rng.lognormal(mean=math.log(target), sigma=0.5))
        return min(remaining, max(128, seg_len))

    def _set_regime(regime: Dict[str, object], *, reset: bool) -> None:
        nonlocal current_regime, spread_log, imbalance, ask_gap_state, bid_gap_state, ask_log_v_state, bid_log_v_state
        current_regime = regime

        ask_gap_mu = np.asarray(regime["log_ask_gap_mean"][: max(0, L - 1)], dtype=np.float64)
        bid_gap_mu = np.asarray(regime["log_bid_gap_mean"][: max(0, L - 1)], dtype=np.float64)
        ask_log_v_mu = np.asarray(regime["log_ask_vol_mean"][:L], dtype=np.float64)
        bid_log_v_mu = np.asarray(regime["log_bid_vol_mean"][:L], dtype=np.float64)

        if reset:
            spread_log = float(regime["log_spread_mean"])
            imbalance = float(regime["imb_mean"])
            ask_gap_state = ask_gap_mu.copy()
            bid_gap_state = bid_gap_mu.copy()
            ask_log_v_state = ask_log_v_mu.copy()
            bid_log_v_state = bid_log_v_mu.copy()
            return

        spread_log = 0.5 * spread_log + 0.5 * float(regime["log_spread_mean"])
        imbalance = 0.5 * imbalance + 0.5 * float(regime["imb_mean"])
        ask_gap_state = 0.5 * ask_gap_state + 0.5 * ask_gap_mu
        bid_gap_state = 0.5 * bid_gap_state + 0.5 * bid_gap_mu
        ask_log_v_state = 0.5 * ask_log_v_state + 0.5 * ask_log_v_mu
        bid_log_v_state = 0.5 * bid_log_v_state + 0.5 * bid_log_v_mu

    for t in range(T):
        if segment_remaining <= 0:
            next_regime = regimes[int(rng.choice(len(regimes), p=weights))]
            _set_regime(next_regime, reset=(t == 0))
            segment_remaining = _sample_segment_length(T - t)
        segment_remaining -= 1

        assert current_regime is not None
        seasonality = current_regime["seasonality_abs_ret"]
        season_idx = min(len(seasonality) - 1, int(len(seasonality) * t / max(1, T)))
        season_scale = max(0.05, float(seasonality[season_idx]))

        spread_phi = float(np.clip(current_regime["spread_phi"], 0.7, 0.995))
        spread_std = max(0.05, float(current_regime["log_spread_std"]))
        spread_noise = spread_std * math.sqrt(max(1.0 - spread_phi ** 2, 1e-4))
        spread_log = (
            float(current_regime["log_spread_mean"])
            + spread_phi * (spread_log - float(current_regime["log_spread_mean"]))
            + spread_noise * rng.normal()
        )
        spread_ticks = max(1.0, round(math.exp(spread_log)))

        imb_phi = float(np.clip(current_regime["imb_phi"], 0.5, 0.995))
        imb_std = max(0.02, float(current_regime["imb_std"]))
        imb_noise = imb_std * math.sqrt(max(1.0 - imb_phi ** 2, 1e-4))
        imbalance = (
            float(current_regime["imb_mean"])
            + imb_phi * (imbalance - float(current_regime["imb_mean"]))
            + imb_noise * rng.normal()
        )
        imbalance = float(np.clip(imbalance, -0.98, 0.98))

        ret_scale = max(0.02, float(current_regime["ret_scale_ticks"])) * math.sqrt(season_scale)
        shock = rng.standard_t(df=5) / math.sqrt(5.0 / 3.0)
        ret_ticks = 0.12 * imbalance + 0.45 * ret_scale * shock
        if rng.random() < float(current_regime["jump_prob_5ticks"]):
            ret_ticks += float(rng.choice((-1.0, 1.0)) * (5.0 + abs(rng.normal(scale=1.5))))
        elif rng.random() < float(current_regime["jump_prob_2ticks"]):
            ret_ticks += float(rng.choice((-1.0, 1.0)) * (2.0 + abs(rng.normal(scale=0.75))))
        if t == 0:
            mid_ticks[t] = round(100.0 / tick_size)
        else:
            mid_ticks[t] = mid_ticks[t - 1] + ret_ticks

        if L > 1:
            gap_rho = 0.9
            ask_gap_mu = np.asarray(current_regime["log_ask_gap_mean"][: L - 1], dtype=np.float64)
            ask_gap_std = np.asarray(current_regime["log_ask_gap_std"][: L - 1], dtype=np.float64)
            bid_gap_mu = np.asarray(current_regime["log_bid_gap_mean"][: L - 1], dtype=np.float64)
            bid_gap_std = np.asarray(current_regime["log_bid_gap_std"][: L - 1], dtype=np.float64)
            gap_scale = math.sqrt(max(1.0 - gap_rho ** 2, 1e-4))
            ask_gap_state = ask_gap_mu + gap_rho * (ask_gap_state - ask_gap_mu) + gap_scale * ask_gap_std * rng.normal(size=L - 1)
            bid_gap_state = bid_gap_mu + gap_rho * (bid_gap_state - bid_gap_mu) + gap_scale * bid_gap_std * rng.normal(size=L - 1)
            ask_gap_ticks = np.maximum(1.0, np.round(np.exp(ask_gap_state)))
            bid_gap_ticks = np.maximum(1.0, np.round(np.exp(bid_gap_state)))
        else:
            ask_gap_ticks = np.empty(0, dtype=np.float64)
            bid_gap_ticks = np.empty(0, dtype=np.float64)

        vol_rho = 0.97
        ask_log_v_mu = np.asarray(current_regime["log_ask_vol_mean"][:L], dtype=np.float64)
        ask_log_v_std = np.asarray(current_regime["log_ask_vol_std"][:L], dtype=np.float64)
        bid_log_v_mu = np.asarray(current_regime["log_bid_vol_mean"][:L], dtype=np.float64)
        bid_log_v_std = np.asarray(current_regime["log_bid_vol_std"][:L], dtype=np.float64)
        vol_scale = math.sqrt(max(1.0 - vol_rho ** 2, 1e-4))
        ask_log_v_state = ask_log_v_mu + vol_rho * (ask_log_v_state - ask_log_v_mu) + vol_scale * ask_log_v_std * rng.normal(size=L)
        bid_log_v_state = bid_log_v_mu + vol_rho * (bid_log_v_state - bid_log_v_mu) + vol_scale * bid_log_v_std * rng.normal(size=L)

        imbalance_tilt = 0.35 * imbalance * level_weights
        ask_v_row = np.exp(np.clip(ask_log_v_state - imbalance_tilt, math.log(eps), 16.0))
        bid_v_row = np.exp(np.clip(bid_log_v_state + imbalance_tilt, math.log(eps), 16.0))

        mid_price = mid_ticks[t] * tick_size
        ask_p[t, 0] = mid_price + 0.5 * spread_ticks * tick_size
        bid_p[t, 0] = mid_price - 0.5 * spread_ticks * tick_size
        for i in range(1, L):
            ask_p[t, i] = ask_p[t, i - 1] + ask_gap_ticks[i - 1] * tick_size
            bid_p[t, i] = bid_p[t, i - 1] - bid_gap_ticks[i - 1] * tick_size

        ask_v[t] = ask_v_row.astype(np.float32)
        bid_v[t] = bid_v_row.astype(np.float32)

    return ask_p, ask_v.astype(np.float32), bid_p, bid_v.astype(np.float32)


def build_dataset_synthetic(
    cfg: OTFlowConfig,
    length: int = DEFAULT_SYNTHETIC_LENGTH,
    seed: int = 0,
    stride: int = 1,
) -> WindowedParamSequenceDataset:
    ask_p, ask_v, bid_p, bid_v = _generate_synthetic_l2(cfg.levels, length, seed, cfg.eps)
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)
    return _build_windowed_dataset(params_raw, mids, cfg, stride=stride)


# -----------------------------
# Split-aware builders (NEW)
# -----------------------------
def _resolve_split_bounds(
    T: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Tuple[int, int]:
    """Return (train_end, val_end) as absolute timestep boundaries in [0, T].

    Splits are interpreted over raw timesteps (params rows).
    """
    if test_frac is None:
        test_frac = 1.0 - train_frac - val_frac

    if train_end is None or val_end is None:
        if train_frac <= 0 or val_frac < 0 or test_frac < 0:
            raise ValueError("Invalid split fractions.")
        s = train_frac + val_frac + test_frac
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {s:.6f}")
        train_end = int(round(T * train_frac))
        val_end = int(round(T * (train_frac + val_frac)))

    train_end = int(train_end)
    val_end = int(val_end)

    if not (0 < train_end < val_end <= T):
        raise ValueError(f"Invalid split bounds: train_end={train_end}, val_end={val_end}, T={T}")

    return train_end, val_end


def _slice_segment_with_history(
    arr: np.ndarray,
    start_t: int,
    end_t: int,
    history_len: int,
) -> Tuple[np.ndarray, int]:
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
    return arr_seg, left


def _segment_starts_from_ends(segment_ends: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.asarray(segment_ends, dtype=np.int64)[:-1])).astype(np.int64)


def _resolve_segment_split_bounds(
    T: int,
    segment_ends: np.ndarray,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: Optional[float],
    train_end: Optional[int],
    val_end: Optional[int],
) -> Tuple[int, int]:
    segment_ends = np.asarray(segment_ends, dtype=np.int64)
    if len(segment_ends) < 3:
        raise ValueError("Need at least 3 segments for train/val/test splits.")
    if int(segment_ends[-1]) != int(T):
        raise ValueError("segment_ends must terminate at T.")

    if test_frac is None:
        test_frac = 1.0 - train_frac - val_frac

    if train_end is None or val_end is None:
        s = train_frac + val_frac + test_frac
        if abs(s - 1.0) > 1e-6:
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

    if not (0 < train_end < val_end <= T):
        raise ValueError(f"Invalid segment split bounds: train_end={train_end}, val_end={val_end}, T={T}")
    return int(train_end), int(val_end)


def _make_windowed_dataset_from_arrays(
    params_full: np.ndarray,
    mids_full: np.ndarray,
    cfg: OTFlowConfig,
    *,
    stride: int,
    start_t: int,
    end_t: int,
    params_mean: Optional[np.ndarray],
    params_std: Optional[np.ndarray],
    cond_full: Optional[np.ndarray],
    cond_mean: Optional[np.ndarray],
    cond_std: Optional[np.ndarray],
    time_features_full: Optional[np.ndarray],
    time_gap_scale: Optional[float],
    time_feature_source: str = "none",
    segment_ends_full: Optional[np.ndarray] = None,
    valid_start_mask_full: Optional[np.ndarray] = None,
    dataset_kind: Optional[str] = None,
    dataset_metadata: Optional[Dict[str, object]] = None,
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

    # Apply pre-fit stats (or keep raw if disabled)
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

    # Restrict targets to exactly [start_t, end_t) in GLOBAL time
    # local target t corresponds to global_offset + t
    g = ds.global_offset + ds.start_indices
    mask = (g >= int(start_t)) & (g < int(end_t))
    ds.start_indices = ds.start_indices[mask]

    if len(ds.start_indices) == 0:
        raise ValueError(
            f"Empty split dataset: start_t={start_t}, end_t={end_t}, "
            f"H={cfg.history_len}, stride={stride}. Increase segment length or reduce history_len."
        )
    return ds


def build_dataset_splits_from_arrays(
    params_raw: np.ndarray,
    mids: np.ndarray,
    cfg: OTFlowConfig,
    *,
    timestamps: Optional[np.ndarray] = None,
    cond_raw_full: Optional[np.ndarray] = None,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
    segment_ends: Optional[np.ndarray] = None,
    valid_start_mask: Optional[np.ndarray] = None,
    dataset_kind: Optional[str] = None,
    dataset_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
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
            T,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            train_end=train_end,
            val_end=val_end,
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

    # Train-only fit stats
    if cfg.standardize:
        p_mu, p_sig = fit_standardizer(params_raw[:train_end])
    else:
        p_mu = p_sig = None

    resolved_cond_raw_full = None if cond_raw_full is None else np.asarray(cond_raw_full, dtype=np.float32)
    c_mu = c_sig = None
    if resolved_cond_raw_full is None and cfg.use_cond_features:
        resolved_cond_raw_full = build_cond_features(params_raw, mids, cfg)
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

    # Build split datasets (each with left history buffer)
    ds_train = _make_windowed_dataset_from_arrays(
        params_full=params_raw,
        mids_full=mids,
        cfg=cfg,
        stride=stride_train,
        start_t=cfg.history_len,  # first valid target with full history
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



def build_dataset_splits_from_npz_l2(
    path: str,
    cfg: OTFlowConfig,
    *,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    """Chronological split for a *preprocessed* standardized L2 NPZ file.

    For datasets that are not off-the-shelf (exchange dumps, Kaggle files, etc.),
    first convert them to the standardized NPZ using `lob_prepare_dataset.py`.
    """
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    data = load_l2_npz(path)

    if "params_raw" in data and "mids" in data:
        params_raw = data["params_raw"]
        mids = data["mids"]
    else:
        for k in ("ask_p", "ask_v", "bid_p", "bid_v"):
            if k not in data:
                raise ValueError(f"NPZ missing required key '{k}'.")
        ask_p, ask_v, bid_p, bid_v = data["ask_p"], data["ask_v"], data["bid_p"], data["bid_v"]
        if ask_p.shape[1] != cfg.levels:
            raise ValueError(f"Levels mismatch: file L={ask_p.shape[1]}, cfg.levels={cfg.levels}")
        params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    return build_dataset_splits_from_arrays(
        params_raw=params_raw,
        mids=mids,
        cfg=cfg,
        timestamps=_timestamps_from_loaded_npz(data),
        stride_train=stride_train,
        stride_eval=stride_eval,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
        segment_ends=data.get("segment_ends"),
    )


def default_cryptos_npz_path() -> str:
    return DEFAULT_CRYPTOS_NPZ


def _download_url_to_path(url: str, destination: str | os.PathLike[str]) -> str:
    resolved = os.fspath(destination)
    os.makedirs(os.path.dirname(os.path.abspath(resolved)), exist_ok=True)
    with urllib.request.urlopen(str(url)) as response, open(resolved, "wb") as out_fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_fh.write(chunk)
    return resolved


def download_cryptos_npz(
    path: str | os.PathLike[str] | None = None,
    *,
    url: str = LOBIFLOW_CRYPTOS_NPZ_URL,
    force: bool = False,
) -> str:
    resolved = os.fspath(path or default_cryptos_npz_path())
    if force or not os.path.exists(resolved):
        _download_url_to_path(str(url), resolved)
    return resolved


def default_es_mbp_10_npz_path() -> str:
    return DEFAULT_ES_MBP_10_NPZ


def default_optiver_npz_path() -> str:
    return DEFAULT_OPTIVER_NPZ


def build_dataset_splits_from_cryptos(
    path: str,
    cfg: OTFlowConfig,
    *,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    """Named dataset helper for the prepared Tardis crypto L2 archive."""
    resolved_path = path or default_cryptos_npz_path()
    if not os.path.exists(resolved_path):
        download_cryptos_npz(resolved_path)
    return build_dataset_splits_from_npz_l2(
        path=resolved_path,
        cfg=cfg,
        stride_train=stride_train,
        stride_eval=stride_eval,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
    )


def build_dataset_splits_from_es_mbp_10(
    path: str,
    cfg: OTFlowConfig,
    *,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    """Named dataset helper for the prepared Databento ES MBP-10 archive."""
    resolved_path = path or default_es_mbp_10_npz_path()
    return build_dataset_splits_from_npz_l2(
        path=resolved_path,
        cfg=cfg,
        stride_train=stride_train,
        stride_eval=stride_eval,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
    )


def build_dataset_splits_from_optiver(
    path: str,
    cfg: OTFlowConfig,
    *,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    """Named dataset helper for the prepared Optiver L2 archive."""
    resolved_path = path or default_optiver_npz_path()
    return build_dataset_splits_from_npz_l2(
        path=resolved_path,
        cfg=cfg,
        stride_train=stride_train,
        stride_eval=stride_eval,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
    )


def build_dataset_splits_synthetic(
    cfg: OTFlowConfig,
    length: int = DEFAULT_SYNTHETIC_LENGTH,
    seed: int = 0,
    *,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    """LOBSTER-calibrated synthetic chronological split with train-only normalization."""
    ask_p, ask_v, bid_p, bid_v = _generate_synthetic_l2(cfg.levels, length, seed, cfg.eps)

    fm = L2FeatureMap(cfg.levels, cfg.eps)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    return build_dataset_splits_from_arrays(
        params_raw=params_raw,
        mids=mids,
        cfg=cfg,
        timestamps=None,
        stride_train=stride_train,
        stride_eval=stride_eval,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
    )


def build_dataset_splits_from_lobster_synthetic(
    profile_path: str,
    cfg: OTFlowConfig,
    *,
    length: int = DEFAULT_SYNTHETIC_LENGTH,
    seed: int = 0,
    stride_train: int = 1,
    stride_eval: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: Optional[float] = None,
    train_end: Optional[int] = None,
    val_end: Optional[int] = None,
) -> Dict[str, object]:
    resolved_profile = profile_path or default_lobster_synth_profile_path()
    if not os.path.exists(resolved_profile):
        download_lobster_synthetic_profile(resolved_profile)
    validate_lobster_synth_profile(load_lobster_synth_profile(resolved_profile), source=resolved_profile)
    ask_p, ask_v, bid_p, bid_v = _generate_synthetic_l2(
        cfg.levels,
        length,
        seed,
        cfg.eps,
        profile_path=resolved_profile,
    )
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)
    return build_dataset_splits_from_arrays(
        params_raw=params_raw,
        mids=mids,
        cfg=cfg,
        timestamps=None,
        stride_train=stride_train,
        stride_eval=stride_eval,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train_end=train_end,
        val_end=val_end,
        dataset_kind=LOBSTER_SYNTHETIC_DATASET_KEY,
        dataset_metadata={
            "dataset_key": LOBSTER_SYNTHETIC_DATASET_KEY,
            "source": "mpstoryfans/lobiflow data/synthetic/lobster_free_sample_profile_10.json",
            "profile_name": os.path.basename(os.fspath(resolved_profile)),
            "length": int(length),
            "seed": int(seed),
        },
    )


# -----------------------------
# Basic metrics (raw space) for quick checks
# -----------------------------
def compute_basic_l2_metrics(ask_p: np.ndarray, ask_v: np.ndarray, bid_p: np.ndarray, bid_v: np.ndarray) -> Dict[str, float]:
    spread = ask_p[:, 0] - bid_p[:, 0]
    depth = ask_v.sum(axis=1) + bid_v.sum(axis=1)
    imb = (bid_v.sum(axis=1) - ask_v.sum(axis=1)) / (depth + 1e-8)
    return {
        "spread_mean": float(np.mean(spread)),
        "spread_std": float(np.std(spread)),
        "depth_mean": float(np.mean(depth)),
        "imb_mean": float(np.mean(imb)),
        "imb_std": float(np.std(imb)),
    }


__all__ = [
    "L2FeatureMap",
    "WindowedParamSequenceDataset",
    "build_dataset_synthetic",
    "build_dataset_splits_from_arrays",
    "build_dataset_splits_from_lobster_synthetic",
    "build_dataset_splits_from_npz_l2",
    "build_dataset_splits_from_es_mbp_10",
    "build_dataset_splits_from_optiver",
    "build_dataset_splits_synthetic",
    "default_cryptos_npz_path",
    "download_cryptos_npz",
    "default_es_mbp_10_npz_path",
    "default_optiver_npz_path",
    "default_lobster_synth_profile_path",
    "default_lobster_synthetic_profile_path",
    "download_lobster_synthetic_profile",
    "load_lobster_synth_profile",
    "validate_lobster_synth_profile",
    "standardize_params",
    "standardize_cond",
    "load_l2_npz",
    "fit_standardizer",
    "apply_standardizer",
    "build_cond_features",
    "compute_basic_l2_metrics",
    "DEFAULT_SYNTHETIC_LENGTH",
    "LOBIFLOW_CRYPTOS_NPZ_URL",
    "LOBIFLOW_SYNTHETIC_PROFILE_URL",
    "LOBSTER_SYNTHETIC_DATASET_KEY",
]
