"""otflow_train_val.py

Training and evaluation utilities for OTFlow backbones.

Adds evaluation helpers:
- Real-vs-real comparison metrics (distribution + temporal + validity)
- Horizon-wise rollout stability
- NFE speed/quality benchmarking
- Generic ablation runner
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from contextlib import contextmanager
import copy
import hashlib
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel

from genode.models.config import OTFlowConfig
from genode.models.modules import EMAModel
from genode.models.otflow_model import OTFlow
from genode.data.otflow_datasets import L2FeatureMap, WindowedParamSequenceDataset, compute_basic_l2_metrics
from genode.data.otflow_medical_datasets import LONG_TERM_ST_DATASET_KEY
from genode.models.otflow_utils import flatten_dict, unflatten_to_nested, microstructure_series


SUPPORTED_MODEL_NAMES = ("otflow",)
CORE_L2_STATS = ("spread", "depth", "imb", "ret")


# -----------------------------
# Basics
# -----------------------------
_NUMPY_SEED_MODULUS = 2**32


def _bounded_numpy_seed(seed: int) -> int:
    seed_i = int(seed)
    if seed_i < 0:
        raise ValueError(f"seed must be non-negative, got {seed!r}")
    return int(seed_i % _NUMPY_SEED_MODULUS)


def seed_all(seed: int = 0):
    normalized_seed = _bounded_numpy_seed(seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    torch.cuda.manual_seed_all(normalized_seed)


def make_loader(
    ds: WindowedParamSequenceDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader with safe defaults for small split sizes.

    `drop_last=False` avoids creating an empty loader when len(ds) < batch_size,
    which would otherwise cause training loops to fail with repeated StopIteration.
    """
    sampler = None
    effective_shuffle = bool(shuffle)
    if bool(shuffle) and getattr(ds, "sampler_weights", None) is not None:
        weights = torch.as_tensor(getattr(ds, "sampler_weights"), dtype=torch.double)
        num_samples = getattr(ds, "sampler_num_samples", None)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=int(num_samples) if num_samples is not None else int(len(weights)),
            replacement=True,
        )
        effective_shuffle = False
    elif bool(shuffle) and bool(getattr(ds, "sampler_replacement", False)):
        num_samples = getattr(ds, "sampler_num_samples", None)
        if num_samples is not None and int(num_samples) > 0:
            sampler = RandomSampler(
                ds,
                replacement=True,
                num_samples=int(num_samples),
            )
            effective_shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def _parse_batch(batch):
    """Unpack the dataset tuple emitted by WindowedParamSequenceDataset.

    Supports:
      - (hist, tgt, meta)
      - (hist, tgt, cond, meta)
      - (hist, tgt, fut, meta)
      - (hist, tgt, fut, cond, meta)

    Batched future horizons are rank-3 tensors [B, H_fut, D], batched
    conditioning features are rank-2 tensors [B, C]. Unbatched examples use
    rank-2 futures [H_fut, D] and rank-1 conditions [C].
    """
    if len(batch) == 3:
        hist, tgt, meta = batch
        return hist, tgt, None, None, meta
    if len(batch) == 4:
        hist, tgt, a, meta = batch
        hist_rank = int(hist.dim())
        arg_rank = int(a.dim())
        if hist_rank == 3 and arg_rank == 2:
            return hist, tgt, None, a, meta
        if hist_rank == 3 and arg_rank == 3:
            return hist, tgt, a, None, meta
        if hist_rank == 2 and arg_rank == 1:
            return hist, tgt, None, a, meta
        if hist_rank == 2 and arg_rank == 2:
            return hist, tgt, a, None, meta
        raise ValueError(f"Unexpected 4-item batch tensor ranks: hist={hist_rank}, item={arg_rank}.")
    if len(batch) == 5:
        hist, tgt, fut, cond, meta = batch
        return hist, tgt, fut, cond, meta
    raise ValueError("Unexpected batch format.")


def _torch_sync(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _amp_enabled(cfg: OTFlowConfig, device: torch.device) -> bool:
    return bool(getattr(cfg.train, "use_amp", False)) and device.type == "cuda" and torch.cuda.is_available()


@contextmanager
def _autocast_context(cfg: OTFlowConfig, device: torch.device):
    if _amp_enabled(cfg, device):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            yield
        return
    yield


@contextmanager
def _temporary_eval_seed(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    normalized_seed = _bounded_numpy_seed(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    torch.cuda.manual_seed_all(normalized_seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def resolve_context_length(max_available: int, *, horizon: int, cfg: Optional[OTFlowConfig]) -> int:
    max_available = max(1, int(max_available))
    if cfg is None or not bool(getattr(cfg, "adaptive_context", False)):
        return max_available

    ratio = float(getattr(cfg, "adaptive_context_ratio", 1.5))
    ctx_min = int(getattr(cfg, "adaptive_context_min", 1))
    ctx_max = int(getattr(cfg, "adaptive_context_max", max_available))
    desired = int(round(max(1, int(horizon)) * ratio))
    desired = max(ctx_min, desired)
    desired = min(desired, ctx_max, max_available)
    return max(1, desired)


def crop_history_window(hist: torch.Tensor, context_len: int) -> torch.Tensor:
    context_len = max(1, int(context_len))
    if hist.dim() == 3:
        return hist[:, -context_len:, :]
    if hist.dim() == 2:
        return hist[-context_len:, :]
    raise ValueError(f"Unsupported history tensor rank: {hist.dim()}")


def sample_training_context_length(max_available: int, cfg: Optional[OTFlowConfig]) -> int:
    max_available = max(1, int(max_available))
    if cfg is None or not bool(getattr(cfg, "train_variable_context", False)):
        return max_available

    min_len = max(1, int(getattr(cfg, "train_context_min", 1)))
    max_len = int(getattr(cfg, "train_context_max", max_available))
    min_len = min(min_len, max_available)
    max_len = min(max(max_len, min_len), max_available)
    return int(np.random.randint(min_len, max_len + 1))


def _model_prediction_horizon(model: torch.nn.Module) -> int:
    model_cfg = getattr(model, "cfg", None)
    if model_cfg is None:
        return 1
    return int(max(1, int(getattr(model_cfg, "prediction_horizon", 1))))


def _model_snapshot_dim(model: torch.nn.Module, context_dim: int) -> int:
    model_cfg = getattr(model, "cfg", None)
    if model_cfg is None:
        return int(context_dim)
    return int(getattr(model_cfg, "snapshot_dim", int(context_dim)))


def _future_time_context_seq(ds, t0: int, horizon: int) -> Optional[torch.Tensor]:
    if hasattr(ds, "future_time_features"):
        features = ds.future_time_features(int(t0), int(horizon))
        if features is not None:
            return features
    if hasattr(ds, "future_time_gap_features"):
        features = ds.future_time_gap_features(int(t0), int(horizon))
        if features is not None:
            return features
    return None


# -----------------------------
# Training
# -----------------------------
def _build_scheduler(opt: torch.optim.Optimizer, cfg: OTFlowConfig, total_steps: int):
    """Build an optional LR scheduler (warmup + cosine decay)."""
    schedule = getattr(cfg, "lr_schedule", "constant").lower()
    warmup = int(getattr(cfg, "lr_warmup_steps", 0))
    if schedule == "constant" and warmup <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        if schedule == "cosine":
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.lower().strip()
    if normalized not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Only model_name='otflow' is supported, got {model_name!r}.")
    return normalized


def _validate_model_dataset_support(model_name: str, cfg: OTFlowConfig) -> None:
    del cfg
    _normalize_model_name(model_name)


def _build_model(model_name: str, cfg: OTFlowConfig, device: torch.device) -> torch.nn.Module:
    _validate_model_dataset_support(model_name, cfg)
    return OTFlow(cfg).to(device)


def _compute_training_loss(
    model: torch.nn.Module,
    *,
    tgt: torch.Tensor,
    hist: torch.Tensor,
    fut: Optional[torch.Tensor],
    cond: Optional[torch.Tensor],
    meta: Any,
    loss_mode: Optional[str],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if isinstance(model, OTFlow):
        return model.loss(tgt, hist, fut=fut, cond=cond, meta=meta)

    del loss_mode
    raise RuntimeError("Unexpected model type; OTFlow is the only supported model.")


@torch.no_grad()
def evaluate_average_loss(
    ds: WindowedParamSequenceDataset,
    model: torch.nn.Module,
    cfg: OTFlowConfig,
    *,
    model_name: str = "otflow",
    max_batches: Optional[int] = None,
    loss_mode: Optional[str] = None,
    shuffle: bool = False,
) -> Dict[str, Any]:
    """Evaluate the mean training objective on a dataset split."""
    _normalize_model_name(model_name)
    device = cfg.device
    loader = make_loader(ds, cfg.batch_size, shuffle=shuffle, drop_last=False)
    if len(loader) == 0:
        raise ValueError("Evaluation loader is empty.")

    was_training = bool(model.training)
    model.eval()
    total_loss = 0.0
    total_examples = 0
    batches = 0
    try:
        for batch in loader:
            hist, tgt, fut, cond, meta = _parse_batch(batch)
            hist = hist.to(device).float()
            tgt = tgt.to(device).float()
            fut = fut.to(device).float() if fut is not None else None
            cond = cond.to(device).float() if cond is not None else None

            context_len = resolve_context_length(hist.shape[1], horizon=_model_prediction_horizon(model), cfg=cfg)
            hist = crop_history_window(hist, context_len)
            loss, logs = _compute_training_loss(
                model,
                tgt=tgt,
                hist=hist,
                fut=fut,
                cond=cond,
                meta=meta,
                loss_mode=loss_mode,
            )
            batch_size = int(hist.shape[0])
            loss_value = float(logs.get("loss", float(loss.detach())))
            total_loss += loss_value * float(batch_size)
            total_examples += batch_size
            batches += 1
            if max_batches is not None and batches >= int(max_batches):
                break
    finally:
        if was_training:
            model.train()

    if total_examples <= 0:
        raise ValueError("Evaluation produced no examples.")
    return {
        "loss": float(total_loss / float(total_examples)),
        "examples": int(total_examples),
        "batches": int(batches),
    }


def train_loop(
    ds: WindowedParamSequenceDataset,
    cfg: OTFlowConfig,
    model_name: str = "otflow",
    steps: int = 10_000,
    log_every: int = 200,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_mode: Optional[str] = None,
    shuffle: bool = True,
    on_step: Optional[Callable[[int, torch.nn.Module, float, Dict[str, float]], None]] = None,
) -> torch.nn.Module:
    """Train a model on next-step prediction in normalized param space.

    Features:
    - EMA model averaging (cfg.ema_decay > 0)
    - LR warmup + cosine decay (cfg.lr_schedule, cfg.lr_warmup_steps)
    """
    device = cfg.device
    loader = make_loader(ds, cfg.batch_size, shuffle=shuffle, drop_last=False)
    if len(loader) == 0:
        raise ValueError(
            "Training loader is empty. Check dataset construction, history_len, "
            "split boundaries, and batch_size."
        )

    model_name = _normalize_model_name(model_name)
    _validate_model_dataset_support(model_name, cfg)
    if model is None:
        model = _build_model(model_name, cfg, device)
    else:
        model = model.to(device)

    opt = optimizer or torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    accum_steps = max(1, int(getattr(cfg.train, "grad_accum_steps", 1)))
    use_amp = _amp_enabled(cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # LR scheduler
    scheduler = _build_scheduler(opt, cfg, steps)

    # EMA
    ema_decay = float(getattr(cfg, "ema_decay", 0.0))
    ema = EMAModel(model, decay=ema_decay) if ema_decay > 0 else None

    # SWA
    use_swa = getattr(cfg, "use_swa", False)
    swa_model = AveragedModel(model) if use_swa else None
    swa_start = int(0.75 * steps)

    model.train()
    it = iter(loader)
    opt.zero_grad(set_to_none=True)
    opt_step = 0
    micro_step = 0
    while opt_step < int(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        hist, tgt, fut, cond, meta = _parse_batch(batch)
        hist = hist.to(device).float()
        tgt = tgt.to(device).float()
        fut = fut.to(device).float() if fut is not None else None
        cond = cond.to(device).float() if cond is not None else None

        train_context_len = sample_training_context_length(hist.shape[1], cfg)
        hist = crop_history_window(hist, train_context_len)

        with _autocast_context(cfg, device):
            loss, logs = _compute_training_loss(
                model,
                tgt=tgt,
                hist=hist,
                fut=fut,
                cond=cond,
                meta=meta,
                loss_mode=loss_mode,
            )

        micro_step += 1
        loss_for_backward = loss / float(accum_steps)
        scaler.scale(loss_for_backward).backward()

        if micro_step % accum_steps != 0:
            continue

        opt_step += 1
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        if swa_model is not None and opt_step >= swa_start:
            swa_model.update_parameters(model)

        latest_train_loss = float(logs.get("loss", float(loss.detach())))
        if on_step is not None:
            on_step(int(opt_step), model, latest_train_loss, dict(logs))

        if opt_step % log_every == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(
                f"[{model_name}] step {opt_step}/{steps}  "
                f"loss={latest_train_loss:.4f}  "
                f"lr={lr_now:.2e}  details={logs}"
            )

    # Apply SWA weights if used
    if swa_model is not None:
        print(f"[{model_name}] Applying SWA weights tracked over the last {steps - swa_start + 1} steps")
        model.load_state_dict(swa_model.module.state_dict())
    # Apply EMA weights for evaluation (overrides SWA if both are enabled, but usually one is chosen)
    elif ema is not None:
        ema.apply_shadow(model)

    return model.eval()


# -----------------------------
# Generation
# -----------------------------
@torch.no_grad()
def generate_continuation(
    model: torch.nn.Module,
    hist: torch.Tensor,
    cond_seq: Optional[torch.Tensor],
    steps: int,
    nfe: int,
    future_context_seq: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Continuation in normalized param space.

    Autoregressive models sample one next state per solve. Non-autoregressive
    OTFlow models sample a future block and advance the rollout by that block.
    """
    B, H, D = hist.shape
    model_cfg = getattr(model, "cfg", None)
    context_len = resolve_context_length(H, horizon=steps, cfg=model_cfg)
    x_hist = crop_history_window(hist, context_len).clone()
    out = []

    snapshot_dim = _model_snapshot_dim(model, context_dim=D)
    extra_dim = max(0, int(D) - int(snapshot_dim))
    if extra_dim > 0 and future_context_seq is None:
        uses_time_context = bool(
            getattr(model_cfg, "use_time_features", False)
            or getattr(model_cfg, "use_time_gaps", False)
        )
        if not uses_time_context:
            raise ValueError(
                "Continuation with context_dim > snapshot_dim requires explicit future_context_seq. "
                "Use a domain-specific rollout path for non-temporal augmented contexts."
            )
    if extra_dim > 0 and future_context_seq is not None:
        if future_context_seq.shape[0] != B or future_context_seq.shape[1] < int(steps) or future_context_seq.shape[2] != extra_dim:
            raise ValueError(
                "future_context_seq must have shape [B, steps, context_extra_dim] "
                f"with B={B}, steps>={int(steps)}, context_extra_dim={extra_dim}; "
                f"got {tuple(future_context_seq.shape)}."
            )

    def _append_context_features(block: torch.Tensor, cursor: int, take: int) -> torch.Tensor:
        if extra_dim <= 0:
            return block
        if future_context_seq is None:
            raise ValueError("future_context_seq is required when context_dim exceeds snapshot_dim.")
        extra = future_context_seq[:, int(cursor) : int(cursor) + int(take), :].to(
            device=block.device,
            dtype=block.dtype,
        )
        return torch.cat([block, extra], dim=-1)

    prediction_horizon = _model_prediction_horizon(model)
    if prediction_horizon > 1:
        cursor = 0
        while cursor < int(steps):
            cond_t = cond_seq[:, cursor, :] if cond_seq is not None else None
            if isinstance(model, OTFlow):
                if not hasattr(model, "sample_future"):
                    raise RuntimeError("Non-autoregressive OTFlow requires sample_future(...).")
                x_block = model.sample_future(x_hist, cond=cond_t, steps=nfe)
            else:
                raise RuntimeError("Non-autoregressive continuation is currently implemented for OTFlow only.")

            take = min(int(prediction_horizon), int(steps) - int(cursor))
            block_slice = x_block[:, :take, :]
            out.append(block_slice)
            hist_block = _append_context_features(block_slice, cursor=cursor, take=take)
            x_hist = torch.cat([x_hist, hist_block], dim=1)
            x_hist = crop_history_window(x_hist, context_len)
            cursor += int(take)
        return torch.cat(out, dim=1)

    for k in range(steps):
        cond_t = cond_seq[:, k, :] if cond_seq is not None else None

        if isinstance(model, OTFlow):
            x_next = model.sample(x_hist, cond=cond_t, steps=nfe)
        else:
            raise RuntimeError("Generation is implemented for OTFlow only.")

        out.append(x_next[:, None, :])
        hist_step = _append_context_features(x_next[:, None, :], cursor=k, take=1)
        x_hist = torch.cat([x_hist, hist_step], dim=1)
        x_hist = crop_history_window(x_hist, context_len)

    return torch.cat(out, dim=1)  # [B,steps,D]


# -----------------------------
# Evaluation metrics (raw + params)
# -----------------------------
def _safe_mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a))}


def _ks_stat(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = np.sort(x)
    y = np.sort(y)
    z = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, z, side="right") / float(x.size)
    cdf_y = np.searchsorted(y, z, side="right") / float(y.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Quantile approximation (no SciPy dependency)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return float("nan")
    q = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return float(np.mean(np.abs(xq - yq)))


def _acf(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 3:
        return np.zeros(max_lag, dtype=np.float64)
    x = x - np.mean(x)
    var = np.var(x) + 1e-12
    out = np.zeros(max_lag, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        if lag >= x.size:
            out[lag - 1] = 0.0
        else:
            out[lag - 1] = float(np.mean(x[:-lag] * x[lag:]) / var)
    return out


def _rolling_volatility(x: np.ndarray, window: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    w = max(2, int(window))
    out = np.zeros_like(x, dtype=np.float64)
    for t in range(x.size):
        s = max(0, t - w + 1)
        out[t] = float(np.std(x[s : t + 1]))
    return out


def _normalized_mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return float("nan")
    scale = float(np.std(y[:n]) + 1e-6)
    return float(np.mean(np.abs(x[:n] - y[:n])) / scale)


def _hist_l1(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float("nan")
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, int(bins) + 1, dtype=np.float64)
    hx, _ = np.histogram(x, bins=edges)
    hy, _ = np.histogram(y, bins=edges)
    px = hx.astype(np.float64) / max(1.0, float(hx.sum()))
    py = hy.astype(np.float64) / max(1.0, float(hy.sum()))
    return float(np.sum(np.abs(px - py)))


def _impact_response_curve(imb: np.ndarray, ret: np.ndarray, lags: Sequence[int] = (1, 5, 10, 20)) -> Dict[str, float]:
    imb = np.asarray(imb, dtype=np.float64).ravel()
    ret = np.asarray(ret, dtype=np.float64).ravel()
    out: Dict[str, float] = {}
    if imb.size == 0 or ret.size == 0:
        return {str(int(lag)): float("nan") for lag in lags}
    ret_csum = np.concatenate(([0.0], np.cumsum(ret, dtype=np.float64)))
    for lag in lags:
        hh = int(lag)
        if hh <= 0 or ret.size <= hh:
            out[str(hh)] = float("nan")
            continue
        future_ret = ret_csum[1 + hh :] - ret_csum[1 : ret.size - hh + 1]
        drive = imb[: ret.size - hh]
        out[str(hh)] = float(np.mean(drive * future_ret))
    return out


def _validity_metrics(ask_p: np.ndarray, ask_v: np.ndarray, bid_p: np.ndarray, bid_v: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    crossed = (ask_p[:, 0] <= bid_p[:, 0]).astype(np.float32)
    ask_monotonic_bad = (np.diff(ask_p, axis=1) <= 0).any(axis=1).astype(np.float32)
    bid_monotonic_bad = (np.diff(bid_p, axis=1) >= 0).any(axis=1).astype(np.float32)
    nonpos_vol = ((ask_v <= eps).any(axis=1) | (bid_v <= eps).any(axis=1)).astype(np.float32)
    invalid_any = np.clip(crossed + ask_monotonic_bad + bid_monotonic_bad + nonpos_vol, 0, 1)
    return {
        "valid_rate": float(1.0 - invalid_any.mean()),
        "crossed_rate": float(crossed.mean()),
        "ask_monotonic_violation_rate": float(ask_monotonic_bad.mean()),
        "bid_monotonic_violation_rate": float(bid_monotonic_bad.mean()),
        "nonpositive_volume_rate": float(nonpos_vol.mean()),
    }


class _SmallMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _downstream_device(cfg: OTFlowConfig) -> torch.device:
    device = getattr(cfg, "device", torch.device("cpu"))
    if isinstance(device, str):
        return torch.device(device)
    return device


def _standardize_pair(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    sig = (train_x.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return ((train_x - mu) / sig).astype(np.float32), ((test_x - mu) / sig).astype(np.float32)


def _macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0:
        return float("nan")

    f1s = []
    for cls in range(int(num_classes)):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        denom = 2.0 * tp + fp + fn
        f1s.append(0.0 if denom <= 0 else (2.0 * tp) / denom)
    return float(np.mean(f1s))


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[y_true == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def _train_small_multiclass_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    hidden_dim: int = 64,
    epochs: int = 12,
    batch_size: int = 512,
    num_classes: Optional[int] = None,
) -> Tuple[torch.nn.Module, int]:
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int64)
    if len(train_x) == 0:
        raise ValueError("Cannot train stage classifier on an empty feature matrix.")
    if np.unique(train_y).size < 2:
        raise ValueError("Stage classifier requires at least two training classes.")
    class_count = int(
        num_classes
        if num_classes is not None
        else int(np.max(train_y)) + 1
    )
    if class_count <= 1:
        raise ValueError(f"Stage classifier requires at least two classes, got {class_count}.")

    torch.manual_seed(seed)
    model = _SmallMLP(train_x.shape[1], class_count, hidden_dim=hidden_dim).to(device)
    counts = np.bincount(train_y, minlength=class_count).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1.0), 0.0)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x_train = torch.from_numpy(train_x).to(device)
    y_train = torch.from_numpy(train_y).to(device)
    for _ in range(int(epochs)):
        perm = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            logits = model(x_train.index_select(0, idx))
            loss = loss_fn(logits, y_train.index_select(0, idx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    return model, class_count


def _predict_small_multiclass_mlp(model: torch.nn.Module, x: np.ndarray, *, device: torch.device) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return np.zeros(0, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).argmax(dim=1).cpu().numpy()
    return np.asarray(pred, dtype=np.int64)


def _train_small_multiclass_mlp_f1(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    hidden_dim: int = 64,
    epochs: int = 12,
    batch_size: int = 512,
    num_classes: Optional[int] = None,
) -> float:
    test_x = np.asarray(test_x, dtype=np.float32)
    test_y = np.asarray(test_y, dtype=np.int64)
    if len(test_x) == 0:
        return float("nan")
    try:
        model, class_count = _train_small_multiclass_mlp(
            train_x,
            train_y,
            device=device,
            seed=seed,
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            num_classes=num_classes,
        )
    except ValueError:
        return float("nan")
    pred = _predict_small_multiclass_mlp(model, test_x, device=device)
    return _macro_f1_score(test_y, pred, num_classes=class_count)


def _train_small_discriminator_auc(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    hidden_dim: int = 64,
    epochs: int = 10,
    batch_size: int = 512,
) -> float:
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    test_y = np.asarray(test_y, dtype=np.int64)
    if len(train_x) == 0 or len(test_x) == 0:
        return float("nan")
    if np.unique(train_y).size < 2 or np.unique(test_y).size < 2:
        return float("nan")

    torch.manual_seed(seed)
    model = _SmallMLP(train_x.shape[1], 1, hidden_dim=hidden_dim).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x_train = torch.from_numpy(train_x).to(device)
    y_train = torch.from_numpy(train_y[:, None]).to(device)
    x_test = torch.from_numpy(test_x).to(device)

    for _ in range(int(epochs)):
        perm = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            logits = model(x_train.index_select(0, idx))
            loss = loss_fn(logits, y_train.index_select(0, idx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_test)).squeeze(1).cpu().numpy()
    return _binary_auc(test_y, probs)


def _future_moves_from_params(params_raw: np.ndarray, label_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    params_raw = np.asarray(params_raw, dtype=np.float32)
    T = int(len(params_raw))
    hh = int(label_horizon)
    if hh <= 0 or T <= hh:
        return np.zeros((0, params_raw.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.float32)

    delta_mid = params_raw[:, 0].astype(np.float64)
    csum = np.concatenate(([0.0], np.cumsum(delta_mid, dtype=np.float64)))
    idx = np.arange(0, T - hh, dtype=np.int64)
    moves = csum[idx + hh + 1] - csum[idx + 1]
    feats = params_raw[: T - hh].astype(np.float32)
    return feats, moves.astype(np.float32)


def _ternary_labels(moves: np.ndarray, threshold: float) -> np.ndarray:
    moves = np.asarray(moves, dtype=np.float32)
    thr = float(max(threshold, 1e-8))
    labels = np.full(moves.shape, 1, dtype=np.int64)
    labels[moves > thr] = 2
    labels[moves < -thr] = 0
    return labels


def _subsample_examples(x: np.ndarray, y: np.ndarray, max_examples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= int(max_examples):
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=int(max_examples), replace=False)
    return x[idx], y[idx]


def _collect_downstream_examples(
    rows: Sequence[Dict[str, Any]],
    *,
    label_horizon: int,
    max_examples_per_split: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    real_x = []
    real_moves = []
    gen_x = []
    gen_moves = []

    for row in rows:
        seq = row["seq"]
        gx, gm = _future_moves_from_params(seq["gen_params_raw"], label_horizon)
        rx, rm = _future_moves_from_params(seq["true_params_raw"], label_horizon)
        if len(gx) == 0 or len(rx) == 0:
            continue
        gen_x.append(gx)
        gen_moves.append(gm)
        real_x.append(rx)
        real_moves.append(rm)

    if not real_x or not gen_x:
        empty_x = np.zeros((0, 1), dtype=np.float32)
        empty_y = np.zeros(0, dtype=np.float32)
        return {
            "real_x": empty_x,
            "real_moves": empty_y,
            "gen_x": empty_x,
            "gen_moves": empty_y,
        }

    real_x_arr = np.concatenate(real_x, axis=0).astype(np.float32)
    real_moves_arr = np.concatenate(real_moves, axis=0).astype(np.float32)
    gen_x_arr = np.concatenate(gen_x, axis=0).astype(np.float32)
    gen_moves_arr = np.concatenate(gen_moves, axis=0).astype(np.float32)

    real_x_arr, real_moves_arr = _subsample_examples(real_x_arr, real_moves_arr, max_examples_per_split, seed + 1)
    gen_x_arr, gen_moves_arr = _subsample_examples(gen_x_arr, gen_moves_arr, max_examples_per_split, seed + 2)
    return {
        "real_x": real_x_arr,
        "real_moves": real_moves_arr,
        "gen_x": gen_x_arr,
        "gen_moves": gen_moves_arr,
    }


def _pairwise_split(
    real_x: np.ndarray,
    gen_x: np.ndarray,
    *,
    seed: int,
    train_frac: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(min(len(real_x), len(gen_x)))
    if n <= 1:
        empty = np.zeros((0, real_x.shape[1] if real_x.ndim == 2 else gen_x.shape[1]), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.int64), empty, np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = max(1, int(round(n * train_frac)))
    n_train = min(n_train, n - 1)
    tr = idx[:n_train]
    te = idx[n_train:]
    train_x = np.concatenate([real_x[tr], gen_x[tr]], axis=0).astype(np.float32)
    train_y = np.concatenate([np.ones(len(tr), dtype=np.int64), np.zeros(len(tr), dtype=np.int64)], axis=0)
    test_x = np.concatenate([real_x[te], gen_x[te]], axis=0).astype(np.float32)
    test_y = np.concatenate([np.ones(len(te), dtype=np.int64), np.zeros(len(te), dtype=np.int64)], axis=0)
    return train_x, train_y, test_x, test_y


def _aggregate_core_l2_distribution_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pooled_gen = {k: [] for k in CORE_L2_STATS}
    pooled_true = {k: [] for k in CORE_L2_STATS}
    per_window = {k: [] for k in CORE_L2_STATS}
    per_window_l1 = {k: [] for k in CORE_L2_STATS}

    for row in rows:
        seq = row["seq"]
        sg = microstructure_series(
            seq["gen"]["ask_p"], seq["gen"]["ask_v"], seq["gen"]["bid_p"], seq["gen"]["bid_v"]
        )
        st = microstructure_series(
            seq["true"]["ask_p"], seq["true"]["ask_v"], seq["true"]["bid_p"], seq["true"]["bid_v"]
        )
        for key in CORE_L2_STATS:
            sg_arr = np.asarray(sg[key], dtype=np.float64)
            st_arr = np.asarray(st[key], dtype=np.float64)
            pooled_gen[key].append(sg_arr)
            pooled_true[key].append(st_arr)
            per_window[key].append(_wasserstein_1d(sg[key], st[key]))
            per_window_l1[key].append(_normalized_mae(sg_arr, st_arr))

    unconditional_by_stat = {}
    conditional_by_stat = {}
    unconditional_l1_by_stat = {}
    conditional_l1_by_stat = {}
    stat_scales = {}
    for key in CORE_L2_STATS:
        pooled_true_arr = np.concatenate(pooled_true[key], axis=0)
        pooled_gen_arr = np.concatenate(pooled_gen[key], axis=0)
        scale = float(np.std(pooled_true_arr) + 1e-6)
        stat_scales[key] = scale
        x_norm = (pooled_gen_arr - float(np.mean(pooled_true_arr))) / scale
        y_norm = (pooled_true_arr - float(np.mean(pooled_true_arr))) / scale
        unconditional_by_stat[key] = _wasserstein_1d(
            pooled_gen_arr,
            pooled_true_arr,
        ) / scale
        unconditional_l1_by_stat[key] = _hist_l1(x_norm, y_norm)
        conditional_by_stat[key] = float(np.mean(per_window[key]) / scale)
        conditional_l1_by_stat[key] = float(np.mean(per_window_l1[key]))

    unconditional = float(np.mean(list(unconditional_by_stat.values())))
    conditional = float(np.mean(list(conditional_by_stat.values())))
    u_l1 = float(np.mean(list(unconditional_l1_by_stat.values())))
    c_l1 = float(np.mean(list(conditional_l1_by_stat.values())))
    return {
        "unconditional_w1": unconditional,
        "conditional_w1": conditional,
        "u_l1": u_l1,
        "c_l1": c_l1,
        "unconditional_w1_by_stat": unconditional_by_stat,
        "conditional_w1_by_stat": conditional_by_stat,
        "unconditional_l1_by_stat": unconditional_l1_by_stat,
        "conditional_l1_by_stat": conditional_l1_by_stat,
        "stat_scales": stat_scales,
    }


def _signal_bandpower(values: np.ndarray, fs_hz: float, low_hz: float, high_hz: float) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return float("nan")
    arr = arr - float(np.mean(arr))
    if not np.any(arr):
        return 0.0
    spectrum = np.fft.rfft(arr)
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / float(fs_hz))
    power = (np.abs(spectrum) ** 2) / max(1, arr.size)
    mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def _signal_feature_vector(
    signal: np.ndarray,
    *,
    sampling_rate_hz: float,
    channel_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D signal array [T, C], got shape {arr.shape}.")

    features: List[float] = []
    names: List[str] = []
    for channel_index, channel_name in enumerate(channel_names):
        channel = arr[:, int(channel_index)]
        channel = channel[np.isfinite(channel)]
        if channel.size == 0:
            channel = np.zeros(1, dtype=np.float64)
        centered = channel - float(np.mean(channel))
        line_length = float(np.mean(np.abs(np.diff(channel)))) if channel.size > 1 else 0.0
        zero_crossings = (
            float(np.mean(np.signbit(centered[1:]) != np.signbit(centered[:-1])))
            if channel.size > 1
            else 0.0
        )
        base_stats = {
            "mean": float(np.mean(channel)),
            "std": float(np.std(channel)),
            "rms": float(np.sqrt(np.mean(channel * channel))),
            "line_length": float(line_length),
            "zero_crossing_rate": float(zero_crossings),
        }
        for stat_name, stat_value in base_stats.items():
            names.append(f"{channel_name}:{stat_name}")
            features.append(float(stat_value))
        if "EEG" in str(channel_name) or "EOG" in str(channel_name):
            for band_name, low_hz, high_hz in (
                ("delta", 0.5, 4.0),
                ("theta", 4.0, 8.0),
                ("alpha", 8.0, 12.0),
                ("beta", 12.0, 30.0),
            ):
                names.append(f"{channel_name}:{band_name}_bandpower")
                features.append(_signal_bandpower(channel, float(sampling_rate_hz), float(low_hz), float(high_hz)))
    return np.asarray(features, dtype=np.float32), names


def _collect_signal_feature_examples(
    rows: Sequence[Dict[str, Any]],
    *,
    sampling_rate_hz: float,
    channel_names: Sequence[str],
) -> Dict[str, Any]:
    real_features: List[np.ndarray] = []
    gen_features: List[np.ndarray] = []
    history_scores: List[float] = []
    feature_names: Optional[List[str]] = None
    for row in rows:
        seq = row.get("seq", {})
        gen_signal = seq.get("gen_signal_raw")
        true_signal = seq.get("true_signal_raw")
        if gen_signal is None or true_signal is None:
            continue
        gen_vec, names = _signal_feature_vector(
            np.asarray(gen_signal, dtype=np.float32),
            sampling_rate_hz=float(sampling_rate_hz),
            channel_names=channel_names,
        )
        true_vec, _ = _signal_feature_vector(
            np.asarray(true_signal, dtype=np.float32),
            sampling_rate_hz=float(sampling_rate_hz),
            channel_names=channel_names,
        )
        if feature_names is None:
            feature_names = list(names)
        gen_features.append(gen_vec)
        real_features.append(true_vec)
        history_signal = seq.get("history_signal_raw")
        if history_signal is None:
            history_scores.append(float(np.mean(np.abs(true_vec))))
        else:
            history_vec, _ = _signal_feature_vector(
                np.asarray(history_signal, dtype=np.float32),
                sampling_rate_hz=float(sampling_rate_hz),
                channel_names=channel_names,
            )
            history_scores.append(float(np.nanmean(np.abs(history_vec))))
    if not real_features:
        empty = np.zeros((0, 1), dtype=np.float32)
        return {"real_x": empty, "gen_x": empty, "feature_names": ["empty"], "condition_labels": np.zeros(0, dtype=np.int64)}
    score_arr = np.asarray(history_scores, dtype=np.float64)
    if score_arr.size >= 3 and np.isfinite(score_arr).any():
        q1, q2 = np.nanquantile(score_arr, [1.0 / 3.0, 2.0 / 3.0])
        condition_labels = np.digitize(score_arr, [float(q1), float(q2)], right=False).astype(np.int64)
    else:
        condition_labels = np.zeros(score_arr.shape[0], dtype=np.int64)
    return {
        "real_x": np.stack(real_features, axis=0).astype(np.float32),
        "gen_x": np.stack(gen_features, axis=0).astype(np.float32),
        "feature_names": list(feature_names or []),
        "condition_labels": condition_labels,
    }


def _aggregate_signal_feature_distances(
    *,
    real_x: np.ndarray,
    gen_x: np.ndarray,
    feature_names: Sequence[str],
    condition_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    real_x = np.asarray(real_x, dtype=np.float64)
    gen_x = np.asarray(gen_x, dtype=np.float64)
    if real_x.shape != gen_x.shape:
        raise ValueError("Signal real_x and gen_x must share shape.")
    if real_x.ndim != 2:
        raise ValueError("Signal feature arrays must be 2D.")

    unconditional_by_stat: Dict[str, float] = {}
    conditional_by_stat: Dict[str, float] = {}
    unconditional_l1_by_stat: Dict[str, float] = {}
    conditional_l1_by_stat: Dict[str, float] = {}
    stat_scales: Dict[str, float] = {}
    labels = None if condition_labels is None else np.asarray(condition_labels, dtype=np.int64).reshape(-1)
    if labels is not None and labels.shape[0] != real_x.shape[0]:
        raise ValueError("Signal condition_labels must match the number of examples.")
    for feature_idx, feature_name in enumerate(feature_names):
        real_col = real_x[:, int(feature_idx)]
        gen_col = gen_x[:, int(feature_idx)]
        scale = float(np.std(real_col) + 1e-6)
        stat_scales[str(feature_name)] = scale
        unconditional_by_stat[str(feature_name)] = _wasserstein_1d(real_col, gen_col) / scale
        unconditional_l1_by_stat[str(feature_name)] = _hist_l1(
            (gen_col - float(np.mean(real_col))) / scale,
            (real_col - float(np.mean(real_col))) / scale,
        )
        per_condition = []
        per_condition_l1 = []
        if labels is not None:
            for label in sorted(set(int(value) for value in labels.tolist())):
                mask = labels == int(label)
                if int(np.count_nonzero(mask)) < 2:
                    continue
                local_scale = max(float(np.std(real_col[mask])), 1e-6)
                per_condition.append(_wasserstein_1d(real_col[mask], gen_col[mask]) / local_scale)
                per_condition_l1.append(
                    _hist_l1(
                        (gen_col[mask] - float(np.mean(real_col[mask]))) / local_scale,
                        (real_col[mask] - float(np.mean(real_col[mask]))) / local_scale,
                    )
                )
        conditional_by_stat[str(feature_name)] = float(np.mean(per_condition)) if per_condition else float("nan")
        conditional_l1_by_stat[str(feature_name)] = float(np.mean(per_condition_l1)) if per_condition_l1 else float("nan")
    unconditional = float(np.nanmean(np.asarray(list(unconditional_by_stat.values()), dtype=np.float64)))
    conditional = float(np.nanmean(np.asarray(list(conditional_by_stat.values()), dtype=np.float64)))
    u_l1 = float(np.nanmean(np.asarray(list(unconditional_l1_by_stat.values()), dtype=np.float64)))
    c_l1 = float(np.nanmean(np.asarray(list(conditional_l1_by_stat.values()), dtype=np.float64)))
    return {
        "unconditional_w1": unconditional,
        "conditional_w1": conditional,
        "u_l1": u_l1,
        "c_l1": c_l1,
        "unconditional_w1_by_stat": unconditional_by_stat,
        "conditional_w1_by_stat": conditional_by_stat,
        "unconditional_l1_by_stat": unconditional_l1_by_stat,
        "conditional_l1_by_stat": conditional_l1_by_stat,
        "stat_scales": stat_scales,
    }


def _compare_signal_sequences(
    gen_signal: np.ndarray,
    true_signal: np.ndarray,
    *,
    sampling_rate_hz: float,
    channel_names: Sequence[str],
    max_acf_lag: int = 20,
) -> Dict[str, Any]:
    gen_arr = np.asarray(gen_signal, dtype=np.float64)
    true_arr = np.asarray(true_signal, dtype=np.float64)
    length = min(int(gen_arr.shape[0]), int(true_arr.shape[0]))
    gen_arr = gen_arr[:length]
    true_arr = true_arr[:length]

    mae_by_channel: Dict[str, float] = {}
    rmse_by_channel: Dict[str, float] = {}
    acf_by_channel: Dict[str, Dict[str, float]] = {}
    spectral_errors: List[float] = []
    signal_errors: List[float] = []

    for channel_index, channel_name in enumerate(channel_names):
        gen_col = gen_arr[:, int(channel_index)]
        true_col = true_arr[:, int(channel_index)]
        diff = gen_col - true_col
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        mae_by_channel[str(channel_name)] = mae
        rmse_by_channel[str(channel_name)] = rmse
        signal_errors.append(mae)
        acf_by_channel[str(channel_name)] = {
            "acf_l1": float(np.mean(np.abs(_acf(gen_col, max_lag=max_acf_lag) - _acf(true_col, max_lag=max_acf_lag)))),
            "acf_l2": float(np.sqrt(np.mean((_acf(gen_col, max_lag=max_acf_lag) - _acf(true_col, max_lag=max_acf_lag)) ** 2))),
        }
        gen_feat, feat_names = _signal_feature_vector(
            gen_arr[:, [int(channel_index)]],
            sampling_rate_hz=float(sampling_rate_hz),
            channel_names=[str(channel_name)],
        )
        true_feat, _ = _signal_feature_vector(
            true_arr[:, [int(channel_index)]],
            sampling_rate_hz=float(sampling_rate_hz),
            channel_names=[str(channel_name)],
        )
        if gen_feat.size > 0:
            spectral_errors.append(float(np.mean(np.abs(gen_feat - true_feat))))

    score_terms = np.asarray(signal_errors + spectral_errors, dtype=np.float64)
    finite_terms = score_terms[np.isfinite(score_terms)]
    return {
        "temporal": acf_by_channel,
        "error": {
            "signal_mae": float(np.mean(signal_errors)) if signal_errors else float("nan"),
            "spectral_mae": float(np.mean(spectral_errors)) if spectral_errors else float("nan"),
            "signal_mae_by_channel": mae_by_channel,
            "signal_rmse_by_channel": rmse_by_channel,
        },
        "score_main": float(np.mean(finite_terms)) if finite_terms.size > 0 else float("nan"),
    }


def _evaluate_generation_main_metrics(
    rows: Sequence[Dict[str, Any]],
    cfg: OTFlowConfig,
    *,
    horizon: int,
    seed: int,
    max_examples_per_split: int = 20_000,
    dataset_kind: str = "l2",
    dataset_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    dataset_kind = str(dataset_kind)
    dataset_metadata = dict(dataset_metadata or {})
    if dataset_kind == LONG_TERM_ST_DATASET_KEY:
        device = _downstream_device(cfg)
        channel_names = list(dataset_metadata.get("channel_names", [])) or ["ECG"]
        sampling_rate_hz = float(dataset_metadata.get("sampling_rate_hz", 100.0))
        downstream = _collect_signal_feature_examples(
            rows,
            sampling_rate_hz=sampling_rate_hz,
            channel_names=channel_names,
        )
        real_x = downstream["real_x"]
        gen_x = downstream["gen_x"]
        disc_auc = float("nan")
        if len(gen_x) > 1 and len(real_x) > 1:
            train_x, train_y, test_x, test_y = _pairwise_split(real_x, gen_x, seed=seed + 17)
            if len(train_x) > 0 and len(test_x) > 0:
                train_x, test_x = _standardize_pair(train_x, test_x)
                disc_auc = _train_small_discriminator_auc(
                    train_x,
                    train_y.astype(np.float32),
                    test_x,
                    test_y,
                    device=device,
                    seed=seed + 47,
                )
        disc_auc_gap = float(abs(disc_auc - 0.5)) if np.isfinite(disc_auc) else float("nan")
        w1_metrics = _aggregate_signal_feature_distances(
            real_x=real_x,
            gen_x=gen_x,
            feature_names=downstream["feature_names"],
            condition_labels=downstream["condition_labels"],
        )
        score_terms = [
            disc_auc_gap,
            np.log1p(w1_metrics["unconditional_w1"]) if np.isfinite(w1_metrics["unconditional_w1"]) else np.nan,
        ]
        finite_terms = np.asarray([term for term in score_terms if np.isfinite(term)], dtype=np.float64)
        score_main = float(finite_terms.mean()) if finite_terms.size > 0 else float("nan")
        return {
            "temporal_tstr_f1": None,
            "temporal_tstr_f1_applicable": False,
            "disc_auc": float(disc_auc),
            "disc_auc_gap": float(disc_auc_gap),
            "temporal_uw1": float(w1_metrics["unconditional_w1"]),
            "temporal_cw1": float(w1_metrics["conditional_w1"]),
            "u_l1": float(w1_metrics["u_l1"]),
            "c_l1": float(w1_metrics["c_l1"]),
            "temporal_uw1_by_stat": w1_metrics["unconditional_w1_by_stat"],
            "temporal_cw1_by_stat": w1_metrics["conditional_w1_by_stat"],
            "unconditional_l1_by_stat": w1_metrics["unconditional_l1_by_stat"],
            "conditional_l1_by_stat": w1_metrics["conditional_l1_by_stat"],
            "stat_scales": w1_metrics["stat_scales"],
            "score_main": float(score_main),
            "label_horizon": int(max(1, horizon)),
            "n_examples_real": int(len(real_x)),
            "n_examples_gen": int(len(gen_x)),
            "threshold_abs_move": float("nan"),
        }
    label_horizon = int(max(1, min(10, max(1, horizon // 10))))
    downstream = _collect_downstream_examples(
        rows,
        label_horizon=label_horizon,
        max_examples_per_split=max_examples_per_split,
        seed=seed,
    )
    device = _downstream_device(cfg)

    real_x = downstream["real_x"]
    real_moves = downstream["real_moves"]
    gen_x = downstream["gen_x"]
    gen_moves = downstream["gen_moves"]

    threshold = float(np.quantile(np.abs(real_moves), 1.0 / 3.0)) if len(real_moves) > 0 else float("nan")
    real_y = _ternary_labels(real_moves, threshold) if np.isfinite(threshold) else np.zeros(0, dtype=np.int64)
    gen_y = _ternary_labels(gen_moves, threshold) if np.isfinite(threshold) else np.zeros(0, dtype=np.int64)

    tstr_macro_f1 = float("nan")
    if len(gen_x) > 0 and len(real_x) > 0 and np.unique(gen_y).size >= 2 and np.unique(real_y).size >= 2:
        x_train, x_test = _standardize_pair(gen_x, real_x)
        tstr_macro_f1 = _train_small_multiclass_mlp_f1(
            x_train,
            gen_y,
            x_test,
            real_y,
            device=device,
            seed=seed + 31,
        )

    disc_auc = float("nan")
    if len(gen_x) > 1 and len(real_x) > 1:
        train_x, train_y, test_x, test_y = _pairwise_split(real_x, gen_x, seed=seed + 17)
        if len(train_x) > 0 and len(test_x) > 0:
            train_x, test_x = _standardize_pair(train_x, test_x)
            disc_auc = _train_small_discriminator_auc(
                train_x,
                train_y.astype(np.float32),
                test_x,
                test_y,
                device=device,
                seed=seed + 47,
            )
    disc_auc_gap = float(abs(disc_auc - 0.5)) if np.isfinite(disc_auc) else float("nan")

    w1_metrics = _aggregate_core_l2_distribution_metrics(rows)

    score_terms = [
        1.0 - tstr_macro_f1 if np.isfinite(tstr_macro_f1) else np.nan,
        disc_auc_gap,
        np.log1p(w1_metrics["unconditional_w1"]) if np.isfinite(w1_metrics["unconditional_w1"]) else np.nan,
        np.log1p(w1_metrics["conditional_w1"]) if np.isfinite(w1_metrics["conditional_w1"]) else np.nan,
    ]
    finite_terms = np.asarray([term for term in score_terms if np.isfinite(term)], dtype=np.float64)
    score_main = float(finite_terms.mean()) if finite_terms.size > 0 else float("nan")

    return {
        "temporal_tstr_f1": float(tstr_macro_f1) if np.isfinite(tstr_macro_f1) else None,
        "temporal_tstr_f1_applicable": True,
        "disc_auc": float(disc_auc),
        "disc_auc_gap": float(disc_auc_gap),
        "temporal_uw1": float(w1_metrics["unconditional_w1"]),
        "temporal_cw1": float(w1_metrics["conditional_w1"]),
        "u_l1": float(w1_metrics["u_l1"]),
        "c_l1": float(w1_metrics["c_l1"]),
        "temporal_uw1_by_stat": w1_metrics["unconditional_w1_by_stat"],
        "temporal_cw1_by_stat": w1_metrics["conditional_w1_by_stat"],
        "unconditional_l1_by_stat": w1_metrics["unconditional_l1_by_stat"],
        "conditional_l1_by_stat": w1_metrics["conditional_l1_by_stat"],
        "stat_scales": w1_metrics["stat_scales"],
        "score_main": float(score_main),
        "label_horizon": int(label_horizon),
        "n_examples_real": int(len(real_x)),
        "n_examples_gen": int(len(gen_x)),
        "threshold_abs_move": float(threshold) if np.isfinite(threshold) else float("nan"),
    }


def _param_horizon_metrics(gen_params: np.ndarray, true_params: np.ndarray, horizons: Sequence[int]) -> Dict[str, Dict[str, float]]:
    T = min(len(gen_params), len(true_params))
    g = gen_params[:T]
    r = true_params[:T]
    out: Dict[str, Dict[str, float]] = {}
    for h in horizons:
        hh = int(h)
        if hh <= 0 or hh > T:
            continue
        dg = g[:hh]
        dr = r[:hh]
        diff = dg - dr
        out[str(hh)] = {
            "params_mae": float(np.mean(np.abs(diff))),
            "params_rmse": float(np.sqrt(np.mean(diff * diff))),
            "delta_mid_rmse": float(np.sqrt(np.mean((dg[:, 0] - dr[:, 0]) ** 2))),
            "log_spread_rmse": float(np.sqrt(np.mean((dg[:, 1] - dr[:, 1]) ** 2))),
        }
    return out


def compare_l2_sequences(
    gen_params: np.ndarray,
    true_params: np.ndarray,
    ask_p_gen: np.ndarray,
    ask_v_gen: np.ndarray,
    bid_p_gen: np.ndarray,
    bid_v_gen: np.ndarray,
    ask_p_true: np.ndarray,
    ask_v_true: np.ndarray,
    bid_p_true: np.ndarray,
    bid_v_true: np.ndarray,
    max_acf_lag: int = 20,
) -> Dict[str, Any]:
    T = int(min(len(gen_params), len(true_params)))
    gen_params = gen_params[:T]
    true_params = true_params[:T]

    sg = microstructure_series(ask_p_gen[:T], ask_v_gen[:T], bid_p_gen[:T], bid_v_gen[:T])
    st = microstructure_series(ask_p_true[:T], ask_v_true[:T], bid_p_true[:T], bid_v_true[:T])
    vol_g = _rolling_volatility(sg["ret"], window=10)
    vol_t = _rolling_volatility(st["ret"], window=10)

    # Distribution fidelity
    dist = {}
    for k in ("spread", "depth", "imb", "ret"):
        dist[k] = {
            "ks": _ks_stat(sg[k], st[k]),
            "w1": _wasserstein_1d(sg[k], st[k]),
        }

    # Temporal fidelity
    temporal = {}
    for k in ("ret", "spread", "depth", "imb"):
        ag = _acf(sg[k], max_lag=max_acf_lag)
        at = _acf(st[k], max_lag=max_acf_lag)
        temporal[k] = {
            "acf_l1": float(np.mean(np.abs(ag - at))),
            "acf_l2": float(np.sqrt(np.mean((ag - at) ** 2))),
        }
    temporal["volatility"] = {
        "acf_l1": float(np.mean(np.abs(_acf(vol_g, max_lag=max_acf_lag) - _acf(vol_t, max_lag=max_acf_lag)))),
        "acf_l2": float(np.sqrt(np.mean((_acf(vol_g, max_lag=max_acf_lag) - _acf(vol_t, max_lag=max_acf_lag)) ** 2))),
    }

    # Cross-level structure: volume correlation matrix similarity
    def _corr_mat(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        if v.shape[0] < 2:
            return np.zeros((v.shape[1], v.shape[1]), dtype=np.float64)
        return np.corrcoef(v.T)

    corr_gen = _corr_mat(np.concatenate([bid_v_gen[:T], ask_v_gen[:T]], axis=1))
    corr_true = _corr_mat(np.concatenate([bid_v_true[:T], ask_v_true[:T]], axis=1))
    corr_diff = corr_gen - corr_true
    structure = {
        "vol_corr_mae": float(np.nanmean(np.abs(corr_diff))),
        "vol_corr_rmse": float(np.sqrt(np.nanmean(corr_diff * corr_diff))),
    }

    # Param-space fit
    p_diff = gen_params - true_params
    params_fit = {
        "params_mae": float(np.mean(np.abs(p_diff))),
        "params_rmse": float(np.sqrt(np.mean(p_diff * p_diff))),
        "delta_mid_rmse": float(np.sqrt(np.mean((gen_params[:, 0] - true_params[:, 0]) ** 2))),
        "log_spread_rmse": float(np.sqrt(np.mean((gen_params[:, 1] - true_params[:, 1]) ** 2))),
    }

    validity = _validity_metrics(ask_p_gen[:T], ask_v_gen[:T], bid_p_gen[:T], bid_v_gen[:T])
    errors = {
        "spread_mae": _normalized_mae(sg["spread"], st["spread"]),
        "imbalance_mae": _normalized_mae(sg["imb"], st["imb"]),
    }
    response_g = _impact_response_curve(sg["imb"], sg["ret"])
    response_t = _impact_response_curve(st["imb"], st["ret"])
    response_err = []
    for lag_key in response_t.keys():
        if np.isfinite(response_g[lag_key]) and np.isfinite(response_t[lag_key]):
            denom = max(abs(response_t[lag_key]), 1e-6)
            response_err.append(abs(response_g[lag_key] - response_t[lag_key]) / denom)
    microstructure = {
        "impact_response_l1": float(np.mean(response_err)) if response_err else float("nan"),
        "impact_response_curve_gen": response_g,
        "impact_response_curve_true": response_t,
    }

    # Compact scalar (lower is better) for Pareto plots
    score_main = (
        dist["spread"]["w1"]
        + dist["imb"]["w1"]
        + dist["ret"]["w1"]
        + temporal["ret"]["acf_l1"]
        + params_fit["params_rmse"]
        + (1.0 - validity["valid_rate"])
    )

    return {
        "dist": dist,
        "temporal": temporal,
        "structure": structure,
        "params_fit": params_fit,
        "validity": validity,
        "error": errors,
        "microstructure": microstructure,
        "score_main": float(score_main),
        "basic_gen": compute_basic_l2_metrics(ask_p_gen[:T], ask_v_gen[:T], bid_p_gen[:T], bid_v_gen[:T]),
        "basic_true": compute_basic_l2_metrics(ask_p_true[:T], ask_v_true[:T], bid_p_true[:T], bid_v_true[:T]),
    }


# -----------------------------
# Evaluation
# -----------------------------
def _valid_eval_indices(ds: WindowedParamSequenceDataset, horizon: int) -> np.ndarray:
    starts = np.asarray(ds.start_indices, dtype=np.int64)
    if len(starts) == 0:
        return starts
    segment_ends = ds.segment_end_for_t(starts)
    return starts[starts + int(horizon) <= segment_ends]


def select_eval_window_starts(ds: WindowedParamSequenceDataset, horizon: int, n_windows: int, seed: int) -> np.ndarray:
    valid_ts = _valid_eval_indices(ds, int(horizon))
    if len(valid_ts) == 0:
        raise ValueError(f"No valid windows for horizon={horizon}.")
    rng = np.random.default_rng(seed)
    if int(n_windows) <= len(valid_ts):
        return np.asarray(rng.choice(valid_ts, size=int(n_windows), replace=False), dtype=np.int64)
    return np.asarray(rng.choice(valid_ts, size=int(n_windows), replace=True), dtype=np.int64)


def _denorm_params_seq(ds: WindowedParamSequenceDataset, x_norm: np.ndarray) -> np.ndarray:
    if ds.params_mean is not None and ds.params_std is not None:
        return (x_norm * ds.params_std[None, :] + ds.params_mean[None, :]).astype(np.float32)
    return x_norm.astype(np.float32)


def _get_dataset_item_by_t(ds: WindowedParamSequenceDataset, t0: int):
    idxs = np.where(np.asarray(ds.start_indices) == int(t0))[0]
    if len(idxs) == 0:
        raise IndexError(f"t={t0} not in dataset start_indices.")
    return ds[int(idxs[0])]


@torch.no_grad()
def eval_one_window(
    ds: WindowedParamSequenceDataset,
    model: torch.nn.Module,
    cfg: OTFlowConfig,
    horizon: int = 200,
    nfe: int = 1,
    seed: int = 0,
    t0: Optional[int] = None,
    horizons_eval: Sequence[int] = (1, 10, 50, 100, 200),
    return_sequences: bool = False,
    main_metrics_only: bool = False,
) -> Dict[str, Any]:
    horizon = int(horizon)
    dataset_kind = str(getattr(ds, "dataset_kind", "l2"))
    dataset_metadata = dict(getattr(ds, "dataset_metadata", {}) or {})

    if t0 is None:
        rng = np.random.default_rng(seed)
        valid_ts = _valid_eval_indices(ds, horizon)
        if len(valid_ts) == 0:
            raise ValueError(f"No valid windows for horizon={horizon}.")
        t0 = int(valid_ts[int(rng.integers(0, len(valid_ts)))])

    batch = _get_dataset_item_by_t(ds, t0)
    hist, _, _, _, meta = _parse_batch(batch)
    hist = hist[None, :, :].to(cfg.device).float()
    context_len = resolve_context_length(hist.shape[1], horizon=horizon, cfg=cfg)

    cond_seq = None
    if ds.cond is not None:
        cond_seq = torch.from_numpy(ds.cond[t0 : t0 + horizon]).to(cfg.device).float()[None, :, :]

    future_context_seq = None
    future_context = _future_time_context_seq(ds, int(t0), int(horizon))
    if future_context is not None:
        future_context_seq = future_context.to(cfg.device).float()[None, :, :]

    # Generate and time
    _torch_sync(cfg.device)
    t_start = time.perf_counter()
    with _temporary_eval_seed(int(seed)):
        gen_norm = generate_continuation(
            model,
            hist,
            cond_seq,
            steps=horizon,
            nfe=nfe,
            future_context_seq=future_context_seq,
        )
    _torch_sync(cfg.device)
    latency_s = time.perf_counter() - t_start
    gen_norm = gen_norm[0].cpu().numpy()

    # True continuation
    true_norm = ds.params[t0 : t0 + horizon].astype(np.float32)
    gen_raw_params = _denorm_params_seq(ds, gen_norm)
    true_raw_params = _denorm_params_seq(ds, true_norm)
    history_start = max(0, int(t0) - int(context_len))
    history_norm = ds.params[history_start:int(t0)].astype(np.float32)
    history_raw_params = _denorm_params_seq(ds, history_norm) if history_norm.size > 0 else np.zeros((0, gen_raw_params.shape[1]), dtype=np.float32)

    if dataset_kind == LONG_TERM_ST_DATASET_KEY:
        channel_names = list(dataset_metadata.get("channel_names", []))
        if not channel_names:
            channel_names = [f"channel_{idx}" for idx in range(int(gen_raw_params.shape[1]))]
        sampling_rate_hz = float(dataset_metadata.get("sampling_rate_hz", 100.0))
        cmp_metrics = (
            {}
            if main_metrics_only
            else _compare_signal_sequences(
                gen_raw_params,
                true_raw_params,
                sampling_rate_hz=sampling_rate_hz,
                channel_names=channel_names,
            )
        )
        horizon_metrics = {}
        if not main_metrics_only:
            horizon_metrics = {}
            for hh in horizons_eval:
                hh = int(hh)
                if hh <= 0 or hh > min(len(gen_raw_params), len(true_raw_params)):
                    continue
                diff = gen_raw_params[:hh] - true_raw_params[:hh]
                horizon_metrics[str(hh)] = {
                    "signal_mae": float(np.mean(np.abs(diff))),
                    "signal_rmse": float(np.sqrt(np.mean(diff * diff))),
                }
        basic_gen = {}
        basic_true = {}
        out: Dict[str, Any] = {
            "gen": basic_gen,
            "true": basic_true,
            "cmp": cmp_metrics,
            "horizon": horizon_metrics,
            "timing": {
                "latency_s_total": float(latency_s),
                "latency_ms_per_sample": float(1000.0 * latency_s),
                "latency_ms_per_step": float(1000.0 * latency_s / max(1, horizon)),
                "throughput_steps_per_s": float(horizon / max(latency_s, 1e-12)),
                "nfe": int(nfe),
                "horizon": int(horizon),
                "main_metrics_only": bool(main_metrics_only),
            },
            "meta": {
                "t": int(t0),
                "context_len": int(context_len),
                "main_metrics_only": bool(main_metrics_only),
                "dataset_kind": dataset_kind,
            },
        }
        if return_sequences:
            out["seq"] = {
                "gen_params_raw": gen_raw_params,
                "true_params_raw": true_raw_params,
                "gen_signal_raw": gen_raw_params,
                "true_signal_raw": true_raw_params,
                "history_signal_raw": history_raw_params,
            }
        return out

    # Decode to raw L2
    fm = L2FeatureMap(cfg.levels, cfg.eps)

    # decode deltas against the previous mid: first row uses mids[t0-1]
    if t0 <= 0:
        raise ValueError("t0 must be >= 1 to decode delta-mid trajectories.")
    init_mid_prev = float(ds.mids[t0 - 1])

    ask_p_g, ask_v_g, bid_p_g, bid_v_g = fm.decode_sequence(gen_raw_params, init_mid=init_mid_prev)
    ask_p_r, ask_v_r, bid_p_r, bid_v_r = fm.decode_sequence(true_raw_params, init_mid=init_mid_prev)

    cmp_metrics = (
        {}
        if main_metrics_only
        else compare_l2_sequences(
            gen_params=gen_raw_params,
            true_params=true_raw_params,
            ask_p_gen=ask_p_g, ask_v_gen=ask_v_g, bid_p_gen=bid_p_g, bid_v_gen=bid_v_g,
            ask_p_true=ask_p_r, ask_v_true=ask_v_r, bid_p_true=bid_p_r, bid_v_true=bid_v_r,
        )
    )
    horizon_metrics = {} if main_metrics_only else _param_horizon_metrics(gen_raw_params, true_raw_params, horizons=horizons_eval)
    basic_gen = (
        {}
        if main_metrics_only
        else compute_basic_l2_metrics(ask_p_g, ask_v_g, bid_p_g, bid_v_g)
    )
    basic_true = (
        {}
        if main_metrics_only
        else compute_basic_l2_metrics(ask_p_r, ask_v_r, bid_p_r, bid_v_r)
    )

    out: Dict[str, Any] = {
        "gen": basic_gen,
        "true": basic_true,
        "cmp": cmp_metrics,
        "horizon": horizon_metrics,
        "timing": {
            "latency_s_total": float(latency_s),
            "latency_ms_per_sample": float(1000.0 * latency_s),
            "latency_ms_per_step": float(1000.0 * latency_s / max(1, horizon)),
            "throughput_steps_per_s": float(horizon / max(latency_s, 1e-12)),
            "nfe": int(nfe),
            "horizon": int(horizon),
            "main_metrics_only": bool(main_metrics_only),
        },
        "meta": {
            "t": int(t0),
            "init_mid_for_window": float(meta["init_mid_for_window"]),
            "init_mid_prev": float(init_mid_prev),
            "context_len": int(context_len),
            "main_metrics_only": bool(main_metrics_only),
            "dataset_kind": dataset_kind,
        },
    }

    if return_sequences:
        out["seq"] = {
            "gen_params_raw": gen_raw_params,
            "true_params_raw": true_raw_params,
            "gen": {"ask_p": ask_p_g, "ask_v": ask_v_g, "bid_p": bid_p_g, "bid_v": bid_v_g},
            "true": {"ask_p": ask_p_r, "ask_v": ask_v_r, "bid_p": bid_p_r, "bid_v": bid_v_r},
        }
    return out


def _aggregate_nested_dicts(dicts):
    flat_rows = [flatten_dict(d) for d in dicts]
    keys = sorted(set().union(*[set(fr.keys()) for fr in flat_rows]))
    aggs: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [fr[k] for fr in flat_rows if k in fr and np.isfinite(fr[k])]
        if not vals:
            continue
        aggs[k] = _safe_mean_std(vals)
    return unflatten_to_nested(aggs)


def _wrap_scalar_as_mean_std(value: float) -> Dict[str, float]:
    return {"mean": float(value), "std": 0.0}


def _wrap_optional_scalar_as_mean_std(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"mean": None, "std": None}
    return _wrap_scalar_as_mean_std(float(value))


@torch.no_grad()
def eval_many_windows(
    ds: WindowedParamSequenceDataset,
    model: torch.nn.Module,
    cfg: OTFlowConfig,
    horizon: int = 200,
    nfe: int = 1,
    n_windows: int = 50,
    seed: int = 0,
    horizons_eval: Sequence[int] = (1, 10, 50, 100, 200),
    chosen_t0s: Optional[Sequence[int]] = None,
    generation_seed_base: Optional[int] = None,
    metrics_seed: Optional[int] = None,
    main_metrics_only: bool = False,
) -> Dict[str, Any]:
    dataset_kind = str(getattr(ds, "dataset_kind", "l2"))
    dataset_metadata = dict(getattr(ds, "dataset_metadata", {}) or {})
    rng = np.random.default_rng(seed)
    valid_ts = _valid_eval_indices(ds, horizon)
    if len(valid_ts) == 0:
        raise ValueError(f"No valid windows for horizon={horizon}.")

    if chosen_t0s is not None:
        chosen = np.asarray([int(t0) for t0 in chosen_t0s], dtype=np.int64)
        if chosen.ndim != 1 or chosen.size == 0:
            raise ValueError("chosen_t0s must be a non-empty 1D sequence of valid window starts.")
        valid_set = set(int(t0) for t0 in valid_ts.tolist())
        invalid = [int(t0) for t0 in chosen.tolist() if int(t0) not in valid_set]
        if invalid:
            raise ValueError(f"chosen_t0s contains invalid window starts: {invalid}")
    else:
        chosen = select_eval_window_starts(ds, horizon=int(horizon), n_windows=int(n_windows), seed=int(seed))

    metrics_seed_value = int(seed if metrics_seed is None else metrics_seed)

    rows = []
    for window_idx, t0 in enumerate(chosen.tolist()):
        window_seed = (
            int(generation_seed_base) + int(window_idx)
            if generation_seed_base is not None
            else int(rng.integers(0, 1_000_000))
        )
        rows.append(
            eval_one_window(
                ds, model, cfg,
                horizon=horizon, nfe=nfe,
                t0=int(t0),
                seed=window_seed,
                horizons_eval=horizons_eval,
                return_sequences=True,
                main_metrics_only=bool(main_metrics_only),
            )
        )

    cmp = _aggregate_nested_dicts([r["cmp"] for r in rows])
    timing = _aggregate_nested_dicts([r["timing"] for r in rows])
    with torch.enable_grad():
        main_metrics = _evaluate_generation_main_metrics(
            rows,
            cfg,
            horizon=horizon,
            seed=metrics_seed_value,
            dataset_kind=dataset_kind,
            dataset_metadata=dataset_metadata,
        )
    cmp["main"] = {
        "temporal_tstr_f1": _wrap_optional_scalar_as_mean_std(main_metrics["temporal_tstr_f1"]),
        "temporal_tstr_f1_applicable": bool(main_metrics["temporal_tstr_f1_applicable"]),
        "disc_auc": _wrap_scalar_as_mean_std(main_metrics["disc_auc"]),
        "disc_auc_gap": _wrap_scalar_as_mean_std(main_metrics["disc_auc_gap"]),
        "temporal_uw1": _wrap_scalar_as_mean_std(main_metrics["temporal_uw1"]),
        "temporal_cw1": _wrap_scalar_as_mean_std(main_metrics["temporal_cw1"]),
        "label_horizon": int(main_metrics["label_horizon"]),
        "n_examples_real": int(main_metrics["n_examples_real"]),
        "n_examples_gen": int(main_metrics["n_examples_gen"]),
        "threshold_abs_move": _wrap_scalar_as_mean_std(main_metrics["threshold_abs_move"]),
        "temporal_uw1_by_stat": {
            key: _wrap_scalar_as_mean_std(val)
            for key, val in main_metrics["temporal_uw1_by_stat"].items()
        },
        "temporal_cw1_by_stat": {
            key: _wrap_scalar_as_mean_std(val)
            for key, val in main_metrics["temporal_cw1_by_stat"].items()
        },
        "stat_scales": {
            key: _wrap_scalar_as_mean_std(val)
            for key, val in main_metrics["stat_scales"].items()
        },
    }
    ret_acf = cmp.get("temporal", {}).get("ret", {}).get("acf_l1", {}).get("mean")
    vol_acf = cmp.get("temporal", {}).get("volatility", {}).get("acf_l1", {}).get("mean")
    ret_vol_terms = [v for v in (ret_acf, vol_acf) if isinstance(v, (int, float)) and np.isfinite(v)]
    if dataset_kind == LONG_TERM_ST_DATASET_KEY:
        temporal_entries = []
        for channel_payload in cmp.get("temporal", {}).values():
            if isinstance(channel_payload, dict):
                value = channel_payload.get("acf_l1", {}).get("mean")
                if isinstance(value, (int, float)) and np.isfinite(value):
                    temporal_entries.append(float(value))
        cmp["extra"] = {
            "u_l1": _wrap_scalar_as_mean_std(main_metrics["u_l1"]),
            "c_l1": _wrap_scalar_as_mean_std(main_metrics["c_l1"]),
            "spread_specific_error": cmp.get("error", {}).get("signal_mae", _wrap_scalar_as_mean_std(float("nan"))),
            "imbalance_specific_error": cmp.get("error", {}).get("spectral_mae", _wrap_scalar_as_mean_std(float("nan"))),
            "ret_vol_acf_error": _wrap_scalar_as_mean_std(
                float(np.mean(temporal_entries)) if temporal_entries else float("nan")
            ),
            "impact_response_error": _wrap_scalar_as_mean_std(float("nan")),
            "efficiency_ms_per_sample": timing.get("latency_ms_per_sample", _wrap_scalar_as_mean_std(float("nan"))),
        }
    else:
        cmp["extra"] = {
            "u_l1": _wrap_scalar_as_mean_std(main_metrics["u_l1"]),
            "c_l1": _wrap_scalar_as_mean_std(main_metrics["c_l1"]),
            "spread_specific_error": cmp.get("error", {}).get("spread_mae", _wrap_scalar_as_mean_std(float("nan"))),
            "imbalance_specific_error": cmp.get("error", {}).get("imbalance_mae", _wrap_scalar_as_mean_std(float("nan"))),
            "ret_vol_acf_error": _wrap_scalar_as_mean_std(float(np.mean(ret_vol_terms)) if ret_vol_terms else float("nan")),
            "impact_response_error": cmp.get("microstructure", {}).get("impact_response_l1", _wrap_scalar_as_mean_std(float("nan"))),
            "efficiency_ms_per_sample": timing.get("latency_ms_per_sample", _wrap_scalar_as_mean_std(float("nan"))),
        }
    cmp["score_main"] = _wrap_scalar_as_mean_std(main_metrics["score_main"])

    return {
        "gen": _aggregate_nested_dicts([r["gen"] for r in rows]),
        "true": _aggregate_nested_dicts([r["true"] for r in rows]),
        "cmp": cmp,
        "horizon": _aggregate_nested_dicts([r["horizon"] for r in rows]),
        "timing": timing,
        "meta": {
            "n_windows": int(len(rows)),
            "nfe": int(nfe),
            "horizon": int(horizon),
            "chosen_t0s": [int(t0) for t0 in chosen.tolist()],
            "chosen_t0s_hash": hashlib.sha256(
                json.dumps([int(t0) for t0 in chosen.tolist()], separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "generation_seed_base": None if generation_seed_base is None else int(generation_seed_base),
            "metrics_seed": int(metrics_seed_value),
            "main_metrics_only": bool(main_metrics_only),
            "dataset_kind": dataset_kind,
        },
    }


@torch.no_grad()
def eval_rollout_horizons(
    ds: WindowedParamSequenceDataset,
    model: torch.nn.Module,
    cfg: OTFlowConfig,
    horizons: Sequence[int] = (1, 10, 50, 100, 200),
    nfe: int = 1,
    n_windows: int = 50,
    seed: int = 0,
) -> Dict[int, Dict[str, Any]]:
    out = {}
    for h in horizons:
        out[int(h)] = eval_many_windows(ds, model, cfg, horizon=int(h), nfe=nfe, n_windows=n_windows, seed=seed, horizons_eval=horizons)
    return out


# -----------------------------
# Speed benchmarking (B)
# -----------------------------
@torch.no_grad()
def benchmark_sampling_latency(
    ds: WindowedParamSequenceDataset,
    model: torch.nn.Module,
    cfg: OTFlowConfig,
    horizon: int = 200,
    nfe: int = 1,
    n_trials: int = 20,
    warmup: int = 3,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    valid_ts = _valid_eval_indices(ds, horizon)
    if len(valid_ts) == 0:
        raise ValueError(f"No valid windows for horizon={horizon}.")
    t0 = int(valid_ts[int(rng.integers(0, len(valid_ts)))])

    hist, _, _, _, _ = _parse_batch(_get_dataset_item_by_t(ds, t0))
    hist = hist[None].to(cfg.device).float()
    context_len = resolve_context_length(hist.shape[1], horizon=horizon, cfg=cfg)

    cond_seq = None
    if ds.cond is not None:
        cond_seq = torch.from_numpy(ds.cond[t0 : t0 + horizon]).to(cfg.device).float()[None]
    future_context_seq = None
    future_context = _future_time_context_seq(ds, int(t0), int(horizon))
    if future_context is not None:
        future_context_seq = future_context.to(cfg.device).float()[None]

    for __ in range(max(0, warmup)):
        generate_continuation(model, hist, cond_seq, steps=horizon, nfe=nfe, future_context_seq=future_context_seq)
    _torch_sync(cfg.device)

    times = []
    for _ in range(max(1, n_trials)):
        _torch_sync(cfg.device)
        t1 = time.perf_counter()
        _ = generate_continuation(model, hist, cond_seq, steps=horizon, nfe=nfe, future_context_seq=future_context_seq)
        _torch_sync(cfg.device)
        times.append(time.perf_counter() - t1)

    arr = np.asarray(times, dtype=np.float64)
    return {
        "latency_s_mean": float(arr.mean()),
        "latency_s_std": float(arr.std()),
        "latency_ms_per_sample_mean": float(1000.0 * arr.mean()),
        "latency_ms_per_step_mean": float(1000.0 * arr.mean() / max(1, horizon)),
        "throughput_steps_per_s_mean": float(horizon / max(arr.mean(), 1e-12)),
        "n_trials": int(n_trials),
        "warmup": int(warmup),
        "nfe": int(nfe),
        "horizon": int(horizon),
        "context_len": int(context_len),
    }


@torch.no_grad()
def eval_speed_quality_nfe(
    ds: WindowedParamSequenceDataset,
    model: torch.nn.Module,
    cfg: OTFlowConfig,
    nfe_list: Sequence[int] = (1, 2, 4, 8, 16, 32),
    horizon: int = 200,
    n_windows: int = 30,
    seed: int = 0,
    n_trials_latency: int = 10,
) -> Dict[int, Dict[str, Any]]:
    results = {}
    for nfe in nfe_list:
        q = eval_many_windows(ds, model, cfg, horizon=horizon, nfe=int(nfe), n_windows=n_windows, seed=seed)
        t = benchmark_sampling_latency(ds, model, cfg, horizon=horizon, nfe=int(nfe), n_trials=n_trials_latency, seed=seed)
        results[int(nfe)] = {"quality": q, "latency": t}
    return results


# -----------------------------
# Ablations (C)
# -----------------------------
def clone_cfg_with_overrides(cfg: OTFlowConfig, overrides: Dict[str, Any]) -> OTFlowConfig:
    cfg2 = copy.deepcopy(cfg)
    cfg2.apply_overrides(**overrides)
    return cfg2


def run_ablation_grid(
    ds_train: WindowedParamSequenceDataset,
    ds_eval: WindowedParamSequenceDataset,
    base_cfg: OTFlowConfig,
    ablations: Sequence[Tuple[str, Dict[str, Any]]],
    model_name: str = "otflow",
    train_steps: int = 10_000,
    stage2_steps_nf: Optional[int] = None,
    eval_horizon: int = 200,
    eval_nfe: int = 1,
    n_windows: int = 30,
    seed: int = 0,
    log_every: int = 200,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    suite = {}
    model_name = _normalize_model_name(model_name)

    for name, overrides in ablations:
        cfg_i = clone_cfg_with_overrides(base_cfg, dict(overrides))
        print(f"\n=== Ablation: {name} | overrides={overrides} ===")

        model = train_loop(ds_train, cfg_i, model_name=model_name, steps=train_steps, log_every=log_every)

        res = eval_many_windows(
            ds_eval, model, cfg_i,
            horizon=eval_horizon, nfe=eval_nfe,
            n_windows=n_windows, seed=int(rng.integers(0, 1_000_000)),
        )

        suite[name] = {
            "overrides": dict(overrides),
            "model_name": model_name,
            "train_steps": int(train_steps),
            "eval": res,
        }
    return suite


def summarize_ablation_for_table(
    ablation_results: Dict[str, Any],
    keys: Sequence[str] = (
        "eval.cmp.score_main.mean",
        "eval.cmp.main.temporal_tstr_f1.mean",
        "eval.cmp.main.disc_auc_gap.mean",
        "eval.cmp.main.temporal_uw1.mean",
        "eval.cmp.main.temporal_cw1.mean",
        "eval.cmp.extra.u_l1.mean",
        "eval.cmp.extra.c_l1.mean",
        "eval.cmp.extra.spread_specific_error.mean",
        "eval.cmp.extra.imbalance_specific_error.mean",
        "eval.cmp.extra.ret_vol_acf_error.mean",
        "eval.cmp.extra.impact_response_error.mean",
        "eval.cmp.extra.efficiency_ms_per_sample.mean",
    ),
):
    rows = []
    for name, payload in ablation_results.items():
        row = {"name": name}
        flat = flatten_dict(payload)
        for k in keys:
            if k in flat:
                row[k] = flat[k]
        rows.append(row)
    rows = sorted(rows, key=lambda r: r.get("eval.cmp.score_main.mean", float("inf")))
    return rows


# -----------------------------
# Qualitative bundle export (for separate viz script)
# -----------------------------
@torch.no_grad()
def save_qualitative_window_npz(
    save_path: str,
    ds: WindowedParamSequenceDataset,
    model: torch.nn.Module,
    cfg: OTFlowConfig,
    horizon: int = 200,
    nfe: int = 1,
    seed: int = 0,
    t0: Optional[int] = None,
):
    res = eval_one_window(ds, model, cfg, horizon=horizon, nfe=nfe, seed=seed, t0=t0, return_sequences=True)
    seq = res["seq"]
    meta = res["meta"]
    np.savez_compressed(
        save_path,
        gen_params_raw=seq["gen_params_raw"],
        true_params_raw=seq["true_params_raw"],
        gen_ask_p=seq["gen"]["ask_p"], gen_ask_v=seq["gen"]["ask_v"],
        gen_bid_p=seq["gen"]["bid_p"], gen_bid_v=seq["gen"]["bid_v"],
        true_ask_p=seq["true"]["ask_p"], true_ask_v=seq["true"]["ask_v"],
        true_bid_p=seq["true"]["bid_p"], true_bid_v=seq["true"]["bid_v"],
        meta_t=np.int64(meta["t"]),
        meta_init_mid_prev=np.float32(meta["init_mid_prev"]),
        horizon=np.int64(horizon),
        nfe=np.int64(nfe),
    )
    print(f"Saved qualitative window to {save_path}")


def save_json(obj: Dict[str, Any], path: str):
    def _conv(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_conv)
    print(f"Saved JSON -> {path}")


__all__ = [
    "seed_all",
    "resolve_context_length",
    "crop_history_window",
    "sample_training_context_length",
    "evaluate_average_loss",
    "train_loop",
    "generate_continuation",
    "select_eval_window_starts",
    "eval_one_window",
    "eval_many_windows",
    "eval_rollout_horizons",
    "benchmark_sampling_latency",
    "eval_speed_quality_nfe",
    "run_ablation_grid",
    "summarize_ablation_for_table",
    "save_qualitative_window_npz",
    "save_json",
    "compare_l2_sequences",
]
